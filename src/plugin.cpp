#include "plugin.h"
#include "log_capture.h"

#include <iostream>

// BPE tokens for multi-byte scripts (Hindi, CJK, emoji) can be partial UTF-8
// sequences. Protobuf string fields reject invalid UTF-8, so replace bad bytes
// with U+FFFD before serializing.
std::string sanitize_utf8(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        int len = 0;
        if (c < 0x80) len = 1;
        else if ((c >> 5) == 0x06) len = 2;
        else if ((c >> 4) == 0x0E) len = 3;
        else if ((c >> 3) == 0x1E) len = 4;

        if (len == 0 || i + len > s.size()) {
            out += "\xEF\xBF\xBD";  // U+FFFD
            i++;
            continue;
        }
        // Validate continuation bytes
        bool valid = true;
        for (int j = 1; j < len; j++) {
            if ((s[i + j] & 0xC0) != 0x80) { valid = false; break; }
        }
        if (valid) {
            out.append(s, i, len);
            i += len;
        } else {
            out += "\xEF\xBF\xBD";
            i++;
        }
    }
    return out;
}

// --- GazePlugin (DomainPluginService) ---

grpc::Status GazePlugin::Metadata(grpc::ServerContext* ctx,
                                   const protocol::Empty* req,
                                   protocol::MetadataResponse* resp) {
    resp->set_name("gaze");
    resp->set_version(PLUGIN_VERSION);
    resp->set_qntx_version(">= 0.1.0");
    resp->set_description("Production LLM inference via llama.cpp with Metal acceleration");
    resp->set_author("teranos");
    resp->set_license("MIT");
    return grpc::Status::OK;
}

grpc::Status GazePlugin::Initialize(grpc::ServerContext* ctx,
                                     const protocol::InitializeRequest* req,
                                     protocol::InitializeResponse* resp) {
    auto config = req->config();

    int n_ctx = 2048;
    auto ctx_it = config.find("n_ctx");
    if (ctx_it != config.end()) {
        try { n_ctx = std::stoi(ctx_it->second); } catch (...) {}
        if (n_ctx <= 0) n_ctx = 2048;
    }

    // Collect model paths: "models" JSON array or legacy "model_path" single entry
    std::vector<std::string> model_paths;
    auto models_it = config.find("models");
    if (models_it != config.end() && !models_it->second.empty()) {
        // Core serializes TOML arrays as JSON: ["/path/a.gguf","/path/b.gguf"]
        const std::string& json = models_it->second;
        // Minimal JSON array parser — extract quoted strings
        size_t pos = 0;
        while ((pos = json.find('"', pos)) != std::string::npos) {
            size_t start = pos + 1;
            size_t end = json.find('"', start);
            if (end == std::string::npos) break;
            std::string path = json.substr(start, end - start);
            if (!path.empty()) model_paths.push_back(path);
            pos = end + 1;
        }
    }
    if (model_paths.empty()) {
        auto it = config.find("model_path");
        if (it != config.end() && !it->second.empty()) {
            model_paths.push_back(it->second);
        }
    }


    // Parse sampler config
    auto parse_int = [&config](const char* key, int fallback) -> int {
        auto it = config.find(key);
        if (it != config.end()) { try { return std::stoi(it->second); } catch (...) {} }
        return fallback;
    };
    auto parse_float = [&config](const char* key, float fallback) -> float {
        auto it = config.find(key);
        if (it != config.end()) { try { return std::stof(it->second); } catch (...) {} }
        return fallback;
    };
    sampler_cfg_.top_k = parse_int("top_k", 0);
    sampler_cfg_.top_p = parse_float("top_p", 1.0f);
    sampler_cfg_.min_p = parse_float("min_p", 0.0f);
    sampler_cfg_.typical_p = parse_float("typical_p", 1.0f);
    sampler_cfg_.penalty_last_n = parse_int("penalty_last_n", 0);
    sampler_cfg_.penalty_repeat = parse_float("repeat_penalty", 1.0f);
    sampler_cfg_.penalty_freq = parse_float("freq_penalty", 0.0f);
    sampler_cfg_.penalty_present = parse_float("presence_penalty", 0.0f);

    // Sampler config details are available via ConfigSchema RPC

    auto log_it = config.find("log_level");
    std::string log_level = (log_it != config.end()) ? log_it->second : "info";
    int idle_seconds = parse_int("idle_unload_seconds", 0);
    if (idle_seconds < 0) idle_seconds = 0;

    // A node initializes a running plugin again when its record changes: what
    // the last Initialize loaded goes, and this record's models are loaded.
    // The answer waits for no load (see plugin.h).
    want(Wanted{model_paths, n_ctx, log_level, idle_seconds});

    resp->set_llm_provider(true);

    auto* version_route = resp->add_http_routes();
    version_route->set_method("GET");
    version_route->set_path("/version");
    version_route->set_description("Returns plugin version string");

    auto* status_route = resp->add_http_routes();
    status_route->set_method("GET");
    status_route->set_path("/status");
    status_route->set_description("Returns JSON with state, version, model count, and current activity");

    auto* responses_route = resp->add_http_routes();
    responses_route->set_method("POST");
    responses_route->set_path("/v1/responses");
    responses_route->set_description("The OpenAI Responses API, answered with a loaded model as one event stream");

    // Its sigils are the routes it declares.
    resp->set_sigils_from(protocol::SIGILS_FROM_HTTP_ROUTES);

    return grpc::Status::OK;
}

GazePlugin::GazePlugin() : loader_(&GazePlugin::loader_loop, this) {}

GazePlugin::~GazePlugin() {
    {
        std::lock_guard<std::mutex> lock(wanted_mu_);
        quit_ = true;
    }
    wanted_cv_.notify_one();
    loader_.join();
}

int64_t GazePlugin::seconds_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void GazePlugin::want(Wanted wanted) {
    {
        std::lock_guard<std::mutex> wanted_lock(wanted_mu_);
        std::lock_guard<std::mutex> engines_lock(engines_mu_);
        engines_.clear();
        still_loading_ = static_cast<int>(wanted.paths.size());
        wanted_ = std::move(wanted);
        idled_ = false;
        last_used_.store(seconds_now());
        asked_.fetch_add(1);
    }
    wanted_cv_.notify_one();
    loaded_cv_.notify_all();
}

// "it needs to kind of get out of the way of other things happening"
void GazePlugin::ready_for_a_request() {
    last_used_.store(seconds_now());
    bool asked = false;
    {
        std::lock_guard<std::mutex> wanted_lock(wanted_mu_);
        if (idled_) {
            std::lock_guard<std::mutex> engines_lock(engines_mu_);
            idled_ = false;
            still_loading_ = static_cast<int>(wanted_.paths.size());
            asked_.fetch_add(1);
            asked = true;
            std::cout << "[idle] A request came: loading " << wanted_.paths.size() << " model(s) again" << std::endl;
        }
    }
    if (asked) wanted_cv_.notify_one();
    // A model takes about a minute to load on a CPU; a request waits up to three.
    std::unique_lock<std::mutex> lock(engines_mu_);
    loaded_cv_.wait_for(lock, std::chrono::minutes(3), [&] { return !engines_.empty() || still_loading_ <= 0; });
}

void GazePlugin::loader_loop() {
    uint64_t loaded_for = 0;
    for (;;) {
        Wanted wanted;
        uint64_t asking;
        {
            std::unique_lock<std::mutex> lock(wanted_mu_);
            for (;;) {
                if (quit_) return;
                if (asked_.load() != loaded_for) break;
                const int idle = wanted_.idle_seconds;
                if (idle <= 0 || idled_) {
                    wanted_cv_.wait(lock);
                    continue;
                }
                const int64_t left = last_used_.load() + idle - seconds_now();
                if (left > 0) {
                    wanted_cv_.wait_for(lock, std::chrono::seconds(left));
                    continue;
                }
                // Idle for long enough: what is loaded goes, unless a request holds it.
                std::lock_guard<std::mutex> engines_lock(engines_mu_);
                bool held = false;
                for (const auto& [_, engine] : engines_) held = held || engine.use_count() > 1;
                if (held || engines_.empty()) {
                    last_used_.store(seconds_now());
                    continue;
                }
                engines_.clear();
                idled_ = true;
                std::cout << "[idle] No request for " << idle << "s: the model(s) are let go" << std::endl;
            }
            wanted = wanted_;
            asking = asked_.load();
        }
        for (const auto& path : wanted.paths) {
            // Asked again since: the rest of this asking is not loaded.
            if (asked_.load() != asking) break;
            auto engine = std::make_shared<InferenceEngine>();
            bool loaded = engine->load_model(path, wanted.n_ctx);
            std::lock_guard<std::mutex> lock(engines_mu_);
            // Asked again while it loaded: it is dropped, and the count is the new asking's.
            if (asked_.load() != asking) break;
            if (loaded) {
                engines_[engine->model_name()] = std::move(engine);
            } else {
                std::cout << "[init] Failed to load model: " << path << std::endl;
            }
            still_loading_--;
            last_used_.store(seconds_now());
            loaded_cv_.notify_all();
        }
        // Flush condensed log summary
        LogCapture::instance().flush_summary(wanted.log_level);
        loaded_for = asking;
    }
}

grpc::Status GazePlugin::Shutdown(grpc::ServerContext* ctx,
                                   const protocol::Empty* req,
                                   protocol::ShutdownResponse* resp) {
    // Nothing is wanted: what is loaded goes, and what is loading is dropped.
    want(Wanted{});
    resp->set_stopped(true);
    return grpc::Status::OK;
}

std::shared_ptr<InferenceEngine> GazePlugin::get_engine(const std::string& model_name) {
    ready_for_a_request();
    std::lock_guard<std::mutex> lock(engines_mu_);
    auto it = engines_.find(model_name);
    if (it != engines_.end()) return it->second;
    return nullptr;
}

std::shared_ptr<InferenceEngine> GazePlugin::default_engine() {
    ready_for_a_request();
    std::lock_guard<std::mutex> lock(engines_mu_);
    if (engines_.empty()) return nullptr;
    return engines_.begin()->second;
}

std::vector<std::string> GazePlugin::model_names() const {
    std::lock_guard<std::mutex> lock(engines_mu_);
    std::vector<std::string> names;
    names.reserve(engines_.size());
    for (const auto& [name, _] : engines_) {
        names.push_back(name);
    }
    return names;
}

grpc::Status GazePlugin::Health(grpc::ServerContext* ctx,
                                 const protocol::Empty* req,
                                 protocol::HealthResponse* resp) {
    auto* details = resp->mutable_details();
    std::lock_guard<std::mutex> lock(engines_mu_);
    int still_loading = still_loading_;

    if (engines_.empty()) {
        resp->set_healthy(true);
        resp->set_message(still_loading > 0
            ? "loading " + std::to_string(still_loading) + " model(s)"
            : "no models loaded");
        return grpc::Status::OK;
    }

    resp->set_healthy(true);

    // Build model list and populate details from first/each engine
    std::string models_str;
    bool first = true;
    for (const auto& [name, engine] : engines_) {
        if (!first) models_str += ", ";
        models_str += name;
        first = false;

        // Per-model details (prefixed if multiple models)
        std::string prefix = (engines_.size() > 1) ? name + "." : "";

        std::string desc = engine->model_desc();
        if (!desc.empty()) {
            (*details)[prefix + "backend"] = desc;
        }

        (*details)[prefix + "context"] = std::to_string(engine->context_length());

        if (engine->has_vision()) {
            (*details)[prefix + "vision"] = "true";
        }

        uint64_t size = engine->model_size_bytes();
        if (size > 0) {
            // Human-readable size
            if (size >= 1024ULL * 1024 * 1024) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.1f GiB", size / (1024.0 * 1024.0 * 1024.0));
                (*details)[prefix + "size"] = buf;
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.0f MiB", size / (1024.0 * 1024.0));
                (*details)[prefix + "size"] = buf;
            }
        }

        uint64_t params = engine->model_n_params();
        if (params > 0) {
            char buf[32];
            if (params >= 1000000000ULL) {
                snprintf(buf, sizeof(buf), "%.1fB", params / 1e9);
            } else {
                snprintf(buf, sizeof(buf), "%.0fM", params / 1e6);
            }
            (*details)[prefix + "params"] = buf;
        }
    }

    // Hardware details from llama.cpp log capture
    auto& logs = LogCapture::instance();
    std::string device = logs.device_name();
    if (!device.empty()) {
        std::string accel = "Metal, " + device;
        std::string family = logs.gpu_family();
        if (!family.empty()) accel += " (" + family + ")";
        int vram = logs.vram_free_mib();
        if (vram > 0) accel += ", " + std::to_string(vram) + " MiB free";
        (*details)["accelerator"] = accel;
    } else {
        (*details)["accelerator"] = "Metal";
    }

    std::string quant = logs.quant_type();
    if (!quant.empty()) {
        (*details)["quantization"] = quant;
    }

    resp->set_message(std::to_string(engines_.size()) + " model(s): " + models_str
        + (still_loading > 0 ? ", loading " + std::to_string(still_loading) + " more" : ""));
    return grpc::Status::OK;
}

grpc::Status GazePlugin::ConfigSchema(grpc::ServerContext* ctx,
                                       const protocol::Empty* req,
                                       protocol::ConfigSchemaResponse* resp) {
    auto* fields = resp->mutable_fields();

    protocol::ConfigFieldSchema model_field;
    model_field.set_type("string");
    model_field.set_description("Path to GGUF model file");
    model_field.set_required(true);
    (*fields)["model_path"] = model_field;

    protocol::ConfigFieldSchema ctx_field;
    ctx_field.set_type("number");
    ctx_field.set_description("Context window size in tokens");
    ctx_field.set_default_value("2048");
    ctx_field.set_min_value("512");
    ctx_field.set_max_value("32768");
    (*fields)["n_ctx"] = ctx_field;

    protocol::ConfigFieldSchema idle_field;
    idle_field.set_type("number");
    idle_field.set_description("Seconds without a request after which the models are let go, and loaded again on the next request; 0 keeps them loaded");
    idle_field.set_default_value("0");
    idle_field.set_min_value("0");
    (*fields)["idle_unload_seconds"] = idle_field;

    protocol::ConfigFieldSchema top_k_field;
    top_k_field.set_type("number");
    top_k_field.set_description("Top-K sampling: keep only the top K tokens (0 = disabled)");
    top_k_field.set_default_value("0");
    top_k_field.set_min_value("0");
    top_k_field.set_max_value("500");
    (*fields)["top_k"] = top_k_field;

    protocol::ConfigFieldSchema top_p_field;
    top_p_field.set_type("number");
    top_p_field.set_description("Top-P (nucleus) sampling: cumulative probability cutoff (1.0 = disabled)");
    top_p_field.set_default_value("1.0");
    top_p_field.set_min_value("0.0");
    top_p_field.set_max_value("1.0");
    (*fields)["top_p"] = top_p_field;

    protocol::ConfigFieldSchema min_p_field;
    min_p_field.set_type("number");
    min_p_field.set_description("Min-P sampling: drop tokens below this fraction of top token (0.0 = disabled)");
    min_p_field.set_default_value("0.0");
    min_p_field.set_min_value("0.0");
    min_p_field.set_max_value("1.0");
    (*fields)["min_p"] = min_p_field;

    protocol::ConfigFieldSchema typical_p_field;
    typical_p_field.set_type("number");
    typical_p_field.set_description("Typical-P sampling: locally typical sampling threshold (1.0 = disabled)");
    typical_p_field.set_default_value("1.0");
    typical_p_field.set_min_value("0.0");
    typical_p_field.set_max_value("1.0");
    (*fields)["typical_p"] = typical_p_field;

    protocol::ConfigFieldSchema repeat_penalty_field;
    repeat_penalty_field.set_type("number");
    repeat_penalty_field.set_description("Repetition penalty (1.0 = disabled)");
    repeat_penalty_field.set_default_value("1.0");
    repeat_penalty_field.set_min_value("0.0");
    repeat_penalty_field.set_max_value("3.0");
    (*fields)["repeat_penalty"] = repeat_penalty_field;

    protocol::ConfigFieldSchema penalty_last_n_field;
    penalty_last_n_field.set_type("number");
    penalty_last_n_field.set_description("Penalty window: number of recent tokens to penalize (0 = disabled)");
    penalty_last_n_field.set_default_value("0");
    penalty_last_n_field.set_min_value("0");
    penalty_last_n_field.set_max_value("2048");
    (*fields)["penalty_last_n"] = penalty_last_n_field;

    return grpc::Status::OK;
}

grpc::Status GazePlugin::RegisterElements(grpc::ServerContext* ctx,
                                          const protocol::Empty* req,
                                          protocol::ElementDefResponse* resp) {
    return grpc::Status::OK;
}

grpc::Status GazePlugin::HandleHTTP(grpc::ServerContext* ctx,
                                     const protocol::HTTPRequest* req,
                                     protocol::HTTPResponse* resp) {
    if (req->method() == "GET" && req->path() == "/version") {
        resp->set_status_code(200);
        resp->set_body(PLUGIN_VERSION);
        return grpc::Status::OK;
    }

    // The node passes the query along with the path; the route is the path alone.
    std::string path = req->path().substr(0, req->path().find('?'));
    if (req->method() == "POST" && path == "/v1/responses") {
        respond_http(req, resp);
        return grpc::Status::OK;
    }

    if (req->method() == "GET" && req->path() == "/status") {
        size_t loaded;
        int still_loading;
        {
            std::lock_guard<std::mutex> lock(engines_mu_);
            loaded = engines_.size();
            still_loading = still_loading_;
        }
        std::string state = loaded > 0 ? "ready" : (still_loading > 0 ? "loading" : "no_model");
        std::string act = activity();
        std::string body = "{\"state\":\"" + state + "\",\"version\":\"" + PLUGIN_VERSION + "\"";
        body += ",\"models\":" + std::to_string(loaded);
        if (!act.empty()) body += ",\"activity\":\"" + act + "\"";
        body += "}";
        resp->set_status_code(200);
        resp->set_body(body);
        auto* ct = resp->add_headers();
        ct->set_name("Content-Type");
        ct->add_values("application/json");
        return grpc::Status::OK;
    }

    resp->set_status_code(404);
    resp->set_body("not found");
    return grpc::Status::OK;
}

grpc::Status GazePlugin::HandleWebSocket(
    grpc::ServerContext* ctx,
    grpc::ServerReaderWriter<protocol::WebSocketMessage,
                             protocol::WebSocketMessage>* stream) {
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "gaze has no WebSocket interface");
}

grpc::Status GazePlugin::ExecuteJob(grpc::ServerContext* ctx,
                                     const protocol::ExecuteJobRequest* req,
                                     protocol::ExecuteJobResponse* resp) {
    resp->set_success(false);
    resp->set_error("gaze does not handle async jobs");
    return grpc::Status::OK;
}

