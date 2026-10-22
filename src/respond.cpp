#include "inference_internal.h"
#include "chat.h"

#include <atomic>
#include <chrono>
#include <iostream>

// Defined in plugin.cpp
std::string sanitize_utf8(const std::string& s);

using ordered_json = nlohmann::ordered_json;

InferenceEngine::Reply InferenceEngine::respond(const ordered_json& messages,
                                                const ordered_json& tools,
                                                float temperature,
                                                int max_tokens,
                                                const SamplerConfig& sampler_cfg) {
    std::lock_guard<std::mutex> lock(mutex_);

    Reply reply;
    if (!model_ || !ctx_) {
        reply.error = "no model loaded";
        return reply;
    }

    common_chat_params params;
    common_chat_templates_inputs inputs;
    try {
        auto templates = common_chat_templates_init(model_, "");
        inputs.messages = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools = common_chat_tools_parse_oaicompat(tools);
        inputs.parallel_tool_calls = true;
        params = common_chat_templates_apply(templates.get(), inputs);
    } catch (const std::exception& e) {
        reply.error = std::string("the chat did not render with the model's template: ") + e.what();
        return reply;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(params.prompt.size() + 32);
    int n_tokens = llama_tokenize(vocab, params.prompt.c_str(), params.prompt.size(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        reply.error = "the rendered prompt did not tokenize";
        return reply;
    }
    tokens.resize(n_tokens);

    // A conversation cut to fit is not the one codex sent, so it is refused, with the numbers.
    int ctx_size = llama_n_ctx(ctx_);
    if (n_tokens + max_tokens > ctx_size) {
        reply.error = "the prompt is " + std::to_string(n_tokens) + " tokens and the answer may take " +
                      std::to_string(max_tokens) + ", past the context of " + std::to_string(ctx_size) +
                      ": raise n_ctx in gaze's record, or send codex fewer tools";
        return reply;
    }

    llama_memory_clear(llama_get_memory(ctx_), true);
    ChatResult result;
    auto t0 = std::chrono::steady_clock::now();
    if (llama_decode(ctx_, llama_batch_get_one(tokens.data(), n_tokens)) != 0) {
        reply.error = "the prompt did not decode";
        return reply;
    }
    result.prompt_eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "[responses] " << n_tokens << " prompt tokens in " << result.prompt_eval_ms << "ms" << std::endl;

    // The template's own stop strings end the answer as the end-of-generation token would.
    std::string written;
    auto stop_at = [&](const std::string& token) -> bool {
        written += token;
        for (const auto& stop : params.additional_stops) {
            if (!stop.empty() && written.size() >= stop.size() &&
                written.compare(written.size() - stop.size(), stop.size(), stop) == 0) {
                return false;
            }
        }
        return true;
    };
    generate(temperature, max_tokens, stop_at, sampler_cfg, result);

    std::string text = result.content;
    for (const auto& stop : params.additional_stops) {
        if (!stop.empty() && text.size() >= stop.size() &&
            text.compare(text.size() - stop.size(), stop.size(), stop) == 0) {
            text.resize(text.size() - stop.size());
            break;
        }
    }

    reply.prompt_tokens = n_tokens;
    reply.completion_tokens = result.completion_tokens;

    common_chat_parser_params parser_params(params);
    parser_params.parse_tool_calls = !inputs.tools.empty();
    common_chat_msg msg;
    try {
        if (!params.parser.empty()) parser_params.parser.load(params.parser);
        msg = common_chat_parse(text, false, parser_params);
    } catch (const std::exception& e) {
        reply.error = std::string("what the model wrote did not parse as its tool format: ") + e.what() +
                      "; it wrote: " + sanitize_utf8(text);
        return reply;
    }

    static std::atomic<long> calls{0};
    reply.content = sanitize_utf8(msg.content);
    for (const auto& call : msg.tool_calls) {
        std::string id = call.id.empty() ? "call_gaze_" + std::to_string(++calls) : call.id;
        reply.tool_calls.push_back({call.name, sanitize_utf8(call.arguments), id});
    }
    return reply;
}

static std::string response_id() {
    static std::atomic<long> n{0};
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return "resp_gaze_" + std::to_string(now) + "_" + std::to_string(++n);
}

static void answer(protocol::HTTPResponse* resp, int status, const std::string& content_type, const std::string& body) {
    resp->set_status_code(status);
    resp->set_body(body);
    auto* ct = resp->add_headers();
    ct->set_name("Content-Type");
    ct->add_values(content_type);
}

static void refuse(protocol::HTTPResponse* resp, int status, const std::string& message) {
    ordered_json body = {{"error", {{"type", "invalid_request_error"}, {"message", message}}}};
    answer(resp, status, "application/json", body.dump());
}

void GazePlugin::respond_http(const protocol::HTTPRequest* req, protocol::HTTPResponse* resp) {
    ordered_json body;
    try {
        body = ordered_json::parse(req->body());
    } catch (const std::exception& e) {
        refuse(resp, 400, std::string("the body is not JSON: ") + e.what());
        return;
    }

    gaze::ChatRequest chat;
    try {
        chat = gaze::chat_request(body);
    } catch (const std::invalid_argument& e) {
        refuse(resp, 400, e.what());
        return;
    }
    for (const auto& name : chat.dropped) {
        std::cout << "[responses] left out tool " << name << ": gaze passes function tools only" << std::endl;
    }

    std::string model = body.value("model", "");
    InferenceEngine* engine = model.empty() ? default_engine() : get_engine(model);
    if (!engine || !engine->is_loaded()) {
        std::string names;
        for (const auto& name : model_names()) names += (names.empty() ? "" : ", ") + name;
        refuse(resp, 404, "gaze has no model '" + model + "'; it serves: " + (names.empty() ? "none" : names));
        return;
    }

    float temperature = body.contains("temperature") && body["temperature"].is_number()
        ? body["temperature"].get<float>() : 0.7f;
    int max_tokens = body.contains("max_output_tokens") && body["max_output_tokens"].is_number_integer()
        ? body["max_output_tokens"].get<int>() : 1024;

    set_activity("responding");
    auto reply = engine->respond(chat.messages, chat.tools, temperature, max_tokens, sampler_config());
    set_activity("");

    std::string id = response_id();
    if (!reply.error.empty()) {
        std::cout << "[responses] " << id << " failed: " << reply.error << std::endl;
        answer(resp, 200, "text/event-stream", gaze::responses_failed_sse(id, engine->model_name(), reply.error));
        return;
    }

    gaze::Response out;
    out.id = id;
    out.model = engine->model_name();
    out.content = reply.content;
    out.tool_calls = reply.tool_calls;
    out.input_tokens = reply.prompt_tokens;
    out.output_tokens = reply.completion_tokens;
    std::cout << "[responses] " << id << ": " << reply.completion_tokens << " tokens, "
              << reply.tool_calls.size() << " tool call(s)" << std::endl;
    answer(resp, 200, "text/event-stream", gaze::responses_sse(out));
}
