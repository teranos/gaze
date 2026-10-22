#include "responses.h"

#include <stdexcept>

namespace gaze {

using ordered_json = nlohmann::ordered_json;

// Content is a string, or a list of parts whose text is joined.
static std::string text_of(const ordered_json& content) {
    if (content.is_string()) return content.get<std::string>();
    std::string text;
    if (!content.is_array()) return text;
    for (const auto& part : content) {
        if (part.contains("text") && part["text"].is_string()) text += part["text"].get<std::string>();
    }
    return text;
}

static bool is_assistant(const ordered_json& messages) {
    return !messages.empty() && messages.back().value("role", "") == "assistant";
}

ChatRequest chat_request(const ordered_json& body) {
    if (!body.contains("input")) throw std::invalid_argument("'input' is required");
    if (body.contains("previous_response_id") && !body["previous_response_id"].is_null()) {
        throw std::invalid_argument("gaze keeps nothing between requests: 'previous_response_id' cannot be answered");
    }

    ChatRequest req;
    if (body.contains("instructions") && body["instructions"].is_string()) {
        req.messages.push_back({{"role", "system"}, {"content", body["instructions"]}});
    }

    const auto& input = body["input"];
    if (input.is_string()) {
        req.messages.push_back({{"role", "user"}, {"content", input}});
    } else if (input.is_array()) {
        for (const auto& item : input) {
            std::string type = item.value("type", "message");
            std::string role = item.value("role", "");
            if (type == "message" && (role == "developer" || role == "system")) {
                // The model's template has a system role; developer is not one of its roles.
                req.messages.push_back({{"role", "system"}, {"content", text_of(item["content"])}});
            } else if (type == "message" && role == "user") {
                req.messages.push_back({{"role", "user"}, {"content", text_of(item["content"])}});
            } else if (type == "message" && role == "assistant") {
                req.messages.push_back({{"role", "assistant"}, {"content", text_of(item["content"])}});
            } else if (type == "function_call") {
                ordered_json call = {
                    {"id", item.value("call_id", "")},
                    {"type", "function"},
                    {"function", {{"name", item.value("name", "")}, {"arguments", item.value("arguments", "")}}},
                };
                // A call belongs to the assistant turn that made it.
                if (!is_assistant(req.messages)) {
                    req.messages.push_back({{"role", "assistant"}, {"content", ""}});
                }
                auto& turn = req.messages.back();
                if (!turn.contains("tool_calls")) turn["tool_calls"] = ordered_json::array();
                turn["tool_calls"].push_back(call);
            } else if (type == "function_call_output") {
                req.messages.push_back({
                    {"role", "tool"},
                    {"tool_call_id", item.value("call_id", "")},
                    {"content", text_of(item["output"])},
                });
            } else if (type == "reasoning") {
                // Another model's encrypted reasoning: nothing this model can read.
                continue;
            } else {
                throw std::invalid_argument("gaze cannot read an input item of type '" + type +
                                            (role.empty() ? "" : "' and role '" + role) + "'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or a list of items");
    }

    if (body.contains("tools") && body["tools"].is_array()) {
        for (const auto& tool : body["tools"]) {
            std::string type = tool.value("type", "");
            if (type != "function") {
                req.dropped.push_back(tool.value("name", type));
                continue;
            }
            ordered_json function = {{"name", tool.value("name", "")}};
            if (tool.contains("description")) function["description"] = tool["description"];
            function["parameters"] = tool.contains("parameters") ? tool["parameters"] : ordered_json::object();
            req.tools.push_back({{"type", "function"}, {"function", function}});
        }
    }
    return req;
}

static void event(std::string& out, const ordered_json& data) {
    out += "event: " + data["type"].get<std::string>() + "\n";
    out += "data: " + data.dump() + "\n\n";
}

std::string responses_sse(const Response& r) {
    std::string out;
    event(out, {{"type", "response.created"},
                {"response", {{"id", r.id}, {"object", "response"}, {"model", r.model}, {"status", "in_progress"}}}});

    int n = 0;
    if (!r.content.empty()) {
        event(out, {{"type", "response.output_item.done"},
                    {"output_index", n++},
                    {"item", {{"type", "message"},
                              {"id", r.id + "_msg"},
                              {"role", "assistant"},
                              {"status", "completed"},
                              {"content", {{{"type", "output_text"}, {"text", r.content}, {"annotations", ordered_json::array()}}}}}}});
    }
    for (const auto& call : r.tool_calls) {
        event(out, {{"type", "response.output_item.done"},
                    {"output_index", n},
                    {"item", {{"type", "function_call"},
                              {"id", r.id + "_fc" + std::to_string(n)},
                              {"status", "completed"},
                              {"name", call.name},
                              {"arguments", call.arguments},
                              {"call_id", call.call_id}}}});
        n++;
    }

    event(out, {{"type", "response.completed"},
                {"response", {{"id", r.id},
                              {"object", "response"},
                              {"model", r.model},
                              {"status", "completed"},
                              {"usage", {{"input_tokens", r.input_tokens},
                                         {"output_tokens", r.output_tokens},
                                         {"total_tokens", r.input_tokens + r.output_tokens}}}}}});
    return out;
}

std::string responses_failed_sse(const std::string& id, const std::string& model, const std::string& message) {
    std::string out;
    event(out, {{"type", "response.created"},
                {"response", {{"id", id}, {"object", "response"}, {"model", model}, {"status", "in_progress"}}}});
    event(out, {{"type", "response.failed"},
                {"response", {{"id", id},
                              {"object", "response"},
                              {"model", model},
                              {"status", "failed"},
                              {"error", {{"code", "server_error"}, {"message", message}}}}}});
    return out;
}

}  // namespace gaze
