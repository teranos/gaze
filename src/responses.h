#pragma once

// The Responses API as codex speaks it, mapped to and from the OpenAI chat
// shape llama.cpp's templates and parser read. Nothing here touches a model.

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace gaze {

// A Responses request, as a chat.
struct ChatRequest {
    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    nlohmann::ordered_json tools = nlohmann::ordered_json::array();
    std::vector<std::string> dropped;  // tools left out, by name
};

// Throws std::invalid_argument naming what it could not read.
ChatRequest chat_request(const nlohmann::ordered_json& body);

struct ToolCall {
    std::string name;
    std::string arguments;
    std::string call_id;
};

// What the model wrote, as one Responses response.
struct Response {
    std::string id;
    std::string model;
    std::string content;
    std::vector<ToolCall> tool_calls;
    int input_tokens = 0;
    int output_tokens = 0;
};

// The whole event stream codex reads: created, an item per output, completed.
std::string responses_sse(const Response& r);

// The stream for a response the model could not give: created, then failed with why.
std::string responses_failed_sse(const std::string& id, const std::string& model, const std::string& message);

}  // namespace gaze
