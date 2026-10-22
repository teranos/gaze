// The Responses API as codex 0.163 speaks it: the request it sends, mapped to
// the chat body llama.cpp's templates read, and the SSE codex parses back.

#include "responses.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

static int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " #cond << std::endl; \
            failures++;                                                          \
        }                                                                        \
    } while (0)

using ordered_json = nlohmann::ordered_json;

// Trimmed from what codex 0.163 sent: two developer messages, a user message of
// input_text parts, a function tool and a web_search tool.
static ordered_json codex_turn() {
    return ordered_json::parse(R"({
        "model": "qwen2.5-3b-instruct",
        "stream": true,
        "store": false,
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are Codex."}]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Sandbox: read-only."}]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "List the "},
                {"type": "input_text", "text": "files."}
            ]}
        ],
        "tools": [
            {"type": "function", "name": "exec_command", "description": "Runs a command.",
             "strict": false, "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
            {"type": "web_search", "external_web_access": false}
        ]
    })");
}

static void test_messages() {
    auto req = gaze::chat_request(codex_turn());
    CHECK(req.messages.size() == 3);
    // Qwen's template has no developer role: a developer message would be dropped from the prompt.
    CHECK(req.messages[0]["role"] == "system");
    CHECK(req.messages[0]["content"] == "You are Codex.");
    CHECK(req.messages[1]["role"] == "system");
    CHECK(req.messages[2]["role"] == "user");
    CHECK(req.messages[2]["content"] == "List the files.");
}

static void test_tools() {
    auto req = gaze::chat_request(codex_turn());
    CHECK(req.tools.size() == 1);
    CHECK(req.tools[0]["type"] == "function");
    CHECK(req.tools[0]["function"]["name"] == "exec_command");
    CHECK(req.tools[0]["function"]["description"] == "Runs a command.");
    CHECK(req.tools[0]["function"]["parameters"]["required"][0] == "cmd");
    // A tool the model cannot call is named, not passed and not lost in silence.
    CHECK(req.dropped.size() == 1);
    CHECK(req.dropped[0] == "web_search");
}

// A second turn: the model called a tool, codex ran it, and sends both back.
static void test_tool_round_trip() {
    auto body = codex_turn();
    body["input"].push_back(ordered_json::parse(R"({"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "Listing."}]})"));
    body["input"].push_back(ordered_json::parse(R"({"type": "function_call", "name": "exec_command",
        "arguments": "{\"cmd\":\"ls\"}", "call_id": "call_1"})"));
    body["input"].push_back(ordered_json::parse(R"({"type": "function_call_output", "call_id": "call_1",
        "output": "a.txt\nb.txt"})"));

    auto req = gaze::chat_request(body);
    CHECK(req.messages.size() == 5);
    // The assistant's text is history the model wrote; it stays.
    CHECK(req.messages[3]["role"] == "assistant");
    CHECK(req.messages[3]["content"] == "Listing.");
    // A call follows the message that made it, as one assistant turn.
    CHECK(req.messages[3]["tool_calls"].size() == 1);
    CHECK(req.messages[3]["tool_calls"][0]["id"] == "call_1");
    CHECK(req.messages[3]["tool_calls"][0]["function"]["name"] == "exec_command");
    CHECK(req.messages[3]["tool_calls"][0]["function"]["arguments"] == "{\"cmd\":\"ls\"}");
    CHECK(req.messages[4]["role"] == "tool");
    CHECK(req.messages[4]["tool_call_id"] == "call_1");
    CHECK(req.messages[4]["content"] == "a.txt\nb.txt");
}

static void test_instructions_and_string_input() {
    auto body = ordered_json::parse(R"({"instructions": "Be brief.", "input": "Hi."})");
    auto req = gaze::chat_request(body);
    CHECK(req.messages.size() == 2);
    CHECK(req.messages[0]["role"] == "system");
    CHECK(req.messages[0]["content"] == "Be brief.");
    CHECK(req.messages[1]["role"] == "user");
    CHECK(req.messages[1]["content"] == "Hi.");
}

static void test_refusals() {
    bool threw = false;
    try { gaze::chat_request(ordered_json::parse(R"({"model": "x"})")); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { gaze::chat_request(ordered_json::parse(R"({"input": [{"type": "computer_call"}]})")); }
    catch (const std::invalid_argument& e) { threw = std::string(e.what()).find("computer_call") != std::string::npos; }
    CHECK(threw);
}

// Every event codex reads, in the order it reads them.
static std::vector<ordered_json> events_of(const std::string& sse) {
    std::vector<ordered_json> out;
    std::istringstream in(sse);
    std::string line, event;
    while (std::getline(in, line)) {
        if (line.rfind("event: ", 0) == 0) event = line.substr(7);
        if (line.rfind("data: ", 0) == 0) {
            auto data = ordered_json::parse(line.substr(6));
            if (data["type"] != event) data["event_mismatch"] = event;
            out.push_back(data);
        }
    }
    return out;
}

static void test_sse_message() {
    gaze::Response r;
    r.id = "resp_7";
    r.model = "qwen2.5-3b-instruct";
    r.content = "Two files.";
    r.input_tokens = 40;
    r.output_tokens = 3;

    auto events = events_of(gaze::responses_sse(r));
    CHECK(events.size() == 3);
    CHECK(events[0]["type"] == "response.created");
    CHECK(events[0]["response"]["id"] == "resp_7");
    CHECK(events[1]["type"] == "response.output_item.done");
    CHECK(events[1]["item"]["type"] == "message");
    CHECK(events[1]["item"]["role"] == "assistant");
    CHECK(events[1]["item"]["content"][0]["type"] == "output_text");
    CHECK(events[1]["item"]["content"][0]["text"] == "Two files.");
    CHECK(events[2]["type"] == "response.completed");
    CHECK(events[2]["response"]["id"] == "resp_7");
    CHECK(events[2]["response"]["usage"]["input_tokens"] == 40);
    CHECK(events[2]["response"]["usage"]["output_tokens"] == 3);
    CHECK(events[2]["response"]["usage"]["total_tokens"] == 43);
    for (const auto& e : events) CHECK(!e.contains("event_mismatch"));
}

static void test_sse_tool_call() {
    gaze::Response r;
    r.id = "resp_8";
    r.model = "qwen2.5-3b-instruct";
    r.tool_calls.push_back({"exec_command", "{\"cmd\":\"ls\"}", "call_9"});

    auto events = events_of(gaze::responses_sse(r));
    // No text was written, so there is no message item.
    CHECK(events.size() == 3);
    CHECK(events[1]["item"]["type"] == "function_call");
    CHECK(events[1]["item"]["name"] == "exec_command");
    CHECK(events[1]["item"]["arguments"] == "{\"cmd\":\"ls\"}");
    CHECK(events[1]["item"]["call_id"] == "call_9");
    CHECK(events[2]["type"] == "response.completed");
}

// What the model could not do reaches codex as its error, not as an empty reply.
static void test_sse_failed() {
    auto events = events_of(gaze::responses_failed_sse("resp_9", "qwen2.5-3b-instruct",
                                                        "the prompt is 9000 tokens and the context 4096"));
    CHECK(events.size() == 2);
    CHECK(events[0]["type"] == "response.created");
    CHECK(events[1]["type"] == "response.failed");
    CHECK(events[1]["response"]["id"] == "resp_9");
    CHECK(events[1]["response"]["status"] == "failed");
    CHECK(events[1]["response"]["error"]["message"] == "the prompt is 9000 tokens and the context 4096");
}

int main() {
    test_messages();
    test_tools();
    test_tool_round_trip();
    test_instructions_and_string_input();
    test_refusals();
    test_sse_message();
    test_sse_tool_call();
    test_sse_failed();
    if (failures > 0) {
        std::cerr << failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "responses_test: all checks passed" << std::endl;
    return 0;
}
