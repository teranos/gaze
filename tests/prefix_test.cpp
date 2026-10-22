// What a request can reuse of what the context already holds: the tokens it
// shares from the start, short of the last, which is decoded for its logits.

#include "prefix.h"

#include <iostream>

static int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " #cond << std::endl; \
            failures++;                                                          \
        }                                                                        \
    } while (0)

int main() {
    // codex sends the whole conversation again, so the last turn's prompt is the start of this one.
    CHECK(gaze::reusable_prefix({1, 2, 3}, {1, 2, 3, 4, 5}) == 3);
    // The same prompt again: everything but the last, which must be decoded to sample from.
    CHECK(gaze::reusable_prefix({1, 2, 3}, {1, 2, 3}) == 2);
    // They part at the third token: two are kept.
    CHECK(gaze::reusable_prefix({1, 2, 9, 9}, {1, 2, 3, 4}) == 2);
    // The context holds more than the prompt shares: only what is shared counts.
    CHECK(gaze::reusable_prefix({1, 2, 3, 4, 5, 6}, {1, 2, 3, 7}) == 3);
    // Nothing held, nothing shared, or nothing asked.
    CHECK(gaze::reusable_prefix({}, {1, 2}) == 0);
    CHECK(gaze::reusable_prefix({5}, {1, 2}) == 0);
    CHECK(gaze::reusable_prefix({1, 2}, {}) == 0);
    CHECK(gaze::reusable_prefix({1, 2}, {1}) == 0);

    if (failures > 0) {
        std::cerr << failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "prefix_test: all checks passed" << std::endl;
    return 0;
}
