#pragma once

// What a prompt can reuse of what the context already holds.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gaze {

// The tokens next shares with held from the start, short of next's last: that
// one is decoded either way, because sampling reads the logits its decode leaves.
inline size_t reusable_prefix(const std::vector<int32_t>& held, const std::vector<int32_t>& next) {
    if (next.empty()) return 0;
    size_t limit = next.size() - 1;
    size_t n = 0;
    while (n < limit && n < held.size() && held[n] == next[n]) n++;
    return n;
}

}  // namespace gaze
