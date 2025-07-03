#pragma once

#include <cstdint>
#include <cstddef>

// SHA-1 context structure
struct sha1_ctx {
    uint32_t state[5]; // Hash state (A, B, C, D, E)
    uint64_t count; // Number of bits processed
    uint8_t buffer[64]; // Input buffer
    size_t buffer_len; // Current buffer length
};

// SHA-1 functions
void sha1_init(sha1_ctx &ctx);

void sha1_update(sha1_ctx &ctx, const void *data, size_t len);

void sha1_final(sha1_ctx &ctx, uint8_t digest[20]);

// Convenience functions for C++ usage
inline void sha1_update(sha1_ctx &ctx, const uint8_t *data, size_t len) {
    sha1_update(ctx, static_cast<const void *>(data), len);
}

inline void sha1_update(sha1_ctx &ctx, const uint32_t *data, size_t len) {
    sha1_update(ctx, static_cast<const void *>(data), len * sizeof(uint32_t));
}
