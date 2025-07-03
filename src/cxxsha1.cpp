#include "cxxsha1.hpp"
#include <cstring>
#include <algorithm>
#include <cstdint>

// SHA-1 constants
constexpr uint32_t K[4] = {
    0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6
};

// Rotate left
static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// SHA-1 round functions
static inline uint32_t f1(uint32_t b, uint32_t c, uint32_t d) {
    return (b & c) | (~b & d);
}

static inline uint32_t f2(uint32_t b, uint32_t c, uint32_t d) {
    return b ^ c ^ d;
}

static inline uint32_t f3(uint32_t b, uint32_t c, uint32_t d) {
    return (b & c) | (b & d) | (c & d);
}

// Initialize SHA-1 context
void sha1_init(sha1_ctx &ctx) {
    ctx.state[0] = 0x67452301;
    ctx.state[1] = 0xEFCDAB89;
    ctx.state[2] = 0x98BADCFE;
    ctx.state[3] = 0x10325476;
    ctx.state[4] = 0xC3D2E1F0;
    ctx.count = 0;
    ctx.buffer_len = 0;
}

// Process a single 512-bit block
static void sha1_process_block(sha1_ctx &ctx, const uint8_t block[64]) {
    uint32_t w[80];

    // Prepare message schedule
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t(block[i * 4]) << 24) |
               (uint32_t(block[i * 4 + 1]) << 16) |
               (uint32_t(block[i * 4 + 2]) << 8) |
               uint32_t(block[i * 4 + 3]);
    }

    for (int i = 16; i < 80; i++) {
        w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    // Working variables
    uint32_t a = ctx.state[0];
    uint32_t b = ctx.state[1];
    uint32_t c = ctx.state[2];
    uint32_t d = ctx.state[3];
    uint32_t e = ctx.state[4];

    // Main loop
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;

        if (i < 20) {
            f = f1(b, c, d);
            k = K[0];
        } else if (i < 40) {
            f = f2(b, c, d);
            k = K[1];
        } else if (i < 60) {
            f = f3(b, c, d);
            k = K[2];
        } else {
            f = f2(b, c, d);
            k = K[3];
        }

        uint32_t temp = rotl32(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Add to state
    ctx.state[0] += a;
    ctx.state[1] += b;
    ctx.state[2] += c;
    ctx.state[3] += d;
    ctx.state[4] += e;
}

// Update SHA-1 with new data
void sha1_update(sha1_ctx &ctx, const void *data, size_t len) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);

    while (len > 0) {
        size_t copy_len = std::min(len, size_t(64 - ctx.buffer_len));
        std::memcpy(ctx.buffer + ctx.buffer_len, bytes, copy_len);

        ctx.buffer_len += copy_len;
        ctx.count += copy_len * 8;
        bytes += copy_len;
        len -= copy_len;

        if (ctx.buffer_len == 64) {
            sha1_process_block(ctx, ctx.buffer);
            ctx.buffer_len = 0;
        }
    }
}

// Finalize SHA-1 and get digest
void sha1_final(sha1_ctx &ctx, uint8_t digest[20]) {
    // Padding
    uint8_t pad[64];
    std::memset(pad, 0, sizeof(pad));
    pad[0] = 0x80;

    uint64_t bit_count = ctx.count;
    size_t pad_len = (ctx.buffer_len < 56) ? (56 - ctx.buffer_len) : (120 - ctx.buffer_len);

    sha1_update(ctx, pad, pad_len);

    // Append length in big-endian
    uint8_t length_bytes[8];
    for (int i = 0; i < 8; i++) {
        length_bytes[i] = (bit_count >> (56 - i * 8)) & 0xFF;
    }
    sha1_update(ctx, length_bytes, 8);

    // Extract digest
    for (int i = 0; i < 5; i++) {
        digest[i * 4] = (ctx.state[i] >> 24) & 0xFF;
        digest[i * 4 + 1] = (ctx.state[i] >> 16) & 0xFF;
        digest[i * 4 + 2] = (ctx.state[i] >> 8) & 0xFF;
        digest[i * 4 + 3] = ctx.state[i] & 0xFF;
    }

    // Clear sensitive data
    std::memset(&ctx, 0, sizeof(ctx));
}
