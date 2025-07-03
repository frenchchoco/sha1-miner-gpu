#pragma once
#include <cstdint>

struct sha1_ctx {
    uint32_t state[5];
    uint64_t count;
    uint8_t buf[64];
};

inline void sha1_init(sha1_ctx &c) {
    c.state[0] = 0x67452301;
    c.state[1] = 0xEFCDAB89;
    c.state[2] = 0x98BADCFE;
    c.state[3] = 0x10325476;
    c.state[4] = 0xC3D2E1F0;
    c.count = 0;
}

inline void sha1_transform(uint32_t s[5], const uint8_t blk[64]) {
#define ROL(x,n) (((x)<<(n))|((x)>>(32-(n))))
    uint32_t w[80]; // message schedule
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(blk[4 * i]) << 24) | (uint32_t(blk[4 * i + 1]) << 16) |
               (uint32_t(blk[4 * i + 2]) << 8) | blk[4 * i + 3];
    }
    for (int i = 16; i < 80; ++i) w[i] = ROL(w[i-3]^w[i-8]^w[i-14]^w[i-16], 1);

    uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }
        uint32_t t = ROL(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = t;
    }
    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
    s[4] += e;
#undef ROL
}

inline void sha1_update(sha1_ctx &c, const void *data, size_t len) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(data);
    while (len) {
        size_t i = c.count & 63;
        size_t n = std::min<size_t>(len, 64 - i);
        std::memcpy(c.buf + i, p, n);
        p += n;
        len -= n;
        c.count += n;
        if (((c.count) & 63) == 0) sha1_transform(c.state, c.buf);
    }
}

inline void sha1_final(sha1_ctx &c, uint8_t out[20]) {
    size_t i = c.count & 63;
    c.buf[i++] = 0x80;
    if (i > 56) {
        std::memset(c.buf + i, 0, 64 - i);
        sha1_transform(c.state, c.buf);
        i = 0;
    }
    std::memset(c.buf + i, 0, 56 - i);
    uint64_t bits = c.count * 8;
    for (int j = 0; j < 8; ++j) c.buf[63 - j] = uint8_t(bits >> (8 * j));
    sha1_transform(c.state, c.buf);
    for (int j = 0; j < 5; ++j) {
        out[4 * j] = uint8_t(c.state[j] >> 24);
        out[4 * j + 1] = uint8_t(c.state[j] >> 16);
        out[4 * j + 2] = uint8_t(c.state[j] >> 8);
        out[4 * j + 3] = uint8_t(c.state[j]);
    }
}
