#pragma once
#include <span>
#include <cstdint>
#include <cstring>
#include <random>

/* -------- portable 8-byte fingerprint of a digest ------------------ */
/*  Works with BOTH fixed-extent and dynamic-extent spans.             */
template<std::size_t N = std::dynamic_extent>
inline uint64_t digest_tag(std::span<const uint8_t, N> d) {
    static_assert(N == std::dynamic_extent || N >= 8,
                  "digest_tag needs at least 8 bytes");
    uint64_t tag;
    std::memcpy(&tag, d.data(), 8);
    return tag;
}

/* -------- fill a span with random bytes ---------------------------- */
inline void fill_rand(std::span<uint8_t> buf) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint32_t> dist;
    for (std::size_t i = 0; i < buf.size(); i += 4) {
        uint32_t x = dist(rng);
        std::memcpy(buf.data() + i, &x,
                    std::min<std::size_t>(4, buf.size() - i));
    }
}
