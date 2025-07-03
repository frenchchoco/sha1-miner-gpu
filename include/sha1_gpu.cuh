#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __host__ static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __host__ static inline uint32_t bswap32(uint32_t v) {
#ifdef __CUDA_ARCH__
    return __byte_perm(v, 0, 0x0123);
#else
#   if defined(__cpp_lib_byteswap) && (__cpp_lib_byteswap >= 202110L)
    return std::byteswap(v);
#   elif defined(_MSC_VER)
    return _byteswap_ulong(v);
#   else
    return __builtin_bswap32(v);
#   endif
#endif
}

/* ---- schedule word: normal function, no template ----------------------- */
__device__ __forceinline__
uint32_t schedule_word(uint32_t w0, uint32_t w1,
                       uint32_t w2, uint32_t w3) {
    uint32_t t = w0 ^ w1 ^ w2 ^ w3;
    return (t << 1) | (t >> 31); // rotl1
}

struct Sha1Ctx {
    uint32_t a, b, c, d, e;
    __device__ __forceinline__
    void round(uint32_t f, uint32_t k, uint32_t w) {
        uint32_t tmp = __funnelshift_l(a, a, 5) + f + e + k + w;
        e = d;
        d = c;
        c = __funnelshift_l(b, b, 30);
        b = a;
        a = tmp;
    }
};

struct Xoroshiro128Plus {
    uint64_t s0, s1;
    __device__ __forceinline__ uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    __device__ __forceinline__ uint64_t next() {
        uint64_t r = s0 + s1, t = s1 << 23;
        s1 ^= s0;
        s0 = rotl64(s0, 24) ^ s1 ^ t;
        s1 = rotl64(s1, 37);
        return r;
    }
};
