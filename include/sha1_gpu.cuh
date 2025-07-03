#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <bit>

__device__ __host__ static inline uint32_t bswap32(uint32_t v) {
#ifdef __CUDA_ARCH__
    // GPU path – use the NVVM “byte-permute” intrinsic
    return __byte_perm(v, 0, 0x0123);
#else
#  if defined(__cpp_lib_byteswap) && (__cpp_lib_byteswap >= 202110L)
    // C++23 / libstdc++-13 / MSVC-v19.38+ in /std:c++23
    return std::byteswap(v);
#  elif defined(_MSC_VER)
    // Any MSVC version: use the intrinsic from <intrin.h>
    return _byteswap_ulong(v);
#  else
    // GCC / Clang: __builtin_bswap32 is available since forever
    return __builtin_bswap32(v);
#  endif
#endif
}

__device__ __host__ static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

static __device__ void sha1_compress(const uint8_t* msg, uint32_t H[5])
{
    uint32_t w[80];

#pragma unroll
    for (int i = 0; i < 16; ++i)
        w[i] = bswap32(*(const uint32_t*)(msg + 4 * i));

#pragma unroll
    for (int i = 16; i < 80; ++i)
        w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4];

#pragma unroll
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20)        { f = (b & c) | (~b & d);      k = 0x5A827999; }
        else if (i < 40)   { f = b ^ c ^ d;               k = 0x6ED9EBA1; }
        else if (i < 60)   { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else               { f = b ^ c ^ d;               k = 0xCA62C1D6; }

        uint32_t tmp = rotl32(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rotl32(b, 30); b = a; a = tmp;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e;
}
