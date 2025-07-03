#include "sha1_gpu.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__
void sha1_double_kernel(uint8_t * __restrict__ out_msgs,
                        uint64_t * __restrict__ out_pairs,
                        uint32_t * __restrict__ ticket,
                        uint64_t seed) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* -------- generate 32-byte random message ------------------------- */
    Xoroshiro128Plus prng{seed ^ tid, seed + tid * 0x9E3779B97F4A7C15ULL};
    uint32_t M0 = prng.next();
    uint32_t M1 = prng.next();
    uint32_t M2 = prng.next();
    uint32_t M3 = prng.next();

    /* -------- first SHA-1 compression (single padded block) ----------- */
    uint32_t w[16];
    w[0] = bswap32(M0);
    w[1] = bswap32(M1);
    w[2] = bswap32(M2);
    w[3] = bswap32(M3);
    w[4] = 0x80000000U;
#pragma unroll
    for (int i = 5; i < 15; ++i) w[i] = 0;
    w[15] = 0x00000100U; // length 256 bits

    Sha1Ctx ctx{
        0x67452301, 0xEFCDAB89, 0x98BADCFE,
        0x10325476, 0xC3D2E1F0
    };

#pragma unroll 80
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) {
            f = (ctx.b & ctx.c) | (~ctx.b & ctx.d);
            k = 0x5A827999;
        } else if (i < 40) {
            f = ctx.b ^ ctx.c ^ ctx.d;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (ctx.b & ctx.c) | (ctx.b & ctx.d) | (ctx.c & ctx.d);
            k = 0x8F1BBCDC;
        } else {
            f = ctx.b ^ ctx.c ^ ctx.d;
            k = 0xCA62C1D6;
        }

        uint32_t wi = (i < 16)
                          ? w[i & 15]
                          : w[i & 15] = schedule_word(
                                w[(i - 3) & 15],
                                w[(i - 8) & 15],
                                w[(i - 14) & 15],
                                w[(i - 16) & 15]);
        ctx.round(f, k, wi);
    }

    uint32_t H0 = 0x67452301 + ctx.a;
    uint32_t H1 = 0xEFCDAB89 + ctx.b;
    uint32_t H2 = 0x98BADCFE + ctx.c;
    uint32_t H3 = 0x10325476 + ctx.d;
    uint32_t H4 = 0xC3D2E1F0 + ctx.e;

    /* -------- second compression on the 20-byte digest --------------- */
    w[0] = H0;
    w[1] = H1;
    w[2] = H2;
    w[3] = H3;
    w[4] = H4;
    w[5] = 0x80000000U;
#pragma unroll
    for (int i = 6; i < 15; ++i) w[i] = 0;
    w[15] = 0x000000A0U; // length 160 bits

    ctx = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE,
        0x10325476, 0xC3D2E1F0
    };

#pragma unroll 80
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) {
            f = (ctx.b & ctx.c) | (~ctx.b & ctx.d);
            k = 0x5A827999;
        } else if (i < 40) {
            f = ctx.b ^ ctx.c ^ ctx.d;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (ctx.b & ctx.c) | (ctx.b & ctx.d) | (ctx.c & ctx.d);
            k = 0x8F1BBCDC;
        } else {
            f = ctx.b ^ ctx.c ^ ctx.d;
            k = 0xCA62C1D6;
        }

        uint32_t wi = (i < 16)
                          ? w[i & 15]
                          : w[i & 15] = schedule_word(
                                w[(i - 3) & 15],
                                w[(i - 8) & 15],
                                w[(i - 14) & 15],
                                w[(i - 16) & 15]);
        ctx.round(f, k, wi);
    }

    H0 += ctx.a;
    H1 += ctx.b;
    H2 += ctx.c;
    H3 += ctx.d;
    H4 += ctx.e;

    /* -------- 64-bit tag & collision filter --------------------------- */
    uint64_t tag = (uint64_t(H0) << 32) | H1;
    __shared__ uint64_t table[64];
    int slot = tag & 63;
    uint64_t old = atomicCAS(table + slot, 0ULL, tag);

    if (old == tag && old) {
        uint32_t pos = atomicAdd(ticket, 1);
        if (pos < (1u << 20)) {
            uint64_t *dst = out_pairs + pos * 6;
            dst[0] = M0;
            dst[1] = M1;
            dst[2] = M2;
            dst[3] = M3;
            dst[4] = old;
            dst[5] = tag;
        }
    }

#if 0   /* Optional: copy message back for inspection */
    uint8_t* p = out_msgs + tid * 32;
    reinterpret_cast<uint32_t*>(p)[0] = M0;
    reinterpret_cast<uint32_t*>(p)[1] = M1;
    reinterpret_cast<uint32_t*>(p)[2] = M2;
    reinterpret_cast<uint32_t*>(p)[3] = M3;
#endif
}
