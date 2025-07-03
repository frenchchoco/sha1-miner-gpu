#include "sha1_gpu.cuh"
#include "job_constants.cuh"
#include <cuda_runtime.h>

/* single device-side constants (populated by upload_new_job) */
__device__ __constant__ uint8_t g_job_msg[32];
__device__ __constant__ uint32_t g_target[5];

/* 20-byte digest stored in shared memory */
struct Digest {
    uint32_t h[5];
}; // 5×4 B = 20 B

extern "C" __global__
void sha1_double_kernel(uint8_t * __restrict__ out_msgs,
                        uint64_t * __restrict__ out_pairs,
                        uint32_t * __restrict__ ticket,
                        uint64_t seed) {
    /* ── 1. load cached 32-B message, mix per-thread nonce ───────────── */
    uint32_t M[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        M[i] = reinterpret_cast<const uint32_t *>(g_job_msg)[i];

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    M[7] ^= tid ^ static_cast<uint32_t>(seed);

    /* ── 2. double-SHA-1 — identical to previous version ─────────────── */
    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = bswap32(M[i]);
    w[8] = 0x80000000U;
#pragma unroll
    for (int i = 9; i < 15; ++i) w[i] = 0;
    w[15] = 0x00000100U;

    Sha1Ctx ctx{0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};

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
                                w[(i - 3) & 15], w[(i - 8) & 15],
                                w[(i - 14) & 15], w[(i - 16) & 15]);
        ctx.round(f, k, wi);
    }
    uint32_t H0 = 0x67452301 + ctx.a,
            H1 = 0xEFCDAB89 + ctx.b,
            H2 = 0x98BADCFE + ctx.c,
            H3 = 0x10325476 + ctx.d,
            H4 = 0xC3D2E1F0 + ctx.e;

    /* second compression on 20-B digest (unchanged) */
    w[0] = H0;
    w[1] = H1;
    w[2] = H2;
    w[3] = H3;
    w[4] = H4;
    w[5] = 0x80000000U;
#pragma unroll
    for (int i = 6; i < 15; ++i) w[i] = 0;
    w[15] = 0x000000A0U;

    ctx = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
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
                                w[(i - 3) & 15], w[(i - 8) & 15],
                                w[(i - 14) & 15], w[(i - 16) & 15]);
        ctx.round(f, k, wi);
    }
    H0 += ctx.a;
    H1 += ctx.b;
    H2 += ctx.c;
    H3 += ctx.d;
    H4 += ctx.e;

    /* ── 3. 160-bit collision filter in 64 buckets ──────────────────── */
    __shared__ Digest buckets[64]; // 1 280 B

    Digest mine = {H0, H1, H2, H3, H4};
    const unsigned int slot = threadIdx.x & 63u;

    /* first writer initialises slot */
    uint32_t prev0 = atomicCAS(&buckets[slot].h[0], 0u, mine.h[0]);

    bool real_hit = false;
    if (prev0 == 0u) {
#pragma unroll
        for (int i = 1; i < 5; ++i)
            buckets[slot].h[i] = mine.h[i];
    } else {
        real_hit = (prev0 == mine.h[0]) &&
                   (buckets[slot].h[1] == mine.h[1]) &&
                   (buckets[slot].h[2] == mine.h[2]) &&
                   (buckets[slot].h[3] == mine.h[3]) &&
                   (buckets[slot].h[4] == mine.h[4]);
    }

    /* ── 4. store candidate when real_hit == true ───────────────────── */
    if (real_hit) {
        uint32_t pos = ticket ? atomicAdd(ticket, 1) : 0;
        if (out_pairs && pos < (1u << 20)) {
            uint64_t *dst = out_pairs + pos * 4;
            dst[0] = (static_cast<uint64_t>(M[1]) << 32) | M[0];
            dst[1] = (static_cast<uint64_t>(M[3]) << 32) | M[2];
            dst[2] = (static_cast<uint64_t>(M[5]) << 32) | M[4];
            dst[3] = (static_cast<uint64_t>(M[7]) << 32) | M[6];
        }
    }

#if 0   // optional debug dump
    if (out_msgs) {
        uint8_t* p = out_msgs + tid * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            reinterpret_cast<uint32_t*>(p)[i] = M[i];
    }
#endif
}
