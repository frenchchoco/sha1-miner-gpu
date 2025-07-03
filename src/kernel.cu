// ─────────────────────────────────── src/kernel.cu ──────────────────────────
#include "sha1_gpu.cuh"
#include "job_constants.cuh"
#include <cuda_runtime.h>

/* device-side constants set by upload_new_job() */
__device__ __constant__ uint8_t g_job_msg[32];
__device__ __constant__ uint32_t g_target[5];

/* ───────────────────────────── kernel ───────────────────────────────────── */
extern "C" __global__
void sha1_double_kernel(uint8_t * __restrict__ out_msgs, // may be nullptr
                        uint64_t * __restrict__ out_pairs, // ring buffer
                        uint32_t * __restrict__ ticket, // candidate counter
                        uint64_t seed) // per-launch seed
{
    /* 1 ─ load the 32-byte cached message and mix a per-thread nonce */
    uint32_t M[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        M[i] = reinterpret_cast<const uint32_t *>(g_job_msg)[i];

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    M[7] ^= tid ^ static_cast<uint32_t>(seed);

    /* 2 ─ first SHA-1 compression (single 512-bit block) */
    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = bswap32(M[i]);
    w[8] = 0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; ++i) w[i] = 0;
    w[15] = 0x00000100u; // 256 bits

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

    /* 3 ─ second compression on the 20-byte digest */
    w[0] = H0;
    w[1] = H1;
    w[2] = H2;
    w[3] = H3;
    w[4] = H4;
    w[5] = 0x80000000u;
#pragma unroll
    for (int i = 6; i < 15; ++i) w[i] = 0;
    w[15] = 0x000000A0u; // 160 bits

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

    /* 4 ─ accept only if the full 160-bit digest matches g_target */
    if (H0 != g_target[0] || H1 != g_target[1] ||
        H2 != g_target[2] || H3 != g_target[3] || H4 != g_target[4])
        return; // not a real collision

    /* 5 ─ store the successful 32-byte message (if buffers provided) */
    uint32_t pos = ticket ? atomicAdd(ticket, 1) : 0;
    if (out_pairs && pos < (1u << 20)) {
        uint64_t *dst = out_pairs + pos * 4;
        dst[0] = (static_cast<uint64_t>(M[1]) << 32) | M[0];
        dst[1] = (static_cast<uint64_t>(M[3]) << 32) | M[2];
        dst[2] = (static_cast<uint64_t>(M[5]) << 32) | M[4];
        dst[3] = (static_cast<uint64_t>(M[7]) << 32) | M[6];
    }

#if 0   // optional: copy raw message back for debugging
    if (out_msgs) {
        uint32_t *p = reinterpret_cast<uint32_t*>(out_msgs + tid * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) p[i] = M[i];
    }
#endif
}
