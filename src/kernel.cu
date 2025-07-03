#include "job_constants.cuh"
#include <cuda_runtime.h>

__device__ __constant__ uint8_t g_job_msg[32];
__device__ __constant__ uint32_t g_target[5];

// SHA-1 constants in registers for faster access
#define K0 0x5A827999u
#define K1 0x6ED9EBA1u
#define K2 0x8F1BBCDCu
#define K3 0xCA62C1D6u

// Initial SHA-1 values
#define H0 0x67452301u
#define H1 0xEFCDAB89u
#define H2 0x98BADCFEu
#define H3 0x10325476u
#define H4 0xC3D2E1F0u

// =================================================================
// PTX Assembly Optimized Functions
// =================================================================

// Optimized rotate left using PTX
__device__ __forceinline__ uint32_t rotl32_asm(uint32_t x, uint32_t n) {
    uint32_t result;
    asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(result) : "r"(x), "r"(n));
    return result;
}

// Optimized byte swap using PTX
__device__ __forceinline__ uint32_t bswap32_asm(uint32_t x) {
    uint32_t result;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(result) : "r"(x));
    return result;
}

// SHA-1 F function for rounds 0-19 and 40-59 using PTX
__device__ __forceinline__ uint32_t sha1_f1_asm(uint32_t b, uint32_t c, uint32_t d) {
    uint32_t result;
    asm("{\n\t"
        ".reg .u32 t1, t2;\n\t"
        "and.b32 t1, %1, %2;\n\t" // t1 = b & c
        "not.b32 t2, %1;\n\t" // t2 = ~b
        "and.b32 t2, t2, %3;\n\t" // t2 = ~b & d
        "or.b32 %0, t1, t2;\n\t" // result = (b & c) | (~b & d)
        "}"
        : "=r"(result) : "r"(b), "r"(c), "r"(d));
    return result;
}

// SHA-1 F function for rounds 20-39 and 60-79 (simple XOR)
__device__ __forceinline__ uint32_t sha1_f2_asm(uint32_t b, uint32_t c, uint32_t d) {
    uint32_t result;
    asm("xor.b32 %0, %1, %2;\n\t"
        "xor.b32 %0, %0, %3;"
        : "=r"(result) : "r"(b), "r"(c), "r"(d));
    return result;
}

// SHA-1 F function for rounds 40-59 (majority) using PTX
__device__ __forceinline__ uint32_t sha1_f3_asm(uint32_t b, uint32_t c, uint32_t d) {
    uint32_t result;
    asm("{\n\t"
        ".reg .u32 t1, t2, t3;\n\t"
        "and.b32 t1, %1, %2;\n\t" // t1 = b & c
        "and.b32 t2, %1, %3;\n\t" // t2 = b & d
        "and.b32 t3, %2, %3;\n\t" // t3 = c & d
        "or.b32 %0, t1, t2;\n\t" // result = (b & c) | (b & d)
        "or.b32 %0, %0, t3;\n\t" // result |= (c & d)
        "}"
        : "=r"(result) : "r"(b), "r"(c), "r"(d));
    return result;
}

// Optimized SHA-1 round computation using PTX
__device__ __forceinline__ void sha1_round_asm(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t &e,
    uint32_t w, uint32_t k, uint32_t f
) {
    uint32_t temp;
    asm("{\n\t"
        ".reg .u32 rot_a, rot_b;\n\t"
        "shf.l.wrap.b32 rot_a, %5, %5, 5;\n\t" // rot_a = rotl32(a, 5)
        "add.u32 %0, rot_a, %4;\n\t" // temp = rot_a + f
        "add.u32 %0, %0, %6;\n\t" // temp += e
        "add.u32 %0, %0, %7;\n\t" // temp += k
        "add.u32 %0, %0, %8;\n\t" // temp += w
        "shf.l.wrap.b32 %2, %3, %3, 30;\n\t" // c = rotl32(b, 30)
        "}"
        : "=r"(temp), "+r"(e), "+r"(c)
        : "r"(b), "r"(f), "r"(a), "r"(e), "r"(k), "r"(w)
    );
    e = d;
    d = c;
    c = b;
    b = a;
    a = temp;
}

// =================================================================
// Ultra-Optimized SHA-1 Implementation with PTX
// =================================================================

__device__ __noinline__ void sha1_transform_asm(const uint32_t M[8], uint32_t result[5]) {
    // Working variables
    uint32_t a = H0;
    uint32_t b = H1;
    uint32_t c = H2;
    uint32_t d = H3;
    uint32_t e = H4;

    // Message schedule
    uint32_t W[16];

    // Initialize first 8 words with byte swap
#pragma unroll
    for (int i = 0; i < 8; i++) {
        W[i] = bswap32_asm(M[i]);
    }

    // Padding
    W[8] = 0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; i++) {
        W[i] = 0;
    }
    W[15] = 0x00000100u;

    // Rounds 0-15
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t f = sha1_f1_asm(b, c, d);
        sha1_round_asm(a, b, c, d, e, W[i], K0, f);
    }

    // Rounds 16-19
#pragma unroll
    for (int i = 16; i < 20; i++) {
        uint32_t wi;
        asm("xor.b32 %0, %1, %2;\n\t"
            "xor.b32 %0, %0, %3;\n\t"
            "xor.b32 %0, %0, %4;\n\t"
            "shf.l.wrap.b32 %0, %0, %0, 1;"
            : "=r"(wi)
            : "r"(W[(i + 13) & 15]), "r"(W[(i + 8) & 15]),
            "r"(W[(i + 2) & 15]), "r"(W[i & 15]));
        W[i & 15] = wi;

        uint32_t f = sha1_f1_asm(b, c, d);
        sha1_round_asm(a, b, c, d, e, wi, K0, f);
    }

    // Rounds 20-39
#pragma unroll
    for (int i = 20; i < 40; i++) {
        uint32_t wi;
        asm("xor.b32 %0, %1, %2;\n\t"
            "xor.b32 %0, %0, %3;\n\t"
            "xor.b32 %0, %0, %4;\n\t"
            "shf.l.wrap.b32 %0, %0, %0, 1;"
            : "=r"(wi)
            : "r"(W[(i + 13) & 15]), "r"(W[(i + 8) & 15]),
            "r"(W[(i + 2) & 15]), "r"(W[i & 15]));
        W[i & 15] = wi;

        uint32_t f = sha1_f2_asm(b, c, d);
        sha1_round_asm(a, b, c, d, e, wi, K1, f);
    }

    // Rounds 40-59
#pragma unroll
    for (int i = 40; i < 60; i++) {
        uint32_t wi;
        asm("xor.b32 %0, %1, %2;\n\t"
            "xor.b32 %0, %0, %3;\n\t"
            "xor.b32 %0, %0, %4;\n\t"
            "shf.l.wrap.b32 %0, %0, %0, 1;"
            : "=r"(wi)
            : "r"(W[(i + 13) & 15]), "r"(W[(i + 8) & 15]),
            "r"(W[(i + 2) & 15]), "r"(W[i & 15]));
        W[i & 15] = wi;

        uint32_t f = sha1_f3_asm(b, c, d);
        sha1_round_asm(a, b, c, d, e, wi, K2, f);
    }

    // Rounds 60-79
#pragma unroll
    for (int i = 60; i < 80; i++) {
        uint32_t wi;
        asm("xor.b32 %0, %1, %2;\n\t"
            "xor.b32 %0, %0, %3;\n\t"
            "xor.b32 %0, %0, %4;\n\t"
            "shf.l.wrap.b32 %0, %0, %0, 1;"
            : "=r"(wi)
            : "r"(W[(i + 13) & 15]), "r"(W[(i + 8) & 15]),
            "r"(W[(i + 2) & 15]), "r"(W[i & 15]));
        W[i & 15] = wi;

        uint32_t f = sha1_f2_asm(b, c, d);
        sha1_round_asm(a, b, c, d, e, wi, K3, f);
    }

    // Final addition using PTX
    asm("add.u32 %0, %0, %5;\n\t"
        "add.u32 %1, %1, %6;\n\t"
        "add.u32 %2, %2, %7;\n\t"
        "add.u32 %3, %3, %8;\n\t"
        "add.u32 %4, %4, %9;"
        : "+r"(a), "+r"(b), "+r"(c), "+r"(d), "+r"(e)
        : "r"(H0), "r"(H1), "r"(H2), "r"(H3), "r"(H4));

    result[0] = a;
    result[1] = b;
    result[2] = c;
    result[3] = d;
    result[4] = e;
}

// =================================================================
// Main Kernel with PTX Optimizations
// =================================================================

extern "C" __global__ __launch_bounds__(256, 4)
void sha1_collision_kernel(
    uint8_t * __restrict__ out_msgs,
    uint64_t * __restrict__ out_pairs,
    uint32_t * __restrict__ ticket,
    uint64_t seed
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    // Load target into shared memory
    __shared__ uint32_t shared_target[5];
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    }
    __syncthreads();

    // Load base message into registers
    uint32_t M_base[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M_base[i] = ((const uint32_t *) g_job_msg)[i];
    }

    // Process 8 nonces per thread
#pragma unroll 8
    for (uint32_t n = 0; n < 8; n++) {
        uint64_t nonce = seed + tid + n * total_threads;

        // Prepare message
        uint32_t M[8];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            M[i] = M_base[i];
        }

        // Apply nonce using PTX
        asm("mov.b32 %0, %2;\n\t"
            "xor.b32 %0, %0, %3;\n\t"
            "mov.b32 %1, %4;\n\t"
            "xor.b32 %1, %1, %5;"
            : "=r"(M[6]), "=r"(M[7])
            : "r"(M_base[6]), "r"((uint32_t) (nonce & 0xFFFFFFFF)),
            "r"(M_base[7]), "r"((uint32_t) (nonce >> 32)));

        // Compute SHA-1
        uint32_t hash[5];
        sha1_transform_asm(M, hash);

        // Check match using PTX comparison
        uint32_t match;
        asm("{\n\t"
            ".reg .pred p0, p1, p2, p3, p4;\n\t"
            "setp.eq.u32 p0, %1, %6;\n\t"
            "setp.eq.u32 p1, %2, %7;\n\t"
            "setp.eq.u32 p2, %3, %8;\n\t"
            "setp.eq.u32 p3, %4, %9;\n\t"
            "setp.eq.u32 p4, %5, %10;\n\t"
            "and.pred p0, p0, p1;\n\t"
            "and.pred p0, p0, p2;\n\t"
            "and.pred p0, p0, p3;\n\t"
            "and.pred p0, p0, p4;\n\t"
            "selp.u32 %0, 1, 0, p0;\n\t"
            "}"
            : "=r"(match)
            : "r"(hash[0]), "r"(hash[1]), "r"(hash[2]), "r"(hash[3]), "r"(hash[4]),
            "r"(shared_target[0]), "r"(shared_target[1]), "r"(shared_target[2]),
            "r"(shared_target[3]), "r"(shared_target[4]));

        if (match) {
            uint32_t pos = atomicAdd(ticket, 1);
            if (out_pairs && pos < (1u << 20)) {
                uint64_t *dst = out_pairs + pos * 4;
                dst[0] = ((uint64_t) M[1] << 32) | M[0];
                dst[1] = ((uint64_t) M[3] << 32) | M[2];
                dst[2] = ((uint64_t) M[5] << 32) | M[4];
                dst[3] = ((uint64_t) M[7] << 32) | M[6];
            }
        }
    }
}

// =================================================================
// Ultra-Performance Kernel with Aggressive PTX Optimization
// =================================================================

extern "C" __global__ __launch_bounds__(128, 8)
void sha1_collision_kernel_extreme_asm(
    uint8_t * __restrict__ out_msgs,
    uint64_t * __restrict__ out_pairs,
    uint32_t * __restrict__ ticket,
    uint64_t seed
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ uint32_t shared_target[5];
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    }
    __syncthreads();

    // Load base message
    uint32_t M_base[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M_base[i] = ((const uint32_t *) g_job_msg)[i];
    }

    // Process 4 iterations of 2 parallel hashes
#pragma unroll 4
    for (uint32_t iter = 0; iter < 4; iter++) {
        uint64_t nonce1 = seed + (tid * 8) + (iter * 2);
        uint64_t nonce2 = nonce1 + 1;

        // Prepare messages with PTX
        uint32_t M1[8], M2[8];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            M1[i] = M_base[i];
            M2[i] = M_base[i];
        }

        // Apply nonces
        asm("xor.b32 %0, %4, %5;\n\t"
            "xor.b32 %1, %6, %7;\n\t"
            "xor.b32 %2, %4, %8;\n\t"
            "xor.b32 %3, %6, %9;"
            : "=r"(M1[6]), "=r"(M1[7]), "=r"(M2[6]), "=r"(M2[7])
            : "r"(M_base[6]), "r"((uint32_t) (nonce1 & 0xFFFFFFFF)),
            "r"(M_base[7]), "r"((uint32_t) (nonce1 >> 32)),
            "r"((uint32_t) (nonce2 & 0xFFFFFFFF)), "r"((uint32_t) (nonce2 >> 32)));

        // Compute both hashes
        uint32_t hash1[5], hash2[5];
        sha1_transform_asm(M1, hash1);
        sha1_transform_asm(M2, hash2);

        // Check both results with PTX
        uint32_t match1, match2;

        // Check hash1
        asm("{\n\t"
            ".reg .pred p;\n\t"
            "setp.eq.u32 p, %1, %6;\n\t"
            "@p setp.eq.u32 p, %2, %7;\n\t"
            "@p setp.eq.u32 p, %3, %8;\n\t"
            "@p setp.eq.u32 p, %4, %9;\n\t"
            "@p setp.eq.u32 p, %5, %10;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(match1)
            : "r"(hash1[0]), "r"(hash1[1]), "r"(hash1[2]), "r"(hash1[3]), "r"(hash1[4]),
            "r"(shared_target[0]), "r"(shared_target[1]), "r"(shared_target[2]),
            "r"(shared_target[3]), "r"(shared_target[4]));

        // Check hash2
        asm("{\n\t"
            ".reg .pred p;\n\t"
            "setp.eq.u32 p, %1, %6;\n\t"
            "@p setp.eq.u32 p, %2, %7;\n\t"
            "@p setp.eq.u32 p, %3, %8;\n\t"
            "@p setp.eq.u32 p, %4, %9;\n\t"
            "@p setp.eq.u32 p, %5, %10;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(match2)
            : "r"(hash2[0]), "r"(hash2[1]), "r"(hash2[2]), "r"(hash2[3]), "r"(hash2[4]),
            "r"(shared_target[0]), "r"(shared_target[1]), "r"(shared_target[2]),
            "r"(shared_target[3]), "r"(shared_target[4]));

        if (match1) {
            uint32_t pos = atomicAdd(ticket, 1);
            if (out_pairs && pos < (1u << 20)) {
                uint64_t *dst = out_pairs + pos * 4;
                dst[0] = ((uint64_t) M1[1] << 32) | M1[0];
                dst[1] = ((uint64_t) M1[3] << 32) | M1[2];
                dst[2] = ((uint64_t) M1[5] << 32) | M1[4];
                dst[3] = ((uint64_t) M1[7] << 32) | M1[6];
            }
        }

        if (match2) {
            uint32_t pos = atomicAdd(ticket, 1);
            if (out_pairs && pos < (1u << 20)) {
                uint64_t *dst = out_pairs + pos * 4;
                dst[0] = ((uint64_t) M2[1] << 32) | M2[0];
                dst[1] = ((uint64_t) M2[3] << 32) | M2[2];
                dst[2] = ((uint64_t) M2[5] << 32) | M2[4];
                dst[3] = ((uint64_t) M2[7] << 32) | M2[6];
            }
        }
    }
}
