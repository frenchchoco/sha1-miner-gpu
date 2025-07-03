#include "job_constants.cuh"
#include <cuda_runtime.h>

__device__ __constant__ uint8_t g_job_msg[32];
__device__ __constant__ uint32_t g_target[5];

// SHA-1 constants
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

// Configuration
#define NONCES_PER_THREAD 16
#define HASHES_PER_BATCH 4

// =================================================================
// PTX Assembly Optimized Functions
// =================================================================

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
    uint32_t result;
    asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(result) : "r"(x), "r"(n));
    return result;
}

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    uint32_t result;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(result) : "r"(x));
    return result;
}

// SHA-1 F functions
__device__ __forceinline__ uint32_t sha1_f1(uint32_t b, uint32_t c, uint32_t d) {
    uint32_t result;
    asm("{\n\t"
        ".reg .u32 t1, t2;\n\t"
        "and.b32 t1, %1, %2;\n\t"
        "not.b32 t2, %1;\n\t"
        "and.b32 t2, t2, %3;\n\t"
        "or.b32 %0, t1, t2;\n\t"
        "}"
        : "=r"(result) : "r"(b), "r"(c), "r"(d));
    return result;
}

__device__ __forceinline__ uint32_t sha1_f2(uint32_t b, uint32_t c, uint32_t d) {
    uint32_t result;
    asm("xor.b32 %0, %1, %2;\n\t"
        "xor.b32 %0, %0, %3;"
        : "=r"(result) : "r"(b), "r"(c), "r"(d));
    return result;
}

__device__ __forceinline__ uint32_t sha1_f3(uint32_t b, uint32_t c, uint32_t d) {
    uint32_t result;
    asm("{\n\t"
        ".reg .u32 t1, t2, t3;\n\t"
        "and.b32 t1, %1, %2;\n\t"
        "and.b32 t2, %1, %3;\n\t"
        "and.b32 t3, %2, %3;\n\t"
        "or.b32 %0, t1, t2;\n\t"
        "or.b32 %0, %0, t3;\n\t"
        "}"
        : "=r"(result) : "r"(b), "r"(c), "r"(d));
    return result;
}

// =================================================================
// Optimized Quad SHA-1 Transform - Correct Implementation
// =================================================================

__device__ void sha1_quad_transform(
    const uint32_t M1[8], const uint32_t M2[8], const uint32_t M3[8], const uint32_t M4[8],
    uint32_t result1[5], uint32_t result2[5],
    uint32_t result3[5], uint32_t result4[5]
) {
    // State variables for 4 parallel hashes
    uint32_t a1 = H0, b1 = H1, c1 = H2, d1 = H3, e1 = H4;
    uint32_t a2 = H0, b2 = H1, c2 = H2, d2 = H3, e2 = H4;
    uint32_t a3 = H0, b3 = H1, c3 = H2, d3 = H3, e3 = H4;
    uint32_t a4 = H0, b4 = H1, c4 = H2, d4 = H3, e4 = H4;

    // Local W arrays - each hash needs its own
    uint32_t W1[16], W2[16], W3[16], W4[16];

    // Initialize first 8 words with byte swap
#pragma unroll
    for (int i = 0; i < 8; i++) {
        W1[i] = bswap32(M1[i]);
        W2[i] = bswap32(M2[i]);
        W3[i] = bswap32(M3[i]);
        W4[i] = bswap32(M4[i]);
    }

    // Padding - correct for 32-byte message
    W1[8] = W2[8] = W3[8] = W4[8] = 0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; i++) {
        W1[i] = W2[i] = W3[i] = W4[i] = 0;
    }
    W1[15] = W2[15] = W3[15] = W4[15] = 0x00000100u; // Length = 256 bits

    // Process 80 rounds
#pragma unroll 2
    for (int i = 0; i < 80; i++) {
        uint32_t wi1, wi2, wi3, wi4;
        uint32_t f1, f2, f3, f4;
        uint32_t k;

        if (i < 16) {
            wi1 = W1[i];
            wi2 = W2[i];
            wi3 = W3[i];
            wi4 = W4[i];
        } else {
            // Compute message schedule
            asm("{\n\t"
                ".reg .u32 t1, t2, t3, t4;\n\t"
                "xor.b32 t1, %4, %5;\n\t"
                "xor.b32 t1, t1, %6;\n\t"
                "xor.b32 t1, t1, %7;\n\t"
                "shf.l.wrap.b32 %0, t1, t1, 1;\n\t"
                "xor.b32 t2, %8, %9;\n\t"
                "xor.b32 t2, t2, %10;\n\t"
                "xor.b32 t2, t2, %11;\n\t"
                "shf.l.wrap.b32 %1, t2, t2, 1;\n\t"
                "xor.b32 t3, %12, %13;\n\t"
                "xor.b32 t3, t3, %14;\n\t"
                "xor.b32 t3, t3, %15;\n\t"
                "shf.l.wrap.b32 %2, t3, t3, 1;\n\t"
                "xor.b32 t4, %16, %17;\n\t"
                "xor.b32 t4, t4, %18;\n\t"
                "xor.b32 t4, t4, %19;\n\t"
                "shf.l.wrap.b32 %3, t4, t4, 1;\n\t"
                "}"
                : "=r"(wi1), "=r"(wi2), "=r"(wi3), "=r"(wi4)
                : "r"(W1[(i - 3) & 15]), "r"(W1[(i - 8) & 15]), "r"(W1[(i - 14) & 15]), "r"(W1[(i - 16) & 15]),
                "r"(W2[(i - 3) & 15]), "r"(W2[(i - 8) & 15]), "r"(W2[(i - 14) & 15]), "r"(W2[(i - 16) & 15]),
                "r"(W3[(i - 3) & 15]), "r"(W3[(i - 8) & 15]), "r"(W3[(i - 14) & 15]), "r"(W3[(i - 16) & 15]),
                "r"(W4[(i - 3) & 15]), "r"(W4[(i - 8) & 15]), "r"(W4[(i - 14) & 15]), "r"(W4[(i - 16) & 15]));

            W1[i & 15] = wi1;
            W2[i & 15] = wi2;
            W3[i & 15] = wi3;
            W4[i & 15] = wi4;
        }

        // Select F function and K based on round
        if (i < 20) {
            f1 = sha1_f1(b1, c1, d1);
            f2 = sha1_f1(b2, c2, d2);
            f3 = sha1_f1(b3, c3, d3);
            f4 = sha1_f1(b4, c4, d4);
            k = K0;
        } else if (i < 40) {
            f1 = sha1_f2(b1, c1, d1);
            f2 = sha1_f2(b2, c2, d2);
            f3 = sha1_f2(b3, c3, d3);
            f4 = sha1_f2(b4, c4, d4);
            k = K1;
        } else if (i < 60) {
            f1 = sha1_f3(b1, c1, d1);
            f2 = sha1_f3(b2, c2, d2);
            f3 = sha1_f3(b3, c3, d3);
            f4 = sha1_f3(b4, c4, d4);
            k = K2;
        } else {
            f1 = sha1_f2(b1, c1, d1);
            f2 = sha1_f2(b2, c2, d2);
            f3 = sha1_f2(b3, c3, d3);
            f4 = sha1_f2(b4, c4, d4);
            k = K3;
        }

        // Perform round computation with interleaved operations
        uint32_t temp1, temp2, temp3, temp4;
        uint32_t new_c1, new_c2, new_c3, new_c4;

        asm("{\n\t"
            ".reg .u32 rot1, rot2, rot3, rot4;\n\t"
            "shf.l.wrap.b32 rot1, %20, %20, 5;\n\t"
            "shf.l.wrap.b32 rot2, %21, %21, 5;\n\t"
            "shf.l.wrap.b32 rot3, %22, %22, 5;\n\t"
            "shf.l.wrap.b32 rot4, %23, %23, 5;\n\t"
            "add.u32 %0, rot1, %8;\n\t"
            "add.u32 %1, rot2, %9;\n\t"
            "add.u32 %2, rot3, %10;\n\t"
            "add.u32 %3, rot4, %11;\n\t"
            "add.u32 %0, %0, %24;\n\t"
            "add.u32 %1, %1, %25;\n\t"
            "add.u32 %2, %2, %26;\n\t"
            "add.u32 %3, %3, %27;\n\t"
            "add.u32 %0, %0, %12;\n\t"
            "add.u32 %1, %1, %12;\n\t"
            "add.u32 %2, %2, %12;\n\t"
            "add.u32 %3, %3, %12;\n\t"
            "add.u32 %0, %0, %13;\n\t"
            "add.u32 %1, %1, %14;\n\t"
            "add.u32 %2, %2, %15;\n\t"
            "add.u32 %3, %3, %16;\n\t"
            // rotate(b*,30)  --  b1-b4 are the 25-28-th operands
            "shf.l.wrap.b32 %4, %25, %25, 30;\n\t"
            "shf.l.wrap.b32 %5, %26, %26, 30;\n\t"
            "shf.l.wrap.b32 %6, %27, %27, 30;\n\t"
            "shf.l.wrap.b32 %7, %28, %28, 30;\n\t"
            "}"
            : "=r"(temp1), "=r"(temp2), "=r"(temp3), "=r"(temp4),
            "=r"(new_c1), "=r"(new_c2), "=r"(new_c3), "=r"(new_c4)
            : "r"(f1), "r"(f2), "r"(f3), "r"(f4),
            "r"(k), "r"(wi1), "r"(wi2), "r"(wi3), "r"(wi4),
            "r"(a1), "r"(a2), "r"(a3), "r"(a4),
            "r"(e1), "r"(e2), "r"(e3), "r"(e4),
            "r"(b1), "r"(b2), "r"(b3), "r"(b4));

        // Update states
        e1 = d1;
        e2 = d2;
        e3 = d3;
        e4 = d4;
        d1 = c1;
        d2 = c2;
        d3 = c3;
        d4 = c4;
        c1 = new_c1;
        c2 = new_c2;
        c3 = new_c3;
        c4 = new_c4;
        b1 = a1;
        b2 = a2;
        b3 = a3;
        b4 = a4;
        a1 = temp1;
        a2 = temp2;
        a3 = temp3;
        a4 = temp4;
    }

    // Final addition
    result1[0] = a1 + H0;
    result1[1] = b1 + H1;
    result1[2] = c1 + H2;
    result1[3] = d1 + H3;
    result1[4] = e1 + H4;
    result2[0] = a2 + H0;
    result2[1] = b2 + H1;
    result2[2] = c2 + H2;
    result2[3] = d2 + H3;
    result2[4] = e2 + H4;
    result3[0] = a3 + H0;
    result3[1] = b3 + H1;
    result3[2] = c3 + H2;
    result3[3] = d3 + H3;
    result3[4] = e3 + H4;
    result4[0] = a4 + H0;
    result4[1] = b4 + H1;
    result4[2] = c4 + H2;
    result4[3] = d4 + H3;
    result4[4] = e4 + H4;
}

// =================================================================
// Vectorized Match Checking
// =================================================================

__device__ __forceinline__ uint32_t check_match_vectorized(
    const uint32_t hash[5], const uint32_t target[5]
) {
    uint32_t match;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %1, %6;\n\t"
        "@p setp.eq.u32 p, %2, %7;\n\t"
        "@p setp.eq.u32 p, %3, %8;\n\t"
        "@p setp.eq.u32 p, %4, %9;\n\t"
        "@p setp.eq.u32 p, %5, %10;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(match)
        : "r"(hash[0]), "r"(hash[1]), "r"(hash[2]), "r"(hash[3]), "r"(hash[4]),
        "r"(target[0]), "r"(target[1]), "r"(target[2]), "r"(target[3]), "r"(target[4]));
    return match;
}

// =================================================================
// Optimized Warp-Level Match Handling
// =================================================================

__device__ void handle_matches_warp_optimized(
    uint32_t matches,
    const uint32_t M1[8], const uint32_t M2[8],
    const uint32_t M3[8], const uint32_t M4[8],
    uint64_t *out_pairs,
    uint32_t *blk_ticket, // <- shared counter (pointer!)
    uint32_t *blk_base, // <- shared base index (pointer!)
    uint32_t *global_ticket) // <- original global ticket
{
    if (!matches) return;

    // Count total matches in warp
    uint32_t warp_matches = __ballot_sync(0xFFFFFFFF, matches != 0);
    if (!warp_matches) return;

    // Count matches per thread
    uint32_t thread_match_count = __popc(matches);

    // Fast warp reduction for total count
    uint32_t total_matches = thread_match_count;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total_matches += __shfl_down_sync(0xFFFFFFFF, total_matches, offset);
    }
    total_matches = __shfl_sync(0xFFFFFFFF, total_matches, 0);

    // Leader reserves space
    uint32_t base_pos; // <- re-added

    // Leader reserves (or flushes) space in the **block buffer**
    if ((threadIdx.x & 31) == 0) {
        base_pos = atomicAdd(blk_ticket, total_matches);

        // will the buffer overflow?
        if (base_pos + total_matches > 1024) {
            if (threadIdx.x == 0) {
                uint32_t old = atomicAdd(global_ticket, *blk_ticket);
                *blk_base = old; // new global base for this block
                *blk_ticket = 0; // reset local counter
                base_pos = 0; // start writing at buffer front
            }
        }
    }

    base_pos = *blk_base + __shfl_sync(0xFFFFFFFF, base_pos, 0);

    // Calculate write position with prefix sum
    uint32_t thread_offset = 0;
    uint32_t lane_id = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < lane_id; i++) {
        uint32_t other_matches = __shfl_sync(0xFFFFFFFF, thread_match_count, i);
        thread_offset += other_matches;
    }

    // Write matches
    if (out_pairs) {
        uint32_t write_pos = base_pos + thread_offset;

        if ((matches & 1) && write_pos < (1u << 20)) {
            uint64_t *dst = out_pairs + write_pos * 4;
            dst[0] = ((uint64_t) M1[1] << 32) | M1[0];
            dst[1] = ((uint64_t) M1[3] << 32) | M1[2];
            dst[2] = ((uint64_t) M1[5] << 32) | M1[4];
            dst[3] = ((uint64_t) M1[7] << 32) | M1[6];
            write_pos++;
        }

        if ((matches & 2) && write_pos < (1u << 20)) {
            uint64_t *dst = out_pairs + write_pos * 4;
            dst[0] = ((uint64_t) M2[1] << 32) | M2[0];
            dst[1] = ((uint64_t) M2[3] << 32) | M2[2];
            dst[2] = ((uint64_t) M2[5] << 32) | M2[4];
            dst[3] = ((uint64_t) M2[7] << 32) | M2[6];
            write_pos++;
        }

        if ((matches & 4) && write_pos < (1u << 20)) {
            uint64_t *dst = out_pairs + write_pos * 4;
            dst[0] = ((uint64_t) M3[1] << 32) | M3[0];
            dst[1] = ((uint64_t) M3[3] << 32) | M3[2];
            dst[2] = ((uint64_t) M3[5] << 32) | M3[4];
            dst[3] = ((uint64_t) M3[7] << 32) | M3[6];
            write_pos++;
        }

        if ((matches & 8) && write_pos < (1u << 20)) {
            uint64_t *dst = out_pairs + write_pos * 4;
            dst[0] = ((uint64_t) M4[1] << 32) | M4[0];
            dst[1] = ((uint64_t) M4[3] << 32) | M4[2];
            dst[2] = ((uint64_t) M4[5] << 32) | M4[4];
            dst[3] = ((uint64_t) M4[7] << 32) | M4[6];
        }
    }
}

// -----------------------------------------------------------------
// Dual SHA-1 transform  (2 digests per thread)
// -----------------------------------------------------------------
__device__ inline void sha1_dual_transform(
    const uint32_t M1[8], const uint32_t M2[8],
    uint32_t out1[5], uint32_t out2[5]) {
    uint32_t a1 = H0, b1 = H1, c1 = H2, d1 = H3, e1 = H4;
    uint32_t a2 = H0, b2 = H1, c2 = H2, d2 = H3, e2 = H4;

    uint32_t W1[16], W2[16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        W1[i] = bswap32(M1[i]);
        W2[i] = bswap32(M2[i]);
    }
    W1[8] = W2[8] = 0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; i++) W1[i] = W2[i] = 0;
    W1[15] = W2[15] = 0x00000100u;

#pragma unroll 2
    for (int i = 0; i < 80; i++) {
        uint32_t wi1, wi2;
        if (i < 16) {
            wi1 = W1[i];
            wi2 = W2[i];
        } else {
            wi1 = rotl32(W1[(i - 3) & 15] ^ W1[(i - 8) & 15] ^
                         W1[(i - 14) & 15] ^ W1[(i - 16) & 15], 1);
            wi2 = rotl32(W2[(i - 3) & 15] ^ W2[(i - 8) & 15] ^
                         W2[(i - 14) & 15] ^ W2[(i - 16) & 15], 1);
            W1[i & 15] = wi1;
            W2[i & 15] = wi2;
        }

        uint32_t f1, f2, k;
        if (i < 20) {
            f1 = sha1_f1(b1, c1, d1);
            f2 = sha1_f1(b2, c2, d2);
            k = K0;
        } else if (i < 40) {
            f1 = sha1_f2(b1, c1, d1);
            f2 = sha1_f2(b2, c2, d2);
            k = K1;
        } else if (i < 60) {
            f1 = sha1_f3(b1, c1, d1);
            f2 = sha1_f3(b2, c2, d2);
            k = K2;
        } else {
            f1 = sha1_f2(b1, c1, d1);
            f2 = sha1_f2(b2, c2, d2);
            k = K3;
        }

        uint32_t t1 = rotl32(a1, 5) + f1 + e1 + k + wi1;
        uint32_t t2 = rotl32(a2, 5) + f2 + e2 + k + wi2;

        e1 = d1;
        d1 = c1;
        c1 = rotl32(b1, 30);
        b1 = a1;
        a1 = t1;
        e2 = d2;
        d2 = c2;
        c2 = rotl32(b2, 30);
        b2 = a2;
        a2 = t2;
    }
    out1[0] = a1 + H0;
    out1[1] = b1 + H1;
    out1[2] = c1 + H2;
    out1[3] = d1 + H3;
    out1[4] = e1 + H4;
    out2[0] = a2 + H0;
    out2[1] = b2 + H1;
    out2[2] = c2 + H2;
    out2[3] = d2 + H3;
    out2[4] = e2 + H4;
}

// =================================================================
// Main Optimized Kernel with Quad Processing
// =================================================================

extern "C" __global__ __launch_bounds__(256, 2)
void sha1_collision_kernel_ultra(
    uint8_t * __restrict__ out_msgs,
    uint64_t * __restrict__ out_pairs,
    uint32_t * __restrict__ ticket,
    uint64_t seed
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ------------------------------------------------------------------
    // 1.  per-block broadcast of the target (already in your code)
    // ------------------------------------------------------------------
    __shared__ uint32_t shared_target[5];
    if (threadIdx.x < 5)
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    __syncthreads();

    // ------------------------------------------------------------------
    // 2.  NEW: per-block ticket buffer & base pointer
    //     (lifetime: entire kernel launch, scope: one thread-block)
    // ------------------------------------------------------------------
    __shared__ uint32_t blk_ticket; // how many matches the block has queued
    __shared__ uint32_t blk_base; // base index in the global out_pairs array

    if (threadIdx.x == 0) {
        // initialise once per block
        blk_ticket = 0;
        blk_base = 0;
    }
    __syncthreads();

    // Load base message into registers
    uint32_t M_base[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M_base[i] = ((const uint32_t *) g_job_msg)[i];
    }

    // Process nonces in batches of 4
#pragma unroll 2
    for (uint32_t iter = 0; iter < NONCES_PER_THREAD; iter += HASHES_PER_BATCH) {
        // Calculate 4 nonces
        uint64_t base_nonce = seed + (uint64_t) tid * NONCES_PER_THREAD + iter;
        uint64_t nonce1 = base_nonce;
        uint64_t nonce2 = base_nonce + 1;
        uint64_t nonce3 = base_nonce + 2;
        uint64_t nonce4 = base_nonce + 3;

        // Prepare 4 messages
        uint32_t M1[8], M2[8], M3[8], M4[8];

#pragma unroll
        for (int i = 0; i < 6; i++) {
            M1[i] = M2[i] = M3[i] = M4[i] = M_base[i];
        }

        // Apply nonces
        M1[6] = M_base[6] ^ (uint32_t) (nonce1 & 0xFFFFFFFF);
        M1[7] = M_base[7] ^ (uint32_t) (nonce1 >> 32);
        M2[6] = M_base[6] ^ (uint32_t) (nonce2 & 0xFFFFFFFF);
        M2[7] = M_base[7] ^ (uint32_t) (nonce2 >> 32);
        M3[6] = M_base[6] ^ (uint32_t) (nonce3 & 0xFFFFFFFF);
        M3[7] = M_base[7] ^ (uint32_t) (nonce3 >> 32);
        M4[6] = M_base[6] ^ (uint32_t) (nonce4 & 0xFFFFFFFF);
        M4[7] = M_base[7] ^ (uint32_t) (nonce4 >> 32);

        // Compute 4 SHA-1 hashes in parallel
        uint32_t h1[5], h2[5], h3[5], h4[5];
        sha1_dual_transform(M1, M2, h1, h2);
        sha1_dual_transform(M3, M4, h3, h4);

        // Check all 4 matches
        uint32_t matches = 0;
        matches |= check_match_vectorized(h1, shared_target) << 0;
        matches |= check_match_vectorized(h2, shared_target) << 1;
        matches |= check_match_vectorized(h3, shared_target) << 2;
        matches |= check_match_vectorized(h4, shared_target) << 3;

        // Handle matches with warp-level coordination
        if (__any_sync(0xFFFFFFFF, matches)) {
            handle_matches_warp_optimized(matches, M1, M2, M3, M4,
                                          out_pairs,
                                          &blk_ticket, &blk_base, // NEW
                                          ticket); // global
        }
    }

    if (threadIdx.x == 0 && blk_ticket) {
        atomicAdd(ticket, blk_ticket); // reserve space once
        /* blk_base already updated inside handle_matches_warp_optimized */
    }
}
