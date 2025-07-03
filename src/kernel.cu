#include "job_constants.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __constant__ uint8_t g_job_msg[32];
__device__ __constant__ uint32_t g_target[5];

// SHA-1 round constants
__device__ __constant__ uint32_t K[4] = {
    0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xCA62C1D6u
};

// =================================================================
// Ultra-Optimized Rotation and Helper Functions
// =================================================================

// Use PTX for optimal rotation on newer architectures
__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
#if __CUDA_ARCH__ >= 350
    uint32_t result;
    asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(result) : "r"(x), "r"(n));
    return result;
#else
    return (x << n) | (x >> (32 - n));
#endif
}

// Byte swap for endianness
__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

// =================================================================
// Message Schedule Optimization
// =================================================================

// Precompute message schedule for rounds 0-15
__device__ __forceinline__ void prepare_message_schedule(
    const uint32_t M[8], uint32_t W[16]
) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        W[i] = bswap32(M[i]);
    }
    W[8] = 0x80000000u; // Padding bit
#pragma unroll
    for (int i = 9; i < 15; i++) {
        W[i] = 0;
    }
    W[15] = 0x00000100u; // Length = 256 bits
}

// Compute message schedule for rounds 16-79 on the fly
__device__ __forceinline__ uint32_t compute_w(
    uint32_t w[16], int round
) {
    int idx = round & 15;
    uint32_t value = w[idx] ^ w[(idx + 2) & 15] ^ w[(idx + 8) & 15] ^ w[(idx + 13) & 15];
    value = rotl32(value, 1);
    w[idx] = value;
    return value;
}

// =================================================================
// SHA-1 Core with Maximum Optimization
// =================================================================

__device__ __forceinline__ void sha1_transform_optimized(
    const uint32_t W[16], uint32_t state[5]
) {
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = W[i];
    }

    // Rounds 0-19: f = (b & c) | (~b & d)
#pragma unroll
    for (int i = 0; i < 20; i++) {
        uint32_t wi = (i < 16) ? w[i] : compute_w(w, i);
        uint32_t f = (b & c) | (~b & d);
        uint32_t temp = rotl32(a, 5) + f + e + K[0] + wi;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 20-39: f = b ^ c ^ d
#pragma unroll
    for (int i = 20; i < 40; i++) {
        uint32_t wi = compute_w(w, i);
        uint32_t f = b ^ c ^ d;
        uint32_t temp = rotl32(a, 5) + f + e + K[1] + wi;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 40-59: f = (b & c) | (b & d) | (c & d)
#pragma unroll
    for (int i = 40; i < 60; i++) {
        uint32_t wi = compute_w(w, i);
        uint32_t f = (b & c) | (b & d) | (c & d);
        uint32_t temp = rotl32(a, 5) + f + e + K[2] + wi;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 60-79: f = b ^ c ^ d
#pragma unroll
    for (int i = 60; i < 80; i++) {
        uint32_t wi = compute_w(w, i);
        uint32_t f = b ^ c ^ d;
        uint32_t temp = rotl32(a, 5) + f + e + K[3] + wi;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

// =================================================================
// Main Collision Mining Kernel - Single SHA-1
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

    // Shared memory for fast target access
    __shared__ uint32_t shared_target[5];
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    }
    __syncthreads();

    // Load base message into registers
    uint32_t M[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M[i] = ((const uint32_t *) g_job_msg)[i];
    }

    // Each thread processes multiple nonces for better efficiency
    const uint32_t NONCES_PER_THREAD = 4;

    for (uint32_t n = 0; n < NONCES_PER_THREAD; n++) {
        // Generate unique nonce with good distribution
        uint64_t nonce = seed + tid + n * total_threads;

        // Apply nonce to message (last 8 bytes)
        uint32_t M_work[8];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            M_work[i] = M[i];
        }

        // Mix nonce into last 64 bits with additional entropy
        M_work[6] = M[6] ^ (uint32_t) (nonce & 0xFFFFFFFF) ^ __funnelshift_l(tid, n, 13);
        M_work[7] = M[7] ^ (uint32_t) (nonce >> 32) ^ __brev(tid + n);

        // Prepare message schedule
        uint32_t W[16];
        prepare_message_schedule(M_work, W);

        // Initialize SHA-1 state
        uint32_t state[5] = {
            0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u
        };

        // Perform SHA-1 transform
        sha1_transform_optimized(W, state);

        // Compare with target using warp-level voting for efficiency
        bool match = true;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            if (state[i] != shared_target[i]) {
                match = false;
                break;
            }
        }

        // If we found a match, save it
        if (match) {
            uint32_t pos = atomicAdd(ticket, 1);
            if (out_pairs && pos < (1u << 20)) {
                uint64_t *dst = out_pairs + pos * 4;
                // Store the message that produced the collision
                dst[0] = ((uint64_t) M_work[1] << 32) | M_work[0];
                dst[1] = ((uint64_t) M_work[3] << 32) | M_work[2];
                dst[2] = ((uint64_t) M_work[5] << 32) | M_work[4];
                dst[3] = ((uint64_t) M_work[7] << 32) | M_work[6];
            }
        }
    }
}

// =================================================================
// Advanced Multi-Stream Kernel for Multi-GPU
// =================================================================

extern "C" __global__ __launch_bounds__(256, 4)
void sha1_collision_kernel_multistream(
    uint8_t * __restrict__ out_msgs,
    uint64_t * __restrict__ out_pairs,
    uint32_t * __restrict__ ticket,
    uint64_t seed,
    uint32_t stream_id,
    uint32_t total_streams
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_tid = tid + stream_id * gridDim.x * blockDim.x;
    const uint32_t stride = gridDim.x * blockDim.x * total_streams;

    // Shared memory setup
    __shared__ uint32_t shared_target[5];
    __shared__ uint32_t warp_found[8];

    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    }
    if (threadIdx.x < 8) {
        warp_found[threadIdx.x] = 0;
    }
    __syncthreads();

    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane_id = threadIdx.x & 31;

    // Load base message
    uint32_t M[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M[i] = ((const uint32_t *) g_job_msg)[i];
    }

    // Process multiple nonces per thread
    const uint32_t NONCES_PER_THREAD = 8;

    for (uint32_t n = 0; n < NONCES_PER_THREAD; n++) {
        uint64_t nonce = seed + global_tid + n * stride;

        // Apply nonce with enhanced mixing
        uint32_t M_work[8];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            M_work[i] = M[i];
        }

        // Advanced nonce mixing for better distribution
        uint32_t mix1 = __funnelshift_l(global_tid, n, 13) ^ __popc(nonce);
        uint32_t mix2 = __brev(global_tid + n) ^ __clz(nonce >> 32);

        M_work[6] = M[6] ^ (uint32_t) (nonce & 0xFFFFFFFF) ^ mix1;
        M_work[7] = M[7] ^ (uint32_t) (nonce >> 32) ^ mix2;

        // SHA-1 computation
        uint32_t W[16];
        prepare_message_schedule(M_work, W);

        uint32_t state[5] = {
            0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u
        };

        sha1_transform_optimized(W, state);

        // Check for match
        bool match = true;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            if (state[i] != shared_target[i]) {
                match = false;
                break;
            }
        }

        // Warp-level coordination for found results
        uint32_t found_mask = __ballot_sync(0xFFFFFFFF, match);

        if (found_mask != 0 && lane_id == 0) {
            uint32_t count = __popc(found_mask);
            uint32_t old = atomicAdd(&warp_found[warp_id], count);

            // Store results cooperatively
            if (old + count <= 4) {
                for (int i = 0; i < 32; i++) {
                    if (found_mask & (1u << i)) {
                        uint32_t pos = atomicAdd(ticket, 1);
                        if (out_pairs && pos < (1u << 20)) {
                            // Reconstruct the message for thread i
                            uint64_t thread_nonce = seed + (tid - lane_id + i) + n * stride;
                            uint32_t thread_tid = tid - lane_id + i;

                            uint64_t *dst = out_pairs + pos * 4;
                            dst[0] = ((uint64_t) M[1] << 32) | M[0];
                            dst[1] = ((uint64_t) M[3] << 32) | M[2];
                            dst[2] = ((uint64_t) M[5] << 32) | M[4];

                            // Reconstruct M[6] and M[7]
                            uint32_t m6 = M[6] ^ (uint32_t) (thread_nonce & 0xFFFFFFFF) ^
                                          __funnelshift_l(thread_tid + stream_id * gridDim.x * blockDim.x, n, 13) ^
                                          __popc(thread_nonce);
                            uint32_t m7 = M[7] ^ (uint32_t) (thread_nonce >> 32) ^
                                          __brev(thread_tid + n) ^
                                          __clz(thread_nonce >> 32);

                            dst[3] = ((uint64_t) m7 << 32) | m6;
                        }
                    }
                }
            }
        }
    }
}

// =================================================================
// Extreme Performance Kernel with Instruction-Level Parallelism
// =================================================================

extern "C" __global__ __launch_bounds__(128, 8)
void sha1_collision_kernel_extreme(
    uint8_t * __restrict__ out_msgs,
    uint64_t * __restrict__ out_pairs,
    uint32_t * __restrict__ ticket,
    uint64_t seed
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Use texture memory for target (if available)
    __shared__ uint32_t shared_target[5];
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    }
    __syncthreads();

    // Load base message
    uint32_t M[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M[i] = ((const uint32_t *) g_job_msg)[i];
    }

    // Process 2 nonces in parallel using ILP
    uint64_t nonce1 = seed + tid * 2;
    uint64_t nonce2 = seed + tid * 2 + 1;

    // Prepare both messages
    uint32_t M1[8], M2[8];
#pragma unroll
    for (int i = 0; i < 6; i++) {
        M1[i] = M[i];
        M2[i] = M[i];
    }

    M1[6] = M[6] ^ (uint32_t) (nonce1 & 0xFFFFFFFF);
    M1[7] = M[7] ^ (uint32_t) (nonce1 >> 32) ^ __brev(tid * 2);

    M2[6] = M[6] ^ (uint32_t) (nonce2 & 0xFFFFFFFF);
    M2[7] = M[7] ^ (uint32_t) (nonce2 >> 32) ^ __brev(tid * 2 + 1);

    // Prepare message schedules
    uint32_t W1[16], W2[16];
    prepare_message_schedule(M1, W1);
    prepare_message_schedule(M2, W2);

    // Initialize states
    uint32_t state1[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    uint32_t state2[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};

    // Interleaved SHA-1 transforms for ILP
    uint32_t a1 = state1[0], b1 = state1[1], c1 = state1[2], d1 = state1[3], e1 = state1[4];
    uint32_t a2 = state2[0], b2 = state2[1], c2 = state2[2], d2 = state2[3], e2 = state2[4];

    uint32_t w1[16], w2[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        w1[i] = W1[i];
        w2[i] = W2[i];
    }

    // Rounds 0-19 interleaved
#pragma unroll
    for (int i = 0; i < 20; i++) {
        uint32_t wi1 = (i < 16) ? w1[i] : compute_w(w1, i);
        uint32_t wi2 = (i < 16) ? w2[i] : compute_w(w2, i);

        uint32_t f1 = (b1 & c1) | (~b1 & d1);
        uint32_t f2 = (b2 & c2) | (~b2 & d2);

        uint32_t temp1 = rotl32(a1, 5) + f1 + e1 + K[0] + wi1;
        uint32_t temp2 = rotl32(a2, 5) + f2 + e2 + K[0] + wi2;

        e1 = d1;
        d1 = c1;
        c1 = rotl32(b1, 30);
        b1 = a1;
        a1 = temp1;
        e2 = d2;
        d2 = c2;
        c2 = rotl32(b2, 30);
        b2 = a2;
        a2 = temp2;
    }

    // Rounds 20-39 interleaved
#pragma unroll
    for (int i = 20; i < 40; i++) {
        uint32_t wi1 = compute_w(w1, i);
        uint32_t wi2 = compute_w(w2, i);

        uint32_t f1 = b1 ^ c1 ^ d1;
        uint32_t f2 = b2 ^ c2 ^ d2;

        uint32_t temp1 = rotl32(a1, 5) + f1 + e1 + K[1] + wi1;
        uint32_t temp2 = rotl32(a2, 5) + f2 + e2 + K[1] + wi2;

        e1 = d1;
        d1 = c1;
        c1 = rotl32(b1, 30);
        b1 = a1;
        a1 = temp1;
        e2 = d2;
        d2 = c2;
        c2 = rotl32(b2, 30);
        b2 = a2;
        a2 = temp2;
    }

    // Rounds 40-59 interleaved
#pragma unroll
    for (int i = 40; i < 60; i++) {
        uint32_t wi1 = compute_w(w1, i);
        uint32_t wi2 = compute_w(w2, i);

        uint32_t f1 = (b1 & c1) | (b1 & d1) | (c1 & d1);
        uint32_t f2 = (b2 & c2) | (b2 & d2) | (c2 & d2);

        uint32_t temp1 = rotl32(a1, 5) + f1 + e1 + K[2] + wi1;
        uint32_t temp2 = rotl32(a2, 5) + f2 + e2 + K[2] + wi2;

        e1 = d1;
        d1 = c1;
        c1 = rotl32(b1, 30);
        b1 = a1;
        a1 = temp1;
        e2 = d2;
        d2 = c2;
        c2 = rotl32(b2, 30);
        b2 = a2;
        a2 = temp2;
    }

    // Rounds 60-79 interleaved
#pragma unroll
    for (int i = 60; i < 80; i++) {
        uint32_t wi1 = compute_w(w1, i);
        uint32_t wi2 = compute_w(w2, i);

        uint32_t f1 = b1 ^ c1 ^ d1;
        uint32_t f2 = b2 ^ c2 ^ d2;

        uint32_t temp1 = rotl32(a1, 5) + f1 + e1 + K[3] + wi1;
        uint32_t temp2 = rotl32(a2, 5) + f2 + e2 + K[3] + wi2;

        e1 = d1;
        d1 = c1;
        c1 = rotl32(b1, 30);
        b1 = a1;
        a1 = temp1;
        e2 = d2;
        d2 = c2;
        c2 = rotl32(b2, 30);
        b2 = a2;
        a2 = temp2;
    }

    // Final addition
    state1[0] += a1;
    state1[1] += b1;
    state1[2] += c1;
    state1[3] += d1;
    state1[4] += e1;
    state2[0] += a2;
    state2[1] += b2;
    state2[2] += c2;
    state2[3] += d2;
    state2[4] += e2;

    // Check both results
    bool match1 = true, match2 = true;
#pragma unroll
    for (int i = 0; i < 5; i++) {
        if (state1[i] != shared_target[i]) match1 = false;
        if (state2[i] != shared_target[i]) match2 = false;
    }

    // Store results
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

// =================================================================
// Ultra Performance Kernel with 4-way ILP
// =================================================================

extern "C" __global__ __launch_bounds__(64, 16)
void sha1_collision_kernel_ultra(
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

    uint32_t M[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M[i] = ((const uint32_t *) g_job_msg)[i];
    }

    // Process 4 nonces in parallel
    uint64_t nonce_base = seed + tid * 4;

    // Prepare 4 messages
    uint32_t M0[8], M1[8], M2[8], M3[8];
#pragma unroll
    for (int i = 0; i < 6; i++) {
        M0[i] = M[i];
        M1[i] = M[i];
        M2[i] = M[i];
        M3[i] = M[i];
    }

    M0[6] = M[6] ^ (uint32_t) ((nonce_base + 0) & 0xFFFFFFFF);
    M0[7] = M[7] ^ (uint32_t) ((nonce_base + 0) >> 32) ^ __brev(tid * 4 + 0);

    M1[6] = M[6] ^ (uint32_t) ((nonce_base + 1) & 0xFFFFFFFF);
    M1[7] = M[7] ^ (uint32_t) ((nonce_base + 1) >> 32) ^ __brev(tid * 4 + 1);

    M2[6] = M[6] ^ (uint32_t) ((nonce_base + 2) & 0xFFFFFFFF);
    M2[7] = M[7] ^ (uint32_t) ((nonce_base + 2) >> 32) ^ __brev(tid * 4 + 2);

    M3[6] = M[6] ^ (uint32_t) ((nonce_base + 3) & 0xFFFFFFFF);
    M3[7] = M[7] ^ (uint32_t) ((nonce_base + 3) >> 32) ^ __brev(tid * 4 + 3);

    // Prepare schedules
    uint32_t W0[16], W1[16], W2[16], W3[16];
    prepare_message_schedule(M0, W0);
    prepare_message_schedule(M1, W1);
    prepare_message_schedule(M2, W2);
    prepare_message_schedule(M3, W3);

    // Initialize states
    const uint32_t H0 = 0x67452301u;
    const uint32_t H1 = 0xEFCDAB89u;
    const uint32_t H2 = 0x98BADCFEu;
    const uint32_t H3 = 0x10325476u;
    const uint32_t H4 = 0xC3D2E1F0u;

    uint32_t a0 = H0, b0 = H1, c0 = H2, d0 = H3, e0 = H4;
    uint32_t a1 = H0, b1 = H1, c1 = H2, d1 = H3, e1 = H4;
    uint32_t a2 = H0, b2 = H1, c2 = H2, d2 = H3, e2 = H4;
    uint32_t a3 = H0, b3 = H1, c3 = H2, d3 = H3, e3 = H4;

    uint32_t w0[16], w1[16], w2[16], w3[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        w0[i] = W0[i];
        w1[i] = W1[i];
        w2[i] = W2[i];
        w3[i] = W3[i];
    }

    // Process all 80 rounds with 4-way ILP
#pragma unroll
    for (int i = 0; i < 80; i++) {
        uint32_t k = (i < 20) ? K[0] : (i < 40) ? K[1] : (i < 60) ? K[2] : K[3];

        uint32_t wi0 = (i < 16) ? w0[i] : compute_w(w0, i);
        uint32_t wi1 = (i < 16) ? w1[i] : compute_w(w1, i);
        uint32_t wi2 = (i < 16) ? w2[i] : compute_w(w2, i);
        uint32_t wi3 = (i < 16) ? w3[i] : compute_w(w3, i);

        uint32_t f0, f1, f2, f3;
        if (i < 20) {
            f0 = (b0 & c0) | (~b0 & d0);
            f1 = (b1 & c1) | (~b1 & d1);
            f2 = (b2 & c2) | (~b2 & d2);
            f3 = (b3 & c3) | (~b3 & d3);
        } else if (i < 40 || i >= 60) {
            f0 = b0 ^ c0 ^ d0;
            f1 = b1 ^ c1 ^ d1;
            f2 = b2 ^ c2 ^ d2;
            f3 = b3 ^ c3 ^ d3;
        } else {
            f0 = (b0 & c0) | (b0 & d0) | (c0 & d0);
            f1 = (b1 & c1) | (b1 & d1) | (c1 & d1);
            f2 = (b2 & c2) | (b2 & d2) | (c2 & d2);
            f3 = (b3 & c3) | (b3 & d3) | (c3 & d3);
        }

        uint32_t temp0 = rotl32(a0, 5) + f0 + e0 + k + wi0;
        uint32_t temp1 = rotl32(a1, 5) + f1 + e1 + k + wi1;
        uint32_t temp2 = rotl32(a2, 5) + f2 + e2 + k + wi2;
        uint32_t temp3 = rotl32(a3, 5) + f3 + e3 + k + wi3;

        e0 = d0;
        d0 = c0;
        c0 = rotl32(b0, 30);
        b0 = a0;
        a0 = temp0;
        e1 = d1;
        d1 = c1;
        c1 = rotl32(b1, 30);
        b1 = a1;
        a1 = temp1;
        e2 = d2;
        d2 = c2;
        c2 = rotl32(b2, 30);
        b2 = a2;
        a2 = temp2;
        e3 = d3;
        d3 = c3;
        c3 = rotl32(b3, 30);
        b3 = a3;
        a3 = temp3;
    }

    // Final addition
    a0 += H0;
    b0 += H1;
    c0 += H2;
    d0 += H3;
    e0 += H4;
    a1 += H0;
    b1 += H1;
    c1 += H2;
    d1 += H3;
    e1 += H4;
    a2 += H0;
    b2 += H1;
    c2 += H2;
    d2 += H3;
    e2 += H4;
    a3 += H0;
    b3 += H1;
    c3 += H2;
    d3 += H3;
    e3 += H4;

    // Check all 4 results
    if (a0 == shared_target[0] && b0 == shared_target[1] && c0 == shared_target[2] &&
        d0 == shared_target[3] && e0 == shared_target[4]) {
        uint32_t pos = atomicAdd(ticket, 1);
        if (out_pairs && pos < (1u << 20)) {
            uint64_t *dst = out_pairs + pos * 4;
            dst[0] = ((uint64_t) M0[1] << 32) | M0[0];
            dst[1] = ((uint64_t) M0[3] << 32) | M0[2];
            dst[2] = ((uint64_t) M0[5] << 32) | M0[4];
            dst[3] = ((uint64_t) M0[7] << 32) | M0[6];
        }
    }

    if (a1 == shared_target[0] && b1 == shared_target[1] && c1 == shared_target[2] &&
        d1 == shared_target[3] && e1 == shared_target[4]) {
        uint32_t pos = atomicAdd(ticket, 1);
        if (out_pairs && pos < (1u << 20)) {
            uint64_t *dst = out_pairs + pos * 4;
            dst[0] = ((uint64_t) M1[1] << 32) | M1[0];
            dst[1] = ((uint64_t) M1[3] << 32) | M1[2];
            dst[2] = ((uint64_t) M1[5] << 32) | M1[4];
            dst[3] = ((uint64_t) M1[7] << 32) | M1[6];
        }
    }

    if (a2 == shared_target[0] && b2 == shared_target[1] && c2 == shared_target[2] &&
        d2 == shared_target[3] && e2 == shared_target[4]) {
        uint32_t pos = atomicAdd(ticket, 1);
        if (out_pairs && pos < (1u << 20)) {
            uint64_t *dst = out_pairs + pos * 4;
            dst[0] = ((uint64_t) M2[1] << 32) | M2[0];
            dst[1] = ((uint64_t) M2[3] << 32) | M2[2];
            dst[2] = ((uint64_t) M2[5] << 32) | M2[4];
            dst[3] = ((uint64_t) M2[7] << 32) | M2[6];
        }
    }

    if (a3 == shared_target[0] && b3 == shared_target[1] && c3 == shared_target[2] &&
        d3 == shared_target[3] && e3 == shared_target[4]) {
        uint32_t pos = atomicAdd(ticket, 1);
        if (out_pairs && pos < (1u << 20)) {
            uint64_t *dst = out_pairs + pos * 4;
            dst[0] = ((uint64_t) M3[1] << 32) | M3[0];
            dst[1] = ((uint64_t) M3[3] << 32) | M3[2];
            dst[2] = ((uint64_t) M3[5] << 32) | M3[4];
            dst[3] = ((uint64_t) M3[7] << 32) | M3[6];
        }
    }
}
