#include "job_constants.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __constant__ uint8_t g_job_msg[32];
__device__ __constant__ uint32_t g_target[5];

// SHA-1 round constants - combine into single array for better cache
__device__ __constant__ uint32_t K[4] = {
    0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xCA62C1D6u
};

// =================================================================
// Critical Performance Optimizations
// =================================================================

// Force inline everything - compiler hints are suggestions, not guarantees
#define DEVICE_INLINE __device__ __forceinline__

// Use intrinsics directly for rotation
DEVICE_INLINE uint32_t rotl32(uint32_t x, uint32_t n) {
    return __funnelshift_l(x, x, n);
}

// Optimized byte swap using intrinsic
DEVICE_INLINE uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

// =================================================================
// Optimized SHA-1 Implementation with Manual Unrolling
// =================================================================

DEVICE_INLINE void sha1_collision_kernel(const uint32_t M[8], uint32_t result[5]) {
    // Initialize working variables
    uint32_t a = 0x67452301u;
    uint32_t b = 0xEFCDAB89u;
    uint32_t c = 0x98BADCFEu;
    uint32_t d = 0x10325476u;
    uint32_t e = 0xC3D2E1F0u;

    // Message schedule - keep in registers as much as possible
    uint32_t W[16];

    // Prepare initial message schedule with padding
    W[0] = bswap32(M[0]);
    W[1] = bswap32(M[1]);
    W[2] = bswap32(M[2]);
    W[3] = bswap32(M[3]);
    W[4] = bswap32(M[4]);
    W[5] = bswap32(M[5]);
    W[6] = bswap32(M[6]);
    W[7] = bswap32(M[7]);
    W[8] = 0x80000000u;
    W[9] = 0;
    W[10] = 0;
    W[11] = 0;
    W[12] = 0;
    W[13] = 0;
    W[14] = 0;
    W[15] = 0x00000100u;

    // Rounds 0-15: Direct message use
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t f = (b & c) | (~b & d);
        uint32_t temp = rotl32(a, 5) + f + e + K[0] + W[i];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 16-19: Complete first phase
#pragma unroll
    for (int i = 16; i < 20; i++) {
        uint32_t wi = W[i & 15] = rotl32(W[(i + 13) & 15] ^ W[(i + 8) & 15] ^ W[(i + 2) & 15] ^ W[i & 15], 1);
        uint32_t f = (b & c) | (~b & d);
        uint32_t temp = rotl32(a, 5) + f + e + K[0] + wi;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 20-39: XOR phase
#pragma unroll
    for (int i = 20; i < 40; i++) {
        uint32_t wi = W[i & 15] = rotl32(W[(i + 13) & 15] ^ W[(i + 8) & 15] ^ W[(i + 2) & 15] ^ W[i & 15], 1);
        uint32_t f = b ^ c ^ d;
        uint32_t temp = rotl32(a, 5) + f + e + K[1] + wi;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 40-59: Majority phase
#pragma unroll
    for (int i = 40; i < 60; i++) {
        uint32_t wi = W[i & 15] = rotl32(W[(i + 13) & 15] ^ W[(i + 8) & 15] ^ W[(i + 2) & 15] ^ W[i & 15], 1);
        uint32_t f = (b & c) | (b & d) | (c & d);
        uint32_t temp = rotl32(a, 5) + f + e + K[2] + wi;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 60-79: XOR phase
#pragma unroll
    for (int i = 60; i < 80; i++) {
        uint32_t wi = W[i & 15] = rotl32(W[(i + 13) & 15] ^ W[(i + 8) & 15] ^ W[(i + 2) & 15] ^ W[i & 15], 1);
        uint32_t f = b ^ c ^ d;
        uint32_t temp = rotl32(a, 5) + f + e + K[3] + wi;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Final addition
    result[0] = 0x67452301u + a;
    result[1] = 0xEFCDAB89u + b;
    result[2] = 0x98BADCFEu + c;
    result[3] = 0x10325476u + d;
    result[4] = 0xC3D2E1F0u + e;
}

// =================================================================
// Optimized Base Kernel - Process 8 nonces per thread
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

    // Load target into shared memory once
    __shared__ uint32_t shared_target[5];
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    }
    __syncthreads();

    // Load base message into registers
    uint32_t M_base[8];
    const uint32_t *msg_ptr = (const uint32_t *) g_job_msg;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M_base[i] = msg_ptr[i];
    }

    // Process 8 nonces per thread for better efficiency
#pragma unroll 8
    for (uint32_t n = 0; n < 8; n++) {
        uint64_t nonce = seed + tid + n * total_threads;

        // Prepare working message
        uint32_t M[8];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            M[i] = M_base[i];
        }

        // Apply nonce - simplified mixing
        M[6] = M_base[6] ^ (uint32_t) (nonce & 0xFFFFFFFF);
        M[7] = M_base[7] ^ (uint32_t) (nonce >> 32);

        // Compute SHA-1
        uint32_t hash[5];
        sha1_collision_kernel(M, hash);

        // Check if we found a match
        if (hash[0] == shared_target[0] &&
            hash[1] == shared_target[1] &&
            hash[2] == shared_target[2] &&
            hash[3] == shared_target[3] &&
            hash[4] == shared_target[4]) {
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
// 2-way ILP Kernel - Process 2 SHA-1 in parallel
// =================================================================

extern "C" __global__ __launch_bounds__(128, 8)
void sha1_collision_kernel_extreme(
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
    const uint32_t *msg_ptr = (const uint32_t *) g_job_msg;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M_base[i] = msg_ptr[i];
    }

    // Process 4 iterations of 2 parallel hashes each
#pragma unroll 4
    for (uint32_t iter = 0; iter < 4; iter++) {
        uint64_t nonce1 = seed + (tid * 8) + (iter * 2);
        uint64_t nonce2 = nonce1 + 1;

        // Prepare two messages
        uint32_t M1[8], M2[8];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            M1[i] = M_base[i];
            M2[i] = M_base[i];
        }

        M1[6] = M_base[6] ^ (uint32_t) (nonce1 & 0xFFFFFFFF);
        M1[7] = M_base[7] ^ (uint32_t) (nonce1 >> 32);
        M2[6] = M_base[6] ^ (uint32_t) (nonce2 & 0xFFFFFFFF);
        M2[7] = M_base[7] ^ (uint32_t) (nonce2 >> 32);

        // Compute both SHA-1 hashes
        uint32_t hash1[5], hash2[5];
        sha1_collision_kernel(M1, hash1);
        sha1_collision_kernel(M2, hash2);

        // Check results
        if (hash1[0] == shared_target[0] && hash1[1] == shared_target[1] &&
            hash1[2] == shared_target[2] && hash1[3] == shared_target[3] &&
            hash1[4] == shared_target[4]) {
            uint32_t pos = atomicAdd(ticket, 1);
            if (out_pairs && pos < (1u << 20)) {
                uint64_t *dst = out_pairs + pos * 4;
                dst[0] = ((uint64_t) M1[1] << 32) | M1[0];
                dst[1] = ((uint64_t) M1[3] << 32) | M1[2];
                dst[2] = ((uint64_t) M1[5] << 32) | M1[4];
                dst[3] = ((uint64_t) M1[7] << 32) | M1[6];
            }
        }

        if (hash2[0] == shared_target[0] && hash2[1] == shared_target[1] &&
            hash2[2] == shared_target[2] && hash2[3] == shared_target[3] &&
            hash2[4] == shared_target[4]) {
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

// =================================================================
// Multi-stream kernel with warp-level optimization
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

    __shared__ uint32_t shared_target[5];
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    }
    __syncthreads();

    uint32_t M_base[8];
    const uint32_t *msg_ptr = (const uint32_t *) g_job_msg;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M_base[i] = msg_ptr[i];
    }

    // Process 8 nonces per thread
#pragma unroll 8
    for (uint32_t n = 0; n < 8; n++) {
        uint64_t nonce = seed + global_tid + n * stride;

        uint32_t M[8];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            M[i] = M_base[i];
        }

        M[6] = M_base[6] ^ (uint32_t) (nonce & 0xFFFFFFFF);
        M[7] = M_base[7] ^ (uint32_t) (nonce >> 32);

        uint32_t hash[5];
        sha1_collision_kernel(M, hash);

        if (hash[0] == shared_target[0] && hash[1] == shared_target[1] &&
            hash[2] == shared_target[2] && hash[3] == shared_target[3] &&
            hash[4] == shared_target[4]) {
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
// Ultra kernel - 4-way ILP with minimal overhead
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

    uint32_t M_base[8];
    const uint32_t *msg_ptr = (const uint32_t *) g_job_msg;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        M_base[i] = msg_ptr[i];
    }

    // Process 2 iterations of 4 parallel hashes
#pragma unroll 2
    for (uint32_t iter = 0; iter < 2; iter++) {
        uint64_t nonce_base = seed + (tid * 8) + (iter * 4);

        // Prepare 4 messages
        uint32_t M0[8], M1[8], M2[8], M3[8];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            M0[i] = M_base[i];
            M1[i] = M_base[i];
            M2[i] = M_base[i];
            M3[i] = M_base[i];
        }

        M0[6] = M_base[6] ^ (uint32_t) (nonce_base & 0xFFFFFFFF);
        M0[7] = M_base[7] ^ (uint32_t) (nonce_base >> 32);
        M1[6] = M_base[6] ^ (uint32_t) ((nonce_base + 1) & 0xFFFFFFFF);
        M1[7] = M_base[7] ^ (uint32_t) ((nonce_base + 1) >> 32);
        M2[6] = M_base[6] ^ (uint32_t) ((nonce_base + 2) & 0xFFFFFFFF);
        M2[7] = M_base[7] ^ (uint32_t) ((nonce_base + 2) >> 32);
        M3[6] = M_base[6] ^ (uint32_t) ((nonce_base + 3) & 0xFFFFFFFF);
        M3[7] = M_base[7] ^ (uint32_t) ((nonce_base + 3) >> 32);

        // Compute 4 SHA-1 hashes
        uint32_t hash0[5], hash1[5], hash2[5], hash3[5];
        sha1_collision_kernel(M0, hash0);
        sha1_collision_kernel(M1, hash1);
        sha1_collision_kernel(M2, hash2);
        sha1_collision_kernel(M3, hash3);

        // Check all 4 results
        if (hash0[0] == shared_target[0] && hash0[1] == shared_target[1] &&
            hash0[2] == shared_target[2] && hash0[3] == shared_target[3] &&
            hash0[4] == shared_target[4]) {
            uint32_t pos = atomicAdd(ticket, 1);
            if (out_pairs && pos < (1u << 20)) {
                uint64_t *dst = out_pairs + pos * 4;
                dst[0] = ((uint64_t) M0[1] << 32) | M0[0];
                dst[1] = ((uint64_t) M0[3] << 32) | M0[2];
                dst[2] = ((uint64_t) M0[5] << 32) | M0[4];
                dst[3] = ((uint64_t) M0[7] << 32) | M0[6];
            }
        }

        if (hash1[0] == shared_target[0] && hash1[1] == shared_target[1] &&
            hash1[2] == shared_target[2] && hash1[3] == shared_target[3] &&
            hash1[4] == shared_target[4]) {
            uint32_t pos = atomicAdd(ticket, 1);
            if (out_pairs && pos < (1u << 20)) {
                uint64_t *dst = out_pairs + pos * 4;
                dst[0] = ((uint64_t) M1[1] << 32) | M1[0];
                dst[1] = ((uint64_t) M1[3] << 32) | M1[2];
                dst[2] = ((uint64_t) M1[5] << 32) | M1[4];
                dst[3] = ((uint64_t) M1[7] << 32) | M1[6];
            }
        }

        if (hash2[0] == shared_target[0] && hash2[1] == shared_target[1] &&
            hash2[2] == shared_target[2] && hash2[3] == shared_target[3] &&
            hash2[4] == shared_target[4]) {
            uint32_t pos = atomicAdd(ticket, 1);
            if (out_pairs && pos < (1u << 20)) {
                uint64_t *dst = out_pairs + pos * 4;
                dst[0] = ((uint64_t) M2[1] << 32) | M2[0];
                dst[1] = ((uint64_t) M2[3] << 32) | M2[2];
                dst[2] = ((uint64_t) M2[5] << 32) | M2[4];
                dst[3] = ((uint64_t) M2[7] << 32) | M2[6];
            }
        }

        if (hash3[0] == shared_target[0] && hash3[1] == shared_target[1] &&
            hash3[2] == shared_target[2] && hash3[3] == shared_target[3] &&
            hash3[4] == shared_target[4]) {
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
}
