#pragma once
#include "gpu_platform.hpp"
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdint.h>

// Only include OpenSSL for host code
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <openssl/sha.h>
#else
// Define SHA_DIGEST_LENGTH for device code if not already defined
#ifndef SHA_DIGEST_LENGTH
#define SHA_DIGEST_LENGTH 20
#endif
#endif

// SHA-1 constants
#define SHA1_BLOCK_SIZE 64
#define SHA1_DIGEST_SIZE 20
#define SHA1_ROUNDS 80
#define MAX_CANDIDATES_PER_BATCH 1024

#ifdef USE_HIP
    #define NONCES_PER_THREAD 8192
    #define DEFAULT_THREADS_PER_BLOCK 256
#else
#define NONCES_PER_THREAD 8192
#define DEFAULT_THREADS_PER_BLOCK 256
#endif

struct MiningJob {
    uint8_t base_message[32]; // Base message to modify
    uint32_t target_hash[5]; // Target hash we're trying to match
    uint32_t difficulty; // Number of bits that must match
    uint64_t nonce_offset; // Starting nonce for this job
};

// Result structure for found candidates
struct MiningResult {
    uint64_t nonce; // The nonce that produced this result
    uint32_t hash[5]; // The resulting hash
    uint32_t matching_bits; // Number of bits matching the target
    uint32_t difficulty_score; // Additional difficulty metric
    uint64_t job_version; // Job version for this result
};

// GPU memory pool for results
struct ResultPool {
    MiningResult *results; // Array of results
    uint32_t *count; // Count of results found
    uint32_t capacity; // Maximum results
    uint64_t *nonces_processed; // Total nonces processed
    uint64_t *job_version; // Current job version (device memory)
};

// Mining statistics
struct MiningStats {
    uint64_t hashes_computed;
    uint64_t candidates_found;
    uint64_t best_match_bits;
    double hash_rate;
};

// Kernel launch parameters
struct KernelConfig {
    int blocks;
    int threads_per_block;
    int shared_memory_size;
    gpuStream_t stream;
};

// Device memory holder for mining job
// Update in sha1_miner.cuh - replace the DeviceMiningJob struct with improved version

struct DeviceMiningJob {
    uint8_t *base_message;
    uint32_t *target_hash;

    DeviceMiningJob() : base_message(nullptr), target_hash(nullptr) {
    }

    ~DeviceMiningJob() {
        free();
    }

    bool allocate() {
        // Free any existing allocations first
        free();

        // Check current device
        int current_device;
        gpuError_t err = gpuGetDevice(&current_device);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to get current device: %s\n",
                    gpuGetErrorString(err));
            return false;
        }

        // Check available memory before allocation
        size_t free_mem, total_mem;
        err = gpuMemGetInfo(&free_mem, &total_mem);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to get memory info: %s\n",
                    gpuGetErrorString(err));
        } else {
            fprintf(stderr, "[DeviceMiningJob] Device %d memory: %zu MB free / %zu MB total\n",
                    current_device, free_mem / (1024 * 1024), total_mem / (1024 * 1024));
        }

        // Allocate base_message (32 bytes)
        fprintf(stderr, "[DeviceMiningJob] Allocating 32 bytes for base_message...\n");
        err = gpuMalloc(&base_message, 32);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] ERROR: Failed to allocate base_message: %s (error code: %d)\n",
                    gpuGetErrorString(err), err);
            base_message = nullptr;

            // Try to understand why allocation failed
#ifdef USE_HIP
            if (err == hipErrorOutOfMemory) {
                fprintf(stderr, "[DeviceMiningJob] Out of GPU memory\n");
            } else if (err == hipErrorInvalidValue) {
                fprintf(stderr, "[DeviceMiningJob] Invalid allocation size\n");
            } else if (err == hipErrorInvalidDevice) {
                fprintf(stderr, "[DeviceMiningJob] Invalid device\n");
            } else if (err == hipErrorNoDevice) {
                fprintf(stderr, "[DeviceMiningJob] No GPU device available\n");
            }
#else
            if (err == cudaErrorMemoryAllocation) {
                fprintf(stderr, "[DeviceMiningJob] Out of GPU memory\n");
            } else if (err == cudaErrorInvalidValue) {
                fprintf(stderr, "[DeviceMiningJob] Invalid allocation size\n");
            } else if (err == cudaErrorInvalidDevice) {
                fprintf(stderr, "[DeviceMiningJob] Invalid device\n");
            } else if (err == cudaErrorNoDevice) {
                fprintf(stderr, "[DeviceMiningJob] No CUDA device available\n");
            }
#endif
            return false;
        }
        fprintf(stderr, "[DeviceMiningJob] Successfully allocated base_message at %p\n", base_message);

        // Allocate target_hash (20 bytes = 5 * sizeof(uint32_t))
        size_t target_size = 5 * sizeof(uint32_t);
        fprintf(stderr, "[DeviceMiningJob] Allocating %zu bytes for target_hash...\n", target_size);
        err = gpuMalloc(&target_hash, target_size);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] ERROR: Failed to allocate target_hash: %s (error code: %d)\n",
                    gpuGetErrorString(err), err);
            // Free the successfully allocated base_message
            gpuFree(base_message);
            base_message = nullptr;
            target_hash = nullptr;
            return false;
        }
        fprintf(stderr, "[DeviceMiningJob] Successfully allocated target_hash at %p\n", target_hash);

        // Verify allocations
        if (!base_message || !target_hash) {
            fprintf(stderr, "[DeviceMiningJob] ERROR: Pointers are null after allocation\n");
            free();
            return false;
        }

        // Clear the allocated memory
        err = gpuMemset(base_message, 0, 32);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Warning: Failed to clear base_message: %s\n",
                    gpuGetErrorString(err));
        }

        err = gpuMemset(target_hash, 0, target_size);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Warning: Failed to clear target_hash: %s\n",
                    gpuGetErrorString(err));
        }

        fprintf(stderr, "[DeviceMiningJob] Allocation successful\n");
        return true;
    }

    void free() {
        if (base_message) {
            gpuError_t err = gpuFree(base_message);
            if (err != gpuSuccess) {
                fprintf(stderr, "[DeviceMiningJob] Warning: Failed to free base_message: %s\n",
                        gpuGetErrorString(err));
            }
            base_message = nullptr;
        }
        if (target_hash) {
            gpuError_t err = gpuFree(target_hash);
            if (err != gpuSuccess) {
                fprintf(stderr, "[DeviceMiningJob] Warning: Failed to free target_hash: %s\n",
                        gpuGetErrorString(err));
            }
            target_hash = nullptr;
        }
    }

    void copyFromHost(const MiningJob &job) const {
        fprintf(stderr, "[DeviceMiningJob] copyFromHost called. Pointers: base_message=%p, target_hash=%p\n", base_message, target_hash);
        if (!base_message || !target_hash) {
            fprintf(stderr, "[DeviceMiningJob] Error: Not allocated (base_message=%p, target_hash=%p)\n",
                            base_message, target_hash);
            return;
        }
        gpuError_t err = gpuMemcpy(base_message, job.base_message, 32, gpuMemcpyHostToDevice);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to copy base_message: %s\n", gpuGetErrorString(err));
        }
        err = gpuMemcpy(target_hash, job.target_hash, 5 * sizeof(uint32_t), gpuMemcpyHostToDevice);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to copy target_hash: %s\n", gpuGetErrorString(err));
        }
        fprintf(stderr, "[DeviceMiningJob] Successfully copied new job data to device memory\n");
    }

    void copyFromHostAsync(const MiningJob &job, gpuStream_t stream) const {
        fprintf(stderr, "[DeviceMiningJob] copyFromHostAsync called. Pointers: base_message=%p, target_hash=%p, stream=%p\n", base_message, target_hash, stream);
        if (!base_message || !target_hash) {
            fprintf(stderr, "[DeviceMiningJob] Error: Not allocated\n");
            return;
        }
        gpuError_t err = gpuMemcpyAsync(base_message, job.base_message, 32, gpuMemcpyHostToDevice, stream);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to async copy base_message: %s\n", gpuGetErrorString(err));
        }
        err = gpuMemcpyAsync(target_hash, job.target_hash, 5 * sizeof(uint32_t),
                                gpuMemcpyHostToDevice, stream);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to async copy target_hash: %s\n",
                            gpuGetErrorString(err));
        }
        fprintf(stderr, "[DeviceMiningJob] Successfully launched async copy to device memory\n");
    }

    void updateFromHost(const MiningJob &job) const {
        copyFromHost(job);
    }

    bool isAllocated() const {
        return base_message != nullptr && target_hash != nullptr;
    }
};

// API functions - Host side
#ifdef __cplusplus
extern "C" {
#endif

MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);

void cleanup_mining_system();

void run_mining_loop(MiningJob job);

#ifdef __cplusplus
}
#endif

// C++ only functions (not extern "C")
#ifdef __cplusplus

// Launch mining kernel - uses C++ references
void launch_mining_kernel(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version
);

#endif // __cplusplus

// Device functions for SHA-1 computation
#if defined(__CUDACC__) || defined(__HIPCC__)

// Platform-optimized rotation function
__gpu_device__ __gpu_forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
#ifdef USE_HIP
    return __builtin_rotateleft32(x, n);
#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return __funnelshift_l(x, x, n);
#else
        return (x << n) | (x >> (32 - n));
#endif
#endif
}

// Platform-optimized endian swap
__gpu_device__ __gpu_forceinline__ uint32_t swap_endian(uint32_t x) {
#ifdef USE_HIP
    return __builtin_bswap32(x);
#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
        return __byte_perm(x, 0, 0x0123);
#else
        return ((x & 0xFF000000) >> 24) |
                ((x & 0x00FF0000) >> 8)  |
                ((x & 0x0000FF00) << 8)  |
                ((x & 0x000000FF) << 24);
#endif
#endif
}

// Count leading zeros - platform optimized
__gpu_device__ __gpu_forceinline__ uint32_t count_leading_zeros(uint32_t x) {
    return __gpu_clz(x);
}

#endif // __CUDACC__ || __HIPCC__

// Helper functions for SHA-1 computation (host side) using OpenSSL
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline std::vector<uint8_t> sha1_binary(const uint8_t *data, size_t len) {
    std::vector<uint8_t> result(SHA_DIGEST_LENGTH);
    SHA1(data, len, result.data());
    return result;
}

inline std::vector<uint8_t> sha1_binary(const std::vector<uint8_t> &data) {
    return sha1_binary(data.data(), data.size());
}

inline std::string sha1_hex(const uint8_t *hash) {
    std::ostringstream oss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        oss << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(hash[i]);
    }
    return oss.str();
}

inline std::vector<uint8_t> hex_to_binary(const std::string &hex) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < hex.length(); i += 2) {
        result.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return result;
}

inline std::vector<uint8_t> calculate_sha1(const std::vector<uint8_t> &data) {
    return sha1_binary(data);
}

inline std::string bytes_to_hex(const std::vector<uint8_t> &bytes) {
    std::ostringstream oss;
    for (uint8_t b: bytes) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return oss.str();
}

inline std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
    return hex_to_binary(hex);
}

#endif // !__CUDACC__ && !__HIPCC__
