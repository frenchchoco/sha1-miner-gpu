#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>
#include <fstream>
#include <sstream>
#include <ctime>
#include <mutex>
#include <memory>
#include <algorithm>
#include <random>
#include <cstring>
#include <deque>

#ifdef _WIN32
#include <windows.h>
#else
    #include <pthread.h>
    #include <sched.h>
#endif

#include "job_upload_api.h"
#include "cxxsha1.hpp"

// Kernel declarations - updated to match our fixed kernels
extern "C" __global__ void sha1_mining_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_warp_collaborative_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_vectorized_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_bitsliced_kernel_correct(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_hashcat_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_hashcat_extreme_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_cooperative_groups_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_multi_hash_kernel(uint64_t *, uint32_t *, uint64_t); // good

extern "C" __global__ void sha1_readonly_cache_kernel(uint64_t *, uint32_t *, uint64_t); // bad

extern "C" __global__ void sha1_simd_vectorized_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_hybrid_warp_simd_kernel(uint64_t *, uint32_t *, uint64_t); // very good?

extern "C" __global__ void sha1_ldg_optimized_kernel(uint64_t *, uint32_t *, uint64_t); // bad

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(1);} \
    }while(0)

// ==================== Configuration ====================
constexpr uint32_t RING_SIZE = 1u << 20; // 1M candidate slots
constexpr int PROGRESS_INTERVAL = 5; // Update every 5 seconds
constexpr int STREAMS_PER_GPU = 8; // More streams for better overlap

// Kernel types
enum KernelType {
    KERNEL_STANDARD,
    KERNEL_WARP_COLLABORATIVE,
    KERNEL_VECTORIZED,
    KERNEL_BITSLICED,
    KERNEL_HASHCAT,
    KERNEL_HASHCAT_EXTREME,
    KERNEL_COOPERATIVE_GROUPS,
    KERNEL_MULTI_HASH,
    KERNEL_READONLY_CACHE,
    KERNEL_SIMD_VECTORIZED,
    KERNEL_HYBRID_WARP_SIMD,
    KERNEL_LDG_OPTIMIZED
};

// Global state for signal handling
std::atomic<bool> g_shutdown(false);
std::atomic<uint64_t> g_total_hashes(0);
std::atomic<uint32_t> g_total_candidates(0);
std::atomic<uint64_t> g_total_kernel_launches(0);
std::chrono::steady_clock::time_point g_start_time;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    std::cout << "\n\nReceived signal " << sig << ", shutting down gracefully...\n";
    g_shutdown.store(true);
}

void setup_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef _WIN32
    std::signal(SIGBREAK, signal_handler);
#endif
}

// Helper to get kernel name
const char *getKernelName(KernelType type) {
    static const char *names[] = {
        "Standard", "Warp Collaborative", "Vectorized", "Bitsliced",
        "HashCat", "HashCat Extreme", "Cooperative Groups", "Multi-Hash",
        "Read-Only Cache", "SIMD Vectorized", "Hybrid Warp-SIMD", "LDG Optimized"
    };
    return names[type];
}

// ==================== GPU Configuration ====================
struct GPUContext {
    int device_id;
    cudaDeviceProp properties;
    std::vector<cudaStream_t> streams;
    std::vector<uint64_t *> d_pairs;
    std::vector<uint32_t *> d_tickets;
    int optimal_blocks;
    int optimal_threads;
    std::atomic<uint64_t> hashes_processed;
    std::atomic<uint64_t> kernels_launched;
    KernelType kernel_type;
    int warps_per_block;
    uint64_t hashes_per_kernel_launch;

    GPUContext(int id, bool force_bitsliced = false, bool use_hashcat = true,
               int force_kernel_type = -1) : device_id(id), hashes_processed(0), kernels_launched(0) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));

        // Determine optimal configuration based on GPU and kernel type
        if (force_kernel_type >= 0 && force_kernel_type <= 11) {
            kernel_type = (KernelType) force_kernel_type;

            // Set configuration based on actual kernel implementation
            switch (kernel_type) {
                case KERNEL_STANDARD:
                    optimal_threads = 256;
                    warps_per_block = optimal_threads / 32;
                    optimal_blocks = properties.multiProcessorCount * 8;
                    // sha1_mining_kernel: NONCES_PER_THREAD = 4
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 4;
                    break;

                case KERNEL_WARP_COLLABORATIVE:
                    optimal_threads = 256;
                    warps_per_block = 8;
                    optimal_blocks = properties.multiProcessorCount * 32;
                    // sha1_warp_collaborative_kernel: BATCHES_PER_WARP = 8, 32 hashes per batch
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * warps_per_block * 8 * 32;
                    break;

                case KERNEL_VECTORIZED:
                    optimal_threads = 128;
                    warps_per_block = optimal_threads / 32;
                    optimal_blocks = properties.multiProcessorCount * 16;
                    // sha1_vectorized_kernel: 2 nonces per thread
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 2;
                    break;

                case KERNEL_BITSLICED:
                    optimal_threads = 128;
                    warps_per_block = 4;
                    optimal_blocks = properties.multiProcessorCount * 8;
                    // sha1_bitsliced_kernel_correct: BATCHES = 4, 32 messages per batch per warp
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * warps_per_block * 4 * 32;
                    break;

                case KERNEL_HASHCAT:
                    optimal_threads = 64;
                    warps_per_block = 2;
                    optimal_blocks = properties.multiProcessorCount * 64;
                    // sha1_hashcat_kernel: HASHES_PER_THREAD = 16
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 16;
                    break;

                case KERNEL_HASHCAT_EXTREME:
                    optimal_threads = 128;
                    warps_per_block = 4;
                    optimal_blocks = properties.multiProcessorCount * 32;
                    // sha1_hashcat_extreme_kernel: HASHES_PER_THREAD = 32
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 32;
                    break;

                case KERNEL_COOPERATIVE_GROUPS:
                    optimal_threads = 256;
                    warps_per_block = 8;
                    optimal_blocks = properties.multiProcessorCount * 16;
                    // sha1_cooperative_groups_kernel: BATCHES = 10, 32 hashes per batch per warp
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * warps_per_block * 10 * 32;
                    break;

                case KERNEL_MULTI_HASH:
                    optimal_threads = 128;
                    warps_per_block = optimal_threads / 32;
                    optimal_blocks = properties.multiProcessorCount * 16;
                    // sha1_multi_hash_kernel: HASHES_PER_THREAD = 8
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 8;
                    break;

                case KERNEL_READONLY_CACHE:
                    optimal_threads = 256;
                    warps_per_block = optimal_threads / 32;
                    optimal_blocks = properties.multiProcessorCount * 16;
                    // sha1_readonly_cache_kernel: NONCES_PER_THREAD = 4
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 4;
                    break;

                case KERNEL_SIMD_VECTORIZED:
                    optimal_threads = 256;
                    warps_per_block = optimal_threads / 32;
                    optimal_blocks = properties.multiProcessorCount * 16;
                    // sha1_simd_vectorized_kernel: VECTORS_PER_THREAD = 2, 4 hashes per vector
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 2 * 4;
                    break;

                case KERNEL_HYBRID_WARP_SIMD:
                    optimal_threads = 128;
                    warps_per_block = 4;
                    optimal_blocks = properties.multiProcessorCount * 16;
                    // sha1_hybrid_warp_simd_kernel: BATCHES = 4, HASHES_PER_THREAD = 2, 32 threads per warp
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * warps_per_block * 4 * 32 * 2;
                    break;

                case KERNEL_LDG_OPTIMIZED:
                    optimal_threads = 256;
                    warps_per_block = optimal_threads / 32;
                    optimal_blocks = properties.multiProcessorCount * 16;
                    // sha1_ldg_optimized_kernel: NONCES_PER_THREAD = 6
                    hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 6;
                    break;
            }
        } else if (force_bitsliced) {
            kernel_type = KERNEL_BITSLICED;
            optimal_threads = 128;
            warps_per_block = 4;
            optimal_blocks = properties.multiProcessorCount * 8;
            hashes_per_kernel_launch = (uint64_t) optimal_blocks * warps_per_block * 4 * 32;
        } else if (use_hashcat) {
            if (properties.major >= 8) {
                kernel_type = KERNEL_HASHCAT_EXTREME;
                optimal_threads = 128;
                warps_per_block = 4;
                optimal_blocks = properties.multiProcessorCount * 32;
                hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 32;
            } else {
                kernel_type = KERNEL_HASHCAT;
                optimal_threads = 64;
                warps_per_block = 2;
                optimal_blocks = properties.multiProcessorCount * 64;
                hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 16;
            }
        } else if (properties.major >= 7) {
            kernel_type = KERNEL_WARP_COLLABORATIVE;
            optimal_threads = 256;
            warps_per_block = 8;
            optimal_blocks = properties.multiProcessorCount * 32;
            hashes_per_kernel_launch = (uint64_t) optimal_blocks * warps_per_block * 8 * 32;
        } else if (properties.major >= 6) {
            kernel_type = KERNEL_VECTORIZED;
            optimal_threads = 128;
            warps_per_block = 4;
            optimal_blocks = properties.multiProcessorCount * 16;
            hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 2;
        } else {
            kernel_type = KERNEL_STANDARD;
            optimal_threads = 256;
            warps_per_block = 8;
            optimal_blocks = properties.multiProcessorCount * 8;
            hashes_per_kernel_launch = (uint64_t) optimal_blocks * optimal_threads * 4;
        }

        // Check available memory
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        // Reserve some memory for system
        free_mem = (size_t) (free_mem * 0.9);

        size_t mem_per_stream = sizeof(uint64_t) * 4 * RING_SIZE + sizeof(uint32_t);
        int num_streams = std::min(STREAMS_PER_GPU, (int) (free_mem / mem_per_stream));

        // Create streams and allocate memory
        streams.resize(num_streams);
        d_pairs.resize(num_streams);
        d_tickets.resize(num_streams);

        for (int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, i % 2));
            CUDA_CHECK(cudaMalloc(&d_pairs[i], sizeof(uint64_t) * 4 * RING_SIZE));
            CUDA_CHECK(cudaMalloc(&d_tickets[i], sizeof(uint32_t)));
            CUDA_CHECK(cudaMemsetAsync(d_tickets[i], 0, sizeof(uint32_t), streams[i]));
        }

        // Set cache configuration based on kernel
        switch (kernel_type) {
            case KERNEL_STANDARD:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_mining_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_WARP_COLLABORATIVE:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_warp_collaborative_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_VECTORIZED:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_vectorized_kernel, cudaFuncCachePreferShared));
                break;
            case KERNEL_BITSLICED:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_bitsliced_kernel_correct, cudaFuncCachePreferL1));
                break;
            case KERNEL_HASHCAT:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_hashcat_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_HASHCAT_EXTREME:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_hashcat_extreme_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_COOPERATIVE_GROUPS:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_cooperative_groups_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_MULTI_HASH:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_multi_hash_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_READONLY_CACHE:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_readonly_cache_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_SIMD_VECTORIZED:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_simd_vectorized_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_HYBRID_WARP_SIMD:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_hybrid_warp_simd_kernel, cudaFuncCachePreferL1));
                break;
            case KERNEL_LDG_OPTIMIZED:
                CUDA_CHECK(cudaFuncSetCacheConfig(sha1_ldg_optimized_kernel, cudaFuncCachePreferL1));
                break;
        }

        printInfo();
    }

    ~GPUContext() {
        CUDA_CHECK(cudaSetDevice(device_id));
        for (size_t i = 0; i < streams.size(); i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaFree(d_pairs[i]));
            CUDA_CHECK(cudaFree(d_tickets[i]));
        }
    }

    void printInfo() {
        std::cout << "=== GPU " << device_id << " Configuration ===\n";
        std::cout << "Device        : " << properties.name << "\n";
        std::cout << "Compute Cap   : " << properties.major << "." << properties.minor << "\n";
        std::cout << "SMs           : " << properties.multiProcessorCount << "\n";
        std::cout << "Clock Rate    : " << properties.clockRate / 1000 << " MHz\n";
        std::cout << "Memory        : " << properties.totalGlobalMem / (1024.0 * 1024 * 1024) << " GB\n";
        std::cout << "Memory Clock  : " << properties.memoryClockRate / 1000 << " MHz\n";
        std::cout << "Memory Bus    : " << properties.memoryBusWidth << " bits\n";
        std::cout << "L2 Cache      : " << properties.l2CacheSize / (1024 * 1024) << " MB\n";

        std::cout << "Kernel        : " << getKernelName(kernel_type) << "\n";
        std::cout << "Configuration : " << optimal_blocks << " blocks × " << optimal_threads << " threads\n";

        // Display kernel-specific information
        switch (kernel_type) {
            case KERNEL_STANDARD:
            case KERNEL_READONLY_CACHE:
                std::cout << "Work per thread: 4 hashes\n";
                break;
            case KERNEL_WARP_COLLABORATIVE:
                std::cout << "Work per warp: 8 batches × 32 hashes = 256 hashes\n";
                break;
            case KERNEL_VECTORIZED:
                std::cout << "Work per thread: 2 hashes\n";
                break;
            case KERNEL_BITSLICED:
                std::cout << "Work per warp: 4 batches × 32 messages = 128 hashes\n";
                break;
            case KERNEL_HASHCAT:
                std::cout << "Work per thread: 16 hashes\n";
                break;
            case KERNEL_HASHCAT_EXTREME:
                std::cout << "Work per thread: 32 hashes\n";
                break;
            case KERNEL_COOPERATIVE_GROUPS:
                std::cout << "Work per warp: 10 batches × 32 hashes = 320 hashes\n";
                break;
            case KERNEL_MULTI_HASH:
                std::cout << "Work per thread: 8 hashes\n";
                break;
            case KERNEL_SIMD_VECTORIZED:
                std::cout << "Work per thread: 2 vectors × 4 hashes = 8 hashes\n";
                break;
            case KERNEL_HYBRID_WARP_SIMD:
                std::cout << "Work per warp: 4 batches × 64 hashes = 256 hashes\n";
                break;
            case KERNEL_LDG_OPTIMIZED:
                std::cout << "Work per thread: 6 hashes\n";
                break;
        }

        std::cout << "Hashes/Launch : " << hashes_per_kernel_launch << "\n";
        std::cout << "Streams       : " << streams.size() << "\n\n";
    }

    void launchKernel(cudaStream_t stream, uint64_t seed) {
        int stream_idx = getStreamIndex(stream);

        switch (kernel_type) {
            case KERNEL_STANDARD:
                sha1_mining_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_WARP_COLLABORATIVE:
                sha1_warp_collaborative_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_VECTORIZED:
                sha1_vectorized_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_BITSLICED:
                sha1_bitsliced_kernel_correct<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_HASHCAT:
                sha1_hashcat_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_HASHCAT_EXTREME:
                sha1_hashcat_extreme_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_COOPERATIVE_GROUPS:
                sha1_cooperative_groups_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_MULTI_HASH:
                sha1_multi_hash_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_READONLY_CACHE:
                sha1_readonly_cache_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_SIMD_VECTORIZED:
                sha1_simd_vectorized_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_HYBRID_WARP_SIMD:
                sha1_hybrid_warp_simd_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
            case KERNEL_LDG_OPTIMIZED:
                sha1_ldg_optimized_kernel<<<optimal_blocks, optimal_threads, 0, stream>>>(
                    d_pairs[stream_idx], d_tickets[stream_idx], seed);
                break;
        }

        kernels_launched++;
        g_total_kernel_launches.fetch_add(1);
    }

private:
    int getStreamIndex(cudaStream_t stream) {
        for (size_t i = 0; i < streams.size(); i++) {
            if (streams[i] == stream) return i;
        }
        return 0;
    }
};

// ==================== Result Handler ====================
class ResultHandler {
private:
    std::ofstream output_file;
    std::mutex file_mutex;
    std::array<uint8_t, 20> target_hash;
    std::array<uint8_t, 32> base_message;

public:
    ResultHandler(const std::string &filename, const uint8_t target[20], const uint8_t base_msg[32]) {
        std::copy(target, target + 20, target_hash.begin());
        std::copy(base_msg, base_msg + 32, base_message.begin());

        output_file.open(filename, std::ios::app);
        if (!output_file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << "\n";
        }

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        output_file << "\n=== SHA-1 Collision Mining Started at "
                << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                << " ===\n";
        output_file << "Target: ";
        for (int i = 0; i < 20; i++) {
            output_file << std::hex << std::setw(2) << std::setfill('0') << (int) target_hash[i];
        }
        output_file << "\n\n";
        output_file.flush();
    }

    void saveResults(uint64_t *h_pairs, uint32_t count, int gpu_id) {
        std::lock_guard<std::mutex> lock(file_mutex);

        output_file << "[GPU " << gpu_id << "] Found " << count << " collision candidates:\n";

        for (uint32_t i = 0; i < std::min(count, 100u); i++) {
            output_file << "Candidate " << i + 1 << ":\n";

            // Reconstruct message
            uint8_t msg[32];
            for (int j = 0; j < 4; j++) {
                uint64_t word = h_pairs[i * 4 + j];
                for (int k = 0; k < 8; k++) {
                    msg[j * 8 + k] = (word >> (k * 8)) & 0xFF;
                }
            }

            output_file << "  Message: ";
            for (int j = 0; j < 32; j++) {
                output_file << std::hex << std::setw(2) << std::setfill('0') << (int) msg[j];
            }
            output_file << "\n";

            // Verify SHA-1
            uint8_t hash[20];
            sha1_ctx ctx;
            sha1_init(ctx);
            sha1_update(ctx, msg, 32);
            sha1_final(ctx, hash);

            output_file << "  SHA-1:   ";
            for (int j = 0; j < 20; j++) {
                output_file << std::hex << std::setw(2) << std::setfill('0') << (int) hash[j];
            }

            bool verified = (std::memcmp(hash, target_hash.data(), 20) == 0);
            output_file << " [" << (verified ? "VERIFIED" : "FAILED") << "]\n\n";
        }

        output_file.flush();

        // Console notification
        std::cout << "\n[GPU " << gpu_id << "] Found " << count << " collision candidates! ";
        std::cout << "Saved to output file.\n";
    }
};

// ==================== Performance Monitor ====================
class PerformanceMonitor {
private:
    std::chrono::steady_clock::time_point last_update;
    uint64_t last_hashes = 0;
    uint64_t last_kernel_launches = 0;
    std::mutex monitor_mutex;
    std::vector<double> rate_history;
    const size_t history_size = 12; // 1 minute of 5-second intervals

public:
    PerformanceMonitor() {
        last_update = std::chrono::steady_clock::now();
        rate_history.reserve(history_size);
    }

    void update(const std::vector<std::unique_ptr<GPUContext> > &gpus) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);

        if (elapsed.count() >= PROGRESS_INTERVAL) {
            std::lock_guard<std::mutex> lock(monitor_mutex);

            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time);
            double total_seconds = total_elapsed.count();

            uint64_t current_hashes = g_total_hashes.load();
            uint64_t current_kernel_launches = g_total_kernel_launches.load();
            uint64_t interval_hashes = current_hashes - last_hashes;
            uint64_t interval_kernel_launches = current_kernel_launches - last_kernel_launches;

            double interval_rate = interval_hashes / elapsed.count() / 1e9;
            double average_rate = current_hashes / total_seconds / 1e9;
            double kernel_rate = interval_kernel_launches / (double) elapsed.count();

            // Update rate history
            rate_history.push_back(interval_rate);
            if (rate_history.size() > history_size) {
                rate_history.erase(rate_history.begin());
            }

            // Calculate moving average
            double moving_avg = 0;
            for (double rate: rate_history) {
                moving_avg += rate;
            }
            moving_avg /= rate_history.size();

            std::cout << "\r[" << formatTime(total_seconds) << "] "
                    << "Hashes: " << std::fixed << std::setprecision(2)
                    << current_hashes / 1e12 << "T | "
                    << "Rate: " << interval_rate << " GH/s (avg: "
                    << average_rate << " GH/s, MA: " << moving_avg << ") | "
                    << "Kernels/s: " << std::fixed << std::setprecision(0) << kernel_rate << " | "
                    << "Candidates: " << g_total_candidates.load();

            // Per-GPU stats
            std::cout << " | GPU: ";
            for (size_t i = 0; i < gpus.size(); i++) {
                double gpu_rate = gpus[i]->hashes_processed.load() / total_seconds / 1e9;
                std::cout << "[" << i << ":" << std::fixed << std::setprecision(1)
                        << gpu_rate << "] ";
            }
            std::cout << "    " << std::flush;

            last_update = now;
            last_hashes = current_hashes;
            last_kernel_launches = current_kernel_launches;
        }
    }

    void finalReport() {
        auto now = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time);
        double total_seconds = total_elapsed.count();
        uint64_t total_hashes = g_total_hashes.load();
        uint64_t total_kernel_launches = g_total_kernel_launches.load();
        double average_rate = total_hashes / total_seconds / 1e9;

        std::cout << "\n\n=== Final Statistics ===\n";
        std::cout << "Total Time    : " << formatTime(total_seconds) << "\n";
        std::cout << "Total Hashes  : " << std::fixed << std::setprecision(3)
                << total_hashes / 1e12 << " trillion\n";
        std::cout << "Average Rate  : " << average_rate << " GH/s\n";
        std::cout << "Kernel Launches: " << total_kernel_launches << "\n";
        std::cout << "Avg Kernel Time: " << std::fixed << std::setprecision(2)
                << (total_seconds * 1000.0) / total_kernel_launches << " ms\n";
        std::cout << "Candidates    : " << g_total_candidates.load() << "\n";
        std::cout << "Efficiency    : " << std::scientific << std::setprecision(2)
                << (double) g_total_candidates.load() / total_hashes * 100 << "%\n";
    }

private:
    std::string formatTime(double seconds) {
        int hours = seconds / 3600;
        int minutes = (seconds - hours * 3600) / 60;
        int secs = seconds - hours * 3600 - minutes * 60;

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(2) << hours << ":"
                << std::setw(2) << minutes << ":"
                << std::setw(2) << secs;
        return ss.str();
    }
};

// ==================== GPU Worker Thread ====================
void gpu_worker(GPUContext *gpu, ResultHandler *results, uint64_t base_seed) {
    CUDA_CHECK(cudaSetDevice(gpu->device_id));

    uint64_t local_seed = base_seed + gpu->device_id * (1ull << 48);

    // Allocate pinned host memory for faster transfers
    std::vector<uint64_t> h_pairs;
    h_pairs.resize(RING_SIZE * 4);

    // Create events for accurate timing
    std::vector<cudaEvent_t> start_events(gpu->streams.size());
    std::vector<cudaEvent_t> stop_events(gpu->streams.size());
    std::vector<bool> kernel_running(gpu->streams.size(), false);
    std::vector<uint64_t> kernel_seeds(gpu->streams.size());

    for (size_t i = 0; i < gpu->streams.size(); i++) {
        CUDA_CHECK(cudaEventCreate(&start_events[i]));
        CUDA_CHECK(cudaEventCreate(&stop_events[i]));
    }

    // Keep track of kernels launched for debugging
    uint64_t local_kernels_launched = 0;
    auto last_debug_time = std::chrono::steady_clock::now();

    while (!g_shutdown.load()) {
        // Launch kernels on available streams
        for (size_t s = 0; s < gpu->streams.size(); s++) {
            if (!kernel_running[s]) {
                // Record start time
                CUDA_CHECK(cudaEventRecord(start_events[s], gpu->streams[s]));

                // Launch kernel
                gpu->launchKernel(gpu->streams[s], local_seed);
                kernel_seeds[s] = local_seed;
                local_seed += gpu->hashes_per_kernel_launch;

                // Record stop time
                CUDA_CHECK(cudaEventRecord(stop_events[s], gpu->streams[s]));

                kernel_running[s] = true;
                local_kernels_launched++;
            }
        }

        // Check completed kernels
        for (size_t s = 0; s < gpu->streams.size(); s++) {
            if (kernel_running[s]) {
                cudaError_t err = cudaEventQuery(stop_events[s]);
                if (err == cudaSuccess) {
                    // Kernel completed
                    kernel_running[s] = false;

                    // Check for results
                    uint32_t found = 0;
                    CUDA_CHECK(cudaMemcpyAsync(&found, gpu->d_tickets[s], sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, gpu->streams[s]));
                    CUDA_CHECK(cudaStreamSynchronize(gpu->streams[s]));

                    if (found > 0) {
                        g_total_candidates.fetch_add(found);

                        uint32_t to_copy = std::min(found, RING_SIZE);
                        CUDA_CHECK(cudaMemcpyAsync(h_pairs.data(), gpu->d_pairs[s],
                            sizeof(uint64_t) * 4 * to_copy, cudaMemcpyDeviceToHost, gpu->streams[s]));
                        CUDA_CHECK(cudaStreamSynchronize(gpu->streams[s]));

                        results->saveResults(h_pairs.data(), to_copy, gpu->device_id);

                        // Reset ticket
                        CUDA_CHECK(cudaMemsetAsync(gpu->d_tickets[s], 0, sizeof(uint32_t), gpu->streams[s]));
                    }

                    // Update hash count
                    g_total_hashes.fetch_add(gpu->hashes_per_kernel_launch);
                    gpu->hashes_processed.fetch_add(gpu->hashes_per_kernel_launch);
                } else if (err != cudaErrorNotReady) {
                    CUDA_CHECK(err);
                }
            }
        }

        // Periodic debug output (every 30 seconds)
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_debug_time);
        if (elapsed.count() >= 30 && gpu->device_id == 0) {
            double kernels_per_sec = local_kernels_launched / (double) elapsed.count();
            std::cout << "\n[GPU 0 Debug] Kernels: " << local_kernels_launched
                    << " in " << elapsed.count() << "s = "
                    << std::fixed << std::setprecision(1) << kernels_per_sec
                    << " kernels/sec, Expected: "
                    << (kernels_per_sec * gpu->hashes_per_kernel_launch / 1e9)
                    << " GH/s\n" << std::flush;
            local_kernels_launched = 0;
            last_debug_time = now;
        }

        // Small yield to prevent CPU spinning
        std::this_thread::yield();
    }

    // Cleanup
    for (auto &stream: gpu->streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    for (size_t i = 0; i < gpu->streams.size(); i++) {
        CUDA_CHECK(cudaEventDestroy(start_events[i]));
        CUDA_CHECK(cudaEventDestroy(stop_events[i]));
    }
}

// ==================== Benchmark Utilities ====================
void benchmarkKernel(GPUContext *gpu, int warmup_runs = 10, int test_runs = 100) {
    CUDA_CHECK(cudaSetDevice(gpu->device_id));

    std::cout << "\n=== Benchmarking Kernel Performance ===\n";
    std::cout << "Kernel: " << getKernelName(gpu->kernel_type) << "\n";
    std::cout << "Configuration: " << gpu->optimal_blocks << " blocks × "
            << gpu->optimal_threads << " threads\n";
    std::cout << "Hashes per launch: " << gpu->hashes_per_kernel_launch << "\n\n";

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        gpu->launchKernel(gpu->streams[0], i * gpu->hashes_per_kernel_launch);
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu->streams[0]));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Time the kernel
    CUDA_CHECK(cudaEventRecord(start, gpu->streams[0]));

    for (int i = 0; i < test_runs; i++) {
        gpu->launchKernel(gpu->streams[0], i * gpu->hashes_per_kernel_launch);
    }

    CUDA_CHECK(cudaEventRecord(stop, gpu->streams[0]));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    double seconds = milliseconds / 1000.0;
    uint64_t total_hashes = (uint64_t) test_runs * gpu->hashes_per_kernel_launch;
    double ghps = total_hashes / seconds / 1e9;

    std::cout << "Results:\n";
    std::cout << "  Total time: " << milliseconds << " ms\n";
    std::cout << "  Time per kernel: " << milliseconds / test_runs << " ms\n";
    std::cout << "  Total hashes: " << total_hashes << "\n";
    std::cout << "  Performance: " << ghps << " GH/s\n";
    std::cout << "  Theoretical max (all streams): " << ghps * gpu->streams.size()
            << " GH/s\n\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ==================== Configuration Parser ====================
struct Config {
    std::vector<int> gpu_ids;
    std::string output_file = "sha1_collisions.txt";
    std::array<uint8_t, 32> target_preimage;
    bool benchmark_mode = false;
    int benchmark_seconds = 60;
    bool force_bitsliced = false;
    bool no_hashcat = false;
    int force_kernel = -1; // -1 = auto, 0-11 = specific kernel
    bool quick_bench = false;
};

Config parse_arguments(int argc, char **argv) {
    Config config;

    // Initialize with zeros
    config.target_preimage.fill(0);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--gpu" && i + 1 < argc) {
            config.gpu_ids.push_back(std::stoi(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            std::string hex = argv[++i];
            if (hex.length() != 64) {
                std::cerr << "Error: Target must be 64 hex characters (32 bytes)\n";
                std::exit(1);
            }
            for (int j = 0; j < 32; j++) {
                std::string byte = hex.substr(j * 2, 2);
                config.target_preimage[j] = std::stoi(byte, nullptr, 16);
            }
        } else if (arg == "--benchmark") {
            config.benchmark_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                config.benchmark_seconds = std::stoi(argv[++i]);
            }
        } else if (arg == "--quick-bench") {
            config.quick_bench = true;
        } else if (arg == "--bitsliced") {
            config.force_bitsliced = true;
        } else if (arg == "--no-hashcat") {
            config.no_hashcat = true;
        } else if (arg == "--kernel" && i + 1 < argc) {
            config.force_kernel = std::stoi(argv[++i]);
        } else if (arg == "--list-kernels") {
            std::cout << "Available kernels:\n";
            std::cout << "  0: Standard (4 hashes/thread)\n";
            std::cout << "  1: Warp Collaborative (256 hashes/warp) - FASTEST on modern GPUs\n";
            std::cout << "  2: Vectorized (2 hashes/thread)\n";
            std::cout << "  3: Bitsliced (128 hashes/warp)\n";
            std::cout << "  4: HashCat (16 hashes/thread)\n";
            std::cout << "  5: HashCat Extreme (32 hashes/thread) - Best for Ampere+\n";
            std::cout << "  6: Cooperative Groups (320 hashes/warp)\n";
            std::cout << "  7: Multi-Hash (8 hashes/thread)\n";
            std::cout << "  8: Read-Only Cache (4 hashes/thread)\n";
            std::cout << "  9: SIMD Vectorized (8 hashes/thread)\n";
            std::cout << " 10: Hybrid Warp-SIMD (256 hashes/warp)\n";
            std::cout << " 11: LDG Optimized (6 hashes/thread)\n";
            std::exit(0);
        } else if (arg == "--help") {
            std::cout << "SHA-1 Collision Miner v3.0\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --gpu <id>       GPU device ID (can be used multiple times)\n";
            std::cout << "  --output <file>  Output file (default: sha1_collisions.txt)\n";
            std::cout << "  --target <hex>   Target preimage in hex (default: zeros)\n";
            std::cout << "  --benchmark [s]  Run benchmark for s seconds (default: 60)\n";
            std::cout << "  --quick-bench    Run quick kernel benchmark\n";
            std::cout << "  --bitsliced      Force bitsliced kernel (slower, for testing)\n";
            std::cout << "  --no-hashcat     Use original kernels instead of HashCat-style\n";
            std::cout << "  --kernel <id>    Force specific kernel (0-11, see --list-kernels)\n";
            std::cout << "  --list-kernels   List all available kernels\n";
            std::cout << "  --help           Show this help\n\n";
            std::cout << "Example:\n";
            std::cout << "  " << argv[0] << " --gpu 0 --gpu 1 --target ";
            std::cout << "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            std::exit(1);
        }
    }

    // Default: use all available GPUs
    if (config.gpu_ids.empty()) {
        int num_gpus;
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        for (int i = 0; i < num_gpus; i++) {
            config.gpu_ids.push_back(i);
        }
    }

    return config;
}

// ==================== Main Function ====================
int main(int argc, char **argv) {
    setup_signal_handlers();

    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|      SHA-1 Collision Miner v3.0         |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Parse configuration
    Config config = parse_arguments(argc, argv);

    // Calculate target SHA-1
    uint8_t target_hash[20];
    sha1_ctx ctx;
    sha1_init(ctx);
    sha1_update(ctx, config.target_preimage.data(), 32);
    sha1_final(ctx, target_hash);

    // Convert to uint32_t array for GPU
    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(target_hash[4 * i]) << 24) |
                    (uint32_t(target_hash[4 * i + 1]) << 16) |
                    (uint32_t(target_hash[4 * i + 2]) << 8) |
                    uint32_t(target_hash[4 * i + 3]);
    }

    // Upload job to GPU constant memory
    upload_new_job(config.target_preimage.data(), target);

    std::cout << "Target Preimage: ";
    for (int i = 0; i < 32; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int) config.target_preimage[i];
    }
    std::cout << "\n";

    std::cout << "Target SHA-1: ";
    for (int i = 0; i < 20; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int) target_hash[i];
    }
    std::cout << std::dec << "\n\n";

    if (config.force_bitsliced) {
        std::cout << "NOTE: Using bitsliced kernel (slower, for testing)\n\n";
    } else if (config.force_kernel >= 0) {
        std::cout << "NOTE: Forcing kernel " << config.force_kernel << " ("
                << getKernelName((KernelType) config.force_kernel) << ")\n\n";
    }

    // Initialize GPUs
    std::vector<std::unique_ptr<GPUContext> > gpus;
    for (int id: config.gpu_ids) {
        try {
            gpus.push_back(std::make_unique<GPUContext>(
                id, config.force_bitsliced, !config.no_hashcat, config.force_kernel));
        } catch (const std::exception &e) {
            std::cerr << "Failed to initialize GPU " << id << ": " << e.what() << "\n";
        }
    }

    if (gpus.empty()) {
        std::cerr << "No GPUs available for mining!\n";
        return 1;
    }

    // Run quick benchmark if requested
    if (config.quick_bench) {
        for (auto &gpu: gpus) {
            benchmarkKernel(gpu.get(), 10, 100);
        }
        return 0;
    }

    // Initialize result handler
    ResultHandler results(config.output_file, target_hash, config.target_preimage.data());

    // Initialize performance monitor
    PerformanceMonitor monitor;

    // Start timing
    g_start_time = std::chrono::steady_clock::now();

    // Generate random base seed
    std::random_device rd;
    uint64_t base_seed = (uint64_t(rd()) << 32) | rd();

    std::cout << "Starting collision search with " << gpus.size() << " GPU(s)...\n";
    std::cout << "Base seed: 0x" << std::hex << base_seed << std::dec << "\n";

    if (config.benchmark_mode) {
        std::cout << "Running in benchmark mode for " << config.benchmark_seconds << " seconds...\n";
    }
    std::cout << "\n";

    // Launch GPU worker threads
    std::vector<std::thread> workers;
    for (auto &gpu: gpus) {
        workers.emplace_back(gpu_worker, gpu.get(), &results, base_seed);
    }

    // Optionally set main thread to high priority
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#else
    struct sched_param param;
    param.sched_priority = 1;
    pthread_setschedparam(pthread_self(), SCHED_RR, &param);
#endif

    // Monitor progress
    if (config.benchmark_mode) {
        auto benchmark_end = std::chrono::steady_clock::now() +
                             std::chrono::seconds(config.benchmark_seconds);

        while (!g_shutdown.load() && std::chrono::steady_clock::now() < benchmark_end) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            monitor.update(gpus);
        }
        g_shutdown.store(true);
    } else {
        while (!g_shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            monitor.update(gpus);
        }
    }

    // Wait for workers to finish
    for (auto &worker: workers) {
        worker.join();
    }

    // Final report
    monitor.finalReport();

    // Additional benchmark statistics
    if (config.benchmark_mode) {
        auto total_time = std::chrono::steady_clock::now() - g_start_time;
        double seconds = std::chrono::duration<double>(total_time).count();
        uint64_t total_hashes = g_total_hashes.load();

        std::cout << "\n=== Benchmark Results ===\n";
        std::cout << "Total GPUs    : " << gpus.size() << "\n";
        std::cout << "Duration      : " << std::fixed << std::setprecision(2) << seconds << " seconds\n";
        std::cout << "Total Hashes  : " << total_hashes / 1e12 << " trillion\n";
        std::cout << "Performance   : " << total_hashes / seconds / 1e9 << " GH/s\n";
        std::cout << "Per GPU       : " << total_hashes / seconds / 1e9 / gpus.size() << " GH/s\n";

        // Detailed per-GPU stats
        std::cout << "\nPer-GPU Performance:\n";
        for (size_t i = 0; i < gpus.size(); i++) {
            uint64_t gpu_hashes = gpus[i]->hashes_processed.load();
            double gpu_ghps = gpu_hashes / seconds / 1e9;
            uint64_t gpu_kernels = gpus[i]->kernels_launched.load();

            std::cout << "  GPU " << gpus[i]->device_id << " ("
                    << gpus[i]->properties.name << ", "
                    << getKernelName(gpus[i]->kernel_type) << "):\n";
            std::cout << "    Performance: " << gpu_ghps << " GH/s\n";
            std::cout << "    Kernels launched: " << gpu_kernels << "\n";
            std::cout << "    Avg kernel time: " << (seconds * 1000.0) / gpu_kernels << " ms\n";
        }
    }

    if (g_total_candidates.load() > 0) {
        std::cout << "\n[SUCCESS] Found " << g_total_candidates.load() << " collisions!\n";
        std::cout << "Results saved to: " << config.output_file << "\n";
    } else if (!config.benchmark_mode) {
        std::cout << "\n[CONTINUING] No collisions found yet. Keep mining!\n";
    }

    return 0;
}
