#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <array>
#include <chrono>
#include "job_upload_api.h"
#include "cxxsha1.hpp"
#include "job_constants.cuh"

// Forward declaration of optimized kernel
extern "C" __global__
void sha1_double_kernel(uint8_t *, uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(x)                                                     \
    do { cudaError_t e = (x);                                             \
         if (e != cudaSuccess) {                                          \
             std::cerr << "CUDA Error: " << cudaGetErrorString(e)        \
                       << " at line " << __LINE__ << '\n';                \
             std::exit(1);                                                \
         }                                                                \
    } while (0)

/* --------------------------------------------------------------------- */
constexpr uint32_t BATCH = 1u << 24; // 16M threads (increased for better GPU utilization)
constexpr int THREADS = 256; // Optimal for modern GPUs
constexpr uint32_t RING_SIZE = 1u << 20; // Ring buffer slots
constexpr int WARMUP_RUNS = 10; // Warmup iterations

// Function to get optimal block configuration for current GPU
void getOptimalLaunchConfig(int &blocks, int &threads, uint32_t &batch_size, int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Calculate optimal configuration based on GPU
    int maxBlocksPerSM = 0;

    if (prop.major >= 8) {
        // Ampere and newer
        threads = 256;
        maxBlocksPerSM = 4;
    } else if (prop.major >= 7) {
        // Volta/Turing
        threads = 256;
        maxBlocksPerSM = 2;
    } else {
        threads = 1024;
        maxBlocksPerSM = 2;
    }

    blocks = prop.multiProcessorCount * maxBlocksPerSM;

    // Adjust batch size based on GPU capability
    if (prop.multiProcessorCount < 40) {
        batch_size = 1u << 22; // 4M for low-end GPUs
    } else if (prop.multiProcessorCount < 80) {
        batch_size = 1u << 23; // 8M for mid-range GPUs
    } else if (prop.multiProcessorCount < 120) {
        batch_size = 1u << 24; // 16M for high-end GPUs
    } else {
        batch_size = 1u << 25; // 32M for top-tier GPUs (RTX 4090, A100)
    }

    std::cout << "=== GPU Configuration ===\n";
    std::cout << "Device         : " << prop.name << "\n";
    std::cout << "Compute Cap    : " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs            : " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads/SM : " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Memory Clock   : " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Memory Bus     : " << prop.memoryBusWidth << " bits\n";
    std::cout << "L2 Cache       : " << prop.l2CacheSize / 1024 / 1024 << " MB\n\n";
}

int main(int argc, char **argv) {
    int iterations = 100;

    // Parse command line arguments
    if (argc > 1) {
        char *end = nullptr;
        long tmp = std::strtol(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && tmp > 0) {
            iterations = static_cast<int>(tmp);
        }
    }

    std::cout << "SHA-1 Double Hash Benchmark\n";

    /* ------------------------------------------------------------------ */
    /* 1. Setup test preimage and target                                 */
    /* ------------------------------------------------------------------ */
    std::array<uint8_t, 32> preimage = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
    };

    // Compute double SHA-1
    uint8_t d1[20], d2[20];
    sha1_ctx c{};
    sha1_init(c);
    sha1_update(c, preimage.data(), 32);
    sha1_final(c, d1);
    sha1_init(c);
    sha1_update(c, d1, 20);
    sha1_final(c, d2);

    uint32_t target[5];
    for (int i = 0; i < 5; ++i) {
        target[i] = (static_cast<uint32_t>(d2[4 * i]) << 24) |
                    (static_cast<uint32_t>(d2[4 * i + 1]) << 16) |
                    (static_cast<uint32_t>(d2[4 * i + 2]) << 8) |
                    static_cast<uint32_t>(d2[4 * i + 3]);
    }

    // Upload job to GPU
    upload_new_job(preimage.data(), target);

    std::cout << "Target SHA-1: ";
    for (uint8_t b: d2) std::printf("%02x", b);
    std::cout << "\n\n";

    /* ------------------------------------------------------------------ */
    /* 2. Get optimal launch configuration                                */
    /* ------------------------------------------------------------------ */
    int optimal_blocks, optimal_threads;
    uint32_t batch_size;
    getOptimalLaunchConfig(optimal_blocks, optimal_threads, batch_size);

    const dim3 blockDim(optimal_threads);
    const dim3 gridDim((batch_size + optimal_threads - 1) / optimal_threads);

    std::cout << "=== Launch Configuration ===\n";
    std::cout << "Batch Size     : " << batch_size << " (" << batch_size / 1e6 << "M threads)\n";
    std::cout << "Grid           : " << gridDim.x << " blocks x " << blockDim.x << " threads\n";
    std::cout << "Total Threads  : " << gridDim.x * blockDim.x << "\n";
    std::cout << "Iterations     : " << iterations << "\n\n";

    /* ------------------------------------------------------------------ */
    /* 3. Allocate GPU memory                                             */
    /* ------------------------------------------------------------------ */
    uint64_t *d_pairs = nullptr;
    uint32_t *d_ticket = nullptr;

    size_t pairs_size = sizeof(uint64_t) * 4 * RING_SIZE;
    CUDA_CHECK(cudaMalloc(&d_pairs, pairs_size));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    // Set cache configuration for better performance
    CUDA_CHECK(cudaFuncSetCacheConfig(sha1_double_kernel, cudaFuncCachePreferL1));

    // Get GPU properties for metrics
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    /* ------------------------------------------------------------------ */
    /* 4. Warmup runs                                                     */
    /* ------------------------------------------------------------------ */
    std::cout << "=== Warming up GPU... ===\n";

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        sha1_double_kernel<<<gridDim, blockDim>>>(
            nullptr, d_pairs, d_ticket, 0xDEADBEEF + i
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    std::cout << "Warmup complete.\n\n";

    /* ------------------------------------------------------------------ */
    /* 5. Main benchmark                                                  */
    /* ------------------------------------------------------------------ */
    std::cout << "=== Running Benchmark ===\n";

    // CPU timing for comparison
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // GPU timing with events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernels
    for (int i = 0; i < iterations; ++i) {
        sha1_double_kernel<<<gridDim, blockDim>>>(
            nullptr, d_pairs, d_ticket, 0xCAFEBABEULL + i
        );
        CUDA_CHECK(cudaPeekAtLastError());

        // Progress indicator for long runs
        if (iterations > 100 && i % (iterations / 10) == 0) {
            std::cout << "." << std::flush;
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    auto cpu_end = std::chrono::high_resolution_clock::now();

    if (iterations > 100) std::cout << "\n";

    // Get timing results
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    double cpu_ms = cpu_duration.count() / 1000.0;

    /* ------------------------------------------------------------------ */
    /* 6. Calculate and display results                                   */
    /* ------------------------------------------------------------------ */
    double total_threads = static_cast<double>(gridDim.x) * blockDim.x;
    double total_hashes = total_threads * iterations;
    double gpu_seconds = gpu_ms / 1000.0;
    double ghps = total_hashes / gpu_seconds / 1e9;
    double mhps = ghps * 1000.0;

    // Get number of candidates found
    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(found), cudaMemcpyDeviceToHost));

    std::cout << "\n=== Performance Results ===\n";
    std::cout << "GPU Time       : " << gpu_seconds << " seconds\n";
    std::cout << "CPU Time       : " << cpu_ms / 1000.0 << " seconds\n";
    std::cout << "Total Hashes   : " << total_hashes / 1e9 << " billion\n";
    std::cout << "Hash Rate      : " << ghps << " GH/s (" << mhps << " MH/s)\n";
    std::cout << "Per Thread     : " << iterations << " hashes\n";
    std::cout << "Candidates     : " << found << "\n\n";

    /* ------------------------------------------------------------------ */
    /* 7. Advanced metrics                                                */
    /* ------------------------------------------------------------------ */
    std::cout << "=== Efficiency Analysis ===\n";

    // Memory bandwidth calculation
    // Each hash: 32B input read + minimal output writes
    double memory_traffic_gb = (total_hashes * 32) / 1e9;
    double memory_bandwidth = memory_traffic_gb / gpu_seconds;
    double theoretical_bandwidth = prop.memoryClockRate * 2.0 * (prop.memoryBusWidth / 8) / 1e6;

    std::cout << "Memory BW Used : " << memory_bandwidth << " GB/s\n";
    std::cout << "Memory BW Max  : " << theoretical_bandwidth << " GB/s\n";
    std::cout << "Memory Effic.  : " << (memory_bandwidth / theoretical_bandwidth * 100) << "%\n";

    // Compute efficiency
    double clock_ghz = prop.clockRate / 1e6;
    double sm_count = prop.multiProcessorCount;
    double ops_per_hash = 80 * 5; // 80 rounds, ~5 ops per round
    double actual_gops = ghps * ops_per_hash;
    double theoretical_gops = sm_count * 64 * clock_ghz; // 64 ops/clock/SM for INT32

    std::cout << "Compute Used   : " << actual_gops << " Gops/s\n";
    std::cout << "Compute Max    : " << theoretical_gops << " Gops/s\n";
    std::cout << "Compute Effic. : " << (actual_gops / theoretical_gops * 100) << "%\n";

    // Occupancy
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int active_threads_per_sm = (optimal_threads * optimal_blocks) / sm_count;
    double occupancy = static_cast<double>(active_threads_per_sm) / max_threads_per_sm * 100;

    std::cout << "Occupancy      : " << occupancy << "%\n";

    // Power efficiency
    /*unsigned int power_mw = 0;
    if (cudaDeviceGetAttribute((int *) &power_mw, cudaDevAttrMaxPower, 0) == cudaSuccess) {
        double power_w = power_mw / 1000.0;
        double hashes_per_watt = mhps * 1e6 / power_w;
        std::cout << "Power Draw     : " << power_w << " W (TDP)\n";
        std::cout << "Efficiency     : " << hashes_per_watt / 1e6 << " MH/W\n";
    }*/

    /* ------------------------------------------------------------------ */
    /* 8. Display found candidates (if any)                               */
    /* ------------------------------------------------------------------ */
    if (found > 0) {
        std::cout << "\n=== Found Candidates ===\n";

        uint32_t to_show = std::min(found, 5u);
        std::vector<uint64_t> candidates(to_show * 4);

        CUDA_CHECK(cudaMemcpy(candidates.data(), d_pairs,
            sizeof(uint64_t) * 4 * to_show,
            cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < to_show; ++i) {
            std::cout << "Candidate " << i + 1 << ": ";
            for (int j = 0; j < 4; ++j) {
                std::printf("%016llx", candidates[i * 4 + j]);
            }
            std::cout << "\n";
        }

        if (found > to_show) {
            std::cout << "... and " << (found - to_show) << " more.\n";
        }
    }

    /* ------------------------------------------------------------------ */
    /* 9. Cleanup                                                         */
    /* ------------------------------------------------------------------ */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));


    std::cout << "------ Benchmark Complete! ------\n";

    return 0;
}
