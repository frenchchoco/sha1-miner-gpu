#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <sstream>
#include <fstream>
#include "cxxsha1.hpp"
#include "job_upload_api.h"

// Declare all kernel functions
extern "C" __global__ void sha1_mining_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_warp_collaborative_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_vectorized_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_bitsliced_kernel_correct(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_hashcat_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_hashcat_extreme_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_cooperative_groups_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_multi_hash_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_readonly_cache_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_simd_vectorized_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_hybrid_warp_simd_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_ldg_optimized_kernel(uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        return false;} \
    }while(0)

#define CUDA_CHECK_VOID(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(1);} \
    }while(0)

// Kernel information structure
struct KernelInfo {
    int id;
    const char *name;

    void (*launch_fn)(dim3, dim3, cudaStream_t, uint64_t *, uint32_t *, uint64_t);

    int optimal_blocks;
    int optimal_threads;
    int work_per_unit; // hashes per thread or warp
    const char *work_unit; // "thread" or "warp"
    bool uses_warps;
    int warps_per_block;
};

// Test vector structure
struct TestVector {
    std::string name;
    std::array<uint8_t, 32> message;
    std::array<uint8_t, 20> expected_hash;
};

// Test result structure
struct TestResult {
    bool passed;
    bool hash_correct;
    bool found_match;
    double performance_ghps;
    double gpu_utilization;
    std::string error_message;
};

// Kernel launch wrappers
void launch_standard(dim3 blocks, dim3 threads, cudaStream_t stream,
                     uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_mining_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_warp_collaborative(dim3 blocks, dim3 threads, cudaStream_t stream,
                               uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_warp_collaborative_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_vectorized(dim3 blocks, dim3 threads, cudaStream_t stream,
                       uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_vectorized_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_bitsliced(dim3 blocks, dim3 threads, cudaStream_t stream,
                      uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_bitsliced_kernel_correct<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_hashcat(dim3 blocks, dim3 threads, cudaStream_t stream,
                    uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_hashcat_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_hashcat_extreme(dim3 blocks, dim3 threads, cudaStream_t stream,
                            uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_hashcat_extreme_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_cooperative_groups(dim3 blocks, dim3 threads, cudaStream_t stream,
                               uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_cooperative_groups_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_multi_hash(dim3 blocks, dim3 threads, cudaStream_t stream,
                       uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_multi_hash_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_readonly_cache(dim3 blocks, dim3 threads, cudaStream_t stream,
                           uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_readonly_cache_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_simd_vectorized(dim3 blocks, dim3 threads, cudaStream_t stream,
                            uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_simd_vectorized_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_hybrid_warp_simd(dim3 blocks, dim3 threads, cudaStream_t stream,
                             uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_hybrid_warp_simd_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

void launch_ldg_optimized(dim3 blocks, dim3 threads, cudaStream_t stream,
                          uint64_t *pairs, uint32_t *ticket, uint64_t seed) {
    sha1_ldg_optimized_kernel<<<blocks, threads, 0, stream>>>(pairs, ticket, seed);
}

// CPU SHA-1 implementation
void cpu_sha1(const uint8_t *msg, size_t len, uint8_t *output) {
    sha1_ctx ctx;
    sha1_init(ctx);
    sha1_update(ctx, msg, len);
    sha1_final(ctx, output);
}

// Generate test vectors
std::vector<TestVector> generate_test_vectors() {
    std::vector<TestVector> vectors;

    // Test 1: All zeros
    {
        TestVector tv;
        tv.name = "All zeros";
        tv.message.fill(0);
        cpu_sha1(tv.message.data(), 32, tv.expected_hash.data());
        vectors.push_back(tv);
    }

    // Test 2: Sequential bytes
    {
        TestVector tv;
        tv.name = "Sequential bytes";
        for (int i = 0; i < 32; i++) tv.message[i] = i;
        cpu_sha1(tv.message.data(), 32, tv.expected_hash.data());
        vectors.push_back(tv);
    }

    // Test 3: All 0xFF
    {
        TestVector tv;
        tv.name = "All 0xFF";
        tv.message.fill(0xFF);
        cpu_sha1(tv.message.data(), 32, tv.expected_hash.data());
        vectors.push_back(tv);
    }

    // Test 4: Pattern
    {
        TestVector tv;
        tv.name = "Pattern 0xDEADBEEF";
        for (int i = 0; i < 32; i += 4) {
            tv.message[i] = 0xDE;
            tv.message[i + 1] = 0xAD;
            tv.message[i + 2] = 0xBE;
            tv.message[i + 3] = 0xEF;
        }
        cpu_sha1(tv.message.data(), 32, tv.expected_hash.data());
        vectors.push_back(tv);
    }

    // Test 5: ASCII text
    {
        TestVector tv;
        tv.name = "ASCII text";
        const char *text = "The quick brown fox jumps over.."; // 32 chars
        memcpy(tv.message.data(), text, 32);
        cpu_sha1(tv.message.data(), 32, tv.expected_hash.data());
        vectors.push_back(tv);
    }

    return vectors;
}

// Initialize kernel information
std::vector<KernelInfo> get_kernel_info(int num_sms) {
    std::vector<KernelInfo> kernels = {
        {0, "Standard", launch_standard, num_sms * 8, 256, 4, "thread", false, 8},
        {1, "Warp Collaborative", launch_warp_collaborative, num_sms * 32, 256, 256, "warp", true, 8},
        {2, "Vectorized", launch_vectorized, num_sms * 16, 128, 2, "thread", false, 4},
        {3, "Bitsliced", launch_bitsliced, num_sms * 8, 128, 128, "warp", true, 4},
        {4, "HashCat", launch_hashcat, num_sms * 64, 64, 16, "thread", false, 2},
        {5, "HashCat Extreme", launch_hashcat_extreme, num_sms * 32, 128, 32, "thread", false, 4},
        {6, "Cooperative Groups", launch_cooperative_groups, num_sms * 16, 256, 320, "warp", true, 8},
        {7, "Multi-Hash", launch_multi_hash, num_sms * 16, 128, 8, "thread", false, 4},
        {8, "Read-Only Cache", launch_readonly_cache, num_sms * 16, 256, 4, "thread", false, 8},
        {9, "SIMD Vectorized", launch_simd_vectorized, num_sms * 16, 256, 8, "thread", false, 8},
        {10, "Hybrid Warp-SIMD", launch_hybrid_warp_simd, num_sms * 16, 128, 256, "warp", true, 4},
        {11, "LDG Optimized", launch_ldg_optimized, num_sms * 16, 256, 6, "thread", false, 8}
    };
    return kernels;
}

// Calculate hashes per kernel launch
uint64_t calculate_hashes_per_launch(const KernelInfo &kernel) {
    if (kernel.uses_warps) {
        return (uint64_t) kernel.optimal_blocks * kernel.warps_per_block * kernel.work_per_unit;
    } else {
        return (uint64_t) kernel.optimal_blocks * kernel.optimal_threads * kernel.work_per_unit;
    }
}

// Test a single kernel with a single test vector
bool test_kernel_correctness(const KernelInfo &kernel, const TestVector &tv,
                             uint64_t *d_pairs, uint32_t *d_ticket, TestResult &result) {
    // Convert expected hash to target format
    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(tv.expected_hash[4 * i]) << 24) |
                    (uint32_t(tv.expected_hash[4 * i + 1]) << 16) |
                    (uint32_t(tv.expected_hash[4 * i + 2]) << 8) |
                    uint32_t(tv.expected_hash[4 * i + 3]);
    }

    // Upload job
    upload_new_job(tv.message.data(), target);

    // Clear ticket
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    // Launch kernel
    kernel.launch_fn(kernel.optimal_blocks, kernel.optimal_threads, 0, d_pairs, d_ticket, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check result
    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    result.found_match = (found > 0);

    if (found > 0) {
        // Verify the found message
        uint64_t h_result[4];
        CUDA_CHECK(cudaMemcpy(h_result, d_pairs, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost));

        // Reconstruct message
        uint8_t gpu_msg[32];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                gpu_msg[i * 8 + j] = (h_result[i] >> (j * 8)) & 0xFF;
            }
        }

        // Compute hash of GPU result
        uint8_t gpu_hash[20];
        cpu_sha1(gpu_msg, 32, gpu_hash);

        result.hash_correct = (memcmp(gpu_hash, tv.expected_hash.data(), 20) == 0);
        result.passed = result.found_match && result.hash_correct;
    } else {
        result.hash_correct = false;
        result.passed = false;
        result.error_message = "No match found";
    }

    return true;
}

// Test kernel performance and check for suspicious behavior
bool test_kernel_performance(const KernelInfo &kernel, uint64_t *d_pairs,
                             uint32_t *d_ticket, TestResult &result) {
    // Dummy job for performance testing
    uint8_t msg[32] = {0};
    uint32_t target[5] = {0};
    upload_new_job(msg, target);

    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel.launch_fn(kernel.optimal_blocks, kernel.optimal_threads, 0,
                         d_pairs, d_ticket, i * calculate_hashes_per_launch(kernel));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time the kernel
    const int iterations = 1000;
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        kernel.launch_fn(kernel.optimal_blocks, kernel.optimal_threads, 0,
                         d_pairs, d_ticket, i * calculate_hashes_per_launch(kernel));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    uint64_t total_hashes = iterations * calculate_hashes_per_launch(kernel);
    result.performance_ghps = total_hashes / (milliseconds * 1e6);

    // Check for suspicious performance (>300 GH/s is likely a bug)
    if (result.performance_ghps > 300) {
        result.error_message = "Suspiciously high performance - kernel may be broken";
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return true;
}

// Main test function
void test_all_kernels() {
    // Get device properties
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK_VOID(cudaGetDeviceProperties(&prop, device));

    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|   SHA-1 Kernel Test Suite v1.0          |\n";
    std::cout << "+------------------------------------------+\n\n";

    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n\n";

    // Allocate GPU memory
    uint64_t *d_pairs;
    uint32_t *d_ticket;
    CUDA_CHECK_VOID(cudaMalloc(&d_pairs, sizeof(uint64_t) * 4 * (1u << 20)));
    CUDA_CHECK_VOID(cudaMalloc(&d_ticket, sizeof(uint32_t)));

    // Get kernel info and test vectors
    auto kernels = get_kernel_info(prop.multiProcessorCount);
    auto test_vectors = generate_test_vectors();

    // Results summary
    std::vector<std::pair<int, bool> > kernel_status;

    // Test each kernel
    for (const auto &kernel: kernels) {
        std::cout << "\n=== Testing Kernel " << kernel.id << ": " << kernel.name << " ===\n";
        std::cout << "Configuration: " << kernel.optimal_blocks << " blocks × "
                << kernel.optimal_threads << " threads\n";
        std::cout << "Work: " << kernel.work_per_unit << " hashes per " << kernel.work_unit << "\n";
        std::cout << "Hashes per launch: " << calculate_hashes_per_launch(kernel) << "\n\n";

        bool all_passed = true;

        // Test correctness with all test vectors
        std::cout << "Correctness tests:\n";
        for (const auto &tv: test_vectors) {
            TestResult result;
            if (!test_kernel_correctness(kernel, tv, d_pairs, d_ticket, result)) {
                std::cout << "  " << tv.name << ": ERROR (failed to run)\n";
                all_passed = false;
                continue;
            }

            std::cout << "  " << tv.name << ": ";
            if (result.passed) {
                std::cout << "PASS\n";
            } else {
                std::cout << "FAIL";
                if (!result.found_match) {
                    std::cout << " (no match found)";
                } else if (!result.hash_correct) {
                    std::cout << " (incorrect hash)";
                }
                std::cout << "\n";
                all_passed = false;
            }
        }

        // Test performance
        std::cout << "\nPerformance test:\n";
        TestResult perf_result;
        if (test_kernel_performance(kernel, d_pairs, d_ticket, perf_result)) {
            std::cout << "  Performance: " << std::fixed << std::setprecision(2)
                    << perf_result.performance_ghps << " GH/s\n";

            if (perf_result.performance_ghps > 300) {
                std::cout << "  WARNING: " << perf_result.error_message << "\n";
                std::cout << "  This suggests the kernel has early exit bugs!\n";
                all_passed = false;
            }
        }

        kernel_status.push_back({kernel.id, all_passed});

        std::cout << "\nKernel " << kernel.id << " overall: "
                << (all_passed ? "PASSED" : "FAILED") << "\n";
    }

    // Summary report
    std::cout << "\n\n+------------------------------------------+\n";
    std::cout << "|              SUMMARY REPORT              |\n";
    std::cout << "+------------------------------------------+\n\n";

    int passed_count = 0;
    int failed_count = 0;

    for (const auto &[id, passed]: kernel_status) {
        const auto &kernel = kernels[id];
        std::cout << "Kernel " << std::setw(2) << id << " (" << std::setw(20) << std::left
                << kernel.name << "): " << (passed ? "PASSED" : "FAILED") << "\n";
        if (passed) passed_count++;
        else failed_count++;
    }

    std::cout << "\nTotal: " << passed_count << " passed, " << failed_count << " failed\n";

    // Write detailed report to file
    std::ofstream report("kernel_test_report.txt");
    if (report.is_open()) {
        report << "SHA-1 Kernel Test Report\n";
        report << "========================\n\n";
        report << "GPU: " << prop.name << "\n";
        report << "Date: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n\n";

        for (const auto &[id, passed]: kernel_status) {
            const auto &kernel = kernels[id];
            report << "Kernel " << id << " (" << kernel.name << "): "
                    << (passed ? "PASSED" : "FAILED") << "\n";
        }

        report.close();
        std::cout << "\nDetailed report written to: kernel_test_report.txt\n";
    }

    // Recommendations
    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|            RECOMMENDATIONS               |\n";
    std::cout << "+------------------------------------------+\n\n";

    std::cout << "Working kernels (use these for production):\n";
    for (const auto &[id, passed]: kernel_status) {
        if (passed) {
            std::cout << "  - Kernel " << id << " (" << kernels[id].name << ")\n";
        }
    }

    std::cout << "\nBroken kernels (do not use):\n";
    for (const auto &[id, passed]: kernel_status) {
        if (!passed) {
            std::cout << "  - Kernel " << id << " (" << kernels[id].name << ")";

            // Add specific warnings
            if (id == 4 || id == 5) {
                std::cout << " [likely has early exit bug]";
            } else if (id == 7 || id == 10) {
                std::cout << " [complex implementation may have bugs]";
            }
            std::cout << "\n";
        }
    }

    // Cleanup
    CUDA_CHECK_VOID(cudaFree(d_pairs));
    CUDA_CHECK_VOID(cudaFree(d_ticket));
}

// GPU utilization monitoring helper
void monitor_gpu_utilization() {
    std::cout << "\n\nTo monitor GPU utilization during tests:\n";
    std::cout << "1. Open another terminal\n";
    std::cout << "2. Run: nvidia-smi dmon -i 0 -s pucvmet\n";
    std::cout << "3. Look for:\n";
    std::cout << "   - sm (GPU utilization): Should be >90% for working kernels\n";
    std::cout << "   - pwr (Power): Should be near TDP for working kernels\n";
    std::cout << "   - If you see 0% utilization with high reported GH/s, the kernel is broken!\n";
}

int main(int argc, char **argv) {
    // Run the comprehensive test
    test_all_kernels();

    // Show GPU monitoring instructions
    monitor_gpu_utilization();

    std::cout << "\n\nTest complete. Press Enter to exit...";
    std::cin.get();

    return 0;
}
