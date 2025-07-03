#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <cuda_runtime.h>
#include <cstring>
#include "cxxsha1.hpp"
#include "job_upload_api.h"

// Kernel declaration for single SHA-1
extern "C" __global__ void sha1_collision_kernel_ultra(uint8_t *, uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) << '\n'; \
        std::exit(1);} \
    }while(0)

// CPU implementation of single SHA-1
void cpu_sha1(const uint8_t *msg, uint8_t *output) {
    sha1_ctx ctx;
    sha1_init(ctx);
    sha1_update(ctx, msg, 32);
    sha1_final(ctx, output);
}

// Test known vectors
void test_known_vectors() {
    std::cout << "=== Testing Known SHA-1 Vectors ===\n\n";

    // Test 1: Zero message
    {
        uint8_t msg[32] = {0};
        uint8_t result[20];

        cpu_sha1(msg, result);

        std::cout << "Test 1 - Zero message:\n";
        std::cout << "Message: ";
        for (int i = 0; i < 32; i++) printf("%02x", msg[i]);
        std::cout << "\nSHA-1:   ";
        for (int i = 0; i < 20; i++) printf("%02x", result[i]);
        std::cout << "\n\n";
    }

    // Test 2: Sequential bytes
    {
        uint8_t msg[32];
        for (int i = 0; i < 32; i++) msg[i] = i;
        uint8_t result[20];

        cpu_sha1(msg, result);

        std::cout << "Test 2 - Sequential bytes (0x00-0x1f):\n";
        std::cout << "Message: ";
        for (int i = 0; i < 32; i++) printf("%02x", msg[i]);
        std::cout << "\nSHA-1:   ";
        for (int i = 0; i < 20; i++) printf("%02x", result[i]);
        std::cout << "\n\n";
    }

    // Test 3: All 0xFF
    {
        uint8_t msg[32];
        for (int i = 0; i < 32; i++) msg[i] = 0xFF;
        uint8_t result[20];

        cpu_sha1(msg, result);

        std::cout << "Test 3 - All 0xFF:\n";
        std::cout << "SHA-1:   ";
        for (int i = 0; i < 20; i++) printf("%02x", result[i]);
        std::cout << "\n\n";
    }
}

// Test GPU kernel with specific inputs
void test_gpu_kernel() {
    std::cout << "=== Testing GPU Kernel ===\n\n";

    // Setup test message
    uint8_t test_msg[32];
    for (int i = 0; i < 32; i++) test_msg[i] = i;

    // Calculate expected result on CPU
    uint8_t expected[20];
    cpu_sha1(test_msg, expected);

    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(expected[4 * i]) << 24) |
                    (uint32_t(expected[4 * i + 1]) << 16) |
                    (uint32_t(expected[4 * i + 2]) << 8) |
                    uint32_t(expected[4 * i + 3]);
    }

    // Upload to GPU
    upload_new_job(test_msg, target);

    // Allocate GPU memory
    uint64_t *d_pairs;
    uint32_t *d_ticket;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 4 * 1024));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    std::cout << "Target SHA-1: ";
    for (int i = 0; i < 20; i++) printf("%02x", expected[i]);
    std::cout << "\n\n";

    // Test 1: Should find the original message (nonce = 0)
    std::cout << "Test 1 - Finding original message:\n";

    // Launch kernel with 1 thread
    sha1_collision_kernel_ultra<<<1, 1>>>(nullptr, d_pairs, d_ticket, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Found candidates: " << found << "\n";

    if (found > 0) {
        uint64_t result[4];
        CUDA_CHECK(cudaMemcpy(result, d_pairs, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost));

        std::cout << "GPU found message: ";
        for (int i = 0; i < 4; i++) {
            printf("%016llx", result[i]);
        }
        std::cout << "\n";

        // Reconstruct message from GPU result
        uint8_t gpu_msg[32];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                gpu_msg[i * 8 + j] = (result[i] >> (j * 8)) & 0xFF;
            }
        }

        // Verify the hash
        uint8_t gpu_hash[20];
        cpu_sha1(gpu_msg, gpu_hash);
        bool hash_matches = (memcmp(gpu_hash, expected, 20) == 0);
        std::cout << "Hash verification: " << (hash_matches ? "PASS" : "FAIL") << "\n";
        if (!hash_matches) {
            std::cout << "Expected: ";
            for (int i = 0; i < 20; i++) printf("%02x", expected[i]);
            std::cout << "\nGot:      ";
            for (int i = 0; i < 20; i++) printf("%02x", gpu_hash[i]);
            std::cout << "\n";
        }
    }

    // Test 2: Search a range
    std::cout << "\nTest 2 - Searching range with modifications:\n";
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    // Launch with more threads
    sha1_collision_kernel_ultra<<<256, 256>>>(nullptr, d_pairs, d_ticket, 0xDEADBEEF);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Found candidates in 65536 attempts: " << found << "\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
}

// Verify a specific collision
void verify_collision(const uint8_t *msg1, const uint8_t *msg2) {
    std::cout << "\n=== Verifying Collision ===\n";

    uint8_t hash1[20], hash2[20];
    cpu_sha1(msg1, hash1);
    cpu_sha1(msg2, hash2);

    std::cout << "Message 1: ";
    for (int i = 0; i < 32; i++) printf("%02x", msg1[i]);
    std::cout << "\nSHA-1:     ";
    for (int i = 0; i < 20; i++) printf("%02x", hash1[i]);

    std::cout << "\n\nMessage 2: ";
    for (int i = 0; i < 32; i++) printf("%02x", msg2[i]);
    std::cout << "\nSHA-1:     ";
    for (int i = 0; i < 20; i++) printf("%02x", hash2[i]);

    bool is_collision = (memcmp(hash1, hash2, 20) == 0);
    bool different_msgs = (memcmp(msg1, msg2, 32) != 0);
    std::cout << "\n\nCollision: " << (is_collision ? "YES" : "NO") << "\n";
    std::cout << "Different messages: " << (different_msgs ? "YES" : "NO") << "\n";
    std::cout << "Valid collision: " << (is_collision && different_msgs ? "VALID!" : "INVALID!") << "\n";
}

// Performance sanity check
void performance_check() {
    std::cout << "\n=== Performance Sanity Check ===\n";

    // Dummy job
    uint8_t msg[32] = {0};
    uint32_t target[5] = {0};
    upload_new_job(msg, target);

    uint64_t *d_pairs;
    uint32_t *d_ticket;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 4 * 1024));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    // Time a batch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int blocks = 256;
    const int threads = 256;
    const int iterations = 100000;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        sha1_collision_kernel_ultra<<<blocks, threads>>>(nullptr, d_pairs, d_ticket, i);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    uint64_t total_hashes = (uint64_t) blocks * threads * iterations * 4; // 4 nonces per thread
    double ghps = total_hashes / (ms / 1000.0) / 1e9;

    std::cout << "Hashes computed: " << total_hashes << "\n";
    std::cout << "Time: " << ms << " ms\n";
    std::cout << "Performance: " << ghps << " GH/s\n";

    if (ghps < 0.1) {
        std::cout << "WARNING: Performance seems too low!\n";
    } else if (ghps > 1000) {
        std::cout << "WARNING: Performance seems unrealistically high!\n";
    } else {
        std::cout << "Performance looks reasonable.\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Test collision finding capability
void test_collision_finding() {
    std::cout << "\n=== Testing Collision Finding ===\n";
    // Create a target that's more likely to have collisions (e.g., with some zero bytes)
    uint8_t base_msg[32] = {0};
    base_msg[0] = 0x01; // Small change from all zeros
    uint8_t target_hash[20];
    cpu_sha1(base_msg, target_hash);
    // Set some bytes to zero to increase collision probability
    target_hash[18] = 0x00;
    target_hash[19] = 0x00;
    std::cout << "Searching for partial collisions with target ending in 0000\n";
    std::cout << "Target hash: ";
    for (int i = 0; i < 20; i++) printf("%02x", target_hash[i]);
    std::cout << "\n";
    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(target_hash[4 * i]) << 24) |
                    (uint32_t(target_hash[4 * i + 1]) << 16) |
                    (uint32_t(target_hash[4 * i + 2]) << 8) |
                    uint32_t(target_hash[4 * i + 3]);
    }
    upload_new_job(base_msg, target);
    uint64_t *d_pairs;
    uint32_t *d_ticket;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 4 * 1024));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));
    // Search with many threads
    std::cout << "Searching with 1M threads...\n";
    sha1_collision_kernel_ultra<<<4096, 256>>>(nullptr, d_pairs, d_ticket, 0x12345678);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Found " << found << " candidates\n";

    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
}

int main(int argc, char **argv) {
    std::cout << "+------------------------------------------+\n";
    std::cout << "|    SHA-1 Collision Mining Verification   |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Run all tests
    test_known_vectors();
    test_gpu_kernel();
    performance_check();
    test_collision_finding();

    // If collision candidates provided on command line
    if (argc == 3) {
        std::cout << "\n=== Verifying Command Line Input ===\n";
        uint8_t msg1[32], msg2[32];

        // Parse hex strings
        for (int i = 0; i < 32; i++) {
            sscanf(argv[1] + i * 2, "%2hhx", &msg1[i]);
            sscanf(argv[2] + i * 2, "%2hhx", &msg2[i]);
        }

        verify_collision(msg1, msg2);
    } else if (argc > 1) {
        std::cout << "\nUsage: " << argv[0] << " [msg1_hex] [msg2_hex]\n";
        std::cout << "Where msg1_hex and msg2_hex are 64-character hex strings (32 bytes)\n";
    }

    std::cout << "\nVerification complete.\n";
    return 0;
}
