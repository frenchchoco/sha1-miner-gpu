#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>
#include <vector>
#include <cuda_runtime.h>
#include "cxxsha1.cpp"
#include "job_upload_api.h"

// Kernel declaration
extern "C" __global__ void sha1_double_kernel(uint8_t *, uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) << '\n'; \
        std::exit(1);} \
    }while(0)

// CPU implementation of double SHA-1
void cpu_double_sha1(const uint8_t *msg, uint8_t *output) {
    uint8_t first_hash[20];

    // First SHA-1
    sha1_ctx ctx;
    sha1_init(ctx);
    sha1_update(ctx, msg, 32);
    sha1_final(ctx, first_hash);

    // Second SHA-1
    sha1_init(ctx);
    sha1_update(ctx, first_hash, 20);
    sha1_final(ctx, output);
}

// Test known vectors
void test_known_vectors() {
    std::cout << "=== Testing Known SHA-1 Vectors ===\n\n";

    // Test 1: Zero message
    {
        uint8_t msg[32] = {0};
        uint8_t expected[20];
        uint8_t result[20];

        // Expected: SHA1(SHA1(zeros))
        // SHA1(zeros) = 5ba93c9db0cff93f52b521d7420e43f6eda2784f
        // SHA1(5ba9...) = bf8b4530d8d246dd74ac53a13471bba17941dff7
        const char *expected_hex = "92b404e556588ced6c1acd4ebf053f6809f73a93";
        for (int i = 0; i < 20; i++) {
            sscanf(expected_hex + i * 2, "%2hhx", &expected[i]);
        }

        cpu_double_sha1(msg, result);

        std::cout << "Test 1 - Zero message:\n";
        std::cout << "Expected: ";
        for (int i = 0; i < 20; i++) printf("%02x", expected[i]);
        std::cout << "\nGot:      ";
        for (int i = 0; i < 20; i++) printf("%02x", result[i]);
        std::cout << "\nStatus:   " << (memcmp(result, expected, 20) == 0 ? "PASS" : "FAIL") << "\n\n";
    }

    // Test 2: Sequential bytes
    {
        uint8_t msg[32];
        for (int i = 0; i < 32; i++) msg[i] = i;
        uint8_t result[20];

        cpu_double_sha1(msg, result);

        std::cout << "Test 2 - Sequential bytes (0x00-0x1f):\n";
        std::cout << "Result:   ";
        for (int i = 0; i < 20; i++) printf("%02x", result[i]);
        std::cout << "\n\n";
    }
}

// Test GPU kernel with specific inputs
void test_gpu_kernel() {
    std::cout << "=== Testing GPU Kernel ===\n\n";

    // Setup test message
    std::array<uint8_t, 32> test_msg = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
    };

    // Calculate expected result on CPU
    uint8_t expected[20];
    cpu_double_sha1(test_msg.data(), expected);

    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(expected[4 * i]) << 24) |
                    (uint32_t(expected[4 * i + 1]) << 16) |
                    (uint32_t(expected[4 * i + 2]) << 8) |
                    uint32_t(expected[4 * i + 3]);
    }

    // Upload to GPU
    upload_new_job(test_msg.data(), target);

    // Allocate GPU memory
    uint64_t *d_pairs;
    uint32_t *d_ticket;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 4 * 1024));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    std::cout << "Target double-SHA-1: ";
    for (int i = 0; i < 20; i++) printf("%02x", expected[i]);
    std::cout << "\n\n";

    // Test 1: Should find the original message (nonce = 0)
    std::cout << "Test 1 - Finding original message:\n";
    sha1_double_kernel<<<1, 1>>>(nullptr, d_pairs, d_ticket, 0);
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

        // Verify it matches
        uint8_t gpu_msg[32];
        memcpy(gpu_msg, result, 32);

        bool matches = true;
        for (int i = 0; i < 32; i++) {
            if (gpu_msg[i] != test_msg[i]) {
                matches = false;
                break;
            }
        }
        std::cout << "Verification: " << (matches ? "PASS" : "FAIL") << "\n";
    }

    // Test 2: Search a range
    std::cout << "\nTest 2 - Searching range with modifications:\n";
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    // This will modify the last 4 bytes with different nonces
    sha1_double_kernel<<<256, 256>>>(nullptr, d_pairs, d_ticket, 0xDEADBEEF);
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
    cpu_double_sha1(msg1, hash1);
    cpu_double_sha1(msg2, hash2);

    std::cout << "Message 1: ";
    for (int i = 0; i < 32; i++) printf("%02x", msg1[i]);
    std::cout << "\nSHA-1^2:   ";
    for (int i = 0; i < 20; i++) printf("%02x", hash1[i]);

    std::cout << "\n\nMessage 2: ";
    for (int i = 0; i < 32; i++) printf("%02x", msg2[i]);
    std::cout << "\nSHA-1^2:   ";
    for (int i = 0; i < 20; i++) printf("%02x", hash2[i]);

    std::cout << "\n\nCollision: " << (memcmp(hash1, hash2, 20) == 0 ? "VALID!" : "INVALID!") << "\n";
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
    const int iterations = 100;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        sha1_double_kernel<<<blocks, threads>>>(nullptr, d_pairs, d_ticket, i);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    uint64_t total_hashes = (uint64_t) blocks * threads * iterations;
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

int main(int argc, char **argv) {
    std::cout << "+------------------------------------------+\n";
    std::cout << "|        SHA-1 Mining Verification         |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Run all scripts
    test_known_vectors();
    test_gpu_kernel();
    performance_check();

    // If collision candidates provided on command line
    if (argc == 3) {
        std::cout << "\n=== Verifying Command Line Input ===\n";
        uint8_t msg1[32], msg2[32];

        for (int i = 0; i < 32; i++) {
            sscanf(argv[1] + i * 2, "%2hhx", &msg1[i]);
            sscanf(argv[2] + i * 2, "%2hhx", &msg2[i]);
        }

        verify_collision(msg1, msg2);
    }

    std::cout << "\nVerification complete.\n";
    return 0;
}
