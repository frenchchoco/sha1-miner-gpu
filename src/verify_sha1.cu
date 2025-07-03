#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <cuda_runtime.h>
#include <cstring>
#include "cxxsha1.hpp"
#include "job_upload_api.h"

// Kernel declaration for bitsliced SHA-1
extern "C" __global__ void sha1_warp_collaborative_kernel(uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
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
    std::cout << "=== Testing Bitsliced GPU Kernel ===\n\n";

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
    std::cout << "Note: Bitsliced kernel processes 32 messages per warp\n";

    // Launch kernel with 32 threads (1 warp) to process first 32 nonces
    sha1_warp_collaborative_kernel<<<1, 32>>>(d_pairs, d_ticket, 0);
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

    // Launch with multiple warps (must be multiple of 32 threads)
    // Bitsliced kernel uses __launch_bounds__(128, 4)
    sha1_warp_collaborative_kernel<<<64, 128>>>(d_pairs, d_ticket, 0xDEADBEEF);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // Each warp processes 32*4 = 128 messages (4 batches of 32)
    uint64_t total_processed = 64 * 4 * 128; // blocks * warps_per_block * messages_per_warp
    std::cout << "Found candidates in " << total_processed << " attempts: " << found << "\n";

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
    std::cout << "\n=== Bitsliced Performance Sanity Check ===\n";

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

    // For bitsliced kernel: threads must be multiple of 32
    // and not exceed __launch_bounds__(128, 4)
    const int blocks = 512;
    const int threads = 256; // 4 warps per block
    const int iterations = 100000;
    const int batches_per_warp = 4; // From kernel implementation
    const int messages_per_batch = 32;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        sha1_warp_collaborative_kernel<<<blocks, threads>>>(d_pairs, d_ticket, i * blocks * threads);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Calculate total hashes: blocks * warps_per_block * batches_per_warp * messages_per_batch * iterations
    uint64_t warps_per_block = threads / 32;
    uint64_t total_hashes = (uint64_t) blocks * warps_per_block * batches_per_warp * messages_per_batch * iterations;

    double ghps = total_hashes / (ms * 1e6); // ms→s and hashes→giga

    std::cout << "Bitsliced implementation stats:\n";
    std::cout << "Blocks          : " << blocks << '\n';
    std::cout << "Threads/block   : " << threads << " (" << warps_per_block << " warps)\n";
    std::cout << "Messages/warp   : " << batches_per_warp * messages_per_batch << '\n';
    std::cout << "Iterations      : " << iterations << '\n';
    std::cout << "Total hashes    : " << total_hashes << '\n';
    std::cout << "Time            : " << ms << " ms\n";
    std::cout << "Performance     : " << std::fixed << std::setprecision(3)
            << ghps << " GH/s\n";

    if (ghps < 0.001) {
        std::cout << "WARNING: Performance seems too low! (Expected 0.1-2 GH/s for bitsliced)\n";
    } else if (ghps > 10) {
        std::cout << "WARNING: Performance seems unrealistically high for bitsliced!\n";
    } else {
        std::cout << "Performance looks reasonable for bitsliced implementation.\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Test collision finding capability
void test_collision_finding() {
    std::cout << "\n=== Testing Bitsliced Collision Finding ===\n";

    // Create a target that's more likely to have collisions
    uint8_t base_msg[32] = {0};
    base_msg[0] = 0x01;
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

    // Search with many warps
    const int blocks = 512;
    const int threads = 128; // 4 warps per block
    uint64_t total_messages = (uint64_t) blocks * (threads / 32) * 4 * 32; // 4 batches of 32 messages per warp

    std::cout << "Searching " << total_messages << " messages with bitsliced kernel...\n";
    sha1_warp_collaborative_kernel<<<blocks, threads>>>(d_pairs, d_ticket, 0x12345678);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Found " << found << " candidates\n";

    if (found > 0 && found <= 10) {
        // Print first few results
        std::vector<uint64_t> results(found * 4);
        CUDA_CHECK(cudaMemcpy(results.data(), d_pairs, sizeof(uint64_t) * 4 * found, cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < found && i < 3; i++) {
            std::cout << "\nCandidate " << i + 1 << ":\n";
            uint8_t msg[32];
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 8; k++) {
                    msg[j * 8 + k] = (results[i * 4 + j] >> (k * 8)) & 0xFF;
                }
            }

            std::cout << "Message: ";
            for (int j = 0; j < 32; j++) printf("%02x", msg[j]);

            uint8_t hash[20];
            cpu_sha1(msg, hash);
            std::cout << "\nHash:    ";
            for (int j = 0; j < 20; j++) printf("%02x", hash[j]);
            std::cout << "\n";
        }
    }

    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
}

// Test bitslicing correctness
void test_bitslicing_correctness() {
    std::cout << "\n=== Testing Bitslicing Correctness ===\n";

    // Test with a specific nonce range where we know the answer
    uint8_t msg[32] = {0};
    msg[0] = 0xAB;
    msg[1] = 0xCD;

    // Compute hash for various nonces on CPU
    std::cout << "CPU reference hashes for first 5 nonces:\n";
    for (uint64_t nonce = 0; nonce < 5; nonce++) {
        uint8_t test_msg[32];
        memcpy(test_msg, msg, 32);

        // Apply nonce
        uint32_t *msg_words = (uint32_t *) test_msg;
        msg_words[6] ^= (uint32_t) (nonce & 0xFFFFFFFF);
        msg_words[7] ^= (uint32_t) (nonce >> 32);

        uint8_t hash[20];
        cpu_sha1(test_msg, hash);

        std::cout << "Nonce " << nonce << ": ";
        for (int i = 0; i < 20; i++) printf("%02x", hash[i]);
        std::cout << "\n";
    }

    // Set target to match nonce=2
    uint8_t target_msg[32];
    memcpy(target_msg, msg, 32);
    uint32_t *target_words = (uint32_t *) target_msg;
    target_words[6] ^= 2; // nonce = 2

    uint8_t target_hash[20];
    cpu_sha1(target_msg, target_hash);

    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(target_hash[4 * i]) << 24) |
                    (uint32_t(target_hash[4 * i + 1]) << 16) |
                    (uint32_t(target_hash[4 * i + 2]) << 8) |
                    uint32_t(target_hash[4 * i + 3]);
    }

    upload_new_job(msg, target);

    uint64_t *d_pairs;
    uint32_t *d_ticket;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 4 * 10));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    // Launch just one warp to test first 32 nonces
    std::cout << "\nLaunching bitsliced kernel for nonces 0-31...\n";
    sha1_warp_collaborative_kernel<<<1, 32>>>(d_pairs, d_ticket, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (found == 1) {
        std::cout << "SUCCESS: Found exactly 1 match (expected nonce=2)\n";

        uint64_t result[4];
        CUDA_CHECK(cudaMemcpy(result, d_pairs, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost));

        // Verify it's nonce=2
        uint32_t found_m6 = result[3] & 0xFFFFFFFF;
        uint32_t found_m7 = result[3] >> 32;
        uint32_t orig_m6 = ((uint32_t *) msg)[6];
        uint32_t orig_m7 = ((uint32_t *) msg)[7];

        uint64_t found_nonce = ((uint64_t) (found_m7 ^ orig_m7) << 32) | (found_m6 ^ orig_m6);
        std::cout << "Found nonce: " << found_nonce << " (expected: 2)\n";
    } else {
        std::cout << "ERROR: Found " << found << " matches (expected exactly 1)\n";
    }

    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
}

int main(int argc, char **argv) {
    std::cout << "+------------------------------------------+\n";
    std::cout << "|  SHA-1 Bitsliced Mining Verification     |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n";
    std::cout << "Warp size: " << prop.warpSize << "\n\n";

    // Run all tests
    test_known_vectors();
    test_gpu_kernel();
    test_bitslicing_correctness();
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
