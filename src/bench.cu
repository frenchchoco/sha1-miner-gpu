#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <array>
#include "job_upload_api.h"

#include "cxxsha1.hpp"
#include "job_constants.cuh"

extern "C" __global__
void sha1_double_kernel(uint8_t *, uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(x)                                                     \
    do { cudaError_t e = (x);                                             \
         if (e != cudaSuccess) {                                          \
             std::cerr << cudaGetErrorString(e) << '\n'; std::exit(1);}   \
    } while (0)

/* --------------------------------------------------------------------- */
constexpr uint32_t BATCH = 1u << 22; // 4 194 304 threads/launch
constexpr int THREADS = 1024;
constexpr uint32_t RING_SIZE = 1u << 20; // ring buffer slots

int main(int argc, char **argv) {
    int repeats = 50;
    if (argc > 1) {
        char *end = nullptr;
        long tmp = std::strtol(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && tmp > 0) repeats = int(tmp);
    }

    /* ------------------------------------------------------------------ */
    /* 1.  Challenge pre-image and its double-SHA-1 digest                */
    /* ------------------------------------------------------------------ */
    std::array<uint8_t, 32> preimage = {
        /* TODO: insert the 32-byte puzzle here */
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
    };

    uint8_t d1[20], d2[20];
    sha1_ctx c{};
    sha1_init(c);
    sha1_update(c, preimage.data(), 32);
    sha1_final(c, d1);
    sha1_init(c);
    sha1_update(c, d1, 20); // second SHA-1
    sha1_final(c, d2);

    uint32_t target[5];
    for (int i = 0; i < 5; ++i) {
        target[i] =
                (static_cast<uint32_t>(d2[4 * i]) << 24) |
                (static_cast<uint32_t>(d2[4 * i + 1]) << 16) |
                (static_cast<uint32_t>(d2[4 * i + 2]) << 8) |
                static_cast<uint32_t>(d2[4 * i + 3]);
    }

    std::cout << "CPU double-SHA-1(preimage): ";
    for (uint8_t b: d2) std::printf("%02x", b);
    std::cout << '\n';

    /* upload target as before */
    upload_new_job(preimage.data(), target);

    /* optional sanity-check: read g_target back */
    uint32_t target_d[5] = {};
    CUDA_CHECK(cudaMemcpyFromSymbol(target_d, g_target, 5*4));
    std::cout << "g_target (device)        : ";
    for (int i = 0; i < 5; ++i)
        std::printf("%08x", target_d[i]);
    std::cout << '\n';

    /* ------------------------------------------------------------------ */
    /* 2.  Allocate ring buffer + ticket                                  */
    /* ------------------------------------------------------------------ */
    uint64_t *d_pairs = nullptr;
    uint32_t *d_ticket = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 6 * RING_SIZE));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    constexpr dim3 blockDim(THREADS);
    constexpr dim3 gridDim((BATCH + THREADS - 1) / THREADS);

    /* ------------------------------------------------------------------ */
    /* 3.  Timing loop                                                    */
    /* ------------------------------------------------------------------ */
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeats; ++i) {
        sha1_double_kernel<<<gridDim, blockDim>>>(
            nullptr, d_pairs, d_ticket, 0xCAFEBABEULL + i);
        CUDA_CHECK(cudaPeekAtLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    /* ------------------------------------------------------------------ */
    /* 4.  Statistics                                                     */
    /* ------------------------------------------------------------------ */
    double hashes = double(gridDim.x) * blockDim.x * repeats;
    double seconds = ms / 1e3;
    double ghps = hashes / seconds / 1e9;

    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    std::cout << "Kernel time   : " << seconds << " s\n"
            << "Hash-rate     : " << ghps * 1000.0
            << " M double-SHA-1/s\n"
            << "PCIe traffic  : 0.00 GiB/s (all data stays on-GPU)\n";

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(found),
        cudaMemcpyDeviceToHost));
    std::cout << "Candidates stored: " << found << '\n';

    if (found) {
        uint64_t first[4] = {};
        CUDA_CHECK(cudaMemcpy(first, d_pairs, sizeof(first),
            cudaMemcpyDeviceToHost));
        uint32_t msg32[8];
        msg32[0] = uint32_t(first[0]);
        msg32[1] = uint32_t(first[0] >> 32);
        msg32[2] = uint32_t(first[1]);
        msg32[3] = uint32_t(first[1] >> 32);
        msg32[4] = uint32_t(first[2]);
        msg32[5] = uint32_t(first[2] >> 32);
        msg32[6] = uint32_t(first[3]);
        msg32[7] = uint32_t(first[3] >> 32);
        std::cout << "\nFirst candidate message : ";
        for (int i = 0; i < 8; ++i) std::printf("%08x", msg32[i]);
        std::cout << '\n';

        /* compute double-SHA-1 on CPU just to show the digest          */
        uint8_t dA[20], dB[20];
        sha1_ctx tmp{};
        sha1_init(tmp);
        sha1_update(tmp, msg32, 32);
        sha1_final(tmp, dA);
        sha1_init(tmp);
        sha1_update(tmp, dA, 20);
        sha1_final(tmp, dB);

        std::cout << "double-SHA-1(candidate) : ";
        for (uint8_t b: dB) std::printf("%02x", b);
        std::cout << "\n------------------------------------------------\n";
    }

    /* ------------------------------------------------------------------ */
    /* 5.  Cleanup                                                        */
    /* ------------------------------------------------------------------ */
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
    return 0;
}
