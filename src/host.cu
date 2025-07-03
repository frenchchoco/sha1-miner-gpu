#include <cuda_runtime.h>
#include <iostream>
#include <array>
#include "job_upload_api.h"
#include "cxxsha1.hpp"

extern "C" __global__
void sha1_double_kernel(uint8_t *, uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ std::cerr<<cudaGetErrorString(_e)<<'\n'; std::exit(1);} }while(0)

constexpr uint64_t TOTAL = 1ull << 30; // 1 Gi messages
constexpr int THREADS = 1024;
constexpr int BLOCKS = int((TOTAL + THREADS - 1) / THREADS);
constexpr uint32_t RING = 1u << 20; // candidate buffer slots

int main() {
    std::array<uint8_t, 32> msg = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
    };

    uint8_t d1[20], d2[20];
    sha1_ctx c;
    sha1_init(c);
    sha1_update(c, msg.data(), 32);
    sha1_final(c, d1);
    sha1_init(c);
    sha1_update(c, d1, 20);
    sha1_final(c, d2);

    uint32_t target[5];
    for (int i = 0; i < 5; ++i)
        target[i] = (uint32_t(d2[4 * i]) << 24) | (uint32_t(d2[4 * i + 1]) << 16) |
                    (uint32_t(d2[4 * i + 2]) << 8) | d2[4 * i + 3];

    upload_new_job(msg.data(), target);

    uint64_t *d_pairs = nullptr;
    uint32_t *d_ticket = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t)*6*RING));
    CUDA_CHECK(cudaMalloc(&d_ticket,sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket,0,sizeof(uint32_t)));

    /* 5. Time a single 1-GiB batch ------------------------------------ */
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));

    sha1_double_kernel<<<BLOCKS,THREADS>>>(nullptr, d_pairs, d_ticket, 0xCAFEBABEULL);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
    double ghps = (double) TOTAL / ms * 1e3 / 1e9;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    std::cout << "Throughput: " << ghps << " G double-SHA-1/s\n";

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found,d_ticket,sizeof(found),cudaMemcpyDeviceToHost));
    std::cout << "Candidates stored: " << found << '\n';

    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
    return 0;
}
