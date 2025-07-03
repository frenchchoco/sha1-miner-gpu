#ifndef BENCH_CUH
#define BENCH_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <span>
#include "util.hpp"

// exported from kernel.cu
extern __global__ void sha1_double_kernel(const uint8_t*, uint8_t*, uint32_t);

constexpr uint32_t  BATCH      = 1 << 22;   // 4 194 304 messages per launch
constexpr int       THREADS    = 256;
constexpr size_t    MSG_BYTES  = 32;
constexpr size_t    OUT_BYTES  = 20;

int main(int argc, char**)
{
    int repeats = (argc > 1) ? std::atoi(argv[1]) : 50;   // default 50 launches

    /* host / device buffers ------------------------------------------------*/
    std::vector<uint8_t> h_in (BATCH * MSG_BYTES);
    std::vector<uint8_t> h_out(BATCH * OUT_BYTES);
    uint8_t *d_in{}, *d_out{};
    cudaMalloc(&d_in,  h_in .size());
    cudaMalloc(&d_out, h_out.size());

    fill_rand(std::span<uint8_t>(h_in));                  // one random batch
    cudaMemcpy(d_in, h_in.data(), h_in.size(), cudaMemcpyHostToDevice);

    /* launch geometry ------------------------------------------------------*/
    dim3 blockDim(THREADS);
    dim3 gridDim((BATCH + THREADS - 1) / THREADS);

    /* timing ----------------------------------------------------------------*/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i)
        sha1_double_kernel<<<gridDim, blockDim>>>(d_in, d_out, BATCH);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double hashes   = 1.0 * BATCH * repeats;
    double seconds  = ms / 1e3;
    double mhps     = hashes / 1e6  / seconds;            // mega-hashes/s
    double gbps_in  = (hashes * MSG_BYTES) / seconds / 1e9;

    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    std::cout << "Kernel time   : " << seconds << "  s\n"
              << "Hashrate      : " << mhps   << "  M double-SHA-1/s\n"
              << "Input traffic : " << gbps_in << "  GiB/s (32 B per msg)\n";

    /* optional – copy one digest batch back just to exercise PCIe --------- */
    cudaMemcpy(h_out.data(), d_out, h_out.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}

#endif //BENCH_CUH
