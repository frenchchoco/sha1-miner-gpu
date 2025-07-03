// ─── bench.cu ─────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

extern "C" __global__
void sha1_double_kernel(uint8_t *out_msgs, // nullptr
                        uint64_t *out_pairs, // ring buffer
                        uint32_t *ticket, // counter
                        uint64_t seed);

#define CUDA_CHECK(x)                                                      \
    do { cudaError_t e = (x);                                              \
         if (e != cudaSuccess) {                                           \
             std::cerr << cudaGetErrorString(e) << '\n'; std::exit(1); }   \
    } while (0)

/* --------------------------------------------------------------------- */
constexpr uint32_t BATCH = 1u << 22; // 4,194,304 messages
constexpr int THREADS = 1024;
constexpr uint32_t RING_SIZE = 1u << 20; // candidate slots

int main(int argc, char **argv) {
    long repeats = (argc > 1) ? std::strtol(argv[1], nullptr, 10) : 50;
    if (repeats <= 0) repeats = 50;

    /* device buffers ---------------------------------------------------- */
    uint64_t *d_pairs = nullptr;
    uint32_t *d_ticket = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 6 * RING_SIZE));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    const dim3 blockDim(THREADS);
    const dim3 gridDim((BATCH + THREADS - 1) / THREADS);

    /* timing ------------------------------------------------------------ */
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeats; ++i) {
        sha1_double_kernel<<<gridDim, blockDim>>>(
            nullptr, // out_msgs (not used)
            d_pairs,
            d_ticket,
            0xCAFEBABEDEADBEEFull + i);
        CUDA_CHECK(cudaPeekAtLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    /* stats ------------------------------------------------------------- */
    double hashes = double(gridDim.x) * blockDim.x * repeats;
    double seconds = ms / 1e3;
    double mhps = hashes / 1e6 / seconds; // mega-hash/s

    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    std::cout << "Kernel time   : " << seconds << " s\n"
            << "Hash-rate     : " << mhps << " M double-SHA-1/s\n"
            << "PCIe traffic  : 0.00 GiB/s (all data stays on-GPU)\n";

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(found),
        cudaMemcpyDeviceToHost));
    std::cout << "Candidates stored: " << found << '\n';

    /* cleanup ----------------------------------------------------------- */
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
    return 0;
}
