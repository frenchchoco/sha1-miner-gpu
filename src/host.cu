#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>          // for std::exit

/* -- add __global__ here -------------------------------------- */
extern "C" __global__
void sha1_double_kernel(uint8_t *, uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t e = (x);                                                 \
        if (e != cudaSuccess) {                                              \
            std::cerr << cudaGetErrorString(e) << '\n';                      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

int main(int argc, char **argv) {
    constexpr uint64_t TOTAL = 1ull << 30; // messages per batch
    constexpr int THREADS = 1024;
    const int blocks = int((TOTAL + THREADS - 1) / THREADS);

    /* device buffers ------------------------------------------------------ */
    uint64_t *d_pairs = nullptr;
    uint32_t *d_ticket = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 6 * (1 << 20)));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    /* timing -------------------------------------------------------------- */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    /* kernel launch ------------------------------------------------------- */
    sha1_double_kernel<<<blocks, THREADS>>>(
        /* out_msgs = */ nullptr,
                         /* out_pairs */ d_pairs,
                         /* ticket    */ d_ticket,
                         /* seed      */ 0xDEADBEEFCAFEBABEULL);

    CUDA_CHECK(cudaDeviceSynchronize());

    /* end timing ---------------------------------------------------------- */
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double hashes = static_cast<double>(TOTAL);
    std::cout << "Throughput: " << hashes / ms * 1e3 / 1e9
            << "  G double-SHA-1/s\n";

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(found),
        cudaMemcpyDeviceToHost));
    std::cout << "Candidates stored: " << found << '\n';

    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
    return 0;
}
