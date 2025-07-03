#include <span>
#include <cuda_runtime.h>
#include "util.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <iostream>

#define CHECK(x)                                                                       \
    do {                                                                               \
        auto err = (x);                                                                \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';            \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

constexpr uint32_t  BATCH      = 1 << 22;  // 4 ,194 ,304 candidates per launch
constexpr size_t    MSG_BYTES  = 32;
constexpr size_t    OUT_BYTES  = 20;

// Forward declaration of the kernel exported from kernel.cu
extern __global__ void sha1_double_kernel(const uint8_t*, uint8_t*, uint32_t);

int main()
{
    // ---------------- Host buffers ------------------------------------------
    std::vector<uint8_t> h_msg(BATCH * MSG_BYTES);
    std::vector<uint8_t> h_out(BATCH * OUT_BYTES);

    // ---------------- Device buffers ----------------------------------------
    uint8_t *d_msg{}, *d_out{};
    CHECK(cudaMalloc(&d_msg, h_msg.size()));
    CHECK(cudaMalloc(&d_out, h_out.size()));

    while (true) {
        fill_rand(std::span<uint8_t>(h_msg));

        CHECK(cudaMemcpy(d_msg, h_msg.data(), h_msg.size(),
                         cudaMemcpyHostToDevice));

        // Launch grid: one 256-thread block per 256 messages
        constexpr int THREADS = 256;
        dim3 blockDim(THREADS);
        dim3 gridDim((BATCH + THREADS - 1) / THREADS);

        sha1_double_kernel<<<gridDim, blockDim>>>(d_msg, d_out, BATCH);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size(),
                         cudaMemcpyDeviceToHost));

        // ---------------- Collision check on CPU ----------------------------
        std::unordered_map<uint64_t, uint32_t> seen;
        for (uint32_t i = 0; i < BATCH; ++i) {
            std::span<const uint8_t> digest(&h_out[i * OUT_BYTES], OUT_BYTES);

            uint64_t tag = digest_tag(digest);
            auto [it, inserted] = seen.emplace(tag, i);

            if (!inserted &&
                std::memcmp(digest.data(),
                            &h_out[it->second * OUT_BYTES], OUT_BYTES) == 0 &&
                std::memcmp(&h_msg[i * MSG_BYTES],
                            &h_msg[it->second * MSG_BYTES], MSG_BYTES) != 0)
            {
                std::cout << "***** SHA-1 double collision found! *****\n";

                auto dump = [&](const uint8_t* p) {
                    for (size_t j = 0; j < MSG_BYTES; ++j)
                        printf("%02x", p[j]);
                    std::cout << '\n';
                };
                dump(&h_msg[i * MSG_BYTES]);
                dump(&h_msg[it->second * MSG_BYTES]);
                return 0;
            }
        }
        std::cout << "batch done, no collision yet\n";
    }
}
