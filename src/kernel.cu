#include "sha1_gpu.cuh"
#include <bit>

constexpr int MSG_BYTES  = 32;
constexpr int OUT_BYTES  = 20;
constexpr int THREADS    = 256;

__global__ void sha1_double_kernel(
        const uint8_t* __restrict__ in,
        uint8_t*       __restrict__ out,
        uint32_t total)
{
    const uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= total) return;

    alignas(4) uint8_t block1[64] = {0};
    #pragma unroll
    for(int i=0;i<MSG_BYTES;i++)
        block1[i] = in[idx*MSG_BYTES + i];

    block1[MSG_BYTES] = 0x80;
    block1[63]        = MSG_BYTES << 3;        // 256 bits length

    uint32_t H[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,
                     0x10325476,0xC3D2E1F0};

    sha1_compress(block1,H);

    alignas(4) uint8_t block2[64] = {0};
    #pragma unroll
    for(int i=0;i<5;i++){
        uint32_t be = std::byteswap(H[i]);   // LE→BE on little-endian GPUs
        *reinterpret_cast<uint32_t*>(block2 + 4*i) = be;
    }
    block2[20] = 0x80;
    block2[63] = 160;                       // 20*8

    uint32_t H2[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,
                      0x10325476,0xC3D2E1F0};
    sha1_compress(block2,H2);

    #pragma unroll
    for(int i=0;i<5;i++){
        uint32_t be = std::byteswap(H2[i]);
        *reinterpret_cast<uint32_t*>(out + idx*OUT_BYTES + 4*i) = be;
    }
}
