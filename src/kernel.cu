#include "sha1_gpu.cuh"

constexpr int MSG_BYTES = 32;
constexpr int OUT_BYTES = 20;

__global__ void sha1_double_kernel(const uint8_t* in, uint8_t* out, uint32_t N)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint8_t block1[64] = {0};
#pragma unroll
    for (int i = 0; i < MSG_BYTES; ++i)
        block1[i] = in[idx * MSG_BYTES + i];

    block1[MSG_BYTES] = 0x80;
    block1[63]        = MSG_BYTES << 3;

    uint32_t H[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0};
    sha1_compress(block1, H);

    uint8_t block2[64] = {0};
#pragma unroll
    for (int i = 0; i < 5; ++i)
        *(uint32_t*)(block2 + 4*i) = bswap32(H[i]);

    block2[20] = 0x80;
    block2[63] = 160;

    uint32_t H2[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0};
    sha1_compress(block2, H2);

#pragma unroll
    for (int i = 0; i < 5; ++i)
        *(uint32_t*)(out + idx * OUT_BYTES + 4*i) = bswap32(H2[i]);
}
