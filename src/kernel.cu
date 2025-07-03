#include "sha1_gpu.cuh"

constexpr int  MSG_BYTES = 32;
constexpr int  OUT_BYTES = 20;

constexpr uint32_t  BATCH   = 1 << 24;
constexpr int       THREADS = 1024;

__global__ void sha1_double_kernel(const uint8_t* __restrict__  in,
                                   uint8_t*       __restrict__  out,
                                   uint32_t N)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    /* ---------------- first 512-bit block ------------------------- */
    uint8_t block1[64] = {0};
#pragma unroll
    for (int i = 0; i < MSG_BYTES; ++i)
        block1[i] = in[idx * MSG_BYTES + i];

    block1[MSG_BYTES] = 0x80;
    block1[63]        = static_cast<uint8_t>(MSG_BYTES << 3);   // 256-bit len

    uint32_t H[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,
                     0x10325476,0xC3D2E1F0};

    sha1_compress(block1, H);          // H =   sha1(M)

    /* ---------------- second 512-bit block (double-SHA) ----------- */
    // Here the message is literally the 20-byte BE digest of H
    uint8_t block2[64] = {0};

#pragma unroll
    for (int i = 0; i < 5; ++i)                     // write BE digest
        *(uint32_t*)(block2 + 4 * i) = bswap32(H[i]);

    block2[20] = 0x80;       // padding 0x80
    block2[63] = 160;        // 20 B * 8 bits

    uint32_t H2[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,
                      0x10325476,0xC3D2E1F0};

    sha1_compress(block2, H2);         // H2 = sha1(sha1(M))
#pragma unroll
    for (int i = 0; i < 5; ++i)
        *(uint32_t*)(out + idx * OUT_BYTES + 4 * i) = bswap32(H2[i]);
}
