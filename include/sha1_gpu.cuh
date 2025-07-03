#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <bit>          // C++20 rotl/rotr, byteswap

static __forceinline__ __device__
uint32_t rotl32(uint32_t x, int n) {  return std::rotl(x, n); }

template<int R>
static __forceinline__ __device__
void round(uint32_t &a,uint32_t &b,uint32_t &c,uint32_t &d,uint32_t &e,
           uint32_t w)
{
    uint32_t f,k;
    if constexpr (R < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
    else if constexpr (R < 40){ f = b ^ c ^ d;       k = 0x6ED9EBA1; }
    else if constexpr (R < 60){ f = (b & c)|(b & d)|(c & d); k = 0x8F1BBCDC; }
    else                     { f = b ^ c ^ d;       k = 0xCA62C1D6; }

    uint32_t tmp = rotl32(a,5) + f + e + k + w;
    e = d;  d = c;  c = rotl32(b,30);  b = a;  a = tmp;
}

static __device__
void sha1_compress(const uint8_t* __restrict__ msg, uint32_t H[5])
{
    uint32_t w[80];

    #pragma unroll
    for(int i=0;i<16;i++)
        w[i] = std::byteswap(*(reinterpret_cast<const uint32_t*>(msg + 4*i)));

    #pragma unroll 64
    for(int i=16;i<80;i++)
        w[i] = rotl32(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);

    uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4];

    #pragma unroll
    for(int i=0;i<80;i++) round<i>(a,b,c,d,e,w[i]);

    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d; H[4]+=e;
}
