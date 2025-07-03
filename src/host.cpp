#include "util.hpp"
#include <cuda_runtime.h>
#include <unordered_map>
#include <iostream>
#include <span>
#include <cstring>
#include <cstdlib>

#define CHECK(x) do{auto e=(x); if(e!=cudaSuccess){          \
    std::cerr<<"CUDA error "<<cudaGetErrorString(e)<<'\n';\
    std::exit(1);} }while(0)

extern __global__ void sha1_double_kernel(const uint8_t*,uint8_t*,uint32_t);

constexpr uint32_t  BATCH     = 1<<22;
constexpr size_t    MSG_BYTES = 32;
constexpr size_t    OUT_BYTES = 20;

int main(){
    std::vector<uint8_t> h_msg(BATCH*MSG_BYTES);
    std::vector<uint8_t> h_out(BATCH*OUT_BYTES);

    uint8_t *d_msg,*d_out;
    CHECK(cudaMalloc(&d_msg,h_msg.size()));
    CHECK(cudaMalloc(&d_out,h_out.size()));

    while(true){
        fill_rand(std::span<uint8_t>(h_msg));

        CHECK(cudaMemcpy(d_msg,h_msg.data(),h_msg.size(),
                         cudaMemcpyHostToDevice));

        dim3 grid((BATCH+255)/256), blk(256);
        sha1_double_kernel<<<grid,blk>>>(d_msg,d_out,BATCH);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_out.data(),d_out,h_out.size(),
                         cudaMemcpyDeviceToHost));

        std::unordered_map<uint64_t,uint32_t> seen;
        for(uint32_t i=0;i<BATCH;i++){
            auto digest = std::span<const uint8_t,OUT_BYTES>(
                              &h_out[i*OUT_BYTES], OUT_BYTES);
            uint64_t tag = digest_tag(digest);
            auto [it,ins] = seen.emplace(tag,i);
            if(!ins && std::memcmp(digest.data(),
                                   &h_out[it->second*OUT_BYTES],OUT_BYTES)==0){
                if(std::memcmp(&h_msg[i*MSG_BYTES],
                               &h_msg[it->second*MSG_BYTES], MSG_BYTES)!=0){
                    std::cout<<"collision!\n";
                    for(auto b: std::span<const uint8_t,MSG_BYTES>(&h_msg[i*MSG_BYTES],MSG_BYTES))
                        printf("%02x",b);
                    std::cout<<"\n";
                    for(auto b: std::span<const uint8_t,MSG_BYTES>(&h_msg[it->second*MSG_BYTES],MSG_BYTES))
                        printf("%02x",b);
                    std::cout<<"\n";
                    return 0;
                }
            }
        }
        std::cout<<"batch done\n";
    }
}
