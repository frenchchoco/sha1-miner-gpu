#pragma once
#include <random>
#include <span>
#include <cstdint>
#include <cstring>

inline void fill_rand(std::span<uint8_t> buf){
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint32_t> d;
    for(size_t i=0;i<buf.size(); i+=4){
        uint32_t x=d(rng);
        std::memcpy(buf.data()+i,&x, std::min<size_t>(4,buf.size()-i));
    }
}

inline uint64_t digest_tag(const std::span<const uint8_t,20> d){
    uint64_t tag; std::memcpy(&tag,d.data(),8);
    return tag;
}
