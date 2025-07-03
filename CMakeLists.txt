cmake_minimum_required(VERSION 3.24)
project(sha1_miner_optimized LANGUAGES CXX CUDA)

# ────────── GLOBAL TOOLCHAIN ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86 89 90 120)
set(CMAKE_CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# host flags (MSVC)
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /MD /arch:AVX2 /GL")
set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")

# device flags
set(CMAKE_CUDA_FLAGS
        "--use_fast_math -Xptxas=-O4,-dlcm=ca,-maxrregcount=64 --allow-unsupported-compiler")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

# ────────── helper to build GPU static libraries ─────────────────────────
function(add_gpu_target tgt)
    add_library(${tgt} STATIC ${ARGN})

    target_include_directories(${tgt} PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS})

    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /GL /arch:AVX2>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od /RTC1>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xptxas=-O4,-dlcm=ca,-maxrregcount=64>)

    set_target_properties(${tgt} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)
endfunction()

# ────────── TARGETS ──────────────────────────────────────────────────────
# 1. Device kernel library
add_gpu_target(kernel src/kernel.cu)

# 2. Mining executable
add_executable(miner src/host.cu)
target_include_directories(miner PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(miner PRIVATE kernel CUDA::cudart)
set_target_properties(miner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 3. Benchmark executable (NEW)
add_executable(bench src/bench.cu)
target_include_directories(bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench PRIVATE kernel CUDA::cudart)
set_target_properties(bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ────────── IPO / LTO -----------------------------------------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok LANGUAGES CXX)
if (ipo_ok)
    set_property(TARGET miner bench kernel PROPERTY
            INTERPROCEDURAL_OPTIMIZATION
            $<BOOL:$<NOT:$<CONFIG:Debug>>>)
endif ()
