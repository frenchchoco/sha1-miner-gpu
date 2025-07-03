cmake_minimum_required(VERSION 3.24)
project(sha1_miner LANGUAGES CXX CUDA)

# Language levels & architectures
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86 89 90 120)     # Ampere / Ada / Hopper / Blackwell

find_package(CUDAToolkit REQUIRED)

add_library(kernel OBJECT src/kernel.cu)

target_compile_options(kernel PRIVATE
        --use_fast_math
        -Xptxas -O3
        -Xptxas -maxrregcount=64)

target_include_directories(kernel PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON     # compile with -rdc=true
        POSITION_INDEPENDENT_CODE ON)

# miner executable   (host.cu + kernel object)
add_executable(miner
        src/host.cu
        $<TARGET_OBJECTS:kernel>)
# set_source_files_properties(src/host.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)    # enables device-link step

target_include_directories(miner PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(miner PRIVATE CUDA::cudart)

# bench executable (bench.cu + kernel object)
add_executable(bench
        src/bench.cu
        $<TARGET_OBJECTS:kernel>)

set_target_properties(bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(bench PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(bench PRIVATE CUDA::cudart)
