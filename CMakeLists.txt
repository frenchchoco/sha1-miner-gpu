cmake_minimum_required(VERSION 3.24)
project(sha1_miner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD   20)
set(CMAKE_CUDA_STANDARD  20)

# Uncomment if CMake fails to find CUDA 12.9
# set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")

set(CMAKE_CUDA_ARCHITECTURES 86 89 90)

include_directories(include)

add_library(kernel OBJECT src/kernel.cu)
set_target_properties(kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON)

add_executable(miner src/host.cpp $<TARGET_OBJECTS:kernel>)
target_link_libraries(miner cuda cudart)
