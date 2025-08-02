cmake_minimum_required(VERSION 3.20)
project(sha1_miner_gpu C CXX CUDA)

# --- Vcpkg and Runtime Library Configuration ---

# Uses a relative path to the vcpkg toolchain file.
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg Toolchain File")
# Sets the vcpkg triplet for static compilation
set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "Vcpkg Target Triplet")
# Ensures all libraries are linked statically
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "MSVC runtime library setting")

# --- Build Options ---

# Force C++17 standard for all compilers
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Define WIN32_LEAN_AND_MEAN to fix the WinSock header conflict
add_compile_definitions(WIN32_LEAN_AND_MEAN)

option(USE_HIP "Use HIP/ROCm backend" OFF)
if(USE_HIP)
    message(STATUS "Using HIP backend")
else()
    message(STATUS "Using CUDA backend")
endif()

# Explicitly find the CUDA package to set up all necessary variables
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 80 86 89)

# --- Dependency Search ---
find_package(Boost 1.88 REQUIRED COMPONENTS system thread program_options date_time regex random chrono atomic)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

# --- Executable and Library Definitions ---

add_library(gpu_kernel include/miner/sha1_kernel.cu)

add_executable(sha1_miner
    src/main.cpp
    src/globals.cpp
    src/mining_system.cpp
    src/multi_gpu_manager.cpp
    net/pool_client.cpp
    net/pool_integration.cpp
    net/pool_protocol.cpp
    net/pool_utils.cpp
    include/miner/kernel_launcher.cpp
)

# Add include paths for both C++ and CUDA headers.
target_include_directories(sha1_miner PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/miner"
    "${CUDA_INCLUDE_DIRS}"
)

# The CUDA library also needs to find the project's headers.
target_include_directories(gpu_kernel PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/miner"
    "${CUDA_INCLUDE_DIRS}"
)

# Link the libraries to the main executable
target_link_libraries(sha1_miner PUBLIC
    gpu_kernel
    Boost::system
    Boost::thread
    Boost::program_options
    Boost::date_time
    Boost::regex
    Boost::random
    Boost::chrono
    Boost::atomic
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
    Threads::Threads
)
