cmake_minimum_required(VERSION 3.20)
project(sha1_miner_gpu)

option(USE_HIP "Use HIP/ROCm backend" OFF)

if(USE_HIP)
    project(sha1_miner_gpu C CXX HIP)
    message(STATUS "Using HIP backend")
else()
    project(sha1_miner_gpu C CXX CUDA)
    message(STATUS "Using CUDA backend")
endif()

# --- Vcpkg and Runtime Library Configuration ---
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg Toolchain File")
set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "Vcpkg Target Triplet")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "MSVC runtime library setting")

# --- Build Options ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(WIN32_LEAN_AND_MEAN)

if(USE_HIP)
    find_package(HIP REQUIRED)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
else()
    find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# --- Dependency Search ---
find_package(Boost 1.88 REQUIRED COMPONENTS system thread program_options date_time regex random chrono atomic)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

# --- Executable and Library Definitions ---
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

# Set up the GPU kernel library conditionally
if(USE_HIP)
    add_library(gpu_kernel SHARED include/miner/sha1_kernel.cpp)
    set_source_files_properties(include/miner/sha1_kernel.cpp PROPERTIES LANGUAGE HIP)
else()
    add_library(gpu_kernel SHARED include/miner/sha1_kernel.cu)
endif()

# Add include paths for both C++ and GPU headers.
target_include_directories(sha1_miner PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/miner"
)
target_include_directories(gpu_kernel PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/miner"
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
