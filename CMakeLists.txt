cmake_minimum_required(VERSION 3.24)
project(sha1_miner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86 89 90 120)
set(CMAKE_CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CMAKE_CXX_FLAGS_RELEASE "/O2 /MD /arch:AVX2 /GL")
set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")

set(CMAKE_CUDA_FLAGS
        "--use_fast_math -Xptxas=-O4,-dlcm=ca,-maxrregcount=64 --allow-unsupported-compiler")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(CUDAToolkit REQUIRED)

# helper
function(add_gpu_target tgt)
    add_library(${tgt} STATIC ${ARGN})
    target_include_directories(${tgt} PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /GL /arch:AVX2>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od /RTC1>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xptxas=-O4,-dlcm=ca,-maxrregcount=64>)
    set_target_properties(${tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

# kernel (contains the __constant__ definitions)
add_gpu_target(kernel src/kernel.cu)

# host-only upload helper
add_library(jobupload STATIC src/job_upload.cpp)
target_include_directories(jobupload PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(jobupload PUBLIC CUDA::cudart)

# SHA-1 library (if needed for verification)
add_library(sha1lib STATIC src/cxxsha1.cpp)
target_include_directories(sha1lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# miner
add_executable(miner src/host.cu)
target_include_directories(miner PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(miner PRIVATE jobupload kernel sha1lib CUDA::cudart)
set_target_properties(miner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# bench
add_executable(bench src/bench.cu)
target_include_directories(bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench PRIVATE jobupload kernel sha1lib CUDA::cudart)
set_target_properties(bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# verify - verification tool
add_executable(verify src/verify_sha1.cpp)
target_include_directories(verify PRIVATE ${CMAKE_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(verify PRIVATE jobupload kernel sha1lib CUDA::cudart)
set_target_properties(verify PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optional: Python script installation
configure_file(${CMAKE_SOURCE_DIR}/scripts/verify_sha1.py
        ${CMAKE_BINARY_DIR}/verify_sha1.py
        COPYONLY)

# IPO / LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok LANGUAGES CXX)
if (ipo_ok)
    set_property(TARGET miner bench kernel verify PROPERTY
            INTERPROCEDURAL_OPTIMIZATION
            $<BOOL:$<NOT:$<CONFIG:Debug>>>)
endif ()

# Print build configuration
message(STATUS "")
message(STATUS "SHA-1 Bitcoin Mining Build Configuration:")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  miner  - Main SHA-1 collision finder")
message(STATUS "  bench  - Performance benchmark")
message(STATUS "  verify - Verification tool")
message(STATUS "")