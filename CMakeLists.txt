cmake_minimum_required(VERSION 3.16)

# Detect if we should use HIP or CUDA
option(USE_HIP "Use HIP for AMD GPUs instead of CUDA" OFF)

# Force static builds for HIP on Windows due to hipcc limitations
if (USE_HIP AND WIN32)
    set(FORCE_STATIC_BUILD ON)
else()
    set(FORCE_STATIC_BUILD OFF)
endif()

# Set MSVC runtime library BEFORE project()
if (MSVC AND FORCE_STATIC_BUILD)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Define directories at the top
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MINER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/miner)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/net)
set(LOGGING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/logging)
set(CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/configs)
set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# Project setup
if (USE_HIP)
    project(SHA1NearCollisionMiner LANGUAGES CXX)
    set(GPU_BACKEND "HIP")
else()
    project(SHA1NearCollisionMiner LANGUAGES CXX CUDA)
    set(GPU_BACKEND "CUDA")
endif()

# C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Directory Overrides
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}")

# Detect vcpkg triplet
if (NOT DEFINED VCPKG_TARGET_TRIPLET)
    if (WIN32)
        if (USE_HIP)
            set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "vcpkg target triplet")
            message(STATUS "AMD/HIP detected on Windows - forcing static triplet")
        else()
            set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
        endif()
    else()
        set(VCPKG_TARGET_TRIPLET "x64-linux-dynamic" CACHE STRING "vcpkg target triplet")
    endif()
endif()

# Set static/dynamic based on triplet
if (VCPKG_TARGET_TRIPLET MATCHES ".*-static$")
    set(USE_STATIC_LIBS ON)
else()
    set(USE_STATIC_LIBS OFF)
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost 1.70 REQUIRED COMPONENTS system thread program_options date_time regex random)
find_package(ZLIB REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Print include directories for debugging
message(STATUS "INCLUDE_DIR: ${INCLUDE_DIR}")
message(STATUS "LOGGING_DIR: ${LOGGING_DIR}")
message(STATUS "MINER_DIR: ${MINER_DIR}")
message(STATUS "SRC_DIR: ${SRC_DIR}")
message(STATUS "NET_DIR: ${NET_DIR}")
message(STATUS "CONFIG_DIR: ${CONFIG_DIR}")
message(STATUS "CORE_DIR: ${CORE_DIR}")
message(STATUS "UTILS_DIR: ${UTILS_DIR}")
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "OPENSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR}")
message(STATUS "ZLIB_INCLUDE_DIRS: ${ZLIB_INCLUDE_DIRS}")
message(STATUS "NLOHMANN_JSON_INCLUDE_DIR: ${NLOHMANN_JSON_INCLUDE_DIR}")

# Platform-specific optimization flags
if (MSVC)
    if (USE_STATIC_LIBS)
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MT")
        set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MTd")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    endif()
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
    if (USE_STATIC_LIBS)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:msvcrt.lib /NODEFAULTLIB:msvcrtd.lib /NODEFAULTLIB:ucrt.lib /NODEFAULTLIB:ucrtd.lib")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -finline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

# GPU Backend configuration
if (GPU_BACKEND STREQUAL "HIP")
    set(ROCM_PATH /opt/rocm CACHE PATH "Path to ROCm installation")
    set(HIP_ARCH_LIST
        "gfx900;gfx906;gfx908;gfx90a;gfx940;gfx1010;gfx1012;gfx1030;gfx1031;gfx1032;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1200;gfx1201"
    )
    string(REPLACE ";" "," HIP_ARCH "${HIP_ARCH_LIST}")
    string(REPLACE "," ";" HIP_ARCH_LIST "${HIP_ARCH}")
    set(HIP_OFFLOAD_ARCH_FLAGS "")
    foreach(arch IN LISTS HIP_ARCH_LIST)
        string(STRIP ${arch} arch)
        list(APPEND HIP_OFFLOAD_ARCH_FLAGS "--offload-arch=${arch}")
    endforeach()

    set(HIPCC_BINARY "${ROCM_PATH}/bin/hipcc")
    set(HIPCC_FLAGS -O3 -ffast-math -fno-gpu-rdc -fPIC)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o
        COMMAND "${HIPCC_BINARY}"
        ARGS ${HIPCC_FLAGS}
             ${HIP_OFFLOAD_ARCH_FLAGS}
             -I${INCLUDE_DIR} -I${MINER_DIR} -I${OPENSSL_INCLUDE_DIR}
             -DUSE_HIP -D__HIP_PLATFORM_AMD__
             -c ${MINER_DIR}/sha1_kernel_amd.hip.cpp
             -o "${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o"
        DEPENDS ${MINER_DIR}/sha1_kernel_amd.hip.cpp
        COMMENT "Compiling HIP kernel for architectures: ${HIP_ARCH}"
        VERBATIM
    )

    add_custom_target(hip_kernel_target ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o)

    set(GPU_INCLUDES ${ROCM_PATH}/include)
    set(GPU_LIBRARIES
        ${ROCM_PATH}/lib/libamdhip64.so
        ${ROCM_PATH}/lib/libhiprtc.so
    )
    if (EXISTS ${ROCM_PATH}/lib/librocm_smi64.so)
        list(APPEND GPU_LIBRARIES ${ROCM_PATH}/lib/librocm_smi64.so)
        add_definitions(-DHAS_ROCM_SMI)
    endif()
    set(GPU_DEFINES -DUSE_HIP -D__HIP_PLATFORM_AMD__)
    set(KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o")
else()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90;90a;120")
        message(STATUS "Auto-selected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -maxrregcount=0 --extra-device-vectorization")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xptxas -O3,-dlcm=ca,-dscm=wt")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -lineinfo")

    if (USE_STATIC_LIBS)
        set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    else()
        set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    endif()

    add_library(gpu_kernel STATIC ${MINER_DIR}/sha1_kernel.cu)
    set_target_properties(gpu_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(gpu_kernel PRIVATE
        ${INCLUDE_DIR}
        ${MINER_DIR}
        ${OPENSSL_INCLUDE_DIR}
    )

    set(GPU_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
    set(GPU_LIBRARIES gpu_kernel CUDA::cudart_static CUDA::cuda_driver)
    set(GPU_DEFINES "")
    set(KERNEL_OBJECT "")
endif()

# Add pool protocol source files
set(POOL_SOURCES "")
if (EXISTS ${NET_DIR}/pool_protocol.cpp)
    list(APPEND POOL_SOURCES ${NET_DIR}/pool_protocol.cpp)
endif()
if (EXISTS ${NET_DIR}/pool_utils.cpp)
    list(APPEND POOL_SOURCES ${NET_DIR}/pool_utils.cpp)
endif()

# Create main executable
add_executable(sha1_miner
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/core/mining_system.cpp
    ${SRC_DIR}/core/multi_gpu_manager.cpp
    ${SRC_DIR}/core/pool_mining.cpp
    ${SRC_DIR}/core/solo_mining.cpp
    ${SRC_DIR}/utils/platform_utils.cpp
    ${SRC_DIR}/utils/test_utils.cpp
    ${SRC_DIR}/configs/config.cpp
    ${SRC_DIR}/globals.cpp
    ${NET_DIR}/pool_client.cpp
    ${NET_DIR}/pool_integration.cpp
    ${POOL_SOURCES}
    ${MINER_DIR}/kernel_launcher.cpp
)

set_target_properties(sha1_miner PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(sha1_miner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/miner
    ${CMAKE_CURRENT_SOURCE_DIR}/net
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/configs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/logging
    ${GPU_INCLUDES}
    ${Boost_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# Compile options
if (MSVC)
    if (USE_STATIC_LIBS)
        target_compile_options(sha1_miner PRIVATE /MT$<$<CONFIG:Debug>:d>)
    else()
        target_compile_options(sha1_miner PRIVATE /MD$<$<CONFIG:Debug>:d>)
    endif()
else()
    target_compile_options(sha1_miner PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math -funroll-loops>
        $<$<CONFIG:Debug>:-O0 -g>
    )
endif()

# Compile definitions
target_compile_definitions(sha1_miner PRIVATE
    ${GPU_DEFINES}
    BOOST_BIND_GLOBAL_PLACEHOLDERS
    BOOST_NO_CXX17_HDR_STRING_VIEW
    BOOST_ALL_NO_LIB
)

if (USE_HIP AND WIN32)
    target_compile_definitions(sha1_miner PRIVATE BOOST_PROGRAM_OPTIONS_STATIC_LINK)
endif()

if (WIN32)
    target_compile_definitions(sha1_miner PRIVATE
        _WIN32_WINNT=0x0A00
        WIN32_LEAN_AND_MEAN
    )
endif()

# Link libraries
if (GPU_BACKEND STREQUAL "HIP")
    add_dependencies(sha1_miner hip_kernel_target)
    target_link_libraries(sha1_miner PRIVATE
        ${KERNEL_OBJECT}
        ${GPU_LIBRARIES}
        ${Boost_LIBRARIES}
        ZLIB::ZLIB
        OpenSSL::SSL
        OpenSSL::Crypto
        Threads::Threads
    )
else()
    target_link_libraries(sha1_miner PRIVATE
        gpu_kernel
        ${GPU_LIBRARIES}
        ${Boost_LIBRARIES}
        ZLIB::ZLIB
        OpenSSL::SSL
        OpenSSL::Crypto
        Threads::Threads
    )
    if (WIN32)
        target_link_libraries(sha1_miner PRIVATE
            synchronization.lib
            ws2_32.lib
            iphlpapi.lib
            userenv.lib
            psapi.lib
        )
    endif()
endif()

if (nlohmann_json_FOUND)
    target_link_libraries(sha1_miner PRIVATE nlohmann_json::nlohmann_json)
endif()

# Enable LTO if supported
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)
    if (ipo_supported)
        set_property(TARGET sha1_miner PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        if (GPU_BACKEND STREQUAL "CUDA")
            set_property(TARGET gpu_kernel PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
        message(STATUS "Link-time optimization enabled")
    endif()
endif()

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Installation
install(TARGETS sha1_miner RUNTIME DESTINATION bin)

# Build configuration summary
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "GPU Backend: ${GPU_BACKEND}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "vcpkg triplet: ${VCPKG_TARGET_TRIPLET}")
message(STATUS "Library linkage: ${USE_STATIC_LIBS}")
if (GPU_BACKEND STREQUAL "CUDA")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
endif()
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost.Beast: Available (header-only)")
message(STATUS "OpenSSL version: ${OPENSSL_VERSION}")
message(STATUS "zlib found: ${ZLIB_FOUND}")
message(STATUS "")
