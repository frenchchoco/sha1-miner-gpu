#!/bin/bash
#
# SHA-1 OP_NET Miner - Build Script for Linux
# Supports both NVIDIA CUDA and AMD ROCm GPUs
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to find the best CUDA installation
find_cuda() {
    local cuda_paths=()

    # Check common CUDA installation paths
    for path in /usr/local/cuda* /opt/cuda* /usr/lib/cuda; do
        if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
            cuda_paths+=("$path")
        fi
    done

    # Sort by version (newest first)
    if [ ${#cuda_paths[@]} -gt 0 ]; then
        # Get the highest version
        local best_cuda=$(printf '%s\n' "${cuda_paths[@]}" | sort -V -r | head -n1)
        echo "$best_cuda"
        return 0
    fi

    # Check if nvcc is in PATH
    if command -v nvcc &> /dev/null; then
        local nvcc_path=$(which nvcc)
        local cuda_root=$(dirname $(dirname "$nvcc_path"))
        if [ -d "$cuda_root" ]; then
            echo "$cuda_root"
            return 0
        fi
    fi

    return 1
}

# Function to detect GPU architecture
detect_gpu_arch() {
    if ! command -v nvidia-smi &> /dev/null; then
        return 1
    fi

    # Get compute capability
    local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')

    if [ -n "$compute_cap" ]; then
        echo "$compute_cap"
        return 0
    fi

    return 1
}

# Default settings
BUILD_TYPE="Release"
GPU_TYPE=""
HIP_ARCH=""
CMAKE_ARGS=""
THREADS=$(nproc)
CLEAN_BUILD=0
CUDA_ARCH=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --cuda)
            GPU_TYPE="CUDA"
            shift
            ;;
        --hip|--amd)
            GPU_TYPE="HIP"
            shift
            ;;
        --arch)
            if [ "$GPU_TYPE" = "HIP" ] || [ -z "$GPU_TYPE" ]; then
                HIP_ARCH="$2"
            else
                CUDA_ARCH="$2"
            fi
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --threads|-j)
            THREADS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug           Build in debug mode"
            echo "  --cuda            Force CUDA build (NVIDIA)"
            echo "  --hip, --amd      Force HIP build (AMD)"
            echo "  --arch <arch>     Specify GPU architecture"
            echo "  --clean           Clean build directory before building"
            echo "  --threads, -j N   Number of build threads (default: $(nproc))"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect GPU type if not specified
if [ -z "$GPU_TYPE" ]; then
    print_info "Auto-detecting GPU type..."

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_TYPE="CUDA"
            print_info "Detected NVIDIA GPU"
        fi
    fi

    # Check for AMD GPU
    if [ -z "$GPU_TYPE" ]; then
        if [ -d /opt/rocm ] || command -v rocm-smi &> /dev/null; then
            # Check if AMD GPU is actually present
            if [ -f /sys/class/kfd/kfd/topology/nodes/1/gpu_id ] || \
               (command -v rocm-smi &> /dev/null && rocm-smi --showid 2>/dev/null | grep -q "GPU"); then
                GPU_TYPE="HIP"
                print_info "Detected AMD GPU"
            fi
        fi
    fi

    if [ -z "$GPU_TYPE" ]; then
        print_error "No supported GPU detected. Please install CUDA (NVIDIA) or ROCm (AMD)."
        print_info "You can force a build type with --cuda or --hip"
        exit 1
    fi
fi

# NVIDIA-specific setup
if [ "$GPU_TYPE" = "CUDA" ]; then
    # Find best CUDA installation
    CUDA_PATH=$(find_cuda)
    if [ $? -ne 0 ] || [ -z "$CUDA_PATH" ]; then
        print_error "CUDA installation not found!"
        print_info "Please install CUDA toolkit or set CUDA_HOME environment variable"
        exit 1
    fi

    print_info "Found CUDA installation: $CUDA_PATH"

    # Set CUDA environment
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    export CUDA_HOME="$CUDA_PATH"
    export CUDACXX="$CUDA_PATH/bin/nvcc"

    # Verify CUDA version
    CUDA_VERSION=$($CUDA_PATH/bin/nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    print_info "CUDA version: $CUDA_VERSION"

    # Auto-detect GPU architecture if not specified
    if [ -z "$CUDA_ARCH" ]; then
        DETECTED_ARCH=$(detect_gpu_arch)
        if [ $? -eq 0 ] && [ -n "$DETECTED_ARCH" ]; then
            CUDA_ARCH="$DETECTED_ARCH"
            print_info "Detected GPU architecture: sm_$CUDA_ARCH"

            # Map to GPU name for user info
            case "$CUDA_ARCH" in
                50|52) GPU_NAME="Maxwell (GTX 900 series)" ;;
                60) GPU_NAME="Pascal (GTX 10xx, P100)" ;;
                61) GPU_NAME="Pascal (GTX 1050-1080)" ;;
                70) GPU_NAME="Volta (V100, Titan V)" ;;
                75) GPU_NAME="Turing (RTX 20xx, GTX 16xx, T4)" ;;
                80) GPU_NAME="Ampere (A100, A30, A40)" ;;
                86) GPU_NAME="Ampere (RTX 30xx consumer)" ;;
                89) GPU_NAME="Ada Lovelace (RTX 40xx, L4, L40/L40S)" ;;
                90) GPU_NAME="Hopper (H100, H200)" ;;
                100) GPU_NAME="Grace Hopper (GH100, GH200)" ;;
                120) GPU_NAME="Blackwell (RTX 50xx, B100, B200, GB200)" ;;
                *) GPU_NAME="Unknown" ;;
            esac
            print_info "GPU Generation: $GPU_NAME"
        else
            print_warning "Could not auto-detect GPU architecture"
            print_info "Will use default CUDA architectures"
        fi
    fi
fi

# AMD-specific setup
if [ "$GPU_TYPE" = "HIP" ]; then
    if [ -n "$ROCM_PATH" ]; then
        print_info "Using ROCM_PATH: $ROCM_PATH"
    elif [ -d /opt/rocm ]; then
        export ROCM_PATH=/opt/rocm
        print_info "Using ROCM_PATH: $ROCM_PATH"
    else
        print_error "ROCm not found. Please install ROCm or set ROCM_PATH"
        exit 1
    fi

    # Check ROCm version
    if [ -f "$ROCM_PATH/.info/version" ]; then
        ROCM_VERSION=$(cat "$ROCM_PATH/.info/version")
        print_info "ROCm version: $ROCM_VERSION"
    fi

    # Auto-detect AMD GPU architecture if not specified
    if [ -z "$HIP_ARCH" ]; then
        print_info "Detecting AMD GPUs..."

        # Try using rocminfo
        if command -v rocminfo &> /dev/null; then
            # Extract gfx architectures from rocminfo
            DETECTED_ARCHS=$(rocminfo 2>/dev/null | grep -E "^\s*Name:\s+gfx" | sed 's/.*gfx/gfx/' | sort -u | tr '\n' ',')
            # Remove trailing comma
            DETECTED_ARCHS=${DETECTED_ARCHS%,}
        fi

        # Fallback to rocm-smi if rocminfo didn't work
        if [ -z "$DETECTED_ARCHS" ] && command -v rocm-smi &> /dev/null; then
            # Try to detect from device name
            GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -E "Card series|GPU" | head -1)
            case "$GPU_NAME" in
                *"RX 5"*) DETECTED_ARCHS="gfx1010" ;;  # RDNA1
                *"RX 6"*) DETECTED_ARCHS="gfx1030" ;;  # RDNA2
                *"RX 7"*) DETECTED_ARCHS="gfx1100" ;;  # RDNA3
                *"MI100"*) DETECTED_ARCHS="gfx908" ;;
                *"MI200"*) DETECTED_ARCHS="gfx90a" ;;
                *"MI300"*) DETECTED_ARCHS="gfx940" ;;
            esac
        fi

        if [ -n "$DETECTED_ARCHS" ]; then
            HIP_ARCH="$DETECTED_ARCHS"
            print_info "Detected AMD GPU architectures:"
            IFS=',' read -ra ARCH_ARRAY <<< "$HIP_ARCH"
            for arch in "${ARCH_ARRAY[@]}"; do
                case "$arch" in
                    gfx900) echo "  - $arch (Vega 10)" ;;
                    gfx906) echo "  - $arch (Vega 20, Radeon VII)" ;;
                    gfx908) echo "  - $arch (MI100)" ;;
                    gfx90a) echo "  - $arch (MI200 series)" ;;
                    gfx940) echo "  - $arch (MI300 series)" ;;
                    gfx1010) echo "  - $arch (RDNA1 - RX 5500/5600/5700)" ;;
                    gfx1030) echo "  - $arch (RDNA2 - RX 6000 series)" ;;
                    gfx1100) echo "  - $arch (RDNA3 - RX 7000 series)" ;;
                    *) echo "  - $arch" ;;
                esac
            done
        else
            print_warning "Could not auto-detect AMD GPU architecture"
            print_info "Please specify with --arch flag (e.g., --arch gfx1030)"
            print_info "Common architectures:"
            print_info "  gfx900  - Vega 10 (RX Vega 56/64)"
            print_info "  gfx906  - Vega 20 (Radeon VII)"
            print_info "  gfx1010 - RDNA1 (RX 5500/5600/5700)"
            print_info "  gfx1030 - RDNA2 (RX 6000 series)"
            print_info "  gfx1100 - RDNA3 (RX 7000 series)"
            exit 1
        fi
    fi

    print_info "Will build for architectures: $HIP_ARCH"
fi

# Create build directory
BUILD_DIR="build"
if [ "$BUILD_TYPE" = "Debug" ]; then
    BUILD_DIR="build-debug"
fi

if [ $CLEAN_BUILD -eq 1 ] && [ -d "$BUILD_DIR" ]; then
    print_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
if [ "$GPU_TYPE" = "HIP" ]; then
    print_info "Configuring for AMD GPUs..."
    print_info "Building for architectures: $HIP_ARCH"
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DUSE_HIP=ON -DHIP_ARCH=\"$HIP_ARCH\""
else
    print_info "Configuring for NVIDIA GPUs..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc"
    CMAKE_ARGS="$CMAKE_ARGS -DCUDAToolkit_ROOT=$CUDA_PATH"

    # Add compiler flags to fix glibc compatibility issue
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_FLAGS=\"-D_DISABLE_MATHCALLS_LEGACY\""

    if [ -n "$CUDA_ARCH" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
        print_info "Building specifically for architecture: sm_$CUDA_ARCH"
    fi
fi

print_info "Running CMake with args: $CMAKE_ARGS"
eval cmake .. $CMAKE_ARGS

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

# Build
print_info "Building with $THREADS threads..."
make -j$THREADS

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

print_success "Build complete!"
print_info "Executable: $BUILD_DIR/sha1_miner"

# Print usage instructions
echo
echo "To run the miner:"
if [ "$GPU_TYPE" = "HIP" ]; then
    echo "  ./$BUILD_DIR/sha1_miner --gpu 0"
    echo
    echo "Note: You may need to set LD_LIBRARY_PATH:"
    echo "  export LD_LIBRARY_PATH=$ROCM_PATH/lib:\$LD_LIBRARY_PATH"
else
    echo "  ./$BUILD_DIR/sha1_miner --gpu 0"
fi
echo
echo "For help:"
echo "  ./$BUILD_DIR/sha1_miner --help"