#!/bin/bash
#
# SHA-1 OP_NET Miner - Linux Dependencies Installer
# This script installs dependencies (system packages) including Intel oneAPI support
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default installation directory
INSTALL_DIR="${1:-$PWD}"

# Wrapper around apt-get that waits for locks and retries (cloud instances run apt on boot)
apt_retry() {
    local max_attempts=12
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if $SUDO apt-get "$@" 2>&1; then
            return 0
        fi
        local exit_code=$?
        # Check if it was a lock error (exit code 100 from apt)
        if [ $attempt -lt $max_attempts ]; then
            print_info "apt-get failed (attempt $attempt/$max_attempts), waiting 10s for lock..."
            sleep 10
        fi
        attempt=$((attempt + 1))
    done
    print_error "apt-get $* failed after $max_attempts attempts"
    return 1
}

# Legacy wrapper kept for compatibility
wait_for_apt() {
    :
}

# Intel mode flag (can be set via environment variable or command line)
INTEL_MODE="${INTEL_MODE:-false}"

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

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --intel)
                INTEL_MODE=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS] [INSTALL_DIR]"
                echo ""
                echo "Options:"
                echo "  --intel    Install Intel oneAPI DPC++ compiler for Intel GPU support"
                echo "  --help     Show this help message"
                echo ""
                echo "Environment variables:"
                echo "  INTEL_MODE=true    Enable Intel oneAPI installation"
                exit 0
                ;;
            *)
                if [ -z "$INSTALL_DIR" ] || [ "$INSTALL_DIR" = "$PWD" ]; then
                    INSTALL_DIR="$1"
                fi
                shift
                ;;
        esac
    done
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
        VER=$(lsb_release -sr)
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        OS=$(echo $DISTRIB_ID | tr '[:upper:]' '[:lower:]')
        VER=$DISTRIB_RELEASE
    elif [ -f /etc/debian_version ]; then
        OS=debian
        VER=$(cat /etc/debian_version)
    else
        OS=$(uname -s)
        VER=$(uname -r)
    fi
}

# Check for sudo (handles root containers without sudo installed)
check_sudo() {
    if [ "$EUID" -eq 0 ] || [ "$(id -u)" -eq 0 ]; then
        SUDO=""
    elif command -v sudo &>/dev/null; then
        if sudo -n true 2>/dev/null; then
            SUDO="sudo"
        else
            print_warning "sudo requires a password. Running as current user (may fail for package installs)."
            SUDO=""
        fi
    else
        print_warning "sudo not installed. Running as current user (may fail for package installs)."
        SUDO=""
    fi
}

# Fix conflicting CUDA apt sources from Docker images (PyTorch, Jupyter, etc.)
fix_cuda_apt_sources() {
    # Only relevant for Debian/Ubuntu
    [ ! -f /etc/debian_version ] && return 0

    print_info "Checking for conflicting CUDA apt sources..."

    local conflicts_found=false

    # Docker images (PyTorch, Jupyter) often add their own CUDA repos
    # that conflict with the official NVIDIA repo (different Signed-By keys)
    for f in /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
        [ ! -f "$f" ] && continue
        if grep -q "developer.download.nvidia.com/compute/cuda" "$f" 2>/dev/null; then
            print_warning "Found potentially conflicting CUDA source: $f"
            $SUDO mv "$f" "$f.bak"
            conflicts_found=true
        fi
    done

    # Remove broken CUDA pinning from Docker images
    for f in /etc/apt/preferences.d/*cuda* /etc/apt/preferences.d/*nvidia*; do
        if [ -f "$f" ]; then
            print_warning "Removing CUDA apt pin: $f"
            $SUDO rm -f "$f"
            conflicts_found=true
        fi
    done

    if [ "$conflicts_found" = true ]; then
        print_info "Cleaned up conflicting sources. Refreshing package lists..."
        apt_retry update
    fi
}

# Install CUDA toolkit (smart version detection from driver)
install_cuda_toolkit() {
    # Add CUDA paths
    for p in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        [ -d "$p" ] && export PATH="$p:$PATH"
    done

    # Skip if nvcc already available
    if command -v nvcc &>/dev/null; then
        local ver
        ver=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        print_success "CUDA toolkit already installed: $ver"
        return 0
    fi

    print_info "CUDA toolkit (nvcc) not found. Installing..."

    # Detect what the driver supports
    local driver_cuda=""
    if command -v nvidia-smi &>/dev/null; then
        driver_cuda=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: //' | sed 's/ .*//')
    fi

    if [ -z "$driver_cuda" ]; then
        print_warning "Cannot detect CUDA version from driver."
        driver_cuda="12.4"
    fi

    local major minor cuda_pkg
    major=$(echo "$driver_cuda" | cut -d. -f1)
    minor=$(echo "$driver_cuda" | cut -d. -f2)
    cuda_pkg="cuda-toolkit-${major}-${minor}"

    print_info "Driver supports CUDA $driver_cuda. Installing $cuda_pkg..."

    if [ -f /etc/debian_version ]; then
        # Clean conflicting sources first
        fix_cuda_apt_sources

        # Detect Ubuntu version
        local ubuntu_ver
        ubuntu_ver=$(grep VERSION_ID /etc/os-release 2>/dev/null | tr -dc '0-9')
        [ -z "$ubuntu_ver" ] && ubuntu_ver="2204"

        # Install cuda-keyring
        local keyring_deb="cuda-keyring_1.1-1_all.deb"
        local keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_ver}/x86_64/${keyring_deb}"

        wget -q "$keyring_url" -O "/tmp/${keyring_deb}" 2>/dev/null && \
            $SUDO dpkg -i "/tmp/${keyring_deb}" && \
            apt_retry update && \
            apt_retry install -y "$cuda_pkg" && \
            print_success "$cuda_pkg installed" && return 0

        # Fallback: try generic cuda-toolkit
        print_warning "Specific version failed, trying generic cuda-toolkit..."
        apt_retry install -y cuda-toolkit 2>/dev/null && return 0

        # Last resort: distro package
        print_warning "NVIDIA repo failed, trying nvidia-cuda-toolkit from distro..."
        apt_retry install -y nvidia-cuda-toolkit 2>/dev/null && return 0

        print_error "Could not install CUDA toolkit. Install manually: https://developer.nvidia.com/cuda-downloads"
        return 1
    else
        print_warning "Non-Debian system. Install CUDA manually: https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
}

# Install Intel oneAPI for Debian/Ubuntu
install_intel_debian() {
    print_info "Installing Intel oneAPI DPC++ compiler for Ubuntu/Debian..."

    # Install prerequisites
    apt_retry update
    apt_retry install -y wget gpg

    # Add Intel's GPG key
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
        $SUDO gpg --dearmor | $SUDO tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

    # Add the repository
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
        $SUDO tee /etc/apt/sources.list.d/oneAPI.list

    # Update and install the compiler
    apt_retry update
    apt_retry install -y intel-oneapi-compiler-dpcpp-cpp

    print_success "Intel oneAPI DPC++ compiler installed successfully"
}

# Install Intel oneAPI for Fedora/RHEL
install_intel_fedora() {
    print_info "Installing Intel oneAPI DPC++ compiler for Fedora/RHEL/CentOS..."

    # Create YUM repository file
    cat << EOF | $SUDO tee /etc/yum.repos.d/oneAPI.repo
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

    # Install the compiler
    $SUDO dnf install -y intel-oneapi-compiler-dpcpp-cpp

    print_success "Intel oneAPI DPC++ compiler installed successfully"
}

# Install Intel oneAPI for openSUSE
install_intel_opensuse() {
    print_info "Installing Intel oneAPI DPC++ compiler for openSUSE..."

    # Add repository
    $SUDO zypper addrepo https://yum.repos.intel.com/oneapi oneAPI

    # Import GPG key
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

    # Install the compiler
    $SUDO zypper install -y intel-oneapi-compiler-dpcpp-cpp

    print_success "Intel oneAPI DPC++ compiler installed successfully"
}

# Install Intel oneAPI for Arch
install_intel_arch() {
    print_info "Installing Intel oneAPI DPC++ compiler for Arch Linux..."
    print_warning "Intel oneAPI is not officially supported on Arch Linux."
    print_info "You can try installing from AUR or use the generic installer:"
    print_info "  yay -S intel-oneapi-basekit"
    print_info "Or download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
}

# Setup Intel oneAPI environment
setup_intel_env() {
    print_info "Setting up Intel oneAPI environment..."

    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        print_info "Sourcing Intel oneAPI environment..."
        source /opt/intel/oneapi/setvars.sh

        # Test installation
        if command -v icpx &> /dev/null; then
            print_success "Intel DPC++ compiler (icpx) is available"
            icpx --version
        else
            print_warning "icpx not found in PATH after sourcing setvars.sh"
        fi

        if command -v sycl-ls &> /dev/null; then
            print_success "SYCL devices list:"
            sycl-ls
        else
            print_warning "sycl-ls not found"
        fi

        # Add to bashrc for persistent setup
        if [ -f ~/.bashrc ]; then
            if ! grep -q "oneapi/setvars.sh" ~/.bashrc; then
                echo "" >> ~/.bashrc
                echo "# Intel oneAPI environment" >> ~/.bashrc
                echo "[ -f /opt/intel/oneapi/setvars.sh ] && source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc
                print_info "Added oneAPI environment setup to ~/.bashrc"
            fi
        fi
    else
        print_warning "Intel oneAPI setvars.sh not found at /opt/intel/oneapi/setvars.sh"
        print_warning "You may need to manually source the environment setup script"
    fi
}

# Install dependencies for Ubuntu/Debian
install_deps_debian() {
    print_info "Installing dependencies for Ubuntu/Debian..."
    apt_retry update
    apt_retry install -y \
        build-essential \
        cmake \
        git \
        libssl-dev \
        libboost-all-dev \
        nlohmann-json3-dev \
        zlib1g-dev \
        libuv1-dev \
        pkg-config \
        wget \
        curl \
        ninja-build \
        ccache

    # Install Intel oneAPI if requested
    if [ "$INTEL_MODE" = true ]; then
        install_intel_debian
    fi
}

# Install dependencies for Fedora/RHEL/CentOS
install_deps_fedora() {
    print_info "Installing dependencies for Fedora/RHEL/CentOS..."
    $SUDO dnf install -y \
        gcc-c++ \
        cmake \
        git \
        openssl-devel \
        boost-devel \
        json-devel \
        zlib-devel \
        libuv-devel \
        pkgconfig \
        wget \
        curl \
        ninja-build

    # Install Intel oneAPI if requested
    if [ "$INTEL_MODE" = true ]; then
        install_intel_fedora
    fi
}

# Install dependencies for Arch Linux
install_deps_arch() {
    print_info "Installing dependencies for Arch Linux..."
    $SUDO pacman -Syu --noconfirm --needed \
        base-devel \
        cmake \
        git \
        openssl \
        boost \
        nlohmann-json \
        zlib \
        libuv \
        pkg-config \
        wget \
        curl \
        ninja

    # Install Intel oneAPI if requested
    if [ "$INTEL_MODE" = true ]; then
        install_intel_arch
    fi
}

# Install dependencies for openSUSE
install_deps_opensuse() {
    print_info "Installing dependencies for openSUSE..."
    $SUDO zypper install -y \
        gcc-c++ \
        cmake \
        git \
        libopenssl-devel \
        boost-devel \
        nlohmann_json-devel \
        zlib-devel \
        libuv-devel \
        pkg-config \
        wget \
        curl \
        ninja

    # Install Intel oneAPI if requested
    if [ "$INTEL_MODE" = true ]; then
        install_intel_opensuse
    fi
}

# Install dependencies for Alpine
install_deps_alpine() {
    print_info "Installing dependencies for Alpine Linux..."
    $SUDO apk add --no-cache \
        build-base \
        cmake \
        git \
        openssl-dev \
        boost-dev \
        nlohmann-json \
        zlib-dev \
        libuv-dev \
        pkgconfig \
        wget \
        curl \
        ninja

    if [ "$INTEL_MODE" = true ]; then
        print_warning "Intel oneAPI is not officially supported on Alpine Linux"
        print_info "Consider using a different distribution for Intel GPU support"
    fi
}

# Check GPU support
check_gpu() {
    print_info "Checking GPU support..."

    local gpu_found=false

    # Check for NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        gpu_found=true
        if ! command -v nvcc &> /dev/null; then
            print_warning "CUDA toolkit not found. You'll need to install CUDA for GPU mining."
            print_warning "Visit: https://developer.nvidia.com/cuda-downloads"
        fi
    fi

    # Check for AMD
    if [ -d /opt/rocm ] || command -v rocm-smi &> /dev/null; then
        print_success "AMD GPU detected"
        gpu_found=true
        if [ ! -d /opt/rocm ]; then
            print_warning "ROCm not found. You'll need to install ROCm for GPU mining."
            print_warning "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
        fi
    fi

    # Check for Intel
    if lspci 2>/dev/null | grep -qi "intel.*graphics\|intel.*gpu\|intel.*arc\|intel.*xe"; then
        print_success "Intel GPU detected"
        gpu_found=true
        if [ "$INTEL_MODE" = true ]; then
            print_info "Intel oneAPI will be installed for Intel GPU support"
        else
            print_warning "Intel GPU detected but --intel flag not set"
            print_info "Run with --intel flag to install Intel oneAPI DPC++ compiler"
        fi
    fi

    if [ "$gpu_found" = false ]; then
        print_warning "No supported GPU detected. You'll need:"
        print_warning "  - CUDA (NVIDIA GPUs)"
        print_warning "  - ROCm (AMD GPUs)"
        print_warning "  - Intel oneAPI DPC++ (Intel GPUs)"
    fi
}

# Main installation
main() {
    [ -t 1 ] && [ -n "${TERM:-}" ] && clear 2>/dev/null || true
    echo "====================================="
    echo "SHA-1 Miner - Linux Dependencies Installer"
    echo "====================================="
    echo

    # Parse arguments
    parse_args "$@"

    echo "Working directory: $INSTALL_DIR"
    if [ "$INTEL_MODE" = true ]; then
        echo -e "${GREEN}Intel mode: ENABLED${NC}"
    fi
    echo

    # Check sudo access
    check_sudo

    # Detect distribution
    detect_distro
    print_info "Detected OS: $OS $VER"
    echo

    # Check GPU
    check_gpu
    echo

    # Install dependencies based on distro
    case $OS in
        ubuntu|debian|linuxmint|pop|elementary|zorin|kali)
            install_deps_debian
            ;;
        fedora|rhel|centos|rocky|almalinux|oracle)
            install_deps_fedora
            ;;
        arch|manjaro|endeavouros|garuda|artix)
            install_deps_arch
            ;;
        opensuse*|suse*)
            install_deps_opensuse
            ;;
        alpine)
            install_deps_alpine
            ;;
        *)
            print_error "Unsupported distribution: $OS"
            print_info "Please install these packages manually:"
            print_info "  - build-essential/base-devel (compiler toolchain)"
            print_info "  - cmake (3.16+)"
            print_info "  - git"
            print_info "  - libssl-dev/openssl-devel"
            print_info "  - libboost-all-dev/boost-devel"
            print_info "  - nlohmann-json3-dev/json-devel"
            print_info "  - zlib1g-dev/zlib-devel"
            print_info "  - libuv1-dev/libuv-devel"
            print_info "  - pkg-config"
            print_info "  - ninja-build (optional but recommended)"
            if [ "$INTEL_MODE" = true ]; then
                print_info "  - Intel oneAPI DPC++ compiler"
                print_info "    Visit: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html"
            fi
            exit 1
            ;;
    esac

    # Setup Intel environment if installed
    if [ "$INTEL_MODE" = true ]; then
        setup_intel_env
    fi

    echo
    print_success "Dependencies installation complete!"
    echo
    echo "Installed packages:"
    echo "  - OpenSSL (SSL/TLS support)"
    echo "  - Boost (system, thread, program-options)"
    echo "  - nlohmann-json (JSON parsing)"
    echo "  - zlib (compression)"
    echo "  - libuv (async I/O)"

    if [ "$INTEL_MODE" = true ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
        echo "  - Intel oneAPI DPC++ compiler (Intel GPU support)"
    fi

    echo
    echo "To build your project:"
    echo "  1. Make sure your source files are in place"

    if [ "$INTEL_MODE" = true ]; then
        echo "  2. Source Intel environment: source /opt/intel/oneapi/setvars.sh"
        echo "  3. Create build directory: mkdir -p build && cd build"
        echo "  4. Configure with Intel compiler: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx"
        echo "  5. Build: make -j\$(nproc)"
    else
        echo "  2. Create build directory: mkdir -p build && cd build"
        echo "  3. Configure: cmake .. -DCMAKE_BUILD_TYPE=Release"
        echo "  4. Build: make -j\$(nproc)"
    fi
    echo

    if [ ! -f "$INSTALL_DIR/CMakeLists.txt" ]; then
        print_warning "No CMakeLists.txt found in current directory."
        print_warning "Make sure you're running this from your project root."
    fi

    if [ "$INTEL_MODE" = true ]; then
        echo
        print_info "Intel oneAPI Notes:"
        print_info "  - Environment setup has been added to ~/.bashrc"
        print_info "  - Run 'source ~/.bashrc' or start a new terminal to load the environment"
        print_info "  - Use 'icpx --version' to verify the compiler is available"
        print_info "  - Use 'sycl-ls' to list available SYCL devices"
    fi
}

# Run main if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi