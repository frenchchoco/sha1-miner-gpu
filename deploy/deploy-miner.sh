#!/bin/bash
#
# Deploy SHA-1 GPU miner to a rented machine and connect it to your solo pool
# via SSH tunnel (no public exposure needed).
#
# Universal script — works on Clore.ai, Vast.ai, RunPod, Lambda Labs, bare-metal.
#
# Usage:
#   ./deploy-miner.sh <user@host> [options]
#   ./deploy-miner.sh <user@host:port> [options]
#   ./deploy-miner.sh --config configs/clore-4080.conf
#
# Examples:
#   ./deploy-miner.sh root@n1.de.clorecloud.net --ssh-port 2396
#   ./deploy-miner.sh root@n1.de.clorecloud.net:2396   # shorthand
#   ./deploy-miner.sh root@ssh5.vast.ai --ssh-port 51234
#   ./deploy-miner.sh root@203.0.113.50 --force-amd
#   ./deploy-miner.sh --config deploy/configs/clore-4080.conf
#
# Prerequisites:
#   - SSH access to the rented GPU machine (key-based auth recommended)
#   - Pool server running on this machine (localhost:3333)
#

set -euo pipefail

# ==================== CONFIGURATION ====================
# Override these via CLI flags, environment variables, or --config file

MINER_REPO="https://github.com/frenchchoco/sha1-miner-gpu.git"
POOL_VM="${POOL_VM:-yoztheripper@192.168.1.12}"
POOL_PORT="${POOL_PORT:-3333}"
WALLET="${WALLET:-opt1pe62t3jaulavawcrerpqz97t2cdrekxe47sxmcyq2pqc8x27v4x0sh3zqvn}"
WORKER_NAME="${WORKER_NAME:-rented-gpu}"
TUNNEL_LOCAL_PORT="${TUNNEL_LOCAL_PORT:-3333}"
MINER_EXTRA_ARGS="${MINER_EXTRA_ARGS:---auto-bench}"

# =======================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()     { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
header()  { echo -e "\n${CYAN}=== $* ===${NC}\n"; }

# ==================== PROVIDER DETECTION ====================

detect_provider() {
    local host="$1"
    case "$host" in
        *clorecloud.net*|*clore.ai*|*cloreai.ru*)
            PROVIDER="clore"
            [[ -z "$SSH_CIPHER" ]] && SSH_CIPHER="aes128-ctr"
            info "Provider: Clore.ai (auto-set cipher=aes128-ctr)"
            ;;
        *vast.ai*)
            PROVIDER="vast"
            info "Provider: Vast.ai"
            ;;
        *runpod.io*|*runpod.net*)
            PROVIDER="runpod"
            info "Provider: RunPod"
            ;;
        *lambda*|*lambdalabs*)
            PROVIDER="lambda"
            info "Provider: Lambda Labs"
            ;;
        *)
            PROVIDER="bare-metal"
            ;;
    esac
}

# ==================== REMOTE PREAMBLE ====================
# Injected at the start of every remote SSH command.
# Fixes: sudo absent, TERM missing, CUDA not in PATH, apt locks.

REMOTE_PREAMBLE='
    # --- Universal preamble ---
    set -e
    export DEBIAN_FRONTEND=noninteractive
    export TERM=${TERM:-dumb}

    # Determine sudo wrapper (containers often run as root without sudo)
    if [ "$(id -u)" -eq 0 ]; then
        SUDO=""
    elif command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
        SUDO="sudo"
    else
        SUDO=""
    fi

    # Add CUDA paths (cloud images pre-install but dont add to PATH)
    for p in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        [ -d "$p" ] && export PATH="$p:$PATH"
    done

    # Mini apt wrapper with retry (handles boot-time apt locks)
    apt_safe() {
        local attempts=0
        while [ $attempts -lt 12 ]; do
            if $SUDO apt-get "$@" 2>&1; then
                return 0
            fi
            attempts=$((attempts + 1))
            echo "[APT] Waiting for lock (attempt $attempts/12)..."
            sleep 10
        done
        echo "[APT] apt-get $* failed after 12 attempts"
        return 1
    }
    # --- End preamble ---
'

# ==================== USAGE ====================

usage() {
    cat <<'EOF'
Usage: deploy-miner.sh <user@host[:port]> [options]

Required:
  <user@host>             SSH target for the rented GPU machine
                          Supports user@host:port shorthand

Options:
  --pool-vm <user@host>   Pool VM SSH target (default: yoztheripper@192.168.1.12)
  --pool-port <port>      Pool WebSocket port (default: 3333)
  --wallet <address>      Wallet address for mining
  --worker <name>         Worker name (default: rented-gpu)
  --force-cuda            Force NVIDIA/CUDA build
  --force-amd             Force AMD/HIP build
  --miner-args <args>     Extra miner arguments (default: --auto-bench)
  --tunnel-mode <mode>    Tunnel mode: "local" or "remote" (default: local)
  --ssh-port <port>       SSH port on rented machine (default: auto-detect or 22)
  --ssh-cipher <cipher>   SSH cipher to use (auto-detected for known providers)
  --config <file>         Load deployment config from file
  --save-config <file>    Save deployment config after successful deploy
  --resume                Resume from last successful step
  --from-step <N>         Start from step N (1-7)
  --dry-run               Print commands without executing
  --help                  Show this help

Providers auto-detected from hostname:
  Clore.ai    (*clorecloud.net)  → auto-sets cipher=aes128-ctr
  Vast.ai     (*vast.ai)         → key-only auth
  RunPod      (*runpod.io)       → SSH may need template config
  Lambda Labs (*lambda*)         → standard SSH
  Bare-metal  (everything else)  → defaults
EOF
    exit 0
}

# ==================== PARSE ARGS ====================

REMOTE_HOST=""
FORCE_GPU=""
SSH_PORT=""
SSH_CIPHER=""
TUNNEL_MODE="local"
DRY_RUN=false
PROVIDER=""
CONFIG_FILE=""
SAVE_CONFIG=""
RESUME=false
START_STEP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pool-vm)      POOL_VM="$2"; shift 2 ;;
        --pool-port)    POOL_PORT="$2"; shift 2 ;;
        --wallet)       WALLET="$2"; shift 2 ;;
        --worker)       WORKER_NAME="$2"; shift 2 ;;
        --force-cuda)   FORCE_GPU="cuda"; shift ;;
        --force-amd)    FORCE_GPU="amd"; shift ;;
        --miner-args)   MINER_EXTRA_ARGS="$2"; shift 2 ;;
        --ssh-port)     SSH_PORT="$2"; shift 2 ;;
        --ssh-cipher)   SSH_CIPHER="$2"; shift 2 ;;
        --tunnel-mode)  TUNNEL_MODE="$2"; shift 2 ;;
        --config)       CONFIG_FILE="$2"; shift 2 ;;
        --save-config)  SAVE_CONFIG="$2"; shift 2 ;;
        --resume)       RESUME=true; shift ;;
        --from-step)    START_STEP="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help|-h)      usage ;;
        -*)             err "Unknown option: $1" ;;
        *)
            if [[ -z "$REMOTE_HOST" ]]; then
                REMOTE_HOST="$1"
            else
                err "Unexpected argument: $1"
            fi
            shift
            ;;
    esac
done

# Load config file if specified
if [[ -n "$CONFIG_FILE" ]] && [[ -f "$CONFIG_FILE" ]]; then
    info "Loading config from $CONFIG_FILE"
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

[[ -z "$REMOTE_HOST" ]] && err "Missing required argument: <user@host>\nRun with --help for usage."

# Parse user@host:port shorthand
if [[ "$REMOTE_HOST" == *:* ]] && [[ -z "$SSH_PORT" ]]; then
    SSH_PORT="${REMOTE_HOST##*:}"
    REMOTE_HOST="${REMOTE_HOST%:*}"
fi

# Default SSH port
[[ -z "$SSH_PORT" ]] && SSH_PORT="22"

# ==================== HELPERS ====================

# Auto-detect provider from hostname
detect_provider "$REMOTE_HOST"

# SSH options (port-aware, cipher-aware)
SSH_OPTS=(-o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p "$SSH_PORT")
[[ -n "$SSH_CIPHER" ]] && SSH_OPTS+=(-c "$SSH_CIPHER")

# Run a single command on the remote host
run_remote() {
    if $DRY_RUN; then
        echo "[DRY-RUN] ssh -p $SSH_PORT $REMOTE_HOST: $*"
    else
        ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
    fi
}

# Run a multi-line script on the remote host (with preamble injected)
run_remote_script() {
    local script
    script="$REMOTE_PREAMBLE"$'\n'"$(cat)"
    if $DRY_RUN; then
        echo "[DRY-RUN] Would run script on $REMOTE_HOST"
    else
        ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "bash -s" <<< "$script"
    fi
}

# Mark a step as completed on the remote machine (for --resume)
mark_step() {
    run_remote "echo 'STEP_$1' > ~/.miner-deploy-state" 2>/dev/null || true
}

# Check if we should skip a step (for --resume / --from-step)
should_skip() {
    local step_num="$1"
    [[ "$step_num" -lt "$START_STEP" ]]
}

# SSH connectivity test with cipher auto-fallback
test_ssh_connection() {
    # First attempt with current settings
    info "Testing: ssh ${SSH_OPTS[*]} $REMOTE_HOST 'echo ok'"
    if timeout 20 ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "echo ok" 2>/dev/null; then
        return 0
    fi

    # If cipher wasn't explicitly set, try aes128-ctr (fixes Clore.ai and others)
    if [[ -z "${USER_SET_CIPHER:-}" ]]; then
        warn "SSH connection failed. Trying cipher aes128-ctr..."
        SSH_CIPHER="aes128-ctr"
        SSH_OPTS+=(-c "$SSH_CIPHER")
        if timeout 20 ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "echo ok" 2>/dev/null; then
            success "Connected with cipher aes128-ctr"
            return 0
        fi
    fi

    return 1
}

# ==================== RESUME SUPPORT ====================

if $RESUME; then
    DEPLOY_STATE=$(run_remote "cat ~/.miner-deploy-state 2>/dev/null" || echo "STEP_0")
    case "$DEPLOY_STATE" in
        *STEP_7*) info "All steps already completed. Use --from-step 1 to force re-deploy."; exit 0 ;;
        *STEP_6*) START_STEP=7 ;;
        *STEP_5*) START_STEP=6 ;;
        *STEP_4*) START_STEP=5 ;;
        *STEP_3*) START_STEP=4 ;;
        *STEP_2*) START_STEP=3 ;;
        *STEP_1*) START_STEP=2 ;;
        *)        START_STEP=1 ;;
    esac
    info "Resuming from step $START_STEP"
fi

# Remember if user explicitly set the cipher (don't auto-fallback if they did)
[[ -n "$SSH_CIPHER" ]] && USER_SET_CIPHER="$SSH_CIPHER"

# ==================== MAIN ====================

header "OP_NET GPU Miner Deployment"
info "Target:     $REMOTE_HOST (port $SSH_PORT${SSH_CIPHER:+, cipher $SSH_CIPHER})"
info "Provider:   $PROVIDER"
info "Pool port:  $POOL_PORT"
info "Wallet:     ${WALLET:0:30}..."
info "Worker:     $WORKER_NAME"
info "Tunnel:     $TUNNEL_MODE"
[[ "$START_STEP" -gt 1 ]] && info "Starting from step: $START_STEP"
echo

# ---- Step 1: Test SSH connectivity ----
if ! should_skip 1; then
    header "Step 1/7: Testing SSH connectivity"
    if $DRY_RUN; then
        info "[DRY-RUN] Would test SSH to $REMOTE_HOST"
    else
        if ! test_ssh_connection; then
            err "Cannot SSH to $REMOTE_HOST. Check your credentials and network."
        fi
        success "SSH connection OK"
    fi
    mark_step 1
fi

# ---- Step 2: Detect GPU type ----
if ! should_skip 2; then
    header "Step 2/7: Detecting GPU on remote machine"

    if [[ -n "$FORCE_GPU" ]]; then
        GPU_TYPE="$FORCE_GPU"
        info "Forced GPU type: $GPU_TYPE"
    else
        GPU_TYPE=$(run_remote 'bash -s' <<'DETECT_EOF'
            if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
                echo "cuda"
            elif [ -d /opt/rocm ] || command -v rocm-smi &>/dev/null; then
                echo "amd"
            else
                echo "unknown"
            fi
DETECT_EOF
        )
        GPU_TYPE=$(echo "$GPU_TYPE" | tr -d '[:space:]')

        case "$GPU_TYPE" in
            cuda) success "Detected NVIDIA GPU" ;;
            amd)  success "Detected AMD GPU" ;;
            *)    err "No supported GPU detected on $REMOTE_HOST. Use --force-cuda or --force-amd." ;;
        esac
    fi

    # Get GPU details
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        GPU_INFO=$(run_remote "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -5" || echo "unknown")
        info "GPU(s): $GPU_INFO"
    elif [[ "$GPU_TYPE" == "amd" ]]; then
        GPU_INFO=$(run_remote "rocm-smi --showproductname 2>/dev/null | head -10 || echo 'AMD GPU detected'" || echo "unknown")
        info "GPU(s): $GPU_INFO"
    fi
    mark_step 2
else
    # Need GPU_TYPE for later steps even when skipping
    GPU_TYPE=$(run_remote 'bash -s' <<'DETECT_EOF'
        if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
            echo "cuda"
        elif [ -d /opt/rocm ] || command -v rocm-smi &>/dev/null; then
            echo "amd"
        else
            echo "unknown"
        fi
DETECT_EOF
    )
    GPU_TYPE=$(echo "$GPU_TYPE" | tr -d '[:space:]')
fi

# ---- Step 3: Clone the miner repo ----
if ! should_skip 3; then
    header "Step 3/7: Cloning miner repo"

    run_remote_script <<CLONE_EOF
    cd ~

    # Ensure git is available (minimal bootstrap)
    if ! command -v git &>/dev/null; then
        echo "[CLONE] Installing git..."
        apt_safe update -qq && apt_safe install -y -qq git 2>/dev/null || \
        \$SUDO dnf install -y git 2>/dev/null || \
        \$SUDO pacman -Sy --noconfirm git 2>/dev/null || true
    fi

    # Clone or update the miner repo
    if [ -d sha1-miner-gpu ]; then
        echo "[CLONE] Updating existing repo..."
        cd sha1-miner-gpu
        git pull --ff-only || { echo "[CLONE] Pull failed, re-cloning..."; cd ~; rm -rf sha1-miner-gpu; git clone $MINER_REPO; cd sha1-miner-gpu; }
    else
        echo "[CLONE] Cloning miner repo..."
        git clone $MINER_REPO
        cd sha1-miner-gpu
    fi
    echo "[CLONE] Repo ready at ~/sha1-miner-gpu"
CLONE_EOF
    success "Repo cloned"
    mark_step 3
fi

# ---- Step 4: Install dependencies using repo's install.sh ----
if ! should_skip 4; then
    header "Step 4/7: Installing build dependencies"

    info "Using the miner's install.sh (handles all distros and packages)"
    run_remote_script <<'DEPS_EOF'
    cd ~/sha1-miner-gpu

    # Run the repo's own install script (supports Ubuntu, Fedora, Arch, openSUSE, Alpine)
    echo "[DEPS] Running install.sh from the miner repo..."
    chmod +x install.sh
    ./install.sh

    echo ""
    echo "[DEPS] Checking GPU toolkit..."
DEPS_EOF

    # Install GPU-specific toolkit if missing — strict version matching
    # Strategy: apt first → .run installer fallback (handles broken repos, Docker images, etc.)
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        CUDA_INSTALL_RESULT=$(run_remote_script <<'CUDA_EOF'
        # 1. Detect driver's max supported CUDA version
        DRIVER_CUDA_VER=""
        if command -v nvidia-smi &>/dev/null; then
            DRIVER_CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: //' | sed 's/ .*//')
        fi

        if [ -z "$DRIVER_CUDA_VER" ]; then
            echo "CUDA_INSTALL_FAIL:no_driver"
            echo "[DEPS] ERROR: nvidia-smi not found or no CUDA version reported"
            exit 0
        fi

        CUDA_MAJOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f2)
        echo "[DEPS] Driver supports CUDA $DRIVER_CUDA_VER (max)"

        # 2. Check if nvcc already exists AND matches driver version
        for p in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
            [ -d "$p" ] && export PATH="$p:$PATH"
        done

        if command -v nvcc &>/dev/null; then
            NVCC_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
            NVCC_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
            NVCC_MINOR=$(echo "$NVCC_VER" | cut -d. -f2)

            if [ "$NVCC_MAJOR" -le "$CUDA_MAJOR" ] && { [ "$NVCC_MAJOR" -lt "$CUDA_MAJOR" ] || [ "$NVCC_MINOR" -le "$CUDA_MINOR" ]; }; then
                echo "[DEPS] CUDA toolkit already installed and compatible: nvcc $NVCC_VER <= driver $DRIVER_CUDA_VER"
                echo "CUDA_INSTALL_OK:existing=$NVCC_VER:driver=$DRIVER_CUDA_VER"
                exit 0
            else
                echo "[DEPS] WARNING: Existing nvcc $NVCC_VER > driver $DRIVER_CUDA_VER — removing and reinstalling"
                $SUDO apt-get remove -y --purge 'cuda-toolkit*' 'nvidia-cuda-toolkit*' 2>/dev/null || true
            fi
        fi

        # 3. Try apt install first (fast if repo works)
        echo "[DEPS] Installing cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}..."
        APT_OK=false

        if [ -f /etc/debian_version ]; then
            # Clean conflicting sources from Docker/Jupyter images
            for f in /etc/apt/sources.list.d/*cuda* /etc/apt/sources.list.d/*nvidia*; do
                [ -f "$f" ] && { echo "[DEPS] Removing conflicting source: $f"; $SUDO rm -f "$f"; }
            done
            $SUDO rm -f /etc/apt/preferences.d/*cuda* /etc/apt/preferences.d/*nvidia* 2>/dev/null || true

            # Detect Ubuntu version
            UBUNTU_VER=$(lsb_release -rs 2>/dev/null | tr -d '.')
            [ -z "$UBUNTU_VER" ] && UBUNTU_VER=$(grep VERSION_ID /etc/os-release 2>/dev/null | tr -dc '0-9')
            [ -z "$UBUNTU_VER" ] && UBUNTU_VER="2204"
            REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/x86_64"

            # Add NVIDIA repo — both keyring AND explicit sources list (DEB822 compat for Ubuntu 24.04+)
            CUDA_DEB="cuda-keyring_1.1-1_all.deb"
            echo "[DEPS] Adding NVIDIA CUDA repo for Ubuntu ${UBUNTU_VER}..."
            wget -q "${REPO_BASE}/${CUDA_DEB}" -O "/tmp/${CUDA_DEB}" 2>/dev/null && \
                $SUDO dpkg -i "/tmp/${CUDA_DEB}" 2>/dev/null

            # Ensure sources list exists (cuda-keyring may only install GPG key on Ubuntu 24.04+)
            if [ ! -f /etc/apt/sources.list.d/cuda*.list ] && [ ! -f /etc/apt/sources.list.d/cuda*.sources ]; then
                echo "[DEPS] Creating explicit CUDA sources list (DEB822 workaround)..."
                echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] ${REPO_BASE}/ /" | \
                    $SUDO tee /etc/apt/sources.list.d/cuda-nvidia.list > /dev/null
            fi

            $SUDO apt-get update -qq 2>/dev/null

            # Verify repo is actually serving packages
            if apt-cache search "^cuda-toolkit-${CUDA_MAJOR}" 2>/dev/null | grep -q "cuda-toolkit"; then
                CUDA_PKG="cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"
                if $SUDO apt-get install -y "$CUDA_PKG" 2>/dev/null; then
                    echo "[DEPS] $CUDA_PKG installed via apt"
                    APT_OK=true
                else
                    echo "[DEPS] $CUDA_PKG failed, trying cuda-toolkit-${CUDA_MAJOR}..."
                    if $SUDO apt-get install -y "cuda-toolkit-${CUDA_MAJOR}" 2>/dev/null; then
                        echo "[DEPS] cuda-toolkit-${CUDA_MAJOR} installed via apt"
                        APT_OK=true
                    fi
                fi
            else
                echo "[DEPS] NVIDIA apt repo has no cuda-toolkit packages — skipping apt"
            fi
        fi

        # 4. Fallback: NVIDIA .run installer (works on any Linux, no repo needed)
        if [ "$APT_OK" = false ]; then
            echo "[DEPS] apt install failed or unavailable — falling back to NVIDIA .run installer"

            # Map driver CUDA version to closest .run installer URL
            # Format: cuda_MAJOR.MINOR.PATCH_DRIVER_linux.run
            # We use the .0 patch release for each minor version
            CUDA_RUN_VER="${CUDA_MAJOR}.${CUDA_MINOR}.0"
            case "${CUDA_MAJOR}.${CUDA_MINOR}" in
                12.0) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run" ;;
                12.1) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run" ;;
                12.2) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run" ;;
                12.3) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run" ;;
                12.4) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run" ;;
                12.5) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run" ;;
                12.6) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run" ;;
                12.8) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run" ;;
                13.0) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_575.51.03_linux.run" ;;
                13.1) CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_575.57.08_linux.run" ;;
                *)
                    echo "CUDA_INSTALL_FAIL:no_run_url"
                    echo "[DEPS] ERROR: No .run installer URL mapped for CUDA ${CUDA_MAJOR}.${CUDA_MINOR}"
                    echo "[DEPS] Check https://developer.nvidia.com/cuda-toolkit-archive for the correct URL"
                    exit 0
                    ;;
            esac

            echo "[DEPS] Downloading CUDA ${CUDA_MAJOR}.${CUDA_MINOR} .run installer (~4GB)..."
            echo "[DEPS] URL: $CUDA_RUN_URL"
            if wget -q --show-progress "$CUDA_RUN_URL" -O /tmp/cuda_installer.run 2>&1; then
                chmod +x /tmp/cuda_installer.run
                echo "[DEPS] Installing CUDA toolkit (--toolkit --silent)..."
                if /tmp/cuda_installer.run --toolkit --silent --override 2>&1; then
                    echo "[DEPS] .run installer completed"
                else
                    echo "CUDA_INSTALL_FAIL:run_failed"
                    echo "[DEPS] ERROR: .run installer failed"
                    rm -f /tmp/cuda_installer.run
                    exit 0
                fi
                rm -f /tmp/cuda_installer.run
            else
                echo "CUDA_INSTALL_FAIL:download_failed"
                echo "[DEPS] ERROR: Failed to download .run installer from $CUDA_RUN_URL"
                exit 0
            fi
        fi

        # 5. Update PATH and verify immediately (within same SSH session)
        for p in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
            [ -d "$p" ] && export PATH="$p:$PATH"
        done

        if command -v nvcc &>/dev/null; then
            FINAL_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
            echo "[DEPS] Verified: nvcc $FINAL_VER installed"
            echo "CUDA_INSTALL_OK:installed=$FINAL_VER:driver=$DRIVER_CUDA_VER"
        else
            echo "CUDA_INSTALL_FAIL:nvcc_missing_after_install"
            echo "[DEPS] ERROR: nvcc still not found after install"
        fi
CUDA_EOF
        )
        # Show full output so the user sees what happened (filter out the result tag)
        echo "$CUDA_INSTALL_RESULT" | grep -v "^CUDA_INSTALL_"
        # Parse the result line
        CUDA_RESULT=$(echo "$CUDA_INSTALL_RESULT" | grep "^CUDA_INSTALL_" | tail -1)

        case "$CUDA_RESULT" in
            CUDA_INSTALL_OK*)
                success "CUDA toolkit: $CUDA_RESULT"
                ;;
            *no_driver*)
                err "No NVIDIA driver found (nvidia-smi failed). This machine cannot run CUDA miners."
                ;;
            *apt_failed*|*run_failed*|*download_failed*|*no_run_url*)
                err "CUDA toolkit install failed (both apt and .run). Details above."
                ;;
            *nvcc_missing_after_install*)
                err "CUDA toolkit installed but nvcc not in PATH. Broken install."
                ;;
            *)
                err "CUDA toolkit install failed: $CUDA_RESULT"
                ;;
        esac
    elif [[ "$GPU_TYPE" == "amd" ]]; then
        run_remote_script <<'AMD_EOF'
        if [ -d /opt/rocm ]; then
            echo "[DEPS] ROCm found at /opt/rocm"
            rocm-smi --version 2>/dev/null || true
        else
            echo "[DEPS] WARNING: ROCm not found at /opt/rocm"
            echo "[DEPS] Most GPU rental providers pre-install ROCm for AMD machines."
            echo "[DEPS] If not, install ROCm 6.2+: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
        fi
AMD_EOF
    fi

    success "Dependencies ready"
    mark_step 4
fi

# ---- Step 5: Build the miner ----
if ! should_skip 5; then
    header "Step 5/7: Building the miner"

    BUILD_FLAG=""
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        BUILD_FLAG="--cuda"
    elif [[ "$GPU_TYPE" == "amd" ]]; then
        BUILD_FLAG="--hip"
    fi

    run_remote_script <<BUILD_EOF
    cd ~/sha1-miner-gpu

    # Clean CMake cache if CUDA compiler path changed (prevents stale cache pointing to wrong nvcc)
    if [ -f build/CMakeCache.txt ]; then
        CACHED_NVCC=\$(grep "CMAKE_CUDA_COMPILER:FILEPATH" build/CMakeCache.txt 2>/dev/null | cut -d= -f2)
        if [ -n "\$CACHED_NVCC" ] && [ ! -f "\$CACHED_NVCC" ]; then
            echo "[BUILD] Stale CMake cache: \$CACHED_NVCC no longer exists — cleaning build dir"
            rm -rf build
        fi
    fi

    # Build
    echo "[BUILD] Building with: ./build.sh $BUILD_FLAG"
    chmod +x build.sh
    ./build.sh $BUILD_FLAG

    # Verify binary exists
    if [ -f build/sha1_miner ]; then
        echo "[BUILD] Binary ready: ~/sha1-miner-gpu/build/sha1_miner"
    elif [ -f build-release/sha1_miner ]; then
        echo "[BUILD] Binary ready: ~/sha1-miner-gpu/build-release/sha1_miner"
    else
        echo "[BUILD] ERROR: sha1_miner binary not found after build!"
        ls -la build*/ 2>/dev/null
        exit 1
    fi
BUILD_EOF
    success "Miner built"
    mark_step 5
fi

# ---- Step 6: Set up SSH tunnel ----
if ! should_skip 6; then
    header "Step 6/7: Setting up SSH tunnel to pool"

    if [[ "$TUNNEL_MODE" == "local" ]]; then
        info "Setting up reverse tunnel: rented:$TUNNEL_LOCAL_PORT -> localhost:$POOL_PORT (this machine)"

        if $DRY_RUN; then
            info "[DRY-RUN] ssh -f -N -p $SSH_PORT -R $TUNNEL_LOCAL_PORT:localhost:$POOL_PORT $REMOTE_HOST"
        else
            # Unique tunnel identifier: includes SSH port to differentiate
            # multiple tunnels to the same provider host (e.g. two machines on clorecloud.net)
            TUNNEL_ID="ssh.*-p $SSH_PORT.*-R.*${TUNNEL_LOCAL_PORT}:localhost:${POOL_PORT}.*$REMOTE_HOST"

            # Kill ONLY the existing tunnel for THIS specific machine (port-aware)
            pkill -f "$TUNNEL_ID" 2>/dev/null || true
            pkill -f "autossh.*-p $SSH_PORT.*$REMOTE_HOST" 2>/dev/null || true
            sleep 1

            # Tunnel SSH options — optimized for low-latency mining
            TUNNEL_SSH_OPTS=(
                -p "$SSH_PORT"
                -o ExitOnForwardFailure=yes
                -o ServerAliveInterval=15
                -o ServerAliveCountMax=3
                -o TCPKeepAlive=yes
                -o Compression=no
                -o IPQoS=lowdelay
            )
            [[ -n "$SSH_CIPHER" ]] && TUNNEL_SSH_OPTS+=(-c "$SSH_CIPHER")

            # Prefer autossh for auto-reconnect, fallback to plain SSH
            if command -v autossh &>/dev/null; then
                info "Using autossh for persistent tunnel"
                AUTOSSH_GATETIME=0 autossh -f -M 0 \
                    -N "${TUNNEL_SSH_OPTS[@]}" \
                    -R "${TUNNEL_LOCAL_PORT}:localhost:${POOL_PORT}" \
                    "$REMOTE_HOST"
                success "Persistent tunnel via autossh (auto-reconnects)"
            else
                # Plain SSH tunnel
                ssh -f -N "${TUNNEL_SSH_OPTS[@]}" \
                    -R "${TUNNEL_LOCAL_PORT}:localhost:${POOL_PORT}" \
                    "$REMOTE_HOST"
                success "SSH tunnel established (rented:$TUNNEL_LOCAL_PORT -> localhost:$POOL_PORT)"

                # Set up cron watchdog for tunnel persistence (unique per SSH port)
                WATCHDOG="/tmp/tunnel-watchdog-$(echo "${REMOTE_HOST}-${SSH_PORT}" | tr -dc 'a-zA-Z0-9').sh"
                cat > "$WATCHDOG" <<WATCHEOF
#!/bin/bash
# Auto-generated tunnel watchdog for $REMOTE_HOST:$SSH_PORT ($WORKER_NAME)
if ! pgrep -f "ssh.*-p $SSH_PORT.*-R.*${TUNNEL_LOCAL_PORT}:localhost:${POOL_PORT}.*$REMOTE_HOST" >/dev/null 2>&1; then
    ssh -f -N ${TUNNEL_SSH_OPTS[*]} -R ${TUNNEL_LOCAL_PORT}:localhost:${POOL_PORT} $REMOTE_HOST 2>/dev/null
fi
WATCHEOF
                chmod +x "$WATCHDOG"
                # Add cron entry every 2 minutes (idempotent — removes only THIS watchdog)
                (crontab -l 2>/dev/null | grep -v "$WATCHDOG"; echo "*/2 * * * * $WATCHDOG") | crontab - 2>/dev/null && \
                    info "Tunnel watchdog cron installed (reconnects every 2 min)" || \
                    warn "Could not install cron watchdog (crontab not available)"
            fi

            info "Tunnel PID: $(pgrep -f "ssh.*-p $SSH_PORT.*-R.*$TUNNEL_LOCAL_PORT.*$REMOTE_HOST" | head -1 || echo 'autossh managed')"
        fi

    elif [[ "$TUNNEL_MODE" == "remote" ]]; then
        warn "Remote tunnel mode requires the rented machine to have SSH access to $POOL_VM"
        info "Setting up tunnel on rented machine..."

        run_remote "bash -s" <<TUNNEL_EOF
            # Kill existing tunnels
            pkill -f "ssh.*-L.*$TUNNEL_LOCAL_PORT.*localhost:$POOL_PORT" 2>/dev/null || true

            # Create tunnel: localhost:3333 on rented -> pool_vm:3333
            ssh -f -N \
                -o ExitOnForwardFailure=yes \
                -o ServerAliveInterval=30 \
                -o StrictHostKeyChecking=accept-new \
                -L ${TUNNEL_LOCAL_PORT}:localhost:${POOL_PORT} \
                ${POOL_VM}

            echo "[TUNNEL] Tunnel established: localhost:${TUNNEL_LOCAL_PORT} -> ${POOL_VM}:${POOL_PORT}"
TUNNEL_EOF
        success "Remote tunnel established"
    fi

    # Verify tunnel is working
    info "Verifying tunnel connectivity..."
    sleep 2
    TUNNEL_CHECK=$(run_remote "curl -s --max-time 5 http://localhost:$TUNNEL_LOCAL_PORT/ 2>/dev/null | head -c 50 || echo 'FAIL'")
    if [[ "$TUNNEL_CHECK" == *"FAIL"* ]] || [[ -z "$TUNNEL_CHECK" ]]; then
        warn "Could not verify tunnel via HTTP (pool uses WebSocket — this is normal)."
    else
        success "Tunnel verified"
    fi
    mark_step 6
fi

# ---- Step 7: Start the miner ----
if ! should_skip 7; then
    header "Step 7/7: Starting the miner"

    POOL_URL="ws://localhost:${TUNNEL_LOCAL_PORT}"

    info "Pool URL:     $POOL_URL"
    info "Wallet:       ${WALLET:0:30}..."
    info "Worker:       $WORKER_NAME"
    info "Extra args:   $MINER_EXTRA_ARGS"

    run_remote_script <<START_EOF
    cd ~/sha1-miner-gpu

    # Find the binary
    MINER_BIN=""
    if [ -f build/sha1_miner ]; then
        MINER_BIN="./build/sha1_miner"
    elif [ -f build-release/sha1_miner ]; then
        MINER_BIN="./build-release/sha1_miner"
    else
        echo "[ERROR] Miner binary not found!"
        exit 1
    fi

    # Kill any existing miner process
    pkill -f sha1_miner 2>/dev/null || true
    sleep 1

    # Start miner in a screen/tmux session (or nohup as fallback)
    MINER_CMD="\$MINER_BIN --pool ${POOL_URL} --wallet ${WALLET} --worker ${WORKER_NAME} --all-gpus ${MINER_EXTRA_ARGS}"
    echo "[MINER] Starting: \$MINER_CMD"

    if command -v screen &>/dev/null; then
        screen -dmS opnet-miner bash -c "\$MINER_CMD 2>&1 | tee ~/miner.log"
        echo "[MINER] Started in screen session 'opnet-miner'"
        echo "[MINER] Attach with: screen -r opnet-miner"
    elif command -v tmux &>/dev/null; then
        tmux kill-session -t opnet-miner 2>/dev/null || true
        tmux new-session -d -s opnet-miner "\$MINER_CMD 2>&1 | tee ~/miner.log"
        echo "[MINER] Started in tmux session 'opnet-miner'"
        echo "[MINER] Attach with: tmux attach -t opnet-miner"
    else
        nohup \$MINER_CMD > ~/miner.log 2>&1 &
        echo "[MINER] Started with nohup (PID: \$!)"
        echo "[MINER] Logs: tail -f ~/miner.log"
    fi

    # Wait and verify miner is running
    sleep 3
    if pgrep -f sha1_miner &>/dev/null; then
        echo "[MINER] Miner is running!"
    else
        echo "[MINER] WARNING: Miner process not found. Check ~/miner.log for errors."
        tail -20 ~/miner.log 2>/dev/null || true
    fi
START_EOF
    mark_step 7
fi

# ---- Health check ----
header "Post-Deployment Health Check"
info "Waiting for miner to initialize (up to 60s)..."

HEALTH_OK=false
HEALTH_FATAL=false
for i in $(seq 1 12); do
    sleep 5
    MINER_LOG=$(run_remote "tail -20 ~/miner.log 2>/dev/null" || echo "")

    # Check for fatal GPU/CUDA errors first — no point waiting 60s
    if echo "$MINER_LOG" | grep -qiE "CUDA driver version is insufficient|Failed to get GPU|no CUDA-capable device|CUDA error|Failed to initialize mining|out of memory"; then
        err_msg=$(echo "$MINER_LOG" | grep -iE "CUDA driver|Failed to get GPU|no CUDA-capable|CUDA error|Failed to initialize|out of memory" | head -1)
        echo
        echo -e "${RED}[FATAL]${NC} GPU/CUDA error detected:"
        echo -e "${RED}        $err_msg${NC}"
        echo
        warn "Full log:"
        echo "$MINER_LOG" | tail -10
        HEALTH_FATAL=true
        break
    fi

    # Check for success
    if echo "$MINER_LOG" | grep -qiE "share|accepted|hashrate|MH/s|GH/s|connected.*pool"; then
        success "Miner is producing output!"
        echo "$MINER_LOG" | tail -3
        HEALTH_OK=true
        break
    fi
    info "  Waiting... ($((i*5))s)"
done

if $HEALTH_FATAL; then
    err "Miner failed to start due to GPU/CUDA error. Fix the issue and re-deploy with: $0 $REMOTE_HOST --ssh-port $SSH_PORT --from-step 4"
elif ! $HEALTH_OK; then
    warn "Miner has not produced share output after 60s."
    warn "Check logs: ssh -p $SSH_PORT $REMOTE_HOST 'tail -50 ~/miner.log'"
fi

# ---- Save config ----
if [[ -n "$SAVE_CONFIG" ]]; then
    mkdir -p "$(dirname "$SAVE_CONFIG")"
    cat > "$SAVE_CONFIG" <<CFGEOF
# Deploy config generated on $(date -Iseconds)
# Provider: $PROVIDER
REMOTE_HOST="$REMOTE_HOST"
SSH_PORT="$SSH_PORT"
SSH_CIPHER="$SSH_CIPHER"
POOL_VM="$POOL_VM"
POOL_PORT="$POOL_PORT"
WALLET="$WALLET"
WORKER_NAME="$WORKER_NAME"
TUNNEL_MODE="$TUNNEL_MODE"
FORCE_GPU="$FORCE_GPU"
MINER_EXTRA_ARGS="$MINER_EXTRA_ARGS"
CFGEOF
    success "Config saved to $SAVE_CONFIG"
fi

# ---- Done ----
header "Deployment Complete"
success "Miner deployed to $REMOTE_HOST ($PROVIDER)"
echo
info "Useful commands:"
info "  Check miner:    ssh -p $SSH_PORT $REMOTE_HOST 'tail -f ~/miner.log'"
info "  Stop miner:     ssh -p $SSH_PORT $REMOTE_HOST 'pkill -f sha1_miner'"
info "  Restart miner:  ssh -p $SSH_PORT $REMOTE_HOST 'screen -r opnet-miner'"
info "  Kill tunnel:    pkill -f 'ssh.*-p $SSH_PORT.*-R.*$TUNNEL_LOCAL_PORT.*$REMOTE_HOST'"
info "  Re-deploy:      $0 --config ${SAVE_CONFIG:-deploy/configs/<name>.conf}"
info "  Resume failed:  $0 $REMOTE_HOST --ssh-port $SSH_PORT --resume"
info "  Pool dashboard: http://$(echo "$POOL_VM" | cut -d@ -f2):8080"
