#!/bin/bash
#
# Deploy SHA-1 GPU miner to a rented machine and connect it to your solo pool
# via SSH tunnel (no public exposure needed).
#
# Usage:
#   ./deploy-miner.sh <user@host> [options]
#
# Examples:
#   ./deploy-miner.sh root@203.0.113.50
#   ./deploy-miner.sh root@203.0.113.50 --pool-vm yoztheripper@192.168.1.12
#   ./deploy-miner.sh root@203.0.113.50 --wallet opt1pe62t... --worker gpu-rented-1
#   ./deploy-miner.sh root@203.0.113.50 --force-amd
#
# Prerequisites:
#   - SSH access to the rented GPU machine (key-based auth recommended)
#   - The rented machine can SSH to your pool VM (for the tunnel)
#   - Or: your pool VM can be reached from here (we set up the tunnel from local)
#

set -euo pipefail

# ==================== CONFIGURATION ====================
# Override these via CLI flags or environment variables

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
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
header()  { echo -e "\n${CYAN}=== $* ===${NC}\n"; }

usage() {
    cat <<'EOF'
Usage: deploy-miner.sh <user@host> [options]

Required:
  <user@host>             SSH target for the rented GPU machine

Options:
  --pool-vm <user@host>   Pool VM SSH target (default: yoztheripper@192.168.1.12)
  --pool-port <port>      Pool WebSocket port (default: 3333)
  --wallet <address>      Wallet address for mining
  --worker <name>         Worker name (default: rented-gpu)
  --force-cuda            Force NVIDIA/CUDA build
  --force-amd             Force AMD/HIP build
  --miner-args <args>     Extra miner arguments (default: --auto-bench)
  --tunnel-mode <mode>    Tunnel mode: "local" or "remote" (default: local)
                          local  = tunnel goes from your machine through to rented
                          remote = rented machine SSHes to pool VM directly
  --ssh-port <port>       SSH port on rented machine (default: 22)
  --dry-run               Print commands without executing
  --help                  Show this help
EOF
    exit 0
}

# ==================== PARSE ARGS ====================

REMOTE_HOST=""
FORCE_GPU=""
SSH_PORT="22"
TUNNEL_MODE="local"
DRY_RUN=false

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
        --tunnel-mode)  TUNNEL_MODE="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help|-h)      usage ;;
        -*)             error "Unknown option: $1" ;;
        *)
            if [[ -z "$REMOTE_HOST" ]]; then
                REMOTE_HOST="$1"
            else
                error "Unexpected argument: $1"
            fi
            shift
            ;;
    esac
done

[[ -z "$REMOTE_HOST" ]] && error "Missing required argument: <user@host>\n$(usage)"

# ==================== HELPER ====================

# SSH options (port-aware)
SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -p "$SSH_PORT")

# Run command on the remote host
run_remote() {
    if $DRY_RUN; then
        echo "[DRY-RUN] ssh -p $SSH_PORT $REMOTE_HOST: $*"
    else
        ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
    fi
}

# ==================== MAIN ====================

header "OP_NET GPU Miner Deployment"
info "Target:     $REMOTE_HOST (port $SSH_PORT)"
info "Pool VM:    $POOL_VM"
info "Pool port:  $POOL_PORT"
info "Wallet:     ${WALLET:0:30}..."
info "Worker:     $WORKER_NAME"
info "Tunnel:     $TUNNEL_MODE"
echo

# ---- Step 1: Test SSH connectivity ----
header "Step 1/7: Testing SSH connectivity"
if $DRY_RUN; then
    info "[DRY-RUN] Would test SSH to $REMOTE_HOST"
else
    if ! ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "echo ok" &>/dev/null; then
        error "Cannot SSH to $REMOTE_HOST. Check your credentials."
    fi
    success "SSH connection OK"
fi

# ---- Step 2: Detect GPU type ----
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
        *)    error "No supported GPU detected on $REMOTE_HOST. Use --force-cuda or --force-amd." ;;
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

# ---- Step 3: Clone the miner repo ----
header "Step 3/7: Cloning miner repo"

run_remote "bash -s" <<CLONE_EOF
    set -e
    cd ~

    # Ensure git is available (minimal bootstrap)
    if ! command -v git &>/dev/null; then
        echo "[CLONE] Installing git..."
        sudo apt-get update -qq && sudo apt-get install -y -qq git 2>/dev/null || \
        sudo dnf install -y git 2>/dev/null || \
        sudo pacman -Sy --noconfirm git 2>/dev/null || true
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

# ---- Step 4: Install dependencies using repo's install.sh ----
header "Step 4/7: Installing build dependencies"

info "Using the miner's install.sh (handles all distros and packages)"
run_remote 'bash -s' <<'DEPS_EOF'
    set -e
    cd ~/sha1-miner-gpu

    # Run the repo's own install script (supports Ubuntu, Fedora, Arch, openSUSE, Alpine)
    echo "[DEPS] Running install.sh from the miner repo..."
    chmod +x install.sh
    ./install.sh

    echo ""
    echo "[DEPS] Checking GPU toolkit..."
DEPS_EOF

# Install GPU-specific toolkit if missing
if [[ "$GPU_TYPE" == "cuda" ]]; then
    run_remote 'bash -s' <<'CUDA_EOF'
        set -e
        export DEBIAN_FRONTEND=noninteractive

        if command -v nvcc &>/dev/null; then
            CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
            echo "[DEPS] CUDA toolkit already installed: $CUDA_VER"
        else
            echo "[DEPS] CUDA toolkit (nvcc) not found. Installing CUDA 12.9..."

            # Use NVIDIA's official repo for CUDA 12.9 (as recommended in README)
            if [ -f /etc/debian_version ]; then
                # Detect Ubuntu version for correct package URL
                UBUNTU_VER=$(lsb_release -rs 2>/dev/null | tr -d '.' || echo "2204")
                CUDA_DEB="cuda-keyring_1.1-1_all.deb"
                CUDA_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/x86_64/${CUDA_DEB}"

                echo "[DEPS] Adding NVIDIA CUDA repo for Ubuntu ${UBUNTU_VER}..."
                wget -q "$CUDA_URL" -O "/tmp/${CUDA_DEB}" 2>/dev/null && \
                    sudo dpkg -i "/tmp/${CUDA_DEB}" && \
                    sudo apt-get update -qq && \
                    sudo apt-get install -y cuda-toolkit-12-9 && \
                    echo "[DEPS] CUDA 12.9 installed successfully" || {
                        echo "[DEPS] NVIDIA repo install failed, trying apt fallback..."
                        sudo apt-get install -y nvidia-cuda-toolkit 2>/dev/null || \
                            echo "[DEPS] WARNING: Could not install CUDA. Install manually: https://developer.nvidia.com/cuda-downloads"
                    }
            else
                echo "[DEPS] Non-Debian system. Install CUDA manually: https://developer.nvidia.com/cuda-downloads"
            fi
        fi
CUDA_EOF
elif [[ "$GPU_TYPE" == "amd" ]]; then
    run_remote 'bash -s' <<'AMD_EOF'
        set -e
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

# ---- Step 5 (was 4): Build the miner ----
header "Step 5/7: Building the miner"

BUILD_FLAG=""
if [[ "$GPU_TYPE" == "cuda" ]]; then
    BUILD_FLAG="--cuda"
elif [[ "$GPU_TYPE" == "amd" ]]; then
    BUILD_FLAG="--hip"
fi

run_remote "bash -s" <<BUILD_EOF
    set -e
    cd ~/sha1-miner-gpu

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

# ---- Step 5: Set up SSH tunnel ----
header "Step 6/7: Setting up SSH tunnel to pool"

if [[ "$TUNNEL_MODE" == "local" ]]; then
    # Local mode: we create a tunnel from our machine
    # Local port on rented machine -> pool VM port
    # This requires the rented machine to be able to reach the pool VM,
    # OR we do a two-hop tunnel: local -> rented -> pool VM
    #
    # Simplest: we forward from our local machine through to pool VM,
    # then the rented machine connects via another tunnel.
    #
    # Actually, the cleanest approach for "local" tunnel:
    # SSH from HERE to the RENTED machine, forwarding rented:3333 -> pool_vm:3333
    # This requires our machine can reach pool_vm.

    info "Setting up tunnel: $REMOTE_HOST:$TUNNEL_LOCAL_PORT -> $POOL_VM:$POOL_PORT"
    info "Starting SSH tunnel in background..."

    if $DRY_RUN; then
        info "[DRY-RUN] ssh -f -N -R $TUNNEL_LOCAL_PORT:localhost:$POOL_PORT $REMOTE_HOST"
        info "[DRY-RUN] (requires $POOL_VM:$POOL_PORT to be reachable from here via another hop)"
    else
        # Kill any existing tunnel to this host
        pkill -f "ssh.*-R.*$TUNNEL_LOCAL_PORT.*$REMOTE_HOST" 2>/dev/null || true

        # We need the rented machine to reach ws://localhost:3333.
        # So we create a REVERSE tunnel: rented:3333 -> here -> pool_vm:3333
        # First, ensure we can reach the pool VM
        POOL_VM_HOST=$(echo "$POOL_VM" | cut -d@ -f2)

        # Option A: Direct - if rented machine can reach pool VM
        # Option B: Two-hop via our machine

        # Try: create reverse forward on rented machine
        # -R 3333:pool_vm_host:3333 means rented:3333 -> pool_vm:3333 (routed through our machine)
        ssh -f -N \
            -p "$SSH_PORT" \
            -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 \
            -R "${TUNNEL_LOCAL_PORT}:${POOL_VM_HOST}:${POOL_PORT}" \
            "$REMOTE_HOST"

        success "SSH tunnel established (rented:$TUNNEL_LOCAL_PORT -> $POOL_VM_HOST:$POOL_PORT)"
        info "Tunnel PID: $(pgrep -f "ssh.*-R.*$TUNNEL_LOCAL_PORT.*$REMOTE_HOST" | head -1)"
    fi

elif [[ "$TUNNEL_MODE" == "remote" ]]; then
    # Remote mode: the rented machine SSHes directly to the pool VM
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
TUNNEL_CHECK=$(run_remote "curl -s --max-time 5 http://localhost:$TUNNEL_LOCAL_PORT/ 2>/dev/null | head -c 50 || echo 'FAIL'")
if [[ "$TUNNEL_CHECK" == *"FAIL"* ]] || [[ -z "$TUNNEL_CHECK" ]]; then
    warn "Could not verify tunnel (pool may use WebSocket only, not HTTP). Proceeding anyway."
else
    success "Tunnel appears to be working"
fi

# ---- Step 6: Start the miner ----
header "Step 7/7: Starting the miner"

POOL_URL="ws://localhost:${TUNNEL_LOCAL_PORT}"

info "Pool URL:     $POOL_URL"
info "Wallet:       ${WALLET:0:30}..."
info "Worker:       $WORKER_NAME"
info "Extra args:   $MINER_EXTRA_ARGS"

run_remote "bash -s" <<START_EOF
    set -e
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
        tmux new-session -d -s opnet-miner "\$MINER_CMD 2>&1 | tee ~/miner.log"
        echo "[MINER] Started in tmux session 'opnet-miner'"
        echo "[MINER] Attach with: tmux attach -t opnet-miner"
    else
        nohup \$MINER_CMD > ~/miner.log 2>&1 &
        echo "[MINER] Started with nohup (PID: \$!)"
        echo "[MINER] Logs: tail -f ~/miner.log"
    fi

    # Wait a moment and check if it's running
    sleep 3
    if pgrep -f sha1_miner &>/dev/null; then
        echo "[MINER] Miner is running!"
        echo "[MINER] Check logs: ssh ${REMOTE_HOST:-\$HOSTNAME} 'tail -f ~/miner.log'"
    else
        echo "[MINER] WARNING: Miner process not found. Check ~/miner.log for errors."
        tail -20 ~/miner.log 2>/dev/null || true
    fi
START_EOF

header "Deployment Complete"
success "Miner deployed to $REMOTE_HOST"
echo
info "Useful commands:"
info "  Check miner:    ssh -p $SSH_PORT $REMOTE_HOST 'tail -f ~/miner.log'"
info "  Stop miner:     ssh -p $SSH_PORT $REMOTE_HOST 'pkill -f sha1_miner'"
info "  Restart miner:  ssh -p $SSH_PORT $REMOTE_HOST 'screen -r opnet-miner'"
info "  Kill tunnel:    pkill -f 'ssh.*-R.*$TUNNEL_LOCAL_PORT.*$REMOTE_HOST'"
info "  Pool dashboard: http://$(echo "$POOL_VM" | cut -d@ -f2):8080"
