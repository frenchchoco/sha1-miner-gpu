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
  --dry-run               Print commands without executing
  --help                  Show this help
EOF
    exit 0
}

# ==================== PARSE ARGS ====================

REMOTE_HOST=""
FORCE_GPU=""
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

# Run command on the remote host
run_remote() {
    if $DRY_RUN; then
        echo "[DRY-RUN] ssh $REMOTE_HOST: $*"
    else
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" "$@"
    fi
}

# ==================== MAIN ====================

header "OP_NET GPU Miner Deployment"
info "Target:     $REMOTE_HOST"
info "Pool VM:    $POOL_VM"
info "Pool port:  $POOL_PORT"
info "Wallet:     ${WALLET:0:30}..."
info "Worker:     $WORKER_NAME"
info "Tunnel:     $TUNNEL_MODE"
echo

# ---- Step 1: Test SSH connectivity ----
header "Step 1/6: Testing SSH connectivity"
if $DRY_RUN; then
    info "[DRY-RUN] Would test SSH to $REMOTE_HOST"
else
    if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" "echo ok" &>/dev/null; then
        error "Cannot SSH to $REMOTE_HOST. Check your credentials."
    fi
    success "SSH connection OK"
fi

# ---- Step 2: Detect GPU type ----
header "Step 2/6: Detecting GPU on remote machine"

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

# ---- Step 3: Install dependencies ----
header "Step 3/6: Installing build dependencies"

if [[ "$GPU_TYPE" == "cuda" ]]; then
    run_remote 'bash -s' <<'DEPS_CUDA_EOF'
        set -e
        export DEBIAN_FRONTEND=noninteractive

        echo "[DEPS] Checking build tools..."
        PKGS=""
        command -v cmake &>/dev/null || PKGS="$PKGS cmake"
        command -v g++   &>/dev/null || PKGS="$PKGS g++ build-essential"
        command -v git   &>/dev/null || PKGS="$PKGS git"
        dpkg -s libboost-all-dev &>/dev/null 2>&1 || PKGS="$PKGS libboost-all-dev"
        dpkg -s libssl-dev       &>/dev/null 2>&1 || PKGS="$PKGS libssl-dev"
        dpkg -s zlib1g-dev       &>/dev/null 2>&1 || PKGS="$PKGS zlib1g-dev"
        dpkg -s nlohmann-json3-dev &>/dev/null 2>&1 || PKGS="$PKGS nlohmann-json3-dev"

        if [[ -n "$PKGS" ]]; then
            echo "[DEPS] Installing: $PKGS"
            sudo apt-get update -qq
            sudo apt-get install -y -qq $PKGS
        else
            echo "[DEPS] All build dependencies already installed"
        fi

        # Check CUDA toolkit
        if ! command -v nvcc &>/dev/null; then
            echo "[DEPS] CUDA toolkit not found, installing..."
            if ! dpkg -s cuda-toolkit &>/dev/null 2>&1; then
                sudo apt-get install -y -qq nvidia-cuda-toolkit 2>/dev/null || \
                    echo "[DEPS] WARNING: Could not install cuda-toolkit via apt. Please install CUDA manually."
            fi
        fi
        echo "[DEPS] Done"
DEPS_CUDA_EOF
elif [[ "$GPU_TYPE" == "amd" ]]; then
    run_remote 'bash -s' <<'DEPS_AMD_EOF'
        set -e
        export DEBIAN_FRONTEND=noninteractive

        echo "[DEPS] Checking build tools..."
        PKGS=""
        command -v cmake &>/dev/null || PKGS="$PKGS cmake"
        command -v g++   &>/dev/null || PKGS="$PKGS g++ build-essential"
        command -v git   &>/dev/null || PKGS="$PKGS git"
        dpkg -s libboost-all-dev &>/dev/null 2>&1 || PKGS="$PKGS libboost-all-dev"
        dpkg -s libssl-dev       &>/dev/null 2>&1 || PKGS="$PKGS libssl-dev"
        dpkg -s zlib1g-dev       &>/dev/null 2>&1 || PKGS="$PKGS zlib1g-dev"
        dpkg -s nlohmann-json3-dev &>/dev/null 2>&1 || PKGS="$PKGS nlohmann-json3-dev"

        if [[ -n "$PKGS" ]]; then
            echo "[DEPS] Installing: $PKGS"
            sudo apt-get update -qq
            sudo apt-get install -y -qq $PKGS
        else
            echo "[DEPS] All build dependencies already installed"
        fi

        # Check ROCm
        if [ ! -d /opt/rocm ]; then
            echo "[DEPS] ROCm not found at /opt/rocm"
            echo "[DEPS] Please ensure ROCm is pre-installed on this machine."
        fi
        echo "[DEPS] Done"
DEPS_AMD_EOF
fi
success "Dependencies ready"

# ---- Step 4: Clone and build the miner ----
header "Step 4/6: Building the miner"

BUILD_FLAG=""
if [[ "$GPU_TYPE" == "cuda" ]]; then
    BUILD_FLAG="--cuda"
elif [[ "$GPU_TYPE" == "amd" ]]; then
    BUILD_FLAG="--hip"
fi

run_remote "bash -s" <<BUILD_EOF
    set -e
    cd ~

    # Clone or update the miner repo
    if [ -d sha1-miner-gpu ]; then
        echo "[BUILD] Updating existing repo..."
        cd sha1-miner-gpu
        git pull --ff-only || { echo "[BUILD] Pull failed, re-cloning..."; cd ~; rm -rf sha1-miner-gpu; git clone $MINER_REPO; cd sha1-miner-gpu; }
    else
        echo "[BUILD] Cloning miner repo..."
        git clone $MINER_REPO
        cd sha1-miner-gpu
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

# ---- Step 5: Set up SSH tunnel ----
header "Step 5/6: Setting up SSH tunnel to pool"

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
header "Step 6/6: Starting the miner"

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
info "  Check miner:    ssh $REMOTE_HOST 'tail -f ~/miner.log'"
info "  Stop miner:     ssh $REMOTE_HOST 'pkill -f sha1_miner'"
info "  Restart miner:  ssh $REMOTE_HOST 'screen -r opnet-miner'"
info "  Kill tunnel:    pkill -f 'ssh.*-R.*$TUNNEL_LOCAL_PORT.*$REMOTE_HOST'"
info "  Pool dashboard: http://$(echo "$POOL_VM" | cut -d@ -f2):8080"
