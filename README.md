# SHA-1 Miner

This is a GPU miner for SHA-1 proof-of-work that works with both NVIDIA and AMD graphics cards. You can run it on
Windows or Linux, and it connects to mining pools through WebSocket connections.

## What You'll Need

You need a GPU to run this miner effectively. For NVIDIA cards, anything from the GTX 900 series
onwards will work. For AMD cards, you'll want something with RDNA architecture (like the RX 9000 series) or
recent GCN cards that support ROCm. Older cards should work too, but performance may vary.

On the software side, you'll need a C++ compiler, CMake for building, and either CUDA for NVIDIA cards or ROCm for AMD
cards. The specific versions matter - we're using CUDA 12.9 and ROCm 6.2 or later.

## Getting Started on Windows

### For NVIDIA Users

First, grab CUDA 12.9 from NVIDIA's website. Just run the installer with the default options - it'll set everything up
for you.

Next, you need Visual Studio Build Tools 2022. When you install it, make sure to select the "Desktop development with
C++" workload. This gives you the compiler and Windows SDK you need.

You'll also want Ninja for faster builds. Download the exe file and either add it to your PATH or just drop it in the
project folder. Same goes for CMake - get version 3.16 or newer and let it add itself to PATH during installation.

Once you have all that set up, clone this repository and run:

```
install.bat
build.bat
```

The install script will set up vcpkg and grab all the C++ libraries we need (Boost, OpenSSL, etc.). The build script
gives you a menu to configure and build the project.

### For AMD Users

The process is similar, but instead of CUDA, you need to install your AMD drivers and then grab ROCm for Windows. Make
sure you get version 6.2 or newer - the Windows port of ROCm is relatively new and earlier versions had issues.

After that, follow the same steps as NVIDIA users. When you run `build.bat`, it'll ask which GPU backend you want -
select AMD.

## Getting Started on Linux

### For NVIDIA Users

Installing CUDA on Linux involves adding NVIDIA's package repository first:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-9
```

You'll also need a bunch of development packages. Here's a command that installs everything you might need:

```bash
sudo apt-get install -y build-essential gcc g++ make python3.6 git manpages-dev \
  libcairo2-dev libatk1.0-0 libatk-bridge2.0-0 libc6 libcairo2 libcups2 \
  libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgdk-pixbuf2.0-0 libglib2.0-0 \
  libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 \
  libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 \
  libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation \
  libnss3 lsb-release xdg-utils libtool autoconf software-properties-common \
  gcc-12 g++-12 gcc-13 g++-13 cmake
```

Then just run:

```bash
./install.sh
./build.sh
```

### For AMD Users

For AMD on Linux, you need to install ROCm first. The installation process varies by distribution, so check AMD's
official ROCm documentation for your specific distro. Once ROCm is installed, the rest of the process is the same as for
NVIDIA users.

### Quick Fix for CUDA Issues

Some Linux distributions have a quirk with CUDA 12.9 where certain math functions cause compilation errors. If you run
into this, you'll need to edit `/usr/local/cuda-12.9/targets/x86_64-linux/include/crt/math_functions.h` and add
`noexcept(true)` to these four functions:

- `sinpi(double x)`
- `sinpif(float x)`
- `cospi(double x)`
- `cospif(float x)`

Just add it before the semicolon at the end of each function declaration. This tells the compiler these functions won't
throw exceptions, which fixes the compilation issue.

## Running the Miner

The basic command to start mining looks like this:

**On Linux:**

```bash
./build/sha1_miner --pool wss://sha1.opnet.org/pool --wallet YOUR_WALLET_P2TR --worker my-worker-1 --debug-level 2
```

**On Windows:**

```cmd
sha1_miner.exe --pool wss://sha1.opnet.org/pool --wallet YOUR_WALLET_P2TR --worker my-worker-1 --debug-level 2
```

Replace `YOUR_WALLET_P2TR` with your actual wallet address. The `--worker` parameter helps you identify different mining
rigs if you're running multiple miners.

## Understanding the Options

The miner has quite a few options, but most users only need a handful of them. Here are the important ones:

`--pool` is where you specify the mining pool's WebSocket URL. Use `wss://` for secure connections or `ws://` for
unencrypted ones.

`--wallet` is your wallet address where the pool will send your earnings.

`--worker` identifies your mining rig. If you don't specify one, it'll use your computer's hostname.

`--gpu` lets you select which GPU to use. The default is GPU 0, but if you have multiple cards, you can use `--all-gpus`
to mine with all of them, or `--gpus 0,2,3` to select specific ones.

`--debug-level` controls how much information you see. Level 2 (INFO) is usually good for normal operation. Use level 3
or 4 if you're troubleshooting issues.

## Some Practical Examples

If you just want to mine with one GPU on a pool:

```bash
sha1_miner --pool wss://sha1.opnet.org/pool --wallet YOUR_WALLET --worker rig1
```

If you have multiple GPUs and want to use them all:

```bash
sha1_miner --all-gpus --pool wss://sha1.opnet.org/pool --wallet YOUR_WALLET
```

Want to benchmark your GPU to see what it can do?

```bash
sha1_miner --benchmark --auto-tune
```

The auto-tune feature is pretty handy - it'll test different configurations to find what works best for your specific
GPU.

## Performance Notes

Modern GPUs can calculate SHA-1 hashes incredibly fast. The key to good performance is keeping the GPU fed with work and
minimizing the overhead of result checking. The `--streams` parameter controls how many concurrent CUDA/HIP streams the
miner uses - more streams can help keep the GPU busy, but too many can cause overhead. The default of 4 works well for
most cards.

The `--threads` parameter controls how many threads run in each GPU block. This is more of a fine-tuning parameter - 256
is a good default that works well across different GPU architectures.

## Pool Compatibility

This miner uses a WebSocket-based protocol that's similar to Stratum but adapted for SHA-1 mining. Make sure your pool
supports this protocol. The miner will automatically reconnect if the connection drops, and you can specify backup pools
with `--backup-pool` for failover support.

## Building Considerations

The project uses CMake with presets to make building easier. The presets handle all the compiler flags and optimization
settings for you. If you want to customize the build, you can always fall back to standard CMake commands, but the
presets should cover most use cases.

On Windows, the build system will automatically detect whether you have NVIDIA or AMD hardware and suggest the
appropriate configuration. On Linux, the build script will do the same thing.

## What's Next?

Once you get the miner running, keep an eye on the output to make sure it's finding shares and submitting them
successfully. The miner will show your hashrate, accepted/rejected shares, and other statistics. If you're not seeing
accepted shares after a few minutes, double-check your wallet address and pool URL.

Remember that mining profitability depends on many factors including your electricity costs, the current network
difficulty, and the value of the coins you're mining. Make sure to do your own calculations before committing
significant resources to mining.