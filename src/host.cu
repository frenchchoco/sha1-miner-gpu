#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>
#include <fstream>
#include <sstream>
#include <ctime>
#include <mutex>

#include "job_upload_api.h"
#include "cxxsha1.hpp"
#include "job_constants.cuh"

// Kernel declaration
extern "C" __global__ void sha1_double_kernel_optimized(uint8_t *, uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(1);} \
    }while(0)

// ==================== Configuration ====================
constexpr uint64_t BATCH_SIZE = 1ull << 30; // 1B hashes per batch
constexpr uint32_t RING_SIZE = 1u << 20; // 1M candidate slots
constexpr int DEFAULT_THREADS = 256; // Threads per block
constexpr int PROGRESS_INTERVAL = 10; // Update every 10 seconds

// Global state for signal handling
std::atomic<bool> g_shutdown(false);
std::atomic<uint64_t> g_total_hashes(0);
std::atomic<uint32_t> g_total_candidates(0);

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    std::cout << "\n\nReceived signal " << sig << ", shutting down gracefully...\n";
    g_shutdown.store(true);
}

// Set up signal handlers (cross-platform)
void setup_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef _WIN32
    std::signal(SIGBREAK, signal_handler);
#endif
}

// ==================== Job Configuration ====================
struct JobConfig {
    std::array<uint8_t, 32> preimage;
    std::array<uint8_t, 20> target_hash;
    uint64_t start_nonce = 0;
    uint64_t end_nonce = UINT64_MAX;
    int gpu_id = 0;
    int threads_per_block = DEFAULT_THREADS;
    std::string output_file = "found_collisions.txt";
};

// ==================== GPU Manager ====================
class GPUManager {
private:
    int device_id;
    cudaDeviceProp prop;
    uint64_t *d_pairs;
    uint32_t *d_ticket;
    int optimal_blocks;
    int optimal_threads;

public:
    GPUManager(int gpu_id, int threads) : device_id(gpu_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

        // Calculate optimal configuration
        optimal_threads = threads;
        if (prop.major >= 8) {
            optimal_blocks = prop.multiProcessorCount * 4;
        } else if (prop.major >= 7) {
            optimal_blocks = prop.multiProcessorCount * 2;
        } else {
            optimal_blocks = prop.multiProcessorCount * 2;
        }

        // Allocate memory
        CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(uint64_t) * 4 * RING_SIZE));
        CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

        // Set cache config
        CUDA_CHECK(cudaFuncSetCacheConfig(sha1_double_kernel_optimized, cudaFuncCachePreferL1));

        printInfo();
    }

    ~GPUManager() {
        CUDA_CHECK(cudaFree(d_pairs));
        CUDA_CHECK(cudaFree(d_ticket));
    }

    void printInfo() {
        std::cout << "=== GPU " << device_id << " Configuration ===\n";
        std::cout << "Device        : " << prop.name << "\n";
        std::cout << "Compute Cap   : " << prop.major << "." << prop.minor << "\n";
        std::cout << "SMs           : " << prop.multiProcessorCount << "\n";
        std::cout << "Memory        : " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n";
        std::cout << "Configuration : " << optimal_blocks << " blocks × "
                << optimal_threads << " threads\n\n";
    }

    dim3 getGridDim(uint64_t work_size) {
        return dim3((work_size + optimal_threads - 1) / optimal_threads);
    }

    dim3 getBlockDim() {
        return dim3(optimal_threads);
    }

    uint64_t *getPairsPtr() { return d_pairs; }
    uint32_t *getTicketPtr() { return d_ticket; }
    int getOptimalBlocks() { return optimal_blocks; }
};

// ==================== Result Handler ====================
class ResultHandler {
private:
    std::ofstream output_file;
    std::mutex file_mutex;

public:
    ResultHandler(const std::string &filename) {
        output_file.open(filename, std::ios::app);
        if (!output_file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << "\n";
        }

        // Write header
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        output_file << "\n=== SHA-1 Bitcoin Mining Started at "
                << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                << " ===\n\n";
        output_file.flush();
    }

    void saveResults(uint64_t *h_pairs, uint32_t count, const uint8_t *target) {
        std::lock_guard<std::mutex> lock(file_mutex);

        output_file << "Found " << count << " candidates:\n";
        output_file << "Target: ";
        for (int i = 0; i < 20; i++) {
            output_file << std::hex << std::setw(2) << std::setfill('0')
                    << (int) target[i];
        }
        output_file << "\n\n";

        for (uint32_t i = 0; i < std::min(count, 100u); i++) {
            output_file << "Candidate " << i + 1 << ":\n";
            output_file << "  Message: ";

            // Reconstruct 32-byte message
            for (int j = 0; j < 4; j++) {
                uint64_t word = h_pairs[i * 4 + j];
                output_file << std::hex << std::setw(16) << std::setfill('0') << word;
            }
            output_file << "\n";

            // Verify on CPU
            uint8_t msg[32];
            for (int j = 0; j < 4; j++) {
                uint64_t word = h_pairs[i * 4 + j];
                msg[j * 8 + 0] = (word >> 0) & 0xFF;
                msg[j * 8 + 1] = (word >> 8) & 0xFF;
                msg[j * 8 + 2] = (word >> 16) & 0xFF;
                msg[j * 8 + 3] = (word >> 24) & 0xFF;
                msg[j * 8 + 4] = (word >> 32) & 0xFF;
                msg[j * 8 + 5] = (word >> 40) & 0xFF;
                msg[j * 8 + 6] = (word >> 48) & 0xFF;
                msg[j * 8 + 7] = (word >> 56) & 0xFF;
            }

            // Compute double SHA-1
            uint8_t d1[20], d2[20];
            sha1_ctx ctx;
            sha1_init(ctx);
            sha1_update(ctx, msg, 32);
            sha1_final(ctx, d1);
            sha1_init(ctx);
            sha1_update(ctx, d1, 20);
            sha1_final(ctx, d2);

            output_file << "  SHA-1^2: ";
            for (int j = 0; j < 20; j++) {
                output_file << std::hex << std::setw(2) << std::setfill('0')
                        << (int) d2[j];
            }
            output_file << "\n\n";
        }

        output_file.flush();

        // Also print to console
        std::cout << "\n[!] Found " << count << " collision candidates! ";
        std::cout << "Saved to output file.\n";
    }
};

// ==================== Progress Monitor ====================
class ProgressMonitor {
private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update;
    uint64_t last_hashes = 0;

public:
    ProgressMonitor() {
        start_time = std::chrono::steady_clock::now();
        last_update = start_time;
    }

    void update(uint64_t total_hashes, uint32_t candidates) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);

        if (elapsed.count() >= PROGRESS_INTERVAL) {
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            double total_seconds = total_elapsed.count();
            double interval_seconds = elapsed.count();

            uint64_t interval_hashes = total_hashes - last_hashes;
            double interval_rate = interval_hashes / interval_seconds / 1e9;
            double average_rate = total_hashes / total_seconds / 1e9;

            std::cout << "\r[" << formatTime(total_seconds) << "] "
                    << "Hashes: " << std::fixed << std::setprecision(2)
                    << total_hashes / 1e9 << "B | "
                    << "Rate: " << interval_rate << " GH/s (avg: "
                    << average_rate << " GH/s) | "
                    << "Candidates: " << candidates << "    " << std::flush;

            last_update = now;
            last_hashes = total_hashes;
        }
    }

    void finalReport(uint64_t total_hashes, uint32_t candidates) {
        auto now = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        double total_seconds = total_elapsed.count();
        double average_rate = total_hashes / total_seconds / 1e9;

        std::cout << "\n\n=== Final Statistics ===\n";
        std::cout << "Total Time    : " << formatTime(total_seconds) << "\n";
        std::cout << "Total Hashes  : " << std::fixed << std::setprecision(3)
                << total_hashes / 1e9 << " billion\n";
        std::cout << "Average Rate  : " << average_rate << " GH/s\n";
        std::cout << "Candidates    : " << candidates << "\n";
        std::cout << "Efficiency    : " << std::scientific << std::setprecision(2)
                << (double) candidates / total_hashes * 100 << "%\n";
    }

private:
    std::string formatTime(double seconds) {
        int hours = seconds / 3600;
        int minutes = (seconds - hours * 3600) / 60;
        int secs = seconds - hours * 3600 - minutes * 60;

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(2) << hours << ":"
                << std::setw(2) << minutes << ":"
                << std::setw(2) << secs;
        return ss.str();
    }
};

// ==================== Main Collision Finder ====================
void runCollisionFinder(const JobConfig &config) {
    // Initialize GPU
    GPUManager gpu(config.gpu_id, config.threads_per_block);

    // Initialize result handler
    ResultHandler results(config.output_file);

    // Initialize progress monitor
    ProgressMonitor progress;

    // Prepare target
    uint32_t target[5];
    for (int i = 0; i < 5; ++i) {
        target[i] = (uint32_t(config.target_hash[4 * i]) << 24) |
                    (uint32_t(config.target_hash[4 * i + 1]) << 16) |
                    (uint32_t(config.target_hash[4 * i + 2]) << 8) |
                    config.target_hash[4 * i + 3];
    }

    // Upload job to GPU
    upload_new_job(config.preimage.data(), target);

    std::cout << "Starting collision search...\n";
    std::cout << "Target: ";
    for (uint8_t b: config.target_hash) {
        std::printf("%02x", b);
    }
    std::cout << "\n\n";

    // Main search loop
    uint64_t current_nonce = config.start_nonce;
    uint64_t batch_id = 0;

    while (!g_shutdown.load() && current_nonce < config.end_nonce) {
        // Calculate work size for this batch
        uint64_t remaining = config.end_nonce - current_nonce;
        uint64_t work_size = std::min(BATCH_SIZE, remaining);

        // Launch kernel
        dim3 grid = gpu.getGridDim(work_size);
        dim3 block = gpu.getBlockDim();

        sha1_double_kernel_optimized<<<grid, block>>>(
            nullptr, gpu.getPairsPtr(), gpu.getTicketPtr(), current_nonce
        );
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update counters
        g_total_hashes.fetch_add(work_size);
        current_nonce += work_size;
        batch_id++;

        // Check for results
        uint32_t found = 0;
        CUDA_CHECK(cudaMemcpy(&found, gpu.getTicketPtr(), sizeof(uint32_t),
            cudaMemcpyDeviceToHost));

        if (found > 0) {
            g_total_candidates.fetch_add(found);

            // Copy results
            uint32_t to_copy = std::min(found, RING_SIZE);
            std::vector<uint64_t> h_pairs(to_copy * 4);
            CUDA_CHECK(cudaMemcpy(h_pairs.data(), gpu.getPairsPtr(),
                sizeof(uint64_t) * 4 * to_copy,
                cudaMemcpyDeviceToHost));

            // Save results
            results.saveResults(h_pairs.data(), to_copy, config.target_hash.data());

            // Reset ticket
            CUDA_CHECK(cudaMemset(gpu.getTicketPtr(), 0, sizeof(uint32_t)));
        }

        // Update progress
        progress.update(g_total_hashes.load(), g_total_candidates.load());

        // Check if we should exit after finding results
        if (found > 0 && config.end_nonce == UINT64_MAX) {
            std::cout << "\nFound collision! Continuing search...\n";
            std::cout << "Press Ctrl+C to stop.\n";
        }
    }

    // Final report
    progress.finalReport(g_total_hashes.load(), g_total_candidates.load());
}

// ==================== Configuration Parser ====================
JobConfig parseConfig(int argc, char **argv) {
    JobConfig config;

    // Default preimage (can be overridden by command line)
    config.preimage = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
    };

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--gpu" && i + 1 < argc) {
            config.gpu_id = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads_per_block = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--start-nonce" && i + 1 < argc) {
            config.start_nonce = std::stoull(argv[++i]);
        } else if (arg == "--end-nonce" && i + 1 < argc) {
            config.end_nonce = std::stoull(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "SHA-1 Bitcoin Mining\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --gpu <id>           GPU device ID (default: 0)\n";
            std::cout << "  --threads <n>        Threads per block (default: 256)\n";
            std::cout << "  --output <file>      Output file (default: found_collisions.txt)\n";
            std::cout << "  --start-nonce <n>    Starting nonce (default: 0)\n";
            std::cout << "  --end-nonce <n>      Ending nonce (default: MAX)\n";
            std::cout << "  --help               Show this help\n";
            std::exit(0);
        }
    }

    // Compute target hash
    uint8_t d1[20], d2[20];
    sha1_ctx ctx;
    sha1_init(ctx);
    sha1_update(ctx, config.preimage.data(), 32);
    sha1_final(ctx, d1);
    sha1_init(ctx);
    sha1_update(ctx, d1, 20);
    sha1_final(ctx, d2);

    std::copy(d2, d2 + 20, config.target_hash.begin());

    return config;
}

// ==================== Main ====================
int main(int argc, char **argv) {
    // Set up signal handlers
    setup_signal_handlers();

    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|        SHA-1 Bitcoin Mining v1.0         |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Parse configuration
    JobConfig config = parseConfig(argc, argv);

    // Run collision finder
    try {
        runCollisionFinder(config);
    } catch (const std::exception &e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }

    if (g_total_candidates.load() > 0) {
        std::cout << "\n[SUCCESS] Found " << g_total_candidates.load()
                << " collisions.\n";
        std::cout << "Results saved to: " << config.output_file << "\n";
    } else {
        std::cout << "\n[FAILED] No collisions found in search space.\n";
    }

    return 0;
}
