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
#include <memory>
#include <algorithm>
#include <random>
#include <cstring>

#include "job_upload_api.h"
#include "cxxsha1.hpp"

// Kernel declarations
extern "C" __global__ void sha1_collision_kernel(uint8_t *, uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_collision_kernel_multistream(uint8_t *, uint64_t *, uint32_t *, uint64_t, uint32_t,
                                                             uint32_t);

extern "C" __global__ void sha1_collision_kernel_extreme(uint8_t *, uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_collision_kernel_ultra(uint8_t *, uint64_t *, uint32_t *, uint64_t);

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(1);} \
    }while(0)

// ==================== Configuration ====================
constexpr uint32_t RING_SIZE = 1u << 20; // 1M candidate slots
constexpr int PROGRESS_INTERVAL = 5; // Update every 5 seconds
constexpr int STREAMS_PER_GPU = 4; // Concurrent streams

// Global state for signal handling
std::atomic<bool> g_shutdown(false);
std::atomic<uint64_t> g_total_hashes(0);
std::atomic<uint32_t> g_total_candidates(0);
std::chrono::steady_clock::time_point g_start_time;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    std::cout << "\n\nReceived signal " << sig << ", shutting down gracefully...\n";
    g_shutdown.store(true);
}

void setup_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef _WIN32
    std::signal(SIGBREAK, signal_handler);
#endif
}

// ==================== GPU Configuration ====================
struct GPUContext {
    int device_id;
    cudaDeviceProp properties;
    std::vector<cudaStream_t> streams;
    std::vector<uint64_t *> d_pairs;
    std::vector<uint32_t *> d_tickets;
    int optimal_blocks;
    int optimal_threads;
    uint64_t hashes_processed;
    bool use_extreme_kernel;
    bool use_ultra_kernel;

    GPUContext(int id) : device_id(id), hashes_processed(0) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));

        // Calculate optimal configuration based on GPU architecture
        if (properties.major >= 8) {
            // Ampere and newer (RTX 30xx, RTX 40xx, A100)
            optimal_threads = 256;
            optimal_blocks = properties.multiProcessorCount * 4;
            use_extreme_kernel = true;
            use_ultra_kernel = (properties.multiProcessorCount >= 100); // RTX 4090, A100
        } else if (properties.major >= 7) {
            // Volta/Turing (V100, RTX 20xx)
            optimal_threads = 256;
            optimal_blocks = properties.multiProcessorCount * 2;
            use_extreme_kernel = true;
            use_ultra_kernel = false;
        } else if (properties.major >= 6) {
            // Pascal (GTX 10xx, P100)
            optimal_threads = 256;
            optimal_blocks = properties.multiProcessorCount * 2;
            use_extreme_kernel = false;
            use_ultra_kernel = false;
        } else {
            optimal_threads = 512;
            optimal_blocks = properties.multiProcessorCount;
            use_extreme_kernel = false;
            use_ultra_kernel = false;
        }

        // Adjust blocks based on available memory
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        size_t mem_per_stream = sizeof(uint64_t) * 4 * RING_SIZE + sizeof(uint32_t);
        int max_streams = std::min(STREAMS_PER_GPU, (int) (free_mem / mem_per_stream / 2));

        // Create streams and allocate memory
        streams.resize(max_streams);
        d_pairs.resize(max_streams);
        d_tickets.resize(max_streams);

        for (int i = 0; i < max_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&d_pairs[i], sizeof(uint64_t) * 4 * RING_SIZE));
            CUDA_CHECK(cudaMalloc(&d_tickets[i], sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(d_tickets[i], 0, sizeof(uint32_t)));
        }

        // Set cache configuration
        CUDA_CHECK(cudaFuncSetCacheConfig(sha1_collision_kernel, cudaFuncCachePreferL1));
        CUDA_CHECK(cudaFuncSetCacheConfig(sha1_collision_kernel_multistream, cudaFuncCachePreferL1));
        if (use_extreme_kernel) {
            CUDA_CHECK(cudaFuncSetCacheConfig(sha1_collision_kernel_extreme, cudaFuncCachePreferL1));
        }
        if (use_ultra_kernel) {
            CUDA_CHECK(cudaFuncSetCacheConfig(sha1_collision_kernel_ultra, cudaFuncCachePreferL1));
        }

        printInfo();
    }

    ~GPUContext() {
        CUDA_CHECK(cudaSetDevice(device_id));
        for (size_t i = 0; i < streams.size(); i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaFree(d_pairs[i]));
            CUDA_CHECK(cudaFree(d_tickets[i]));
        }
    }

    void printInfo() {
        std::cout << "=== GPU " << device_id << " Configuration ===\n";
        std::cout << "Device        : " << properties.name << "\n";
        std::cout << "Compute Cap   : " << properties.major << "." << properties.minor << "\n";
        std::cout << "SMs           : " << properties.multiProcessorCount << "\n";
        std::cout << "Memory        : " << properties.totalGlobalMem / (1024.0 * 1024 * 1024) << " GB\n";
        std::cout << "Memory Clock  : " << properties.memoryClockRate / 1000 << " MHz\n";
        std::cout << "Memory Bus    : " << properties.memoryBusWidth << " bits\n";
        std::cout << "L2 Cache      : " << properties.l2CacheSize / (1024 * 1024) << " MB\n";
        std::cout << "Configuration : " << optimal_blocks << " blocks × " << optimal_threads << " threads\n";
        std::cout << "Streams       : " << streams.size() << "\n";
        std::cout << "Kernel Mode   : ";
        if (use_ultra_kernel) std::cout << "Ultra (4-way ILP)";
        else if (use_extreme_kernel) std::cout << "Extreme (2-way ILP)";
        else std::cout << "Standard";
        std::cout << "\n\n";
    }
};

// ==================== Result Handler ====================
class ResultHandler {
private:
    std::ofstream output_file;
    std::mutex file_mutex;
    std::array<uint8_t, 20> target_hash;

public:
    ResultHandler(const std::string &filename, const uint8_t target[20]) {
        std::copy(target, target + 20, target_hash.begin());

        output_file.open(filename, std::ios::app);
        if (!output_file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << "\n";
        }

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        output_file << "\n=== SHA-1 Collision Mining Started at "
                << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                << " ===\n";
        output_file << "Target: ";
        for (int i = 0; i < 20; i++) {
            output_file << std::hex << std::setw(2) << std::setfill('0') << (int) target_hash[i];
        }
        output_file << "\n\n";
        output_file.flush();
    }

    void saveResults(uint64_t *h_pairs, uint32_t count, int gpu_id) {
        std::lock_guard<std::mutex> lock(file_mutex);

        output_file << "[GPU " << gpu_id << "] Found " << count << " collision candidates:\n";

        for (uint32_t i = 0; i < std::min(count, 100u); i++) {
            output_file << "Candidate " << i + 1 << ":\n";

            // Reconstruct message
            uint8_t msg[32];
            for (int j = 0; j < 4; j++) {
                uint64_t word = h_pairs[i * 4 + j];
                for (int k = 0; k < 8; k++) {
                    msg[j * 8 + k] = (word >> (k * 8)) & 0xFF;
                }
            }

            output_file << "  Message: ";
            for (int j = 0; j < 32; j++) {
                output_file << std::hex << std::setw(2) << std::setfill('0') << (int) msg[j];
            }
            output_file << "\n";

            // Verify SHA-1
            uint8_t hash[20];
            sha1_ctx ctx;
            sha1_init(ctx);
            sha1_update(ctx, msg, 32);
            sha1_final(ctx, hash);

            output_file << "  SHA-1:   ";
            for (int j = 0; j < 20; j++) {
                output_file << std::hex << std::setw(2) << std::setfill('0') << (int) hash[j];
            }

            bool verified = (std::memcmp(hash, target_hash.data(), 20) == 0);
            output_file << " [" << (verified ? "VERIFIED" : "FAILED") << "]\n\n";
        }

        output_file.flush();

        // Console notification
        std::cout << "\n[GPU " << gpu_id << "] Found " << count << " collision candidates! ";
        std::cout << "Saved to output file.\n";
    }
};

// ==================== Performance Monitor ====================
class PerformanceMonitor {
private:
    std::chrono::steady_clock::time_point last_update;
    uint64_t last_hashes = 0;
    std::mutex monitor_mutex;

public:
    PerformanceMonitor() {
        last_update = std::chrono::steady_clock::now();
    }

    void update(const std::vector<std::unique_ptr<GPUContext> > &gpus) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);

        if (elapsed.count() >= PROGRESS_INTERVAL) {
            std::lock_guard<std::mutex> lock(monitor_mutex);

            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time);
            double total_seconds = total_elapsed.count();

            uint64_t current_hashes = g_total_hashes.load();
            uint64_t interval_hashes = current_hashes - last_hashes;
            double interval_rate = interval_hashes / elapsed.count() / 1e9;
            double average_rate = current_hashes / total_seconds / 1e9;

            std::cout << "\r[" << formatTime(total_seconds) << "] "
                    << "Hashes: " << std::fixed << std::setprecision(2)
                    << current_hashes / 1e9 << "B | "
                    << "Rate: " << interval_rate << " GH/s (avg: "
                    << average_rate << " GH/s) | "
                    << "Candidates: " << g_total_candidates.load();

            // Per-GPU stats
            std::cout << " | GPU: ";
            for (size_t i = 0; i < gpus.size(); i++) {
                double gpu_rate = gpus[i]->hashes_processed / total_seconds / 1e9;
                std::cout << "[" << i << ":" << std::fixed << std::setprecision(1)
                        << gpu_rate << "] ";
            }
            std::cout << "    " << std::flush;

            last_update = now;
            last_hashes = current_hashes;
        }
    }

    void finalReport() {
        auto now = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time);
        double total_seconds = total_elapsed.count();
        uint64_t total_hashes = g_total_hashes.load();
        double average_rate = total_hashes / total_seconds / 1e9;

        std::cout << "\n\n=== Final Statistics ===\n";
        std::cout << "Total Time    : " << formatTime(total_seconds) << "\n";
        std::cout << "Total Hashes  : " << std::fixed << std::setprecision(3)
                << total_hashes / 1e9 << " billion\n";
        std::cout << "Average Rate  : " << average_rate << " GH/s\n";
        std::cout << "Candidates    : " << g_total_candidates.load() << "\n";
        std::cout << "Efficiency    : " << std::scientific << std::setprecision(2)
                << (double) g_total_candidates.load() / total_hashes * 100 << "%\n";
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

// ==================== GPU Worker Thread ====================
void gpu_worker(GPUContext *gpu, ResultHandler *results, uint64_t base_seed) {
    CUDA_CHECK(cudaSetDevice(gpu->device_id));

    // Calculate work based on kernel type
    uint64_t work_per_kernel;
    if (gpu->use_ultra_kernel) {
        work_per_kernel = (uint64_t) gpu->optimal_blocks * 64 * 4; // 64 threads, 4 nonces each
    } else if (gpu->use_extreme_kernel) {
        work_per_kernel = (uint64_t) gpu->optimal_blocks * 128 * 2; // 128 threads, 2 nonces each
    } else {
        work_per_kernel = (uint64_t) gpu->optimal_blocks * gpu->optimal_threads * 8; // 8 nonces per thread
    }

    uint64_t local_seed = base_seed + gpu->device_id * (1ull << 48);

    std::vector<uint64_t> h_pairs(RING_SIZE * 4);

    while (!g_shutdown.load()) {
        // Launch kernels on all streams
        for (size_t s = 0; s < gpu->streams.size(); s++) {
            if (gpu->use_ultra_kernel) {
                // Ultra kernel for highest-end GPUs
                dim3 grid(gpu->optimal_blocks);
                dim3 block(64);

                sha1_collision_kernel_ultra<<<grid, block, 0, gpu->streams[s]>>>(
                    nullptr,
                    gpu->d_pairs[s],
                    gpu->d_tickets[s],
                    local_seed
                );
            } else if (gpu->use_extreme_kernel) {
                // Extreme kernel for high-end GPUs
                dim3 grid(gpu->optimal_blocks);
                dim3 block(128);

                sha1_collision_kernel_extreme<<<grid, block, 0, gpu->streams[s]>>>(
                    nullptr,
                    gpu->d_pairs[s],
                    gpu->d_tickets[s],
                    local_seed
                );
            } else {
                // Multi-stream kernel for standard GPUs
                dim3 grid(gpu->optimal_blocks);
                dim3 block(gpu->optimal_threads);

                sha1_collision_kernel_multistream<<<grid, block, 0, gpu->streams[s]>>>(
                    nullptr,
                    gpu->d_pairs[s],
                    gpu->d_tickets[s],
                    local_seed,
                    s,
                    gpu->streams.size()
                );
            }

            local_seed += work_per_kernel;
        }

        // Check results from all streams
        for (size_t s = 0; s < gpu->streams.size(); s++) {
            CUDA_CHECK(cudaStreamSynchronize(gpu->streams[s]));

            uint32_t found = 0;
            CUDA_CHECK(cudaMemcpy(&found, gpu->d_tickets[s], sizeof(uint32_t), cudaMemcpyDeviceToHost));

            if (found > 0) {
                g_total_candidates.fetch_add(found);

                uint32_t to_copy = std::min(found, RING_SIZE);
                CUDA_CHECK(cudaMemcpy(h_pairs.data(), gpu->d_pairs[s],
                    sizeof(uint64_t) * 4 * to_copy, cudaMemcpyDeviceToHost));

                results->saveResults(h_pairs.data(), to_copy, gpu->device_id);

                // Reset ticket
                CUDA_CHECK(cudaMemset(gpu->d_tickets[s], 0, sizeof(uint32_t)));
            }

            // Update counters
            uint64_t hashes = work_per_kernel;
            g_total_hashes.fetch_add(hashes);
            gpu->hashes_processed += hashes;
        }
    }
}

// ==================== Configuration Parser ====================
struct Config {
    std::vector<int> gpu_ids;
    std::string output_file = "sha1_collisions.txt";
    std::array<uint8_t, 32> target_preimage;
    bool benchmark_mode = false;
    int benchmark_seconds = 60;
};

Config parse_arguments(int argc, char **argv) {
    Config config;

    // Initialize with zeros
    config.target_preimage.fill(0);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--gpu" && i + 1 < argc) {
            config.gpu_ids.push_back(std::stoi(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            std::string hex = argv[++i];
            if (hex.length() != 64) {
                std::cerr << "Error: Target must be 64 hex characters (32 bytes)\n";
                std::exit(1);
            }
            for (int j = 0; j < 32; j++) {
                std::string byte = hex.substr(j * 2, 2);
                config.target_preimage[j] = std::stoi(byte, nullptr, 16);
            }
        } else if (arg == "--benchmark") {
            config.benchmark_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                config.benchmark_seconds = std::stoi(argv[++i]);
            }
        } else if (arg == "--help") {
            std::cout << "SHA-1 Collision Miner v2.0 (Ultra)\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --gpu <id>       GPU device ID (can be used multiple times)\n";
            std::cout << "  --output <file>  Output file (default: sha1_collisions.txt)\n";
            std::cout << "  --target <hex>   Target preimage in hex (default: zeros)\n";
            std::cout << "  --benchmark [s]  Run benchmark for s seconds (default: 60)\n";
            std::cout << "  --help           Show this help\n\n";
            std::cout << "Example:\n";
            std::cout << "  " << argv[0] << " --gpu 0 --gpu 1 --target ";
            std::cout << "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            std::exit(1);
        }
    }

    // Default: use all available GPUs
    if (config.gpu_ids.empty()) {
        int num_gpus;
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        for (int i = 0; i < num_gpus; i++) {
            config.gpu_ids.push_back(i);
        }
    }

    return config;
}

// ==================== Main Function ====================
int main(int argc, char **argv) {
    setup_signal_handlers();

    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|    SHA-1 Collision Miner v2.0 (Ultra)    |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Parse configuration
    Config config = parse_arguments(argc, argv);

    // Calculate target SHA-1
    uint8_t target_hash[20];
    sha1_ctx ctx;
    sha1_init(ctx);
    sha1_update(ctx, config.target_preimage.data(), 32);
    sha1_final(ctx, target_hash);

    // Convert to uint32_t array for GPU
    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(target_hash[4 * i]) << 24) |
                    (uint32_t(target_hash[4 * i + 1]) << 16) |
                    (uint32_t(target_hash[4 * i + 2]) << 8) |
                    uint32_t(target_hash[4 * i + 3]);
    }

    // Upload job to GPU constant memory
    upload_new_job(config.target_preimage.data(), target);

    std::cout << "Target Preimage: ";
    for (int i = 0; i < 32; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int) config.target_preimage[i];
    }
    std::cout << "\n";

    std::cout << "Target SHA-1: ";
    for (int i = 0; i < 20; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int) target_hash[i];
    }
    std::cout << std::dec << "\n\n";

    // Initialize GPUs
    std::vector<std::unique_ptr<GPUContext> > gpus;
    for (int id: config.gpu_ids) {
        try {
            gpus.push_back(std::make_unique<GPUContext>(id));
        } catch (const std::exception &e) {
            std::cerr << "Failed to initialize GPU " << id << ": " << e.what() << "\n";
        }
    }

    if (gpus.empty()) {
        std::cerr << "No GPUs available for mining!\n";
        return 1;
    }

    // Initialize result handler
    ResultHandler results(config.output_file, target_hash);

    // Initialize performance monitor
    PerformanceMonitor monitor;

    // Start timing
    g_start_time = std::chrono::steady_clock::now();

    // Generate random base seed
    std::random_device rd;
    uint64_t base_seed = (uint64_t(rd()) << 32) | rd();

    std::cout << "Starting collision search with " << gpus.size() << " GPU(s)...\n";
    std::cout << "Base seed: 0x" << std::hex << base_seed << std::dec << "\n";

    if (config.benchmark_mode) {
        std::cout << "Running in benchmark mode for " << config.benchmark_seconds << " seconds...\n";
    }
    std::cout << "\n";

    // Launch GPU worker threads
    std::vector<std::thread> workers;
    for (auto &gpu: gpus) {
        workers.emplace_back(gpu_worker, gpu.get(), &results, base_seed);
    }

    // Monitor progress
    if (config.benchmark_mode) {
        auto benchmark_end = std::chrono::steady_clock::now() +
                             std::chrono::seconds(config.benchmark_seconds);

        while (!g_shutdown.load() && std::chrono::steady_clock::now() < benchmark_end) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            monitor.update(gpus);
        }
        g_shutdown.store(true);
    } else {
        while (!g_shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            monitor.update(gpus);
        }
    }

    // Wait for workers to finish
    for (auto &worker: workers) {
        worker.join();
    }

    // Final report
    monitor.finalReport();

    // Additional benchmark statistics
    if (config.benchmark_mode) {
        auto total_time = std::chrono::steady_clock::now() - g_start_time;
        double seconds = std::chrono::duration<double>(total_time).count();
        uint64_t total_hashes = g_total_hashes.load();

        std::cout << "\n=== Benchmark Results ===\n";
        std::cout << "Total GPUs    : " << gpus.size() << "\n";
        std::cout << "Duration      : " << std::fixed << std::setprecision(2) << seconds << " seconds\n";
        std::cout << "Total Hashes  : " << total_hashes / 1e9 << " billion\n";
        std::cout << "Performance   : " << total_hashes / seconds / 1e9 << " GH/s\n";
        std::cout << "Per GPU       : " << total_hashes / seconds / 1e9 / gpus.size() << " GH/s\n";

        // Detailed per-GPU stats
        std::cout << "\nPer-GPU Performance:\n";
        for (size_t i = 0; i < gpus.size(); i++) {
            double gpu_ghps = gpus[i]->hashes_processed / seconds / 1e9;
            std::cout << "  GPU " << gpus[i]->device_id << " ("
                    << gpus[i]->properties.name << "): "
                    << gpu_ghps << " GH/s\n";
        }
    }

    if (g_total_candidates.load() > 0) {
        std::cout << "\n[SUCCESS] Found " << g_total_candidates.load() << " collisions!\n";
        std::cout << "Results saved to: " << config.output_file << "\n";
    } else if (!config.benchmark_mode) {
        std::cout << "\n[CONTINUING] No collisions found yet. Keep mining!\n";
    }

    return 0;
}
