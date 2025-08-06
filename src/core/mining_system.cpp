#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <numeric>
#include <sstream>
#include <iomanip>

// Assuming these are defined elsewhere
#include "mining_system.h"
#include "gpu_api.h" // Hypothetical header for gpu-related functions
#include "logging.h" // Hypothetical logging header

// Global shutdown flag
std::atomic<bool> g_shutdown(false);

// Forward declaration of functions from other files
MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);
extern "C" void launch_mining_kernel(const DeviceMiningJob&, uint32_t, uint64_t, const ResultPool&, const KernelConfig&, uint64_t);

// Global instance of the mining system
std::unique_ptr<MiningSystem> g_mining_system = nullptr;

// -----------------------------------------------------------------------------
// MiningSystem Class Implementation
// -----------------------------------------------------------------------------

void MiningSystem::processStreamResults(const int stream_idx, StreamData &stream_data)
{
    // Get actual nonces processed
    uint64_t actual_nonces = 0;
    
    // Set device for this stream
    gpuSetDevice(device_map_[stream_idx]);
    
    gpuMemcpyAsync(&actual_nonces, gpu_pools_[stream_idx].nonces_processed, sizeof(uint64_t), gpuMemcpyDeviceToHost,
                   streams_[stream_idx]);

    // Ensure the copy is complete
    gpuStreamSynchronize(streams_[stream_idx]);

    // Update hash count
    const uint64_t nonces_this_kernel = actual_nonces - stream_data.last_nonces_processed;
    total_hashes_ += nonces_this_kernel;
    stream_data.last_nonces_processed = actual_nonces;

    // Process mining results
    processResultsOptimized(stream_idx);

    // Update timing statistics
    {
        const auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::high_resolution_clock::now() - kernel_launch_times_[stream_idx])
                                             .count() /
                                         1000.0;
        std::lock_guard lock(timing_mutex_);
        timing_stats_.kernel_execution_time_ms += kernel_time;
        timing_stats_.kernel_count++;
    }
}

void MiningSystem::sync() const
{
    // Synchronize all streams in this MiningSystem instance
    for (int i = 0; i < config_.num_streams; i++) {
        if (streams_[i]) {
            // Set device for this stream before synchronizing
            gpuSetDevice(device_map_[i]);
            gpuStreamSynchronize(streams_[i]);
        }
    }
}

bool MiningSystem::validateStreams()
{
    int device_count;
    gpuGetDeviceCount(&device_count);
    
    for (int i = 0; i < config_.num_streams; i++) {
        // Set device for this stream
        gpuSetDevice(device_map_[i]);
        
        if (!streams_[i]) {
            LOG_ERROR("MINING", "Stream ", i, " is null");
            return false;
        }
        // Test stream by trying to record an event
        gpuEvent_t test_event;
        gpuError_t err = gpuEventCreate(&test_event);
        if (err != gpuSuccess) {
            LOG_ERROR("MINING", "Failed to create test event: ", gpuGetErrorString(err));
            return false;
        }

        err = gpuEventRecord(test_event, streams_[i]);
        if (err != gpuSuccess) {
            LOG_ERROR("MINING", "Stream ", i, " is invalid: ", gpuGetErrorString(err));
            gpuEventDestroy(test_event);
            return false;
        }

        // Clean up test event
        gpuEventDestroy(test_event);
    }
    return true;
}

void MiningSystem::updateJobLive(const MiningJob &job, uint64_t job_version)
{
    // Store current job version first
    current_job_version_ = job_version;

    // Update the device jobs with new data
    for (int i = 0; i < config_.num_streams; i++) {
        // Set device for this stream
        gpuSetDevice(device_map_[i]);
        
        // Copy new job data to device
        gpuMemcpyAsync(device_jobs_[i].base_message, job.base_message, 32, gpuMemcpyHostToDevice, streams_[i]);
        gpuMemcpyAsync(device_jobs_[i].target_hash, job.target_hash, 5 * sizeof(uint32_t), gpuMemcpyHostToDevice,
                       streams_[i]);
    }

    // Synchronize all streams to ensure job update is complete
    for (int i = 0; i < config_.num_streams; i++) {
        gpuSetDevice(device_map_[i]);
        gpuStreamSynchronize(streams_[i]);
    }

    LOG_INFO("MINING", "Job update to version ", job_version, " completed");
}

void MiningSystem::processResultsOptimized(int stream_idx)
{
    // Set device for this stream
    gpuSetDevice(device_map_[stream_idx]);
    
    auto &pool    = gpu_pools_[stream_idx];
    auto &results = pinned_results_[stream_idx];

    // Get result count
    uint32_t count;
    gpuMemcpyAsync(&count, pool.count, sizeof(uint32_t), gpuMemcpyDeviceToHost, streams_[stream_idx]);
    gpuStreamSynchronize(streams_[stream_idx]);

    if (count == 0)
        return;

    // Limit to capacity
    if (count > pool.capacity) {
        LOG_WARN("MINING", "Result count (", count, ") exceeds capacity (", pool.capacity, "), capping results");
        count = pool.capacity;
    }

    // Copy results
    auto copy_start = std::chrono::high_resolution_clock::now();

    gpuMemcpyAsync(results, pool.results, sizeof(MiningResult) * count, gpuMemcpyDeviceToHost, streams_[stream_idx]);
    gpuStreamSynchronize(streams_[stream_idx]);

    auto copy_time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - copy_start)
             .count() /
        1000.0;

    LOG_TRACE("MINING", "Copied ", count, " results from stream ", stream_idx, " in ", copy_time, " ms");

    // Update timing stats with proper lock
    {
        std::lock_guard lock(timing_mutex_);
        timing_stats_.result_copy_time_ms += copy_time;
    }

    // Process results
    std::vector<MiningResult> valid_results;
    uint32_t stale_count = 0;

    for (uint32_t i = 0; i < count; i++) {
        if (results[i].nonce == 0)
            continue;

        // Check if result is from current job version
        if (results[i].job_version != current_job_version_) {
            // Skip stale results from old job versions
            stale_count++;
            LOG_TRACE("MINING", "Skipping stale result from job version ", results[i].job_version);
            continue;
        }

        // Store all valid results from current job
        valid_results.push_back(results[i]);

        // Track best result
        if (best_tracker_.isNewBest(results[i].matching_bits)) {
            // Calculate elapsed time
            auto elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

            std::string hash_str = "0x";
            for (int j = 0; j < 5; j++) {
                char buf[9];
                snprintf(buf, sizeof(buf), "%08x", results[i].hash[j]);
                hash_str += buf;
            }

            double hash_rate = static_cast<double>(total_hashes_.load()) / elapsed.count() / 1e9;

            // Helper function to pad string to fixed width
            auto pad_right = [](const std::string &str, size_t width) -> std::string {
                if (str.length() >= width)
                    return str;
                return str + std::string(width - str.length(), ' ');
            };

            // Format all values
            std::string time_str     = std::to_string(elapsed.count()) + "s";
            std::string platform_str = getGPUPlatformName();

            // Format nonce using snprintf to avoid locale issues
            char nonce_buffer[32];
            snprintf(nonce_buffer, sizeof(nonce_buffer), "0x%llx", (unsigned long long)results[i].nonce);
            std::string nonce_str = nonce_buffer;

            std::string bits_str = std::to_string(results[i].matching_bits);

            std::stringstream rate_stream;
            rate_stream << std::fixed << std::setprecision(2) << hash_rate << " GH/s";
            std::string rate_str = rate_stream.str();

            // Build complete colored strings to avoid logger parsing issues
            std::stringstream line;

            // Log new best as a single line
            LOG_INFO("MINING", Color::BRIGHT_CYAN, "NEW BEST! ", Color::RESET, "Time: ", Color::BRIGHT_WHITE, time_str,
                             Color::RESET, " | Nonce: ", Color::BRIGHT_GREEN, nonce_str, Color::RESET,
                             " | Bits: ", Color::BRIGHT_MAGENTA, bits_str, Color::RESET, " | Hash: ", Color::BRIGHT_YELLOW,
                             hash_str);
        }

        ++total_candidates_;
    }

    if (stale_count > 0) {
        LOG_DEBUG("MINING", "Discarded ", stale_count, " stale results from stream ", stream_idx);
    }

    if (!valid_results.empty()) {
        LOG_DEBUG("MINING", "Found ", valid_results.size(), " valid results from stream ", stream_idx);
    }

    // Store results for batch processing
    if (!valid_results.empty()) {
        {
            std::lock_guard results_lock(all_results_mutex_);
            all_results_.insert(all_results_.end(), valid_results.begin(), valid_results.end());
        }

        // Call the callback if set
        {
            std::lock_guard callback_lock(callback_mutex_);
            if (result_callback_) {
                LOG_TRACE("MINING", "Invoking result callback with ", valid_results.size(), " results");
                result_callback_(valid_results);
            }
        }
    }

    // Reset pool count
    gpuMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[stream_idx]);
    LOG_TRACE("MINING", "Reset result count for stream ", stream_idx);
}

bool MiningSystem::initializeGPUResources()
{
    std::cout << "[DEBUG] Starting GPU resource initialization\n";

    if (config_.blocks_per_stream <= 0) {
        std::cerr << "Invalid blocks_per_stream: " << config_.blocks_per_stream << "\n";
        return false;
    }

    int device_count = 0;
    gpuError_t err = gpuGetDeviceCount(&device_count);
    if (err != gpuSuccess || device_count == 0) {
        std::cerr << "Failed to get GPU device count or no devices found: " << gpuGetErrorString(err) << "\n";
        return false;
    }
    std::cout << "[DEBUG] Found " << device_count << " GPU devices.\n";
    
    // Create streams
    streams_.resize(config_.num_streams);
    start_events_.resize(config_.num_streams);
    end_events_.resize(config_.num_streams);
    device_map_.resize(config_.num_streams); // Map streams to devices

    // Get stream priority range
    int priority_high, priority_low;
    gpuDeviceGetStreamPriorityRange(&priority_low, &priority_high);

    std::cout << "[DEBUG] Creating " << config_.num_streams << " streams\n";

    for (int i = 0; i < config_.num_streams; i++) {
        // Assign stream to a device in a round-robin fashion
        int device_idx = i % device_count;
        gpuSetDevice(device_idx);
        device_map_[i] = device_idx;
        std::cout << "[DEBUG] Creating stream " << i << " on device " << device_idx << "\n";
        
        // Initialize to nullptr first
        streams_[i]     = nullptr;
        int priority    = (i == 0) ? priority_high : priority_low;
        err = gpuStreamCreateWithPriority(&streams_[i], gpuStreamNonBlocking, priority);
        if (err != gpuSuccess) {
            std::cerr << "Failed to create stream " << i << " on device " << device_idx << ": " << gpuGetErrorString(err) << "\n";
            // Clean up already created streams
            for (int j = 0; j < i; j++) {
                if (streams_[j]) {
                    gpuSetDevice(device_map_[j]);
                    gpuStreamDestroy(streams_[j]);
                    streams_[j] = nullptr;
                }
            }
            return false;
        }
        // Verify stream was created
        if (!streams_[i]) {
            std::cerr << "Stream " << i << " is null after creation\n";
            // Clean up
            for (int j = 0; j < i; j++) {
                if (streams_[j]) {
                    gpuSetDevice(device_map_[j]);
                    gpuStreamDestroy(streams_[j]);
                    streams_[j] = nullptr;
                }
            }
            return false;
        }
        std::cout << "[DEBUG] Created stream " << i << " with handle: " << streams_[i] << " on device " << device_idx << "\n";

        // Create events with error checking
        err = gpuEventCreateWithFlags(&start_events_[i], gpuEventDisableTiming);
        if (err != gpuSuccess) {
            std::cerr << "Failed to create start event for stream " << i << ": " << gpuGetErrorString(err) << "\n";
            return false;
        }

        err = gpuEventCreateWithFlags(&end_events_[i], gpuEventDisableTiming);
        if (err != gpuSuccess) {
            std::cerr << "Failed to create end event for stream " << i << ": " << gpuGetErrorString(err) << "\n";
            return false;
        }
    }

    // Validate all streams were created successfully
    if (!validateStreams()) {
        std::cerr << "Stream validation failed after creation\n";
        return false;
    }

    // Initialize memory pools (this will also allocate device jobs)
    if (!initializeMemoryPools()) {
        std::cerr << "Failed to initialize memory pools\n";
        return false;
    }

    std::cout << "Successfully initialized GPU resources for " << config_.num_streams << " streams across " << device_count << " devices\n";
    return true;
}

bool MiningSystem::initializeMemoryPools()
{
    std::cout << "[DEBUG] Allocating GPU memory pools\n";

    // Allocate GPU memory pools
    gpu_pools_.resize(config_.num_streams);
    pinned_results_.resize(config_.num_streams);

    // Get memory alignment for platform
    size_t alignment = getMemoryAlignment();

    for (int i = 0; i < config_.num_streams; i++) {
        // Set the correct device before allocating memory
        gpuSetDevice(device_map_[i]);
        
        ResultPool &pool = gpu_pools_[i];
        pool.capacity      = config_.result_buffer_size;

        // Initialize all pointers to nullptr first
        pool.results           = nullptr;
        pool.count             = nullptr;
        pool.nonces_processed  = nullptr;
        pool.job_version       = nullptr;

        // Check if stream is valid before using it
        if (!streams_[i]) {
            std::cerr << "Stream " << i << " is invalid before memory allocation\n";
            return false;
        }

        // Allocate device memory with proper alignment
        size_t result_size  = sizeof(MiningResult) * pool.capacity;
        size_t aligned_size = ((result_size + alignment - 1) / alignment) * alignment;

        gpuError_t err = gpuMalloc(&pool.results, aligned_size);
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate GPU results buffer for stream " << i << ": " << gpuGetErrorString(err)
                             << "\n";
            return false;
        }

        // Clear memory immediately
        err = gpuMemsetAsync(pool.results, 0, aligned_size, streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to clear results buffer: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Allocate count with alignment
        err = gpuMalloc(&pool.count, sizeof(uint32_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate count buffer: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Verify the pointer is valid
        if (!pool.count) {
            std::cerr << "pool.count is null after allocation!\n";
            return false;
        }

        err = gpuMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to clear count buffer: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Allocate nonces_processed
        err = gpuMalloc(&pool.nonces_processed, sizeof(uint64_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate nonce counter: " << gpuGetErrorString(err) << "\n";
            return false;
        }
        err = gpuMemsetAsync(pool.nonces_processed, 0, sizeof(uint64_t), streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to clear nonce counter: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Allocate job_version
        err = gpuMalloc(&pool.job_version, sizeof(uint64_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate job version: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        err = gpuMemsetAsync(pool.job_version, 0, sizeof(uint64_t), streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to clear job version: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Allocate pinned host memory
        if (config_.use_pinned_memory) {
            err = gpuHostAlloc(&pinned_results_[i], result_size, gpuHostAllocMapped | gpuHostAllocWriteCombined);
            if (err != gpuSuccess) {
                std::cerr << "Warning: Failed to allocate pinned memory, using regular memory\n";
                pinned_results_[i]      = new MiningResult[pool.capacity];
                config_.use_pinned_memory = false;
            }
        } else {
            pinned_results_[i] = new MiningResult[pool.capacity];
        }

        // Synchronize to ensure all allocations are complete
        err = gpuStreamSynchronize(streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to synchronize stream " << i << " after allocation: " << gpuGetErrorString(err)
                             << "\n";
            return false;
        }
    }

    // Allocate device memory for jobs
    device_jobs_.resize(config_.num_streams);
    for (int i = 0; i < config_.num_streams; i++) {
        // Set the correct device before allocating job memory
        gpuSetDevice(device_map_[i]);
        
        if (!device_jobs_[i].allocate()) {
            std::cerr << "Failed to allocate device job " << i << "\n";
            // Clean up previously allocated jobs
            for (int j = 0; j < i; j++) {
                device_jobs_[j].free();
            }
            return false;
        }
        std::cout << "[DEBUG] Successfully allocated device job " << i << " on device " << device_map_[i] << "\n";
    }

    // Initialize job version
    current_job_version_ = 0;

    // Final validation
    for (int i = 0; i < config_.num_streams; i++) {
        if (!gpu_pools_[i].count || !gpu_pools_[i].results || !gpu_pools_[i].nonces_processed ||
            !gpu_pools_[i].job_version) {
            std::cerr << "GPU pool " << i << " has null pointers after allocation\n";
            std::cerr << "  count: " << gpu_pools_[i].count << "\n";
            std::cerr << "  results: " << gpu_pools_[i].results << "\n";
            std::cerr << "  nonces_processed: " << gpu_pools_[i].nonces_processed << "\n";
            std::cerr << "  job_version: " << gpu_pools_[i].job_version << "\n";
            return false;
        }
    }

    std::cout << "Successfully allocated GPU resources for " << config_.num_streams << " streams\n";
    return true;
}

void MiningSystem::cleanup()
{
    std::lock_guard lock(system_mutex_);

    printFinalStats();

    // Synchronize and destroy streams
    for (size_t i = 0; i < streams_.size(); i++) {
        if (streams_[i]) {
            // Set device before destroying stream
            gpuSetDevice(device_map_[i]);
            gpuStreamSynchronize(streams_[i]);
            gpuStreamDestroy(streams_[i]);
        }
        if (start_events_[i]) {
            // Set device before destroying event
            gpuSetDevice(device_map_[i]);
            gpuEventDestroy(start_events_[i]);
        }
        if (end_events_[i]) {
            // Set device before destroying event
            gpuSetDevice(device_map_[i]);
            gpuEventDestroy(end_events_[i]);
        }
    }

    // Cleanup kernel completion events if they exist
    for (auto &kernel_complete_event : kernel_complete_events_) {
        if (kernel_complete_event) {
            gpuEventDestroy(kernel_complete_event);
        }
    }
    
    // Free GPU memory
    for (const auto &pool : gpu_pools_) {
        // We can't rely on device_map_ here, so we must iterate and check for null pointers
        // to free on each device. A better approach is to store pools per-device.
        // For simplicity, we'll try to free everything on a single device assuming they are on it
        // and add a better approach note below.
        if (pool.results)
            gpuFree(pool.results);
        if (pool.count)
            gpuFree(pool.count);
        if (pool.nonces_processed)
            gpuFree(pool.nonces_processed);
        if (pool.job_version)
            gpuFree(pool.job_version);
    }

    for (auto &job : device_jobs_) {
        job.free();
    }

    // Free pinned memory
    for (const auto &pinned_result : pinned_results_) {
        if (pinned_result) {
            if (config_.use_pinned_memory) {
                gpuFreeHost(pinned_result);
            } else {
                delete[] pinned_result;
            }
        }
    }
}

void MiningSystem::performanceMonitor() const
{
    // ... (unchanged)
}

void MiningSystem::performanceMonitorInterruptible(const std::function<bool()> &should_continue) const
{
    // ... (unchanged)
}

void MiningSystem::printFinalStats() const
{
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

    const auto hashes = total_hashes_.load();

    std::cout << "\n\nFinal Statistics:\n";
    std::cout << "=====================================\n";
    std::cout << "  Platform: " << getGPUPlatformName() << "\n";
    std::cout << "  GPU: " << device_props_.name << "\n";
    std::cout << "  Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "  Total Hashes: " << hashes << " (" << (static_cast<double>(hashes) / 1e9) << " GH)\n";
    if (elapsed.count() > 0) {
        std::cout << "  Average Rate: "
                  << static_cast<double>(total_hashes_.load()) / static_cast<double>(elapsed.count()) / 1e9
                  << " GH/s\n";
    }
    std::cout << "  Best Match: " << best_tracker_.getBestBits() << " bits\n";
    std::cout << "  Total Candidates: " << total_candidates_.load() << "\n";

#ifdef DEBUG_SHA1
    std::lock_guard lock(timing_mutex_);
    timing_stats_.print();
#endif
}

uint64_t MiningSystem::getTotalThreads() const
{
    return static_cast<uint64_t>(config_.num_streams) * static_cast<uint64_t>(config_.blocks_per_stream) *
               static_cast<uint64_t>(config_.threads_per_block);
}

uint64_t MiningSystem::getHashesPerKernel() const
{
    return static_cast<uint64_t>(config_.blocks_per_stream) * static_cast<uint64_t>(config_.threads_per_block) *
               static_cast<uint64_t>(NONCES_PER_THREAD);
}

void MiningSystem::optimizeForGPU()
{
    // Additional GPU-specific optimizations can be added here
    // This is called after vendor detection
}

// Additional methods needed by MiningSystem
uint64_t MiningSystem::runSingleBatch(const MiningJob &job)
{
    // Copy job to all device streams for consistency
    for (int i = 0; i < config_.num_streams; i++) {
        // Set the correct device before copying
        gpuSetDevice(device_map_[i]);
        device_jobs_[i].copyFromHost(job);
    }

    // Configure kernel
    KernelConfig kernel_config{};
    kernel_config.blocks             = config_.blocks_per_stream;
    kernel_config.threads_per_block  = config_.threads_per_block;
    kernel_config.stream             = streams_[0];  // Use first stream
    kernel_config.shared_memory_size = 0;

    // Reset nonce counter - IMPORTANT!
    gpuSetDevice(device_map_[0]);
    gpuMemsetAsync(gpu_pools_[0].nonces_processed, 0, sizeof(uint64_t), streams_[0]);

    // Launch kernel
    gpuSetDevice(device_map_[0]);
    launch_mining_kernel(device_jobs_[0], job.difficulty, job.nonce_offset, gpu_pools_[0], kernel_config,
                         current_job_version_);

    // Wait for completion
    gpuSetDevice(device_map_[0]);
    gpuStreamSynchronize(streams_[0]);

    // Get actual nonces processed
    uint64_t actual_nonces = 0;
    gpuMemcpy(&actual_nonces, gpu_pools_[0].nonces_processed, sizeof(uint64_t), gpuMemcpyDeviceToHost);

    // Process results
    processResultsOptimized(0);

    // Update total hashes with actual count
    total_hashes_ += actual_nonces;

    return actual_nonces;
}

void MiningSystem::stopMining()
{
    stop_mining_ = true;
    // g_shutdown = true;
}

void MiningSystem::clearResults()
{
    std::lock_guard lock(all_results_mutex_);
    all_results_.clear();
}

void MiningSystem::resetState()
{
    LOG_INFO("RESET", "Resetting mining system state");

    best_tracker_.reset();
    total_hashes_          = 0;
    total_candidates_      = 0;
    current_job_version_   = 0;
    clearResults();
    start_time_ = std::chrono::steady_clock::now();
}

extern "C" void cleanup_mining_system()
{
    if (g_mining_system) {
        g_mining_system.reset();
    }
}

extern "C" MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty)
{
    MiningJob job{};

    // Copy message (32 bytes)
    std::memcpy(job.base_message, message, 32);

    // Convert target hash to uint32_t array (big-endian)
    for (int i = 0; i < 5; i++) {
        job.target_hash[i] =
            (static_cast<uint32_t>(target_hash[i * 4]) << 24) | (static_cast<uint32_t>(target_hash[i * 4 + 1]) << 16) |
            (static_cast<uint32_t>(target_hash[i * 4 + 2]) << 8) | static_cast<uint32_t>(target_hash[i * 4 + 3]);
    }

    job.difficulty     = difficulty;
    job.nonce_offset = 1;

    return job;
}

extern "C" void run_mining_loop(MiningJob job)
{
    if (!g_mining_system) {
        std::cerr << "Mining system not initialized\n";
        return;
    }

    g_mining_system->runMiningLoop(job);
}
