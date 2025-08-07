#include "multi_gpu_manager.hpp"
#ifdef USE_HIP
    #include "architecture/gpu_architecture.hpp"
#endif
#include <algorithm>
#include <iomanip>
#include <iostream>

#include "config_utils.hpp"

MultiGPUManager::MultiGPUManager()
{
    start_time_ = std::chrono::steady_clock::now();
}

MultiGPUManager::~MultiGPUManager()
{
    if (!shutdown_.load()) {
        stopMining();
    }

    // Force cleanup any remaining resources
    for (const auto &worker : workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            LOG_WARN("MULTI_GPU", "Destructor: Force detaching worker thread for GPU ", worker->device_id);
            worker->worker_thread->detach();
        }
    }
}

bool MultiGPUManager::initialize(const std::vector<int> &gpu_ids)
{
    std::cout << "\nInitializing Multi-GPU Mining System\n";
    std::cout << "=====================================\n";

    for (int gpu_id : gpu_ids) {
        auto worker       = std::make_unique<GPUWorker>();
        worker->device_id = gpu_id;

        // Get device properties
        gpuDeviceProp props;
        if (gpuError_t err = gpuGetDeviceProperties(&props, gpu_id); err != gpuSuccess) {
            std::cerr << "Failed to get properties for GPU " << gpu_id << "\n";
            continue;
        }

        std::cout << "\nInitializing GPU " << gpu_id << ": " << props.name << "\n";

#ifdef USE_HIP
        // AMD-specific initialization
        AMDArchitecture arch      = AMDGPUDetector::detectArchitecture(props);
        AMDArchParams arch_params = AMDGPUDetector::getArchitectureParams(arch, props);

        std::cout << "  Architecture: " << AMDGPUDetector::getArchitectureName(arch) << " (" << props.gcnArchName
                  << ")\n";
        std::cout << "  Wave size: " << arch_params.wave_size << "\n";
        std::cout << "  Max waves per CU: " << arch_params.waves_per_cu << "\n";
        std::cout << "  Compute units: " << props.multiProcessorCount << "\n";

        if (AMDGPUDetector::hasKnownIssues(arch, props.name)) {
            std::cout << "WARNING: GPU " << gpu_id << " has known compatibility issues.\n";

            // Check ROCm version
            int version;
            if (hipRuntimeGetVersion(&version) == hipSuccess) {
                int major = version / 10000000;
                int minor = (version % 10000000) / 100000;
                int patch = (version % 100000) / 100;
                std::cout << "Current ROCm version: " << major << "." << minor << "." << patch << "\n";

                if (version < 50700000 && arch == AMDArchitecture::RDNA3) {
                    std::cout << "RDNA3 requires ROCm 5.7 or later. Skipping GPU " << gpu_id << "\n";
                    continue;
                }
            }
        }
#endif

        // Create mining system configuration
        MiningSystem::Config config;
        config.device_id = gpu_id;

        // Apply user config if available
        if (user_config_) {
            // Cast to MiningConfig and apply settings
            auto mining_config = static_cast<const MiningConfig *>(user_config_);
            // Apply ONLY the performance settings, NOT the device_id
            if (mining_config->user_specified.num_streams && mining_config->num_streams > 0) {
                config.num_streams = mining_config->num_streams;
            } else {
                config.num_streams = 8;  // Default
            }
            if (mining_config->user_specified.threads_per_block && mining_config->threads_per_block > 0) {
                config.threads_per_block = mining_config->threads_per_block;
            } else {
                config.threads_per_block = DEFAULT_THREADS_PER_BLOCK;
            }
            if (mining_config->user_specified.blocks_per_stream && mining_config->blocks_per_stream > 0) {
                config.blocks_per_stream = mining_config->blocks_per_stream;
            }

            if (mining_config->user_specified.result_buffer_size && mining_config->result_buffer_size > 0) {
                config.result_buffer_size = mining_config->result_buffer_size;
            } else {
                config.result_buffer_size = 1024;  // Default
            }

            config.use_pinned_memory = mining_config->use_pinned_memory;
        } else {
            // Default values if no user config
            config.num_streams        = 8;
            config.threads_per_block  = DEFAULT_THREADS_PER_BLOCK;
            config.use_pinned_memory  = true;
            config.result_buffer_size = 1024;
        }

        // Create and initialize mining system
        worker->mining_system = std::make_unique<MiningSystem>(config);

        if (user_config_) {
            worker->mining_system->setUserConfig(user_config_);
        }

        try {
            if (!worker->mining_system->initialize()) {
                std::cerr << "Failed to initialize GPU " << gpu_id << "\n";
                continue;
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception initializing GPU " << gpu_id << ": " << e.what() << "\n";
            continue;
        } catch (...) {
            std::cerr << "Unknown exception initializing GPU " << gpu_id << "\n";
            continue;
        }

        workers_.push_back(std::move(worker));
        std::cout << "Successfully initialized GPU " << gpu_id << "\n";
    }

    if (workers_.empty()) {
        std::cerr << "No GPUs were successfully initialized\n";
        return false;
    }

    std::cout << "\nSuccessfully initialized " << workers_.size() << " GPU(s) for mining\n";
    std::cout << "=====================================\n\n";

    return true;
}

void MultiGPUManager::stopMining()
{
    if (shutdown_.load()) {
        return;
    }

    LOG_INFO("MULTI_GPU", "Stopping all mining operations...");

    // Signal shutdown FIRST
    shutdown_ = true;

    // Stop all worker mining systems to interrupt any blocking operations
    for (auto &worker : workers_) {
        if (worker->mining_system) {
            worker->mining_system->stopMining();
        }
    }

    // Give threads a moment to notice the shutdown flag
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Wait for all worker threads with timeout
    waitForWorkersWithTimeout(std::chrono::seconds(5));

    LOG_INFO("MULTI_GPU", "All mining operations stopped");
}

void MultiGPUManager::waitForWorkers() const
{
    for (auto &worker : workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            worker->worker_thread->join();
        }
        worker->active = false;
    }
}

void MultiGPUManager::waitForWorkersWithTimeout(const std::chrono::seconds timeout) const
{
    auto start = std::chrono::steady_clock::now();
    for (auto &worker : workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            // Calculate remaining timeout
            auto elapsed   = std::chrono::steady_clock::now() - start;
            auto remaining = timeout - std::chrono::duration_cast<std::chrono::seconds>(elapsed);
            if (remaining.count() <= 0) {
                LOG_ERROR("MULTI_GPU", "Timeout waiting for worker threads to stop");
                // Force detach remaining threads
                for (auto &w : workers_) {
                    if (w->worker_thread && w->worker_thread->joinable()) {
                        LOG_WARN("MULTI_GPU", "Force detaching worker thread for GPU ", w->device_id);
                        w->worker_thread->detach();
                    }
                }
                break;
            }
            // Try to join with timeout using a condition variable
            std::mutex m;
            std::condition_variable cv;
            bool thread_finished = false;

            std::thread joiner([&]() {
                worker->worker_thread->join();
                std::lock_guard<std::mutex> lock(m);
                thread_finished = true;
                cv.notify_all();
            });

            std::unique_lock<std::mutex> lock(m);
            if (cv.wait_for(lock, remaining, [&]() { return thread_finished; })) {
                joiner.join();
                LOG_INFO("MULTI_GPU", "Worker thread for GPU ", worker->device_id, " stopped cleanly");
            } else {
                LOG_WARN("MULTI_GPU", "Worker thread for GPU ", worker->device_id, " did not stop in time");
                joiner.detach();
                worker->worker_thread->detach();
            }
        }
        worker->active = false;
    }
}

uint64_t MultiGPUManager::getNextNonceBatch()
{
    return global_nonce_counter_.fetch_add(NONCE_BATCH_SIZE);
}

void MultiGPUManager::processWorkerResults(GPUWorker *worker, const std::vector<MiningResult> &results) const
{
    // Update worker stats
    worker->candidates_found += results.size();

    // Check for new best
    for (const auto &result : results) {
        if (result.matching_bits > worker->best_match_bits) {
            worker->best_match_bits = result.matching_bits;

            // Check if this is a global best
            if (global_best_tracker_.isNewBest(result.matching_bits)) {
                std::lock_guard lock(stats_mutex_);
                auto elapsed =
                    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

                std::cout << "\n[GPU " << worker->device_id << " - NEW BEST!] Time: " << elapsed.count() << "s\n";
                std::cout << "  Nonce: 0x" << std::hex << result.nonce << std::dec << "\n";
                std::cout << "  Matching bits: " << result.matching_bits << "\n";
                std::cout << "  Hash: ";
                for (int j = 0; j < 5; j++) {
                    std::cout << std::hex << std::setw(8) << std::setfill('0') << result.hash[j];
                    if (j < 4)
                        std::cout << " ";
                }
                std::cout << std::dec << "\n\n";
            }
        }
    }

    // Forward to global callback
    {
        std::lock_guard lock(callback_mutex_);
        if (result_callback_) {
            result_callback_(results);
        }
    }
}

void MultiGPUManager::workerThread(GPUWorker *worker, const MiningJob &job)
{
    // CRITICAL: Set GPU context for this thread at the very beginning
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Failed to set device context: ", gpuGetErrorString(err));
        return;
    }

    // Synchronize to ensure device is ready
    err = gpuDeviceSynchronize();
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Device not ready: ", gpuGetErrorString(err));
        return;
    }

    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker thread started");
    worker->active = true;

    // Set result callback
    auto worker_callback = [this, worker](const std::vector<MiningResult> &results) {
        processWorkerResults(worker, results);
    };
    worker->mining_system->setResultCallback(worker_callback);

    // Error handling
    int consecutive_errors           = 0;
    const int max_consecutive_errors = 5;

    // Get initial nonce batch
    uint64_t current_nonce_base   = getNextNonceBatch();
    uint64_t nonces_used_in_batch = 0;

    // Run until shutdown
    while (!shutdown_ && consecutive_errors < max_consecutive_errors) {
        try {
            // IMPORTANT: Verify we still have the correct device context
            int current_device;
            gpuGetDevice(&current_device);
            if (current_device != worker->device_id) {
                LOG_WARN("MULTI_GPU", "GPU ", worker->device_id, " - Context switched, resetting");
                err = gpuSetDevice(worker->device_id);
                if (err != gpuSuccess) {
                    LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id,
                              " - Failed to reset context: ", gpuGetErrorString(err));
                    consecutive_errors++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
            }

            // Create job with current nonce offset
            MiningJob worker_job    = job;
            worker_job.nonce_offset = current_nonce_base + nonces_used_in_batch;

            // Run a single kernel batch
            uint64_t hashes_this_round = 0;
            try {
                hashes_this_round = worker->mining_system->runSingleBatch(worker_job);
            } catch (const std::runtime_error &e) {
                LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Kernel launch failed: ", e.what());
                consecutive_errors++;

                // Try to recover by resetting the mining system
                if (consecutive_errors >= 3) {
                    LOG_WARN("MULTI_GPU", "GPU ", worker->device_id, " - Attempting recovery");
                    worker->mining_system->resetState();
                    // Clear any GPU errors
                    gpuGetLastError();
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
                continue;
            }

            if (hashes_this_round == 0) {
                // Fallback estimation
                auto config = worker->mining_system->getConfig();
                hashes_this_round =
                    static_cast<uint64_t>(config.blocks_per_stream) * config.threads_per_block * NONCES_PER_THREAD;
            }

            // Update stats
            worker->hashes_computed += hashes_this_round;
            nonces_used_in_batch += hashes_this_round;

            // Check if we need a new nonce batch
            if (nonces_used_in_batch < NONCE_BATCH_SIZE * 0.9) {
                current_nonce_base   = getNextNonceBatch();
                nonces_used_in_batch = 0;
            }

            // Reset error counter on success
            consecutive_errors = 0;
        } catch (const std::exception &e) {
            LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Error: ", e.what());
            consecutive_errors++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    worker->active = false;
    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker thread finished");
}

void MultiGPUManager::workerThreadInterruptibleWithOffset(GPUWorker *worker, const MiningJob &job,
                                                          std::function<bool()> should_continue,
                                                          std::atomic<uint64_t> &shared_nonce_counter)
{
    // CRITICAL: Set GPU context for this thread at the very beginning
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Failed to set device context: ", gpuGetErrorString(err));
        return;
    }

    // Synchronize to ensure device is ready
    err = gpuDeviceSynchronize();
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Device not ready: ", gpuGetErrorString(err));
        return;
    }

    // Find this GPU's index in the workers array
    int gpu_index = -1;
    for (size_t i = 0; i < workers_.size(); i++) {
        if (workers_[i].get() == worker) {
            gpu_index = static_cast<int>(i);
            break;
        }
    }
    if (gpu_index == -1) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Failed to find worker index!");
        return;
    }
    const int total_gpus = static_cast<int>(workers_.size());
    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker thread started (worker index ", gpu_index, " of ",
             total_gpus, " total workers)");
    worker->active = true;

    // Set result callback
    auto worker_callback = [this, worker](const std::vector<MiningResult> &results) {
        processWorkerResults(worker, results);
    };
    worker->mining_system->setResultCallback(worker_callback);

    // Error handling
    int consecutive_errors               = 0;
    constexpr int max_consecutive_errors = 5;

    // Each GPU will grab nonce batches from the shared counter
    constexpr uint64_t gpu_batch_size = NONCE_BATCH_SIZE;

    // Run until shutdown OR should_continue returns false
    while (!shutdown_.load(std::memory_order_relaxed) && consecutive_errors < max_consecutive_errors &&
           should_continue()) {
        try {
            // IMPORTANT: Verify we still have the correct device context
            int current_device;
            gpuGetDevice(&current_device);
            if (current_device != worker->device_id) {
                LOG_WARN("MULTI_GPU", "GPU ", worker->device_id, " - Context switched, resetting");
                err = gpuSetDevice(worker->device_id);
                if (err != gpuSuccess) {
                    LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id,
                              " - Failed to reset context: ", gpuGetErrorString(err));
                    consecutive_errors++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
            }

            // Get next nonce batch atomically
            const uint64_t batch_start = shared_nonce_counter.fetch_add(gpu_batch_size);
            const uint64_t batch_end   = batch_start + gpu_batch_size;

            LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Processing batch: ", batch_start, " to ", batch_end,
                     " (", gpu_batch_size, " nonces)");

            // Create job for this batch
            MiningJob batch_job    = job;
            batch_job.nonce_offset = batch_start;

            // Create continuation function for this batch
            auto batch_continue = [&should_continue, &shutdown = this->shutdown_]() -> bool {
                return should_continue() && !shutdown.load();
            };

            // Run the entire batch
            uint64_t final_nonce =
                worker->mining_system->runMiningLoopInterruptibleWithOffset(batch_job, batch_continue, batch_start);

            // Calculate how many nonces were actually processed
            uint64_t nonces_processed = 0;
            if (final_nonce > batch_start) {
                nonces_processed = final_nonce - batch_start;

                // Cap at batch size if needed
                if (nonces_processed > gpu_batch_size) {
                    LOG_WARN("MULTI_GPU", "GPU ", worker->device_id,
                             " processed more than batch size: ", nonces_processed, " > ", gpu_batch_size);
                    nonces_processed = gpu_batch_size;
                }
            }

            LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Batch complete. Processed ", nonces_processed,
                     " nonces");

            // Reset error counter on success
            consecutive_errors = 0;

        } catch (const std::exception &e) {
            LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Error: ", e.what());
            consecutive_errors++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (shutdown_.load(std::memory_order_relaxed)) {
            LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Shutdown detected, breaking loop");
            break;
        }
    }

    worker->active = false;

    if (!should_continue()) {
        LOG_INFO("MULTI_GPU", "GPU ", worker->device_id,
                 " - Worker stopped (external signal) at global nonce: ", shared_nonce_counter.load());
    } else if (consecutive_errors >= max_consecutive_errors) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Worker stopped (too many errors)");
    } else {
        LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker finished (shutdown)");
    }
}

uint64_t MultiGPUManager::getTotalHashes() const
{
    uint64_t total = 0;
    for (const auto &worker : workers_) {
        if (worker->mining_system && worker->active.load()) {
            auto stats = worker->mining_system->getStats();
            total += stats.hashes_computed;
        }
    }
    return total;
}

void MultiGPUManager::runMining(const MiningJob &job)
{
    current_difficulty_ = job.difficulty;
    shutdown_           = false;
    start_time_         = std::chrono::steady_clock::now();
    global_best_tracker_.reset();

    // Reset all worker stats
    for (auto &worker : workers_) {
        worker->hashes_computed  = 0;
        worker->candidates_found = 0;
        worker->best_match_bits  = 0;
    }

    // Start worker threads
    for (auto &worker : workers_) {
        worker->worker_thread = std::make_unique<std::thread>(&MultiGPUManager::workerThread, this, worker.get(), job);
    }
    waitForWorkers();
}

void MultiGPUManager::runMiningInterruptibleWithOffset(const MiningJob &job,
                                                       const std::function<bool()> &should_continue,
                                                       std::atomic<uint64_t> &global_nonce_offset)
{
    current_difficulty_ = job.difficulty;
    shutdown_           = false;
    start_time_         = std::chrono::steady_clock::now();
    global_best_tracker_.reset();

    // Reset all worker stats
    for (const auto &worker : workers_) {
        worker->hashes_computed  = 0;
        worker->candidates_found = 0;
        worker->best_match_bits  = 0;
    }

    // CRITICAL: Log initial state
    uint64_t initial_nonce = global_nonce_offset.load();
    LOG_INFO("MULTI_GPU", "Starting multi-GPU mining from nonce offset: ", initial_nonce);

    // CRITICAL: Give each GPU a unique starting offset to prevent overlap
    const size_t num_gpus          = workers_.size();
    constexpr uint64_t gpu_spacing = NONCE_BATCH_SIZE * 10000;  // Space GPUs apart by 1000 batches

    // Start worker threads with proper spacing
    for (size_t i = 0; i < workers_.size(); i++) {
        auto &worker = workers_[i];

        // Set this GPU's initial offset
        uint64_t gpu_start_offset = initial_nonce + (i * gpu_spacing);

        // Create a per-GPU nonce counter
        auto *gpu_nonce_counter = new std::atomic<uint64_t>(gpu_start_offset);

        // Create a lambda that includes GPU-specific offset
        auto gpu_should_continue = [should_continue, gpu_id = worker->device_id]() -> bool {
            const bool cont = should_continue();
            if (!cont) {
                LOG_DEBUG("MULTI_GPU", "GPU ", gpu_id, " - should_continue returned false");
            }
            return cont;
        };

        // Pass the per-GPU counter instead of the global one
        worker->worker_thread = std::make_unique<std::thread>(
            [this, worker_ptr = worker.get(), job, gpu_should_continue, gpu_nonce_counter, &global_nonce_offset]() {
                // Call the original function but with per-GPU counter
                this->workerThreadInterruptibleWithOffset(worker_ptr, job, gpu_should_continue, *gpu_nonce_counter);

                // Update global offset with this GPU's final position
                uint64_t final_gpu_nonce = gpu_nonce_counter->load();
                uint64_t current_global  = global_nonce_offset.load();
                while (current_global < final_gpu_nonce) {
                    if (global_nonce_offset.compare_exchange_weak(current_global, final_gpu_nonce)) {
                        break;
                    }
                }

                delete gpu_nonce_counter;
            });
    }

    // Wait for workers to finish
    waitForWorkers();

    uint64_t final_nonce = global_nonce_offset.load();
    LOG_INFO("MULTI_GPU", "Multi-GPU mining complete. Nonces processed: ", final_nonce - initial_nonce, " (from ",
             initial_nonce, " to ", final_nonce, ")");

    if (!should_continue()) {
        LOG_INFO("MULTI_GPU", "External stop signal received - all workers stopped at nonce: ", final_nonce);
    }
}

void MultiGPUManager::sync() const
{
    for (const auto &worker : workers_) {
        if (worker->mining_system && worker->active) {
            worker->mining_system->sync();
        }
    }
}

void MultiGPUManager::updateJobLive(const MiningJob &job, uint64_t job_version) const
{
    LOG_INFO("MULTI_GPU", "Updating job to version ", job_version, " on all GPUs");

    // Update job on all active GPUs
    for (const auto &worker : workers_) {
        if (worker->mining_system) {
            worker->mining_system->updateJobLive(job, job_version);
        }
    }

    // Ensure all GPUs have completed the update
    sync();

    LOG_INFO("MULTI_GPU", "Job update to version ", job_version, " completed on all active GPUs");
}

double MultiGPUManager::getTotalHashRate() const
{
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);

    // Need at least 1 second of data
    if (elapsed.count() < 1000)
        return 0.0;

    uint64_t total_hashes = 0;

    // CRITICAL: Get actual hashes from each GPU's mining system
    for (const auto &worker : workers_) {
        if (worker->mining_system && worker->active.load()) {
            auto stats = worker->mining_system->getStats();
            total_hashes += stats.hashes_computed;
        }
    }

    // Convert to seconds for accurate rate
    double seconds = elapsed.count() / 1000.0;
    return static_cast<double>(total_hashes) / seconds;
}

size_t MultiGPUManager::getActiveWorkerCount() const
{
    size_t count = 0;
    for (const auto &worker : workers_) {
        if (worker->active.load()) {
            count++;
        }
    }
    return count;
}

bool MultiGPUManager::allWorkersReady() const
{
    for (const auto &worker : workers_) {
        if (!worker->active.load()) {
            return false;
        }
    }
    return true;
}
