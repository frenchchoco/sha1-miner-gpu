#include "include/multi_gpu_manager.hpp"
#include "logging/logger.hpp"
#include "core/gpu_api.h"
#include "include/miner/kernel_launcher.hpp"
#include "core/mining_system.hpp"
#ifdef USE_HIP
#include "architecture/gpu_architecture.hpp"
#endif
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <thread>

#include "config_utils.hpp"

// Define a safe sleep duration to prevent a hard spin
constexpr std::chrono::milliseconds WORKER_THREAD_SLEEP_MS = std::chrono::milliseconds(5);
constexpr std::chrono::seconds MONITOR_THREAD_SLEEP_S = std::chrono::seconds(5);

MultiGPUManager::MultiGPUManager()
{
    start_time_ = std::chrono::steady_clock::now();
}

MultiGPUManager::~MultiGPUManager()
{
    stopMining();
}

// Private helper to print GPU properties and check for compatibility issues
bool MultiGPUManager::logAndCheckGpu(int gpu_id, const gpuDeviceProp &props)
{
    gpuError_t err = gpuSetDevice(gpu_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "Failed to set device context for GPU ", gpu_id, ": ", gpuGetErrorString(err));
        return false;
    }

    std::cout << "\nInitializing GPU " << gpu_id << ": " << props.name << "\n";
#ifdef USE_HIP
    // AMD-specific initialization
    AMDArchitecture arch = AMDGPUDetector::detectArchitecture(props);
    AMDArchParams arch_params = AMDGPUDetector::getArchitectureParams(arch, props);

    std::cout << "  Architecture: " << AMDGPUDetector::getArchitectureName(arch) << " (" << props.gcnArchName << ")\n";
    std::cout << "  Wave size: " << arch_params.wave_size << "\n";
    std::cout << "  Max waves per CU: " << arch_params.waves_per_cu << "\n";
    std::cout << "  Compute units: " << props.multiProcessorCount << "\n";

    if (AMDGPUDetector::hasKnownIssues(arch, props.name)) {
        LOG_WARN("MULTI_GPU", "GPU ", gpu_id, " has known compatibility issues.");
        int version;
        if (hipRuntimeGetVersion(&version) == hipSuccess) {
            int major = version / 10000000;
            int minor = (version % 10000000) / 100000;
            int patch = (version % 100000) / 100;
            std::cout << "Current ROCm version: " << major << "." << minor << "." << patch << "\n";

            if (version < 50700000 && arch == AMDArchitecture::RDNA3) {
                LOG_ERROR("MULTI_GPU", "RDNA3 requires ROCm 5.7 or later. Skipping GPU ", gpu_id);
                return false;
            }
        }
    }
#endif
    return true;
}

// Private helper to create and initialize the MiningSystem for a worker
bool MultiGPUManager::createWorker(int gpu_id)
{
    gpuError_t err = gpuSetDevice(gpu_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "Failed to set device context for GPU ", gpu_id, ": ", gpuGetErrorString(err));
        return false;
    }

    auto worker = std::make_unique<GPUWorker>();
    worker->device_id = gpu_id;

    gpuDeviceProp props;
    err = gpuGetDeviceProperties(&props, gpu_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "Failed to get properties for GPU ", gpu_id, ": ", gpuGetErrorString(err));
        return false;
    }

    if (!logAndCheckGpu(gpu_id, props)) {
        return false;
    }

    // Create mining system configuration
    MiningSystem::Config config;
    config.device_id = gpu_id;

    // Apply user config if available
    if (user_config_) {
        auto mining_config = static_cast<const MiningConfig *>(user_config_);
        ConfigUtils::applyMiningConfig(*mining_config, config);
    } else {
        config.num_streams = 8;
        config.threads_per_block = DEFAULT_THREADS_PER_BLOCK;
        config.use_pinned_memory = true;
        config.result_buffer_size = 1024;
        config.blocks_per_stream = 1024; // Default value, adjust as needed
    }

    // Create and initialize mining system
    worker->mining_system = std::make_unique<MiningSystem>(config);

    if (user_config_) {
        worker->mining_system->setUserConfig(user_config_);
    }

    try {
        if (!worker->mining_system->initialize()) {
            LOG_ERROR("MULTI_GPU", "Failed to initialize GPU ", gpu_id);
            return false;
        }
    } catch (const std::exception &e) {
        LOG_ERROR("MULTI_GPU", "Exception initializing GPU ", gpu_id, ": ", e.what());
        return false;
    }

    workers_.push_back(std::move(worker));
    LOG_INFO("MULTI_GPU", "Successfully initialized GPU ", gpu_id);
    return true;
}

bool MultiGPUManager::initialize(const std::vector<int> &gpu_ids)
{
    std::cout << "\nInitializing Multi-GPU Mining System\n";
    std::cout << "=====================================\n";

    for (int gpu_id : gpu_ids) {
        if (!createWorker(gpu_id)) {
            LOG_ERROR("MULTI_GPU", "Failed to create worker for GPU ", gpu_id);
        }
    }

    if (workers_.empty()) {
        LOG_ERROR("MULTI_GPU", "No GPUs were successfully initialized");
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

    shutdown_ = true;

    for (auto &worker : workers_) {
        if (worker->mining_system) {
            gpuError_t err = gpuSetDevice(worker->device_id);
            if (err != gpuSuccess) {
                LOG_ERROR("MULTI_GPU", "Failed to set device context for GPU ", worker->device_id, ": ", gpuGetErrorString(err));
                continue;
            }
            worker->mining_system->stopMining();
        }
    }

    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }

    waitForWorkers();

    LOG_INFO("MULTI_GPU", "All mining operations stopped");
}

void MultiGPUManager::waitForWorkers()
{
    for (auto &worker : workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            worker->worker_thread->join();
        }
        worker->active = false;
    }
}

uint64_t MultiGPUManager::getNextNonceBatch()
{
    return global_nonce_counter_.fetch_add(NONCE_BATCH_SIZE);
}

void MultiGPUManager::processWorkerResults(GPUWorker *worker, const std::vector<MiningResult> &results)
{
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "Failed to set device context for GPU ", worker->device_id, ": ", gpuGetErrorString(err));
        return;
    }

    worker->candidates_found += results.size();

    for (const auto &result : results) {
        if (result.matching_bits > worker->best_match_bits) {
            worker->best_match_bits = result.matching_bits;

            if (global_best_tracker_.isNewBest(result.matching_bits)) {
                std::lock_guard lock(stats_mutex_);
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

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

    std::lock_guard lock(callback_mutex_);
    if (result_callback_) {
        result_callback_(results);
    }
}

void MultiGPUManager::workerThread(GPUWorker *worker, const MiningJob &job)
{
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Failed to set device context: ", gpuGetErrorString(err));
        return;
    }

    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker thread started");
    worker->active = true;

    auto worker_callback = [this, worker](const std::vector<MiningResult> &results) {
        processWorkerResults(worker, results);
    };
    worker->mining_system->setResultCallback(worker_callback);

    int consecutive_errors = 0;
    const int max_consecutive_errors = 5;
    uint64_t current_nonce_base = getNextNonceBatch();
    uint64_t nonces_used_in_batch = 0;

    while (!shutdown_ && consecutive_errors < max_consecutive_errors) {
        try {
            MiningJob worker_job = job;
            worker_job.nonce_offset = current_nonce_base + nonces_used_in_batch;

            uint64_t hashes_this_round = worker->mining_system->runSingleBatch(worker_job);

            if (hashes_this_round == 0) {
                auto config = worker->mining_system->getConfig();
                hashes_this_round = static_cast<uint64_t>(config.blocks_per_stream) * config.threads_per_block * NONCES_PER_THREAD;
            }

            worker->hashes_computed += hashes_this_round;
            nonces_used_in_batch += hashes_this_round;

            if (nonces_used_in_batch >= NONCE_BATCH_SIZE) {
                current_nonce_base = getNextNonceBatch();
                nonces_used_in_batch = 0;
            }

            consecutive_errors = 0;
        } catch (const std::exception &e) {
            LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Error: ", e.what());
            consecutive_errors++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::this_thread::sleep_for(WORKER_THREAD_SLEEP_MS);
    }

    worker->active = false;
    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker thread finished");
}

void MultiGPUManager::workerThreadInterruptibleWithOffset(GPUWorker *worker, const MiningJob &job,
    std::function<bool()> should_continue, std::atomic<uint64_t> &shared_nonce_counter)
{
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Failed to set device context: ", gpuGetErrorString(err));
        return;
    }

    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker thread started (continuous nonce mode)");
    worker->active = true;

    auto worker_callback = [this, worker](const std::vector<MiningResult> &results) {
        processWorkerResults(worker, results);
    };
    worker->mining_system->setResultCallback(worker_callback);

    int consecutive_errors = 0;
    const int max_consecutive_errors = 5;
    const uint64_t gpu_batch_size = NONCE_BATCH_SIZE;

    while (!shutdown_ && consecutive_errors < max_consecutive_errors && should_continue()) {
        try {
            uint64_t batch_start = shared_nonce_counter.fetch_add(gpu_batch_size);

            LOG_TRACE("MULTI_GPU", "GPU ", worker->device_id, " - Got nonce batch starting at: ", batch_start);

            MiningJob worker_job = job;
            worker_job.nonce_offset = batch_start;

            uint64_t nonces_to_process = gpu_batch_size;
            uint64_t nonces_processed = 0;

            const uint64_t chunk_size = gpu_batch_size / 10;
            if (chunk_size == 0) {
                nonces_to_process = 1;
            }

            while (nonces_processed < nonces_to_process && should_continue() && !shutdown_) {
                worker_job.nonce_offset = batch_start + nonces_processed;
                uint64_t hashes_this_round = worker->mining_system->runSingleBatch(worker_job);

                if (hashes_this_round == 0) {
                    auto config = worker->mining_system->getConfig();
                    hashes_this_round = static_cast<uint64_t>(config.blocks_per_stream) * config.threads_per_block * NONCES_PER_THREAD;
                }

                worker->hashes_computed += hashes_this_round;
                nonces_processed += hashes_this_round;

                if (nonces_processed >= nonces_to_process) {
                    break;
                }
            }

            consecutive_errors = 0;
        } catch (const std::exception &e) {
            LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Error: ", e.what());
            consecutive_errors++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::this_thread::sleep_for(WORKER_THREAD_SLEEP_MS);
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

void MultiGPUManager::monitorThread(std::function<bool()> should_continue)
{
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_total_hashes = 0;

    while (!shutdown_ && should_continue()) {
        std::this_thread::sleep_for(MONITOR_THREAD_SLEEP_S);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        uint64_t total_hashes = 0;
        uint64_t total_candidates = 0;
        uint32_t best_bits = global_best_tracker_.getBestBits();

        std::vector<double> gpu_rates;
        for (const auto& worker : workers_) {
            if (!worker->active) continue;

            uint64_t gpu_hash_count = worker->hashes_computed.load();
            total_hashes += gpu_hash_count;
            total_candidates += worker->candidates_found.load();

            double gpu_rate = 0.0;
            if (elapsed.count() > 0) {
                gpu_rate = static_cast<double>(gpu_hash_count) / elapsed.count() / 1e9;
            }
            gpu_rates.push_back(gpu_rate);
        }

        uint64_t hash_diff = total_hashes - last_total_hashes;
        auto interval = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        double instant_rate = 0.0;
        double average_rate = 0.0;

        if (interval.count() > 0) {
            instant_rate = static_cast<double>(hash_diff) / interval.count() / 1e9;
        }
        if (elapsed.count() > 0) {
            average_rate = static_cast<double>(total_hashes) / elapsed.count() / 1e9;
        }

        std::cout << "\r[" << elapsed.count() << "s] "
                  << "Rate: " << std::fixed << std::setprecision(2)
                  << instant_rate << " GH/s"
                  << " (avg: " << average_rate << " GH/s) | "
                  << "Best: " << best_bits << " bits | "
                  << "GPUs: ";

        for (size_t i = 0; i < gpu_rates.size(); i++) {
            if (i > 0) std::cout << "+";
            std::cout << std::fixed << std::setprecision(1) << gpu_rates[i];
        }

        std::cout << " | Total: " << std::fixed << std::setprecision(3)
                  << static_cast<double>(total_hashes) / 1e12
                  << " TH" << std::flush;

        last_update = now;
        last_total_hashes = total_hashes;
    }
}

void MultiGPUManager::runMining(const MiningJob &job)
{
    current_difficulty_ = job.difficulty;
    shutdown_ = false;
    start_time_ = std::chrono::steady_clock::now();
    global_best_tracker_.reset();

    for (auto &worker : workers_) {
        gpuError_t err = gpuSetDevice(worker->device_id);
        if (err != gpuSuccess) {
            LOG_ERROR("MULTI_GPU", "Failed to set device context for GPU ", worker->device_id, ": ", gpuGetErrorString(err));
            continue;
        }
        worker->hashes_computed = 0;
        worker->candidates_found = 0;
        worker->best_match_bits = 0;
    }

    for (auto &worker : workers_) {
        worker->worker_thread = std::make_unique<std::thread>(&MultiGPUManager::workerThread, this moulding, worker.get(), job);
    }

    monitor_thread_ = std::make_unique<std::thread>(
        &MultiGPUManager::monitorThread, this, []() { return true; }
    );

    waitForWorkers();

    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }
}

void MultiGPUManager::runMiningInterruptibleWithOffset(const MiningJob &job, std::function<bool()> should_continue,
    std::atomic<uint64_t> &global_nonce_offset)
{
    current_difficulty_ = job.difficulty;
    shutdown_ = false;
    start_time_ = std::chrono::steady_clock::now();
    global_best_tracker_.reset();

    for (auto &worker : workers_) {
        gpuError_t err = gpuSetDevice(worker->device_id);
        if (err != gpuSuccess) {
            LOG_ERROR("MULTI_GPU", "Failed to set device context for GPU ", worker->device_id, ": ", gpuGetErrorString(err));
            continue;
        }
        worker->hashes_computed = 0;
        worker->candidates_found = 0;
        worker->best_match_bits = 0;
    }

    LOG_INFO("MULTI_GPU", "Starting multi-GPU mining from nonce offset: ", global_nonce_offset.load());

    for (auto &worker : workers_) {
        worker->worker_thread =
            std::make_unique<std::thread>(&MultiGPUManager::workerThreadInterruptibleWithOffset, this, worker.get(),
                                         job, should_continue, std::ref(global_nonce_offset));
    }

    monitor_thread_ = std::make_unique<std::thread>(
        &MultiGPUManager::monitorThread, this, should_continue
    );

    waitForWorkers();

    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }

    if (!should_continue()) {
        LOG_INFO("MULTI_GPU",
                 "External stop signal received - all workers stopped at nonce: ", global_nonce_offset.load());
    }
}

void MultiGPUManager::sync() const
{
    for (const auto &worker : workers_) {
        if (worker->mining_system && worker->active) {
            gpuError_t err = gpuSetDevice(worker->device_id);
            if (err != gpuSuccess) {
                LOG_ERROR("MULTI_GPU", "Failed to set device context for GPU ", worker->device_id, ": ", gpuGetErrorString(err));
                continue;
            }
            worker->mining_system->sync();
        }
    }
}

void MultiGPUManager::updateJobLive(const MiningJob &job, uint64_t job_version) const
{
    for (const auto &worker : workers_) {
        if (worker->mining_system && worker->active) {
            gpuError_t err = gpuSetDevice(worker->device_id);
            if (err != gpuSuccess) {
                LOG_ERROR("MULTI_GPU", "Failed to set device context for GPU ", worker->device_id, ": ", gpuGetErrorString(err));
                continue;
            }
            worker->mining_system->updateJobLive(job, job_version);
        }
    }
    LOG_INFO("MULTI_GPU", "Updated job on all active GPUs to version ", job_version);
}

double MultiGPUManager::getTotalHashRate() const
{
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);
    if (elapsed.count() == 0)
        return 0.0;
    uint64_t total_hashes = 0;
    for (const auto &worker : workers_) {
        total_hashes += worker->hashes_computed.load();
    }
    return static_cast<double>(total_hashes) / elapsed.count();
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
