#ifndef MULTI_GPU_MANAGER_HPP
#define MULTI_GPU_MANAGER_HPP

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "sha1_miner.cuh"
#include "core/mining_system.hpp"

// Callback type for multi-GPU results
using MiningResultCallback = std::function<void(const std::vector<MiningResult> &)>;

// Global batch size for nonce distribution across GPUs
constexpr uint64_t NONCE_BATCH_SIZE = 1ULL << 32;  // 4B nonces per batch

struct GPUWorker
{
    int device_id;
    std::unique_ptr<MiningSystem> mining_system;
    std::unique_ptr<std::thread> worker_thread;
    std::atomic<uint64_t> hashes_computed{0};
    std::atomic<uint64_t> candidates_found{0};
    std::atomic<uint32_t> best_match_bits{0};
    std::atomic<bool> active{false};
};

class MultiGPUManager final
{
public:
    MultiGPUManager();
    ~MultiGPUManager();

    void setUserConfig(const void *user_config) { user_config_ = user_config; }

    bool initialize(const std::vector<int> &gpu_ids);
    void setResultCallback(const MiningResultCallback &callback)
    {
        std::lock_guard lock(callback_mutex_);
        result_callback_ = callback;
    }
    void runMining(const MiningJob &job);
    void sync() const;
    void runMiningInterruptibleWithOffset(const MiningJob &job, std::function<bool()> should_continue,
                                         std::atomic<uint64_t> &global_nonce_offset);
    void stopMining();
    void updateJobLive(const MiningJob &job, uint64_t job_version) const;
    double getTotalHashRate() const;
    size_t getActiveWorkerCount() const;

private:
    const void *user_config_ = nullptr;
    std::vector<std::unique_ptr<GPUWorker>> workers_;
    std::atomic<bool> shutdown_{false};
    std::atomic<uint64_t> global_nonce_counter_{1};
    mutable std::chrono::steady_clock::time_point start_time_;
    mutable BestResultTracker global_best_tracker_;
    mutable std::mutex stats_mutex_;
    uint32_t current_difficulty_{0};
    mutable std::mutex callback_mutex_;
    MiningResultCallback result_callback_;
    std::unique_ptr<std::thread> monitor_thread_;
    void workerThread(GPUWorker *worker, const MiningJob &job);
    void workerThreadInterruptibleWithOffset(GPUWorker *worker, const MiningJob &job,
                                            std::function<bool()> should_continue,
                                            std::atomic<uint64_t> &shared_nonce_counter);
    uint64_t getNextNonceBatch();
    void processWorkerResults(GPUWorker *worker, const std::vector<MiningResult> &results);
    bool allWorkersReady() const;
    void waitForWorkers();
};

#endif  // MULTI_GPU_MANAGER_HPP
