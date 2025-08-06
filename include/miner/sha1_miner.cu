#include "sha1_miner.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <thread>

// Use the actual kernel from sha1_kernel.cu
extern void launch_mining_kernel_nvidia(const DeviceMiningJob &device_job, uint32_t difficulty,
                                       uint64_t nonce_offset, const ResultPool &pool,
                                       const KernelConfig &config, uint64_t job_version);

MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty) {
    MiningJob job;
    memcpy(job.base_message, message, 32);
    memcpy(job.target_hash, target_hash, 5 * sizeof(uint32_t));
    job.difficulty = difficulty;
    job.nonce_offset = 0;
    job.job_version = 1;
    return job;
}

void cleanup_mining_system() {
    cudaDeviceReset();
    fprintf(stderr, "[INFO] Cleaning up mining system\n");
}

void run_mining_loop(MiningJob job) {
    int num_gpus;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus < 1) {
        fprintf(stderr, "[ERROR] Failed to get GPU count: %s\n", cudaGetErrorString(err));
        return;
    }
    fprintf(stderr, "[INFO] Using %d GPUs\n", num_gpus);

    std::vector<DeviceMiningJob> device_jobs(num_gpus);
    std::vector<std::vector<cudaStream_t>> streams(num_gpus);
    std::vector<ResultPool> pools(num_gpus);
    std::vector<std::thread> worker_threads(num_gpus);

    for (int device = 0; device < num_gpus; device++) {
        err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ERROR] Failed to set device %d: %s\n", device, cudaGetErrorString(err));
            return;
        }
        fprintf(stderr, "[DEBUG] Set device to %d\n", device);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        fprintf(stderr, "[INFO] GPU %d: %s, Compute capability: %d.%d\n",
                device, prop.name, prop.major, prop.minor);

        if (!device_jobs[device].allocate()) {
            fprintf(stderr, "[ERROR] Failed to allocate device job for GPU %d\n", device);
            return;
        }
        device_jobs[device].copyFromHost(job);

        KernelConfig config;
        config.blocks = 1024;
        config.threads_per_block = 128;
        config.shared_memory_size = 0;
        config.streams = 7;

        streams[device].resize(config.streams);
        for (int s = 0; s < config.streams; s++) {
            err = cudaStreamCreate(&streams[device][s]);
            if (err != cudaSuccess) {
                fprintf(stderr, "[ERROR] Failed to create stream %d for GPU %d: %s\n",
                        s, device, cudaGetErrorString(err));
                return;
            }
            fprintf(stderr, "[DEBUG] Created stream %d for GPU %d: %p\n",
                    s, device, streams[device][s]);
        }

        pools[device].capacity = MAX_CANDIDATES_PER_BATCH;
        err = cudaMalloc(&pools[device].results, pools[device].capacity * sizeof(MiningResult));
        if (err != cudaSuccess) {
            fprintf(stderr, "[ERROR] Failed to allocate results for GPU %d: %s\n",
                    device, cudaGetErrorString(err));
            return;
        }
        err = cudaMalloc(&pools[device].count, sizeof(uint32_t));
        err = cudaMalloc(&pools[device].nonces_processed, sizeof(uint64_t));
        err = cudaMalloc(&pools[device].job_version, sizeof(uint64_t));
        cudaMemset(pools[device].count, 0, sizeof(uint32_t));
        cudaMemset(pools[device].nonces_processed, 0, sizeof(uint64_t));
        cudaMemset(pools[device].job_version, job.job_version, sizeof(uint64_t));
    }

    for (int device = 0; device < num_gpus; device++) {
        worker_threads[device] = std::thread([device, &device_jobs, &pools, &config, &job]() {
            cudaSetDevice(device);
            fprintf(stderr, "[INFO] GPU %d - Worker thread started\n", device);

            uint64_t nonce_offset = job.nonce_offset + (uint64_t)device * 4294967296ULL;
            for (int s = 0; s < config.streams; s++) {
                config.stream = streams[device][s];
                launch_mining_kernel_nvidia(device_jobs[device], job.difficulty, nonce_offset,
                                            pools[device], config, job.job_version);
                cudaError_t err = cudaStreamSynchronize(streams[device][s]);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[ERROR] Stream synchronization failed on GPU %d, stream %d: %s\n",
                            device, s, cudaGetErrorString(err));
                }
                nonce_offset += config.blocks * config.threads_per_block * NONCES_PER_THREAD;
            }
        });
    }

    for (int device = 0; device < num_gpus; device++) {
        worker_threads[device].join();
    }

    for (int device = 0; device < num_gpus; device++) {
        cudaSetDevice(device);
        for (int s = 0; s < streams[device].size(); s++) {
            cudaStreamDestroy(streams[device][s]);
        }
        device_jobs[device].free();
        cudaFree(pools[device].results);
        cudaFree(pools[device].count);
        cudaFree(pools[device].nonces_processed);
        cudaFree(pools[device].job_version);
    }
}

int main() {
    uint8_t message[32] = {0};
    uint8_t target[20] = {0};
    MiningJob job = create_mining_job(message, target, 35);
    run_mining_loop(job);
    cleanup_mining_system();
    return 0;
}
