#include <cuda_runtime.h>
#include "job_constants.cuh"
#include "job_upload_api.h"

extern "C" void upload_new_job(const uint8_t msg32[32],
                               const uint32_t digest[5]) {
    cudaMemcpyToSymbol(g_job_msg, msg32, 32, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_target, digest, 5 * 4, 0, cudaMemcpyHostToDevice);
}
