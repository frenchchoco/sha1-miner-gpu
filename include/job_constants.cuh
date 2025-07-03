#pragma once
#include <stdint.h>

/* One definition lives in kernel.cu; everywhere else sees `extern` */
extern __device__ __constant__ uint8_t g_job_msg[32];
extern __device__ __constant__ uint32_t g_target[5];
