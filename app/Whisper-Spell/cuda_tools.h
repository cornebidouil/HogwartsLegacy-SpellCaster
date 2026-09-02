#pragma once

#include <cuda_runtime.h>

bool isCudaCompatible() {
    int cudaDeviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&cudaDeviceCount);
    if (cudaStatus == cudaSuccess && cudaDeviceCount > 0) {
        return true;
    }
    return false;
}