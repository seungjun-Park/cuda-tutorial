#pragma once

#include <cuda_runtime.h>

static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const* const func, const char* const file,
    int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void CudaMalloc(void** dst, size_t size)
{
    checkCudaErrors(cudaMalloc(dst, size));
}

void CudaMemcpyHostToDevice(void* dst, const void* src, size_t size)
{
    checkCudaErrors(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void CudaMemcpyDeviceToHost(void* dst, const void* src, size_t size)
{
    checkCudaErrors(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}