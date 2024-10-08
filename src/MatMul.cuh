#pragma once
#include <algorithm>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <iostream>
#include <cooperative_groups.h>

#include "Common.h"
#include "GPUInfo.h"
#include "Matrix.cuh"

template<int dim>
void MatMulCPU(const Matrix<dim>& A, const Matrix<dim>& B, Matrix<dim>& C)
{
    assert(dim > 1 && A.shape[dim - 1] == B.shape[dim - 2]);

    int aIdx = 0;
    int bIdx = 0;
    int curShape[dim];

    int aDiv = 1;
    int bDiv = 1;
    int curDiv = 1;

    auto startTime = std::chrono::system_clock::now();

    for (size_t idx = 0; idx < C.size; idx++)
    {
        aIdx = 0;
        bIdx = 0;

        aDiv = 1;
        bDiv = 1;
        curDiv = 1;

        for (int i = dim - 1; i >= 0; i--)
        {
            curShape[i] = (idx / curDiv) % C.shape[i];
            if (i < dim - 2)
            {
                aIdx += curShape[i] * aDiv;
                bIdx += curShape[i] * bDiv;
            }
            curDiv *= C.shape[i];
            aDiv *= A.shape[i];
            bDiv *= B.shape[i];
        }

        float value = 0.f;
        for (size_t e = 0; e < A.shape[dim - 1]; e++)
        {
            value += A[aIdx + curShape[dim - 2] * A.shape[dim - 1] + e] * B[bIdx + e * B.shape[dim - 1] + curShape[dim - 1]];
        }
        C[idx] = value;
    }
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;
}

template<int dim>
__global__ void MatMulGPUKernel(
    const Matrix<dim> A,
    const Matrix<dim> B,
    Matrix<dim> C
);

template<int dim>
void MatMulGPU(const Matrix<dim>& A, const Matrix<dim>& B, Matrix<dim>& C)
{
    // Load GPU info
    GPUInfo info = GPUInfo(false);
    const int deviceCount = info.GetDeviceCount();
    const cudaDeviceProp* deviceProps = info.GetDeviceProps();

    size_t outputSize = C.size;
    size_t numThreads = std::min((int)outputSize, deviceProps[0].maxThreadsPerBlock);
    dim3 dimBlock(numThreads);
    size_t gridSize = std::ceil((float)outputSize / numThreads);
    gridSize = std::min((int)gridSize, deviceProps[0].maxGridSize[0]);
    dim3 dimGrid(gridSize);
    std::cout << "outputSize: " << outputSize << ", numThreads: " << numThreads << ", gridSize: " << gridSize << std::endl;

    // CPU -> GPU Memory allocate
    Matrix<dim> d_A(A);
    Matrix<dim> d_B(B);
    Matrix<dim> d_C(C);

    CudaMalloc((void**)&d_A.elements, A.size * sizeof(float));
    CudaMalloc((void**)&d_B.elements, B.size * sizeof(float));
    CudaMalloc((void**)&d_C.elements, C.size * sizeof(float));

    CudaMemcpyHostToDevice(d_A.elements, A.elements, A.size * sizeof(float));
    CudaMemcpyHostToDevice(d_B.elements, B.elements, B.size * sizeof(float));
    CudaMemcpyHostToDevice(d_C.elements, C.elements, C.size * sizeof(float));

    auto startTime = std::chrono::system_clock::now();
    MatMulGPUKernel<<<dimGrid, dimBlock>>>(
        d_A,
        d_B,
        d_C
    );

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    // GPU -> CPU Memory allocate
    
    CudaMemcpyDeviceToHost(C.elements, d_C.elements, C.size * sizeof(float));

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

template<int dim>
__global__ void MatMulGPUKernel(
    const Matrix<dim> A,
    const Matrix<dim> B,
    Matrix<dim> C
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C.size)
        return;

    int curShape[dim];

    int aIdx = 0;
    int bIdx = 0;

    int aDiv = 1;
    int bDiv = 1;
    int curDiv = 1;

    for (int i = dim - 1; i >= 0; i--)
    {
        curShape[i] = idx / curDiv % C.shape[i];
        if (i < dim - 2)
        {
            aIdx += curShape[i] * aDiv;
            bIdx += curShape[i] * bDiv;
        }
        curDiv *= C.shape[i];
        aDiv *= A.shape[i];
        bDiv *= B.shape[i];
    }

    float value = 0.f;
    for (int e = 0; e < A.shape[dim - 1]; ++e)
    {
        value += A[aIdx + curShape[dim - 2] * A.shape[dim - 1] + e] * B[bIdx + e * B.shape[dim - 1] + curShape[dim - 1]];
    }
    C[idx] = value;
}

template<int dim>
__global__ void MatMulGPUKernelWithSharedMemory(
    const Matrix<dim> A,
    const Matrix<dim> B,
    Matrix<dim> C,
    size_t sizePerBlock
);


template<int dim>
void MatMulGPUWithSharedMemory(const Matrix<dim>& A, const Matrix<dim>& B, Matrix<dim>& C)
{
    // Load GPU info
    GPUInfo info = GPUInfo(false);
    const int deviceCount = info.GetDeviceCount();
    const cudaDeviceProp* deviceProps = info.GetDeviceProps();

    // Invoke kernel
    size_t outputSize = C.size;
    size_t sharedMemSize = deviceProps[0].sharedMemPerBlock / sizeof(float);
    size_t maxBlockSize = std::floor(std::sqrt(std::min((size_t)deviceProps[0].maxThreadsPerBlock, sharedMemSize / 2)));
    size_t blockSize = std::_Gcd(C.shape[dim - 1], C.shape[dim - 2]);

    if (blockSize > maxBlockSize)
    {
        for (size_t i = maxBlockSize; i > 0; i--)
        {
            if (blockSize % i == 0)
            {
                blockSize = i;
                break;
            }
        }
    }

    size_t numThreads = blockSize * blockSize;
    dim3 dimBlock(numThreads);
    size_t gridSize = std::ceil((float)outputSize / numThreads);
    gridSize = std::min((int)gridSize, deviceProps[0].maxGridSize[0]);
    dim3 dimGrid(gridSize);

    std::cout << "outputSize: " << outputSize << ", numThreads: " << numThreads << ", gridSize: " << gridSize << ", blockSize: " << blockSize << std::endl;

    // CPU -> GPU Memory allocate
    Matrix<dim> d_A(A);
    Matrix<dim> d_B(B);
    Matrix<dim> d_C(C);

    CudaMalloc((void**)&d_A.elements, A.size * sizeof(float));
    CudaMalloc((void**)&d_B.elements, B.size * sizeof(float));
    CudaMalloc((void**)&d_C.elements, C.size * sizeof(float));

    CudaMemcpyHostToDevice(d_A.elements, A.elements, A.size * sizeof(float));
    CudaMemcpyHostToDevice(d_B.elements, B.elements, B.size * sizeof(float));
    CudaMemcpyHostToDevice(d_C.elements, C.elements, C.size * sizeof(float));

    auto startTime = std::chrono::system_clock::now();
    MatMulGPUKernelWithSharedMemory<<<dimGrid, dimBlock, numThreads * 2 * sizeof(float)>>>(
        d_A,
        d_B,
        d_C,
        blockSize
    );

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    // GPU -> CPU Memory allocate
    CudaMemcpyDeviceToHost(C.elements, d_C.elements, C.size * sizeof(float));

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

template<int dim>
__global__ void MatMulGPUKernelWithSharedMemory(
    const Matrix<dim> A,
    const Matrix<dim> B,
    Matrix<dim> C,
    size_t sizePerBlock
)
{
    extern __shared__ float sharedMemory[];

    float value = 0.f;

    // To reduce global memory latency
    int blockShape[dim];
    int curBlockShape[dim];

    for (size_t i = 0; i < dim; i++)
    {
        blockShape[i] = C.shape[i];
        if (i >= dim - 2)
        {
            blockShape[i] /= sizePerBlock;
        }
    }

    int aDiv = 1;
    int bDiv = 1;
    int cDiv = 1;
    int curDiv = 1;

    int AIdx = 0;
    int BIdx = 0;
    int CIdx = 0;

    for (int i = dim - 1; i >= 0; i--)
    {
        curBlockShape[i] = (blockIdx.x / curDiv) % blockShape[i];
        if (i < dim - 2)
        {
            AIdx += curBlockShape[i] * aDiv;
            BIdx += curBlockShape[i] * bDiv;
            CIdx += curBlockShape[i] * cDiv;
        }
        curDiv *= blockShape[i];
        aDiv *= A.shape[i];
        bDiv *= B.shape[i];
        cDiv *= C.shape[i];
    }

    int row = threadIdx.x / sizePerBlock;
    int col = threadIdx.x % sizePerBlock;

    CIdx += (curBlockShape[dim - 2] * C.shape[dim - 1] + curBlockShape[dim - 1]) * sizePerBlock;

    int e = A.shape[dim - 1];
    int blockSize;

    int ASubIdx = curBlockShape[dim - 2] * A.shape[dim - 1] * sizePerBlock;
    int BSubIdx = curBlockShape[dim - 1] * sizePerBlock;

    while (e > 0)
    {
        blockSize = (e >= sizePerBlock) ? sizePerBlock : e;

        // Shared memory used to store Asub and Bsub respectively
        float* As = (float*)sharedMemory;
        float* Bs = (float*)(sharedMemory + (blockSize * sizePerBlock));

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        if (col < blockSize)
        {
            As[row * blockSize + col] = A[AIdx + ASubIdx + row * A.shape[dim - 1] + col];
        }
        if (row < blockSize)
        {
            Bs[row * sizePerBlock + col] = B[BIdx + BSubIdx + row * B.shape[dim - 1] + col];
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int i = 0; i < blockSize; i++)
        {
            value += As[row * blockSize + i] * Bs[i * sizePerBlock + col];
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        e -= blockSize;
        ASubIdx += blockSize;
        BSubIdx += (B.shape[dim - 1] * blockSize);
        __syncthreads();
    }

    C[CIdx + row * C.shape[dim - 1] + col] = value;
}

void MatMulTest()
{
    constexpr int dim = 3;

    Matrix<dim> A = Randn<dim>({ 1, 10, 10 }, true);
    Matrix<dim> B = Randn<dim>({ 1, 10, 10 }, true);
    Matrix<dim> C = Zeros<dim>({ 1, 10, 10 }, true);
    Matrix<dim> D = Zeros<dim>({ 1, 10, 10 }, true);
    Matrix<dim> E = Zeros<dim>({ 1, 10, 10 }, true);

    MatMulCPU(A, B, C);
    MatMulGPU(A, B, D);
    MatMulGPUWithSharedMemory(A, B, E);

    std::cout << (C == D) << std::endl;
    std::cout << (C == E) << std::endl;

    return;
}