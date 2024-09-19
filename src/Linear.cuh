#pragma once
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include <device_launch_parameters.h>

#include "Common.h"
#include "GPUInfo.h"
#include "Matrix.cuh"

//////////////////////////////////		Declaration		//////////////////////////////////

typedef struct Linear
{
public:
    __host__ Linear(int inFeatures, int outFeatures, bool useBias = true)
    {
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        wSize = inFeatures * outFeatures;
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::normal_distribution<float> nd(0, 1);

        weight = new float[wSize];
        for (size_t i = 0; i < wSize; i++)
        {
            weight[i] = nd(gen);
        }

        if (useBias)
        {
            bias = new float[outFeatures];
            bSize = outFeatures;
            for (size_t i = 0; i < outFeatures; i++)
            {
                bias[i] = nd(gen);
            }
        }
    }

public:
    int inFeatures;
    int outFeatures;
	float* weight = nullptr;
    int wSize = 1;
    float* bias = nullptr;
    int bSize = 0;
};

template<int dim>
void LinearCPU(const Matrix<dim>& inp, const Linear& k, Matrix<dim>& out)
{
    assert(inp.shape[dim - 1] == k.inFeatures && dim > 1);

    int curShape[dim];

    int inDiv;
    int inIdx;
    int curDiv;

    auto startTime = std::chrono::system_clock::now();
    for (size_t idx = 0; idx < out.size; idx++)
    {
        inIdx = 0;
        inDiv = 1;
        curDiv = 1;

        for (int i = dim - 1; i >= 0; i--)
        {
            curShape[i] = (idx / curDiv) % out.shape[i];
            if (i < dim - 2)
            {
                inIdx += curShape[i] * inDiv;
            }
            curDiv *= out.shape[i];
            inDiv *= inp.shape[i];
        }

        float value = 0.f;
        if (k.bias != nullptr)
        {
            value += k.bias[curShape[dim - 1]];
        }
        for (size_t e = 0; e < inp.shape[dim - 1]; e++)
        {
            value += inp[inIdx + curShape[dim - 2] * inp.shape[dim - 1] + e] * k.weight[e * k.outFeatures + curShape[dim - 1]];
        }
        out[idx] = value;
    }
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;
}

template<int dim>
__global__ void LinearGPUKernel(const Matrix<dim> inp, const Linear k, Matrix<dim> out);

template<int dim>
void LinearGPU(const Matrix<dim>& inp, const Linear& k, Matrix<dim>& out)
{
    assert(inp.shape[dim - 1] == k.inFeatures && dim > 1);

    // Load GPU info
    GPUInfo info = GPUInfo(false);
    const int deviceCount = info.GetDeviceCount();
    const cudaDeviceProp* deviceProps = info.GetDeviceProps();

    size_t outputSize = out.size;
    size_t numThreads = std::min((int)outputSize, deviceProps[0].maxThreadsPerBlock);
    dim3 dimBlock(numThreads);
    size_t gridSize = std::ceil((float)outputSize / numThreads);
    gridSize = std::min((int)gridSize, deviceProps[0].maxGridSize[0]);
    dim3 dimGrid(gridSize);
    std::cout << "outputSize: " << outputSize << ", numThreads: " << numThreads << ", gridSize: " << gridSize << std::endl;

    Matrix<dim> d_inp(inp);
    Linear d_k(k);
    Matrix<dim> d_out(out);

    CudaMalloc((void**)&d_inp.elements, inp.size * sizeof(float));
    CudaMalloc((void**)&d_k.weight, k.wSize * sizeof(float));
    CudaMalloc((void**)&d_out.elements, out.size * sizeof(float));

    CudaMemcpyHostToDevice(d_inp.elements, inp.elements, inp.size * sizeof(float));
    CudaMemcpyHostToDevice(d_k.weight, k.weight, k.wSize * sizeof(float));
    CudaMemcpyHostToDevice(d_out.elements, out.elements, out.size * sizeof(float));

    if (k.bias != nullptr)
    {
        CudaMalloc((void**)&d_k.bias, k.bSize * sizeof(float));
        CudaMemcpyHostToDevice(d_k.bias, k.bias, k.bSize * sizeof(float));
    }

    auto startTime = std::chrono::system_clock::now();
    LinearGPUKernel<<<dimGrid, dimBlock>>>(
        d_inp,
        d_k,
        d_out
    );

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    CudaMemcpyDeviceToHost(out.elements, d_out.elements, out.size * sizeof(float));

    cudaFree(d_inp.elements);
    cudaFree(d_k.weight);
    cudaFree(d_out.elements);

    if (k.bias != nullptr)
    {
        cudaFree(d_k.bias);
    }
}

template<int dim>
__global__ void LinearGPUKernel(const Matrix<dim> inp, const Linear k, Matrix<dim> out)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= out.size)
    {
        return;
    }

    int curShape[dim];

    int inIdx = 0;
    int inDiv = 1;
    int curDiv = 1;

    for (int i = dim - 1; i >= 0; i--)
    {
        curShape[i] = (idx / curDiv) % out.shape[i];
        if (i < dim - 2)
        {
            inIdx += curShape[i] * inDiv;
        }
        curDiv *= out.shape[i];
        inDiv *= inp.shape[i];
    }

    float value = 0.f;
    if (k.bias != nullptr)
    {
        value += k.bias[curShape[dim - 1]];
    }
    for (size_t e = 0; e < inp.shape[dim - 1]; e++)
    {
        value += inp[inIdx + curShape[dim - 2] * inp.shape[dim - 1] + e] * k.weight[e * k.outFeatures + curShape[dim - 1]];
    }
    out[idx] = value;
}

template<int dim>
__global__ void LinearGPUWithSharedMemoryKernel(const Matrix<dim> inp, const Linear k, Matrix<dim> out, size_t sizePerBlock, bool useCPU);

template<int dim>
void LinearGPUWithSharedMemory(const Matrix<dim>& inp, const Linear& k, Matrix<dim>& out)
{
    assert(inp.shape[dim - 1] == k.inFeatures && dim > 1);

    // Load GPU info
    GPUInfo info = GPUInfo(false);
    const int deviceCount = info.GetDeviceCount();
    const cudaDeviceProp* deviceProps = info.GetDeviceProps();

    // Invoke kernel
    size_t outputSize = out.size;
    size_t sharedMemSize = deviceProps[0].sharedMemPerBlock / sizeof(float);
    size_t maxBlockSize = std::floor(std::sqrt(std::min((size_t)deviceProps[0].maxThreadsPerBlock, sharedMemSize / 3)));
    size_t blockSize = std::_Gcd(out.shape[dim - 1], out.shape[dim - 2]);

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
    Matrix<dim> d_inp(inp);
    Linear d_k(k);
    Matrix<dim> d_out(out);

    CudaMalloc((void**)&d_inp.elements, inp.size * sizeof(float));
    CudaMalloc((void**)&d_k.weight, k.wSize * sizeof(float));
    CudaMalloc((void**)&d_out.elements, out.size * sizeof(float));

    CudaMemcpyHostToDevice(d_inp.elements, inp.elements, inp.size * sizeof(float));
    CudaMemcpyHostToDevice(d_k.weight, k.weight, k.wSize * sizeof(float));
    CudaMemcpyHostToDevice(d_out.elements, out.elements, out.size * sizeof(float));

    if (k.bias != nullptr)
    {
        CudaMalloc((void**)&d_k.bias, k.bSize * sizeof(float));
        CudaMemcpyHostToDevice(d_k.bias, k.bias, k.bSize * sizeof(float));
    }

    auto startTime = std::chrono::system_clock::now();
    LinearGPUWithSharedMemoryKernel << <dimGrid, dimBlock, numThreads * 3 * sizeof(float) >> > (
        d_inp,
        d_k,
        d_out,
        blockSize,
        (k.bias != nullptr)
        );

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    // GPU -> CPU Memory allocate
    CudaMemcpyDeviceToHost(out.elements, d_out.elements, out.size * sizeof(float));

    cudaFree(d_inp.elements);
    cudaFree(d_k.weight);
    cudaFree(d_out.elements);

    if (k.bias != nullptr)
    {
        cudaFree(d_k.bias);
    }
}

template<int dim>
__global__ void LinearGPUWithSharedMemoryKernel(
    const Matrix<dim> inp,
    const Linear k, 
    Matrix<dim> out, 
    size_t sizePerBlock,
    bool useBias
)
{
    extern __shared__ float sharedMemory[];

    // To reduce global memory latency
    int blockShape[dim];
    int curBlockShape[dim];

    for (size_t i = 0; i < dim; i++)
    {
        blockShape[i] = out.shape[i];
        if (i >= dim - 2)
        {
            blockShape[i] /= sizePerBlock;
        }
    }

    int inpDiv = 1;
    int outDiv = 1;
    int curDiv = 1;

    int inpIdx = 0;
    int outIdx = 0;

    for (int i = dim - 1; i >= 0; i--)
    {
        curBlockShape[i] = (blockIdx.x / curDiv) % blockShape[i];
        if (i < dim - 2)
        {
            inpIdx += curBlockShape[i] * inpDiv;
            outIdx += curBlockShape[i] * outDiv;
        }
        curDiv *= blockShape[i];
        inpDiv *= inp.shape[i];
        outDiv *= out.shape[i];
    }

    float value = 0.f;

    int row = threadIdx.x / sizePerBlock;
    int col = threadIdx.x % sizePerBlock;

    outIdx += (curBlockShape[dim - 2] * out.shape[dim - 1] + curBlockShape[dim - 1]) * sizePerBlock;

    int e = inp.shape[dim - 1];
    int blockSize;

    int inpSubIdx = curBlockShape[dim - 2] * inp.shape[dim - 1] * sizePerBlock;
    int kSubIdx = curBlockShape[dim - 1] * sizePerBlock;
    while (e > 0)
    {
        blockSize = (e >= sizePerBlock) ? sizePerBlock : e;

        // Shared memory used to store Asub and Bsub respectively
        float* inps = (float*)(sharedMemory);
        float* ks = (float*)(sharedMemory + (blockSize * sizePerBlock));

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        if (col < blockSize)
        {
            inps[row * blockSize + col] = inp[inpIdx + inpSubIdx + row * inp.shape[dim - 1] + col];
        }
        if (row < blockSize)
        {
            ks[row * sizePerBlock + col] = k.weight[kSubIdx + row * k.outFeatures + col];
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int i = 0; i < blockSize; i++)
        {
            value += inps[row * blockSize + i] * ks[i * sizePerBlock + col];
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        e -= blockSize;
        inpSubIdx += blockSize;
        kSubIdx += (k.outFeatures * blockSize);
        __syncthreads();
    }

    if (useBias)
    {
        int biasIdx = curBlockShape[dim - 1] * sizePerBlock;
        float* bias = (float*)(sharedMemory);
        bias[col] = k.bias[biasIdx + col];

        __syncthreads();

        value += bias[col];
    }

    out[outIdx + row * out.shape[dim - 1] + col] = value;
}

void LinearTest()
{
    constexpr int dim = 4;

    Matrix<dim> in = Randn<dim>({2, 8, 8, 512}, true);
    Linear k(512, 512, true);
    Matrix<dim> outCPU = Zeros<dim>({ 2, 8, 8, 512 }, true);
    Matrix<dim> outGPU = Zeros<dim>({ 2, 8, 8, 512 }, true);
    Matrix<dim> outGPUSM = Zeros<dim>({ 2, 8, 8, 512 }, true);

    LinearCPU(in, k, outCPU);
    LinearGPU(in, k, outGPU);
    LinearGPUWithSharedMemory(in, k, outGPUSM);

    std::cout << (outCPU == outGPU) << std::endl;
    std::cout << (outCPU == outGPUSM) << std::endl;

    return;
}