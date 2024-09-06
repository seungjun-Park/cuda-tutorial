#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#include "Conv.h"
#include "GPUInfo.h"


void Conv2dCPU(const Matrix& m, const Matrix& k, Matrix& out)
{
	int kW = k.GetWidth();
	int mW = m.GetWidth();
	int oW = out.GetWidth();

	for (size_t row = 0; row < out.GetHeight(); row++)
	{
		for (size_t col = 0; col < out.GetWidth(); col++)
		{
			float outValue = 0;
			for (size_t kRow = 0; kRow < k.GetHeight(); kRow++)
			{
				for (size_t kCol = 0; kCol < k.GetWidth(); kCol++)
				{
					outValue += k[kRow * k.GetWidth() + kCol] * m[(row + kRow) * m.GetWidth() + (col + kCol)];
				}
			}
			out[row * oW + col] = outValue;
		}
	}
}

__global__ void Conv2dGPUKernel(
    const MatrixD m, 
    const MatrixD k, 
    MatrixD out
);
__global__ void Conv2dGPUKernelWithSharedMemory(
    const MatrixD m, 
    const MatrixD k,
    MatrixD out,
    size_t blockSize,
    size_t blockWidth
);

void Conv2dGPU(Matrix& m, Matrix& k, Matrix& out)
{
    // CPU -> GPU Memory allocate
    m.ToDevice();
    k.ToDevice();
    out.ToDevice();
    MatrixD d_m;
    MatrixD d_k;
    MatrixD d_out;

    d_m.width = m.GetWidth();
    d_m.height = m.GetHeight();
    d_m.elements = m.GetElements();

    d_k.width = k.GetWidth();
    d_k.height = k.GetHeight();
    d_k.elements = k.GetElements();

    d_out.width = out.GetWidth();
    d_out.height = out.GetHeight();
    d_out.elements = out.GetElements();

    // Load GPU info
    GPUInfo info = GPUInfo(false);
    const int deviceCount = info.GetDeviceCount();
    const cudaDeviceProp* deviceProps = info.GetDeviceProps();

    // Invoke kernel
    size_t outputSize = d_out.height * d_out.width;
    size_t numThreads = std::min(outputSize, (size_t)deviceProps[0].maxThreadsPerBlock);
    dim3 dimBlock(numThreads);
    size_t gridSize = std::ceil((float)outputSize / numThreads);
    gridSize = std::min((int)gridSize, deviceProps[0].maxGridSize[0]);
    dim3 dimGrid(gridSize);

    std::cout << "outputSize: " << outputSize << ", numThreads: " << numThreads << ", gridSize: " << gridSize << std::endl;

    auto startTime = std::chrono::system_clock::now();
    Conv2dGPUKernel<<<dimGrid, dimBlock>>>(
        d_m,
        d_k,
        d_out
        );

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    // GPU -> CPU Memory allocate
    m.ToHost();
    k.ToHost();
    out.ToHost();
}

void Conv2dGPUWithSharedMemory(Matrix& m, Matrix& k, Matrix& out)
{
    // CPU -> GPU Memory allocate
    m.ToDevice();
    k.ToDevice();
    out.ToDevice();
    MatrixD d_m;
    MatrixD d_k;
    MatrixD d_out;

    d_m.width = m.GetWidth();
    d_m.height = m.GetHeight();
    d_m.elements = m.GetElements();

    d_k.width = k.GetWidth();
    d_k.height = k.GetHeight();
    d_k.elements = k.GetElements();

    d_out.width = out.GetWidth();
    d_out.height = out.GetHeight();
    d_out.elements = out.GetElements();

    // Load GPU info
    GPUInfo info = GPUInfo(false);
    const int deviceCount = info.GetDeviceCount();
    const cudaDeviceProp* deviceProps = info.GetDeviceProps();

    // Invoke kernel
    size_t outputSize = d_out.height * d_out.width;
    size_t numThreads = std::min(outputSize, (size_t)deviceProps[0].maxThreadsPerBlock);
    dim3 dimBlock(numThreads);
    size_t gridSize = std::ceil((float)outputSize / numThreads);
    gridSize = std::min((int)gridSize, deviceProps[0].maxGridSize[0]);
    dim3 dimGrid(gridSize);

    std::cout << "outputSize: " << outputSize << ", numThreads: " << numThreads << ", gridSize: " << gridSize << std::endl;

    auto startTime = std::chrono::system_clock::now();
    Conv2dGPUKernel << <dimGrid, dimBlock >> > (
        d_m,
        d_k,
        d_out
        );

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    // GPU -> CPU Memory allocate
    m.ToHost();
    k.ToHost();
    out.ToHost();
}

__global__ void Conv2dGPUKernel(
    const MatrixD m, 
    const MatrixD k, 
    MatrixD out
)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= out.width * out.height)
	{
		return;
	}
	
	int row = idx / out.width;
	int col = idx % out.width;

	float outValue = 0.f;

	for (size_t kRow = 0; kRow < k.height; kRow++)
	{
		for (size_t kCol = 0; kCol < k.width; kCol++)
		{
			outValue += k[kRow * k.width + kCol] * m[(row + kRow) * m.width + (col + kCol)];
		}
	}
	out[row * out.width + col] = outValue;
}

__global__ void Conv2dGPUKernelWithSharedMemory(
    const MatrixD m,
    const MatrixD k, 
    MatrixD out,
    size_t blockSize,
    size_t kSize,
    size_t blockWidth
)
{
    extern __shared__ float sharedMemory[];
    
    float outValue = 0.f;
    // Shared memory used to store Asub and Bsub respectively
    float* ms = (float*)sharedMemory;
    float* ks = (float*)(sharedMemory + blockSize * blockSize);
    
    int blockRow = blockIdx.x / blockWidth;
    int blockCol = blockIdx.x % blockWidth;
    int row = threadIdx.x / blockSize;
    int col = threadIdx.x % blockSize;

    int outSubIdx = (blockRow * out.width + blockCol) * blockSize;
    int i = 0;
    //while (curAWidth > 0)
    //{
    //    blockSize = (curAWidth > blockSize) ? blockSize : curAWidth;

    //    int mSubIdx;
    //    int kSubIdx;

    //    ms[row * blockSize + col] = m[mSubIdx + row * m.width + col];
    //    ks[row * blockSize + col] = k[kSubIdx + row * k.width + col];

    //    __syncthreads();
    //    // Multiply Asub and Bsub together
    //    for (int e = 0; e < blockSize; ++e)
    //    {
    //        outValue += ms[row * blockSize + e] * ks[e * blockSize + col];
    //    }
    //    // Synchronize to make sure that the preceding
    //    // computation is done before loading two new
    //    // sub-matrices of A and B in the next iteration
    //    __syncthreads();
    //    curAWidth -= blockSize;
    //    m++;
    //}
    //out[outSubIdx + row * out.width + col] = outValue;
}