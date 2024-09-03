#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include <chrono>
#include <cooperative_groups.h>

#include "MatMul.h"
#include "GPUInfo.h"

typedef struct MatrixD {
    int width;
    int height;
    float* elements;

public:
    __host__ __device__ float& operator[](int index)
    {
        return elements[index];
    }

    __host__ __device__ float operator[](int index) const
    {
        return elements[index];
    }
};

__global__ void MatMulGPUKernel(
    const MatrixD A,
    const MatrixD B,
    MatrixD C
);

__global__ void MatMulGPUKernelWithSharedMemory(
    MatrixD A,
    MatrixD B,
    MatrixD C,
    size_t sizePerBlock,
    size_t blockWidth
);

void MatMulCPU(const Matrix& A, const Matrix& B, Matrix& C)
{
    assert(A.GetWidth() == B.GetHeight(), "A.width should be equal to B.height!");
    auto startTime = std::chrono::system_clock::now();
    for (size_t row = 0; row < A.GetHeight(); row++)
    {
        for (size_t col = 0; col < B.GetWidth(); col++)
        {
            float cvalue = 0;
            for (size_t i = 0; i < B.GetHeight(); i++)
            {
                cvalue += A[row * A.GetWidth() + i] * B[i * B.GetWidth() + col];
            }
            C[row * C.GetWidth() + col] = cvalue;
        }
    }
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;
}

void MatMulGPU(Matrix& A, Matrix& B, Matrix& C)
{
    // CPU -> GPU Memory allocate
    A.ToDevice();
    B.ToDevice();
    C.ToDevice();
    MatrixD d_A;
    MatrixD d_B;
    MatrixD d_C;

    d_A.width = A.GetWidth();
    d_A.height = A.GetHeight();
    d_A.elements = A.GetElements();

    d_B.width = B.GetWidth();
    d_B.height = B.GetHeight();
    d_B.elements = B.GetElements();

    d_C.width = C.GetWidth();
    d_C.height = C.GetHeight();
    d_C.elements = C.GetElements();

    // Load GPU info
    GPUInfo info = GPUInfo(false);
    const int deviceCount = info.GetDeviceCount();
    const cudaDeviceProp* deviceProps = info.GetDeviceProps();

    // Invoke kernel
    size_t outputSize = C.GetHeight() * C.GetWidth();
    size_t numThreads = std::min((int)outputSize, deviceProps[0].maxThreadsPerBlock);
    dim3 dimBlock(numThreads);
    size_t gridSize = std::ceil((float)outputSize / numThreads);
    gridSize = std::min((int)gridSize, deviceProps[0].maxGridSize[0]);
    dim3 dimGrid(gridSize);
    std::cout << "outputSize: " << outputSize << ", numThreads: " << numThreads << ", gridSize: " << gridSize << std::endl;

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
    A.ToHost();
    B.ToHost();
    C.ToHost();
}

void MatMulGPUWithSharedMemory(Matrix& A, Matrix& B, Matrix& C)
{
    // CPU -> GPU Memory allocate
    A.ToDevice();
    B.ToDevice();
    C.ToDevice();
    MatrixD d_A;
    MatrixD d_B;
    MatrixD d_C;

    d_A.width = A.GetWidth();
    d_A.height = A.GetHeight();
    d_A.elements = A.GetElements();

    d_B.width = B.GetWidth();
    d_B.height = B.GetHeight();
    d_B.elements = B.GetElements();

    d_C.width = C.GetWidth();
    d_C.height = C.GetHeight();
    d_C.elements = C.GetElements();

    // Load GPU info
    GPUInfo info = GPUInfo(false);
    const int deviceCount = info.GetDeviceCount();
    const cudaDeviceProp* deviceProps = info.GetDeviceProps();

    // Invoke kernel
    size_t outputSize = d_C.height * d_C.width;
    size_t sharedMemSize = deviceProps[0].sharedMemPerBlock / sizeof(float);
    size_t maxBlockSize = std::floor(std::sqrt(std::min((size_t)deviceProps[0].maxThreadsPerBlock, sharedMemSize / 2)));
    size_t blockSize = std::_Gcd(d_C.height, d_C.width);
    
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
    size_t blockWidth = std::ceil((float)d_C.width / blockSize);

    std::cout << "outputSize: " << outputSize << ", numThreads: " << numThreads << ", gridSize: " << gridSize << ", blockSize: " << blockSize << std::endl;

    auto startTime = std::chrono::system_clock::now();
    MatMulGPUKernelWithSharedMemory<<<dimGrid, dimBlock, numThreads * 2 * sizeof(float)>>>(
        d_A,
        d_B,
        d_C,
        blockSize,
        blockWidth
    );

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    // GPU -> CPU Memory allocate
    A.ToHost();
    B.ToHost();
    C.ToHost();
}

__global__ void MatMulGPUKernel(
    const MatrixD A,
    const MatrixD B,
    MatrixD C
)
{
	float CValue = 0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C.height * C.width) 
        return;
    int row = idx / C.width;
    int col = idx % C.width;

	for (int e = 0; e < A.width; ++e)
	{
		CValue += A[row * A.width + e] * B[e * B.width + col];
	}
	C[row * C.width + col] = CValue;
}

__global__ void MatMulGPUKernelWithSharedMemory(
    MatrixD A,
    MatrixD B,
    MatrixD C,
    size_t blockSize,
    size_t blockWidth
)
{
    extern __shared__ float subsetMatrix[];
    // Shared memory used to store Asub and Bsub respectively
    float* As = (float*)subsetMatrix;
    float* Bs = (float*)(subsetMatrix + blockSize * blockSize);

    float CValue = 0;
    int blockRow = blockIdx.x / blockWidth;
    int blockCol = blockIdx.x % blockWidth;
    int row = threadIdx.x / blockSize;
    int col = threadIdx.x % blockSize;
    
    int CSubIdx = (blockRow * C.width + blockCol) * blockSize;
    int curAWidth = A.width;
    int m = 0;
    while (curAWidth > 0)
    {
        blockSize = (curAWidth > blockSize) ? blockSize : curAWidth;

        int ASubIdx = (blockRow * A.width + m) * blockSize;
        int BSubIdx = (m * B.width + blockCol) * blockSize;

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row * blockSize + col] = A[ASubIdx + row * A.width + col];
        Bs[row * blockSize + col] = B[BSubIdx + row * B.width + col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < blockSize; ++e)
        {
            CValue += As[row * blockSize + e] * Bs[e * blockSize + col];
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
        curAWidth -= blockSize;
        m++;
    }
    C[CSubIdx + row * C.width + col] = CValue;
}

void MatMulTest()
{
    Matrix A(1000, 1000, true);
    Matrix B(1000, 1000, true);
    Matrix C(1000, 1000, true);
    Matrix D(1000, 1000, true);
    Matrix E(1000, 1000, true);

    FillRandn(A);
    FillRandn(B);
    FillZeros(C);
    FillZeros(D);
    FillZeros(E);

    MatMulCPU(A, B, C);
    MatMulGPU(A, B, D);
    MatMulGPUWithSharedMemory(A, B, E);

    // std::cout << (C == D) << std::endl;
    std::cout << (D == E) << std::endl;

    return;
}