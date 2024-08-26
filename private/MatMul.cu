#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include <chrono>

#include "MatMul.cuh"

#define BLOCK_SIZE 16

__global__ void MatMulGPUKernel(
    float* elementsA,
    float* elementsB,
    float* elementsC,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    int widthC
);

__global__ void MatMulGPUKernelWithSharedMemory(
    float* elementsA,
    float* elementsB,
    float* elementsC,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    int widthC
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
    A.ConvertDevice();
    B.ConvertDevice();
    C.ConvertDevice();

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.GetWidth() / dimBlock.x + 1, A.GetHeight() / dimBlock.y + 1);

    auto startTime = std::chrono::system_clock::now();
    MatMulGPUKernel<<<dimGrid, dimBlock>>>(
        A.GetElements(), 
        B.GetElements(), 
        C.GetElements(), 
        A.GetWidth(), 
        A.GetHeight(),
        B.GetWidth(), 
        B.GetHeight(),
        C.GetWidth());

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    // GPU -> CPU Memory allocate
    C.ConvertDevice();
}

void MatMulGPUWithSharedMemory(Matrix& A, Matrix& B, Matrix& C)
{
    // CPU -> GPU Memory allocate
    A.ConvertDevice();
    B.ConvertDevice();
    C.ConvertDevice();

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.GetWidth() / dimBlock.x + 1, A.GetHeight() / dimBlock.y + 1);

    auto startTime = std::chrono::system_clock::now();
    MatMulGPUKernelWithSharedMemory<<<dimGrid, dimBlock>>>(
        A.GetElements(),
        B.GetElements(),
        C.GetElements(),
        A.GetWidth(),
        A.GetHeight(),
        B.GetWidth(),
        B.GetHeight(),
        C.GetWidth());

    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << duration.count() << "ms" << std::endl;

    // GPU -> CPU Memory allocate
    C.ConvertDevice();
}

__global__ void MatMulGPUKernel(
	float* elementsA,
	float* elementsB,
	float* elementsC,
	int widthA,
    int heightA,
	int widthB,
    int heightB,
	int widthC
)
{
	float CValue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= heightA || col >= widthB)
    {
        return;
    }

	for (int e = 0; e < widthA; ++e)
	{
		CValue += elementsA[row * widthA + e] * elementsB[e * widthB + col];
	}
	elementsC[row * widthC + col] = CValue;
}

__global__ void MatMulGPUKernelWithSharedMemory(
    float* elementsA,
    float* elementsB,
    float* elementsC,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    int widthC
)
{
    float CValue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= heightA || col >= widthB)
    {
        return;
    }

    for (int e = 0; e < widthA; ++e)
    {
        CValue += elementsA[row * widthA + e] * elementsB[e * widthB + col];
    }
    elementsC[row * widthC + col] = CValue;
}