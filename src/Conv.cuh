#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Common.h"
#include "Matrix.cuh"

template<int dim>
struct ConvNd
{
public:
	ConvNd(
		int inChannels,
		int outChannels,
		int kernelSize,
		int stride = 1,
		int padding = 0,
		int dillation = 1,
		int groups = 1,
		bool useBias = true
	)
	{
		assert(
			inChannels % groups == 0 && 
			outChannels % groups == 0 &&
			stride > 0 && groups > 0 && dillation > 0 && padding >= 0 &&
			kernelSize > 0
		);

		this->inChannels = inChannels / groups;
		this->outChannels = outChannels;
		this->groups = groups;
		this->wSize = (inChannels / groups) * outChannels;
		this->kSize = 1;
		for (size_t i = 0; i < dim; i++)
		{
			this->kernelSize[i] = kernelSize;
			this->stride[i] = stride;
			this->padding[i] = padding;
			this->dillation[i] = dillation;
			this->wSize *= kernelSize;
			this->kSize *= kernelSize;
		}

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
			bias = new float[outChannels];
			bSize = outChannels;
			for (size_t i = 0; i < bSize; i++)
			{
				bias[i] = nd(gen);
			}
		}
	}

	ConvNd(
		int inChannels,
		int outChannels,
		std::vector<int> kernelSize,
		std::vector<int> stride,
		std::vector<int> padding,
		std::vector<int> dillation,
		int groups = 1,
		bool useBias = true)
	{
		assert(
			inChannels % groups == 0 &&
			outChannels % groups == 0 &&
			dim == kernelSize.size() &&
			dim == stride.size() &&
			dim == padding.size() &&
			dim == dillation.size() &&
			groups > 0
		);

		this->inChannels = inChannels / groups;
		this->outChannels = outChannels;
		this->groups = groups;
		this->wSize = (inChannels / groups) * outChannels;
		this->kSize = 1;
		for (size_t i = 0; i < dim; i++)
		{
			assert(
				stride[i] > 0 && dillation[i] > 0 && kernelSize[i] > 0 && padding[i] >= 0
			);

			this->kernelSize[i] = kernelSize[i];
			this->stride[i] = stride[i];
			this->padding[i] = padding[i];
			this->dillation[i] = dillation[i];
			this->wSize *= kernelSize[i];
			this->kSize *= kernelSize[i];
		}

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
			bias = new float[outChannels];
			bSize = outChannels;
			for (size_t i = 0; i < bSize; i++)
			{
				bias[i] = nd(gen);
			}
		}
	}

public:
	int inChannels;
	int outChannels;
	int kernelSize[dim];
	int kSize = 1;
	int stride[dim];
	int padding[dim];
	int dillation[dim];

	int groups;

	float* weight = nullptr;
	int wSize = 0;

	float* bias = nullptr;
	int bSize = 0;
};

typedef ConvNd<1> Conv1d;
typedef ConvNd<2> Conv2d;
typedef ConvNd<3> Conv3d;

template<int dim, int n>
void ConvNdCPU(Matrix<dim>& inp, const ConvNd<n>& k, Matrix<dim>& out)
{	
	assert(dim - n >= 1);
	constexpr int chIdx = dim - n - 1;
	
	bool usePadding = false;
	std::vector<int> pad;
	for (size_t i = 0; i < n; i++)
	{
		int v = k.padding[i];
		if (v != 0)
		{
			usePadding = true;
		}

		pad.push_back(v);
		pad.push_back(v);
	}

	if (usePadding)
	{
		Pad(inp, pad, 0.f);
	}

	int curInpShape[n];
	int curOutShape[dim];
	int curKShape[n];

	int curDiv;
	int curKDiv;
	int inpDiv;

	int inpIdx = 0;
	int inpKIdx = 0;
	int curGroups = 0;

	int baseKIdx;

	auto startTime = std::chrono::system_clock::now();

	for (size_t idx = 0; idx < out.size; idx++)
	{
		curDiv = 1;
		inpDiv = 1;
		inpIdx = 0;
		for (int i = dim - 1; i >= 0; i--)
		{
			curOutShape[i] = (idx / curDiv) % out.shape[i];
			if (i < chIdx)
			{
				inpIdx += curOutShape[i] * inpDiv;
			}
			curDiv *= out.shape[i];
			inpDiv *= inp.shape[i];
		}

		for (int i = n - 1; i >= 0; i--)
		{
			curInpShape[i] = k.stride[i] * (curOutShape[i - n + dim] - 1) + 1 + k.dillation[i] * (k.kernelSize[i] - 1) - 2 * k.padding[i];
		}

		curGroups = k.groups * curOutShape[chIdx] / out.shape[chIdx];
		baseKIdx = curOutShape[chIdx] * k.inChannels * k.kSize;

		float value = 0.f;

		for (size_t kCh = 0; kCh < k.inChannels ; kCh++)
		{
			for (size_t kIdx = 0; kIdx < k.kSize; kIdx++)
			{
				inpKIdx = 0;
				inpDiv = 1;
				curKDiv = 1;
				for (int i = n - 1; i >= 0; i--)
				{
					curKShape[i] = (kIdx / curKDiv) % k.kernelSize[i];
					inpKIdx += (curInpShape[i] + curKShape[i]) * inpDiv;
					inpDiv *= inp.shape[dim - n + i];
					curKDiv *= k.kernelSize[i];
				}
				value += inp[inpIdx + inpKIdx + (curGroups * k.inChannels + kCh) * inpDiv] * k.weight[baseKIdx + kCh * k.kSize + kIdx];
			}
		}

		if (k.bias != nullptr)
		{
			value += k.bias[curOutShape[chIdx]];
		}
		out[idx] = value;
	}

	auto endTime = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	std::cout << duration.count() << "ms" << std::endl;
}

template<int dim, int n>
__global__ void ConvNdGPUKernel(Matrix<dim> in, const ConvNd<n> k, Matrix<dim> out);

template<int dim, int n>
void ConvNdGPU(Matrix<dim>& inp, const ConvNd<n>& k, Matrix<dim>& out)
{
	assert(dim - n >= 1);
	constexpr int chIdx = dim - n - 1;

	bool usePadding = false;
	std::vector<int> pad;
	for (size_t i = 0; i < n; i++)
	{
		int v = k.padding[i];
		if (v != 0)
		{
			usePadding = true;
		}

		pad.push_back(v);
		pad.push_back(v);
	}

	if (usePadding)
	{
		Pad(inp, pad, 0.f);
	}

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
	ConvNd<n> d_k(k);
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
	ConvNdGPUKernel<<<dimGrid, dimBlock>>>(
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

template<int dim, int n>
__global__ void ConvNdGPUKernel(Matrix<dim> inp, const ConvNd<n> k, Matrix<dim> out)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= out.size)
	{
		return;
	}

	constexpr int chIdx = dim - n - 1;

	int curInpShape[n];
	int curOutShape[dim];
	int curKShape[n];

	int curDiv = 1;
	int curKDiv = 1;
	int inpDiv = 1;

	int inpIdx = 0;
	int inpKIdx = 0;

	for (int i = dim - 1; i >= 0; i--)
	{
		curOutShape[i] = (idx / curDiv) % out.shape[i];
		if (i < chIdx)
		{
			inpIdx += curOutShape[i] * inpDiv;
		}
		curDiv *= out.shape[i];
		inpDiv *= inp.shape[i];
	}

	for (int i = n - 1; i >= 0; i--)
	{
		curInpShape[i] = k.stride[i] * (curOutShape[i - n + dim] - 1) + 1 + k.dillation[i] * (k.kernelSize[i] - 1) - 2 * k.padding[i];
	}

	int curGroups = k.groups * curOutShape[chIdx] / out.shape[chIdx];
	int baseKIdx = curOutShape[chIdx] * k.inChannels * k.kSize;

	float value = 0.f;

	for (size_t kCh = 0; kCh < k.inChannels; kCh++)
	{
		for (size_t kIdx = 0; kIdx < k.kSize; kIdx++)
		{
			inpKIdx = 0;
			inpDiv = 1;
			curKDiv = 1;
			for (int i = n - 1; i >= 0; i--)
			{
				curKShape[i] = (kIdx / curKDiv) % k.kernelSize[i];
				inpKIdx += (curInpShape[i] + curKShape[i]) * inpDiv;
				inpDiv *= inp.shape[dim - n + i];
				curKDiv *= k.kernelSize[i];
			}
			value += inp[inpIdx + inpKIdx + (curGroups * k.inChannels + kCh) * inpDiv] * k.weight[baseKIdx + kCh * k.kSize + kIdx];
		}
	}

	if (k.bias != nullptr)
	{
		value += k.bias[curOutShape[chIdx]];
	}

	out[idx] = value;
}

template<int dim, int n>
__global__ void ConvNdGPUKernelWithSharedMemory(const Matrix<dim> in, const ConvNd<n> k, Matrix<dim> out);

template<int dim, int n>
void ConvNdGPUWithSharedMemory(Matrix<dim> inp, const ConvNd<n> k, Matrix<dim> out)
{
	assert(dim - n >= 1);
	constexpr int chIdx = dim - n - 1;

	bool usePadding = false;
	std::vector<int> pad;
	for (size_t i = 0; i < n; i++)
	{
		int v = k.padding[i];
		if (v != 0)
		{
			usePadding = true;
		}

		pad.push_back(v);
		pad.push_back(v);
	}

	if (usePadding)
	{
		Pad(inp, pad, 0.f);
	}

	// Load GPU info
	GPUInfo info = GPUInfo(false);
	const int deviceCount = info.GetDeviceCount();
	const cudaDeviceProp* deviceProps = info.GetDeviceProps();

	// Invoke kernel
	size_t outputSize = out.size;
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

	Matrix<dim> d_inp(inp);
	ConvNd<n> d_k(k);
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
	ConvNdGPUKernelWithSharedMemory<<<dimGrid, dimBlock>>>(
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

template<int dim, int n>
__global__ void ConvNdGPUKernelWithSharedMemory (const Matrix<dim> in, const ConvNd<n> k, Matrix<dim> out)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= out.size)
	{
		return;
	}

	constexpr int chIdx = dim - n - 1;

	int curInpShape[n];
	int curOutShape[dim];
	int curKShape[n];

	int curDiv = 1;
	int curKDiv = 1;
	int inpDiv = 1;

	int inpIdx = 0;
	int inpKIdx = 0;

	for (int i = dim - 1; i >= 0; i--)
	{
		curOutShape[i] = (idx / curDiv) % out.shape[i];
		if (i < chIdx)
		{
			inpIdx += curOutShape[i] * inpDiv;
		}
		curDiv *= out.shape[i];
		inpDiv *= inp.shape[i];
	}

	for (int i = n - 1; i >= 0; i--)
	{
		curInpShape[i] = k.stride[i] * (curOutShape[i - n + dim] - 1) + 1 + k.dillation[i] * (k.kernelSize[i] - 1) - 2 * k.padding[i];
	}

	int curGroups = k.groups * curOutShape[chIdx] / out.shape[chIdx];
	int baseKIdx = curOutShape[chIdx] * k.inChannels * k.kSize;

	float value = 0.f;

	for (size_t kCh = 0; kCh < k.inChannels; kCh++)
	{
		for (size_t kIdx = 0; kIdx < k.kSize; kIdx++)
		{
			inpKIdx = 0;
			inpDiv = 1;
			curKDiv = 1;
			for (int i = n - 1; i >= 0; i--)
			{
				curKShape[i] = (kIdx / curKDiv) % k.kernelSize[i];
				inpKIdx += (curInpShape[i] + curKShape[i]) * inpDiv;
				inpDiv *= inp.shape[dim - n + i];
				curKDiv *= k.kernelSize[i];
			}
			value += inp[inpIdx + inpKIdx + (curGroups * k.inChannels + kCh) * inpDiv] * k.weight[baseKIdx + kCh * k.kSize + kIdx];
		}
	}

	if (k.bias != nullptr)
	{
		value += k.bias[curOutShape[chIdx]];
	}

	out[idx] = value;
}

void ConvNdTest()
{
	constexpr int dim = 4;
	constexpr int n = 2;

	Matrix<dim> a = Ones<dim>({ 1, 3, 512, 512 });
	Matrix<dim> b = Ones<dim>({ 1, 3, 512, 512 });
	Conv2d k(3, 32, 3, 1, 1, 1, 1);
	for (size_t i = 0; i < k.wSize; i++)
	{
		if (i < k.wSize / k.groups)
		{
			k.weight[i] = 1.f;
		}
		else
		{
			k.weight[i] = 2.f;
		}
	}

	for (size_t i = 0; i < k.bSize; i++)
	{
		k.bias[i] = 0.f;
	}

	Matrix<dim> outCPU = Zeros<dim>({ 1, 32, 512, 512 });
	Matrix<dim> outGPU = Zeros<dim>({ 1, 32, 512, 512 });
	Matrix<dim> outGPUSM = Zeros<dim>({ 1, 32, 512, 512 });

	ConvNdCPU(a, k, outCPU);
	ConvNdGPU(b, k, outGPU);

	std::cout << (outCPU == outGPU) << std::endl;
}