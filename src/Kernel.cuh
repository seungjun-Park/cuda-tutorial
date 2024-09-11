#pragma once
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Common.h"

template<int dim>
class Kernel final
{
public:
	Kernel(int batch, int channels, int size, int stride, int dillations, bool useCPU = true);
	Kernel(int batch, int channels, int* sizes, int* strides, int* dillations, bool useCPU = true);
	Kernel(const Kernel<dim>& other);
	Kernel(Kernel<dim>&& other);

	~Kernel();

	Kernel<dim>& operator=(const Kernel<dim>& other);
	Kernel<dim>& operator=(Kernel<dim>&& other);

	float& operator[](int index);
	float operator[](int index) const;

	int GetBatch() const;
	int GetChannels() const;
	const int* GetSize() const;
	int GetDim() const;
	const int* Getdillations() const;
	bool IsCPU() const;

	void ToDevice();
	void ToHost();

private:
	Kernel();

private:
	int sizes[dim];
	int strides[dim];
	int dillations[dim];
	int batch;
	int channels;
	float* elements;
	int dim;
	int length = 1;
	bool useCPU;
};

typedef Kernel<1> Kernel1d;
typedef Kernel<2> Kernel2d;
typedef Kernel<3> Kernel3d;

template<int dim>
struct KernelD
{
	const int sizes[dim];
	int strides[dim];
	int dillations[dim];
	int batch;
	int channels;
	float* elements;
	int length;

public:
	__host__ __device__ float operator[](int index) const;
	__host__ __device__ float& operator[](int index);
};

typedef KernelD<1> KernelD1d;
typedef KernelD<2> KernelD2d;
typedef KernelD<3> KernelD3d;

template<int dim>
Kernel<dim>::Kernel(int batch, int channels, int size, int stride, int dillation, bool useCPU) :
	dim(dim),
	useCPU(useCPU),
	batch(batch),
	channels(channels)
{
	for (size_t i = 0; i < dim; i++)
	{
		length *= size;
		this->sizes[i] = size;
		this->strides[i] = stride;
		this->dillations[i] = dillation;
	}
	float* tmp = new float[length];
	for (size_t i = 0; i < length; i++)
	{
		tmp[i] = 1.f;
	}
	if (useCPU)
	{
		elements = tmp;
		tmp = nullptr;
	}
	else
	{
		checkCudaErrors(cudaMalloc(&elements, length * sizeof(float)));
		checkCudaErrors(cudaMemcpy(elements, tmp, length * sizeof(float), cudaMemcpyHostToDevice));
	}
}

template<int dim>
Kernel<dim>::Kernel(int batch, int channels, int* sizes, int* strides, int* dillations, bool useCPU) :
	dim(dim),
	useCPU(useCPU),
	batch(batch),
	channels(channels)
{
	for (size_t i = 0; i < dim; i++)
	{
		length *= sizes[i];
		this->sizes[i] = sizes[i];
		this->strides[i] = strides[i];
		this->dillations[i] = dillations[i];
	}

	float* tmp = new float[length];
	for (size_t i = 0; i < length; i++)
	{
		tmp[i] = 1.f;
	}
	if (useCPU)
	{
		elements = tmp;
		tmp = nullptr;
	}
	else
	{
		checkCudaErrors(cudaMalloc(&elements, length * sizeof(float)));
		checkCudaErrors(cudaMemcpy(elements, tmp, length * sizeof(float), cudaMemcpyHostToDevice));
	}
}

template<int dim>
Kernel<dim>::Kernel(const Kernel& other) :
	dim(dim),
	useCPU(other.useCPU),
	batch(other.batch),
	channels(other.channels)
{
	length = other.length;
	for (size_t i = 0; i < dim; i++)
	{
		this->sizes[i] = other.sizes[i];
		this->strides[i] = other.strides[i];
		this->dillations[i] = other.dillations[i];
	}

	if (useCPU)
	{
		elements = new float[length];
		for (size_t i = 0; i < length; i++)
		{
			elements[i] = other.elements[i];
		}
	}
	else
	{
		checkCudaErrors(cudaMalloc(&elements, other.elements, length * sizeof(float)));
		checkCudaErrors(cudaMemcpy(elements, other.elements, length * sizeof(float), cudaMemcpyDeviceToDevice));
	}
}

template<int dim>
Kernel<dim>::Kernel(Kernel&& other) :
	dim(dim),
	useCPU(other.useCPU),
	batch(other.batch),
	channels(other.channels)
{
	for (size_t i = 0; i < dim; i++)
	{
		this->sizes[i] = other.sizes[i];
		this->strides[i] = other.strides[i];
		this->dillations[i] = other.dillations[i];
	}
	elements = other.elements;
	other.elements = nullptr;
}

template<int dim>
Kernel<dim>::~Kernel()
{
	if (elements != nullptr)
	{
		if (useCPU)
		{
			delete[] elements;
		}
		else
		{
			cudaFree(elements);
		}
	}
}

template<int dim>
Kernel<dim>& Kernel<dim>::operator=(const Kernel& other)
{
	dim = dim;
	useCPU = other.useCPU;
	batch = other.batch;
	channels = other.channels;
	for (size_t i = 0; i < dim; i++)
	{
		this->sizes[i] = other.sizes[i];
		this->strides[i] = other.strides[i];
		this->dillations[i] = other.dillations[i];
	}
	elements = other.elements;
	other.elements = nullptr;

	return *this;
}

template<int dim>
Kernel<dim>& Kernel<dim>::operator=(Kernel&& other)
{
	dim = dim;
	useCPU = other.useCPU;
	batch = other.batch;
	channels = other.channels;
	for (size_t i = 0; i < dim; i++)
	{
		this->sizes[i] = other.sizes[i];
		this->strides[i] = other.strides[i];
		this->dillations[i] = other.dillations[i];
	}
	elements = other.elements;
	other.elements = nullptr;

	return *this;
}

template<int dim>
float& Kernel<dim>::operator[](int index)
{
	assert(index < length && printf("index %d must be smaller than length %d.\n", index, length));
	return elements[index];
}

template<int dim>
float Kernel<dim>::operator[](int index) const
{
	assert(index < length && printf("index %d must be smaller than length %d.\n", index, length));
	return elements[index];
}

template<int dim>
const int* Kernel<dim>::GetSize() const
{
	return (int*)&sizes;
}

template<int dim>
int Kernel<dim>::GetDim() const
{
	return dim;
}

template<int dim>
int Kernel<dim>::GetBatch() const
{
	return batch;
}

template<int dim>
int Kernel<dim>::GetChannels() const
{
	return channels;
}


template<int dim>
const int* Kernel<dim>::Getdillations() const
{
	return (int*)& dillations;
}

template<int dim>
bool Kernel<dim>::IsCPU() const
{
	return useCPU;
}

template<int dim>
void Kernel<dim>::ToDevice()
{
	if (!useCPU)
	{
		return;
	}

	float* tmp = elements;
	elements = nullptr;
	checkCudaErrors(cudaMalloc(&elements, length * sizeof(float)));
	checkCudaErrors(cudaMemcpy(elements, tmp, length * sizeof(float), cudaMemcpyHostToDevice));
	delete[] tmp;
	useCPU = false;
}

template<int dim>
void Kernel<dim>::ToHost()
{
	if (useCPU)
	{
		return;
	}

	float* tmp = elements;
	elements = new float[length];
	checkCudaErrors(cudaMemcpy(elements, tmp, length * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(tmp);
	useCPU = true;
}


template<int dim>
__host__ __device__ float KernelD<dim>::operator[](int index) const
{
	assert(index < length && printf("index %d must be smaller than length %d.\n", index, length));
	return elements[index];
}

template<int dim>
__host__ __device__ float& KernelD<dim>::operator[](int index)
{
	assert(index < length && printf("index %d must be smaller than length %d.\n", index, length));
	return elements[index];
}