#include <assert.h>
#include <cuda_runtime.h>
#include <string>
#include <random>
#include "Matrix.h"

Matrix::Matrix(int height, int width, int stride, bool useCPU) :
	width(width),
	height(height),
	stride(stride),
	useCPU(useCPU)
{
	if (useCPU)
	{
		size_t size = this->width * this->height;
		this->elements = new float[size];
		for (size_t i = 0; i < size; i++)
		{
			this->elements[i] = i;
		}
	}
	else
	{
		size_t size = width * height * sizeof(float);
		cudaError_t cudaStatus = cudaMalloc((void**)&this->elements, size);
		assert(cudaStatus == cudaError_t::cudaSuccess, "cudaMalloc was failed in Matrix class!");
	}
}

Matrix::Matrix(const Matrix& other) :
	width(other.width),
	height(other.height),
	stride(other.stride),
	useCPU(other.useCPU)
{
	size_t size = width * height * sizeof(float);
	cudaMemcpyKind cudaMemcpyType;
	cudaError_t cudaStatus = cudaError_t::cudaSuccess;
	if (useCPU)
	{
		this->elements = new float[width * height];
		cudaMemcpyType = cudaMemcpyHostToHost;
	}
	else
	{
		cudaStatus = cudaMalloc((void**)&this->elements, size);
		assert(cudaStatus == cudaError_t::cudaSuccess, "cudaMalloc was failed in Matrix class!");
		cudaMemcpyType = cudaMemcpyDeviceToDevice;
	}
	cudaStatus = cudaMemcpy(this->elements, other.elements, size, cudaMemcpyType);
	assert(cudaStatus == cudaError_t::cudaSuccess);
}

Matrix::Matrix(Matrix&& other) noexcept:
	width(other.width),
	height(other.height),
	stride(other.stride),
	useCPU(other.useCPU)
{
	this->elements = other.elements;
	other.elements = nullptr;
}

Matrix::~Matrix()
{
	if (this->elements != nullptr)
	{
		if (useCPU)
		{
			delete[] this->elements;
		}

		else
		{
			cudaFree(this->elements);
		}
	}
}

Matrix& Matrix::operator=(const Matrix& other)
{
	assert((this->height == other.height) && (this->width == other.width));

	size_t size = this->width * this->height * sizeof(float);
	cudaMemcpyKind cudaMemcpyType;
	if (this->useCPU)
	{
		if (other.useCPU)
		{
			cudaMemcpyType = cudaMemcpyHostToHost;
		}
		else
		{
			cudaMemcpyType = cudaMemcpyDeviceToHost;
		}
	}
	else
	{
		if (other.useCPU)
		{
			cudaMemcpyType = cudaMemcpyHostToDevice;
		}
		else
		{
			cudaMemcpyType = cudaMemcpyDeviceToDevice;
		}
	}

	cudaMemcpy(this->elements, other.elements, size, cudaMemcpyType);

	return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
	assert((this->height == other.height) && (this->width == other.width));

	if (this->elements != nullptr)
	{
		if (useCPU)
		{
			delete[] this->elements;
		}
		else
		{
			cudaFree(this->elements);
		}
	}

	this->elements = other.elements;
	this->useCPU = other.useCPU;

	return *this;
}

float& Matrix::operator[](int index)
{
	return this->elements[index];
}

float Matrix::operator[](int index) const
{
	return this->elements[index];
}

int Matrix::GetWidth() const
{
	return this->width;
}

int Matrix::GetHeight() const
{
	return this->height;
}

int Matrix::GetStride() const
{
	return this->stride;
}

bool Matrix::IsCPU() const
{
	return this->useCPU;
}

float* Matrix::GetElements()
{
	return this->elements;
}

void Matrix::ConvertDevice()
{
	cudaError_t cudaStatus = cudaError_t::cudaSuccess;
	if (this->useCPU)
	{
		float* tmp = this->elements;
		this->elements = nullptr;
		size_t size = this->width * this->height * sizeof(float);
		cudaStatus = cudaMalloc((void**)&this->elements, size);
		assert(cudaStatus == cudaError_t::cudaSuccess);
		cudaStatus = cudaMemcpy(this->elements, tmp, size, cudaMemcpyHostToDevice);
		assert(cudaStatus == cudaError_t::cudaSuccess);
		delete[] tmp;
		this->useCPU = false;
	}
	else
	{
		float* tmp = this->elements;
		this->elements = nullptr;
		size_t size = this->width * this->height * sizeof(float);
		this->elements = new float[width * height];
		cudaStatus = cudaMemcpy(this->elements, tmp, size, cudaMemcpyDeviceToHost);
		assert(cudaStatus == cudaError_t::cudaSuccess);
		cudaFree(tmp);
		this->useCPU = true;
	}
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
	int width = matrix.GetWidth();
	for (size_t row = 0; row < matrix.GetHeight(); row++)
	{
		for (size_t col = 0; col < matrix.GetWidth(); col++)
		{
			size_t index = row * width + col;
			os << matrix[index] << ", ";
		}
		os << "\n";
	}

	return os;
}

void FillZeros(Matrix& matrix)
{
	int width = matrix.GetWidth();
	for (size_t row = 0; row < matrix.GetHeight(); row++)
	{
		for (size_t col = 0; col < matrix.GetWidth(); col++)
		{
			matrix[row * width + col] = 0.f;
		}
	}
}

void FillRandn(Matrix& matrix)
{
	int width = matrix.GetWidth();
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::normal_distribution<float> nd(0, 1);
	for (size_t row = 0; row < matrix.GetHeight(); row++)
	{
		for (size_t col = 0; col < matrix.GetWidth(); col++)
		{
			matrix[row * width + col] = nd(gen);
		}
	}
}