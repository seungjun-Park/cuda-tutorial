#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <random>
#include "Matrix.h"
#include "GPUInfo.h"

Matrix::Matrix(int height, int width, bool useCPU) :
	width(width),
	height(height),
	useCPU(useCPU)
{
	assert(elements == nullptr);
	size_t size = width * height;
	float* tmp = new float[size];
	for (size_t i = 0; i < size; i++)
	{
		tmp[i] = i;
	}
	if (useCPU)
	{
		elements = tmp;
		tmp = nullptr;
	}
	else
	{
		size *= sizeof(float);
		cudaError cudaStatus = cudaMalloc((void**)&elements, size);
		assert(cudaStatus == cudaError::cudaSuccess);
		cudaStatus = cudaMemcpy(elements, tmp, size, cudaMemcpyHostToDevice);
		assert(cudaStatus == cudaError::cudaSuccess);
	}
}

Matrix::Matrix(const Matrix& other) :
	width(other.width),
	height(other.height),
	useCPU(other.useCPU)
{
	assert(elements == nullptr);
	size_t size = width * height;
	if (useCPU)
	{
		elements = new float[size];
		for (size_t i = 0; i < size; i++)
		{
			elements[i] = other.elements[i];
		}
	}
	else
	{
		size *= sizeof(float);
		cudaError cudaStatus = cudaMalloc((void**)&elements, size);
		assert(cudaStatus == cudaError::cudaSuccess);
		cudaStatus = cudaMemcpy(elements, other.elements, size, cudaMemcpyDeviceToDevice);
		assert(cudaStatus == cudaError::cudaSuccess);
	}
}

Matrix::Matrix(Matrix&& other) noexcept:
	width(other.width),
	height(other.height),
	useCPU(other.useCPU)
{
	elements = other.elements;
	other.elements = nullptr;
}

Matrix::~Matrix()
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

Matrix& Matrix::operator=(const Matrix& other)
{
	assert((height== other.height) && (width == other.width) && (useCPU == other.useCPU));

	if (useCPU)
	{
		for (size_t i = 0; i < width * height; i++)
		{
			elements[i] = other.elements[i];
		}
	}
	else
	{
		cudaError_t cudaStatus = cudaMemcpy(elements, other.elements, sizeof(float) * width * height, cudaMemcpyDeviceToDevice);
		assert(cudaStatus == cudaError_t::cudaSuccess);
	}

	return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
	assert((height == other.height) && (width == other.width));

	elements = other.elements;
	useCPU = other.useCPU;

	return *this;
}

float& Matrix::operator[](int index)
{
	return elements[index];
}

float Matrix::operator[](int index) const
{
	return elements[index];
}

int Matrix::GetWidth() const
{
	return width;
}

int Matrix::GetHeight() const
{
	return height;
}

bool Matrix::IsCPU() const
{
	return useCPU;
}

float* Matrix::GetElements()
{
	return elements;
}

void Matrix::ToDevice()
{
	if (!useCPU)
	{
		return;
	}

	float* tmp = elements;
	elements = nullptr;
	size_t size = width * height * sizeof(float);
	cudaError cudaStatus = cudaMalloc((void**)&elements, size);
	assert(cudaStatus == cudaError::cudaSuccess);
	cudaStatus = cudaMemcpy(elements, tmp, size, cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaError::cudaSuccess);
	useCPU = false;
}

void Matrix::ToHost()
{
	if (useCPU)
	{
		return;
	}
	float* tmp = elements;
	elements = nullptr;
	elements = new float[width * height];
	size_t size = width * height * sizeof(float);
	cudaError_t cudaStatus = cudaMemcpy(elements, tmp, size, cudaMemcpyDeviceToHost);
	assert(cudaStatus == cudaError::cudaSuccess);
	useCPU = true;
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
	if (matrix.IsCPU())
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
	else
	{
		GPUInfo info = GPUInfo(false);
		const int deviceCount = info.GetDeviceCount();
		const cudaDeviceProp* deviceProps = info.GetDeviceProps();

		// Invoke kernel
		int outputSize = matrix.GetWidth() * matrix.GetHeight();
		size_t blockSize = std::min((int)outputSize, deviceProps[0].maxThreadsPerBlock);
		dim3 dimBlock(blockSize);
		size_t gridSize = (outputSize / blockSize) + ((outputSize % blockSize > 0) ? 1 : 0);
		gridSize = std::min((int)gridSize, deviceProps[0].maxGridSize[0]);
		dim3 dimGrid(gridSize);
		
		// FillZerosKernel<<<dimGrid, dimBlock>>>(matrix.GetElements(), matrix.GetWidth(), outputSize);

		cudaDeviceSynchronize();
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

bool operator==(const Matrix& left, const Matrix& right)
{
	assert(left.width == right.width && left.height == right.height);

	for (size_t row = 0; row < left.height; row++)
	{
		for (size_t col = 0; col < left.width; col++)
		{
			if (std::abs(left.elements[row * left.width + col] - right.elements[row * right.width + col]) > 1e-4)
			{
				std::cout << "row: " << row << ", col: " << col << std::endl;
				std::cout << "left: " << left.elements[row * left.width + col] << ", right: " << right.elements[row * right.width + col] << std::endl;
				return false;
			}
		}
	}

	return true;
}


