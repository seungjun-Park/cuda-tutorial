#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <random>
#include "Matrix.h"
#include "GPUInfo.h"
#include "Common.h"

Matrix::Matrix(int batch, int channels, int height, int width, bool useCPU) :
	batch(batch),
	channels(channels),
	width(width),
	height(height),
	useCPU(useCPU)
{
	assert(elements == nullptr);
	size_t size = batch * channels * width * height;
	float* tmp = new float[size];
	for (size_t i = 0; i < size; i++)
	{
		tmp[i] = 0.f;
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
	batch(other.batch),
	channels(other.channels),
	width(other.width),
	height(other.height),
	useCPU(other.useCPU)
{
	assert(elements == nullptr);
	size_t size = width * height * channels * batch;
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
	batch(other.batch),
	channels(other.channels),
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
		cudaError_t cudaStatus = cudaMemcpy(elements, other.elements, sizeof(float) * width * height * batch * channels, cudaMemcpyDeviceToDevice);
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

int Matrix::GetChannels() const
{
	return channels;
}

int Matrix::GetBatch() const
{
	return batch;
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
	size_t size = width * height * batch * channels;
	cudaError cudaStatus = cudaMalloc((void**)&elements, size * sizeof(float));
	assert(cudaStatus == cudaError::cudaSuccess);
	checkCudaErrors(cudaMemcpy(elements, tmp, size * sizeof(float), cudaMemcpyHostToDevice));
	useCPU = false;
	delete[] tmp;
}

void Matrix::ToHost()
{
	if (useCPU)
	{
		return;
	}
	float* tmp = elements;
	elements = nullptr;
	size_t size = width * height * batch * channels;
	elements = new float[size];
	checkCudaErrors(cudaMemcpy(elements, tmp, size * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(tmp);
	useCPU = true;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
	int batch = matrix.GetBatch();
	int channels = matrix.GetChannels();
	int height = matrix.GetHeight();
	int width = matrix.GetWidth();

	for (size_t b = 0; b < batch; b++)
	{
		for (size_t ch = 0; ch < channels; ch++)
		{
			for (size_t row = 0; row < height; row++)
			{
				for (size_t col = 0; col < width; col++)
				{
					int idx = b * channels * width * height + ch * width * height + row * width + col;
					os << matrix[idx] << ", ";
				}
				os << "\n";
			}
			os << "\n";
		}
		os << "\n";
	}

	return os;
}

void FillZeros(Matrix& matrix)
{
	int batch = matrix.GetBatch();
	int channels = matrix.GetChannels();
	int height = matrix.GetHeight();
	int width = matrix.GetWidth();

	for (size_t b = 0; b < batch; b++)
	{
		for (size_t ch = 0; ch < channels; ch++)
		{
			for (size_t row = 0; row < height; row++)
			{
				for (size_t col = 0; col < width; col++)
				{
					int idx = b * channels * width * height + ch * width * height + row * width + col;
					matrix[idx] = 0.f;
				}
			}
		}
	}
}

void FillRandn(Matrix& matrix)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::normal_distribution<float> nd(0, 1);

	int batch = matrix.GetBatch();
	int channels = matrix.GetChannels();
	int height = matrix.GetHeight();
	int width = matrix.GetWidth();

	for (size_t b = 0; b < batch; b++)
	{
		for (size_t ch = 0; ch < channels; ch++)
		{
			for (size_t row = 0; row < height; row++)
			{
				for (size_t col = 0; col < width; col++)
				{
					int idx = b * channels * width * height + ch * width * height + row * width + col;
					matrix[idx] = nd(gen);
				}
			}
		}
	}
}

bool operator==(const Matrix& left, const Matrix& right)
{
	assert(left.width == right.width && left.height == right.height && left.batch == right.batch && left.channels == right.channels);

	int width = left.width;
	int height = left.height;
	int channels = left.channels;
	int batch = left.batch;

	for (size_t b = 0; b < batch; b++)
	{
		for (size_t ch = 0; ch < channels; ch++)
		{
			for (size_t row = 0; row < height; row++)
			{
				for (size_t col = 0; col < width; col++)
				{
					int idx = b * channels * width * height + ch * width * height + row * width + col;
					if (std::abs(left[idx] - right[idx]) > 1e-4)
					{
						std::cout << "row: " << row << ", col: " << col << std::endl;
						std::cout << "left: " << left[idx] << ", right: " << right[idx] << std::endl;
						return false;
					}
				}
			}
		}
	}

	return true;
}


