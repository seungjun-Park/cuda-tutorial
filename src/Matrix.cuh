#pragma once
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

///////////////////////////// Declaration ///////////////////////////////

template<int dim>
class Matrix final
{
public:
	Matrix(std::vector<int> shape, bool useCPU = true);
	Matrix(const Matrix& other);
	Matrix(Matrix&& other) noexcept;

	~Matrix();

	Matrix<dim>& operator=(const Matrix& other);
	Matrix<dim>& operator=(Matrix&& other) noexcept;
	float& operator[](int index);
	float operator[](int index) const;

	const int* GetShape() const;
	constexpr int GetDim() const;
	int GetSize() const;
	float* GetElements();
	bool IsCPU() const;
	void ToDevice();
	void ToHost();

private:
	Matrix();

private:
	int* shape;
	int size;
	bool useCPU;

	float* elements = nullptr;
};

typedef Matrix<1> Matrix1d;
typedef Matrix<2> Matrix2d;
typedef Matrix<3> Matrix3d;
typedef Matrix<4> Matrix4d;
typedef Matrix<5> Matrix5d;

template<int dim>
struct MatrixD {
	const int* shape;
	int size;

	float* elements;

public:
	__host__ __device__ float& operator[](int index);
	__host__ __device__ float operator[](int index) const;
};

typedef MatrixD<1> MatrixD1d;
typedef MatrixD<2> MatrixD2d;
typedef MatrixD<3> MatrixD3d;
typedef MatrixD<4> MatrixD4d;
typedef MatrixD<5> MatrixD5d;

template<int dim>
std::ostream& operator<<(std::ostream&, const Matrix<dim>&);

template<int dim>
bool operator==(const Matrix<dim>& left, const Matrix<dim>& right);

template<int dim>
Matrix<dim> Zeros(std::vector<int> shape, bool useCPU = true);

template<int dim>
void Zeros(Matrix<dim>& m);

template<int dim>
Matrix<dim> Randn(std::vector<int> shape, bool useCPU = true);

template<int dim>
void Randn(Matrix<dim>& m);


//////////////////////////////// Implmentation //////////////////////////////
#include <assert.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <random>
#include "GPUInfo.h"
#include "Common.h"

template<int dim>
Matrix<dim>::Matrix(std::vector<int> shape, bool useCPU) :
	useCPU(useCPU)
{
	assert(elements == nullptr && dim == shape.size());

	size = 1;
	this->shape = new int[dim];
	for (size_t i = 0; i < dim; i++)
	{
		this->shape[i] = shape[i];
		size *= shape[i];
	}
	elements = new float[size];
	for (size_t i = 0; i < size; i++)
	{
		elements[i] = 0.f;
	}
	if (!useCPU)
	{
		float* tmpElements = elements;
		elements = nullptr;
		int* tmpShape = this->shape;
		this->shape = nullptr;

		checkCudaErrors(cudaMalloc((void**)&elements, size * sizeof(float)));
		checkCudaErrors(cudaMemcpy(elements, tmpElements, size * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&this->shape, dim * sizeof(int)));
		checkCudaErrors(cudaMemcpy(this->shape, tmpShape, dim * sizeof(int), cudaMemcpyHostToDevice));
	}
}

template<int dim>
Matrix<dim>::Matrix(const Matrix& other) :
	useCPU(other.useCPU),
	size(other.size)
{
	assert(elements == nullptr && shape == nullptr && dim == other.dim);
	if (useCPU)
	{
		elements = new float[size];
		for (size_t i = 0; i < size; i++)
		{
			elements[i] = other.elements[i];
		}
		shape = new int[dim];
		for (size_t i = 0; i < dim; i++)
		{
			this->shape[i] = other.shape[i];
		}
	}
	else
	{
		checkCudaErrors(cudaMalloc((void**)&elements, size * sizeof(float)));
		checkCudaErrors(cudaMemcpy(elements, other.elements, size * sizeof(float), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMalloc((void**)&shape, dim * sizeof(int)));
		checkCudaErrors(cudaMemcpy(shape, other.shape, dim * sizeof(int), cudaMemcpyDeviceToDevice));
	}
}

template<int dim>
Matrix<dim>::Matrix(Matrix&& other) noexcept :
	useCPU(other.useCPU),
	size(other.size)
{
	assert(elements == nullptr && shape == nullptr && dim == other.dim);
	shape = other.shape;
	other.shape = nullptr;
	elements = other.elements;
	other.elements = nullptr;
}

template<int dim>
Matrix<dim>::~Matrix()
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

	if (shape != nullptr)
	{
		if (useCPU)
		{
			delete[] shape;
		}
		else
		{
			cudaFree(shape);
		}
	}
}

template<int dim>
Matrix<dim>& Matrix<dim>::operator=(const Matrix& other)
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

	if (shape != nullptr)
	{
		if (useCPU)
		{
			delete[] shape;
		}
		else
		{
			cudaFree(shape);
		}
	}

	useCPU = other.useCPU;
	dim = other.dim;
	size = other.size;

	if (other.useCPU)
	{
		elements = new float[size];
		for (size_t i = 0; i < size; i++)
		{
			elements[i] = other.elements[i];
		}

		shape = new int[dim];
		for (size_t i = 0; i < dim; i++)
		{
			shape[i] = other.shape[i];
		}
	}
	else
	{
		checkCudaErrors(cudaMalloc((void**)&elements, size * sizeof(float)));
		checkCudaErrors(cudaMemcpy(elements, other.elements, sizeof(float) * size, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMalloc((void**)&shape, dim * sizeof(int)));
		checkCudaErrors(cudaMemcpy(shape, other.shape, dim * sizeof(int), cudaMemcpyDeviceToDevice));
	}

	return *this;
}

template<int dim>
Matrix<dim>& Matrix<dim>::operator=(Matrix&& other) noexcept
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

	if (shape != nullptr)
	{
		if (useCPU)
		{
			delete[] shape;
		}
		else
		{
			cudaFree(shape);
		}
	}

	elements = other.elements;
	shape = other.shape;
	useCPU = other.useCPU;
	other.elements = nullptr;
	other.shape = nullptr;
	return *this;
}

template<int dim>
float& Matrix<dim>::operator[](int index)
{
	assert(index < size);
	return elements[index];
}

template<int dim>
float Matrix<dim>::operator[](int index) const
{
	assert(index < size);
	return elements[index];
}

template<int dim>
bool Matrix<dim>::IsCPU() const
{
	return useCPU;
}

template<int dim>
float* Matrix<dim>::GetElements()
{
	return elements;
}

template<int dim>
const int* Matrix<dim>::GetShape() const
{
	return shape;
}

template<int dim>
constexpr int Matrix<dim>::GetDim() const
{
	return dim;
}

template<int dim>
int Matrix<dim>::GetSize() const
{
	return size;
}

template<int dim>
void Matrix<dim>::ToDevice()
{
	if (!useCPU)
	{
		return;
	}

	float* tmp = elements;
	elements = nullptr;
	checkCudaErrors(cudaMalloc((void**)&elements, size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(elements, tmp, size * sizeof(float), cudaMemcpyHostToDevice));

	int* tmpShape = shape;
	shape = nullptr;
	checkCudaErrors(cudaMalloc((void**)&shape, dim * sizeof(int)));
	checkCudaErrors(cudaMemcpy(shape, tmpShape, dim * sizeof(int), cudaMemcpyHostToDevice));

	useCPU = false;
	delete[] tmp;
	delete[] tmpShape;
}

template<int dim>
void Matrix<dim>::ToHost()
{
	if (useCPU)
	{
		return;
	}

	float* tmp = elements;
	elements = nullptr;
	elements = new float[size];
	checkCudaErrors(cudaMemcpy(elements, tmp, size * sizeof(float), cudaMemcpyDeviceToHost));

	int* tmpShape = shape;
	shape = nullptr;
	shape = new int[dim];
	checkCudaErrors(cudaMemcpy(shape, tmpShape, dim * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(tmp);
	cudaFree(tmpShape);
	useCPU = true;
}

template<int dim>
Matrix<dim> Zeros(std::vector<int> shape, bool useCPU)
{
	Matrix<dim> m(shape, true);

	for (size_t i = 0; i < m.GetSize(); i++)
	{
		m[i] = 0.f;
	}
	if (!useCPU)
	{
		m.ToDevice();
	}
	return m;
}


template<int dim>
void Zeros(Matrix<dim>& m)
{
	assert(m.IsCPU());
	for (size_t i = 0; i < m.GetSize(); i++)
	{
		m[i] = 0.f;
	}
}

template<int dim>
Matrix<dim> Randn(std::vector<int> shape, bool useCPU)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::normal_distribution<float> nd(0, 1);
	Matrix<dim> m(shape, true);

	for (size_t i = 0; i < m.GetSize(); i++)
	{
		m[i] = nd(gen);
	}

	if (!useCPU)
	{
		m.ToDevice();
	}

	return m;
}

template<int dim>
void Randn(Matrix<dim>& m)
{
	assert(m.IsCPU());

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::normal_distribution<float> nd(0, 1);

	for (size_t i = 0; i < m.GetSize(); i++)
	{
		m[i] = nd(gen);
	}
}

template<int dim>
bool operator==(const Matrix<dim>& left, const Matrix<dim>& right)
{
	for (size_t i = 0; i < left.GetSize(); i++)
	{
		if ((left[i] - right[i]) > 1e-4)
		{
			std::cout << "index: " << i << "(" << left[i] << ", " << right[i] << ")" << " is not eqaul." << std::endl;
			return false;
		}
	}
	return true;
}

template<int dim>
__host__ __device__
float MatrixD<dim>::operator[](int index) const
{
	if (index >= size)
	{
		printf("index: %d >= %d\n", index, size);
	}
	assert(index < size);
	return elements[index];
}

template<int dim>
__host__ __device__
float& MatrixD<dim>::operator[](int index)
{
	if (index >= size)
	{
		printf("index: %d >= %d\n", index, size);
	}
	assert(index < size);
	return elements[index];
}

template<int dim>
std::ostream& operator<<(std::ostream& os, const Matrix<dim>& matrix)
{
	const int* shape = matrix.GetShape();
	int curShape[dim];
	int div = 1;
	for (int i = dim - 1; i >= 0; i--)
	{
		curShape[i] = 0;
	}

	for (size_t idx = 0; idx < matrix.GetSize(); idx++)
	{
		os << matrix[idx] << ", ";

		div = 1;
		for (int i = dim - 1; i >= 0; i--)
		{
			curShape[i] = (idx / div) % shape[i];
			div *= shape[i];
		}
		curShape[dim - 1] += 1;
		for (int i = dim - 1; i > 0; i--)
		{
			if (curShape[i] < shape[i])
			{
				break;
			}
			curShape[i - 1] += 1;
			os << "\n";
		}
	}

	return os;
}