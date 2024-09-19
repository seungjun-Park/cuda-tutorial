#pragma once
#include <assert.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <random>

#include "GPUInfo.h"
#include "Common.h"

///////////////////////////// Declaration ///////////////////////////////

template<int dim>
struct Matrix
{
public:
	__host__ __device__ float operator[](int index) const;
	__host__ __device__ float& operator[](int index);

public:
	int shape[dim];
	int size = 1;

	float* elements = nullptr;
};

typedef Matrix<1> Matrix1d;
typedef Matrix<2> Matrix2d;
typedef Matrix<3> Matrix3d;
typedef Matrix<4> Matrix4d;
typedef Matrix<5> Matrix5d;

template<int dim>
float Matrix<dim>::operator[](int index) const
{
	assert(index < size);
	return elements[index];
}

template<int dim>
float& Matrix<dim>::operator[](int index)
{
	assert(index < size);
	return elements[index];
}

template<int dim>
std::ostream& operator<<(std::ostream& os, const Matrix<dim>& matrix)
{
	int curShape[dim];

	int div = 1;
	for (int i = dim - 1; i >= 0; i--)
	{
		curShape[i] = 0;
	}

	for (size_t idx = 0; idx < matrix.size; idx++)
	{
		os << matrix[idx] << ", ";

		div = 1;
		for (int i = dim - 1; i >= 0; i--)
		{
			curShape[i] = (idx / div) % matrix.shape[i];
			div *= matrix.shape[i];
		}
		curShape[dim - 1] += 1;
		for (int i = dim - 1; i > 0; i--)
		{
			if (curShape[i] < matrix.shape[i])
			{
				break;
			}
			curShape[i - 1] += 1;
			os << "\n";
		}
	}

	return os;
}

template<int dim>
bool operator==(const Matrix<dim>& left, const Matrix<dim>& right)
{
	for (size_t i = 0; i < left.size; i++)
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
Matrix<dim> Zeros(std::vector<int> shape, bool useCPU = true)
{
	assert(shape.size() == dim);
	Matrix<dim> m;

	for (size_t i = 0; i < dim; i++)
	{
		m.shape[i] = shape[i];
		m.size *= shape[i];
	}

	m.elements = new float[m.size];

	for (size_t i = 0; i < m.size; i++)
	{
		m[i] = 0.f;
	}

	return m;
}

template<int dim>
Matrix<dim> Randn(std::vector<int> shape, bool useCPU = true)
{
	assert(shape.size() == dim);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::normal_distribution<float> nd(0, 1);
	Matrix<dim> m;

	for (size_t i = 0; i < dim; i++)
	{
		m.shape[i] = shape[i];
		m.size *= shape[i];
	}

	m.elements = new float[m.size];

	for (size_t i = 0; i < m.size; i++)
	{
		m[i] = nd(gen);
	}

	return m;
}

template<int dim>
Matrix<dim> Ones(std::vector<int> shape, bool useCPU = true)
{
	assert(shape.size() == dim);
	Matrix<dim> m;

	for (size_t i = 0; i < dim; i++)
	{
		m.shape[i] = shape[i];
		m.size *= shape[i];
	}

	m.elements = new float[m.size];

	for (size_t i = 0; i < m.size; i++)
	{
		m[i] = 1.f;
	}

	return m;
}

template<int dim>
void Pad(Matrix<dim>& m, std::vector<int> shape, int value = 0)
{
	assert(shape.size() % 2 == 0 && (shape.size() / 2) <= dim);

	int newShape[dim];

	for (size_t i = 0; i < dim; i++)
	{
		newShape[i] = m.shape[i];
	}

	for (size_t i = 0; i < shape.size() / 2; i++)
	{
		newShape[dim - i - 1] += (shape[i * 2] + shape[i * 2 + 1]);
	}

	int newSize = 1;

	for (size_t i = 0; i < dim; i++)
	{
		newSize *= newShape[i];
	}

	float* newElements = new float[newSize];

	int curShape[dim];
	int div;
	int oldIdx = 0;
	bool useValue;
	for (size_t idx = 0; idx < newSize; idx++)
	{
		int i = dim - 1;
		int j = 0;
		div = 1;
		useValue = false;
		while (i >= 0)
		{
			curShape[i] = (idx / div) % newShape[i];
			div *= newShape[i];
			if (j < (shape.size() / 2))
			{
				if ((curShape[i] < shape[j * 2] || (newShape[i] - curShape[i] - 1) < shape[j * 2 + 1]))
				{
					useValue = true;
				}
			}
			i--;
			j++;
		}
		if (useValue)
		{
			newElements[idx] = value;
		}
		else
		{
			newElements[idx] = m.elements[oldIdx];
			oldIdx++;
		}
	}

	for (size_t i = 0; i < dim; i++)
	{
		m.shape[i] = newShape[i];
	}
	float* tmp = m.elements;
	m.elements = newElements;
	m.size = newSize;
	delete[] tmp;
}
