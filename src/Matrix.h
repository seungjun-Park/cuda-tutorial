#pragma once
#include <iostream>
#include <cuda_runtime.h>

class Matrix final
{
public:
	Matrix(int batch, int channels, int height, int width, bool useCPU = true);
	Matrix(const Matrix& other);
	Matrix(Matrix&& other) noexcept;

	~Matrix();

	Matrix& operator=(const Matrix& other);
	Matrix& operator=(Matrix&& other) noexcept;
	float& operator[](int index);
	float operator[](int index) const;

	friend bool operator==(const Matrix& left, const Matrix& right);

	int GetWidth() const;
	int GetHeight() const;
	int GetChannels() const;
	int GetBatch() const;

	float* GetElements();
	bool IsCPU() const;
	void ToDevice();
	void ToHost();

private:
	Matrix();

private:
	int width;
	int height;
	int channels;
	int batch;

	bool useCPU;

	float* elements = nullptr;
};


typedef struct MatrixD {
	int width;
	int height;
	int channels;
	int batch;
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

std::ostream& operator<<(std::ostream&, const Matrix&);

void FillZeros(Matrix&);
void FillRandn(Matrix&);