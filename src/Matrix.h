#pragma once
#include <iostream>
#include <cuda_runtime.h>

class Matrix final
{
public:
	Matrix(int height, int width, bool useCPU = true);
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
	float* GetElements();
	bool IsCPU() const;
	void ToDevice();
	void ToHost();

private:
	Matrix();

private:
	int width;
	int height;
	bool useCPU;
	float* elements = nullptr;
};

std::ostream& operator<<(std::ostream&, const Matrix&);

void FillZeros(Matrix&);
void FillRandn(Matrix&);