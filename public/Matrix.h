#pragma once
#include <iostream>

class Matrix
{
public:
	// Host methods
	Matrix(int height, int width, int stride, bool useCPU=true);
	Matrix(const Matrix& other);
	Matrix(Matrix&& other) noexcept;

	virtual ~Matrix();

	Matrix& operator=(const Matrix& other);
	Matrix& operator=(Matrix&& other) noexcept;
	float& operator[](int index);
	float operator[](int index) const;

	int GetWidth() const;
	int GetHeight() const;
	int GetStride() const;
	bool IsCPU() const;

	float* GetElements();
	void ConvertDevice();


private:
	Matrix();

private:
	int width;
	int height;
	int stride;
	bool useCPU;
	float* elements = nullptr;
};

std::ostream& operator<<(std::ostream&, const Matrix&);

void FillZeros(Matrix&);
void FillRandn(Matrix&);