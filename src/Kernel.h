#pragma once


template<int dim>
class Kernel final
{
public:
	Kernel(int* sizes, int* strides, int* dillations);
	Kernel(const Kernel& other);
	Kernel(Kernel&& other);

	~Kernel();

	Kernel& operator=(const Kernel& other);
	Kernel& operator=(Kernel&& other);

	const int* GetSize() const;
	int GetDim() const;
	int Getdillations() const;

private:
	Kernel();

private:
	int sizes[dim];
	int strides[dim];
	int dillations[dim];
	int* elements;
};