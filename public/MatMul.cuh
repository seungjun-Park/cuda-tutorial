#pragma once
#include "Matrix.h"

void MatMulCPU(const Matrix& A, const Matrix& B, Matrix& C);
void MatMulGPU(Matrix& A, Matrix& B, Matrix& C);

//__global__ void MatMulGPUWithSharedMemoryKernel(const Matrix&, const Matrix&, Matrix&);