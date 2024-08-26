#pragma once
#include "Matrix.h"

void MatMulCPU(const Matrix& A, const Matrix& B, Matrix& C);
void MatMulGPU(Matrix& A, Matrix& B, Matrix& C);
void MatMulGPUWithSharedMemory(Matrix& A, Matrix& B, Matrix& C);