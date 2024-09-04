#pragma once

#include "Matrix.h"

void Conv2dCPU(const Matrix& m, const Matrix& k, Matrix& out);
void Conv2dGPU(Matrix& m, Matrix& k, Matrix& out);
void Conv2dGPUWithSharedMemory(Matrix& m, Matrix& k, Matrix& out);