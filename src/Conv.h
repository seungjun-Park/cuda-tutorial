#pragma once

#include "Matrix.h"

void Conv2dCPU(const Matrix& m, const Matrix& k, Matrix& out);
void Conv2dGPU();
void Conv2dGPUWithSharedMemory();