#include "Matrix.cuh"
#include "MatMul.cuh"
#include "Linear.cuh"
#include "Conv.cuh"

#include <cuda_runtime.h>
#include "Common.h"

int main()
{
	// MatMulTest();
	// LinearTest();
	ConvNdTest();

	return 0;
}