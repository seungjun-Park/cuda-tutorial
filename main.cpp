#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "MatMul.cuh"
#include "Matrix.h"



int main()
{
    Matrix* A = new Matrix(1000, 1000, 0, true);
    Matrix* B = new Matrix(1000, 1000, 0, true);
    Matrix* C = new Matrix(1000, 1000, 0, true);

    FillRandn(*A);
    FillRandn(*B);
    FillZeros(*C);

    /*std::cout << *A << std::endl;
    std::cout << *B << std::endl;*/

    MatMulCPU(*A, *B, *C);
    MatMulGPU(*A, *B, *C);

    delete A;
    delete B;
    delete C;

    return 0;
}