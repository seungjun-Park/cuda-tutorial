#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Conv.h"



void Conv2dCPU(const Matrix& m, const Matrix& k, Matrix& out)
{
	for (size_t row = 0; row < out.GetHeight(); row++)
	{
		for (size_t col = 0; col < out.GetWidth(); col++)
		{
			float outValue = 0;
			for (size_t kRow = 0; kRow < k.GetHeight(); kRow++)
			{
				for (size_t kCol = 0; kCol < k.GetWidth(); kCol++)
				{
					outValue += k[kRow * k.GetWidth() + kCol] * m[(row + kRow) * m.GetWidth() + (col + kCol)];
				}
			}
			out[row * out.GetWidth() + col] = outValue;
		}
	}
}
