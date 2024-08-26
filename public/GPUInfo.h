#pragma once
#include <cuda_runtime.h>

class GPUInfo final
{
public:
	GPUInfo(bool printInfo=true);
	~GPUInfo();

	int GetDeviceCount() const;
	const cudaDeviceProp* GetDeviceProps() const;

private:
	void checkDevice(bool printInfo = true);

private:
	int deviceCount = -1;
	cudaDeviceProp* deviceProps = nullptr;
};