#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>

__global__ void LaplaceKernel(float* C1, float* C2, uint64_t x, uint64_t y, uint64_t z);

namespace CudaWrapper {
	void runKernel(dim3 dimGrid, dim3 dimBlock, float* firstArray, float* secondArray, uint64_t x, uint64_t y, uint64_t z);
};