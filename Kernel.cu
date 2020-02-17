#include "Kernel.cuh"

__global__ void LaplaceKernel(float* firstArray, float* secondArray, uint64_t x, uint64_t y, uint64_t z)
{
	uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t k = blockIdx.y * blockDim.y + threadIdx.y;
	uint64_t i = blockIdx.z * blockDim.z + threadIdx.z;


	uint64_t P = i * x * y + j * x + k;
	uint64_t S = i * x * y + (j + 1) * x + k;
	uint64_t N = i * x * y + (j - 1) * x + k;
	uint64_t E = i * x * y + j * x + (k + 1);
	uint64_t W = i * x * y + j * x + (k - 1);
	uint64_t U = (i + 1) * x * y + j * x + k;
	uint64_t D = (i - 1) * x * y + j * x + k;

	if (i > 0 && i < z - 1 && j > 0 && j < x - 1 && k > 0 && k < y - 1)
	{
		secondArray[P] = 0.166666667 * (firstArray[E] + firstArray[W] + firstArray[N] + firstArray[S] + firstArray[U] + firstArray[D]);
	}
}

namespace CudaWrapper {
	void runKernel(dim3 dimGrid, dim3 dimBlock, float* firstArray, float* secondArray, uint64_t x, uint64_t y, uint64_t z)
	{
		LaplaceKernel<<<dimGrid, dimBlock>>> (firstArray, secondArray, x, y, z);
	}
}