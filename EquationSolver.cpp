#include "EquationSolver.h"

EquationSolver::EquationSolver(uint16_t arraySize, uint32_t numberOfIterations) 
	: x(arraySize), y(arraySize), z(arraySize), numberOfIterations(numberOfIterations)
{
	this->hostArray = std::make_unique<float[]>(static_cast<uint32_t>(x * y * z));
	this->deviceArray = std::make_unique<float[]>(static_cast<uint32_t>(x * y * z));
}

EquationSolver::~EquationSolver()
{
}

void EquationSolver::solveEquations()
{
	// CPU
	std::cout << "Start calculation on CPU...." << std::endl;

	auto startCPU = std::chrono::system_clock::now();
	this->finiteDifferenceMethodCPU();
	auto endCPU = std::chrono::system_clock::now();

	std::cout << "Computations on CPU takes " << std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count() << " ms.\n" << std::endl;


	//GPU
	std::cout << "Start calculation on GPU...." << std::endl;

	auto startGPU = std::chrono::system_clock::now();
	this->finiteDifferenceMethodGPU();
	auto endGPU = std::chrono::system_clock::now();

	std::cout << "Computations on GPU takes " << std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - startGPU).count() << " ms.\n" << std::endl;
}

void EquationSolver::saveResultsToFile(std::string fileName)
{
	std::ofstream resultsFile(fileName, std::ios::out);

	for (uint64_t i = 0; i < z; i++)
	{
		for (uint64_t j = 0; j < x; j++)
		{
			for (uint64_t k = 0; k < y; k++)
			{
				uint32_t index = i * x * y + j * x + k;
				resultsFile << deviceArray[index] << "\t";
			}
			resultsFile << std::endl;
		}
		resultsFile << std::endl;
	}

	resultsFile.close();
}

void EquationSolver::initializeArrays()
{
	for (uint64_t i = 0; i < z; i++)
	{
		for (uint64_t j = 0; j < x; j++)
		{
			for (uint64_t k = 0; k < y; k++)
			{
				uint32_t index = i * x * y + j * x + k;
				hostArray[index] = 0.0;
				deviceArray[index] = 0.0;
			}
		}
	}

	// set the boundary conditions
	for (uint64_t j = 0; j < x; j++)
	{
		for (uint64_t k = 0; k < y; k++)
		{
			uint32_t index = j * x + k;
			hostArray[index] = 50.0;
			deviceArray[index] = 50.0;
		}
	}
	for (uint64_t j = 0; j < x; j++)
	{
		for (uint64_t k = 0; k < y; k++)
		{
			uint32_t index = x * y * (z - 1) + j * y + k;
			hostArray[index] = -50.0;
			deviceArray[index] = -50.0;
		}
	}
}

void EquationSolver::finiteDifferenceMethodCPU()
{
	uint32_t iterations = 0;
	while (iterations < this->numberOfIterations)
	{
		for (uint64_t i = 1; i < z - 5; i++)
		{
			for (uint64_t j = 1; j < x - 5; j++)
			{
				for (uint64_t k = 1; k < y - 5; k++)
				{
					uint64_t P = i * x * y + j * x + k;
					uint64_t S = i * x * y + (j + 1) * x + k;
					uint64_t N = i * x * y + (j - 1) * x + k;
					uint64_t E = i * x * y + j * x + (k + 1);
					uint64_t W = i * x * y + j * x + (k - 1);
					uint64_t U = (i + 1) * x * y + j * x + k;
					uint64_t D = (i - 1) * x * y + j * x + k;

					hostArray[P] = 0.166666667f *
						(hostArray[E] + hostArray[W] + hostArray[N] + hostArray[S] + hostArray[U] + hostArray[D]);
				}
			}
		}
		iterations++;
	}
}

void EquationSolver::finiteDifferenceMethodGPU()
{
	auto deleter = [&](float* ptr) { cudaFree(ptr); };
	std::unique_ptr<float[], decltype(deleter)> deviceCalculationArray1(new float[x * y * z], deleter);
	std::unique_ptr<float[], decltype(deleter)> deviceCalculationArray2(new float[x * y * z], deleter);

	cudaMalloc((void**)&deviceCalculationArray1, x * y * z * sizeof(float));
	cudaMalloc((void**)&deviceCalculationArray2, x * y * z * sizeof(float));

	cudaMemcpy(deviceCalculationArray1.get(), deviceArray.get(), x * y * z * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceCalculationArray2.get(), deviceArray.get(), x * y * z * sizeof(float), cudaMemcpyHostToDevice);

	uint32_t ThreadsPerBlock = 10;
	dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock, ThreadsPerBlock);
	dim3 dimGrid(getGridDimension(x, dimBlock.x), getGridDimension(y, dimBlock.y), getGridDimension(z, dimBlock.z));

	int iterations = 0;
	while (iterations < this->numberOfIterations)
	{
		CudaWrapper::runKernel(dimGrid, dimBlock, deviceCalculationArray1.get(), deviceCalculationArray2.get(), x, y, z);
		CudaWrapper::runKernel(dimGrid, dimBlock, deviceCalculationArray2.get(), deviceCalculationArray1.get(), x, y, z);
		iterations += 2;
	}

	cudaDeviceSynchronize();
	cudaMemcpy(deviceArray.get(), deviceCalculationArray2.get(), x * y * z * sizeof(float), cudaMemcpyDeviceToHost);
}

float EquationSolver::getGridDimension(uint64_t dimension, uint32_t blockDimension)
{
	auto fDimension = static_cast<float>(dimension);
	auto fBlockDimension = static_cast<float>(blockDimension);

	return ceil(fDimension / fBlockDimension);
}
