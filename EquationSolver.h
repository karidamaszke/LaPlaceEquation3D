#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Kernel.cuh"

#include <cstdint>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>

class EquationSolver
{
public:
	EquationSolver() = delete;
	EquationSolver(uint16_t arraySize, uint32_t numberOfIterations);
	~EquationSolver();

	void solveEquations();
	void saveResultsToFile(std::string fileName);

private:
	uint32_t numberOfIterations;
	uint64_t x;
	uint64_t y;
	uint64_t z;

	std::unique_ptr<float[]> hostArray;
	std::unique_ptr<float[]> deviceArray;

	void initializeArrays();
	void finiteDifferenceMethodCPU();
	void finiteDifferenceMethodGPU();
	static float getGridDimension(uint64_t dimension, uint32_t blockDimension);
};

