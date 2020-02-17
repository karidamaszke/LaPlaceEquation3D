#include "EquationSolver.h"
#include <memory>
#include <iostream>

// TODO: get params from user
const uint16_t ARRAY_SIZE = 128;
const uint16_t NUMBER_OF_ITERATIONS = 1000;
const std::string FILE_NAME = "LaPlaceEquation.txt";

int main()
{
	std::unique_ptr<EquationSolver> equationSolver;
	try {
		equationSolver = std::make_unique<EquationSolver>(ARRAY_SIZE, NUMBER_OF_ITERATIONS);

		equationSolver->solveEquations();
		equationSolver->saveResultsToFile(FILE_NAME);
	}
	catch (std::exception &e) {
		std::cerr << "Exception! -> " << e.what() << std::endl;
	}

	return 0;
}

