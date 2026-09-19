#pragma once

#include <cstddef>
#include <vector>


double squared_distance(const double* a, const double* b, std::size_t D);

//Gaussian-kernel mean shift with input X.
//No radius cutoff or neighbor approximation.
std::vector<double> mean_shift(const std::vector<double>& X, std::size_t D,
                               double h, double epsilon, int max_iterations);

