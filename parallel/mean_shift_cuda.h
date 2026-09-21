#pragma once

#include <vector>

struct CudaResult {
    std::vector<double> modes;
    std::vector<unsigned int> labels;

    unsigned int unconverged_seeds = 0;

    double shift_seconds = 0.0;
    double merge_seconds = 0.0;
};

CudaResult mean_shift_cuda(
    const std::vector<double>& X,
    double h,
    double epsilon,
    double merge_radius,
    int max_iterations
);
