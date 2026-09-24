#pragma once

#include <vector>

struct CudaResult {
    std::vector<float> modes;
    std::vector<unsigned int> labels;

    unsigned int unconverged_seeds = 0;

    double shift_seconds = 0.0;
    double merge_seconds = 0.0;
};

CudaResult mean_shift_cuda(
    const std::vector<float>& X,
    float h,
    float epsilon,
    float merge_radius,
    int max_iterations
);
