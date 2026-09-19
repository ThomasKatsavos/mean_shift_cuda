#include "mean_shift.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

// For calculating the squared euclidean distance
double squared_distance(const double* a, const double* b, size_t D) {
    double sum = 0.0;
    for (size_t d = 0; d < D; ++d) {
        const double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

// The mean shift implementation used as a baseline for Project 4. It is one of the simplest(dumbest :) ) implementations of the algorithm, but it will do its job for benchmarking.
vector<double> mean_shift(const vector<double>& X, size_t D,
                          double h, double epsilon, int max_iterations) {
	//Input parameters check
    if (D == 0 || X.empty() || X.size() % D != 0 ||
        !isfinite(h) || h <= 0 || !isfinite(epsilon) || epsilon <= 0 ||
        max_iterations <= 0) {
        throw invalid_argument("Invalid mean shift input or parameters.");
    }

    const size_t N = X.size() / D;
    const double denominator = 2.0 * h * h;
    const double epsilon_squared = epsilon * epsilon;
    vector<double> shifted(X.size());

    //Allocate space once, not once per point/iteration.
    vector<double> center(D), next(D);

    for (size_t i = 0; i < N; ++i) {
	//Use copy_n(), parameters are: (Start index, size count, result)
        copy_n(X.data() + i * D, D, center.data());

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            fill(next.begin(), next.end(), 0.0);
            double weight_sum = 0.0;

            for (size_t j = 0; j < N; ++j) {
                const double* point = X.data() + j * D;
                const double dist2 = squared_distance(center.data(), point, D);
                const double weight = exp(-dist2 / denominator);

                for (size_t d = 0; d < D; ++d) {
                    next[d] += weight * point[d];
                }
                weight_sum += weight;
            }

            if (weight_sum == 0.0) break;
            for (size_t d = 0; d < D; ++d) next[d] /= weight_sum;

            const double movement = squared_distance(center.data(), next.data(), D);
            center.swap(next);
            if (movement <= epsilon_squared) break;
        }
        copy_n(center.data(), D, shifted.data() + i * D);
    }
    return shifted;
}

