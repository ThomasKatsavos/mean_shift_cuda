#include "mean_shift_cuda_float.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace std;

constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int UNASSIGNED = 0xffffffffu;

// Check for CUDA errors.
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) {
        cerr << "CUDA error: "
             << cudaGetErrorString(error) << '\n';

        exit(EXIT_FAILURE);
    }
}

//Squared distance between two RGB points.
__device__ float squared_distance_gpu(
    const float* a,
    const float* b) {
    float sum = 0.0f;

    for (int d = 0; d < 3; ++d) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }

    return sum;
}

// Attempt 1: one block per seed.

__global__ void shift_seeds_kernel(
    const float* X,
    float* shifted,
    unsigned int N,
    float denominator,
    float epsilon_squared,
    int max_iterations,
    unsigned int* unconverged
) {
    const unsigned int seed = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    __shared__ float center[3];

    //Four sums per thread: weights for R, G, B and total weight.
    __shared__ float sums[4][BLOCK_SIZE];

    __shared__ int stop;
    __shared__ int converged;

    // Initialize this block's center
    if (tid == 0) {
        for (int d = 0; d < 3; ++d) {
            center[d] = X[size_t(seed) * 3 + d];
        }

        stop = 0;
        converged = 0;
    }

    __syncthreads();

    for (int iteration = 0;
         iteration < max_iterations;
         ++iteration) {

        float red_sum = 0.0f;
        float green_sum = 0.0f;
        float blue_sum = 0.0f;
        float weight_sum = 0.0f;

        // Threads divide the original points among themselves.
        for (size_t j = tid; j < N; j += BLOCK_SIZE) {
            const float* point = X + j * 3;

            float dist2 =
                squared_distance_gpu(center, point);

            float weight = expf(-dist2 / denominator);

            red_sum += weight * point[0];
            green_sum += weight * point[1];
            blue_sum += weight * point[2];
            weight_sum += weight;
        }

        // Store each thread's partial sums in shared memory.
        sums[0][tid] = red_sum;
        sums[1][tid] = green_sum;
        sums[2][tid] = blue_sum;
        sums[3][tid] = weight_sum;

        __syncthreads();

        // Tree reduction 
        for (unsigned int stride = BLOCK_SIZE / 2;
             stride > 0;
             stride /= 2) {

            if (tid < stride) {
                for (int channel = 0; channel < 4; ++channel) {
                    sums[channel][tid] +=
                        sums[channel][tid + stride];
                }
            }

            __syncthreads();
        }

        // Thread 0 calculates the new center.
        if (tid == 0) {
            if (sums[3][0] == 0.0f) {
                stop = 1;
            } else {
                float next[3];

                for (int d = 0; d < 3; ++d) {
                    next[d] = sums[d][0] / sums[3][0];
                }

                float movement =
                    squared_distance_gpu(center, next);

                for (int d = 0; d < 3; ++d) {
                    center[d] = next[d];
                }

                converged = movement <= epsilon_squared;
                stop = converged;
            }
        }

        // Everyone must have access to the updated center and stop flag.
        __syncthreads();

        if (stop) {
            break;
        }
    }

    // Save this seed's final position
    if (tid == 0) {
        for (int d = 0; d < 3; ++d) {
            shifted[size_t(seed) * 3 + d] = center[d];
        }

        if (!converged) {
            atomicAdd(unconverged, 1u);
        }
    }
}


// Assign unlabelled endpoints to one selected mode.
// Also find the first endpoint that remains unlabelled.

__global__ void merge_mode_kernel(
    const float* shifted,
    unsigned int* labels,
    float* modes,
    unsigned int* next_seed,
    unsigned int N,
    unsigned int seed,
    unsigned int cluster,
    float radius_squared) {
    const unsigned int tid = threadIdx.x;

    const size_t i =
        size_t(blockIdx.x) * BLOCK_SIZE + tid;

    __shared__ unsigned int candidates[BLOCK_SIZE];

    unsigned int candidate = UNASSIGNED;

    // Exactly one thread records the current representative.
    if (i == 0) {
        for (int d = 0; d < 3; ++d) {
            modes[size_t(cluster) * 3 + d] =
                shifted[size_t(seed) * 3 + d];
        }
    }

    if (i < N && labels[i] == UNASSIGNED) {
        float dist2 = squared_distance_gpu(
            shifted + i * 3,
            shifted + size_t(seed) * 3
        );

        if (dist2 <= radius_squared) {
            labels[i] = cluster;
        } else {
            candidate = static_cast<unsigned int>(i);
        }
    }

    candidates[tid] = candidate;

    __syncthreads();

    // Find the smallest unassigned index inside this block.
    for (unsigned int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            unsigned int other = candidates[tid + stride];

            if (other < candidates[tid]) {
                candidates[tid] = other;
            }
        }

        __syncthreads();
    }

    // Combine the minimum indices from all blocks.
    if (tid == 0) {
        atomicMin(next_seed, candidates[0]);
    }
}

// Allocation, launches, transfers, cleanup.

CudaResult mean_shift_cuda(
    const vector<float>& X,
    float h,
    float epsilon,
    float merge_radius,
    int max_iterations) {
    if (X.empty() || X.size() % 3 != 0 ||
        max_iterations <= 0) {
        throw invalid_argument("Invalid RGB input or iteration limit.");
    }

    for (float value : {h, epsilon, merge_radius}) {
        if (!isfinite(value) || value <= 0.0f ||
            !isfinite(2.0f * value * value) ||
            value * value == 0.0f) {
            throw invalid_argument("Invalid bandwidth or tolerance.");
        }
    }

    // This version expects RGB values normalized to [0,1].
    for (float value : X) {
        if (!isfinite(value) || value < 0.0f || value > 1.0f) {
            throw invalid_argument("Expected RGB values in [0,1].");
        }
    }

    int device = 0;
    cuda_check(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device));

    const size_t count = X.size() / 3;

    if (count >= UNASSIGNED ||
        count > static_cast<size_t>(properties.maxGridSize[0])) {
        throw invalid_argument("Too many pixels for this launch.");
    }

    const unsigned int N = static_cast<unsigned int>(count);

    const size_t rgb_bytes = X.size() * sizeof(float);
    const size_t label_bytes = count * sizeof(unsigned int);

    float* d_X = nullptr;
    float* d_shifted = nullptr;
    float* d_modes = nullptr;

    unsigned int* d_labels = nullptr;
    unsigned int* d_next_seed = nullptr;
    unsigned int* d_unconverged = nullptr;

    cuda_check(cudaMalloc(&d_X, rgb_bytes));
    cuda_check(cudaMalloc(&d_shifted, rgb_bytes));

    // Worst case: one mode per pixel.
    cuda_check(cudaMalloc(&d_modes, rgb_bytes));

    cuda_check(cudaMalloc(&d_labels, label_bytes));
    cuda_check(cudaMalloc(&d_next_seed, sizeof(unsigned int)));
    cuda_check(cudaMalloc(&d_unconverged, sizeof(unsigned int)));

    cuda_check(cudaMemcpy(
        d_X, X.data(), rgb_bytes, cudaMemcpyHostToDevice
    ));

    cuda_check(cudaMemset(
        d_unconverged, 0, sizeof(unsigned int)
    ));

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&stop));

    CudaResult result;
    float milliseconds = 0.0f;


    cout << "GPU: " << properties.name << '\n'
         << "Updating " << N << " seeds..." << endl;

    cuda_check(cudaEventRecord(start));

    shift_seeds_kernel<<<N, BLOCK_SIZE>>>(
        d_X,
        d_shifted,
        N,
        2.0f * h * h,
        epsilon * epsilon,
        max_iterations,
        d_unconverged
    );

    cuda_check(cudaGetLastError());
    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));

    cuda_check(cudaEventElapsedTime(
        &milliseconds, start, stop
    ));

    result.shift_seconds = milliseconds / 1000.0;

    

    cout << "Seed updates completed. Merging..." << endl;

    cuda_check(cudaEventRecord(start));

    // All bits set to UNASSIGNED.
    cuda_check(cudaMemset(d_labels, 0xff, label_bytes));

    const unsigned int blocks = static_cast<unsigned int>(
        (count + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    unsigned int seed = 0;
    unsigned int clusters = 0;

    while (seed != UNASSIGNED) {
        cuda_check(cudaMemset(
            d_next_seed, 0xff, sizeof(unsigned int)
        ));

        merge_mode_kernel<<<blocks, BLOCK_SIZE>>>(
            d_shifted,
            d_labels,
            d_modes,
            d_next_seed,
            N,
            seed,
            clusters,
            merge_radius * merge_radius
        );

        cuda_check(cudaGetLastError());

        // Only one scalar returns to CPU per merge round.
        // This blocking copy also waits for the current round.
        cuda_check(cudaMemcpy(
            &seed,
            d_next_seed,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        ));

        ++clusters;
    }

    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));

    cuda_check(cudaEventElapsedTime(
        &milliseconds, start, stop
    ));

    result.merge_seconds = milliseconds / 1000.0;

    // ---------------- Return results ----------------

    result.labels.resize(count);
    result.modes.resize(size_t(clusters) * 3);

    cuda_check(cudaMemcpy(
        result.labels.data(),
        d_labels,
        label_bytes,
        cudaMemcpyDeviceToHost
    ));

    cuda_check(cudaMemcpy(
        result.modes.data(),
        d_modes,
        result.modes.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    cuda_check(cudaMemcpy(
        &result.unconverged_seeds,
        d_unconverged,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ));

    cuda_check(cudaFree(d_X));
    cuda_check(cudaFree(d_shifted));
    cuda_check(cudaFree(d_modes));
    cuda_check(cudaFree(d_labels));
    cuda_check(cudaFree(d_next_seed));
    cuda_check(cudaFree(d_unconverged));

    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(stop));

    return result;
}
