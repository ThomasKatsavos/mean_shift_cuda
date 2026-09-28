#include "mean_shift_cuda.h"

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

// Check for CUDA errors
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) {
        cerr << "CUDA error: "
             << cudaGetErrorString(error) << '\n';

        exit(EXIT_FAILURE);
    }
}

//Squared distance between two RGB points.
__device__ double squared_distance_gpu(
    const double* a,
    const double* b) {
    double sum = 0.0;

    for (int d = 0; d < 3; ++d) {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }

    return sum;
}

// Attempt 2: one warp per seed.


constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

static_assert(BLOCK_SIZE % WARP_SIZE == 0,
              "BLOCK_SIZE must be a multiple of 32.");

__global__ void shift_seeds_warp_kernel(
    const double* X,
    double* shifted,
    unsigned int N,
    double denominator,
    double epsilon_squared,
    int max_iterations,
    unsigned int* unconverged
) {
    const unsigned int tid = threadIdx.x;

    // Position within the warp: 0..31.
    const unsigned int lane = tid % WARP_SIZE;

    // Position of this warp within the block: 0..7.
    const unsigned int warp = tid / WARP_SIZE;

    // Each warp handles one seed.
    const size_t seed =
        static_cast<size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;

    const bool valid_seed = seed < N;

    // All 32 lanes participate in the shuffle operations.
    constexpr unsigned int FULL_MASK = 0xffffffffu;

    // One tile contains BLOCK_SIZE RGB points.
    // Separate rows for R, G and B.
    __shared__ double tile[3][BLOCK_SIZE];

    // Each thread keeps a copy of its warp's center.
    double center[3] = {0.0, 0.0, 0.0};

    // Only lane 0 loads the initial center.
    if (lane == 0 && valid_seed) {
        for (int d = 0; d < 3; ++d) {
            center[d] = X[3 * seed + d];
        }
    }

    // Broadcast the center from lane 0 to its entire warp.
    for (int d = 0; d < 3; ++d) {
        center[d] = __shfl_sync(FULL_MASK, center[d], 0);
    }

    // Identical across all lanes of the same warp.
    int active = valid_seed ? 1 : 0;

    // Only lane 0 updates and uses this flag.
    int converged = 0;

    for (int iteration = 0;
         iteration < max_iterations;
         ++iteration) {

        // Block-wide synchronization and termination check.
        // Every thread participates, including inactive warps.
        if (__syncthreads_count(active) == 0) {
            break;
        }

        double red_sum = 0.0;
        double green_sum = 0.0;
        double blue_sum = 0.0;
        double weight_sum = 0.0;

        // Scan the ORIGINAL dataset, one tile at a time.
        for (size_t base = 0; base < N; base += BLOCK_SIZE) {

            const size_t remaining = static_cast<size_t>(N) - base;

            const unsigned int tile_count =
                static_cast<unsigned int>(
                    remaining < BLOCK_SIZE ? remaining : BLOCK_SIZE
                );

            // Each thread loads one RGB point.
            // Inactive warps must also help load the tile.
            if (tid < tile_count) {
                const size_t point_index = base + tid;

                for (int d = 0; d < 3; ++d) {
                    tile[d][tid] = X[3 * point_index + d];
                }
            }

            // Wait until the entire tile has been loaded.
            __syncthreads();

            if (active) {
                // Each warp scans the same tile.
                // Its lanes divide the points among themselves.
                for (unsigned int j = lane;
                     j < tile_count;
                     j += WARP_SIZE) {

                    double dist_squared = 0.0;

                    for (int d = 0; d < 3; ++d) {
                        const double diff = center[d] - tile[d][j];
                        dist_squared += diff * diff;
                    }

                    const double weight =
                        exp(-dist_squared / denominator);

                    red_sum   += weight * tile[0][j];
                    green_sum += weight * tile[1][j];
                    blue_sum  += weight * tile[2][j];
                    weight_sum += weight;
                }
            }

            // All readers must finish before the next tile
            // overwrites shared memory.
            __syncthreads();
        }

        // Reduce the four sums independently within each warp.
        // All lanes execute these operations, even inactive ones.
        for (int offset = WARP_SIZE / 2;
             offset > 0;
             offset /= 2) {

            red_sum +=
                __shfl_down_sync(FULL_MASK, red_sum, offset);

            green_sum +=
                __shfl_down_sync(FULL_MASK, green_sum, offset);

            blue_sum +=
                __shfl_down_sync(FULL_MASK, blue_sum, offset);

            weight_sum +=
                __shfl_down_sync(FULL_MASK, weight_sum, offset);
        }

        // Lane 0 now holds the complete sums for this seed.
        if (lane == 0 && active) {
            if (weight_sum == 0.0) {
                active = 0;
            } else {
                double next[3];

                next[0] = red_sum / weight_sum;
                next[1] = green_sum / weight_sum;
                next[2] = blue_sum / weight_sum;

                const double movement =
                    squared_distance_gpu(center, next);

                for (int d = 0; d < 3; ++d) {
                    center[d] = next[d];
                }

                converged = movement <= epsilon_squared;

                if (converged) {
                    active = 0;
                }
            }
        }

        // All lanes receive the new center and active state.
        for (int d = 0; d < 3; ++d) {
            center[d] = __shfl_sync(FULL_MASK, center[d], 0);
        }

        active = __shfl_sync(FULL_MASK, active, 0);
    }

    // One thread writes the result of each valid seed.
    if (lane == 0 && valid_seed) {
        for (int d = 0; d < 3; ++d) {
            shifted[3 * seed + d] = center[d];
        }

        if (!converged) {
            atomicAdd(unconverged, 1u);
        }
    }
}
        

// Assign unlabelled endpoints to one selected mode.
// Also find the first endpoint that remains unlabelled.

__global__ void merge_mode_kernel(
    const double* shifted,
    unsigned int* labels,
    double* modes,
    unsigned int* next_seed,
    unsigned int N,
    unsigned int seed,
    unsigned int cluster,
    double radius_squared) {
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
        double dist2 = squared_distance_gpu(
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
    const vector<double>& X,
    double h,
    double epsilon,
    double merge_radius,
    int max_iterations) {
    if (X.empty() || X.size() % 3 != 0 ||
        max_iterations <= 0) {
        throw invalid_argument("Invalid RGB input or iteration limit.");
    }

    for (double value : {h, epsilon, merge_radius}) {
        if (!isfinite(value) || value <= 0.0 ||
            !isfinite(2.0 * value * value) ||
            value * value == 0.0) {
            throw invalid_argument("Invalid bandwidth or tolerance.");
        }
    }

    // This version expects RGB values normalized to [0,1].
    for (double value : X) {
        if (!isfinite(value) || value < 0.0 || value > 1.0) {
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

    const size_t rgb_bytes = X.size() * sizeof(double);
    const size_t label_bytes = count * sizeof(unsigned int);

    double* d_X = nullptr;
    double* d_shifted = nullptr;
    double* d_modes = nullptr;

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



    // Each block handles 8 seeds.
    // Round up to cover the final partial block.
    const unsigned int shift_blocks =  N / WARPS_PER_BLOCK + (N % WARPS_PER_BLOCK != 0);

    cuda_check(cudaEventRecord(start));


    shift_seeds_warp_kernel<<<shift_blocks, BLOCK_SIZE>>>(
    	d_X,
    	d_shifted,
    	N,
    	2.0 * h * h,
    	epsilon * epsilon,
    	max_iterations,
    	d_unconverged);

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
        result.modes.size() * sizeof(double),
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
