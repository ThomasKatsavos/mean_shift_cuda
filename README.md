# Mean Shift: Serial and CUDA Implementations

## Introduction

This project implements **color-domain Mean Shift clustering for RGB images**, with both a serial CPU implementation and several parallel CUDA implementations.

The goal is to investigate how the computationally intensive Mean Shift algorithm can be accelerated using different GPU parallelization strategies while preserving the same underlying RGB clustering procedure.

Each image pixel is represented as a three-dimensional RGB vector:

$$
x_i = (R_i, G_i, B_i)
$$

The project includes a straightforward serial implementation that serves as the baseline, with multiple CUDA implementations designed to exploit GPU parallelism.


## Description

### RGB Mean Shift

For every input pixel, Mean Shift initializes a seed at its RGB value and iteratively moves this seed towards a region of higher color density.

At each iteration, the Euclidean distance(in feature space - color domain) between the current center and every input RGB vector is calculated. A Gaussian kernel is then used to assign a weight to each point:

$$
w_{ij} =
\exp\left(
-\frac{\|y_i-x_j\|^2}{2h^2}
\right)
$$

The new center is calculated as the weighted mean of all RGB vectors:

$$
y_i =
\frac{\sum_j w_{ij}x_j}
{\sum_j w_{ij}}
$$

The process continues until the center converges or the maximum number of iterations is reached.

Since each pixel is treated as an independent seed, the algorithm contains a significant amount of parallel work.

### Computational Complexity

For \(N\) pixels and \(T\) Mean Shift iterations, the dominant computational complexity is:

$$
O(N^2T)
$$

The RGB dimensions are fixed to \(D=3\), so it is treated as a constant.

The quadratic dependence on the number of pixels makes Mean Shift particularly expensive for high-resolution images and provides a suitable workload for GPU acceleration.

### Implementations

The project contains the following implementations:

| Implementation          | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| `mean_shift_gpu_serial` | Serial CPU baseline                                  |
| `mean_shift_gpu`        | CUDA block-level implementation                         |
| `mean_shift_gpu_warp`   | CUDA warp-level implementation                         |
| `mean_shift_gpu_thread` | CUDA thread-level implementation                       |
| `mean_shift_gpu_float`  | CUDA implementation using single-precision floating point |

The CUDA implementations use different mappings between Mean Shift seeds and GPU threads while maintaining the same underlying algorithm.


## Project Structure

```text

.
├── Makefile
├── README.md
├── logs_diagrams
│   ├── gpu_full_ship_input.png
│   ├── gpu_main_input.png
│   ├── result-2625810.log
│   ├── result-2625813.log
│   ├── result-2625814.log
│   ├── result-2626358.log
│   ├── result-2627875.log
│   ├── result-2628028.log
│   ├── result-2628949.log
│   ├── result_gpu128.png
│   ├── result_gpu256.png
│   ├── result_gpu512.png
│   ├── result_gpu64t.png
│   ├── result_gpu_full_ship.png
│   ├── result_gpu_fullt.png
│   ├── speedups_mean.png
│   └── times_mean.png
├── parallel
│   ├── Makefile
│   ├── libs
│   │   ├── stb_image.h
│   │   ├── stb_image_resize2.h
│   │   └── stb_image_write.h
│   ├── main_cuda.cpp
│   ├── mean_shift_cuda.cu
│   ├── mean_shift_cuda.h
│   ├── mean_shift_cuda_thread.cu
│   ├── mean_shift_cuda_warp.cu
│   ├── photo.jpeg
│   ├── run.sbatch
│   ├── run_thread.sbatch
│   └── run_warp.sbatch
├── parallel_float
│   ├── Makefile
│   ├── libs
│   │   ├── stb_image.h
│   │   ├── stb_image_resize2.h
│   │   └── stb_image_write.h
│   ├── main_cuda_float.cpp
│   ├── mean_shift_cuda_float.cu
│   ├── mean_shift_cuda_float.h
│   └── run.sbatch
└── sequential
    ├── Makefile
    ├── libs
    │   ├── stb_image.h
    │   ├── stb_image_resize2.h
    │   └── stb_image_write.h
    ├── main.cpp
    ├── mean_shift.cpp
    ├── mean_shift.h
    ├── photo2.jpeg
    └── seq.sbatch

```

---

## Installation

### Requirements

The project requires:

* Access to Aristotelis Cluster
* A C++17-compatible compiler (g++)
* NVIDIA CUDA Toolkit (nvcc)


The CUDA implementations are compiled using:

```bash
-std=c++17 -O3 -arch=sm_80
```

### Clone the repository

```bash
git clone https://github.com/ThomasKatsavos/mean_shift_cuda.git
cd mean_shift_cuda
```

### Load required modules

```bash
module load gcc/12.2.0 cuda
```

### Build the project

The repository contains a root Makefile which invokes the Makefile in each implementation directory.

Build all implementations with:

```bash
make
```

This produces:

```text
parallel/mean_shift_gpu
parallel/mean_shift_gpu_warp
parallel/mean_shift_gpu_thread

parallel_float/mean_shift_gpu_float

sequential/mean_shift_gpu_serial
```

To remove the generated executables:

```bash
make clean
```

Individual implementations can also be built separately:

```bash
make -C parallel
make -C parallel_float
make -C sequential
```


## Running

All implementations follow the same general command-line interface:

```bash
./executable input_image output_image max_pixel_dimension
```

For example:

```bash
./mean_shift_gpu input.jpeg output.png 512
```

The third argument specifies the maximum image dimension used for the execution. If there is no such argument, 64 is chosen by default, whereas if 0 is inserted, the full dimensions image is used. 

There are ready scripts for submitting the executables as Slurm jobs on Aristotelis. The command for submitting a Slurm script named 'run.sbatch' for a specific execution is:

```bash
sbatch run.sbatch
```
Each subdirectory contains already configured, ready-to-run Slurm scripts, that can be executed from the terminal on Aristotelis. The serial implementation is set to run on the 'ondemand' partition, whereas the GPU kernels on 'ampere'.
To execute each one of those scripts with the command above suggests, the user must be in the respective working directory. The output of the execution
is an automatically generated log file, as well as the produced processed image, all in the same directory with the executable and the Slurm script.

## Benchmarking

The implementations can be compared using the same input image and maximum dimensions.

Suggested test dimensions are:

```text
64
128
256
512
Full resolution
```

For each configuration, the execution time can be recorded and compared between the serial and CUDA implementations.

The speedup relative to the serial implementation is calculated as:

$$
Speedup =
\frac{T_{serial}}
{T_{parallel}}
$$

This allows the effect of the different CUDA parallelization strategies to be evaluated as the input image size increases.

---

## Notes

The serial implementation is intentionally straightforward and serves primarily as a correctness and performance baseline. The CUDA implementations retain the same RGB Mean Shift procedure while reorganizing its computational workload to exploit GPU parallelism.

The project focuses exclusively on the **RGB feature space**. Pixel spatial coordinates (x,y) are not included in the Mean Shift feature vector.
