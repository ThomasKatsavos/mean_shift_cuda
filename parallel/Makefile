NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -arch=sm_80
INCLUDES = -Ilibs

TARGETS = mean_shift_gpu mean_shift_gpu_warp mean_shift_gpu_thread

.PHONY: all clean

all: $(TARGETS)

mean_shift_gpu: main_cuda.cpp mean_shift_cuda.cu mean_shift_cuda.h
	$(NVCC) $(CXXFLAGS) $(INCLUDES) main_cuda.cpp mean_shift_cuda.cu -o $@

mean_shift_gpu_warp: main_cuda.cpp mean_shift_cuda_warp.cu mean_shift_cuda.h
	$(NVCC) $(CXXFLAGS) $(INCLUDES) main_cuda.cpp mean_shift_cuda_warp.cu -o $@

mean_shift_gpu_thread: main_cuda.cpp mean_shift_cuda_thread.cu mean_shift_cuda.h
	$(NVCC) $(CXXFLAGS) $(INCLUDES) main_cuda.cpp mean_shift_cuda_thread.cu -o $@

clean:
	rm -f $(TARGETS)
