NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -arch=sm_80 -DUSE_FLOAT
INCLUDES = -Ilibs

TARGET = mean_shift_gpu_float

.PHONY: all clean

all: $(TARGET)

$(TARGET): main_cuda_float.cpp mean_shift_cuda_float.cu mean_shift_cuda_float.h
	$(NVCC) $(CXXFLAGS) $(INCLUDES) main_cuda_float.cpp mean_shift_cuda_float.cu -o $@

clean:
	rm -f $(TARGET)
