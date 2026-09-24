CXX = g++
CXXFLAGS = -std=c++17 -O3
INCLUDES = -Ilibs

TARGET = mean_shift_gpu_serial

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cpp mean_shift.cpp mean_shift.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.cpp mean_shift.cpp -o $@

clean:
	rm -f $(TARGET)
