#include "mean_shift.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Package for image handling and resize only.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

using namespace std;

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            cerr << "Usage: ./mean_shift_rgb input.jpg output.png [max_side]\n"
                 << "max_side (longer image dimension) defaults to 64; use 0 for original resolution.\n";
            return 1;
        }

        const filesystem::path input_path = argv[1];
        const filesystem::path output_path = argv[2];
        if (output_path.extension() != ".png") {
            throw runtime_error("Use a .png output filename.");
        }
        const filesystem::path preview_path = output_path.parent_path() /
            (output_path.stem().string() + "_input.png");
        if (filesystem::weakly_canonical(input_path) == filesystem::weakly_canonical(output_path) ||
            filesystem::weakly_canonical(input_path) == filesystem::weakly_canonical(preview_path)) {
            throw runtime_error("Choose output paths different from the input image.");
        }

        int max_side = 64;
        if (argc == 4) {
            size_t parsed = 0;
            const string argument = argv[3];
            max_side = stoi(argument, &parsed);
            if (parsed != argument.size() || max_side < 0) {
                throw runtime_error("max_side must be a nonnegative integer.");
            }
        }

        //Decode to 8-bit RGB.
        int original_width = 0, original_height = 0, source_channels = 0;
        unique_ptr<unsigned char, decltype(&stbi_image_free)> raw(
            stbi_load(argv[1], &original_width, &original_height, &source_channels, 3),
            stbi_image_free);
        
	if (!raw) throw runtime_error(string("Cannot load image: ") + stbi_failure_reason());

        //Optional resize, preserving aspect ratio.
        // The exact processed input is saved below.
        int width = original_width, height = original_height;
        if (max_side > 0 && max(width, height) > max_side) {
            const double scale = static_cast<double>(max_side) / max(width, height);
            width = max(1, static_cast<int>(lround(width * scale)));
            height = max(1, static_cast<int>(lround(height * scale)));
        }

        const size_t N = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t D = 3;
        if (N > numeric_limits<uint32_t>::max() ||
            N > vector<double>().max_size() / D ||
            width > numeric_limits<int>::max() / 3) {
            throw runtime_error("Image is too large for this demo.");
        }

        vector<unsigned char> pixels(N * D);
        if (width != original_width || height != original_height) {
            if (!stbir_resize_uint8_srgb(raw.get(), original_width, original_height, 0,
                                        pixels.data(), width, height, 0, STBIR_RGB)) {
                throw runtime_error("Image resize failed.");
            }
        } else {
            copy_n(raw.get(), pixels.size(), pixels.data());
        }
        raw.reset(); 

        if (!stbi_write_png(preview_path.string().c_str(), width, height, 3,
                            pixels.data(), width * 3)) {
            throw runtime_error("Cannot write processed input PNG.");
        }

        //One array in use, with the form of:  R0,G0,B0,R1,G1,B1,..., normalized to [0,1].
        //X[i*3+d] is channel d of pixel i, i = row*width + column.
        vector<double> X(N * D);
        for (size_t k = 0; k < X.size(); ++k) X[k] = pixels[k] / 255.0;

        const double h = 0.10;          //Gaussian bandwidth in normalized RGB.
        const double epsilon = 1e-4;    //Convergence tolerance.
        const double merge_radius = 0.03;
        const int max_iterations = 300;

        cout << "Original: " << original_width << 'x' << original_height
             << "; processing: " << width << 'x' << height << "; N=" << N << '\n'
             << "Gaussian mean shift, RGB only, double precision, h=" << h << endl;

        //Sequential mean shift, followed by a greedy endpoint merge
        //Decode-resize-normalization and image writes are outside these timers
        using Clock = chrono::steady_clock;
        const auto start = Clock::now();
        const auto shifted = mean_shift(X, D, h, epsilon, max_iterations);
        const auto after_shift = Clock::now();

        vector<double> modes;
        vector<uint32_t> labels(N);
        for (size_t i = 0; i < N; ++i) {
            size_t j = 0;
            while (j < modes.size() / D &&
                   squared_distance(shifted.data() + i * D, modes.data() + j * D, D)
                       > merge_radius * merge_radius) {
                ++j;
            }
            if (j == modes.size() / D) {
                const double* point = shifted.data() + i * D;
                modes.insert(modes.end(), point, point + D);
            }
            labels[i] = static_cast<uint32_t>(j);
        }
        const auto after_merge = Clock::now();

        //Color every pixel using its cluster's representative mode
        //Reuse the input byte buffer- pixel positions stay unchanged
        for (size_t i = 0; i < N; ++i) {
            for (size_t d = 0; d < D; ++d) {
                const double value = modes[static_cast<size_t>(labels[i]) * D + d];
                pixels[i * D + d] = static_cast<unsigned char>(
                    lround(clamp(value, 0.0, 1.0) * 255.0));
            }
        }
        if (!stbi_write_png(argv[2], width, height, 3, pixels.data(), width * 3)) {
            throw runtime_error("Cannot write output PNG.");
        }

        cout << fixed << setprecision(6)
             << "Clusters: " << modes.size() / D << '\n'
             << "Mean shift (s): " << chrono::duration<double>(after_shift - start).count() << '\n'
             << "Merge (s): " << chrono::duration<double>(after_merge - after_shift).count() << '\n'
             << "Clustering total (s): " << chrono::duration<double>(after_merge - start).count() << '\n'
             << "Processed input: " << preview_path.string() << '\n'
             << "Result: " << output_path.string() << '\n';
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

