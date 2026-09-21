#include "mean_shift_cuda.h"

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

// Image loading.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// PNG writing.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Image resizing.
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

using namespace std;

int main(int argc, char** argv) {
    try {
        // Arguments: input.jpeg output.png [max_side]
        if (argc < 3 || argc > 4) {
            cerr << "Usage: ./mean_shift_gpu "
                 << "input.jpeg output.png [max_side]\n"
                 << "Default max_side: 64. "
                 << "Use 0 for original resolution.\n";

            return 1;
        }

        const filesystem::path input_path = argv[1];
        const filesystem::path output_path = argv[2];

        if (output_path.extension() != ".png") {
            throw runtime_error(
                "The output filename must end in .png"
            );
        }

        // Example: result.png -> result_input.png
        const filesystem::path preview_path =
            output_path.parent_path() /
            (output_path.stem().string() + "_input.png");

        // Avoid overwriting the original image.
        if (filesystem::weakly_canonical(input_path) ==
                filesystem::weakly_canonical(output_path) ||
            filesystem::weakly_canonical(input_path) ==
                filesystem::weakly_canonical(preview_path)) {
            throw runtime_error(
                "Input and output paths must be different."
            );
        }


        //Read the optional resize limit.
        int max_side = 64;

        if (argc == 4) {
            const string argument = argv[3];
            size_t parsed = 0;

            max_side = stoi(argument, &parsed);

            if (parsed != argument.size() || max_side < 0) {
                throw runtime_error(
                    "max_side must be a nonnegative integer."
                );
            }
        }

        // Load the image as tightly packed 8-bit RGB.
        int original_width = 0;
        int original_height = 0;
        int source_channels = 0;

        // Automatically calls stbi_image_free when released.
        unique_ptr<unsigned char, decltype(&stbi_image_free)> raw(
            stbi_load(
                argv[1],
                &original_width,
                &original_height,
                &source_channels,
                3
            ),
            stbi_image_free
        );

        if (!raw) {
            throw runtime_error(
                "Cannot load image: " + input_path.string()
            );
        }


        //Calculate the processing resolution
        int width = original_width;
        int height = original_height;

        if (max_side > 0 && max(width, height) > max_side) {
            const double scale =
                static_cast<double>(max_side) /
                max(width, height);

            width = max(
                1,
                static_cast<int>(lround(width * scale))
            );

            height = max(
                1,
                static_cast<int>(lround(height * scale))
            );
        }

        const size_t N =
            static_cast<size_t>(width) *
            static_cast<size_t>(height);

        const size_t D = 3;

        if (N > numeric_limits<uint32_t>::max() ||
            N > vector<double>().max_size() / D ||
            width > numeric_limits<int>::max() / 3) {
            throw runtime_error("Image is too large.");
        }

       //RGB values buffer: R0,G0,B0,R1,G1,B1,...
        vector<unsigned char> pixels(N * D);

        if (width != original_width ||
            height != original_height) {

            unsigned char* resized = stbir_resize_uint8_srgb(
                raw.get(),
                original_width,
                original_height,
                0,
                pixels.data(),
                width,
                height,
                0,
                STBIR_RGB
            );

            if (!resized) {
                throw runtime_error("Image resize failed.");
            }
        } else {
            copy_n(raw.get(), pixels.size(), pixels.data());
        }

        // Release the original full-resolution image.
        raw.reset();

        // Save the image that will be processed
        if (!stbi_write_png(
                preview_path.string().c_str(),
                width,
                height,
                3,
                pixels.data(),
                width * 3)) {
            throw runtime_error(
                "Cannot save the processed input image."
            );
        }

        
        //Convert RGB bytes to doubles in [0,1]
        vector<double> X(N * D);

        for (size_t k = 0; k < X.size(); ++k) {
            X[k] = pixels[k] / 255.0;
        }

        //Same parameters as the serial image implementation.
        const double h = 0.10;
        const double epsilon = 1e-4;
        const double merge_radius = 0.03;
        const int max_iterations = 300;

        cout << "Original image: "
             << original_width << " x "
             << original_height << '\n'
             << "Processing image: "
             << width << " x " << height << '\n'
             << "Pixels: " << N << '\n'
             << "Gaussian bandwidth: " << h << endl;

         
	//CUDA mean shift + merging
        using Clock = chrono::steady_clock;

        const auto start = Clock::now();

        const CudaResult result = mean_shift_cuda(
            X,
            h,
            epsilon,
            merge_radius,
            max_iterations
        );

        const auto finish = Clock::now();

        const auto& modes = result.modes;
        const auto& labels = result.labels;


        //Reconstruct the output image.
        //Each pixel gets its cluster's representative color
        for (size_t i = 0; i < N; ++i) {
            const size_t cluster = labels[i];

            for (size_t d = 0; d < D; ++d) {
                const double value = modes[cluster * D + d];

                pixels[i * D + d] =
                    static_cast<unsigned char>(
                        lround(clamp(value, 0.0, 1.0) * 255.0)
                    );
            }
        }


        //Save the result as PNG
        if (!stbi_write_png(
                output_path.string().c_str(),
                width,
                height,
                3,
                pixels.data(),
                width * 3)) {
            throw runtime_error("Cannot save the output image.");
        }

        // ----------------------------------------------------
        // Print results and timing
        // ----------------------------------------------------

        const double total_seconds =
            chrono::duration<double>(finish - start).count();

        cout << fixed << setprecision(6)
             << "Clusters: " << modes.size() / D << '\n'
             << "GPU mean shift (s): "
             << result.shift_seconds << '\n'
             << "GPU merge workflow (s): "
             << result.merge_seconds << '\n'
             << "CUDA total including setup/transfers (s): "
             << total_seconds << '\n'
             << "Seeds without convergence: "
             << result.unconverged_seeds << '/' << N << '\n'
             << "Processed input: "
             << preview_path.string() << '\n'
             << "Result: "
             << output_path.string() << '\n';

        if (result.unconverged_seeds != 0) {
            cerr << "Warning: some seeds stopped without "
                 << "convergence; their last positions were used.\n";
        }

        return 0;
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
