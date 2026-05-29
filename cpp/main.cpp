#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>

// fc1.weight first 10:
// tensor([-0.0003,  0.0192, -0.0294, -0.0263, -0.0138,  0.0096, -0.0007,  0.0283, -0.0032,  0.0095])

void linear(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    std::vector<float>& output,
    int in_features,
    int out_features
) {
    for (int i = 0; i < out_features; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_features; j++) {
            sum += weight[i * in_features + j] * input[j];
        }
        output[i] = sum;
    }
}

void relu(std::vector<float>& x) {
    for (float& value : x) {
        if (value < 0.0f) {
            value = 0.0f;
        }
    }
}

std::vector<float> load_float_file(const std::string& path, int expected_count) {
    std::vector<float> data(expected_count);

    std::ifstream file(path, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open file: " << path << "\n";
        std::exit(1);
    }

    file.read(
        reinterpret_cast<char*>(data.data()),
        expected_count * sizeof(float)
    );

    if (!file) {
        std::cerr << "Failed to read expected bytes from: " << path << "\n";
        std::exit(1);
    }

    return data;
}

std::vector<int64_t> load_int64_file(const std::string& path, int expected_count) {

    std::vector<int64_t> data(expected_count);
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open file: " << path << "\n";
        std::exit(1);
    }

    file.read(
        reinterpret_cast<char*>(data.data()), 
        expected_count * sizeof(int64_t)
    );

    if (!file) {
        std::cerr << "Failed to read expected bytes from: " << path << "\n";
        std::exit(1);
    }

    return data;

}

int argmax(const std::vector<float>& values) {
    int best_index = 0;
    float best_value = values[0];

    for (int i = 1; i < static_cast<int>(values.size()); i++) {
        if (values[i] > best_value) {
            best_value = values[i];
            best_index = i;
        }
    }
    return best_index;
}

int main() {
    const int FC1_OUT = 128;
    const int FC1_IN = 784;
    const int FC2_IN = 128;
    const int FC2_OUT = 10;
    const int NUM_TEST = 10000;

    std::vector<float> test_images = load_float_file("data/test_images.bin", NUM_TEST*FC1_IN);
    std::vector<int64_t> test_labels = load_int64_file("data/test_labels.bin", NUM_TEST);

    std::vector<float> fc1_weight = load_float_file("data/fc1_weight.bin", FC1_OUT * FC1_IN);
    std::vector<float> fc2_weight = load_float_file("data/fc2_weight.bin", FC2_OUT * FC2_IN);


    std::vector<float> fc1_bias = load_float_file("data/fc1_bias.bin", FC1_OUT);
    std::vector<float> fc2_bias = load_float_file("data/fc2_bias.bin", FC2_OUT);


    std::vector<float> image(FC1_IN);
    std::vector<float> hidden(FC1_OUT);
    std::vector<float> logits(FC2_OUT);

    auto start = std::chrono::high_resolution_clock::now();

    int correct = 0;

    for(int n = 0; n < NUM_TEST; n++) {
        for (int j = 0; j < FC1_IN; j++) {
            image[j] = test_images[n * FC1_IN + j];
        }

        linear(image, fc1_weight, fc1_bias, hidden, FC1_IN, FC1_OUT);
        relu(hidden);
        linear(hidden, fc2_weight, fc2_bias, logits, FC2_IN, FC2_OUT);

        int prediction = argmax(logits);

        if (prediction == test_labels[n]) {
            correct++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    float accuracy = static_cast<float>(correct) / NUM_TEST;
    double total_seconds = elapsed.count();
    double ms_per_image = (total_seconds * 1000.0) / NUM_TEST;
    double images_per_second = NUM_TEST / total_seconds;

    std::cout << "correct: " << correct << " / " << NUM_TEST << "\n";
    std::cout << "accuracy: " << accuracy << "\n";
    std::cout << "total time: " << total_seconds << " seconds\n";
    std::cout << "ms per image: " << ms_per_image << "\n";
    std::cout << "images/sec: " << images_per_second << "\n";

    return 0;
}

