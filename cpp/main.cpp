#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

int main() {
    const int FC1_OUT = 128;
    const int FC1_IN = 784;

    std::vector<float> fc1_weight = load_float_file("data/fc1_weight.bin", FC1_OUT * FC1_IN);

    std::cout << "Loaded fc1_weight values: " << fc1_weight.size() << "\n";

    std::cout << "first 10 fc1_weight values:\n";
    for (int i = 0; i < 10; i ++) {
        std::cout << fc1_weight[i] << "\n";
    }

    std::vector<float> fc1_bias = load_float_file("data/fc1_bias.bin", FC1_OUT);
    std::vector<float> image = load_float_file("data/test_image_0.bin", FC1_IN);

    std::vector<float> hidden(FC1_OUT);

    linear(image, fc1_weight, fc1_bias, hidden, FC1_IN, FC1_OUT);

    std::cout << "first 10 fc1 pre-ReLU values:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << hidden[i] << "\n";
    }

    relu(hidden);

    std::cout << "first 10 hidden post-ReLU values:\n";
    for(int i = 0; i < 10; i++) {
        std::cout << hidden[i] << "\n";
    }

    return 0;
}

