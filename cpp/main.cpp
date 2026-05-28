#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

    return 0;
}

