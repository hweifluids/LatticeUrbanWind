#include <iostream>
#include <string_view>

int main(int argc, char* argv[]) {
    if (argc != 2 || std::string_view(argv[1]) != "detect_opencl") {
        std::cerr << "fake launcher expected exactly: detect_opencl\n";
        return 9;
    }
    std::cout << "FAKE_NATIVE_OPENCL_REPORT\n";
    return 3;
}
