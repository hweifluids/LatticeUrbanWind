#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    const std::string command = argc > 1 ? argv[1] : std::string();
    std::cout << "FAKE_NATIVE_STEP=" << command;
    for (int index = 2; index < argc; ++index) {
        std::cout << "|ARG=" << argv[index];
    }
    std::cout << '\n';
    const char* forceSuccess = std::getenv("LUW_MAKELUW_FAKE_ALL_SUCCESS");
    const bool allSuccess = forceSuccess != nullptr &&
                            std::string(forceSuccess) == "1";
    return !allSuccess && command == "luwcut" ? 9 : 0;
}
