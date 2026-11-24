#include "luw/detect_opencl_command.hpp"

#include <iostream>
#include <sstream>
#include <string>

#ifndef LUW_HAS_OPENCL
#define LUW_HAS_OPENCL 0
#endif

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_contains(const std::string& text,
                     const std::string& needle,
                     const char* message) {
    expect(text.find(needle) != std::string::npos, message);
}

} // namespace

int main() {
    using luw::cli::opencl_diagnostics_detail::device_type_string;
    using luw::cli::opencl_diagnostics_detail::human_bytes;
    using luw::cli::opencl_diagnostics_detail::platform_enumeration_exit;

    expect(human_bytes(0) == "0.00 B", "zero bytes formatting");
    expect(human_bytes(1023) == "1023.00 B", "byte boundary formatting");
    expect(human_bytes(1024) == "1.00 KB", "KiB formatting");
    expect(human_bytes(22ULL * 1024ULL * 1024ULL * 1024ULL) == "22.00 GB",
           "GiB formatting");
    expect(human_bytes(5ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL) == "5.00 TB",
           "TiB formatting");

    expect(device_type_string(0) == "0", "unknown zero device type");
    expect(device_type_string(2) == "ALL | CPU", "CPU device type");
    expect(device_type_string(4) == "ALL | GPU", "GPU device type");
    expect(device_type_string(8) == "ACCELERATOR | ALL", "accelerator device type");
    expect(device_type_string(16) == "ALL | CUSTOM", "custom device type");
    expect(device_type_string(6) == "ALL | CPU | GPU", "combined device type");
    expect(device_type_string(1) == "ALL | DEFAULT | _is_bitfield",
           "pyopencl default-bit compatibility");
    expect(device_type_string(32) == "ALL", "unknown bit compatibility");

    {
        std::ostringstream branchOutput;
        const auto status = platform_enumeration_exit(-1001, 0, branchOutput);
        expect(status == 3, "platform-not-found status");
        expect(branchOutput.str() ==
                   "No OpenCL platforms found.\n"
                   "LogicError('clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR')\n",
               "platform-not-found legacy message");
    }
    {
        std::ostringstream branchOutput;
        const auto status = platform_enumeration_exit(0, 0, branchOutput);
        expect(status == 3, "empty platform list status");
        expect(branchOutput.str() == "No OpenCL platforms found.\n",
               "empty platform list message");
    }
    {
        std::ostringstream branchOutput;
        const auto status = platform_enumeration_exit(-30, 0, branchOutput);
        expect(status == 1, "unexpected platform query status");
        expect(branchOutput.str() ==
                   "Unexpected error while querying OpenCL.\n"
                   "OpenCL error -30 while enumerating OpenCL platforms.\n",
               "unexpected platform query message");
    }
    {
        std::ostringstream branchOutput;
        const auto status = platform_enumeration_exit(0, 2, branchOutput);
        expect(!status.has_value(), "non-empty platform list continues");
        expect(branchOutput.str().empty(), "non-empty platform list emits no prefix");
    }

    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_detect_opencl(
        {"--legacy-arguments-are-ignored"}, output, error);
    const std::string report = output.str();
    expect(error.str().empty(), "diagnostic writes through legacy stdout stream");

#if LUW_HAS_OPENCL
    expect(status == 0 || status == 3,
           "enabled diagnostic reports platforms or the no-platform status");
    if (status == 0) {
        expect(report.rfind("==== OpenCL Environment Report ====\n", 0) == 0,
               "legacy report title");
        expect_contains(report, "[Platform] ", "platform report line");
        if (report.find("  [Device] ") != std::string::npos) {
            expect_contains(report, "           Type: ", "device type field");
            expect_contains(report, "           Vendor: ", "device vendor field");
            expect_contains(report, "           OpenCL: ", "device OpenCL field");
            expect_contains(report, "           Driver: ", "device driver field");
            expect_contains(report, "           Compute Units: ", "compute unit field");
            expect_contains(report, "           Max Clock: ", "clock field");
            expect_contains(report, "           Global Memory: ", "memory field");
        } else {
            expect(report.find("  No devices found on this platform.\n") != std::string::npos ||
                       report.find("  Failed to enumerate devices on this platform.\n") !=
                           std::string::npos,
                   "device-less platform report");
        }
    } else {
        expect(report.rfind("No OpenCL platforms found.\n", 0) == 0,
               "no-platform diagnostic");
    }
#else
    expect(status == 2, "missing native OpenCL dependency status");
    expect(report ==
               "Native OpenCL diagnostics are not available in this build.\n"
               "Reconfigure with OpenCL headers and the OpenCL loader library.\n",
           "missing native OpenCL dependency message");
#endif

    if (failures == 0) {
        std::cout << "detect_opencl compatibility tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
