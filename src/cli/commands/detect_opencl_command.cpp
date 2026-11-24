#include "luw/detect_opencl_command.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LUW_HAS_OPENCL
#define LUW_HAS_OPENCL 0
#endif

#if LUW_HAS_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace luw::cli {

namespace opencl_diagnostics_detail {

std::string human_bytes(std::uint64_t bytes) {
    static constexpr const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream formatted;
    formatted << std::fixed << std::setprecision(2) << value << ' ' << units[unit];
    return formatted.str();
}

std::string device_type_string(std::uint64_t type) {
    // pyopencl's bitfield formatter includes ALL for every non-zero device
    // type and orders the public constant names lexicographically.  Preserve
    // that slightly surprising output for the old diagnostic's console ABI.
    if (type == 0) return "0";

    constexpr std::uint64_t defaultType = 1ULL << 0;
    constexpr std::uint64_t cpuType = 1ULL << 1;
    constexpr std::uint64_t gpuType = 1ULL << 2;
    constexpr std::uint64_t acceleratorType = 1ULL << 3;
    constexpr std::uint64_t customType = 1ULL << 4;

    std::vector<std::string_view> names;
    if ((type & acceleratorType) != 0) names.emplace_back("ACCELERATOR");
    names.emplace_back("ALL");
    if ((type & cpuType) != 0) names.emplace_back("CPU");
    if ((type & customType) != 0) names.emplace_back("CUSTOM");
    if ((type & defaultType) != 0) names.emplace_back("DEFAULT");
    if ((type & gpuType) != 0) names.emplace_back("GPU");
    if ((type & defaultType) != 0) names.emplace_back("_is_bitfield");

    std::ostringstream formatted;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (index != 0) formatted << " | ";
        formatted << names[index];
    }
    return formatted.str();
}

std::optional<int> platform_enumeration_exit(int status,
                                             std::uint32_t platformCount,
                                             std::ostream& output) {
    constexpr int success = 0;
    constexpr int platformNotFoundKhr = -1001;
    if (status == platformNotFoundKhr) {
        output << "No OpenCL platforms found.\n"
               << "LogicError('clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR')\n";
        return 3;
    }
    if (status != success) {
        output << "Unexpected error while querying OpenCL.\n"
               << "OpenCL error " << status
               << " while enumerating OpenCL platforms.\n";
        return 1;
    }
    if (platformCount == 0) {
        output << "No OpenCL platforms found.\n";
        return 3;
    }
    return std::nullopt;
}

} // namespace opencl_diagnostics_detail

namespace {

#if LUW_HAS_OPENCL

class OpenClQueryError final : public std::runtime_error {
public:
    OpenClQueryError(cl_int status, std::string operation)
        : std::runtime_error("OpenCL error " + std::to_string(status)
                             + " while " + std::move(operation) + '.') {}
};

void require_success(cl_int status, std::string operation) {
    if (status != CL_SUCCESS) {
        throw OpenClQueryError(status, std::move(operation));
    }
}

std::string platform_string(cl_platform_id platform,
                            cl_platform_info field,
                            std::string operation) {
    std::size_t size = 0;
    require_success(clGetPlatformInfo(platform, field, 0, nullptr, &size), operation);
    if (size == 0) return {};

    std::vector<char> buffer(size);
    require_success(clGetPlatformInfo(platform, field, buffer.size(), buffer.data(), nullptr),
                    std::move(operation));
    while (!buffer.empty() && buffer.back() == '\0') buffer.pop_back();
    return std::string(buffer.begin(), buffer.end());
}

std::string device_string(cl_device_id device,
                          cl_device_info field,
                          std::string operation) {
    std::size_t size = 0;
    require_success(clGetDeviceInfo(device, field, 0, nullptr, &size), operation);
    if (size == 0) return {};

    std::vector<char> buffer(size);
    require_success(clGetDeviceInfo(device, field, buffer.size(), buffer.data(), nullptr),
                    std::move(operation));
    while (!buffer.empty() && buffer.back() == '\0') buffer.pop_back();
    return std::string(buffer.begin(), buffer.end());
}

template <typename Value>
Value device_value(cl_device_id device, cl_device_info field, std::string operation) {
    Value value{};
    require_success(clGetDeviceInfo(device, field, sizeof(value), &value, nullptr),
                    std::move(operation));
    return value;
}

std::string strip_ascii(std::string value) {
    const auto notSpace = [](unsigned char character) {
        return std::isspace(character) == 0;
    };
    const auto first = std::find_if(value.begin(), value.end(), notSpace);
    const auto last = std::find_if(value.rbegin(), value.rend(), notSpace).base();
    if (first >= last) return {};
    return std::string(first, last);
}

void print_device_enumeration_failure(std::ostream& output,
                                      cl_int status,
                                      std::string_view operation) {
    output << "  Failed to enumerate devices on this platform.\n"
           << "  OpenCL error " << status << " while " << operation << ".\n";
}

#endif

} // namespace

int run_detect_opencl(const CommandArguments& arguments,
                      std::ostream& output,
                      std::ostream& error) {
    // The legacy script accepted and ignored arbitrary arguments, including
    // --help.  Keep that behavior for the direct compatibility wrapper.
    (void)arguments;
    (void)error;

#if !LUW_HAS_OPENCL
    output << "Native OpenCL diagnostics are not available in this build.\n"
           << "Reconfigure with OpenCL headers and the OpenCL loader library.\n";
    return 2;
#else
    try {
        cl_uint platformCount = 0;
        const cl_int countStatus = clGetPlatformIDs(0, nullptr, &platformCount);
        if (const auto earlyExit =
                opencl_diagnostics_detail::platform_enumeration_exit(
                    countStatus, platformCount, output)) {
            return *earlyExit;
        }

        std::vector<cl_platform_id> platforms(platformCount);
        require_success(clGetPlatformIDs(platformCount, platforms.data(), nullptr),
                        "enumerating OpenCL platforms");

        output << "==== OpenCL Environment Report ====\n";
        for (cl_platform_id platform : platforms) {
            const std::string name = platform_string(
                platform, CL_PLATFORM_NAME, "querying the OpenCL platform name");
            const std::string vendor = platform_string(
                platform, CL_PLATFORM_VENDOR, "querying the OpenCL platform vendor");
            const std::string version = platform_string(
                platform, CL_PLATFORM_VERSION, "querying the OpenCL platform version");
            output << "[Platform] " << name << " | " << vendor << " | " << version << '\n';

            cl_uint deviceCount = 0;
            const cl_int deviceCountStatus =
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
            if (deviceCountStatus == CL_DEVICE_NOT_FOUND ||
                (deviceCountStatus == CL_SUCCESS && deviceCount == 0)) {
                output << "  No devices found on this platform.\n";
                continue;
            }
            if (deviceCountStatus != CL_SUCCESS) {
                print_device_enumeration_failure(output, deviceCountStatus,
                                                 "enumerating devices");
                continue;
            }

            std::vector<cl_device_id> devices(deviceCount);
            const cl_int deviceStatus = clGetDeviceIDs(
                platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
            if (deviceStatus != CL_SUCCESS) {
                print_device_enumeration_failure(output, deviceStatus,
                                                 "enumerating devices");
                continue;
            }

            for (cl_device_id device : devices) {
                const cl_device_type type = device_value<cl_device_type>(
                    device, CL_DEVICE_TYPE, "querying the OpenCL device type");
                const cl_ulong globalMemory = device_value<cl_ulong>(
                    device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    "querying the OpenCL device global memory");

                output << "  [Device] "
                       << strip_ascii(device_string(
                              device, CL_DEVICE_NAME, "querying the OpenCL device name"))
                       << '\n';
                output << "           Type: "
                       << opencl_diagnostics_detail::device_type_string(
                              static_cast<std::uint64_t>(type))
                       << '\n';
                output << "           Vendor: "
                       << strip_ascii(device_string(
                              device, CL_DEVICE_VENDOR, "querying the OpenCL device vendor"))
                       << '\n';
                output << "           OpenCL: "
                       << device_string(
                              device, CL_DEVICE_VERSION, "querying the OpenCL device version")
                       << '\n';
                output << "           Driver: "
                       << device_string(device, CL_DRIVER_VERSION,
                                        "querying the OpenCL driver version")
                       << '\n';
                output << "           Compute Units: "
                       << device_value<cl_uint>(
                              device, CL_DEVICE_MAX_COMPUTE_UNITS,
                              "querying the OpenCL compute-unit count")
                       << '\n';
                output << "           Max Clock: "
                       << device_value<cl_uint>(
                              device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                              "querying the OpenCL maximum clock frequency")
                       << " MHz\n";
                output << "           Global Memory: "
                       << opencl_diagnostics_detail::human_bytes(
                              static_cast<std::uint64_t>(globalMemory))
                       << '\n';
            }
        }
        return 0;
    } catch (const OpenClQueryError& exception) {
        output << "Unexpected error while querying OpenCL.\n"
               << exception.what() << '\n';
    } catch (const std::exception& exception) {
        output << "Unexpected error while querying OpenCL.\n"
               << "Exception: " << exception.what() << '\n';
    } catch (...) {
        output << "Unexpected error while querying OpenCL.\n"
               << "Unknown native exception.\n";
    }
    return 1;
#endif
}

} // namespace luw::cli
