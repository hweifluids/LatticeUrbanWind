#pragma once

#include "luw/command_registry.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>

namespace luw::cli {

int run_detect_opencl(const CommandArguments& arguments,
                      std::ostream& output,
                      std::ostream& error);

// Kept independent of OpenCL headers so the compatibility formatting can be
// unit-tested even in an inspection build without an OpenCL SDK.
namespace opencl_diagnostics_detail {

[[nodiscard]] std::string human_bytes(std::uint64_t bytes);
[[nodiscard]] std::string device_type_string(std::uint64_t type);
[[nodiscard]] std::optional<int> platform_enumeration_exit(
    int status, std::uint32_t platformCount, std::ostream& output);

} // namespace opencl_diagnostics_detail

} // namespace luw::cli
