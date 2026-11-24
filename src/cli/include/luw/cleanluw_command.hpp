#pragma once

#include "luw/command_registry.hpp"

#include <iosfwd>

namespace luw::cli {

int run_cleanluw(const CommandArguments& arguments,
                 std::ostream& output,
                 std::ostream& error);

} // namespace luw::cli
