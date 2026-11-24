#pragma once

#include "luw/command_registry.hpp"

#include <iosfwd>

namespace luw::cli {

// Native replacement for core/bridge_core/1_buildBC.py.
int run_luwbc(const CommandArguments& arguments,
              std::ostream& output,
              std::ostream& error);

} // namespace luw::cli
