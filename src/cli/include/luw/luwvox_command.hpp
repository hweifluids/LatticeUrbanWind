#pragma once

#include "luw/command_registry.hpp"

namespace luw::cli {

// Native replacement for core/bridge_core/3_voxelization.py.
int run_luwvox(const CommandArguments& arguments, std::ostream& output,
               std::ostream& error);

} // namespace luw::cli
