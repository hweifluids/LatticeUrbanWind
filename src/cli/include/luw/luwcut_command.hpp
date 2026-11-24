#pragma once

#include "luw/command_registry.hpp"

namespace luw::cli {

int run_luwcut(const CommandArguments &arguments, std::ostream &output,
               std::ostream &error);

} // namespace luw::cli
