#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace luw::cli {

int runDeckCommand(const std::vector<std::string>& arguments,
                   std::ostream& output,
                   std::ostream& error);

} // namespace luw::cli

