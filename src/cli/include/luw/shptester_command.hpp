#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace luw::cli {

int run_shptester(const std::vector<std::string>& args,
                  std::ostream& out,
                  std::ostream& err);

}  // namespace luw::cli
