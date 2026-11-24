#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace luw::cli {

int run_luwval(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);
int run_dgprepare(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

} // namespace luw::cli
