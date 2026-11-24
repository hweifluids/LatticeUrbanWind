#pragma once

// Fixed-width scalar aliases and numeric constants of the CULBM host code.
// Every name here has the exact type the solver sources expect, so plain
// literals and default arguments keep their former meaning.

#include <cstdint>

namespace culbm {
namespace core {

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using slong = std::int64_t;
using ulong = std::uint64_t;

inline constexpr double pi = 3.141592653589793; // double literal, as used in host formulas
inline constexpr float pif = 3.1415927f; // float literal of pi
inline constexpr unsigned int max_uint = 4294967295u;
inline constexpr unsigned long long max_ulong = 18446744073709551615ull; // literal type, also where ulong is `unsigned long`

// Console line width shared by every boxed table the solver prints.
inline constexpr int CONSOLE_WIDTH = 94;

} // namespace core
} // namespace culbm
