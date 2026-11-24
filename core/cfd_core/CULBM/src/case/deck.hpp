#pragma once

// LUW deck files: line syntax (key = value, // comments outside quotes,
// normalized keys and aliases), typed value parsers, and the ASCII text
// helpers of the case setup.

#include "../core/core.hpp"

#include <cstddef>
#include <istream>
#include <unordered_map>

namespace culbm::cases {

std::string deck_trim(std::string s);
std::string deck_unquote(std::string s);
// Position of a // comment outside quotes, npos when there is none.
size_t deck_comment_index(const std::string& line);
// Lower case, '-' and blanks to '_', aliases resolved.
std::string deck_normalize_key(std::string key);
bool deck_try_parse_bool(const std::string& raw, bool& out);
bool deck_try_parse_finite_double(const std::string& raw, double& out);
bool deck_try_parse_uint(const std::string& raw, uint& out);

struct ParsedDeck {
    std::unordered_map<std::string, std::string> values;
};

// Every key = value line; a later line overrides an earlier one.
ParsedDeck read_deck_entries(std::istream& in);

std::string to_lower_ascii(std::string s);
std::string to_upper_ascii(std::string s);
std::string strip_all_whitespace(std::string s);
std::string trim_trailing_decimal_zeros(std::string s);
std::string format_decimal_trimmed(double value, int precision = 6);
// Characters other than [0-9A-Za-z_.-] become '_'.
std::string sanitize_filename_component(std::string s);
std::string trim_copy(std::string s);

} // namespace culbm::cases
