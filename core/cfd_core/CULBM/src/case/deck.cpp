#include "deck.hpp"

#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>

namespace culbm::cases {

std::string deck_trim(std::string s) {
    const char* ws = " \t\r\n";
    const size_t begin = s.find_first_not_of(ws);
    if (begin == std::string::npos) {
        return std::string();
    }
    const size_t end = s.find_last_not_of(ws);
    return s.substr(begin, end - begin + 1u);
}

std::string deck_unquote(std::string s) {
    s = deck_trim(std::move(s));
    if (s.size() >= 2u) {
        const char q = s.front();
        if ((q == '"' || q == '\'') && s.back() == q) {
            s = deck_trim(s.substr(1u, s.size() - 2u));
        }
    }
    return s;
}

size_t deck_comment_index(const std::string& line) {
    bool in_single_quote = false;
    bool in_double_quote = false;
    for (size_t i = 0u; i + 1u < line.size(); ++i) {
        const char ch = line[i];
        const char next = line[i + 1u];
        if (ch == '\'' && !in_double_quote) {
            in_single_quote = !in_single_quote;
            continue;
        }
        if (ch == '"' && !in_single_quote) {
            in_double_quote = !in_double_quote;
            continue;
        }
        if (!in_single_quote && !in_double_quote && ch == '/' && next == '/') {
            return i;
        }
    }
    return std::string::npos;
}

std::string deck_normalize_key(std::string key) {
    key = deck_trim(std::move(key));
    std::string normalized;
    normalized.reserve(key.size());
    bool last_was_separator = false;
    for (unsigned char ch : key) {
        if (ch == '-' || std::isspace(ch)) {
            if (!normalized.empty() && !last_was_separator) {
                normalized.push_back('_');
            }
            last_was_separator = true;
            continue;
        }
        normalized.push_back((char)std::tolower(ch));
        last_was_separator = false;
    }
    while (!normalized.empty() && normalized.front() == '_') {
        normalized.erase(normalized.begin());
    }
    while (!normalized.empty() && normalized.back() == '_') {
        normalized.pop_back();
    }

    static const std::unordered_map<std::string, std::string> aliases = {
        {"vk_inlet_enable", "turb_inflow_enable"},
        {"vk_inlet_anisotropy_scale", "vk_inlet_anisotropy"},
        {"vk_inlet_aniso_scale", "vk_inlet_anisotropy"},
        {"culbm_temperature_reference_k", "culbm_boussinesq_reference_temperature_k"},
    };
    const auto alias_it = aliases.find(normalized);
    return alias_it != aliases.end() ? alias_it->second : normalized;
}

bool deck_try_parse_bool(const std::string& raw, bool& out) {
    std::string normalized = deck_unquote(raw);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    if (normalized.empty()) {
        return false;
    }

    static const std::unordered_map<std::string, bool> bool_tokens = {
        {"1", true},
        {"true", true},
        {"t", true},
        {"yes", true},
        {"y", true},
        {"on", true},
        {"enable", true},
        {"enabled", true},
        {"0", false},
        {"false", false},
        {"f", false},
        {"no", false},
        {"n", false},
        {"off", false},
        {"disable", false},
        {"disabled", false},
    };
    const auto token_it = bool_tokens.find(normalized);
    if (token_it != bool_tokens.end()) {
        out = token_it->second;
        return true;
    }

    char* endptr = nullptr;
    const double parsed = std::strtod(normalized.c_str(), &endptr);
    if (endptr == normalized.c_str() || *endptr != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed != 0.0;
    return true;
}

bool deck_try_parse_finite_double(const std::string& raw, double& out) {
    const std::string normalized = deck_unquote(raw);
    if (normalized.empty()) return false;
    char* endptr = nullptr;
    const double parsed = std::strtod(normalized.c_str(), &endptr);
    if (endptr == normalized.c_str() || *endptr != '\0' ||
        !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

bool deck_try_parse_uint(const std::string& raw, uint& out) {
    const std::string normalized = deck_unquote(raw);
    if (normalized.empty() || normalized.front() == '-') return false;
    char* endptr = nullptr;
    const unsigned long long parsed =
        std::strtoull(normalized.c_str(), &endptr, 10);
    if (endptr == normalized.c_str() || *endptr != '\0' ||
        parsed > (unsigned long long)std::numeric_limits<uint>::max()) {
        return false;
    }
    out = (uint)parsed;
    return true;
}

ParsedDeck read_deck_entries(std::istream& in) {
    ParsedDeck deck;
    std::string line;
    while (std::getline(in, line)) {
        const size_t comment = deck_comment_index(line);
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        const std::string key = deck_normalize_key(line.substr(0u, eq));
        if (key.empty()) {
            continue;
        }
        deck.values[key] = deck_trim(line.substr(eq + 1u));
    }
    return deck;
}

std::string to_lower_ascii(std::string s) {
    for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

std::string to_upper_ascii(std::string s) {
    for (char& ch : s) ch = (char)std::toupper((unsigned char)ch);
    return s;
}

std::string strip_all_whitespace(std::string s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }), s.end());
    return s;
}

std::string trim_trailing_decimal_zeros(std::string s) {
    const size_t dot = s.find('.');
    if (dot == std::string::npos) return s;
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    if (s.empty()) return std::string("0");
    return s;
}

std::string format_decimal_trimmed(const double value, const int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return trim_trailing_decimal_zeros(oss.str());
}

std::string sanitize_filename_component(std::string s) {
    for (char& ch : s) {
        const bool ok =
            (ch >= '0' && ch <= '9') ||
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            ch == '_' || ch == '-' || ch == '.';
        if (!ok) ch = '_';
    }
    while (!s.empty() && (s.back() == '.' || s.back() == ' ')) s.pop_back();
    return s.empty() ? std::string("probe") : s;
}

std::string trim_copy(std::string s) {
    const char* ws = " \t\r\n";
    size_t b = s.find_first_not_of(ws);
    size_t e = s.find_last_not_of(ws);
    return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
}

} // namespace culbm::cases
