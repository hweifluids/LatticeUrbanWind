#include "surface_ibm_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace culbm::surface_ibm {
namespace {

std::mutex config_mutex;
Config published_config;
thread_local const Config* construction_config = nullptr;

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string unquote(std::string value) {
    value = trim(std::move(value));
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1, value.size() - 2);
    }
    return trim(std::move(value));
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string normalize_key(std::string key) {
    key = lower(trim(std::move(key)));
    while (!key.empty() && key.front() == '-') key.erase(key.begin());
    std::replace_if(key.begin(), key.end(), [](char c) {
        return c == '-' || c == '.' || std::isspace(static_cast<unsigned char>(c)) != 0;
    }, '_');
    return key;
}

bool starts_with(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool parse_bool(const std::string& text, bool& value) {
    const std::string token = lower(unquote(text));
    if (token == "1" || token == "true" || token == "t" ||
        token == "yes" || token == "y" || token == "on" ||
        token == "enable" || token == "enabled") {
        value = true;
        return true;
    }
    if (token == "0" || token == "false" || token == "f" ||
        token == "no" || token == "n" || token == "off" ||
        token == "disable" || token == "disabled") {
        value = false;
        return true;
    }
    char* end = nullptr;
    const double numeric = std::strtod(token.c_str(), &end);
    if (end != token.c_str() && *end == '\0' && std::isfinite(numeric)) {
        value = numeric != 0.0;
        return true;
    }
    return false;
}

bool parse_finite_float(const std::string& text, float& value) {
    try {
        const std::string token = unquote(text);
        std::size_t consumed = 0;
        const double parsed = std::stod(token, &consumed);
        if (consumed != token.size() || !std::isfinite(parsed) ||
            parsed < -static_cast<double>(std::numeric_limits<float>::max()) ||
            parsed > static_cast<double>(std::numeric_limits<float>::max())) {
            return false;
        }
        value = static_cast<float>(parsed);
        return std::isfinite(value);
    } catch (...) {
        return false;
    }
}

bool parse_u64(const std::string& text, std::uint64_t& value) {
    try {
        const std::string token = unquote(text);
        if (token.empty() || token.front() == '-') return false;
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(token, &consumed, 10);
        if (consumed != token.size()) return false;
        value = static_cast<std::uint64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_unsigned(const std::string& text, unsigned& value) {
    std::uint64_t parsed = 0;
    if (!parse_u64(text, parsed) || parsed > std::numeric_limits<unsigned>::max()) return false;
    value = static_cast<unsigned>(parsed);
    return true;
}

template <typename T, typename Parser>
bool assign_value(const std::unordered_map<std::string, std::string>& deck,
                  const char* key,
                  T& destination,
                  Parser parser,
                  std::string& error) {
    const auto it = deck.find(key);
    if (it == deck.end()) return true;
    if (!parser(it->second, destination)) {
        error = std::string("invalid value for ") + key + ": " + it->second;
        return false;
    }
    return true;
}

}  // namespace

const char* delta_kernel_name(DeltaKernel kernel) {
    switch (kernel) {
        case DeltaKernel::Roma3Point: return "roma3_point";
    }
    return "unknown";
}

const char* geometry_policy_name(GeometryPolicy policy) {
    switch (policy) {
        case GeometryPolicy::ExteriorUnion: return "exterior_union";
        case GeometryPolicy::AllTriangles: return "all_triangles";
    }
    return "unknown";
}

bool validate_config(const Config& config, std::string& error) {
    error.clear();
    if (!((config.support_z_min == -1 && config.support_z_max == -1) ||
          (config.support_z_min >= 1 &&
           config.support_z_max >= config.support_z_min))) {
        error = "surface-IBM native slab bounds are invalid";
        return false;
    }
    if (config.delta_kernel != DeltaKernel::Roma3Point) {
        error = "unsupported surface-IBM delta kernel";
        return false;
    }
    if (config.geometry_policy != GeometryPolicy::ExteriorUnion &&
        config.geometry_policy != GeometryPolicy::AllTriangles) {
        error = "unsupported surface-IBM geometry policy";
        return false;
    }
    if (!std::isfinite(config.marker_spacing_lu) || config.marker_spacing_lu <= 0.0f) {
        error = "culbm_surface_ibm_marker_spacing_lu must be finite and > 0";
        return false;
    }
    if (config.forcing_iterations < 1 || config.forcing_iterations > 32) {
        error = "culbm_surface_ibm_forcing_iterations must be in [1, 32]";
        return false;
    }
    if (!std::isfinite(config.relaxation) || config.relaxation <= 0.0f || config.relaxation >= 2.0f) {
        error = "culbm_surface_ibm_relaxation must be finite and in (0, 2)";
        return false;
    }
    if (!std::isfinite(config.residual_tolerance) ||
        config.residual_tolerance <= 0.0f || config.residual_tolerance > 1.0f) {
        error = "culbm_surface_ibm_residual_tolerance must be finite and in (0, 1]";
        return false;
    }
    if (!std::isfinite(config.retraction_lu) || config.retraction_lu < 0.0f) {
        error = "culbm_surface_ibm_retraction_lu must be finite and >= 0";
        return false;
    }
    if (!std::isfinite(config.support_thickness_lu) || config.support_thickness_lu <= 0.0f) {
        error = "culbm_surface_ibm_support_thickness_lu must be finite and > 0";
        return false;
    }
    if (config.max_markers == 0) {
        error = "culbm_surface_ibm_max_markers must be > 0";
        return false;
    }
    return true;
}

bool parse_config(const RawDeck& raw, Config& config, std::string& error) {
    error.clear();
    std::unordered_map<std::string, std::string> deck;
    deck.reserve(raw.size());
    for (const auto& entry : raw) {
        const std::string key = normalize_key(entry.first);
        if (key.empty()) continue;
        if (!deck.emplace(key, entry.second).second) {
            error = "duplicate normalized configuration key: " + key;
            return false;
        }
    }

    static const std::unordered_map<std::string, bool> known_keys = {
        {"culbm_surface_ibm_delta_kernel", true},
        {"culbm_surface_ibm_geometry_policy", true},
        {"culbm_surface_ibm_marker_spacing_lu", true},
        {"culbm_surface_ibm_forcing_iterations", true},
        {"culbm_surface_ibm_relaxation", true},
        {"culbm_surface_ibm_residual_tolerance", true},
        {"culbm_surface_ibm_retraction_lu", true},
        {"culbm_surface_ibm_support_thickness_lu", true},
        {"culbm_surface_ibm_max_markers", true},
        {"culbm_surface_ibm_strict_geometry", true},
        {"culbm_surface_ibm_diagnostics_interval", true},
    };
    for (const auto& entry : deck) {
        if (starts_with(entry.first, "culbm_surface_ibm_") && known_keys.count(entry.first) == 0) {
            error = "unknown surface-IBM configuration key: " + entry.first;
            return false;
        }
    }

    Config parsed;
    bool master_enabled = false;
    const bool has_master = deck.count("ibm_enabler") != 0;
    if (has_master && !parse_bool(deck.at("ibm_enabler"), master_enabled)) {
        error = "invalid value for ibm_enabler: " + deck.at("ibm_enabler");
        return false;
    }
    parsed.enabled = has_master ? master_enabled : false;

    const auto delta_it = deck.find("culbm_surface_ibm_delta_kernel");
    if (delta_it != deck.end()) {
        std::string token = lower(unquote(delta_it->second));
        std::replace(token.begin(), token.end(), '-', '_');
        if (token == "roma3" || token == "roma_3" || token == "roma3point" ||
            token == "roma3_point" || token == "roma_3_point") {
            parsed.delta_kernel = DeltaKernel::Roma3Point;
        } else {
            error = "unsupported culbm_surface_ibm_delta_kernel: " + delta_it->second;
            return false;
        }
    }

    const auto policy_it = deck.find("culbm_surface_ibm_geometry_policy");
    if (policy_it != deck.end()) {
        std::string token = lower(unquote(policy_it->second));
        std::replace(token.begin(), token.end(), '-', '_');
        if (token == "exterior_union" || token == "union" || token == "exterior") {
            parsed.geometry_policy = GeometryPolicy::ExteriorUnion;
        } else if (token == "all_triangles" || token == "all") {
            parsed.geometry_policy = GeometryPolicy::AllTriangles;
        } else {
            error = "unsupported culbm_surface_ibm_geometry_policy: " + policy_it->second;
            return false;
        }
    }

    if (!assign_value(deck, "culbm_surface_ibm_marker_spacing_lu", parsed.marker_spacing_lu,
                      parse_finite_float, error) ||
        !assign_value(deck, "culbm_surface_ibm_forcing_iterations", parsed.forcing_iterations,
                      parse_unsigned, error) ||
        !assign_value(deck, "culbm_surface_ibm_relaxation", parsed.relaxation,
                      parse_finite_float, error) ||
        !assign_value(deck, "culbm_surface_ibm_residual_tolerance", parsed.residual_tolerance,
                      parse_finite_float, error) ||
        !assign_value(deck, "culbm_surface_ibm_retraction_lu", parsed.retraction_lu,
                      parse_finite_float, error) ||
        !assign_value(deck, "culbm_surface_ibm_max_markers", parsed.max_markers,
                      parse_u64, error) ||
        !assign_value(deck, "culbm_surface_ibm_strict_geometry", parsed.strict_geometry,
                      parse_bool, error) ||
        !assign_value(deck, "culbm_surface_ibm_diagnostics_interval", parsed.diagnostics_interval,
                      parse_unsigned, error)) {
        return false;
    }

    const bool support_explicit = deck.count("culbm_surface_ibm_support_thickness_lu") != 0;
    if (support_explicit) {
        if (!assign_value(deck, "culbm_surface_ibm_support_thickness_lu",
                          parsed.support_thickness_lu, parse_finite_float, error)) {
            return false;
        }
    } else {
        parsed.support_thickness_lu = parsed.marker_spacing_lu;
    }

    if (!validate_config(parsed, error)) return false;
    config = parsed;
    return true;
}

void set_config(const Config& config) {
    std::string error;
    if (!validate_config(config, error)) {
        throw std::invalid_argument("invalid surface-IBM configuration: " + error);
    }
    const std::lock_guard<std::mutex> lock(config_mutex);
    published_config = config;
}

Config get_config() {
    if (construction_config) return *construction_config;
    const std::lock_guard<std::mutex> lock(config_mutex);
    return published_config;
}

ScopedConfig::ScopedConfig(const Config& config)
    : local_(config), previous_(construction_config) {
    std::string error;
    if (!validate_config(config, error)) throw std::invalid_argument(error);
    construction_config = &local_;
}

ScopedConfig::~ScopedConfig() { construction_config = previous_; }

}  // namespace culbm::surface_ibm
