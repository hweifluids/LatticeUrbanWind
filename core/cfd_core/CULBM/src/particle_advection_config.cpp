#include "particle_advection_config.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

namespace culbm::particle_advection {
namespace {

std::string trim(std::string text) {
    constexpr const char* whitespace = " \t\r\n";
    const std::size_t begin = text.find_first_not_of(whitespace);
    if (begin == std::string::npos) return {};
    const std::size_t end = text.find_last_not_of(whitespace);
    return text.substr(begin, end - begin + 1u);
}

std::string unquote(std::string text) {
    text = trim(std::move(text));
    if (text.size() >= 2u) {
        const char quote = text.front();
        if ((quote == '"' || quote == '\'') && text.back() == quote) {
            return trim(text.substr(1u, text.size() - 2u));
        }
    }
    return text;
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::string normalize_key(const std::string& raw) {
    const std::string input = trim(raw);
    std::string normalized;
    normalized.reserve(input.size());
    bool separator = false;
    for (const unsigned char ch : input) {
        if (ch == '-' || std::isspace(ch)) {
            if (!normalized.empty()) separator = true;
            continue;
        }
        if (separator && !normalized.empty() && normalized.back() != '_') {
            normalized.push_back('_');
        }
        normalized.push_back(static_cast<char>(std::tolower(ch)));
        separator = false;
    }
    while (!normalized.empty() && normalized.front() == '_') {
        normalized.erase(normalized.begin());
    }
    while (!normalized.empty() && normalized.back() == '_') {
        normalized.pop_back();
    }
    return normalized;
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() &&
        text.compare(0u, prefix.size(), prefix) == 0;
}

bool parse_bool(const std::string& raw, bool& value) {
    const std::string token = lower(unquote(raw));
    if (token == "1" || token == "true" || token == "t" || token == "yes" ||
        token == "y" || token == "on" || token == "enable" || token == "enabled") {
        value = true;
        return true;
    }
    if (token == "0" || token == "false" || token == "f" || token == "no" ||
        token == "n" || token == "off" || token == "disable" || token == "disabled") {
        value = false;
        return true;
    }
    if (token.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double numeric = std::strtod(token.c_str(), &end);
    if (errno == ERANGE || end == token.c_str() || *end != '\0' || !std::isfinite(numeric)) {
        return false;
    }
    value = numeric != 0.0;
    return true;
}

bool parse_finite_double(const std::string& raw, double& value) {
    const std::string token = trim(raw);
    if (token.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(token.c_str(), &end);
    if (errno == ERANGE || end == token.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_positive_u64(const std::string& raw, std::uint64_t& value) {
    const std::string token = trim(raw);
    if (token.empty() || token.front() == '+' || token.front() == '-') return false;
    for (const unsigned char ch : token) {
        if (!std::isdigit(ch)) return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(token.c_str(), &end, 10);
    if (errno == ERANGE || end == token.c_str() || *end != '\0' || parsed == 0u) {
        return false;
    }
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

std::string strip_line_comment(std::string line) {
    const std::size_t hash = line.find('#');
    const std::size_t slash = line.find("//");
    std::size_t comment = std::string::npos;
    if (hash != std::string::npos) comment = hash;
    if (slash != std::string::npos) {
        comment = comment == std::string::npos ? slash : std::min(comment, slash);
    }
    if (comment != std::string::npos) line.erase(comment);
    return trim(std::move(line));
}

std::vector<std::string> split_particle_columns(std::string line) {
    for (char& ch : line) {
        if (ch == ',' || ch == ';') ch = ' ';
    }
    std::vector<std::string> columns;
    std::istringstream stream(line);
    std::string column;
    while (stream >> column) columns.push_back(column);
    return columns;
}

bool has_empty_punctuation_column(const std::string& line) {
    bool previous_punctuation_separator = false;
    bool saw_nonspace_since_separator = false;
    for (const unsigned char ch : line) {
        if (ch == ',' || ch == ';') {
            if (previous_punctuation_separator || !saw_nonspace_since_separator) {
                return true;
            }
            previous_punctuation_separator = true;
            saw_nonspace_since_separator = false;
        }
        else if (!std::isspace(ch)) {
            previous_punctuation_separator = false;
            saw_nonspace_since_separator = true;
        }
    }
    return previous_punctuation_separator;
}

std::string normalized_header_token(std::string token) {
    token = lower(trim(std::move(token)));
    token.erase(std::remove_if(token.begin(), token.end(), [](const unsigned char ch) {
        return ch == '"' || ch == '\'' || std::isspace(ch);
    }), token.end());
    return token;
}

bool is_axis_header(const std::string& raw, const char axis) {
    const std::string token = normalized_header_token(raw);
    const std::string prefix(1u, axis);
    return token == prefix || token == prefix + "_m" || token == prefix + "[m]" ||
        token == prefix + "(m)";
}

bool is_time_header(const std::string& raw) {
    const std::string token = normalized_header_token(raw);
    return token == "t" || token == "time" || token == "t_s" || token == "t[s]" ||
        token == "time_s" || token == "time[s]" || token == "release_time" ||
        token == "release_time_s" || token == "release_time[s]";
}

bool is_particle_header(const std::vector<std::string>& columns) {
    if (columns.size() != 3u && columns.size() != 4u) return false;
    return is_axis_header(columns[0], 'x') && is_axis_header(columns[1], 'y') &&
        is_axis_header(columns[2], 'z') &&
        (columns.size() == 3u || is_time_header(columns[3]));
}

std::string line_error(const std::string& source_name,
                       const std::size_t line,
                       const std::string& detail) {
    return source_name + ":" + std::to_string(line) + ": " + detail;
}

bool validate_nonnegative_seconds(const std::optional<double>& seconds,
                                  const char* name,
                                  std::string& error) {
    if (!seconds.has_value()) return true;
    if (!std::isfinite(*seconds) || *seconds < 0.0) {
        error = std::string(name) + " must be finite and >= 0 seconds";
        return false;
    }
    return true;
}

bool validate_step_inputs(const double seconds,
                          const double seconds_per_step,
                          const char* seconds_name,
                          std::string& error) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        error = std::string(seconds_name) + " must be finite and >= 0";
        return false;
    }
    if (!std::isfinite(seconds_per_step) || seconds_per_step <= 0.0) {
        error = "seconds_per_step must be finite and > 0";
        return false;
    }
    return true;
}

bool long_double_to_u64(const long double value,
                        std::uint64_t& converted,
                        std::string& error,
                        const char* quantity) {
    const long double limit_exclusive = std::ldexp(1.0L, 64);
    if (!std::isfinite(value) || value < 0.0L || value >= limit_exclusive) {
        error = std::string(quantity) + " exceeds the uint64 lattice-step range";
        return false;
    }
    converted = static_cast<std::uint64_t>(value);
    return true;
}

long double quantization_tolerance(const long double scaled) {
    const long double relative =
        8.0L * static_cast<long double>(std::numeric_limits<double>::epsilon()) *
        std::max(1.0L, std::fabs(scaled));
    return std::min(relative, 1.0e-7L);
}

std::string canonical_advection_key(const std::string& key) {
    if (key == "culbm_advection_enable") return "culbm_advection_enabled";
    if (key == "culbm_advection_method" || key == "culbm_advection_scheme") {
        return "culbm_advection_integrator";
    }
    if (key == "culbm_advection_file") return "culbm_advection_particle_file";
    if (key == "culbm_advection_global_release_time_s") {
        return "culbm_advection_release_time_s";
    }
    if (key == "culbm_advection_particle_lifetime_s" ||
        key == "culbm_advection_duration_s") {
        return "culbm_advection_lifetime_s";
    }
    if (key == "culbm_advection_delta_s" ||
        key == "culbm_advection_auto_interval_s" ||
        key == "culbm_advection_automatic_interval_s" ||
        key == "culbm_advection_auto_release_interval_s" ||
        key == "culbm_advection_automatic_release_interval_s") {
        return "culbm_advection_release_interval_s";
    }
    if (key == "culbm_advection_automatic_spacing_m") {
        return "culbm_advection_auto_spacing_m";
    }
    if (key == "culbm_advection_automatic_min_m") {
        return "culbm_advection_auto_min_m";
    }
    if (key == "culbm_advection_automatic_max_m") {
        return "culbm_advection_auto_max_m";
    }
    return key;
}

bool is_known_advection_key(const std::string& key) {
    return key == "culbm_advection_enabled" ||
        key == "culbm_advection_mode" ||
        key == "culbm_advection_integrator" ||
        key == "culbm_advection_particle_file" ||
        key == "culbm_advection_release_time_s" ||
        key == "culbm_advection_lifetime_s" ||
        key == "culbm_advection_output_interval_s" ||
        key == "culbm_advection_max_particles" ||
        key == "culbm_advection_release_interval_s" ||
        key == "culbm_advection_auto_spacing_m" ||
        key == "culbm_advection_auto_min_m" ||
        key == "culbm_advection_auto_max_m";
}

} // namespace

const char* mode_name(const Mode mode) {
    switch (mode) {
    case Mode::Prescribed: return "prescribed";
    case Mode::Automatic: return "automatic";
    }
    return "unknown";
}

const char* integrator_name(const Integrator integrator) {
    switch (integrator) {
    case Integrator::Euler: return "euler";
    case Integrator::HeunRk2: return "heun-rk2";
    case Integrator::Rk4: return "rk4";
    case Integrator::Rk6: return "rk6";
    }
    return "unknown";
}

bool parse_strict_triplet(const std::string& raw,
                          std::array<double, 3u>& values,
                          std::string& error) {
    error.clear();
    std::string text = trim(raw);
    if (text.empty()) {
        error = "triplet is empty";
        return false;
    }
    const bool begins_bracket = text.front() == '[';
    const bool ends_bracket = text.back() == ']';
    if (begins_bracket != ends_bracket) {
        error = "triplet has unmatched brackets";
        return false;
    }
    if (begins_bracket) text = trim(text.substr(1u, text.size() - 2u));
    if (text.empty() || text.find('[') != std::string::npos ||
        text.find(']') != std::string::npos) {
        error = "triplet has invalid brackets";
        return false;
    }

    bool previous_punctuation_separator = false;
    bool saw_nonspace_since_separator = false;
    for (const unsigned char ch : text) {
        if (ch == ',' || ch == ';') {
            if (previous_punctuation_separator || !saw_nonspace_since_separator) {
                error = "triplet contains an empty item";
                return false;
            }
            previous_punctuation_separator = true;
            saw_nonspace_since_separator = false;
        }
        else if (!std::isspace(ch)) {
            previous_punctuation_separator = false;
            saw_nonspace_since_separator = true;
        }
    }
    if (previous_punctuation_separator || !saw_nonspace_since_separator) {
        error = "triplet contains an empty item";
        return false;
    }

    for (char& ch : text) {
        if (ch == ',' || ch == ';') ch = ' ';
    }
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) tokens.push_back(token);
    if (tokens.size() != 3u) {
        error = "triplet must contain exactly 3 finite numbers";
        return false;
    }
    std::array<double, 3u> parsed{};
    for (std::size_t i = 0u; i < parsed.size(); ++i) {
        if (!parse_finite_double(tokens[i], parsed[i])) {
            error = "triplet item " + std::to_string(i + 1u) +
                " is not a finite number";
            return false;
        }
    }
    values = parsed;
    return true;
}

bool parse_config(const RawDeck& raw_deck, Config& config, std::string& error) {
    error.clear();
    std::unordered_map<std::string, std::string> entries;
    for (const auto& [raw_key, raw_value] : raw_deck) {
        const std::string normalized = normalize_key(raw_key);
        if (!starts_with(normalized, "culbm_advection_")) continue;
        const std::string canonical = canonical_advection_key(normalized);
        if (!is_known_advection_key(canonical)) {
            error = "unknown CULBM particle-advection key '" + raw_key + "'";
            return false;
        }
        if (!entries.emplace(canonical, raw_value).second) {
            error = "duplicate CULBM particle-advection setting for '" + canonical + "'";
            return false;
        }
    }

    Config parsed;
    if (const auto it = entries.find("culbm_advection_enabled"); it != entries.end()) {
        if (!parse_bool(it->second, parsed.enabled)) {
            error = "culbm_advection_enabled expects a bool-like value";
            return false;
        }
    }
    if (const auto it = entries.find("culbm_advection_mode"); it != entries.end()) {
        const std::string value = lower(unquote(it->second));
        if (value == "prescribed" || value == "given" || value == "file" ||
            value == "manual") {
            parsed.mode = Mode::Prescribed;
        }
        else if (value == "automatic" || value == "auto") {
            parsed.mode = Mode::Automatic;
        }
        else {
            error = "culbm_advection_mode must be prescribed or automatic";
            return false;
        }
    }
    if (const auto it = entries.find("culbm_advection_integrator"); it != entries.end()) {
        std::string value = lower(unquote(it->second));
        std::replace(value.begin(), value.end(), '-', '_');
        if (value == "euler" || value == "explicit_euler" || value == "rk1") {
            parsed.integrator = Integrator::Euler;
        }
        else if (value == "heun" || value == "heun_rk2" || value == "rk2") {
            parsed.integrator = Integrator::HeunRk2;
        }
        else if (value == "rk4" || value == "runge_kutta_4") {
            parsed.integrator = Integrator::Rk4;
        }
        else if (value == "rk6" || value == "runge_kutta_6") {
            parsed.integrator = Integrator::Rk6;
        }
        else {
            error = "culbm_advection_integrator must be euler, heun-rk2, rk4, or rk6";
            return false;
        }
    }
    if (const auto it = entries.find("culbm_advection_particle_file"); it != entries.end()) {
        const std::string value = unquote(it->second);
        if (!value.empty()) parsed.particle_file = std::filesystem::path(value);
    }

    auto parse_optional_seconds = [&](const char* key,
                                      std::optional<double>& destination) -> bool {
        const auto it = entries.find(key);
        if (it == entries.end() || unquote(it->second).empty()) {
            destination.reset();
            return true;
        }
        double value = 0.0;
        if (!parse_finite_double(unquote(it->second), value)) {
            error = std::string(key) + " must be a finite number";
            return false;
        }
        destination = value;
        return true;
    };
    if (!parse_optional_seconds("culbm_advection_release_time_s", parsed.release_time_s) ||
        !parse_optional_seconds("culbm_advection_lifetime_s", parsed.lifetime_s) ||
        !parse_optional_seconds("culbm_advection_output_interval_s",
                                parsed.output_interval_s) ||
        !parse_optional_seconds("culbm_advection_release_interval_s",
                                parsed.release_interval_s)) {
        return false;
    }
    if (!validate_nonnegative_seconds(parsed.release_time_s,
                                      "culbm_advection_release_time_s", error)) {
        return false;
    }
    if (parsed.lifetime_s.has_value() &&
        (!std::isfinite(*parsed.lifetime_s) || *parsed.lifetime_s <= 0.0)) {
        error = "culbm_advection_lifetime_s must be finite and > 0 when set";
        return false;
    }
    if (parsed.output_interval_s.has_value() &&
        (!std::isfinite(*parsed.output_interval_s) ||
         *parsed.output_interval_s <= 0.0)) {
        error = "culbm_advection_output_interval_s must be finite and > 0 when set";
        return false;
    }
    if (parsed.release_interval_s.has_value() &&
        (!std::isfinite(*parsed.release_interval_s) ||
         *parsed.release_interval_s <= 0.0)) {
        error = "culbm_advection_release_interval_s must be finite and > 0 when set";
        return false;
    }
    if (const auto it = entries.find("culbm_advection_max_particles");
        it != entries.end()) {
        const std::string value = unquote(it->second);
        if (!parse_positive_u64(value, parsed.max_particles)) {
            error = "culbm_advection_max_particles must be a positive integer";
            return false;
        }
    }

    auto parse_optional_triplet = [&](const char* key,
                                      std::optional<std::array<double, 3u>>& destination) -> bool {
        const auto it = entries.find(key);
        if (it == entries.end() || unquote(it->second).empty()) {
            destination.reset();
            return true;
        }
        std::array<double, 3u> values{};
        std::string detail;
        if (!parse_strict_triplet(unquote(it->second), values, detail)) {
            error = std::string(key) + ": " + detail;
            return false;
        }
        destination = values;
        return true;
    };
    if (!parse_optional_triplet("culbm_advection_auto_spacing_m",
                                parsed.automatic_spacing_m) ||
        !parse_optional_triplet("culbm_advection_auto_min_m",
                                parsed.automatic_min_m) ||
        !parse_optional_triplet("culbm_advection_auto_max_m",
                                parsed.automatic_max_m)) {
        return false;
    }
    if (parsed.automatic_spacing_m.has_value()) {
        for (const double spacing : *parsed.automatic_spacing_m) {
            if (!(spacing > 0.0)) {
                error = "culbm_advection_auto_spacing_m values must all be > 0";
                return false;
            }
        }
    }
    if (parsed.automatic_min_m.has_value() != parsed.automatic_max_m.has_value()) {
        error = "culbm_advection_auto_min_m and auto_max_m must be set together";
        return false;
    }
    if (parsed.automatic_min_m.has_value()) {
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            if (!((*parsed.automatic_min_m)[axis] <
                  (*parsed.automatic_max_m)[axis])) {
                error = "automatic bounds require min < max on every axis";
                return false;
            }
        }
    }
    if (parsed.enabled && parsed.mode == Mode::Automatic) {
        if (!parsed.release_interval_s.has_value()) {
            error = "automatic mode requires culbm_advection_release_interval_s";
            return false;
        }
        if (!parsed.automatic_spacing_m.has_value()) {
            error = "automatic mode requires culbm_advection_auto_spacing_m";
            return false;
        }
    }

    config = std::move(parsed);
    return true;
}

bool parse_particle_stream(std::istream& input,
                           const std::string& source_name,
                           const std::optional<double>& release_time_override_s,
                           std::vector<ParticleSeed>& particles,
                           std::string& error,
                           ParticleFileInfo* info) {
    error.clear();
    particles.clear();
    ParticleFileInfo parsed_info;
    if (info != nullptr) *info = {};
    if (!validate_nonnegative_seconds(release_time_override_s,
                                      "release_time_override_s", error)) {
        return false;
    }

    std::string line;
    std::size_t line_number = 0u;
    bool first_content_line = true;
    while (std::getline(input, line)) {
        ++line_number;
        if (line_number == 1u && line.size() >= 3u &&
            static_cast<unsigned char>(line[0]) == 0xEFu &&
            static_cast<unsigned char>(line[1]) == 0xBBu &&
            static_cast<unsigned char>(line[2]) == 0xBFu) {
            line.erase(0u, 3u);
        }
        line = strip_line_comment(std::move(line));
        if (line.empty()) continue;
        if (has_empty_punctuation_column(line)) {
            error = line_error(source_name, line_number,
                               "particle row contains an empty column");
            particles.clear();
            return false;
        }
        const std::vector<std::string> columns = split_particle_columns(line);
        if (first_content_line && is_particle_header(columns)) {
            first_content_line = false;
            continue;
        }
        first_content_line = false;
        if (columns.size() != 3u && columns.size() != 4u) {
            error = line_error(source_name, line_number,
                               "expected exactly x y z [release_time_s]");
            particles.clear();
            return false;
        }

        ParticleSeed particle;
        particle.source_line = line_number;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            if (!parse_finite_double(columns[axis], particle.position_m[axis])) {
                error = line_error(source_name, line_number,
                                   "coordinate " + std::to_string(axis + 1u) +
                                   " is not a finite number");
                particles.clear();
                return false;
            }
        }
        double row_release_time_s = 0.0;
        if (columns.size() == 4u) {
            if (!parse_finite_double(columns[3], row_release_time_s)) {
                error = line_error(source_name, line_number,
                                   "release time is not a finite number");
                particles.clear();
                return false;
            }
            ++parsed_info.rows_with_explicit_release_time;
        }
        if (row_release_time_s < 0.0) {
            error = line_error(source_name, line_number,
                               "release time must be >= 0 seconds");
            particles.clear();
            return false;
        }
        particle.release_time_s =
            release_time_override_s.value_or(row_release_time_s);
        if (release_time_override_s.has_value()) {
            ++parsed_info.rows_using_global_release_time;
        }
        particles.push_back(particle);
    }
    if (input.bad()) {
        error = source_name + ": failed while reading particle data";
        particles.clear();
        return false;
    }
    if (particles.empty()) {
        error = source_name + ": no particle rows were found";
        return false;
    }
    parsed_info.release_time_override_applied =
        release_time_override_s.has_value();
    if (info != nullptr) *info = parsed_info;
    return true;
}

bool read_particle_file(const std::filesystem::path& path,
                        const std::optional<double>& release_time_override_s,
                        std::vector<ParticleSeed>& particles,
                        std::string& error,
                        ParticleFileInfo* info) {
    std::ifstream input(path);
    if (!input.is_open()) {
        particles.clear();
        if (info != nullptr) *info = {};
        error = "could not open particle file '" + path.string() + "'";
        return false;
    }
    return parse_particle_stream(input, path.string(), release_time_override_s,
                                 particles, error, info);
}

bool release_step_nearest(const double release_time_s,
                          const double seconds_per_step,
                          std::uint64_t& release_step,
                          std::string& error) {
    error.clear();
    if (!validate_step_inputs(release_time_s, seconds_per_step,
                              "release_time_s", error)) {
        return false;
    }
    const long double scaled =
        static_cast<long double>(release_time_s) /
        static_cast<long double>(seconds_per_step);
    const long double rounded =
        std::floor(scaled + 0.5L + quantization_tolerance(scaled));
    return long_double_to_u64(rounded, release_step, error, "release time");
}

bool lifetime_steps_floor(const std::optional<double>& lifetime_s,
                          const double seconds_per_step,
                          LifetimeSteps& lifetime,
                          std::string& error) {
    error.clear();
    lifetime = {};
    if (!std::isfinite(seconds_per_step) || seconds_per_step <= 0.0) {
        error = "seconds_per_step must be finite and > 0";
        return false;
    }
    if (!lifetime_s.has_value()) return true;
    if (!std::isfinite(*lifetime_s) || *lifetime_s <= 0.0) {
        error = "lifetime_s must be finite and > 0 when set";
        return false;
    }
    const long double scaled =
        static_cast<long double>(*lifetime_s) /
        static_cast<long double>(seconds_per_step);
    const long double floored =
        std::floor(scaled + quantization_tolerance(scaled));
    std::uint64_t steps = 0u;
    if (!long_double_to_u64(floored, steps, error, "particle lifetime")) {
        return false;
    }
    lifetime.infinite = false;
    lifetime.steps = steps;
    return true;
}

bool quantize_particle_releases(const std::vector<ParticleSeed>& particles,
                                const double seconds_per_step,
                                std::vector<QuantizedParticleSeed>& quantized,
                                std::string& error) {
    error.clear();
    quantized.clear();
    quantized.reserve(particles.size());
    for (const ParticleSeed& particle : particles) {
        QuantizedParticleSeed converted;
        converted.position_m = particle.position_m;
        converted.source_line = particle.source_line;
        if (!release_step_nearest(particle.release_time_s, seconds_per_step,
                                  converted.release_step, error)) {
            error = "particle source line " + std::to_string(particle.source_line) +
                ": " + error;
            quantized.clear();
            return false;
        }
        quantized.push_back(converted);
    }
    return true;
}

bool build_automatic_release_schedule(
    const double start_time_s,
    const double interval_s,
    const double seconds_per_step,
    const std::uint64_t final_step,
    std::vector<ReleaseBatch>& schedule,
    std::string& error,
    const std::uint64_t maximum_nominal_releases) {
    error.clear();
    schedule.clear();
    if (!validate_step_inputs(start_time_s, seconds_per_step,
                              "start_time_s", error)) {
        return false;
    }
    if (!std::isfinite(interval_s) || interval_s <= 0.0) {
        error = "interval_s must be finite and > 0";
        return false;
    }
    if (maximum_nominal_releases == 0u) {
        error = "maximum_nominal_releases must be > 0";
        return false;
    }

    for (std::uint64_t ordinal = 0u;; ++ordinal) {
        const long double event_seconds =
            static_cast<long double>(start_time_s) +
            static_cast<long double>(ordinal) *
                static_cast<long double>(interval_s);
        if (!std::isfinite(event_seconds)) {
            error = "automatic release time overflow";
            schedule.clear();
            return false;
        }
        const long double scaled =
            event_seconds / static_cast<long double>(seconds_per_step);
        const long double rounded =
            std::floor(scaled + 0.5L + quantization_tolerance(scaled));
        std::uint64_t step = 0u;
        if (!long_double_to_u64(rounded, step, error,
                                "automatic release time")) {
            schedule.clear();
            return false;
        }
        if (step > final_step) return true;
        if (ordinal >= maximum_nominal_releases) {
            error = "automatic release schedule exceeds maximum_nominal_releases";
            schedule.clear();
            return false;
        }
        if (!schedule.empty() && schedule.back().step == step) {
            if (schedule.back().multiplicity ==
                std::numeric_limits<std::uint64_t>::max()) {
                error = "automatic release multiplicity overflow";
                schedule.clear();
                return false;
            }
            ++schedule.back().multiplicity;
        }
        else {
            schedule.push_back({step, 1u});
        }
        if (ordinal == std::numeric_limits<std::uint64_t>::max()) {
            error = "automatic release ordinal overflow";
            schedule.clear();
            return false;
        }
    }
}

} // namespace culbm::particle_advection
