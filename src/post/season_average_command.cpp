#include "luw/vtk_post_commands.hpp"

#include "luw/legacy_vtk.hpp"
#include "post_internal.hpp"
#include "season_average_internal.hpp"

#include <QByteArray>
#include <QFile>
#include <QString>
#include <QTemporaryDir>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#elif __has_include(<iconv.h>)
#include <cerrno>
#include <iconv.h>
#define LUW_SEASON_HAS_ICONV 1
#else
#define LUW_SEASON_HAS_ICONV 0
#endif

namespace luw::cli {

namespace {

namespace fs = std::filesystem;
using luw::post::LegacyStructuredPoints;
using luw::post::LegacyVtkAssociation;
using luw::post::LegacyVtkField;
using luw::post::LegacyVtkPlaneReader;
using luw::post::detail::pathUtf8;

constexpr std::array<double, 16> angleSequence{
    0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,
    180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5};
constexpr std::array<std::string_view, 16> angleDirections{
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"};
constexpr std::uint64_t accumulatorMemoryThreshold = 512ULL * 1024ULL * 1024ULL;

class ArgumentError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Arguments final {
    fs::path configPath;
    std::optional<fs::path> vtkDirectory;
    std::optional<std::array<double, 3>> outputSpacing;
    bool skipFigures = false;
    bool help = false;
};

struct SpeedBin final {
    std::size_t columnIndex = 0;
    std::string header;
    int classId = 0;
    double lower = 0.0;
    std::optional<double> upper;
    double targetSpeed = 0.0;
};

struct DirectionWeight final {
    double angle = 0.0;
    std::string label;
    double probability = 0.0;
    double velocityWeight = 0.0;
    double tkeWeight = 0.0;
};

struct VtkDirectoryChoice final {
    std::string label;
    fs::path directory;
    std::map<double, fs::path> angleToFile;
};

struct WindFieldSelection final {
    const LegacyVtkField* vectorField = nullptr;
    std::array<const LegacyVtkField*, 3> componentFields{};

    [[nodiscard]] bool hasComponents() const noexcept {
        return componentFields[0] != nullptr && componentFields[1] != nullptr
            && componentFields[2] != nullptr;
    }
    [[nodiscard]] std::string displayName() const {
        if (vectorField != nullptr) return vectorField->name;
        if (hasComponents()) {
            return componentFields[0]->name + "/" + componentFields[1]->name + "/"
                + componentFields[2]->name;
        }
        return "-";
    }
};

struct ProjectLayout final {
    fs::path configPath;
    fs::path projectDirectory;
    fs::path windBcDirectory;
    fs::path profilePath;
    fs::path windrosePath;
    fs::path outputDirectory;
    fs::path figureDirectory;
    fs::path summaryPath;
    fs::path outputVtkPath;
    int figureDpi = 1200;
    std::optional<double> zLimitSi;
};

struct Profile final {
    std::vector<double> z;
    std::vector<double> u;
};

struct Windrose final {
    std::vector<SpeedBin> bins;
    std::map<double, std::vector<double>> probabilities;
    double totalProbability = 0.0;
};

struct HeaderSelection final {
    LegacyStructuredPoints reference;
    std::map<double, LegacyStructuredPoints> byAngle;
};

struct AxisResample final {
    std::vector<std::size_t> lower;
    std::vector<std::size_t> upper;
    std::vector<float> weight;
};

struct ResamplePlan final {
    std::array<int, 3> dimensions{};
    std::array<double, 3> origin{};
    std::array<double, 3> spacing{};
    AxisResample x;
    AxisResample y;
    AxisResample z;
};

std::string lowerAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return text;
}

std::string upperAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return text;
}

std::string trimAscii(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

fs::path pathFromUtf8(const std::string& text) {
#ifdef _WIN32
    return fs::path(QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size())).toStdWString());
#else
    return fs::u8path(text);
#endif
}

QString qPath(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring());
#else
    const std::string text = pathUtf8(path);
    return QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()));
#endif
}

fs::path normalizedAbsolute(const fs::path& path) {
    std::error_code error;
    fs::path result = fs::absolute(path, error);
    if (error) result = path;
    const fs::path canonical = fs::weakly_canonical(result, error);
    return (error ? result : canonical).lexically_normal();
}

std::tm localTimeNow() {
    const std::time_t raw = std::time(nullptr);
    std::tm result{};
#ifdef _WIN32
    localtime_s(&result, &raw);
#else
    localtime_r(&raw, &result);
#endif
    return result;
}

void log(std::ostream& output, const std::string& message) {
    const std::tm local = localTimeNow();
    output << '[' << std::put_time(&local, "%H:%M:%S") << "] " << message << std::endl;
}

void logRule(std::ostream& output, const std::string& title) {
    log(output, std::string(24, '=') + " " + title + " " + std::string(24, '='));
}

std::string humanBytes(std::uint64_t bytes) {
    if (bytes < 1024ULL) return std::to_string(bytes) + " B";
    constexpr std::array<std::string_view, 4> units{"KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    for (const std::string_view unit : units) {
        value /= 1024.0;
        if (value < 1024.0) {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(2) << value << ' ' << unit;
            return stream.str();
        }
    }
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value / 1024.0 << " PiB";
    return stream.str();
}

std::string formatNumber(double value) {
    if (!std::isfinite(value)) {
        if (std::isnan(value)) return "nan";
        return value < 0.0 ? "-inf" : "inf";
    }
    if (std::abs(value - std::round(value)) < 1.0e-9) {
        return std::to_string(static_cast<long long>(std::llround(value)));
    }
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    std::string text = stream.str();
    while (!text.empty() && text.back() == '0') text.pop_back();
    if (!text.empty() && text.back() == '.') text.pop_back();
    return text;
}

std::string pythonFloatString(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value < 0.0 ? "-inf" : "inf";
    if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";
    std::array<char, 128> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (converted.ec != std::errc{}) {
        std::ostringstream fallback;
        fallback << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        return fallback.str();
    }
    std::string text(buffer.data(), converted.ptr);
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}

std::string tupleText(const std::array<int, 3>& values) {
    return "(" + std::to_string(values[0]) + ", " + std::to_string(values[1]) + ", "
        + std::to_string(values[2]) + ")";
}

std::string tupleText(const std::array<double, 3>& values) {
    return "(" + pythonFloatString(values[0]) + ", " + pythonFloatString(values[1]) + ", "
        + pythonFloatString(values[2]) + ")";
}

void printUsage(std::ostream& output) {
    output << "usage: season_average <config_path>\n\n"
              "Synthesize seasonal average wind/tke VTK from directional average VTK files and windrose probabilities.\n\n"
              "positional arguments:\n"
              "  config_path           Path to LUW-family config (*.luw, *.luwdg, *.luwpf)\n\n"
              "options:\n"
              "  -h, --help            show this help message and exit\n"
              "  --vtk-dir VTK_DIR     Directory containing source *avg*.vtk files. Relative paths are resolved from the project directory.\n"
              "  --output-spacing M [M ...]\n"
              "                        Resample output VTK spacing in meters. Pass one value for x/y/z or three values for x y z.\n"
              "  --skip-figures        Only write VTK and summary; skip PNG figures.\n";
}

double parseFinitePositive(const std::string& option, const std::string& token) {
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(token, &consumed);
    } catch (const std::exception&) {
        throw ArgumentError("argument " + option + ": invalid float value: '" + token + "'");
    }
    if (consumed != token.size()) {
        throw ArgumentError("argument " + option + ": invalid float value: '" + token + "'");
    }
    return value;
}

Arguments parseArguments(const std::vector<std::string>& arguments) {
    Arguments parsed;
    std::vector<double> spacing;
    bool haveConfig = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "-h" || argument == "--help") {
            parsed.help = true;
        } else if (argument == "--skip-figures") {
            parsed.skipFigures = true;
        } else if (argument == "--vtk-dir") {
            if (index + 1 >= arguments.size()) {
                throw ArgumentError("argument --vtk-dir: expected one argument");
            }
            parsed.vtkDirectory = pathFromUtf8(arguments[++index]);
        } else if (argument == "--output-spacing") {
            if (!spacing.empty()) throw ArgumentError("argument --output-spacing: specified more than once");
            while (index + 1 < arguments.size() && !arguments[index + 1].starts_with("--")) {
                spacing.push_back(parseFinitePositive("--output-spacing", arguments[++index]));
            }
            if (spacing.empty()) throw ArgumentError("argument --output-spacing: expected at least one argument");
        } else if (argument.starts_with('-')) {
            throw ArgumentError("unrecognized arguments: " + argument);
        } else if (!haveConfig) {
            parsed.configPath = pathFromUtf8(argument);
            haveConfig = true;
        } else {
            throw ArgumentError("unrecognized arguments: " + argument);
        }
    }
    if (!parsed.help && !haveConfig) throw ArgumentError("the following arguments are required: config_path");
    if (!spacing.empty()) {
        if (spacing.size() == 1) {
            parsed.outputSpacing = std::array<double, 3>{spacing[0], spacing[0], spacing[0]};
        } else if (spacing.size() == 3) {
            parsed.outputSpacing = std::array<double, 3>{spacing[0], spacing[1], spacing[2]};
        } else {
            throw ArgumentError("--output-spacing expects either one value or three values (x y z)");
        }
        for (const double value : *parsed.outputSpacing) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw ArgumentError("Output spacing values must be positive finite numbers: "
                                    + tupleText(*parsed.outputSpacing));
            }
        }
    }
    return parsed;
}

std::vector<unsigned char> readBytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open file: " + pathUtf8(path));
    input.seekg(0, std::ios::end);
    const auto length = input.tellg();
    if (length < 0) throw std::runtime_error("Failed to determine file size: " + pathUtf8(path));
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> result(static_cast<std::size_t>(length));
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
    if (input.gcount() != static_cast<std::streamsize>(result.size())) {
        throw std::runtime_error("Failed to read file: " + pathUtf8(path));
    }
    return result;
}

bool utf8Continuation(unsigned char value) noexcept { return (value & 0xc0U) == 0x80U; }

std::optional<std::string> decodeUtf8Strict(const std::vector<unsigned char>& source,
                                            bool stripBom) {
    std::size_t index = 0;
    if (stripBom && source.size() >= 3 && source[0] == 0xefU && source[1] == 0xbbU
        && source[2] == 0xbfU) {
        index = 3;
    }
    const std::size_t firstOutput = index;
    while (index < source.size()) {
        const unsigned char first = source[index];
        std::size_t length = 0;
        if (first <= 0x7fU) {
            length = 1;
        } else if (first >= 0xc2U && first <= 0xdfU && index + 1 < source.size()
                   && utf8Continuation(source[index + 1])) {
            length = 2;
        } else if (first >= 0xe0U && first <= 0xefU && index + 2 < source.size()) {
            const unsigned char second = source[index + 1];
            const unsigned char third = source[index + 2];
            const bool validSecond = (first == 0xe0U && second >= 0xa0U && second <= 0xbfU)
                || (first == 0xedU && second >= 0x80U && second <= 0x9fU)
                || (((first >= 0xe1U && first <= 0xecU) || (first >= 0xeeU && first <= 0xefU))
                    && utf8Continuation(second));
            if (validSecond && utf8Continuation(third)) length = 3;
        } else if (first >= 0xf0U && first <= 0xf4U && index + 3 < source.size()) {
            const unsigned char second = source[index + 1];
            const unsigned char third = source[index + 2];
            const unsigned char fourth = source[index + 3];
            const bool validSecond = (first == 0xf0U && second >= 0x90U && second <= 0xbfU)
                || (first == 0xf4U && second >= 0x80U && second <= 0x8fU)
                || (first >= 0xf1U && first <= 0xf3U && utf8Continuation(second));
            if (validSecond && utf8Continuation(third) && utf8Continuation(fourth)) length = 4;
        }
        if (length == 0) return std::nullopt;
        index += length;
    }
    return std::string(reinterpret_cast<const char*>(source.data() + firstOutput),
                       source.size() - firstOutput);
}

std::string decodeUtf8IgnoringErrors(const std::vector<unsigned char>& source) {
    std::string valid;
    valid.reserve(source.size());
    std::size_t index = 0;
    while (index < source.size()) {
        const unsigned char first = source[index];
        std::size_t length = 0;
        if (first <= 0x7fU) {
            length = 1;
        } else if (first >= 0xc2U && first <= 0xdfU && index + 1 < source.size()
                   && utf8Continuation(source[index + 1])) {
            length = 2;
        } else if (first >= 0xe0U && first <= 0xefU && index + 2 < source.size()) {
            const unsigned char second = source[index + 1];
            const unsigned char third = source[index + 2];
            const bool validSecond = (first == 0xe0U && second >= 0xa0U && second <= 0xbfU)
                || (first == 0xedU && second >= 0x80U && second <= 0x9fU)
                || (((first >= 0xe1U && first <= 0xecU) || (first >= 0xeeU && first <= 0xefU))
                    && utf8Continuation(second));
            if (validSecond && utf8Continuation(third)) length = 3;
        } else if (first >= 0xf0U && first <= 0xf4U && index + 3 < source.size()) {
            const unsigned char second = source[index + 1];
            const unsigned char third = source[index + 2];
            const unsigned char fourth = source[index + 3];
            const bool validSecond = (first == 0xf0U && second >= 0x90U && second <= 0xbfU)
                || (first == 0xf4U && second >= 0x80U && second <= 0x8fU)
                || (first >= 0xf1U && first <= 0xf3U && utf8Continuation(second));
            if (validSecond && utf8Continuation(third) && utf8Continuation(fourth)) length = 4;
        }
        if (length == 0) {
            ++index;
            continue;
        }
        valid.append(reinterpret_cast<const char*>(source.data() + index), length);
        index += length;
    }
    return valid;
}

std::optional<std::string> decodeGbk(const std::vector<unsigned char>& source) {
#ifdef _WIN32
    if (source.empty()) return std::string{};
    const int wideLength = MultiByteToWideChar(936U, MB_ERR_INVALID_CHARS,
                                                reinterpret_cast<const char*>(source.data()),
                                                static_cast<int>(source.size()), nullptr, 0);
    if (wideLength <= 0) return std::nullopt;
    std::wstring wide(static_cast<std::size_t>(wideLength), L'\0');
    if (MultiByteToWideChar(936U, MB_ERR_INVALID_CHARS,
                            reinterpret_cast<const char*>(source.data()),
                            static_cast<int>(source.size()), wide.data(), wideLength) <= 0) {
        return std::nullopt;
    }
    const int utf8Length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(),
                                                wideLength, nullptr, 0, nullptr, nullptr);
    if (utf8Length <= 0) return std::nullopt;
    std::string result(static_cast<std::size_t>(utf8Length), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), wideLength,
                            result.data(), utf8Length, nullptr, nullptr) <= 0) {
        return std::nullopt;
    }
    return result;
#elif LUW_SEASON_HAS_ICONV
    iconv_t converter = iconv_open("UTF-8", "GBK");
    if (converter == reinterpret_cast<iconv_t>(-1)) return std::nullopt;
    std::string result(std::max<std::size_t>(source.size() * 3U + 16U, 32U), '\0');
    char* inputPointer = const_cast<char*>(reinterpret_cast<const char*>(source.data()));
    std::size_t inputRemaining = source.size();
    char* outputPointer = result.data();
    std::size_t outputRemaining = result.size();
    while (inputRemaining > 0) {
        if (iconv(converter, &inputPointer, &inputRemaining, &outputPointer, &outputRemaining)
            != static_cast<std::size_t>(-1)) {
            continue;
        }
        if (errno != E2BIG) {
            iconv_close(converter);
            return std::nullopt;
        }
        const std::size_t written = result.size() - outputRemaining;
        result.resize(result.size() * 2U);
        outputPointer = result.data() + written;
        outputRemaining = result.size() - written;
    }
    iconv_close(converter);
    result.resize(result.size() - outputRemaining);
    return result;
#else
    (void)source;
    return std::nullopt;
#endif
}

std::string decodeLatin1(const std::vector<unsigned char>& source) {
    const QByteArray bytes(reinterpret_cast<const char*>(source.data()),
                           static_cast<qsizetype>(source.size()));
    return QString::fromLatin1(bytes).toUtf8().toStdString();
}

std::string stripInlineComment(std::string line) {
    const auto slash = line.find("//");
    if (slash != std::string::npos) line.resize(slash);
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    return trimAscii(std::move(line));
}

std::optional<std::string> parseScalar(const std::string& text, const std::string& key) {
    const std::regex pattern("^\\s*" + key + "\\s*=\\s*(.+?)\\s*$",
                             std::regex_constants::icase);
    std::istringstream lines(text);
    for (std::string line; std::getline(lines, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::smatch match;
        if (!std::regex_match(line, match, pattern)) continue;
        std::string value = stripInlineComment(match[1].str());
        if (value.empty()) return std::nullopt;
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"')
                                  || (value.front() == '\'' && value.back() == '\''))) {
            value = trimAscii(value.substr(1, value.size() - 2));
        }
        return value.empty() ? std::nullopt : std::optional<std::string>(std::move(value));
    }
    return std::nullopt;
}

std::optional<double> safeFloat(const std::optional<std::string>& text) {
    if (!text) return std::nullopt;
    std::size_t consumed = 0;
    try {
        const double value = std::stod(*text, &consumed);
        if (consumed == text->size() && std::isfinite(value)) return value;
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

std::optional<std::pair<double, double>> parsePair(const std::string& text,
                                                   const std::string& key) {
    const auto scalar = parseScalar(text, key);
    if (!scalar || scalar->size() < 2 || scalar->front() != '[' || scalar->back() != ']') {
        return std::nullopt;
    }
    const std::string inside = scalar->substr(1, scalar->size() - 2);
    const auto comma = inside.find(',');
    if (comma == std::string::npos) return std::nullopt;
    try {
        std::size_t consumed0 = 0;
        std::size_t consumed1 = 0;
        const std::string first = trimAscii(inside.substr(0, comma));
        std::string second = trimAscii(inside.substr(comma + 1));
        const auto nextComma = second.find(',');
        if (nextComma != std::string::npos) second.resize(nextComma);
        const double a = std::stod(first, &consumed0);
        const double b = std::stod(second, &consumed1);
        if (consumed0 == first.size() && consumed1 == second.size()) return std::pair{a, b};
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

ProjectLayout loadProjectLayout(const fs::path& configInput) {
    const fs::path configPath = normalizedAbsolute(configInput);
    const std::string extension = lowerAscii(pathUtf8(configPath.extension()));
    if (extension != ".luw" && extension != ".luwdg" && extension != ".luwpf") {
        throw std::runtime_error("Config file must have extension .luw, .luwdg, or .luwpf");
    }
    if (!fs::is_regular_file(configPath)) {
        throw std::runtime_error("Config file not found: " + pathUtf8(configPath));
    }
    const fs::path project = configPath.parent_path();
    const fs::path windBc = project / "wind_bc";
    if (!fs::is_directory(windBc)) {
        throw std::runtime_error("wind_bc directory not found: " + pathUtf8(windBc));
    }
    const fs::path profile = windBc / "profile.dat";
    if (!fs::is_regular_file(profile)) {
        throw std::runtime_error("profile.dat not found: " + pathUtf8(profile));
    }
    std::vector<fs::path> windroses;
    for (const fs::directory_entry& entry : fs::directory_iterator(windBc)) {
        std::error_code error;
        if (!entry.is_regular_file(error) || error) continue;
        const std::string name = pathUtf8(entry.path().filename());
#ifdef _WIN32
        const std::string matchName = lowerAscii(name);
#else
        const std::string& matchName = name;
#endif
        if (matchName.starts_with("windrose_") && matchName.ends_with("m.csv")) {
            windroses.push_back(entry.path());
        }
    }
    std::sort(windroses.begin(), windroses.end());
    if (windroses.empty()) {
        throw std::runtime_error("No windrose_*m.csv found under: " + pathUtf8(windBc));
    }
    if (windroses.size() != 1) {
        std::string message = "Expected exactly one windrose_*m.csv under wind_bc, found: ";
        for (std::size_t index = 0; index < windroses.size(); ++index) {
            if (index != 0) message += ", ";
            message += pathUtf8(windroses[index].filename());
        }
        throw std::runtime_error(message);
    }

    const std::string rawConfig = decodeUtf8IgnoringErrors(readBytes(configPath));
    const auto dpiRaw = safeFloat(parseScalar(rawConfig, "crop_vis_dpi"));
    int figureDpi = dpiRaw ? static_cast<int>(std::nearbyint(*dpiRaw)) : 1200;
    if (figureDpi <= 0) figureDpi = 1200;
    std::optional<double> zLimit = safeFloat(parseScalar(rawConfig, "z_limit"));
    if (!zLimit) {
        const auto pair = parsePair(rawConfig, "si_z_cfd");
        const auto base = safeFloat(parseScalar(rawConfig, "base_height"));
        if (pair && base) {
            const double candidate = std::max(pair->first, pair->second) - *base;
            if (std::isfinite(candidate) && candidate > 0.0) zLimit = candidate;
        }
    }

    const fs::path output = project / "RESULTS" / "season_average";
    const fs::path figures = output / "figures";
    fs::create_directories(output);
    fs::create_directories(figures);
    return ProjectLayout{configPath,
                         project,
                         windBc,
                         profile,
                         normalizedAbsolute(windroses.front()),
                         normalizedAbsolute(output),
                         normalizedAbsolute(figures),
                         normalizedAbsolute(output / "season_average_summary.txt"),
                         normalizedAbsolute(output / "season_average.vtk"),
                         figureDpi,
                         zLimit};
}

Profile parseProfile(const fs::path& path, std::optional<double> zLimit,
                     std::ostream& output) {
    const std::string text = decodeUtf8IgnoringErrors(readBytes(path));
    std::vector<std::pair<double, double>> samples;
    std::istringstream lines(text);
    std::string raw;
    std::size_t lineNumber = 0;
    while (std::getline(lines, raw)) {
        ++lineNumber;
        std::string line = stripInlineComment(raw);
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), ';', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream values(line);
        std::string zToken;
        std::string uToken;
        if (!(values >> zToken >> uToken)) continue;
        try {
            std::size_t zConsumed = 0;
            std::size_t uConsumed = 0;
            const double z = std::stod(zToken, &zConsumed);
            const double u = std::stod(uToken, &uConsumed);
            if (zConsumed != zToken.size() || uConsumed != uToken.size()) continue;
            if (!std::isfinite(z) || !std::isfinite(u)) {
                log(output, "WARNING: invalid profile sample at line " + std::to_string(lineNumber)
                                + ", skipped");
                continue;
            }
            samples.emplace_back(z, u);
        } catch (const std::exception&) {
        }
    }
    if (samples.size() < 2) {
        throw std::runtime_error("profile.dat must contain at least two valid z/U samples");
    }
    std::stable_sort(samples.begin(), samples.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    Profile profile;
    for (const auto& [z, u] : samples) {
        if (!profile.z.empty() && std::abs(z - profile.z.back()) < 1.0e-9) {
            profile.u.back() = u;
        } else {
            profile.z.push_back(z);
            profile.u.push_back(u);
        }
    }
    if (profile.z.size() < 2) {
        throw std::runtime_error("profile.dat collapsed to fewer than two unique z samples");
    }
    if (zLimit && *zLimit > 1.0 && profile.z.back() <= 1.5) {
        log(output, "profile.dat vertical coordinate looks normalized. Scale z by z_limit="
                        + formatNumber(*zLimit) + " m.");
        for (double& z : profile.z) z *= *zLimit;
    }
    if (*std::max_element(profile.u.begin(), profile.u.end()) <= 0.0) {
        throw std::runtime_error("profile.dat has non-positive wind speeds only");
    }
    return profile;
}

std::pair<double, std::string> interpolateProfile(const Profile& profile, double target) {
    if (target <= profile.z.front()) {
        if (std::abs(target - profile.z.front()) <= 1.0e-9) {
            return {profile.u.front(), "exact at lower bound"};
        }
        const double factor = (target - profile.z[0]) / (profile.z[1] - profile.z[0]);
        return {profile.u[0] + factor * (profile.u[1] - profile.u[0]),
                "linear extrapolation below profile range"};
    }
    if (target >= profile.z.back()) {
        if (std::abs(target - profile.z.back()) <= 1.0e-9) {
            return {profile.u.back(), "exact at upper bound"};
        }
        const std::size_t last = profile.z.size() - 1;
        const double factor = (target - profile.z[last - 1])
            / (profile.z[last] - profile.z[last - 1]);
        return {profile.u[last - 1] + factor * (profile.u[last] - profile.u[last - 1]),
                "linear extrapolation above profile range"};
    }
    const auto found = std::lower_bound(profile.z.begin(), profile.z.end(), target);
    const std::size_t upper = static_cast<std::size_t>(found - profile.z.begin());
    const std::size_t lower = upper - 1;
    if (std::abs(target - profile.z[lower]) <= 1.0e-9) {
        return {profile.u[lower], "exact sample"};
    }
    if (std::abs(target - profile.z[upper]) <= 1.0e-9) {
        return {profile.u[upper], "exact sample"};
    }
    const double factor = (target - profile.z[lower])
        / (profile.z[upper] - profile.z[lower]);
    return {profile.u[lower] + factor * (profile.u[upper] - profile.u[lower]),
            "linear interpolation"};
}

double parseCustomFloat(std::string text, bool allowInf = true) {
    text = lowerAscii(trimAscii(std::move(text)));
    auto eraseAll = [&](std::string_view needle) {
        for (std::size_t pos = text.find(needle); pos != std::string::npos;
             pos = text.find(needle)) {
            text.erase(pos, needle.size());
        }
    };
    eraseAll("%");
    eraseAll("m/s");
    eraseAll("ms-1");
    eraseAll(" ");
    std::replace(text.begin(), text.end(), 'p', '.');
    if (allowInf && (text == "inf" || text == "+inf" || text == "infinity"
                     || text == "+infinity")) {
        return std::numeric_limits<double>::infinity();
    }
    std::replace(text.begin(), text.end(), ',', '.');
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size()) throw std::invalid_argument("invalid number");
    return value;
}

double parseWindroseHeight(const fs::path& path) {
    const std::string stem = pathUtf8(path.stem());
    const std::regex pattern(R"(([0-9]+(?:[p\.][0-9]+)?)m)", std::regex_constants::icase);
    std::optional<std::string> last;
    for (std::sregex_iterator it(stem.begin(), stem.end(), pattern), end; it != end; ++it) {
        last = (*it)[1].str();
    }
    if (!last) {
        throw std::runtime_error("Cannot parse reference height from windrose file name: "
                                 + pathUtf8(path.filename()));
    }
    return parseCustomFloat(*last, false);
}

struct DecodedCsv final {
    std::vector<std::vector<std::string>> rows;
    std::string encoding;
};

std::vector<std::vector<std::string>> parseCsv(const std::string& utf8) {
    const QString text = QString::fromUtf8(utf8.data(), static_cast<qsizetype>(utf8.size()));
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    QString field;
    bool quoted = false;
    for (qsizetype index = 0; index < text.size(); ++index) {
        const QChar character = text[index];
        if (character == QChar('"')) {
            if (quoted && index + 1 < text.size() && text[index + 1] == QChar('"')) {
                field.append(QChar('"'));
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (!quoted && character == QChar(',')) {
            row.push_back(field.trimmed().toUtf8().toStdString());
            field.clear();
        } else if (!quoted && (character == QChar('\n') || character == QChar('\r'))) {
            if (character == QChar('\r') && index + 1 < text.size()
                && text[index + 1] == QChar('\n')) {
                ++index;
            }
            row.push_back(field.trimmed().toUtf8().toStdString());
            field.clear();
            rows.push_back(std::move(row));
            row.clear();
        } else {
            field.append(character);
        }
    }
    if (!field.isEmpty() || !row.empty() || (!text.isEmpty() && text.back() == QChar(','))) {
        row.push_back(field.trimmed().toUtf8().toStdString());
        rows.push_back(std::move(row));
    }
    return rows;
}

DecodedCsv readCsv(const fs::path& path) {
    const std::vector<unsigned char> bytes = readBytes(path);
    if (const auto utf8 = decodeUtf8Strict(bytes, true)) {
        return {parseCsv(*utf8), "utf-8-sig"};
    }
    if (const auto gbk = decodeGbk(bytes)) return {parseCsv(*gbk), "gbk"};
    return {parseCsv(decodeLatin1(bytes)), "latin-1"};
}

std::string normalizeLabel(const std::string& text) {
    std::string result;
    for (const unsigned char character : upperAscii(text)) {
        if ((character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9')
            || character == '.') {
            result.push_back(static_cast<char>(character));
        }
    }
    return result;
}

const std::unordered_map<std::string, double>& compassAngles() {
    static const std::unordered_map<std::string, double> values{
        {"N",0.0},{"NORTH",0.0},{"NNE",22.5},{"NORTHNORTHEAST",22.5},
        {"NE",45.0},{"NORTHEAST",45.0},{"ENE",67.5},{"EASTNORTHEAST",67.5},
        {"E",90.0},{"EAST",90.0},{"ESE",112.5},{"EASTSOUTHEAST",112.5},
        {"SE",135.0},{"SOUTHEAST",135.0},{"SSE",157.5},{"SOUTHSOUTHEAST",157.5},
        {"S",180.0},{"SOUTH",180.0},{"SSW",202.5},{"SOUTHSOUTHWEST",202.5},
        {"SW",225.0},{"SOUTHWEST",225.0},{"WSW",247.5},{"WESTSOUTHWEST",247.5},
        {"W",270.0},{"WEST",270.0},{"WNW",292.5},{"WESTNORTHWEST",292.5},
        {"NW",315.0},{"NORTHWEST",315.0},{"NNW",337.5},{"NORTHNORTHWEST",337.5}};
    return values;
}

std::optional<double> snapAngle(double angle) {
    double wrapped = std::fmod(angle, 360.0);
    if (wrapped < 0.0) wrapped += 360.0;
    const auto found = std::min_element(angleSequence.begin(), angleSequence.end(),
        [wrapped](double left, double right) {
            return std::abs(left - wrapped) < std::abs(right - wrapped);
        });
    if (found == angleSequence.end() || std::abs(*found - wrapped) > 0.51) return std::nullopt;
    return *found;
}

std::optional<double> parseDirectionLabel(const std::string& text) {
    const std::string raw = trimAscii(text);
    if (raw.empty()) return std::nullopt;
    const std::string normalized = normalizeLabel(raw);
    static const std::set<std::string> ignored{
        "", "ALL", "ALLDIRECTIONS", "TOTAL", "SUM", "CALM", "CALMS", "VARIABLE", "VAR"};
    if (ignored.contains(normalized)) return std::nullopt;
    if (const auto found = compassAngles().find(normalized); found != compassAngles().end()) {
        return found->second;
    }
    const std::regex number(R"([-+]?\d+(?:[\.]\d+)?)");
    std::smatch match;
    if (!std::regex_search(raw, match, number)) return std::nullopt;
    try {
        return snapAngle(std::stod(match[0].str()));
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<SpeedBin> parseSpeedBins(const std::vector<std::string>& headers,
                                     std::ostream& output,
                                     bool require = true) {
    const std::regex pattern(
        R"(^C(\d+)_([-+]?[0-9]+(?:[p\.][0-9]+)?)_([-+]?[0-9]+(?:[p\.][0-9]+)?|inf)$)",
        std::regex_constants::icase);
    struct Raw final {
        std::size_t index;
        std::string header;
        int classId;
        double lower;
        std::optional<double> upper;
    };
    std::vector<Raw> raw;
    std::vector<double> widths;
    for (std::size_t index = 0; index < headers.size(); ++index) {
        const std::string header = trimAscii(headers[index]);
        std::smatch match;
        if (!std::regex_match(header, match, pattern)) continue;
        const int classId = std::stoi(match[1].str());
        const double lower = parseCustomFloat(match[2].str(), false);
        std::optional<double> upper;
        if (lowerAscii(match[3].str()) != "inf") {
            upper = parseCustomFloat(match[3].str(), false);
            const double width = *upper - lower;
            if (width <= 0.0) {
                throw std::runtime_error("Invalid finite windrose bin width in header: " + header);
            }
            widths.push_back(width);
        }
        raw.push_back(Raw{index, header, classId, lower, upper});
    }
    if (raw.empty()) {
        if (require) {
            throw std::runtime_error("No speed-bin columns like C7_7p0_9p4 found in windrose CSV header");
        }
        return {};
    }
    if (widths.empty()) {
        throw std::runtime_error("Windrose CSV contains no finite speed bins, cannot infer target speeds");
    }
    const double widthSum = std::accumulate(widths.begin(), widths.end(), 0.0);
    std::vector<SpeedBin> result;
    result.reserve(raw.size());
    for (const Raw& item : raw) {
        double target = 0.0;
        if (item.upper) {
            target = 0.5 * (item.lower + *item.upper);
        } else {
            target = item.lower + 0.5 * widthSum;
            log(output, "Infinite speed bin target inferred: " + item.header + " -> "
                            + formatNumber(target) + " m/s (lower " + formatNumber(item.lower)
                            + " + 0.5 * finite-width-sum " + formatNumber(widthSum) + ")");
        }
        result.push_back(SpeedBin{item.index, item.header, item.classId,
                                  item.lower, item.upper, target});
    }
    return result;
}

Windrose parseWindrose(const fs::path& path, std::ostream& output) {
    const DecodedCsv decoded = readCsv(path);
    log(output, "Windrose CSV encoding used: " + decoded.encoding);
    std::optional<std::size_t> headerIndex;
    std::vector<SpeedBin> bins;
    const std::regex speedPrefix(R"(^C\d+_)", std::regex_constants::icase);
    for (std::size_t rowIndex = 0; rowIndex < decoded.rows.size(); ++rowIndex) {
        bool potential = false;
        for (const std::string& cell : decoded.rows[rowIndex]) {
            if (std::regex_search(trimAscii(cell), speedPrefix)) {
                potential = true;
                break;
            }
        }
        if (!potential) continue;
        std::vector<SpeedBin> candidate = parseSpeedBins(decoded.rows[rowIndex], output, false);
        if (!candidate.empty()) {
            headerIndex = rowIndex;
            bins = std::move(candidate);
            break;
        }
    }
    if (!headerIndex) {
        throw std::runtime_error("Failed to locate windrose CSV header row containing speed-bin columns");
    }
    Windrose result;
    result.bins = bins;
    for (const double angle : angleSequence) {
        result.probabilities.emplace(angle, std::vector<double>(bins.size(), 0.0));
    }
    const std::size_t firstSpeedColumn = std::min_element(
        bins.begin(), bins.end(), [](const SpeedBin& left, const SpeedBin& right) {
            return left.columnIndex < right.columnIndex;
        })->columnIndex;
    for (std::size_t rowIndex = *headerIndex + 1; rowIndex < decoded.rows.size(); ++rowIndex) {
        const auto& row = decoded.rows[rowIndex];
        if (row.empty() || std::none_of(row.begin(), row.end(), [](const std::string& value) {
                return !trimAscii(value).empty();
            })) {
            continue;
        }
        std::optional<double> angle;
        std::optional<double> fallback;
        const std::size_t limit = std::min(firstSpeedColumn, row.size());
        for (std::size_t column = 0; column < limit; ++column) {
            const std::string cell = trimAscii(row[column]);
            if (cell.empty()) continue;
            const std::string normalized = normalizeLabel(cell);
            if (const auto found = compassAngles().find(normalized); found != compassAngles().end()) {
                angle = found->second;
                break;
            }
            if (!fallback) fallback = parseDirectionLabel(cell);
        }
        if (!angle) angle = fallback;
        if (!angle) continue;
        auto& probabilities = result.probabilities.at(*angle);
        for (std::size_t binIndex = 0; binIndex < bins.size(); ++binIndex) {
            const SpeedBin& bin = bins[binIndex];
            if (bin.columnIndex >= row.size()) continue;
            const std::string cell = trimAscii(row[bin.columnIndex]);
            if (cell.empty()) continue;
            const std::string normalized = lowerAscii(cell);
            if (normalized == "-" || normalized == "--" || normalized == "nan"
                || normalized == "na" || normalized == "n/a") {
                continue;
            }
            try {
                const double value = parseCustomFloat(cell, false);
                if (std::isfinite(value)) probabilities[binIndex] += value;
            } catch (const std::exception&) {
            }
        }
    }
    double rawTotal = 0.0;
    for (const auto& [angle, values] : result.probabilities) {
        (void)angle;
        rawTotal += std::accumulate(values.begin(), values.end(), 0.0);
    }
    if (rawTotal <= 0.0) {
        throw std::runtime_error("Windrose CSV contains no valid probability values in direction-speed cells");
    }
    const double scale = rawTotal > 1.5 ? 0.01 : 1.0;
    const std::string mode = rawTotal > 1.5 ? "percentage" : "fraction";
    for (auto& [angle, values] : result.probabilities) {
        (void)angle;
        for (double& value : values) value *= scale;
    }
    result.totalProbability = 0.0;
    for (const auto& [angle, values] : result.probabilities) {
        (void)angle;
        result.totalProbability += std::accumulate(values.begin(), values.end(), 0.0);
    }
    log(output, "Windrose raw probability total: " + formatNumber(rawTotal) + " (" + mode
                    + " mode -> scaled total " + formatNumber(result.totalProbability) + ")");
    return result;
}

std::vector<DirectionWeight> computeWeights(const Windrose& windrose,
                                            double referenceSpeed,
                                            std::ostream& output) {
    if (referenceSpeed <= 0.0) throw std::runtime_error("Reference speed must be positive");
    if (windrose.totalProbability <= 0.0) {
        throw std::runtime_error("Total windrose probability must be positive");
    }
    const double normalization = 1.0 / windrose.totalProbability;
    std::vector<double> ratios;
    ratios.reserve(windrose.bins.size());
    for (const SpeedBin& bin : windrose.bins) ratios.push_back(bin.targetSpeed / referenceSpeed);

    std::vector<DirectionWeight> result;
    result.reserve(angleSequence.size());
    logRule(output, "Direction Weights");
    for (std::size_t angleIndex = 0; angleIndex < angleSequence.size(); ++angleIndex) {
        const double angle = angleSequence[angleIndex];
        const auto found = windrose.probabilities.find(angle);
        const std::vector<double> empty(windrose.bins.size(), 0.0);
        const std::vector<double>& joint = found == windrose.probabilities.end() ? empty : found->second;
        double probability = 0.0;
        double velocityWeight = 0.0;
        double tkeWeight = 0.0;
        for (std::size_t index = 0; index < windrose.bins.size(); ++index) {
            const double normalized = joint[index] * normalization;
            probability += normalized;
            velocityWeight += normalized * ratios[index];
            tkeWeight += normalized * ratios[index] * ratios[index];
        }
        DirectionWeight weight{angle, std::string(angleDirections[angleIndex]),
                               probability, velocityWeight, tkeWeight};
        result.push_back(weight);
        std::ostringstream line;
        line << "Angle " << std::setw(5) << formatNumber(angle) << " deg / "
             << std::setw(3) << weight.label << ": prob=" << std::fixed << std::setprecision(6)
             << probability << ", vel_weight=" << velocityWeight
             << ", tke_weight=" << tkeWeight;
        log(output, line.str());
    }
    return result;
}

std::optional<int> extractStep(const std::string& stem) {
    static const std::regex pattern(R"(_avg[-_](\d+)(?:_cropped)?$)",
                                    std::regex_constants::icase);
    std::smatch match;
    if (!std::regex_search(stem, match, pattern)) return std::nullopt;
    return std::stoi(match[1].str());
}

std::optional<double> tokenToFloat(std::string token) {
    token = lowerAscii(trimAscii(std::move(token)));
    std::replace(token.begin(), token.end(), 'p', '.');
    try {
        std::size_t consumed = 0;
        const double value = std::stod(token, &consumed);
        if (consumed == token.size() && std::isfinite(value)) return value;
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

std::optional<double> parseAngleFromVtkName(const fs::path& path) {
    const std::string stem = pathUtf8(path.stem());
    std::vector<std::string> tokens;
    std::size_t begin = 0;
    while (begin <= stem.size()) {
        const std::size_t separator = stem.find('_', begin);
        const std::size_t end = separator == std::string::npos ? stem.size() : separator;
        if (end > begin) tokens.push_back(stem.substr(begin, end - begin));
        if (separator == std::string::npos) break;
        begin = separator + 1;
    }
    std::size_t limit = tokens.size();
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (lowerAscii(tokens[index]).find("avg") != std::string::npos) {
            limit = index;
            break;
        }
    }
    std::vector<double> candidates;
    for (std::size_t index = 0; index < limit; ++index) {
        const auto numeric = tokenToFloat(tokens[index]);
        if (!numeric) continue;
        const auto snapped = snapAngle(*numeric);
        if (snapped) candidates.push_back(*snapped);
    }
    return candidates.empty() ? std::nullopt : std::optional<double>(candidates.back());
}

bool preferredVtk(const fs::path& left, const fs::path& right) {
    const std::string leftStem = pathUtf8(left.stem());
    const std::string rightStem = pathUtf8(right.stem());
    const int leftCropped = lowerAscii(leftStem).find("cropped") != std::string::npos ? 1 : 0;
    const int rightCropped = lowerAscii(rightStem).find("cropped") != std::string::npos ? 1 : 0;
    if (leftCropped != rightCropped) return leftCropped > rightCropped;
    const int leftStep = extractStep(leftStem).value_or(-1);
    const int rightStep = extractStep(rightStem).value_or(-1);
    if (leftStep != rightStep) return leftStep > rightStep;
    std::error_code leftError;
    std::error_code rightError;
    const auto leftTime = fs::last_write_time(left, leftError);
    const auto rightTime = fs::last_write_time(right, rightError);
    if (!leftError && !rightError && leftTime != rightTime) return leftTime > rightTime;
    return pathUtf8(left.filename()) > pathUtf8(right.filename());
}

std::optional<VtkDirectoryChoice> scanVtkDirectory(const fs::path& directory,
                                                   const std::string& label,
                                                   std::ostream& output) {
    if (!fs::is_directory(directory)) return std::nullopt;
    std::vector<fs::path> files;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        std::error_code error;
        if (!entry.is_regular_file(error) || error) continue;
        const std::string name = pathUtf8(entry.path().filename());
#ifdef _WIN32
        const std::string matchName = lowerAscii(name);
#else
        const std::string& matchName = name;
#endif
        if (matchName.find("avg") != std::string::npos && matchName.ends_with(".vtk")) {
            files.push_back(normalizedAbsolute(entry.path()));
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) return std::nullopt;
    std::map<double, std::vector<fs::path>> grouped;
    std::vector<std::string> ignored;
    for (const fs::path& file : files) {
        const auto angle = parseAngleFromVtkName(file);
        if (angle) grouped[*angle].push_back(file);
        else ignored.push_back(pathUtf8(file.filename()));
    }
    if (!ignored.empty()) {
        std::string message = label + ": ignored " + std::to_string(ignored.size())
            + " avg-like VTK file(s) without a resolvable angle token: ";
        const std::size_t shown = std::min<std::size_t>(6, ignored.size());
        for (std::size_t index = 0; index < shown; ++index) {
            if (index != 0) message += ", ";
            message += ignored[index];
        }
        if (ignored.size() > shown) message += " ...";
        log(output, message);
    }
    if (grouped.empty()) return std::nullopt;
    VtkDirectoryChoice choice{label, normalizedAbsolute(directory), {}};
    for (auto& [angle, candidates] : grouped) {
        std::stable_sort(candidates.begin(), candidates.end(), preferredVtk);
        choice.angleToFile.emplace(angle, candidates.front());
        if (candidates.size() > 1) {
            log(output, label + ": multiple VTK files for angle " + formatNumber(angle)
                            + " deg, choose " + pathUtf8(candidates.front().filename()) + " from "
                            + std::to_string(candidates.size()) + " candidates.");
        }
    }
    return choice;
}

VtkDirectoryChoice chooseVtkSource(const fs::path& project,
                                   const std::vector<DirectionWeight>& weights,
                                   const std::optional<fs::path>& explicitDirectory,
                                   std::ostream& output) {
    std::vector<VtkDirectoryChoice> candidates;
    fs::path resolvedExplicit;
    if (explicitDirectory) {
        resolvedExplicit = explicitDirectory->is_absolute()
            ? *explicitDirectory : project / *explicitDirectory;
        if (auto found = scanVtkDirectory(resolvedExplicit, "explicit vtk-dir", output)) {
            candidates.push_back(std::move(*found));
        }
    } else {
        if (auto found = scanVtkDirectory(project / "RESULTS" / "crop" / "cropped_vtk",
                                          "preferred cropped_vtk", output)) {
            candidates.push_back(std::move(*found));
        }
        if (auto found = scanVtkDirectory(project / "RESULTS" / "vtk",
                                          "fallback RESULTS/vtk", output)) {
            candidates.push_back(std::move(*found));
        }
    }
    if (candidates.empty()) {
        if (explicitDirectory) {
            throw std::runtime_error("No suitable *avg*.vtk files found under explicit vtk-dir: "
                                     + pathUtf8(resolvedExplicit));
        }
        throw std::runtime_error("No suitable *avg*.vtk files found under "
                                 + pathUtf8(project / "RESULTS" / "crop" / "cropped_vtk")
                                 + " or " + pathUtf8(project / "RESULTS" / "vtk"));
    }
    std::vector<double> required;
    for (const DirectionWeight& weight : weights) {
        if (weight.probability > 1.0e-10) required.push_back(weight.angle);
    }
    logRule(output, "VTK Directory Selection");
    for (VtkDirectoryChoice& candidate : candidates) {
        log(output, "Candidate " + candidate.label + ": " + pathUtf8(candidate.directory));
        std::string available;
        for (const auto& [angle, path] : candidate.angleToFile) {
            if (!available.empty()) available += ", ";
            available += formatNumber(angle) + ":" + pathUtf8(path.filename());
        }
        log(output, "  Available angles: " + available);
        std::vector<double> missing;
        for (const double angle : required) {
            if (!candidate.angleToFile.contains(angle)) missing.push_back(angle);
        }
        if (missing.empty()) {
            log(output, "Selected source directory: " + pathUtf8(candidate.directory)
                            + " (" + candidate.label + ")");
            return std::move(candidate);
        }
        std::string message = "  Missing required non-zero-probability angles: ";
        for (std::size_t index = 0; index < missing.size(); ++index) {
            if (index != 0) message += ", ";
            message += formatNumber(missing[index]);
        }
        log(output, message);
    }
    throw std::runtime_error(
        "No VTK directory covers all directions with non-zero windrose probability. "
        "Check the selected VTK directory contents.");
}

std::string normalizeFieldToken(const std::string& name) {
    std::string result;
    for (const unsigned char value : lowerAscii(name)) {
        if ((value >= 'a' && value <= 'z') || (value >= '0' && value <= '9')) {
            result.push_back(static_cast<char>(value));
        }
    }
    return result;
}

const LegacyVtkField* findFieldCandidate(const LegacyStructuredPoints& vtk,
                                         const std::vector<std::string>& candidates,
                                         std::optional<std::size_t> exactComponents = std::nullopt,
                                         std::optional<std::size_t> minimumComponents = std::nullopt) {
    for (const std::string& candidate : candidates) {
        const std::string candidateLower = lowerAscii(candidate);
        const std::string candidateNormalized = normalizeFieldToken(candidate);
        for (const LegacyVtkField& field : vtk.fields) {
            if (lowerAscii(field.name) != candidateLower
                && normalizeFieldToken(field.name) != candidateNormalized) {
                continue;
            }
            if (exactComponents && field.components != *exactComponents) continue;
            if (minimumComponents && field.components < *minimumComponents) continue;
            return &field;
        }
    }
    return nullptr;
}

WindFieldSelection resolveWind(const LegacyStructuredPoints& vtk) {
    if (const LegacyVtkField* vector = findFieldCandidate(
            vtk, {"u_avg", "velocity", "wind", "uvw", "u"}, std::nullopt, 3U)) {
        return WindFieldSelection{vector, {nullptr, nullptr, nullptr}};
    }
    for (const LegacyVtkField& field : vtk.fields) {
        if (field.tuples == vtk.pointCount && field.components >= 3
            && field.association == LegacyVtkAssociation::PointData) {
            return WindFieldSelection{&field, {nullptr, nullptr, nullptr}};
        }
    }
    const LegacyVtkField* u = findFieldCandidate(
        vtk, {"u", "u_avg_x", "ux", "u_x", "velocity_x", "u_filtered"}, 1U);
    const LegacyVtkField* v = findFieldCandidate(
        vtk, {"v", "u_avg_y", "uy", "u_y", "velocity_y", "v_filtered"}, 1U);
    const LegacyVtkField* w = findFieldCandidate(
        vtk, {"w", "u_avg_z", "uz", "u_z", "velocity_z", "w_filtered"}, 1U);
    if (u != nullptr && v != nullptr && w != nullptr) {
        return WindFieldSelection{nullptr, {u, v, w}};
    }
    std::string fields;
    for (const LegacyVtkField& field : vtk.fields) {
        if (!fields.empty()) fields += ", ";
        fields += field.name;
    }
    throw std::runtime_error("VTK missing wind vector field. Available fields=[" + fields + "]");
}

const LegacyVtkField& resolveTke(const LegacyStructuredPoints& vtk) {
    if (const LegacyVtkField* exact = vtk.field("tke")) return *exact;
    if (const LegacyVtkField* field = findFieldCandidate(
            vtk, {"tke", "k", "tke_avg", "k_avg", "sgs_tke", "turbulence_k",
                  "tkenergy", "tke_filtered"})) {
        return *field;
    }
    std::string fields;
    for (const LegacyVtkField& field : vtk.fields) {
        if (!fields.empty()) fields += ", ";
        fields += field.name;
    }
    throw std::runtime_error("VTK missing TKE scalar field. Available fields=[" + fields + "]");
}

HeaderSelection validateHeaders(const VtkDirectoryChoice& choice,
                                const std::vector<DirectionWeight>& weights,
                                std::ostream& output) {
    HeaderSelection result;
    bool haveReference = false;
    for (const DirectionWeight& weight : weights) {
        if (weight.probability <= 1.0e-10) continue;
        const fs::path& path = choice.angleToFile.at(weight.angle);
        log(output, "Read header for angle " + formatNumber(weight.angle) + " deg from "
                        + pathUtf8(path.filename()));
        LegacyStructuredPoints vtk = LegacyStructuredPoints::readHeader(path);
        const WindFieldSelection wind = resolveWind(vtk);
        const LegacyVtkField& tke = resolveTke(vtk);
        log(output, "  Grid signature: dims=" + tupleText(vtk.dimensions)
                        + ", origin=" + tupleText(vtk.origin) + ", spacing="
                        + tupleText(vtk.spacing) + ", wind=" + wind.displayName()
                        + ", tke=" + tke.name);
        if (!haveReference) {
            result.reference = vtk;
            haveReference = true;
        } else if (vtk.dimensions != result.reference.dimensions
                   || vtk.origin != result.reference.origin
                   || vtk.spacing != result.reference.spacing) {
            throw std::runtime_error(
                "Input VTK grids are not identical across directions. Reference=("
                + tupleText(result.reference.dimensions) + ", " + tupleText(result.reference.origin)
                + ", " + tupleText(result.reference.spacing) + "), current=("
                + tupleText(vtk.dimensions) + ", " + tupleText(vtk.origin) + ", "
                + tupleText(vtk.spacing) + "), file=" + pathUtf8(path));
        }
        result.byAngle.emplace(weight.angle, std::move(vtk));
    }
    if (!haveReference) {
        throw std::runtime_error("No non-zero-probability direction remained after windrose weighting");
    }
    return result;
}

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           std::string_view context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::runtime_error("Size overflow for " + std::string(context));
    }
    return left * right;
}

class MappedFloatFile final {
public:
    MappedFloatFile() = default;
    MappedFloatFile(const MappedFloatFile&) = delete;
    MappedFloatFile& operator=(const MappedFloatFile&) = delete;
    ~MappedFloatFile() {
        if (mapped_ != nullptr && file_) file_->unmap(mapped_);
        if (file_) file_->close();
    }

    void create(const fs::path& path, std::size_t count) {
        if (count == 0 || count > static_cast<std::size_t>(std::numeric_limits<qint64>::max() / 4)) {
            throw std::runtime_error("Invalid mapped accumulator size");
        }
        count_ = count;
        file_ = std::make_unique<QFile>(qPath(path));
        if (!file_->open(QIODevice::ReadWrite | QIODevice::Truncate)) {
            throw std::runtime_error("Failed to create mapped accumulator: " + pathUtf8(path));
        }
        const qint64 bytes = static_cast<qint64>(count * sizeof(float));
        if (!file_->resize(bytes)) {
            throw std::runtime_error("Failed to resize mapped accumulator: " + pathUtf8(path));
        }
        mapped_ = file_->map(0, bytes);
        if (mapped_ == nullptr) {
            throw std::runtime_error("Failed to memory-map accumulator: " + pathUtf8(path));
        }
        std::memset(mapped_, 0, static_cast<std::size_t>(bytes));
    }

    [[nodiscard]] float* data() noexcept { return reinterpret_cast<float*>(mapped_); }
    [[nodiscard]] const float* data() const noexcept { return reinterpret_cast<const float*>(mapped_); }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    void flush() { if (file_) file_->flush(); }

private:
    std::unique_ptr<QFile> file_;
    unsigned char* mapped_ = nullptr;
    std::size_t count_ = 0;
};

class Accumulators final {
public:
    Accumulators(const fs::path& outputDirectory,
                 const std::array<int, 3>& dimensions,
                 std::ostream& output) {
        const std::size_t nx = static_cast<std::size_t>(dimensions[0]);
        const std::size_t ny = static_cast<std::size_t>(dimensions[1]);
        const std::size_t nz = static_cast<std::size_t>(dimensions[2]);
        points_ = checkedProduct(checkedProduct(nx, ny, "accumulator"), nz, "accumulator");
        const std::uint64_t totalBytes = static_cast<std::uint64_t>(points_) * 5ULL * sizeof(float);
        mapped_ = totalBytes > accumulatorMemoryThreshold;
        if (mapped_) {
            tempDirectory_ = std::make_unique<QTemporaryDir>(
                qPath(outputDirectory / "season_average_tmp_XXXXXX"));
            if (!tempDirectory_->isValid()) {
                throw std::runtime_error("Failed to create seasonal-average accumulator directory");
            }
            const fs::path temp = pathFromUtf8(tempDirectory_->path().toUtf8().toStdString());
            log(output, "Accumulator strategy: disk-backed memmap (estimated data volume "
                            + humanBytes(totalBytes) + ") at " + pathUtf8(temp));
            mappedU_.create(temp / "u_accum.bin", checkedProduct(points_, 3U, "u accumulator"));
            mappedVm_.create(temp / "vm_accum.bin", points_);
            mappedTke_.create(temp / "tke_accum.bin", points_);
        } else {
            log(output, "Accumulator strategy: in-memory arrays (estimated data volume "
                            + humanBytes(totalBytes) + ")");
            u_.assign(checkedProduct(points_, 3U, "u accumulator"), 0.0F);
            vm_.assign(points_, 0.0F);
            tke_.assign(points_, 0.0F);
        }
    }

    [[nodiscard]] float* u() noexcept { return mapped_ ? mappedU_.data() : u_.data(); }
    [[nodiscard]] float* vm() noexcept { return mapped_ ? mappedVm_.data() : vm_.data(); }
    [[nodiscard]] float* tke() noexcept { return mapped_ ? mappedTke_.data() : tke_.data(); }
    [[nodiscard]] const float* u() const noexcept { return mapped_ ? mappedU_.data() : u_.data(); }
    [[nodiscard]] const float* vm() const noexcept { return mapped_ ? mappedVm_.data() : vm_.data(); }
    [[nodiscard]] const float* tke() const noexcept { return mapped_ ? mappedTke_.data() : tke_.data(); }
    [[nodiscard]] std::size_t points() const noexcept { return points_; }
    void flush() {
        if (!mapped_) return;
        mappedU_.flush();
        mappedVm_.flush();
        mappedTke_.flush();
    }

private:
    bool mapped_ = false;
    std::size_t points_ = 0;
    std::unique_ptr<QTemporaryDir> tempDirectory_;
    MappedFloatFile mappedU_;
    MappedFloatFile mappedVm_;
    MappedFloatFile mappedTke_;
    std::vector<float> u_;
    std::vector<float> vm_;
    std::vector<float> tke_;
};

int progressStep(std::size_t total) {
    if (total <= 50) return 1;
    if (total <= 150) return 2;
    return 5;
}

int logProgress(std::ostream& output, const std::string& prefix,
                std::size_t current, std::size_t total, int lastPercent) {
    const int percent = static_cast<int>(std::nearbyint(
        100.0 * static_cast<double>(current) / static_cast<double>(std::max<std::size_t>(total, 1))));
    if (current == total || lastPercent < 0 || percent - lastPercent >= progressStep(total)) {
        log(output, prefix + ": " + std::to_string(current) + "/" + std::to_string(total)
                        + " (" + std::to_string(percent) + "%)");
        return percent;
    }
    return lastPercent;
}

void accumulateDirection(const LegacyStructuredPoints& vtk,
                         const WindFieldSelection& wind,
                         const LegacyVtkField& tkeField,
                         float velocityWeight,
                         float tkeWeight,
                         Accumulators& accumulators,
                         const std::string& angleLabel,
                         std::ostream& output) {
    const std::size_t nx = static_cast<std::size_t>(vtk.dimensions[0]);
    const std::size_t ny = static_cast<std::size_t>(vtk.dimensions[1]);
    const std::size_t nz = static_cast<std::size_t>(vtk.dimensions[2]);
    const std::size_t planePoints = checkedProduct(nx, ny, "VTK plane");
    LegacyVtkPlaneReader reader(vtk);
    if (std::abs(velocityWeight) > 0.0F) {
        log(output, "Start vector accumulation for " + angleLabel + ": field="
                        + wind.displayName() + ", velocity_weight=" + [&] {
                            std::ostringstream stream;
                            stream << std::fixed << std::setprecision(6) << velocityWeight;
                            return stream.str();
                        }());
        if (wind.hasComponents()) {
            log(output, "  Component scalar fields: u=" + wind.componentFields[0]->name
                            + ", v=" + wind.componentFields[1]->name
                            + ", w=" + wind.componentFields[2]->name);
        }
        int lastPercent = -1;
        for (std::size_t z = 0; z < nz; ++z) {
            std::vector<float> vector;
            std::array<std::vector<float>, 3> components;
            if (wind.vectorField != nullptr) {
                vector = reader.readPointPlane(*wind.vectorField, z);
                if (wind.vectorField->components < 3 || vector.size() < planePoints * 3U) {
                    throw std::runtime_error("Vector field '" + wind.vectorField->name + "' in "
                                             + pathUtf8(vtk.path) + " must have 3 components");
                }
            } else {
                for (std::size_t component = 0; component < 3; ++component) {
                    components[component] = reader.readPointPlane(*wind.componentFields[component], z);
                    if (wind.componentFields[component]->components != 1
                        || components[component].size() != planePoints) {
                        throw std::runtime_error("Component fields in " + pathUtf8(vtk.path)
                                                 + " must be scalar");
                    }
                }
            }
            const std::size_t pointBase = z * planePoints;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::int64_t signedPoint = 0;
                 signedPoint < static_cast<std::int64_t>(planePoints); ++signedPoint) {
                const std::size_t point = static_cast<std::size_t>(signedPoint);
                const float u = wind.vectorField != nullptr
                    ? vector[point * wind.vectorField->components] : components[0][point];
                const float v = wind.vectorField != nullptr
                    ? vector[point * wind.vectorField->components + 1U] : components[1][point];
                const float w = wind.vectorField != nullptr
                    ? vector[point * wind.vectorField->components + 2U] : components[2][point];
                const std::size_t global = pointBase + point;
                accumulators.u()[global * 3U] += velocityWeight * u;
                accumulators.u()[global * 3U + 1U] += velocityWeight * v;
                accumulators.u()[global * 3U + 2U] += velocityWeight * w;
                const float magnitude = std::sqrt(u * u + v * v + w * w);
                accumulators.vm()[global] += velocityWeight * magnitude;
            }
            lastPercent = logProgress(output, "  " + angleLabel + " vector progress",
                                      z + 1, nz, lastPercent);
        }
    } else {
        log(output, "Skip vector accumulation for " + angleLabel + ": velocity_weight is zero");
    }

    if (std::abs(tkeWeight) > 0.0F) {
        std::ostringstream weightText;
        weightText << std::fixed << std::setprecision(6) << tkeWeight;
        log(output, "Start TKE accumulation for " + angleLabel + ": field=" + tkeField.name
                        + ", tke_weight=" + weightText.str());
        int lastPercent = -1;
        for (std::size_t z = 0; z < nz; ++z) {
            const std::vector<float> tke = reader.readPointPlane(tkeField, z);
            if (tkeField.components != 1 || tke.size() != planePoints) {
                throw std::runtime_error("TKE field '" + tkeField.name + "' in "
                                         + pathUtf8(vtk.path) + " must be scalar");
            }
            const std::size_t pointBase = z * planePoints;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::int64_t signedPoint = 0;
                 signedPoint < static_cast<std::int64_t>(planePoints); ++signedPoint) {
                const std::size_t point = static_cast<std::size_t>(signedPoint);
                accumulators.tke()[pointBase + point] += tkeWeight * tke[point];
            }
            lastPercent = logProgress(output, "  " + angleLabel + " TKE progress",
                                      z + 1, nz, lastPercent);
        }
    } else {
        log(output, "Skip TKE accumulation for " + angleLabel + ": tke_weight is zero");
    }
}

std::unique_ptr<Accumulators> synthesize(const VtkDirectoryChoice& choice,
                                         const std::vector<DirectionWeight>& weights,
                                         HeaderSelection& headers,
                                         const fs::path& outputDirectory,
                                         std::ostream& output) {
    auto accumulators = std::make_unique<Accumulators>(
        outputDirectory, headers.reference.dimensions, output);
    for (const DirectionWeight& weight : weights) {
        if (weight.probability <= 1.0e-10) {
            log(output, "Skip angle " + formatNumber(weight.angle) + " deg / " + weight.label
                            + ": windrose probability is zero");
            continue;
        }
        LegacyStructuredPoints& vtk = headers.byAngle.at(weight.angle);
        const WindFieldSelection wind = resolveWind(vtk);
        const LegacyVtkField& tke = resolveTke(vtk);
        logRule(output, "Accumulate " + formatNumber(weight.angle) + " deg / " + weight.label);
        log(output, "Source VTK: " + pathUtf8(choice.angleToFile.at(weight.angle)));
        std::ostringstream probability;
        probability << std::fixed << std::setprecision(6) << weight.probability;
        log(output, "Probability contribution: " + probability.str());
        accumulateDirection(vtk, wind, tke,
                            static_cast<float>(weight.velocityWeight),
                            static_cast<float>(weight.tkeWeight),
                            *accumulators,
                            formatNumber(weight.angle) + " deg / " + weight.label,
                            output);
    }
    accumulators->flush();
    return accumulators;
}

void writeBigEndianFloatBlock(std::ofstream& output, std::span<const float> values) {
    constexpr std::size_t chunkValues = 262144;
    std::vector<unsigned char> bytes(std::min(chunkValues, values.size()) * sizeof(float));
    for (std::size_t offset = 0; offset < values.size(); offset += chunkValues) {
        const std::size_t count = std::min(chunkValues, values.size() - offset);
        bytes.resize(count * sizeof(float));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t signedIndex = 0;
             signedIndex < static_cast<std::int64_t>(count); ++signedIndex) {
            const std::size_t index = static_cast<std::size_t>(signedIndex);
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(values[offset + index]);
            unsigned char* destination = bytes.data() + index * sizeof(float);
            destination[0] = static_cast<unsigned char>((bits >> 24U) & 0xffU);
            destination[1] = static_cast<unsigned char>((bits >> 16U) & 0xffU);
            destination[2] = static_cast<unsigned char>((bits >> 8U) & 0xffU);
            destination[3] = static_cast<unsigned char>(bits & 0xffU);
        }
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) throw std::runtime_error("Failed while writing legacy VTK binary values");
    }
}

void writeHeader(std::ofstream& output,
                 const std::array<int, 3>& dimensions,
                 const std::array<double, 3>& origin,
                 const std::array<double, 3>& spacing,
                 std::string_view windroseName) {
    const std::size_t pointCount = checkedProduct(
        checkedProduct(static_cast<std::size_t>(dimensions[0]),
                       static_cast<std::size_t>(dimensions[1]), "output VTK"),
        static_cast<std::size_t>(dimensions[2]), "output VTK");
    output << "# vtk DataFile Version 3.0\n"
           << "Season average synthesized from " << windroseName << "\n"
           << "BINARY\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS " << dimensions[0] << ' ' << dimensions[1] << ' ' << dimensions[2] << "\n"
           << "ORIGIN " << pythonFloatString(origin[0]) << ' ' << pythonFloatString(origin[1]) << ' '
           << pythonFloatString(origin[2]) << "\n"
           << "SPACING " << pythonFloatString(spacing[0]) << ' ' << pythonFloatString(spacing[1]) << ' '
           << pythonFloatString(spacing[2]) << "\n\n"
           << "POINT_DATA " << pointCount << "\n";
}

void writeScalarHeader(std::ofstream& output, std::string_view name) {
    output << "SCALARS " << name << " float\nLOOKUP_TABLE default\n";
}

void fillComponentPlane(const float* interleaved,
                        std::size_t pointOffset,
                        std::size_t pointCount,
                        std::size_t component,
                        std::vector<float>& plane) {
    plane.resize(pointCount);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t signedPoint = 0;
         signedPoint < static_cast<std::int64_t>(pointCount); ++signedPoint) {
        const std::size_t point = static_cast<std::size_t>(signedPoint);
        plane[point] = interleaved[(pointOffset + point) * 3U + component];
    }
}

void writeLegacyVtk(const fs::path& path,
                    const LegacyStructuredPoints& reference,
                    const Accumulators& accumulators,
                    std::string_view windroseName,
                    std::ostream& output) {
    const std::size_t nx = static_cast<std::size_t>(reference.dimensions[0]);
    const std::size_t ny = static_cast<std::size_t>(reference.dimensions[1]);
    const std::size_t nz = static_cast<std::size_t>(reference.dimensions[2]);
    const std::size_t planePoints = checkedProduct(nx, ny, "output VTK plane");
    logRule(output, "Write Output VTK");
    log(output, "Writing synthesized VTK to: " + pathUtf8(path));
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Failed to create VTK file: " + pathUtf8(path));
    writeHeader(file, reference.dimensions, reference.origin, reference.spacing, windroseName);

    log(output, "  Write VECTORS u_avg");
    file << "VECTORS u_avg float\n";
    int lastPercent = -1;
    for (std::size_t z = 0; z < nz; ++z) {
        writeBigEndianFloatBlock(file, std::span<const float>(
            accumulators.u() + z * planePoints * 3U, planePoints * 3U));
        lastPercent = logProgress(output, "  u_avg write progress", z + 1, nz, lastPercent);
    }

    std::vector<float> componentPlane;
    for (std::size_t component = 0; component < 3; ++component) {
        const std::string name(1, static_cast<char>('u' + component));
        log(output, "  Write SCALARS " + name);
        writeScalarHeader(file, name);
        lastPercent = -1;
        for (std::size_t z = 0; z < nz; ++z) {
            fillComponentPlane(accumulators.u(), z * planePoints, planePoints,
                               component, componentPlane);
            writeBigEndianFloatBlock(file, componentPlane);
            lastPercent = logProgress(output, "  " + name + " write progress",
                                      z + 1, nz, lastPercent);
        }
    }
    for (const auto& [name, values] : std::array<std::pair<std::string_view, const float*>, 2>{
             std::pair<std::string_view, const float*>{"vm", accumulators.vm()},
             std::pair<std::string_view, const float*>{"tke", accumulators.tke()}}) {
        log(output, "  Write SCALARS " + std::string(name));
        writeScalarHeader(file, name);
        lastPercent = -1;
        for (std::size_t z = 0; z < nz; ++z) {
            writeBigEndianFloatBlock(file,
                std::span<const float>(values + z * planePoints, planePoints));
            lastPercent = logProgress(output, "  " + std::string(name) + " write progress",
                                      z + 1, nz, lastPercent);
        }
    }
    file.flush();
    if (!file) throw std::runtime_error("Failed while finalizing VTK file: " + pathUtf8(path));
    log(output, "Finished writing synthesized VTK: " + pathUtf8(path));
}

std::pair<int, double> resampledAxisSize(int inputSize,
                                        double sourceSpacing,
                                        double requestedSpacing) {
    if (requestedSpacing <= 0.0 || !std::isfinite(requestedSpacing)) {
        throw std::runtime_error("Output spacing must be positive, got "
                                 + pythonFloatString(requestedSpacing));
    }
    if (inputSize <= 1) return {1, requestedSpacing};
    const double extent = std::abs(sourceSpacing) * static_cast<double>(inputSize - 1);
    const double rounded = std::nearbyint(extent / requestedSpacing);
    if (rounded > static_cast<double>(std::numeric_limits<int>::max() - 1)) {
        throw std::runtime_error("Resampled VTK dimension is too large");
    }
    const int outputSize = std::max(2, static_cast<int>(rounded) + 1);
    double actual = extent / static_cast<double>(outputSize - 1);
    if (sourceSpacing < 0.0) actual = -actual;
    return {outputSize, actual};
}

AxisResample buildAxisResample(int inputSize, int outputSize) {
    AxisResample axis;
    axis.lower.resize(static_cast<std::size_t>(outputSize));
    axis.upper.resize(static_cast<std::size_t>(outputSize));
    axis.weight.resize(static_cast<std::size_t>(outputSize));
    for (int index = 0; index < outputSize; ++index) {
        double position = outputSize <= 1 ? 0.0
            : static_cast<double>(index) * static_cast<double>(inputSize - 1)
                / static_cast<double>(outputSize - 1);
        if (index == outputSize - 1) position = static_cast<double>(inputSize - 1);
        const std::size_t lower = static_cast<std::size_t>(std::floor(position));
        axis.lower[static_cast<std::size_t>(index)] = lower;
        axis.upper[static_cast<std::size_t>(index)] = std::min(
            lower + 1U, static_cast<std::size_t>(inputSize - 1));
        axis.weight[static_cast<std::size_t>(index)] = static_cast<float>(position - lower);
    }
    return axis;
}

ResamplePlan buildResamplePlan(const LegacyStructuredPoints& reference,
                               const std::array<double, 3>& requested) {
    ResamplePlan plan;
    plan.origin = reference.origin;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto [size, spacing] = resampledAxisSize(
            reference.dimensions[axis], reference.spacing[axis], requested[axis]);
        plan.dimensions[axis] = size;
        plan.spacing[axis] = spacing;
    }
    plan.x = buildAxisResample(reference.dimensions[0], plan.dimensions[0]);
    plan.y = buildAxisResample(reference.dimensions[1], plan.dimensions[1]);
    plan.z = buildAxisResample(reference.dimensions[2], plan.dimensions[2]);
    return plan;
}

float lerpFloat(float a, float b, float weight) noexcept {
    return a * (1.0F - weight) + b * weight;
}

void resampleComponentPlane(const float* source,
                            std::size_t sourceNx,
                            std::size_t sourceNy,
                            std::size_t stride,
                            std::size_t component,
                            const ResamplePlan& plan,
                            std::size_t outputZ,
                            std::vector<float>& destination) {
    const std::size_t outputNx = static_cast<std::size_t>(plan.dimensions[0]);
    const std::size_t outputNy = static_cast<std::size_t>(plan.dimensions[1]);
    destination.resize(checkedProduct(outputNx, outputNy, "resampled plane"));
    const std::size_t z0 = plan.z.lower[outputZ];
    const std::size_t z1 = plan.z.upper[outputZ];
    const float wz = plan.z.weight[outputZ];
    auto sourceValue = [&](std::size_t x, std::size_t y) {
        const std::size_t first = ((z0 * sourceNy + y) * sourceNx + x) * stride + component;
        if (z0 == z1 || wz == 0.0F) return source[first];
        const std::size_t second = ((z1 * sourceNy + y) * sourceNx + x) * stride + component;
        return lerpFloat(source[first], source[second], wz);
    };
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t signedY = 0;
         signedY < static_cast<std::int64_t>(outputNy); ++signedY) {
        const std::size_t y = static_cast<std::size_t>(signedY);
        const std::size_t y0 = plan.y.lower[y];
        const std::size_t y1 = plan.y.upper[y];
        const float wy = plan.y.weight[y];
        for (std::size_t x = 0; x < outputNx; ++x) {
            const std::size_t x0 = plan.x.lower[x];
            const std::size_t x1 = plan.x.upper[x];
            const float wx = plan.x.weight[x];
            const float lower = lerpFloat(sourceValue(x0, y0), sourceValue(x1, y0), wx);
            const float upper = lerpFloat(sourceValue(x0, y1), sourceValue(x1, y1), wx);
            destination[y * outputNx + x] = lerpFloat(lower, upper, wy);
        }
    }
}

void writeLegacyVtkResampled(const fs::path& path,
                             const LegacyStructuredPoints& reference,
                             const Accumulators& accumulators,
                             std::string_view windroseName,
                             const std::array<double, 3>& requested,
                             std::ostream& output) {
    const ResamplePlan plan = buildResamplePlan(reference, requested);
    const std::size_t nx = static_cast<std::size_t>(plan.dimensions[0]);
    const std::size_t ny = static_cast<std::size_t>(plan.dimensions[1]);
    const std::size_t nz = static_cast<std::size_t>(plan.dimensions[2]);
    const std::size_t planePoints = checkedProduct(nx, ny, "resampled plane");
    const std::size_t outputPoints = checkedProduct(planePoints, nz, "resampled VTK");
    QTemporaryDir tempDirectory(qPath(path.parent_path() / "season_average_resample_tmp_XXXXXX"));
    if (!tempDirectory.isValid()) {
        throw std::runtime_error("Failed to create resampling cache directory");
    }
    const fs::path temp = pathFromUtf8(tempDirectory.path().toUtf8().toStdString());
    std::array<std::unique_ptr<MappedFloatFile>, 3> caches;
    for (std::size_t component = 0; component < 3; ++component) {
        caches[component] = std::make_unique<MappedFloatFile>();
        caches[component]->create(temp / (std::string(1, static_cast<char>('u' + component))
                                           + "_resampled.bin"), outputPoints);
    }

    logRule(output, "Write Resampled Output VTK");
    log(output, "Requested output spacing: " + tupleText(requested));
    log(output, "Resolved output grid: dims=" + tupleText(plan.dimensions)
                    + ", origin=" + tupleText(plan.origin) + ", spacing="
                    + tupleText(plan.spacing) + ", points=" + std::to_string(outputPoints));
    log(output, "Component cache directory: " + pathUtf8(temp));
    log(output, "Writing synthesized VTK to: " + pathUtf8(path));

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Failed to create VTK file: " + pathUtf8(path));
    writeHeader(file, plan.dimensions, plan.origin, plan.spacing, windroseName);
    file << "VECTORS u_avg float\n";
    log(output, "  Write VECTORS u_avg");
    std::array<std::vector<float>, 3> componentPlanes;
    std::vector<float> vectorPlane(checkedProduct(planePoints, 3U, "resampled vector plane"));
    int lastPercent = -1;
    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t component = 0; component < 3; ++component) {
            resampleComponentPlane(accumulators.u(),
                                   static_cast<std::size_t>(reference.dimensions[0]),
                                   static_cast<std::size_t>(reference.dimensions[1]),
                                   3U, component, plan, z, componentPlanes[component]);
            std::memcpy(caches[component]->data() + z * planePoints,
                        componentPlanes[component].data(), planePoints * sizeof(float));
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t signedPoint = 0;
             signedPoint < static_cast<std::int64_t>(planePoints); ++signedPoint) {
            const std::size_t point = static_cast<std::size_t>(signedPoint);
            for (std::size_t component = 0; component < 3; ++component) {
                vectorPlane[point * 3U + component] = componentPlanes[component][point];
            }
        }
        writeBigEndianFloatBlock(file, vectorPlane);
        lastPercent = logProgress(output, "  u_avg write progress", z + 1, nz, lastPercent);
    }
    for (auto& cache : caches) cache->flush();
    for (std::size_t component = 0; component < 3; ++component) {
        const std::string name(1, static_cast<char>('u' + component));
        log(output, "  Write SCALARS " + name);
        writeScalarHeader(file, name);
        lastPercent = -1;
        for (std::size_t z = 0; z < nz; ++z) {
            writeBigEndianFloatBlock(file, std::span<const float>(
                caches[component]->data() + z * planePoints, planePoints));
            lastPercent = logProgress(output, "  " + name + " write progress",
                                      z + 1, nz, lastPercent);
        }
    }
    for (const auto& [name, source] : std::array<std::pair<std::string_view, const float*>, 2>{
             std::pair<std::string_view, const float*>{"vm", accumulators.vm()},
             std::pair<std::string_view, const float*>{"tke", accumulators.tke()}}) {
        log(output, "  Write SCALARS " + std::string(name));
        writeScalarHeader(file, name);
        lastPercent = -1;
        for (std::size_t z = 0; z < nz; ++z) {
            std::vector<float> plane;
            resampleComponentPlane(source,
                                   static_cast<std::size_t>(reference.dimensions[0]),
                                   static_cast<std::size_t>(reference.dimensions[1]),
                                   1U, 0U, plan, z, plane);
            writeBigEndianFloatBlock(file, plane);
            lastPercent = logProgress(output, "  " + std::string(name) + " write progress",
                                      z + 1, nz, lastPercent);
        }
    }
    file.flush();
    if (!file) throw std::runtime_error("Failed while finalizing VTK file: " + pathUtf8(path));
    log(output, "Finished writing synthesized VTK: " + pathUtf8(path));
}

void writeSummary(const ProjectLayout& layout,
                  double referenceSpeed,
                  const std::string& referenceMode,
                  double windroseHeight,
                  const Windrose& windrose,
                  const std::vector<DirectionWeight>& weights,
                  const VtkDirectoryChoice& choice,
                  const LegacyStructuredPoints& input,
                  const LegacyStructuredPoints& outputVtk,
                  std::ostream& output) {
    // Python's Path.write_text uses the platform text newline convention.
    // Keep text mode so Windows emits CRLF while POSIX retains LF.
    std::ofstream summary(layout.summaryPath, std::ios::trunc);
    if (!summary) {
        throw std::runtime_error("Failed to create summary: " + pathUtf8(layout.summaryPath));
    }
    summary << "Season Average Summary\n"
            << "======================\n"
            << "Config path: " << pathUtf8(layout.configPath) << "\n"
            << "Project dir: " << pathUtf8(layout.projectDirectory) << "\n"
            << "Windrose file: " << pathUtf8(layout.windrosePath) << "\n"
            << "Profile file: " << pathUtf8(layout.profilePath) << "\n"
            << "Selected VTK dir: " << pathUtf8(choice.directory) << "\n"
            << "Output VTK: " << pathUtf8(layout.outputVtkPath) << "\n"
            << "Figure dir: " << pathUtf8(layout.figureDirectory) << "\n\n"
            << "Windrose reference height: " << formatNumber(windroseHeight) << " m\n"
            << "Reference speed from profile: " << formatNumber(referenceSpeed) << " m/s ("
            << referenceMode << ")\n"
            << "Normalized total probability used: " << std::fixed << std::setprecision(9)
            << windrose.totalProbability << "\n"
            << "Input grid signature: dims=" << tupleText(input.dimensions)
            << ", origin=" << tupleText(input.origin) << ", spacing=" << tupleText(input.spacing)
            << "\n"
            << "Output grid signature: dims=" << tupleText(outputVtk.dimensions)
            << ", origin=" << tupleText(outputVtk.origin) << ", spacing="
            << tupleText(outputVtk.spacing) << "\n\n"
            << "Speed bins:\n";
    for (const SpeedBin& bin : windrose.bins) {
        summary << "  " << bin.header << ": lower=" << formatNumber(bin.lower)
                << " m/s, upper=" << (bin.upper ? formatNumber(*bin.upper) : "inf")
                << " m/s, target=" << formatNumber(bin.targetSpeed) << " m/s\n";
    }
    summary << "\nDirection weights:\n";
    for (const DirectionWeight& weight : weights) {
        const auto file = choice.angleToFile.find(weight.angle);
        const std::string vtkName = file == choice.angleToFile.end()
            ? "-" : pathUtf8(file->second.filename());
        summary << "  " << std::setw(5) << formatNumber(weight.angle) << " deg / "
                << std::setw(3) << weight.label << ": prob=" << std::fixed << std::setprecision(9)
                << weight.probability << ", vel_weight=" << weight.velocityWeight
                << ", tke_weight=" << weight.tkeWeight << ", vtk=" << vtkName << "\n";
    }
    summary.flush();
    if (!summary) {
        throw std::runtime_error("Failed while writing summary: " + pathUtf8(layout.summaryPath));
    }
    log(output, "Saved summary: " + pathUtf8(layout.summaryPath));
}

int execute(const Arguments& arguments, std::ostream& output) {
    const auto started = std::chrono::steady_clock::now();
    const ProjectLayout layout = loadProjectLayout(arguments.configPath);
    logRule(output, "Season Average Synthesis");
    log(output, "Config path: " + pathUtf8(layout.configPath));
    log(output, "Project dir: " + pathUtf8(layout.projectDirectory));
    log(output, "wind_bc dir: " + pathUtf8(layout.windBcDirectory));
    log(output, "profile.dat: " + pathUtf8(layout.profilePath));
    log(output, "windrose CSV: " + pathUtf8(layout.windrosePath));
    log(output, "Output dir: " + pathUtf8(layout.outputDirectory));
    log(output, "Figures dir: " + pathUtf8(layout.figureDirectory));
    log(output, "Figure DPI: " + std::to_string(layout.figureDpi));
    if (arguments.vtkDirectory) {
        log(output, "Explicit VTK dir: " + pathUtf8(*arguments.vtkDirectory));
    }
    if (arguments.outputSpacing) {
        log(output, "Requested output spacing: " + tupleText(*arguments.outputSpacing));
    }
    if (arguments.skipFigures) log(output, "Figure generation: skipped by --skip-figures");
    if (layout.zLimitSi) {
        log(output, "Profile z_limit candidate from config: " + formatNumber(*layout.zLimitSi) + " m");
    } else {
        log(output, "Profile z_limit candidate from config: not available");
    }

    const Profile profile = parseProfile(layout.profilePath, layout.zLimitSi, output);
    const auto [minimumU, maximumU] = std::minmax_element(profile.u.begin(), profile.u.end());
    log(output, "profile.dat parsed: " + std::to_string(profile.z.size())
                    + " unique samples, z range " + formatNumber(profile.z.front()) + " -> "
                    + formatNumber(profile.z.back()) + " m, U range " + formatNumber(*minimumU)
                    + " -> " + formatNumber(*maximumU) + " m/s");
    const double windroseHeight = parseWindroseHeight(layout.windrosePath);
    const auto [referenceSpeed, referenceMode] = interpolateProfile(profile, windroseHeight);
    log(output, "Windrose reference height parsed from file name: "
                    + formatNumber(windroseHeight) + " m; profile reference speed = "
                    + formatNumber(referenceSpeed) + " m/s (" + referenceMode + ")");

    const Windrose windrose = parseWindrose(layout.windrosePath, output);
    logRule(output, "Speed Bins");
    for (const SpeedBin& bin : windrose.bins) {
        log(output, bin.header + ": lower=" + formatNumber(bin.lower) + " m/s, upper="
                        + (bin.upper ? formatNumber(*bin.upper) : "inf") + " m/s, target="
                        + formatNumber(bin.targetSpeed) + " m/s");
    }
    const std::vector<DirectionWeight> weights = computeWeights(
        windrose, referenceSpeed, output);
    const VtkDirectoryChoice choice = chooseVtkSource(
        layout.projectDirectory, weights, arguments.vtkDirectory, output);
    HeaderSelection headers = validateHeaders(choice, weights, output);
    std::unique_ptr<Accumulators> accumulators = synthesize(
        choice, weights, headers, layout.outputDirectory, output);

    if (arguments.outputSpacing) {
        writeLegacyVtkResampled(layout.outputVtkPath, headers.reference, *accumulators,
                                pathUtf8(layout.windrosePath.filename()),
                                *arguments.outputSpacing, output);
    } else {
        writeLegacyVtk(layout.outputVtkPath, headers.reference, *accumulators,
                       pathUtf8(layout.windrosePath.filename()), output);
    }
    const LegacyStructuredPoints outputVtk = LegacyStructuredPoints::readHeader(layout.outputVtkPath);
    writeSummary(layout, referenceSpeed, referenceMode, windroseHeight, windrose, weights,
                 choice, headers.reference, outputVtk, output);
    if (!arguments.skipFigures) {
        luw::post::detail::renderSeasonAverageFigures(
            layout.outputVtkPath, layout.figureDirectory,
            std::max(80, layout.figureDpi), output);
    }
    accumulators.reset();

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    logRule(output, "Finished");
    log(output, "Synthesized VTK: " + pathUtf8(layout.outputVtkPath));
    log(output, "Summary file: " + pathUtf8(layout.summaryPath));
    log(output, "Figures directory: " + pathUtf8(layout.figureDirectory));
    std::ostringstream elapsedText;
    elapsedText << std::fixed << std::setprecision(2) << elapsed;
    log(output, "Elapsed time: " + elapsedText.str() + " s");
    return 0;
}

} // namespace

int run_season_average(const std::vector<std::string>& arguments,
                       std::ostream& output,
                       std::ostream& error) {
    try {
        const Arguments parsed = parseArguments(arguments);
        if (parsed.help) {
            printUsage(output);
            return 0;
        }
        return execute(parsed, output);
    } catch (const ArgumentError& exception) {
        error << "usage: season_average [-h] [--vtk-dir VTK_DIR] "
                 "[--output-spacing M [M ...]] [--skip-figures] config_path\n"
              << "season_average: error: " << exception.what() << '\n';
        return 2;
    }
}

} // namespace luw::cli
