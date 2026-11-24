#include "luw/vtk_post_commands.hpp"

#include "luw/deck_document.hpp"
#include "luw/legacy_vtk.hpp"
#include "post_internal.hpp"
#include "tke_viz_internal.hpp"

#include <QHash>
#include <QString>
#include <QVector>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace luw::cli {

namespace {

namespace fs = std::filesystem;
using luw::post::LegacyStructuredPoints;
using luw::post::detail::Projection;
using luw::post::detail::pathUtf8;
namespace tke = luw::post::tke;

class CommandExit final : public std::runtime_error {
public:
    explicit CommandExit(std::string message, int code = 1)
        : std::runtime_error(std::move(message)), code_(code) {}
    [[nodiscard]] int code() const noexcept { return code_; }
private:
    int code_ = 1;
};

[[nodiscard]] fs::path resolved(const fs::path& path) {
    std::error_code error;
    fs::path result = fs::absolute(path, error);
    if (error) result = path;
    return result.lexically_normal();
}

[[nodiscard]] std::string number(double value, int precision = 6) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

[[nodiscard]] std::string pythonFloat(double value) {
    std::array<char, 64> buffer{};
    const auto [end, status] = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                             value, std::chars_format::general);
    std::string result;
    if (status == std::errc{}) result.assign(buffer.data(), end);
    else {
        std::ostringstream fallback;
        fallback << std::setprecision(17) << value;
        result = fallback.str();
    }
    if (std::isfinite(value) && result.find_first_of(".eE") == std::string::npos) result += ".0";
    return result;
}

[[nodiscard]] double parseDouble(std::string_view text, std::string_view option) {
    std::string owned(text);
    std::size_t consumed = 0;
    try {
        const double value = std::stod(owned, &consumed);
        if (consumed == owned.size()) return value;
    } catch (const std::exception&) {
    }
    throw CommandExit("argument " + std::string(option) + ": invalid float value: '" + owned + "'", 2);
}

[[nodiscard]] int parseInt(std::string_view text, std::string_view option) {
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw CommandExit("argument " + std::string(option) + ": invalid int value: '" + std::string(text) + "'", 2);
    }
    return value;
}

[[nodiscard]] fs::path pathFromUtf8(const std::string& value) {
#ifdef _WIN32
    return fs::path(QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size())).toStdWString());
#else
    return fs::path(value);
#endif
}

struct TkeArguments final {
    std::optional<fs::path> vtkPath;
    bool skipConfig = false;
    bool localXy = false;
    std::optional<fs::path> dataDirectory;
    std::optional<fs::path> outputDirectory;
    std::optional<std::string> targetCrs;
    std::optional<double> minLon;
    std::optional<double> maxLon;
    std::optional<double> minLat;
    std::optional<double> maxLat;
    std::optional<double> domainLonMin;
    std::optional<double> domainLonMax;
    std::optional<double> domainLatMin;
    std::optional<double> domainLatMax;
    std::optional<std::string> utmCrs;
    std::optional<double> rotateDegrees;
    std::optional<double> cellSize;
    std::optional<double> minX;
    std::optional<double> maxX;
    std::optional<double> minY;
    std::optional<double> maxY;
    int quiverStep = 50;
    int dpi = 220;
    bool exportCroppedVtk = false;
    std::optional<fs::path> croppedVtkPath;
    bool show = false;
    bool help = false;
};

void tkeUsage(std::ostream& output, std::string_view program) {
    output << "usage: " << program << " [-h] [--skip-config] [--local-xy] [--data-dir DATA_DIR]\n"
           << "                    [--output-dir OUTPUT_DIR] [--target-crs TARGET_CRS]\n"
           << "                    [--min-lon MIN_LON] [--max-lon MAX_LON]\n"
           << "                    [--min-lat MIN_LAT] [--max-lat MAX_LAT]\n"
           << "                    [--domain-lon-min DOMAIN_LON_MIN]\n"
           << "                    [--domain-lon-max DOMAIN_LON_MAX]\n"
           << "                    [--domain-lat-min DOMAIN_LAT_MIN]\n"
           << "                    [--domain-lat-max DOMAIN_LAT_MAX] [--utm-crs UTM_CRS]\n"
           << "                    [--rotate-deg ROTATE_DEG] [--cell-size CELL_SIZE]\n"
           << "                    [--min-x MIN_X] [--max-x MAX_X] [--min-y MIN_Y] [--max-y MAX_Y]\n"
           << "                    [--quiver-step QUIVER_STEP] [--dpi DPI] [--export-cropped-vtk]\n"
           << "                    [--cropped-vtk-path CROPPED_VTK_PATH] [--show]\n"
           << "                    [vtk_file]\n\n"
           << "Visualize 9 layers of wind arrows and TKE from one VTK\n";
}

TkeArguments parseTkeArguments(const std::vector<std::string>& arguments) {
    TkeArguments result;
    auto required = [&](std::size_t& index, std::string_view option) -> const std::string& {
        if (index + 1 >= arguments.size()) {
            throw CommandExit("argument " + std::string(option) + ": expected one argument", 2);
        }
        return arguments[++index];
    };
    auto assignment = [](const std::string& argument, std::string_view option) -> std::optional<std::string_view> {
        if (argument.size() > option.size() && argument.compare(0, option.size(), option) == 0
            && argument[option.size()] == '=') return std::string_view(argument).substr(option.size() + 1);
        return std::nullopt;
    };
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        auto value = [&](std::string_view option) -> std::optional<std::string_view> {
            if (argument == option) return required(index, option);
            return assignment(argument, option);
        };
        if (argument == "-h" || argument == "--help") result.help = true;
        else if (argument == "--skip-config") result.skipConfig = true;
        else if (argument == "--local-xy") result.localXy = true;
        else if (argument == "--export-cropped-vtk") result.exportCroppedVtk = true;
        else if (argument == "--show") result.show = true;
        else if (const auto dataDirectoryValue = value("--data-dir")) result.dataDirectory = pathFromUtf8(std::string(*dataDirectoryValue));
        else if (const auto outputDirectoryValue = value("--output-dir")) result.outputDirectory = pathFromUtf8(std::string(*outputDirectoryValue));
        else if (const auto targetCrsValue = value("--target-crs")) result.targetCrs = std::string(*targetCrsValue);
        else if (const auto minLonValue = value("--min-lon")) result.minLon = parseDouble(*minLonValue, "--min-lon");
        else if (const auto maxLonValue = value("--max-lon")) result.maxLon = parseDouble(*maxLonValue, "--max-lon");
        else if (const auto minLatValue = value("--min-lat")) result.minLat = parseDouble(*minLatValue, "--min-lat");
        else if (const auto maxLatValue = value("--max-lat")) result.maxLat = parseDouble(*maxLatValue, "--max-lat");
        else if (const auto domainLonMinValue = value("--domain-lon-min")) result.domainLonMin = parseDouble(*domainLonMinValue, "--domain-lon-min");
        else if (const auto domainLonMaxValue = value("--domain-lon-max")) result.domainLonMax = parseDouble(*domainLonMaxValue, "--domain-lon-max");
        else if (const auto domainLatMinValue = value("--domain-lat-min")) result.domainLatMin = parseDouble(*domainLatMinValue, "--domain-lat-min");
        else if (const auto domainLatMaxValue = value("--domain-lat-max")) result.domainLatMax = parseDouble(*domainLatMaxValue, "--domain-lat-max");
        else if (const auto utmCrsValue = value("--utm-crs")) result.utmCrs = std::string(*utmCrsValue);
        else if (const auto rotateDegreesValue = value("--rotate-deg")) result.rotateDegrees = parseDouble(*rotateDegreesValue, "--rotate-deg");
        else if (const auto cellSizeValue = value("--cell-size")) result.cellSize = parseDouble(*cellSizeValue, "--cell-size");
        else if (const auto minXValue = value("--min-x")) result.minX = parseDouble(*minXValue, "--min-x");
        else if (const auto maxXValue = value("--max-x")) result.maxX = parseDouble(*maxXValue, "--max-x");
        else if (const auto minYValue = value("--min-y")) result.minY = parseDouble(*minYValue, "--min-y");
        else if (const auto maxYValue = value("--max-y")) result.maxY = parseDouble(*maxYValue, "--max-y");
        else if (const auto quiverStepValue = value("--quiver-step")) result.quiverStep = parseInt(*quiverStepValue, "--quiver-step");
        else if (const auto dpiValue = value("--dpi")) result.dpi = parseInt(*dpiValue, "--dpi");
        else if (const auto croppedVtkPathValue = value("--cropped-vtk-path")) result.croppedVtkPath = pathFromUtf8(std::string(*croppedVtkPathValue));
        else if (!argument.empty() && argument.front() == '-') throw CommandExit("unrecognized arguments: " + argument, 2);
        else if (result.vtkPath) throw CommandExit("unrecognized arguments: " + argument, 2);
        else result.vtkPath = pathFromUtf8(argument);
    }
    if (result.domainLonMin.has_value() != result.domainLonMax.has_value()) {
        throw CommandExit("--domain-lon-min and --domain-lon-max must be provided together", 2);
    }
    if (result.domainLatMin.has_value() != result.domainLatMax.has_value()) {
        throw CommandExit("--domain-lat-min and --domain-lat-max must be provided together", 2);
    }
    return result;
}

[[nodiscard]] std::string trim(std::string value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

[[nodiscard]] std::string stripPythonComment(std::string value) {
    bool single = false;
    bool doubleQuote = false;
    bool escaped = false;
    for (std::size_t index = 0; index < value.size(); ++index) {
        const char character = value[index];
        if (escaped) { escaped = false; continue; }
        if (character == '\\') { escaped = true; continue; }
        if (character == '\'' && !doubleQuote) single = !single;
        else if (character == '"' && !single) doubleQuote = !doubleQuote;
        else if (character == '#' && !single && !doubleQuote) return trim(value.substr(0, index));
    }
    return trim(value);
}

[[nodiscard]] std::optional<std::string> pythonScalar(std::string_view text, std::string_view key) {
    const std::regex pattern("(?:^|\\n)\\s*" + std::string(key) + R"(\s*=\s*([^\r\n]+))");
    const std::string owned(text);
    std::smatch match;
    if (!std::regex_search(owned, match, pattern)) return std::nullopt;
    std::string value = stripPythonComment(match[1].str());
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"')
        || (value.front() == '\'' && value.back() == '\''))) value = trim(value.substr(1, value.size() - 2));
    return value.empty() ? std::nullopt : std::optional<std::string>(value);
}

[[nodiscard]] std::optional<double> optionalPythonFloat(std::string_view text, std::string_view key) {
    const auto raw = pythonScalar(text, key);
    if (!raw) return std::nullopt;
    try {
        std::size_t consumed = 0;
        const double value = std::stod(*raw, &consumed);
        if (consumed == raw->size()) return value;
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::pair<double, double>> pythonPair(std::string_view text,
                                                                  std::string_view key) {
    const auto raw = pythonScalar(text, key);
    if (!raw || raw->size() < 2 || raw->front() != '[' || raw->back() != ']') return std::nullopt;
    const std::size_t comma = raw->find(',');
    if (comma == std::string::npos) return std::nullopt;
    try {
        const double first = std::stod(trim(raw->substr(1, comma - 1)));
        const double second = std::stod(trim(raw->substr(comma + 1, raw->size() - comma - 2)));
        return std::pair{first, second};
    } catch (const std::exception&) { return std::nullopt; }
}

struct PythonConfig final {
    fs::path dataDirectory = ".";
    fs::path outputDirectory = "./tke_viz_output";
    std::optional<std::string> targetCrs;
    std::optional<double> minLon;
    std::optional<double> maxLon;
    std::optional<double> minLat;
    std::optional<double> maxLat;
    std::optional<double> cellSize;
    std::optional<std::pair<double, double>> domainLon;
    std::optional<std::pair<double, double>> domainLat;
    std::optional<std::string> utmCrs;
    std::optional<double> rotateDegrees;
};

[[nodiscard]] std::optional<PythonConfig> loadPythonConfig(std::string& sourceDetail) {
    const fs::path path = fs::current_path() / "config.py";
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        sourceDetail = "No module named 'config'";
        return std::nullopt;
    }
    const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    PythonConfig config;
    if (const auto value = pythonScalar(text, "DATA_DIR")) config.dataDirectory = pathFromUtf8(*value);
    if (const auto value = pythonScalar(text, "OUTPUT_DIR")) config.outputDirectory = pathFromUtf8(*value);
    config.targetCrs = pythonScalar(text, "TARGET_CRS");
    config.minLon = optionalPythonFloat(text, "min_lon");
    config.maxLon = optionalPythonFloat(text, "max_lon");
    config.minLat = optionalPythonFloat(text, "min_lat");
    config.maxLat = optionalPythonFloat(text, "max_lat");
    config.cellSize = optionalPythonFloat(text, "cell_size");
    // A compact, literal WIND_VTK_CONFIG is the deployed form.  Restrict the
    // search to its braces so unrelated similarly named variables do not win.
    const std::size_t dictName = text.find("WIND_VTK_CONFIG");
    if (dictName != std::string::npos) {
        const std::size_t open = text.find('{', dictName);
        const std::size_t close = open == std::string::npos ? std::string::npos : text.find('}', open + 1);
        if (open != std::string::npos && close != std::string::npos) {
            const std::string dict = text.substr(open + 1, close - open - 1);
            auto dictScalar = [&](std::string_view key) -> std::optional<std::string> {
                const std::regex expression("['\"]" + std::string(key) + R"(['\"]\s*:\s*([^,\r\n}]+))");
                std::smatch match;
                if (!std::regex_search(dict, match, expression)) return std::nullopt;
                std::string value = trim(match[1].str());
                if (value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'')
                    || (value.front() == '"' && value.back() == '"'))) value = value.substr(1, value.size() - 2);
                return value;
            };
            auto dictPair = [&](std::string_view key) -> std::optional<std::pair<double,double>> {
                const std::regex expression("['\"]" + std::string(key) + R"(['\"]\s*:\s*\[([^\]]+)\])");
                std::smatch match;
                if (!std::regex_search(dict, match, expression)) return std::nullopt;
                const std::string inside = match[1].str();
                const std::size_t comma = inside.find(',');
                if (comma == std::string::npos) return std::nullopt;
                try { return std::pair{std::stod(trim(inside.substr(0, comma))),
                                      std::stod(trim(inside.substr(comma + 1)))}; }
                catch (const std::exception&) { return std::nullopt; }
            };
            config.domainLon = dictPair("cut_lon_manual");
            if (!config.domainLon) config.domainLon = dictPair("domain_lon");
            config.domainLat = dictPair("cut_lat_manual");
            if (!config.domainLat) config.domainLat = dictPair("domain_lat");
            config.utmCrs = dictScalar("utm_crs");
            if (const auto value = dictScalar("rotate_deg")) {
                try { config.rotateDegrees = std::stod(*value); } catch (const std::exception&) {}
            }
        }
    }
    sourceDetail.clear();
    return config;
}

tke::RuntimeConfig buildTkeConfig(const TkeArguments& args,
                                  const std::optional<PythonConfig>& external) {
    tke::RuntimeConfig result;
    result.dataDirectory = args.dataDirectory.value_or(external ? external->dataDirectory : fs::path("."));
    result.outputDirectory = args.outputDirectory.value_or(
        external ? external->outputDirectory : fs::path("./tke_viz_output"));
    const auto target = args.targetCrs ? args.targetCrs : (external ? external->targetCrs : std::nullopt);
    const auto minLon = args.minLon ? args.minLon : (external ? external->minLon : std::nullopt);
    const auto maxLon = args.maxLon ? args.maxLon : (external ? external->maxLon : std::nullopt);
    const auto minLat = args.minLat ? args.minLat : (external ? external->minLat : std::nullopt);
    const auto maxLat = args.maxLat ? args.maxLat : (external ? external->maxLat : std::nullopt);
    if (!args.localXy) {
        std::vector<std::string> missing;
        if (!target) missing.push_back("--target-crs");
        if (!minLon) missing.push_back("--min-lon");
        if (!maxLon) missing.push_back("--max-lon");
        if (!minLat) missing.push_back("--min-lat");
        if (!maxLat) missing.push_back("--max-lat");
        if (!missing.empty()) {
            std::ostringstream message;
            message << "Geo mode requires geospatial config. Missing: ";
            for (std::size_t index = 0; index < missing.size(); ++index) {
                if (index != 0) message << ", ";
                message << missing[index];
            }
            message << ". Provide these args or use --local-xy.";
            throw CommandExit(message.str(), 2);
        }
    }
    result.targetCrs = target.value_or("LOCAL_XY");
    result.minLon = minLon.value_or(0.0);
    result.maxLon = maxLon.value_or(0.0);
    result.minLat = minLat.value_or(0.0);
    result.maxLat = maxLat.value_or(0.0);
    if (result.minLon > result.maxLon) std::swap(result.minLon, result.maxLon);
    if (result.minLat > result.maxLat) std::swap(result.minLat, result.maxLat);
    result.cellSize = args.cellSize.value_or(external && external->cellSize ? *external->cellSize : 10.0);
    if (!std::isfinite(result.cellSize) || result.cellSize <= 0.0) result.cellSize = 10.0;
    result.domainLon = external && external->domainLon
        ? *external->domainLon : std::pair{result.minLon, result.maxLon};
    result.domainLat = external && external->domainLat
        ? *external->domainLat : std::pair{result.minLat, result.maxLat};
    if (args.domainLonMin) result.domainLon = {*args.domainLonMin, *args.domainLonMax};
    if (args.domainLatMin) result.domainLat = {*args.domainLatMin, *args.domainLatMax};
    if (result.domainLon.first > result.domainLon.second) std::swap(result.domainLon.first, result.domainLon.second);
    if (result.domainLat.first > result.domainLat.second) std::swap(result.domainLat.first, result.domainLat.second);
    result.utmCrs = args.utmCrs.value_or(external && external->utmCrs
        ? *external->utmCrs : result.targetCrs);
    result.rotateDegrees = args.rotateDegrees ? args.rotateDegrees
        : (external ? external->rotateDegrees : std::nullopt);
    return result;
}

[[nodiscard]] fs::path resolveTkeInput(const TkeArguments& args, const fs::path& dataDirectory) {
    if (args.vtkPath) {
        std::error_code error;
        if (args.vtkPath->is_absolute() && fs::exists(*args.vtkPath, error) && !error) return *args.vtkPath;
        error.clear();
        if (fs::exists(*args.vtkPath, error) && !error) return resolved(*args.vtkPath);
        const fs::path candidate = dataDirectory / *args.vtkPath;
        error.clear();
        if (fs::exists(candidate, error) && !error) return resolved(candidate);
        throw std::runtime_error("VTK file not found: " + pathUtf8(*args.vtkPath));
    }
    std::vector<fs::path> candidates;
    std::error_code iteratorError;
    for (fs::directory_iterator iterator(dataDirectory, iteratorError), end;
         !iteratorError && iterator != end; iterator.increment(iteratorError)) {
        std::error_code typeError;
        if (!iterator->is_regular_file(typeError) || typeError) continue;
        std::string name = pathUtf8(iterator->path().filename());
        std::string lowerName = name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (lowerName.ends_with(".vtk")) candidates.push_back(iterator->path());
    }
    if (iteratorError) throw std::runtime_error("Cannot enumerate VTK directory: " + iteratorError.message());
    if (candidates.empty()) throw std::runtime_error("No VTK found under: " + pathUtf8(dataDirectory));
    return resolved(*std::max_element(candidates.begin(), candidates.end(), [](const fs::path& left, const fs::path& right) {
        std::error_code leftError, rightError;
        const auto leftTime = fs::last_write_time(left, leftError);
        const auto rightTime = fs::last_write_time(right, rightError);
        if (leftError || rightError) return pathUtf8(left) < pathUtf8(right);
        return leftTime < rightTime;
    }));
}

int executeTke(const TkeArguments& args, std::ostream& output) {
    std::string configDetail;
    std::optional<PythonConfig> external;
    if (args.skipConfig) configDetail = "--skip-config";
    else external = loadPythonConfig(configDetail);
    const tke::RuntimeConfig config = buildTkeConfig(args, external);
    if (external) output << "Config source: config.py + CLI overrides\n";
    else output << "Config source: CLI only (" << configDetail << ")\n";

    const fs::path vtkPath = resolveTkeInput(args, config.dataDirectory);
    fs::create_directories(config.outputDirectory);
    const LegacyStructuredPoints vtk = LegacyStructuredPoints::readHeader(vtkPath);
    const tke::Bounds bounds = args.localXy
        ? tke::localBounds(vtk, args.minX, args.maxX, args.minY, args.maxY)
        : tke::geographicBounds(config, args.minX, args.maxX, args.minY, args.maxY);
    const tke::Crop crop = args.localXy ? tke::localCrop(vtk, bounds)
        : tke::geographicCrop(vtk, config, bounds);
    const tke::TargetGrid grid = args.localXy ? tke::localTargetGrid(vtk, config, bounds)
        : tke::geographicTargetGrid(vtk, config, bounds, crop);
    const std::string targetLabel = args.localXy ? "LOCAL_XY" : config.targetCrs;

    output << "VTK file: " << pathUtf8(vtkPath) << '\n'
           << "Dimensions: (" << vtk.dimensions[0] << ", " << vtk.dimensions[1] << ", " << vtk.dimensions[2] << ")\n"
           << "Origin: (" << pythonFloat(vtk.origin[0]) << ", " << pythonFloat(vtk.origin[1])
           << ", " << pythonFloat(vtk.origin[2]) << ")\n"
           << "Spacing: (" << pythonFloat(vtk.spacing[0]) << ", " << pythonFloat(vtk.spacing[1])
           << ", " << pythonFloat(vtk.spacing[2]) << ")\n"
           << "Fields: [";
    for (std::size_t index = 0; index < vtk.fields.size(); ++index) {
        if (index != 0) output << ", ";
        output << '\'' << vtk.fields[index].name << '\'';
    }
    output << "]\nOutput dir: " << pathUtf8(config.outputDirectory) << '\n'
           << "Default XY bounds (" << targetLabel << "):\n"
           << "  X=[" << number(bounds.defaultMinX, 2) << ", " << number(bounds.defaultMaxX, 2)
           << "], Y=[" << number(bounds.defaultMinY, 2) << ", " << number(bounds.defaultMaxY, 2) << "]\n"
           << "Using XY bounds (" << targetLabel << "):\n"
           << "  X=[" << number(bounds.minX, 2) << ", " << number(bounds.maxX, 2)
           << "], Y=[" << number(bounds.minY, 2) << ", " << number(bounds.maxY, 2) << "]\n"
           << "Mapped VTK crop:\n"
           << "  X_idx=[" << crop.xStart << ", " << crop.xEnd << "), Y_idx=[" << crop.yStart
           << ", " << crop.yEnd << "), size=" << crop.nx() << 'x' << crop.ny() << "\nTransform chain:\n";
    if (args.localXy) output << "  local XY (standalone mode, no lon/lat conversion)\n";
    else {
        output << "  local XY -> lon/lat -> " << targetLabel << ", utm_crs=" << crop.utmCrs
               << ", rotate_deg=" << number(crop.rotateDegrees, 6) << "\n"
               << "Domain lon/lat for transform: lon=[" << number(crop.domainLon.first, 6) << ", "
               << number(crop.domainLon.second, 6) << "], lat=[" << number(crop.domainLat.first, 6)
               << ", " << number(crop.domainLat.second, 6) << "]\n";
    }
    output << "Crop mask coverage inside bounds: " << crop.insideCount << '/'
           << static_cast<std::size_t>(crop.nx()) * static_cast<std::size_t>(crop.ny())
           << " (" << number(crop.insideRatio * 100.0, 2) << "%)\n";
    const std::size_t validCount = static_cast<std::size_t>(std::count(grid.valid.begin(), grid.valid.end(), 1));
    output << "Regular target grid: " << grid.nx << 'x' << grid.ny << ", step=" << number(grid.step, 2)
           << "m, valid=" << number(static_cast<double>(validCount) / grid.valid.size() * 100.0, 2) << "%\n";
    const auto plan = tke::heightPlan(vtk);
    output << "\nHeight mapping (fixed): height=-50+10*z\n";
    for (const auto& item : plan) {
        output << "  " << std::setw(4) << item.targetHeight << "m -> z=" << std::setw(3) << item.zIndex
               << " (" << (item.valid ? "OK" : "OUT_OF_RANGE") << ")\n";
    }
    const std::string stem = pathUtf8(vtkPath.stem());
    const fs::path windOutput = config.outputDirectory / (stem + "_wind_arrows_9layers.png");
    const fs::path tkeOutput = config.outputDirectory / (stem + "_tke_9layers.png");
    const fs::path rangeOutput = config.outputDirectory / (stem + "_range_diagnostic.png");
    const fs::path croppedOutput = args.croppedVtkPath.value_or(
        config.outputDirectory / (stem + "_cropped.vtk"));
    if (args.exportCroppedVtk) {
        tke::exportCroppedVtk(vtk, crop, tke::computeZCrop(vtk, std::nullopt), croppedOutput, output);
    }
    tke::renderRangeDiagnostic(crop, rangeOutput, targetLabel, args.dpi);
    output << "Saved range diagnostic figure: " << pathUtf8(rangeOutput) << '\n';
    tke::renderWindFigure(vtk, tke::selectWindFields(vtk, true), plan, grid,
                          args.quiverStep, windOutput, args.dpi);
    output << "Saved wind figure: " << pathUtf8(windOutput) << '\n';
    tke::renderTkeFigure(vtk, tke::selectTkeField(vtk, true), plan, grid, tkeOutput, args.dpi);
    output << "Saved TKE figure: " << pathUtf8(tkeOutput) << '\n';
    (void)args.show; // Native rendering is deliberately headless.
    return 0;
}

[[nodiscard]] std::optional<double> deckFloat(const luw::DeckDocument& deck,
                                              std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (deck.has(QString::fromLatin1(key))) return deck.floating(QString::fromLatin1(key));
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> deckText(const luw::DeckDocument& deck,
                                                  std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (!deck.has(QString::fromLatin1(key))) continue;
        const auto value = deck.text(QString::fromLatin1(key));
        if (value) return value->toStdString();
        return std::nullopt;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::pair<double,double>> deckPair(
    const luw::DeckDocument& deck,
    std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (!deck.has(QString::fromLatin1(key))) continue;
        return deck.pair(QString::fromLatin1(key));
    }
    return std::nullopt;
}

[[nodiscard]] std::string inferUtm(double longitude, double latitude) {
    const int zone = std::clamp(static_cast<int>(std::floor((longitude + 180.0) / 6.0)) + 1, 1, 60);
    return "EPSG:" + std::to_string((latitude >= 0.0 ? 32600 : 32700) + zone);
}

struct BatchArguments final {
    fs::path configPath;
    std::optional<double> cropZMax;
};

BatchArguments parseBatchArguments(const std::vector<std::string>& arguments) {
    BatchArguments result;
    bool haveConfig = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "--crop-z-max") {
            if (index + 1 >= arguments.size()) throw CommandExit("argument --crop-z-max: expected one argument", 2);
            result.cropZMax = parseDouble(arguments[++index], "--crop-z-max");
        } else if (argument.starts_with("--crop-z-max=")) {
            result.cropZMax = parseDouble(argument.substr(13), "--crop-z-max");
        } else if (!argument.empty() && argument.front() == '-') {
            throw CommandExit("unrecognized arguments: " + argument, 2);
        } else if (haveConfig) {
            throw CommandExit("unrecognized arguments: " + argument, 2);
        } else {
            result.configPath = pathFromUtf8(argument);
            haveConfig = true;
        }
    }
    if (!haveConfig) throw CommandExit("the following arguments are required: config_path", 2);
    return result;
}

tke::RuntimeConfig loadBatchConfig(const fs::path& inputPath) {
    const fs::path configPath = resolved(inputPath);
    std::error_code fileError;
    if (!fs::is_regular_file(configPath, fileError) || fileError) {
        throw std::runtime_error("Config file not found: " + pathUtf8(configPath));
    }
    const luw::DeckDocument deck = luw::DeckDocument::load(
        QString::fromUtf8(pathUtf8(configPath).c_str()));
    const auto domainLon = deckPair(deck, {"cut_lon_manual", "domain_lon"});
    const auto domainLat = deckPair(deck, {"cut_lat_manual", "domain_lat"});
    const auto cropMinLon = deckFloat(deck, {"crop_min_lon"});
    const auto cropMaxLon = deckFloat(deck, {"crop_max_lon"});
    const auto cropMinLat = deckFloat(deck, {"crop_min_lat"});
    const auto cropMaxLat = deckFloat(deck, {"crop_max_lat"});
    if (cropMinLon.has_value() != cropMaxLon.has_value()) {
        throw std::runtime_error("Config must define both crop_min_lon and crop_max_lon together.");
    }
    if (cropMinLat.has_value() != cropMaxLat.has_value()) {
        throw std::runtime_error("Config must define both crop_min_lat and crop_max_lat together.");
    }
    if (!cropMinLon && !domainLon) {
        throw std::runtime_error("Config missing crop_min_lon/crop_max_lon and cut_lon_manual/domain_lon");
    }
    if (!cropMinLat && !domainLat) {
        throw std::runtime_error("Config missing crop_min_lat/crop_max_lat and cut_lat_manual/domain_lat");
    }
    tke::RuntimeConfig result;
    result.configPath = configPath;
    result.minLon = cropMinLon ? *cropMinLon : domainLon->first;
    result.maxLon = cropMaxLon ? *cropMaxLon : domainLon->second;
    result.minLat = cropMinLat ? *cropMinLat : domainLat->first;
    result.maxLat = cropMaxLat ? *cropMaxLat : domainLat->second;
    if (result.minLon > result.maxLon) std::swap(result.minLon, result.maxLon);
    if (result.minLat > result.maxLat) std::swap(result.minLat, result.maxLat);
    result.domainLon = domainLon.value_or(std::pair{result.minLon, result.maxLon});
    result.domainLat = domainLat.value_or(std::pair{result.minLat, result.maxLat});
    if (result.domainLon.first > result.domainLon.second) std::swap(result.domainLon.first, result.domainLon.second);
    if (result.domainLat.first > result.domainLat.second) std::swap(result.domainLat.first, result.domainLat.second);
    const auto targetCrs = deckText(deck, {"target_crs", "TARGET_CRS", "utm_crs"});
    result.targetCrs = targetCrs.value_or(inferUtm(
        0.5 * (result.minLon + result.maxLon), 0.5 * (result.minLat + result.maxLat)));
    result.utmCrs = deckText(deck, {"utm_crs", "UTM_CRS"}).value_or(result.targetCrs);
    result.rotateDegrees = deckFloat(deck, {"rotate_deg"});
    result.cellSize = deckFloat(deck, {"crop_cell_size", "cell_size"}).value_or(10.0);
    result.visDpi = static_cast<int>(std::nearbyint(deckFloat(deck, {"crop_vis_dpi"}).value_or(1200.0)));
    if (result.visDpi <= 0) result.visDpi = 1200;
    result.cropZMax = deckFloat(deck, {"crop_z_max", "crop_max_z", "crop_z_limit", "crop_max_height"});
    result.fileGlob = deckText(deck, {"crop_debug_file_glob"}).value_or("*_avg-*.vtk,*_avg_*.vtk");
    const fs::path baseDirectory = configPath.parent_path();
    if (const auto debugInput = deckText(deck, {"crop_debug_input_dir"})) {
        std::string normalized = trim(*debugInput);
        while (!normalized.empty() && (normalized.front() == '/' || normalized.front() == '\\')) normalized.erase(normalized.begin());
        result.dataDirectory = resolved(baseDirectory / pathFromUtf8(normalized));
        std::error_code directoryError;
        if (!fs::is_directory(result.dataDirectory, directoryError) || directoryError) {
            throw std::runtime_error("crop_debug_input_dir resolved to a non-directory path: "
                                     + pathUtf8(result.dataDirectory));
        }
        result.dataDirectorySource = "crop_debug_input_dir=" + *debugInput;
    } else {
        const std::array<std::pair<fs::path, std::string>, 4> candidates{{
            {baseDirectory / "RESULTS" / "vtk", "RESULTS/vtk"},
            {baseDirectory / "RESULTS", "RESULTS"},
            {baseDirectory / "proj_temp" / "vtk", "proj_temp/vtk"},
            {baseDirectory, "."}}};
        bool found = false;
        for (const auto& [candidate, label] : candidates) {
            std::error_code directoryError;
            if (fs::is_directory(candidate, directoryError) && !directoryError) {
                result.dataDirectory = resolved(candidate);
                result.dataDirectorySource = "auto:" + label;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "Could not auto-discover an input VTK directory relative to the config file. Expected one of: "
                "<config_dir>/RESULTS/vtk, <config_dir>/RESULTS, <config_dir>/proj_temp/vtk, or <config_dir>.");
        }
    }
    result.outputDirectory = resolved(baseDirectory / "RESULTS" / "crop");
    return result;
}

[[nodiscard]] std::string configType(const fs::path& path) {
    std::string extension = pathUtf8(path.extension());
    if (!extension.empty() && extension.front() == '.') extension.erase(extension.begin());
    return extension.empty() ? "config" : extension;
}

void outputIntent(std::ostream& output, std::string_view label, const fs::path& path) {
    std::error_code error;
    output << "  - " << (fs::exists(path, error) && !error ? "Overwriting " : "Writing ")
           << label << ": " << pathUtf8(path) << '\n';
}

[[nodiscard]] std::string gridKey(const LegacyStructuredPoints& vtk) {
    std::ostringstream key;
    key << std::setprecision(17) << vtk.dimensions[0] << ',' << vtk.dimensions[1] << ',' << vtk.dimensions[2]
        << '|' << vtk.origin[0] << ',' << vtk.origin[1] << ',' << vtk.origin[2]
        << '|' << vtk.spacing[0] << ',' << vtk.spacing[1] << ',' << vtk.spacing[2];
    return key.str();
}

int executeBatch(const BatchArguments& args, std::ostream& output) {
    tke::RuntimeConfig config = loadBatchConfig(args.configPath);
    if (args.cropZMax) config.cropZMax = args.cropZMax;
    const fs::path outputRoot = config.outputDirectory;
    const fs::path croppedDirectory = outputRoot / "cropped_vtk";
    const fs::path figureDirectory = outputRoot / "figures";
    fs::create_directories(croppedDirectory);
    fs::create_directories(figureDirectory);
    output << "=== Batch Geo Crop / Visualization ===\n"
           << "Config path: " << pathUtf8(config.configPath) << '\n'
           << "Config type: " << configType(config.configPath) << '\n'
           << "Config directory: " << pathUtf8(config.configPath.parent_path()) << '\n'
           << "Input directory resolution order: <config_dir>/RESULTS/vtk -> <config_dir>/RESULTS -> "
              "<config_dir>/proj_temp/vtk -> <config_dir>\n"
           << "Input directory override key: crop_debug_input_dir (resolved relative to <config_dir>)\n"
           << "Resolved input directory: " << pathUtf8(config.dataDirectory) << '\n'
           << "Input directory source: " << config.dataDirectorySource << '\n'
           << "Resolved output root: " << pathUtf8(outputRoot) << '\n'
           << "Output policy: existing files with the same name will be overwritten.\n"
           << "crop_debug_file_glob: " << config.fileGlob << '\n'
           << "Crop lon/lat bounds: lon=[" << number(config.minLon) << ", " << number(config.maxLon)
           << "], lat=[" << number(config.minLat) << ", " << number(config.maxLat) << "]\n"
           << "VTK domain lon/lat: lon=[" << number(config.domainLon.first) << ", " << number(config.domainLon.second)
           << "], lat=[" << number(config.domainLat.first) << ", " << number(config.domainLat.second) << "]\n"
           << "Target CRS: " << config.targetCrs << '\n'
           << "Crop grid step: " << number(config.cellSize, 3) << " m\n";
    if (config.cropZMax) output << "Z crop upper bound: z <= " << number(*config.cropZMax) << " m\n";
    else output << "Z crop upper bound: disabled (all layers retained)\n";
    output << "Figure DPI: " << config.visDpi << "\nDiscovering input VTK files...\n";
    const std::vector<fs::path> files = tke::discoverVtkFiles(config.dataDirectory, config.fileGlob);
    const tke::Bounds bounds = tke::geographicBounds(config, {}, {}, {}, {});
    output << "Found " << files.size() << " VTK file(s) to process.\n"
           << "Using XY bounds (" << config.targetCrs << "): X=[" << number(bounds.minX, 2) << ", "
           << number(bounds.maxX, 2) << "], Y=[" << number(bounds.minY, 2) << ", "
           << number(bounds.maxY, 2) << "]\n";

    std::map<std::string, tke::Crop> crops;
    std::map<std::string, tke::TargetGrid> grids;
    std::map<std::string, tke::ZCrop> zCrops;
    for (std::size_t fileIndex = 0; fileIndex < files.size(); ++fileIndex) {
        const fs::path& path = files[fileIndex];
        output << "\n[" << fileIndex + 1 << '/' << files.size() << "] Starting " << pathUtf8(path.filename()) << '\n';
        const LegacyStructuredPoints vtk = LegacyStructuredPoints::readHeader(path);
        output << "  - Grid signature: dimensions=(" << vtk.dimensions[0] << ", " << vtk.dimensions[1] << ", "
               << vtk.dimensions[2] << "), origin=(" << pythonFloat(vtk.origin[0]) << ", "
               << pythonFloat(vtk.origin[1]) << ", " << pythonFloat(vtk.origin[2]) << "), spacing=("
               << pythonFloat(vtk.spacing[0]) << ", " << pythonFloat(vtk.spacing[1]) << ", "
               << pythonFloat(vtk.spacing[2]) << ")\n  - Available fields: [";
        for (std::size_t index = 0; index < vtk.fields.size(); ++index) {
            if (index != 0) output << ", ";
            output << '\'' << vtk.fields[index].name << '\'';
        }
        output << "]\n";
        const tke::WindFields windFields = tke::selectWindFields(vtk, false);
        const auto& tkeField = tke::selectTkeField(vtk, false);
        output << "  - Wind field selected for visualization: " << windFields.label() << '\n'
               << "  - TKE field selected for visualization: " << tkeField.name << '\n';
        const std::string key = gridKey(vtk);
        if (!crops.contains(key)) {
            output << "  - Computing crop window and target grid for this grid signature...\n";
            crops.emplace(key, tke::geographicCrop(vtk, config, bounds));
            grids.emplace(key, tke::geographicTargetGrid(vtk, config, bounds, crops.at(key)));
            zCrops.emplace(key, tke::computeZCrop(vtk, config.cropZMax));
            const auto& crop = crops.at(key);
            const auto& grid = grids.at(key);
            const auto& zCrop = zCrops.at(key);
            const std::size_t valid = static_cast<std::size_t>(std::count(grid.valid.begin(), grid.valid.end(), 1));
            output << "  - Crop computed: X_idx=[" << crop.xStart << ", " << crop.xEnd << "), Y_idx=["
                   << crop.yStart << ", " << crop.yEnd << "), size=" << crop.nx() << 'x' << crop.ny() << '\n'
                   << "  - Z crop computed: Z_idx=[" << zCrop.zStart << ", " << zCrop.zEnd << "), z=["
                   << number(zCrop.zMin, 3) << ", " << number(zCrop.zMaxKept, 3) << "] m, layers="
                   << zCrop.nzOut << '/' << vtk.dimensions[2] << '\n'
                   << "  - Target grid computed: " << grid.nx << 'x' << grid.ny << ", step="
                   << number(grid.step, 2) << "m, valid="
                   << number(static_cast<double>(valid) / grid.valid.size() * 100.0, 2) << "%\n";
        } else {
            output << "  - Reusing cached crop window and target grid for matching grid signature.\n";
        }
        const std::string stem = pathUtf8(path.stem());
        const fs::path croppedOutput = croppedDirectory / (stem + "_cropped.vtk");
        const fs::path rangeOutput = figureDirectory / (stem + "_range_diagnostic.png");
        const fs::path windOutput = figureDirectory / (stem + "_wind_arrows_9layers.png");
        const fs::path tkeOutput = figureDirectory / (stem + "_tke_9layers.png");
        outputIntent(output, "cropped VTK", croppedOutput);
        tke::exportCroppedVtk(vtk, crops.at(key), zCrops.at(key), croppedOutput, output);
        outputIntent(output, "range diagnostic figure", rangeOutput);
        tke::renderRangeDiagnostic(crops.at(key), rangeOutput, config.targetCrs, std::max(80, config.visDpi));
        output << "Saved range diagnostic figure: " << pathUtf8(rangeOutput) << '\n';
        outputIntent(output, "wind figure", windOutput);
        tke::renderWindFigure(vtk, windFields, tke::heightPlan(vtk), grids.at(key), 50,
                              windOutput, std::max(80, config.visDpi));
        output << "Saved wind figure: " << pathUtf8(windOutput) << '\n';
        outputIntent(output, "TKE figure", tkeOutput);
        tke::renderTkeFigure(vtk, tkeField, tke::heightPlan(vtk), grids.at(key),
                             tkeOutput, std::max(80, config.visDpi));
        output << "Saved TKE figure: " << pathUtf8(tkeOutput) << '\n'
               << '[' << fileIndex + 1 << '/' << files.size() << "] Finished " << pathUtf8(path.filename()) << '\n';
    }
    output << "\nProcessing complete.\nProcessed files: " << files.size()
           << "\nCropped VTK directory: " << pathUtf8(croppedDirectory)
           << "\nFigure directory: " << pathUtf8(figureDirectory) << '\n';
    return 0;
}

int runTkeNamed(std::string_view program,
                const std::vector<std::string>& arguments,
                std::ostream& output,
                std::ostream& error) {
    try {
        const TkeArguments parsed = parseTkeArguments(arguments);
        if (parsed.help) { tkeUsage(output, program); return 0; }
        return executeTke(parsed, output);
    } catch (const CommandExit& exception) {
        if (exception.code() == 2) {
            error << "usage: " << program << " [options] [vtk_file]\n"
                  << program << ": error: " << exception.what() << '\n';
        } else error << "ERROR: " << exception.what() << '\n';
        return exception.code();
    } catch (const std::exception& exception) {
        error << "ERROR: " << exception.what() << '\n';
        return 1;
    }
}

} // namespace

int run_tke_viz(const std::vector<std::string>& arguments,
                std::ostream& output,
                std::ostream& error) {
    return runTkeNamed("tke_viz", arguments, output, error);
}

int run_cut_vis(const std::vector<std::string>& arguments,
                std::ostream& output,
                std::ostream& error) {
    return runTkeNamed("cut_vis", arguments, output, error);
}

int run_batch_tke_geo_viz(const std::vector<std::string>& arguments,
                          std::ostream& output,
                          std::ostream& error) {
    try {
        return executeBatch(parseBatchArguments(arguments), output);
    } catch (const CommandExit& exception) {
        error << "usage: batch_tke_geo_viz <config_path>\n"
              << "batch_tke_geo_viz: error: " << exception.what() << '\n';
        return exception.code();
    } catch (const std::exception& exception) {
        error << "ERROR: " << exception.what() << '\n';
        return 1;
    }
}

} // namespace luw::cli
