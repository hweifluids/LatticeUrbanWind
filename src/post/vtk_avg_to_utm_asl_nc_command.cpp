#include "luw/vtk_post_commands.hpp"

#include "luw/deck_document.hpp"
#include "luw/legacy_vtk.hpp"
#include "post_internal.hpp"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace luw::cli {

namespace {

namespace fs = std::filesystem;
using luw::post::LegacyStructuredPoints;
using luw::post::LegacyVtkAssociation;
using luw::post::LegacyVtkField;
using luw::post::LegacyVtkPlaneReader;
using luw::post::detail::NetcdfFile;
using luw::post::detail::Projection;
using luw::post::detail::defineCompressedDoubleVariable;
using luw::post::detail::defineCompressedFloatVariable;
using luw::post::detail::ncCheck;
using luw::post::detail::pathUtf8;
using luw::post::detail::setTextAttribute;

constexpr std::array<std::string_view, 4> caseOrder{
    "beijing", "chongqing", "shanghai", "singapore"};

struct AverageArguments final {
    fs::path root = LR"(D:\averagedField\New0518_average)";
    std::optional<fs::path> caseDirectory;
    std::optional<fs::path> config;
    std::optional<fs::path> rangeFile;
    std::optional<double> terrainMinAsl;
    std::optional<std::vector<std::string>> cases;
    fs::path inputSubdirectory = fs::path("RESULTS") / "crop" / "cropped_vtk_raw_assembled";
    std::string inputGlob = "ANG_*_avg-*_cropped.vtk";
    fs::path outputSubdirectory = fs::path("RESULTS") / "nc_utm_asl";
    double pedestalHeight = 50.0;
    int compressionLevel = 3;
    std::optional<int> limit;
    bool overwrite = false;
    bool help = false;
};

struct CaseConfig final {
    fs::path configPath;
    fs::path projectHome;
    std::string casename;
    std::optional<std::string> datetime;
    std::pair<double, double> cutLon;
    std::pair<double, double> cutLat;
    std::string utmCrs;
    double rotateDegrees = 0.0;
};

struct TransformModel final {
    std::string utmCrs;
    double rotateDegrees = 0.0;
    double pivotX = 0.0;
    double pivotY = 0.0;
    double xOriginRotated = 0.0;
    double yOriginRotated = 0.0;
};

struct FieldComponent final {
    const LegacyVtkField* field = nullptr;
    std::size_t component = 0;
};

struct WindSelection final {
    FieldComponent u;
    FieldComponent v;
    FieldComponent w;
    std::vector<const LegacyVtkField*> scalarFields;
};

class AverageArgumentError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &time);
#else
    localtime_r(&time, &local);
#endif
    std::ostringstream stream;
    stream << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
    return stream.str();
}

void log(std::ostream& output, const std::string& message) {
    output << "[INFO] " << timestamp() << " | " << message << '\n';
}

void warn(std::ostream& output, const std::string& message) {
    output << "[WARN] " << timestamp() << " | " << message << '\n';
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return text;
}

fs::path resolved(const fs::path& path) {
    std::error_code error;
    fs::path absolute = fs::absolute(path, error);
    return (error ? path : absolute).lexically_normal();
}

double parseDoubleOption(const std::string& option, const std::string& text) {
    std::size_t consumed = 0;
    double result = 0.0;
    try {
        result = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw AverageArgumentError(option + " expects a number, got '" + text + "'");
    }
    if (consumed != text.size()) throw AverageArgumentError(option + " expects a number, got '" + text + "'");
    return result;
}

int parseIntOption(const std::string& option, const std::string& text) {
    std::size_t consumed = 0;
    long result = 0;
    try {
        result = std::stol(text, &consumed, 10);
    } catch (const std::exception&) {
        throw AverageArgumentError(option + " expects an integer, got '" + text + "'");
    }
    if (consumed != text.size() || result < std::numeric_limits<int>::min()
        || result > std::numeric_limits<int>::max()) {
        throw AverageArgumentError(option + " expects an integer, got '" + text + "'");
    }
    return static_cast<int>(result);
}

void printAverageUsage(std::ostream& output) {
    output << "usage: vtk_avg_to_utm_asl_nc [-h] [--root ROOT] [--case-dir CASE_DIR]\n"
              "                                [--config CONFIG] [--range-file RANGE_FILE]\n"
              "                                [--terrain-min-asl M] [--cases [CASE ...]]\n"
              "                                [--input-subdir PATH] [--input-glob GLOB]\n"
              "                                [--output-subdir PATH] [--pedestal-height M]\n"
              "                                [--compression-level 0..9] [--limit N] [--overwrite]\n\n"
              "Convert cropped avg VTK files to UTM/ASL NetCDF.\n";
}

AverageArguments parseAverageArguments(const std::vector<std::string>& arguments) {
    AverageArguments parsed;
    auto valueAfter = [&](std::size_t& index, const std::string& option) -> const std::string& {
        if (++index >= arguments.size()) throw AverageArgumentError(option + " expects one argument");
        return arguments[index];
    };
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "-h" || argument == "--help") parsed.help = true;
        else if (argument == "--root") parsed.root = valueAfter(index, argument);
        else if (argument == "--case-dir") parsed.caseDirectory = valueAfter(index, argument);
        else if (argument == "--config") parsed.config = valueAfter(index, argument);
        else if (argument == "--range-file") parsed.rangeFile = valueAfter(index, argument);
        else if (argument == "--terrain-min-asl") {
            parsed.terrainMinAsl = parseDoubleOption(argument, valueAfter(index, argument));
        } else if (argument == "--cases") {
            std::vector<std::string> values;
            while (index + 1 < arguments.size() && !arguments[index + 1].starts_with('-')) {
                values.push_back(arguments[++index]);
            }
            parsed.cases = std::move(values);
        } else if (argument == "--input-subdir") parsed.inputSubdirectory = valueAfter(index, argument);
        else if (argument == "--input-glob") parsed.inputGlob = valueAfter(index, argument);
        else if (argument == "--output-subdir") parsed.outputSubdirectory = valueAfter(index, argument);
        else if (argument == "--pedestal-height") {
            parsed.pedestalHeight = parseDoubleOption(argument, valueAfter(index, argument));
        } else if (argument == "--compression-level") {
            parsed.compressionLevel = parseIntOption(argument, valueAfter(index, argument));
        } else if (argument == "--limit") {
            parsed.limit = parseIntOption(argument, valueAfter(index, argument));
        } else if (argument == "--overwrite") parsed.overwrite = true;
        else throw AverageArgumentError("unrecognized argument: " + argument);
    }
    parsed.compressionLevel = std::clamp(parsed.compressionLevel, 0, 9);
    return parsed;
}

std::string inferUtm(double longitude, double latitude) {
    int zone = static_cast<int>(std::floor((longitude + 180.0) / 6.0) + 1.0);
    zone = std::clamp(zone, 1, 60);
    return "EPSG:" + std::to_string((latitude >= 0.0 ? 32600 : 32700) + zone);
}

double rotationEstimate(const std::pair<double, double>& longitude,
                        const std::pair<double, double>& latitude,
                        const std::string& utmCrs) {
    Projection projection("EPSG:4326", utmCrs);
    const auto p00 = projection.apply(longitude.first, latitude.first);
    const auto p10 = projection.apply(longitude.second, latitude.first);
    constexpr double radiansToDegrees = 57.2957795130823208768;
    return -std::atan2(p10.second - p00.second, p10.first - p00.first) * radiansToDegrees;
}

CaseConfig parseCaseConfig(const fs::path& configPath, std::ostream& output) {
    const fs::path absolute = resolved(configPath);
    if (!fs::is_regular_file(absolute)) throw std::runtime_error("LUW file not found: " + absolute.string());
    const std::string configUtf8 = pathUtf8(absolute);
    const luw::DeckDocument deck = luw::DeckDocument::load(
        QString::fromUtf8(configUtf8.data(), static_cast<qsizetype>(configUtf8.size())));
    CaseConfig config;
    config.configPath = absolute;
    config.projectHome = absolute.parent_path();
    if (const auto value = deck.text(QStringLiteral("casename"))) config.casename = value->toStdString();
    if (config.casename.empty()) throw std::runtime_error("Missing key 'casename' in LUW file.");
    if (const auto value = deck.text(QStringLiteral("datetime")); value && !value->isEmpty()) {
        config.datetime = value->toStdString();
    }
    const auto longitude = deck.pair(QStringLiteral("cut_lon_manual"));
    const auto latitude = deck.pair(QStringLiteral("cut_lat_manual"));
    if (!longitude || !latitude) {
        throw std::runtime_error("Missing 'cut_lon_manual' or 'cut_lat_manual' in LUW file.");
    }
    config.cutLon = *longitude;
    config.cutLat = *latitude;
    if (const auto value = deck.text(QStringLiteral("utm_crs")); value && !value->isEmpty()) {
        config.utmCrs = value->toStdString();
    } else {
        config.utmCrs = inferUtm((longitude->first + longitude->second) * 0.5,
                                 (latitude->first + latitude->second) * 0.5);
        warn(output, "'utm_crs' not found in LUW. Auto-inferred CRS: " + config.utmCrs);
    }
    const double estimated = rotationEstimate(*longitude, *latitude, config.utmCrs);
    if (const auto value = deck.floating(QStringLiteral("rotate_deg"))) {
        config.rotateDegrees = *value;
        std::ostringstream message;
        message << std::fixed << std::setprecision(9)
                << "Rotation from LUW=" << config.rotateDegrees
                << " deg, recomputed=" << estimated
                << " deg, abs diff=" << std::scientific << std::abs(config.rotateDegrees - estimated) << '.';
        log(output, message.str());
    } else {
        config.rotateDegrees = estimated;
        std::ostringstream message;
        message << std::fixed << std::setprecision(9)
                << "Rotation not found in LUW. Using recomputed rotation " << estimated << " deg.";
        log(output, message.str());
    }
    return config;
}

std::pair<double, double> rotatePoint(double x, double y, double degrees,
                                     double pivotX, double pivotY) {
    constexpr double degreesToRadians = 0.0174532925199432957692;
    const double angle = degrees * degreesToRadians;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return {cosine * (x - pivotX) - sine * (y - pivotY) + pivotX,
            sine * (x - pivotX) + cosine * (y - pivotY) + pivotY};
}

TransformModel buildTransformModel(const CaseConfig& config, std::ostream& output) {
    Projection projection("EPSG:4326", config.utmCrs);
    std::array<std::pair<double, double>, 4> corners{
        projection.apply(config.cutLon.first, config.cutLat.first),
        projection.apply(config.cutLon.second, config.cutLat.first),
        projection.apply(config.cutLon.second, config.cutLat.second),
        projection.apply(config.cutLon.first, config.cutLat.second)};
    double pivotX = 0.0;
    double pivotY = 0.0;
    for (const auto& corner : corners) {
        pivotX += corner.first * 0.25;
        pivotY += corner.second * 0.25;
    }
    double minimumX = std::numeric_limits<double>::infinity();
    double minimumY = std::numeric_limits<double>::infinity();
    for (const auto& corner : corners) {
        const auto rotated = rotatePoint(corner.first, corner.second,
                                         config.rotateDegrees, pivotX, pivotY);
        minimumX = std::min(minimumX, rotated.first);
        minimumY = std::min(minimumY, rotated.second);
    }
    TransformModel model{config.utmCrs, config.rotateDegrees, pivotX, pivotY, minimumX, minimumY};
    std::ostringstream message;
    message << std::fixed << std::setprecision(9)
            << "Transform model built: utm_crs=" << model.utmCrs
            << ", rotate_deg=" << model.rotateDegrees << std::setprecision(3)
            << ", pivot=(" << model.pivotX << ", " << model.pivotY << ')'
            << ", local_origin_rot=(" << model.xOriginRotated << ", "
            << model.yOriginRotated << ").";
    log(output, message.str());
    return model;
}

const LegacyVtkField* fieldCaseInsensitive(const LegacyStructuredPoints& vtk, std::string_view name) {
    const LegacyVtkField* result = vtk.fieldCaseInsensitive(name);
    return result && result->association == LegacyVtkAssociation::PointData ? result : nullptr;
}

WindSelection selectWind(const LegacyStructuredPoints& vtk) {
    const LegacyVtkField* u = fieldCaseInsensitive(vtk, "u");
    const LegacyVtkField* v = fieldCaseInsensitive(vtk, "v");
    const LegacyVtkField* w = fieldCaseInsensitive(vtk, "w");
    WindSelection selection;
    std::set<std::string> excluded;
    if (u && v && w) {
        if (u->components != 1 || v->components != 1 || w->components != 1) {
            throw std::runtime_error(vtk.path.filename().string()
                                     + ": scalar u/v/w fields must have one component each");
        }
        selection.u = {u, 0};
        selection.v = {v, 0};
        selection.w = {w, 0};
        excluded = {"u", "v", "w"};
    } else {
        std::vector<const LegacyVtkField*> vectors;
        for (const LegacyVtkField& field : vtk.fields) {
            if (field.association == LegacyVtkAssociation::PointData
                && field.tuples == vtk.pointCount && field.components >= 3) {
                vectors.push_back(&field);
            }
        }
        const LegacyVtkField* main = nullptr;
        for (const std::string_view name : {"u_avg", "u", "data", "uvw", "velocity"}) {
            const auto found = std::find_if(vectors.begin(), vectors.end(), [&](const LegacyVtkField* field) {
                return lower(field->name) == name;
            });
            if (found != vectors.end()) {
                main = *found;
                break;
            }
        }
        if (!main && !vectors.empty()) main = vectors.front();
        if (!main) {
            std::ostringstream names;
            for (std::size_t index = 0; index < vtk.fields.size(); ++index) {
                if (index) names << ", ";
                names << vtk.fields[index].name << "(ncomp=" << vtk.fields[index].components << ')';
            }
            throw std::runtime_error(vtk.path.filename().string()
                                     + ": expected u/v/w scalars or a 3-component wind field. Available: "
                                     + names.str());
        }
        selection.u = {main, 0};
        selection.v = {main, 1};
        selection.w = {main, 2};
        excluded.insert(lower(main->name));
    }
    for (const LegacyVtkField& field : vtk.fields) {
        if (field.association == LegacyVtkAssociation::PointData
            && field.tuples == vtk.pointCount && field.components == 1
            && !excluded.contains(lower(field.name))) {
            selection.scalarFields.push_back(&field);
        }
    }
    return selection;
}

std::string wildcardRegex(std::string_view pattern) {
    std::string result = "^";
    for (const char value : pattern) {
        if (value == '*') result += ".*";
        else if (value == '?') result += '.';
        else {
            if (std::string_view(".^$|()[]{}+\\").find(value) != std::string_view::npos) result += '\\';
            result += value;
        }
    }
    return result + '$';
}

std::vector<fs::path> discoverVtkFiles(const fs::path& caseDirectory,
                                       const fs::path& subdirectory,
                                       const std::string& glob) {
    const fs::path directory = caseDirectory / subdirectory;
    if (!fs::is_directory(directory)) {
        throw std::runtime_error("VTK input directory not found: " + directory.string());
    }
    const std::regex matcher(wildcardRegex(glob), std::regex::ECMAScript | std::regex::icase);
    std::vector<fs::path> result;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), matcher)) {
            result.push_back(resolved(entry.path()));
        }
    }
    std::sort(result.begin(), result.end());
    if (result.empty()) {
        throw std::runtime_error("No avg VTK files matching '" + glob + "' found in " + directory.string());
    }
    return result;
}

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::map<std::string, double> parseRangeAsl(const fs::path& path,
                                            const std::optional<std::vector<std::string>>& required) {
    if (!fs::is_regular_file(path)) throw std::runtime_error("Range file not found: " + path.string());
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Range file not found: " + path.string());
    std::vector<std::string> wanted;
    if (required) {
        for (const std::string& name : *required) wanted.push_back(lower(name));
    } else {
        for (const auto name : caseOrder) wanted.emplace_back(name);
    }
    const std::regex cityPattern(R"(^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$)");
    const std::regex explicitPattern(R"(terrain_min_asl_m\s*=\s*([-+]?\d+(?:\.\d+)?))", std::regex::icase);
    const std::regex plainPattern(R"(^\s*([-+]?\d+(?:\.\d+)?)\s*m\s*$)", std::regex::icase);
    const std::regex numberPattern(R"(([-+]?\d+(?:\.\d+)?)\s*m\b)", std::regex::icase);
    std::map<std::string, double> values;
    std::optional<std::string> current;
    for (std::string line; std::getline(input, line);) {
        line = trim(std::move(line));
        if (line.empty()) continue;
        const std::string lowered = lower(line);
        for (const std::string& name : wanted) {
            if (lowered.find(name) == std::string::npos) continue;
            std::optional<std::string> last;
            for (std::sregex_iterator match(line.begin(), line.end(), numberPattern), end;
                 match != end; ++match) last = (*match)[1].str();
            if (last) {
                values[name] = std::stod(*last);
                current.reset();
            }
            break;
        }
        std::smatch match;
        if (std::regex_match(line, match, cityPattern)) {
            current = lower(match[1].str());
            continue;
        }
        if (!current) continue;
        if (std::regex_search(line, match, explicitPattern)) values[*current] = std::stod(match[1].str());
        else if (std::regex_match(line, match, plainPattern)) values[*current] = std::stod(match[1].str());
    }
    if (required) {
        std::vector<std::string> missing;
        for (const std::string& name : *required) {
            if (!values.contains(lower(name))) missing.push_back(lower(name));
        }
        if (!missing.empty()) {
            std::ostringstream message;
            for (std::size_t index = 0; index < missing.size(); ++index) {
                if (index) message << ", ";
                message << missing[index];
            }
            throw std::runtime_error("Missing ASL values in " + path.string() + ": " + message.str());
        }
    }
    return values;
}

std::optional<double> prepSummaryAsl(const fs::path& caseDirectory, std::ostream& output) {
    const fs::path path = caseDirectory / "prep_summary.json";
    if (!fs::is_regular_file(path)) return std::nullopt;
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::ReadOnly)) {
        warn(output, "Failed to read " + path.string() + ": cannot open file");
        return std::nullopt;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        warn(output, "Failed to read " + path.string() + ": " + parseError.errorString().toStdString());
        return std::nullopt;
    }
    const QJsonValue value = document.object().value(QStringLiteral("dem_absolute_min_m"));
    double result = 0.0;
    if (value.isDouble()) {
        result = value.toDouble();
    } else if (value.isBool()) {
        // Python's float(), used by the legacy reader, accepts JSON booleans.
        result = value.toBool() ? 1.0 : 0.0;
    } else if (value.isString()) {
        // prep_summary.json files produced by adjacent workflows have also
        // stored this value as a numeric string.  Preserve the legacy
        // float(value) coercion instead of silently falling through to Range.txt.
        bool converted = false;
        result = value.toString().trimmed().toDouble(&converted);
        if (!converted) return std::nullopt;
    } else {
        return std::nullopt;
    }
    return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
}

fs::path findCaseConfig(const fs::path& caseDirectory, const std::optional<fs::path>& explicitPath) {
    if (explicitPath) {
        const fs::path path = resolved(*explicitPath);
        if (!fs::is_regular_file(path)) throw std::runtime_error("Config file not found: " + path.string());
        return path;
    }
    for (const std::string_view name : {"conf.luwpf", "conf.luwdg", "conf.luw"}) {
        const fs::path path = caseDirectory / name;
        if (fs::is_regular_file(path)) return path;
    }
    throw std::runtime_error("No conf.luwpf/conf.luwdg/conf.luw found in " + caseDirectory.string());
}

double resolveTerrainAsl(const fs::path& caseDirectory,
                         const std::string& caseName,
                         const std::optional<double>& explicitValue,
                         const std::optional<fs::path>& rangePath,
                         std::ostream& output) {
    if (explicitValue) {
        if (!std::isfinite(*explicitValue)) {
            throw std::runtime_error("Invalid --terrain-min-asl: " + std::to_string(*explicitValue));
        }
        return *explicitValue;
    }
    if (const auto value = prepSummaryAsl(caseDirectory, output)) {
        std::ostringstream message;
        message << std::fixed << std::setprecision(9)
                << "Terrain min ASL from prep_summary.json: " << *value << " m";
        log(output, message.str());
        return *value;
    }
    if (rangePath && fs::is_regular_file(*rangePath)) {
        const auto values = parseRangeAsl(*rangePath, std::nullopt);
        if (const auto found = values.find(lower(caseName)); found != values.end()) return found->second;
    }
    throw std::runtime_error(
        "Terrain minimum ASL was not provided and could not be inferred. "
        "Use --terrain-min-asl or provide prep_summary.json with dem_absolute_min_m.");
}

std::string sanitizeVariableName(std::string name) {
    for (char& value : name) {
        const unsigned char ch = static_cast<unsigned char>(value);
        if (!std::isalnum(ch) && value != '_') value = '_';
    }
    name = lower(trim(std::move(name)));
    if (name.empty()) name = "scalar";
    if (std::isdigit(static_cast<unsigned char>(name.front()))) name = "v_" + name;
    return name;
}

std::string angleToken(const fs::path& path) {
    std::smatch match;
    const std::string name = path.filename().string();
    if (std::regex_search(name, match, std::regex(R"(ANG_([^_]+)_)", std::regex::icase))) {
        return match[1].str();
    }
    if (std::regex_search(name, match, std::regex(R"(DG_[^_]+_([^_]+)_)", std::regex::icase))) {
        return match[1].str();
    }
    return {};
}

std::string commaTriplet(const std::array<double, 3>& values) {
    std::ostringstream stream;
    stream << std::setprecision(12) << values[0] << ',' << values[1] << ',' << values[2];
    return stream.str();
}

std::optional<int> epsgCode(const std::string& crs) {
    std::smatch match;
    if (!std::regex_match(crs, match, std::regex(R"(EPSG\s*:\s*(\d+))", std::regex::icase))) {
        return std::nullopt;
    }
    return std::stoi(match[1].str());
}

std::string crsWkt(const std::string& crs) {
    PJ_CONTEXT* context = proj_context_create();
    if (!context) return {};
    PJ* object = proj_create(context, crs.c_str());
    std::string result;
    if (object) {
        // pyproj.CRS.to_wkt(), used by the legacy script, emits a compact
        // single-line WKT2 string unless explicitly asked for pretty output.
        const char* options[]{"MULTILINE=NO", nullptr};
        if (const char* wkt = proj_as_wkt(context, object, PJ_WKT2_2019, options)) result = wkt;
        proj_destroy(object);
    }
    proj_context_destroy(context);
    return result;
}

struct OutputScalar final {
    const LegacyVtkField* field = nullptr;
    std::string name;
    int variable = -1;
};

int defineDataVariable(int ncid, const std::string& name, const std::string& longName,
                       const std::string& units, const std::vector<int>& dimensions,
                       int compression, std::size_t ny, std::size_t nx) {
    const int variable = defineCompressedFloatVariable(ncid, name.c_str(), dimensions,
                                                       compression, true, true);
    const std::array<std::size_t, 3> chunks{1, std::min<std::size_t>(ny, 256),
                                            std::min<std::size_t>(nx, 256)};
    ncCheck(nc_def_var_chunking(ncid, variable, NC_CHUNKED, chunks.data()),
            "Failed to configure chunks for '" + name + "'");
    setTextAttribute(ncid, variable, "long_name", longName);
    if (!units.empty()) setTextAttribute(ncid, variable, "units", units);
    setTextAttribute(ncid, variable, "coordinates", "z easting northing");
    setTextAttribute(ncid, variable, "grid_mapping", "crs");
    return variable;
}

fs::path convertAverageFile(const fs::path& vtkPath,
                            const CaseConfig& config,
                            double terrainMinAsl,
                            const fs::path& outputDirectory,
                            double pedestalHeight,
                            int compression,
                            bool overwrite,
                            std::ostream& output) {
    const LegacyStructuredPoints vtk = LegacyStructuredPoints::readHeader(vtkPath);
    const WindSelection wind = selectWind(vtk);
    const int nx = vtk.dimensions[0];
    const int ny = vtk.dimensions[1];
    const int nz = vtk.dimensions[2];
    std::vector<std::size_t> kept;
    std::vector<float> zAsl;
    for (int index = 0; index < nz; ++index) {
        const double z = vtk.origin[2] + index * vtk.spacing[2];
        if (z >= pedestalHeight - 1.0e-6) {
            kept.push_back(static_cast<std::size_t>(index));
            zAsl.push_back(static_cast<float>(z - pedestalHeight + terrainMinAsl));
        }
    }
    if (kept.empty()) {
        throw std::runtime_error(vtkPath.filename().string()
                                 + ": no z layers remain after removing pedestal height "
                                 + std::to_string(pedestalHeight) + " m");
    }
    const fs::path outputPath = outputDirectory / (vtkPath.stem().string() + "_utm_asl.nc");
    if (fs::exists(outputPath) && !overwrite) {
        log(output, "Skip existing: " + outputPath.string());
        return outputPath;
    }

    const TransformModel transform = buildTransformModel(config, output);
    std::vector<double> xLocal(static_cast<std::size_t>(nx));
    std::vector<double> yLocal(static_cast<std::size_t>(ny));
    for (int x = 0; x < nx; ++x) xLocal[static_cast<std::size_t>(x)] = vtk.origin[0] + x * vtk.spacing[0];
    for (int y = 0; y < ny; ++y) yLocal[static_cast<std::size_t>(y)] = vtk.origin[1] + y * vtk.spacing[1];
    std::vector<double> easting(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    std::vector<double> northing(easting.size());
    std::size_t cursor = 0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const auto projected = rotatePoint(xLocal[static_cast<std::size_t>(x)] + transform.xOriginRotated,
                                               yLocal[static_cast<std::size_t>(y)] + transform.yOriginRotated,
                                               -transform.rotateDegrees,
                                               transform.pivotX, transform.pivotY);
            easting[cursor] = projected.first;
            northing[cursor] = projected.second;
            ++cursor;
        }
    }

    fs::create_directories(outputDirectory);
    NetcdfFile file(outputPath);
    const int ncid = file.id();
    int dimZ = -1;
    int dimY = -1;
    int dimX = -1;
    ncCheck(nc_def_dim(ncid, "z", kept.size(), &dimZ), "Failed to define z dimension");
    ncCheck(nc_def_dim(ncid, "y", static_cast<std::size_t>(ny), &dimY), "Failed to define y dimension");
    ncCheck(nc_def_dim(ncid, "x", static_cast<std::size_t>(nx), &dimX), "Failed to define x dimension");
    const int varX = defineCompressedDoubleVariable(ncid, "x", {dimX}, compression);
    const int varY = defineCompressedDoubleVariable(ncid, "y", {dimY}, compression);
    const int varZ = defineCompressedFloatVariable(ncid, "z", {dimZ}, compression, true, false);
    const int varEasting = defineCompressedDoubleVariable(ncid, "easting", {dimY, dimX}, compression);
    const int varNorthing = defineCompressedDoubleVariable(ncid, "northing", {dimY, dimX}, compression);
    setTextAttribute(ncid, varX, "long_name", "local CFD x coordinate before UTM inverse rotation");
    setTextAttribute(ncid, varX, "units", "m");
    setTextAttribute(ncid, varY, "long_name", "local CFD y coordinate before UTM inverse rotation");
    setTextAttribute(ncid, varY, "units", "m");
    setTextAttribute(ncid, varZ, "standard_name", "altitude");
    setTextAttribute(ncid, varZ, "long_name", "altitude above mean sea level");
    setTextAttribute(ncid, varZ, "units", "m");
    setTextAttribute(ncid, varZ, "positive", "up");
    setTextAttribute(ncid, varEasting, "standard_name", "projection_x_coordinate");
    setTextAttribute(ncid, varEasting, "long_name", "UTM easting (" + config.utmCrs + ')');
    setTextAttribute(ncid, varEasting, "units", "m");
    setTextAttribute(ncid, varNorthing, "standard_name", "projection_y_coordinate");
    setTextAttribute(ncid, varNorthing, "long_name", "UTM northing (" + config.utmCrs + ')');
    setTextAttribute(ncid, varNorthing, "units", "m");

    int varCrs = -1;
    ncCheck(nc_def_var(ncid, "crs", NC_INT, 0, nullptr, &varCrs), "Failed to define crs variable");
    setTextAttribute(ncid, varCrs, "long_name", "coordinate reference system");
    setTextAttribute(ncid, varCrs, "grid_mapping_name", "transverse_mercator");
    const std::string wkt = crsWkt(config.utmCrs);
    if (!wkt.empty()) {
        setTextAttribute(ncid, varCrs, "spatial_ref", wkt);
        setTextAttribute(ncid, varCrs, "crs_wkt", wkt);
    }
    if (const auto epsg = epsgCode(config.utmCrs)) {
        ncCheck(nc_put_att_int(ncid, varCrs, "epsg_code", NC_INT, 1, &*epsg),
                "Failed to set EPSG code");
    }

    const std::vector<int> dataDimensions{dimZ, dimY, dimX};
    const int varU = defineDataVariable(ncid, "u", "eastward wind component in UTM CRS",
                                        "m s-1", dataDimensions, compression, ny, nx);
    const int varV = defineDataVariable(ncid, "v", "northward wind component in UTM CRS",
                                        "m s-1", dataDimensions, compression, ny, nx);
    const int varW = defineDataVariable(ncid, "w", "vertical wind component",
                                        "m s-1", dataDimensions, compression, ny, nx);
    setTextAttribute(ncid, varU, "standard_name", "eastward_wind");
    setTextAttribute(ncid, varV, "standard_name", "northward_wind");
    setTextAttribute(ncid, varW, "standard_name", "upward_air_velocity");
    setTextAttribute(ncid, varU, "source_field", wind.u.field->name + '[' + std::to_string(wind.u.component) + ']');
    setTextAttribute(ncid, varV, "source_field", wind.v.field->name + '[' + std::to_string(wind.v.component) + ']');
    setTextAttribute(ncid, varW, "source_field", wind.w.field->name + '[' + std::to_string(wind.w.component) + ']');

    std::set<std::string> used{"u", "v", "w"};
    std::vector<OutputScalar> outputScalars;
    for (const LegacyVtkField* scalar : wind.scalarFields) {
        const std::string base = sanitizeVariableName(scalar->name);
        std::string name = base;
        for (int suffix = 2; used.contains(name); ++suffix) name = base + '_' + std::to_string(suffix);
        used.insert(name);
        const int variable = defineDataVariable(ncid, name, scalar->name,
                                                name == "tke" ? "m2 s-2" : "",
                                                dataDimensions, compression, ny, nx);
        setTextAttribute(ncid, variable, "source_field", scalar->name + "[0]");
        outputScalars.push_back({scalar, name, variable});
    }

    setTextAttribute(ncid, NC_GLOBAL, "source_vtk", pathUtf8(vtk.path));
    setTextAttribute(ncid, NC_GLOBAL, "casename", config.casename);
    if (config.datetime) setTextAttribute(ncid, NC_GLOBAL, "case_datetime", *config.datetime);
    setTextAttribute(ncid, NC_GLOBAL, "angle", angleToken(vtk.path));
    setTextAttribute(ncid, NC_GLOBAL, "utm_crs", config.utmCrs);
    ncCheck(nc_put_att_double(ncid, NC_GLOBAL, "rotate_deg", NC_DOUBLE, 1, &config.rotateDegrees),
            "Failed to set rotate_deg");
    setTextAttribute(ncid, NC_GLOBAL, "vtk_origin", commaTriplet(vtk.origin));
    setTextAttribute(ncid, NC_GLOBAL, "vtk_spacing", commaTriplet(vtk.spacing));
    ncCheck(nc_put_att_double(ncid, NC_GLOBAL, "terrain_min_asl_m", NC_DOUBLE, 1, &terrainMinAsl),
            "Failed to set terrain_min_asl_m");
    ncCheck(nc_put_att_double(ncid, NC_GLOBAL, "pedestal_height_removed_m", NC_DOUBLE, 1, &pedestalHeight),
            "Failed to set pedestal_height_removed_m");
    setTextAttribute(ncid, NC_GLOBAL, "vertical_processing",
                     "Kept VTK layers where z_vtk >= pedestal_height_removed_m; "
                     "stored z = z_vtk - pedestal_height_removed_m + terrain_min_asl_m.");
    setTextAttribute(ncid, NC_GLOBAL, "vector_processing",
                     "Horizontal VTK u/v components were interpreted as model rotated x/y components "
                     "and rotated by -rotate_deg into UTM east/north components.");
    setTextAttribute(ncid, NC_GLOBAL, "history", timestamp() + " converted by vtk_avg_to_utm_asl_nc.py");
    ncCheck(nc_enddef(ncid), "Failed to finish NetCDF schema");

    ncCheck(nc_put_var_double(ncid, varX, xLocal.data()), "Failed to write local x");
    ncCheck(nc_put_var_double(ncid, varY, yLocal.data()), "Failed to write local y");
    ncCheck(nc_put_var_float(ncid, varZ, zAsl.data()), "Failed to write ASL z");
    ncCheck(nc_put_var_double(ncid, varEasting, easting.data()), "Failed to write easting");
    ncCheck(nc_put_var_double(ncid, varNorthing, northing.data()), "Failed to write northing");

    constexpr double degreesToRadians = 0.0174532925199432957692;
    const double angle = config.rotateDegrees * degreesToRadians;
    const float cosine = static_cast<float>(std::cos(angle));
    const float sine = static_cast<float>(std::sin(angle));
    const std::size_t planeSize = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const std::array<std::size_t, 3> count{1, static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
    std::vector<float> uModel;
    std::vector<float> vModel;
    std::vector<float> wModel;
    std::vector<float> uUtm(planeSize);
    std::vector<float> vUtm(planeSize);
    const auto startTime = std::chrono::steady_clock::now();
    LegacyVtkPlaneReader reader(vtk);
    for (std::size_t outputZ = 0; outputZ < kept.size(); ++outputZ) {
        const std::size_t sourceZ = kept[outputZ];
        if (wind.u.field == wind.v.field && wind.u.field == wind.w.field) {
            const std::vector<float> tuples = reader.readPointPlane(*wind.u.field, sourceZ);
            uModel.resize(planeSize);
            vModel.resize(planeSize);
            wModel.resize(planeSize);
            for (std::size_t point = 0; point < planeSize; ++point) {
                uModel[point] = tuples[point * wind.u.field->components + wind.u.component];
                vModel[point] = tuples[point * wind.v.field->components + wind.v.component];
                wModel[point] = tuples[point * wind.w.field->components + wind.w.component];
            }
        } else {
            uModel = reader.readPointComponentPlane(*wind.u.field, sourceZ, wind.u.component);
            vModel = reader.readPointComponentPlane(*wind.v.field, sourceZ, wind.v.component);
            wModel = reader.readPointComponentPlane(*wind.w.field, sourceZ, wind.w.component);
        }
        for (std::size_t point = 0; point < planeSize; ++point) {
            uUtm[point] = cosine * uModel[point] + sine * vModel[point];
            vUtm[point] = -sine * uModel[point] + cosine * vModel[point];
        }
        const std::array<std::size_t, 3> start{outputZ, 0, 0};
        ncCheck(nc_put_vara_float(ncid, varU, start.data(), count.data(), uUtm.data()), "Failed to write u");
        ncCheck(nc_put_vara_float(ncid, varV, start.data(), count.data(), vUtm.data()), "Failed to write v");
        ncCheck(nc_put_vara_float(ncid, varW, start.data(), count.data(), wModel.data()), "Failed to write w");
        for (const OutputScalar& scalar : outputScalars) {
            const std::vector<float> values = reader.readPointComponentPlane(*scalar.field, sourceZ, 0);
            ncCheck(nc_put_vara_float(ncid, scalar.variable, start.data(), count.data(), values.data()),
                    "Failed to write " + scalar.name);
        }
        if ((outputZ + 1) % std::max<std::size_t>(1, kept.size() / 10) == 0 || outputZ + 1 == kept.size()) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
            std::ostringstream message;
            message << vtkPath.filename().string() << ": wrote " << outputZ + 1 << '/' << kept.size()
                    << " z layers (" << std::fixed << std::setprecision(1) << seconds << "s elapsed)";
            log(output, message.str());
        }
    }
    file.close();
    log(output, "Wrote NetCDF: " + outputPath.string());
    return outputPath;
}

std::vector<fs::path> discoverCases(const fs::path& root,
                                    const std::optional<std::vector<std::string>>& requested,
                                    std::ostream& output) {
    std::set<std::string> wanted;
    if (requested && !requested->empty()) {
        for (const std::string& name : *requested) wanted.insert(lower(name));
    } else {
        for (const auto name : caseOrder) wanted.emplace(name);
    }
    std::vector<fs::path> result;
    for (const auto name : caseOrder) {
        if (!wanted.contains(std::string(name))) continue;
        const fs::path directory = root / name;
        if (!fs::is_regular_file(directory / "conf.luwpf")) {
            warn(output, "Skip case without conf.luwpf: " + directory.string());
            continue;
        }
        result.push_back(directory);
    }
    if (fs::is_directory(root)) {
        std::vector<fs::path> extras;
        for (const fs::directory_entry& entry : fs::directory_iterator(root)) {
            if (!entry.is_directory()) continue;
            const std::string name = lower(entry.path().filename().string());
            const bool known = std::any_of(caseOrder.begin(), caseOrder.end(),
                                           [&](std::string_view value) { return name == value; });
            if (wanted.contains(name) && !known && fs::is_regular_file(entry.path() / "conf.luwpf")) {
                extras.push_back(entry.path());
            }
        }
        std::sort(extras.begin(), extras.end());
        result.insert(result.end(), extras.begin(), extras.end());
    }
    if (result.empty()) throw std::runtime_error("No case directories found under " + root.string());
    return result;
}

std::size_t selectedCount(std::size_t count, const std::optional<int>& limit) {
    if (!limit || *limit <= 0) return count;
    return std::min(count, static_cast<std::size_t>(*limit));
}

int executeAverage(const AverageArguments& arguments, std::ostream& output) {
    const fs::path root = resolved(arguments.root);
    const fs::path rangePath = arguments.rangeFile ? resolved(*arguments.rangeFile) : root / "Range.txt";
    if (arguments.caseDirectory) {
        const fs::path caseDirectory = resolved(*arguments.caseDirectory);
        const fs::path configPath = findCaseConfig(caseDirectory, arguments.config);
        const CaseConfig config = parseCaseConfig(configPath, output);
        const double terrain = resolveTerrainAsl(
            caseDirectory, config.casename, arguments.terrainMinAsl,
            fs::is_regular_file(rangePath) ? std::optional<fs::path>(rangePath) : std::nullopt, output);
        const std::vector<fs::path> files = discoverVtkFiles(
            caseDirectory, arguments.inputSubdirectory, arguments.inputGlob);
        const std::size_t count = selectedCount(files.size(), arguments.limit);
        const fs::path outputDirectory = caseDirectory / arguments.outputSubdirectory;
        log(output, "Case dir: " + caseDirectory.string());
        log(output, "Config: " + configPath.string());
        log(output, "Input glob: " + (caseDirectory / arguments.inputSubdirectory / arguments.inputGlob).string());
        {
            std::ostringstream message;
            message << std::fixed << std::setprecision(3)
                    << "Pedestal height removed: " << arguments.pedestalHeight << " m";
            log(output, message.str());
        }
        {
            std::ostringstream message;
            message << "Files: " << count << '/' << files.size() << ", terrain_min_asl_m="
                    << std::fixed << std::setprecision(9) << terrain << ", out=" << outputDirectory.string();
            log(output, message.str());
        }
        std::vector<std::pair<fs::path, std::string>> failed;
        for (std::size_t index = 0; index < count; ++index) {
            log(output, '[' + std::to_string(index + 1) + '/' + std::to_string(count) + "] "
                        + files[index].filename().string());
            try {
                (void)convertAverageFile(files[index], config, terrain, outputDirectory,
                                         arguments.pedestalHeight, arguments.compressionLevel,
                                         arguments.overwrite, output);
            } catch (const std::exception& exception) {
                failed.emplace_back(files[index], exception.what());
                warn(output, "Failed " + files[index].string() + ": " + exception.what());
            }
        }
        log(output, std::string(88, '='));
        log(output, "Summary: ok=" + std::to_string(count - failed.size())
                    + ", failed=" + std::to_string(failed.size()));
        for (const auto& [path, message] : failed) warn(output, "FAIL " + path.string() + " | " + message);
        return failed.empty() ? 0 : 2;
    }

    std::vector<std::string> required;
    for (const auto name : caseOrder) required.emplace_back(name);
    const auto aslValues = parseRangeAsl(rangePath, required);
    const std::vector<fs::path> cases = discoverCases(root, arguments.cases, output);
    log(output, "Root: " + root.string());
    log(output, "Range file: " + rangePath.string());
    {
        std::ostringstream message;
        message << std::fixed << std::setprecision(3)
                << "Pedestal height removed: " << arguments.pedestalHeight << " m";
        log(output, message.str());
    }
    std::size_t succeeded = 0;
    std::vector<std::pair<fs::path, std::string>> failed;
    for (const fs::path& caseDirectory : cases) {
        const std::string caseName = lower(caseDirectory.filename().string());
        const double terrain = aslValues.at(caseName);
        const CaseConfig config = parseCaseConfig(caseDirectory / "conf.luwpf", output);
        const std::vector<fs::path> files = discoverVtkFiles(
            caseDirectory, arguments.inputSubdirectory, arguments.inputGlob);
        const std::size_t count = selectedCount(files.size(), arguments.limit);
        const fs::path outputDirectory = caseDirectory / arguments.outputSubdirectory;
        log(output, std::string(88, '='));
        {
            std::ostringstream message;
            message << "Case " << caseName << ": files=" << count << '/' << files.size()
                    << ", terrain_min_asl_m=" << std::fixed << std::setprecision(6) << terrain
                    << ", out=" << outputDirectory.string();
            log(output, message.str());
        }
        for (std::size_t index = 0; index < count; ++index) {
            log(output, '[' + caseName + ' ' + std::to_string(index + 1) + '/'
                        + std::to_string(count) + "] " + files[index].filename().string());
            try {
                (void)convertAverageFile(files[index], config, terrain, outputDirectory,
                                         arguments.pedestalHeight, arguments.compressionLevel,
                                         arguments.overwrite, output);
                ++succeeded;
            } catch (const std::exception& exception) {
                failed.emplace_back(files[index], exception.what());
                warn(output, "Failed " + files[index].string() + ": " + exception.what());
            }
        }
    }
    log(output, std::string(88, '='));
    log(output, "Summary: ok=" + std::to_string(succeeded)
                + ", failed=" + std::to_string(failed.size()));
    for (const auto& [path, message] : failed) warn(output, "FAIL " + path.string() + " | " + message);
    return failed.empty() ? 0 : 2;
}

} // namespace

int run_vtk_avg_to_utm_asl_nc(const std::vector<std::string>& arguments,
                              std::ostream& output,
                              std::ostream& error) {
    try {
        const AverageArguments parsed = parseAverageArguments(arguments);
        if (parsed.help) {
            printAverageUsage(output);
            return 0;
        }
        return executeAverage(parsed, output);
    } catch (const AverageArgumentError& exception) {
        error << "usage: vtk_avg_to_utm_asl_nc [options]\n"
              << "vtk_avg_to_utm_asl_nc: error: " << exception.what() << '\n';
        return 2;
    } catch (const std::exception& exception) {
        error << "[ERROR] " << timestamp() << " | " << exception.what() << '\n';
        return 1;
    }
}

} // namespace luw::cli
