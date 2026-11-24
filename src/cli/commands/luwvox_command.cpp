#include "luw/luwvox_command.hpp"
#include "luw/luwvox_kriging_compat.hpp"
#include "luw/luwvox_opencl.hpp"

#include "luw/deck_document.hpp"
#include "luw/dem_tif_cache.hpp"
#include "luw/preprocess_compute_config.hpp"
#include "luw/progress.hpp"
#include "luw/runtime_paths.hpp"
#include "luw/terrain_voxel_config.hpp"

#include <QString>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <numbers>
#include <optional>
#include <ostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <cpl_conv.h>
#include <gdal_alg.h>
#include <gdal_priv.h>
#include <mapbox/earcut.hpp>
#include <ogr_spatialref.h>
#include <ogrsf_frmts.h>
#endif

namespace luw::cli {

namespace {

namespace fs = std::filesystem;

QString qString(std::string_view value) {
    return QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
}

fs::path pathFromUtf8(const std::string& value) {
#ifdef _WIN32
    return fs::path(qString(value).toStdWString());
#else
    return fs::path(value);
#endif
}

std::string pathToUtf8(const fs::path& value) {
#ifdef _WIN32
    return QString::fromStdWString(value.wstring()).toUtf8().toStdString();
#else
    return value.string();
#endif
}

QString pathToQString(const fs::path& value) {
#ifdef _WIN32
    return QString::fromStdWString(value.wstring());
#else
    return QString::fromUtf8(value.string());
#endif
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string fixed(double value, int precision) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string general(double value) {
    std::array<char, 128> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                      value, std::chars_format::general);
    if (result.ec == std::errc{}) {
        std::string rendered(buffer.data(), result.ptr);
        if (rendered.find_first_of(".eE") == std::string::npos) rendered += ".0";
        return rendered;
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return stream.str();
}

double parseDouble(std::string_view text, std::string_view option) {
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(owned.c_str(), &end);
    if (errno == ERANGE || end == owned.c_str() ||
        end != owned.c_str() + owned.size()) {
        throw std::invalid_argument("argument " + std::string(option) +
                                    ": invalid float value: '" + owned + "'");
    }
    return value;
}

int parseInteger(std::string_view text, std::string_view option) {
    const std::string_view original = text;
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0)
        text.remove_prefix(1U);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0)
        text.remove_suffix(1U);
    std::string normalized;
    normalized.reserve(text.size());
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char character = text[index];
        if (character == '_') {
            const bool betweenDigits = index > 0U && index + 1U < text.size() &&
                std::isdigit(static_cast<unsigned char>(text[index - 1U])) != 0 &&
                std::isdigit(static_cast<unsigned char>(text[index + 1U])) != 0;
            if (!betweenDigits) {
                throw std::invalid_argument(
                    "argument " + std::string(option) +
                    ": invalid int value: '" + std::string(original) + "'");
            }
            continue;
        }
        normalized.push_back(character);
    }
    text = normalized;
    if (!text.empty() && text.front() == '+') text.remove_prefix(1U);
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument("argument " + std::string(option) +
                                    ": invalid int value: '" + std::string(original) + "'");
    }
    return value;
}

void printUsage(std::ostream& stream) {
    stream
        << "usage: luwvox [-h] [--height-field HEIGHT_FIELD] [--min-height MIN_HEIGHT]\n"
           "              [--no-reproject] [--dem-path DEM_PATH]\n"
           "              [--dem-source {auto,shp,pkl}]\n"
           "              [--terrain-resolution TERRAIN_RESOLUTION]\n"
           "              [--max-buildings MAX_BUILDINGS]\n"
           "              [--elevation-scale ELEVATION_SCALE]\n"
           "              [--smooth-sigma SMOOTH_SIGMA] [--idw-power IDW_POWER]\n"
           "              [--idw-neighbors IDW_NEIGHBORS]\n"
           "              [--terrain-approach {idw,kriging}]\n"
           "              [--base-height BASE_HEIGHT]\n"
           "              conf\n";
}

void printHelp(std::ostream& stream) {
    printUsage(stream);
    stream <<
        "\nVoxelization of buildings with terrain-aware base elevation\n"
        "\npositional arguments:\n"
        "  conf                  path to deck file\n"
        "\noptions:\n"
        "  -h, --help            show this help message and exit\n"
        "  --height-field HEIGHT_FIELD\n"
        "                        height field in shapefile, or 'Inferred' (legacy\n"
        "                        'auto') to detect (Height, Elevation, etc.). If\n"
        "                        omitted, deck key terr_voxel_height_field is used\n"
        "                        before falling back to 'Inferred'.\n"
        "  --min-height MIN_HEIGHT\n"
        "                        minimum valid height in meters. Buildings with height\n"
        "                        less than or equal to this will be ignored. If\n"
        "                        omitted, deck key terr_voxel_ignore_under is used\n"
        "                        before falling back to 0.0.\n"
        "  --no-reproject        do not reproject shapefile to UTM, use original CRS\n"
        "  --dem-path DEM_PATH   path to DEM shapefile. If not provided, will look for\n"
        "                        DEM folder in project home\n"
        "  --dem-source {auto,shp,pkl}\n"
        "                        DEM source selection. auto=prefer shapefile/GeoTIFF\n"
        "                        then fallback to PKL. Default: auto\n"
        "  --terrain-resolution TERRAIN_RESOLUTION\n"
        "                        terrain interpolation grid resolution in meters. If\n"
        "                        omitted, deck key terr_voxel_grid_resolution is used\n"
        "                        before falling back to 50.0.\n"
        "  --max-buildings MAX_BUILDINGS\n"
        "                        maximum number of buildings to process (for testing).\n"
        "                        Default: process all buildings\n"
        "  --elevation-scale ELEVATION_SCALE\n"
        "                        scale factor for elevation differences (for\n"
        "                        visualization/testing). Default: 1.0\n"
        "  --smooth-sigma SMOOTH_SIGMA\n"
        "                        Gaussian smoothing strength for terrain interpolation\n"
        "                        (0=no smoothing). If omitted, deck key\n"
        "                        terr_voxel_idw_sigma is used before falling back to\n"
        "                        1.0.\n"
        "  --idw-power IDW_POWER\n"
        "                        IDW power parameter (higher = more weight to closer\n"
        "                        points, 1-3 recommended). If omitted, deck key\n"
        "                        terr_voxel_idw_power is used before falling back to\n"
        "                        2.0.\n"
        "  --idw-neighbors IDW_NEIGHBORS\n"
        "                        Number of nearest neighbors for terrain interpolation.\n"
        "                        If omitted, deck key terr_voxel_idw_neighbors is used\n"
        "                        before falling back to 12.\n"
        "  --terrain-approach {idw,kriging}\n"
        "                        terrain interpolation approach. If omitted, deck key\n"
        "                        terr_voxel_approach is used before falling back to\n"
        "                        idw. The compute backend is selected independently.\n"
        "  --base-height BASE_HEIGHT\n"
        "                        Base layer height in meters (terrain sits on top of\n"
        "                        this). If not set, will read base_height from conf\n"
        "                        (fallback 50.0).\n";
}

constexpr std::array<std::string_view, 14> kLongOptions{
    "--help", "--height-field", "--min-height", "--no-reproject",
    "--dem-path", "--dem-source", "--terrain-resolution",
    "--max-buildings", "--elevation-scale", "--smooth-sigma",
    "--idw-power", "--idw-neighbors", "--terrain-approach", "--base-height",
};

struct LongOptionMatch {
    std::string_view spelling;
    std::string_view suffix;
    std::vector<std::string_view> candidates;
};

LongOptionMatch matchLongOption(const std::string& argument) {
    LongOptionMatch result;
    if (argument.size() <= 2U || argument.rfind("--", 0) != 0)
        return result;
    const std::size_t equals = argument.find('=');
    result.spelling = std::string_view(argument).substr(0, equals);
    if (equals != std::string::npos)
        result.suffix = std::string_view(argument).substr(equals);
    for (const std::string_view option : kLongOptions) {
        if (option == result.spelling) {
            result.candidates.assign(1U, option);
            return result;
        }
    }
    for (const std::string_view option : kLongOptions) {
        if (option.starts_with(result.spelling))
            result.candidates.push_back(option);
    }
    return result;
}

bool argparseNegativeNumber(std::string_view value) {
    if (value.size() < 2U || value.front() != '-') return false;
    value.remove_prefix(1U);
    if (value.empty()) return false;
    const std::size_t decimal = value.find('.');
    if (decimal == std::string_view::npos) {
        return std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isdigit(character) != 0;
        });
    }
    if (value.find('.', decimal + 1U) != std::string_view::npos) return false;
    const std::string_view before = value.substr(0, decimal);
    const std::string_view after = value.substr(decimal + 1U);
    if (after.empty()) return false;
    const auto digits = [](std::string_view part) {
        return std::all_of(part.begin(), part.end(), [](unsigned char character) {
            return std::isdigit(character) != 0;
        });
    };
    return digits(before) && digits(after);
}

struct Options {
    fs::path conf;
    std::optional<std::string> heightField;
    std::optional<double> minimumHeight;
    bool noReproject = false;
    std::optional<fs::path> demPath;
    std::string demSource = "auto";
    std::optional<double> terrainResolution;
    std::optional<int> maximumBuildings;
    double elevationScale = 1.0;
    std::optional<double> smoothSigma;
    std::optional<double> idwPower;
    std::optional<int> idwNeighbors;
    std::optional<std::string> terrainApproach;
    std::optional<double> baseHeight;
};

Options parseOptions(const CommandArguments& arguments) {
    Options options;
    bool hasConf = false;
    bool positionalOnly = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        std::string argument = arguments[index];
        if (!positionalOnly && argument == "--") {
            positionalOnly = true;
            continue;
        }
        if (positionalOnly) {
            if (!hasConf) {
                options.conf = pathFromUtf8(argument);
                hasConf = true;
                continue;
            }
            throw std::invalid_argument("unrecognized arguments: " + argument);
        }
        const LongOptionMatch longOption = matchLongOption(argument);
        if (longOption.candidates.size() > 1U) {
            std::ostringstream message;
            message << "ambiguous option: " << argument << " could match ";
            for (std::size_t candidate = 0;
                 candidate < longOption.candidates.size(); ++candidate) {
                if (candidate != 0U) message << ", ";
                message << longOption.candidates[candidate];
            }
            throw std::invalid_argument(message.str());
        }
        if (longOption.candidates.size() == 1U) {
            const std::string_view canonical = longOption.candidates.front();
            if (!longOption.suffix.empty() &&
                (canonical == "--help" || canonical == "--no-reproject")) {
                const std::string_view explicitValue = longOption.suffix.substr(1U);
                throw std::invalid_argument(
                    std::string("argument ") +
                    (canonical == "--help" ? "-h/--help" : "--no-reproject") +
                    ": ignored explicit argument '" +
                    std::string(explicitValue) + "'");
            }
            argument = std::string(canonical) + std::string(longOption.suffix);
        }
        const auto valueAfter = [&](std::string_view name) -> std::string {
            const std::string prefix = std::string(name) + '=';
            if (argument.rfind(prefix, 0) == 0) return argument.substr(prefix.size());
            if (argument != name || index + 1U >= arguments.size()) {
                throw std::invalid_argument("argument " + std::string(name) +
                                            ": expected one argument");
            }
            const std::string& candidate = arguments[index + 1U];
            if (!candidate.empty() && candidate.front() == '-' &&
                !argparseNegativeNumber(candidate)) {
                throw std::invalid_argument("argument " + std::string(name) +
                                            ": expected one argument");
            }
            ++index;
            return candidate;
        };
        const auto isOption = [&](std::string_view name) {
            return argument == name || argument.rfind(std::string(name) + '=', 0) == 0;
        };

        if (argument == "-h" || argument == "--help") continue;
        if (argument == "--no-reproject") {
            options.noReproject = true;
        } else if (isOption("--height-field")) {
            options.heightField = valueAfter("--height-field");
        } else if (isOption("--min-height")) {
            options.minimumHeight = parseDouble(valueAfter("--min-height"), "--min-height");
        } else if (isOption("--dem-path")) {
            options.demPath = pathFromUtf8(valueAfter("--dem-path"));
        } else if (isOption("--dem-source")) {
            options.demSource = valueAfter("--dem-source");
            if (options.demSource != "auto" && options.demSource != "shp" &&
                options.demSource != "pkl") {
                throw std::invalid_argument(
                    "argument --dem-source: invalid choice: '" + options.demSource +
                    "' (choose from 'auto', 'shp', 'pkl')");
            }
        } else if (isOption("--terrain-resolution")) {
            options.terrainResolution =
                parseDouble(valueAfter("--terrain-resolution"), "--terrain-resolution");
        } else if (isOption("--max-buildings")) {
            options.maximumBuildings =
                parseInteger(valueAfter("--max-buildings"), "--max-buildings");
        } else if (isOption("--elevation-scale")) {
            options.elevationScale =
                parseDouble(valueAfter("--elevation-scale"), "--elevation-scale");
        } else if (isOption("--smooth-sigma")) {
            options.smoothSigma =
                parseDouble(valueAfter("--smooth-sigma"), "--smooth-sigma");
        } else if (isOption("--idw-power")) {
            options.idwPower = parseDouble(valueAfter("--idw-power"), "--idw-power");
        } else if (isOption("--idw-neighbors")) {
            options.idwNeighbors =
                parseInteger(valueAfter("--idw-neighbors"), "--idw-neighbors");
        } else if (isOption("--terrain-approach")) {
            options.terrainApproach = valueAfter("--terrain-approach");
            const std::string& choice = *options.terrainApproach;
            if (choice != "idw" && choice != "kriging") {
                throw std::invalid_argument(
                    "argument --terrain-approach: invalid choice: '" + choice +
                    "' (choose from 'idw', 'kriging')");
            }
        } else if (isOption("--base-height")) {
            options.baseHeight =
                parseDouble(valueAfter("--base-height"), "--base-height");
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::invalid_argument("unrecognized arguments: " + argument);
        } else if (!hasConf) {
            options.conf = pathFromUtf8(argument);
            hasConf = true;
        } else {
            throw std::invalid_argument("unrecognized arguments: " + argument);
        }
    }
    if (!hasConf) throw std::invalid_argument("the following arguments are required: conf");
    return options;
}

bool hasHelpOption(const CommandArguments& arguments) {
    for (const std::string& argument : arguments) {
        if (argument == "--") return false;
        if (argument == "-h") return true;
        const LongOptionMatch option = matchLongOption(argument);
        if (option.candidates.size() == 1U && option.candidates.front() == "--help" &&
            option.suffix.empty())
            return true;
    }
    return false;
}

#if LUW_HAS_GDAL

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Triangle {
    Vec3 a;
    Vec3 b;
    Vec3 c;
};

struct Polygon2D {
    std::vector<Vec2> exterior;
    std::vector<std::vector<Vec2>> holes;
};

struct DemData {
    std::vector<Vec2> points;
    std::vector<double> elevations;
    bool fromPkl = false;
    std::optional<double> pklBaseHeight;
    std::optional<double> xMin;
    std::optional<double> yMin;
    std::optional<double> xMax;
    std::optional<double> yMax;
    std::optional<double> rotateDegrees;
    std::optional<Vec2> pivot;
};

struct Grid {
    std::size_t nx = 0;
    std::size_t ny = 0;
    double xMin = 0.0;
    double xMax = 0.0;
    double yMin = 0.0;
    double yMax = 0.0;
    std::vector<double> z;

    double x(std::size_t column) const {
        return nx <= 1U ? xMin
                        : xMin + (xMax - xMin) * static_cast<double>(column) /
                                     static_cast<double>(nx - 1U);
    }
    double y(std::size_t row) const {
        return ny <= 1U ? yMin
                        : yMin + (yMax - yMin) * static_cast<double>(row) /
                                     static_cast<double>(ny - 1U);
    }
    double& at(std::size_t row, std::size_t column) { return z[row * nx + column]; }
    double at(std::size_t row, std::size_t column) const { return z[row * nx + column]; }
};

struct DatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) GDALClose(dataset);
    }
};
struct FeatureCloser {
    void operator()(OGRFeature* feature) const noexcept {
        if (feature != nullptr) OGRFeature::DestroyFeature(feature);
    }
};
struct GeometryCloser {
    void operator()(OGRGeometry* geometry) const noexcept {
        if (geometry != nullptr) OGRGeometryFactory::destroyGeometry(geometry);
    }
};
struct TransformCloser {
    void operator()(OGRCoordinateTransformation* transform) const noexcept {
        if (transform != nullptr) OCTDestroyCoordinateTransformation(transform);
    }
};
using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;
using FeaturePtr = std::unique_ptr<OGRFeature, FeatureCloser>;
using GeometryPtr = std::unique_ptr<OGRGeometry, GeometryCloser>;
using TransformPtr = std::unique_ptr<OGRCoordinateTransformation, TransformCloser>;

std::unique_ptr<OGRSpatialReference> epsgReference(int epsg) {
    auto reference = std::make_unique<OGRSpatialReference>();
    if (reference->importFromEPSG(epsg) != OGRERR_NONE) {
        throw std::runtime_error("Cannot construct EPSG:" + std::to_string(epsg));
    }
    reference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return reference;
}

std::unique_ptr<OGRSpatialReference> cloneReference(const OGRSpatialReference* source) {
    if (source == nullptr) return {};
    std::unique_ptr<OGRSpatialReference> result(source->Clone());
    result->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return result;
}

TransformPtr makeTransform(OGRSpatialReference* source, OGRSpatialReference* target,
                           std::string_view context) {
    TransformPtr result(OGRCreateCoordinateTransformation(source, target));
    if (!result) throw std::runtime_error("Cannot create coordinate transformation for " +
                                          std::string(context));
    return result;
}

std::string referenceText(OGRSpatialReference& reference) {
    std::unique_ptr<OGRSpatialReference> copy(reference.Clone());
    copy->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    (void)copy->AutoIdentifyEPSG();
    const char* authority = copy->GetAuthorityName(nullptr);
    const char* code = copy->GetAuthorityCode(nullptr);
    if (authority != nullptr && code != nullptr && lowerAscii(authority) == "epsg") {
        return std::string("EPSG:") + code;
    }
    char* wkt = nullptr;
    if (copy->exportToWkt(&wkt) == OGRERR_NONE && wkt != nullptr) {
        std::string value(wkt);
        CPLFree(wkt);
        return value;
    }
    CPLFree(wkt);
    return "None";
}

int utmZone(double longitude) {
    return std::clamp(static_cast<int>(std::floor((longitude + 180.0) / 6.0)) + 1,
                      1, 60);
}

int utmEpsg(double longitude, double latitude) {
    return (latitude >= 0.0 ? 32600 : 32700) + utmZone(longitude);
}

// Restricted pickle VM for the exact NumPy ndarray/dict subset written by
// 1_buildBC.py (protocol 3) and recent NumPy (protocol 4).  It intentionally
// does not execute globals or constructors.
struct PickleNode;
using PicklePtr = std::shared_ptr<PickleNode>;

struct PickleNode {
    enum class Kind { Null, Marker, Boolean, Integer, Floating, String, Bytes,
                      Tuple, Dictionary, Global, Array };
    Kind kind = Kind::Null;
    bool boolean = false;
    std::int64_t integer = 0;
    double floating = 0.0;
    std::string text;
    std::vector<std::uint8_t> bytes;
    std::vector<PicklePtr> sequence;
    std::map<std::string, PicklePtr> dictionary;
    std::vector<std::size_t> shape;
    std::string dtype = "f8";
    bool fortran = false;
};

PicklePtr pickleNode(PickleNode::Kind kind = PickleNode::Kind::Null) {
    auto node = std::make_shared<PickleNode>();
    node->kind = kind;
    return node;
}

class SafeNumpyPickleReader final {
public:
    explicit SafeNumpyPickleReader(std::vector<std::uint8_t> bytes)
        : bytes_(std::move(bytes)) {}

    PicklePtr read() {
        while (position_ < bytes_.size()) {
            const std::uint8_t opcode = byte();
            switch (opcode) {
            case 0x80: (void)byte(); break;                         // PROTO
            case 0x95: position_ += checkedLength(8U); break;       // FRAME
            case '}': push(pickleNode(PickleNode::Kind::Dictionary)); break;
            case '(': push(pickleNode(PickleNode::Kind::Marker)); break;
            case ')': push(pickleNode(PickleNode::Kind::Tuple)); break;
            case 'N': push(pickleNode()); break;
            case 0x88: { auto n = pickleNode(PickleNode::Kind::Boolean); n->boolean = true; push(n); break; }
            case 0x89: { auto n = pickleNode(PickleNode::Kind::Boolean); n->boolean = false; push(n); break; }
            case 'K': pushInteger(byte()); break;
            case 'M': pushInteger(unsignedLittle(2U)); break;
            case 'J': pushInteger(signedLittle(4U)); break;
            case 0x8a: readLong1(); break;
            case 'G': readBigEndianDouble(); break;
            case 'X': readString(unsignedLittle(4U)); break;
            case 0x8c: readString(byte()); break;
            case 'B': readBytes(unsignedLittle(4U)); break;
            case 'C': readBytes(byte()); break;
            case 0x8e: readBytes(unsignedLittle(8U)); break;
            case 'c': readGlobal(); break;
            case 0x93: stackGlobal(); break;
            case 0x85: makeTuple(1U); break;
            case 0x86: makeTuple(2U); break;
            case 0x87: makeTuple(3U); break;
            case 't': makeMarkedTuple(); break;
            case 'R': reduce(); break;
            case 'b': build(); break;
            case 'q': memo_[byte()] = top(); break;
            case 'r': memo_[unsignedLittle(4U)] = top(); break;
            case 0x94: memo_[nextMemo_++] = top(); break;
            case 'h': pushMemo(byte()); break;
            case 'j': pushMemo(unsignedLittle(4U)); break;
            case 'u': setItems(); break;
            case 's': setItem(); break;
            case '.':
                if (stack_.empty()) throw std::runtime_error("Empty pickle result");
                return stack_.back();
            default:
                throw std::runtime_error("Unsupported pickle opcode 0x" + hex(opcode));
            }
        }
        throw std::runtime_error("Pickle ended before STOP opcode");
    }

private:
    std::uint8_t byte() {
        if (position_ >= bytes_.size()) throw std::runtime_error("Truncated pickle");
        return bytes_[position_++];
    }

    std::size_t checkedLength(std::size_t count) {
        if (count > bytes_.size() - position_) throw std::runtime_error("Truncated pickle payload");
        return count;
    }

    std::uint64_t unsignedLittle(std::size_t count) {
        if (count > 8U) throw std::runtime_error("Oversized pickle integer");
        checkedLength(count);
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < count; ++i) value |= std::uint64_t(byte()) << (i * 8U);
        return value;
    }

    std::int64_t signedLittle(std::size_t count) {
        const std::uint64_t raw = unsignedLittle(count);
        if (count < 8U && (raw & (std::uint64_t(1) << (count * 8U - 1U))) != 0U) {
            return static_cast<std::int64_t>(raw | (~std::uint64_t(0) << (count * 8U)));
        }
        return static_cast<std::int64_t>(raw);
    }

    void push(const PicklePtr& node) { stack_.push_back(node); }
    PicklePtr pop() {
        if (stack_.empty()) throw std::runtime_error("Malformed pickle stack");
        PicklePtr value = stack_.back();
        stack_.pop_back();
        return value;
    }
    PicklePtr top() const {
        if (stack_.empty()) throw std::runtime_error("Malformed pickle stack");
        return stack_.back();
    }

    static std::string hex(std::uint8_t value) {
        const char* digits = "0123456789abcdef";
        std::string result(2, '0');
        result[0] = digits[(value >> 4U) & 0xFU];
        result[1] = digits[value & 0xFU];
        return result;
    }

    void pushInteger(std::int64_t value) {
        auto node = pickleNode(PickleNode::Kind::Integer);
        node->integer = value;
        push(node);
    }

    void readLong1() {
        const std::size_t count = byte();
        pushInteger(signedLittle(count));
    }

    void readBigEndianDouble() {
        checkedLength(8U);
        std::uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) bits = (bits << 8U) | byte();
        auto node = pickleNode(PickleNode::Kind::Floating);
        node->floating = std::bit_cast<double>(bits);
        push(node);
    }

    void readString(std::uint64_t length) {
        if (length > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("Pickle string too large");
        const std::size_t count = static_cast<std::size_t>(length);
        checkedLength(count);
        auto node = pickleNode(PickleNode::Kind::String);
        node->text.assign(reinterpret_cast<const char*>(bytes_.data() + position_), count);
        position_ += count;
        push(node);
    }

    void readBytes(std::uint64_t length) {
        if (length > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("Pickle bytes too large");
        const std::size_t count = static_cast<std::size_t>(length);
        checkedLength(count);
        auto node = pickleNode(PickleNode::Kind::Bytes);
        node->bytes.assign(bytes_.begin() + static_cast<std::ptrdiff_t>(position_),
                           bytes_.begin() + static_cast<std::ptrdiff_t>(position_ + count));
        position_ += count;
        push(node);
    }

    std::string line() {
        const std::size_t begin = position_;
        while (position_ < bytes_.size() && bytes_[position_] != '\n') ++position_;
        if (position_ >= bytes_.size()) throw std::runtime_error("Truncated pickle GLOBAL");
        std::string result(reinterpret_cast<const char*>(bytes_.data() + begin), position_ - begin);
        ++position_;
        return result;
    }

    void readGlobal() {
        auto node = pickleNode(PickleNode::Kind::Global);
        const std::string module = line();
        const std::string name = line();
        node->text = module + " " + name;
        push(node);
    }

    void stackGlobal() {
        const PicklePtr name = pop();
        const PicklePtr module = pop();
        if (name->kind != PickleNode::Kind::String || module->kind != PickleNode::Kind::String) {
            throw std::runtime_error("Malformed STACK_GLOBAL");
        }
        auto node = pickleNode(PickleNode::Kind::Global);
        node->text = module->text + " " + name->text;
        push(node);
    }

    void makeTuple(std::size_t count) {
        if (stack_.size() < count) throw std::runtime_error("Malformed pickle tuple");
        auto tuple = pickleNode(PickleNode::Kind::Tuple);
        tuple->sequence.assign(stack_.end() - static_cast<std::ptrdiff_t>(count), stack_.end());
        stack_.resize(stack_.size() - count);
        push(tuple);
    }

    std::size_t markerIndex() const {
        for (std::size_t i = stack_.size(); i > 0U; --i) {
            if (stack_[i - 1U]->kind == PickleNode::Kind::Marker) return i - 1U;
        }
        throw std::runtime_error("Pickle MARK not found");
    }

    void makeMarkedTuple() {
        const std::size_t marker = markerIndex();
        auto tuple = pickleNode(PickleNode::Kind::Tuple);
        tuple->sequence.assign(stack_.begin() + static_cast<std::ptrdiff_t>(marker + 1U), stack_.end());
        stack_.resize(marker);
        push(tuple);
    }

    void reduce() {
        const PicklePtr args = pop();
        const PicklePtr callable = pop();
        if (callable->kind != PickleNode::Kind::Global || args->kind != PickleNode::Kind::Tuple) {
            throw std::runtime_error("Malformed pickle REDUCE");
        }
        if (callable->text.find("_reconstruct") != std::string::npos) {
            push(pickleNode(PickleNode::Kind::Array));
            return;
        }
        if (callable->text.find(" dtype") != std::string::npos) {
            auto dtype = pickleNode(PickleNode::Kind::Global);
            dtype->text = "dtype";
            if (!args->sequence.empty() && args->sequence.front()->kind == PickleNode::Kind::String) {
                dtype->text += ":" + args->sequence.front()->text;
            }
            push(dtype);
            return;
        }
        throw std::runtime_error("Disallowed pickle callable: " + callable->text);
    }

    static std::optional<std::int64_t> integerValue(const PicklePtr& node) {
        if (node && node->kind == PickleNode::Kind::Integer) return node->integer;
        return std::nullopt;
    }

    void build() {
        const PicklePtr state = pop();
        const PicklePtr instance = top();
        if (instance->kind == PickleNode::Kind::Global && instance->text.rfind("dtype", 0) == 0) {
            if (state->kind == PickleNode::Kind::Tuple && state->sequence.size() > 1U &&
                state->sequence[1U]->kind == PickleNode::Kind::String) {
                const std::string base = instance->text.substr(instance->text.find(':') + 1U);
                instance->text = "dtype:" + state->sequence[1U]->text + base;
            }
            return;
        }
        if (instance->kind != PickleNode::Kind::Array || state->kind != PickleNode::Kind::Tuple ||
            state->sequence.size() < 5U) {
            throw std::runtime_error("Unsupported pickle BUILD state");
        }
        const PicklePtr shape = state->sequence[1U];
        const PicklePtr dtype = state->sequence[2U];
        const PicklePtr fortran = state->sequence[3U];
        const PicklePtr data = state->sequence[4U];
        if (shape->kind != PickleNode::Kind::Tuple || data->kind != PickleNode::Kind::Bytes) {
            throw std::runtime_error("Malformed NumPy ndarray pickle state");
        }
        instance->shape.clear();
        for (const PicklePtr& extent : shape->sequence) {
            const auto value = integerValue(extent);
            if (!value || *value < 0) throw std::runtime_error("Invalid NumPy ndarray shape");
            instance->shape.push_back(static_cast<std::size_t>(*value));
        }
        if (dtype->kind == PickleNode::Kind::Global && dtype->text.rfind("dtype:", 0) == 0) {
            instance->dtype = dtype->text.substr(6U);
        }
        instance->fortran = fortran->kind == PickleNode::Kind::Boolean && fortran->boolean;
        instance->bytes = data->bytes;
    }

    void pushMemo(std::uint64_t index) {
        const auto found = memo_.find(index);
        if (found == memo_.end()) throw std::runtime_error("Invalid pickle memo reference");
        push(found->second);
    }

    void setItems() {
        const std::size_t marker = markerIndex();
        if (marker == 0U || (stack_.size() - marker - 1U) % 2U != 0U) {
            throw std::runtime_error("Malformed SETITEMS");
        }
        const PicklePtr dict = stack_[marker - 1U];
        if (dict->kind != PickleNode::Kind::Dictionary) throw std::runtime_error("SETITEMS target is not dict");
        for (std::size_t i = marker + 1U; i < stack_.size(); i += 2U) {
            if (stack_[i]->kind != PickleNode::Kind::String) throw std::runtime_error("Non-string DEM pickle key");
            dict->dictionary[stack_[i]->text] = stack_[i + 1U];
        }
        stack_.resize(marker);
    }

    void setItem() {
        const PicklePtr value = pop();
        const PicklePtr key = pop();
        const PicklePtr dict = top();
        if (dict->kind != PickleNode::Kind::Dictionary || key->kind != PickleNode::Kind::String) {
            throw std::runtime_error("Malformed SETITEM");
        }
        dict->dictionary[key->text] = value;
    }

    std::vector<std::uint8_t> bytes_;
    std::size_t position_ = 0;
    std::vector<PicklePtr> stack_;
    std::unordered_map<std::uint64_t, PicklePtr> memo_;
    std::uint64_t nextMemo_ = 0;
};

std::optional<double> pickleDouble(const PicklePtr& node) {
    if (!node) return std::nullopt;
    if (node->kind == PickleNode::Kind::Floating) return node->floating;
    if (node->kind == PickleNode::Kind::Integer) return static_cast<double>(node->integer);
    return std::nullopt;
}

std::vector<double> pickleArrayDoubles(const PicklePtr& node) {
    if (!node || node->kind != PickleNode::Kind::Array) throw std::runtime_error("DEM pickle value is not ndarray");
    std::size_t count = 1U;
    for (const std::size_t extent : node->shape) {
        if (extent != 0U && count > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::runtime_error("DEM ndarray is too large");
        }
        count *= extent;
    }
    const bool isF8 = node->dtype.find("f8") != std::string::npos;
    const bool isF4 = node->dtype.find("f4") != std::string::npos;
    const std::size_t width = isF8 ? 8U : (isF4 ? 4U : 0U);
    if (width == 0U || node->bytes.size() != count * width) {
        throw std::runtime_error("Unsupported or inconsistent DEM ndarray dtype: " + node->dtype);
    }
    if (node->fortran) throw std::runtime_error("Fortran-ordered DEM ndarray is not supported");
    const bool bigEndian = !node->dtype.empty() && node->dtype.front() == '>';
    std::vector<double> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (isF8) {
            std::uint64_t bits = 0;
            if (bigEndian) {
                for (std::size_t j = 0; j < 8U; ++j) bits = (bits << 8U) | node->bytes[i * 8U + j];
            } else {
                for (std::size_t j = 0; j < 8U; ++j) bits |= std::uint64_t(node->bytes[i * 8U + j]) << (j * 8U);
            }
            result[i] = std::bit_cast<double>(bits);
        } else {
            std::uint32_t bits = 0;
            if (bigEndian) {
                for (std::size_t j = 0; j < 4U; ++j) bits = (bits << 8U) | node->bytes[i * 4U + j];
            } else {
                for (std::size_t j = 0; j < 4U; ++j) bits |= std::uint32_t(node->bytes[i * 4U + j]) << (j * 8U);
            }
            result[i] = static_cast<double>(std::bit_cast<float>(bits));
        }
    }
    return result;
}

DemData loadDemPickle(const fs::path& path, std::ostream& output) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open DEM pkl: " + pathToUtf8(path));
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    if (length < 0) throw std::runtime_error("Cannot determine DEM pkl size");
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    input.read(reinterpret_cast<char*>(bytes.data()), length);
    if (!input) throw std::runtime_error("Cannot read DEM pkl");

    const PicklePtr root = SafeNumpyPickleReader(std::move(bytes)).read();
    if (!root || root->kind != PickleNode::Kind::Dictionary) throw std::runtime_error("DEM pkl root is not a dict");
    const auto required = [&](std::string_view key) -> PicklePtr {
        const auto found = root->dictionary.find(std::string(key));
        if (found == root->dictionary.end()) throw std::runtime_error("DEM pkl missing key '" + std::string(key) + "'");
        return found->second;
    };
    const PicklePtr demNode = required("dem_grid");
    const std::vector<double> demGrid = pickleArrayDoubles(demNode);
    const std::vector<double> xGrid = pickleArrayDoubles(required("x_grid"));
    const std::vector<double> yGrid = pickleArrayDoubles(required("y_grid"));
    if (demNode->shape.size() != 2U || demNode->shape[0] != yGrid.size() ||
        demNode->shape[1] != xGrid.size()) {
        throw std::runtime_error("DEM pkl grid dimensions are inconsistent");
    }
    const double xMin = pickleDouble(required("x_min")).value_or(0.0);
    const double yMin = pickleDouble(required("y_min")).value_or(0.0);

    DemData result;
    result.fromPkl = true;
    result.points.reserve(demGrid.size());
    result.elevations = demGrid;
    for (std::size_t row = 0; row < yGrid.size(); ++row) {
        for (std::size_t column = 0; column < xGrid.size(); ++column) {
            result.points.push_back({xMin + xGrid[column], yMin + yGrid[row]});
        }
    }
    const auto scalar = [&](std::string_view key) -> std::optional<double> {
        const auto found = root->dictionary.find(std::string(key));
        return found == root->dictionary.end() ? std::nullopt : pickleDouble(found->second);
    };
    result.pklBaseHeight = scalar("base_height");
    result.xMin = scalar("x_min");
    result.yMin = scalar("y_min");
    // The Python consumer derives the upper bounds from the coordinate arrays
    // even when legacy caches also carry x_max/y_max scalar fields.
    if (result.xMin && !xGrid.empty())
        result.xMax = *result.xMin + *std::max_element(xGrid.begin(), xGrid.end());
    if (result.yMin && !yGrid.empty())
        result.yMax = *result.yMin + *std::max_element(yGrid.begin(), yGrid.end());
    result.rotateDegrees = scalar("rotate_deg");
    const auto pivot = root->dictionary.find("pivot_xy");
    if (pivot != root->dictionary.end() && pivot->second->kind == PickleNode::Kind::Tuple &&
        pivot->second->sequence.size() == 2U) {
        const auto x = pickleDouble(pivot->second->sequence[0]);
        const auto y = pickleDouble(pivot->second->sequence[1]);
        if (x && y) result.pivot = Vec2{*x, *y};
    }

    const auto finite = [](double value) { return std::isfinite(value); };
    auto minimum = std::numeric_limits<double>::infinity();
    auto maximum = -std::numeric_limits<double>::infinity();
    for (const double value : result.elevations) {
        if (finite(value)) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    if (!std::isfinite(minimum)) throw std::runtime_error("DEM pkl contains no finite elevations");
    output << "[INFO] Loaded DEM data from pkl: " << pathToUtf8(path) << '\n'
           << "[INFO] DEM grid shape: (" << yGrid.size() << ", " << xGrid.size() << ")\n"
           << "[INFO] DEM elevation difference range: " << fixed(minimum, 2) << "m to "
           << fixed(maximum, 2) << "m\n";
    if (std::abs(minimum) > 1.0e-6) {
        for (double& value : result.elevations) if (finite(value)) value -= minimum;
        output << "[INFO] Adjusted PKL DEM to datum (lowest point = 0). Datum offset applied: "
               << fixed(minimum, 2) << "m\n";
    }
    output << "[INFO] Converted DEM grid to " << result.elevations.size()
           << " points in absolute UTM coordinates\n";
    return result;
}

class KdTree2D final {
public:
    explicit KdTree2D(const std::vector<Vec2>& points) : points_(points) {
        order_.resize(points.size());
        std::iota(order_.begin(), order_.end(), std::size_t(0));
        nodes_.reserve(points.size());
        root_ = build(0U, order_.size(), 0);
    }

    struct Neighbor {
        double distance = 0.0;
        std::size_t index = 0;
    };

    std::vector<Neighbor> nearest(const Vec2& point, std::size_t count) const {
        count = std::min(count, points_.size());
        using Candidate = std::pair<double, std::size_t>;
        std::priority_queue<Candidate> heap;
        query(root_, point, count, heap);
        std::vector<Neighbor> result(heap.size());
        for (std::size_t i = heap.size(); i > 0U; --i) {
            const Candidate candidate = heap.top();
            heap.pop();
            result[i - 1U] = {std::sqrt(candidate.first), candidate.second};
        }
        return result;
    }

private:
    struct Node {
        std::size_t point = 0;
        int axis = 0;
        int left = -1;
        int right = -1;
    };

    int build(std::size_t begin, std::size_t end, int depth) {
        if (begin >= end) return -1;
        const int axis = depth & 1;
        const std::size_t middle = begin + (end - begin) / 2U;
        std::nth_element(order_.begin() + static_cast<std::ptrdiff_t>(begin),
                         order_.begin() + static_cast<std::ptrdiff_t>(middle),
                         order_.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](std::size_t left, std::size_t right) {
                             const double lv = axis == 0 ? points_[left].x : points_[left].y;
                             const double rv = axis == 0 ? points_[right].x : points_[right].y;
                             return lv == rv ? left < right : lv < rv;
                         });
        const int nodeIndex = static_cast<int>(nodes_.size());
        nodes_.push_back({order_[middle], axis, -1, -1});
        const int left = build(begin, middle, depth + 1);
        const int right = build(middle + 1U, end, depth + 1);
        nodes_[static_cast<std::size_t>(nodeIndex)].left = left;
        nodes_[static_cast<std::size_t>(nodeIndex)].right = right;
        return nodeIndex;
    }

    void query(int nodeIndex, const Vec2& point, std::size_t count,
               std::priority_queue<std::pair<double, std::size_t>>& heap) const {
        if (nodeIndex < 0 || count == 0U) return;
        const Node& node = nodes_[static_cast<std::size_t>(nodeIndex)];
        const Vec2& sample = points_[node.point];
        const double dx = point.x - sample.x;
        const double dy = point.y - sample.y;
        const double squared = dx * dx + dy * dy;
        if (heap.size() < count) {
            heap.emplace(squared, node.point);
        } else if (squared < heap.top().first ||
                   (squared == heap.top().first && node.point < heap.top().second)) {
            heap.pop();
            heap.emplace(squared, node.point);
        }
        const double split = node.axis == 0 ? dx : dy;
        const int nearChild = split <= 0.0 ? node.left : node.right;
        const int farChild = split <= 0.0 ? node.right : node.left;
        query(nearChild, point, count, heap);
        const double threshold = heap.size() < count
            ? std::numeric_limits<double>::infinity()
            : heap.top().first;
        if (split * split <= threshold) query(farChild, point, count, heap);
    }

    const std::vector<Vec2>& points_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

double idwPrediction(const std::vector<KdTree2D::Neighbor>& neighbors,
                     const std::vector<double>& elevations, double power) {
    if (neighbors.empty()) return 0.0;
    double minimumDistance = std::numeric_limits<double>::infinity();
    for (const auto& neighbor : neighbors) {
        minimumDistance =
            std::min(minimumDistance, std::max(neighbor.distance, 1.0e-10));
    }
    double weighted = 0.0;
    double sum = 0.0;
    for (const auto& neighbor : neighbors) {
        const double distance = std::max(neighbor.distance, 1.0e-10);
        const double weight = std::pow(minimumDistance / distance, power);
        weighted += weight * elevations[neighbor.index];
        sum += weight;
    }
    return sum > 0.0 ? weighted / sum : 0.0;
}

double exponentialVariogram(double distance, double range, double partialSill,
                            double nugget) {
    const double gamma = partialSill * (1.0 - std::exp(-std::max(distance, 0.0) / range));
    return gamma + (distance > 0.0 ? nugget : 0.0);
}

bool solveDense(std::vector<double>& matrix, std::vector<double>& rhs,
                std::size_t size) {
    for (std::size_t column = 0; column < size; ++column) {
        std::size_t pivot = column;
        double magnitude = std::abs(matrix[column * size + column]);
        for (std::size_t row = column + 1U; row < size; ++row) {
            const double candidate = std::abs(matrix[row * size + column]);
            if (candidate > magnitude) {
                pivot = row;
                magnitude = candidate;
            }
        }
        if (!std::isfinite(magnitude) || magnitude <= 1.0e-14) return false;
        if (pivot != column) {
            for (std::size_t j = column; j < size; ++j) {
                std::swap(matrix[column * size + j], matrix[pivot * size + j]);
            }
            std::swap(rhs[column], rhs[pivot]);
        }
        const double diagonal = matrix[column * size + column];
        for (std::size_t row = column + 1U; row < size; ++row) {
            const double factor = matrix[row * size + column] / diagonal;
            matrix[row * size + column] = 0.0;
            for (std::size_t j = column + 1U; j < size; ++j) {
                matrix[row * size + j] -= factor * matrix[column * size + j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for (std::size_t ii = size; ii > 0U; --ii) {
        const std::size_t row = ii - 1U;
        double value = rhs[row];
        for (std::size_t column = row + 1U; column < size; ++column) {
            value -= matrix[row * size + column] * rhs[column];
        }
        const double diagonal = matrix[row * size + row];
        if (!std::isfinite(diagonal) || std::abs(diagonal) <= 1.0e-14) return false;
        rhs[row] = value / diagonal;
    }
    return true;
}

struct KrigingModel {
    double range = 1.0;
    double partialSill = 1.0;
    double nugget = 0.02;
};

KrigingModel estimateKrigingModel(const std::vector<Vec2>& points,
                                  const std::vector<double>& elevations,
                                  std::vector<double>& positiveDistances) {
    const double mean = std::accumulate(elevations.begin(), elevations.end(), 0.0) /
                        static_cast<double>(std::max<std::size_t>(elevations.size(), 1U));
    double variance = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const double elevation : elevations) {
        const double delta = elevation - mean;
        variance += delta * delta;
        minimum = std::min(minimum, elevation);
        maximum = std::max(maximum, elevation);
    }
    variance /= static_cast<double>(std::max<std::size_t>(elevations.size(), 1U));
    if (!std::isfinite(variance) || variance <= 1.0e-12) {
        variance = std::max(std::pow((maximum - minimum) * 0.25, 2.0), 1.0);
    }
    double range = 1.0;
    if (!positiveDistances.empty()) {
        const std::size_t middle = positiveDistances.size() / 2U;
        std::nth_element(positiveDistances.begin(),
                         positiveDistances.begin() + static_cast<std::ptrdiff_t>(middle),
                         positiveDistances.end());
        double median = positiveDistances[middle];
        if (positiveDistances.size() % 2U == 0U) {
            const auto lower = std::max_element(positiveDistances.begin(),
                                                positiveDistances.begin() + static_cast<std::ptrdiff_t>(middle));
            median = (*lower + median) * 0.5;
        }
        range = std::max(median * 3.0, 1.0);
    } else if (!points.empty()) {
        double minX = points.front().x, maxX = points.front().x;
        double minY = points.front().y, maxY = points.front().y;
        for (const Vec2& point : points) {
            minX = std::min(minX, point.x); maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y); maxY = std::max(maxY, point.y);
        }
        range = std::max(std::hypot(maxX - minX, maxY - minY) / 6.0, 1.0);
    }
    const double nugget = std::max(variance * 0.02, 1.0e-9);
    return {range, std::max(variance - nugget, 1.0e-9), nugget};
}

double krigingPrediction(const Vec2& query,
                         const std::vector<KdTree2D::Neighbor>& neighbors,
                         const std::vector<Vec2>& points,
                         const std::vector<double>& elevations,
                         const KrigingModel& model, double fallbackPower) {
    if (neighbors.empty()) return 0.0;
    for (const auto& neighbor : neighbors) {
        if (neighbor.distance <= 1.0e-10) return elevations[neighbor.index];
    }
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    const double firstElevation = elevations[neighbors.front().index];
    std::vector<double> localElevations;
    localElevations.reserve(neighbors.size());
    for (const auto& neighbor : neighbors) {
        const double elevation = elevations[neighbor.index];
        localMin = std::min(localMin, elevation);
        localMax = std::max(localMax, elevation);
        localElevations.push_back(elevation);
    }
    if (neighbors.size() == 1U || detail::krigingBatchConstant(localElevations))
        return firstElevation;

    const std::size_t count = neighbors.size();
    const std::size_t systemSize = count + 1U;
    std::vector<double> matrix(systemSize * systemSize, 0.0);
    std::vector<double> rhs(systemSize, 1.0);
    for (std::size_t row = 0; row < count; ++row) {
        const Vec2& a = points[neighbors[row].index];
        for (std::size_t column = 0; column < count; ++column) {
            if (row == column) continue;
            const Vec2& b = points[neighbors[column].index];
            matrix[row * systemSize + column] = exponentialVariogram(
                std::hypot(a.x - b.x, a.y - b.y), model.range,
                model.partialSill, model.nugget);
        }
        matrix[row * systemSize + row] += std::max(model.nugget * 1.0e-6, 1.0e-10);
        matrix[row * systemSize + count] = 1.0;
        matrix[count * systemSize + row] = 1.0;
        rhs[row] = exponentialVariogram(neighbors[row].distance, model.range,
                                        model.partialSill, model.nugget);
    }
    matrix[count * systemSize + count] = 0.0;
    rhs[count] = 1.0;
    if (!solveDense(matrix, rhs, systemSize)) {
        if (detail::krigingRowwiseAllClose(localElevations)) return firstElevation;
        return idwPrediction(neighbors, elevations, fallbackPower);
    }
    double value = 0.0;
    for (std::size_t i = 0; i < count; ++i) value += rhs[i] * elevations[neighbors[i].index];
    if (!std::isfinite(value)) value = idwPrediction(neighbors, elevations, fallbackPower);
    (void)query;
    return std::clamp(value, localMin, localMax);
}

class GpuPreprocessUnavailable final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::vector<double> interpolateOpenClKriging(
    const std::vector<std::vector<KdTree2D::Neighbor>>& neighbors,
    const std::vector<Vec2>& points, const std::vector<double>& elevations,
    const KrigingModel& model, double fallbackPower, int gpuDevice,
    std::ostream& output) {
    if (neighbors.empty()) return {};
    const std::size_t neighborCount = neighbors.front().size();
    OpenClKrigingRequest request;
    request.deviceIndex = gpuDevice;
    request.queryCount = neighbors.size();
    request.neighborCount = neighborCount;
    request.localPointsXY.resize(request.queryCount * neighborCount * 2U);
    request.localElevations.resize(request.queryCount * neighborCount);
    request.localDistances.resize(request.queryCount * neighborCount);
    request.range = static_cast<float>(model.range);
    request.partialSill = static_cast<float>(model.partialSill);
    request.nugget = static_cast<float>(model.nugget);
    request.fallbackPower = static_cast<float>(fallbackPower);
    for (std::size_t row = 0; row < neighbors.size(); ++row) {
        if (neighbors[row].size() != neighborCount) {
            throw GpuPreprocessUnavailable("inconsistent GPU kriging neighbor counts");
        }
        const Vec2 origin = points[neighbors[row].front().index];
        for (std::size_t column = 0; column < neighborCount; ++column) {
            const std::size_t sample = row * neighborCount + column;
            const auto& neighbor = neighbors[row][column];
            const Vec2& point = points[neighbor.index];
            request.localPointsXY[sample * 2U] =
                static_cast<float>(point.x - origin.x);
            request.localPointsXY[sample * 2U + 1U] =
                static_cast<float>(point.y - origin.y);
            request.localElevations[sample] =
                static_cast<float>(elevations[neighbor.index]);
            request.localDistances[sample] =
                static_cast<float>(neighbor.distance);
        }
    }
    const bool warmupNeeded = nativeOpenClKrigingRequiresWarmup(gpuDevice);
    if (warmupNeeded) {
        output << "[INFO] OpenCL kriging kernel is compiling for first use. "
                  "This one-time warmup may take a little longer.\n";
    }
    try {
        OpenClKrigingResult result = runNativeOpenClKriging(request);
        if (result.compiledThisCall) {
            output << "[INFO] OpenCL kriging kernel warmup finished. "
                      "Continuing with GPU interpolation.\n";
        }
        output << "[INFO] OpenCL kriging backend: " << result.device << '\n';
        std::vector<double> predictions(result.predictions.size());
        std::transform(result.predictions.begin(), result.predictions.end(),
                       predictions.begin(), [](float value) {
                           return static_cast<double>(value);
                       });
        return predictions;
    } catch (const OpenClKrigingError& exception) {
        throw GpuPreprocessUnavailable(exception.what());
    }
}

std::vector<double> interpolateOpenClIdw(
    const std::vector<std::vector<KdTree2D::Neighbor>>& neighbors,
    const std::vector<double>& elevations, double power, int gpuDevice,
    std::ostream& output) {
    if (neighbors.empty()) return {};
    const std::size_t neighborCount = neighbors.front().size();
    if (neighborCount != 0U
        && neighbors.size() > std::numeric_limits<std::size_t>::max() / neighborCount) {
        throw GpuPreprocessUnavailable("GPU IDW request dimensions overflow");
    }
    OpenClIdwRequest request;
    request.deviceIndex = gpuDevice;
    request.queryCount = neighbors.size();
    request.neighborCount = neighborCount;
    request.elevations.resize(request.queryCount * neighborCount);
    request.distances.resize(request.queryCount * neighborCount);
    request.power = static_cast<float>(power);
    for (std::size_t row = 0; row < neighbors.size(); ++row) {
        if (neighbors[row].size() != neighborCount) {
            throw GpuPreprocessUnavailable("inconsistent GPU IDW neighbor counts");
        }
        for (std::size_t column = 0; column < neighborCount; ++column) {
            const std::size_t sample = row * neighborCount + column;
            const auto& neighbor = neighbors[row][column];
            request.elevations[sample] =
                static_cast<float>(elevations[neighbor.index]);
            request.distances[sample] = static_cast<float>(neighbor.distance);
        }
    }
    try {
        const OpenClIdwResult result = runNativeOpenClIdw(request);
        output << "[INFO] OpenCL IDW backend: " << result.device << '\n';
        std::vector<double> predictions(result.predictions.size());
        std::transform(result.predictions.begin(), result.predictions.end(),
                       predictions.begin(), [](float value) {
                           return static_cast<double>(value);
                       });
        return predictions;
    } catch (const OpenClKrigingError& exception) {
        throw GpuPreprocessUnavailable(exception.what());
    }
}

void gaussianSmoothOpenCl(Grid& grid, double sigma, int gpuDevice,
                          std::ostream& output) {
    OpenClGaussianRequest request;
    request.deviceIndex = gpuDevice;
    request.width = grid.nx;
    request.height = grid.ny;
    request.sigma = sigma;
    request.values.resize(grid.z.size());
    std::transform(grid.z.begin(), grid.z.end(), request.values.begin(),
                   [](double value) { return static_cast<float>(value); });
    try {
        const OpenClGaussianResult result = runNativeOpenClGaussian(request);
        output << "[INFO] OpenCL Gaussian backend: " << result.device << '\n';
        grid.z.resize(result.values.size());
        std::transform(result.values.begin(), result.values.end(), grid.z.begin(),
                       [](float value) { return static_cast<double>(value); });
    } catch (const OpenClKrigingError& exception) {
        throw GpuPreprocessUnavailable(exception.what());
    }
}

std::size_t reflectedIndex(std::ptrdiff_t index, std::size_t size) {
    if (size <= 1U) return 0U;
    const std::ptrdiff_t extent = static_cast<std::ptrdiff_t>(size);
    while (index < 0 || index >= extent) {
        if (index < 0) index = -index - 1;
        if (index >= extent) index = 2 * extent - index - 1;
    }
    return static_cast<std::size_t>(index);
}

void gaussianSmooth(Grid& grid, double sigma) {
    if (sigma <= 0.0 || grid.z.empty()) return;
    const int radius = std::max(1, static_cast<int>(4.0 * sigma + 0.5));
    std::vector<double> kernel(static_cast<std::size_t>(radius * 2 + 1));
    double sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const double value = std::exp(-0.5 * std::pow(static_cast<double>(offset) / sigma, 2.0));
        kernel[static_cast<std::size_t>(offset + radius)] = value;
        sum += value;
    }
    for (double& value : kernel) value /= sum;
    std::vector<double> temporary(grid.z.size(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(grid.ny); ++row) {
        for (std::size_t column = 0; column < grid.nx; ++column) {
            double value = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                value += kernel[static_cast<std::size_t>(offset + radius)] *
                         grid.at(static_cast<std::size_t>(row),
                                 reflectedIndex(static_cast<std::ptrdiff_t>(column) + offset,
                                                grid.nx));
            }
            temporary[static_cast<std::size_t>(row) * grid.nx + column] = value;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(grid.ny); ++row) {
        for (std::size_t column = 0; column < grid.nx; ++column) {
            double value = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                value += kernel[static_cast<std::size_t>(offset + radius)] *
                         temporary[reflectedIndex(row + offset, grid.ny) * grid.nx + column];
            }
            grid.at(static_cast<std::size_t>(row), column) = value;
        }
    }
}

Grid interpolateSurface(double minX, double minY, double maxX, double maxY,
                        const DemData& dem, double resolution, double elevationScale,
                        double smoothSigma, double sigmaFactor,
                        const std::string& approach, double idwPower,
                        int requestedNeighbors, bool useGpu, int gpuDevice,
                        const Grid* reusableRawGrid, Grid* capturedRawGrid,
                        std::string_view surfaceName,
                        std::string_view progressStage, std::string_view progressLabel,
                        std::ostream& output, ProgressEmitter& progress) {
    Grid grid;
    if (dem.points.empty() || dem.elevations.empty()) return grid;
    grid.xMin = minX; grid.xMax = maxX; grid.yMin = minY; grid.yMax = maxY;
    grid.nx = static_cast<std::size_t>((maxX - minX) / resolution) + 1U;
    grid.ny = static_cast<std::size_t>((maxY - minY) / resolution) + 1U;
    grid.nx = std::max<std::size_t>(grid.nx, 1U);
    grid.ny = std::max<std::size_t>(grid.ny, 1U);
    if (grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny) {
        throw std::runtime_error("Terrain interpolation grid is too large");
    }
    grid.z.resize(grid.nx * grid.ny);
    const bool reuseInterpolation = reusableRawGrid != nullptr;
    if (reuseInterpolation) {
        if (reusableRawGrid->nx != grid.nx || reusableRawGrid->ny != grid.ny
            || reusableRawGrid->z.size() != grid.z.size()) {
            throw std::runtime_error("Reusable terrain interpolation grid dimensions changed");
        }
        grid.z = reusableRawGrid->z;
    }
    output << "[INFO] Creating " << surfaceName << " grid (resolution: " << general(resolution)
           << "m, approach: " << approach << ", backend: "
           << (useGpu ? "OpenCL GPU" : "CPU") << ")...\n";
    const std::size_t neighborCount = std::min<std::size_t>(
        static_cast<std::size_t>(std::max(requestedNeighbors, 1)), dem.points.size());
    std::optional<KdTree2D> tree;
    if (!useGpu) tree.emplace(dem.points);
    const double appliedSigma = smoothSigma * sigmaFactor;

    if (reuseInterpolation) {
        output << "[INFO] Reusing the unsmoothed terrain interpolation for "
               << surfaceName << ".\n";
    } else if (approach == "kriging" && useGpu) {
        output << "[INFO] Interpolating " << surfaceName
               << " using GPU-accelerated ordinary kriging with " << requestedNeighbors
               << " neighboring points on GPU device " << gpuDevice << "...\n";
    } else if (approach == "kriging") {
        if (grid.z.size() > 50000U) {
            output << "[WARN] Kriging over " << grid.z.size()
                   << " grid points may take noticeably longer than IDW.\n";
        }
        output << "[INFO] Interpolating " << surfaceName
               << " using local ordinary kriging with " << requestedNeighbors
               << " neighboring points...\n";
    } else if (useGpu) {
        output << "[INFO] Interpolating " << surfaceName
               << " using GPU-accelerated Inverse Distance Weighting (IDW) with "
               << requestedNeighbors << " neighboring points on GPU device "
               << gpuDevice << "...\n";
    } else {
        output << "[INFO] Interpolating " << surfaceName
               << " using Inverse Distance Weighting (IDW) with " << requestedNeighbors
               << " neighboring points...\n";
    }

    std::optional<KrigingModel> krigingModel;
    std::vector<std::vector<KdTree2D::Neighbor>> interpolationNeighbors;
    if (!reuseInterpolation && !useGpu && approach == "kriging") {
        progress.stage(qString(progressStage),
                       QStringLiteral("Finding neighboring DEM points for %1").arg(qString(surfaceName)),
                       qString(progressLabel));
        interpolationNeighbors.resize(grid.z.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t linear = 0;
             linear < static_cast<std::ptrdiff_t>(grid.z.size()); ++linear) {
            const std::size_t index = static_cast<std::size_t>(linear);
            const std::size_t row = index / grid.nx;
            const std::size_t column = index % grid.nx;
            interpolationNeighbors[index] =
                tree->nearest({grid.x(column), grid.y(row)}, neighborCount);
        }
    }
    if (!reuseInterpolation && !useGpu && approach == "kriging") {
        std::vector<double> positiveDistances;
        positiveDistances.reserve(grid.z.size() * neighborCount);
        for (const auto& neighbors : interpolationNeighbors) {
            for (const auto& neighbor : neighbors) {
                if (std::isfinite(neighbor.distance) && neighbor.distance > 0.0)
                    positiveDistances.push_back(neighbor.distance);
            }
        }
        krigingModel = estimateKrigingModel(dem.points, dem.elevations, positiveDistances);
    }

    const std::size_t total = grid.z.size();
    if (!reuseInterpolation) {
        progress.progress(qString(progressStage), 0, static_cast<std::int64_t>(std::max<std::size_t>(total, 1U)),
                      QStringLiteral("Interpolating %1 with %2 (0/%3 points)")
                          .arg(qString(surfaceName),
                               useGpu
                                   ? (approach == "kriging"
                                          ? QStringLiteral("ordinary kriging (GPU)")
                                          : QStringLiteral("IDW (GPU)"))
                                   : (approach == "kriging"
                                          ? QStringLiteral("ordinary kriging")
                                          : QStringLiteral("IDW")))
                          .arg(static_cast<qulonglong>(total)),
                          qString(progressLabel), true);
    }
    bool gpuComplete = false;
    std::vector<double> gpuSmoothedValues;
    std::string gpuBackendDevice;
    if (!reuseInterpolation && useGpu) {
        try {
            OpenClTerrainInterpolationRequest request;
            request.deviceIndex = gpuDevice;
            request.width = grid.nx;
            request.height = grid.ny;
            request.neighborCount = neighborCount;
            request.xMin = grid.xMin;
            request.xMax = grid.xMax;
            request.yMin = grid.yMin;
            request.yMax = grid.yMax;
            request.pointsXY.resize(dem.points.size() * 2U);
            for (std::size_t index = 0U; index < dem.points.size(); ++index) {
                request.pointsXY[index * 2U] = dem.points[index].x;
                request.pointsXY[index * 2U + 1U] = dem.points[index].y;
            }
            request.elevations = dem.elevations;
            request.approach = approach == "kriging"
                ? OpenClTerrainApproach::Kriging
                : OpenClTerrainApproach::Idw;
            request.idwPower = idwPower;
            request.smoothSigma = appliedSigma;
            const OpenClTerrainInterpolationResult result =
                runNativeOpenClTerrainInterpolation(request);
            gpuBackendDevice = result.device;
            grid.z.resize(result.rawValues.size());
            std::transform(result.rawValues.begin(), result.rawValues.end(),
                           grid.z.begin(), [](float value) {
                               return static_cast<double>(value);
                           });
            gpuSmoothedValues.resize(result.values.size());
            std::transform(result.values.begin(), result.values.end(),
                           gpuSmoothedValues.begin(), [](float value) {
                               return static_cast<double>(value);
                           });
            output << "[INFO] OpenCL DEM neighbor backend: " << result.device
                   << " ("
                   << (result.usedRegularGridFastPath
                           ? "complete regular-grid direct indexing"
                           : "uniform-bin CSR search")
                   << ")\n"
                   << "[INFO] OpenCL "
                   << (approach == "kriging" ? "kriging" : "IDW")
                   << " backend: " << result.device << '\n';
            gpuComplete = true;
        } catch (const OpenClKrigingError& exception) {
            throw GpuPreprocessUnavailable(
                "GPU preprocessing was explicitly requested for " +
                std::string(surfaceName) +
                ", but the OpenCL backend is unavailable: " + exception.what());
        }
    }
    if (!reuseInterpolation && !gpuComplete) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t linear = 0;
             linear < static_cast<std::ptrdiff_t>(total); ++linear) {
            const std::size_t index = static_cast<std::size_t>(linear);
            const std::size_t row = index / grid.nx;
            const std::size_t column = index % grid.nx;
            const Vec2 query{grid.x(column), grid.y(row)};
            if (approach == "kriging") {
                grid.z[index] = krigingPrediction(
                    query, interpolationNeighbors[index], dem.points, dem.elevations,
                    *krigingModel, idwPower);
            } else {
                const auto neighbors = tree->nearest(query, neighborCount);
                grid.z[index] = idwPrediction(neighbors, dem.elevations, idwPower);
            }
        }
    }
    if (!reuseInterpolation) {
        progress.progress(qString(progressStage), static_cast<std::int64_t>(total),
                          static_cast<std::int64_t>(std::max<std::size_t>(total, 1U)),
                          QStringLiteral("Interpolating %1 complete").arg(qString(surfaceName)),
                          qString(progressLabel), true);
    }
    if (capturedRawGrid != nullptr) *capturedRawGrid = grid;

    if (appliedSigma > 0.0) {
        progress.stage(qString(progressStage),
                       QStringLiteral("Smoothing %1 (sigma=%2)")
                           .arg(qString(surfaceName)).arg(appliedSigma, 0, 'f', 2),
                       qString(progressLabel));
        output << "[INFO] Applying Gaussian smoothing to " << surfaceName << " (sigma="
               << fixed(appliedSigma, 2) << ")...\n";
        if (useGpu) {
            if (!reuseInterpolation) {
                grid.z = std::move(gpuSmoothedValues);
                output << "[INFO] OpenCL Gaussian backend: "
                       << gpuBackendDevice << '\n';
            } else {
                gaussianSmoothOpenCl(grid, appliedSigma, gpuDevice, output);
            }
        } else {
            gaussianSmooth(grid, appliedSigma);
        }
    } else {
        output << "[INFO] Smoothing disabled for " << surfaceName << '\n';
    }
    double minimum = *std::min_element(grid.z.begin(), grid.z.end());
    if (minimum < 0.0) {
        output << "[WARN] " << static_cast<char>(std::toupper(static_cast<unsigned char>(surfaceName.front())))
               << surfaceName.substr(1) << " contains negative values (min=" << fixed(minimum, 2)
               << "m), clipping to 0\n";
        for (double& value : grid.z) value = std::max(value, 0.0);
        minimum = 0.0;
    }
    if (std::abs(minimum) > 1.0e-9) {
        output << "[INFO] Normalizing " << surfaceName
               << " to local datum (grid min -> 0). Offset removed: " << fixed(minimum, 3) << "m\n";
        for (double& value : grid.z) value = std::max(0.0, value - minimum);
    }
    if (elevationScale != 1.0) {
        output << "[INFO] Applying elevation scale factor to " << surfaceName << ": "
               << general(elevationScale) << "x\n";
        for (double& value : grid.z) value *= elevationScale;
    }
    const auto [minIt, maxIt] = std::minmax_element(grid.z.begin(), grid.z.end());
    std::string title(surfaceName);
    if (!title.empty()) title.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(title.front())));
    output << "[INFO] " << title << " elevation range: " << fixed(*minIt, 2) << " to "
           << fixed(*maxIt, 2) << " meters\n";
    progress.complete(qString(progressStage), qString(title + " ready"), qString(progressLabel));
    return grid;
}

double lookupElevation(const Grid& grid, double x, double y) {
    if (grid.z.empty() || grid.nx < 2U || grid.ny < 2U || x < grid.xMin ||
        x > grid.xMax || y < grid.yMin || y > grid.yMax) {
        return 0.0;
    }
    const double dx = (grid.xMax - grid.xMin) / static_cast<double>(grid.nx - 1U);
    const double dy = (grid.yMax - grid.yMin) / static_cast<double>(grid.ny - 1U);
    if (dx == 0.0 || dy == 0.0) return 0.0;
    std::size_t column = static_cast<std::size_t>((x - grid.xMin) / dx);
    std::size_t row = static_cast<std::size_t>((y - grid.yMin) / dy);
    column = std::min(column, grid.nx - 2U);
    row = std::min(row, grid.ny - 2U);
    const double x0 = grid.x(column), x1 = grid.x(column + 1U);
    const double y0 = grid.y(row), y1 = grid.y(row + 1U);
    const double wx = x1 != x0 ? (x - x0) / (x1 - x0) : 0.0;
    const double wy = y1 != y0 ? (y - y0) / (y1 - y0) : 0.0;
    return (1.0 - wx) * (1.0 - wy) * grid.at(row, column) +
           wx * (1.0 - wy) * grid.at(row, column + 1U) +
           (1.0 - wx) * wy * grid.at(row + 1U, column) +
           wx * wy * grid.at(row + 1U, column + 1U);
}

void appendOrientedTriangle(std::vector<Triangle>& triangles, Vec3 a, Vec3 b,
                            Vec3 c, bool upward) {
    const double crossZ = (b.x - a.x) * (c.y - a.y) -
                          (b.y - a.y) * (c.x - a.x);
    if ((upward && crossZ < 0.0) || (!upward && crossZ > 0.0)) std::swap(b, c);
    triangles.push_back({a, b, c});
}

void appendQuad(std::vector<Triangle>& triangles, const Vec3& a, const Vec3& b,
                const Vec3& c, const Vec3& d) {
    triangles.push_back({a, b, c});
    triangles.push_back({a, c, d});
}

bool equalPoint(const Vec2& left, const Vec2& right) {
    return left.x == right.x && left.y == right.y;
}

void normalizeRing(std::vector<Vec2>& ring) {
    while (ring.size() > 1U && equalPoint(ring.front(), ring.back())) ring.pop_back();
    std::vector<Vec2> compact;
    compact.reserve(ring.size());
    for (const Vec2& point : ring) {
        if (compact.empty() || !equalPoint(compact.back(), point)) compact.push_back(point);
    }
    ring = std::move(compact);
}

std::vector<Triangle> extrudePolygon(const Polygon2D& input, double height,
                                     double baseZ = 0.0) {
    if (height <= 0.0) return {};
    Polygon2D polygon = input;
    normalizeRing(polygon.exterior);
    for (auto& hole : polygon.holes) normalizeRing(hole);
    polygon.holes.erase(
        std::remove_if(polygon.holes.begin(), polygon.holes.end(),
                       [](const auto& ring) { return ring.size() < 3U; }),
        polygon.holes.end());
    if (polygon.exterior.size() < 3U) throw std::runtime_error("Polygon has fewer than three vertices");

    using EarPoint = std::array<double, 2>;
    std::vector<std::vector<EarPoint>> rings;
    rings.reserve(1U + polygon.holes.size());
    const auto addRing = [&](const std::vector<Vec2>& source) {
        std::vector<EarPoint> target;
        target.reserve(source.size());
        for (const Vec2& point : source) target.push_back({point.x, point.y});
        rings.push_back(std::move(target));
    };
    addRing(polygon.exterior);
    for (const auto& hole : polygon.holes) addRing(hole);

    std::vector<Vec2> flattened;
    for (const auto& ring : rings) {
        for (const auto& point : ring) flattened.push_back({point[0], point[1]});
    }
    const std::vector<std::uint32_t> indices = mapbox::earcut<std::uint32_t>(rings);
    if (indices.empty()) throw std::runtime_error("No triangulation engine could triangulate polygon");
    std::vector<Triangle> triangles;
    std::size_t sideEdges = polygon.exterior.size();
    for (const auto& hole : polygon.holes) sideEdges += hole.size();
    triangles.reserve(indices.size() / 3U * 2U + sideEdges * 2U);
    for (std::size_t i = 0; i + 2U < indices.size(); i += 3U) {
        const Vec2& a = flattened[indices[i]];
        const Vec2& b = flattened[indices[i + 1U]];
        const Vec2& c = flattened[indices[i + 2U]];
        appendOrientedTriangle(triangles, {a.x, a.y, baseZ + height},
                               {b.x, b.y, baseZ + height},
                               {c.x, c.y, baseZ + height}, true);
        appendOrientedTriangle(triangles, {a.x, a.y, baseZ}, {b.x, b.y, baseZ},
                               {c.x, c.y, baseZ}, false);
    }
    const auto addSides = [&](const std::vector<Vec2>& ring) {
        for (std::size_t i = 0; i < ring.size(); ++i) {
            const Vec2& a = ring[i];
            const Vec2& b = ring[(i + 1U) % ring.size()];
            appendQuad(triangles, {a.x, a.y, baseZ}, {b.x, b.y, baseZ},
                       {b.x, b.y, baseZ + height}, {a.x, a.y, baseZ + height});
        }
    };
    addSides(polygon.exterior);
    for (const auto& hole : polygon.holes) addSides(hole);
    return triangles;
}

Polygon2D rectangle(double minX, double minY, double maxX, double maxY) {
    return {{{minX, minY}, {maxX, minY}, {maxX, maxY}, {minX, maxY}}, {}};
}

std::vector<Triangle> terrainTriangles(const Grid& grid, double baseHeight) {
    if (grid.z.empty() || grid.nx < 2U || grid.ny < 2U) return {};
    const std::size_t cells = (grid.nx - 1U) * (grid.ny - 1U);
    const std::size_t edges = 2U * (grid.nx - 1U) + 2U * (grid.ny - 1U);
    std::vector<Triangle> triangles;
    triangles.reserve(cells * 4U + edges * 2U);
    for (std::size_t row = 0; row + 1U < grid.ny; ++row) {
        for (std::size_t column = 0; column + 1U < grid.nx; ++column) {
            const Vec3 v0{grid.x(column), grid.y(row), baseHeight + grid.at(row, column)};
            const Vec3 v1{grid.x(column + 1U), grid.y(row), baseHeight + grid.at(row, column + 1U)};
            const Vec3 v2{grid.x(column), grid.y(row + 1U), baseHeight + grid.at(row + 1U, column)};
            const Vec3 v3{grid.x(column + 1U), grid.y(row + 1U), baseHeight + grid.at(row + 1U, column + 1U)};
            triangles.push_back({v0, v1, v2});
            triangles.push_back({v1, v3, v2});
            const Vec3 b0{v0.x, v0.y, baseHeight};
            const Vec3 b1{v1.x, v1.y, baseHeight};
            const Vec3 b2{v2.x, v2.y, baseHeight};
            const Vec3 b3{v3.x, v3.y, baseHeight};
            triangles.push_back({b0, b2, b1});
            triangles.push_back({b1, b2, b3});
        }
    }
    for (std::size_t row = 0; row + 1U < grid.ny; ++row) {
        const Vec3 lt{grid.x(0), grid.y(row), baseHeight + grid.at(row, 0)};
        const Vec3 ltn{grid.x(0), grid.y(row + 1U), baseHeight + grid.at(row + 1U, 0)};
        appendQuad(triangles, lt, ltn, {ltn.x, ltn.y, baseHeight}, {lt.x, lt.y, baseHeight});
        const std::size_t last = grid.nx - 1U;
        const Vec3 rt{grid.x(last), grid.y(row), baseHeight + grid.at(row, last)};
        const Vec3 rtn{grid.x(last), grid.y(row + 1U), baseHeight + grid.at(row + 1U, last)};
        appendQuad(triangles, rt, {rt.x, rt.y, baseHeight}, {rtn.x, rtn.y, baseHeight}, rtn);
    }
    for (std::size_t column = 0; column + 1U < grid.nx; ++column) {
        const Vec3 ft{grid.x(column), grid.y(0), baseHeight + grid.at(0, column)};
        const Vec3 ftn{grid.x(column + 1U), grid.y(0), baseHeight + grid.at(0, column + 1U)};
        appendQuad(triangles, ft, {ft.x, ft.y, baseHeight}, {ftn.x, ftn.y, baseHeight}, ftn);
        const std::size_t last = grid.ny - 1U;
        const Vec3 bt{grid.x(column), grid.y(last), baseHeight + grid.at(last, column)};
        const Vec3 btn{grid.x(column + 1U), grid.y(last), baseHeight + grid.at(last, column + 1U)};
        appendQuad(triangles, bt, btn, {btn.x, btn.y, baseHeight}, {bt.x, bt.y, baseHeight});
    }
    return triangles;
}

std::vector<Triangle> terrainTrianglesOpenCl(const Grid& grid,
                                             double baseHeight,
                                             int gpuDevice,
                                             std::ostream& output) {
    OpenClTerrainMeshRequest request;
    request.deviceIndex = gpuDevice;
    request.width = grid.nx;
    request.height = grid.ny;
    request.xMin = grid.xMin;
    request.xMax = grid.xMax;
    request.yMin = grid.yMin;
    request.yMax = grid.yMax;
    request.baseHeight = baseHeight;
    request.elevations = grid.z;
    try {
        const OpenClTerrainMeshResult result =
            runNativeOpenClTerrainMesh(request);
        if (result.triangleVertices.size() % 9U != 0U) {
            throw GpuPreprocessUnavailable(
                "OpenCL terrain mesh returned an inconsistent vertex buffer");
        }
        std::vector<Triangle> triangles(
            result.triangleVertices.size() / 9U);
        for (std::size_t index = 0U; index < triangles.size(); ++index) {
            const double* values =
                result.triangleVertices.data() + index * 9U;
            triangles[index] = {
                {values[0], values[1], values[2]},
                {values[3], values[4], values[5]},
                {values[6], values[7], values[8]}};
        }
        output << "[INFO] OpenCL terrain mesh backend: " << result.device
               << " (allocation-bounded triangle batches, stable order)\n";
        return triangles;
    } catch (const OpenClKrigingError& exception) {
        throw GpuPreprocessUnavailable(
            "GPU preprocessing was explicitly requested for terrain mesh "
            "generation, but the OpenCL backend is unavailable: " +
            std::string(exception.what()));
    }
}

void writeFloat32(std::ofstream& output, float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::array<char, 4> bytes{
        static_cast<char>(bits & 0xFFU), static_cast<char>((bits >> 8U) & 0xFFU),
        static_cast<char>((bits >> 16U) & 0xFFU), static_cast<char>((bits >> 24U) & 0xFFU)};
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void writeUint32(std::ofstream& output, std::uint32_t value) {
    const std::array<char, 4> bytes{
        static_cast<char>(value & 0xFFU), static_cast<char>((value >> 8U) & 0xFFU),
        static_cast<char>((value >> 16U) & 0xFFU), static_cast<char>((value >> 24U) & 0xFFU)};
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void writeBinaryStl(const fs::path& path, const std::vector<Triangle>& triangles) {
    if (triangles.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("STL has too many triangles");
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create STL: " + pathToUtf8(path));
    std::array<char, 80> header{};
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    writeUint32(output, static_cast<std::uint32_t>(triangles.size()));
    for (const Triangle& triangle : triangles) {
        const Vec3 ab{triangle.b.x - triangle.a.x, triangle.b.y - triangle.a.y,
                      triangle.b.z - triangle.a.z};
        const Vec3 ac{triangle.c.x - triangle.a.x, triangle.c.y - triangle.a.y,
                      triangle.c.z - triangle.a.z};
        Vec3 normal{ab.y * ac.z - ab.z * ac.y,
                    ab.z * ac.x - ab.x * ac.z,
                    ab.x * ac.y - ab.y * ac.x};
        const double length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (length > 0.0) { normal.x /= length; normal.y /= length; normal.z /= length; }
        for (const double value : {normal.x, normal.y, normal.z,
                                   triangle.a.x, triangle.a.y, triangle.a.z,
                                   triangle.b.x, triangle.b.y, triangle.b.z,
                                   triangle.c.x, triangle.c.y, triangle.c.z}) {
            writeFloat32(output, static_cast<float>(value));
        }
        const std::array<char, 2> attribute{};
        output.write(attribute.data(), 2);
    }
    if (!output) throw std::runtime_error("Failed while writing STL: " + pathToUtf8(path));
}

void writeDemCsv(const fs::path& projectTemp, const Grid& grid,
                 std::ostream& output) {
    const fs::path interpolated = projectTemp / "interpolated_dem.csv";
    output << "[INFO] Exporting interpolated DEM to " << pathToUtf8(interpolated) << '\n';
    std::ofstream all(interpolated, std::ios::trunc);
    if (!all) throw std::runtime_error("Cannot create " + pathToUtf8(interpolated));
    all.imbue(std::locale::classic());
    all << "x,y,elevation\n" << std::fixed << std::setprecision(3);
    for (std::size_t row = 0; row < grid.ny; ++row) {
        for (std::size_t column = 0; column < grid.nx; ++column) {
            all << grid.x(column) << ',' << grid.y(row) << ',' << grid.at(row, column) << '\n';
        }
    }
    if (!all) throw std::runtime_error("Failed while writing " + pathToUtf8(interpolated));
    output << "[INFO] Exported " << grid.z.size() << " interpolated elevation points to "
           << pathToUtf8(interpolated) << '\n';

    const fs::path boundary = projectTemp / "boundary_dem.csv";
    output << "[INFO] Exporting boundary DEM to " << pathToUtf8(boundary) << '\n';
    std::ofstream edges(boundary, std::ios::trunc);
    if (!edges) throw std::runtime_error("Cannot create " + pathToUtf8(boundary));
    edges.imbue(std::locale::classic());
    edges << "x,y,elevation\n" << std::fixed << std::setprecision(3);
    std::size_t count = 0U;
    for (std::size_t row = 0; row < grid.ny; ++row) {
        for (std::size_t column = 0; column < grid.nx; ++column) {
            if (row != 0U && row + 1U != grid.ny && column != 0U && column + 1U != grid.nx) continue;
            edges << grid.x(column) << ',' << grid.y(row) << ',' << grid.at(row, column) << '\n';
            ++count;
        }
    }
    if (!edges) throw std::runtime_error("Failed while writing " + pathToUtf8(boundary));
    output << "[INFO] Exported " << count << " boundary elevation points to "
           << pathToUtf8(boundary) << '\n';
}

Vec2 rotatePoint(Vec2 point, double degrees, Vec2 pivot) {
    if (degrees == 0.0) return point;
    const double radians = degrees * std::numbers::pi / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    const double x = point.x - pivot.x;
    const double y = point.y - pivot.y;
    return {cosine * x - sine * y + pivot.x,
            sine * x + cosine * y + pivot.y};
}

void affineGeometry(OGRGeometry* geometry, double rotateDegrees, Vec2 pivot,
                    double translateX = 0.0, double translateY = 0.0) {
    if (geometry == nullptr) return;
    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    if (type == wkbPoint) {
        auto* point = geometry->toPoint();
        const Vec2 transformed = rotatePoint({point->getX(), point->getY()}, rotateDegrees, pivot);
        point->setX(transformed.x + translateX);
        point->setY(transformed.y + translateY);
        return;
    }
    if (OGR_GT_IsCurve(type)) {
        auto* curve = dynamic_cast<OGRLineString*>(geometry);
        if (curve != nullptr) {
            for (int i = 0; i < curve->getNumPoints(); ++i) {
                const Vec2 transformed = rotatePoint({curve->getX(i), curve->getY(i)}, rotateDegrees, pivot);
                curve->setPoint(i, transformed.x + translateX, transformed.y + translateY);
            }
        }
        return;
    }
    if (type == wkbPolygon) {
        auto* polygon = geometry->toPolygon();
        affineGeometry(polygon->getExteriorRing(), rotateDegrees, pivot, translateX, translateY);
        for (int i = 0; i < polygon->getNumInteriorRings(); ++i) {
            affineGeometry(polygon->getInteriorRing(i), rotateDegrees, pivot, translateX, translateY);
        }
        return;
    }
    auto* collection = dynamic_cast<OGRGeometryCollection*>(geometry);
    if (collection != nullptr) {
        for (int i = 0; i < collection->getNumGeometries(); ++i) {
            affineGeometry(collection->getGeometryRef(i), rotateDegrees, pivot, translateX, translateY);
        }
    }
}

GeometryPtr makeValidPolygonal(const OGRGeometry* geometry) {
    if (geometry == nullptr || geometry->IsEmpty()) return {};
    GeometryPtr valid(geometry->MakeValid());
    if (!valid) valid.reset(geometry->Buffer(0.0, 30));
    return valid;
}

std::vector<Vec2> ringPoints(const OGRLinearRing* ring) {
    std::vector<Vec2> points;
    if (ring == nullptr) return points;
    points.reserve(static_cast<std::size_t>(ring->getNumPoints()));
    for (int i = 0; i < ring->getNumPoints(); ++i) points.push_back({ring->getX(i), ring->getY(i)});
    normalizeRing(points);
    return points;
}

void appendPolygons(const OGRGeometry* geometry, std::vector<Polygon2D>& polygons) {
    if (geometry == nullptr || geometry->IsEmpty()) return;
    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    if (type == wkbPolygon) {
        const auto* polygon = geometry->toPolygon();
        Polygon2D result;
        result.exterior = ringPoints(polygon->getExteriorRing());
        for (int i = 0; i < polygon->getNumInteriorRings(); ++i) {
            result.holes.push_back(ringPoints(polygon->getInteriorRing(i)));
        }
        if (result.exterior.size() >= 3U) polygons.push_back(std::move(result));
        return;
    }
    const auto* collection = dynamic_cast<const OGRGeometryCollection*>(geometry);
    if (collection != nullptr) {
        for (int i = 0; i < collection->getNumGeometries(); ++i) {
            appendPolygons(collection->getGeometryRef(i), polygons);
        }
    }
}

GeometryPtr rectangleGeometry(double minX, double minY, double maxX, double maxY) {
    auto ring = std::make_unique<OGRLinearRing>();
    ring->addPoint(minX, minY); ring->addPoint(maxX, minY);
    ring->addPoint(maxX, maxY); ring->addPoint(minX, maxY); ring->addPoint(minX, minY);
    auto polygon = GeometryPtr(new OGRPolygon());
    polygon->toPolygon()->addRingDirectly(ring.release());
    return polygon;
}

struct BuildingRecord {
    GeometryPtr geometry;
    double height = 0.0;
};

std::string detectHeightField(OGRFeatureDefn& definition,
                              const std::string& requested) {
    const std::string normalized = lowerAscii(requested);
    if (normalized == "auto" || normalized == "inferred") {
        constexpr std::array<std::string_view, 6> candidates{
            "Height", "Elevation", "height", "elevation", "HEIGHT", "ELEVATION"};
        for (const std::string_view candidate : candidates) {
            for (int index = 0; index < definition.GetFieldCount(); ++index) {
                if (candidate == definition.GetFieldDefn(index)->GetNameRef()) return std::string(candidate);
            }
        }
        std::ostringstream message;
        message << "Cannot auto-detect height field. Available fields: [";
        for (int index = 0; index < definition.GetFieldCount(); ++index) {
            if (index != 0) message << ", ";
            message << definition.GetFieldDefn(index)->GetNameRef();
        }
        message << ']';
        throw std::runtime_error(message.str());
    }
    for (int index = 0; index < definition.GetFieldCount(); ++index) {
        if (requested == definition.GetFieldDefn(index)->GetNameRef()) return requested;
    }
    std::ostringstream message;
    message << "Specified height field '" << requested << "' not found. Available fields: [";
    for (int index = 0; index < definition.GetFieldCount(); ++index) {
        if (index != 0) message << ", ";
        message << definition.GetFieldDefn(index)->GetNameRef();
    }
    message << ']';
    throw std::runtime_error(message.str());
}

std::vector<BuildingRecord> loadBuildings(const fs::path& path,
                                          OGRSpatialReference& workReference,
                                          bool noReproject,
                                          std::string& heightField,
                                          std::ostream& output) {
    DatasetPtr dataset(static_cast<GDALDataset*>(GDALOpenEx(
        pathToUtf8(path).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
    if (!dataset) throw std::runtime_error("Cannot read building shapefile: " + pathToUtf8(path));
    OGRLayer* layer = dataset->GetLayer(0);
    if (layer == nullptr) throw std::runtime_error("Shapefile has no readable layer");
    const auto featureCount = layer->GetFeatureCount(true);
    if (featureCount == 0) throw std::runtime_error("Shapefile has no features");
    output << "[INFO] Read " << featureCount << " features\n";
    std::unique_ptr<OGRSpatialReference> source = cloneReference(layer->GetSpatialRef());
    if (!source) source = epsgReference(4326);
    output << "[INFO] Source CRS: " << referenceText(*source) << '\n';
    std::optional<TransformPtr> transform;
    if (!noReproject && !source->IsSame(&workReference)) {
        transform.emplace(makeTransform(source.get(), &workReference, "building reprojection"));
    }
    OGRFeatureDefn* definition = layer->GetLayerDefn();
    if (definition == nullptr) throw std::runtime_error("Building layer has no schema");
    heightField = detectHeightField(*definition, heightField);
    const int heightIndex = definition->GetFieldIndex(heightField.c_str());
    std::vector<BuildingRecord> records;
    if (featureCount > 0) records.reserve(static_cast<std::size_t>(featureCount));
    layer->ResetReading();
    while (OGRFeature* raw = layer->GetNextFeature()) {
        FeaturePtr feature(raw);
        OGRGeometry* sourceGeometry = feature->GetGeometryRef();
        if (sourceGeometry == nullptr || sourceGeometry->IsEmpty() ||
            !feature->IsFieldSetAndNotNull(heightIndex)) continue;
        const double height = feature->GetFieldAsDouble(heightIndex);
        GeometryPtr geometry(sourceGeometry->clone());
        geometry->flattenTo2D();
        if (transform && geometry->transform(transform->get()) != OGRERR_NONE) continue;
        records.push_back({std::move(geometry), height});
    }
    return records;
}

std::optional<int> elevationField(OGRFeatureDefn& definition) {
    constexpr std::array<std::string_view, 7> candidates{
        "elevation", "Elevation", "ELEVATION", "height", "Height", "z", "Z"};
    for (const std::string_view candidate : candidates) {
        for (int i = 0; i < definition.GetFieldCount(); ++i) {
            if (candidate == definition.GetFieldDefn(i)->GetNameRef()) return i;
        }
    }
    return std::nullopt;
}

DemData loadDemVector(const fs::path& path, OGRSpatialReference& workReference,
                      std::ostream& output, ProgressEmitter& progress) {
    output << "[INFO] Loading DEM data from: " << pathToUtf8(path) << '\n';
    DatasetPtr dataset(static_cast<GDALDataset*>(GDALOpenEx(
        pathToUtf8(path).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
    if (!dataset) throw std::runtime_error("Cannot read DEM shapefile: " + pathToUtf8(path));
    OGRLayer* layer = dataset->GetLayer(0);
    if (layer == nullptr || layer->GetFeatureCount(true) == 0) {
        output << "[WARN] DEM shapefile is empty, will use flat terrain\n";
        return {};
    }
    OGRFeatureDefn* definition = layer->GetLayerDefn();
    if (definition == nullptr) return {};
    const auto field = elevationField(*definition);
    if (!field) {
        output << "[WARN] No elevation column found in DEM, will use flat terrain\n";
        return {};
    }
    output << "[INFO] Ensuring DEM geometries are 2D...\n";
    std::unique_ptr<OGRSpatialReference> source = cloneReference(layer->GetSpatialRef());
    if (!source) source = cloneReference(&workReference);
    std::optional<TransformPtr> transform;
    if (!source->IsSame(&workReference)) {
        output << "[INFO] Reprojecting DEM from " << referenceText(*source) << " to "
               << referenceText(workReference) << '\n';
        transform.emplace(makeTransform(source.get(), &workReference, "DEM reprojection"));
    }
    output << "[INFO] Using elevation column: "
           << definition->GetFieldDefn(*field)->GetNameRef() << '\n';
    DemData dem;
    const auto total = static_cast<std::size_t>(std::max<GIntBig>(layer->GetFeatureCount(true), 0));
    progress.progress(QStringLiteral("load_dem"), 0,
                      static_cast<std::int64_t>(std::max<std::size_t>(total, 1U)),
                      QStringLiteral("Sampling DEM points 0/%1").arg(static_cast<qulonglong>(total)),
                      {}, true);
    layer->ResetReading();
    std::size_t current = 0U;
    while (OGRFeature* raw = layer->GetNextFeature()) {
        FeaturePtr feature(raw);
        ++current;
        OGRGeometry* geometry = feature->GetGeometryRef();
        if (geometry != nullptr && !geometry->IsEmpty() && feature->IsFieldSetAndNotNull(*field)) {
            GeometryPtr clone(geometry->clone());
            clone->flattenTo2D();
            if (!transform || clone->transform(transform->get()) == OGRERR_NONE) {
                OGRPoint centroid;
                if (clone->Centroid(&centroid) == OGRERR_NONE) {
                    const double elevation = feature->GetFieldAsDouble(*field);
                    if (std::isfinite(elevation)) {
                        dem.points.push_back({centroid.getX(), centroid.getY()});
                        dem.elevations.push_back(elevation);
                    }
                }
            }
        }
        progress.progress(QStringLiteral("load_dem"), static_cast<std::int64_t>(current),
                          static_cast<std::int64_t>(std::max<std::size_t>(total, 1U)),
                          QStringLiteral("Sampling DEM points %1/%2")
                              .arg(static_cast<qulonglong>(current))
                              .arg(static_cast<qulonglong>(total)));
    }
    if (dem.points.empty()) {
        output << "[WARN] No valid DEM points found, will use flat terrain\n";
        return {};
    }
    const auto [minimum, maximum] = std::minmax_element(dem.elevations.begin(), dem.elevations.end());
    output << "[INFO] Loaded " << dem.elevations.size() << " DEM points\n"
           << "[INFO] Original elevation range (sea level): " << fixed(*minimum, 2) << " to "
           << fixed(*maximum, 2) << " meters\n";
    const double datum = *minimum;
    for (double& value : dem.elevations) value -= datum;
    output << "[INFO] Adjusted to datum (lowest point = 0): 0.00 to "
           << fixed(*maximum - datum, 2) << " meters\n"
           << "[INFO] Datum offset applied: " << fixed(datum, 2) << "m\n";
    return dem;
}

std::vector<fs::path> sortedFiles(const fs::path& directory,
                                  std::initializer_list<std::string_view> extensions) {
    std::vector<fs::path> paths;
    std::error_code error;
    if (!fs::is_directory(directory, error) || error) return paths;
    for (fs::directory_iterator iterator(directory, error), end;
         !error && iterator != end; iterator.increment(error)) {
        std::error_code typeError;
        if (!iterator->is_regular_file(typeError) || typeError) continue;
        const std::string extension = lowerAscii(pathToUtf8(iterator->path().extension()));
        if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
            paths.push_back(iterator->path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

DemData loadDemRaster(const fs::path& path, OGRSpatialReference& workReference,
                      double lonMin, double lonMax, double latMin, double latMax,
                      std::ostream& output) {
    output << "[INFO] DEM path: " << pathToUtf8(path) << '\n';
    DatasetPtr dataset(static_cast<GDALDataset*>(GDALOpenEx(
        pathToUtf8(path).c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
    if (!dataset || dataset->GetRasterCount() < 1) throw std::runtime_error("Cannot read DEM GeoTIFF");
    GDALRasterBand* band = dataset->GetRasterBand(1);
    double transform[6]{};
    if (dataset->GetGeoTransform(transform) != CE_None) throw std::runtime_error("DEM GeoTIFF has no geotransform");
    std::unique_ptr<OGRSpatialReference> rasterReference;
    const OGRSpatialReference* datasetReference = dataset->GetSpatialRef();
    if (datasetReference != nullptr) rasterReference = cloneReference(datasetReference);
    if (!rasterReference) {
        output << "[WARN] DEM has no CRS. Assuming raster bounds are EPSG:4326 for coverage check.\n";
        rasterReference = epsgReference(4326);
    }
    output << "[INFO] DEM CRS: " << referenceText(*rasterReference) << '\n'
           << "[INFO] Cutting raster by expanded bbox ...\n";
    auto wgs84 = epsgReference(4326);
    TransformPtr toRaster = makeTransform(wgs84.get(), rasterReference.get(), "DEM crop bounds");
    const double lonSpan = lonMax - lonMin;
    const double latSpan = latMax - latMin;
    std::array<double, 4> xs{lonMin - 0.2 * lonSpan, lonMax + 0.2 * lonSpan,
                             lonMax + 0.2 * lonSpan, lonMin - 0.2 * lonSpan};
    std::array<double, 4> ys{latMin - 0.2 * latSpan, latMin - 0.2 * latSpan,
                             latMax + 0.2 * latSpan, latMax + 0.2 * latSpan};
    if (!toRaster->Transform(4, xs.data(), ys.data())) throw std::runtime_error("Cannot project DEM crop bounds");
    double inverse[6]{};
    if (!GDALInvGeoTransform(transform, inverse)) throw std::runtime_error("DEM geotransform is not invertible");
    int xStart = dataset->GetRasterXSize(), yStart = dataset->GetRasterYSize();
    int xStop = 0, yStop = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double pixel = inverse[0] + inverse[1] * xs[i] + inverse[2] * ys[i];
        const double line = inverse[3] + inverse[4] * xs[i] + inverse[5] * ys[i];
        xStart = std::min(xStart, static_cast<int>(std::floor(pixel)) - 1);
        xStop = std::max(xStop, static_cast<int>(std::ceil(pixel)) + 1);
        yStart = std::min(yStart, static_cast<int>(std::floor(line)) - 1);
        yStop = std::max(yStop, static_cast<int>(std::ceil(line)) + 1);
    }
    xStart = std::clamp(xStart, 0, dataset->GetRasterXSize());
    xStop = std::clamp(xStop, 0, dataset->GetRasterXSize());
    yStart = std::clamp(yStart, 0, dataset->GetRasterYSize());
    yStop = std::clamp(yStop, 0, dataset->GetRasterYSize());
    if (xStart >= xStop || yStart >= yStop) throw std::runtime_error("GeoTIFF does not overlap target bounds");
    int hasNoData = 0;
    const double noData = band->GetNoDataValue(&hasNoData);
    DemData dem;
    const int windowWidth = xStop - xStart;
    const int windowHeight = yStop - yStart;
    const std::size_t windowCells = static_cast<std::size_t>(windowWidth) *
                                    static_cast<std::size_t>(windowHeight);
    dem.points.reserve(windowCells);
    dem.elevations.reserve(dem.points.capacity());
    OGRLinearRing coverageRing;
    for (std::size_t index = 0; index < xs.size(); ++index)
        coverageRing.addPoint(xs[index], ys[index]);
    coverageRing.addPoint(xs.front(), ys.front());
    OGRPolygon coveragePolygon;
    coveragePolygon.addRing(&coverageRing);
    const bool axisAlignedCoverage =
        xs[0] == xs[3] && xs[1] == xs[2] &&
        ys[0] == ys[1] && ys[2] == ys[3];
    const auto [coverageMinXIt, coverageMaxXIt] =
        std::minmax_element(xs.begin(), xs.end());
    const auto [coverageMinYIt, coverageMaxYIt] =
        std::minmax_element(ys.begin(), ys.end());
    OGRPreparedGeometryUniquePtr preparedCoverage;
    if (!axisAlignedCoverage && OGRHasPreparedGeometrySupport()) {
        preparedCoverage.reset(OGRCreatePreparedGeometry(
            OGRGeometry::ToHandle(&coveragePolygon)));
    }
    int blockWidth = 0;
    int blockHeight = 0;
    band->GetBlockSize(&blockWidth, &blockHeight);
    (void)blockWidth;
    constexpr std::size_t targetReadBytes = 8U * 1024U * 1024U;
    const std::size_t bytesPerRow =
        static_cast<std::size_t>(windowWidth) * sizeof(double);
    const std::size_t targetRows = std::max<std::size_t>(
        1U, targetReadBytes / std::max<std::size_t>(bytesPerRow, 1U));
    const std::size_t rowBudget = std::min<std::size_t>(
        static_cast<std::size_t>(windowHeight),
        std::min<std::size_t>(targetRows, 1024U));
    std::size_t alignedRows = rowBudget;
    if (blockHeight > 1) {
        const std::size_t sourceBlockRows =
            static_cast<std::size_t>(blockHeight);
        alignedRows = sourceBlockRows <= rowBudget
            ? (rowBudget / sourceBlockRows) * sourceBlockRows
            : 1U;
    }
    const int readRows = static_cast<int>(std::max<std::size_t>(alignedRows, 1U));
    std::vector<double> rasterBlock(
        static_cast<std::size_t>(windowWidth) *
        static_cast<std::size_t>(readRows));
    for (int blockY = yStart; blockY < yStop; blockY += readRows) {
        const int currentRows = std::min(readRows, yStop - blockY);
        if (band->RasterIO(GF_Read, xStart, blockY, windowWidth, currentRows,
                           rasterBlock.data(), windowWidth, currentRows,
                           GDT_Float64, 0, 0, nullptr) != CE_None) {
            throw std::runtime_error("Failed reading DEM GeoTIFF pixels");
        }
        for (int localY = 0; localY < currentRows; ++localY) {
            const int y = blockY + localY;
            const std::size_t rowOffset =
                static_cast<std::size_t>(localY) *
                static_cast<std::size_t>(windowWidth);
            for (int x = xStart; x < xStop; ++x) {
                const double value = rasterBlock[
                    rowOffset + static_cast<std::size_t>(x - xStart)];
                if (!std::isfinite(value) || value == -9999.0 ||
                    (hasNoData && value == noData)) {
                    continue;
                }
                const double centerX =
                    transform[0] + (static_cast<double>(x) + 0.5) * transform[1] +
                    (static_cast<double>(y) + 0.5) * transform[2];
                const double centerY =
                    transform[3] + (static_cast<double>(x) + 0.5) * transform[4] +
                    (static_cast<double>(y) + 0.5) * transform[5];
                bool covered = false;
                if (axisAlignedCoverage) {
                    covered = centerX >= *coverageMinXIt &&
                              centerX <= *coverageMaxXIt &&
                              centerY >= *coverageMinYIt &&
                              centerY <= *coverageMaxYIt;
                } else {
                    OGRPoint center(centerX, centerY);
                    covered = preparedCoverage
                        ? OGRPreparedGeometryIntersects(
                              preparedCoverage.get(),
                              OGRGeometry::ToHandle(&center)) != 0
                        : coveragePolygon.Intersects(&center);
                }
                if (!covered) continue;
                dem.points.push_back({centerX, centerY});
                dem.elevations.push_back(value);
            }
        }
    }
    if (dem.points.empty()) throw std::runtime_error("No valid elevation cells found within expanded bbox");
    if (!rasterReference->IsSame(&workReference)) {
        output << "[INFO] Reprojecting DEM from " << referenceText(*rasterReference)
               << " to " << referenceText(workReference) << '\n';
        TransformPtr toWork = makeTransform(rasterReference.get(), &workReference, "DEM raster reprojection");
        constexpr std::size_t chunkSize = 65536U;
        std::vector<double> chunkX(std::min(chunkSize, dem.points.size()));
        std::vector<double> chunkY(chunkX.size());
        for (std::size_t start = 0; start < dem.points.size(); start += chunkSize) {
            const std::size_t count = std::min(chunkSize, dem.points.size() - start);
            for (std::size_t i = 0; i < count; ++i) {
                chunkX[i] = dem.points[start + i].x;
                chunkY[i] = dem.points[start + i].y;
            }
            if (!toWork->Transform(static_cast<int>(count), chunkX.data(), chunkY.data())) {
                throw std::runtime_error("Failed reprojecting DEM pixels");
            }
            for (std::size_t i = 0; i < count; ++i) dem.points[start + i] = {chunkX[i], chunkY[i]};
        }
    }
    const auto [minimum, maximum] = std::minmax_element(dem.elevations.begin(), dem.elevations.end());
    const double datum = *minimum;
    for (double& value : dem.elevations) value -= datum;
    output << "[INFO] Loaded " << dem.elevations.size() << " DEM points\n"
           << "[INFO] Original elevation range (sea level): " << fixed(datum, 2) << " to "
           << fixed(*maximum, 2) << " meters\n"
           << "[INFO] Adjusted to datum (lowest point = 0): 0.00 to "
           << fixed(*maximum - datum, 2) << " meters\n"
           << "[INFO] Datum offset applied: " << fixed(datum, 2) << "m\n";
    return dem;
}

std::unique_ptr<OGRSpatialReference> vectorReference(const fs::path& path) {
    DatasetPtr dataset(static_cast<GDALDataset*>(GDALOpenEx(
        pathToUtf8(path).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
    if (!dataset || dataset->GetLayer(0) == nullptr) return {};
    return cloneReference(dataset->GetLayer(0)->GetSpatialRef());
}

struct Domain {
    double minX = 0.0;
    double minY = 0.0;
    double maxX = 0.0;
    double maxY = 0.0;
    double rotateDegrees = 0.0;
    Vec2 pivot;
    std::array<Vec2, 4> projectedCorners{};
};

Domain configDomain(double lonMin, double lonMax, double latMin, double latMax,
                    OGRSpatialReference& workReference,
                    const std::optional<double>& configuredRotation) {
    auto wgs84 = epsgReference(4326);
    TransformPtr transform = makeTransform(wgs84.get(), &workReference, "configuration domain projection");
    std::array<double, 4> x{lonMin, lonMax, lonMax, lonMin};
    std::array<double, 4> y{latMin, latMin, latMax, latMax};
    if (!transform->Transform(4, x.data(), y.data())) {
        throw std::runtime_error("Failed to project configuration domain corners");
    }
    Domain domain;
    for (std::size_t i = 0; i < 4U; ++i) domain.projectedCorners[i] = {x[i], y[i]};
    domain.pivot = {(x[0] + x[1] + x[2] + x[3]) * 0.25,
                    (y[0] + y[1] + y[2] + y[3]) * 0.25};
    domain.rotateDegrees = configuredRotation.value_or(
        -std::atan2(y[1] - y[0], x[1] - x[0]) * 180.0 / std::numbers::pi);
    domain.minX = domain.minY = std::numeric_limits<double>::infinity();
    domain.maxX = domain.maxY = -std::numeric_limits<double>::infinity();
    for (const Vec2& corner : domain.projectedCorners) {
        const Vec2 rotated = rotatePoint(corner, domain.rotateDegrees, domain.pivot);
        domain.minX = std::min(domain.minX, rotated.x);
        domain.maxX = std::max(domain.maxX, rotated.x);
        domain.minY = std::min(domain.minY, rotated.y);
        domain.maxY = std::max(domain.maxY, rotated.y);
    }
    return domain;
}

void translatePolygon(Polygon2D& polygon, double xOffset, double yOffset) {
    for (Vec2& point : polygon.exterior) { point.x += xOffset; point.y += yOffset; }
    for (auto& hole : polygon.holes) {
        for (Vec2& point : hole) { point.x += xOffset; point.y += yOffset; }
    }
}

std::pair<std::vector<Polygon2D>, std::vector<double>>
prepareBuildings(std::vector<BuildingRecord>& records, double minimumHeight,
                 double rotateDegrees, Vec2 pivot, const Domain& domain,
                 std::optional<int> maximumBuildings, std::ostream& output,
                 ProgressEmitter& progress) {
    progress.stage(QStringLiteral("filter_buildings"),
                   QStringLiteral("Filtering buildings by height"));
    output << "[INFO] Filtering building polygons by height\n";
    std::vector<GeometryPtr> rotated;
    std::vector<double> rotatedHeights;
    progress.progress(QStringLiteral("filter_buildings"), 0,
                      static_cast<std::int64_t>(std::max<std::size_t>(records.size(), 1U)),
                      QStringLiteral("Filtering buildings 0/%1")
                          .arg(static_cast<qulonglong>(records.size())), {}, true);
    for (std::size_t index = 0; index < records.size(); ++index) {
        BuildingRecord& record = records[index];
        if (std::isfinite(record.height) && record.height > minimumHeight && record.geometry &&
            !record.geometry->IsEmpty()) {
            GeometryPtr valid = makeValidPolygonal(record.geometry.get());
            if (valid && !valid->IsEmpty()) {
                affineGeometry(valid.get(), rotateDegrees, pivot);
                rotated.push_back(std::move(valid));
                rotatedHeights.push_back(record.height);
            }
        }
        progress.progress(QStringLiteral("filter_buildings"),
                          static_cast<std::int64_t>(index + 1U),
                          static_cast<std::int64_t>(std::max<std::size_t>(records.size(), 1U)),
                          QStringLiteral("Filtering buildings %1/%2")
                              .arg(static_cast<qulonglong>(index + 1U))
                              .arg(static_cast<qulonglong>(records.size())));
    }
    if (rotated.empty()) throw std::runtime_error("[ERROR] No valid building polygons available");

    progress.stage(QStringLiteral("clip_buildings"),
                   QStringLiteral("Clipping buildings to CFD domain"));
    output << "[INFO] Clipping buildings to boundary box\n";
    GeometryPtr boundary = rectangleGeometry(domain.minX, domain.minY, domain.maxX, domain.maxY);
    std::vector<Polygon2D> polygons;
    std::vector<double> heights;
    progress.progress(QStringLiteral("clip_buildings"), 0,
                      static_cast<std::int64_t>(std::max<std::size_t>(rotated.size(), 1U)),
                      QStringLiteral("Clipping polygons 0/%1")
                          .arg(static_cast<qulonglong>(rotated.size())), {}, true);
    for (std::size_t index = 0; index < rotated.size(); ++index) {
        GeometryPtr clipped(rotated[index]->Intersection(boundary.get()));
        if (clipped && !clipped->IsEmpty() &&
            OGR_G_Area(reinterpret_cast<OGRGeometryH>(clipped.get())) > 0.0) {
            std::vector<Polygon2D> pieces;
            appendPolygons(clipped.get(), pieces);
            for (Polygon2D& polygon : pieces) {
                polygons.push_back(std::move(polygon));
                heights.push_back(rotatedHeights[index]);
            }
        }
        progress.progress(QStringLiteral("clip_buildings"),
                          static_cast<std::int64_t>(index + 1U),
                          static_cast<std::int64_t>(std::max<std::size_t>(rotated.size(), 1U)),
                          QStringLiteral("Clipping polygons %1/%2")
                              .arg(static_cast<qulonglong>(index + 1U))
                              .arg(static_cast<qulonglong>(rotated.size())));
    }
    output << "[INFO] Clipped buildings: " << polygons.size() << " polygons remaining\n";
    if (!heights.empty()) {
        const auto [minimum, maximum] = std::minmax_element(heights.begin(), heights.end());
        output << "[INFO] Building height stats: min " << fixed(*minimum, 3) << "m, max "
               << fixed(*maximum, 3) << "m\n";
    }
    if (maximumBuildings &&
        (*maximumBuildings < 0 ||
         polygons.size() > static_cast<std::size_t>(*maximumBuildings))) {
        const int requested = *maximumBuildings;
        output << "[INFO] Limiting to first " << requested
               << " buildings (--max-buildings parameter)\n";
        // Python's sequence[:negative] keeps all but the trailing |negative|
        // entries; preserve that argparse-visible edge case as well.
        const std::size_t retained = requested >= 0
            ? static_cast<std::size_t>(requested)
            : (static_cast<std::size_t>(-static_cast<std::int64_t>(requested)) >=
                       polygons.size()
                   ? 0U
                   : polygons.size() - static_cast<std::size_t>(
                                           -static_cast<std::int64_t>(requested)));
        polygons.resize(retained);
        heights.resize(retained);
    }
    return {std::move(polygons), std::move(heights)};
}

void appendAll(std::vector<Triangle>& destination, std::vector<Triangle>&& source) {
    destination.insert(destination.end(),
                       std::make_move_iterator(source.begin()),
                       std::make_move_iterator(source.end()));
}

void updateDomainMetadata(DeckDocument& deck, const fs::path& conf,
                          int utmCode, double rotateDegrees,
                          double sizeX, double sizeY, double terrainTop,
                          double meshTop, double baseHeight,
                          std::ostream& output) {
    double sizeZ = meshTop;
    if (const std::optional<double> zLimit = deck.floating(QStringLiteral("z_limit"));
        zLimit && std::isfinite(*zLimit) && *zLimit > 0.0) {
        sizeZ = terrainTop + *zLimit;
    }
    sizeZ = std::max({sizeZ, meshTop, baseHeight});
    deck.setPair(QStringLiteral("si_x_cfd"), {0.0, sizeX}, 6);
    deck.setPair(QStringLiteral("si_y_cfd"), {0.0, sizeY}, 6);
    deck.setPair(QStringLiteral("si_z_cfd"), {0.0, sizeZ}, 6);
    deck.setText(QStringLiteral("utm_crs"), QStringLiteral("EPSG:%1").arg(utmCode), true);
    deck.setInteger(QStringLiteral("utm_epsg"), utmCode);
    deck.setInteger(QStringLiteral("utm"), utmCode);
    const int zone = utmCode >= 32701 ? utmCode - 32700 : utmCode - 32600;
    if (zone >= 1 && zone <= 60) deck.setInteger(QStringLiteral("utm_zone"), zone);
    deck.setText(QStringLiteral("utm_hemisphere"), utmCode >= 32701 ? QStringLiteral("S")
                                                                     : QStringLiteral("N"));
    deck.setFloat(QStringLiteral("rotate_deg"), rotateDegrees, 6);
    if (const auto longitude = deck.pair(QStringLiteral("cut_lon_manual"))) {
        deck.setFloat(QStringLiteral("center_lon"), (longitude->first + longitude->second) * 0.5, 8);
    }
    if (const auto latitude = deck.pair(QStringLiteral("cut_lat_manual"))) {
        deck.setFloat(QStringLiteral("center_lat"), (latitude->first + latitude->second) * 0.5, 8);
    }
    deck.save(pathToQString(conf));
    output << "[INFO] Wrote projected domain metadata to config: si_x=" << fixed(sizeX, 3)
           << "m, si_y=" << fixed(sizeY, 3) << "m, si_z=" << fixed(sizeZ, 3)
           << "m, utm_crs=EPSG:" << utmCode << ", rotate_deg="
           << fixed(rotateDegrees, 6) << '\n';
}

int runLuwvoxWithGdal(const CommandArguments& arguments, std::ostream& output,
                      std::ostream& error) {
    if (hasHelpOption(arguments)) {
        printHelp(output);
        return 0;
    }
    Options options;
    try {
        options = parseOptions(arguments);
    } catch (const std::invalid_argument& exception) {
        printUsage(error);
        error << "luwvox: error: " << exception.what() << '\n';
        return 2;
    }

    const auto started = std::chrono::steady_clock::now();
    output << "[INFO] Parsed CLI arguments\n";
    ProgressEmitter progress(QStringLiteral("Generating Voxel Domain"));
    progress.stage(QStringLiteral("load_shapefile"),
                   QStringLiteral("Loading cropped building geometries"));
    std::error_code absoluteError;
    options.conf = fs::absolute(options.conf, absoluteError).lexically_normal();
    if (absoluteError || !fs::is_regular_file(options.conf)) {
        throw std::runtime_error("conf file not found: " + pathToUtf8(options.conf));
    }
    const fs::path projectHome = options.conf.parent_path();
    const fs::path projectTemp = projectHome / "proj_temp";

    DeckDocument deck = DeckDocument::load(pathToQString(options.conf));
    const std::string caseName = deck.textOr(QStringLiteral("casename")).toUtf8().toStdString();
    if (caseName.empty()) throw std::runtime_error("casename not found in conf");
    const auto longitude = deck.pair(QStringLiteral("cut_lon_manual"));
    const auto latitude = deck.pair(QStringLiteral("cut_lat_manual"));
    if (!longitude || !latitude) throw std::runtime_error("conf 中未找到 cut_lon_manual/cut_lat_manual");
    const double lonMin = longitude->first, lonMax = longitude->second;
    const double latMin = latitude->first, latMax = latitude->second;
    const double lonCenter = (lonMin + lonMax) * 0.5;
    const double latCenter = (latMin + latMax) * 0.5;
    const int utmCode = utmEpsg(lonCenter, latCenter);
    output << "[INFO] Base height: ";

    TerrainVoxelOverrides overrides;
    if (options.terrainApproach) overrides.approach = qString(*options.terrainApproach);
    if (options.heightField) overrides.heightField = qString(*options.heightField);
    overrides.ignoreUnder = options.minimumHeight;
    overrides.gridResolution = options.terrainResolution;
    overrides.idwSigma = options.smoothSigma;
    overrides.idwPower = options.idwPower;
    overrides.idwNeighbors = options.idwNeighbors;
    auto [voxelConfig, sources] = resolveTerrainVoxelConfig(
        &deck, overrides, [&](const QString& warning) {
            output << "[WARN] " << warning.toUtf8().toStdString() << '\n';
        });

    auto [computeConfig, computeSources] = resolvePreprocessComputeConfig(
        &deck, {}, [&](const QString& warning) {
            output << "[WARN] " << warning.toUtf8().toStdString() << '\n';
        });
    output << "[INFO] Preprocess compute policy: "
           << preprocessComputePolicyText(computeConfig).toStdString()
           << " (use_gpu source: "
           << computeSources.value(QStringLiteral("use_gpu")).toStdString()
           << ", device source: "
           << computeSources.value(QStringLiteral("gpu_device")).toStdString()
           << ")\n";
    if (computeConfig.useGpu) {
        output
            << "[INFO] CPU topology stages: vector geometry repair/triangulation, "
               "building extrusion, and final heterogeneous mesh combination. "
               "OpenCL is used for DEM neighbor lookup, interpolation, smoothing, "
               "and regular terrain-grid triangle generation.\n";
    }

    if (computeConfig.useGpu) {
        const bool compilationExpected =
            nativeOpenClVoxelRequiresWarmup(computeConfig.gpuDevice);
        if (compilationExpected) {
            output << "[INFO] OpenCL voxel kernels are compiling during GPU "
                      "backend preflight. This one-time warmup may take a little longer.\n";
        }
        try {
            const OpenClVoxelBackendInfo backend =
                requireNativeOpenClVoxelBackend(computeConfig.gpuDevice);
            if (backend.compiledThisCall) {
                output << "[INFO] OpenCL voxel kernel preflight compilation finished.\n";
            }
            output << "[INFO] OpenCL voxel backend preflight (GPU device "
                   << computeConfig.gpuDevice << "): "
                   << backend.device << '\n';
        } catch (const OpenClKrigingError& exception) {
            throw GpuPreprocessUnavailable(
                "GPU preprocessing was explicitly requested, but no usable OpenCL "
                "GPU backend is available: " + std::string(exception.what()));
        }
    }
    std::error_code directoryError;
    fs::create_directories(projectTemp, directoryError);
    if (directoryError) {
        throw std::runtime_error("Cannot create proj_temp: " +
                                 directoryError.message());
    }
    double baseHeight = options.baseHeight.value_or(
        deck.floating(QStringLiteral("base_height")).value_or(50.0));
    if (!std::isfinite(baseHeight) || baseHeight < 0.0) baseHeight = 50.0;
    output << general(baseHeight) << " m\n"
           << "[INFO] UTM CRS for CFD domain: EPSG:" << utmCode << '\n';

    const fs::path primaryBuilding = projectTemp / "cutted_shp" / (caseName + ".shp");
    const fs::path fallbackBuilding = projectTemp / (caseName + ".shp");
    fs::path buildingPath;
    if (fs::is_regular_file(primaryBuilding)) {
        buildingPath = primaryBuilding;
    } else if (fs::is_regular_file(fallbackBuilding)) {
        output << "[WARN] Primary shapefile not found: " << pathToUtf8(primaryBuilding.filename())
               << ". Fallback to " << pathToUtf8(fallbackBuilding.filename()) << '\n';
        buildingPath = fallbackBuilding;
    } else {
        throw std::runtime_error("Shapefile not found: " + pathToUtf8(primaryBuilding) +
                                 " or " + pathToUtf8(fallbackBuilding));
    }
    output << "[INFO] Input Shapefile: " << pathToUtf8(buildingPath) << '\n'
           << "[INFO] Terrain approach: " << voxelConfig.approach.toStdString()
           << " (source: " << sources.value(QStringLiteral("approach")).toStdString() << ")\n";
    const std::string configuredHeightField = voxelConfig.heightField.toUtf8().toStdString();
    output << "[INFO] Height field: "
           << (lowerAscii(configuredHeightField) == "auto" ? "Inferred" : configuredHeightField)
           << " (source: " << sources.value(QStringLiteral("height_field")).toStdString() << ")\n"
           << "[INFO] Min valid height: " << general(voxelConfig.ignoreUnder)
           << " m (source: " << sources.value(QStringLiteral("ignore_under")).toStdString() << ")\n"
           << "[INFO] Terrain grid resolution: " << general(voxelConfig.gridResolution)
           << " m (source: " << sources.value(QStringLiteral("grid_resolution")).toStdString() << ")\n"
           << "[INFO] Terrain smoothing sigma: " << general(voxelConfig.idwSigma)
           << " (source: " << sources.value(QStringLiteral("idw_sigma")).toStdString() << ")\n"
           << "[INFO] IDW power: " << general(voxelConfig.idwPower)
           << " (source: " << sources.value(QStringLiteral("idw_power")).toStdString() << ")\n"
           << "[INFO] Neighboring points: " << voxelConfig.idwNeighbors
           << " (source: " << sources.value(QStringLiteral("idw_neighbors")).toStdString() << ")\n"
           << "[INFO] Reading features from file\n";

    auto utmReference = epsgReference(utmCode);
    std::unique_ptr<OGRSpatialReference> workReference;
    if (options.noReproject) {
        workReference = vectorReference(buildingPath);
        if (!workReference) workReference = epsgReference(4326);
    } else {
        workReference = cloneReference(utmReference.get());
    }
    std::string heightField = configuredHeightField;
    std::vector<BuildingRecord> buildings = loadBuildings(
        buildingPath, *workReference, options.noReproject, heightField, output);
    output << "[INFO] Working CRS for CFD domain: " << referenceText(*workReference) << '\n';

    Domain domain = configDomain(lonMin, lonMax, latMin, latMax, *workReference,
                                 deck.floating(QStringLiteral("rotate_deg")));

    DemData dem;
    auto loadSpatialDem = [&]() -> bool {
        fs::path selected;
        if (options.demPath) {
            selected = *options.demPath;
            if (selected.is_relative()) selected = fs::absolute(selected).lexically_normal();
            output << "[INFO] Using user-provided DEM shapefile: " << pathToUtf8(selected) << '\n';
        } else {
            for (const std::string_view folder : {std::string_view("terrain_db"), std::string_view("DEM")}) {
                const auto files = sortedFiles(projectHome / folder, {".shp"});
                if (!files.empty()) {
                    selected = files.front();
                    output << "[INFO] Auto-detected DEM file: " << pathToUtf8(selected) << '\n';
                    break;
                }
                if (fs::is_directory(projectHome / folder)) {
                    output << "[WARN] No DEM shapefile found in " << folder << " folder\n";
                }
            }
        }
        if (!selected.empty() && fs::is_regular_file(selected)) {
            const std::string extension = lowerAscii(pathToUtf8(selected.extension()));
            dem = extension == ".tif" || extension == ".tiff"
                ? loadDemRaster(selected, *workReference, lonMin, lonMax, latMin, latMax, output)
                : loadDemVector(selected, *workReference, output, progress);
            return !dem.points.empty();
        }
        output << "[INFO] No DEM shapefile found. Trying GeoTIFF fallback...\n";
        const QString installed = resolveInstallFile(QStringLiteral("database/dem.tif"));
        const DemTifCacheRequest request{
            projectHome,
            pathFromUtf8(installed.toUtf8().toStdString()),
            caseName,
            {lonMin, lonMax, latMin, latMax}};
        try {
            const auto cached = ensureDemShapefileFromTif(request, output);
            if (!cached || !fs::is_regular_file(cached->shapefile)) return false;
            dem = loadDemVector(cached->shapefile, *workReference, output, progress);
            return !dem.points.empty();
        } catch (const DemTifCacheCancelled&) {
            throw;
        } catch (const std::exception& exception) {
            output << "[WARN] Failed to process GeoTIFF: "
                   << exception.what() << '\n';
            return false;
        }
    };

    if (options.demSource == "auto" || options.demSource == "shp") {
        progress.stage(QStringLiteral("load_dem"), QStringLiteral("Loading terrain DEM"));
        output << "[INFO] Loading DEM from shapefile/GeoTIFF ...\n";
        (void)loadSpatialDem();
    }
    const fs::path picklePath = projectTemp / "dem_grid.pkl";
    if (dem.points.empty() && (options.demSource == "auto" || options.demSource == "pkl" ||
                               options.demSource == "shp")) {
        if (options.demSource == "shp") {
            output << "[WARN] DEM shapefile/GeoTIFF not available. Falling back to PKL if present.\n";
        }
        progress.stage(QStringLiteral("load_dem"), QStringLiteral("Loading cached terrain DEM"));
        output << "[INFO] Loading DEM from PKL ...\n";
        if (fs::is_regular_file(picklePath)) {
            try {
                dem = loadDemPickle(picklePath, output);
                output << "[INFO] Using DEM data from pkl file (dem_grid.pkl)\n";
                if (dem.pklBaseHeight && std::abs(*dem.pklBaseHeight - baseHeight) > 1.0e-6) {
                    output << "[WARN] PKL base_height=" << fixed(*dem.pklBaseHeight, 3)
                           << "m differs from current base_height=" << fixed(baseHeight, 3)
                           << "m. STL will use base_height=" << fixed(baseHeight, 3) << "m.\n";
                }
            } catch (const std::exception& exception) {
                output << "[WARN] Failed to load DEM pkl: " << exception.what() << '\n';
            }
        }
    }
    if (dem.points.empty()) output << "[INFO] No DEM data available, will create flat terrain\n";

    const std::string normalizedHeightField = lowerAscii(configuredHeightField);
    output << "[INFO] " << (normalizedHeightField == "auto" ||
                                     normalizedHeightField == "inferred"
                                 ? "Auto-detected height field: "
                                 : "Using specified height field: ")
           << heightField << '\n'
           << "[INFO] Config bounds: lon=[" << general(lonMin) << ", " << general(lonMax)
           << "], lat=[" << general(latMin) << ", " << general(latMax) << "]\n"
           << "[INFO] Projected corners (UTM): (" << fixed(domain.projectedCorners[0].x, 3)
           << ',' << fixed(domain.projectedCorners[0].y, 3) << ") ("
           << fixed(domain.projectedCorners[1].x, 3) << ',' << fixed(domain.projectedCorners[1].y, 3)
           << ") (" << fixed(domain.projectedCorners[2].x, 3) << ','
           << fixed(domain.projectedCorners[2].y, 3) << ") ("
           << fixed(domain.projectedCorners[3].x, 3) << ','
           << fixed(domain.projectedCorners[3].y, 3) << ")\n"
           << "[INFO] Rotation from conf/config: rotate_deg=" << fixed(domain.rotateDegrees, 6)
           << ", pivot=(" << fixed(domain.pivot.x, 3) << ", " << fixed(domain.pivot.y, 3)
           << ")\n"
           << "[INFO] Domain size in rotated frame: " << fixed(domain.maxX - domain.minX, 3)
           << " x " << fixed(domain.maxY - domain.minY, 3) << " m\n";
    if (dem.fromPkl && dem.xMin && dem.yMin && dem.xMax && dem.yMax) {
        domain.minX = *dem.xMin; domain.minY = *dem.yMin;
        domain.maxX = *dem.xMax; domain.maxY = *dem.yMax;
        domain.rotateDegrees = dem.rotateDegrees.value_or(0.0);
        domain.pivot = dem.pivot.value_or(Vec2{(domain.minX + domain.maxX) * 0.5,
                                               (domain.minY + domain.maxY) * 0.5});
        output << "[INFO] Using pkl DEM bounds: x=[" << fixed(domain.minX, 3) << ", "
               << fixed(domain.maxX, 3) << "], y=[" << fixed(domain.minY, 3) << ", "
               << fixed(domain.maxY, 3) << "]\n"
               << "[INFO] Using pkl rotation: rotate_deg=" << fixed(domain.rotateDegrees, 6)
               << ", pivot=(" << fixed(domain.pivot.x, 3) << ", "
               << fixed(domain.pivot.y, 3) << ")\n";
    } else {
        output << "[INFO] Using config-rotated bounding box (match buildBC CSV frame)\n";
        if (!dem.points.empty() && domain.rotateDegrees != 0.0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dem.points.size()); ++i) {
                dem.points[static_cast<std::size_t>(i)] =
                    rotatePoint(dem.points[static_cast<std::size_t>(i)], domain.rotateDegrees, domain.pivot);
            }
        }
    }

    auto [polygons, heights] = prepareBuildings(
        buildings, voxelConfig.ignoreUnder, domain.rotateDegrees, domain.pivot,
        domain, options.maximumBuildings, output, progress);
    const double translationX = -domain.minX;
    const double translationY = -domain.minY;
    for (Polygon2D& polygon : polygons) translatePolygon(polygon, translationX, translationY);
    for (Vec2& point : dem.points) { point.x += translationX; point.y += translationY; }
    const double sizeX = domain.maxX - domain.minX;
    const double sizeY = domain.maxY - domain.minY;
    output << "[INFO] Translation offsets: dx " << fixed(translationX, 3) << ", dy "
           << fixed(translationY, 3) << '\n';

    progress.stage(QStringLiteral("create_base"),
                   QStringLiteral("Creating base layer to %1 m").arg(baseHeight, 0, 'f', 2));
    output << "[INFO] Creating base layer from z=0 to z=" << general(baseHeight) << "m\n";
    std::vector<Triangle> baseMesh = extrudePolygon(rectangle(0.0, 0.0, sizeX, sizeY), baseHeight);
    output << "[INFO] Base layer created\n";

    Grid terrainGrid;
    Grid rawTerrainGrid;
    std::vector<Triangle> terrainMesh;
    progress.stage(QStringLiteral("terrain_mesh"), QStringLiteral("Creating terrain mesh"));
    if (dem.points.empty()) {
        output << "[INFO] No DEM data, creating flat base\n";
    } else {
        terrainGrid = interpolateSurface(
            0.0, 0.0, sizeX, sizeY, dem, voxelConfig.gridResolution,
            options.elevationScale, voxelConfig.idwSigma, 1.0,
            voxelConfig.approach.toStdString(), voxelConfig.idwPower,
            voxelConfig.idwNeighbors, computeConfig.useGpu, computeConfig.gpuDevice,
            nullptr, &rawTerrainGrid,
            "terrain mesh", "terrain_mesh",
            "Creating terrain mesh", output, progress);
        output << "[INFO] Grid size: " << terrainGrid.nx << " x " << terrainGrid.ny
               << " = " << terrainGrid.z.size() << " points\n";
        terrainMesh = computeConfig.useGpu
            ? terrainTrianglesOpenCl(terrainGrid, baseHeight,
                                     computeConfig.gpuDevice, output)
            : terrainTriangles(terrainGrid, baseHeight);
        writeDemCsv(projectTemp, terrainGrid, output);
        output << "[INFO] Terrain mesh created: " << terrainGrid.z.size() * 2U
               << " vertices, " << terrainMesh.size() << " faces\n"
               << "[INFO] Terrain is a solid volume from z=" << general(baseHeight)
               << "m to terrain surface\n";
    }
    const bool demUsed = !terrainGrid.z.empty();
    const fs::path combinedPath = projectTemp /
        (caseName + std::string(demUsed ? "_DEM.stl" : ".stl"));
    output << "[INFO] Combined STL output path: " << pathToUtf8(combinedPath) << '\n';

    progress.stage(QStringLiteral("terrain_lookup"),
                   QStringLiteral("Preparing terrain lookup grid"));
    Grid lookupGrid;
    if (!dem.points.empty()) {
        lookupGrid = interpolateSurface(
            0.0, 0.0, sizeX, sizeY, dem, voxelConfig.gridResolution,
            options.elevationScale, voxelConfig.idwSigma, 0.5,
            voxelConfig.approach.toStdString(), voxelConfig.idwPower,
            voxelConfig.idwNeighbors, computeConfig.useGpu, computeConfig.gpuDevice,
            &rawTerrainGrid, nullptr,
            "elevation lookup", "terrain_lookup",
            "Preparing terrain lookup", output, progress);
        output << "[INFO] Elevation lookup grid created: " << lookupGrid.nx << " x "
               << lookupGrid.ny << " = " << lookupGrid.z.size() << " points\n";
    }
    double terrainTop = baseHeight;
    if (demUsed) {
        const auto [minimum, maximum] = std::minmax_element(terrainGrid.z.begin(), terrainGrid.z.end());
        output << "[INFO] Terrain layer elevation range: " << fixed(baseHeight + *minimum, 2)
               << "m to " << fixed(baseHeight + *maximum, 2) << "m\n"
               << "[INFO] Terrain relief: " << fixed(*maximum - *minimum, 2) << "m\n";
        terrainTop = baseHeight + *maximum;
    }

    const std::size_t totalBuildings = polygons.size();
    progress.progress(QStringLiteral("voxelize"), 0,
                      static_cast<std::int64_t>(std::max<std::size_t>(totalBuildings, 1U)),
                      QStringLiteral("Extruding buildings 0/%1")
                          .arg(static_cast<qulonglong>(totalBuildings)), {}, true);
    output << "[INFO] Extruding " << totalBuildings << " buildings in parallel...\n";
    const unsigned int hardware = std::max(1U, std::thread::hardware_concurrency());
    output << "[INFO] Using up to " << std::min(64U, hardware) << " parallel workers\n";
    const std::size_t batchCount = (totalBuildings + 499U) / 500U;
    if (!progress.enabled()) {
        output << "[INFO] Using " << batchCount << " batches for 500 parallel tasks |";
        output.flush();
    }
    const auto extrusionStarted = std::chrono::steady_clock::now();
    std::vector<std::vector<Triangle>> buildingMeshes(totalBuildings);
    std::vector<double> buildingBases(totalBuildings, std::numeric_limits<double>::quiet_NaN());
    std::vector<std::string> buildingErrors(totalBuildings);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(std::min(32U, hardware))
#endif
    for (std::ptrdiff_t linear = 0; linear < static_cast<std::ptrdiff_t>(totalBuildings); ++linear) {
        const std::size_t index = static_cast<std::size_t>(linear);
        try {
            double terrainElevation = 0.0;
            if (!polygons[index].exterior.empty()) {
                terrainElevation = std::numeric_limits<double>::infinity();
                for (const Vec2& point : polygons[index].exterior) {
                    terrainElevation = std::min(terrainElevation,
                                                lookupElevation(lookupGrid, point.x, point.y));
                }
                if (!std::isfinite(terrainElevation)) terrainElevation = 0.0;
            }
            buildingBases[index] = baseHeight + terrainElevation;
            buildingMeshes[index] = extrudePolygon(polygons[index], heights[index], buildingBases[index]);
        } catch (const std::exception& exception) {
            buildingErrors[index] = exception.what();
        }
    }
    if (!progress.enabled()) {
        if (totalBuildings == 0U) {
            output << "|Finished|\n";
        } else {
            output << "####|20%|####|40%|####|60%|####|80%|####|100%||Finished|\n";
        }
    }
    progress.progress(QStringLiteral("voxelize"), static_cast<std::int64_t>(totalBuildings),
                      static_cast<std::int64_t>(std::max<std::size_t>(totalBuildings, 1U)),
                      QStringLiteral("Extruding buildings %1/%1")
                          .arg(static_cast<qulonglong>(totalBuildings)), {}, true);
    std::size_t failed = 0U;
    std::vector<double> successfulBases;
    std::size_t successful = 0U;
    for (std::size_t index = 0; index < totalBuildings; ++index) {
        if (!buildingErrors[index].empty()) {
            ++failed;
            if (failed <= 5U) output << "[WARN] Building " << index
                                     << " extrusion failed: " << buildingErrors[index] << '\n';
        } else {
            ++successful;
            successfulBases.push_back(buildingBases[index]);
        }
    }
    const double extrusionSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - extrusionStarted).count();
    output << "[INFO] Extruding all buildings took " << fixed(extrusionSeconds, 2) << " seconds\n";
    if (failed != 0U) output << "[WARN] Total " << failed << " buildings failed to extrude\n";
    output << "[INFO] Extruded " << successful << " building meshes\n";
    if (!successfulBases.empty()) {
        const auto [minimum, maximum] = std::minmax_element(successfulBases.begin(), successfulBases.end());
        const double mean = std::accumulate(successfulBases.begin(), successfulBases.end(), 0.0) /
                            static_cast<double>(successfulBases.size());
        double variance = 0.0;
        for (double value : successfulBases) variance += (value - mean) * (value - mean);
        variance /= static_cast<double>(successfulBases.size());
        output << "[INFO] Building base elevation stats:\n"
               << "       Min: " << fixed(*minimum, 2) << "m, Max: " << fixed(*maximum, 2) << "m\n"
               << "       Mean: " << fixed(mean, 2) << "m, Std: " << fixed(std::sqrt(variance), 2) << "m\n";
    }

    std::vector<Triangle> finalMesh;
    std::size_t triangleCount = baseMesh.size() + terrainMesh.size();
    for (const auto& mesh : buildingMeshes) triangleCount += mesh.size();
    finalMesh.reserve(triangleCount);
    appendAll(finalMesh, std::move(baseMesh));
    if (demUsed) appendAll(finalMesh, std::move(terrainMesh));
    for (auto& mesh : buildingMeshes) appendAll(finalMesh, std::move(mesh));
    if (!demUsed && successful == 0U) {
        output << "[INFO] Only base layer, no terrain or buildings\n";
    } else {
        output << "[INFO] Combining layers: base";
        if (demUsed) output << ", terrain";
        if (successful != 0U) output << ", " << successful << " buildings";
        output << "\n[INFO] Mesh combination succeeded\n";
    }
    progress.stage(QStringLiteral("save_stl"),
                   QStringLiteral("Saving STL: %1").arg(qString(pathToUtf8(combinedPath.filename()))));
    writeBinaryStl(combinedPath, finalMesh);
    progress.complete(QStringLiteral("save_stl"),
                      QStringLiteral("Saved STL: %1").arg(qString(pathToUtf8(combinedPath.filename()))));
    output << "[INFO] Exported STL with terrain: "
           << pathToUtf8(fs::absolute(combinedPath).lexically_normal()) << '\n';

    output << "[RESULT] Base rectangle corners in meters\n"
           << "[RESULT] Lower left: (0.000, 0.000)\n"
           << "[RESULT] Lower right: (" << fixed(sizeX, 3) << ", 0.000)\n"
           << "[RESULT] Upper right: (" << fixed(sizeX, 3) << ", " << fixed(sizeY, 3) << ")\n"
           << "[RESULT] Upper left: (0.000, " << fixed(sizeY, 3) << ")\n";
    double zMinimum = std::numeric_limits<double>::infinity();
    double zMaximum = -std::numeric_limits<double>::infinity();
    for (const Triangle& triangle : finalMesh) {
        for (const Vec3& vertex : {triangle.a, triangle.b, triangle.c}) {
            zMinimum = std::min(zMinimum, vertex.z);
            zMaximum = std::max(zMaximum, vertex.z);
        }
    }
    if (!std::isfinite(zMinimum)) zMinimum = zMaximum = 0.0;
    updateDomainMetadata(deck, options.conf, utmCode, domain.rotateDegrees,
                         sizeX, sizeY, terrainTop, zMaximum, baseHeight, output);
    output << "[RESULT] STL Z range: min " << fixed(zMinimum, 3) << "m, max "
           << fixed(zMaximum, 3) << "m\n"
           << "[RESULT] Total height: " << fixed(zMaximum - zMinimum, 3) << "m\n"
           << "[INFO] Program total time: "
           << fixed(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(), 2)
           << " seconds\n";
    return 0;
}

#endif // LUW_HAS_GDAL

} // namespace

int run_luwvox(const CommandArguments& arguments, std::ostream& output,
               std::ostream& error) {
    if (hasHelpOption(arguments)) {
        printHelp(output);
        return 0;
    }
#if LUW_HAS_GDAL
    CommandArguments resolved = arguments;
    if (resolved.empty() && !resolveDeckArgument("luwvox", resolved, error)) return 2;
    try {
        return runLuwvoxWithGdal(resolved, output, error);
    } catch (const DemTifCacheCancelled&) {
        return 1;
    } catch (const GpuPreprocessUnavailable& exception) {
        error << "luwvox: error: " << exception.what() << '\n';
        return 1;
    } catch (const std::invalid_argument& exception) {
        error << "luwvox: error: invalid preprocessing configuration: "
              << exception.what() << '\n';
        return 1;
    }
#else
    try {
        (void)parseOptions(arguments);
    } catch (const std::invalid_argument& exception) {
        printUsage(error);
        error << "luwvox: error: " << exception.what() << '\n';
        return 2;
    }
    (void)output;
    error << "ERROR: Native luwvox requires GDAL/OGR with GEOS support; rebuild with "
             "LUW_ENABLE_GEOSPATIAL=ON and a discoverable GDAL installation.\n";
    return 1;
#endif
}

} // namespace luw::cli
