#include "luw/vtk_post_commands.hpp"

#include "luw/deck_document.hpp"
#include "luw/legacy_vtk.hpp"
#include "post_internal.hpp"

#include <QVector>

#include <vtkErrorCode.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtk_netcdf.h>
#include <vtk_zlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <regex>
#include <set>
#include <span>
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
using luw::post::LegacyVtkScalarType;
using luw::post::legacyVtkLittleEndianScalarAsFloat;
using luw::post::detail::NetcdfFile;
using luw::post::detail::Projection;
using luw::post::detail::defineCompressedDoubleVariable;
using luw::post::detail::defineCompressedFloatVariable;
using luw::post::detail::ncCheck;
using luw::post::detail::pathUtf8;
using luw::post::detail::setTextAttribute;

class SoftExit final : public std::runtime_error {
public:
    explicit SoftExit(std::string message, int code = 1)
        : std::runtime_error(std::move(message)), code_(code) {}
    [[nodiscard]] int code() const noexcept { return code_; }
private:
    int code_ = 1;
};

struct Arguments final {
    fs::path configPath;
    std::optional<double> lonMin;
    std::optional<double> lonMax;
    std::optional<double> latMin;
    std::optional<double> latMax;
    std::optional<int> layers;
    std::optional<fs::path> outputDirectory;
    std::optional<fs::path> ncOutput;
    bool exportNc = false;
    bool help = false;
};

struct Config final {
    std::string casename;
    std::string datetime;
    std::optional<std::vector<double>> cutLon;
    std::optional<std::vector<double>> cutLat;
    std::optional<std::vector<double>> siX;
    std::optional<std::vector<double>> siY;
    std::optional<std::int64_t> utmEpsg;
    std::optional<std::int64_t> utm;
    std::optional<std::int64_t> utmZone;
    std::string utmHemisphere;
    std::optional<double> centerLon;
    std::optional<double> centerLat;
};

struct Crop final {
    double lonMin = 0.0;
    double lonMax = 0.0;
    double latMin = 0.0;
    double latMax = 0.0;
};

struct GridData final {
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    std::vector<float> wind;
    // Empty for the normal float32 FluidX3D path. Other VTK scalar types are
    // retained exactly, in C-order and little-endian form, for wind_data.npy.
    std::vector<std::uint8_t> windLittleEndian;
    LegacyVtkScalarType windScalarType = LegacyVtkScalarType::Float32;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> zVtk;
    std::vector<double> heights;
    double zminVtk = 0.0;
    double zmaxVtk = 0.0;
    double centerLon = 0.0;
    double centerLat = 0.0;
    std::pair<double, double> centerUtm{};
    std::string vectorName;
};

void usage(std::ostream& output) {
    output << "usage: visluw [-h] [--lon-min LON_MIN] [--lon-max LON_MAX] "
              "[--lat-min LAT_MIN] [--lat-max LAT_MAX] [--layers LAYERS] "
              "[--output-dir OUTPUT_DIR] [--export-nc] [--nc-output NC_OUTPUT] config_path\n\n"
              "Generate visualization figures from a LUW result.\n\n"
              "positional arguments:\n"
              "  config_path           Path to the LUW deck file\n\n"
              "options:\n"
              "  -h, --help            show this help message and exit\n"
              "  --lon-min LON_MIN     Crop longitude minimum\n"
              "  --lon-max LON_MAX     Crop longitude maximum\n"
              "  --lat-min LAT_MIN     Crop latitude minimum\n"
              "  --lat-max LAT_MAX     Crop latitude maximum\n"
              "  --layers LAYERS       Number of section layers to export\n"
              "  --output-dir OUTPUT_DIR\n"
              "                        Export root directory for NPZ and section figures\n"
              "  --export-nc           Also export a 3D NetCDF file\n"
              "  --nc-output NC_OUTPUT Optional NetCDF output path\n";
}

double parseDouble(std::string_view text, std::string_view option) {
    std::string owned(text);
    std::size_t consumed = 0;
    try {
        const double value = std::stod(owned, &consumed);
        if (consumed == owned.size()) return value;
    } catch (const std::exception&) {
    }
    throw SoftExit("argument " + std::string(option) + ": invalid float value: '" + owned + "'", 2);
}

int parseInt(std::string_view text, std::string_view option) {
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw SoftExit("argument " + std::string(option) + ": invalid int value: '" + std::string(text) + "'", 2);
    }
    return value;
}

Arguments parseArguments(const std::vector<std::string>& arguments) {
    Arguments parsed;
    bool haveConfig = false;
    auto requiredValue = [&](std::size_t& index, std::string_view option) -> const std::string& {
        if (index + 1 >= arguments.size()) {
            throw SoftExit("argument " + std::string(option) + ": expected one argument", 2);
        }
        return arguments[++index];
    };
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "-h" || argument == "--help") {
            parsed.help = true;
        } else if (argument == "--export-nc") {
            parsed.exportNc = true;
        } else if (argument == "--lon-min") {
            parsed.lonMin = parseDouble(requiredValue(index, argument), argument);
        } else if (argument == "--lon-max") {
            parsed.lonMax = parseDouble(requiredValue(index, argument), argument);
        } else if (argument == "--lat-min") {
            parsed.latMin = parseDouble(requiredValue(index, argument), argument);
        } else if (argument == "--lat-max") {
            parsed.latMax = parseDouble(requiredValue(index, argument), argument);
        } else if (argument == "--layers") {
            parsed.layers = parseInt(requiredValue(index, argument), argument);
        } else if (argument == "--output-dir") {
            parsed.outputDirectory = fs::path(requiredValue(index, argument));
        } else if (argument == "--nc-output") {
            parsed.ncOutput = fs::path(requiredValue(index, argument));
        } else if (argument.starts_with("--lon-min=")) {
            parsed.lonMin = parseDouble(argument.substr(10), "--lon-min");
        } else if (argument.starts_with("--lon-max=")) {
            parsed.lonMax = parseDouble(argument.substr(10), "--lon-max");
        } else if (argument.starts_with("--lat-min=")) {
            parsed.latMin = parseDouble(argument.substr(10), "--lat-min");
        } else if (argument.starts_with("--lat-max=")) {
            parsed.latMax = parseDouble(argument.substr(10), "--lat-max");
        } else if (argument.starts_with("--layers=")) {
            parsed.layers = parseInt(argument.substr(9), "--layers");
        } else if (argument.starts_with("--output-dir=")) {
            parsed.outputDirectory = fs::path(argument.substr(13));
        } else if (argument.starts_with("--nc-output=")) {
            parsed.ncOutput = fs::path(argument.substr(12));
        } else if (!argument.empty() && argument.front() == '-') {
            throw SoftExit("unrecognized arguments: " + argument, 2);
        } else if (haveConfig) {
            throw SoftExit("unrecognized arguments: " + argument, 2);
        } else {
            parsed.configPath = fs::path(argument);
            haveConfig = true;
        }
    }
    if (!parsed.help && !haveConfig) {
        throw SoftExit("the following arguments are required: config_path", 2);
    }
    return parsed;
}

fs::path resolved(const fs::path& path) {
    std::error_code error;
    fs::path result = fs::absolute(path, error);
    if (error) result = path;
    return result.lexically_normal();
}

std::string number(double value, int precision = 15) {
    std::ostringstream stream;
    stream << std::setprecision(precision) << value;
    std::string text = stream.str();
    if (std::isfinite(value) && text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}

std::string fixed(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::optional<std::vector<double>> list2(const luw::DeckDocument& deck, const QString& key) {
    const QVector<double> values = deck.floatList(key);
    if (values.size() != 2) return std::nullopt;
    return std::vector<double>{values[0], values[1]};
}

Config loadConfig(const fs::path& path) {
    const std::string utf8 = pathUtf8(path);
    const luw::DeckDocument deck = luw::DeckDocument::load(
        QString::fromUtf8(utf8.data(), static_cast<qsizetype>(utf8.size())));
    Config result;
    if (const auto value = deck.text(QStringLiteral("casename"))) result.casename = value->trimmed().toStdString();
    if (const auto value = deck.integer(QStringLiteral("datetime"))) result.datetime = std::to_string(*value);
    result.cutLon = list2(deck, QStringLiteral("cut_lon_manual"));
    result.cutLat = list2(deck, QStringLiteral("cut_lat_manual"));
    result.siX = list2(deck, QStringLiteral("si_x_cfd"));
    result.siY = list2(deck, QStringLiteral("si_y_cfd"));
    result.utmEpsg = deck.integer(QStringLiteral("utm_epsg"));
    result.utm = deck.integer(QStringLiteral("utm"));
    result.utmZone = deck.integer(QStringLiteral("utm_zone"));
    result.utmHemisphere = deck.textOr(QStringLiteral("utm_hemisphere"), QStringLiteral("N")).toStdString();
    result.centerLon = deck.floating(QStringLiteral("center_lon"));
    result.centerLat = deck.floating(QStringLiteral("center_lat"));
    return result;
}

int inferEpsg(const Config& config,
              const std::optional<double>& fallbackLon,
              const std::optional<double>& fallbackLat) {
    if (config.utmEpsg) return static_cast<int>(*config.utmEpsg);
    if (config.utm) return static_cast<int>(*config.utm);
    if (config.utmZone) {
        std::string hemisphere = config.utmHemisphere;
        std::transform(hemisphere.begin(), hemisphere.end(), hemisphere.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        return (hemisphere == "N" ? 32600 : 32700) + static_cast<int>(*config.utmZone);
    }
    if (fallbackLon && fallbackLat) {
        const int zone = static_cast<int>(std::floor((*fallbackLon + 180.0) / 6.0)) + 1;
        return (*fallbackLat >= 0.0 ? 32600 : 32700) + zone;
    }
    throw SoftExit("Cannot infer UTM EPSG from LUW. Please provide utm_epsg, utm, or utm_zone/utm_hemisphere.");
}

double promptFloat(std::ostream& output, std::string_view prompt,
                   const std::optional<double>& defaultValue) {
    while (true) {
        output << prompt;
        if (defaultValue) output << " [default " << number(*defaultValue) << ']';
        output << ": " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) throw SoftExit("Aborted by user during input.");
        const auto first = line.find_first_not_of(" \t\r\n");
        const auto last = line.find_last_not_of(" \t\r\n");
        line = first == std::string::npos ? std::string{} : line.substr(first, last - first + 1);
        if (line.empty() && defaultValue) return *defaultValue;
        try {
            std::size_t consumed = 0;
            const double value = std::stod(line, &consumed);
            if (consumed == line.size()) return value;
        } catch (const std::exception&) {
        }
        output << "Invalid number. Please re-enter.\n";
    }
}

int promptPositiveInt(std::ostream& output, std::string_view prompt, int defaultValue) {
    while (true) {
        output << prompt << " [default " << defaultValue << "]: " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) throw SoftExit("Aborted by user during input.");
        const auto first = line.find_first_not_of(" \t\r\n");
        const auto last = line.find_last_not_of(" \t\r\n");
        line = first == std::string::npos ? std::string{} : line.substr(first, last - first + 1);
        if (line.empty()) return defaultValue;
        int value = 0;
        const auto [end, status] = std::from_chars(line.data(), line.data() + line.size(), value);
        if (status == std::errc{} && end == line.data() + line.size()) {
            if (value > 0) return value;
            output << "Please enter a positive integer.\n";
        } else {
            output << "Invalid integer. Please re-enter.\n";
        }
    }
}

bool confirm(std::ostream& output, std::string_view prompt) {
    output << prompt << ' ' << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) return false;
    const auto first = line.find_first_not_of(" \t\r\n");
    const auto last = line.find_last_not_of(" \t\r\n");
    line = first == std::string::npos ? std::string{} : line.substr(first, last - first + 1);
    std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return line.empty() || line == "y" || line == "yes";
}

fs::path findResultsDirectory(const fs::path& configPath) {
    const fs::path first = configPath.parent_path() / "RESULTS";
    const fs::path second = configPath.parent_path().parent_path() / "RESULTS";
    if (fs::is_directory(first)) return first;
    if (fs::is_directory(second)) return second;
    fs::create_directories(first);
    return first;
}

std::pair<fs::path, std::string> locateVtk(const fs::path& results,
                                           const std::string& casename,
                                           const std::string& datetime) {
    const std::string uvwPrefix = "uvw-" + casename + '_' + datetime;
    const std::string rhoPrefix = "uvwrho-" + casename + '_' + datetime;
    const fs::path uvw = results / (uvwPrefix + ".vtk");
    const fs::path uvwrho = results / (rhoPrefix + ".vtk");
    if (fs::is_regular_file(uvw)) return {uvw, uvw.stem().string()};
    if (fs::is_regular_file(uvwrho)) return {uvwrho, uvwrho.stem().string()};
    for (const fs::directory_entry& entry : fs::directory_iterator(results)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".vtk") continue;
        const std::string name = entry.path().filename().string();
        if (name.starts_with(uvwPrefix) || name.starts_with(rhoPrefix)) {
            return {entry.path(), entry.path().stem().string()};
        }
    }
    throw SoftExit("VTK not found in RESULTS. Expected " + uvwPrefix + ".vtk or " + rhoPrefix + ".vtk");
}

const LegacyVtkField& selectVector(const LegacyStructuredPoints& vtk) {
    const std::array<std::string_view, 6> priority{"U", "velocity", "Velocity", "UVW", "data", "Data"};
    for (const std::string_view wanted : priority) {
        for (const LegacyVtkField& field : vtk.fields) {
            if (field.association == LegacyVtkAssociation::PointData
                && field.name == wanted && field.components == 3) return field;
        }
    }
    for (const LegacyVtkField& field : vtk.fields) {
        std::string lower = field.name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (field.association == LegacyVtkAssociation::PointData
            && lower != "data_rho" && field.components == 3) return field;
    }
    std::ostringstream names;
    names << '[';
    bool first = true;
    for (const LegacyVtkField& field : vtk.fields) {
        if (!first) names << ", ";
        names << '\'' << field.name << '\'';
        first = false;
    }
    names << ']';
    throw SoftExit("No 3-component vector array found. Available arrays: " + names.str());
}

std::vector<double> axis(double origin, double spacing, int count) {
    std::vector<double> values(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) values[static_cast<std::size_t>(index)] = origin + index * spacing;
    return values;
}

std::vector<std::size_t> indicesIn(const std::vector<double>& values, double minimum, double maximum) {
    std::vector<std::size_t> indices;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index] >= minimum && values[index] <= maximum) indices.push_back(index);
    }
    return indices;
}

double scaleFor(const std::optional<std::vector<double>>& configured,
                double vtkMinimum, double vtkMaximum) {
    if (!configured || configured->size() != 2) return 1.0;
    return ((*configured)[1] - (*configured)[0]) / std::max(vtkMaximum - vtkMinimum, 1.0e-12);
}

GridData extract(const LegacyStructuredPoints& vtk,
                 const LegacyVtkField& field,
                 const Config& config,
                 const Crop& requested,
                 double centerLon,
                 double centerLat,
                 Projection& forward,
                 std::ostream& output) {
    output << "[INFO] Mesh dimensions: (" << vtk.dimensions[0] << ", " << vtk.dimensions[1]
           << ", " << vtk.dimensions[2] << ")\n";
    output << "[INFO] Mesh origin: (" << number(vtk.origin[0]) << ", " << number(vtk.origin[1])
           << ", " << number(vtk.origin[2]) << ")\n";
    output << "[INFO] Mesh spacing: (" << number(vtk.spacing[0]) << ", " << number(vtk.spacing[1])
           << ", " << number(vtk.spacing[2]) << ")\n";

    const auto xFull = axis(vtk.origin[0], vtk.spacing[0], vtk.dimensions[0]);
    const auto yFull = axis(vtk.origin[1], vtk.spacing[1], vtk.dimensions[1]);
    const auto zFull = axis(vtk.origin[2], vtk.spacing[2], vtk.dimensions[2]);
    const auto [xBoundMin, xBoundMax] = std::minmax_element(xFull.begin(), xFull.end());
    const auto [yBoundMin, yBoundMax] = std::minmax_element(yFull.begin(), yFull.end());
    const auto [zBoundMin, zBoundMax] = std::minmax_element(zFull.begin(), zFull.end());
    const double zmin = *zBoundMin;
    const double zmax = *zBoundMax;
    output << "[INFO] Mesh bounds: X [" << fixed(*xBoundMin, 3) << ", " << fixed(*xBoundMax, 3)
           << "], Y [" << fixed(*yBoundMin, 3) << ", " << fixed(*yBoundMax, 3)
           << "], Z [" << fixed(zmin, 3) << ", " << fixed(zmax, 3) << "]\n";

    const double xFullMin = vtk.origin[0];
    const double xFullMax = vtk.origin[0] + (vtk.dimensions[0] - 1) * vtk.spacing[0];
    const double yFullMin = vtk.origin[1];
    const double yFullMax = vtk.origin[1] + (vtk.dimensions[1] - 1) * vtk.spacing[1];
    const double sx = scaleFor(config.siX, xFullMin, xFullMax);
    const double sy = scaleFor(config.siY, yFullMin, yFullMax);
    const auto centerUtm = forward.apply(centerLon, centerLat);
    const auto lowerUtm = forward.apply(requested.lonMin, requested.latMin);
    const auto upperUtm = forward.apply(requested.lonMax, requested.latMax);
    const double xMappedA = xFullMin + (lowerUtm.first - centerUtm.first) / sx;
    const double xMappedB = xFullMin + (upperUtm.first - centerUtm.first) / sx;
    const double yMappedA = yFullMin + (lowerUtm.second - centerUtm.second) / sy;
    const double yMappedB = yFullMin + (upperUtm.second - centerUtm.second) / sy;
    const double xCropMin = std::min(xMappedA, xMappedB);
    const double xCropMax = std::max(xMappedA, xMappedB);
    const double yCropMin = std::min(yMappedA, yMappedB);
    const double yCropMax = std::max(yMappedA, yMappedB);

    std::vector<std::size_t> zIndices;
    for (std::size_t index = 0; index < zFull.size(); ++index) {
        if (zFull[index] - (zmin + 50.0) >= 0.0) zIndices.push_back(index);
    }
    if (zIndices.empty()) {
        throw SoftExit("After removing the ground slab [zmin, zmin+50), no vertical slices remain.");
    }
    output << "[INFO] Planned crop in VTK-relative coordinates:\n"
           << "       X [" << fixed(xCropMin, 3) << ", " << fixed(xCropMax, 3) << "]\n"
           << "       Y [" << fixed(yCropMin, 3) << ", " << fixed(yCropMax, 3) << "]\n"
           << "       Ground reference: zmin + 50 m; kept vertical layers >= 0 m real height\n";

    const std::vector<std::size_t> xIndices = indicesIn(xFull, xCropMin, xCropMax);
    const std::vector<std::size_t> yIndices = indicesIn(yFull, yCropMin, yCropMax);
    if (xIndices.empty() || yIndices.empty() || zIndices.empty()) {
        throw SoftExit("Crop box does not intersect the dataset after index mapping.");
    }

    GridData grid;
    grid.nx = xIndices.size();
    grid.ny = yIndices.size();
    grid.nz = zIndices.size();
    grid.wind.resize(grid.nx * grid.ny * grid.nz * 3U);
    grid.windScalarType = field.scalarType;
    if (field.scalarType != LegacyVtkScalarType::Float32) {
        if (grid.wind.size() > std::numeric_limits<std::size_t>::max() / field.scalarSize()) {
            throw std::runtime_error("Cropped wind array is too large");
        }
        grid.windLittleEndian.resize(grid.wind.size() * field.scalarSize());
    }
    grid.x.reserve(grid.nx);
    grid.y.reserve(grid.ny);
    grid.zVtk.reserve(grid.nz);
    grid.heights.reserve(grid.nz);
    for (const std::size_t index : xIndices) grid.x.push_back(xFull[index]);
    for (const std::size_t index : yIndices) grid.y.push_back(yFull[index]);
    for (const std::size_t index : zIndices) {
        grid.zVtk.push_back(zFull[index]);
        grid.heights.push_back(zFull[index] - (zmin + 50.0));
    }
    grid.zminVtk = zmin;
    grid.zmaxVtk = zmax;
    grid.centerLon = centerLon;
    grid.centerLat = centerLat;
    grid.centerUtm = centerUtm;
    grid.vectorName = field.name;

    LegacyVtkPlaneReader reader(vtk);
    for (std::size_t outputZ = 0; outputZ < zIndices.size(); ++outputZ) {
        if (field.scalarType == LegacyVtkScalarType::Float32) {
            const std::vector<float> plane = reader.readPointPlane(field, zIndices[outputZ]);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::int64_t ixSigned = 0; ixSigned < static_cast<std::int64_t>(grid.nx); ++ixSigned) {
                const std::size_t ix = static_cast<std::size_t>(ixSigned);
                for (std::size_t iy = 0; iy < grid.ny; ++iy) {
                    const std::size_t sourcePoint = yIndices[iy] * static_cast<std::size_t>(vtk.dimensions[0]) + xIndices[ix];
                    const std::size_t destination = ((ix * grid.ny + iy) * grid.nz + outputZ) * 3U;
                    for (std::size_t component = 0; component < 3; ++component) {
                        grid.wind[destination + component] = plane[sourcePoint * field.components + component];
                    }
                }
            }
            continue;
        }

        const std::vector<std::uint8_t> plane =
            reader.readPointPlaneLittleEndian(field, zIndices[outputZ]);
        const std::size_t scalarBytes = field.scalarSize();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t ixSigned = 0; ixSigned < static_cast<std::int64_t>(grid.nx); ++ixSigned) {
            const std::size_t ix = static_cast<std::size_t>(ixSigned);
            for (std::size_t iy = 0; iy < grid.ny; ++iy) {
                const std::size_t sourcePoint = yIndices[iy] * static_cast<std::size_t>(vtk.dimensions[0]) + xIndices[ix];
                const std::size_t destination = ((ix * grid.ny + iy) * grid.nz + outputZ) * 3U;
                for (std::size_t component = 0; component < 3; ++component) {
                    const std::uint8_t* source = plane.data()
                        + (sourcePoint * field.components + component) * scalarBytes;
                    std::uint8_t* exactDestination = grid.windLittleEndian.data()
                        + (destination + component) * scalarBytes;
                    std::copy_n(source, scalarBytes, exactDestination);
                    grid.wind[destination + component] =
                        legacyVtkLittleEndianScalarAsFloat(source, field.scalarType);
                }
            }
        }
    }
    output << "[INFO] Cropped wind array shape: (" << grid.nx << ", " << grid.ny << ", "
           << grid.nz << ", 3)\n";
    output << "[INFO] Lowest plotted height: "
           << fixed(*std::min_element(grid.heights.begin(), grid.heights.end()), 3) << " m\n";
    return grid;
}

// ---- NumPy NPY/NPZ compatibility -------------------------------------------------

void appendU16(std::vector<unsigned char>& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<unsigned char>(value & 0xffU));
    bytes.push_back(static_cast<unsigned char>((value >> 8U) & 0xffU));
}

void appendU32(std::vector<unsigned char>& bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xffU));
}

void appendU64(std::vector<unsigned char>& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xffU));
}

std::vector<unsigned char> npyHeader(std::string_view descriptor, std::string shape) {
    std::string dictionary = "{'descr': '" + std::string(descriptor)
        + "', 'fortran_order': False, 'shape': " + std::move(shape) + ", }";
    constexpr std::size_t prefix = 10;
    std::size_t padding = 64 - ((prefix + dictionary.size() + 1U) % 64U);
    if (padding == 64) padding = 0;
    dictionary.append(padding, ' ');
    dictionary.push_back('\n');
    if (dictionary.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("NPY header is too large");
    }
    std::vector<unsigned char> result{0x93U, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    appendU16(result, static_cast<std::uint16_t>(dictionary.size()));
    result.insert(result.end(), dictionary.begin(), dictionary.end());
    return result;
}

void pickleUnicode(std::vector<unsigned char>& output, std::string_view value) {
    output.push_back('X');
    appendU32(output, static_cast<std::uint32_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
}

void pickleFloat(std::vector<unsigned char>& output, double value) {
    output.push_back('G');
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    for (int shift = 56; shift >= 0; shift -= 8) output.push_back(static_cast<unsigned char>((bits >> shift) & 0xffU));
}

void pickleInt(std::vector<unsigned char>& output, std::size_t value) {
    if (value <= 255U) {
        output.push_back('K');
        output.push_back(static_cast<unsigned char>(value));
        return;
    }
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("NumPy compatibility array is too large for pickle protocol 2");
    }
    output.push_back('J');
    appendU32(output, static_cast<std::uint32_t>(value));
}

std::string latin1AsUtf8(std::span<const unsigned char> bytes) {
    std::string text;
    text.reserve(bytes.size() * 2U);
    for (const unsigned char byte : bytes) {
        if (byte < 0x80U) {
            text.push_back(static_cast<char>(byte));
        } else {
            text.push_back(static_cast<char>(0xc0U | (byte >> 6U)));
            text.push_back(static_cast<char>(0x80U | (byte & 0x3fU)));
        }
    }
    return text;
}

void pickleNumpyDoubleArray(std::vector<unsigned char>& output, std::span<const double> values) {
    static constexpr char reconstruct[] = "cnumpy.core.multiarray\n_reconstruct\n";
    static constexpr char ndarray[] = "cnumpy\nndarray\n";
    static constexpr char encode[] = "c_codecs\nencode\n";
    static constexpr char dtype[] = "cnumpy\ndtype\n";
    output.insert(output.end(), reconstruct, reconstruct + sizeof(reconstruct) - 1U);
    output.insert(output.end(), ndarray, ndarray + sizeof(ndarray) - 1U);
    output.push_back('K'); output.push_back(0); output.push_back(0x85U);
    output.insert(output.end(), encode, encode + sizeof(encode) - 1U);
    pickleUnicode(output, "b"); pickleUnicode(output, "latin1");
    output.push_back(0x86U); output.push_back('R'); output.push_back(0x87U); output.push_back('R');
    output.push_back('('); output.push_back('K'); output.push_back(1);
    pickleInt(output, values.size()); output.push_back(0x85U);
    output.insert(output.end(), dtype, dtype + sizeof(dtype) - 1U);
    pickleUnicode(output, "f8"); output.push_back(0x89U); output.push_back(0x88U);
    output.push_back(0x87U); output.push_back('R');
    output.push_back('('); output.push_back('K'); output.push_back(3);
    pickleUnicode(output, "<"); output.push_back('N'); output.push_back('N'); output.push_back('N');
    output.push_back('J'); appendU32(output, 0xffffffffU);
    output.push_back('J'); appendU32(output, 0xffffffffU);
    output.push_back('K'); output.push_back(0); output.push_back('t'); output.push_back('b');
    output.push_back(0x89U);
    std::vector<unsigned char> bytes(values.size() * sizeof(double));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(values[index]);
        for (unsigned byte = 0; byte < 8; ++byte) {
            bytes[index * 8U + byte] = static_cast<unsigned char>((bits >> (byte * 8U)) & 0xffU);
        }
    }
    output.insert(output.end(), encode, encode + sizeof(encode) - 1U);
    pickleUnicode(output, latin1AsUtf8(bytes)); pickleUnicode(output, "latin1");
    output.push_back(0x86U); output.push_back('R'); output.push_back('t'); output.push_back('b');
}

std::vector<unsigned char> gridInfoPickle(const GridData& grid) {
    // Protocol-2 ndarray reconstruction remains readable by NumPy 1.x and 2.x.
    // The outer 0-D object ndarray and the four float64 ndarray members match
    // np.savez_compressed(..., grid_info=dict) closely enough for existing
    // callers to keep using np.load(..., allow_pickle=True)[...].item().
    std::vector<unsigned char> result;
    static constexpr char prefix[] =
        "\x80\x02"
        "cnumpy.core.multiarray\n_reconstruct\nq\x00"
        "cnumpy\nndarray\nq\x01K\x00\x85q\x02"
        "c_codecs\nencode\nq\x03X\x01\x00\x00\x00" "b" "q\x04"
        "X\x06\x00\x00\x00" "latin1" "q\x05\x86q\x06Rq\x07\x87q\x08Rq\x09"
        "(K\x01)"
        "cnumpy\ndtype\nq\x0aX\x02\x00\x00\x00" "O8" "q\x0b\x89\x88\x87q\x0cRq\x0d"
        "(K\x03X\x01\x00\x00\x00" "|" "q\x0eNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?tq\x0f" "b"
        "\x89]q\x10}q\x11(";
    result.insert(result.end(), reinterpret_cast<const unsigned char*>(prefix),
                  reinterpret_cast<const unsigned char*>(prefix) + sizeof(prefix) - 1U);
    auto addList = [&](std::string_view key, const std::vector<double>& values) {
        pickleUnicode(result, key);
        pickleNumpyDoubleArray(result, values);
    };
    auto addFloat = [&](std::string_view key, double value) {
        pickleUnicode(result, key);
        pickleFloat(result, value);
    };
    addList("x_coords_vtk", grid.x);
    addList("y_coords_vtk", grid.y);
    addList("z_coords_vtk", grid.zVtk);
    addList("z_heights", grid.heights);
    addFloat("zmin_vtk", grid.zminVtk);
    addFloat("zmax_vtk", grid.zmaxVtk);
    addFloat("center_lon", grid.centerLon);
    addFloat("center_lat", grid.centerLat);
    pickleUnicode(result, "center_utm");
    pickleFloat(result, grid.centerUtm.first);
    pickleFloat(result, grid.centerUtm.second);
    result.push_back(0x86U); // TUPLE2
    pickleUnicode(result, "vector_name");
    pickleUnicode(result, grid.vectorName);
    const std::array<unsigned char, 5> suffix{'u', 'a', 't', 'b', '.'};
    result.insert(result.end(), suffix.begin(), suffix.end());
    return result;
}

std::uint32_t crc32Update(std::uint32_t crc, const unsigned char* data, std::size_t size) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> values{};
        for (std::uint32_t index = 0; index < values.size(); ++index) {
            std::uint32_t value = index;
            for (int bit = 0; bit < 8; ++bit) value = (value >> 1U) ^ (0xedb88320U & (0U - (value & 1U)));
            values[index] = value;
        }
        return values;
    }();
    for (std::size_t index = 0; index < size; ++index) crc = table[(crc ^ data[index]) & 0xffU] ^ (crc >> 8U);
    return crc;
}

struct BytePart final {
    const unsigned char* data = nullptr;
    std::uint64_t size = 0;
};

struct ZipEntry final {
    std::string name;
    std::vector<BytePart> parts;
    std::uint64_t uncompressedSize = 0;
    std::uint64_t compressedSize = 0;
    std::uint64_t localOffset = 0;
    std::uint32_t crc = 0;
};

void writeBytes(std::ofstream& output, const std::vector<unsigned char>& bytes) {
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("Failed while writing NPZ");
}

std::uint64_t writeDeflate(std::ofstream& output, const ZipEntry& entry) {
    z_stream stream{};
    const int initialized = deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                                         -MAX_WBITS, 8, Z_DEFAULT_STRATEGY);
    if (initialized != Z_OK) throw std::runtime_error("Cannot initialize zlib for NPZ compression");
    struct End final {
        z_stream* stream;
        ~End() { (void)deflateEnd(stream); }
    } end{&stream};
    std::array<unsigned char, 262144> compressed{};
    std::uint64_t written = 0;
    auto pump = [&](int flush) {
        int status = Z_OK;
        do {
            stream.next_out = compressed.data();
            stream.avail_out = static_cast<uInt>(compressed.size());
            status = deflate(&stream, flush);
            if (status != Z_OK && status != Z_STREAM_END) {
                throw std::runtime_error("zlib failed while compressing NPZ member");
            }
            const std::size_t produced = compressed.size() - stream.avail_out;
            output.write(reinterpret_cast<const char*>(compressed.data()),
                         static_cast<std::streamsize>(produced));
            written += produced;
        } while (stream.avail_in != 0 || (flush == Z_FINISH && status != Z_STREAM_END));
        return status;
    };
    for (const BytePart& part : entry.parts) {
        std::uint64_t offset = 0;
        while (offset < part.size) {
            const std::uint64_t chunk = std::min<std::uint64_t>(part.size - offset,
                                                                std::numeric_limits<uInt>::max());
            stream.next_in = const_cast<Bytef*>(part.data + offset);
            stream.avail_in = static_cast<uInt>(chunk);
            while (stream.avail_in != 0) (void)pump(Z_NO_FLUSH);
            offset += chunk;
        }
    }
    if (pump(Z_FINISH) != Z_STREAM_END) throw std::runtime_error("zlib did not finish the NPZ member");
    if (!output) throw std::runtime_error("Failed while writing compressed NPZ payload");
    return written;
}

void writeZip(const fs::path& path, std::vector<ZipEntry>& entries) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create NPZ: " + pathUtf8(path));
    for (ZipEntry& entry : entries) {
        entry.uncompressedSize = 0;
        std::uint32_t crc = 0xffffffffU;
        for (const BytePart& part : entry.parts) {
            entry.uncompressedSize += part.size;
            crc = crc32Update(crc, part.data, static_cast<std::size_t>(part.size));
        }
        entry.crc = crc ^ 0xffffffffU;
        entry.localOffset = static_cast<std::uint64_t>(output.tellp());
        // Deflate can expand incompressible input slightly; reserve ZIP64 near
        // the 4 GiB boundary even though the final size is not known yet.
        const bool zip64 = entry.uncompressedSize > 0xffe00000ULL;
        std::vector<unsigned char> header;
        appendU32(header, 0x04034b50U);
        appendU16(header, static_cast<std::uint16_t>(zip64 ? 45 : 20));
        appendU16(header, 8); // sizes and CRC follow in a data descriptor
        appendU16(header, 8); // raw DEFLATE, matching np.savez_compressed
        appendU16(header, 0); appendU16(header, 0);
        appendU32(header, 0);
        appendU32(header, zip64 ? 0xffffffffU : 0U);
        appendU32(header, zip64 ? 0xffffffffU : 0U);
        appendU16(header, static_cast<std::uint16_t>(entry.name.size()));
        appendU16(header, static_cast<std::uint16_t>(zip64 ? 20 : 0));
        header.insert(header.end(), entry.name.begin(), entry.name.end());
        if (zip64) {
            appendU16(header, 1); appendU16(header, 16);
            appendU64(header, entry.uncompressedSize); appendU64(header, 0);
        }
        writeBytes(output, header);
        entry.compressedSize = writeDeflate(output, entry);
        std::vector<unsigned char> descriptor;
        appendU32(descriptor, 0x08074b50U); appendU32(descriptor, entry.crc);
        if (zip64) {
            appendU64(descriptor, entry.compressedSize); appendU64(descriptor, entry.uncompressedSize);
        } else {
            appendU32(descriptor, static_cast<std::uint32_t>(entry.compressedSize));
            appendU32(descriptor, static_cast<std::uint32_t>(entry.uncompressedSize));
        }
        writeBytes(output, descriptor);
    }

    const std::uint64_t centralOffset = static_cast<std::uint64_t>(output.tellp());
    for (const ZipEntry& entry : entries) {
        const bool zip64 = entry.uncompressedSize > 0xffffffffULL || entry.compressedSize > 0xffffffffULL
            || entry.localOffset > 0xffffffffULL;
        std::vector<unsigned char> header;
        appendU32(header, 0x02014b50U);
        appendU16(header, 45); appendU16(header, static_cast<std::uint16_t>(zip64 ? 45 : 20));
        appendU16(header, 8); appendU16(header, 8);
        appendU16(header, 0); appendU16(header, 0);
        appendU32(header, entry.crc);
        appendU32(header, zip64 ? 0xffffffffU : static_cast<std::uint32_t>(entry.compressedSize));
        appendU32(header, zip64 ? 0xffffffffU : static_cast<std::uint32_t>(entry.uncompressedSize));
        appendU16(header, static_cast<std::uint16_t>(entry.name.size()));
        appendU16(header, static_cast<std::uint16_t>(zip64 ? 28 : 0));
        appendU16(header, 0); appendU16(header, 0); appendU16(header, 0);
        appendU32(header, 0);
        appendU32(header, zip64 ? 0xffffffffU : static_cast<std::uint32_t>(entry.localOffset));
        header.insert(header.end(), entry.name.begin(), entry.name.end());
        if (zip64) {
            appendU16(header, 1); appendU16(header, 24);
            appendU64(header, entry.uncompressedSize); appendU64(header, entry.compressedSize);
            appendU64(header, entry.localOffset);
        }
        writeBytes(output, header);
    }
    const std::uint64_t centralSize = static_cast<std::uint64_t>(output.tellp()) - centralOffset;
    const bool zip64 = centralOffset > 0xffffffffULL || centralSize > 0xffffffffULL;
    if (zip64) {
        const std::uint64_t zip64Offset = static_cast<std::uint64_t>(output.tellp());
        std::vector<unsigned char> record;
        appendU32(record, 0x06064b50U); appendU64(record, 44);
        appendU16(record, 45); appendU16(record, 45);
        appendU32(record, 0); appendU32(record, 0);
        appendU64(record, entries.size()); appendU64(record, entries.size());
        appendU64(record, centralSize); appendU64(record, centralOffset);
        appendU32(record, 0x07064b50U); appendU32(record, 0); appendU64(record, zip64Offset); appendU32(record, 1);
        writeBytes(output, record);
    }
    std::vector<unsigned char> end;
    appendU32(end, 0x06054b50U); appendU16(end, 0); appendU16(end, 0);
    appendU16(end, static_cast<std::uint16_t>(entries.size()));
    appendU16(end, static_cast<std::uint16_t>(entries.size()));
    appendU32(end, zip64 ? 0xffffffffU : static_cast<std::uint32_t>(centralSize));
    appendU32(end, zip64 ? 0xffffffffU : static_cast<std::uint32_t>(centralOffset));
    appendU16(end, 0);
    writeBytes(output, end);
}

void saveNpz(const fs::path& path, const GridData& grid) {
    const std::string shape = '(' + std::to_string(grid.nx) + ", " + std::to_string(grid.ny)
        + ", " + std::to_string(grid.nz) + ", 3)";
    const auto descriptor = [](LegacyVtkScalarType type) -> std::string_view {
        switch (type) {
        case LegacyVtkScalarType::Int8: return "|i1";
        case LegacyVtkScalarType::UInt8: return "|u1";
        case LegacyVtkScalarType::Int16: return "<i2";
        case LegacyVtkScalarType::UInt16: return "<u2";
        case LegacyVtkScalarType::Int32: return "<i4";
        case LegacyVtkScalarType::UInt32: return "<u4";
        case LegacyVtkScalarType::Float32: return "<f4";
        case LegacyVtkScalarType::Float64: return "<f8";
        }
        throw std::runtime_error("Unhandled legacy VTK scalar type");
    };
    std::vector<unsigned char> windHeader = npyHeader(descriptor(grid.windScalarType), shape);
    std::vector<unsigned char> object = npyHeader("|O", "()");
    const std::vector<unsigned char> pickle = gridInfoPickle(grid);
    object.insert(object.end(), pickle.begin(), pickle.end());
    const bool nativeFloat = grid.windScalarType == LegacyVtkScalarType::Float32;
    const auto* windBytes = nativeFloat
        ? reinterpret_cast<const unsigned char*>(grid.wind.data())
        : grid.windLittleEndian.data();
    const std::uint64_t windSize = nativeFloat
        ? static_cast<std::uint64_t>(grid.wind.size()) * sizeof(float)
        : static_cast<std::uint64_t>(grid.windLittleEndian.size());
    std::vector<ZipEntry> entries{
        {"wind_data.npy", {{windHeader.data(), windHeader.size()}, {windBytes, windSize}}},
        {"grid_info.npy", {{object.data(), object.size()}}},
    };
    writeZip(path, entries);
}

// ---- Lightweight deterministic plot rendering ----------------------------------

struct Color final { unsigned char red = 0; unsigned char green = 0; unsigned char blue = 0; };

class Canvas final {
public:
    Canvas(int width, int height) : width_(width), height_(height), pixels_(static_cast<std::size_t>(width * height), {255,255,255}) {}
    void pixel(int x, int y, Color color) {
        if (x < 0 || y < 0 || x >= width_ || y >= height_) return;
        pixels_[static_cast<std::size_t>(y * width_ + x)] = color;
    }
    void fill(int left, int top, int right, int bottom, Color color) {
        left = std::max(left, 0); top = std::max(top, 0); right = std::min(right, width_); bottom = std::min(bottom, height_);
        for (int y = top; y < bottom; ++y) for (int x = left; x < right; ++x) pixel(x, y, color);
    }
    void line(int x0, int y0, int x1, int y1, Color color) {
        const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        while (true) {
            pixel(x0, y0, color);
            if (x0 == x1 && y0 == y1) break;
            const int twice = 2 * error;
            if (twice >= dy) { error += dy; x0 += sx; }
            if (twice <= dx) { error += dx; y0 += sy; }
        }
    }
    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] const std::vector<Color>& pixels() const noexcept { return pixels_; }
private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Color> pixels_;
};

std::array<unsigned char, 7> glyph(char character) {
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    switch (c) {
    case 'A': return {14,17,17,31,17,17,17}; case 'B': return {30,17,17,30,17,17,30};
    case 'C': return {14,17,16,16,16,17,14}; case 'D': return {30,17,17,17,17,17,30};
    case 'E': return {31,16,16,30,16,16,31}; case 'F': return {31,16,16,30,16,16,16};
    case 'G': return {14,17,16,23,17,17,15}; case 'H': return {17,17,17,31,17,17,17};
    case 'I': return {31,4,4,4,4,4,31}; case 'J': return {7,2,2,2,18,18,12};
    case 'K': return {17,18,20,24,20,18,17}; case 'L': return {16,16,16,16,16,16,31};
    case 'M': return {17,27,21,21,17,17,17}; case 'N': return {17,25,21,19,17,17,17};
    case 'O': return {14,17,17,17,17,17,14}; case 'P': return {30,17,17,30,16,16,16};
    case 'Q': return {14,17,17,17,21,18,13}; case 'R': return {30,17,17,30,20,18,17};
    case 'S': return {15,16,16,14,1,1,30}; case 'T': return {31,4,4,4,4,4,4};
    case 'U': return {17,17,17,17,17,17,14}; case 'V': return {17,17,17,17,17,10,4};
    case 'W': return {17,17,17,21,21,21,10}; case 'X': return {17,17,10,4,10,17,17};
    case 'Y': return {17,17,10,4,4,4,4}; case 'Z': return {31,1,2,4,8,16,31};
    case '0': return {14,17,19,21,25,17,14}; case '1': return {4,12,4,4,4,4,14};
    case '2': return {14,17,1,2,4,8,31}; case '3': return {30,1,1,14,1,1,30};
    case '4': return {2,6,10,18,31,2,2}; case '5': return {31,16,16,30,1,1,30};
    case '6': return {14,16,16,30,17,17,14}; case '7': return {31,1,2,4,8,8,8};
    case '8': return {14,17,17,14,17,17,14}; case '9': return {14,17,17,15,1,1,14};
    case '.': return {0,0,0,0,0,6,6}; case ',': return {0,0,0,0,6,6,4};
    case ':': return {0,6,6,0,6,6,0}; case '-': return {0,0,0,31,0,0,0};
    case '+': return {0,4,4,31,4,4,0}; case '/': return {1,2,2,4,8,8,16};
    case '(': return {2,4,8,8,8,4,2}; case ')': return {8,4,2,2,2,4,8};
    case '[': return {14,8,8,8,8,8,14}; case ']': return {14,2,2,2,2,2,14};
    case '_': return {0,0,0,0,0,0,31}; case '=': return {0,31,0,31,0,0,0};
    default: return {};
    }
}

int textWidth(std::string_view text, int scale) { return static_cast<int>(text.size()) * 6 * scale; }

void drawText(Canvas& canvas, int x, int y, std::string_view text, int scale, Color color) {
    for (const char character : text) {
        const auto rows = glyph(character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[static_cast<std::size_t>(row)] & (1U << (4 - column))) == 0) continue;
                canvas.fill(x + column * scale, y + row * scale,
                            x + (column + 1) * scale, y + (row + 1) * scale, color);
            }
        }
        x += 6 * scale;
    }
}

Color jet(double normalized) {
    const double value = std::clamp(normalized, 0.0, 1.0);
    const auto channel = [value](double offset) {
        return static_cast<unsigned char>(std::lround(255.0 * std::clamp(1.5 - std::abs(4.0 * value - offset), 0.0, 1.0)));
    };
    return {channel(3.0), channel(2.0), channel(1.0)};
}

void writePng(const fs::path& path, const Canvas& canvas) {
    fs::create_directories(path.parent_path());
    vtkNew<vtkImageData> image;
    image->SetDimensions(canvas.width(), canvas.height(), 1);
    image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    for (int y = 0; y < canvas.height(); ++y) {
        auto* destination = static_cast<unsigned char*>(image->GetScalarPointer(0, canvas.height() - 1 - y, 0));
        for (int x = 0; x < canvas.width(); ++x) {
            const Color color = canvas.pixels()[static_cast<std::size_t>(y * canvas.width() + x)];
            destination[x * 3] = color.red; destination[x * 3 + 1] = color.green; destination[x * 3 + 2] = color.blue;
        }
    }
    vtkNew<vtkPNGWriter> writer;
    const std::string utf8 = pathUtf8(path);
    writer->SetFileName(utf8.c_str());
    writer->SetInputData(image);
    writer->Write();
    if (writer->GetErrorCode() != vtkErrorCode::NoError || !fs::is_regular_file(path)) {
        throw std::runtime_error("Failed to write PNG: " + utf8);
    }
}

std::vector<float> speedsAt(const GridData& grid, std::size_t zIndex) {
    std::vector<float> result(grid.nx * grid.ny);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t ixSigned = 0; ixSigned < static_cast<std::int64_t>(grid.nx); ++ixSigned) {
        const std::size_t ix = static_cast<std::size_t>(ixSigned);
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            const std::size_t source = ((ix * grid.ny + iy) * grid.nz + zIndex) * 3U;
            const double u = grid.wind[source], v = grid.wind[source + 1], w = grid.wind[source + 2];
            result[ix * grid.ny + iy] = static_cast<float>(std::sqrt(u*u + v*v + w*w));
        }
    }
    return result;
}

double globalMaximumSpeed(const GridData& grid) {
    double maximum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel
    {
        double localMaximum = 0.0;
#pragma omp for nowait schedule(static)
        for (std::int64_t point = 0; point < static_cast<std::int64_t>(grid.wind.size() / 3U); ++point) {
            const std::size_t source = static_cast<std::size_t>(point) * 3U;
            const double u = grid.wind[source], v = grid.wind[source + 1], w = grid.wind[source + 2];
            localMaximum = std::max(localMaximum, std::sqrt(u*u + v*v + w*w));
        }
#pragma omp critical(luw_visluw_maximum)
        maximum = std::max(maximum, localMaximum);
    }
#else
    for (std::size_t point = 0; point < grid.wind.size() / 3U; ++point) {
        const std::size_t source = point * 3U;
        const double u = grid.wind[source], v = grid.wind[source + 1], w = grid.wind[source + 2];
        maximum = std::max(maximum, std::sqrt(u*u + v*v + w*w));
    }
#endif
    return maximum;
}

struct GeographicGrid final {
    std::vector<double> longitude;
    std::vector<double> latitude;
    double lonMin = 0.0;
    double lonMax = 0.0;
    double latMin = 0.0;
    double latMax = 0.0;
};

GeographicGrid mapLonLat(const GridData& grid,
                         const LegacyStructuredPoints& vtk,
                         const Config& config,
                         Projection& inverse) {
    const double xFullMin = vtk.origin[0];
    const double xFullMax = vtk.origin[0] + (vtk.dimensions[0] - 1) * vtk.spacing[0];
    const double yFullMin = vtk.origin[1];
    const double yFullMax = vtk.origin[1] + (vtk.dimensions[1] - 1) * vtk.spacing[1];
    const double sx = scaleFor(config.siX, xFullMin, xFullMax);
    const double sy = scaleFor(config.siY, yFullMin, yFullMax);
    std::vector<PJ_COORD> coordinates(grid.nx * grid.ny);
    std::size_t cursor = 0;
    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            coordinates[cursor++] = proj_coord(
                grid.centerUtm.first + (grid.x[ix] - xFullMin) * sx,
                grid.centerUtm.second + (grid.y[iy] - yFullMin) * sy, 0.0, 0.0);
        }
    }
    inverse.apply(coordinates);
    GeographicGrid result;
    result.longitude.resize(coordinates.size()); result.latitude.resize(coordinates.size());
    result.lonMin = result.latMin = std::numeric_limits<double>::infinity();
    result.lonMax = result.latMax = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        result.longitude[index] = coordinates[index].xy.x;
        result.latitude[index] = coordinates[index].xy.y;
        result.lonMin = std::min(result.lonMin, result.longitude[index]);
        result.lonMax = std::max(result.lonMax, result.longitude[index]);
        result.latMin = std::min(result.latMin, result.latitude[index]);
        result.latMax = std::max(result.latMax, result.latitude[index]);
    }
    return result;
}

void renderSection(const fs::path& path, const GridData& grid,
                   const GeographicGrid& geographic, std::size_t zIndex,
                   double maximumSpeed) {
    // Matplotlib's 8x6 inch, 150 dpi figure is cropped by bbox_inches="tight"
    // to this canvas for the reference layout used by the legacy command.
    Canvas canvas(1053, 771);
    const Color black{0,0,0}; const Color gray{205,205,205};
    int left = 115, top = 75, width = 700, height = 570;
    const double lonRange = std::max(geographic.lonMax - geographic.lonMin, 1.0e-12);
    const double latRange = std::max(geographic.latMax - geographic.latMin, 1.0e-12);
    const double aspect = lonRange / latRange;
    if (static_cast<double>(width) / height > aspect) {
        const int desired = std::max(1, static_cast<int>(std::lround(height * aspect)));
        left += (width - desired) / 2; width = desired;
    } else {
        const int desired = std::max(1, static_cast<int>(std::lround(width / aspect)));
        top += (height - desired) / 2; height = desired;
    }
    const std::vector<float> speed = speedsAt(grid, zIndex);
    for (int py = 0; py < height; ++py) {
        const double sourceY = height == 1 ? 0.0 : (static_cast<double>(height - 1 - py) / (height - 1)) * (grid.ny - 1);
        const std::size_t y0 = static_cast<std::size_t>(std::floor(sourceY));
        const std::size_t y1 = std::min(y0 + 1, grid.ny - 1); const double fy = sourceY - y0;
        for (int px = 0; px < width; ++px) {
            const double sourceX = width == 1 ? 0.0 : (static_cast<double>(px) / (width - 1)) * (grid.nx - 1);
            const std::size_t x0 = static_cast<std::size_t>(std::floor(sourceX));
            const std::size_t x1 = std::min(x0 + 1, grid.nx - 1); const double fx = sourceX - x0;
            const double a = speed[x0 * grid.ny + y0] * (1.0 - fx) + speed[x1 * grid.ny + y0] * fx;
            const double b = speed[x0 * grid.ny + y1] * (1.0 - fx) + speed[x1 * grid.ny + y1] * fx;
            canvas.pixel(left + px, top + py, jet(maximumSpeed > 0.0 ? (a * (1.0 - fy) + b * fy) / maximumSpeed : 0.0));
        }
    }
    canvas.line(left, top, left + width, top, black); canvas.line(left, top + height, left + width, top + height, black);
    canvas.line(left, top, left, top + height, black); canvas.line(left + width, top, left + width, top + height, black);
    for (int tick = 0; tick <= 4; ++tick) {
        const int x = left + width * tick / 4; const int y = top + height - height * tick / 4;
        canvas.line(x, top + height, x, top + height + 7, black); canvas.line(left - 7, y, left, y, black);
        const std::string lon = fixed(geographic.lonMin + lonRange * tick / 4.0, 6);
        const std::string lat = fixed(geographic.latMin + latRange * tick / 4.0, 6);
        drawText(canvas, x - textWidth(lon, 1) / 2, top + height + 14, lon, 1, black);
        drawText(canvas, left - 12 - textWidth(lat, 1), y - 3, lat, 1, black);
    }
    const std::string title = "WIND SPEED AT " + fixed(grid.heights[zIndex], 1) + " M";
    drawText(canvas, left + width / 2 - textWidth(title, 3) / 2, 28, title, 3, black);
    drawText(canvas, left + width / 2 - textWidth("LONGITUDE (DEG)", 2) / 2, 730, "LONGITUDE (DEG)", 2, black);
    drawText(canvas, 18, 65, "LATITUDE (DEG)", 2, black);
    const int barX = 865, barY = top, barWidth = 32;
    for (int y = 0; y < height; ++y) canvas.fill(barX, barY + y, barX + barWidth, barY + y + 1, jet(1.0 - static_cast<double>(y) / std::max(height - 1, 1)));
    canvas.line(barX, barY, barX + barWidth, barY, black); canvas.line(barX, barY + height, barX + barWidth, barY + height, black);
    for (int tick = 0; tick <= 4; ++tick) {
        const int y = barY + height - height * tick / 4;
        canvas.line(barX + barWidth, y, barX + barWidth + 7, y, black);
        drawText(canvas, barX + barWidth + 12, y - 3, fixed(maximumSpeed * tick / 4.0, 2), 1, black);
    }
    drawText(canvas, 835, 690, "WIND SPEED (M/S)", 1, black);
    (void)gray;
    writePng(path, canvas);
}

void renderHistogram(const fs::path& path, const GridData& grid, double maximumSpeed) {
    // Reference 10x6 inch figure after Matplotlib tight bounding-box crop.
    Canvas canvas(1268, 817); const Color black{0,0,0}; const Color blue{80,130,220}; const Color gridColor{210,210,210};
    constexpr int binsCount = 50; std::array<std::uint64_t, binsCount> bins{}; std::uint64_t maximumBin = 0;
    double minimumFilteredSpeed = std::numeric_limits<double>::infinity();
    if (maximumSpeed > 0.1) {
        for (std::size_t point = 0; point < grid.wind.size() / 3U; ++point) {
            const std::size_t i = point * 3U;
            const double speed = std::sqrt(static_cast<double>(grid.wind[i]) * grid.wind[i]
                                           + static_cast<double>(grid.wind[i+1]) * grid.wind[i+1]
                                           + static_cast<double>(grid.wind[i+2]) * grid.wind[i+2]);
            if (speed > 0.1) minimumFilteredSpeed = std::min(minimumFilteredSpeed, speed);
        }
        const double histogramRange = std::max(maximumSpeed - minimumFilteredSpeed,
                                               std::numeric_limits<double>::epsilon());
        for (std::size_t point = 0; point < grid.wind.size() / 3U; ++point) {
            const std::size_t i = point * 3U;
            const double speed = std::sqrt(static_cast<double>(grid.wind[i]) * grid.wind[i]
                                           + static_cast<double>(grid.wind[i+1]) * grid.wind[i+1]
                                           + static_cast<double>(grid.wind[i+2]) * grid.wind[i+2]);
            if (speed <= 0.1) continue;
            const int bin = std::clamp(static_cast<int>((speed - minimumFilteredSpeed)
                                                        / histogramRange * binsCount), 0, binsCount - 1);
            maximumBin = std::max(maximumBin, ++bins[static_cast<std::size_t>(bin)]);
        }
    }
    const int left = 135, top = 80, width = 1050, height = 600;
    for (int tick = 0; tick <= 5; ++tick) {
        const int y = top + height - height * tick / 5; canvas.line(left, y, left + width, y, gridColor);
        drawText(canvas, left - 20 - textWidth(std::to_string(maximumBin * tick / 5), 1), y - 3,
                 std::to_string(maximumBin * tick / 5), 1, black);
    }
    for (int bin = 0; bin < binsCount; ++bin) {
        const int x0 = left + width * bin / binsCount; const int x1 = left + width * (bin + 1) / binsCount;
        const int bar = maximumBin == 0 ? 0 : static_cast<int>(height * bins[static_cast<std::size_t>(bin)] / maximumBin);
        canvas.fill(x0 + 1, top + height - bar, std::max(x0 + 2, x1 - 1), top + height, blue);
        canvas.line(x0 + 1, top + height - bar, x1 - 1, top + height - bar, black);
    }
    canvas.line(left, top + height, left + width, top + height, black); canvas.line(left, top, left, top + height, black);
    for (int tick = 0; tick <= 5; ++tick) {
        const int x = left + width * tick / 5; canvas.line(x, top + height, x, top + height + 7, black);
        const double lower = std::isfinite(minimumFilteredSpeed) ? minimumFilteredSpeed : 0.0;
        const std::string label = fixed(lower + (maximumSpeed - lower) * tick / 5.0, 2);
        drawText(canvas, x - textWidth(label, 1) / 2, top + height + 15, label, 1, black);
    }
    drawText(canvas, left + width / 2 - textWidth("WIND SPEED DISTRIBUTION", 3) / 2, 30,
             "WIND SPEED DISTRIBUTION", 3, black);
    drawText(canvas, left + width / 2 - textWidth("WIND SPEED (M/S)", 2) / 2, 780,
             "WIND SPEED (M/S)", 2, black);
    drawText(canvas, 20, 70, "FREQUENCY", 2, black);
    writePng(path, canvas);
}

std::vector<std::size_t> selectedLayers(std::size_t nz, int requested) {
    const std::size_t count = std::max<std::size_t>(1, std::min<std::size_t>(static_cast<std::size_t>(std::max(requested, 1)), nz));
    std::vector<std::size_t> layers;
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t layer = count == 1 ? 0 : static_cast<std::size_t>(
            static_cast<double>(index) * static_cast<double>(nz - 1) / static_cast<double>(count - 1));
        if (layers.empty() || layers.back() != layer) layers.push_back(layer);
    }
    return layers;
}

void visualize(const fs::path& directory, const GridData& grid,
               const GeographicGrid& geographic, int numberOfLayers,
               std::ostream& output) {
    fs::create_directories(directory);
    const double maximum = globalMaximumSpeed(grid);
    std::set<std::string> used;
    for (const std::size_t index : selectedLayers(grid.nz, numberOfLayers)) {
        const long long rounded = std::llround(grid.heights[index]);
        std::string name = "wind_" + std::to_string(rounded) + "m.png";
        if (used.contains(name)) name = "wind_" + std::to_string(rounded) + "m_" + std::to_string(index) + ".png";
        used.insert(name);
        const fs::path path = directory / name;
        renderSection(path, grid, geographic, index, maximum);
        output << "[INFO] Saved section: " << pathUtf8(path) << '\n';
    }
    const fs::path distribution = directory / "wind_speed_distribution.png";
    renderHistogram(distribution, grid, maximum);
    output << "[INFO] Saved distribution: " << pathUtf8(distribution) << '\n';
}

void writeNetcdf(const fs::path& path, const GridData& grid,
                 const GeographicGrid& geographic, int epsg,
                 const std::string& casename, const std::string& datetime,
                 const std::string& vtkBasename) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    NetcdfFile file(path); const int ncid = file.id();
    int dimHeight = -1, dimY = -1, dimX = -1;
    ncCheck(nc_def_dim(ncid, "height", grid.nz, &dimHeight), "Failed to define height dimension");
    ncCheck(nc_def_dim(ncid, "y", grid.ny, &dimY), "Failed to define y dimension");
    ncCheck(nc_def_dim(ncid, "x", grid.nx, &dimX), "Failed to define x dimension");
    int varY = -1, varX = -1, varHeight = -1, varCrs = -1;
    ncCheck(nc_def_var(ncid, "y", NC_INT, 1, &dimY, &varY), "Failed to define y variable");
    ncCheck(nc_def_var(ncid, "x", NC_INT, 1, &dimX, &varX), "Failed to define x variable");
    ncCheck(nc_def_var(ncid, "height", NC_FLOAT, 1, &dimHeight, &varHeight), "Failed to define height variable");
    const int varLat = defineCompressedDoubleVariable(ncid, "lat", {dimY, dimX}, 4, true);
    const int varLon = defineCompressedDoubleVariable(ncid, "lon", {dimY, dimX}, 4, true);
    const int varU = defineCompressedFloatVariable(ncid, "u_grid", {dimHeight, dimY, dimX}, 4, true, false);
    const int varV = defineCompressedFloatVariable(ncid, "v_grid", {dimHeight, dimY, dimX}, 4, true, false);
    const int varW = defineCompressedFloatVariable(ncid, "w", {dimHeight, dimY, dimX}, 4, true, false);
    ncCheck(nc_def_var(ncid, "crs", NC_INT, 0, nullptr, &varCrs), "Failed to define crs variable");
    setTextAttribute(ncid, varY, "long_name", "grid y index"); setTextAttribute(ncid, varY, "units", "1");
    setTextAttribute(ncid, varX, "long_name", "grid x index"); setTextAttribute(ncid, varX, "units", "1");
    setTextAttribute(ncid, varHeight, "units", "m");
    setTextAttribute(ncid, varHeight, "long_name", "height above ground, reference zmin + 50 m");
    setTextAttribute(ncid, varHeight, "axis", "Z"); setTextAttribute(ncid, varHeight, "positive", "up");
    setTextAttribute(ncid, varLat, "units", "degrees_north"); setTextAttribute(ncid, varLat, "standard_name", "latitude");
    setTextAttribute(ncid, varLat, "long_name", "latitude on curvilinear grid");
    setTextAttribute(ncid, varLon, "units", "degrees_east"); setTextAttribute(ncid, varLon, "standard_name", "longitude");
    setTextAttribute(ncid, varLon, "long_name", "longitude on curvilinear grid");
    setTextAttribute(ncid, varU, "units", "m s-1"); setTextAttribute(ncid, varU, "long_name", "wind component along model x axis");
    setTextAttribute(ncid, varV, "units", "m s-1"); setTextAttribute(ncid, varV, "long_name", "wind component along model y axis");
    setTextAttribute(ncid, varW, "units", "m s-1"); setTextAttribute(ncid, varW, "long_name", "vertical wind component");
    for (const int variable : {varU, varV, varW}) setTextAttribute(ncid, variable, "coordinates", "lon lat height");
    setTextAttribute(ncid, varCrs, "long_name", "coordinate reference system");
    setTextAttribute(ncid, varCrs, "spatial_ref", "EPSG:" + std::to_string(epsg));
    setTextAttribute(ncid, varCrs, "grid_mapping_name", "latitude_longitude");
    setTextAttribute(ncid, NC_GLOBAL, "Conventions", "CF-1.8");
    setTextAttribute(ncid, NC_GLOBAL, "title", "Wind field on lon/lat grid with height from zmin + 50 m");
    setTextAttribute(ncid, NC_GLOBAL, "history", "Created by wind_from_luw.py");
    setTextAttribute(ncid, NC_GLOBAL, "source_vtk", vtkBasename);
    const long long epsg64 = epsg;
    ncCheck(nc_put_att_longlong(ncid, NC_GLOBAL, "crs_epsg", NC_INT64, 1, &epsg64), "Failed to set crs_epsg");
    ncCheck(nc_put_att_double(ncid, NC_GLOBAL, "center_lon", NC_DOUBLE, 1, &grid.centerLon), "Failed to set center_lon");
    ncCheck(nc_put_att_double(ncid, NC_GLOBAL, "center_lat", NC_DOUBLE, 1, &grid.centerLat), "Failed to set center_lat");
    setTextAttribute(ncid, NC_GLOBAL, "note", "Components are aligned with model x and y axes, not rotated to true east and north.");
    setTextAttribute(ncid, NC_GLOBAL, "case", casename); setTextAttribute(ncid, NC_GLOBAL, "datetime_tag", datetime);
    ncCheck(nc_enddef(ncid), "Failed to finish NetCDF schema");
    std::vector<int> xIndex(grid.nx), yIndex(grid.ny); std::vector<float> height(grid.nz);
    for (std::size_t i = 0; i < grid.nx; ++i) xIndex[i] = static_cast<int>(i);
    for (std::size_t i = 0; i < grid.ny; ++i) yIndex[i] = static_cast<int>(i);
    for (std::size_t i = 0; i < grid.nz; ++i) height[i] = static_cast<float>(grid.heights[i]);
    ncCheck(nc_put_var_int(ncid, varX, xIndex.data()), "Failed to write x index");
    ncCheck(nc_put_var_int(ncid, varY, yIndex.data()), "Failed to write y index");
    ncCheck(nc_put_var_float(ncid, varHeight, height.data()), "Failed to write height");
    ncCheck(nc_put_var_double(ncid, varLon, geographic.longitude.data()), "Failed to write longitude");
    ncCheck(nc_put_var_double(ncid, varLat, geographic.latitude.data()), "Failed to write latitude");
    std::vector<float> plane(grid.nx * grid.ny);
    const std::array<std::size_t, 3> count{1, grid.ny, grid.nx};
    for (std::size_t iz = 0; iz < grid.nz; ++iz) {
        const std::array<std::size_t, 3> start{iz, 0, 0};
        for (std::size_t component = 0; component < 3; ++component) {
            for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                plane[iy * grid.nx + ix] = grid.wind[((ix * grid.ny + iy) * grid.nz + iz) * 3U + component];
            }
            const int variable = component == 0 ? varU : component == 1 ? varV : varW;
            ncCheck(nc_put_vara_float(ncid, variable, start.data(), count.data(), plane.data()), "Failed to write wind component");
        }
    }
    file.close();
}

int execute(const Arguments& arguments, std::ostream& output) {
    const fs::path configPath = resolved(arguments.configPath);
    output << "[INFO] LUW: " << pathUtf8(configPath) << '\n';
    Config config;
    try { config = loadConfig(configPath); }
    catch (const SoftExit&) { throw; }
    catch (const std::exception& exception) { throw SoftExit("Failed to read LUW. Detail: " + std::string(exception.what())); }
    if (config.casename.empty()) throw SoftExit("Missing casename in LUW. Please add: casename = your_case");
    if (!std::regex_match(config.datetime, std::regex("[0-9]{14}"))) {
        throw SoftExit("Missing or invalid datetime in LUW. Expected format yyyymmddHHMMSS. Run tranluw first.");
    }
    const std::optional<double> lonMinDefault = config.cutLon ? std::optional<double>((*config.cutLon)[0]) : std::nullopt;
    const std::optional<double> lonMaxDefault = config.cutLon ? std::optional<double>((*config.cutLon)[1]) : std::nullopt;
    const std::optional<double> latMinDefault = config.cutLat ? std::optional<double>((*config.cutLat)[0]) : std::nullopt;
    const std::optional<double> latMaxDefault = config.cutLat ? std::optional<double>((*config.cutLat)[1]) : std::nullopt;
    std::optional<double> epsgLon, epsgLat;
    if (lonMinDefault && lonMaxDefault && latMinDefault && latMaxDefault) {
        epsgLon = 0.5 * (*lonMinDefault + *lonMaxDefault); epsgLat = 0.5 * (*latMinDefault + *latMaxDefault);
    }
    const int epsg = inferEpsg(config, epsgLon, epsgLat);
    std::unique_ptr<Projection> forward, inverse;
    try {
        forward = std::make_unique<Projection>("EPSG:4326", "EPSG:" + std::to_string(epsg));
        inverse = std::make_unique<Projection>("EPSG:" + std::to_string(epsg), "EPSG:4326");
    } catch (const std::exception&) { throw SoftExit("Invalid EPSG in LUW: " + std::to_string(epsg)); }
    output << "[INFO] Using UTM CRS EPSG:" << epsg << '\n';
    const bool interactiveCrop = !arguments.lonMin || !arguments.lonMax || !arguments.latMin || !arguments.latMax;
    Crop crop;
    if (interactiveCrop) {
        output << "[INPUT] Please provide target crop in lon/lat.\n";
        crop.lonMin = promptFloat(output, "Enter target longitude min", lonMinDefault);
        crop.lonMax = promptFloat(output, "Enter target longitude max", lonMaxDefault);
        crop.latMin = promptFloat(output, "Enter target latitude min", latMinDefault);
        crop.latMax = promptFloat(output, "Enter target latitude max", latMaxDefault);
        output << "[CONFIRM] Crop box lon [" << number(crop.lonMin) << ", " << number(crop.lonMax)
               << "], lat [" << number(crop.latMin) << ", " << number(crop.latMax) << "]\n";
        if (!confirm(output, "Press ENTER to proceed, or type 'y'/'yes'. Any other key to abort:")) {
            throw SoftExit("User did not confirm the crop box.");
        }
    } else {
        crop.lonMin = *arguments.lonMin; crop.lonMax = *arguments.lonMax;
        crop.latMin = *arguments.latMin; crop.latMax = *arguments.latMax;
        output << "[INFO] Using CLI crop box lon [" << number(crop.lonMin) << ", " << number(crop.lonMax)
               << "], lat [" << number(crop.latMin) << ", " << number(crop.latMax) << "]\n";
    }
    const double centerLon = config.centerLon.value_or(lonMinDefault.value_or(crop.lonMin));
    const double centerLat = config.centerLat.value_or(latMinDefault.value_or(crop.latMin));
    output << "[INFO] Mapping origin set to SW corner at lon " << number(centerLon)
           << ", lat " << number(centerLat) << '\n';
    const int layers = arguments.layers ? std::max(*arguments.layers, 1)
                                        : promptPositiveInt(output, "Enter number of layers to export", 6);
    const fs::path results = findResultsDirectory(configPath);
    output << "[INFO] RESULTS directory: " << pathUtf8(results) << '\n';
    const auto [vtkPath, baseName] = locateVtk(results, config.casename, config.datetime);
    output << "[INFO] VTK located: " << pathUtf8(vtkPath) << '\n';
    const fs::path exportRoot = arguments.outputDirectory ? resolved(*arguments.outputDirectory) : results;
    fs::create_directories(exportRoot);
    const fs::path npzPath = exportRoot / (baseName + ".npz");
    const fs::path sections = exportRoot / "sections";
    fs::create_directories(sections);
    output << "[INFO] Export root: " << pathUtf8(exportRoot) << '\n';
    try {
        output << "[INFO] Reading VTK: " << pathUtf8(vtkPath) << '\n';
        const LegacyStructuredPoints vtk = LegacyStructuredPoints::readHeader(vtkPath);
        const LegacyVtkField& vector = selectVector(vtk);
        GridData grid = extract(vtk, vector, config, crop, centerLon, centerLat, *forward, output);
        saveNpz(npzPath, grid);
        output << "[INFO] Saved NPZ: " << pathUtf8(npzPath) << " ("
               << fixed(static_cast<double>(fs::file_size(npzPath)) / 1024.0 / 1024.0, 1) << " MB)\n";
        const GeographicGrid geographic = mapLonLat(grid, vtk, config, *inverse);
        visualize(sections, grid, geographic, layers, output);
        bool exportNc = arguments.exportNc;
        if (!exportNc && !interactiveCrop && arguments.layers) {
            output << "[INFO] NetCDF export skipped.\n";
        } else if (!exportNc) {
            output << "[INPUT] Do you want to export a 3D NetCDF file in lon/lat coordinates?\n";
            exportNc = confirm(output, "Press ENTER, 'y', or 'yes' to export. Any other key to skip:");
        }
        if (exportNc) {
            output << "[INFO] NetCDF lon/lat range after mapping: lon[" << fixed(geographic.lonMin, 6)
                   << ", " << fixed(geographic.lonMax, 6) << "], lat[" << fixed(geographic.latMin, 6)
                   << ", " << fixed(geographic.latMax, 6) << "]\n";
            const fs::path ncPath = arguments.ncOutput ? resolved(*arguments.ncOutput)
                                                       : exportRoot / (baseName + "_visluw.nc");
            writeNetcdf(ncPath, grid, geographic, epsg, config.casename, config.datetime, baseName);
            output << "[INFO] Saved NetCDF: " << pathUtf8(ncPath) << " ("
                   << fixed(static_cast<double>(fs::file_size(ncPath)) / 1024.0 / 1024.0, 1) << " MB)\n";
        } else if (interactiveCrop || !arguments.layers) {
            output << "[INFO] NetCDF export skipped.\n";
        }
    } catch (const SoftExit&) { throw; }
    catch (const std::exception& exception) { throw SoftExit("Processing failed. Detail: " + std::string(exception.what())); }
    output << "[INFO] Completed successfully.\n"
           << "[INFO] NPZ: " << pathUtf8(npzPath) << '\n'
           << "[INFO] Figures directory: " << pathUtf8(sections) << '\n';
    return 0;
}

} // namespace

int run_visluw(const std::vector<std::string>& arguments,
               std::ostream& output,
               std::ostream& error) {
    try {
        const Arguments parsed = parseArguments(arguments);
        if (parsed.help) { usage(output); return 0; }
        return execute(parsed, output);
    } catch (const SoftExit& exception) {
        if (exception.code() == 2) {
            error << "usage: visluw [-h] [options] config_path\n"
                  << "visluw: error: " << exception.what() << '\n';
        } else {
            output << "[ERROR] " << exception.what() << '\n';
        }
        return exception.code();
    } catch (const std::exception& exception) {
        output << "[ERROR] Processing failed. Detail: " << exception.what() << '\n';
        return 1;
    }
}

} // namespace luw::cli
