#include "luw/vtk_post_commands.hpp"

#include "luw/deck_document.hpp"
#include "luw/legacy_vtk.hpp"
#include "post_internal.hpp"

#include <QFileInfo>
#include <QString>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
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
    int code_;
};

struct Vtk2NcArguments final {
    fs::path configPath;
    std::optional<fs::path> outputPath;
    bool help = false;
};

struct Vtk2NcConfig final {
    std::string casename;
    std::string datetime;
    std::optional<std::pair<double, double>> siX;
    std::optional<std::pair<double, double>> siY;
    std::optional<std::pair<double, double>> cutLon;
    std::optional<std::pair<double, double>> cutLat;
};

void printUsage(std::ostream& output) {
    output << "usage: vtk2nc [-h] [--output OUTPUT] config_path\n\n"
              "Export a LUW solver result to NetCDF.\n\n"
              "positional arguments:\n"
              "  config_path     Path to the LUW deck file\n\n"
              "options:\n"
              "  -h, --help      show this help message and exit\n"
              "  --output OUTPUT Optional NetCDF output path\n";
}

Vtk2NcArguments parseArguments(const std::vector<std::string>& arguments) {
    Vtk2NcArguments parsed;
    bool haveConfig = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "-h" || argument == "--help") {
            parsed.help = true;
            continue;
        }
        if (argument == "--output") {
            if (index + 1 >= arguments.size()) {
                throw SoftExit("argument --output: expected one argument", 2);
            }
            parsed.outputPath = fs::path(arguments[++index]);
            continue;
        }
        if (argument.starts_with("--output=")) {
            parsed.outputPath = fs::path(argument.substr(std::string("--output=").size()));
            continue;
        }
        if (!argument.empty() && argument.front() == '-') {
            throw SoftExit("unrecognized arguments: " + argument, 2);
        }
        if (haveConfig) throw SoftExit("unrecognized arguments: " + argument, 2);
        parsed.configPath = fs::path(argument);
        haveConfig = true;
    }
    if (!parsed.help && !haveConfig) {
        throw SoftExit("the following arguments are required: config_path", 2);
    }
    return parsed;
}

fs::path resolved(const fs::path& path) {
    std::error_code error;
    fs::path absolute = fs::absolute(path, error);
    if (error) absolute = path;
    return absolute.lexically_normal();
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return text;
}

std::string pyFloat(double value) {
    std::ostringstream stream;
    stream << std::setprecision(15) << value;
    std::string text = stream.str();
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}

template <typename T>
std::string tuple3(const std::array<T, 3>& values) {
    std::ostringstream stream;
    stream << '(';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) stream << ", ";
        if constexpr (std::is_floating_point_v<T>) stream << pyFloat(values[index]);
        else stream << values[index];
    }
    return stream.str() + ')';
}

std::vector<double> linearAxis(double origin, double spacing, int count) {
    std::vector<double> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        result[static_cast<std::size_t>(index)] = origin + index * spacing;
    }
    return result;
}

std::vector<double> linspace(double first, double last, int count) {
    std::vector<double> result(static_cast<std::size_t>(count));
    if (count == 1) {
        result.front() = first;
        return result;
    }
    for (int index = 0; index < count; ++index) {
        const double fraction = static_cast<double>(index) / static_cast<double>(count - 1);
        result[static_cast<std::size_t>(index)] = first + (last - first) * fraction;
    }
    return result;
}

Vtk2NcConfig loadConfig(const fs::path& path) {
    const std::string utf8 = pathUtf8(path);
    const luw::DeckDocument deck = luw::DeckDocument::load(
        QString::fromUtf8(utf8.data(), static_cast<qsizetype>(utf8.size())));
    Vtk2NcConfig config;
    if (const auto value = deck.text(QStringLiteral("casename"))) config.casename = value->toStdString();
    if (const auto value = deck.text(QStringLiteral("datetime"))) config.datetime = value->toStdString();
    config.siX = deck.pair(QStringLiteral("si_x_cfd"));
    config.siY = deck.pair(QStringLiteral("si_y_cfd"));
    config.cutLon = deck.pair(QStringLiteral("cut_lon_manual"));
    config.cutLat = deck.pair(QStringLiteral("cut_lat_manual"));
    return config;
}

const LegacyVtkField& selectWindField(const LegacyStructuredPoints& vtk) {
    std::size_t pointArrayCount = 0;
    for (const LegacyVtkField& field : vtk.fields) {
        if (field.association != LegacyVtkAssociation::PointData) continue;
        ++pointArrayCount;
        if (field.name == "data" && field.components >= 3) return field;
    }
    if (pointArrayCount == 0) throw SoftExit("No point arrays found in VTK dataset.");
    for (const LegacyVtkField& field : vtk.fields) {
        if (field.association != LegacyVtkAssociation::PointData) continue;
        if (field.name == "data_rho") continue;
        if (field.components >= 3) return field;
    }
    throw SoftExit("No suitable 3 component wind array found. Expected an array named 'data' or any 3 component array.");
}

struct GeographicGrid final {
    std::vector<float> longitude;
    std::vector<float> latitude;
};

GeographicGrid buildGeographicGrid(const Vtk2NcConfig& config,
                                    const LegacyStructuredPoints& vtk,
                                    const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    std::ostream& output) {
    const int nx = vtk.dimensions[0];
    const int ny = vtk.dimensions[1];
    std::vector<PJ_COORD> coordinates(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    std::size_t cursor = 0;
    if (config.cutLon && config.cutLat) {
        Projection forward("EPSG:4326", "EPSG:32650");
        const auto anchor = forward.apply(config.cutLon->first, config.cutLat->first);
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                coordinates[cursor++] = proj_coord(
                    anchor.first + static_cast<double>(ix) * vtk.spacing[0],
                    anchor.second + static_cast<double>(iy) * vtk.spacing[1], 0.0, 0.0);
            }
        }
        output << "[Info] Anchored lon/lat using cut_lon_manual[0]=" << pyFloat(config.cutLon->first)
               << ", cut_lat_manual[0]=" << pyFloat(config.cutLat->first) << '\n';
    } else {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                coordinates[cursor++] = proj_coord(x[static_cast<std::size_t>(ix)],
                                                   y[static_cast<std::size_t>(iy)], 0.0, 0.0);
            }
        }
        output << "[Info] No manual lon/lat anchor provided. Interpreting x,y as absolute UTM 50N.\n";
    }
    Projection inverse("EPSG:32650", "EPSG:4326");
    inverse.apply(coordinates);
    GeographicGrid result;
    result.longitude.resize(coordinates.size());
    result.latitude.resize(coordinates.size());
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        result.longitude[index] = static_cast<float>(coordinates[index].xy.x);
        result.latitude[index] = static_cast<float>(coordinates[index].xy.y);
    }
    return result;
}

void defineTextAttributes(int file, int variable,
                          const char* longName, const char* units) {
    setTextAttribute(file, variable, "long_name", longName);
    setTextAttribute(file, variable, "units", units);
}

void writeNetcdf(const fs::path& outputPath,
                 const LegacyStructuredPoints& vtk,
                 const LegacyVtkField& wind,
                 const std::vector<std::size_t>& keptZ,
                 const std::vector<double>& x,
                 const std::vector<double>& y,
                 const std::vector<float>& height,
                 const GeographicGrid& geographic) {
    if (!outputPath.parent_path().empty()) fs::create_directories(outputPath.parent_path());
    NetcdfFile file(outputPath);
    const int ncid = file.id();

    int dimH = -1;
    int dimY = -1;
    int dimX = -1;
    ncCheck(nc_def_dim(ncid, "h", keptZ.size(), &dimH), "Failed to define h dimension");
    ncCheck(nc_def_dim(ncid, "y", static_cast<std::size_t>(vtk.dimensions[1]), &dimY),
            "Failed to define y dimension");
    ncCheck(nc_def_dim(ncid, "x", static_cast<std::size_t>(vtk.dimensions[0]), &dimX),
            "Failed to define x dimension");

    const std::vector<int> dims3{dimH, dimY, dimX};
    const std::vector<int> dims2{dimY, dimX};
    const int varU = defineCompressedFloatVariable(ncid, "u", dims3);
    const int varV = defineCompressedFloatVariable(ncid, "v", dims3);
    const int varW = defineCompressedFloatVariable(ncid, "w", dims3);
    const int varLat = defineCompressedFloatVariable(ncid, "lat", dims2);
    const int varLon = defineCompressedFloatVariable(ncid, "lon", dims2);
    const int varX = defineCompressedFloatVariable(ncid, "x", {dimX});
    const int varY = defineCompressedFloatVariable(ncid, "y", {dimY});
    const int varH = defineCompressedFloatVariable(ncid, "h", {dimH});

    defineTextAttributes(ncid, varU, "u wind", "m s-1");
    defineTextAttributes(ncid, varV, "v wind", "m s-1");
    defineTextAttributes(ncid, varW, "w wind", "m s-1");
    defineTextAttributes(ncid, varLat, "latitude", "degree_north");
    defineTextAttributes(ncid, varLon, "longitude", "degree_east");
    defineTextAttributes(ncid, varX, "x", "meters");
    defineTextAttributes(ncid, varY, "y", "meters");
    defineTextAttributes(ncid, varH, "height above ground", "meters");

    setTextAttribute(ncid, NC_GLOBAL, "institution",
                     "National Meteorological Information Center, CMA, Beijing, China");
    setTextAttribute(ncid, NC_GLOBAL, "source", "CFD-LBM V1.0");
    setTextAttribute(ncid, NC_GLOBAL, "contact", "Dr. Han Shuai, hans@cma.gov.cn, 010-58993096");
    setTextAttribute(ncid, NC_GLOBAL, "history", "Created by Dr. Han Shuai, NMIC/CMA, 2025-07-27");
    setTextAttribute(ncid, NC_GLOBAL, "note",
                     "VTK vertical coordinate adjusted by subtracting 50 m then removing below ground layers.");

    ncCheck(nc_enddef(ncid), "Failed to finish NetCDF schema");
    ncCheck(nc_put_var_double(ncid, varX, x.data()), "Failed to write x coordinates");
    ncCheck(nc_put_var_double(ncid, varY, y.data()), "Failed to write y coordinates");
    ncCheck(nc_put_var_float(ncid, varH, height.data()), "Failed to write h coordinates");
    ncCheck(nc_put_var_float(ncid, varLat, geographic.latitude.data()), "Failed to write latitude");
    ncCheck(nc_put_var_float(ncid, varLon, geographic.longitude.data()), "Failed to write longitude");

    const std::size_t planeSize = static_cast<std::size_t>(vtk.dimensions[0])
        * static_cast<std::size_t>(vtk.dimensions[1]);
    const std::array<std::size_t, 3> count{1, static_cast<std::size_t>(vtk.dimensions[1]),
                                           static_cast<std::size_t>(vtk.dimensions[0])};
    std::vector<float> components[3];
    LegacyVtkPlaneReader reader(vtk);
    for (std::size_t outputZ = 0; outputZ < keptZ.size(); ++outputZ) {
        const std::vector<float> tuples = reader.readPointPlane(wind, keptZ[outputZ]);
        for (auto& component : components) component.resize(planeSize);
        for (std::size_t point = 0; point < planeSize; ++point) {
            components[0][point] = tuples[point * wind.components];
            components[1][point] = tuples[point * wind.components + 1];
            components[2][point] = tuples[point * wind.components + 2];
        }
        const std::array<std::size_t, 3> start{outputZ, 0, 0};
        ncCheck(nc_put_vara_float(ncid, varU, start.data(), count.data(), components[0].data()),
                "Failed to write u wind");
        ncCheck(nc_put_vara_float(ncid, varV, start.data(), count.data(), components[1].data()),
                "Failed to write v wind");
        ncCheck(nc_put_vara_float(ncid, varW, start.data(), count.data(), components[2].data()),
                "Failed to write w wind");
    }
    file.close();
}

int execute(const Vtk2NcArguments& arguments, std::ostream& output) {
    const fs::path configPath = resolved(arguments.configPath);
    if (!fs::is_regular_file(configPath)) {
        throw SoftExit("Config file not found: " + configPath.string());
    }
    if (lower(configPath.extension().string()) != ".luw") {
        output << "[Warn] Config file does not have .luw suffix: " << configPath.filename().string() << '\n';
    }

    const Vtk2NcConfig config = loadConfig(configPath);
    if (config.casename.empty() || config.datetime.empty()) {
        throw SoftExit("Config must provide both 'casename' and 'datetime'.");
    }
    output << "[Info] Loaded config, casename=" << config.casename
           << ", datetime=" << config.datetime << '\n';

    const fs::path resultsDirectory = configPath.parent_path() / "RESULTS";
    if (!fs::is_directory(resultsDirectory)) {
        throw SoftExit("RESULTS folder not found at: " + resultsDirectory.string());
    }
    const fs::path uvwrho = resultsDirectory / ("uvwrho-" + config.casename + '_' + config.datetime + ".vtk");
    const fs::path uvw = resultsDirectory / ("uvw-" + config.casename + '_' + config.datetime + ".vtk");
    fs::path vtkPath;
    if (fs::is_regular_file(uvwrho)) vtkPath = uvwrho;
    else if (fs::is_regular_file(uvw)) vtkPath = uvw;
    else {
        throw SoftExit("No VTK file found. Tried: " + uvwrho.filename().string()
                       + " and " + uvw.filename().string() + " under " + resultsDirectory.string());
    }
    output << "[Info] Using VTK file: " << vtkPath.string() << '\n';

    LegacyStructuredPoints vtk;
    try {
        vtk = LegacyStructuredPoints::readHeader(vtkPath);
    } catch (const std::exception& exception) {
        throw SoftExit("Failed to read VTK file: " + vtkPath.string() + ", reason: " + exception.what());
    }
    for (const int dimension : vtk.dimensions) {
        if (dimension <= 0) throw SoftExit("Invalid grid dimensions: " + tuple3(vtk.dimensions));
    }
    for (const double spacing : vtk.spacing) {
        if (std::abs(spacing) < 1.0e-12) throw SoftExit("Invalid grid spacing: " + tuple3(vtk.spacing));
    }
    output << "[Info] Grid dims: " << tuple3(vtk.dimensions) << '\n'
           << "[Info] Grid origin: " << tuple3(vtk.origin) << '\n'
           << "[Info] Grid spacing: " << tuple3(vtk.spacing) << '\n';

    std::vector<double> x = linearAxis(vtk.origin[0], vtk.spacing[0], vtk.dimensions[0]);
    std::vector<double> y = linearAxis(vtk.origin[1], vtk.spacing[1], vtk.dimensions[1]);
    if (config.siX && config.siY && !(config.cutLon && config.cutLat)) {
        x = linspace(config.siX->first, config.siX->second, vtk.dimensions[0]);
        y = linspace(config.siY->first, config.siY->second, vtk.dimensions[1]);
        output << "[Info] Using horizontal range from config, x=[" << pyFloat(config.siX->first)
               << ", " << pyFloat(config.siX->second) << "], y=[" << pyFloat(config.siY->first)
               << ", " << pyFloat(config.siY->second) << "]\n";
    } else {
        output << "[Info] Using meter coordinates relative to VTK origin for x and y.\n";
    }

    std::vector<std::size_t> keptZ;
    std::vector<float> height;
    for (int index = 0; index < vtk.dimensions[2]; ++index) {
        const double shifted = vtk.origin[2] + index * vtk.spacing[2] - 50.0;
        if (shifted >= 0.0) {
            keptZ.push_back(static_cast<std::size_t>(index));
            height.push_back(static_cast<float>(shifted));
        }
    }
    if (keptZ.empty()) {
        throw SoftExit("After removing the lowest 50 m, no vertical layers remain. Check your grid and spacing.");
    }
    output << "[Info] Vertical layers before cut: " << vtk.dimensions[2]
           << ", after cut: " << keptZ.size() << '\n';

    const LegacyVtkField& wind = selectWindField(vtk);
    if (wind.tuples != vtk.pointCount) {
        throw SoftExit("Wind array row count " + std::to_string(wind.tuples)
                       + " does not match number of points " + std::to_string(vtk.pointCount) + ".");
    }
    const std::size_t expectedPoints = static_cast<std::size_t>(vtk.dimensions[0])
        * static_cast<std::size_t>(vtk.dimensions[1]) * static_cast<std::size_t>(vtk.dimensions[2]);
    if (vtk.pointCount != expectedPoints) {
        throw SoftExit("Number of points " + std::to_string(vtk.pointCount)
                       + " does not match dims product " + std::to_string(expectedPoints) + ".");
    }

    GeographicGrid geographic;
    try {
        geographic = buildGeographicGrid(config, vtk, x, y, output);
    } catch (const std::exception& exception) {
        throw SoftExit("Geodetic transform failed: " + std::string(exception.what()));
    }

    const fs::path outputPath = arguments.outputPath
        ? resolved(*arguments.outputPath)
        : resultsDirectory / ("uvw-" + config.casename + '_' + config.datetime + ".nc");
    output << "[Info] NetCDF output path: " << outputPath.string() << '\n';
    try {
        writeNetcdf(outputPath, vtk, wind, keptZ, x, y, height, geographic);
    } catch (const std::exception& exception) {
        throw SoftExit("Failed to write NetCDF file: " + std::string(exception.what()));
    }
    output << "[OK] Saved NetCDF: " << outputPath.string() << '\n';
    return 0;
}

} // namespace

int run_vtk2nc(const std::vector<std::string>& arguments,
               std::ostream& output,
               std::ostream& error) {
    try {
        const Vtk2NcArguments parsed = parseArguments(arguments);
        if (parsed.help) {
            printUsage(output);
            return 0;
        }
        return execute(parsed, output);
    } catch (const SoftExit& exit) {
        if (exit.code() == 2) {
            error << "usage: vtk2nc [-h] [--output OUTPUT] config_path\n"
                  << "vtk2nc: error: " << exit.what() << '\n';
        } else {
            output << "[Exit] " << exit.what() << '\n';
        }
        return exit.code();
    } catch (const std::exception& exception) {
        output << "[Exit] Unhandled error: " << exception.what() << '\n';
        return 1;
    }
}

} // namespace luw::cli
