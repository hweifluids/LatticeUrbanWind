#include "luw/vtk_post_commands.hpp"

#include "luw/deck_document.hpp"
#include "luw/legacy_vtk.hpp"
#include "post_internal.hpp"

#include <QFileInfo>
#include <QString>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace luw::cli {

namespace {

namespace fs = std::filesystem;
using luw::post::LegacyStructuredPoints;
using luw::post::LegacyVtkAssociation;
using luw::post::LegacyVtkEncoding;
using luw::post::LegacyVtkField;
using luw::post::LegacyVtkFieldKind;
using luw::post::LegacyVtkPlaneReader;
using luw::post::detail::NetcdfFile;
using luw::post::detail::Projection;
using luw::post::detail::ncCheck;
using luw::post::detail::pathUtf8;
using luw::post::detail::setTextAttribute;

constexpr int maximumWorkers = 12;
constexpr int cubicPadding = 12;

class FatalExit final : public std::exception {
public:
    explicit FatalExit(int code) noexcept : code_(code) {}
    [[nodiscard]] int code() const noexcept { return code_; }
    [[nodiscard]] const char* what() const noexcept override { return "vtk2nc_new fatal exit"; }

private:
    int code_ = 1;
};

std::string timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream stream;
    stream << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
    return stream.str();
}

void writeLog(std::ostream& output, std::string_view level, const std::string& message) {
    output << '[' << level << "] " << timestamp() << " | " << message << '\n' << std::flush;
}

void info(std::ostream& output, const std::string& message) {
    writeLog(output, "INFO", message);
}

void warn(std::ostream& output, const std::string& message) {
    writeLog(output, "WARN", message);
}

[[noreturn]] void fail(std::ostream& output, const std::string& message, int code = 1) {
    writeLog(output, "ERROR", message);
    throw FatalExit(code);
}

std::string fixed(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string scientific(double value, int precision) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(precision) << value;
    return stream.str();
}

std::string general(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

fs::path pathFromUtf8(const std::string& value) {
#if defined(_WIN32)
    const QString text = QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
    return fs::path(text.toStdWString());
#else
    return fs::path(value);
#endif
}

fs::path resolved(const fs::path& path) {
    std::error_code error;
    fs::path absolute = fs::absolute(path, error);
    if (error) absolute = path;
    error.clear();
    fs::path result = fs::weakly_canonical(absolute, error);
    if (!error) return result;
    return absolute.lexically_normal();
}

struct CaseConfig final {
    fs::path luwPath;
    fs::path projectHome;
    std::string casename;
    std::optional<std::string> datetime;
    std::pair<double, double> cutLon{};
    std::pair<double, double> cutLat{};
    std::string utmCrs;
    double rotateDegrees = 0.0;
};

std::string inferUtm(double longitude, double latitude) {
    int zone = static_cast<int>(std::floor((longitude + 180.0) / 6.0) + 1.0);
    zone = std::clamp(zone, 1, 60);
    return "EPSG:" + std::to_string((latitude >= 0.0 ? 32600 : 32700) + zone);
}

CaseConfig parseConfig(const fs::path& requestedPath, std::ostream& output) {
    const fs::path luwPath = resolved(requestedPath);
    std::error_code typeError;
    if (!fs::is_regular_file(luwPath, typeError) || typeError) {
        fail(output, "LUW file not found: " + pathUtf8(luwPath));
    }

    luw::DeckDocument deck;
    try {
        const std::string utf8 = pathUtf8(luwPath);
        deck = luw::DeckDocument::load(
            QString::fromUtf8(utf8.data(), static_cast<qsizetype>(utf8.size())));
    } catch (const std::exception& exception) {
        fail(output, "Failed to read LUW file: " + pathUtf8(luwPath) + " | " + exception.what());
    }

    CaseConfig config;
    config.luwPath = luwPath;
    config.projectHome = luwPath.parent_path();
    if (const auto value = deck.text(QStringLiteral("casename")); value && !value->isEmpty()) {
        config.casename = value->toUtf8().toStdString();
    }
    if (config.casename.empty()) fail(output, "Missing key 'casename' in LUW file.");

    if (const auto value = deck.text(QStringLiteral("datetime")); value && !value->isEmpty()) {
        config.datetime = value->toUtf8().toStdString();
    }

    const auto longitude = deck.pair(QStringLiteral("cut_lon_manual"));
    const auto latitude = deck.pair(QStringLiteral("cut_lat_manual"));
    if (!longitude || !latitude) {
        fail(output, "Missing 'cut_lon_manual' or 'cut_lat_manual' in LUW file.");
    }
    config.cutLon = *longitude;
    config.cutLat = *latitude;

    if (const auto value = deck.text(QStringLiteral("utm_crs")); value && !value->isEmpty()) {
        config.utmCrs = value->toUtf8().toStdString();
    } else {
        config.utmCrs = inferUtm(0.5 * (config.cutLon.first + config.cutLon.second),
                                 0.5 * (config.cutLat.first + config.cutLat.second));
        warn(output, "'utm_crs' not found in LUW. Auto-inferred CRS: " + config.utmCrs);
    }

    Projection toUtm("EPSG:4326", config.utmCrs);
    const auto lowerLeft = toUtm.apply(config.cutLon.first, config.cutLat.first);
    const auto lowerRight = toUtm.apply(config.cutLon.second, config.cutLat.first);
    const double estimate = -std::atan2(lowerRight.second - lowerLeft.second,
                                        lowerRight.first - lowerLeft.first)
        * (180.0 / std::acos(-1.0));
    const std::optional<double> configured = deck.floating(QStringLiteral("rotate_deg"));
    config.rotateDegrees = configured.value_or(estimate);
    if (configured) {
        info(output, "Rotation from LUW=" + fixed(*configured, 9)
                     + " deg, recomputed=" + fixed(estimate, 9)
                     + " deg, abs diff=" + scientific(std::abs(*configured - estimate), 9) + '.');
    } else {
        info(output, "Rotation not found in LUW. Using recomputed rotation "
                     + fixed(config.rotateDegrees, 9) + " deg.");
    }
    return config;
}

std::vector<fs::path> vtkFilesIn(const fs::path& directory) {
    std::vector<fs::path> result;
    std::error_code typeError;
    if (!fs::is_directory(directory, typeError) || typeError) return result;
    std::error_code iteratorError;
    for (fs::directory_iterator iterator(directory, iteratorError), end;
         !iteratorError && iterator != end; iterator.increment(iteratorError)) {
        std::error_code entryError;
        if (!iterator->is_regular_file(entryError) || entryError) continue;
        if (lowerAscii(pathUtf8(iterator->path().extension())) == ".vtk") {
            result.push_back(iterator->path());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<fs::path> matchingDatetime(const std::vector<fs::path>& paths,
                                       const std::optional<std::string>& datetime) {
    if (!datetime) return paths;
    std::vector<fs::path> result;
    for (const fs::path& path : paths) {
        if (pathUtf8(path.filename()).find(*datetime) != std::string::npos) result.push_back(path);
    }
    return result;
}

std::vector<fs::path> discoverVtk(const CaseConfig& config, std::ostream& output) {
    std::vector<fs::path> candidates;
    const fs::path temporary = config.projectHome / "proj_temp" / "vtk";
    const std::vector<fs::path> temporaryFiles = vtkFilesIn(temporary);
    if (!temporaryFiles.empty() || fs::is_directory(temporary)) {
        if (config.datetime) {
            candidates = matchingDatetime(temporaryFiles, config.datetime);
            if (candidates.empty()) {
                warn(output, "No VTK in " + pathUtf8(temporary) + " matched datetime="
                             + *config.datetime
                             + ". Falling back to all VTK files in this folder.");
                candidates = temporaryFiles;
            }
        } else {
            candidates = temporaryFiles;
        }
    }

    if (candidates.empty()) {
        const fs::path results = config.projectHome / "RESULTS";
        const std::vector<fs::path> resultFiles = vtkFilesIn(results);
        if (config.datetime) {
            candidates = matchingDatetime(resultFiles, config.datetime);
            if (candidates.empty()) candidates = resultFiles;
        } else {
            candidates = resultFiles;
        }
    }

    std::set<fs::path> unique;
    for (const fs::path& path : candidates) unique.insert(resolved(path));
    std::vector<fs::path> result(unique.begin(), unique.end());
    if (result.empty()) {
        fail(output, "No VTK files found for case '" + config.casename + "'. Checked: "
                     + pathUtf8(config.projectHome / "proj_temp" / "vtk") + " and "
                     + pathUtf8(config.projectHome / "RESULTS"));
    }

    info(output, "Discovered " + std::to_string(result.size()) + " VTK file(s) for processing.");
    for (const fs::path& path : result) info(output, "  - " + pathUtf8(path));
    return result;
}

struct TransformModel final {
    std::string utmCrs;
    double rotateDegrees = 0.0;
    double pivotX = 0.0;
    double pivotY = 0.0;
    double originRotX = 0.0;
    double originRotY = 0.0;
    Projection llToUtm;
    Projection utmToLl;

    TransformModel(std::string crs, double rotate, double pivotXValue, double pivotYValue,
                   double originXValue, double originYValue)
        : utmCrs(std::move(crs)), rotateDegrees(rotate), pivotX(pivotXValue), pivotY(pivotYValue),
          originRotX(originXValue), originRotY(originYValue),
          llToUtm("EPSG:4326", utmCrs), utmToLl(utmCrs, "EPSG:4326") {}

    [[nodiscard]] std::pair<double, double> rotated(double x, double y, double degrees) const {
        const double radians = degrees * (std::acos(-1.0) / 180.0);
        const double cosine = std::cos(radians);
        const double sine = std::sin(radians);
        return {cosine * (x - pivotX) - sine * (y - pivotY) + pivotX,
                sine * (x - pivotX) + cosine * (y - pivotY) + pivotY};
    }

    void localToLonLat(const std::vector<double>& localX, const std::vector<double>& localY,
                       std::vector<double>& longitude, std::vector<double>& latitude) const {
        if (localX.size() != localY.size()) throw std::invalid_argument("Coordinate size mismatch");
        std::vector<PJ_COORD> coordinates(localX.size());
        for (std::size_t index = 0; index < localX.size(); ++index) {
            const auto utm = rotated(localX[index] + originRotX,
                                     localY[index] + originRotY, -rotateDegrees);
            coordinates[index] = proj_coord(utm.first, utm.second, 0.0, 0.0);
        }
        utmToLl.apply(coordinates);
        longitude.resize(coordinates.size());
        latitude.resize(coordinates.size());
        for (std::size_t index = 0; index < coordinates.size(); ++index) {
            longitude[index] = coordinates[index].xy.x;
            latitude[index] = coordinates[index].xy.y;
        }
    }

    void lonLatToLocal(const std::vector<double>& longitude, const std::vector<double>& latitude,
                       std::vector<double>& localX, std::vector<double>& localY) const {
        if (longitude.size() != latitude.size()) throw std::invalid_argument("Coordinate size mismatch");
        std::vector<PJ_COORD> coordinates(longitude.size());
        for (std::size_t index = 0; index < longitude.size(); ++index) {
            coordinates[index] = proj_coord(longitude[index], latitude[index], 0.0, 0.0);
        }
        llToUtm.apply(coordinates);
        localX.resize(coordinates.size());
        localY.resize(coordinates.size());
        for (std::size_t index = 0; index < coordinates.size(); ++index) {
            const auto local = rotated(coordinates[index].xy.x, coordinates[index].xy.y,
                                       rotateDegrees);
            localX[index] = local.first - originRotX;
            localY[index] = local.second - originRotY;
        }
    }
};

TransformModel buildTransform(const CaseConfig& config, std::ostream& output) {
    Projection forward("EPSG:4326", config.utmCrs);
    const std::array<std::pair<double, double>, 4> geographic{{
        {config.cutLon.first, config.cutLat.first},
        {config.cutLon.second, config.cutLat.first},
        {config.cutLon.second, config.cutLat.second},
        {config.cutLon.first, config.cutLat.second},
    }};
    std::array<double, 4> x{};
    std::array<double, 4> y{};
    for (std::size_t index = 0; index < geographic.size(); ++index) {
        const auto projected = forward.apply(geographic[index].first, geographic[index].second);
        x[index] = projected.first;
        y[index] = projected.second;
    }
    const double pivotX = (x[0] + x[1] + x[2] + x[3]) / 4.0;
    const double pivotY = (y[0] + y[1] + y[2] + y[3]) / 4.0;
    const double radians = config.rotateDegrees * (std::acos(-1.0) / 180.0);
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    double originX = std::numeric_limits<double>::infinity();
    double originY = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < x.size(); ++index) {
        const double rotatedX = cosine * (x[index] - pivotX) - sine * (y[index] - pivotY) + pivotX;
        const double rotatedY = sine * (x[index] - pivotX) + cosine * (y[index] - pivotY) + pivotY;
        originX = std::min(originX, rotatedX);
        originY = std::min(originY, rotatedY);
    }
    info(output, "Transform model built: utm_crs=" + config.utmCrs
                 + ", rotate_deg=" + fixed(config.rotateDegrees, 9)
                 + ", pivot=(" + fixed(pivotX, 3) + ", " + fixed(pivotY, 3)
                 + "), local_origin_rot=(" + fixed(originX, 3) + ", " + fixed(originY, 3) + ").");
    return TransformModel(config.utmCrs, config.rotateDegrees, pivotX, pivotY, originX, originY);
}

LegacyStructuredPoints parseVtk(const fs::path& path, std::ostream& output) {
    LegacyStructuredPoints vtk;
    try {
        vtk = LegacyStructuredPoints::readHeader(path);
    } catch (const std::exception& exception) {
        fail(output, exception.what());
    }
    if (vtk.encoding == LegacyVtkEncoding::Binary) {
        for (const LegacyVtkField& field : vtk.fields) {
            if (field.association != LegacyVtkAssociation::PointData
                || field.kind != LegacyVtkFieldKind::Scalars) {
                fail(output, "Unsupported VTK data section header for field '" + field.name
                             + "' in " + pathUtf8(path));
            }
        }
    } else {
        warn(output, "ASCII VTK compatibility extension enabled for file: " + pathUtf8(path));
        vtk.fields.erase(std::remove_if(vtk.fields.begin(), vtk.fields.end(),
            [](const LegacyVtkField& field) {
                return field.association != LegacyVtkAssociation::PointData;
            }), vtk.fields.end());
    }
    if (vtk.fields.empty()) fail(output, "No SCALARS fields found in VTK file: " + pathUtf8(path));
    return vtk;
}

std::string sanitizeName(std::string_view raw) {
    std::string trimmed(raw);
    const auto first = trimmed.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) trimmed.clear();
    else trimmed = trimmed.substr(first, trimmed.find_last_not_of(" \t\r\n") - first + 1);
    std::string result;
    bool replacing = false;
    for (const unsigned char character : trimmed) {
        const bool upper = character >= 'A' && character <= 'Z';
        const bool lower = character >= 'a' && character <= 'z';
        const bool digit = character >= '0' && character <= '9';
        const bool valid = upper || lower || digit || character == '_';
        if (valid) {
            result.push_back(static_cast<char>(upper ? character - 'A' + 'a' : character));
            replacing = false;
        } else if (!replacing) {
            result.push_back('_');
            replacing = true;
        }
    }
    if (result.empty()) result = "var";
    if (result.front() >= '0' && result.front() <= '9') result = "v_" + result;
    return result;
}

struct OutputSpec final {
    std::string outputName;
    const LegacyVtkField* field = nullptr;
    std::size_t component = 0;
    std::string longName;
    std::string units;
    int variable = -1;
};

std::vector<OutputSpec> buildOutputSpecs(const LegacyStructuredPoints& vtk, std::ostream& output) {
    const std::set<std::string> priority{"data", "u", "u_avg", "uvw", "velocity"};
    const LegacyVtkField* mainVector = nullptr;
    for (const LegacyVtkField& field : vtk.fields) {
        if (field.components == 3 && priority.contains(lowerAscii(field.name))) {
            mainVector = &field;
            break;
        }
    }
    if (!mainVector) {
        for (const LegacyVtkField& field : vtk.fields) {
            if (field.components == 3) {
                mainVector = &field;
                break;
            }
        }
    }

    std::map<std::string, int> used;
    auto unique = [&](std::string base) {
        if (base == "height" || base == "lat" || base == "lon") base += "_var";
        const int index = used[base];
        if (index == 0) {
            used[base] = 1;
            return base;
        }
        used[base] = index + 1;
        return base + '_' + std::to_string(index + 1);
    };

    std::vector<OutputSpec> result;
    if (mainVector) {
        result.push_back({"u", mainVector, 0, mainVector->name + " component x", "m s-1"});
        result.push_back({"v", mainVector, 1, mainVector->name + " component y", "m s-1"});
        result.push_back({"w", mainVector, 2, mainVector->name + " component z", "m s-1"});
        used["u"] = 1;
        used["v"] = 1;
        used["w"] = 1;
    }

    for (const LegacyVtkField& field : vtk.fields) {
        if (&field == mainVector) continue;
        std::string base = sanitizeName(field.name);
        if (field.components == 1) {
            if (base == "data") {
                base = lowerAscii(pathUtf8(vtk.path.stem())).find("rho") != std::string::npos
                    ? "rho" : "scalar";
            }
            result.push_back({unique(base), &field, 0, field.name + " scalar", ""});
        } else if (field.components == 3) {
            constexpr std::array<std::string_view, 3> suffixes{"x", "y", "z"};
            for (std::size_t component = 0; component < suffixes.size(); ++component) {
                const std::string suffix(suffixes[component]);
                result.push_back({unique(base + '_' + suffix), &field, component,
                                  field.name + " component " + suffix, ""});
            }
        } else {
            for (std::size_t component = 0; component < field.components; ++component) {
                result.push_back({unique(base + "_c" + std::to_string(component)), &field, component,
                                  field.name + " component " + std::to_string(component), ""});
            }
        }
    }
    if (result.empty()) fail(output, "No supported fields found in VTK file: " + pathUtf8(vtk.path));

    std::ostringstream message;
    message << "Output variables for " << pathUtf8(vtk.path.filename()) << ": ";
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (index) message << ", ";
        message << result[index].outputName;
    }
    info(output, message.str());
    return result;
}

std::vector<double> axis(double origin, double spacing, int count) {
    std::vector<double> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        result[static_cast<std::size_t>(index)] = origin + static_cast<double>(index) * spacing;
    }
    return result;
}

std::vector<double> linspace(double first, double last, int count) {
    std::vector<double> result(static_cast<std::size_t>(count));
    if (count <= 1) {
        if (count == 1) result.front() = first;
        return result;
    }
    const double step = (last - first) / static_cast<double>(count - 1);
    for (int index = 0; index < count; ++index) {
        result[static_cast<std::size_t>(index)] = first + static_cast<double>(index) * step;
    }
    result.back() = last;
    return result;
}

struct CompleteBounds final {
    double longitudeLow = 0.0;
    double longitudeHigh = 0.0;
    double latitudeLow = 0.0;
    double latitudeHigh = 0.0;
};

CompleteBounds completeBounds(const TransformModel& model,
                              const std::vector<double>& x,
                              const std::vector<double>& y,
                              std::ostream& output) {
    std::vector<double> leftX(y.size(), x.front());
    std::vector<double> rightX(y.size(), x.back());
    std::vector<double> longitudeLeft;
    std::vector<double> latitudeUnused;
    std::vector<double> longitudeRight;
    model.localToLonLat(leftX, y, longitudeLeft, latitudeUnused);
    model.localToLonLat(rightX, y, longitudeRight, latitudeUnused);
    double longitudeLow = -std::numeric_limits<double>::infinity();
    double longitudeHigh = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < y.size(); ++index) {
        longitudeLow = std::max(longitudeLow, std::min(longitudeLeft[index], longitudeRight[index]));
        longitudeHigh = std::min(longitudeHigh, std::max(longitudeLeft[index], longitudeRight[index]));
    }

    std::vector<double> bottomY(x.size(), y.front());
    std::vector<double> topY(x.size(), y.back());
    std::vector<double> longitudeUnused;
    std::vector<double> latitudeBottom;
    std::vector<double> latitudeTop;
    model.localToLonLat(x, bottomY, longitudeUnused, latitudeBottom);
    model.localToLonLat(x, topY, longitudeUnused, latitudeTop);
    double latitudeLow = -std::numeric_limits<double>::infinity();
    double latitudeHigh = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < x.size(); ++index) {
        latitudeLow = std::max(latitudeLow, std::min(latitudeBottom[index], latitudeTop[index]));
        latitudeHigh = std::min(latitudeHigh, std::max(latitudeBottom[index], latitudeTop[index]));
    }

    if (longitudeHigh < longitudeLow) std::swap(longitudeLow, longitudeHigh);
    if (latitudeHigh < latitudeLow) std::swap(latitudeLow, latitudeHigh);
    if (!std::isfinite(longitudeLow) || !std::isfinite(longitudeHigh)
        || !std::isfinite(latitudeLow) || !std::isfinite(latitudeHigh)) {
        fail(output, "Non-finite complete lon/lat bounds detected.");
    }
    if (longitudeHigh <= longitudeLow || latitudeHigh <= latitudeLow) {
        fail(output, "Failed to derive a valid complete lon/lat range. lon=["
                     + general(longitudeLow) + ", " + general(longitudeHigh) + "], lat=["
                     + general(latitudeLow) + ", " + general(latitudeHigh) + ']');
    }
    return {longitudeLow, longitudeHigh, latitudeLow, latitudeHigh};
}

double medianAbsoluteDifference(const std::vector<double>& values) {
    if (values.size() <= 1) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> differences(values.size() - 1);
    for (std::size_t index = 1; index < values.size(); ++index) {
        differences[index - 1] = std::abs(values[index] - values[index - 1]);
    }
    std::sort(differences.begin(), differences.end());
    const std::size_t middle = differences.size() / 2;
    if ((differences.size() & 1U) != 0U) return differences[middle];
    return 0.5 * (differences[middle - 1] + differences[middle]);
}

std::pair<double, double> estimateResolution(const TransformModel& model,
                                             const std::vector<double>& x,
                                             const std::vector<double>& y,
                                             double longitudeSpan,
                                             double latitudeSpan,
                                             std::ostream& output) {
    const double xMiddle = x[x.size() / 2];
    const double yMiddle = y[y.size() / 2];
    std::vector<double> atY(x.size(), yMiddle);
    std::vector<double> longitude;
    std::vector<double> ignored;
    model.localToLonLat(x, atY, longitude, ignored);
    std::vector<double> atX(y.size(), xMiddle);
    std::vector<double> latitude;
    model.localToLonLat(atX, y, ignored, latitude);
    double longitudeStep = medianAbsoluteDifference(longitude);
    double latitudeStep = medianAbsoluteDifference(latitude);
    if (!std::isfinite(longitudeStep) || longitudeStep <= 0.0) {
        longitudeStep = longitudeSpan / static_cast<double>(std::max<std::size_t>(1, x.size() - 1));
    }
    if (!std::isfinite(latitudeStep) || latitudeStep <= 0.0) {
        latitudeStep = latitudeSpan / static_cast<double>(std::max<std::size_t>(1, y.size() - 1));
    }
    if (longitudeStep <= 0.0 || latitudeStep <= 0.0) {
        fail(output, "Failed to estimate valid lon/lat resolution.");
    }
    return {longitudeStep, latitudeStep};
}

struct TargetMapping final {
    std::vector<double> longitude;
    std::vector<double> latitude;
    std::vector<double> sourceX;
    std::vector<double> sourceY;
    double outsideRatio = 0.0;
};

int roundedGridCount(double span, double reference) {
    const double rounded = std::nearbyint(span / reference);
    if (!std::isfinite(rounded) || rounded > static_cast<double>(std::numeric_limits<int>::max() - 1)) {
        throw std::overflow_error("Target grid dimension exceeds integer range");
    }
    return std::max(2, static_cast<int>(rounded) + 1);
}

TargetMapping targetMapping(const TransformModel& model,
                            const std::vector<double>& x,
                            const std::vector<double>& y,
                            const CompleteBounds& bounds,
                            std::ostream& output) {
    const std::size_t nx = x.size();
    const std::size_t ny = y.size();
    const double spacingX = nx > 1 ? x[1] - x[0] : 1.0;
    const double spacingY = ny > 1 ? y[1] - y[0] : 1.0;
    if (std::abs(spacingX) < 1.0e-12 || std::abs(spacingY) < 1.0e-12) {
        fail(output, "Invalid source spacing for mapping: sx=" + general(spacingX)
                     + ", sy=" + general(spacingY));
    }
    const auto resolution = estimateResolution(model, x, y,
        bounds.longitudeHigh - bounds.longitudeLow,
        bounds.latitudeHigh - bounds.latitudeLow, output);
    std::array<double, 4> current{bounds.longitudeLow, bounds.longitudeHigh,
                                  bounds.latitudeLow, bounds.latitudeHigh};
    double lastOutside = 1.0;
    for (int attempt = 0; attempt < 12; ++attempt) {
        const double longitudeLow = current[0];
        const double longitudeHigh = current[1];
        const double latitudeLow = current[2];
        const double latitudeHigh = current[3];
        if (longitudeHigh <= longitudeLow || latitudeHigh <= latitudeLow) {
            fail(output, "Cropping collapsed while searching complete lon/lat mapping range.");
        }

        int outputNx = roundedGridCount(longitudeHigh - longitudeLow, resolution.first);
        int outputNy = roundedGridCount(latitudeHigh - latitudeLow, resolution.second);
        const std::size_t maximumNx = std::max<std::size_t>(2, nx * 4);
        const std::size_t maximumNy = std::max<std::size_t>(2, ny * 4);
        outputNx = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(outputNx), maximumNx));
        outputNy = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(outputNy), maximumNy));
        std::vector<double> longitude = linspace(longitudeLow, longitudeHigh, outputNx);
        std::vector<double> latitude = linspace(latitudeLow, latitudeHigh, outputNy);
        const std::size_t targetSize = static_cast<std::size_t>(outputNx)
            * static_cast<std::size_t>(outputNy);
        std::vector<double> longitude2d(targetSize);
        std::vector<double> latitude2d(targetSize);
        for (int iy = 0; iy < outputNy; ++iy) {
            for (int ix = 0; ix < outputNx; ++ix) {
                const std::size_t index = static_cast<std::size_t>(iy) * static_cast<std::size_t>(outputNx)
                    + static_cast<std::size_t>(ix);
                longitude2d[index] = longitude[static_cast<std::size_t>(ix)];
                latitude2d[index] = latitude[static_cast<std::size_t>(iy)];
            }
        }
        std::vector<double> localX;
        std::vector<double> localY;
        model.lonLatToLocal(longitude2d, latitude2d, localX, localY);
        std::size_t insideCount = 0;
        for (std::size_t index = 0; index < targetSize; ++index) {
            localX[index] = (localX[index] - x.front()) / spacingX;
            localY[index] = (localY[index] - y.front()) / spacingY;
            if (localX[index] >= -1.0e-6 && localX[index] <= static_cast<double>(nx - 1) + 1.0e-6
                && localY[index] >= -1.0e-6 && localY[index] <= static_cast<double>(ny - 1) + 1.0e-6) {
                ++insideCount;
            }
        }
        const double outside = 1.0 - static_cast<double>(insideCount) / static_cast<double>(targetSize);
        info(output, "Target mapping attempt " + std::to_string(attempt + 1)
                     + ": grid=" + std::to_string(outputNx) + 'x' + std::to_string(outputNy)
                     + ", outside ratio=" + scientific(outside, 6)
                     + ", lon=[" + fixed(longitudeLow, 8) + ", " + fixed(longitudeHigh, 8)
                     + "], lat=[" + fixed(latitudeLow, 8) + ", " + fixed(latitudeHigh, 8) + ']');
        if (outside <= 0.0) {
            for (std::size_t index = 0; index < targetSize; ++index) {
                localX[index] = std::clamp(localX[index], 0.0, static_cast<double>(nx - 1));
                localY[index] = std::clamp(localY[index], 0.0, static_cast<double>(ny - 1));
            }
            return {std::move(longitude), std::move(latitude), std::move(localX),
                    std::move(localY), outside};
        }
        if (outside >= lastOutside - 1.0e-9 && attempt >= 2) {
            fail(output, "Could not build a complete Cartesian lon/lat target grid. Last outside ratio="
                         + scientific(outside, 6));
        }
        lastOutside = outside;
        const double longitudeMargin = std::max(resolution.first * 2.0,
                                                 (longitudeHigh - longitudeLow) * 0.02);
        const double latitudeMargin = std::max(resolution.second * 2.0,
                                                (latitudeHigh - latitudeLow) * 0.02);
        current = {longitudeLow + longitudeMargin, longitudeHigh - longitudeMargin,
                   latitudeLow + latitudeMargin, latitudeHigh - latitudeMargin};
    }
    fail(output, "Exceeded max attempts while searching a complete lon/lat mapping range.");
}

void splineFilterLine(double* coefficients, std::size_t count, std::size_t stride) {
    if (count == 1) return;
    // Cubic pole and reflect/nearest initial conditions from SciPy's
    // ni_splines.c. map_coordinates pre-pads nearest mode by 12 cells and
    // then calls this filter in float64.
    constexpr double pole = -0.267949192431122706472553658494127633;
    constexpr double gain = (1.0 - pole) * (1.0 - 1.0 / pole);
    for (std::size_t index = 0; index < count; ++index) coefficients[index * stride] *= gain;

    const double poleToN = std::pow(pole, static_cast<double>(count));
    const double originalFirst = coefficients[0];
    coefficients[0] += poleToN * coefficients[(count - 1) * stride];
    double polePower = pole;
    for (std::size_t index = 1; index < count; ++index) {
        coefficients[0] += polePower
            * (coefficients[index * stride]
               + poleToN * coefficients[(count - 1 - index) * stride]);
        polePower *= pole;
    }
    coefficients[0] *= pole / (1.0 - poleToN * poleToN);
    coefficients[0] += originalFirst;

    for (std::size_t index = 1; index < count; ++index) {
        coefficients[index * stride] += pole * coefficients[(index - 1) * stride];
    }
    coefficients[(count - 1) * stride] *= pole / (pole - 1.0);
    for (std::size_t index = count - 1; index-- > 0;) {
        coefficients[index * stride] = pole
            * (coefficients[(index + 1) * stride] - coefficients[index * stride]);
    }
}

std::vector<float> cubicMapCoordinates(const std::vector<float>& plane,
                                       std::size_t nx, std::size_t ny,
                                       const TargetMapping& mapping,
                                       int workers) {
    const std::size_t paddedNx = nx + 2U * cubicPadding;
    const std::size_t paddedNy = ny + 2U * cubicPadding;
    std::vector<double> coefficients(paddedNx * paddedNy);
    for (std::size_t py = 0; py < paddedNy; ++py) {
        const std::size_t sourceY = std::clamp<std::ptrdiff_t>(
            static_cast<std::ptrdiff_t>(py) - cubicPadding, 0,
            static_cast<std::ptrdiff_t>(ny - 1));
        for (std::size_t px = 0; px < paddedNx; ++px) {
            const std::size_t sourceX = std::clamp<std::ptrdiff_t>(
                static_cast<std::ptrdiff_t>(px) - cubicPadding, 0,
                static_cast<std::ptrdiff_t>(nx - 1));
            coefficients[py * paddedNx + px] = plane[sourceY * nx + sourceX];
        }
    }

    // scipy.ndimage.spline_filter applies axis 0 and then axis 1.
    for (std::size_t column = 0; column < paddedNx; ++column) {
        splineFilterLine(coefficients.data() + column, paddedNy, paddedNx);
    }
    for (std::size_t row = 0; row < paddedNy; ++row) {
        splineFilterLine(coefficients.data() + row * paddedNx, paddedNx, 1);
    }

    std::vector<float> result(mapping.sourceX.size());
    const std::ptrdiff_t targetCount = static_cast<std::ptrdiff_t>(result.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(workers) if(targetCount > 1024)
#endif
    for (std::ptrdiff_t rawIndex = 0; rawIndex < targetCount; ++rawIndex) {
        const std::size_t index = static_cast<std::size_t>(rawIndex);
        const double x = mapping.sourceX[index] + cubicPadding;
        const double y = mapping.sourceY[index] + cubicPadding;
        const std::ptrdiff_t floorX = static_cast<std::ptrdiff_t>(std::floor(x));
        const std::ptrdiff_t floorY = static_cast<std::ptrdiff_t>(std::floor(y));
        const double tx = x - static_cast<double>(floorX);
        const double ty = y - static_cast<double>(floorY);
        const double zx = 1.0 - tx;
        const double zy = 1.0 - ty;
        std::array<double, 4> wx{{
            zx * zx * zx / 6.0,
            (tx * tx * (tx - 2.0) * 3.0 + 4.0) / 6.0,
            (zx * zx * (zx - 2.0) * 3.0 + 4.0) / 6.0,
            1.0,
        }};
        std::array<double, 4> wy{{
            zy * zy * zy / 6.0,
            (ty * ty * (ty - 2.0) * 3.0 + 4.0) / 6.0,
            (zy * zy * (zy - 2.0) * 3.0 + 4.0) / 6.0,
            1.0,
        }};
        for (std::size_t weight = 0; weight < 3; ++weight) {
            wx[3] -= wx[weight];
            wy[3] -= wy[weight];
        }
        double value = 0.0;
        for (std::size_t jy = 0; jy < 4; ++jy) {
            const std::size_t coefficientY = static_cast<std::size_t>(floorY - 1
                + static_cast<std::ptrdiff_t>(jy));
            double rowValue = 0.0;
            for (std::size_t ix = 0; ix < 4; ++ix) {
                const std::size_t coefficientX = static_cast<std::size_t>(floorX - 1
                    + static_cast<std::ptrdiff_t>(ix));
                rowValue += wx[ix] * coefficients[coefficientY * paddedNx + coefficientX];
            }
            value += wy[jy] * rowValue;
        }
        result[index] = static_cast<float>(value);
    }
    return result;
}

int defineCoordinateVariable(int ncid, const char* name, nc_type type, int dimension) {
    int variable = -1;
    ncCheck(nc_def_var(ncid, name, type, 1, &dimension, &variable),
            "Failed to define NetCDF coordinate variable '" + std::string(name) + "'");
    ncCheck(nc_def_var_deflate(ncid, variable, 1, 1, 3),
            "Failed to configure NetCDF compression for '" + std::string(name) + "'");
    return variable;
}

int defineDataVariable(int ncid, const std::string& name, const std::array<int, 3>& dimensions,
                       const std::array<std::size_t, 3>& chunks) {
    int variable = -1;
    ncCheck(nc_def_var(ncid, name.c_str(), NC_FLOAT, 3, dimensions.data(), &variable),
            "Failed to define NetCDF variable '" + name + "'");
    ncCheck(nc_def_var_deflate(ncid, variable, 1, 1, 3),
            "Failed to configure NetCDF compression for '" + name + "'");
    ncCheck(nc_def_var_chunking(ncid, variable, NC_CHUNKED, chunks.data()),
            "Failed to configure NetCDF chunks for '" + name + "'");
    const float fill = std::numeric_limits<float>::quiet_NaN();
    ncCheck(nc_def_var_fill(ncid, variable, 0, &fill),
            "Failed to configure NetCDF fill value for '" + name + "'");
    return variable;
}

fs::path processVtk(const LegacyStructuredPoints& vtk, const TransformModel& model,
                    const fs::path& outputDirectory, int workers, std::ostream& output) {
    const int nx = vtk.dimensions[0];
    const int ny = vtk.dimensions[1];
    const int nz = vtk.dimensions[2];
    info(output, "Processing " + pathUtf8(vtk.path.filename()) + ": dims=("
                 + std::to_string(nx) + ", " + std::to_string(ny) + ", " + std::to_string(nz)
                 + "), origin=(" + fixed(vtk.origin[0], 6) + ", " + fixed(vtk.origin[1], 6)
                 + ", " + fixed(vtk.origin[2], 6) + "), spacing=(" + fixed(vtk.spacing[0], 6)
                 + ", " + fixed(vtk.spacing[1], 6) + ", " + fixed(vtk.spacing[2], 6) + ')');

    const std::vector<double> x = axis(vtk.origin[0], vtk.spacing[0], nx);
    const std::vector<double> y = axis(vtk.origin[1], vtk.spacing[1], ny);
    const std::vector<double> z = axis(vtk.origin[2], vtk.spacing[2], nz);
    const CompleteBounds bounds = completeBounds(model, x, y, output);
    info(output, "Largest complete lon/lat axis-aligned range: lon=["
                 + fixed(bounds.longitudeLow, 8) + ", " + fixed(bounds.longitudeHigh, 8)
                 + "], lat=[" + fixed(bounds.latitudeLow, 8) + ", "
                 + fixed(bounds.latitudeHigh, 8) + ']');
    const TargetMapping mapping = targetMapping(model, x, y, bounds, output);
    const std::size_t outputNx = mapping.longitude.size();
    const std::size_t outputNy = mapping.latitude.size();
    info(output, "Target Cartesian lon/lat grid built: nx=" + std::to_string(outputNx)
                 + ", ny=" + std::to_string(outputNy)
                 + ", outside ratio=" + scientific(mapping.outsideRatio, 6));
    std::vector<OutputSpec> specs = buildOutputSpecs(vtk, output);

    fs::create_directories(outputDirectory);
    fs::path outputName = vtk.path.stem();
    outputName += ".nc";
    const fs::path outputPath = outputDirectory / outputName;
    NetcdfFile file(outputPath);
    const int ncid = file.id();
    int dimensionHeight = -1;
    int dimensionLatitude = -1;
    int dimensionLongitude = -1;
    ncCheck(nc_def_dim(ncid, "height", static_cast<std::size_t>(nz), &dimensionHeight),
            "Failed to define height dimension");
    ncCheck(nc_def_dim(ncid, "lat", outputNy, &dimensionLatitude), "Failed to define lat dimension");
    ncCheck(nc_def_dim(ncid, "lon", outputNx, &dimensionLongitude), "Failed to define lon dimension");
    const int variableHeight = defineCoordinateVariable(ncid, "height", NC_FLOAT, dimensionHeight);
    const int variableLatitude = defineCoordinateVariable(ncid, "lat", NC_DOUBLE, dimensionLatitude);
    const int variableLongitude = defineCoordinateVariable(ncid, "lon", NC_DOUBLE, dimensionLongitude);
    setTextAttribute(ncid, variableHeight, "long_name", "height above local model origin");
    setTextAttribute(ncid, variableHeight, "units", "m");
    setTextAttribute(ncid, variableLatitude, "long_name", "latitude");
    setTextAttribute(ncid, variableLatitude, "units", "degree_north");
    setTextAttribute(ncid, variableLongitude, "long_name", "longitude");
    setTextAttribute(ncid, variableLongitude, "units", "degree_east");

    const std::array<int, 3> dimensions{dimensionHeight, dimensionLatitude, dimensionLongitude};
    const std::array<std::size_t, 3> chunks{1, std::min<std::size_t>(outputNy, 256),
                                            std::min<std::size_t>(outputNx, 256)};
    for (OutputSpec& spec : specs) {
        spec.variable = defineDataVariable(ncid, spec.outputName, dimensions, chunks);
        if (!spec.longName.empty()) setTextAttribute(ncid, spec.variable, "long_name", spec.longName);
        if (!spec.units.empty()) setTextAttribute(ncid, spec.variable, "units", spec.units);
    }
    setTextAttribute(ncid, NC_GLOBAL, "source_vtk", pathUtf8(vtk.path));
    setTextAttribute(ncid, NC_GLOBAL, "utm_crs", model.utmCrs);
    setTextAttribute(ncid, NC_GLOBAL, "rotate_deg", fixed(model.rotateDegrees, 12));
    setTextAttribute(ncid, NC_GLOBAL, "transform_pivot_xy_utm",
                     fixed(model.pivotX, 6) + ',' + fixed(model.pivotY, 6));
    setTextAttribute(ncid, NC_GLOBAL, "transform_local_origin_rot_xy",
                     fixed(model.originRotX, 6) + ',' + fixed(model.originRotY, 6));
    setTextAttribute(ncid, NC_GLOBAL, "note",
        "Converted from FluidX3D legacy VTK to strict Cartesian lon/lat grid using inverse UTM-rotation-origin-shift chain.");
    ncCheck(nc_enddef(ncid), "Failed to finish NetCDF schema");
    std::vector<float> height(z.size());
    std::transform(z.begin(), z.end(), height.begin(), [](double value) { return static_cast<float>(value); });
    ncCheck(nc_put_var_float(ncid, variableHeight, height.data()), "Failed to write height coordinates");
    ncCheck(nc_put_var_double(ncid, variableLatitude, mapping.latitude.data()), "Failed to write latitude");
    ncCheck(nc_put_var_double(ncid, variableLongitude, mapping.longitude.data()), "Failed to write longitude");

    info(output, "Start cubic interpolation for " + pathUtf8(vtk.path.filename())
                 + ": levels=" + std::to_string(nz) + ", variables=" + std::to_string(specs.size())
                 + ", workers=" + std::to_string(workers));
    const auto startTime = std::chrono::steady_clock::now();
    const std::array<std::size_t, 3> count{1, outputNy, outputNx};
    LegacyVtkPlaneReader reader(vtk);
    for (int level = 0; level < nz; ++level) {
        const std::array<std::size_t, 3> start{static_cast<std::size_t>(level), 0, 0};
        for (const OutputSpec& spec : specs) {
            const std::vector<float> plane = reader.readPointComponentPlane(
                *spec.field, static_cast<std::size_t>(level), spec.component);
            const std::vector<float> interpolated = cubicMapCoordinates(
                plane, static_cast<std::size_t>(nx), static_cast<std::size_t>(ny), mapping, workers);
            ncCheck(nc_put_vara_float(ncid, spec.variable, start.data(), count.data(), interpolated.data()),
                    "Failed to write " + spec.outputName);
        }
        const int interval = std::max(1, nz / 20);
        if ((level + 1) % interval == 0 || level + 1 == nz) {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - startTime).count();
            const std::string completion = workers > 1 && nz > 1 ? " levels completed (" : " levels (";
            info(output, "Interpolation progress: " + std::to_string(level + 1) + '/'
                         + std::to_string(nz) + completion + fixed(seconds, 1) + "s elapsed).");
        }
    }
    const double totalSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - startTime).count();
    info(output, "Finished interpolation and NetCDF writing in " + fixed(totalSeconds, 1)
                 + "s: " + pathUtf8(outputPath));
    file.close();
    return outputPath;
}

int execute(const std::vector<std::string>& arguments, std::ostream& output) {
    if (arguments.size() != 1) fail(output, "Usage: python vtk2nc_new.py <path_to_case.luw>");
    const CaseConfig config = parseConfig(pathFromUtf8(arguments.front()), output);
    info(output, "LUW path: " + pathUtf8(config.luwPath));
    info(output, "Case name: " + config.casename);
    if (config.datetime) info(output, "Case datetime: " + *config.datetime);
    else warn(output, "Case datetime is missing in LUW. VTK discovery falls back to all available VTK files.");
    const TransformModel model = buildTransform(config, output);
    const std::vector<fs::path> vtkFiles = discoverVtk(config, output);
    const unsigned int cpuCountRaw = std::thread::hardware_concurrency();
    const int cpuCount = static_cast<int>(cpuCountRaw == 0 ? 1 : cpuCountRaw);
    const int workers = std::max(1, std::min(cpuCount, maximumWorkers));
    info(output, "Runtime worker threads set to " + std::to_string(workers)
                 + " (CPU count=" + std::to_string(cpuCount) + ").");

    std::vector<fs::path> succeeded;
    std::vector<std::pair<fs::path, std::string>> failed;
    const fs::path outputDirectory = config.projectHome / "RESULTS";
    for (std::size_t index = 0; index < vtkFiles.size(); ++index) {
        info(output, std::string(88, '='));
        info(output, '[' + std::to_string(index + 1) + '/' + std::to_string(vtkFiles.size())
                     + "] Start file: " + pathUtf8(vtkFiles[index]));
        try {
            const LegacyStructuredPoints vtk = parseVtk(vtkFiles[index], output);
            succeeded.push_back(processVtk(vtk, model, outputDirectory, workers, output));
        } catch (const FatalExit&) {
            throw;
        } catch (const std::exception& exception) {
            failed.emplace_back(vtkFiles[index], exception.what());
            warn(output, "Failed file " + pathUtf8(vtkFiles[index]) + ": " + exception.what());
        }
    }
    info(output, std::string(88, '='));
    info(output, "Conversion summary: success=" + std::to_string(succeeded.size())
                 + ", failed=" + std::to_string(failed.size()));
    for (const fs::path& path : succeeded) info(output, "  [OK] " + pathUtf8(path));
    for (const auto& [path, message] : failed) {
        warn(output, "  [FAIL] " + pathUtf8(path) + " | " + message);
    }
    if (!failed.empty()) {
        fail(output, std::to_string(failed.size()) + " file(s) failed during conversion.", 2);
    }
    return 0;
}

} // namespace

int run_vtk2nc_new(const std::vector<std::string>& arguments,
                   std::ostream& output,
                   std::ostream& error) {
    (void)error;
    try {
        return execute(arguments, output);
    } catch (const FatalExit& exit) {
        return exit.code();
    } catch (const std::exception& exception) {
        writeLog(output, "ERROR", exception.what());
        return 1;
    }
}

} // namespace luw::cli
