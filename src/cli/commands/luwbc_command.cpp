#include "luw/luwbc_command.hpp"
#include "luw/luwbc_opencl.hpp"

#include "luw/deck_document.hpp"
#include "luw/dem_tif_cache.hpp"
#include "luw/preprocess_compute_config.hpp"
#include "luw/progress.hpp"
#include "luw/runtime_paths.hpp"

#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDelaunay2D.h>
#include <vtkFloatArray.h>
#include <vtkIdList.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkStaticCellLocator.h>
#include <vtkStaticPointLocator.h>
#include <vtkXMLImageDataWriter.h>
#include <vtk_libproj.h>
#include <vtk_netcdf.h>

#include <QFileInfo>
#include <QString>
#include <QVariant>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <limits>
#include <locale>
#include <mutex>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <cpl_conv.h>
#include <cpl_error.h>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#endif

namespace luw::cli {
namespace {

namespace fs = std::filesystem;

constexpr double kPi = 3.141592653589793238462643383279502884;

std::string pathToUtf8(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring()).toUtf8().toStdString();
#else
    return path.string();
#endif
}

fs::path pathFromUtf8(const std::string& text) {
#ifdef _WIN32
    return fs::path(QString::fromUtf8(text).toStdWString());
#else
    return fs::u8path(text);
#endif
}

QString pathToQString(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring());
#else
    return QString::fromUtf8(path.string());
#endif
}

fs::path absoluteNormalized(const fs::path& path) {
    std::error_code error;
    fs::path result = fs::absolute(path, error);
    if (error) result = path;
    return result.lexically_normal();
}

std::string lowerAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return text;
}

std::vector<std::uint64_t> exactDoubleBits(const std::vector<double>& values) {
    std::vector<std::uint64_t> result;
    result.reserve(values.size());
    for (const double value : values) {
        result.push_back(std::bit_cast<std::uint64_t>(value));
    }
    return result;
}

void info(std::ostream& output, const std::string& message) {
    output << "[INFO] " << message << '\n';
    output.flush();
}

void warn(std::ostream& output, const std::string& message) {
    output << "[WARN] " << message << '\n';
    output.flush();
}

void errorLine(std::ostream& output, const std::string& message) {
    output << "[ERROR] " << message << '\n';
    output.flush();
}

std::string fixed(double value, int precision) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string pythonFloat(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value < 0.0 ? "-inf" : "inf";
    std::array<char, 128> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                      std::chars_format::general);
    if (result.ec != std::errc{}) {
        std::ostringstream fallback;
        fallback.imbue(std::locale::classic());
        fallback << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        return fallback.str();
    }
    std::string text(buffer.data(), result.ptr);
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}

std::size_t checkedProduct(const std::vector<std::size_t>& shape) {
    std::size_t product = 1U;
    for (const std::size_t value : shape) {
        if (value != 0U && product > std::numeric_limits<std::size_t>::max() / value) {
            throw std::runtime_error("NetCDF variable is too large for this process");
        }
        product *= value;
    }
    return product;
}

void ncCheck(int status, const std::string& context) {
    if (status != NC_NOERR) {
        throw std::runtime_error(context + ": " + nc_strerror(status));
    }
}

class NcFile final {
public:
    explicit NcFile(const fs::path& path) {
        ncCheck(nc_open(pathToUtf8(path).c_str(), NC_NOWRITE, &id_),
                "Cannot open NetCDF " + pathToUtf8(path));
    }

    NcFile(const NcFile&) = delete;
    NcFile& operator=(const NcFile&) = delete;

    ~NcFile() {
        if (id_ >= 0) nc_close(id_);
    }

    [[nodiscard]] int id() const noexcept { return id_; }

private:
    int id_ = -1;
};

struct NcVariable {
    int id = -1;
    std::string name;
    nc_type type = NC_NAT;
    std::vector<int> dimensionIds;
    std::vector<std::string> dimensions;
    std::vector<std::size_t> shape;
};

class NcDataset final {
public:
    explicit NcDataset(const fs::path& path) : file_(path) {
        int dimensions = 0;
        int variables = 0;
        ncCheck(nc_inq(file_.id(), &dimensions, &variables, nullptr, nullptr),
                "Cannot inspect NetCDF structure");
        for (int variableId = 0; variableId < variables; ++variableId) {
            char name[NC_MAX_NAME + 1]{};
            nc_type type = NC_NAT;
            int dimensionCount = 0;
            ncCheck(nc_inq_var(file_.id(), variableId, name, &type, &dimensionCount,
                               nullptr, nullptr),
                    "Cannot inspect NetCDF variable");
            NcVariable variable;
            variable.id = variableId;
            variable.name = name;
            variable.type = type;
            variable.dimensionIds.resize(static_cast<std::size_t>(dimensionCount));
            if (dimensionCount > 0) {
                ncCheck(nc_inq_vardimid(file_.id(), variableId,
                                        variable.dimensionIds.data()),
                        "Cannot inspect dimensions for " + variable.name);
            }
            for (const int dimensionId : variable.dimensionIds) {
                char dimensionName[NC_MAX_NAME + 1]{};
                std::size_t length = 0U;
                ncCheck(nc_inq_dim(file_.id(), dimensionId, dimensionName, &length),
                        "Cannot inspect a NetCDF dimension");
                variable.dimensions.emplace_back(dimensionName);
                variable.shape.push_back(length);
                dimensionLengths_[dimensionName] = length;
            }
            variables_.emplace(variable.name, std::move(variable));
        }
    }

    [[nodiscard]] int id() const noexcept { return file_.id(); }

    [[nodiscard]] bool has(std::string_view name) const {
        return variables_.find(std::string(name)) != variables_.end();
    }

    [[nodiscard]] const NcVariable& variable(std::string_view name) const {
        const auto found = variables_.find(std::string(name));
        if (found == variables_.end()) {
            throw std::runtime_error("NetCDF variable not found: " + std::string(name));
        }
        return found->second;
    }

    [[nodiscard]] std::optional<std::string> firstVariable(
        std::initializer_list<std::string_view> candidates) const {
        for (const std::string_view name : candidates) {
            if (has(name)) return std::string(name);
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t> dimensionLength(std::string_view name) const {
        const auto found = dimensionLengths_.find(std::string(name));
        if (found == dimensionLengths_.end()) return std::nullopt;
        return found->second;
    }

    [[nodiscard]] std::optional<double> numericAttribute(const NcVariable& variable,
                                                         const char* name) const {
        nc_type type = NC_NAT;
        std::size_t count = 0U;
        if (nc_inq_att(id(), variable.id, name, &type, &count) != NC_NOERR || count == 0U) {
            return std::nullopt;
        }
        double value = 0.0;
        if (nc_get_att_double(id(), variable.id, name, &value) != NC_NOERR) {
            return std::nullopt;
        }
        return value;
    }

    [[nodiscard]] std::string textAttribute(const NcVariable& variable,
                                            const char* name) const {
        nc_type type = NC_NAT;
        std::size_t count = 0U;
        if (nc_inq_att(id(), variable.id, name, &type, &count) != NC_NOERR || count == 0U) {
            return {};
        }
        if (type == NC_CHAR) {
            std::string result(count, '\0');
            if (nc_get_att_text(id(), variable.id, name, result.data()) != NC_NOERR) return {};
            while (!result.empty() && result.back() == '\0') result.pop_back();
            return result;
        }
        return {};
    }

    [[nodiscard]] std::vector<double> read(const NcVariable& variable,
                                           const std::vector<std::size_t>& start,
                                           const std::vector<std::size_t>& count) const {
        if (start.size() != variable.shape.size() || count.size() != variable.shape.size()) {
            throw std::logic_error("Invalid NetCDF hyperslab rank for " + variable.name);
        }
        std::vector<double> data(checkedProduct(count));
        if (!data.empty()) {
            ncCheck(nc_get_vara_double(id(), variable.id, start.data(), count.data(), data.data()),
                    "Cannot read NetCDF variable " + variable.name);
        }

        const auto fillValue = numericAttribute(variable, "_FillValue");
        const auto missingValue = numericAttribute(variable, "missing_value");
        const double scale = numericAttribute(variable, "scale_factor").value_or(1.0);
        const double offset = numericAttribute(variable, "add_offset").value_or(0.0);
        for (double& value : data) {
            const bool missing = (fillValue && value == *fillValue)
                || (missingValue && value == *missingValue);
            value = missing ? std::numeric_limits<double>::quiet_NaN()
                            : value * scale + offset;
        }
        return data;
    }

    [[nodiscard]] std::vector<double> readAll(const NcVariable& variable) const {
        return read(variable, std::vector<std::size_t>(variable.shape.size(), 0U),
                    variable.shape);
    }

private:
    NcFile file_;
    std::map<std::string, NcVariable, std::less<>> variables_;
    std::map<std::string, std::size_t, std::less<>> dimensionLengths_;
};

std::string logicalDimension(std::string_view name) {
    if (name == "west_east") return "lon";
    if (name == "south_north") return "lat";
    if (name == "west_east_stag") return "lon_stag";
    if (name == "south_north_stag") return "lat_stag";
    return std::string(name);
}

bool isTimeDimension(std::string_view name) {
    return name == "Time" || name == "time";
}

std::optional<std::size_t> dimensionIndex(const NcVariable& variable,
                                          std::string_view logicalName) {
    for (std::size_t index = 0; index < variable.dimensions.size(); ++index) {
        if (logicalDimension(variable.dimensions[index]) == logicalName) return index;
    }
    return std::nullopt;
}

struct Crop {
    std::size_t x0 = 0U;
    std::size_t x1 = 0U;
    std::size_t y0 = 0U;
    std::size_t y1 = 0U;

    [[nodiscard]] std::size_t nx() const { return x1 - x0 + 1U; }
    [[nodiscard]] std::size_t ny() const { return y1 - y0 + 1U; }
};

struct Field3D {
    std::size_t nz = 0U;
    std::size_t ny = 0U;
    std::size_t nx = 0U;
    std::vector<float> values;

    Field3D() = default;
    Field3D(std::size_t z, std::size_t y, std::size_t x)
        : nz(z), ny(y), nx(x), values(z * y * x) {}

    [[nodiscard]] std::size_t index(std::size_t z,
                                    std::size_t y,
                                    std::size_t x) const {
        return (z * ny + y) * nx + x;
    }

    float& at(std::size_t z, std::size_t y, std::size_t x) {
        return values[index(z, y, x)];
    }

    [[nodiscard]] float at(std::size_t z, std::size_t y, std::size_t x) const {
        return values[index(z, y, x)];
    }
};

struct Surface2D {
    std::size_t ny = 0U;
    std::size_t nx = 0U;
    std::vector<double> values;

    [[nodiscard]] double at(std::size_t y, std::size_t x) const {
        return values[y * nx + x];
    }
};

std::size_t flatIndex(const std::vector<std::size_t>& shape,
                      const std::vector<std::size_t>& indices) {
    std::size_t result = 0U;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        result = result * shape[axis] + indices[axis];
    }
    return result;
}

Field3D readField3D(const NcDataset& dataset,
                    const std::string& name,
                    const std::string& verticalDimension,
                    const Crop& crop,
                    bool allowExtraFirst,
                    std::ostream& output) {
    const NcVariable& variable = dataset.variable(name);
    const auto xMass = dimensionIndex(variable, "lon");
    const auto xStag = dimensionIndex(variable, "lon_stag");
    const auto yMass = dimensionIndex(variable, "lat");
    const auto yStag = dimensionIndex(variable, "lat_stag");
    const auto zMass = dimensionIndex(variable, verticalDimension);
    const auto zStag = verticalDimension == "bottom_top"
        ? dimensionIndex(variable, "bottom_top_stag") : std::nullopt;
    const auto xAxis = xMass ? xMass : xStag;
    const auto yAxis = yMass ? yMass : yStag;
    const auto zAxis = zMass ? zMass : zStag;
    if (!xAxis || !yAxis || !zAxis) {
        throw std::runtime_error("Variable " + name + " does not have required dimensions (" +
                                 verticalDimension + ", lat, lon)");
    }

    std::vector<std::size_t> start(variable.shape.size(), 0U);
    std::vector<std::size_t> count = variable.shape;
    for (std::size_t axis = 0; axis < variable.shape.size(); ++axis) {
        const std::string logical = logicalDimension(variable.dimensions[axis]);
        if (isTimeDimension(variable.dimensions[axis])) {
            count[axis] = 1U;
        } else if (axis == *xAxis) {
            start[axis] = crop.x0;
            count[axis] = crop.nx() + (xStag ? 1U : 0U);
        } else if (axis == *yAxis) {
            start[axis] = crop.y0;
            count[axis] = crop.ny() + (yStag ? 1U : 0U);
        } else if (axis == *zAxis) {
            // Retain every vertical point; bottom_top_stag is averaged below.
        } else if (variable.shape[axis] != 1U) {
            if (!allowExtraFirst) {
                throw std::runtime_error("Variable " + name + " has unsupported extra dimension " +
                                         variable.dimensions[axis]);
            }
            warn(output, "Temperature field " + name + " has extra dimension " +
                         variable.dimensions[axis] + "; use the first index on that dimension");
            count[axis] = 1U;
        }
        if (start[axis] + count[axis] > variable.shape[axis]) {
            throw std::runtime_error("Crop range exceeds dimension " + variable.dimensions[axis] +
                                     " for variable " + name);
        }
        (void)logical;
    }

    if (xStag) info(output, "Destagger along lon_stag to lon for WRF staggered grid");
    if (yStag) info(output, "Destagger along lat_stag to lat for WRF staggered grid");
    if (zStag) info(output, "Destagger along bottom_top_stag to bottom_top for WRF staggered grid");

    const std::vector<double> raw = dataset.read(variable, start, count);
    const std::size_t nz = count[*zAxis] - (zStag ? 1U : 0U);
    const std::size_t ny = count[*yAxis] - (yStag ? 1U : 0U);
    const std::size_t nx = count[*xAxis] - (xStag ? 1U : 0U);
    Field3D result(nz, ny, nx);
    std::vector<std::size_t> indices(variable.shape.size(), 0U);

    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                double sum = 0.0;
                std::size_t samples = 0U;
                const std::size_t zCorners = zStag ? 2U : 1U;
                const std::size_t yCorners = yStag ? 2U : 1U;
                const std::size_t xCorners = xStag ? 2U : 1U;
                for (std::size_t dz = 0; dz < zCorners; ++dz) {
                    for (std::size_t dy = 0; dy < yCorners; ++dy) {
                        for (std::size_t dx = 0; dx < xCorners; ++dx) {
                            indices[*zAxis] = z + dz;
                            indices[*yAxis] = y + dy;
                            indices[*xAxis] = x + dx;
                            sum += raw[flatIndex(count, indices)];
                            ++samples;
                        }
                    }
                }
                result.at(z, y, x) = static_cast<float>(sum / static_cast<double>(samples));
            }
        }
    }
    return result;
}

Surface2D readSurface2D(const NcDataset& dataset,
                        const std::string& name,
                        const Crop& crop) {
    const NcVariable& variable = dataset.variable(name);
    const auto xAxis = dimensionIndex(variable, "lon");
    const auto yAxis = dimensionIndex(variable, "lat");
    if (!xAxis || !yAxis) throw std::runtime_error("Surface variable lacks lat/lon: " + name);
    std::vector<std::size_t> start(variable.shape.size(), 0U);
    std::vector<std::size_t> count = variable.shape;
    for (std::size_t axis = 0; axis < variable.shape.size(); ++axis) {
        if (isTimeDimension(variable.dimensions[axis])) count[axis] = 1U;
        else if (axis == *xAxis) { start[axis] = crop.x0; count[axis] = crop.nx(); }
        else if (axis == *yAxis) { start[axis] = crop.y0; count[axis] = crop.ny(); }
        else count[axis] = 1U;
    }
    const std::vector<double> raw = dataset.read(variable, start, count);
    Surface2D result{crop.ny(), crop.nx(), std::vector<double>(crop.ny() * crop.nx())};
    std::vector<std::size_t> indices(variable.shape.size(), 0U);
    for (std::size_t y = 0; y < crop.ny(); ++y) {
        for (std::size_t x = 0; x < crop.nx(); ++x) {
            indices[*xAxis] = x;
            indices[*yAxis] = y;
            result.values[y * crop.nx() + x] = raw[flatIndex(count, indices)];
        }
    }
    return result;
}

struct GeoGrid {
    std::size_t ny = 0U;
    std::size_t nx = 0U;
    bool rectilinear = false;
    std::vector<double> longitude;
    std::vector<double> latitude;

    [[nodiscard]] double lon(std::size_t y, std::size_t x) const {
        return rectilinear ? longitude[x] : longitude[y * nx + x];
    }

    [[nodiscard]] double lat(std::size_t y, std::size_t x) const {
        return rectilinear ? latitude[y] : latitude[y * nx + x];
    }
};

std::vector<double> readCoordinate1D(const NcDataset& dataset,
                                     const NcVariable& variable,
                                     std::size_t expected,
                                     std::string_view logicalDimensionName) {
    const auto axis = dimensionIndex(variable, logicalDimensionName);
    if (!axis || variable.shape[*axis] != expected) return {};
    std::vector<std::size_t> start(variable.shape.size(), 0U);
    std::vector<std::size_t> count(variable.shape.size(), 1U);
    count[*axis] = expected;
    return dataset.read(variable, start, count);
}

Surface2D readGeoCoordinate2D(const NcDataset& dataset,
                              const NcVariable& variable,
                              std::size_t massNy,
                              std::size_t massNx) {
    const auto xMass = dimensionIndex(variable, "lon");
    const auto xStag = dimensionIndex(variable, "lon_stag");
    const auto yMass = dimensionIndex(variable, "lat");
    const auto yStag = dimensionIndex(variable, "lat_stag");
    const auto xAxis = xMass ? xMass : xStag;
    const auto yAxis = yMass ? yMass : yStag;
    if (!xAxis || !yAxis) return {};
    std::vector<std::size_t> start(variable.shape.size(), 0U);
    std::vector<std::size_t> count(variable.shape.size(), 1U);
    count[*xAxis] = variable.shape[*xAxis];
    count[*yAxis] = variable.shape[*yAxis];
    const std::vector<double> raw = dataset.read(variable, start, count);
    const std::size_t nx = count[*xAxis] - (xStag ? 1U : 0U);
    const std::size_t ny = count[*yAxis] - (yStag ? 1U : 0U);
    if (nx != massNx || ny != massNy) return {};
    Surface2D result{ny, nx, std::vector<double>(ny * nx)};
    std::vector<std::size_t> indices(variable.shape.size(), 0U);
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            double sum = 0.0;
            std::size_t samples = 0U;
            for (std::size_t dy = 0; dy < (yStag ? 2U : 1U); ++dy) {
                for (std::size_t dx = 0; dx < (xStag ? 2U : 1U); ++dx) {
                    indices[*yAxis] = y + dy;
                    indices[*xAxis] = x + dx;
                    sum += raw[flatIndex(count, indices)];
                    ++samples;
                }
            }
            result.values[y * nx + x] = sum / static_cast<double>(samples);
        }
    }
    return result;
}

GeoGrid readGeoGrid(const NcDataset& dataset,
                    std::size_t massNy,
                    std::size_t massNx,
                    std::ostream& output) {
    if (dataset.has("lon") && dataset.has("lat")) {
        const NcVariable& lonVariable = dataset.variable("lon");
        const NcVariable& latVariable = dataset.variable("lat");
        std::vector<double> lon = readCoordinate1D(dataset, lonVariable, massNx, "lon");
        std::vector<double> lat = readCoordinate1D(dataset, latVariable, massNy, "lat");
        if (!lon.empty() && !lat.empty()) {
            return GeoGrid{massNy, massNx, true, std::move(lon), std::move(lat)};
        }
        const Surface2D lon2d = readGeoCoordinate2D(dataset, lonVariable, massNy, massNx);
        const Surface2D lat2d = readGeoCoordinate2D(dataset, latVariable, massNy, massNx);
        if (!lon2d.values.empty() && !lat2d.values.empty()) {
            return GeoGrid{massNy, massNx, false, lon2d.values, lat2d.values};
        }
    }

    const auto lonName = dataset.firstVariable({"XLONG", "XLONG_M", "XLONG_U", "XLONG_V"});
    const auto latName = dataset.firstVariable({"XLAT", "XLAT_M", "XLAT_U", "XLAT_V"});
    if (!lonName || !latName) {
        throw std::runtime_error("Cannot find lon/lat or XLONG/XLAT in dataset");
    }
    const Surface2D lon = readGeoCoordinate2D(dataset, dataset.variable(*lonName), massNy, massNx);
    const Surface2D lat = readGeoCoordinate2D(dataset, dataset.variable(*latName), massNy, massNx);
    if (lon.values.empty() || lat.values.empty()) {
        throw std::runtime_error("WRF longitude/latitude arrays do not match the mass grid");
    }
    info(output, "Using WRF lon/lat coords: " + *lonName + "/" + *latName);
    return GeoGrid{massNy, massNx, false, lon.values, lat.values};
}

GeoGrid cropGeoGrid(const GeoGrid& source, const Crop& crop) {
    GeoGrid result;
    result.nx = crop.nx();
    result.ny = crop.ny();
    result.rectilinear = source.rectilinear;
    if (source.rectilinear) {
        result.longitude.assign(source.longitude.begin() + static_cast<std::ptrdiff_t>(crop.x0),
                                source.longitude.begin() + static_cast<std::ptrdiff_t>(crop.x1 + 1U));
        result.latitude.assign(source.latitude.begin() + static_cast<std::ptrdiff_t>(crop.y0),
                               source.latitude.begin() + static_cast<std::ptrdiff_t>(crop.y1 + 1U));
    } else {
        result.longitude.resize(result.nx * result.ny);
        result.latitude.resize(result.nx * result.ny);
        for (std::size_t y = 0; y < result.ny; ++y) {
            for (std::size_t x = 0; x < result.nx; ++x) {
                result.longitude[y * result.nx + x] = source.lon(crop.y0 + y, crop.x0 + x);
                result.latitude[y * result.nx + x] = source.lat(crop.y0 + y, crop.x0 + x);
            }
        }
    }
    return result;
}

struct Bounds {
    double lonMin = 0.0;
    double lonMax = 0.0;
    double latMin = 0.0;
    double latMax = 0.0;
};

Bounds geoBounds(const GeoGrid& grid) {
    Bounds bounds{std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()};
    for (std::size_t y = 0; y < grid.ny; ++y) {
        for (std::size_t x = 0; x < grid.nx; ++x) {
            const double lon = grid.lon(y, x);
            const double lat = grid.lat(y, x);
            if (std::isfinite(lon)) {
                bounds.lonMin = std::min(bounds.lonMin, lon);
                bounds.lonMax = std::max(bounds.lonMax, lon);
            }
            if (std::isfinite(lat)) {
                bounds.latMin = std::min(bounds.latMin, lat);
                bounds.latMax = std::max(bounds.latMax, lat);
            }
        }
    }
    if (!std::isfinite(bounds.lonMin) || !std::isfinite(bounds.latMin)) {
        throw std::runtime_error("Longitude/latitude coordinates contain no finite values");
    }
    return bounds;
}

std::optional<Bounds> targetBounds(const DeckDocument& deck) {
    const auto lon = deck.pair(QStringLiteral("cut_lon_manual"));
    const auto lat = deck.pair(QStringLiteral("cut_lat_manual"));
    if (!lon || !lat) return std::nullopt;
    return Bounds{lon->first, lon->second, lat->first, lat->second};
}

bool contains(const Bounds& target, const Bounds& input) {
    return input.lonMin <= target.lonMin && input.lonMax >= target.lonMax
        && input.latMin <= target.latMin && input.latMax >= target.latMax;
}

double maximumMissPercent(const Bounds& target, const Bounds& input) {
    const double lonSpan = std::abs(target.lonMax - target.lonMin);
    const double latSpan = std::abs(target.latMax - target.latMin);
    if (lonSpan <= 0.0 || latSpan <= 0.0) return 100.0;
    const double missLon = std::max({0.0, input.lonMin - target.lonMin,
                                    target.lonMax - input.lonMax});
    const double missLat = std::max({0.0, input.latMin - target.latMin,
                                    target.latMax - input.latMax});
    return std::max(missLon / lonSpan, missLat / latSpan) * 100.0;
}

void confirmCoverage(std::string_view kind,
                     const Bounds& target,
                     const Bounds& input,
                     std::ostream& output) {
    if (contains(target, input)) return;
    const double miss = maximumMissPercent(target, input);
    if (miss < 0.1) {
        warn(output, std::string(kind) + " bounds are slightly smaller than target (max miss " +
             fixed(miss, 4) + "% < 0.1%). Continue without interruption.");
        return;
    }
    warn(output, std::string(kind) + " bounds do not fully cover the target area.");
    warn(output, "Target lon/lat bounds: lon[" + fixed(target.lonMin, 6) + ", " +
         fixed(target.lonMax, 6) + "] lat[" + fixed(target.latMin, 6) + ", " +
         fixed(target.latMax, 6) + "]");
    warn(output, "Input  lon/lat bounds: lon[" + fixed(input.lonMin, 6) + ", " +
         fixed(input.lonMax, 6) + "] lat[" + fixed(input.latMin, 6) + ", " +
         fixed(input.latMax, 6) + "]");
    // Native workflow children commonly receive no interactive stdin.  Match the
    // legacy timeout result without holding an unattended workflow for five seconds.
    warn(output, "No input received (timeout 5s). Continuing by default.");
}

struct XY {
    double x = 0.0;
    double y = 0.0;
};

int utmEpsgForBounds(const std::optional<Bounds>& bounds) {
    if (!bounds) return 32651;
    const double lon = 0.5 * (bounds->lonMin + bounds->lonMax);
    const double lat = 0.5 * (bounds->latMin + bounds->latMax);
    int zone = static_cast<int>(std::floor((lon + 180.0) / 6.0)) + 1;
    zone = std::clamp(zone, 1, 60);
    return (lat >= 0.0 ? 32600 : 32700) + zone;
}

class CoordinateProjector final {
public:
    explicit CoordinateProjector(int epsg) : epsg_(epsg) {
        projContext_ = proj_context_create();
        const std::string targetName = "EPSG:" + std::to_string(epsg_);
        PJ* raw = projContext_ != nullptr
            ? proj_create_crs_to_crs(projContext_, "EPSG:4326", targetName.c_str(), nullptr)
            : nullptr;
        if (raw != nullptr) {
            projOperation_ = proj_normalize_for_visualization(projContext_, raw);
            proj_destroy(raw);
        }
#if LUW_HAS_GDAL
        source_.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        target_.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        if (source_.importFromEPSG(4326) == OGRERR_NONE
            && target_.importFromEPSG(epsg_) == OGRERR_NONE) {
            transform_.reset(OGRCreateCoordinateTransformation(&source_, &target_));
        }
#endif
    }

    CoordinateProjector(const CoordinateProjector&) = delete;
    CoordinateProjector& operator=(const CoordinateProjector&) = delete;

    ~CoordinateProjector() {
        if (projOperation_ != nullptr) proj_destroy(projOperation_);
        if (projContext_ != nullptr) proj_context_destroy(projContext_);
    }

    [[nodiscard]] int epsg() const noexcept { return epsg_; }

    [[nodiscard]] XY project(double longitude, double latitude) const {
        if (projOperation_ != nullptr) {
            const PJ_COORD input = proj_coord(longitude, latitude, 0.0, 0.0);
            const PJ_COORD projected = proj_trans(projOperation_, PJ_FWD, input);
            if (std::isfinite(projected.xy.x) && std::isfinite(projected.xy.y)) {
                return {projected.xy.x, projected.xy.y};
            }
        }
        return projectFallback(longitude, latitude);
    }

    [[nodiscard]] std::vector<XY> projectMany(
        const std::vector<double>& longitude,
        const std::vector<double>& latitude) const {
        if (longitude.size() != latitude.size()) {
            throw std::runtime_error(
                "Projected coordinate batch has mismatched longitude/latitude lengths");
        }
        std::vector<XY> result(longitude.size());
        if (longitude.empty()) return result;
        if (projOperation_ != nullptr) {
            std::vector<PJ_COORD> coordinates(longitude.size());
            for (std::size_t index = 0U; index < longitude.size(); ++index) {
                coordinates[index] = proj_coord(
                    longitude[index], latitude[index], 0.0, 0.0);
            }
            proj_errno_reset(projOperation_);
            (void)proj_trans_array(projOperation_, PJ_FWD,
                                   coordinates.size(), coordinates.data());
            for (std::size_t index = 0U; index < coordinates.size(); ++index) {
                const PJ_COORD projected = coordinates[index];
                if (std::isfinite(projected.xy.x)
                    && std::isfinite(projected.xy.y)) {
                    result[index] = {projected.xy.x, projected.xy.y};
                } else {
                    result[index] = projectFallback(
                        longitude[index], latitude[index]);
                }
            }
            return result;
        }
        for (std::size_t index = 0U; index < longitude.size(); ++index) {
            result[index] = projectFallback(longitude[index], latitude[index]);
        }
        return result;
    }

private:
    [[nodiscard]] XY projectFallback(double longitude, double latitude) const {
#if LUW_HAS_GDAL
        if (transform_) {
            double x = longitude;
            double y = latitude;
            if (transform_->Transform(1, &x, &y) != 0) return {x, y};
        }
#endif
        return projectUtmWgs84(longitude, latitude, epsg_);
    }

    static XY projectUtmWgs84(double longitude, double latitude, int epsg) {
        const bool north = epsg >= 32601 && epsg <= 32660;
        const bool south = epsg >= 32701 && epsg <= 32760;
        const int zone = north ? epsg - 32600 : (south ? epsg - 32700 : 51);
        const double a = 6378137.0;
        const double inverseFlattening = 298.257223563;
        const double f = 1.0 / inverseFlattening;
        const double eccentricitySquared = f * (2.0 - f);
        const double secondEccentricitySquared = eccentricitySquared / (1.0 - eccentricitySquared);
        const double k0 = 0.9996;
        const double phi = latitude * kPi / 180.0;
        const double lambda = longitude * kPi / 180.0;
        const double centralLongitude = ((zone - 1) * 6 - 180 + 3) * kPi / 180.0;
        const double sinPhi = std::sin(phi);
        const double cosPhi = std::cos(phi);
        const double tanPhi = std::tan(phi);
        const double n = a / std::sqrt(1.0 - eccentricitySquared * sinPhi * sinPhi);
        const double t = tanPhi * tanPhi;
        const double c = secondEccentricitySquared * cosPhi * cosPhi;
        const double aa = cosPhi * (lambda - centralLongitude);
        const double e2 = eccentricitySquared;
        const double meridian = a * ((1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0
                                       - 5.0 * e2 * e2 * e2 / 256.0) * phi
            - (3.0 * e2 / 8.0 + 3.0 * e2 * e2 / 32.0
               + 45.0 * e2 * e2 * e2 / 1024.0) * std::sin(2.0 * phi)
            + (15.0 * e2 * e2 / 256.0 + 45.0 * e2 * e2 * e2 / 1024.0)
                * std::sin(4.0 * phi)
            - (35.0 * e2 * e2 * e2 / 3072.0) * std::sin(6.0 * phi));
        const double x = k0 * n * (aa + (1.0 - t + c) * std::pow(aa, 3.0) / 6.0
            + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * secondEccentricitySquared)
                * std::pow(aa, 5.0) / 120.0) + 500000.0;
        double y = k0 * (meridian + n * tanPhi * (aa * aa / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c * c) * std::pow(aa, 4.0) / 24.0
            + (61.0 - 58.0 * t + t * t + 600.0 * c
               - 330.0 * secondEccentricitySquared) * std::pow(aa, 6.0) / 720.0));
        if (!north || latitude < 0.0) y += 10000000.0;
        return {x, y};
    }

    int epsg_ = 32651;
    PJ_CONTEXT* projContext_ = nullptr;
    PJ* projOperation_ = nullptr;
#if LUW_HAS_GDAL
    struct TransformDeleter {
        void operator()(OGRCoordinateTransformation* value) const noexcept {
            if (value != nullptr) OCTDestroyCoordinateTransformation(value);
        }
    };
    OGRSpatialReference source_;
    OGRSpatialReference target_;
    std::unique_ptr<OGRCoordinateTransformation, TransformDeleter> transform_;
#endif
};

XY rotatePoint(const XY& point, double degrees, const XY& pivot) {
    const double angle = degrees * kPi / 180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return {cosine * (point.x - pivot.x) - sine * (point.y - pivot.y) + pivot.x,
            sine * (point.x - pivot.x) + cosine * (point.y - pivot.y) + pivot.y};
}

class RotationOperator final {
public:
    RotationOperator(double degrees, XY pivot)
        : pivot_(pivot) {
        const double angle = degrees * kPi / 180.0;
        cosine_ = std::cos(angle);
        sine_ = std::sin(angle);
    }

    [[nodiscard]] XY apply(const XY& point) const noexcept {
        return {cosine_ * (point.x - pivot_.x)
                    - sine_ * (point.y - pivot_.y) + pivot_.x,
                sine_ * (point.x - pivot_.x)
                    + cosine_ * (point.y - pivot_.y) + pivot_.y};
    }

private:
    XY pivot_;
    double cosine_ = 1.0;
    double sine_ = 0.0;
};

struct RotationFrame {
    double degrees = 0.0;
    XY pivot;
    double xOrigin = 0.0;
    double yOrigin = 0.0;
    double xRange = 0.0;
    double yRange = 0.0;
    bool fromTargetBounds = false;
};

std::array<XY, 4> projectCorners(const Bounds& bounds,
                                 const CoordinateProjector& projector) {
    const std::vector<XY> projected = projector.projectMany(
        {bounds.lonMin, bounds.lonMax, bounds.lonMax, bounds.lonMin},
        {bounds.latMin, bounds.latMin, bounds.latMax, bounds.latMax});
    return {projected[0], projected[1], projected[2], projected[3]};
}

template <typename Key, typename Plan>
class BoundedExactPlanCache final {
public:
    struct Lookup {
        std::shared_ptr<const Plan> plan;
        bool hit = false;
        bool retained = false;
        std::size_t evicted = 0U;
    };

    BoundedExactPlanCache(std::size_t maximumEntries,
                          std::size_t maximumBytes)
        : maximumEntries_(maximumEntries), maximumBytes_(maximumBytes) {}

    template <typename Builder>
    [[nodiscard]] Lookup acquire(const Key& key, Builder&& builder) {
        {
            std::scoped_lock lock(mutex_);
            if (Entry* entry = find(key)) {
                entry->lastUse = ++clock_;
                return {entry->plan, true, true, 0U};
            }
        }

        std::shared_ptr<const Plan> built =
            std::make_shared<const Plan>(std::forward<Builder>(builder)());
        const std::size_t keyBytes = key.memoryBytes();
        const std::size_t planBytes = built->memoryBytes();
        const std::size_t bytes = planBytes > std::numeric_limits<std::size_t>::max() - keyBytes
            ? std::numeric_limits<std::size_t>::max() : keyBytes + planBytes;

        std::scoped_lock lock(mutex_);
        if (Entry* entry = find(key)) {
            entry->lastUse = ++clock_;
            return {entry->plan, true, true, 0U};
        }
        if (maximumEntries_ == 0U || bytes > maximumBytes_) {
            return {std::move(built), false, false, 0U};
        }
        std::size_t evicted = 0U;
        while (!entries_.empty()
               && (entries_.size() >= maximumEntries_
                   || bytes_ > maximumBytes_ - bytes)) {
            const auto oldest = std::min_element(
                entries_.begin(), entries_.end(),
                [](const Entry& first, const Entry& second) {
                    return first.lastUse < second.lastUse;
                });
            bytes_ -= oldest->bytes;
            entries_.erase(oldest);
            ++evicted;
        }
        entries_.push_back(
            Entry{key, built, bytes, ++clock_});
        bytes_ += bytes;
        return {std::move(built), false, true, evicted};
    }

private:
    struct Entry {
        Key key;
        std::shared_ptr<const Plan> plan;
        std::size_t bytes = 0U;
        std::uint64_t lastUse = 0U;
    };

    [[nodiscard]] Entry* find(const Key& key) {
        const auto found = std::find_if(
            entries_.begin(), entries_.end(),
            [&](const Entry& entry) { return entry.key == key; });
        return found == entries_.end() ? nullptr : &*found;
    }

    std::mutex mutex_;
    std::vector<Entry> entries_;
    std::size_t bytes_ = 0U;
    std::uint64_t clock_ = 0U;
    std::size_t maximumEntries_ = 0U;
    std::size_t maximumBytes_ = 0U;
};

struct ProjectedGridKey {
    std::string scope;
    int epsg = 0;
    std::size_t nx = 0U;
    std::size_t ny = 0U;
    bool rectilinear = false;
    std::vector<std::uint64_t> longitude;
    std::vector<std::uint64_t> latitude;

    [[nodiscard]] bool operator==(const ProjectedGridKey&) const = default;

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return scope.capacity() + longitude.capacity() * sizeof(std::uint64_t)
            + latitude.capacity() * sizeof(std::uint64_t);
    }
};

struct ProjectedGridPlan {
    std::vector<XY> points;

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return points.capacity() * sizeof(XY);
    }
};

using ProjectedGridCache = BoundedExactPlanCache<ProjectedGridKey, ProjectedGridPlan>;

ProjectedGridCache& projectedGridCache() {
    static ProjectedGridCache cache(3U, 256U * 1024U * 1024U);
    return cache;
}

ProjectedGridCache::Lookup acquireProjectedGrid(
    const std::string& scope,
    const GeoGrid& grid,
    const CoordinateProjector& projector) {
    ProjectedGridKey key;
    key.scope = scope;
    key.epsg = projector.epsg();
    key.nx = grid.nx;
    key.ny = grid.ny;
    key.rectilinear = grid.rectilinear;
    key.longitude = exactDoubleBits(grid.longitude);
    key.latitude = exactDoubleBits(grid.latitude);
    return projectedGridCache().acquire(key, [&]() {
        const std::size_t pointCount = checkedProduct({grid.nx, grid.ny});
        std::vector<double> longitude(pointCount);
        std::vector<double> latitude(pointCount);
#pragma omp parallel for if(pointCount > 4096)
        for (std::int64_t pointSigned = 0;
             pointSigned < static_cast<std::int64_t>(pointCount); ++pointSigned) {
            const std::size_t point = static_cast<std::size_t>(pointSigned);
            const std::size_t y = point / grid.nx;
            const std::size_t x = point % grid.nx;
            longitude[point] = grid.lon(y, x);
            latitude[point] = grid.lat(y, x);
        }
        return ProjectedGridPlan{projector.projectMany(longitude, latitude)};
    });
}

void reportPlanCache(std::ostream& output,
                     std::string_view name,
                     bool hit,
                     bool retained,
                     std::size_t evicted) {
    info(output, std::string(name) + " mapping plan cache: "
         + (hit ? "hit" : "miss")
         + (retained ? "" : " (entry exceeds cache budget; not retained)")
         + (evicted == 0U ? "" : "; evicted " + std::to_string(evicted)
                                      + " least-recently-used entry(s)"));
}

std::pair<double, XY> rotationFromCorners(const std::array<XY, 4>& points) {
    XY pivot;
    for (const XY& point : points) {
        pivot.x += point.x * 0.25;
        pivot.y += point.y * 0.25;
    }
    const double angle = -std::atan2(points[1].y - points[0].y,
                                     points[1].x - points[0].x) * 180.0 / kPi;
    return {angle, pivot};
}

Crop cropByDeck(const GeoGrid& grid,
                const DeckDocument& deck,
                const CoordinateProjector& projector,
                const std::vector<XY>& projectedGrid,
                std::ostream& output) {
    if (projectedGrid.size() != grid.nx * grid.ny) {
        throw std::runtime_error(
            "Projected wind grid dimensions do not match the geographic grid");
    }
    Crop full{0U, grid.nx - 1U, 0U, grid.ny - 1U};
    std::optional<std::pair<double, double>> xBounds = deck.pair(QStringLiteral("cut_utm_x"));
    std::optional<std::pair<double, double>> yBounds = deck.pair(QStringLiteral("cut_utm_y"));
    if (xBounds && yBounds) {
        info(output, "Using direct UTM bounds from config: X=[" + pythonFloat(xBounds->first) +
             ", " + pythonFloat(xBounds->second) + "], Y=[" + pythonFloat(yBounds->first) +
             ", " + pythonFloat(yBounds->second) + "]");
    } else {
        const auto bounds = targetBounds(deck);
        if (!bounds) {
            info(output, "cut_lon_manual or cut_lat_manual not provided, skip cropping");
            return full;
        }
        const std::vector<XY> projectedBounds = projector.projectMany(
            {bounds->lonMin, bounds->lonMax},
            {bounds->latMin, bounds->latMax});
        const XY lower = projectedBounds[0];
        const XY upper = projectedBounds[1];
        xBounds = std::minmax(lower.x, upper.x);
        yBounds = std::minmax(lower.y, upper.y);
        info(output, "Converted lon/lat bounds [" + fixed(bounds->lonMin, 6) + ", " +
             fixed(bounds->lonMax, 6) + "] x [" + fixed(bounds->latMin, 6) + ", " +
             fixed(bounds->latMax, 6) + "]");
        info(output, "  to UTM bounds X=[" + fixed(xBounds->first, 2) + ", " +
             fixed(xBounds->second, 2) + "], Y=[" + fixed(yBounds->first, 2) + ", " +
             fixed(yBounds->second, 2) + "]");
    }

    std::size_t x0 = grid.nx;
    std::size_t x1 = 0U;
    std::size_t y0 = grid.ny;
    std::size_t y1 = 0U;
    bool any = false;
    for (std::size_t y = 0; y < grid.ny; ++y) {
        for (std::size_t x = 0; x < grid.nx; ++x) {
            const XY projected = projectedGrid[y * grid.nx + x];
            if (projected.x >= xBounds->first && projected.x <= xBounds->second
                && projected.y >= yBounds->first && projected.y <= yBounds->second) {
                any = true;
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y);
            }
        }
    }
    if (!any) {
        warn(output, "No wind data points within UTM bounds, returning original dataset");
        return full;
    }
    info(output, "Cropped wind data by UTM bounds: lat [" + std::to_string(y0) + ", " +
         std::to_string(y1) + "] lon [" + std::to_string(x0) + ", " +
         std::to_string(x1) + "]");
    return Crop{x0, x1, y0, y1};
}

std::vector<XY> cropProjectedGrid(const std::vector<XY>& source,
                                  std::size_t sourceNx,
                                  const Crop& crop) {
    std::vector<XY> result(crop.nx() * crop.ny());
    for (std::size_t y = 0U; y < crop.ny(); ++y) {
        const auto begin = source.begin()
            + static_cast<std::ptrdiff_t>((crop.y0 + y) * sourceNx + crop.x0);
        std::copy_n(begin, static_cast<std::ptrdiff_t>(crop.nx()),
                    result.begin() + static_cast<std::ptrdiff_t>(y * crop.nx()));
    }
    return result;
}

std::vector<double> linearCoordinates(double first, double last, std::size_t count) {
    std::vector<double> result(count, first);
    if (count <= 1U) return result;
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = first + (last - first) * static_cast<double>(index)
            / static_cast<double>(count - 1U);
    }
    return result;
}

struct Bracket {
    std::size_t lower = 0U;
    std::size_t upper = 0U;
    double fraction = 0.0;
};

Bracket bracketCoordinate(const std::vector<double>& coordinates, double query) {
    if (coordinates.empty()) return {};
    const bool increasing = coordinates.back() >= coordinates.front();
    if ((increasing && query <= coordinates.front())
        || (!increasing && query >= coordinates.front())) return {0U, 0U, 0.0};
    if ((increasing && query >= coordinates.back())
        || (!increasing && query <= coordinates.back())) {
        return {coordinates.size() - 1U, coordinates.size() - 1U, 0.0};
    }
    std::size_t upper = 1U;
    if (increasing) {
        upper = static_cast<std::size_t>(std::lower_bound(coordinates.begin(), coordinates.end(), query)
                                         - coordinates.begin());
    } else {
        upper = static_cast<std::size_t>(std::lower_bound(coordinates.begin(), coordinates.end(), query,
                                                          std::greater<double>())
                                         - coordinates.begin());
    }
    const std::size_t lower = upper - 1U;
    const double denominator = coordinates[upper] - coordinates[lower];
    const double fraction = denominator == 0.0 ? 0.0 : (query - coordinates[lower]) / denominator;
    return {lower, upper, fraction};
}

double lerpNan(double first, double second, double fraction) {
    if (fraction <= 0.0) return first;
    if (fraction >= 1.0) return second;
    return first * (1.0 - fraction) + second * fraction;
}

Field3D resampleHorizontalLinear(const Field3D& source,
                                 const std::vector<double>& sourceX,
                                 const std::vector<double>& sourceY,
                                 const std::vector<double>& targetX,
                                 const std::vector<double>& targetY) {
    Field3D result(source.nz, targetY.size(), targetX.size());
    std::vector<Bracket> xBrackets(targetX.size());
    std::vector<Bracket> yBrackets(targetY.size());
    std::transform(targetX.begin(), targetX.end(), xBrackets.begin(),
                   [&](double value) { return bracketCoordinate(sourceX, value); });
    std::transform(targetY.begin(), targetY.end(), yBrackets.begin(),
                   [&](double value) { return bracketCoordinate(sourceY, value); });

#pragma omp parallel for if(source.nz * targetY.size() > 16)
    for (std::int64_t zSigned = 0; zSigned < static_cast<std::int64_t>(source.nz); ++zSigned) {
        const std::size_t z = static_cast<std::size_t>(zSigned);
        for (std::size_t y = 0; y < targetY.size(); ++y) {
            const Bracket by = yBrackets[y];
            for (std::size_t x = 0; x < targetX.size(); ++x) {
                const Bracket bx = xBrackets[x];
                const double a = lerpNan(source.at(z, by.lower, bx.lower),
                                         source.at(z, by.lower, bx.upper), bx.fraction);
                const double b = lerpNan(source.at(z, by.upper, bx.lower),
                                         source.at(z, by.upper, bx.upper), bx.fraction);
                result.at(z, y, x) = static_cast<float>(lerpNan(a, b, by.fraction));
            }
        }
    }
    return result;
}

std::vector<double> pchipSlopes(const std::vector<double>& x,
                                const std::vector<double>& y) {
    const std::size_t count = x.size();
    std::vector<double> slopes(count, 0.0);
    if (count <= 1U) return slopes;
    if (count == 2U) {
        const double value = (y[1] - y[0]) / (x[1] - x[0]);
        slopes[0] = value;
        slopes[1] = value;
        return slopes;
    }
    std::vector<double> h(count - 1U);
    std::vector<double> delta(count - 1U);
    for (std::size_t index = 0; index + 1U < count; ++index) {
        h[index] = x[index + 1U] - x[index];
        delta[index] = (y[index + 1U] - y[index]) / h[index];
    }
    for (std::size_t index = 1U; index + 1U < count; ++index) {
        if (delta[index - 1U] == 0.0 || delta[index] == 0.0
            || std::signbit(delta[index - 1U]) != std::signbit(delta[index])) {
            slopes[index] = 0.0;
        } else {
            const double w1 = 2.0 * h[index] + h[index - 1U];
            const double w2 = h[index] + 2.0 * h[index - 1U];
            slopes[index] = (w1 + w2) / (w1 / delta[index - 1U] + w2 / delta[index]);
        }
    }
    auto endpoint = [](double h0, double h1, double d0, double d1) {
        double slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
        if (std::signbit(slope) != std::signbit(d0)) slope = 0.0;
        else if (std::signbit(d0) != std::signbit(d1) && std::abs(slope) > 3.0 * std::abs(d0)) {
            slope = 3.0 * d0;
        }
        return slope;
    };
    slopes.front() = endpoint(h[0], h[1], delta[0], delta[1]);
    slopes.back() = endpoint(h[count - 2U], h[count - 3U],
                             delta[count - 2U], delta[count - 3U]);
    return slopes;
}

double pchipEvaluate(const std::vector<double>& x,
                     const std::vector<double>& y,
                     const std::vector<double>& slopes,
                     double query) {
    if (query <= x.front()) return y.front();
    if (query >= x.back()) return y.back();
    const std::size_t upper = static_cast<std::size_t>(
        std::upper_bound(x.begin(), x.end(), query) - x.begin());
    const std::size_t lower = upper - 1U;
    const double h = x[upper] - x[lower];
    const double t = (query - x[lower]) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * y[lower]
        + (t3 - 2.0 * t2 + t) * h * slopes[lower]
        + (-2.0 * t3 + 3.0 * t2) * y[upper]
        + (t3 - t2) * h * slopes[upper];
}

Field3D resampleVerticalPchip(const Field3D& source,
                              const std::vector<double>& sourceZ,
                              const std::vector<double>& targetZ) {
    if (source.nz != sourceZ.size()) {
        throw std::runtime_error("Vertical coordinate length does not match wind field");
    }
    Field3D result(targetZ.size(), source.ny, source.nx);
#pragma omp parallel for if(source.nx * source.ny > 64)
    for (std::int64_t pointSigned = 0;
         pointSigned < static_cast<std::int64_t>(source.nx * source.ny);
         ++pointSigned) {
        const std::size_t point = static_cast<std::size_t>(pointSigned);
        const std::size_t yIndex = point / source.nx;
        const std::size_t xIndex = point % source.nx;
        std::vector<double> column(source.nz);
        for (std::size_t z = 0; z < source.nz; ++z) column[z] = source.at(z, yIndex, xIndex);
        const std::vector<double> slopes = pchipSlopes(sourceZ, column);
        for (std::size_t z = 0; z < targetZ.size(); ++z) {
            result.at(z, yIndex, xIndex) = static_cast<float>(
                pchipEvaluate(sourceZ, column, slopes, targetZ[z]));
        }
    }
    return result;
}

GeoGrid resampleGeoGridLinear(const GeoGrid& source,
                              std::size_t targetNy,
                              std::size_t targetNx) {
    if (targetNx == source.nx && targetNy == source.ny) return source;
    GeoGrid result;
    result.nx = targetNx;
    result.ny = targetNy;
    const std::vector<double> sx = linearCoordinates(0.0, static_cast<double>(source.nx - 1U), source.nx);
    const std::vector<double> sy = linearCoordinates(0.0, static_cast<double>(source.ny - 1U), source.ny);
    const std::vector<double> tx = linearCoordinates(0.0, static_cast<double>(source.nx - 1U), targetNx);
    const std::vector<double> ty = linearCoordinates(0.0, static_cast<double>(source.ny - 1U), targetNy);
    if (source.rectilinear) {
        result.rectilinear = true;
        result.longitude.resize(targetNx);
        result.latitude.resize(targetNy);
        for (std::size_t x = 0; x < targetNx; ++x) {
            const Bracket bracket = bracketCoordinate(sx, tx[x]);
            result.longitude[x] = lerpNan(source.longitude[bracket.lower],
                                          source.longitude[bracket.upper], bracket.fraction);
        }
        for (std::size_t y = 0; y < targetNy; ++y) {
            const Bracket bracket = bracketCoordinate(sy, ty[y]);
            result.latitude[y] = lerpNan(source.latitude[bracket.lower],
                                         source.latitude[bracket.upper], bracket.fraction);
        }
        return result;
    }
    result.rectilinear = false;
    result.longitude.resize(targetNx * targetNy);
    result.latitude.resize(targetNx * targetNy);
    for (std::size_t y = 0; y < targetNy; ++y) {
        const Bracket by = bracketCoordinate(sy, ty[y]);
        for (std::size_t x = 0; x < targetNx; ++x) {
            const Bracket bx = bracketCoordinate(sx, tx[x]);
            const auto blend = [&](const std::vector<double>& values) {
                const double a = lerpNan(values[by.lower * source.nx + bx.lower],
                                         values[by.lower * source.nx + bx.upper], bx.fraction);
                const double b = lerpNan(values[by.upper * source.nx + bx.lower],
                                         values[by.upper * source.nx + bx.upper], bx.fraction);
                return lerpNan(a, b, by.fraction);
            };
            result.longitude[y * targetNx + x] = blend(source.longitude);
            result.latitude[y * targetNx + x] = blend(source.latitude);
        }
    }
    return result;
}

void reverseVertical(Field3D& field) {
    for (std::size_t z = 0; z < field.nz / 2U; ++z) {
        const std::size_t other = field.nz - 1U - z;
        for (std::size_t y = 0; y < field.ny; ++y) {
            for (std::size_t x = 0; x < field.nx; ++x) {
                std::swap(field.at(z, y, x), field.at(other, y, x));
            }
        }
    }
}

void forwardFillWholeLayers(Field3D& primary,
                            const std::vector<Field3D*>& companions) {
    std::optional<std::size_t> firstValid;
    for (std::size_t z = 0; z < primary.nz; ++z) {
        bool allNan = true;
        for (std::size_t point = z * primary.nx * primary.ny;
             point < (z + 1U) * primary.nx * primary.ny; ++point) {
            if (!std::isnan(primary.values[point])) { allNan = false; break; }
        }
        if (!allNan) { firstValid = z; break; }
    }
    if (!firstValid) throw std::runtime_error("All vertical layers are NaN, invalid data");
    std::vector<Field3D*> arrays{&primary};
    arrays.insert(arrays.end(), companions.begin(), companions.end());
    auto copyLayer = [&](std::size_t destination, std::size_t source) {
        for (Field3D* field : arrays) {
            const std::size_t layerSize = field->nx * field->ny;
            std::copy_n(field->values.begin() + static_cast<std::ptrdiff_t>(source * layerSize),
                        static_cast<std::ptrdiff_t>(layerSize),
                        field->values.begin() + static_cast<std::ptrdiff_t>(destination * layerSize));
        }
    };
    for (std::size_t z = 0; z < *firstValid; ++z) copyLayer(z, *firstValid);
    for (std::size_t z = *firstValid + 1U; z < primary.nz; ++z) {
        bool allNan = true;
        const std::size_t begin = z * primary.nx * primary.ny;
        for (std::size_t point = begin; point < begin + primary.nx * primary.ny; ++point) {
            if (!std::isnan(primary.values[point])) { allNan = false; break; }
        }
        if (allNan) copyLayer(z, z - 1U);
    }
}

bool containsNan(const Field3D& field) {
    return std::any_of(field.values.begin(), field.values.end(),
                       [](float value) { return std::isnan(value); });
}

void fillHorizontalNearest(Field3D& field,
                           const std::vector<double>& x,
                           const std::vector<double>& y) {
    const std::size_t layerSize = field.nx * field.ny;
    if (x.size() != layerSize || y.size() != layerSize) return;
    for (std::size_t z = 0; z < field.nz; ++z) {
        vtkNew<vtkPoints> points;
        std::vector<std::size_t> sourceIndex;
        std::vector<std::size_t> badIndex;
        for (std::size_t point = 0; point < layerSize; ++point) {
            const float value = field.values[z * layerSize + point];
            if (std::isfinite(value) && std::isfinite(x[point]) && std::isfinite(y[point])) {
                points->InsertNextPoint(x[point], y[point], 0.0);
                sourceIndex.push_back(point);
            } else if (!std::isfinite(value)) {
                badIndex.push_back(point);
            }
        }
        if (badIndex.empty() || sourceIndex.empty()) continue;
        vtkNew<vtkPolyData> data;
        data->SetPoints(points);
        vtkNew<vtkStaticPointLocator> locator;
        locator->SetDataSet(data);
        locator->BuildLocator();
        for (const std::size_t point : badIndex) {
            const double query[3]{x[point], y[point], 0.0};
            const vtkIdType nearest = locator->FindClosestPoint(query);
            if (nearest >= 0) {
                field.values[z * layerSize + point] =
                    field.values[z * layerSize + sourceIndex[static_cast<std::size_t>(nearest)]];
            }
        }
    }
}

struct HorizontalWeight {
    std::array<std::size_t, 3> indices{};
    std::array<double, 3> weights{};
    std::size_t count = 0U;
};

class HorizontalPlan final {
public:
    HorizontalPlan(const std::vector<double>& sourceX,
                   const std::vector<double>& sourceY,
                   const std::vector<double>& targetX,
                   const std::vector<double>& targetY) {
        if (sourceX.size() != sourceY.size()) {
            throw std::runtime_error("Projected source grid has mismatched X/Y lengths");
        }
        vtkNew<vtkPoints> points;
        validSource_.reserve(sourceX.size());
        for (std::size_t index = 0; index < sourceX.size(); ++index) {
            if (!std::isfinite(sourceX[index]) || !std::isfinite(sourceY[index])) continue;
            points->InsertNextPoint(sourceX[index], sourceY[index], 0.0);
            validSource_.push_back(index);
        }
        if (validSource_.size() < 3U) {
            throw std::runtime_error("Not enough valid points for triangulation after dropping NaNs");
        }
        vtkNew<vtkPolyData> input;
        input->SetPoints(points);
        vtkNew<vtkDelaunay2D> delaunay;
        delaunay->SetInputData(input);
        delaunay->BoundingTriangulationOff();
        delaunay->Update();
        vtkPolyData* triangles = delaunay->GetOutput();
        if (triangles == nullptr || triangles->GetNumberOfCells() == 0) {
            throw std::runtime_error("Delaunay triangulation failed for projected wind grid");
        }

        vtkNew<vtkStaticCellLocator> cellLocator;
        cellLocator->SetDataSet(triangles);
        cellLocator->BuildLocator();
        vtkNew<vtkStaticPointLocator> pointLocator;
        pointLocator->SetDataSet(triangles);
        pointLocator->BuildLocator();

        weights_.resize(targetX.size() * targetY.size());
        for (std::size_t y = 0; y < targetY.size(); ++y) {
            for (std::size_t x = 0; x < targetX.size(); ++x) {
                double query[3]{targetX[x], targetY[y], 0.0};
                HorizontalWeight weight;
                const vtkIdType cellId = cellLocator->FindCell(query);
                if (cellId >= 0) {
                    vtkCell* cell = triangles->GetCell(cellId);
                    if (cell != nullptr && cell->GetNumberOfPoints() == 3) {
                        std::array<XY, 3> p{};
                        for (vtkIdType corner = 0; corner < 3; ++corner) {
                            const vtkIdType id = cell->GetPointId(corner);
                            double point[3]{};
                            triangles->GetPoint(id, point);
                            p[static_cast<std::size_t>(corner)] = {point[0], point[1]};
                            weight.indices[static_cast<std::size_t>(corner)] =
                                validSource_.at(static_cast<std::size_t>(id));
                        }
                        const double denominator = (p[1].y - p[2].y) * (p[0].x - p[2].x)
                            + (p[2].x - p[1].x) * (p[0].y - p[2].y);
                        if (std::abs(denominator) > 1e-20) {
                            const double w0 = ((p[1].y - p[2].y) * (query[0] - p[2].x)
                                + (p[2].x - p[1].x) * (query[1] - p[2].y)) / denominator;
                            const double w1 = ((p[2].y - p[0].y) * (query[0] - p[2].x)
                                + (p[0].x - p[2].x) * (query[1] - p[2].y)) / denominator;
                            weight.weights = {w0, w1, 1.0 - w0 - w1};
                            weight.count = 3U;
                        }
                    }
                }
                if (weight.count == 0U) {
                    const vtkIdType nearest = pointLocator->FindClosestPoint(query);
                    if (nearest < 0) throw std::runtime_error("Nearest-neighbor lookup failed");
                    weight.indices[0] = validSource_.at(static_cast<std::size_t>(nearest));
                    weight.weights[0] = 1.0F;
                    weight.count = 1U;
                }
                weights_[y * targetX.size() + x] = weight;
            }
        }
        sourceLayerSize_ = sourceX.size();
        targetNx_ = targetX.size();
        targetNy_ = targetY.size();
        validSource_.clear();
        validSource_.shrink_to_fit();
    }

    [[nodiscard]] Field3D apply(const Field3D& source) const {
        if (source.nx * source.ny != sourceLayerSize_) {
            throw std::runtime_error("Horizontal interpolation plan/source mismatch");
        }
        Field3D result(source.nz, targetNy_, targetNx_);
#pragma omp parallel for if(source.nz * weights_.size() > 4096)
        for (std::int64_t zSigned = 0; zSigned < static_cast<std::int64_t>(source.nz); ++zSigned) {
            const std::size_t z = static_cast<std::size_t>(zSigned);
            const std::size_t sourceOffset = z * source.nx * source.ny;
            const std::size_t targetOffset = z * targetNx_ * targetNy_;
            for (std::size_t point = 0; point < weights_.size(); ++point) {
                const HorizontalWeight& weight = weights_[point];
                double value = 0.0;
                for (std::size_t corner = 0; corner < weight.count; ++corner) {
                    value += weight.weights[corner]
                        * source.values[sourceOffset + weight.indices[corner]];
                }
                result.values[targetOffset + point] = static_cast<float>(value);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<Field3D> applyBatch(
        const std::vector<const Field3D*>& fields) const {
        if (fields.empty() || fields.front() == nullptr) {
            throw std::runtime_error(
                "Horizontal interpolation batch received no fields");
        }
        const Field3D& first = *fields.front();
        if (first.nx * first.ny != sourceLayerSize_) {
            throw std::runtime_error("Horizontal interpolation plan/source mismatch");
        }
        std::vector<Field3D> result;
        result.reserve(fields.size());
        for (const Field3D* field : fields) {
            if (field == nullptr || field->nz != first.nz || field->ny != first.ny
                || field->nx != first.nx) {
                throw std::runtime_error(
                    "Horizontal interpolation batch fields have mismatched dimensions");
            }
            result.emplace_back(first.nz, targetNy_, targetNx_);
        }

#pragma omp parallel for if(first.nz * weights_.size() > 4096)
        for (std::int64_t zSigned = 0;
             zSigned < static_cast<std::int64_t>(first.nz); ++zSigned) {
            const std::size_t z = static_cast<std::size_t>(zSigned);
            const std::size_t sourceOffset = z * sourceLayerSize_;
            const std::size_t targetOffset = z * targetNx_ * targetNy_;
            for (std::size_t point = 0U; point < weights_.size(); ++point) {
                const HorizontalWeight& weight = weights_[point];
                for (std::size_t component = 0U; component < fields.size(); ++component) {
                    double value = 0.0;
                    for (std::size_t corner = 0U; corner < weight.count; ++corner) {
                        value += weight.weights[corner]
                            * fields[component]->values[
                                sourceOffset + weight.indices[corner]];
                    }
                    result[component].values[targetOffset + point] =
                        static_cast<float>(value);
                }
            }
        }
        return result;
    }

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return validSource_.capacity() * sizeof(std::size_t)
            + weights_.capacity() * sizeof(HorizontalWeight);
    }

    [[nodiscard]] std::vector<Field3D> applyOpenCl(
        const std::vector<const Field3D*>& fields,
        int gpuDevice,
        std::ostream& output) const {
        if (fields.size() != 3U && fields.size() != 4U) {
            throw OpenClLuwbcError(
                "luwbc fused horizontal OpenCL path requires U/V/W[/T]");
        }
        if (fields.front() == nullptr) {
            throw OpenClLuwbcError(
                "luwbc fused horizontal OpenCL path received a null field");
        }
        const Field3D& first = *fields.front();
        const std::size_t sourceLayerSize = first.nx * first.ny;
        if (sourceLayerSize != sourceLayerSize_) {
            throw OpenClLuwbcError(
                "luwbc horizontal interpolation plan/source mismatch");
        }
        OpenClLuwbcHorizontalRequest request;
        request.deviceIndex = gpuDevice;
        request.sourceLayerSize = sourceLayerSize;
        request.layerCount = first.nz;
        request.targetPointCount = weights_.size();
        request.componentCount = fields.size();
        request.values.reserve(fields.size() * first.values.size());
        for (const Field3D* field : fields) {
            if (field == nullptr || field->nz != first.nz || field->ny != first.ny
                || field->nx != first.nx) {
                throw OpenClLuwbcError(
                    "luwbc fused horizontal OpenCL fields have mismatched dimensions");
            }
            request.values.insert(request.values.end(), field->values.begin(),
                                  field->values.end());
        }
        request.indices.resize(weights_.size() * 3U, 0U);
        request.weights.resize(weights_.size() * 3U, 0.0);
        request.counts.resize(weights_.size());
        for (std::size_t target = 0U; target < weights_.size(); ++target) {
            const HorizontalWeight& weight = weights_[target];
            request.counts[target] = static_cast<std::uint32_t>(weight.count);
            for (std::size_t corner = 0U; corner < weight.count; ++corner) {
                if (weight.indices[corner] >
                    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                    throw OpenClLuwbcError(
                        "luwbc horizontal source index exceeds OpenCL uint32 capacity");
                }
                const std::size_t offset = target * 3U + corner;
                request.indices[offset] =
                    static_cast<std::uint32_t>(weight.indices[corner]);
                request.weights[offset] = weight.weights[corner];
            }
        }

        const OpenClLuwbcResult result =
            runNativeOpenClLuwbcHorizontal(request);
        output << "[INFO] OpenCL fused horizontal resampling backend: "
               << result.device << '\n';
        std::vector<Field3D> converted;
        converted.reserve(fields.size());
        const std::size_t valuesPerField =
            first.nz * targetNy_ * targetNx_;
        for (std::size_t component = 0U; component < fields.size(); ++component) {
            Field3D field(first.nz, targetNy_, targetNx_);
            const auto begin = result.values.begin()
                + static_cast<std::ptrdiff_t>(component * valuesPerField);
            std::copy_n(begin, static_cast<std::ptrdiff_t>(valuesPerField),
                        field.values.begin());
            converted.push_back(std::move(field));
        }
        return converted;
    }

private:
    std::vector<std::size_t> validSource_;
    std::vector<HorizontalWeight> weights_;
    std::size_t sourceLayerSize_ = 0U;
    std::size_t targetNx_ = 0U;
    std::size_t targetNy_ = 0U;
};

struct HorizontalPlanKey {
    std::string scope;
    std::vector<std::uint64_t> sourceX;
    std::vector<std::uint64_t> sourceY;
    std::vector<std::uint64_t> targetX;
    std::vector<std::uint64_t> targetY;

    [[nodiscard]] bool operator==(const HorizontalPlanKey&) const = default;

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return scope.capacity()
            + (sourceX.capacity() + sourceY.capacity() + targetX.capacity()
               + targetY.capacity()) * sizeof(std::uint64_t);
    }
};

using HorizontalMappingCache =
    BoundedExactPlanCache<HorizontalPlanKey, HorizontalPlan>;

HorizontalMappingCache& horizontalMappingCache() {
    static HorizontalMappingCache cache(3U, 256U * 1024U * 1024U);
    return cache;
}

HorizontalMappingCache::Lookup acquireHorizontalPlan(
    const std::string& scope,
    const std::vector<double>& sourceX,
    const std::vector<double>& sourceY,
    const std::vector<double>& targetX,
    const std::vector<double>& targetY) {
    HorizontalPlanKey key;
    key.scope = scope;
    key.sourceX = exactDoubleBits(sourceX);
    key.sourceY = exactDoubleBits(sourceY);
    key.targetX = exactDoubleBits(targetX);
    key.targetY = exactDoubleBits(targetY);
    return horizontalMappingCache().acquire(key, [&]() {
        return HorizontalPlan(sourceX, sourceY, targetX, targetY);
    });
}

double meanFinite(const Field3D& field) {
    long double sum = 0.0L;
    std::size_t count = 0U;
    for (const float value : field.values) {
        if (std::isfinite(value)) { sum += value; ++count; }
    }
    return count == 0U ? std::numeric_limits<double>::quiet_NaN()
                       : static_cast<double>(sum / static_cast<long double>(count));
}

struct VerticalWeight {
    std::size_t lower = 0U;
    std::size_t upper = 0U;
    double fraction = 0.0;
};

VerticalWeight verticalWeight(const std::vector<double>& z, double query) {
    if (z.empty()) {
        throw std::runtime_error("Cannot sample an empty vertical coordinate");
    }
    if (query <= z.front()) return {0U, 0U, 0.0};
    if (query >= z.back()) return {z.size() - 1U, z.size() - 1U, 0.0};
    const std::size_t upper = static_cast<std::size_t>(
        std::lower_bound(z.begin(), z.end(), query) - z.begin());
    const std::size_t lower = upper - 1U;
    const double lowerDistance = std::abs(query - z[lower]);
    const double upperDistance = std::abs(query - z[upper]);
    if (lowerDistance < 1e-6) return {lower, lower, 0.0};
    if (upperDistance < 1e-6) return {upper, upper, 0.0};
    // The legacy IDW weights between two bracketing levels are algebraically
    // identical to linear interpolation.
    return {lower, upper, lowerDistance / (lowerDistance + upperDistance)};
}

float verticalLinearSample(const Field3D& field,
                           std::size_t y,
                           std::size_t x,
                           const VerticalWeight& weight) {
    const float lower = field.at(weight.lower, y, x);
    if (weight.lower == weight.upper) return lower;
    return static_cast<float>(
        lerpNan(lower, field.at(weight.upper, y, x), weight.fraction));
}

float verticalLinearSample(const Field3D& field,
                           std::size_t y,
                           std::size_t x,
                           double query,
                           const std::vector<double>& z) {
    return verticalLinearSample(field, y, x, verticalWeight(z, query));
}

std::optional<std::size_t> firstDimensionLength(
    const NcDataset& dataset,
    std::initializer_list<std::string_view> names) {
    for (const std::string_view name : names) {
        if (const auto length = dataset.dimensionLength(name)) return length;
    }
    return std::nullopt;
}

std::vector<double> readMassDimensionCoordinates(const NcDataset& dataset,
                                                 std::string_view logicalName,
                                                 const Crop& crop) {
    const std::size_t begin = logicalName == "lon" ? crop.x0 : crop.y0;
    const std::size_t countExpected = logicalName == "lon" ? crop.nx() : crop.ny();
    if (dataset.has(logicalName)) {
        const NcVariable& variable = dataset.variable(logicalName);
        const auto axis = dimensionIndex(variable, logicalName);
        if (axis && variable.shape[*axis] >= begin + countExpected) {
            std::vector<std::size_t> start(variable.shape.size(), 0U);
            std::vector<std::size_t> count(variable.shape.size(), 1U);
            start[*axis] = begin;
            count[*axis] = countExpected;
            return dataset.read(variable, start, count);
        }
    }
    std::vector<double> result(countExpected);
    for (std::size_t index = 0; index < countExpected; ++index) {
        result[index] = static_cast<double>(begin + index);
    }
    return result;
}

std::vector<double> readOneDimensionalVariable(const NcDataset& dataset,
                                               const NcVariable& variable,
                                               std::string_view logicalName) {
    const auto axis = dimensionIndex(variable, logicalName);
    if (!axis) return {};
    std::vector<std::size_t> start(variable.shape.size(), 0U);
    std::vector<std::size_t> count(variable.shape.size(), 1U);
    count[*axis] = variable.shape[*axis];
    return dataset.read(variable, start, count);
}

struct VerticalInfo {
    std::string dimension;
    std::vector<double> levels;
    std::vector<double> dimensionCoordinates;
    std::string units;
};

VerticalInfo detectVertical(const NcDataset& dataset,
                            const Crop& crop,
                            std::ostream& output) {
    for (const std::string candidate : {"lev", "height_agl", "height", "elevation"}) {
        const auto length = dataset.dimensionLength(candidate);
        if (!length) continue;
        std::vector<double> levels;
        std::string units;
        if (dataset.has(candidate)) {
            const NcVariable& variable = dataset.variable(candidate);
            levels = readOneDimensionalVariable(dataset, variable, candidate);
            units = lowerAscii(dataset.textAttribute(variable, "units"));
        }
        if (levels.size() != *length) {
            levels = linearCoordinates(0.0, static_cast<double>(*length - 1U), *length);
        }
        return VerticalInfo{candidate, levels, levels, units};
    }

    const auto massLevels = dataset.dimensionLength("bottom_top");
    if (!massLevels) {
        throw std::runtime_error(
            "Vertical coord not found. Expect lev or height_agl or height or elevation or bottom_top");
    }
    std::vector<double> dimensionCoordinates =
        linearCoordinates(0.0, static_cast<double>(*massLevels - 1U), *massLevels);
    if (dataset.has("bottom_top")) {
        const std::vector<double> coordinate = readOneDimensionalVariable(
            dataset, dataset.variable("bottom_top"), "bottom_top");
        if (coordinate.size() == *massLevels) dimensionCoordinates = coordinate;
    }
    std::vector<double> levels;
    if (dataset.has("PH") && dataset.has("PHB")
        && dataset.dimensionLength("bottom_top_stag")) {
        try {
            Field3D ph = readField3D(dataset, "PH", "bottom_top", crop, false, output);
            Field3D phb = readField3D(dataset, "PHB", "bottom_top", crop, false, output);
            if (ph.nz == *massLevels && phb.nz == ph.nz && ph.nx == crop.nx()
                && ph.ny == crop.ny()) {
                std::optional<Surface2D> terrain;
                if (dataset.has("HGT")) terrain = readSurface2D(dataset, "HGT", crop);
                levels.resize(ph.nz);
                for (std::size_t z = 0; z < ph.nz; ++z) {
                    long double sum = 0.0L;
                    std::size_t count = 0U;
                    for (std::size_t y = 0; y < ph.ny; ++y) {
                        for (std::size_t x = 0; x < ph.nx; ++x) {
                            double value = (static_cast<double>(ph.at(z, y, x))
                                + static_cast<double>(phb.at(z, y, x))) / 9.81;
                            if (terrain) value -= terrain->at(y, x);
                            if (std::isfinite(value)) { sum += value; ++count; }
                        }
                    }
                    if (count == 0U) throw std::runtime_error("WRF PH/PHB height is invalid");
                    levels[z] = static_cast<double>(sum / static_cast<long double>(count));
                }
            }
        } catch (const std::exception& exception) {
            warn(output, "Failed to derive WRF AGL height from PH/PHB; use bottom_top coordinate. Reason: "
                 + std::string(exception.what()));
            levels.clear();
        }
    }
    if (levels.empty() && dataset.has("bottom_top")) {
        levels = readOneDimensionalVariable(dataset, dataset.variable("bottom_top"), "bottom_top");
    }
    if (levels.size() != *massLevels) levels = dimensionCoordinates;
    for (std::size_t index = 1U; index < levels.size(); ++index) {
        if (levels[index] <= levels[index - 1U]) levels[index] = levels[index - 1U] + 1e-3;
    }
    return VerticalInfo{"bottom_top", levels, dimensionCoordinates, "m"};
}

Field3D resizeHorizontalByIndex(const Field3D& source,
                               std::size_t targetNy,
                               std::size_t targetNx) {
    if (source.nx == targetNx && source.ny == targetNy) return source;
    const std::vector<double> sx = linearCoordinates(0.0, static_cast<double>(source.nx - 1U), source.nx);
    const std::vector<double> sy = linearCoordinates(0.0, static_cast<double>(source.ny - 1U), source.ny);
    const std::vector<double> tx = linearCoordinates(0.0, static_cast<double>(source.nx - 1U), targetNx);
    const std::vector<double> ty = linearCoordinates(0.0, static_cast<double>(source.ny - 1U), targetNy);
    return resampleHorizontalLinear(source, sx, sy, tx, ty);
}

void ensureFieldShape(Field3D& field,
                      std::size_t expectedNz,
                      std::size_t expectedNy,
                      std::size_t expectedNx,
                      const std::string& name,
                      std::ostream& output) {
    if (field.nx != expectedNx || field.ny != expectedNy) {
        warn(output, "Wind/temperature field " + name +
             " does not match the U mass grid after destaggering; interpolate to the U grid");
        field = resizeHorizontalByIndex(field, expectedNy, expectedNx);
    }
    if (field.nz != expectedNz) {
        const std::vector<double> source = linearCoordinates(0.0,
            static_cast<double>(field.nz > 0U ? field.nz - 1U : 0U), field.nz);
        const std::vector<double> target = linearCoordinates(0.0,
            static_cast<double>(field.nz > 0U ? field.nz - 1U : 0U), expectedNz);
        warn(output, "Wind/temperature field " + name +
             " does not match the mass vertical grid after destaggering; interpolate to the U grid");
        field = resampleVerticalPchip(field, source, target);
    }
}

void denseInterpolateNoTerrain(Field3D& u,
                               Field3D& v,
                               Field3D& w,
                               std::optional<Field3D>& temperature,
                               GeoGrid& geo,
                               std::vector<double>& xCoordinates,
                               std::vector<double>& yCoordinates,
                               VerticalInfo& vertical,
                               std::ostream& output,
                               ProgressEmitter& progress) {
    const std::size_t targetNx = std::max<std::size_t>(u.nx, 200U);
    const std::size_t targetNy = std::max<std::size_t>(u.ny, 200U);
    const std::size_t targetNz = std::max<std::size_t>(u.nz, 10U);
    if (targetNx == u.nx && targetNy == u.ny && targetNz == u.nz) return;
    if (u.nx < targetNx) info(output, "Dimension lon has " + std::to_string(u.nx)
                              + " points, upsample to " + std::to_string(targetNx));
    if (u.ny < targetNy) info(output, "Dimension lat has " + std::to_string(u.ny)
                              + " points, upsample to " + std::to_string(targetNy));
    if (u.nz < targetNz) info(output, "Dimension " + vertical.dimension + " has "
                              + std::to_string(u.nz) + " points, upsample to "
                              + std::to_string(targetNz));
    progress.stage(QStringLiteral("dense_interpolation"),
                   QStringLiteral("Interpolating wind field to dense compute grid"));
    info(output, "Start interpolation on lon, lat and vertical (Dense Compute)...");

    if (targetNx != u.nx || targetNy != u.ny) {
        info(output, "Interpolating horizontal dimensions with linear method...");
        const std::vector<double> targetX = linearCoordinates(xCoordinates.front(),
                                                               xCoordinates.back(), targetNx);
        const std::vector<double> targetY = linearCoordinates(yCoordinates.front(),
                                                               yCoordinates.back(), targetNy);
        u = resampleHorizontalLinear(u, xCoordinates, yCoordinates, targetX, targetY);
        v = resampleHorizontalLinear(v, xCoordinates, yCoordinates, targetX, targetY);
        w = resampleHorizontalLinear(w, xCoordinates, yCoordinates, targetX, targetY);
        if (temperature) {
            *temperature = resampleHorizontalLinear(*temperature, xCoordinates, yCoordinates,
                                                    targetX, targetY);
        }
        geo = resampleGeoGridLinear(geo, targetNy, targetNx);
        xCoordinates = targetX;
        yCoordinates = targetY;
    }

    if (targetNz != u.nz) {
        info(output, "Interpolating vertical dimension with pchip method...");
        const std::vector<double> targetDimension = linearCoordinates(
            vertical.dimensionCoordinates.front(), vertical.dimensionCoordinates.back(), targetNz);
        u = resampleVerticalPchip(u, vertical.dimensionCoordinates, targetDimension);
        v = resampleVerticalPchip(v, vertical.dimensionCoordinates, targetDimension);
        w = resampleVerticalPchip(w, vertical.dimensionCoordinates, targetDimension);
        if (temperature) {
            *temperature = resampleVerticalPchip(*temperature,
                                                 vertical.dimensionCoordinates, targetDimension);
        }
        const std::vector<double> slopes = pchipSlopes(vertical.dimensionCoordinates,
                                                       vertical.levels);
        std::vector<double> levels(targetNz);
        for (std::size_t index = 0; index < targetNz; ++index) {
            levels[index] = pchipEvaluate(vertical.dimensionCoordinates, vertical.levels,
                                          slopes, targetDimension[index]);
        }
        vertical.dimensionCoordinates = targetDimension;
        vertical.levels = levels;
    }
    info(output, "Finished lon/lat/vertical interpolation");
    info(output, "Refreshed vertical levels after interpolation: " + vertical.dimension +
         " (" + std::to_string(vertical.levels.size()) + " levels)");
}

std::string resolveWindVariable(const NcDataset& dataset,
                                std::string_view preferred,
                                std::initializer_list<std::string_view> fallbacks,
                                std::string_view kind,
                                std::ostream& output) {
    if (dataset.has(preferred)) return std::string(preferred);
    for (const std::string_view fallback : fallbacks) {
        if (dataset.has(fallback)) {
            warn(output, "Wind variable " + std::string(preferred) + " not found, use "
                 + std::string(fallback) + " for " + std::string(kind));
            return std::string(fallback);
        }
    }
    throw std::runtime_error("Wind variable " + std::string(preferred)
                             + " not found, and fallbacks also missing for " + std::string(kind));
}

struct TemperatureVariable {
    std::string name;
    bool celsius = false;
};

std::optional<TemperatureVariable> resolveTemperature(const NcDataset& dataset,
                                                      std::ostream& output) {
    const auto kelvin = dataset.firstVariable(
        {"TK", "tk", "T", "t", "Temp", "temp", "Temperature", "temperature"});
    const auto celsius = dataset.firstVariable({"TC", "tc"});
    if (celsius) {
        if (kelvin) {
            warn(output, "Both Kelvin and Celsius temperature fields found (" + *kelvin + ", "
                 + *celsius + "); use " + *celsius + " (Celsius) with conversion to Kelvin");
        }
        return TemperatureVariable{*celsius, true};
    }
    if (kelvin) return TemperatureVariable{*kelvin, false};
    return std::nullopt;
}

struct DemData {
    std::vector<XY> points;
    std::vector<double> elevations;
};

#if LUW_HAS_GDAL

struct GdalDatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) GDALClose(dataset);
    }
};

struct OgrFeatureCloser {
    void operator()(OGRFeature* feature) const noexcept {
        if (feature != nullptr) OGRFeature::DestroyFeature(feature);
    }
};

struct OgrTransformCloser {
    void operator()(OGRCoordinateTransformation* transform) const noexcept {
        if (transform != nullptr) OCTDestroyCoordinateTransformation(transform);
    }
};

using GdalDatasetPtr = std::unique_ptr<GDALDataset, GdalDatasetCloser>;
using OgrFeaturePtr = std::unique_ptr<OGRFeature, OgrFeatureCloser>;
using OgrTransformPtr = std::unique_ptr<OGRCoordinateTransformation, OgrTransformCloser>;

OgrTransformPtr makeOgrTransform(const OGRSpatialReference* sourceReference, int targetEpsg) {
    OGRSpatialReference source;
    if (sourceReference != nullptr) source = *sourceReference;
    else source.importFromEPSG(4326);
    OGRSpatialReference target;
    if (target.importFromEPSG(targetEpsg) != OGRERR_NONE) {
        return {};
    }
    source.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    target.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return OgrTransformPtr(OGRCreateCoordinateTransformation(&source, &target));
}

std::optional<DemData> loadDemShapefile(const fs::path& path,
                                        int targetEpsg,
                                        std::ostream& output) {
    info(output, "Loading DEM from: " + pathToUtf8(path));
    GdalDatasetPtr dataset(static_cast<GDALDataset*>(
        GDALOpenEx(pathToUtf8(path).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
                   nullptr, nullptr, nullptr)));
    if (!dataset) throw std::runtime_error("GDAL/OGR could not open DEM shapefile");
    OGRLayer* layer = dataset->GetLayer(0);
    if (layer == nullptr) throw std::runtime_error("DEM shapefile has no vector layer");
    OGRFeatureDefn* definition = layer->GetLayerDefn();
    if (definition == nullptr) throw std::runtime_error("DEM shapefile has no schema");
    int elevationIndex = -1;
    std::string elevationName;
    for (const char* name : {"elevation", "Elevation", "ELEVATION", "height", "Height", "z", "Z"}) {
        const int index = definition->GetFieldIndex(name);
        if (index >= 0) { elevationIndex = index; elevationName = name; break; }
    }
    if (elevationIndex < 0) {
        warn(output, "Elevation field not found in DEM shapefile");
        return std::nullopt;
    }
    info(output, "Using elevation column: " + elevationName);
    OgrTransformPtr transform = makeOgrTransform(layer->GetSpatialRef(), targetEpsg);
    if (!transform) throw std::runtime_error("Cannot transform DEM shapefile to working UTM CRS");
    DemData result;
    std::vector<double> centroidX;
    std::vector<double> centroidY;
    std::vector<double> centroidElevation;
    layer->ResetReading();
    while (true) {
        OgrFeaturePtr feature(layer->GetNextFeature());
        if (!feature) break;
        if (!feature->IsFieldSetAndNotNull(elevationIndex)) continue;
        const double elevation = feature->GetFieldAsDouble(elevationIndex);
        if (!std::isfinite(elevation)) continue;
        OGRGeometry* geometry = feature->GetGeometryRef();
        if (geometry == nullptr || geometry->IsEmpty()) continue;
        OGRPoint centroid;
        if (geometry->Centroid(&centroid) != OGRERR_NONE) continue;
        centroidX.push_back(centroid.getX());
        centroidY.push_back(centroid.getY());
        centroidElevation.push_back(elevation);
    }
    constexpr std::size_t kTransformChunk = 65536U;
    result.points.reserve(centroidX.size());
    result.elevations.reserve(centroidElevation.size());
    for (std::size_t start = 0U; start < centroidX.size(); start += kTransformChunk) {
        const std::size_t count = std::min(kTransformChunk, centroidX.size() - start);
        std::vector<int> success(count, 0);
        (void)transform->Transform(static_cast<int>(count),
                                   centroidX.data() + start,
                                   centroidY.data() + start,
                                   nullptr, success.data());
        for (std::size_t offset = 0U; offset < count; ++offset) {
            if (success[offset] == 0) continue;
            result.points.push_back(
                {centroidX[start + offset], centroidY[start + offset]});
            result.elevations.push_back(centroidElevation[start + offset]);
        }
    }
    if (result.points.empty()) {
        warn(output, "No valid DEM points found");
        return std::nullopt;
    }
    return result;
}

#endif // LUW_HAS_GDAL

std::optional<DemData> loadDem(const fs::path& project,
                               int targetEpsg,
                               const std::string& caseName,
                               const std::optional<Bounds>& requestedBounds,
                               std::ostream& output) {
    const fs::path folder = project / "terrain_db";
    std::vector<fs::path> shapefiles;
    std::vector<fs::path> rasters;
    std::error_code iteratorError;
    if (fs::is_directory(folder)) {
        for (fs::directory_iterator iterator(folder, iteratorError), end;
             !iteratorError && iterator != end; iterator.increment(iteratorError)) {
            std::error_code typeError;
            if (!iterator->is_regular_file(typeError) || typeError) continue;
            const std::string extension = lowerAscii(pathToUtf8(iterator->path().extension()));
            if (extension == ".shp") shapefiles.push_back(iterator->path());
            else if (extension == ".tif" || extension == ".tiff") rasters.push_back(iterator->path());
        }
    }
    std::sort(shapefiles.begin(), shapefiles.end());
    std::sort(rasters.begin(), rasters.end());
    if (shapefiles.empty()) {
        warn(output, "No DEM shapefile found in terrain_db. Trying GeoTIFF fallback...");
        if (!requestedBounds) {
            warn(output, "Failed to parse DEM bounds from conf: cut_lon_manual must be a list/tuple of length 2, e.g. [113.302, 113.342].");
        }
    }
#if LUW_HAS_GDAL
    try {
        GDALAllRegister();
        if (shapefiles.empty() && requestedBounds) {
            if (requestedBounds->lonMin == requestedBounds->lonMax ||
                requestedBounds->latMin == requestedBounds->latMax) {
                warn(output, "Failed to parse DEM bounds from conf: Invalid bbox: lon/lat range has zero size.");
            } else {
                const QString installed =
                    resolveInstallFile(QStringLiteral("database/dem.tif"));
                const DemTifCacheRequest request{
                    project,
                    pathFromUtf8(installed.toUtf8().toStdString()),
                    caseName,
                    {requestedBounds->lonMin, requestedBounds->lonMax,
                     requestedBounds->latMin, requestedBounds->latMax}};
                if (const auto cached = ensureDemShapefileFromTif(request, output);
                    cached && fs::is_regular_file(cached->shapefile)) {
                    shapefiles.push_back(cached->shapefile);
                }
            }
        }
        std::optional<DemData> result;
        if (!shapefiles.empty()) result = loadDemShapefile(shapefiles.front(), targetEpsg, output);
        if (!result || result->points.empty()) {
            warn(output, "No DEM shapefile available, proceeding without terrain data");
            return std::nullopt;
        }
        const auto [minimum, maximum] = std::minmax_element(result->elevations.begin(),
                                                             result->elevations.end());
        info(output, "Loaded " + std::to_string(result->elevations.size()) + " DEM points");
        info(output, "Original elevation range (sea level): " + fixed(*minimum, 2) + " to "
             + fixed(*maximum, 2) + " meters");
        const double datum = *minimum;
        for (double& value : result->elevations) value -= datum;
        const double adjustedMaximum = *maximum - datum;
        info(output, "Adjusted to datum (lowest point = 0): 0.00 to "
             + fixed(adjustedMaximum, 2) + " meters");
        info(output, "Datum offset applied: " + fixed(datum, 2) + "m");
        return result;
    } catch (const DemTifCacheCancelled&) {
        throw;
    } catch (const std::exception& exception) {
        warn(output, "Failed to load DEM data: " + std::string(exception.what()));
        return std::nullopt;
    }
#else
    (void)targetEpsg;
    (void)caseName;
    (void)requestedBounds;
    if (!shapefiles.empty() || !rasters.empty()) {
        warn(output, "DEM data found, but this build has no GDAL/OGR support; proceeding without terrain data");
    } else {
        warn(output, "No DEM shapefile available, proceeding without terrain data");
    }
    return std::nullopt;
#endif
}

struct DemNeighborPlanKey {
    std::string scope;
    std::vector<std::uint64_t> sourcePoints;
    std::vector<std::uint64_t> targetX;
    std::vector<std::uint64_t> targetY;

    [[nodiscard]] bool operator==(const DemNeighborPlanKey&) const = default;

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return scope.capacity()
            + (sourcePoints.capacity() + targetX.capacity() + targetY.capacity())
                * sizeof(std::uint64_t);
    }
};

struct DemNeighborPlan {
    std::size_t neighborCount = 0U;
    std::vector<std::uint32_t> indices;

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return indices.capacity() * sizeof(std::uint32_t);
    }
};

using DemNeighborCache = BoundedExactPlanCache<DemNeighborPlanKey, DemNeighborPlan>;

DemNeighborCache& demNeighborCache() {
    static DemNeighborCache cache(2U, 192U * 1024U * 1024U);
    return cache;
}

DemNeighborCache::Lookup acquireDemNeighborPlan(
    const std::string& scope,
    const std::vector<XY>& sourcePoints,
    const std::vector<double>& targetX,
    const std::vector<double>& targetY) {
    if (sourcePoints.empty()) {
        throw std::runtime_error("Cannot build a DEM neighbor plan without source points");
    }
    if (sourcePoints.size()
        > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("DEM point count exceeds uint32 neighbor-plan capacity");
    }
    DemNeighborPlanKey key;
    key.scope = scope;
    key.sourcePoints.reserve(
        checkedProduct({sourcePoints.size(), std::size_t{2U}}));
    for (const XY& point : sourcePoints) {
        key.sourcePoints.push_back(std::bit_cast<std::uint64_t>(point.x));
        key.sourcePoints.push_back(std::bit_cast<std::uint64_t>(point.y));
    }
    key.targetX = exactDoubleBits(targetX);
    key.targetY = exactDoubleBits(targetY);
    return demNeighborCache().acquire(key, [&]() {
        vtkNew<vtkPoints> vtkPointsData;
        for (const XY& point : sourcePoints) {
            vtkPointsData->InsertNextPoint(point.x, point.y, 0.0);
        }
        vtkNew<vtkPolyData> data;
        data->SetPoints(vtkPointsData);
        vtkNew<vtkStaticPointLocator> locator;
        locator->SetDataSet(data);
        locator->BuildLocator();
        DemNeighborPlan plan;
        plan.neighborCount = std::min<std::size_t>(12U, sourcePoints.size());
        const std::size_t targetCount = checkedProduct(
            {targetX.size(), targetY.size()});
        plan.indices.resize(checkedProduct(
            {targetCount, plan.neighborCount}));
#pragma omp parallel for if(targetCount > 512)
        for (std::int64_t pointSigned = 0;
             pointSigned < static_cast<std::int64_t>(targetCount); ++pointSigned) {
            const std::size_t target = static_cast<std::size_t>(pointSigned);
            const std::size_t y = target / targetX.size();
            const std::size_t x = target % targetX.size();
            const double query[3]{targetX[x], targetY[y], 0.0};
            vtkNew<vtkIdList> ids;
            locator->FindClosestNPoints(
                static_cast<int>(plan.neighborCount), query, ids);
            for (std::size_t entry = 0U; entry < plan.neighborCount; ++entry) {
                plan.indices[target * plan.neighborCount + entry] =
                    static_cast<std::uint32_t>(
                        ids->GetId(static_cast<vtkIdType>(entry)));
            }
        }
        return plan;
    });
}

std::vector<double> interpolateDemGrid(const DemData& dem,
                                       const RotationFrame& frame,
                                       const std::vector<double>& xGrid,
                                       const std::vector<double>& yGrid,
                                       std::vector<XY>* shiftedPoints,
                                       const std::string& mappingCacheScope,
                                       const PreprocessComputeConfig& computePolicy,
                                       std::ostream& output) {
    std::vector<XY> points(dem.points.size());
    const RotationOperator rotation(frame.degrees, frame.pivot);
#pragma omp parallel for if(points.size() > 4096)
    for (std::int64_t indexSigned = 0;
         indexSigned < static_cast<std::int64_t>(points.size()); ++indexSigned) {
        const std::size_t index = static_cast<std::size_t>(indexSigned);
        const XY rotated = rotation.apply(dem.points[index]);
        points[index] = {rotated.x - frame.xOrigin,
                         rotated.y - frame.yOrigin};
    }
    if (shiftedPoints != nullptr) *shiftedPoints = points;
    const double buffer = std::max(xGrid.back() - xGrid.front(), yGrid.back() - yGrid.front()) * 0.1;
    std::vector<XY> croppedPoints;
    std::vector<double> croppedElevation;
    for (std::size_t index = 0; index < points.size(); ++index) {
        const XY point = points[index];
        if (point.x >= xGrid.front() - buffer && point.x <= xGrid.back() + buffer
            && point.y >= yGrid.front() - buffer && point.y <= yGrid.back() + buffer) {
            croppedPoints.push_back(point);
            croppedElevation.push_back(dem.elevations[index]);
        }
    }
    info(output, "Cropped DEM from " + std::to_string(points.size()) + " to "
         + std::to_string(croppedPoints.size()) + " points within grid bounds");
    if (croppedPoints.empty()) {
        warn(output, "No DEM points within grid bounds, using zeros");
        return std::vector<double>(xGrid.size() * yGrid.size(), 0.0);
    }
    const DemNeighborCache::Lookup neighborPlan = acquireDemNeighborPlan(
        mappingCacheScope, croppedPoints, xGrid, yGrid);
    reportPlanCache(output, "DEM IDW neighbor", neighborPlan.hit,
                    neighborPlan.retained, neighborPlan.evicted);
    const std::size_t neighbors = neighborPlan.plan->neighborCount;
    std::vector<double> result(checkedProduct({xGrid.size(), yGrid.size()}));
    if (computePolicy.useGpu) {
        OpenClLuwbcDemIdwRequest request;
        request.deviceIndex = computePolicy.gpuDevice;
        request.neighborCount = neighbors;
        request.sourceX.reserve(croppedPoints.size());
        request.sourceY.reserve(croppedPoints.size());
        for (const XY& point : croppedPoints) {
            request.sourceX.push_back(point.x);
            request.sourceY.push_back(point.y);
        }
        request.sourceValues = croppedElevation;
        request.targetX.resize(result.size());
        request.targetY.resize(result.size());
        request.neighborIndices = neighborPlan.plan->indices;
#pragma omp parallel for if(result.size() > 512)
        for (std::int64_t pointSigned = 0;
             pointSigned < static_cast<std::int64_t>(result.size()); ++pointSigned) {
            const std::size_t pointIndex = static_cast<std::size_t>(pointSigned);
            const std::size_t y = pointIndex / xGrid.size();
            const std::size_t x = pointIndex % xGrid.size();
            request.targetX[pointIndex] = xGrid[x];
            request.targetY[pointIndex] = yGrid[y];
        }
        OpenClLuwbcDoubleResult gpu = runNativeOpenClLuwbcDemIdw(request);
        if (gpu.values.size() != result.size()) {
            throw OpenClLuwbcError(
                "OpenCL DEM IDW returned an inconsistent target count");
        }
        result = std::move(gpu.values);
        info(output, "OpenCL DEM IDW interpolation backend: " + gpu.device
             + "; targets=" + std::to_string(result.size())
             + ", neighbors=" + std::to_string(neighbors));
    } else {
#pragma omp parallel for if(result.size() > 512)
        for (std::int64_t pointSigned = 0;
             pointSigned < static_cast<std::int64_t>(result.size()); ++pointSigned) {
            const std::size_t pointIndex = static_cast<std::size_t>(pointSigned);
            const std::size_t y = pointIndex / xGrid.size();
            const std::size_t x = pointIndex % xGrid.size();
            const double query[3]{xGrid[x], yGrid[y], 0.0};
            long double weighted = 0.0L;
            long double weights = 0.0L;
            for (std::size_t entry = 0U; entry < neighbors; ++entry) {
                const std::size_t id = neighborPlan.plan->indices[
                    pointIndex * neighbors + entry];
                const double dx = croppedPoints[id].x - query[0];
                const double dy = croppedPoints[id].y - query[1];
                const double distance = std::max(std::sqrt(dx * dx + dy * dy), 1e-10);
                const double weight = 1.0 / (distance * distance);
                weighted += weight * croppedElevation[id];
                weights += weight;
            }
            result[pointIndex] = static_cast<double>(weighted / weights);
        }
    }
    double minimum = *std::min_element(result.begin(), result.end());
    if (minimum < 0.0) {
        warn(output, "Interpolation created negative values (min=" + fixed(minimum, 2)
             + "m), clipping to 0");
        for (double& value : result) value = std::max(value, 0.0);
        minimum = 0.0;
    }
    const double maximum = *std::max_element(result.begin(), result.end());
    info(output, "DEM interpolated to grid: absolute elevation range " + fixed(minimum, 2)
         + " to " + fixed(maximum, 2) + " meters");
    for (double& value : result) value -= minimum;
    info(output, "DEM elevation difference range: 0.00 to " + fixed(maximum - minimum, 2)
         + " meters (relative to grid minimum)");
    info(output, "Grid minimum elevation (datum): " + fixed(minimum, 2) + " meters");
    return result;
}

void pickleAppendUnicode(std::vector<std::uint8_t>& bytes, std::string_view value) {
    bytes.push_back('X'); // BINUNICODE
    const std::uint32_t size = static_cast<std::uint32_t>(value.size());
    for (int shift = 0; shift < 32; shift += 8) bytes.push_back(static_cast<std::uint8_t>(size >> shift));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void pickleAppendInt(std::vector<std::uint8_t>& bytes, std::int64_t value) {
    if (value >= 0 && value <= 255) {
        bytes.push_back('K');
        bytes.push_back(static_cast<std::uint8_t>(value));
    } else {
        bytes.push_back('J');
        const std::int32_t narrowed = static_cast<std::int32_t>(value);
        for (int shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<std::uint8_t>(static_cast<std::uint32_t>(narrowed) >> shift));
        }
    }
}

void pickleAppendDouble(std::vector<std::uint8_t>& bytes, double value) {
    bytes.push_back('G'); // BINFLOAT uses big endian IEEE754
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    for (int shift = 56; shift >= 0; shift -= 8) bytes.push_back(static_cast<std::uint8_t>(bits >> shift));
}

void pickleAppendBytes(std::vector<std::uint8_t>& bytes, const std::vector<double>& values) {
    bytes.push_back('B'); // BINBYTES
    const std::uint64_t byteCountWide = static_cast<std::uint64_t>(values.size()) * sizeof(double);
    if (byteCountWide > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("DEM array is too large for the compatibility pickle writer");
    }
    const std::uint32_t byteCount = static_cast<std::uint32_t>(byteCountWide);
    for (int shift = 0; shift < 32; shift += 8) bytes.push_back(static_cast<std::uint8_t>(byteCount >> shift));
    for (double value : values) {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
        for (int shift = 0; shift < 64; shift += 8) bytes.push_back(static_cast<std::uint8_t>(bits >> shift));
    }
}

void pickleAppendNumpyArray(std::vector<std::uint8_t>& bytes,
                            const std::vector<double>& values,
                            const std::vector<std::size_t>& shape) {
    const std::string reconstruct = "numpy.core.multiarray\n_reconstruct\n";
    bytes.push_back('c'); bytes.insert(bytes.end(), reconstruct.begin(), reconstruct.end());
    const std::string ndarray = "numpy\nndarray\n";
    bytes.push_back('c'); bytes.insert(bytes.end(), ndarray.begin(), ndarray.end());
    pickleAppendInt(bytes, 0); bytes.push_back(0x85); // TUPLE1
    bytes.push_back('C'); bytes.push_back(1); bytes.push_back('b'); // SHORT_BINBYTES
    bytes.push_back(0x87); // TUPLE3
    bytes.push_back('R'); // REDUCE
    bytes.push_back('('); // MARK for ndarray state
    pickleAppendInt(bytes, 1);
    if (shape.size() == 1U) {
        pickleAppendInt(bytes, static_cast<std::int64_t>(shape[0])); bytes.push_back(0x85);
    } else {
        bytes.push_back('(');
        for (const std::size_t dimension : shape) {
            pickleAppendInt(bytes, static_cast<std::int64_t>(dimension));
        }
        bytes.push_back('t');
    }
    const std::string dtype = "numpy\ndtype\n";
    bytes.push_back('c'); bytes.insert(bytes.end(), dtype.begin(), dtype.end());
    pickleAppendUnicode(bytes, "f8");
    bytes.push_back(0x89); bytes.push_back(0x88); bytes.push_back(0x87); bytes.push_back('R');
    bytes.push_back('('); pickleAppendInt(bytes, 3); pickleAppendUnicode(bytes, "<");
    bytes.push_back('N'); bytes.push_back('N'); bytes.push_back('N');
    pickleAppendInt(bytes, -1); pickleAppendInt(bytes, -1); pickleAppendInt(bytes, 0);
    bytes.push_back('t'); bytes.push_back('b'); // dtype BUILD
    bytes.push_back(0x89); // not Fortran order
    pickleAppendBytes(bytes, values);
    bytes.push_back('t'); bytes.push_back('b'); // ndarray BUILD
}

void writeDemPickle(const fs::path& path,
                    const std::vector<double>& demGrid,
                    std::size_t ny,
                    std::size_t nx,
                    const std::vector<double>& xGrid,
                    const std::vector<double>& yGrid,
                    double dx,
                    double dy,
                    double baseHeight,
                    const RotationFrame& frame) {
    // Compatibility cache contract consumed by legacy 3_voxelization.py:
    // protocol-3 pickle dict; dem_grid is C-order float64 [ny,nx], x_grid and
    // y_grid are float64 vectors, followed by dx/dy/base_height/rotate_deg,
    // pivot_xy, and absolute rotated-frame x_min/y_min/x_max/y_max scalars.
    std::vector<std::uint8_t> bytes{0x80, 0x03, static_cast<std::uint8_t>('}'),
                                    static_cast<std::uint8_t>('(')};
    auto key = [&](std::string_view name) { pickleAppendUnicode(bytes, name); };
    key("dem_grid"); pickleAppendNumpyArray(bytes, demGrid, {ny, nx});
    key("x_grid"); pickleAppendNumpyArray(bytes, xGrid, {nx});
    key("y_grid"); pickleAppendNumpyArray(bytes, yGrid, {ny});
    key("dx"); pickleAppendDouble(bytes, dx);
    key("dy"); pickleAppendDouble(bytes, dy);
    key("base_height"); pickleAppendDouble(bytes, baseHeight);
    key("rotate_deg"); pickleAppendDouble(bytes, frame.degrees);
    key("pivot_xy"); pickleAppendDouble(bytes, frame.pivot.x);
    pickleAppendDouble(bytes, frame.pivot.y); bytes.push_back(0x86);
    key("x_min"); pickleAppendDouble(bytes, frame.xOrigin);
    key("y_min"); pickleAppendDouble(bytes, frame.yOrigin);
    key("x_max"); pickleAppendDouble(bytes, frame.xOrigin + frame.xRange);
    key("y_max"); pickleAppendDouble(bytes, frame.yOrigin + frame.yRange);
    bytes.push_back('u'); bytes.push_back('.');
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot write DEM cache: " + pathToUtf8(path));
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("Failed while writing DEM cache: " + pathToUtf8(path));
}

bool solveSmallSystem(std::vector<double> matrix,
                      std::vector<double> right,
                      std::size_t dimension,
                      std::vector<double>& solution) {
    solution.assign(dimension, 0.0);
    for (std::size_t pivot = 0; pivot < dimension; ++pivot) {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1U; row < dimension; ++row) {
            if (std::abs(matrix[row * dimension + pivot])
                > std::abs(matrix[best * dimension + pivot])) best = row;
        }
        if (std::abs(matrix[best * dimension + pivot]) <= 1e-18) return false;
        if (best != pivot) {
            for (std::size_t column = pivot; column < dimension; ++column) {
                std::swap(matrix[pivot * dimension + column],
                          matrix[best * dimension + column]);
            }
            std::swap(right[pivot], right[best]);
        }
        const double diagonal = matrix[pivot * dimension + pivot];
        for (std::size_t column = pivot; column < dimension; ++column) {
            matrix[pivot * dimension + column] /= diagonal;
        }
        right[pivot] /= diagonal;
        for (std::size_t row = 0; row < dimension; ++row) {
            if (row == pivot) continue;
            const double factor = matrix[row * dimension + pivot];
            if (factor == 0.0) continue;
            for (std::size_t column = pivot; column < dimension; ++column) {
                matrix[row * dimension + column] -= factor * matrix[pivot * dimension + column];
            }
            right[row] -= factor * right[pivot];
        }
    }
    solution = std::move(right);
    return true;
}

struct MlsWeight {
    std::vector<std::size_t> indices;
    std::vector<float> weights;
};

double idwAtPoint(vtkStaticPointLocator& locator,
                  const std::vector<XY>& points,
                  const std::vector<double>& values,
                  const XY& query,
                  int neighbors) {
    vtkNew<vtkIdList> ids;
    const double query3[3]{query.x, query.y, 0.0};
    locator.FindClosestNPoints(neighbors, query3, ids);
    long double sum = 0.0L;
    long double weightSum = 0.0L;
    for (vtkIdType entry = 0; entry < ids->GetNumberOfIds(); ++entry) {
        const std::size_t id = static_cast<std::size_t>(ids->GetId(entry));
        const double dx = points[id].x - query.x;
        const double dy = points[id].y - query.y;
        const double distance = std::max(std::sqrt(dx * dx + dy * dy), 1e-10);
        const double weight = 1.0 / (distance * distance);
        sum += weight * values[id];
        weightSum += weight;
    }
    return static_cast<double>(sum / weightSum);
}

class TerrainFirstPlan final {
public:
    TerrainFirstPlan(const std::vector<double>& sourceX,
                     const std::vector<double>& sourceY,
                     const std::vector<XY>& demPoints,
                     const std::vector<double>& demElevation,
                     const std::vector<double>& targetX,
                     const std::vector<double>& targetY,
                     std::optional<int> gpuDevice,
                     std::ostream& output) {
        if (sourceX.size() != sourceY.size()) throw std::runtime_error("source X/Y mismatch");
        vtkNew<vtkPoints> sourcePoints;
        std::vector<XY> validPoints;
        for (std::size_t index = 0; index < sourceX.size(); ++index) {
            if (!std::isfinite(sourceX[index]) || !std::isfinite(sourceY[index])) continue;
            sourcePoints->InsertNextPoint(sourceX[index], sourceY[index], 0.0);
            validPoints.push_back({sourceX[index], sourceY[index]});
            validSourceIndex_.push_back(index);
        }
        if (validPoints.empty()) throw std::runtime_error("no valid source points");
        vtkNew<vtkPolyData> sourceData;
        sourceData->SetPoints(sourcePoints);
        vtkNew<vtkStaticPointLocator> sourceLocator;
        sourceLocator->SetDataSet(sourceData);
        sourceLocator->BuildLocator();

        vtkNew<vtkPoints> demVtkPoints;
        for (const XY& point : demPoints) demVtkPoints->InsertNextPoint(point.x, point.y, 0.0);
        if (demPoints.empty()) throw std::runtime_error("DEM points are empty after coordinate transform");
        vtkNew<vtkPolyData> demData;
        demData->SetPoints(demVtkPoints);
        vtkNew<vtkStaticPointLocator> demLocator;
        demLocator->SetDataSet(demData);
        demLocator->BuildLocator();
        const int demNeighbors = static_cast<int>(std::min<std::size_t>(12U, demPoints.size()));
        sourceTerrain_.resize(sourceX.size(), 0.0F);
        std::vector<double> validTerrain(validPoints.size());
        if (gpuDevice) {
            if (demPoints.size()
                > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw OpenClLuwbcError(
                    "luwbc terrain DEM point count exceeds OpenCL uint32 index capacity");
            }
            OpenClLuwbcDemIdwRequest request;
            request.deviceIndex = *gpuDevice;
            request.neighborCount = static_cast<std::size_t>(demNeighbors);
            request.sourceX.reserve(demPoints.size());
            request.sourceY.reserve(demPoints.size());
            for (const XY& point : demPoints) {
                request.sourceX.push_back(point.x);
                request.sourceY.push_back(point.y);
            }
            request.sourceValues = demElevation;
            request.targetX.reserve(validPoints.size());
            request.targetY.reserve(validPoints.size());
            request.neighborIndices.resize(
                checkedProduct({validPoints.size(), request.neighborCount}));
            for (std::size_t index = 0U; index < validPoints.size(); ++index) {
                request.targetX.push_back(validPoints[index].x);
                request.targetY.push_back(validPoints[index].y);
                const double query[3]{validPoints[index].x, validPoints[index].y, 0.0};
                vtkNew<vtkIdList> ids;
                demLocator->FindClosestNPoints(demNeighbors, query, ids);
                for (int entry = 0; entry < demNeighbors; ++entry) {
                    request.neighborIndices[
                        index * request.neighborCount
                        + static_cast<std::size_t>(entry)] =
                        static_cast<std::uint32_t>(ids->GetId(entry));
                }
            }
            OpenClLuwbcDoubleResult gpu = runNativeOpenClLuwbcDemIdw(request);
            if (gpu.values.size() != validTerrain.size()) {
                throw OpenClLuwbcError(
                    "OpenCL terrain DEM IDW returned an inconsistent target count");
            }
            validTerrain = std::move(gpu.values);
            info(output, "OpenCL terrain source-column DEM IDW backend: "
                 + gpu.device + "; targets=" + std::to_string(validTerrain.size())
                 + ", neighbors=" + std::to_string(demNeighbors));
        } else {
            for (std::size_t index = 0; index < validPoints.size(); ++index) {
                validTerrain[index] = idwAtPoint(*demLocator, demPoints, demElevation,
                                                 validPoints[index], demNeighbors);
            }
        }
        const double minimum = *std::min_element(validTerrain.begin(), validTerrain.end());
        for (std::size_t index = 0; index < validPoints.size(); ++index) {
            sourceTerrain_[validSourceIndex_[index]] =
                static_cast<float>(std::max(0.0, validTerrain[index] - minimum));
        }

        const int neighbors = static_cast<int>(std::min<std::size_t>(24U, validPoints.size()));
        const std::size_t basisDimension = neighbors >= 6 ? 6U : (neighbors >= 3 ? 3U : 1U);
        weights_.resize(targetX.size() * targetY.size());
        if (gpuDevice) {
            if (validPoints.size()
                > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw OpenClLuwbcError(
                    "luwbc terrain source point count exceeds OpenCL uint32 index capacity");
            }
            OpenClLuwbcMlsRequest request;
            request.deviceIndex = *gpuDevice;
            request.neighborCount = static_cast<std::size_t>(neighbors);
            request.sourceX.reserve(validPoints.size());
            request.sourceY.reserve(validPoints.size());
            for (const XY& point : validPoints) {
                request.sourceX.push_back(point.x);
                request.sourceY.push_back(point.y);
            }
            request.targetX.resize(weights_.size());
            request.targetY.resize(weights_.size());
            request.neighborIndices.resize(
                checkedProduct({weights_.size(), request.neighborCount}));
#pragma omp parallel for if(weights_.size() > 256)
            for (std::int64_t targetSigned = 0;
                 targetSigned < static_cast<std::int64_t>(weights_.size());
                 ++targetSigned) {
                const std::size_t target = static_cast<std::size_t>(targetSigned);
                const std::size_t yIndex = target / targetX.size();
                const std::size_t xIndex = target % targetX.size();
                const XY query{targetX[xIndex], targetY[yIndex]};
                request.targetX[target] = query.x;
                request.targetY[target] = query.y;
                const double query3[3]{query.x, query.y, 0.0};
                vtkNew<vtkIdList> ids;
                sourceLocator->FindClosestNPoints(neighbors, query3, ids);
                MlsWeight item;
                item.indices.resize(static_cast<std::size_t>(neighbors));
                item.weights.resize(static_cast<std::size_t>(neighbors));
                for (int entry = 0; entry < neighbors; ++entry) {
                    const std::size_t local =
                        static_cast<std::size_t>(ids->GetId(entry));
                    item.indices[static_cast<std::size_t>(entry)] =
                        validSourceIndex_[local];
                    request.neighborIndices[
                        target * request.neighborCount
                        + static_cast<std::size_t>(entry)] =
                        static_cast<std::uint32_t>(local);
                }
                weights_[target] = std::move(item);
            }
            OpenClLuwbcResult gpu = runNativeOpenClLuwbcMlsWeights(request);
            if (gpu.values.size()
                != checkedProduct({weights_.size(), request.neighborCount})) {
                throw OpenClLuwbcError(
                    "OpenCL terrain MLS returned an inconsistent weight count");
            }
            for (std::size_t target = 0U; target < weights_.size(); ++target) {
                std::copy_n(
                    gpu.values.begin()
                        + static_cast<std::ptrdiff_t>(
                            target * request.neighborCount),
                    static_cast<std::ptrdiff_t>(request.neighborCount),
                    weights_[target].weights.begin());
            }
            info(output, "OpenCL terrain-first MLS weight backend: " + gpu.device
                 + "; targets=" + std::to_string(weights_.size())
                 + ", neighbors=" + std::to_string(neighbors)
                 + ", basis=" + std::to_string(basisDimension));
        } else {
#pragma omp parallel for if(weights_.size() > 256)
            for (std::int64_t targetSigned = 0;
                 targetSigned < static_cast<std::int64_t>(weights_.size()); ++targetSigned) {
                const std::size_t target = static_cast<std::size_t>(targetSigned);
                const std::size_t yIndex = target / targetX.size();
                const std::size_t xIndex = target % targetX.size();
                const XY query{targetX[xIndex], targetY[yIndex]};
                const double query3[3]{query.x, query.y, 0.0};
                vtkNew<vtkIdList> ids;
                sourceLocator->FindClosestNPoints(neighbors, query3, ids);
                MlsWeight item;
                item.indices.resize(static_cast<std::size_t>(neighbors));
                item.weights.resize(static_cast<std::size_t>(neighbors));
                std::vector<double> distance(static_cast<std::size_t>(neighbors));
                double h = 0.0;
                for (int entry = 0; entry < neighbors; ++entry) {
                    const std::size_t local = static_cast<std::size_t>(ids->GetId(entry));
                    item.indices[static_cast<std::size_t>(entry)] = validSourceIndex_[local];
                    const double dx = validPoints[local].x - query.x;
                    const double dy = validPoints[local].y - query.y;
                    distance[static_cast<std::size_t>(entry)] = std::sqrt(dx * dx + dy * dy);
                    h = std::max(h, distance[static_cast<std::size_t>(entry)]);
                }
            if (!std::isfinite(h) || h <= 1e-12) h = 1.0;
            h *= 1.000001;
            std::vector<double> kernel(static_cast<std::size_t>(neighbors));
            std::vector<std::vector<double>> basis(static_cast<std::size_t>(neighbors),
                                                    std::vector<double>(basisDimension, 0.0));
            double kernelSum = 0.0;
            for (int entry = 0; entry < neighbors; ++entry) {
                const std::size_t id = static_cast<std::size_t>(ids->GetId(entry));
                const double xn = (validPoints[id].x - query.x) / h;
                const double yn = (validPoints[id].y - query.y) / h;
                const double radius = std::sqrt(xn * xn + yn * yn);
                const double compact = std::max(0.0, 1.0 - radius);
                kernel[static_cast<std::size_t>(entry)] = std::pow(compact, 4.0)
                    * (4.0 * radius + 1.0);
                kernelSum += kernel[static_cast<std::size_t>(entry)];
                basis[static_cast<std::size_t>(entry)][0] = 1.0;
                if (basisDimension >= 3U) {
                    basis[static_cast<std::size_t>(entry)][1] = xn;
                    basis[static_cast<std::size_t>(entry)][2] = yn;
                }
                if (basisDimension == 6U) {
                    basis[static_cast<std::size_t>(entry)][3] = xn * xn;
                    basis[static_cast<std::size_t>(entry)][4] = xn * yn;
                    basis[static_cast<std::size_t>(entry)][5] = yn * yn;
                }
            }
            if (!std::isfinite(kernelSum) || kernelSum <= 1e-14) {
                std::fill(kernel.begin(), kernel.end(), 1.0);
                kernelSum = static_cast<double>(neighbors);
            }
            std::vector<double> coefficients(static_cast<std::size_t>(neighbors));
            if (basisDimension == 1U) {
                for (int entry = 0; entry < neighbors; ++entry) {
                    coefficients[static_cast<std::size_t>(entry)] =
                        kernel[static_cast<std::size_t>(entry)] / kernelSum;
                }
            } else {
                std::vector<double> matrix(basisDimension * basisDimension, 0.0);
                for (int entry = 0; entry < neighbors; ++entry) {
                    const std::size_t e = static_cast<std::size_t>(entry);
                    for (std::size_t row = 0; row < basisDimension; ++row) {
                        for (std::size_t column = 0; column < basisDimension; ++column) {
                            matrix[row * basisDimension + column] +=
                                kernel[e] * basis[e][row] * basis[e][column];
                        }
                    }
                }
                double trace = 0.0;
                for (std::size_t diagonal = 0; diagonal < basisDimension; ++diagonal) {
                    trace += matrix[diagonal * basisDimension + diagonal];
                }
                const double regularization = 1e-10 * trace / static_cast<double>(basisDimension) + 1e-12;
                for (std::size_t diagonal = 0; diagonal < basisDimension; ++diagonal) {
                    matrix[diagonal * basisDimension + diagonal] += regularization;
                }
                std::vector<double> right(basisDimension, 0.0);
                right[0] = 1.0;
                std::vector<double> solved;
                if (solveSmallSystem(matrix, right, basisDimension, solved)) {
                    double sum = 0.0;
                    for (int entry = 0; entry < neighbors; ++entry) {
                        const std::size_t e = static_cast<std::size_t>(entry);
                        coefficients[e] = kernel[e] * std::inner_product(
                            basis[e].begin(), basis[e].end(), solved.begin(), 0.0);
                        sum += coefficients[e];
                    }
                    if (!std::isfinite(sum) || std::abs(sum) <= 1e-14) {
                        for (std::size_t e = 0; e < coefficients.size(); ++e) {
                            coefficients[e] = kernel[e] / kernelSum;
                        }
                    } else {
                        for (double& value : coefficients) value /= sum;
                    }
                    double negativeMass = 0.0;
                    for (const double value : coefficients) if (value < 0.0) negativeMass += std::abs(value);
                    if (negativeMass > 0.08) {
                        const double alpha = std::min(1.0, (negativeMass - 0.08) / 0.30);
                        double limitedSum = 0.0;
                        for (std::size_t e = 0; e < coefficients.size(); ++e) {
                            coefficients[e] = (1.0 - alpha) * coefficients[e]
                                + alpha * kernel[e] / kernelSum;
                            limitedSum += coefficients[e];
                        }
                        for (double& value : coefficients) value /= limitedSum;
                    }
                } else {
                    for (std::size_t e = 0; e < coefficients.size(); ++e) {
                        coefficients[e] = kernel[e] / kernelSum;
                    }
                }
            }
            for (std::size_t entry = 0; entry < coefficients.size(); ++entry) {
                item.weights[entry] = static_cast<float>(coefficients[entry]);
            }
            weights_[target] = std::move(item);
        }
        }
        targetNx_ = targetX.size();
        validSourceIndex_.clear();
        validSourceIndex_.shrink_to_fit();
    }

    [[nodiscard]] std::array<double, 4> sampleBatch(
        const std::array<const Field3D*, 4>& fields,
        std::size_t fieldCount,
        std::size_t targetY,
        std::size_t targetX,
        double absoluteZ,
        double baseHeight,
        double elevationScale,
        const std::vector<double>& zAgl) const {
        if (fieldCount < 3U || fieldCount > fields.size()
            || fields.front() == nullptr) {
            throw std::runtime_error(
                "Terrain-first boundary batch received no fields");
        }
        const Field3D& first = *fields.front();
        for (std::size_t component = 0U; component < fieldCount; ++component) {
            const Field3D* field = fields[component];
            if (field == nullptr || field->nz != first.nz || field->ny != first.ny
                || field->nx != first.nx) {
                throw std::runtime_error(
                    "Terrain-first boundary batch fields have mismatched dimensions");
            }
        }
        const MlsWeight& item = weights_.at(targetY * targetNx_ + targetX);
        std::array<double, 4> result{};
        for (std::size_t entry = 0; entry < item.indices.size(); ++entry) {
            const std::size_t sourcePoint = item.indices[entry];
            const std::size_t sourceY = sourcePoint / first.nx;
            const std::size_t sourceX = sourcePoint % first.nx;
            const double ground = baseHeight
                + static_cast<double>(sourceTerrain_[sourcePoint]) * elevationScale;
            const double queryAgl = std::clamp(absoluteZ - ground, zAgl.front(), zAgl.back());
            const VerticalWeight vertical = verticalWeight(zAgl, queryAgl);
            for (std::size_t component = 0U; component < fieldCount; ++component) {
                result[component] += static_cast<double>(item.weights[entry])
                    * verticalLinearSample(*fields[component], sourceY, sourceX,
                                           vertical);
            }
        }
        return result;
    }

    [[nodiscard]] std::size_t neighborCount() const {
        return weights_.empty() ? 0U : weights_.front().indices.size();
    }

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        std::size_t bytes = validSourceIndex_.capacity() * sizeof(std::size_t)
            + sourceTerrain_.capacity() * sizeof(float)
            + weights_.capacity() * sizeof(MlsWeight);
        for (const MlsWeight& weight : weights_) {
            bytes += weight.indices.capacity() * sizeof(std::size_t)
                + weight.weights.capacity() * sizeof(float);
        }
        return bytes;
    }

private:
    std::vector<std::size_t> validSourceIndex_;
    std::vector<float> sourceTerrain_;
    std::vector<MlsWeight> weights_;
    std::size_t targetNx_ = 0U;
};

struct TerrainFirstPlanKey {
    std::string scope;
    int backendDevice = -1;
    std::vector<std::uint64_t> sourceX;
    std::vector<std::uint64_t> sourceY;
    std::vector<std::uint64_t> demPoints;
    std::vector<std::uint64_t> demElevation;
    std::vector<std::uint64_t> targetX;
    std::vector<std::uint64_t> targetY;

    [[nodiscard]] bool operator==(const TerrainFirstPlanKey&) const = default;

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return scope.capacity()
            + (sourceX.capacity() + sourceY.capacity() + demPoints.capacity()
               + demElevation.capacity() + targetX.capacity() + targetY.capacity())
                * sizeof(std::uint64_t);
    }
};

using TerrainMappingCache =
    BoundedExactPlanCache<TerrainFirstPlanKey, TerrainFirstPlan>;

TerrainMappingCache& terrainMappingCache() {
    static TerrainMappingCache cache(2U, 256U * 1024U * 1024U);
    return cache;
}

TerrainMappingCache::Lookup acquireTerrainFirstPlan(
    const std::string& scope,
    const std::vector<double>& sourceX,
    const std::vector<double>& sourceY,
    const std::vector<XY>& demPoints,
    const std::vector<double>& demElevation,
    const std::vector<double>& targetX,
    const std::vector<double>& targetY,
    std::optional<int> gpuDevice,
    std::ostream& output) {
    TerrainFirstPlanKey key;
    key.scope = scope;
    key.backendDevice = gpuDevice.value_or(-1);
    key.sourceX = exactDoubleBits(sourceX);
    key.sourceY = exactDoubleBits(sourceY);
    if (demPoints.size() > std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::runtime_error("Terrain point cache key size overflow");
    }
    key.demPoints.reserve(demPoints.size() * 2U);
    for (const XY& point : demPoints) {
        key.demPoints.push_back(std::bit_cast<std::uint64_t>(point.x));
        key.demPoints.push_back(std::bit_cast<std::uint64_t>(point.y));
    }
    key.demElevation = exactDoubleBits(demElevation);
    key.targetX = exactDoubleBits(targetX);
    key.targetY = exactDoubleBits(targetY);
    return terrainMappingCache().acquire(key, [&]() {
        return TerrainFirstPlan(sourceX, sourceY, demPoints, demElevation,
                                targetX, targetY, gpuDevice, output);
    });
}

class VerticalLinearPlan final {
public:
    VerticalLinearPlan(const std::vector<double>& sourceZ,
                       const std::vector<double>& targetZ)
        : sourceLevelCount_(sourceZ.size()) {
        if (sourceZ.empty()) {
            throw std::runtime_error(
                "Cannot build a vertical interpolation plan without source levels");
        }
        weights_.reserve(targetZ.size());
        for (const double target : targetZ) {
            weights_.push_back(verticalWeight(sourceZ, target));
        }
    }

    [[nodiscard]] std::vector<Field3D> applyBatch(
        const std::vector<const Field3D*>& fields) const {
        if (fields.empty() || fields.front() == nullptr) {
            throw std::runtime_error(
                "Vertical interpolation batch received no fields");
        }
        const Field3D& first = *fields.front();
        if (first.nz != sourceLevelCount_) {
            throw std::runtime_error(
                "Vertical coordinate length does not match source data");
        }
        std::vector<Field3D> result;
        result.reserve(fields.size());
        for (const Field3D* field : fields) {
            if (field == nullptr || field->nz != first.nz || field->ny != first.ny
                || field->nx != first.nx) {
                throw std::runtime_error(
                    "Vertical interpolation batch fields have mismatched dimensions");
            }
            result.emplace_back(weights_.size(), first.ny, first.nx);
        }
        const std::size_t pointCount = first.nx * first.ny;
#pragma omp parallel for if(pointCount > 64)
        for (std::int64_t pointSigned = 0;
             pointSigned < static_cast<std::int64_t>(pointCount); ++pointSigned) {
            const std::size_t point = static_cast<std::size_t>(pointSigned);
            const std::size_t y = point / first.nx;
            const std::size_t x = point % first.nx;
            for (std::size_t z = 0U; z < weights_.size(); ++z) {
                for (std::size_t component = 0U; component < fields.size();
                     ++component) {
                    result[component].at(z, y, x) = verticalLinearSample(
                        *fields[component], y, x, weights_[z]);
                }
            }
        }
        return result;
    }

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return weights_.capacity() * sizeof(VerticalWeight);
    }

private:
    std::size_t sourceLevelCount_ = 0U;
    std::vector<VerticalWeight> weights_;
};

struct VerticalPlanKey {
    std::string scope;
    std::vector<std::uint64_t> sourceZ;
    std::vector<std::uint64_t> targetZ;

    [[nodiscard]] bool operator==(const VerticalPlanKey&) const = default;

    [[nodiscard]] std::size_t memoryBytes() const noexcept {
        return scope.capacity()
            + (sourceZ.capacity() + targetZ.capacity()) * sizeof(std::uint64_t);
    }
};

using VerticalMappingCache =
    BoundedExactPlanCache<VerticalPlanKey, VerticalLinearPlan>;

VerticalMappingCache& verticalMappingCache() {
    static VerticalMappingCache cache(8U, 8U * 1024U * 1024U);
    return cache;
}

VerticalMappingCache::Lookup acquireVerticalPlan(
    const std::string& scope,
    const std::vector<double>& sourceZ,
    const std::vector<double>& targetZ) {
    VerticalPlanKey key;
    key.scope = scope;
    key.sourceZ = exactDoubleBits(sourceZ);
    key.targetZ = exactDoubleBits(targetZ);
    return verticalMappingCache().acquire(key, [&]() {
        return VerticalLinearPlan(sourceZ, targetZ);
    });
}

using VerticalFieldDomain = std::vector<const Field3D*>;

std::vector<std::vector<Field3D>> resampleVerticalLinearOpenClBatched(
    const std::vector<VerticalFieldDomain>& domains,
    const std::vector<double>& sourceZ,
    const std::vector<double>& targetZ,
    int gpuDevice,
    std::ostream& output) {
    if (domains.empty()) {
        throw OpenClLuwbcError(
            "luwbc fused vertical OpenCL path received no point domains");
    }
    const std::size_t componentCount = domains.front().size();
    if (componentCount != 3U && componentCount != 4U) {
        throw OpenClLuwbcError(
            "luwbc fused vertical OpenCL path requires U/V/W[/T]");
    }
    std::vector<std::size_t> pointCounts(domains.size());
    std::vector<std::vector<Field3D>> converted(domains.size());
    for (std::size_t domain = 0U; domain < domains.size(); ++domain) {
        if (domains[domain].size() != componentCount
            || domains[domain].front() == nullptr) {
            throw OpenClLuwbcError(
                "luwbc fused vertical OpenCL domains have mismatched fields");
        }
        const Field3D& first = *domains[domain].front();
        if (first.nz != sourceZ.size()) {
            throw OpenClLuwbcError(
                "luwbc fused vertical OpenCL source level mismatch");
        }
        pointCounts[domain] = first.nx * first.ny;
        converted[domain].reserve(componentCount);
        for (const Field3D* field : domains[domain]) {
            if (field == nullptr || field->nz != first.nz || field->ny != first.ny
                || field->nx != first.nx) {
                throw OpenClLuwbcError(
                    "luwbc fused vertical OpenCL fields have mismatched dimensions");
            }
            converted[domain].emplace_back(targetZ.size(), first.ny, first.nx);
        }
    }

    constexpr std::size_t kPackedHostBudget = 256U * 1024U * 1024U;
    const auto packedBytes = [&](std::size_t points) {
        const std::size_t planes = componentCount * (sourceZ.size() + targetZ.size());
        if (planes != 0U
            && points > std::numeric_limits<std::size_t>::max() / planes
                / sizeof(float)) {
            throw OpenClLuwbcError(
                "luwbc vertical host transfer batch size overflow");
        }
        return planes * points * sizeof(float);
    };

    for (std::size_t batchBegin = 0U; batchBegin < domains.size();) {
        std::size_t batchEnd = batchBegin;
        std::size_t totalPoints = 0U;
        while (batchEnd < domains.size()) {
            if (pointCounts[batchEnd]
                > std::numeric_limits<std::size_t>::max() - totalPoints) {
                throw OpenClLuwbcError(
                    "luwbc vertical point-domain batch size overflow");
            }
            const std::size_t candidate = totalPoints + pointCounts[batchEnd];
            if (batchEnd != batchBegin && packedBytes(candidate) > kPackedHostBudget) break;
            totalPoints = candidate;
            ++batchEnd;
        }

        OpenClLuwbcVerticalRequest request;
        request.deviceIndex = gpuDevice;
        request.pointCount = totalPoints;
        request.componentCount = componentCount;
        request.sourceLevels = sourceZ;
        request.targetLevels = targetZ;
        request.values.resize(
            checkedProduct({componentCount, sourceZ.size(), totalPoints}));
        std::size_t pointPrefix = 0U;
        for (std::size_t domain = batchBegin; domain < batchEnd; ++domain) {
            const std::size_t domainPoints = pointCounts[domain];
            for (std::size_t component = 0U; component < componentCount;
                 ++component) {
                for (std::size_t level = 0U; level < sourceZ.size(); ++level) {
                    const auto source = domains[domain][component]->values.begin()
                        + static_cast<std::ptrdiff_t>(level * domainPoints);
                    auto destination = request.values.begin()
                        + static_cast<std::ptrdiff_t>(
                            (component * sourceZ.size() + level) * totalPoints
                            + pointPrefix);
                    std::copy_n(source, static_cast<std::ptrdiff_t>(domainPoints),
                                destination);
                }
            }
            pointPrefix += domainPoints;
        }

        const OpenClLuwbcResult result = runNativeOpenClLuwbcVertical(request);
        output << "[INFO] OpenCL fused vertical resampling backend: "
               << result.device << '\n';
        output << "[INFO] OpenCL vertical transfer batch: "
               << (batchEnd - batchBegin) << " point domain(s), "
               << totalPoints
               << " points; bounded backend chunks with a pipeline-ready packed layout\n";

        pointPrefix = 0U;
        for (std::size_t domain = batchBegin; domain < batchEnd; ++domain) {
            const std::size_t domainPoints = pointCounts[domain];
            for (std::size_t component = 0U; component < componentCount;
                 ++component) {
                for (std::size_t level = 0U; level < targetZ.size(); ++level) {
                    const auto source = result.values.begin()
                        + static_cast<std::ptrdiff_t>(
                            (component * targetZ.size() + level) * totalPoints
                            + pointPrefix);
                    auto destination = converted[domain][component].values.begin()
                        + static_cast<std::ptrdiff_t>(level * domainPoints);
                    std::copy_n(source, static_cast<std::ptrdiff_t>(domainPoints),
                                destination);
                }
            }
            pointPrefix += domainPoints;
        }
        batchBegin = batchEnd;
    }
    return converted;
}

void writeVti(const fs::path& path,
              const Field3D& u,
              const Field3D& v,
              const Field3D& w,
              double dx,
              double dy,
              double dz) {
    vtkNew<vtkImageData> grid;
    grid->SetOrigin(0.0, 0.0, 0.0);
    grid->SetSpacing(dx, dy, dz);
    grid->SetDimensions(static_cast<int>(u.nx), static_cast<int>(u.ny),
                        static_cast<int>(u.nz));
    auto add = [&](const Field3D& field, const char* name) {
        vtkNew<vtkFloatArray> array;
        array->SetName(name);
        array->SetNumberOfComponents(1);
        array->SetNumberOfTuples(static_cast<vtkIdType>(field.values.size()));
        std::copy(field.values.begin(), field.values.end(), array->GetPointer(0));
        grid->GetPointData()->AddArray(array);
    };
    add(u, "u"); add(v, "v"); add(w, "w");
    grid->GetPointData()->SetActiveScalars("u");
    vtkNew<vtkXMLImageDataWriter> writer;
    writer->SetFileName(pathToUtf8(path).c_str());
    writer->SetInputData(grid);
    writer->SetDataModeToBinary();
    if (writer->Write() == 0) {
        throw std::runtime_error("VTK writer failed for " + pathToUtf8(path));
    }
}

struct InputSelection {
    fs::path deckPath;
    fs::path project;
    fs::path netcdfPath;
    DeckDocument deck;
    std::string caseName;
    std::string dateTime;
};

InputSelection selectInput(const fs::path& requestedDeck, std::ostream& output) {
    const fs::path deckPath = absoluteNormalized(requestedDeck);
    std::error_code error;
    if (!fs::is_regular_file(deckPath, error) || error) {
        errorLine(output, "conf file not found: " + pathToUtf8(deckPath));
        throw std::runtime_error(pathToUtf8(deckPath));
    }
    DeckDocument deck = DeckDocument::load(pathToQString(deckPath));
    const QStringList duplicates = deck.duplicateKeys();
    if (!duplicates.isEmpty()) {
        warn(output, "Duplicate deck keys found; keep the last value and rewrite canonically: "
             + duplicates.join(QStringLiteral(", ")).toStdString());
        deck.save();
    }
    const QString caseNameText = deck.textOr(QStringLiteral("casename"));
    if (caseNameText.trimmed().isEmpty()) {
        errorLine(output, "casename not found in conf");
        throw std::runtime_error("casename missing in conf");
    }
    std::string dateTime = deck.textOr(QStringLiteral("datetime")).trimmed().toStdString();
    if (!std::regex_match(dateTime, std::regex("[0-9]{14}"))) {
        dateTime = "20990101120000";
        warn(output, "datetime not provided in conf, use 20990101120000 and write it back");
        deck.setText(QStringLiteral("datetime"), QString::fromStdString(dateTime));
        deck.save();
    }
    const fs::path project = deckPath.parent_path();
    const fs::path wind = project / "wind_bc";
    const std::string caseName = caseNameText.toStdString();
    const fs::path expected = wind / (caseName + "_" + dateTime + ".nc");
    if (fs::is_regular_file(expected)) {
        return {deckPath, project, expected, std::move(deck), caseName, dateTime};
    }
    if (!fs::is_directory(wind)) {
        errorLine(output, "wind_bc folder not found under " + pathToUtf8(project));
        throw std::runtime_error(pathToUtf8(expected));
    }
    std::vector<fs::path> candidates;
    for (fs::directory_iterator iterator(wind, error), end;
         !error && iterator != end; iterator.increment(error)) {
        std::error_code typeError;
        if (iterator->is_regular_file(typeError) && !typeError
            && lowerAscii(pathToUtf8(iterator->path().extension())) == ".nc") {
            candidates.push_back(iterator->path());
        }
    }
    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() == 1U) {
        warn(output, "Expected " + pathToUtf8(expected.filename()) + " not found. Use the only nc file "
             + pathToUtf8(candidates.front().filename()) + " under wind_bc.");
        return {deckPath, project, candidates.front(), std::move(deck), caseName, dateTime};
    }
    if (candidates.empty()) {
        errorLine(output, "No nc file found under " + pathToUtf8(wind) + ". Expected "
                  + pathToUtf8(expected.filename()) + ".");
    } else {
        errorLine(output, "Expected " + pathToUtf8(expected.filename())
                  + " not found and multiple nc files present under " + pathToUtf8(wind) + ".");
    }
    throw std::runtime_error(pathToUtf8(expected));
}

struct BuildOptions {
    fs::path deck;
    double elevationScale = 1.0;
    bool writeVtk = false;
};

std::optional<BuildOptions> parseOptions(const CommandArguments& arguments,
                                         std::ostream& output,
                                         std::ostream& error) {
    if (arguments.size() == 1U && (arguments.front() == "--help" || arguments.front() == "-h")) {
        output << "usage: luwbc [-h] [--elevation-scale ELEVATION_SCALE] [--write-vtk] conf_file\n\n"
               << "Build boundary conditions with DEM terrain support\n\n"
               << "positional arguments:\n"
               << "  conf_file             Path to configuration file (conf.luw)\n\n"
               << "options:\n"
               << "  -h, --help            show this help message and exit\n"
               << "  --elevation-scale ELEVATION_SCALE\n"
               << "                        Scale factor for elevation differences (for visualization/testing). Default: 1.0\n"
               << "  --write-vtk           Enable VTK (*.vti) output. Disabled by default.\n";
        return std::nullopt;
    }
    BuildOptions options;
    bool hasDeck = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "--write-vtk") {
            options.writeVtk = true;
        } else if (argument == "--elevation-scale") {
            if (++index >= arguments.size()) {
                error << "luwbc: error: argument --elevation-scale: expected one argument\n";
                return std::nullopt;
            }
            try {
                std::size_t consumed = 0U;
                options.elevationScale = std::stod(arguments[index], &consumed);
                if (consumed != arguments[index].size()) throw std::invalid_argument("trailing text");
            } catch (...) {
                error << "luwbc: error: argument --elevation-scale: invalid float value: '"
                      << arguments[index] << "'\n";
                return std::nullopt;
            }
        } else if (!argument.empty() && argument.front() == '-') {
            error << "luwbc: error: unrecognized arguments: " << argument << '\n';
            return std::nullopt;
        } else if (!hasDeck) {
            options.deck = pathFromUtf8(argument);
            hasDeck = true;
        } else {
            error << "luwbc: error: unrecognized arguments: " << argument << '\n';
            return std::nullopt;
        }
    }
    if (!hasDeck) {
        error << "luwbc: error: the following arguments are required: conf_file\n";
        return std::nullopt;
    }
    return options;
}

void buildBoundaryConditions(const BuildOptions& options,
                             std::ostream& output) {
    InputSelection input = selectInput(options.deck, output);
    const std::string mappingCacheScope =
        pathToUtf8(absoluteNormalized(input.deckPath));
    const PreprocessComputeConfig computePolicy =
        resolvePreprocessComputeConfig(&input.deck).first;
    info(output, "Preprocessing compute policy: "
         + preprocessComputePolicyText(computePolicy).toStdString());
    if (computePolicy.useGpu) {
        if (nativeOpenClLuwbcRequiresWarmup(computePolicy.gpuDevice)) {
            info(output, "OpenCL luwbc kernels are compiling during GPU backend "
                         "preflight. This one-time warmup may take a little longer.");
        }
        const OpenClLuwbcBackendInfo backend =
            requireNativeOpenClLuwbcBackend(computePolicy.gpuDevice);
        if (backend.compiledThisCall) {
            info(output, "OpenCL luwbc kernel preflight compilation finished.");
        }
        info(output, "OpenCL luwbc backend preflight (GPU device "
             + std::to_string(computePolicy.gpuDevice) + "): " + backend.device);
    } else {
        info(output, "Boundary-condition numerical resampling backend: CPU");
    }
    ProgressEmitter progress(QStringLiteral("Generating Boundary Conditions"));
    progress.stage(QStringLiteral("load_wind"),
                   QStringLiteral("Loading wind climate: %1")
                       .arg(QString::fromStdString(pathToUtf8(input.netcdfPath.filename()))));
    info(output, "Open NetCDF: " + pathToUtf8(input.netcdfPath));

    unsigned int workerCount = std::max(1U, std::thread::hardware_concurrency());
#ifdef _WIN32
    char* configuredBuffer = nullptr;
    std::size_t configuredLength = 0U;
    _dupenv_s(&configuredBuffer, &configuredLength, "BC_WORKERS");
    std::unique_ptr<char, decltype(&std::free)> configuredOwner(configuredBuffer, &std::free);
    const char* configured = configuredOwner.get();
#else
    const char* configured = std::getenv("BC_WORKERS");
#endif
    if (configured != nullptr && *configured != '\0') {
        try { workerCount = std::min<unsigned int>(160U, static_cast<unsigned int>(std::stoul(configured))); }
        catch (...) { workerCount = std::min(160U, workerCount); }
    } else {
        workerCount = std::min(160U, workerCount);
    }
    workerCount = std::max(1U, workerCount);
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(workerCount));
#endif
    info(output, "Set parallel workers: " + std::to_string(workerCount));

    const std::optional<Bounds> requestedBounds = targetBounds(input.deck);
    if (!requestedBounds) {
        output << "[auto_UTM] cut_lon_manual or cut_lat_manual not found in raw text, use default EPSG:32651\n";
    }
    const int utmEpsg = utmEpsgForBounds(requestedBounds);
    const std::string utmCrs = "EPSG:" + std::to_string(utmEpsg);
    const CoordinateProjector projector(utmEpsg);

    progress.stage(QStringLiteral("load_dem"),
                   QStringLiteral("Loading terrain DEM and boundary context"));
    const std::optional<DemData> dem = loadDem(input.project, utmEpsg,
                                                input.caseName, requestedBounds,
                                                output);
    if (!requestedBounds) {
        output << "[auto_UTM] cut_lon_manual or cut_lat_manual not found in raw text, use default EPSG:32651\n";
    }

    NcDataset dataset(input.netcdfPath);
    const std::size_t timeCount = dataset.dimensionLength("Time")
        .value_or(dataset.dimensionLength("time").value_or(1U));
    if (timeCount > 1U) {
        info(output, "Input NetCDF contains " + std::to_string(timeCount)
             + " time steps; the existing SurfData output contract selects Time index 0. "
               "Spatial and vertical plans remain reusable across sequential timestamp runs.");
    }
    const std::optional<TemperatureVariable> temperatureVariable =
        resolveTemperature(dataset, output);
    if (!temperatureVariable) {
        info(output, "No temperature field found in input NC; CSV will keep columns X,Y,Z,u,v,w");
    } else if (temperatureVariable->celsius) {
        info(output, "Detected temperature field " + temperatureVariable->name
             + " (Celsius); will convert to Kelvin for CSV");
    } else {
        info(output, "Detected temperature field " + temperatureVariable->name + " (Kelvin)");
    }

    std::vector<std::string> staggered;
    for (const char* name : {"west_east_stag", "south_north_stag", "bottom_top_stag"}) {
        if (dataset.dimensionLength(name)) staggered.emplace_back(name);
    }
    if (!staggered.empty()) {
        std::ostringstream message;
        message << "Detected WRF staggered dimensions in input: ";
        for (std::size_t index = 0; index < staggered.size(); ++index) {
            if (index > 0U) message << ", ";
            message << staggered[index];
        }
        info(output, message.str());
    }
    std::vector<std::pair<std::string, std::string>> renames;
    if (dataset.dimensionLength("west_east") && !dataset.dimensionLength("lon"))
        renames.emplace_back("west_east", "lon");
    if (dataset.dimensionLength("south_north") && !dataset.dimensionLength("lat"))
        renames.emplace_back("south_north", "lat");
    if (dataset.dimensionLength("west_east_stag") && !dataset.dimensionLength("lon_stag"))
        renames.emplace_back("west_east_stag", "lon_stag");
    if (dataset.dimensionLength("south_north_stag") && !dataset.dimensionLength("lat_stag"))
        renames.emplace_back("south_north_stag", "lat_stag");
    if (!renames.empty()) {
        std::ostringstream message;
        message << "Renamed WRF dims to internal names: {";
        for (std::size_t index = 0; index < renames.size(); ++index) {
            if (index > 0U) message << ", ";
            message << '\'' << renames[index].first << "': '" << renames[index].second << '\'';
        }
        message << '}';
        info(output, message.str());
    }

    const auto massNx = firstDimensionLength(dataset, {"lon", "west_east"});
    const auto massNy = firstDimensionLength(dataset, {"lat", "south_north"});
    if (!massNx || !massNy || *massNx == 0U || *massNy == 0U) {
        throw std::runtime_error("Cannot find horizontal lon/lat or WRF mass dimensions");
    }
    const GeoGrid fullGeo = readGeoGrid(dataset, *massNy, *massNx, output);
    if (requestedBounds) confirmCoverage("Wind NC", *requestedBounds, geoBounds(fullGeo), output);
    const ProjectedGridCache::Lookup fullProjection = acquireProjectedGrid(
        mappingCacheScope, fullGeo, projector);
    reportPlanCache(output, "PROJ wind-grid", fullProjection.hit,
                    fullProjection.retained, fullProjection.evicted);
    progress.stage(QStringLiteral("crop_wind"),
                   QStringLiteral("Cropping wind field to target domain"));
    const Crop crop = cropByDeck(fullGeo, input.deck, projector,
                                 fullProjection.plan->points, output);
    GeoGrid geo = cropGeoGrid(fullGeo, crop);
    std::vector<XY> projectedGeo = cropProjectedGrid(
        fullProjection.plan->points, fullGeo.nx, crop);
    VerticalInfo vertical = detectVertical(dataset, crop, output);

    const std::string uName = resolveWindVariable(dataset, "u", {"u", "U"}, "u", output);
    const std::string vName = resolveWindVariable(dataset, "v", {"v", "V"}, "v", output);
    Field3D u = readField3D(dataset, uName, vertical.dimension, crop, false, output);
    Field3D v = readField3D(dataset, vName, vertical.dimension, crop, false, output);
    Field3D w;
    if (dataset.has("w")) {
        w = readField3D(dataset, "w", vertical.dimension, crop, false, output);
    } else if (dataset.has("W")) {
        warn(output, "Wind variable w not found, use W for w");
        w = readField3D(dataset, "W", vertical.dimension, crop, false, output);
    } else {
        w = Field3D(u.nz, u.ny, u.nx);
        std::fill(w.values.begin(), w.values.end(), 0.0F);
    }
    std::optional<Field3D> temperature;
    if (temperatureVariable) {
        temperature = readField3D(dataset, temperatureVariable->name, vertical.dimension,
                                  crop, true, output);
        if (temperatureVariable->celsius) {
            for (float& value : temperature->values) value += 273.15F;
        }
    }
    ensureFieldShape(v, u.nz, u.ny, u.nx, vName, output);
    ensureFieldShape(w, u.nz, u.ny, u.nx, "w", output);
    if (temperature) ensureFieldShape(*temperature, u.nz, u.ny, u.nx,
                                      temperatureVariable->name, output);
    if (vertical.levels.size() != u.nz) {
        warn(output, "Invalid vertical levels detected, fallback to index-based z levels (1 m spacing)");
        vertical.levels = linearCoordinates(0.0, static_cast<double>(u.nz - 1U), u.nz);
        vertical.dimensionCoordinates = vertical.levels;
    }

    std::vector<double> xCoordinates = readMassDimensionCoordinates(dataset, "lon", crop);
    std::vector<double> yCoordinates = readMassDimensionCoordinates(dataset, "lat", crop);
    const bool needsDense = u.nx < 200U || u.ny < 200U || u.nz < 10U;
    if (dem && needsDense) {
        info(output, "Skip Dense Compute interpolation because terrain-first BC sampling must use native NC spacing before horizontal interpolation");
    } else if (!dem) {
        denseInterpolateNoTerrain(u, v, w, temperature, geo, xCoordinates,
                                  yCoordinates, vertical, output, progress);
    }
    if (geo.nx != crop.nx() || geo.ny != crop.ny()) {
        const ProjectedGridCache::Lookup denseProjection = acquireProjectedGrid(
            mappingCacheScope, geo, projector);
        reportPlanCache(output, "PROJ dense wind-grid", denseProjection.hit,
                        denseProjection.retained, denseProjection.evicted);
        projectedGeo = denseProjection.plan->points;
    }

    info(output, "Data shape nz=" + std::to_string(u.nz) + " ny=" + std::to_string(u.ny)
         + " nx=" + std::to_string(u.nx));
    if (containsNan(u) || containsNan(v)) {
        warn(output, "NaNs found. Apply vertical forward fill");
        forwardFillWholeLayers(u, {&v});
    }
    if (temperature && containsNan(*temperature)) {
        warn(output, "NaNs found in temperature field. Apply vertical forward fill");
        forwardFillWholeLayers(*temperature, {});
    }

    const Bounds dataGeoBounds = geoBounds(geo);
    info(output, "Using UTM CRS for CFD domain: " + utmCrs);
    const auto dataCorners = projectCorners(dataGeoBounds, projector);
    auto [rotationDegrees, pivot] = rotationFromCorners(dataCorners);
    info(output, "Estimated convergence angle = " + fixed(rotationDegrees, 6)
         + " deg, use projected bbox centroid as pivot");
    RotationFrame frame;
    frame.degrees = rotationDegrees;
    frame.pivot = pivot;
    if (requestedBounds) {
        const auto targetCornersProjected = projectCorners(*requestedBounds, projector);
        std::tie(frame.degrees, frame.pivot) = rotationFromCorners(targetCornersProjected);
        info(output, "Override rotation/pivot from config bounds: rotate_deg="
             + fixed(frame.degrees, 6) + ", pivot=(" + fixed(frame.pivot.x, 3) + ", "
             + fixed(frame.pivot.y, 3) + ")");
        double xMin = std::numeric_limits<double>::infinity();
        double xMax = -std::numeric_limits<double>::infinity();
        double yMin = std::numeric_limits<double>::infinity();
        double yMax = -std::numeric_limits<double>::infinity();
        for (const XY& point : targetCornersProjected) {
            const XY rotated = rotatePoint(point, frame.degrees, frame.pivot);
            xMin = std::min(xMin, rotated.x); xMax = std::max(xMax, rotated.x);
            yMin = std::min(yMin, rotated.y); yMax = std::max(yMax, rotated.y);
        }
        frame.xOrigin = xMin; frame.yOrigin = yMin;
        frame.xRange = xMax - xMin; frame.yRange = yMax - yMin;
        frame.fromTargetBounds = true;
        info(output, "Target domain (rotated) from config: " + fixed(frame.xRange, 3)
             + " x " + fixed(frame.yRange, 3) + " m; origin=(" + fixed(frame.xOrigin, 3)
             + ", " + fixed(frame.yOrigin, 3) + ")");
    }

    std::vector<double> xSource(geo.nx * geo.ny);
    std::vector<double> ySource(geo.nx * geo.ny);
    double dataXMin = std::numeric_limits<double>::infinity();
    double dataXMax = -std::numeric_limits<double>::infinity();
    double dataYMin = std::numeric_limits<double>::infinity();
    double dataYMax = -std::numeric_limits<double>::infinity();
    const RotationOperator windRotation(frame.degrees, frame.pivot);
    const std::size_t projectedPointCount = geo.nx * geo.ny;
#pragma omp parallel for if(projectedPointCount > 4096)
    for (std::int64_t pointSigned = 0;
         pointSigned < static_cast<std::int64_t>(projectedPointCount);
         ++pointSigned) {
        const std::size_t point = static_cast<std::size_t>(pointSigned);
        const XY rotated = windRotation.apply(projectedGeo[point]);
        xSource[point] = rotated.x;
        ySource[point] = rotated.y;
    }
    for (std::size_t point = 0U; point < projectedPointCount; ++point) {
        if (std::isfinite(xSource[point])) {
            dataXMin = std::min(dataXMin, xSource[point]);
            dataXMax = std::max(dataXMax, xSource[point]);
        }
        if (std::isfinite(ySource[point])) {
            dataYMin = std::min(dataYMin, ySource[point]);
            dataYMax = std::max(dataYMax, ySource[point]);
        }
    }
    info(output, "Wind data range after rotation X [" + fixed(dataXMin, 3) + ", "
         + fixed(dataXMax, 3) + "] Y [" + fixed(dataYMin, 3) + ", "
         + fixed(dataYMax, 3) + "]");
    if (!frame.fromTargetBounds) {
        frame.xOrigin = dataXMin; frame.yOrigin = dataYMin;
        frame.xRange = dataXMax - dataXMin; frame.yRange = dataYMax - dataYMin;
    } else {
        info(output, "Use config-rotated origin shift: X0=" + fixed(frame.xOrigin, 3)
             + " Y0=" + fixed(frame.yOrigin, 3));
    }
    for (double& value : xSource) value -= frame.xOrigin;
    for (double& value : ySource) value -= frame.yOrigin;
    const bool needsHorizontalFill = containsNan(u) || containsNan(v) || containsNan(w)
        || (temperature && containsNan(*temperature));
    if (needsHorizontalFill) {
        warn(output, "NaNs found in wind/temperature fields, fill horizontally by nearest neighbor in projected space");
        fillHorizontalNearest(u, xSource, ySource);
        fillHorizontalNearest(v, xSource, ySource);
        fillHorizontalNearest(w, xSource, ySource);
        if (temperature) fillHorizontalNearest(*temperature, xSource, ySource);
    }
    if (frame.fromTargetBounds) {
        info(output, "Using target domain size for interpolation: " + fixed(frame.xRange, 3)
             + " x " + fixed(frame.yRange, 3) + " m");
    } else {
        info(output, "Using data range for interpolation: " + fixed(frame.xRange, 3)
             + " x " + fixed(frame.yRange, 3) + " m");
    }

    double baseHeight = input.deck.floating(QStringLiteral("base_height")).value_or(50.0);
    if (!std::isfinite(baseHeight) || baseHeight < 0.0) baseHeight = 50.0;
    std::optional<double> zLimit = input.deck.floating(QStringLiteral("z_limit"));
    if (zLimit && (!std::isfinite(*zLimit) || *zLimit <= 0.0)) zLimit.reset();
    double meshBase = input.deck.floating(QStringLiteral("midmesh_basesize")).value_or(50.0);
    if (!std::isfinite(meshBase) || meshBase <= 0.0) meshBase = 50.0;
    info(output, "midmesh_basesize for output grid = " + fixed(meshBase, 3)
         + " m (default 50.0 m if not set)");
    const auto choosePointCount = [&](double length) {
        if (!std::isfinite(length) || length <= 0.0) return std::size_t{1U};
        const auto cells = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::nearbyint(length / meshBase)));
        return static_cast<std::size_t>(cells + 1);
    };
    const std::size_t nx = choosePointCount(frame.xRange);
    const std::size_t ny = choosePointCount(frame.yRange);
    info(output, "Target grid points (nx, ny) = (" + std::to_string(nx) + ", "
         + std::to_string(ny) + ") -> nominal spacing ~ "
         + fixed(frame.xRange / static_cast<double>(std::max<std::size_t>(nx - 1U, 1U)), 3)
         + " m x "
         + fixed(frame.yRange / static_cast<double>(std::max<std::size_t>(ny - 1U, 1U)), 3)
         + " m");
    const std::vector<double> xGrid = linearCoordinates(0.0, frame.xRange, nx);
    const std::vector<double> yGrid = linearCoordinates(0.0, frame.yRange, ny);
    const double dx = nx > 1U ? frame.xRange / static_cast<double>(nx - 1U) : 0.0;
    const double dy = ny > 1U ? frame.yRange / static_cast<double>(ny - 1U) : 0.0;

    Field3D uSourceNative = u;
    Field3D vSourceNative = v;
    Field3D wSourceNative = w;
    std::optional<Field3D> tSourceNative = temperature;
    info(output, "Total vertical levels to interpolate: " + std::to_string(u.nz));
    info(output, "Start horizontal interpolation onto uniform grid: precomputed barycentric weights with nearest fallback");
    progress.progress(QStringLiteral("horizontal_interpolation"), 0,
                      static_cast<std::int64_t>(std::max<std::size_t>(u.nz, 1U)),
                      QStringLiteral("Interpolating horizontal levels 0/%1")
                          .arg(static_cast<qulonglong>(u.nz)), {}, true);
    if (!progress.enabled()) {
        output << "[INFO] Horizontal interpolation progress (# is 5%): |";
        output.flush();
    }
    const HorizontalMappingCache::Lookup horizontal = acquireHorizontalPlan(
        mappingCacheScope, xSource, ySource, xGrid, yGrid);
    reportPlanCache(output, "Horizontal barycentric", horizontal.hit,
                    horizontal.retained, horizontal.evicted);
    Field3D uUniform;
    Field3D vUniform;
    Field3D wUniform;
    std::optional<Field3D> tUniform;
    if (computePolicy.useGpu) {
        std::vector<const Field3D*> fields{&u, &v, &w};
        if (temperature) fields.push_back(&*temperature);
        std::vector<Field3D> uniform = horizontal.plan->applyOpenCl(
            fields, computePolicy.gpuDevice, output);
        uUniform = std::move(uniform[0]);
        vUniform = std::move(uniform[1]);
        wUniform = std::move(uniform[2]);
        if (temperature) tUniform = std::move(uniform[3]);
    } else {
        std::vector<const Field3D*> fields{&u, &v, &w};
        if (temperature) fields.push_back(&*temperature);
        std::vector<Field3D> uniform = horizontal.plan->applyBatch(fields);
        info(output, "CPU fused horizontal interpolation batch: "
             + std::to_string(fields.size()) + " fields");
        uUniform = std::move(uniform[0]);
        vUniform = std::move(uniform[1]);
        wUniform = std::move(uniform[2]);
        if (temperature) tUniform = std::move(uniform[3]);
    }
    if (!progress.enabled()) {
        output << "####|20%|####|40%|####|60%|####|80%|####|100%||Finished|\n";
        output.flush();
    }
    progress.progress(QStringLiteral("horizontal_interpolation"),
                      static_cast<std::int64_t>(u.nz), static_cast<std::int64_t>(u.nz),
                      QStringLiteral("Interpolating horizontal levels %1/%1")
                          .arg(static_cast<qulonglong>(u.nz)));
    info(output, "Horizontal interpolation finished");
    progress.complete(QStringLiteral("horizontal_interpolation"),
                      QStringLiteral("Interpolated %1 vertical levels")
                          .arg(static_cast<qulonglong>(u.nz)));

    std::optional<std::vector<double>> demGrid;
    std::vector<XY> shiftedDemPoints;
    if (dem) {
        progress.stage(QStringLiteral("terrain_grid"),
                       QStringLiteral("Interpolating terrain onto CFD grid"));
        info(output, "Interpolating DEM to wind field grid");
        demGrid = interpolateDemGrid(*dem, frame, xGrid, yGrid,
                                     &shiftedDemPoints, mappingCacheScope,
                                     computePolicy, output);
    } else {
        info(output, "No DEM data, proceeding with flat terrain");
    }

    // Build every data-dependent terrain GPU plan before deck/output creation.
    // An explicit GPU error therefore cannot be mistaken for a successful
    // target-column fallback after partial boundary artifacts were written.
    std::shared_ptr<const TerrainFirstPlan> terrainPlan;
    if (demGrid && dem && !shiftedDemPoints.empty()) {
        progress.stage(QStringLiteral("terrain_sampling"),
                       QStringLiteral("Preparing terrain-following boundary sampling"));
        try {
            const TerrainMappingCache::Lookup terrainLookup =
                acquireTerrainFirstPlan(
                    mappingCacheScope, xSource, ySource, shiftedDemPoints,
                    dem->elevations, xGrid, yGrid,
                    computePolicy.useGpu
                        ? std::optional<int>(computePolicy.gpuDevice)
                        : std::nullopt,
                    output);
            reportPlanCache(output, "Terrain-first MLS", terrainLookup.hit,
                            terrainLookup.retained, terrainLookup.evicted);
            terrainPlan = terrainLookup.plan;
            info(output, "Boundary sampling order switched to terrain-first: uplift on source columns first, then horizontal MLS interpolation (quadratic, Wendland-C2, k="
                 + std::to_string(terrainPlan->neighborCount()) + ") to target points");
        } catch (const OpenClLuwbcError&) {
            throw;
        } catch (const std::exception& exception) {
            if (computePolicy.useGpu) throw;
            warn(output, "Terrain-first boundary sampling initialization failed ("
                 + std::string(exception.what()) + "); fallback to target-column sampling");
            terrainPlan.reset();
        }
    }

    std::vector<double> levels = vertical.levels;
    if (levels.size() != uUniform.nz
        || !std::all_of(levels.begin(), levels.end(), [](double value) { return std::isfinite(value); })) {
        warn(output, "Invalid vertical levels detected, fallback to index-based z levels (1 m spacing)");
        levels = linearCoordinates(0.0, static_cast<double>(uUniform.nz - 1U), uUniform.nz);
    }
    const std::string units = lowerAscii(vertical.units);
    const std::string verticalName = lowerAscii(vertical.dimension);
    const bool lengthUnits = units == "m" || units == "meter" || units == "meters"
        || units == "metre" || units == "metres";
    const bool pressureUnits = units == "pa" || units == "hpa" || units == "mb"
        || units.find("mbar") != std::string::npos || units.find("millibar") != std::string::npos;
    const std::set<std::string> pressureNames{"plev", "pressure", "isobaric", "isobaric1",
                                               "isobaric_inpa", "isobaric_inhpa"};
    if (pressureUnits || (pressureNames.contains(verticalName) && !lengthUnits)) {
        const auto [minimum, maximum] = std::minmax_element(levels.begin(), levels.end());
        if (*minimum > 10.0 && *maximum < 2000.0 && *maximum - *minimum > 10.0) {
            warn(output, "Vertical levels appear to be pressure; keep index-based meters for VTK/CFD compatibility");
            levels = linearCoordinates(0.0, static_cast<double>(uUniform.nz - 1U), uUniform.nz);
        }
    }
    if (levels.size() >= 2U && levels[1] < levels[0]) {
        std::reverse(levels.begin(), levels.end());
        reverseVertical(uUniform); reverseVertical(vUniform); reverseVertical(wUniform);
        reverseVertical(uSourceNative); reverseVertical(vSourceNative); reverseVertical(wSourceNative);
        if (tUniform) reverseVertical(*tUniform);
        if (tSourceNative) reverseVertical(*tSourceNative);
    }
    for (std::size_t index = 1U; index < levels.size(); ++index) {
        if (levels[index] <= levels[index - 1U]) levels[index] = levels[index - 1U] + 1e-3;
    }
    double rawMinimumZ = levels.front();
    double rawTopZ = levels.back();
    if (!std::isfinite(rawTopZ) || rawTopZ <= 0.0) {
        warn(output, "Invalid z_top detected, fallback to 1 m vertical spacing");
        levels = linearCoordinates(0.0, static_cast<double>(uUniform.nz - 1U), uUniform.nz);
        rawMinimumZ = levels.front(); rawTopZ = levels.back();
    }
    progress.stage(QStringLiteral("vertical_resample"),
                   QStringLiteral("Resampling vertical CFD grid"));
    const std::size_t zCells = static_cast<std::size_t>(std::max<std::int64_t>(
        1, static_cast<std::int64_t>(std::nearbyint(rawTopZ / meshBase))));
    const std::vector<double> zAgl = linearCoordinates(0.0, rawTopZ, zCells + 1U);
    const double dz = zAgl.size() > 1U ? zAgl[1] - zAgl[0] : 0.0;
    info(output, "Resample vertical grid to ~" + fixed(meshBase, 3)
         + " m with zmin-below nearest fill and ground=0: z_min_raw=" + fixed(rawMinimumZ, 3)
         + " m, z_top=" + fixed(rawTopZ, 3) + " m, nz_src=" + std::to_string(levels.size())
         + " -> nz_new=" + std::to_string(zAgl.size()) + ", dz_new=" + fixed(dz, 3) + " m");
    const VerticalMappingCache::Lookup verticalPlan = acquireVerticalPlan(
        mappingCacheScope, levels, zAgl);
    reportPlanCache(output, "Vertical axis", verticalPlan.hit,
                    verticalPlan.retained, verticalPlan.evicted);
    VerticalFieldDomain uniformFields{&uUniform, &vUniform, &wUniform};
    if (tUniform) uniformFields.push_back(&*tUniform);
    VerticalFieldDomain nativeFields{
        &uSourceNative, &vSourceNative, &wSourceNative};
    if (tSourceNative) nativeFields.push_back(&*tSourceNative);
    if (computePolicy.useGpu) {
        std::vector<std::vector<Field3D>> resampled =
            resampleVerticalLinearOpenClBatched(
                {uniformFields, nativeFields}, levels, zAgl,
                computePolicy.gpuDevice, output);
        std::vector<Field3D>& resampledUniform = resampled[0];
        uUniform = std::move(resampledUniform[0]);
        vUniform = std::move(resampledUniform[1]);
        wUniform = std::move(resampledUniform[2]);
        if (tUniform) *tUniform = std::move(resampledUniform[3]);
        std::vector<Field3D>& resampledNative = resampled[1];
        uSourceNative = std::move(resampledNative[0]);
        vSourceNative = std::move(resampledNative[1]);
        wSourceNative = std::move(resampledNative[2]);
        if (tSourceNative) *tSourceNative = std::move(resampledNative[3]);
    } else {
        std::vector<Field3D> resampledUniform =
            verticalPlan.plan->applyBatch(uniformFields);
        std::vector<Field3D> resampledNative =
            verticalPlan.plan->applyBatch(nativeFields);
        info(output, "CPU fused vertical interpolation batch: 2 point domains, "
             + std::to_string(uniformFields.size()) + " fields each");
        uUniform = std::move(resampledUniform[0]);
        vUniform = std::move(resampledUniform[1]);
        wUniform = std::move(resampledUniform[2]);
        if (tUniform) *tUniform = std::move(resampledUniform[3]);
        uSourceNative = std::move(resampledNative[0]);
        vSourceNative = std::move(resampledNative[1]);
        wSourceNative = std::move(resampledNative[2]);
        if (tSourceNative) *tSourceNative = std::move(resampledNative[3]);
    }

    double groundMaximum = baseHeight;
    if (demGrid) {
        double maximum = *std::max_element(demGrid->begin(), demGrid->end()) * options.elevationScale;
        if (!std::isfinite(maximum) || maximum < 0.0) maximum = 0.0;
        groundMaximum += maximum;
    }
    double outputTopAgl = static_cast<double>(zAgl.size() - 1U) * dz;
    if (zLimit) {
        if (outputTopAgl > *zLimit) {
            info(output, "Apply global top clipping by z_limit AGL: wind_top_agl="
                 + fixed(outputTopAgl, 3) + " m, z_limit=" + fixed(*zLimit, 3) + " m");
        } else {
            info(output, "Skip global top clipping: wind_top_agl " + fixed(outputTopAgl, 3)
                 + " m <= z_limit " + fixed(*zLimit, 3) + " m");
        }
        outputTopAgl = std::min(outputTopAgl, *zLimit);
    }
    const double outputTop = groundMaximum + outputTopAgl;
    info(output, "Global top plane: ground_max=" + fixed(groundMaximum, 3) + " m + top_agl="
         + fixed(outputTopAgl, 3) + " m -> top=" + fixed(outputTop, 3) + " m");

    const std::array<double, 3> volumeMean{meanFinite(uUniform), meanFinite(vUniform),
                                           meanFinite(wUniform)};
    if (frame.fromTargetBounds) {
        info(output, "Using target domain size for si_x/y_cfd: " + fixed(frame.xRange, 3)
             + " x " + fixed(frame.yRange, 3) + " m");
    } else {
        info(output, "Using data bounds for si_x/y_cfd: " + fixed(frame.xRange, 3)
             + " x " + fixed(frame.yRange, 3) + " m");
    }
    input.deck.setPair(QStringLiteral("si_x_cfd"), {0.0, frame.xRange});
    input.deck.setPair(QStringLiteral("si_y_cfd"), {0.0, frame.yRange});
    input.deck.setPair(QStringLiteral("si_z_cfd"), {0.0, outputTop});
    input.deck.setText(QStringLiteral("utm_crs"), QString::fromStdString(utmCrs), true);
    input.deck.setFloat(QStringLiteral("rotate_deg"), frame.degrees);
    input.deck.setBoolean(QStringLiteral("origin_shift_applied"), true);
    input.deck.save();
    info(output, "Wrote lon/lat and SI ranges to deck file");

    const fs::path projectTemp = input.project / "proj_temp";
    std::error_code createError;
    fs::create_directories(projectTemp, createError);
    if (createError) throw std::runtime_error("Cannot create proj_temp: " + createError.message());
    if (options.writeVtk) {
        fs::path vtiPath = projectTemp / input.netcdfPath.filename();
        vtiPath.replace_extension(".vti");
        writeVti(vtiPath, uUniform, vUniform, wUniform, dx, dy, dz);
        std::error_code sizeError;
        const double megabytes = static_cast<double>(fs::file_size(vtiPath, sizeError)) / 1e6;
        info(output, "Wrote VTI: " + pathToUtf8(vtiPath) + " size ~ " + fixed(megabytes, 3) + " MB");
    } else {
        info(output, "Skip VTI output (default). Use --write-vtk to enable.");
    }
    if (demGrid) {
        const fs::path picklePath = projectTemp / "dem_grid.pkl";
        progress.stage(QStringLiteral("write_dem_cache"),
                       QStringLiteral("Saving terrain cache: dem_grid.pkl"));
        writeDemPickle(picklePath, *demGrid, ny, nx, xGrid, yGrid, dx, dy,
                       baseHeight, frame);
        info(output, "Saved DEM grid data to: " + pathToUtf8(picklePath));
    }

    if (terrainPlan) {
        info(output, "Applying terrain-following coordinates with fixed top at "
             + fixed(outputTop, 2) + "m");
        const auto [demMinimum, demMaximum] = std::minmax_element(demGrid->begin(), demGrid->end());
        info(output, "DEM elevation difference range: " + fixed(*demMinimum, 2) + "m to "
             + fixed(*demMaximum, 2) + "m");
        info(output, "Bottom elevation range: " + fixed(baseHeight + *demMinimum, 2)
             + "m to " + fixed(baseHeight + *demMaximum, 2) + "m");
    } else if (!(demGrid && dem && !shiftedDemPoints.empty())) {
        progress.stage(QStringLiteral("terrain_sampling"),
                       QStringLiteral("Preparing flat-terrain boundary sampling"));
        info(output, "No DEM data, applying flat base offset: Z = Z_grid + "
             + pythonFloat(baseHeight) + "m");
    }

    struct Sample { double u; double v; double w; std::optional<double> temperature; };
    const std::size_t boundaryFieldCount = tUniform ? 4U : 3U;
    const std::array<const Field3D*, 4> uniformBoundaryFields{
        &uUniform, &vUniform, &wUniform, tUniform ? &*tUniform : nullptr};
    const std::array<const Field3D*, 4> nativeBoundaryFields{
        &uSourceNative, &vSourceNative, &wSourceNative,
        tSourceNative ? &*tSourceNative : nullptr};
    const auto sample = [&](std::size_t y, std::size_t x, double absoluteZ,
                             double ground) -> Sample {
        if (terrainPlan) {
            const std::array<double, 4> values = terrainPlan->sampleBatch(
                nativeBoundaryFields, boundaryFieldCount, y, x, absoluteZ,
                baseHeight, options.elevationScale, zAgl);
            return {values[0], values[1], values[2],
                    tSourceNative ? std::optional<double>(values[3])
                                  : std::nullopt};
        }
        const double queryAgl = std::clamp(absoluteZ - ground, zAgl.front(), zAgl.back());
        const VerticalWeight vertical = verticalWeight(zAgl, queryAgl);
        std::array<double, 4> values{};
        for (std::size_t component = 0U; component < boundaryFieldCount;
             ++component) {
            values[component] = verticalLinearSample(
                *uniformBoundaryFields[component], y, x, vertical);
        }
        return {values[0], values[1], values[2],
                tUniform ? std::optional<double>(values[3]) : std::nullopt};
    };

    const fs::path csvPath = projectTemp / "SurfData_Latest.csv";
    const bool writeTemperature = tUniform.has_value();
    if (options.elevationScale != 1.0) {
        info(output, "Applying elevation scale " + pythonFloat(options.elevationScale)
             + "x to CSV output");
    }
    if (writeTemperature) info(output, "Append temperature column T (Kelvin) to boundary CSV");
    info(output, "Append patch column to boundary CSV: top=1, south=2, north=3, west=4, east=5, bottom=0");
    info(output, "Preserve sampled vertical velocity w on top and side boundary patches");
    const double groundEpsilon = dz > 0.0 && std::isfinite(dz)
        ? std::max(1e-3, std::min(0.1, 0.05 * dz)) : 0.05;
    info(output, "Bottom patch points are sampled at ground + " + fixed(groundEpsilon, 4) + " m");
    std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
    if (!csv) throw std::runtime_error("Cannot write boundary CSV: " + pathToUtf8(csvPath));
    progress.progress(QStringLiteral("save_boundary_csv"), 0, 6,
                      QStringLiteral("Writing boundary CSV: SurfData_Latest.csv (0/6 faces)"), {}, true);
    csv << (writeTemperature ? "X,Y,Z,u,v,w,T,patch\n" : "X,Y,Z,u,v,w,patch\n");
    std::array<long double, 3> boundarySum{};
    std::size_t boundaryCount = 0U;
    const auto groundAt = [&](std::size_t y, std::size_t x) {
        const double terrain = demGrid ? (*demGrid)[y * nx + x] : 0.0;
        return baseHeight + terrain * options.elevationScale;
    };
    const auto writeRow = [&](double x, double y, double z, const Sample& value, int patch) {
        csv << fixed(x, 3) << ',' << fixed(y, 3) << ',' << fixed(z, 3) << ','
            << pythonFloat(value.u) << ',' << pythonFloat(value.v) << ','
            << pythonFloat(value.w) << ',';
        if (writeTemperature) csv << pythonFloat(value.temperature.value_or(0.0)) << ',';
        csv << patch << '\n';
        boundarySum[0] += value.u; boundarySum[1] += value.v; boundarySum[2] += value.w;
        ++boundaryCount;
    };

    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            const double ground = groundAt(y, x);
            if (!std::isfinite(ground) || outputTop <= ground) continue;
            const double bottom = ground + std::min(groundEpsilon, 0.5 * (outputTop - ground));
            if (bottom >= outputTop - 1e-6) continue;
            writeRow(xGrid[x], yGrid[y], bottom, sample(y, x, bottom, ground), 0);
        }
    }
    progress.progress(QStringLiteral("save_boundary_csv"), 1, 6,
                      QStringLiteral("Writing boundary CSV: bottom face 1/6"));
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            const double ground = groundAt(y, x);
            if (!std::isfinite(ground) || ground >= outputTop) continue;
            writeRow(xGrid[x], yGrid[y], outputTop, sample(y, x, outputTop, ground), 1);
        }
    }
    progress.progress(QStringLiteral("save_boundary_csv"), 2, 6,
                      QStringLiteral("Writing boundary CSV: top face 2/6"));
    int faceProgress = 2;
    for (const std::size_t y : {std::size_t{0U}, ny - 1U}) {
        const int patch = y == 0U ? 2 : 3;
        for (std::size_t x = 0; x < nx; ++x) {
            const double ground = groundAt(y, x);
            if (!std::isfinite(ground) || outputTop <= ground) continue;
            writeRow(xGrid[x], yGrid[y], ground, sample(y, x, ground, ground), patch);
            if (dz <= 0.0 || !std::isfinite(dz)) continue;
            std::size_t maximumLevel = static_cast<std::size_t>(std::max<double>(
                0.0, std::floor((outputTop - ground) / dz + 1e-6)));
            maximumLevel = std::min(maximumLevel, zAgl.size() - 1U);
            for (std::size_t level = 1U; level <= maximumLevel; ++level) {
                const double z = ground + static_cast<double>(level) * dz;
                if (z >= outputTop - 1e-6) continue;
                writeRow(xGrid[x], yGrid[y], z, sample(y, x, z, ground), patch);
            }
        }
        ++faceProgress;
        progress.progress(QStringLiteral("save_boundary_csv"), faceProgress, 6,
                          QStringLiteral("Writing boundary CSV: %1 face %2/6")
                              .arg(y == 0U ? QStringLiteral("south") : QStringLiteral("north"))
                              .arg(faceProgress));
    }
    for (const std::size_t x : {std::size_t{0U}, nx - 1U}) {
        const int patch = x == 0U ? 4 : 5;
        for (std::size_t y = 0; y < ny; ++y) {
            const double ground = groundAt(y, x);
            if (!std::isfinite(ground) || outputTop <= ground) continue;
            writeRow(xGrid[x], yGrid[y], ground, sample(y, x, ground, ground), patch);
            if (dz <= 0.0 || !std::isfinite(dz)) continue;
            std::size_t maximumLevel = static_cast<std::size_t>(std::max<double>(
                0.0, std::floor((outputTop - ground) / dz + 1e-6)));
            maximumLevel = std::min(maximumLevel, zAgl.size() - 1U);
            for (std::size_t level = 1U; level <= maximumLevel; ++level) {
                const double z = ground + static_cast<double>(level) * dz;
                if (z >= outputTop - 1e-6) continue;
                writeRow(xGrid[x], yGrid[y], z, sample(y, x, z, ground), patch);
            }
        }
        ++faceProgress;
        progress.progress(QStringLiteral("save_boundary_csv"), faceProgress, 6,
                          QStringLiteral("Writing boundary CSV: %1 face %2/6")
                              .arg(x == 0U ? QStringLiteral("west") : QStringLiteral("east"))
                              .arg(faceProgress));
    }
    csv.close();
    if (!csv) throw std::runtime_error("Failed while writing boundary CSV");
    info(output, "Wrote boundary CSV: " + pathToUtf8(csvPath));
    const fs::path timestamped = projectTemp / ("SurfData_" + input.dateTime + ".csv");
    std::error_code copyError;
    fs::copy_file(csvPath, timestamped, fs::copy_options::overwrite_existing, copyError);
    if (copyError) throw std::runtime_error("Cannot copy timestamped CSV: " + copyError.message());
    info(output, "Copied timestamped CSV: " + pathToUtf8(timestamped));
    progress.complete(QStringLiteral("save_boundary_csv"),
                      QStringLiteral("Saved boundary CSV: %1")
                          .arg(QString::fromStdString(pathToUtf8(timestamped.filename()))));

    const std::array<double, 3> boundaryMean{
        boundaryCount > 0U ? static_cast<double>(boundarySum[0] / boundaryCount) : 0.0,
        boundaryCount > 0U ? static_cast<double>(boundarySum[1] / boundaryCount) : 0.0,
        boundaryCount > 0U ? static_cast<double>(boundarySum[2] / boundaryCount) : 0.0};
    std::string downstream;
    double parallel = 0.0;
    double perpendicular = 0.0;
    if (std::abs(volumeMean[0]) >= std::abs(volumeMean[1])) {
        downstream = volumeMean[0] >= 0.0 ? "+x" : "-x";
        parallel = std::abs(volumeMean[0]); perpendicular = volumeMean[1];
    } else {
        downstream = volumeMean[1] >= 0.0 ? "+y" : "-y";
        parallel = std::abs(volumeMean[1]); perpendicular = volumeMean[0];
    }
    const double theta = parallel != 0.0
        ? std::atan2(std::abs(perpendicular), std::abs(parallel)) * 180.0 / kPi : 90.0;
    const double yaw = (perpendicular >= 0.0 ? 1.0 : -1.0) * theta;
    QVariantList volumeList;
    QVariantList boundaryList;
    for (double value : volumeMean) volumeList.push_back(value);
    for (double value : boundaryMean) boundaryList.push_back(value);
    input.deck.setList(QStringLiteral("um_vol"), volumeList);
    input.deck.setList(QStringLiteral("um_bc"), boundaryList);
    input.deck.setText(QStringLiteral("downstream_bc"), QString::fromStdString(downstream), true);
    input.deck.setFloat(QStringLiteral("downstream_bc_yaw"), yaw, 2);
    input.deck.save();
    info(output, "Wrote mean velocity and downstream info to deck file");

    info(output, "Lon range " + fixed(dataGeoBounds.lonMin, 6) + " to "
         + fixed(dataGeoBounds.lonMax, 6) + " Lat range " + fixed(dataGeoBounds.latMin, 6)
         + " to " + fixed(dataGeoBounds.latMax, 6));
    info(output, "Target domain range X 0.000 to " + fixed(frame.xRange, 3)
         + " m Y 0.000 to " + fixed(frame.yRange, 3) + " m");
    info(output, "Grid spacing dx=" + fixed(dx, 3) + " m dy=" + fixed(dy, 3)
         + " m dz=" + fixed(dz, 3) + " m");
    info(output, "buildBC_dev finished");
}

} // namespace

int run_luwbc(const CommandArguments& arguments,
              std::ostream& output,
              std::ostream& error) {
    const std::optional<BuildOptions> options = parseOptions(arguments, output, error);
    if (!options) {
        const bool help = arguments.size() == 1U
            && (arguments.front() == "--help" || arguments.front() == "-h");
        return help ? 0 : 2;
    }
    try {
        buildBoundaryConditions(*options, output);
        return 0;
    } catch (const DemTifCacheCancelled&) {
        return 1;
    } catch (const std::exception& exception) {
        errorLine(error, "Execution failed: " + std::string(exception.what()));
        return 1;
    }
}

} // namespace luw::cli
