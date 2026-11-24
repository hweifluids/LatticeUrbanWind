#include "luw/buildingscale_command.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <cpl_error.h>
#include <gdal_priv.h>
#include <gdal_version.h>
#include <ogrsf_frmts.h>
#endif

namespace luw::cli {

namespace {

namespace fs = std::filesystem;

struct Arguments final {
    fs::path shapefile;
    std::optional<fs::path> output;
    bool help = false;
};

std::string pathUtf8(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const std::u8string text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

fs::path pathFromUtf8(std::string_view value) {
#if defined(__cpp_lib_char8_t)
    const auto* begin = reinterpret_cast<const char8_t*>(value.data());
    return fs::path(std::u8string(begin, begin + value.size()));
#else
    return fs::u8path(value.begin(), value.end());
#endif
}

void usage(std::ostream& stream) {
    stream
        << "usage: buildingscale [-h] [--out OUT] shp_path\n\n"
           "Plot empirical CDF of building short-side lengths from a shapefile.\n\n"
           "positional arguments:\n"
           "  shp_path    Path to the input .shp file\n\n"
           "options:\n"
           "  -h, --help  show this help message and exit\n"
           "  --out OUT   Output image path (PNG). Default:\n"
           "              <shp_dir>/<shp_stem>_short_side_cdf.png\n";
}

int argumentError(std::ostream& error, std::string_view message) {
    error << "usage: buildingscale [-h] [--out OUT] shp_path\n"
          << "buildingscale: error: " << message << '\n';
    return 2;
}

std::optional<Arguments> parseArguments(const CommandArguments& arguments,
                                        std::ostream& output,
                                        std::ostream& error,
                                        int& status) {
    Arguments parsed;
    std::vector<std::string> positional;
    bool positionalOnly = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        const bool abbreviatedHelp = !positionalOnly && argument.size() >= 3U
            && std::string_view("--help").starts_with(argument);
        if (!positionalOnly && (argument == "-h" || abbreviatedHelp)) {
            usage(output);
            parsed.help = true;
            status = 0;
            return parsed;
        }
        if (!positionalOnly && argument == "--") {
            positionalOnly = true;
            continue;
        }
        const bool abbreviatedOut = !positionalOnly && argument.size() >= 3U
            && std::string_view("--out").starts_with(argument);
        if (abbreviatedOut) {
            if (index + 1 >= arguments.size()) {
                status = argumentError(error, "argument --out: expected one argument");
                return std::nullopt;
            }
            parsed.output = pathFromUtf8(arguments[++index]);
            continue;
        }
        if (!positionalOnly && argument.starts_with("--")) {
            const std::size_t equals = argument.find('=');
            const std::size_t nameLength = equals == std::string::npos ? argument.size() : equals;
            const std::string_view name(argument.data(), nameLength);
            if (name.size() >= 3U && std::string_view("--out").starts_with(name)
                && equals != std::string::npos) {
                parsed.output = pathFromUtf8(std::string_view(argument).substr(equals + 1U));
                continue;
            }
        }
        if (!positionalOnly && !argument.empty() && argument.front() == '-') {
            status = argumentError(error, "unrecognized arguments: " + argument);
            return std::nullopt;
        }
        positional.push_back(argument);
    }
    if (positional.empty()) {
        status = argumentError(error, "the following arguments are required: shp_path");
        return std::nullopt;
    }
    if (positional.size() > 1) {
        std::string extras;
        for (std::size_t index = 1; index < positional.size(); ++index) {
            if (!extras.empty()) extras.push_back(' ');
            extras += positional[index];
        }
        status = argumentError(error, "unrecognized arguments: " + extras);
        return std::nullopt;
    }
    parsed.shapefile = pathFromUtf8(positional.front());
    status = 0;
    return parsed;
}

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) throw std::invalid_argument("Cannot calculate an empty percentile.");
    const double position = fraction * static_cast<double>(sorted.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if (values.size() % 2U != 0U) return values[middle];
    return (values[middle - 1U] + values[middle]) / 2.0;
}

#if LUW_HAS_GDAL

struct DatasetCloser final {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) GDALClose(dataset);
    }
};

struct GeometryCloser final {
    void operator()(OGRGeometry* geometry) const noexcept {
        if (geometry != nullptr) OGRGeometryFactory::destroyGeometry(geometry);
    }
};

struct FeatureCloser final {
    void operator()(OGRFeature* feature) const noexcept {
        if (feature != nullptr) OGRFeature::DestroyFeature(feature);
    }
};

struct TransformCloser final {
    void operator()(OGRCoordinateTransformation* transformation) const noexcept {
        if (transformation != nullptr) OCTDestroyCoordinateTransformation(transformation);
    }
};

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;
using GeometryPtr = std::unique_ptr<OGRGeometry, GeometryCloser>;
using FeaturePtr = std::unique_ptr<OGRFeature, FeatureCloser>;
using TransformPtr = std::unique_ptr<OGRCoordinateTransformation, TransformCloser>;

struct FeatureRecord final {
    GeometryPtr geometry;
    std::vector<double> numericFields;
};

struct Point final {
    double x = 0.0;
    double y = 0.0;
};

bool pointLess(const Point& left, const Point& right) noexcept {
    return left.x < right.x || (left.x == right.x && left.y < right.y);
}

bool pointEqual(const Point& left, const Point& right) noexcept {
    return left.x == right.x && left.y == right.y;
}

double cross(const Point& origin, const Point& first, const Point& second) noexcept {
    return (first.x - origin.x) * (second.y - origin.y)
        - (first.y - origin.y) * (second.x - origin.x);
}

std::vector<Point> convexHull(std::vector<Point> points) {
    points.erase(std::remove_if(points.begin(), points.end(), [](const Point& point) {
        return !std::isfinite(point.x) || !std::isfinite(point.y);
    }), points.end());
    std::sort(points.begin(), points.end(), pointLess);
    points.erase(std::unique(points.begin(), points.end(), pointEqual), points.end());
    if (points.size() < 3U) return points;
    std::vector<Point> hull;
    hull.reserve(points.size() * 2U);
    for (const Point& point : points) {
        while (hull.size() >= 2U
               && cross(hull[hull.size() - 2U], hull.back(), point) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(point);
    }
    const std::size_t lowerSize = hull.size();
    for (auto iterator = points.rbegin() + 1; iterator != points.rend(); ++iterator) {
        while (hull.size() > lowerSize
               && cross(hull[hull.size() - 2U], hull.back(), *iterator) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(*iterator);
    }
    if (!hull.empty()) hull.pop_back();
    return hull;
}

double polygonArea(const OGRPolygon& polygon) noexcept {
    return OGR_G_Area(reinterpret_cast<OGRGeometryH>(const_cast<OGRPolygon*>(&polygon)));
}

const OGRPolygon* largestPolygon(const OGRGeometry* geometry) noexcept {
    if (geometry == nullptr || geometry->IsEmpty()) return nullptr;
    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    if (type == wkbPolygon) return dynamic_cast<const OGRPolygon*>(geometry);
    if (type != wkbMultiPolygon) return nullptr;
    const auto* collection = dynamic_cast<const OGRGeometryCollection*>(geometry);
    if (collection == nullptr || collection->getNumGeometries() == 0) return nullptr;
    const OGRPolygon* largest = nullptr;
    double largestArea = -std::numeric_limits<double>::infinity();
    for (int index = 0; index < collection->getNumGeometries(); ++index) {
        const auto* polygon = dynamic_cast<const OGRPolygon*>(collection->getGeometryRef(index));
        if (polygon == nullptr) continue;
        const double area = polygonArea(*polygon);
        if (area > largestArea) {
            largestArea = area;
            largest = polygon;
        }
    }
    return largest;
}

double footprintArea(const OGRGeometry* geometry) noexcept {
    const OGRPolygon* polygon = largestPolygon(geometry);
    return polygon == nullptr ? std::numeric_limits<double>::quiet_NaN()
                              : polygonArea(*polygon);
}

double shortSideLength(const OGRGeometry* geometry) {
    const OGRPolygon* polygon = largestPolygon(geometry);
    if (polygon == nullptr) return std::numeric_limits<double>::quiet_NaN();
    const OGRLinearRing* exterior = polygon->getExteriorRing();
    if (exterior == nullptr || exterior->getNumPoints() < 4) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(exterior->getNumPoints()));
    for (int index = 0; index < exterior->getNumPoints(); ++index) {
        points.push_back({exterior->getX(index), exterior->getY(index)});
    }
    const std::vector<Point> hull = convexHull(std::move(points));
    if (hull.size() < 3U) return std::numeric_limits<double>::quiet_NaN();

    double minimumArea = std::numeric_limits<double>::infinity();
    double result = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t edge = 0; edge < hull.size(); ++edge) {
        const Point& first = hull[edge];
        const Point& second = hull[(edge + 1U) % hull.size()];
        const double dx = second.x - first.x;
        const double dy = second.y - first.y;
        const double length = std::hypot(dx, dy);
        if (!(length > 0.0)) continue;
        const double ux = dx / length;
        const double uy = dy / length;
        const double vx = -uy;
        const double vy = ux;
        double uMin = std::numeric_limits<double>::infinity();
        double uMax = -std::numeric_limits<double>::infinity();
        double vMin = std::numeric_limits<double>::infinity();
        double vMax = -std::numeric_limits<double>::infinity();
        for (const Point& point : hull) {
            const double u = point.x * ux + point.y * uy;
            const double v = point.x * vx + point.y * vy;
            uMin = std::min(uMin, u); uMax = std::max(uMax, u);
            vMin = std::min(vMin, v); vMax = std::max(vMax, v);
        }
        const double width = uMax - uMin;
        const double height = vMax - vMin;
        const double area = width * height;
        if (area < minimumArea) {
            minimumArea = area;
            result = std::min(width, height);
        }
    }
    return result;
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

double numericField(const OGRFeature& feature, int index) noexcept {
    if (!feature.IsFieldSetAndNotNull(index)) return std::numeric_limits<double>::quiet_NaN();
    const char* raw = feature.GetFieldAsString(index);
    if (raw == nullptr) return std::numeric_limits<double>::quiet_NaN();
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw) return std::numeric_limits<double>::quiet_NaN();
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end != '\0' || errno == ERANGE) return std::numeric_limits<double>::quiet_NaN();
    return value;
}

std::vector<FeatureRecord> readFeatures(OGRLayer& layer) {
    OGRFeatureDefn* definition = layer.GetLayerDefn();
    if (definition == nullptr) throw std::runtime_error("Input shapefile has no layer definition.");
    std::vector<FeatureRecord> records;
    layer.ResetReading();
    while (OGRFeature* raw = layer.GetNextFeature()) {
        FeaturePtr feature(raw);
        OGRGeometry* source = feature->GetGeometryRef();
        if (source == nullptr) continue;
        const OGRwkbGeometryType sourceType = wkbFlatten(source->getGeometryType());
        if (sourceType != wkbPolygon && sourceType != wkbMultiPolygon) continue;
        GeometryPtr geometry(source->clone());
        if (GeometryPtr repaired(source->Buffer(0.0)); repaired) geometry = std::move(repaired);
        FeatureRecord record;
        record.geometry = std::move(geometry);
        record.numericFields.reserve(static_cast<std::size_t>(definition->GetFieldCount()));
        for (int index = 0; index < definition->GetFieldCount(); ++index) {
            record.numericFields.push_back(numericField(*feature, index));
        }
        records.push_back(std::move(record));
    }
    return records;
}

int chooseHeightField(const OGRFeatureDefn& definition,
                      const std::vector<FeatureRecord>& records) {
    static constexpr std::array<std::string_view, 10> exact{{
        "height", "hgt", "building_height", "bldg_height", "bldg_h",
        "bldgheight", "roof_height", "roof_h", "eave_height", "z"}};
    const auto hasFinite = [&](int field) {
        return std::any_of(records.begin(), records.end(), [&](const FeatureRecord& record) {
            return std::isfinite(record.numericFields[static_cast<std::size_t>(field)]);
        });
    };
    for (int field = 0; field < definition.GetFieldCount(); ++field) {
        const OGRFieldDefn* item = definition.GetFieldDefn(field);
        const std::string name = lowerAscii(item == nullptr ? std::string{} : item->GetNameRef());
        if (std::find(exact.begin(), exact.end(), name) != exact.end() && hasFinite(field)) {
            return field;
        }
    }
    for (int field = 0; field < definition.GetFieldCount(); ++field) {
        const OGRFieldDefn* item = definition.GetFieldDefn(field);
        const std::string name = lowerAscii(item == nullptr ? std::string{} : item->GetNameRef());
        if ((name.find("height") != std::string::npos || name.find("hgt") != std::string::npos)
            && hasFinite(field)) {
            return field;
        }
    }
    return -1;
}

GeometryPtr geometryUnion(const std::vector<FeatureRecord>& records) {
    GeometryPtr result;
    for (const FeatureRecord& record : records) {
        if (record.geometry == nullptr || record.geometry->IsEmpty()) continue;
        if (!result) {
            result.reset(record.geometry->clone());
        } else {
            GeometryPtr united(result->Union(record.geometry.get()));
            if (!united) {
                throw std::runtime_error("Could not form the geometry union for CRS selection.");
            }
            result = std::move(united);
        }
    }
    if (!result) throw std::runtime_error("Could not form the geometry union for CRS selection.");
    return result;
}

int utmEpsg(double longitude, double latitude) {
    const int zone = static_cast<int>(std::floor((longitude + 180.0) / 6.0)) + 1;
    return (latitude >= 0.0 ? 32600 : 32700) + zone;
}

std::string chooseMetricCrs(std::vector<FeatureRecord>& records,
                            const OGRSpatialReference* layerReference) {
    if (layerReference == nullptr || layerReference->IsEmpty()) {
        throw std::runtime_error(
            "Input data has no CRS. Please set a CRS before measuring lengths.");
    }
    std::unique_ptr<OGRSpatialReference> source(layerReference->Clone());
    if (!source) throw std::runtime_error("Could not clone input CRS.");
#if GDAL_VERSION_NUM >= 3000000
    source->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif
    if (!source->IsGeographic()) {
        return "Using existing projected CRS for metric lengths.";
    }

    GeometryPtr united = geometryUnion(records);
    OGRPoint centroid;
    if (united->Centroid(&centroid) != OGRERR_NONE
        || !std::isfinite(centroid.getX()) || !std::isfinite(centroid.getY())) {
        throw std::runtime_error("Could not calculate the geographic union centroid.");
    }
    const int epsg = utmEpsg(centroid.getX(), centroid.getY());
    OGRSpatialReference target;
    if (target.importFromEPSG(epsg) != OGRERR_NONE) {
        throw std::runtime_error("Could not create metric CRS EPSG:" + std::to_string(epsg));
    }
#if GDAL_VERSION_NUM >= 3000000
    target.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif
    TransformPtr transformation(OGRCreateCoordinateTransformation(source.get(), &target));
    if (!transformation) {
        throw std::runtime_error("Could not create transformation to EPSG:" + std::to_string(epsg));
    }
    for (FeatureRecord& record : records) {
        if (record.geometry != nullptr
            && record.geometry->transform(transformation.get()) != OGRERR_NONE) {
            throw std::runtime_error("Could not reproject a building geometry to EPSG:"
                                     + std::to_string(epsg));
        }
    }
    return "Reprojected from geographic CRS to EPSG:" + std::to_string(epsg)
        + " for metric lengths.";
}

struct Color final {
    unsigned char red = 255;
    unsigned char green = 255;
    unsigned char blue = 255;
};
static_assert(sizeof(Color) == 3U);

class Canvas final {
public:
    Canvas(int width, int height)
        : width_(width), height_(height), pixels_(static_cast<std::size_t>(width)
                                                   * static_cast<std::size_t>(height)) {}

    void pixel(int x, int y, Color color) noexcept {
        if (x < 0 || y < 0 || x >= width_ || y >= height_) return;
        pixels_[static_cast<std::size_t>(y) * static_cast<std::size_t>(width_)
                + static_cast<std::size_t>(x)] = color;
    }

    void fill(int left, int top, int right, int bottom, Color color) noexcept {
        left = std::clamp(left, 0, width_); right = std::clamp(right, 0, width_);
        top = std::clamp(top, 0, height_); bottom = std::clamp(bottom, 0, height_);
        for (int y = top; y < bottom; ++y) {
            for (int x = left; x < right; ++x) pixel(x, y, color);
        }
    }

    void line(int x0, int y0, int x1, int y1, Color color, int thickness = 1) noexcept {
        const int dx = std::abs(x1 - x0); const int sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0); const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        for (;;) {
            fill(x0 - thickness / 2, y0 - thickness / 2,
                 x0 + (thickness + 1) / 2, y0 + (thickness + 1) / 2, color);
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
    int width_;
    int height_;
    std::vector<Color> pixels_;
};

std::array<unsigned char, 7> glyph(char character) noexcept {
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
    case '.': return {0,0,0,0,0,6,6}; case '-': return {0,0,0,31,0,0,0};
    case '/': return {1,2,2,4,8,8,16}; case '(': return {2,4,8,8,8,4,2};
    case ')': return {8,4,2,2,2,4,8}; case ' ': return {};
    default: return {};
    }
}

void text(Canvas& canvas, int x, int y, std::string_view value, int scale,
          Color color = {35, 35, 35}) noexcept {
    for (char character : value) {
        const auto rows = glyph(character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[static_cast<std::size_t>(row)] & (1U << (4 - column))) != 0U) {
                    canvas.fill(x + column * scale, y + row * scale,
                                x + (column + 1) * scale, y + (row + 1) * scale, color);
                }
            }
        }
        x += 6 * scale;
    }
}

void writeImage(const fs::path& path, const Canvas& canvas) {
    const std::string extension = lowerAscii(pathUtf8(path.extension()));
    const char* driverName = nullptr;
    if (extension == ".png") {
        driverName = "PNG";
    } else if (extension == ".jpg" || extension == ".jpeg") {
        driverName = "JPEG";
    } else {
        throw std::runtime_error("Unsupported output image format '" + extension
                                 + "'; supported formats are .png, .jpg, and .jpeg.");
    }
    GDALDriver* memoryDriver = GetGDALDriverManager()->GetDriverByName("MEM");
    GDALDriver* imageDriver = GetGDALDriverManager()->GetDriverByName(driverName);
    if (memoryDriver == nullptr || imageDriver == nullptr) {
        throw std::runtime_error(std::string("GDAL MEM/") + driverName
                                 + " drivers are unavailable.");
    }
    DatasetPtr memory(memoryDriver->Create("", canvas.width(), canvas.height(), 3, GDT_Byte, nullptr));
    if (!memory) throw std::runtime_error("Could not allocate the PNG canvas.");
    std::array<int, 3> bands{1, 2, 3};
    auto* bytes = const_cast<Color*>(canvas.pixels().data());
    if (memory->RasterIO(GF_Write, 0, 0, canvas.width(), canvas.height(), bytes,
                         canvas.width(), canvas.height(), GDT_Byte, 3, bands.data(),
                         static_cast<GSpacing>(sizeof(Color)),
                         static_cast<GSpacing>(sizeof(Color) * canvas.width()), 1, nullptr)
        != CE_None) {
        throw std::runtime_error("Could not populate the PNG canvas.");
    }
    memory->GetRasterBand(1)->SetColorInterpretation(GCI_RedBand);
    memory->GetRasterBand(2)->SetColorInterpretation(GCI_GreenBand);
    memory->GetRasterBand(3)->SetColorInterpretation(GCI_BlueBand);
    DatasetPtr output(imageDriver->CreateCopy(pathUtf8(path).c_str(), memory.get(), FALSE,
                                               nullptr, nullptr, nullptr));
    if (!output) throw std::runtime_error("Could not save image: " + pathUtf8(path));
}

void plotDistribution(const BuildingScaleDistribution& distribution,
                      const fs::path& outputPath) {
    constexpr int width = 3000;
    constexpr int height = 900;
    constexpr int left = 260;
    constexpr int right = 2710;
    constexpr int top = 85;
    constexpr int bottom = 710;
    Canvas canvas(width, height);
    const Color grid{180, 180, 180};
    const Color axes{40, 40, 40};
    const Color pdfColor{43, 120, 177};
    const Color cdfColor{249, 129, 44};
    for (int step = 0; step <= 5; ++step) {
        const int y = top + (bottom - top) * step / 5;
        for (int x = left; x <= right; x += 16) canvas.line(x, y, std::min(x + 8, right), y, grid);
        const int x = left + (right - left) * step / 5;
        for (int yDash = top; yDash <= bottom; yDash += 16) {
            canvas.line(x, yDash, x, std::min(yDash + 8, bottom), grid);
        }
    }
    canvas.line(left, top, left, bottom, axes, 3);
    canvas.line(right, top, right, bottom, axes, 3);
    canvas.line(left, bottom, right, bottom, axes, 3);

    const double maximumX = distribution.valuesDescending.front();
    const double maximumPdf = *std::max_element(distribution.probabilityDensity.begin(),
                                                distribution.probabilityDensity.end());
    const auto mapX = [&](double value) {
        return left + static_cast<int>(std::lround((maximumX - value) / maximumX * (right - left)));
    };
    const auto mapPdf = [&](double value) {
        return bottom - static_cast<int>(std::lround(value / maximumPdf * (bottom - top)));
    };
    const auto mapCdf = [&](double value) {
        return bottom - static_cast<int>(std::lround(value * (bottom - top)));
    };
    for (double reference : {2.0, 5.0, 10.0, 20.0, 50.0}) {
        if (reference > maximumX) continue;
        const int x = mapX(reference);
        for (int y = top; y <= bottom; y += 14) canvas.line(x, y, x, std::min(y + 7, bottom), axes);
    }
    for (std::size_t index = 1; index < distribution.binCenters.size(); ++index) {
        canvas.line(mapX(distribution.binCenters[index - 1U]),
                    mapPdf(distribution.probabilityDensity[index - 1U]),
                    mapX(distribution.binCenters[index]),
                    mapPdf(distribution.probabilityDensity[index]), pdfColor, 5);
    }
    if (distribution.binCenters.size() == 1U) {
        canvas.fill(mapX(distribution.binCenters.front()) - 3,
                    mapPdf(distribution.probabilityDensity.front()) - 3,
                    mapX(distribution.binCenters.front()) + 4,
                    mapPdf(distribution.probabilityDensity.front()) + 4, pdfColor);
    }
    for (std::size_t index = 1; index < distribution.valuesDescending.size(); ++index) {
        canvas.line(mapX(distribution.valuesDescending[index - 1U]),
                    mapCdf(distribution.cumulativeWeights[index - 1U]),
                    mapX(distribution.valuesDescending[index]),
                    mapCdf(distribution.cumulativeWeights[index]), cdfColor, 5);
    }
    canvas.line(520, 120, 650, 120, pdfColor, 6);
    text(canvas, 675, 102, "PROBABILITY DENSITY FUNCTION", 3);
    canvas.line(1500, 120, 1630, 120, cdfColor, 6);
    text(canvas, 1655, 102, "CUMULATIVE DISTRIBUTION FUNCTION", 3);
    text(canvas, 1130, 790, "SHORT-SIDE LENGTH (M)", 4);
    text(canvas, 25, 35, "PROBABILITY DENSITY FUNCTION", 2);
    text(canvas, 2220, 35, "CUMULATIVE DISTRIBUTION FUNCTION", 2);
    writeImage(outputPath, canvas);
}

int runWithGdal(const Arguments& arguments, std::ostream& output) {
    std::error_code error;
    if (!fs::exists(arguments.shapefile, error) || error) {
        throw std::runtime_error("Input shapefile not found: " + pathUtf8(arguments.shapefile));
    }
    fs::path defaultFilename = arguments.shapefile.stem();
    defaultFilename += "_short_side_cdf.png";
    const fs::path outputPath = arguments.output && !arguments.output->empty()
        ? *arguments.output : arguments.shapefile.parent_path() / defaultFilename;
    fs::path imagePath = outputPath;
    if (imagePath.extension().empty()) imagePath += ".png";
    if ((fs::exists(outputPath, error) && !error)
        || (imagePath != outputPath && fs::exists(imagePath, error) && !error)) {
        throw std::runtime_error("Output image already exists; refusing to overwrite: "
                                 + pathUtf8(imagePath));
    }

    GDALAllRegister();
    DatasetPtr dataset(static_cast<GDALDataset*>(GDALOpenEx(
        pathUtf8(arguments.shapefile).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
        nullptr, nullptr, nullptr)));
    if (!dataset) throw std::runtime_error("Could not read input shapefile: "
                                           + pathUtf8(arguments.shapefile));
    OGRLayer* layer = dataset->GetLayer(0);
    if (layer == nullptr) throw std::runtime_error("Input shapefile contains no readable layer.");
    OGRFeatureDefn* definition = layer->GetLayerDefn();
    if (definition == nullptr) throw std::runtime_error("Input shapefile has no layer definition.");
    std::vector<FeatureRecord> records = readFeatures(*layer);
    if (records.empty()) {
        throw std::runtime_error("No Polygon/MultiPolygon geometries found in the shapefile.");
    }

    output << chooseMetricCrs(records, layer->GetSpatialRef()) << '\n';
    const int heightField = chooseHeightField(*definition, records);
    if (heightField < 0) {
        output << "Height field: not found, using 1.0 for all features.\n";
    } else {
        output << "Height field: " << definition->GetFieldDefn(heightField)->GetNameRef() << '\n';
    }

    std::vector<double> lengths;
    std::vector<double> weights;
    lengths.reserve(records.size());
    weights.reserve(records.size());
    for (const FeatureRecord& record : records) {
        const double length = shortSideLength(record.geometry.get());
        const double area = footprintArea(record.geometry.get());
        const double heightValue = heightField < 0
            ? 1.0 : record.numericFields[static_cast<std::size_t>(heightField)];
        const double weight = area * heightValue;
        if (std::isfinite(length) && std::isfinite(weight)
            && length > 0.0 && weight > 0.0) {
            lengths.push_back(length);
            weights.push_back(weight);
        }
    }
    if (lengths.empty()) {
        throw std::runtime_error(
            "No valid weighted short-side lengths computed (check geometries, height field, and CRS).");
    }
    const BuildingScaleDistribution distribution =
        calculateBuildingScaleDistribution(lengths, weights);
    plotDistribution(distribution, imagePath);

    output << "Output saved to: " << pathUtf8(outputPath) << '\n'
           << "Count: " << lengths.size() << '\n'
           << std::fixed << std::setprecision(3)
           << "Min/Median/Max (m): " << *std::min_element(lengths.begin(), lengths.end())
           << " / " << median(lengths)
           << " / " << *std::max_element(lengths.begin(), lengths.end()) << '\n';
    return 0;
}

#endif // LUW_HAS_GDAL

} // namespace

BuildingScaleDistribution calculateBuildingScaleDistribution(
    const std::vector<double>& values,
    const std::vector<double>& weights) {
    if (values.empty() || values.size() != weights.size()) {
        throw std::invalid_argument("Building-scale values and weights must have equal non-zero size.");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index]) || !std::isfinite(weights[index])
            || weights[index] <= 0.0) {
            throw std::invalid_argument("Building-scale values/weights must be finite and weights positive.");
        }
    }
    const double totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(totalWeight > 0.0) || !std::isfinite(totalWeight)) {
        throw std::invalid_argument("Building-scale total weight must be finite and positive.");
    }

    BuildingScaleDistribution result;
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        if (values[left] != values[right]) return values[left] > values[right];
        // np.argsort(values)[::-1] reverses the ordinary ascending tie order
        // for the small/repeated inputs used by the original plotting path.
        return left > right;
    });
    result.valuesDescending.reserve(values.size());
    result.cumulativeWeights.reserve(values.size());
    double cumulative = 0.0;
    for (const std::size_t index : order) {
        result.valuesDescending.push_back(values[index]);
        cumulative += weights[index];
        result.cumulativeWeights.push_back(cumulative / totalWeight);
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double dataMin = sorted.front();
    const double dataMax = sorted.back();
    double edgeMin = dataMin;
    double edgeMax = dataMax;
    if (edgeMin == edgeMax) {
        edgeMin -= 0.5;
        edgeMax += 0.5;
    }
    const double iqr = percentile(sorted, 0.75) - percentile(sorted, 0.25);
    const double width = 2.0 * iqr / std::cbrt(static_cast<double>(values.size()));
    std::size_t binCount = 1U;
    if (width > 0.0 && std::isfinite(width)) {
        const double raw = std::ceil((edgeMax - edgeMin) / width);
        if (raw > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error("Freedman-Diaconis histogram has too many bins.");
        }
        binCount = std::max<std::size_t>(1U, static_cast<std::size_t>(raw));
    }
    result.binEdges.resize(binCount + 1U);
    for (std::size_t index = 0; index <= binCount; ++index) {
        result.binEdges[index] = edgeMin
            + (edgeMax - edgeMin) * static_cast<double>(index) / static_cast<double>(binCount);
    }
    result.binEdges.back() = edgeMax;
    std::vector<double> weightedBins(binCount, 0.0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::size_t bin = 0;
        if (values[index] == edgeMax) {
            bin = binCount - 1U;
        } else {
            const auto upper = std::upper_bound(result.binEdges.begin(), result.binEdges.end(), values[index]);
            bin = static_cast<std::size_t>(std::distance(result.binEdges.begin(), upper) - 1);
            bin = std::min(bin, binCount - 1U);
        }
        weightedBins[bin] += weights[index];
    }
    result.binCenters.resize(binCount);
    result.probabilityDensity.resize(binCount);
    for (std::size_t bin = 0; bin < binCount; ++bin) {
        const double binWidth = result.binEdges[bin + 1U] - result.binEdges[bin];
        result.binCenters[bin] = (result.binEdges[bin] + result.binEdges[bin + 1U]) / 2.0;
        result.probabilityDensity[bin] = weightedBins[bin] / (totalWeight * binWidth);
    }
    return result;
}

int run_buildingscale(const CommandArguments& arguments,
                      std::ostream& output,
                      std::ostream& error) {
    int status = 0;
    const std::optional<Arguments> parsed = parseArguments(arguments, output, error, status);
    if (!parsed || parsed->help) return status;
#if LUW_HAS_GDAL
    try {
        return runWithGdal(*parsed, output);
    } catch (const std::exception& exception) {
        error << "ERROR: buildingscale failed: " << exception.what() << '\n';
        return 1;
    }
#else
    error << "ERROR: Native buildingscale requires GDAL/OGR with GEOS support; rebuild with "
             "LUW_ENABLE_GEOSPATIAL=ON and a discoverable GDAL installation.\n";
    return 1;
#endif
}

} // namespace luw::cli
