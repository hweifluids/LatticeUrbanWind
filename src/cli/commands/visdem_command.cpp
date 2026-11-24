#include "luw/visdem_command.hpp"

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <cpl_error.h>
#include <gdal_priv.h>
#include <gdalwarper.h>
#include <ogr_spatialref.h>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace luw::cli {

namespace {

namespace fs = std::filesystem;

constexpr std::string_view default_dem_filename =
    "AST14DEM_00408232025023230_20251011022044.tif";

std::string replace_all(std::string value,
                        std::string_view needle,
                        std::string_view replacement) {
    std::size_t offset = 0;
    while ((offset = value.find(needle, offset)) != std::string::npos) {
        value.replace(offset, needle.size(), replacement);
        offset += replacement.size();
    }
    return value;
}

struct OutputPaths {
    std::string overview_text;
    std::string surface_text;
    fs::path overview;
    fs::path surface;
};

fs::path path_from_utf8(std::string_view value) {
#if defined(__cpp_lib_char8_t)
    const auto* begin = reinterpret_cast<const char8_t*>(value.data());
    return fs::path(std::u8string(begin, begin + value.size()));
#else
    return fs::u8path(value.begin(), value.end());
#endif
}

std::string path_to_utf8(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const std::u8string value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
#else
    return path.u8string();
#endif
}

OutputPaths output_paths(const std::string& input) {
    OutputPaths result;
    result.overview_text = replace_all(input, ".tif", "_visualization.png");
    result.surface_text = replace_all(input, ".tif", "_3d_visualization.png");
    result.overview = path_from_utf8(result.overview_text);
    result.surface = path_from_utf8(result.surface_text);
    return result;
}

#if LUW_HAS_GDAL

struct DatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) {
            GDALClose(dataset);
        }
    }
};

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;

class QuietGdalErrors final {
public:
    QuietGdalErrors() {
        CPLErrorReset();
        CPLPushErrorHandler(CPLQuietErrorHandler);
    }
    ~QuietGdalErrors() { CPLPopErrorHandler(); }
    QuietGdalErrors(const QuietGdalErrors&) = delete;
    QuietGdalErrors& operator=(const QuietGdalErrors&) = delete;
};

std::string last_gdal_error(std::string fallback) {
    const char* message = CPLGetLastErrorMsg();
    return message != nullptr && *message != '\0' ? message : std::move(fallback);
}

std::string python_float(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value < 0.0 ? "-inf" : "inf";
    std::array<char, 64> buffer{};
    const auto [end, error] =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (error != std::errc{}) {
        std::ostringstream stream;
        stream << std::setprecision(17) << value;
        return stream.str();
    }
    std::string result(buffer.data(), end);
    const std::size_t exponent_marker = result.find_first_of("eE");
    if (exponent_marker != std::string::npos) {
        int exponent = 0;
        std::string_view exponent_text(result.data() + exponent_marker + 1,
                                       result.size() - exponent_marker - 1);
        if (!exponent_text.empty() && exponent_text.front() == '+') {
            exponent_text.remove_prefix(1);
        }
        const auto [parsed_end, parsed_error] = std::from_chars(
            exponent_text.data(), exponent_text.data() + exponent_text.size(),
            exponent);
        if (parsed_error == std::errc{}
            && parsed_end == exponent_text.data() + exponent_text.size()
            && exponent >= -4 && exponent < 16) {
            const auto [fixed_end, fixed_error] = std::to_chars(
                buffer.data(), buffer.data() + buffer.size(), value,
                std::chars_format::fixed);
            if (fixed_error == std::errc{}) {
                result.assign(buffer.data(), fixed_end);
            }
        }
    }
    if (result.find_first_of(".eE") == std::string::npos) result += ".0";
    return result;
}

std::string fixed(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string crs_text(const OGRSpatialReference* source) {
    if (source == nullptr) return "None";
    std::unique_ptr<OGRSpatialReference> reference(source->Clone());
    if (!reference) return "None";
    (void)reference->AutoIdentifyEPSG();
    const char* authority = reference->GetAuthorityName(nullptr);
    const char* code = reference->GetAuthorityCode(nullptr);
    if (authority != nullptr && code != nullptr) {
        return std::string(authority) + ':' + code;
    }
    char* wkt = nullptr;
    if (reference->exportToWkt(&wkt) == OGRERR_NONE && wkt != nullptr) {
        std::string result(wkt);
        CPLFree(wkt);
        return result;
    }
    return "None";
}

struct Bounds {
    double left = 0.0;
    double bottom = 0.0;
    double right = 0.0;
    double top = 0.0;
};

Bounds raster_bounds(const double transform[6], int width, int height) {
    Bounds bounds{std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()};
    for (const auto& corner : std::array<std::array<double, 2>, 4>{{
             {0.0, 0.0}, {static_cast<double>(width), 0.0},
             {0.0, static_cast<double>(height)},
             {static_cast<double>(width), static_cast<double>(height)}}}) {
        double x = 0.0;
        double y = 0.0;
        GDALApplyGeoTransform(const_cast<double*>(transform), corner[0], corner[1],
                              &x, &y);
        bounds.left = std::min(bounds.left, x);
        bounds.right = std::max(bounds.right, x);
        bounds.bottom = std::min(bounds.bottom, y);
        bounds.top = std::max(bounds.top, y);
    }
    return bounds;
}

struct DemGrid {
    int width = 0;
    int height = 0;
    std::array<double, 6> transform{};
    std::vector<float> values;
};

DemGrid reproject_to_wgs84(GDALDataset& input,
                           GDALRasterBand& band,
                           const double source_transform[6]) {
    GDALDriver* memory_driver =
        GetGDALDriverManager()->GetDriverByName("MEM");
    if (memory_driver == nullptr) {
        throw std::runtime_error("GDAL MEM driver is unavailable.");
    }
    const int width = input.GetRasterXSize();
    const int height = input.GetRasterYSize();
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("DEM has no raster pixels.");
    }
    const std::size_t cell_count = static_cast<std::size_t>(width)
        * static_cast<std::size_t>(height);
    if (cell_count / static_cast<std::size_t>(width)
        != static_cast<std::size_t>(height)) {
        throw std::overflow_error("DEM dimensions exceed addressable memory.");
    }

    std::vector<double> source_values(cell_count);
    if (band.RasterIO(GF_Read, 0, 0, width, height, source_values.data(), width,
                      height, GDT_Float64, 0, 0, nullptr) != CE_None) {
        throw std::runtime_error(last_gdal_error("Could not read DEM band 1."));
    }
    int has_nodata = 0;
    const double nodata = band.GetNoDataValue(&has_nodata);
    std::vector<float> filtered(cell_count);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t index = 0; index < cell_count; ++index) {
        const double value = source_values[index];
        filtered[index] = value == -9999.0
                || (has_nodata != 0 && value == nodata)
            ? nan
            : static_cast<float>(value);
    }

    DatasetPtr source(memory_driver->Create("", width, height, 1, GDT_Float32,
                                            nullptr));
    if (!source || source->SetGeoTransform(const_cast<double*>(source_transform))
                       != CE_None) {
        throw std::runtime_error("Could not create the in-memory source DEM.");
    }
    const char* source_projection = input.GetProjectionRef();
    if (source_projection == nullptr || *source_projection == '\0') {
        throw std::runtime_error("Missing src_crs.");
    }
    if (source->SetProjection(source_projection) != CE_None) {
        throw std::runtime_error("Could not assign the source DEM CRS.");
    }
    GDALRasterBand* source_band = source->GetRasterBand(1);
    if (source_band == nullptr
        || source_band->SetNoDataValue(
               static_cast<double>(std::numeric_limits<float>::quiet_NaN()))
               != CE_None
        || source_band->RasterIO(GF_Write, 0, 0, width, height, filtered.data(),
                                 width, height, GDT_Float32, 0, 0, nullptr)
               != CE_None) {
        throw std::runtime_error("Could not populate the in-memory source DEM.");
    }

    OGRSpatialReference target_reference;
    if (target_reference.importFromEPSG(4326) != OGRERR_NONE) {
        throw std::runtime_error("Could not initialize EPSG:4326.");
    }
    target_reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    char* target_wkt_raw = nullptr;
    if (target_reference.exportToWkt(&target_wkt_raw) != OGRERR_NONE
        || target_wkt_raw == nullptr) {
        throw std::runtime_error("Could not export EPSG:4326.");
    }
    const std::unique_ptr<char, decltype(&CPLFree)> target_wkt(target_wkt_raw,
                                                               &CPLFree);

    DatasetPtr warped(static_cast<GDALDataset*>(GDALAutoCreateWarpedVRT(
        reinterpret_cast<GDALDatasetH>(source.get()), source_projection,
        target_wkt.get(), GRA_Bilinear, 0.0, nullptr)));
    if (!warped) {
        throw std::runtime_error(last_gdal_error(
            "Could not calculate the EPSG:4326 reprojection grid."));
    }

    DemGrid result;
    result.width = warped->GetRasterXSize();
    result.height = warped->GetRasterYSize();
    if (result.width <= 0 || result.height <= 0
        || warped->GetGeoTransform(result.transform.data()) != CE_None) {
        throw std::runtime_error("Invalid EPSG:4326 reprojection grid.");
    }
    const std::size_t result_count = static_cast<std::size_t>(result.width)
        * static_cast<std::size_t>(result.height);
    if (result_count / static_cast<std::size_t>(result.width)
        != static_cast<std::size_t>(result.height)) {
        throw std::overflow_error("Reprojected DEM dimensions exceed addressable memory.");
    }
    result.values.resize(result_count, nan);
    GDALRasterBand* warped_band = warped->GetRasterBand(1);
    if (warped_band == nullptr
        || warped_band->RasterIO(GF_Read, 0, 0, result.width, result.height,
                                 result.values.data(), result.width,
                                 result.height, GDT_Float32, 0, 0, nullptr)
               != CE_None) {
        throw std::runtime_error(last_gdal_error(
            "Could not read the reprojected DEM."));
    }
    return result;
}

struct Statistics {
    std::size_t count = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double standard_deviation = 0.0;
};

Statistics statistics(const std::vector<float>& values) {
    Statistics result;
    double sum = 0.0;
    result.minimum = std::numeric_limits<double>::infinity();
    result.maximum = -std::numeric_limits<double>::infinity();
    for (const float value : values) {
        if (std::isnan(value)) continue;
        ++result.count;
        result.minimum = std::min(result.minimum, static_cast<double>(value));
        result.maximum = std::max(result.maximum, static_cast<double>(value));
        sum += static_cast<double>(value);
    }
    if (result.count == 0) return result;
    result.mean = sum / static_cast<double>(result.count);
    double squared = 0.0;
    for (const float value : values) {
        if (std::isnan(value)) continue;
        const double delta = static_cast<double>(value) - result.mean;
        squared += delta * delta;
    }
    result.standard_deviation =
        std::sqrt(squared / static_cast<double>(result.count));
    return result;
}

std::vector<float> hillshade(const DemGrid& grid) {
    if (grid.width < 2 || grid.height < 2) {
        throw std::runtime_error(
            "Shape of array too small to calculate a numerical gradient.");
    }
    const double dx = std::abs(grid.transform[1]);
    const double dy = -std::abs(grid.transform[5]);
    if (!(dx > 0.0) || !(std::abs(dy) > 0.0)) {
        throw std::runtime_error("Reprojected DEM has zero pixel resolution.");
    }
    std::vector<float> result(grid.values.size());
    bool has_nan = false;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    const auto value = [&](int x, int y) {
        return 2.0 * static_cast<double>(grid.values[
            static_cast<std::size_t>(y) * static_cast<std::size_t>(grid.width)
            + static_cast<std::size_t>(x)]);
    };
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const double gradient_x = x == 0
                ? (value(1, y) - value(0, y)) / dx
                : (x == grid.width - 1
                       ? (value(x, y) - value(x - 1, y)) / dx
                       : (value(x + 1, y) - value(x - 1, y)) / (2.0 * dx));
            const double gradient_y = y == 0
                ? (value(x, 1) - value(x, 0)) / dy
                : (y == grid.height - 1
                       ? (value(x, y) - value(x, y - 1)) / dy
                       : (value(x, y + 1) - value(x, y - 1)) / (2.0 * dy));
            const double nx = -gradient_x;
            const double ny = -gradient_y;
            const double magnitude = std::sqrt(nx * nx + ny * ny + 1.0);
            const double intensity =
                (nx * -0.5 + ny * 0.5 + std::sqrt(0.5)) / magnitude;
            const std::size_t index = static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(grid.width)
                + static_cast<std::size_t>(x);
            result[index] = static_cast<float>(intensity);
            if (std::isnan(intensity)) {
                has_nan = true;
            } else {
                minimum = std::min(minimum, intensity);
                maximum = std::max(maximum, intensity);
            }
        }
    }
    // numpy.ndarray.min propagates a NaN.  In that case Matplotlib keeps the
    // unscaled dot product and only clips its finite cells.
    const bool stretch = !has_nan && maximum - minimum > 1.0e-6;
    for (float& intensity : result) {
        if (stretch) {
            intensity = static_cast<float>(
                (static_cast<double>(intensity) - minimum) / (maximum - minimum));
        }
        intensity = std::clamp(intensity, 0.0F, 1.0F);
    }
    return result;
}

struct Color {
    std::uint8_t red = 255;
    std::uint8_t green = 255;
    std::uint8_t blue = 255;
    std::uint8_t alpha = 255;
};

static_assert(sizeof(Color) == 4);

Color blend(Color bottom, Color top, double alpha) {
    const auto channel = [alpha](std::uint8_t below, std::uint8_t above) {
        return static_cast<std::uint8_t>(std::clamp(
            std::lround(static_cast<double>(below) * (1.0 - alpha)
                        + static_cast<double>(above) * alpha),
            0L, 255L));
    };
    return {channel(bottom.red, top.red), channel(bottom.green, top.green),
            channel(bottom.blue, top.blue), 255};
}

Color interpolate(Color left, Color right, double amount) {
    return blend(left, right, amount);
}

Color terrain_color(double normalized) {
    struct Stop {
        double position;
        Color color;
    };
    constexpr std::array<Stop, 6> stops{{
        {0.00, {51, 51, 153, 255}},
        {0.15, {0, 153, 255, 255}},
        {0.25, {0, 204, 102, 255}},
        {0.50, {255, 255, 153, 255}},
        {0.75, {128, 92, 84, 255}},
        {1.00, {255, 255, 255, 255}},
    }};
    normalized = std::clamp(normalized, 0.0, 1.0);
    for (std::size_t index = 1; index < stops.size(); ++index) {
        if (normalized <= stops[index].position) {
            const double fraction = (normalized - stops[index - 1U].position)
                / (stops[index].position - stops[index - 1U].position);
            return interpolate(stops[index - 1U].color, stops[index].color,
                               fraction);
        }
    }
    return stops.back().color;
}

class Canvas {
public:
    Canvas(int width, int height)
        : width_(width), height_(height) {
        if (width <= 0 || height <= 0
            || static_cast<std::size_t>(width)
                    > std::numeric_limits<std::size_t>::max()
                        / static_cast<std::size_t>(height)) {
            throw std::overflow_error("PNG canvas dimensions are invalid.");
        }
        pixels_.resize(static_cast<std::size_t>(width)
                       * static_cast<std::size_t>(height));
    }

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] std::vector<Color>& pixels() noexcept { return pixels_; }
    [[nodiscard]] const std::vector<Color>& pixels() const noexcept {
        return pixels_;
    }

    void pixel(int x, int y, Color color) noexcept {
        if (x < 0 || y < 0 || x >= width_ || y >= height_) return;
        pixels_[static_cast<std::size_t>(y) * static_cast<std::size_t>(width_)
                + static_cast<std::size_t>(x)] = color;
    }

    void fill(int left, int top, int right, int bottom, Color color) noexcept {
        left = std::clamp(left, 0, width_);
        right = std::clamp(right, 0, width_);
        top = std::clamp(top, 0, height_);
        bottom = std::clamp(bottom, 0, height_);
        for (int y = top; y < bottom; ++y) {
            std::fill(pixels_.begin()
                          + static_cast<std::ptrdiff_t>(y)
                              * static_cast<std::ptrdiff_t>(width_)
                          + left,
                      pixels_.begin()
                          + static_cast<std::ptrdiff_t>(y)
                              * static_cast<std::ptrdiff_t>(width_)
                          + right,
                      color);
        }
    }

    void line(int x0, int y0, int x1, int y1, Color color,
              int thickness = 1) noexcept {
        const int dx = std::abs(x1 - x0);
        const int sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0);
        const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        while (true) {
            fill(x0 - thickness / 2, y0 - thickness / 2,
                 x0 + (thickness + 1) / 2, y0 + (thickness + 1) / 2,
                 color);
            if (x0 == x1 && y0 == y1) break;
            const int doubled = 2 * error;
            if (doubled >= dy) {
                error += dy;
                x0 += sx;
            }
            if (doubled <= dx) {
                error += dx;
                y0 += sy;
            }
        }
    }

    void triangle(double ax, double ay, double bx, double by,
                  double cx, double cy, Color color) noexcept {
        const int left = std::max(0, static_cast<int>(std::floor(
                                         std::min({ax, bx, cx}))));
        const int right = std::min(width_ - 1, static_cast<int>(std::ceil(
                                                std::max({ax, bx, cx}))));
        const int top = std::max(0, static_cast<int>(std::floor(
                                        std::min({ay, by, cy}))));
        const int bottom = std::min(height_ - 1, static_cast<int>(std::ceil(
                                                  std::max({ay, by, cy}))));
        const double area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (area == 0.0) return;
        for (int y = top; y <= bottom; ++y) {
            for (int x = left; x <= right; ++x) {
                const double px = static_cast<double>(x) + 0.5;
                const double py = static_cast<double>(y) + 0.5;
                const double first =
                    ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
                const double second =
                    ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
                const double third = 1.0 - first - second;
                if (first >= -1.0e-9 && second >= -1.0e-9
                    && third >= -1.0e-9) {
                    pixel(x, y, color);
                }
            }
        }
    }

private:
    int width_;
    int height_;
    std::vector<Color> pixels_;
};

std::array<std::uint8_t, 7> glyph(char character) {
    switch (static_cast<char>(std::toupper(
        static_cast<unsigned char>(character)))) {
    case 'A': return {14,17,17,31,17,17,17}; case 'B': return {30,17,17,30,17,17,30};
    case 'C': return {14,17,16,16,16,17,14}; case 'D': return {30,17,17,17,17,17,30};
    case 'E': return {31,16,16,30,16,16,31}; case 'F': return {31,16,16,30,16,16,16};
    case 'G': return {14,17,16,23,17,17,15}; case 'H': return {17,17,17,31,17,17,17};
    case 'I': return {14,4,4,4,4,4,14}; case 'J': return {1,1,1,1,17,17,14};
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
    case '=': return {0,31,0,31,0,0,0}; case '%': return {17,2,4,8,17,0,0};
    default: return {};
    }
}

int text_width(std::string_view value, int scale) {
    return static_cast<int>(value.size()) * 6 * scale;
}

void draw_text(Canvas& canvas, int x, int y, std::string_view value,
               int scale, Color color = {35, 35, 35, 255}) {
    for (const char character : value) {
        const auto rows = glyph(character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[static_cast<std::size_t>(row)]
                     & (1U << (4 - column))) != 0U) {
                    canvas.fill(x + column * scale, y + row * scale,
                                x + (column + 1) * scale,
                                y + (row + 1) * scale, color);
                }
            }
        }
        x += 6 * scale;
    }
}

void draw_text_vertical(Canvas& canvas, int x, int y,
                        std::string_view value, int scale,
                        Color color = {35, 35, 35, 255}) {
    const int source_width = static_cast<int>(value.size()) * 6;
    for (std::size_t character_index = 0;
         character_index < value.size(); ++character_index) {
        const auto rows = glyph(value[character_index]);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[static_cast<std::size_t>(row)]
                     & (1U << (4 - column))) == 0U) {
                    continue;
                }
                const int source_x =
                    static_cast<int>(character_index) * 6 + column;
                const int destination_x = x + row * scale;
                const int destination_y =
                    y + (source_width - 1 - source_x) * scale;
                canvas.fill(destination_x, destination_y,
                            destination_x + scale,
                            destination_y + scale, color);
            }
        }
    }
}

float bilinear(const std::vector<float>& values, int width, int height,
               double x, double y) {
    x = std::clamp(x, 0.0, static_cast<double>(width - 1));
    y = std::clamp(y, 0.0, static_cast<double>(height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float samples[4]{
        values[static_cast<std::size_t>(y0) * static_cast<std::size_t>(width)
               + static_cast<std::size_t>(x0)],
        values[static_cast<std::size_t>(y0) * static_cast<std::size_t>(width)
               + static_cast<std::size_t>(x1)],
        values[static_cast<std::size_t>(y1) * static_cast<std::size_t>(width)
               + static_cast<std::size_t>(x0)],
        values[static_cast<std::size_t>(y1) * static_cast<std::size_t>(width)
               + static_cast<std::size_t>(x1)]};
    if (std::any_of(std::begin(samples), std::end(samples),
                    [](float value) { return std::isnan(value); })) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    const double upper = static_cast<double>(samples[0]) * (1.0 - fx)
        + static_cast<double>(samples[1]) * fx;
    const double lower = static_cast<double>(samples[2]) * (1.0 - fx)
        + static_cast<double>(samples[3]) * fx;
    return static_cast<float>(upper * (1.0 - fy) + lower * fy);
}

void draw_axes(Canvas& canvas, int left, int top, int right, int bottom,
               const Bounds& bounds) {
    const Color axes{55, 55, 55, 255};
    const Color grid{205, 205, 205, 255};
    for (int tick = 0; tick <= 4; ++tick) {
        const int x = left + (right - left) * tick / 4;
        const int y = top + (bottom - top) * tick / 4;
        for (int dash = top; dash <= bottom; dash += 28) {
            canvas.line(x, dash, x, std::min(dash + 12, bottom), grid, 2);
        }
        for (int dash = left; dash <= right; dash += 28) {
            canvas.line(dash, y, std::min(dash + 12, right), y, grid, 2);
        }
        const std::string longitude = fixed(
            bounds.left + (bounds.right - bounds.left) * tick / 4.0, 3);
        const std::string latitude = fixed(
            bounds.top - (bounds.top - bounds.bottom) * tick / 4.0, 3);
        draw_text(canvas, x - text_width(longitude, 2) / 2, bottom + 16,
                  longitude, 2);
        draw_text(canvas, left - text_width(latitude, 2) - 18, y - 7,
                  latitude, 2);
    }
    canvas.line(left, top, left, bottom, axes, 3);
    canvas.line(left, bottom, right, bottom, axes, 3);
    canvas.line(right, top, right, bottom, axes, 2);
    canvas.line(left, top, right, top, axes, 2);
    const std::string x_label = "LONGITUDE (DEG)";
    draw_text(canvas, (left + right - text_width(x_label, 3)) / 2,
              bottom + 62, x_label, 3);
    const int y_label_height = text_width("LATITUDE", 3);
    draw_text_vertical(canvas, left - 175,
                       (top + bottom - y_label_height) / 2,
                       "LATITUDE", 3);
}

enum class MapKind { Elevation, Hillshade, Overlay };

void draw_map_panel(Canvas& canvas, const DemGrid& grid,
                    const std::vector<float>& shade, const Statistics& stats,
                    const Bounds& bounds, int panel_left, int panel_top,
                    int panel_right, int panel_bottom, std::string_view title,
                    MapKind kind, bool colorbar) {
    const int plot_left = panel_left + 190;
    const int plot_top = panel_top + 105;
    const int plot_right = panel_right - (colorbar ? 240 : 80);
    const int plot_bottom = panel_bottom - 145;
    draw_text(canvas,
              (panel_left + panel_right - text_width(title, 4)) / 2,
              panel_top + 24, title, 4, {20, 20, 20, 255});
    const int draw_width = std::max(1, plot_right - plot_left);
    const int draw_height = std::max(1, plot_bottom - plot_top);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int py = 0; py < draw_height; ++py) {
        const double source_y = draw_height == 1 ? 0.0
            : static_cast<double>(py) * static_cast<double>(grid.height - 1)
                / static_cast<double>(draw_height - 1);
        for (int px = 0; px < draw_width; ++px) {
            const double source_x = draw_width == 1 ? 0.0
                : static_cast<double>(px) * static_cast<double>(grid.width - 1)
                    / static_cast<double>(draw_width - 1);
            const float elevation = bilinear(grid.values, grid.width,
                                             grid.height, source_x, source_y);
            const float illumination = bilinear(shade, grid.width, grid.height,
                                                source_x, source_y);
            Color color{255, 255, 255, 255};
            if (kind == MapKind::Elevation && !std::isnan(elevation)) {
                const double normalized = stats.maximum > stats.minimum
                    ? (static_cast<double>(elevation) - stats.minimum)
                        / (stats.maximum - stats.minimum)
                    : 0.5;
                color = terrain_color(normalized);
            } else if (kind == MapKind::Hillshade
                       && !std::isnan(illumination)) {
                const auto channel = static_cast<std::uint8_t>(std::clamp(
                    std::lround(static_cast<double>(illumination) * 255.0),
                    0L, 255L));
                color = {channel, channel, channel, 255};
            } else if (kind == MapKind::Overlay && !std::isnan(elevation)
                       && !std::isnan(illumination)) {
                const auto channel = static_cast<std::uint8_t>(std::clamp(
                    std::lround(static_cast<double>(illumination) * 255.0),
                    0L, 255L));
                Color base = blend({255, 255, 255, 255},
                                   {channel, channel, channel, 255}, 0.5);
                const double normalized = stats.maximum > stats.minimum
                    ? (static_cast<double>(elevation) - stats.minimum)
                        / (stats.maximum - stats.minimum)
                    : 0.5;
                color = blend(base, terrain_color(normalized), 0.6);
            }
            canvas.pixel(plot_left + px, plot_top + py, color);
        }
    }
    draw_axes(canvas, plot_left, plot_top, plot_right, plot_bottom, bounds);
    if (colorbar) {
        const int bar_left = plot_right + 50;
        const int bar_right = bar_left + 48;
        for (int y = plot_top; y < plot_bottom; ++y) {
            const double normalized = 1.0
                - static_cast<double>(y - plot_top)
                    / static_cast<double>(std::max(1, plot_bottom - plot_top - 1));
            canvas.fill(bar_left, y, bar_right, y + 1,
                        terrain_color(normalized));
        }
        canvas.line(bar_left, plot_top, bar_right, plot_top,
                    {30, 30, 30, 255}, 2);
        canvas.line(bar_right, plot_top, bar_right, plot_bottom,
                    {30, 30, 30, 255}, 2);
        canvas.line(bar_left, plot_bottom, bar_right, plot_bottom,
                    {30, 30, 30, 255}, 2);
        for (int tick = 0; tick <= 4; ++tick) {
            const int y = plot_bottom - (plot_bottom - plot_top) * tick / 4;
            const std::string label = fixed(
                stats.minimum + (stats.maximum - stats.minimum) * tick / 4.0,
                1);
            draw_text(canvas, bar_right + 12, y - 7, label, 2);
        }
        draw_text(canvas, bar_left - 10, plot_bottom + 60, "ELEVATION (M)", 2);
    }
}

void draw_histogram_panel(Canvas& canvas, const std::vector<float>& values,
                          const Statistics& stats, int panel_left,
                          int panel_top, int panel_right, int panel_bottom) {
    constexpr std::size_t bins = 50;
    draw_text(canvas,
              (panel_left + panel_right
               - text_width("ELEVATION DISTRIBUTION", 4))
                  / 2,
              panel_top + 24, "ELEVATION DISTRIBUTION", 4,
              {20, 20, 20, 255});
    const int left = panel_left + 190;
    const int right = panel_right - 90;
    const int top = panel_top + 105;
    const int bottom = panel_bottom - 145;
    std::array<std::size_t, bins> counts{};
    if (stats.count > 0) {
        if (!std::isfinite(stats.minimum) || !std::isfinite(stats.maximum)) {
            throw std::runtime_error(
                "supplied range of [inf, inf] is not finite");
        }
        for (const float value : values) {
            if (std::isnan(value)) continue;
            std::size_t bin = 0;
            if (stats.maximum > stats.minimum) {
                const double position = (static_cast<double>(value) - stats.minimum)
                    / (stats.maximum - stats.minimum);
                bin = std::min<std::size_t>(
                    bins - 1U, static_cast<std::size_t>(std::max(
                                   0.0, std::floor(position * bins))));
            }
            ++counts[bin];
        }
    }
    const std::size_t maximum_count =
        *std::max_element(counts.begin(), counts.end());
    for (int tick = 0; tick <= 5; ++tick) {
        const int y = top + (bottom - top) * tick / 5;
        for (int x = left; x <= right; x += 28) {
            canvas.line(x, y, std::min(x + 12, right), y,
                        {205, 205, 205, 255}, 2);
        }
    }
    for (std::size_t bin = 0; bin < bins; ++bin) {
        const int x0 = left + static_cast<int>((right - left) * bin / bins);
        const int x1 = left
            + static_cast<int>((right - left) * (bin + 1U) / bins);
        const int y = maximum_count == 0 ? bottom
            : bottom - static_cast<int>((bottom - top) * counts[bin]
                                        / maximum_count);
        canvas.fill(x0 + 1, y, std::max(x0 + 2, x1 - 1), bottom,
                    {126, 168, 202, 255});
        canvas.line(x0, y, x1, y, {0, 0, 0, 255}, 1);
    }
    canvas.line(left, top, left, bottom, {45, 45, 45, 255}, 3);
    canvas.line(left, bottom, right, bottom, {45, 45, 45, 255}, 3);
    draw_text(canvas, (left + right - text_width("ELEVATION (M)", 3)) / 2,
              bottom + 62, "ELEVATION (M)", 3);
    const int y_label_height = text_width("FREQUENCY", 3);
    draw_text_vertical(canvas, left - 175,
                       (top + bottom - y_label_height) / 2,
                       "FREQUENCY", 3);
    if (stats.count > 0) {
        const int box_right = right - 30;
        const int box_left = box_right - 420;
        const int box_top = top + 35;
        canvas.fill(box_left, box_top, box_right, box_top + 205,
                    {248, 226, 174, 255});
        const std::array<std::string, 4> labels{{
            "MIN: " + fixed(stats.minimum, 1) + " M",
            "MAX: " + fixed(stats.maximum, 1) + " M",
            "MEAN: " + fixed(stats.mean, 1) + " M",
            "STD: " + fixed(stats.standard_deviation, 1) + " M"}};
        for (std::size_t index = 0; index < labels.size(); ++index) {
            draw_text(canvas, box_left + 24,
                      box_top + 24 + static_cast<int>(index) * 42,
                      labels[index], 3);
        }
    }
}

Canvas render_overview(const DemGrid& grid, const Statistics& stats,
                       const Bounds& bounds) {
    Canvas canvas(4770, 3570);
    const std::vector<float> shade = hillshade(grid);
    draw_map_panel(canvas, grid, shade, stats, bounds,
                   0, 0, 2385, 1785, "ELEVATION MAP (EPSG:4326)",
                   MapKind::Elevation, true);
    draw_map_panel(canvas, grid, shade, stats, bounds,
                   2385, 0, 4770, 1785, "HILLSHADE (SHADED RELIEF)",
                   MapKind::Hillshade, false);
    draw_map_panel(canvas, grid, shade, stats, bounds,
                   0, 1785, 2385, 3570,
                   "ELEVATION + HILLSHADE OVERLAY", MapKind::Overlay, true);
    draw_histogram_panel(canvas, grid.values, stats,
                         2385, 1785, 4770, 3570);
    return canvas;
}

struct SurfaceVertex {
    double x = 0.0;
    double y = 0.0;
    double depth = 0.0;
};

struct SurfaceTriangle {
    std::array<SurfaceVertex, 3> vertices;
    double depth = 0.0;
    Color color;
};

SurfaceVertex project_surface(double x, double y, double z) {
    constexpr double azimuth = -1.0471975511965977462;  // -60 degrees
    constexpr double elevation = 0.5235987755982988731; // 30 degrees
    const std::array<double, 3> point{x, y, z * 0.85};
    const std::array<double, 3> right{-std::sin(azimuth),
                                      std::cos(azimuth), 0.0};
    const std::array<double, 3> view{
        std::cos(elevation) * std::cos(azimuth),
        std::cos(elevation) * std::sin(azimuth), std::sin(elevation)};
    const std::array<double, 3> up{
        -std::sin(elevation) * std::cos(azimuth),
        -std::sin(elevation) * std::sin(azimuth), std::cos(elevation)};
    const double horizontal = point[0] * right[0] + point[1] * right[1];
    const double vertical = point[0] * up[0] + point[1] * up[1]
        + point[2] * up[2];
    const double depth = point[0] * view[0] + point[1] * view[1]
        + point[2] * view[2];
    return {1475.0 + horizontal * 850.0, 1325.0 - vertical * 780.0,
            depth};
}

Canvas render_surface(const DemGrid& grid, const Statistics& stats) {
    Canvas canvas(2955, 2498);
    draw_text(canvas,
              (canvas.width()
               - text_width("3D TERRAIN SURFACE (EPSG:4326)", 5))
                  / 2,
              90, "3D TERRAIN SURFACE (EPSG:4326)", 5,
              {20, 20, 20, 255});
    const int step = std::max(1, std::min(grid.width, grid.height) / 200);
    std::vector<int> xs;
    std::vector<int> ys;
    for (int x = 0; x < grid.width; x += step) xs.push_back(x);
    for (int y = 0; y < grid.height; y += step) ys.push_back(y);
    if (xs.empty() || xs.back() != grid.width - 1) xs.push_back(grid.width - 1);
    if (ys.empty() || ys.back() != grid.height - 1) ys.push_back(grid.height - 1);
    std::vector<SurfaceTriangle> triangles;
    if (xs.size() > 1U && ys.size() > 1U) {
        triangles.reserve((xs.size() - 1U) * (ys.size() - 1U) * 2U);
    }
    const auto normalized_point = [&](int x, int y) {
        const std::size_t index = static_cast<std::size_t>(y)
                * static_cast<std::size_t>(grid.width)
            + static_cast<std::size_t>(x);
        const double horizontal = grid.width == 1 ? 0.0
            : static_cast<double>(x) / static_cast<double>(grid.width - 1)
                * 2.0 - 1.0;
        const double vertical = grid.height == 1 ? 0.0
            : 1.0 - static_cast<double>(y)
                    / static_cast<double>(grid.height - 1)
                    * 2.0;
        const double elevation = stats.maximum > stats.minimum
            ? (static_cast<double>(grid.values[index]) - stats.minimum)
                / (stats.maximum - stats.minimum)
            : 0.5;
        return std::array<double, 4>{horizontal, vertical, elevation,
                                     static_cast<double>(grid.values[index])};
    };
    const auto append_triangle = [&](const std::array<double, 4>& first,
                                     const std::array<double, 4>& second,
                                     const std::array<double, 4>& third) {
        if (std::isnan(first[3]) || std::isnan(second[3])
            || std::isnan(third[3])) return;
        SurfaceTriangle triangle;
        triangle.vertices = {{project_surface(first[0], first[1], first[2]),
                              project_surface(second[0], second[1], second[2]),
                              project_surface(third[0], third[1], third[2])}};
        triangle.depth = (triangle.vertices[0].depth
                          + triangle.vertices[1].depth
                          + triangle.vertices[2].depth)
            / 3.0;
        triangle.color = blend(
            {255, 255, 255, 255},
            terrain_color((first[2] + second[2] + third[2]) / 3.0), 0.8);
        triangles.push_back(triangle);
    };
    for (std::size_t row = 1; row < ys.size(); ++row) {
        for (std::size_t column = 1; column < xs.size(); ++column) {
            const auto a = normalized_point(xs[column - 1U], ys[row - 1U]);
            const auto b = normalized_point(xs[column], ys[row - 1U]);
            const auto c = normalized_point(xs[column], ys[row]);
            const auto d = normalized_point(xs[column - 1U], ys[row]);
            append_triangle(a, b, c);
            append_triangle(a, c, d);
        }
    }
    std::sort(triangles.begin(), triangles.end(),
              [](const SurfaceTriangle& left, const SurfaceTriangle& right) {
                  return left.depth < right.depth;
              });
    for (const SurfaceTriangle& triangle : triangles) {
        canvas.triangle(triangle.vertices[0].x, triangle.vertices[0].y,
                        triangle.vertices[1].x, triangle.vertices[1].y,
                        triangle.vertices[2].x, triangle.vertices[2].y,
                        triangle.color);
    }
    const SurfaceVertex origin = project_surface(-1.0, -1.0, 0.0);
    const SurfaceVertex x_axis = project_surface(1.15, -1.0, 0.0);
    const SurfaceVertex y_axis = project_surface(-1.0, 1.15, 0.0);
    const SurfaceVertex z_axis = project_surface(-1.0, -1.0, 1.3);
    const Color axes{35, 35, 35, 255};
    canvas.line(static_cast<int>(origin.x), static_cast<int>(origin.y),
                static_cast<int>(x_axis.x), static_cast<int>(x_axis.y), axes, 4);
    canvas.line(static_cast<int>(origin.x), static_cast<int>(origin.y),
                static_cast<int>(y_axis.x), static_cast<int>(y_axis.y), axes, 4);
    canvas.line(static_cast<int>(origin.x), static_cast<int>(origin.y),
                static_cast<int>(z_axis.x), static_cast<int>(z_axis.y), axes, 4);
    draw_text(canvas, static_cast<int>(x_axis.x), static_cast<int>(x_axis.y),
              "LONGITUDE (DEG)", 3);
    draw_text(canvas, static_cast<int>(y_axis.x) - 180,
              static_cast<int>(y_axis.y) - 30, "LATITUDE (DEG)", 3);
    draw_text(canvas, static_cast<int>(z_axis.x) - 150,
              static_cast<int>(z_axis.y) - 30, "ELEVATION (M)", 3);

    const int bar_left = 2570;
    const int bar_right = 2630;
    const int bar_top = 610;
    const int bar_bottom = 1780;
    for (int y = bar_top; y < bar_bottom; ++y) {
        const double normalized = 1.0
            - static_cast<double>(y - bar_top)
                / static_cast<double>(bar_bottom - bar_top - 1);
        canvas.fill(bar_left, y, bar_right, y + 1,
                    terrain_color(normalized));
    }
    canvas.line(bar_left, bar_top, bar_right, bar_top, axes, 3);
    canvas.line(bar_right, bar_top, bar_right, bar_bottom, axes, 3);
    canvas.line(bar_left, bar_bottom, bar_right, bar_bottom, axes, 3);
    for (int tick = 0; tick <= 4; ++tick) {
        const int y = bar_bottom - (bar_bottom - bar_top) * tick / 4;
        draw_text(canvas, bar_right + 18, y - 8,
                  fixed(stats.minimum
                            + (stats.maximum - stats.minimum) * tick / 4.0,
                        1),
                  2);
    }
    draw_text(canvas, bar_left - 25, bar_bottom + 60, "ELEVATION (M)", 2);
    return canvas;
}

void write_png(const fs::path& path, const Canvas& canvas) {
    if (path.extension() != ".png") {
        throw std::runtime_error(
            "Unsupported output path generated by legacy '.tif' replacement: "
            + path_to_utf8(path));
    }
    GDALDriver* memory_driver =
        GetGDALDriverManager()->GetDriverByName("MEM");
    GDALDriver* png_driver =
        GetGDALDriverManager()->GetDriverByName("PNG");
    if (memory_driver == nullptr || png_driver == nullptr) {
        throw std::runtime_error("GDAL MEM/PNG drivers are unavailable.");
    }
    DatasetPtr memory(memory_driver->Create("", canvas.width(), canvas.height(),
                                            4, GDT_Byte, nullptr));
    if (!memory) throw std::runtime_error("Could not allocate the PNG canvas.");
    std::array<int, 4> bands{1, 2, 3, 4};
    auto* pixels = const_cast<Color*>(canvas.pixels().data());
    if (memory->RasterIO(
            GF_Write, 0, 0, canvas.width(), canvas.height(), pixels,
            canvas.width(), canvas.height(), GDT_Byte, 4, bands.data(),
            static_cast<GSpacing>(sizeof(Color)),
            static_cast<GSpacing>(sizeof(Color)
                                  * static_cast<std::size_t>(canvas.width())),
            1, nullptr)
        != CE_None) {
        throw std::runtime_error("Could not populate the PNG canvas.");
    }
    memory->GetRasterBand(1)->SetColorInterpretation(GCI_RedBand);
    memory->GetRasterBand(2)->SetColorInterpretation(GCI_GreenBand);
    memory->GetRasterBand(3)->SetColorInterpretation(GCI_BlueBand);
    memory->GetRasterBand(4)->SetColorInterpretation(GCI_AlphaBand);
    DatasetPtr written(png_driver->CreateCopy(path_to_utf8(path).c_str(),
                                               memory.get(), FALSE, nullptr,
                                               nullptr, nullptr));
    if (!written) {
        throw std::runtime_error(last_gdal_error(
            "Could not save PNG: " + path_to_utf8(path)));
    }
    written.reset();
    if (!fs::is_regular_file(path)) {
        throw std::runtime_error("Could not save PNG: " + path_to_utf8(path));
    }
}

void ensure_new_outputs(const OutputPaths& paths) {
    std::error_code error;
    if ((fs::exists(paths.overview, error) && !error)
        || (fs::exists(paths.surface, error) && !error)) {
        const fs::path existing = fs::exists(paths.overview) ? paths.overview
                                                             : paths.surface;
        throw std::runtime_error(
            "Output image already exists; refusing to overwrite: "
            + path_to_utf8(existing));
    }
    if (paths.overview == paths.surface) {
        throw std::runtime_error(
            "Legacy output names collide; refusing to overwrite the input: "
            + path_to_utf8(paths.overview));
    }
}

int run_with_gdal(const std::string& input_text, std::ostream& output) {
    const OutputPaths paths = output_paths(input_text);
    ensure_new_outputs(paths);
    GDALAllRegister();
    QuietGdalErrors quiet_errors;
    DatasetPtr input(static_cast<GDALDataset*>(GDALOpenEx(
        input_text.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY
                                | GDAL_OF_VERBOSE_ERROR,
        nullptr, nullptr, nullptr)));
    if (!input) {
        throw std::runtime_error(last_gdal_error(
            input_text + ": No such file or directory"));
    }
    if (input->GetRasterCount() < 1 || input->GetRasterBand(1) == nullptr) {
        throw std::runtime_error("DEM contains no raster band 1.");
    }
    double source_transform[6]{};
    if (input->GetGeoTransform(source_transform) != CE_None) {
        throw std::runtime_error("DEM has no geotransform.");
    }
    GDALRasterBand* band = input->GetRasterBand(1);
    const int source_width = input->GetRasterXSize();
    const int source_height = input->GetRasterYSize();
    const Bounds source_bounds =
        raster_bounds(source_transform, source_width, source_height);
    const double x_resolution = std::hypot(source_transform[1], source_transform[4]);
    const double y_resolution = std::hypot(source_transform[2], source_transform[5]);
    int has_nodata = 0;
    const double nodata = band->GetNoDataValue(&has_nodata);

    output << "\nOriginal DEM Metadata:\n"
           << "  Shape: (" << source_height << ", " << source_width << ")\n"
           << "  CRS: " << crs_text(input->GetSpatialRef()) << '\n'
           << "  Bounds: BoundingBox(left=" << python_float(source_bounds.left)
           << ", bottom=" << python_float(source_bounds.bottom)
           << ", right=" << python_float(source_bounds.right)
           << ", top=" << python_float(source_bounds.top) << ")\n"
           << "  Resolution: (" << python_float(x_resolution) << ", "
           << python_float(y_resolution) << ")\n"
           << "  NoData value: "
           << (has_nodata != 0 ? python_float(nodata) : "None") << "\n\n"
           << "Reprojecting to EPSG:4326 (WGS84)...\n";

    DemGrid grid = reproject_to_wgs84(*input, *band, source_transform);
    const double lon_min = grid.transform[0];
    const double lat_max = grid.transform[3];
    const double lon_max = grid.transform[0]
        + static_cast<double>(grid.width) * grid.transform[1];
    const double lat_min = grid.transform[3]
        + static_cast<double>(grid.height) * grid.transform[5];
    const Bounds target_bounds{lon_min, lat_min, lon_max, lat_max};
    const Statistics stats = statistics(grid.values);
    output << "\nReprojected DEM Metadata:\n"
           << "  Shape: (" << grid.height << ", " << grid.width << ")\n"
           << "  CRS: EPSG:4326\n"
           << "  Lat range: [" << fixed(lat_min, 6) << ", "
           << fixed(lat_max, 6) << "]\n"
           << "  Lon range: [" << fixed(lon_min, 6) << ", "
           << fixed(lon_max, 6) << "]\n";
    if (stats.count > 0) {
        output << "\nElevation Statistics (after filtering -9999):\n"
               << "  Min: " << fixed(stats.minimum, 2) << " m\n"
               << "  Max: " << fixed(stats.maximum, 2) << " m\n"
               << "  Mean: " << fixed(stats.mean, 2) << " m\n"
               << "  Std: " << fixed(stats.standard_deviation, 2) << " m\n"
               << "  Valid pixels: " << stats.count << " / "
               << grid.values.size() << " ("
               << fixed(100.0 * static_cast<double>(stats.count)
                            / static_cast<double>(grid.values.size()),
                        1)
               << "%)\n";
    }

    Canvas overview = render_overview(grid, stats, target_bounds);
    write_png(paths.overview, overview);
    overview = Canvas(1, 1);
    output << "\nVisualization saved to: " << paths.overview_text << "\n\n"
           << "Generating 3D surface plot...\n";
    Canvas surface = render_surface(grid, stats);
    write_png(paths.surface, surface);
    output << "3D visualization saved to: " << paths.surface_text << '\n';
    return 0;
}

#endif

}  // namespace

int run_visdem(const std::vector<std::string>& args,
               std::ostream& out,
               std::ostream& err) {
    const std::string input = args.empty() ? std::string(default_dem_filename)
                                           : args.front();
    out << "Loading DEM file: " << input << '\n';
#if LUW_HAS_GDAL
    try {
        return run_with_gdal(input, out);
    } catch (const std::exception& exception) {
        err << "ERROR: visdem failed: " << exception.what() << '\n';
        return 1;
    }
#else
    err << "ERROR: Native visdem requires GDAL raster/reprojection support; "
           "rebuild with LUW_ENABLE_GEOSPATIAL=ON and a discoverable GDAL "
           "installation.\n";
    return 1;
#endif
}

}  // namespace luw::cli
