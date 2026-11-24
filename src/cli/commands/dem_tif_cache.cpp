#include "luw/dem_tif_cache.hpp"

#include <QString>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <conio.h>
#else
#include <sys/select.h>
#include <unistd.h>
#endif

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <cpl_string.h>
#include <gdal_alg.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <ogrsf_frmts.h>
#endif

namespace luw::cli {
namespace {

namespace fs = std::filesystem;

std::string pathToUtf8(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring()).toUtf8().toStdString();
#else
    return path.string();
#endif
}

fs::path pathFromUtf8(std::string_view text) {
#ifdef _WIN32
    return fs::path(QString::fromUtf8(text.data(),
                                     static_cast<qsizetype>(text.size())).toStdWString());
#else
    return fs::path(std::string(text));
#endif
}

void info(std::ostream& output, const std::string& message) {
    output << "[INFO] " << message << '\n';
    output.flush();
}

void warn(std::ostream& output, const std::string& message) {
    output << "[WARN] " << message << '\n';
    output.flush();
}

std::string fixed(double value, int precision) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string pythonFloat(double value) {
    std::array<char, 128> buffer{};
    const auto rendered = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                        value, std::chars_format::general);
    if (rendered.ec == std::errc{}) {
        std::string result(buffer.data(), rendered.ptr);
        if (result.find_first_of(".eE") == std::string::npos) result += ".0";
        return result;
    }
    std::ostringstream fallback;
    fallback.imbue(std::locale::classic());
    fallback << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return fallback.str();
}

std::string lowerAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return text;
}

bool regularFile(const fs::path& path) {
    std::error_code error;
    return fs::is_regular_file(path, error) && !error;
}

std::vector<fs::path> tifCandidates(const fs::path& terrainDirectory) {
    std::vector<fs::path> tifs;
    std::vector<fs::path> tiffs;
    std::error_code iteratorError;
    if (!fs::is_directory(terrainDirectory, iteratorError) || iteratorError) return {};
    for (fs::directory_iterator iterator(terrainDirectory, iteratorError), end;
         !iteratorError && iterator != end; iterator.increment(iteratorError)) {
        std::error_code typeError;
        if (!iterator->is_regular_file(typeError) || typeError) continue;
        const std::string extension = lowerAscii(pathToUtf8(iterator->path().extension()));
        if (extension == ".tif") tifs.push_back(iterator->path());
        else if (extension == ".tiff") tiffs.push_back(iterator->path());
    }
    std::sort(tifs.begin(), tifs.end());
    std::sort(tiffs.begin(), tiffs.end());
    tifs.insert(tifs.end(), tiffs.begin(), tiffs.end());
    return tifs;
}

constexpr std::array<std::string_view, 7> shapefileExtensions{
    ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"};

bool sidecarCollision(const fs::path& outputShapefile, fs::path& collision) {
    fs::path base = outputShapefile;
    base.replace_extension();
    for (const std::string_view extension : shapefileExtensions) {
        fs::path candidate = base;
        candidate += extension;
        std::error_code error;
        if (fs::exists(candidate, error) && !error) {
            collision = std::move(candidate);
            return true;
        }
    }
    return false;
}

std::optional<fs::path> safeOutputShapefile(const fs::path& terrainDirectory,
                                            const std::string& caseName,
                                            std::ostream& output) {
    if (caseName.empty() || caseName.find('\0') != std::string::npos) {
        warn(output, "Refusing unsafe DEM cache casename.");
        return std::nullopt;
    }
    const fs::path component = pathFromUtf8(caseName + ".shp");
    if (component.empty() || component.has_root_name() ||
        component.has_root_directory() || component.has_parent_path() ||
        component.filename() != component || component == "." || component == "..") {
        warn(output, "Refusing unsafe DEM cache casename: " + caseName);
        return std::nullopt;
    }
    std::error_code terrainError;
    const fs::path terrainAbsolute =
        fs::absolute(terrainDirectory, terrainError).lexically_normal();
    if (terrainError) {
        warn(output, "Failed to resolve terrain_db directory: " +
                         terrainError.message());
        return std::nullopt;
    }
    const fs::path outputShapefile =
        (terrainAbsolute / component).lexically_normal();
    if (outputShapefile.parent_path() != terrainAbsolute) {
        warn(output, "Refusing DEM cache path outside terrain_db: " +
                         pathToUtf8(outputShapefile));
        return std::nullopt;
    }
    return outputShapefile;
}

DemLonLatBounds expandedBounds(const DemLonLatBounds& bounds) {
    const double longitudeSpan = bounds.lonMax - bounds.lonMin;
    const double latitudeSpan = bounds.latMax - bounds.latMin;
    return {bounds.lonMin - 0.2 * longitudeSpan,
            bounds.lonMax + 0.2 * longitudeSpan,
            bounds.latMin - 0.2 * latitudeSpan,
            bounds.latMax + 0.2 * latitudeSpan};
}

bool contains(const DemLonLatBounds& target, const DemLonLatBounds& input) {
    return input.lonMin <= target.lonMin && input.lonMax >= target.lonMax &&
           input.latMin <= target.latMin && input.latMax >= target.latMax;
}

double maximumMissPercent(const DemLonLatBounds& target,
                          const DemLonLatBounds& input) {
    const double longitudeSpan = std::abs(target.lonMax - target.lonMin);
    const double latitudeSpan = std::abs(target.latMax - target.latMin);
    if (longitudeSpan <= 0.0 || latitudeSpan <= 0.0) return 100.0;
    const double longitudeMiss = std::max(
        {0.0, input.lonMin - target.lonMin, target.lonMax - input.lonMax});
    const double latitudeMiss = std::max(
        {0.0, input.latMin - target.latMin, target.latMax - input.latMax});
    return std::max(longitudeMiss / longitudeSpan,
                    latitudeMiss / latitudeSpan) * 100.0;
}

std::string formattedBounds(const DemLonLatBounds& bounds) {
    return "lon[" + fixed(bounds.lonMin, 6) + ", " + fixed(bounds.lonMax, 6) +
           "] lat[" + fixed(bounds.latMin, 6) + ", " + fixed(bounds.latMax, 6) + "]";
}

std::optional<std::string> timedInputLine(std::ostream& output,
                                          std::istream* injectedInput,
                                          std::string_view prompt,
                                          double timeoutSeconds) {
    output << prompt;
    output.flush();
    if (injectedInput != nullptr) {
        std::string line;
        if (!std::getline(*injectedInput, line)) return std::nullopt;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return line;
    }

    const double boundedTimeout = std::max(0.0, timeoutSeconds);
#ifdef _WIN32
    std::string buffer;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(boundedTimeout);
    while (std::chrono::steady_clock::now() < deadline) {
        if (_kbhit() != 0) {
            const int character = _getch();
            if (character == '\r' || character == '\n') {
                output << '\n';
                output.flush();
                return buffer;
            }
            if (character == 3) throw std::runtime_error("Interrupted");
            if (character == '\b') {
                if (!buffer.empty()) {
                    buffer.pop_back();
                    output << "\b \b";
                    output.flush();
                }
                continue;
            }
            if (character == 0 || character == 0xE0) {
                if (_kbhit() != 0) (void)_getch();
                continue;
            }
            buffer.push_back(static_cast<char>(character));
            output << static_cast<char>(character);
            output.flush();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
#else
    fd_set readSet;
    FD_ZERO(&readSet);
    FD_SET(STDIN_FILENO, &readSet);
    timeval timeout{};
    timeout.tv_sec = static_cast<long>(boundedTimeout);
    timeout.tv_usec = static_cast<long>(
        (boundedTimeout - static_cast<double>(timeout.tv_sec)) * 1000000.0);
    const int status =
        select(STDIN_FILENO + 1, &readSet, nullptr, nullptr, &timeout);
    if (status > 0) {
        std::string line;
        if (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return line;
        }
    }
#endif
    output << '\n';
    output.flush();
    return std::nullopt;
}

void confirmCoverage(const DemLonLatBounds& target,
                     const DemLonLatBounds& input,
                     const DemTifCacheRequest& request,
                     std::ostream& output) {
    if (contains(target, input)) return;
    const double miss = maximumMissPercent(target, input);
    if (miss < 0.1) {
        warn(output, "DEM GeoTIFF bounds are slightly smaller than target (max miss " +
                         fixed(miss, 4) +
                         "% < 0.1%). Continue without interruption.");
        return;
    }
    warn(output, "DEM GeoTIFF bounds do not fully cover the target area.");
    warn(output, "Target lon/lat bounds: " + formattedBounds(target));
    warn(output, "Input  lon/lat bounds: " + formattedBounds(input));
    const std::optional<std::string> answer = timedInputLine(
        output, request.coverageInput,
        "Continue anyway? (Y/N) [auto-continue in 5s]: ",
        request.coverageTimeoutSeconds);
    if (!answer) {
        warn(output, "No input received (timeout 5s). Continuing by default.");
        return;
    }
    std::string normalized = lowerAscii(*answer);
    const std::size_t first = normalized.find_first_not_of(" \t\r\n");
    const std::size_t last = normalized.find_last_not_of(" \t\r\n");
    normalized = first == std::string::npos
        ? std::string{}
        : normalized.substr(first, last - first + 1U);
    if (normalized == "n" || normalized == "no") {
        info(output, "User canceled. Exiting.");
        throw DemTifCacheCancelled{};
    }
    if (normalized.empty() || normalized == "y" || normalized == "yes") {
        warn(output, "Continuing despite bounds mismatch.");
        return;
    }
    warn(output, "Invalid input '" + *answer + "'. Continuing by default.");
}

#if LUW_HAS_GDAL

struct DatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) GDALClose(dataset);
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

struct FeatureCloser {
    void operator()(OGRFeature* feature) const noexcept {
        if (feature != nullptr) OGRFeature::DestroyFeature(feature);
    }
};

struct StringListCloser {
    void operator()(char** list) const noexcept {
        if (list != nullptr) CSLDestroy(list);
    }
};

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;
using GeometryPtr = std::unique_ptr<OGRGeometry, GeometryCloser>;
using TransformPtr = std::unique_ptr<OGRCoordinateTransformation, TransformCloser>;
using FeaturePtr = std::unique_ptr<OGRFeature, FeatureCloser>;
using StringListPtr = std::unique_ptr<char*, StringListCloser>;

std::unique_ptr<OGRSpatialReference> cloneReference(
    const OGRSpatialReference* reference) {
    if (reference == nullptr) return {};
    std::unique_ptr<OGRSpatialReference> result(reference->Clone());
    if (result) result->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return result;
}

std::unique_ptr<OGRSpatialReference> wgs84Reference() {
    auto result = std::make_unique<OGRSpatialReference>();
    if (result->importFromEPSG(4326) != OGRERR_NONE) return {};
    result->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return result;
}

TransformPtr coordinateTransform(OGRSpatialReference* source,
                                 OGRSpatialReference* target) {
    if (source == nullptr || target == nullptr) return {};
    source->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    target->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return TransformPtr(OGRCreateCoordinateTransformation(source, target));
}

std::string referenceText(const OGRSpatialReference* reference) {
    if (reference == nullptr) return "None";
    const char* authorityName = reference->GetAuthorityName(nullptr);
    const char* authorityCode = reference->GetAuthorityCode(nullptr);
    if (authorityName != nullptr && authorityCode != nullptr) {
        return std::string(authorityName) + ':' + authorityCode;
    }
    char* text = nullptr;
    if (reference->exportToWkt(&text) == OGRERR_NONE && text != nullptr) {
        std::string result(text);
        CPLFree(text);
        return result;
    }
    return "unknown";
}

GeometryPtr rectangle(const DemLonLatBounds& bounds) {
    auto polygon = std::make_unique<OGRPolygon>();
    OGRLinearRing ring;
    ring.addPoint(bounds.lonMin, bounds.latMin);
    ring.addPoint(bounds.lonMax, bounds.latMin);
    ring.addPoint(bounds.lonMax, bounds.latMax);
    ring.addPoint(bounds.lonMin, bounds.latMax);
    ring.addPoint(bounds.lonMin, bounds.latMin);
    if (polygon->addRing(&ring) != OGRERR_NONE) return {};
    return GeometryPtr(polygon.release());
}

bool isLegacyWgs84(const OGRSpatialReference& reference) {
    const char* authorityName = reference.GetAuthorityName(nullptr);
    const char* authorityCode = reference.GetAuthorityCode(nullptr);
    return authorityName != nullptr && authorityCode != nullptr &&
           lowerAscii(authorityName) == "epsg" &&
           std::string_view(authorityCode) == "4326";
}

double roundedCoordinate(double value) {
    constexpr double scale = 1000000.0;
    return std::round(value * scale) / scale;
}

void roundLineCoordinates(OGRLineString& line) {
    for (int index = 0; index < line.getNumPoints(); ++index) {
        line.setPoint(index, roundedCoordinate(line.getX(index)),
                      roundedCoordinate(line.getY(index)));
    }
}

void roundGeometryCoordinates(OGRGeometry& geometry) {
    const OGRwkbGeometryType type = wkbFlatten(geometry.getGeometryType());
    if (type == wkbPolygon) {
        auto* polygon = dynamic_cast<OGRPolygon*>(&geometry);
        if (polygon == nullptr) return;
        if (OGRLinearRing* exterior = polygon->getExteriorRing()) {
            roundLineCoordinates(*exterior);
        }
        for (int index = 0; index < polygon->getNumInteriorRings(); ++index) {
            if (OGRLinearRing* interior = polygon->getInteriorRing(index)) {
                roundLineCoordinates(*interior);
            }
        }
        return;
    }
    if (OGR_GT_IsSubClassOf(type, wkbGeometryCollection)) {
        auto* collection = dynamic_cast<OGRGeometryCollection*>(&geometry);
        for (int index = 0; collection != nullptr &&
                            index < collection->getNumGeometries();
             ++index) {
            if (OGRGeometry* child = collection->getGeometryRef(index)) {
                roundGeometryCoordinates(*child);
            }
        }
    }
}

std::optional<DemLonLatBounds> rasterBoundsInWgs84(
    GDALDataset& dataset,
    OGRSpatialReference* rasterReference,
    std::ostream& output) {
    double geotransform[6]{};
    if (dataset.GetGeoTransform(geotransform) != CE_None) return std::nullopt;
    constexpr int pointsPerEdge = 23; // rasterio transform_bounds densify_pts=21
    std::vector<double> x;
    std::vector<double> y;
    x.reserve(pointsPerEdge * 4U);
    y.reserve(pointsPerEdge * 4U);
    const double width = static_cast<double>(dataset.GetRasterXSize());
    const double height = static_cast<double>(dataset.GetRasterYSize());
    const auto append = [&](double pixel, double line) {
        x.push_back(geotransform[0] + pixel * geotransform[1] + line * geotransform[2]);
        y.push_back(geotransform[3] + pixel * geotransform[4] + line * geotransform[5]);
    };
    for (int index = 0; index < pointsPerEdge; ++index) {
        const double fraction = static_cast<double>(index) /
                                static_cast<double>(pointsPerEdge - 1);
        append(fraction * width, 0.0);
        append(width, fraction * height);
        append((1.0 - fraction) * width, height);
        append(0.0, (1.0 - fraction) * height);
    }
    if (rasterReference != nullptr) {
        auto wgs84 = wgs84Reference();
        TransformPtr transform = coordinateTransform(rasterReference, wgs84.get());
        if (!transform || !transform->Transform(static_cast<int>(x.size()), x.data(), y.data())) {
            warn(output,
                 "Failed to transform DEM bounds to EPSG:4326 for coverage check.");
            return std::nullopt;
        }
    }
    DemLonLatBounds result{std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};
    for (std::size_t index = 0; index < x.size(); ++index) {
        if (!std::isfinite(x[index]) || !std::isfinite(y[index])) continue;
        result.lonMin = std::min(result.lonMin, x[index]);
        result.lonMax = std::max(result.lonMax, x[index]);
        result.latMin = std::min(result.latMin, y[index]);
        result.latMax = std::max(result.latMax, y[index]);
    }
    if (!std::isfinite(result.lonMin) || !std::isfinite(result.latMin)) return std::nullopt;
    return result;
}

struct PolygonizedRaster {
    DatasetPtr vectorDataset;
    OGRLayer* layer = nullptr;
    std::size_t featureCount = 0U;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
};

std::optional<PolygonizedRaster> polygonizeCandidate(
    const fs::path& rasterPath,
    const DemLonLatBounds& expanded,
    const DemTifCacheRequest& request,
    std::ostream& output) {
    DatasetPtr dataset(static_cast<GDALDataset*>(GDALOpenEx(
        pathToUtf8(rasterPath).c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY,
        nullptr, nullptr, nullptr)));
    if (!dataset || dataset->GetRasterCount() < 1) {
        warn(output, "Failed to process GeoTIFF: Cannot read DEM GeoTIFF");
        return std::nullopt;
    }
    GDALRasterBand* band = dataset->GetRasterBand(1);
    if (band == nullptr) {
        warn(output, "Failed to process GeoTIFF: Cannot read DEM GeoTIFF band");
        return std::nullopt;
    }
    double geotransform[6]{};
    if (dataset->GetGeoTransform(geotransform) != CE_None) {
        warn(output, "Failed to process GeoTIFF: DEM GeoTIFF has no geotransform");
        return std::nullopt;
    }

    std::unique_ptr<OGRSpatialReference> rasterReference =
        cloneReference(dataset->GetSpatialRef());
    if (!rasterReference) {
        warn(output,
             "DEM has no CRS. Assuming raster bounds are EPSG:4326 for coverage check.");
    }
    if (const auto inputBounds =
            rasterBoundsInWgs84(*dataset, rasterReference.get(), output)) {
        confirmCoverage(request.targetBounds, *inputBounds, request, output);
    }

    GeometryPtr cropGeometry = rectangle(expanded);
    if (!cropGeometry) {
        warn(output, "Failed to process GeoTIFF: Cannot construct DEM crop bounds");
        return std::nullopt;
    }
    if (!rasterReference) {
        warn(output, "DEM has no CRS. Assuming bbox is already in raster CRS.");
    } else if (!isLegacyWgs84(*rasterReference)) {
        auto wgs84 = wgs84Reference();
        TransformPtr toRaster = coordinateTransform(wgs84.get(), rasterReference.get());
        if (!toRaster || cropGeometry->transform(toRaster.get()) != OGRERR_NONE) {
            warn(output, "Failed to transform bbox from EPSG:4326 to DEM CRS (" +
                             referenceText(rasterReference.get()) + ").");
            return std::nullopt;
        }
        // rasterio.warp.transform_geom(..., precision=6) rounds every
        // transformed vertex before rasterio.mask derives its crop window.
        roundGeometryCoordinates(*cropGeometry);
    }

    info(output, "DEM path: " + pathToUtf8(rasterPath));
    info(output, "DEM CRS: " + referenceText(rasterReference.get()));
    info(output, "Cutting raster by expanded bbox ...");

    double inverse[6]{};
    if (!GDALInvGeoTransform(geotransform, inverse)) {
        warn(output, "Failed to process GeoTIFF: DEM geotransform is not invertible");
        return std::nullopt;
    }
    OGREnvelope envelope;
    cropGeometry->getEnvelope(&envelope);
    const std::array<double, 4> envelopeX{
        envelope.MinX, envelope.MaxX, envelope.MaxX, envelope.MinX};
    const std::array<double, 4> envelopeY{
        envelope.MinY, envelope.MinY, envelope.MaxY, envelope.MaxY};
    int xStart = dataset->GetRasterXSize();
    int xStop = 0;
    int yStart = dataset->GetRasterYSize();
    int yStop = 0;
    for (std::size_t index = 0; index < envelopeX.size(); ++index) {
        const double pixel = inverse[0] + inverse[1] * envelopeX[index] +
                             inverse[2] * envelopeY[index];
        const double line = inverse[3] + inverse[4] * envelopeX[index] +
                            inverse[5] * envelopeY[index];
        xStart = std::min(xStart, static_cast<int>(std::floor(pixel)));
        xStop = std::max(xStop, static_cast<int>(std::ceil(pixel)));
        yStart = std::min(yStart, static_cast<int>(std::floor(line)));
        yStop = std::max(yStop, static_cast<int>(std::ceil(line)));
    }
    xStart = std::clamp(xStart, 0, dataset->GetRasterXSize());
    xStop = std::clamp(xStop, 0, dataset->GetRasterXSize());
    yStart = std::clamp(yStart, 0, dataset->GetRasterYSize());
    yStop = std::clamp(yStop, 0, dataset->GetRasterYSize());
    if (xStart >= xStop || yStart >= yStop) {
        warn(output, "GeoTIFF does not overlap target bounds.");
        return std::nullopt;
    }

    const int width = xStop - xStart;
    const int height = yStop - yStart;
    int hasNoData = 0;
    const double noData = band->GetNoDataValue(&hasNoData);
    GDALDriver* rasterMemoryDriver =
        GetGDALDriverManager()->GetDriverByName("MEM");
    if (rasterMemoryDriver == nullptr) {
        warn(output, "Polygonize failed: cannot create in-memory raster mask.");
        return std::nullopt;
    }
    double croppedTransform[6]{geotransform[0], geotransform[1], geotransform[2],
                               geotransform[3], geotransform[4], geotransform[5]};
    croppedTransform[0] = geotransform[0] + static_cast<double>(xStart) * geotransform[1] +
                          static_cast<double>(yStart) * geotransform[2];
    croppedTransform[3] = geotransform[3] + static_cast<double>(xStart) * geotransform[4] +
                          static_cast<double>(yStart) * geotransform[5];

    DatasetPtr geometryMaskRaster(rasterMemoryDriver->Create(
        "", width, height, 1, GDT_Byte, nullptr));
    GDALRasterBand* geometryMaskBand =
        geometryMaskRaster ? geometryMaskRaster->GetRasterBand(1) : nullptr;
    if (!geometryMaskRaster || geometryMaskBand == nullptr ||
        geometryMaskRaster->SetGeoTransform(croppedTransform) != CE_None ||
        (rasterReference &&
         geometryMaskRaster->SetSpatialRef(rasterReference.get()) != CE_None) ||
        geometryMaskBand->Fill(0.0) != CE_None) {
        warn(output, "Failed to process GeoTIFF: Cannot construct raster geometry mask");
        return std::nullopt;
    }
    int maskBandIndex = 1;
    OGRGeometryH cropGeometryHandle =
        reinterpret_cast<OGRGeometryH>(cropGeometry.get());
    double maskBurnValue = 1.0;
    if (GDALRasterizeGeometries(
            reinterpret_cast<GDALDatasetH>(geometryMaskRaster.get()), 1,
            &maskBandIndex, 1, &cropGeometryHandle, nullptr, nullptr,
            &maskBurnValue, nullptr, nullptr, nullptr) != CE_None) {
        warn(output, "Failed to process GeoTIFF: Cannot rasterize DEM crop bounds");
        return std::nullopt;
    }

    DatasetPtr croppedRaster(rasterMemoryDriver->Create(
        "", width, height, 1, GDT_Float64, nullptr));
    if (!croppedRaster) {
        warn(output, "Polygonize failed: cannot create in-memory cropped raster.");
        return std::nullopt;
    }
    if (croppedRaster->SetGeoTransform(croppedTransform) != CE_None ||
        (rasterReference && croppedRaster->SetSpatialRef(rasterReference.get()) != CE_None)) {
        warn(output, "Polygonize failed: cannot configure in-memory cropped raster.");
        return std::nullopt;
    }
    GDALRasterBand* valueBand = croppedRaster->GetRasterBand(1);
    if (valueBand == nullptr) {
        warn(output, "Polygonize failed: cannot access in-memory cropped raster.");
        return std::nullopt;
    }

    GDALRasterBand* sourceMaskBand = band->GetMaskBand();
    const bool readSourceMask = sourceMaskBand != nullptr &&
                                (band->GetMaskFlags() & GMF_ALL_VALID) == 0;

    // rasterio.mask(..., filled=True) fills geometry/source-mask exclusions
    // with src.nodata, or zero when the source declares no nodata value. The
    // legacy code subsequently considers that zero valid in the latter case.
    const double fillValue = hasNoData != 0 ? noData : 0.0;

    struct RasterStrip {
        int offset = 0;
        int rows = 0;
        std::vector<double> values;
        std::vector<std::uint8_t> sourceMask;
        std::string error;
    };

    int sourceBlockWidth = 0;
    int sourceBlockRows = 0;
    band->GetBlockSize(&sourceBlockWidth, &sourceBlockRows);
    (void)sourceBlockWidth;
    sourceBlockRows = std::max(1, sourceBlockRows);
    constexpr std::size_t targetStripBytes = 8U * 1024U * 1024U;
    const std::size_t bytesPerRow = std::max<std::size_t>(
        1U, static_cast<std::size_t>(width) *
                (sizeof(double) + sizeof(std::uint8_t)));
    const std::size_t budgetRows = std::clamp<std::size_t>(
        targetStripBytes / bytesPerRow, 1U, 1024U);
    int stripRows = sourceBlockRows;
    if (budgetRows >= static_cast<std::size_t>(sourceBlockRows)) {
        stripRows = static_cast<int>(budgetRows /
                                     static_cast<std::size_t>(sourceBlockRows)) *
                    sourceBlockRows;
    }
    stripRows = std::clamp(stripRows, 1, height);

    const auto readStrip = [&](RasterStrip& strip, int offset, int rows) {
        strip.offset = offset;
        strip.rows = rows;
        strip.error.clear();
        const std::size_t stripCells = static_cast<std::size_t>(width) *
                                       static_cast<std::size_t>(rows);
        strip.values.resize(stripCells);
        strip.sourceMask.assign(stripCells, 255U);
        if (band->RasterIO(GF_Read, xStart, yStart + offset, width, rows,
                           strip.values.data(), width, rows, GDT_Float64,
                           0, 0, nullptr) != CE_None) {
            strip.error = "Failed reading DEM GeoTIFF pixels";
            return false;
        }
        if (readSourceMask &&
            sourceMaskBand->RasterIO(GF_Read, xStart, yStart + offset,
                                     width, rows, strip.sourceMask.data(),
                                     width, rows, GDT_Byte, 0, 0,
                                     nullptr) != CE_None) {
            strip.error = "Failed reading DEM validity mask";
            return false;
        }
        return true;
    };

    std::array<RasterStrip, 2> strips;
    std::vector<std::uint8_t> geometryMask;
    std::size_t validCount = 0U;
    int currentBuffer = 0;
    const int firstRows = std::min(stripRows, height);
    std::future<bool> pendingRead = std::async(
        std::launch::async, readStrip, std::ref(strips[currentBuffer]),
        0, firstRows);
    for (int offset = 0; offset < height; offset += stripRows) {
        RasterStrip& strip = strips[currentBuffer];
        if (!pendingRead.get()) {
            warn(output, "Failed to process GeoTIFF: " + strip.error);
            return std::nullopt;
        }

        const int nextOffset = offset + strip.rows;
        const int nextBuffer = 1 - currentBuffer;
        std::future<bool> nextRead;
        if (nextOffset < height) {
            const int nextRows = std::min(stripRows, height - nextOffset);
            nextRead = std::async(
                std::launch::async, readStrip, std::ref(strips[nextBuffer]),
                nextOffset, nextRows);
        }

        const std::size_t stripCells = static_cast<std::size_t>(width) *
                                       static_cast<std::size_t>(strip.rows);
        geometryMask.resize(stripCells);
        if (geometryMaskBand->RasterIO(
                GF_Read, 0, strip.offset, width, strip.rows,
                geometryMask.data(), width, strip.rows, GDT_Byte,
                0, 0, nullptr) != CE_None) {
            warn(output,
                 "Failed to process GeoTIFF: Cannot read DEM crop mask");
            return std::nullopt;
        }
        for (std::size_t cell = 0U; cell < stripCells; ++cell) {
            if (geometryMask[cell] == 0U || strip.sourceMask[cell] == 0U) {
                strip.values[cell] = fillValue;
            }
            const double value = strip.values[cell];
            const bool valid = value != -9999.0 &&
                               (hasNoData == 0 || value != noData);
            geometryMask[cell] = valid ? 1U : 0U;
            validCount += valid ? 1U : 0U;
        }
        if (valueBand->RasterIO(
                GF_Write, 0, strip.offset, width, strip.rows,
                strip.values.data(), width, strip.rows, GDT_Float64,
                0, 0, nullptr) != CE_None ||
            geometryMaskBand->RasterIO(
                GF_Write, 0, strip.offset, width, strip.rows,
                geometryMask.data(), width, strip.rows, GDT_Byte,
                0, 0, nullptr) != CE_None) {
            warn(output,
                 "Polygonize failed: cannot populate in-memory cropped raster.");
            return std::nullopt;
        }

        if (nextOffset < height) {
            pendingRead = std::move(nextRead);
            currentBuffer = nextBuffer;
        }
    }
    if (validCount == 0U) {
        warn(output, "No valid elevation cells found within expanded bbox.");
        return std::nullopt;
    }
    GDALRasterBand* maskBand = geometryMaskBand;

    GDALDriver* vectorMemoryDriver =
        GetGDALDriverManager()->GetDriverByName("MEM");
    const char* vectorCapability =
        vectorMemoryDriver != nullptr
            ? vectorMemoryDriver->GetMetadataItem(GDAL_DCAP_VECTOR)
            : nullptr;
    if (vectorCapability == nullptr || !EQUAL(vectorCapability, "YES")) {
        // Before GDAL 3.11, MEM was raster-only and the vector driver used
        // the legacy Memory name.
        vectorMemoryDriver =
            GetGDALDriverManager()->GetDriverByName("Memory");
    }
    DatasetPtr vectorDataset;
    if (vectorMemoryDriver != nullptr) {
        vectorDataset.reset(vectorMemoryDriver->Create(
            "", 0, 0, 0, GDT_Unknown, nullptr));
    }
    if (!vectorDataset) {
        warn(output, "Polygonize failed: cannot create in-memory vector dataset.");
        return std::nullopt;
    }
    OGRLayer* layer = vectorDataset->CreateLayer(
        request.caseName.c_str(), rasterReference.get(), wkbPolygon, nullptr);
    OGRFieldDefn elevationField("elevation", OFTReal);
    elevationField.SetWidth(24);
    elevationField.SetPrecision(15);
    OGRFieldDefn caseField("casename", OFTString);
    caseField.SetWidth(80);
    if (layer == nullptr || layer->CreateField(&elevationField) != OGRERR_NONE ||
        layer->CreateField(&caseField) != OGRERR_NONE) {
        warn(output, "Polygonize failed: cannot create DEM vector schema.");
        return std::nullopt;
    }
    info(output, "Polygonizing raster cells ...");
    const int elevationIndex = layer->GetLayerDefn()->GetFieldIndex("elevation");
    if (elevationIndex < 0 ||
        GDALFPolygonize(reinterpret_cast<GDALRasterBandH>(valueBand),
                        reinterpret_cast<GDALRasterBandH>(maskBand),
                        reinterpret_cast<OGRLayerH>(layer), elevationIndex,
                        nullptr, nullptr, nullptr) != CE_None) {
        warn(output, "Polygonize failed.");
        return std::nullopt;
    }

    std::vector<double> statistics;
    std::size_t featureCount = 0U;
    layer->ResetReading();
    while (OGRFeature* rawFeature = layer->GetNextFeature()) {
        FeaturePtr feature(rawFeature);
        ++featureCount;
        feature->SetField("casename", request.caseName.c_str());
        if (layer->SetFeature(feature.get()) != OGRERR_NONE) {
            warn(output, "Polygonize failed: cannot set DEM feature attributes.");
            return std::nullopt;
        }
        if (!feature->IsFieldSetAndNotNull(elevationIndex)) continue;
        const double elevation = feature->GetFieldAsDouble(elevationIndex);
        // GeoPandas reductions skip NaN but retain infinities. Preserve that
        // distinction for the legacy min/max/mean summary.
        if (!std::isnan(elevation)) statistics.push_back(elevation);
    }
    if (featureCount == 0U) {
        warn(output,
             "Polygonize produced no features (all cells may be nodata after masking).");
        return std::nullopt;
    }
    PolygonizedRaster result;
    result.vectorDataset = std::move(vectorDataset);
    result.layer = layer;
    result.featureCount = featureCount;
    if (statistics.empty()) {
        result.minimum = std::numeric_limits<double>::quiet_NaN();
        result.maximum = std::numeric_limits<double>::quiet_NaN();
        result.mean = std::numeric_limits<double>::quiet_NaN();
    } else {
        const auto [minimum, maximum] =
            std::minmax_element(statistics.begin(), statistics.end());
        result.minimum = *minimum;
        result.maximum = *maximum;
        result.mean = std::accumulate(statistics.begin(), statistics.end(), 0.0) /
                      static_cast<double>(statistics.size());
    }
    return result;
}

std::optional<fs::path> createStagingDirectory(
    const fs::path& terrainDirectory,
    std::ostream& output) {
    static std::atomic<std::uint64_t> sequence{0U};
    const auto stamp = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    for (std::uint64_t attempt = 0U; attempt < 1024U; ++attempt) {
        const std::uint64_t suffix =
            sequence.fetch_add(1U, std::memory_order_relaxed);
        const fs::path candidate = terrainDirectory /
            (".luw-dem-cache-staging-" + std::to_string(stamp) + "-" +
             std::to_string(suffix));
        std::error_code createError;
        if (fs::create_directory(candidate, createError)) return candidate;
        if (createError && createError != std::errc::file_exists) {
            warn(output, "Failed to create DEM cache staging directory: " +
                             createError.message());
            return std::nullopt;
        }
    }
    warn(output, "Failed to allocate a unique DEM cache staging directory.");
    return std::nullopt;
}

bool validateStagedShapefile(const fs::path& stagedShapefile,
                             std::size_t expectedFeatureCount,
                             std::ostream& output) {
    for (const std::string_view extension :
         {std::string_view(".shp"), std::string_view(".shx"),
          std::string_view(".dbf"), std::string_view(".cpg")}) {
        fs::path component = stagedShapefile;
        component.replace_extension(extension);
        if (!regularFile(component)) {
            warn(output, "Failed to write shapefile: staged component is missing: " +
                             pathToUtf8(component.filename()));
            return false;
        }
    }
    DatasetPtr verification(static_cast<GDALDataset*>(GDALOpenEx(
        pathToUtf8(stagedShapefile).c_str(),
        GDAL_OF_VECTOR | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
    OGRLayer* layer = verification ? verification->GetLayer(0) : nullptr;
    if (layer == nullptr || layer->GetLayerDefn() == nullptr ||
        layer->GetLayerDefn()->GetFieldIndex("elevation") < 0 ||
        layer->GetLayerDefn()->GetFieldIndex("casename") < 0 ||
        layer->GetFeatureCount(TRUE) !=
            static_cast<GIntBig>(expectedFeatureCount)) {
        warn(output, "Failed to write shapefile: staged dataset validation failed.");
        return false;
    }
    return true;
}

bool publishStagedShapefile(const fs::path& stagedShapefile,
                            const fs::path& outputShapefile,
                            std::ostream& output) {
    fs::path collision;
    if (sidecarCollision(outputShapefile, collision)) {
        warn(output, "Refusing to overwrite existing DEM shapefile component: " +
                         pathToUtf8(collision));
        return false;
    }

    // Copy with copy_options::none so every destination component is created
    // only when absent. Publish .shp last: readers never observe the cache's
    // entry point before its index, attributes, CRS, and encoding are present.
    constexpr std::array<std::string_view, 7> publishOrder{
        ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix", ".shp"};
    for (const std::string_view extension : publishOrder) {
        fs::path stagedComponent = stagedShapefile;
        stagedComponent.replace_extension(extension);
        if (!regularFile(stagedComponent)) continue;
        fs::path outputComponent = outputShapefile;
        outputComponent.replace_extension(extension);
        std::error_code copyError;
        if (!fs::copy_file(stagedComponent, outputComponent,
                           fs::copy_options::none, copyError) || copyError) {
            warn(output,
                 "Refusing to overwrite or partially replace DEM shapefile component: " +
                     pathToUtf8(outputComponent));
            return false;
        }
    }
    return regularFile(outputShapefile);
}

bool writeShapefile(const fs::path& outputShapefile,
                    const std::string& caseName,
                    PolygonizedRaster& polygonized,
                    std::ostream& output) {
    const auto stagingDirectory =
        createStagingDirectory(outputShapefile.parent_path(), output);
    if (!stagingDirectory) return false;
    const fs::path stagedShapefile =
        *stagingDirectory / outputShapefile.filename();

    GDALDriver* driver =
        GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    if (driver == nullptr) {
        warn(output, "Failed to write shapefile: ESRI Shapefile driver unavailable.");
        return false;
    }
    {
        DatasetPtr destination(driver->Create(pathToUtf8(stagedShapefile).c_str(),
                                              0, 0, 0, GDT_Unknown, nullptr));
        if (!destination) {
            warn(output, "Failed to write shapefile: cannot create staged dataset.");
            return false;
        }
        char** rawOptions = nullptr;
        rawOptions = CSLSetNameValue(rawOptions, "ENCODING", "UTF-8");
        StringListPtr options(rawOptions);
        OGRLayer* destinationLayer = destination->CreateLayer(
            pathToUtf8(stagedShapefile.stem()).c_str(),
            polygonized.layer->GetSpatialRef(), wkbPolygon, options.get());
        OGRFieldDefn elevationField("elevation", OFTReal);
        elevationField.SetWidth(24);
        elevationField.SetPrecision(15);
        OGRFieldDefn caseField("casename", OFTString);
        caseField.SetWidth(80);
        if (destinationLayer == nullptr ||
            destinationLayer->CreateField(&elevationField) != OGRERR_NONE ||
            destinationLayer->CreateField(&caseField) != OGRERR_NONE) {
            warn(output, "Failed to write shapefile: cannot create destination schema.");
            return false;
        }
        const int destinationElevationIndex =
            destinationLayer->GetLayerDefn()->GetFieldIndex("elevation");
        polygonized.layer->ResetReading();
        while (OGRFeature* rawFeature = polygonized.layer->GetNextFeature()) {
            FeaturePtr sourceFeature(rawFeature);
            FeaturePtr destinationFeature(
                OGRFeature::CreateFeature(destinationLayer->GetLayerDefn()));
            if (!destinationFeature ||
                destinationFeature->SetFrom(sourceFeature.get(), TRUE) != OGRERR_NONE) {
                warn(output, "Failed to write shapefile: cannot write DEM feature.");
                return false;
            }
            // Fiona serializes pandas NaN as a DBF null while retaining
            // positive and negative infinity.  OGR otherwise writes the
            // literal text "nan" into an OFTReal Shapefile field.
            if (destinationElevationIndex >= 0 &&
                destinationFeature->IsFieldSetAndNotNull(
                    destinationElevationIndex) &&
                std::isnan(destinationFeature->GetFieldAsDouble(
                    destinationElevationIndex))) {
                destinationFeature->SetFieldNull(destinationElevationIndex);
            }
            destinationFeature->SetField("casename", caseName.c_str());
            if (destinationLayer->CreateFeature(destinationFeature.get()) != OGRERR_NONE) {
                warn(output, "Failed to write shapefile: cannot write DEM feature.");
                return false;
            }
        }
        if (destinationLayer->SyncToDisk() != OGRERR_NONE) {
            warn(output, "Failed to write shapefile: cannot flush destination dataset.");
            return false;
        }
    }
    if (!validateStagedShapefile(stagedShapefile,
                                 polygonized.featureCount, output)) {
        return false;
    }
    return publishStagedShapefile(stagedShapefile, outputShapefile, output);
}

#endif // LUW_HAS_GDAL

} // namespace

std::optional<DemTifCacheResult> ensureDemShapefileFromTif(
    const DemTifCacheRequest& request,
    std::ostream& output) {
    const fs::path terrainDirectory = request.projectHome / "terrain_db";
    std::error_code directoryError;
    if (!fs::is_directory(terrainDirectory, directoryError)) {
        directoryError.clear();
        fs::create_directories(terrainDirectory, directoryError);
        if (directoryError) {
            warn(output, "Failed to create terrain_db directory: " +
                             directoryError.message());
            return std::nullopt;
        }
        info(output, "Created terrain_db directory: " + pathToUtf8(terrainDirectory));
    }

    const auto safeOutput =
        safeOutputShapefile(terrainDirectory, request.caseName, output);
    if (!safeOutput) return std::nullopt;
    const fs::path outputShapefile = *safeOutput;
    if (regularFile(outputShapefile)) {
        info(output, "DEM shapefile already exists; skip GeoTIFF conversion.");
        return DemTifCacheResult{outputShapefile, {}, false, 0U};
    }
    fs::path collision;
    if (sidecarCollision(outputShapefile, collision)) {
        warn(output, "Refusing to overwrite existing DEM shapefile component: " +
                         pathToUtf8(collision));
        return std::nullopt;
    }

    std::vector<fs::path> candidates = tifCandidates(terrainDirectory);
    const bool fromTerrainDirectory = !candidates.empty();
    if (candidates.empty() && regularFile(request.installDatabaseDem)) {
        candidates.push_back(request.installDatabaseDem);
    }
    if (candidates.empty()) {
        info(output, "No DEM GeoTIFF found in terrain_db or install database.");
        return std::nullopt;
    }
    if (fromTerrainDirectory) {
        info(output, "Found DEM GeoTIFF(s) in terrain_db.");
    } else {
        info(output, "Using install database DEM: " + pathToUtf8(candidates.front()));
    }

    const DemLonLatBounds expanded = expandedBounds(request.targetBounds);
    info(output, "DEM cut bbox (EPSG:4326) expanded 20%: lon[" +
                     pythonFloat(expanded.lonMin) + ", " +
                     pythonFloat(expanded.lonMax) + "], lat[" +
                     pythonFloat(expanded.latMin) + ", " +
                     pythonFloat(expanded.latMax) + "]");

#if LUW_HAS_GDAL
    GDALAllRegister();
    for (const fs::path& candidate : candidates) {
        info(output, "Trying DEM GeoTIFF: " + pathToUtf8(candidate));
        auto polygonized = polygonizeCandidate(
            candidate, expanded, request, output);
        if (!polygonized) continue;
        info(output, "Writing shapefile: " + pathToUtf8(outputShapefile));
        if (!writeShapefile(outputShapefile, request.caseName, *polygonized, output)) {
            continue;
        }
        info(output, "DEM GeoTIFF conversion done.");
        info(output, "Feature count: " + std::to_string(polygonized->featureCount));
        info(output, "Elevation min: " + fixed(polygonized->minimum, 3));
        info(output, "Elevation max: " + fixed(polygonized->maximum, 3));
        info(output, "Elevation mean: " + fixed(polygonized->mean, 3));
        info(output, "Elevation range: " +
                         fixed(polygonized->maximum - polygonized->minimum, 3));
        return DemTifCacheResult{outputShapefile, candidate, true,
                                 polygonized->featureCount};
    }
    warn(output, "No GeoTIFF covers target bounds or contains valid cells.");
    return std::nullopt;
#else
    (void)expanded;
    warn(output,
         "GeoTIFF conversion requires a build with GDAL/OGR support.");
    return std::nullopt;
#endif
}

} // namespace luw::cli
