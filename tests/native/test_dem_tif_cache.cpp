#include "luw/dem_tif_cache.hpp"

#include <QString>

#include <gdal_priv.h>
#include <ogrsf_frmts.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

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

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;
using FeaturePtr = std::unique_ptr<OGRFeature, FeatureCloser>;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireContains(const std::string& text,
                     std::string_view expected,
                     const std::string& context) {
    require(text.find(expected) != std::string::npos,
            context + ": expected output to contain '" + std::string(expected) +
                "'\nActual output:\n" + text);
}

std::string utf8(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring()).toUtf8().toStdString();
#else
    return path.string();
#endif
}

fs::path retainedRunRoot() {
    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
#ifdef _WIN32
    const int process = _getpid();
#else
    const int process = getpid();
#endif
    const fs::path root = fs::path(LUW_DEM_TIF_CACHE_TEST_WORK_ROOT) /
        ("run-" + std::to_string(stamp) + "-" + std::to_string(process));
    fs::create_directories(root);
    return root;
}

void createRaster(const fs::path& path) {
    fs::create_directories(path.parent_path());
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    require(driver != nullptr, "GTiff fixture driver is unavailable");
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), 10, 10, 1,
                                      GDT_Float64, nullptr));
    require(dataset != nullptr, "cannot create GeoTIFF fixture");
    double transform[6]{0.0, 1.0, 0.0, 10.0, 0.0, -1.0};
    require(dataset->SetGeoTransform(transform) == CE_None,
            "cannot set GeoTIFF fixture transform");
    OGRSpatialReference wgs84;
    require(wgs84.importFromEPSG(4326) == OGRERR_NONE,
            "cannot construct WGS84 fixture CRS");
    wgs84.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    require(dataset->SetSpatialRef(&wgs84) == CE_None,
            "cannot set GeoTIFF fixture CRS");
    std::vector<double> values(100U);
    for (std::size_t row = 0U; row < 10U; ++row) {
        for (std::size_t column = 0U; column < 10U; ++column) {
            values[row * 10U + column] =
                static_cast<double>(row * 10U + column);
        }
    }
    // This sentinel is inside the legacy 20%-expanded crop and must not become
    // a polygon. The three retained values form the stable Python-oracle set.
    values[5U * 10U + 3U] = -9999.0;
    GDALRasterBand* band = dataset->GetRasterBand(1);
    require(band != nullptr, "GeoTIFF fixture has no raster band");
    require(band->SetNoDataValue(-9999.0) == CE_None,
            "cannot set GeoTIFF fixture nodata");
    require(band->RasterIO(GF_Write, 0, 0, 10, 10, values.data(), 10, 10,
                           GDT_Float64, 0, 0, nullptr) == CE_None,
            "cannot write GeoTIFF fixture values");
}

void createProjectedRasterWithoutNodata(const fs::path& path) {
    fs::create_directories(path.parent_path());
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    require(driver != nullptr, "GTiff projected fixture driver is unavailable");
    constexpr int width = 27;
    constexpr int height = 38;
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), width, height, 1,
                                      GDT_Float32, nullptr));
    require(dataset != nullptr, "cannot create projected GeoTIFF fixture");
    double transform[6]{642671.2405177942, 100.0, 0.0,
                        5543561.799950907, 0.0, -100.0};
    require(dataset->SetGeoTransform(transform) == CE_None,
            "cannot set projected GeoTIFF transform");
    OGRSpatialReference utm;
    require(utm.importFromEPSG(32631) == OGRERR_NONE,
            "cannot construct projected fixture CRS");
    utm.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    require(dataset->SetSpatialRef(&utm) == CE_None,
            "cannot set projected GeoTIFF CRS");
    std::vector<float> values(static_cast<std::size_t>(width * height));
    for (std::size_t index = 0U; index < values.size(); ++index) {
        values[index] = static_cast<float>(index + 1U);
    }
    GDALRasterBand* band = dataset->GetRasterBand(1);
    require(band != nullptr &&
                band->RasterIO(GF_Write, 0, 0, width, height, values.data(),
                               width, height, GDT_Float32, 0, 0,
                               nullptr) == CE_None,
            "cannot write projected GeoTIFF values");
}

void createNonFiniteRaster(const fs::path& path) {
    fs::create_directories(path.parent_path());
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    require(driver != nullptr, "GTiff nonfinite fixture driver is unavailable");
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), 10, 10, 1,
                                      GDT_Float32, nullptr));
    require(dataset != nullptr, "cannot create nonfinite GeoTIFF fixture");
    double transform[6]{0.0, 1.0, 0.0, 10.0, 0.0, -1.0};
    require(dataset->SetGeoTransform(transform) == CE_None,
            "cannot set nonfinite GeoTIFF transform");
    OGRSpatialReference reference;
    require(reference.importFromEPSG(4326) == OGRERR_NONE,
            "cannot construct nonfinite fixture CRS");
    reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    require(dataset->SetSpatialRef(&reference) == CE_None,
            "cannot set nonfinite GeoTIFF CRS");
    std::vector<float> values(100U);
    for (std::size_t index = 0U; index < values.size(); ++index) {
        values[index] = static_cast<float>(index);
    }
    values[5U * 10U + 3U] = std::numeric_limits<float>::quiet_NaN();
    values[5U * 10U + 4U] = std::numeric_limits<float>::infinity();
    values[6U * 10U + 3U] = -std::numeric_limits<float>::infinity();
    GDALRasterBand* band = dataset->GetRasterBand(1);
    require(band != nullptr &&
                band->RasterIO(GF_Write, 0, 0, 10, 10, values.data(),
                               10, 10, GDT_Float32, 0, 0,
                               nullptr) == CE_None,
            "cannot write nonfinite GeoTIFF values");
}

void createMultiStripRaster(const fs::path& path) {
    fs::create_directories(path.parent_path());
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    require(driver != nullptr, "GTiff multi-strip fixture driver is unavailable");
    constexpr int width = 16384;
    constexpr int height = 65;
    char tiled[] = "TILED=YES";
    char blockX[] = "BLOCKXSIZE=256";
    char blockY[] = "BLOCKYSIZE=16";
    char* options[]{tiled, blockX, blockY, nullptr};
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), width, height, 1,
                                      GDT_Float32, options));
    require(dataset != nullptr, "cannot create multi-strip GeoTIFF fixture");
    double transform[6]{0.0, 0.0001, 0.0, 0.0065, 0.0, -0.0001};
    require(dataset->SetGeoTransform(transform) == CE_None,
            "cannot set multi-strip GeoTIFF transform");
    OGRSpatialReference reference;
    require(reference.importFromEPSG(4326) == OGRERR_NONE,
            "cannot construct multi-strip fixture CRS");
    reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    require(dataset->SetSpatialRef(&reference) == CE_None,
            "cannot set multi-strip GeoTIFF CRS");
    std::vector<float> values(static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height),
                              7.0F);
    GDALRasterBand* band = dataset->GetRasterBand(1);
    require(band != nullptr && band->SetNoDataValue(-9999.0) == CE_None &&
                band->RasterIO(GF_Write, 0, 0, width, height, values.data(),
                               width, height, GDT_Float32, 0, 0,
                               nullptr) == CE_None,
            "cannot write multi-strip GeoTIFF values");
}

struct ShapefileSnapshot {
    std::vector<std::string> fieldNames;
    std::vector<int> fieldWidths;
    std::vector<int> fieldPrecisions;
    std::vector<double> elevations;
    std::vector<std::string> caseNames;
    std::vector<OGREnvelope> envelopes;
    std::size_t nullElevationCount = 0U;
    int authorityCode = 0;
};

ShapefileSnapshot inspectShapefile(const fs::path& path) {
    DatasetPtr dataset(static_cast<GDALDataset*>(GDALOpenEx(
        utf8(path).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
        nullptr, nullptr, nullptr)));
    require(dataset != nullptr, "cannot open generated DEM shapefile");
    OGRLayer* layer = dataset->GetLayer(0);
    require(layer != nullptr, "generated DEM shapefile has no layer");
    OGRFeatureDefn* definition = layer->GetLayerDefn();
    require(definition != nullptr, "generated DEM shapefile has no schema");
    ShapefileSnapshot result;
    for (int index = 0; index < definition->GetFieldCount(); ++index) {
        OGRFieldDefn* field = definition->GetFieldDefn(index);
        result.fieldNames.emplace_back(field->GetNameRef());
        result.fieldWidths.push_back(field->GetWidth());
        result.fieldPrecisions.push_back(field->GetPrecision());
    }
    if (const OGRSpatialReference* reference = layer->GetSpatialRef()) {
        const char* authority = reference->GetAuthorityCode(nullptr);
        if (authority != nullptr) result.authorityCode = std::stoi(authority);
    }
    const int elevationIndex = definition->GetFieldIndex("elevation");
    const int caseIndex = definition->GetFieldIndex("casename");
    require(elevationIndex >= 0, "generated cache lacks elevation field");
    require(caseIndex >= 0, "generated cache lacks casename field");
    layer->ResetReading();
    while (OGRFeature* rawFeature = layer->GetNextFeature()) {
        FeaturePtr feature(rawFeature);
        if (feature->IsFieldSetAndNotNull(elevationIndex)) {
            result.elevations.push_back(feature->GetFieldAsDouble(elevationIndex));
        } else {
            ++result.nullElevationCount;
        }
        result.caseNames.emplace_back(feature->GetFieldAsString(caseIndex));
        OGRGeometry* geometry = feature->GetGeometryRef();
        require(geometry != nullptr && !geometry->IsEmpty(),
                "generated cache has empty geometry");
        OGREnvelope envelope;
        geometry->getEnvelope(&envelope);
        result.envelopes.push_back(envelope);
    }
    return result;
}

std::string readAll(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read retained safety fixture");
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

void testTerrainDatabaseRaster(const fs::path& root) {
    const fs::path project = root / "terrain-source";
    const fs::path raster = project / "terrain_db" / "arbitrary-city-surface.tiff";
    createRaster(raster);
    const luw::cli::DemTifCacheRequest request{
        project, root / "missing-install-dem.tif", "oracle_case",
        {3.0, 5.0, 3.0, 5.0}};
    std::ostringstream output;
    const auto result = luw::cli::ensureDemShapefileFromTif(request, output);
    require(result.has_value(), "terrain_db GeoTIFF was not converted\n" + output.str());
    require(result->created, "first cache call did not report creation");
    require(result->sourceRaster == raster, "arbitrary .tiff source was not selected");
    require(result->shapefile == project / "terrain_db" / "oracle_case.shp",
            "cache path does not use <casename>.shp");
    require(result->featureCount == 3U,
            "20%-expanded crop produced unexpected feature count");
    requireContains(output.str(), "Found DEM GeoTIFF(s) in terrain_db.",
                    "terrain source notice");
    requireContains(output.str(),
                    "DEM cut bbox (EPSG:4326) expanded 20%: lon[2.6, 5.4], lat[2.6, 5.4]",
                    "expanded bbox notice");
    requireContains(output.str(), "Feature count: 3", "conversion summary");

    ShapefileSnapshot snapshot = inspectShapefile(result->shapefile);
    require(snapshot.fieldNames == std::vector<std::string>({"elevation", "casename"}),
            "generated cache schema differs from legacy lowercase field contract");
    require(snapshot.fieldWidths == std::vector<int>({24, 80}) &&
                snapshot.fieldPrecisions == std::vector<int>({15, 0}),
            "generated cache field widths differ from the Fiona legacy oracle");
    require(snapshot.authorityCode == 4326,
            "generated cache CRS differs from the EPSG:4326 legacy oracle");
    require(readAll(project / "terrain_db" / "oracle_case.cpg") == "UTF-8",
            "generated cache encoding marker differs from the legacy oracle");
    std::sort(snapshot.elevations.begin(), snapshot.elevations.end());
    require(snapshot.elevations == std::vector<double>({54.0, 63.0, 64.0}),
            "generated elevations differ from retained Python-oracle values");
    require(std::all_of(snapshot.caseNames.begin(), snapshot.caseNames.end(),
                        [](const std::string& value) {
                            return value == "oracle_case";
                        }),
            "casename attribute was not copied to every polygon");
    require(snapshot.envelopes.size() == 3U,
            "generated polygon envelopes are incomplete");
    std::vector<std::array<double, 4>> geometryBounds;
    for (const OGREnvelope& envelope : snapshot.envelopes) {
        geometryBounds.push_back(
            {envelope.MinX, envelope.MinY, envelope.MaxX, envelope.MaxY});
    }
    std::sort(geometryBounds.begin(), geometryBounds.end());
    require(geometryBounds == std::vector<std::array<double, 4>>({
                                  {3.0, 3.0, 4.0, 4.0},
                                  {4.0, 3.0, 5.0, 4.0},
                                  {4.0, 4.0, 5.0, 5.0}}),
            "generated polygon bounds differ from the retained Python oracle");

    const auto writeTime = fs::last_write_time(result->shapefile);
    const auto byteSize = fs::file_size(result->shapefile);
    std::ostringstream repeatOutput;
    const auto repeated =
        luw::cli::ensureDemShapefileFromTif(request, repeatOutput);
    require(repeated.has_value() && !repeated->created,
            "existing cache was not returned through the skip path");
    require(fs::last_write_time(result->shapefile) == writeTime &&
                fs::file_size(result->shapefile) == byteSize,
            "existing cache was modified by the skip path");
    requireContains(repeatOutput.str(),
                    "DEM shapefile already exists; skip GeoTIFF conversion.",
                    "existing cache notice");
}

void testInstallDatabaseFallback(const fs::path& root) {
    const fs::path raster = root / "install-fixture" / "database-dem.tif";
    createRaster(raster);
    const fs::path project = root / "install-fallback-project";
    const luw::cli::DemTifCacheRequest request{
        project, raster, "install_case", {3.0, 5.0, 3.0, 5.0}};
    std::ostringstream output;
    const auto result = luw::cli::ensureDemShapefileFromTif(request, output);
    require(result.has_value() && result->created,
            "installation database fallback did not create cache\n" + output.str());
    require(result->sourceRaster == raster,
            "installation database fallback selected the wrong raster");
    requireContains(output.str(), "Created terrain_db directory:",
                    "terrain_db creation notice");
    requireContains(output.str(), "Using install database DEM:",
                    "installation database notice");
    const ShapefileSnapshot snapshot = inspectShapefile(result->shapefile);
    require(snapshot.elevations.size() == 3U,
            "installation fallback cache differs from terrain_db cache");
}

void testSidecarCollisionIsPreserved(const fs::path& root) {
    const fs::path project = root / "sidecar-collision";
    const fs::path terrain = project / "terrain_db";
    createRaster(terrain / "source.tif");
    const fs::path existing = terrain / "blocked_case.dbf";
    {
        std::ofstream output(existing, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(output), "cannot create sidecar safety fixture");
        output << "do-not-overwrite";
    }
    const luw::cli::DemTifCacheRequest request{
        project, {}, "blocked_case", {3.0, 5.0, 3.0, 5.0}};
    std::ostringstream output;
    const auto result = luw::cli::ensureDemShapefileFromTif(request, output);
    require(!result.has_value(), "sidecar collision unexpectedly created a cache");
    require(readAll(existing) == "do-not-overwrite",
            "sidecar collision was overwritten");
    require(!fs::exists(terrain / "blocked_case.shp"),
            "sidecar collision created a partial .shp");
    requireContains(output.str(),
                    "Refusing to overwrite existing DEM shapefile component:",
                    "sidecar safety notice");
}

void testProjectedRasterioMaskCompatibility(const fs::path& root) {
    const fs::path project = root / "projected-no-nodata";
    const fs::path raster =
        project / "terrain_db" / "projected-no-nodata.tiff";
    createProjectedRasterWithoutNodata(raster);
    const luw::cli::DemTifCacheRequest request{
        project, {}, "projected_case", {5.0, 5.02, 50.0, 50.02}};
    std::ostringstream output;
    const auto result = luw::cli::ensureDemShapefileFromTif(request, output);
    require(result.has_value() && result->created,
            "projected no-nodata raster was not converted\n" + output.str());
    require(result->featureCount == 631U,
            "projected crop feature count differs from retained rasterio.mask oracle");
    const ShapefileSnapshot snapshot = inspectShapefile(result->shapefile);
    require(snapshot.authorityCode == 32631,
            "projected cache did not preserve EPSG:32631");
    require(snapshot.nullElevationCount == 0U &&
                snapshot.elevations.size() == 631U,
            "projected cache contains unexpected null elevations");
    require(std::count(snapshot.elevations.begin(), snapshot.elevations.end(), 0.0) == 3,
            "no-nodata mask fill does not match three zero-valued rasterio polygons");
    const auto [minimum, maximum] =
        std::minmax_element(snapshot.elevations.begin(), snapshot.elevations.end());
    require(*minimum == 0.0 && *maximum == 928.0,
            "projected cache value range differs from retained Python oracle");
    requireContains(output.str(), "Feature count: 631",
                    "projected rasterio-mask feature summary");
    requireContains(output.str(), "Elevation mean: 502.520",
                    "projected rasterio-mask mean summary");
}

void testNonFiniteRasterCompatibility(const fs::path& root) {
    const fs::path project = root / "nonfinite";
    createNonFiniteRaster(project / "terrain_db" / "nonfinite.tif");
    const luw::cli::DemTifCacheRequest request{
        project, {}, "nonfinite_case", {3.0, 5.0, 3.0, 5.0}};
    std::ostringstream output;
    const auto result = luw::cli::ensureDemShapefileFromTif(request, output);
    require(result.has_value() && result->featureCount == 5U,
            "NaN/Inf raster differs from retained Python feature count\n" +
                output.str());
    const ShapefileSnapshot snapshot = inspectShapefile(result->shapefile);
    require(snapshot.nullElevationCount == 1U &&
                snapshot.elevations.size() == 4U,
            "NaN shapefile field does not match Fiona null serialization");
    require(std::count_if(snapshot.elevations.begin(), snapshot.elevations.end(),
                          [](double value) { return std::isinf(value); }) == 2,
            "infinite elevations were not retained like the Python implementation");
    require(std::count(snapshot.elevations.begin(), snapshot.elevations.end(), 0.0) == 1 &&
                std::count(snapshot.elevations.begin(), snapshot.elevations.end(), 64.0) == 1,
            "finite values in the NaN/Inf fixture differ from the Python oracle");
    requireContains(output.str(), "Feature count: 5",
                    "nonfinite feature summary");
}

void testDoubleBufferedMultiStripRead(const fs::path& root) {
    const fs::path project = root / "multi-strip";
    createMultiStripRaster(project / "terrain_db" / "multi-strip.tif");
    const luw::cli::DemTifCacheRequest request{
        project, {}, "multi_strip_case", {0.2, 1.4, 0.0015, 0.0050}};
    std::ostringstream output;
    const auto result = luw::cli::ensureDemShapefileFromTif(request, output);
    require(result.has_value() && result->created,
            "multi-strip raster was not converted\n" + output.str());
    require(result->featureCount == 1U,
            "multi-strip block boundary split or dropped the constant polygon");
    const ShapefileSnapshot snapshot = inspectShapefile(result->shapefile);
    require(snapshot.nullElevationCount == 0U &&
                snapshot.elevations.size() == 1U &&
                snapshot.elevations.front() == 7.0,
            "multi-strip conversion changed the constant elevation");
}

void testCoveragePromptBranches(const fs::path& root) {
    const auto makeRequest = [&](const fs::path& project,
                                 const std::string& caseName) {
        createRaster(project / "terrain_db" / "source.tif");
        return luw::cli::DemTifCacheRequest{
            project, {}, caseName, {9.0, 11.0, 3.0, 5.0}};
    };

    auto cancelRequest = makeRequest(root / "coverage-cancel", "cancel_case");
    std::istringstream cancelInput("n\n");
    cancelRequest.coverageInput = &cancelInput;
    std::ostringstream cancelOutput;
    bool cancelled = false;
    try {
        (void)luw::cli::ensureDemShapefileFromTif(cancelRequest, cancelOutput);
    } catch (const luw::cli::DemTifCacheCancelled&) {
        cancelled = true;
    }
    require(cancelled, "coverage answer N did not cancel conversion");
    require(!fs::exists(cancelRequest.projectHome / "terrain_db" / "cancel_case.shp"),
            "coverage cancellation created a final shapefile");
    requireContains(cancelOutput.str(), "Continue anyway? (Y/N) [auto-continue in 5s]: ",
                    "coverage cancel prompt");
    requireContains(cancelOutput.str(), "[INFO] User canceled. Exiting.",
                    "coverage cancel result");

    auto continueRequest = makeRequest(root / "coverage-continue", "continue_case");
    std::istringstream continueInput("yes\n");
    continueRequest.coverageInput = &continueInput;
    std::ostringstream continueOutput;
    const auto continued =
        luw::cli::ensureDemShapefileFromTif(continueRequest, continueOutput);
    require(continued.has_value() && continued->created,
            "coverage answer Y did not continue conversion\n" +
                continueOutput.str());
    requireContains(continueOutput.str(),
                    "[WARN] Continuing despite bounds mismatch.",
                    "coverage continue result");

    auto timeoutRequest = makeRequest(root / "coverage-timeout", "timeout_case");
    timeoutRequest.coverageTimeoutSeconds = 0.0;
    std::ostringstream timeoutOutput;
    const auto timedOut =
        luw::cli::ensureDemShapefileFromTif(timeoutRequest, timeoutOutput);
    require(timedOut.has_value() && timedOut->created,
            "coverage timeout did not default to continue\n" +
                timeoutOutput.str());
    requireContains(timeoutOutput.str(),
                    "[WARN] No input received (timeout 5s). Continuing by default.",
                    "coverage timeout result");
}

void testUnsafeCaseNameAndCandidateFallback(const fs::path& root) {
    const fs::path unsafeProject = root / "unsafe-case";
    createRaster(unsafeProject / "terrain_db" / "source.tif");
    const luw::cli::DemTifCacheRequest unsafeRequest{
        unsafeProject, {}, "../escape", {3.0, 5.0, 3.0, 5.0}};
    std::ostringstream unsafeOutput;
    require(!luw::cli::ensureDemShapefileFromTif(unsafeRequest, unsafeOutput),
            "unsafe casename unexpectedly created a DEM cache");
    require(!fs::exists(unsafeProject / "escape.shp"),
            "unsafe casename escaped terrain_db");
    requireContains(unsafeOutput.str(), "Refusing unsafe DEM cache casename:",
                    "unsafe casename diagnostic");

    const fs::path fallbackProject = root / "candidate-fallback";
    const fs::path terrain = fallbackProject / "terrain_db";
    fs::create_directories(terrain);
    {
        std::ofstream corrupt(terrain / "00-corrupt.tif", std::ios::binary);
        require(static_cast<bool>(corrupt), "cannot create corrupt candidate fixture");
        corrupt << "not-a-geotiff";
    }
    const fs::path usable = terrain / "01-usable.tiff";
    createRaster(usable);
    const luw::cli::DemTifCacheRequest fallbackRequest{
        fallbackProject, {}, "fallback_case", {3.0, 5.0, 3.0, 5.0}};
    std::ostringstream fallbackOutput;
    const auto fallback =
        luw::cli::ensureDemShapefileFromTif(fallbackRequest, fallbackOutput);
    require(fallback.has_value() && fallback->sourceRaster == usable,
            "helper did not continue after the first invalid tif candidate\n" +
                fallbackOutput.str());
}

} // namespace

int main() {
    try {
        GDALAllRegister();
        const fs::path root = retainedRunRoot();
        testTerrainDatabaseRaster(root);
        testInstallDatabaseFallback(root);
        testSidecarCollisionIsPreserved(root);
        testProjectedRasterioMaskCompatibility(root);
        testNonFiniteRasterCompatibility(root);
        testDoubleBufferedMultiStripRead(root);
        testCoveragePromptBranches(root);
        testUnsafeCaseNameAndCandidateFallback(root);
        std::cout << "Native DEM GeoTIFF cache compatibility tests passed; retained artifacts: "
                  << utf8(root) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Native DEM GeoTIFF cache compatibility test failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
