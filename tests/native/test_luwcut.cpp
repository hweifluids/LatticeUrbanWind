#include "luw/luwcut_command.hpp"

#include <QString>

#include <gdal_priv.h>
#include <ogr_api.h>
#include <ogrsf_frmts.h>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct DatasetCloser {
    void operator()(GDALDataset *dataset) const noexcept {
        if (dataset != nullptr)
            GDALClose(dataset);
    }
};

struct GeometryCloser {
    void operator()(OGRGeometry *geometry) const noexcept {
        if (geometry != nullptr)
            OGRGeometryFactory::destroyGeometry(geometry);
    }
};

struct TransformCloser {
    void operator()(OGRCoordinateTransformation *transform) const noexcept {
        if (transform != nullptr)
            OCTDestroyCoordinateTransformation(transform);
    }
};

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;
using GeometryPtr = std::unique_ptr<OGRGeometry, GeometryCloser>;
using TransformPtr =
    std::unique_ptr<OGRCoordinateTransformation, TransformCloser>;

std::string utf8(const fs::path &path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring()).toUtf8().toStdString();
#else
    return path.string();
#endif
}

void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

void requireContains(const std::string &text, std::string_view expected,
                     const std::string &context) {
    require(text.find(expected) != std::string::npos,
            context + ": expected output to contain '" + std::string(expected) +
                "'\nActual output:\n" + text);
}

std::vector<unsigned char> readBytes(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read retained preview bytes");
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

struct JfifDensity {
    int unit = 0;
    int x = 0;
    int y = 0;
};

std::optional<JfifDensity> readJfifDensity(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::nullopt;
    std::array<unsigned char, 256> bytes{};
    input.read(reinterpret_cast<char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    const std::size_t count = static_cast<std::size_t>(input.gcount());
    for (std::size_t index = 0U; index + 15U < count; ++index) {
        if (bytes[index] != 0xFFU || bytes[index + 1U] != 0xE0U)
            continue;
        const std::size_t payload = index + 4U;
        if (bytes[payload] != 'J' || bytes[payload + 1U] != 'F' ||
            bytes[payload + 2U] != 'I' || bytes[payload + 3U] != 'F' ||
            bytes[payload + 4U] != 0U)
            continue;
        return JfifDensity{
            bytes[payload + 7U],
            (static_cast<int>(bytes[payload + 8U]) << 8) |
                static_cast<int>(bytes[payload + 9U]),
            (static_cast<int>(bytes[payload + 10U]) << 8) |
                static_cast<int>(bytes[payload + 11U])};
    }
    return std::nullopt;
}

OGRSpatialReference epsg(int code) {
    OGRSpatialReference reference;
    require(reference.importFromEPSG(code) == OGRERR_NONE,
            "could not construct fixture CRS");
    reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return reference;
}

TransformPtr transform(OGRSpatialReference &source,
                       OGRSpatialReference &target) {
    TransformPtr result(OGRCreateCoordinateTransformation(&source, &target));
    require(result != nullptr,
            "could not construct fixture coordinate transform");
    return result;
}

GeometryPtr rectangle(double minX, double minY, double maxX, double maxY) {
    auto result = GeometryPtr(new OGRPolygon());
    OGRLinearRing ring;
    ring.addPoint(maxX, minY);
    ring.addPoint(maxX, maxY);
    ring.addPoint(minX, maxY);
    ring.addPoint(minX, minY);
    ring.addPoint(maxX, minY);
    require(dynamic_cast<OGRPolygon *>(result.get())->addRing(&ring) ==
                OGRERR_NONE,
            "could not construct fixture rectangle");
    return result;
}

void addRectangularRing(OGRPolygon &polygon, double minX, double minY,
                        double maxX, double maxY) {
    OGRLinearRing ring;
    ring.addPoint(maxX, minY);
    ring.addPoint(maxX, maxY);
    ring.addPoint(minX, maxY);
    ring.addPoint(minX, minY);
    ring.addPoint(maxX, minY);
    require(polygon.addRing(&ring) == OGRERR_NONE,
            "could not add fixture polygon ring");
}

GeometryPtr polygonWithHoles() {
    auto result = GeometryPtr(new OGRPolygon());
    auto *polygon = dynamic_cast<OGRPolygon *>(result.get());
    addRectangularRing(*polygon, 3.0020, 1.0005, 3.0030, 1.0015);
    addRectangularRing(*polygon, 3.00210, 1.00060, 3.002108, 1.000608);
    addRectangularRing(*polygon, 3.00240, 1.00085, 3.00245, 1.00090);
    return result;
}

GeometryPtr invalidBowTie() {
    auto result = GeometryPtr(new OGRPolygon());
    OGRLinearRing ring;
    ring.addPoint(3.0020, 1.0020);
    ring.addPoint(3.0028, 1.0028);
    ring.addPoint(3.0020, 1.0028);
    ring.addPoint(3.0028, 1.0020);
    ring.addPoint(3.0020, 1.0020);
    require(dynamic_cast<OGRPolygon *>(result.get())->addRing(&ring) ==
                OGRERR_NONE,
            "could not construct invalid fixture polygon");
    return result;
}

void writeFeature(OGRLayer &layer, OGRCoordinateTransformation *toSource,
                  GeometryPtr geometry, int height, const char *name) {
    require(geometry != nullptr, "fixture geometry is null");
    require(geometry->transform(toSource) == OGRERR_NONE,
            "could not project fixture geometry");
    OGRFeature feature(layer.GetLayerDefn());
    feature.SetField("height", height);
    feature.SetField("name", name);
    require(feature.SetGeometry(geometry.get()) == OGRERR_NONE,
            "could not attach fixture geometry");
    require(layer.CreateFeature(&feature) == OGRERR_NONE,
            "could not write fixture feature");
}

void createBuildingFixture(const fs::path &path) {
    GDALAllRegister();
    GDALDriver *driver =
        GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    require(driver != nullptr, "ESRI Shapefile fixture driver is unavailable");
    DatasetPtr dataset(
        driver->Create(utf8(path).c_str(), 0, 0, 0, GDT_Unknown, nullptr));
    require(dataset != nullptr, "could not create building fixture");
    OGRSpatialReference wgs84 = epsg(4326);
    OGRSpatialReference webMercator = epsg(3857);
    TransformPtr toSource = transform(wgs84, webMercator);
    OGRLayer *layer =
        dataset->CreateLayer("buildings", &webMercator, wkbPolygon, nullptr);
    require(layer != nullptr, "could not create building fixture layer");
    OGRFieldDefn height("height", OFTInteger);
    OGRFieldDefn name("name", OFTString);
    name.SetWidth(32);
    require(layer->CreateField(&height) == OGRERR_NONE,
            "could not create height fixture field");
    require(layer->CreateField(&name) == OGRERR_NONE,
            "could not create name fixture field");

    writeFeature(*layer, toSource.get(),
                 rectangle(3.0005, 1.0005, 3.0011, 1.0011), 10, "overlap_a");
    writeFeature(*layer, toSource.get(),
                 rectangle(3.0008, 1.0008, 3.0014, 1.0014), 25, "overlap_b");
    writeFeature(*layer, toSource.get(), polygonWithHoles(), 12, "holes");
    writeFeature(*layer, toSource.get(), invalidBowTie(), 7, "invalid_fixed");
    writeFeature(*layer, toSource.get(),
                 rectangle(3.00160, 1.00160, 3.00162, 1.00162), 30,
                 "tiny_drop");

    // Tiny polygons make total_bounds cover the requested deck rectangle, then
    // disappear under the same 20 m^2 area filter as the Python implementation.
    writeFeature(*layer, toSource.get(),
                 rectangle(2.999999, 0.999999, 3.000001, 1.000001), 2,
                 "bound_sw");
    writeFeature(*layer, toSource.get(),
                 rectangle(3.003999, 1.003999, 3.004001, 1.004001), 2,
                 "bound_ne");
    require(layer->SyncToDisk() == OGRERR_NONE,
            "could not flush building fixture");
}

void createNoCrsFixture(const fs::path &path) {
    GDALDriver *driver =
        GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    require(driver != nullptr,
            "ESRI Shapefile no-CRS fixture driver is unavailable");
    DatasetPtr dataset(
        driver->Create(utf8(path).c_str(), 0, 0, 0, GDT_Unknown, nullptr));
    require(dataset != nullptr, "could not create no-CRS building fixture");
    OGRLayer *layer =
        dataset->CreateLayer("buildings", nullptr, wkbPolygon, nullptr);
    require(layer != nullptr, "could not create no-CRS fixture layer");
    OGRFieldDefn code("code", OFTInteger);
    require(layer->CreateField(&code) == OGRERR_NONE,
            "could not create no-CRS fixture field");
    GeometryPtr geometry = rectangle(4.0, 2.0, 4.001, 2.001);
    OGRFeature feature(layer->GetLayerDefn());
    feature.SetField("code", 9);
    require(feature.SetGeometry(geometry.get()) == OGRERR_NONE,
            "could not attach no-CRS fixture geometry");
    require(layer->CreateFeature(&feature) == OGRERR_NONE,
            "could not write no-CRS fixture feature");
    require(layer->SyncToDisk() == OGRERR_NONE,
            "could not flush no-CRS fixture");
}

int flattenedType(const OGRGeometry *geometry) {
    return geometry != nullptr
               ? static_cast<int>(wkbFlatten(geometry->getGeometryType()))
               : static_cast<int>(wkbUnknown);
}

double geometryArea(const OGRGeometry *geometry) {
    return OGR_G_Area(
        reinterpret_cast<OGRGeometryH>(const_cast<OGRGeometry *>(geometry)));
}

void verifyOutput(const fs::path &path) {
    DatasetPtr dataset(static_cast<GDALDataset *>(
        GDALOpenEx(utf8(path).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
                   nullptr, nullptr, nullptr)));
    require(dataset != nullptr, "native luwcut output is unreadable");
    OGRLayer *layer = dataset->GetLayer(0);
    require(layer != nullptr, "native luwcut output has no layer");
    const OGRSpatialReference *sourceReference = layer->GetSpatialRef();
    require(sourceReference != nullptr, "native luwcut output lost its CRS");
    std::unique_ptr<OGRSpatialReference> reference(sourceReference->Clone());
    require(reference != nullptr, "native luwcut could not clone output CRS");
    reference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    (void)reference->AutoIdentifyEPSG();
    const char *authorityCode = reference->GetAuthorityCode(nullptr);
    require(authorityCode != nullptr && std::string(authorityCode) == "3857",
            "native luwcut did not restore the source CRS");
    require(layer->GetFeatureCount(TRUE) == 7,
            "native luwcut wrote the wrong number of footprints");

    const int heightIndex = layer->GetLayerDefn()->GetFieldIndex("height");
    const int nameIndex = layer->GetLayerDefn()->GetFieldIndex("name");
    const int cornerIndex = layer->GetLayerDefn()->GetFieldIndex("corner_id");
    require(heightIndex >= 0 && nameIndex >= 0 && cornerIndex >= 0,
            "native luwcut output fields are incomplete");
    OGRSpatialReference utm = epsg(32631);
    TransformPtr toUtm = transform(*reference, utm);
    int corners = 0;
    int holes = 0;
    bool sawMergedMaximum = false;
    bool sawInvalidRepair = false;
    layer->ResetReading();
    while (OGRFeature *raw = layer->GetNextFeature()) {
        std::unique_ptr<OGRFeature, decltype(&OGRFeature::DestroyFeature)>
            feature(raw, &OGRFeature::DestroyFeature);
        const OGRGeometry *geometry = feature->GetGeometryRef();
        require(flattenedType(geometry) == wkbPolygon ||
                    flattenedType(geometry) == wkbMultiPolygon,
                "native luwcut output contains a non-polygonal feature");
        if (feature->IsFieldSetAndNotNull(cornerIndex)) {
            ++corners;
            GeometryPtr projected(geometry->clone());
            require(projected->transform(toUtm.get()) == OGRERR_NONE,
                    "could not project output corner marker");
            const double area = geometryArea(projected.get());
            require(area > 0.95 && area < 1.05,
                    "native luwcut corner marker is not 1 m by 1 m");
            require(std::abs(feature->GetFieldAsDouble(heightIndex) - 1.0) <
                        1e-12,
                    "native luwcut corner marker height changed");
            continue;
        }
        const std::string nameValue = feature->GetFieldAsString(nameIndex);
        if (nameValue == "overlap_a") {
            sawMergedMaximum =
                std::abs(feature->GetFieldAsDouble(heightIndex) - 25.0) < 1e-12;
        } else if (nameValue == "invalid_fixed") {
            sawInvalidRepair = true;
        }
        const auto countHoles = [&](const OGRGeometry *candidate) {
            int result = 0;
            if (flattenedType(candidate) == wkbPolygon) {
                result += dynamic_cast<const OGRPolygon *>(candidate)
                              ->getNumInteriorRings();
            } else if (flattenedType(candidate) == wkbMultiPolygon) {
                const auto *collection =
                    dynamic_cast<const OGRGeometryCollection *>(candidate);
                for (int index = 0; index < collection->getNumGeometries();
                     ++index) {
                    result += dynamic_cast<const OGRPolygon *>(
                                  collection->getGeometryRef(index))
                                  ->getNumInteriorRings();
                }
            }
            return result;
        };
        holes += countHoles(geometry);
    }
    require(corners == 4,
            "native luwcut did not append exactly four corner markers");
    require(holes == 1, "native luwcut did not remove only the sub-5 m^2 hole");
    require(
        sawMergedMaximum,
        "native luwcut did not retain the maximum height for an overlap union");
    require(sawInvalidRepair,
            "native luwcut dropped a repairable invalid polygon");
}

void testFullPipeline(const fs::path &root) {
    const fs::path project = root / "projected-source";
    const fs::path buildings = project / "building_db";
    fs::create_directories(buildings);
    const fs::path deck = project / "conf.luw";
    std::ofstream(deck) << "[metadata]\n"
                        << "casename = native_cut\n"
                        << "[domain]\n"
                        << "cut_lon_manual = [3.000000, 3.004000]\n"
                        << "cut_lat_manual = [1.000000, 1.004000]\n";
    createBuildingFixture(buildings / "buildings.shp");

    std::ostringstream output;
    std::ostringstream error;
    const int status =
        luw::cli::run_luwcut({utf8(deck), "--validate"}, output, error);
    require(status == 0, "native luwcut full pipeline failed: " + error.str());
    require(error.str().empty(), "native luwcut emitted an unexpected error");
    requireContains(output.str(), "Original CRS: EPSG:3857",
                    "source CRS message");
    requireContains(output.str(), "Clipping to rotated-rectangle bbox",
                    "rotated clipping message");
    requireContains(output.str(),
                    "Merged components with size greater than one: 1",
                    "overlap union message");
    requireContains(output.str(), "Small interior rings removed: 1",
                    "hole filtering message");
    requireContains(output.str(),
                    "Processing finished. Feature count written: 7",
                    "output feature-count message");
    requireContains(output.str(),
                    "Preview generation skipped. Use --preview to enable.",
                    "preview compatibility message");
    verifyOutput(project / "proj_temp" / "cutted_shp" / "native_cut.shp");
}

void testMissingCrsAndFallbackHeight(const fs::path &root) {
    const fs::path project = root / "missing-crs";
    const fs::path buildings = project / "building_db";
    fs::create_directories(buildings);
    const fs::path deck = project / "conf.luw";
    std::ofstream(deck) << "casename = no_crs_cut\n"
                        << "cut_lon_manual = [4.000000, 4.001000]\n"
                        << "cut_lat_manual = [2.000000, 2.001000]\n";
    createNoCrsFixture(buildings / "buildings.shp");

    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_luwcut({utf8(deck), "--preview"}, output, error) == 0,
            "native luwcut no-CRS pipeline failed: " + error.str());
    requireContains(output.str(),
                    "Notice: source SHP has no CRS. Assuming EPSG 4326",
                    "missing CRS compatibility notice");
    requireContains(output.str(), "Height column for processing: elevation",
                    "fallback height field message");
    requireContains(output.str(),
                    "Preview step uses auto detected UTM CRS for plotting:",
                    "native preview CRS message");
    requireContains(output.str(),
                    "Preview raster size: 38400x38400 pixels (8x8 inches at 4800 DPI).",
                    "native preview dimensions message");
    requireContains(output.str(),
                    "Encoding preview JPEG scanline-by-scanline with bounded memory.",
                    "native preview bounded-memory message");
    requireContains(output.str(), "Preview generation finished.",
                    "native preview completion message");

    const fs::path outputPath =
        project / "proj_temp" / "cutted_shp" / "no_crs_cut.shp";
    DatasetPtr dataset(static_cast<GDALDataset *>(
        GDALOpenEx(utf8(outputPath).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
                   nullptr, nullptr, nullptr)));
    require(dataset != nullptr, "native luwcut no-CRS output is unreadable");
    OGRLayer *layer = dataset->GetLayer(0);
    require(layer != nullptr && layer->GetFeatureCount(TRUE) == 5,
            "native luwcut no-CRS output feature count changed");
    require(layer->GetSpatialRef() != nullptr,
            "native luwcut did not assign EPSG:4326 to no-CRS input");
    const int elevation = layer->GetLayerDefn()->GetFieldIndex("elevation");
    const int corner = layer->GetLayerDefn()->GetFieldIndex("corner_id");
    require(elevation >= 0 && corner >= 0,
            "native luwcut did not add fallback height/corner fields");
    int cornerCount = 0;
    layer->ResetReading();
    while (OGRFeature *raw = layer->GetNextFeature()) {
        std::unique_ptr<OGRFeature, decltype(&OGRFeature::DestroyFeature)>
            feature(raw, &OGRFeature::DestroyFeature);
        if (!feature->IsFieldSetAndNotNull(corner)) {
            require(
                !feature->IsFieldSetAndNotNull(elevation),
                "native luwcut assigned fallback height to a real building");
            continue;
        }
        ++cornerCount;
        require(std::abs(feature->GetFieldAsDouble(elevation) - 1.0) < 1e-12,
                "native luwcut fallback corner height changed");
    }
    require(cornerCount == 4,
            "native luwcut no-CRS case lost fallback corner markers");

    const fs::path previewPath =
        project / "proj_temp" / "cutted_shp" / "no_crs_cut_preview.jpg";
    DatasetPtr preview(static_cast<GDALDataset *>(
        GDALOpenEx(utf8(previewPath).c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY,
                   nullptr, nullptr, nullptr)));
    require(preview != nullptr, "native luwcut preview JPEG is unreadable");
    require(preview->GetRasterXSize() == 38400 &&
                preview->GetRasterYSize() == 38400 &&
                preview->GetRasterCount() == 3,
            "native luwcut preview raster shape changed");
    const std::optional<JfifDensity> density = readJfifDensity(previewPath);
    require(density && density->unit == 1 && density->x == 4800 &&
                density->y == 4800,
            "native luwcut preview does not carry 4800-DPI JFIF metadata");
}

void testExistingPreviewIsPreserved(const fs::path &root) {
    const fs::path project = root / "existing-preview";
    const fs::path buildings = project / "building_db";
    const fs::path outputDirectory = project / "proj_temp" / "cutted_shp";
    fs::create_directories(buildings);
    fs::create_directories(outputDirectory);
    const fs::path deck = project / "conf.luw";
    std::ofstream(deck) << "casename = preserve_preview\n"
                        << "cut_lon_manual = [4.000000, 4.001000]\n"
                        << "cut_lat_manual = [2.000000, 2.001000]\n";
    createNoCrsFixture(buildings / "buildings.shp");
    const fs::path previewPath =
        outputDirectory / "preserve_preview_preview.jpg";
    const std::vector<unsigned char> sentinel{
        'd', 'o', '-', 'n', 'o', 't', '-', 'o', 'v', 'e', 'r', 'w', 'r', 'i', 't', 'e'};
    {
        std::ofstream preview(previewPath, std::ios::binary);
        require(static_cast<bool>(preview),
                "cannot create existing preview safety fixture");
        preview.write(reinterpret_cast<const char *>(sentinel.data()),
                      static_cast<std::streamsize>(sentinel.size()));
    }

    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_luwcut({utf8(deck), "--preview"}, output, error) == 0,
            "existing-preview safety run failed: " + error.str());
    require(readBytes(previewPath) == sentinel,
            "luwcut overwrote a pre-existing preview image");
    requireContains(
        output.str(),
        "Preview image already exists; preserving it without overwrite:",
        "existing preview preservation diagnostic");
}

void testUsageAndUniqueShapefileContract(const fs::path &root) {
    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_luwcut({}, output, error) == 2,
            "native luwcut missing-argument exit code changed");
    requireContains(output.str(), "Usage: python 2_shpCutter.py",
                    "usage message");

    const fs::path project = root / "no-shapefile";
    fs::create_directories(project / "building_db");
    const fs::path deck = project / "conf.luw";
    std::ofstream(deck) << "casename = missing_shp\n"
                        << "cut_lon_manual = [3.0, 3.1]\n"
                        << "cut_lat_manual = [1.0, 1.1]\n";
    bool threw = false;
    try {
        (void)luw::cli::run_luwcut({utf8(deck)}, output, error);
    } catch (const std::runtime_error &exception) {
        threw = true;
        requireContains(exception.what(), "Expected exactly one .shp",
                        "unique SHP error");
        requireContains(exception.what(), "found 0", "unique SHP count");
    }
    require(threw, "native luwcut accepted a project without a building SHP");

    const fs::path multipleProject = root / "multiple-shapefiles";
    const fs::path multipleBuildings = multipleProject / "building_db";
    fs::create_directories(multipleBuildings);
    const fs::path multipleDeck = multipleProject / "conf.luw";
    std::ofstream(multipleDeck) << "casename = multiple_shp\n"
                                << "cut_lon_manual = [3.0, 3.1]\n"
                                << "cut_lat_manual = [1.0, 1.1]\n";
    std::ofstream(multipleBuildings / "a.shp") << "retained fixture A\n";
    std::ofstream(multipleBuildings / "b.shp") << "retained fixture B\n";
    threw = false;
    try {
        (void)luw::cli::run_luwcut({utf8(multipleDeck)}, output, error);
    } catch (const std::runtime_error &exception) {
        threw = true;
        requireContains(exception.what(), "Expected exactly one .shp",
                        "multiple SHP error");
        requireContains(exception.what(), "found 2", "multiple SHP count");
    }
    require(threw,
            "native luwcut accepted a project with multiple building SHPs");
}

} // namespace

int main() {
    try {
        const auto nonce = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        const fs::path root = fs::path(LUW_LUWCUT_TEST_WORK_ROOT) /
                              ("run_" + std::to_string(nonce));
        fs::create_directories(root);
        testFullPipeline(root);
        testMissingCrsAndFallbackHeight(root);
        testExistingPreviewIsPreserved(root);
        testUsageAndUniqueShapefileContract(root);
        std::cout << "Native luwcut compatibility tests passed.\n"
                  << "Generated fixtures retained at: " << utf8(root) << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "Native luwcut compatibility test failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
