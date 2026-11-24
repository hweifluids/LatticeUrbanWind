#include "luw/deck_document.hpp"
#include "luw/luwvox_command.hpp"
#include "luw/preprocess_compute_config.hpp"

#include <QString>

#include <gdal_priv.h>
#include <ogrsf_frmts.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string utf8(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring()).toUtf8().toStdString();
#else
    return path.string();
#endif
}

QString qpath(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring());
#else
    return QString::fromUtf8(path.string());
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
    fs::path root = fs::path(LUW_LUWVOX_TEST_WORK_ROOT) /
                    ("run-" + std::to_string(stamp) + "-" + std::to_string(process));
    fs::create_directories(root);
    return root;
}

void writeDeck(const fs::path& path, const std::string& caseName) {
    std::ofstream output(path, std::ios::trunc);
    require(static_cast<bool>(output), "cannot create test deck");
    output << "casename = \"" << caseName << "\"\n"
           << "cut_lon_manual = [-0.001, 0.001]\n"
           << "cut_lat_manual = [51.000, 51.002]\n"
           << "base_height = 10\n"
           << "z_limit = 100\n"
           << "terr_voxel_approach = \"idw\"\n"
           << "terr_voxel_height_field = \"auto\"\n"
           << "terr_voxel_ignore_under = 0\n"
           << "terr_voxel_grid_resolution = 30\n"
           << "terr_voxel_idw_sigma = 0\n"
           << "terr_voxel_idw_power = 2\n"
           << "terr_voxel_idw_neighbors = 4\n";
}

std::unique_ptr<OGRSpatialReference> wgs84() {
    auto reference = std::make_unique<OGRSpatialReference>();
    require(reference->importFromEPSG(4326) == OGRERR_NONE, "cannot create WGS84");
    reference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return reference;
}

void createBuildings(const fs::path& path) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    require(driver != nullptr, "ESRI Shapefile driver unavailable");
    fs::create_directories(path.parent_path());
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), 0, 0, 0, GDT_Unknown, nullptr));
    require(dataset != nullptr, "cannot create building shapefile");
    auto reference = wgs84();
    OGRLayer* layer = dataset->CreateLayer("buildings", reference.get(), wkbPolygon, nullptr);
    require(layer != nullptr, "cannot create building layer");
    OGRFieldDefn height("Height", OFTReal);
    require(layer->CreateField(&height) == OGRERR_NONE, "cannot create Height field");

    OGRFeature* raw = OGRFeature::CreateFeature(layer->GetLayerDefn());
    require(raw != nullptr, "cannot create building feature");
    std::unique_ptr<OGRFeature, decltype(&OGRFeature::DestroyFeature)>
        feature(raw, &OGRFeature::DestroyFeature);
    feature->SetField("Height", 20.0);
    OGRLinearRing ring;
    ring.addPoint(-0.00045, 51.00070);
    ring.addPoint(0.00045, 51.00070);
    ring.addPoint(0.00045, 51.00130);
    ring.addPoint(-0.00045, 51.00130);
    ring.addPoint(-0.00045, 51.00070);
    OGRPolygon polygon;
    polygon.addRing(&ring);
    feature->SetGeometry(&polygon);
    require(layer->CreateFeature(feature.get()) == OGRERR_NONE, "cannot write building feature");
}

void createDem(const fs::path& path) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    require(driver != nullptr, "ESRI Shapefile driver unavailable");
    fs::create_directories(path.parent_path());
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), 0, 0, 0, GDT_Unknown, nullptr));
    require(dataset != nullptr, "cannot create DEM shapefile");
    auto reference = wgs84();
    OGRLayer* layer = dataset->CreateLayer("dem", reference.get(), wkbPoint, nullptr);
    require(layer != nullptr, "cannot create DEM layer");
    OGRFieldDefn elevation("elevation", OFTReal);
    require(layer->CreateField(&elevation) == OGRERR_NONE, "cannot create elevation field");
    const std::array<std::array<double, 3>, 9> samples{{
        {{-0.0010, 51.0000, 100.0}}, {{0.0, 51.0000, 104.0}}, {{0.0010, 51.0000, 108.0}},
        {{-0.0010, 51.0010, 105.0}}, {{0.0, 51.0010, 112.0}}, {{0.0010, 51.0010, 116.0}},
        {{-0.0010, 51.0020, 109.0}}, {{0.0, 51.0020, 117.0}}, {{0.0010, 51.0020, 121.0}},
    }};
    for (const auto& sample : samples) {
        OGRFeature* raw = OGRFeature::CreateFeature(layer->GetLayerDefn());
        require(raw != nullptr, "cannot create DEM feature");
        std::unique_ptr<OGRFeature, decltype(&OGRFeature::DestroyFeature)>
            feature(raw, &OGRFeature::DestroyFeature);
        feature->SetField("elevation", sample[2]);
        OGRPoint point(sample[0], sample[1]);
        feature->SetGeometry(&point);
        require(layer->CreateFeature(feature.get()) == OGRERR_NONE, "cannot write DEM feature");
    }
}

void createDemRaster(const fs::path& path) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    require(driver != nullptr, "GTiff driver unavailable");
    fs::create_directories(path.parent_path());
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), 8, 8, 1,
                                      GDT_Float32, nullptr));
    require(dataset != nullptr, "cannot create DEM GeoTIFF");
    double transform[6]{-0.002, 0.0005, 0.0, 51.003, 0.0, -0.0005};
    require(dataset->SetGeoTransform(transform) == CE_None,
            "cannot set DEM GeoTIFF transform");
    auto reference = wgs84();
    require(dataset->SetSpatialRef(reference.get()) == CE_None,
            "cannot set DEM GeoTIFF CRS");
    std::vector<float> values(64U);
    for (std::size_t row = 0U; row < 8U; ++row) {
        for (std::size_t column = 0U; column < 8U; ++column) {
            values[row * 8U + column] =
                static_cast<float>(100.0 + 2.0 * row + column);
        }
    }
    GDALRasterBand* band = dataset->GetRasterBand(1);
    require(band != nullptr &&
                band->RasterIO(GF_Write, 0, 0, 8, 8, values.data(), 8, 8,
                               GDT_Float32, 0, 0, nullptr) == CE_None,
            "cannot write DEM GeoTIFF values");
}

void pickleU32(std::vector<unsigned char>& bytes, std::uint32_t value) {
    bytes.push_back(static_cast<unsigned char>(value & 0xFFU));
    bytes.push_back(static_cast<unsigned char>((value >> 8U) & 0xFFU));
    bytes.push_back(static_cast<unsigned char>((value >> 16U) & 0xFFU));
    bytes.push_back(static_cast<unsigned char>((value >> 24U) & 0xFFU));
}

void pickleString(std::vector<unsigned char>& bytes, const std::string& value) {
    bytes.push_back('X');
    pickleU32(bytes, static_cast<std::uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void pickleDoubleValue(std::vector<unsigned char>& bytes, double value) {
    bytes.push_back('G');
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    for (int shift = 56; shift >= 0; shift -= 8) {
        bytes.push_back(static_cast<unsigned char>((bits >> shift) & 0xFFU));
    }
}

void pickleSmallInteger(std::vector<unsigned char>& bytes, std::size_t value) {
    if (value <= 255U) {
        bytes.push_back('K');
        bytes.push_back(static_cast<unsigned char>(value));
    } else {
        bytes.push_back('M');
        bytes.push_back(static_cast<unsigned char>(value & 0xFFU));
        bytes.push_back(static_cast<unsigned char>((value >> 8U) & 0xFFU));
    }
}

void pickleArray(std::vector<unsigned char>& bytes, const std::vector<double>& values,
                 const std::vector<std::size_t>& shape) {
    const auto global = [&](const std::string& module, const std::string& name) {
        bytes.push_back('c');
        bytes.insert(bytes.end(), module.begin(), module.end());
        bytes.push_back('\n');
        bytes.insert(bytes.end(), name.begin(), name.end());
        bytes.push_back('\n');
    };
    global("numpy._core.multiarray", "_reconstruct");
    global("numpy", "ndarray");
    pickleSmallInteger(bytes, 0U);
    bytes.push_back(0x85); // TUPLE1
    bytes.push_back('C'); bytes.push_back(1); bytes.push_back('b');
    bytes.push_back(0x87); // TUPLE3
    bytes.push_back('R');
    bytes.push_back('(');
    pickleSmallInteger(bytes, 1U);
    for (const std::size_t extent : shape) pickleSmallInteger(bytes, extent);
    bytes.push_back(shape.size() == 1U ? 0x85 : (shape.size() == 2U ? 0x86 : 't'));
    global("numpy", "dtype");
    pickleString(bytes, "f8");
    bytes.push_back(0x89); bytes.push_back(0x88); bytes.push_back(0x87);
    bytes.push_back('R');
    bytes.push_back('(');
    pickleSmallInteger(bytes, 3U);
    pickleString(bytes, "<");
    bytes.push_back('N'); bytes.push_back('N'); bytes.push_back('N');
    bytes.push_back('J'); pickleU32(bytes, 0xFFFFFFFFU);
    bytes.push_back('J'); pickleU32(bytes, 0xFFFFFFFFU);
    pickleSmallInteger(bytes, 0U);
    bytes.push_back('t');
    bytes.push_back('b');
    bytes.push_back(0x89);
    bytes.push_back('B');
    pickleU32(bytes, static_cast<std::uint32_t>(values.size() * sizeof(double)));
    for (const double value : values) {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
        for (int shift = 0; shift <= 56; shift += 8) {
            bytes.push_back(static_cast<unsigned char>((bits >> shift) & 0xFFU));
        }
    }
    bytes.push_back('t');
    bytes.push_back('b');
}

void createProtocol3DemPickle(const fs::path& path) {
    std::vector<unsigned char> bytes{0x80, 0x03, '}', '('};
    const auto key = [&](const std::string& value) { pickleString(bytes, value); };
    key("dem_grid");
    pickleArray(bytes, {0.0, 2.0, 4.0, 3.0, 6.0, 9.0, 5.0, 9.0, 12.0}, {3U, 3U});
    key("x_grid"); pickleArray(bytes, {0.0, 0.001, 0.002}, {3U});
    key("y_grid"); pickleArray(bytes, {0.0, 0.001, 0.002}, {3U});
    key("dx"); pickleDoubleValue(bytes, 0.001);
    key("dy"); pickleDoubleValue(bytes, 0.001);
    key("base_height"); pickleDoubleValue(bytes, 10.0);
    key("rotate_deg"); pickleDoubleValue(bytes, 0.0);
    key("pivot_xy"); pickleDoubleValue(bytes, 0.0); pickleDoubleValue(bytes, 51.001); bytes.push_back(0x86);
    key("x_min"); pickleDoubleValue(bytes, -0.001);
    key("y_min"); pickleDoubleValue(bytes, 51.0);
    key("x_max"); pickleDoubleValue(bytes, 0.001);
    key("y_max"); pickleDoubleValue(bytes, 51.002);
    bytes.push_back('u');
    bytes.push_back('.');
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot create protocol-3 DEM pickle");
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(output), "cannot write protocol-3 DEM pickle");
}

std::uint32_t littleU32(const std::array<unsigned char, 4>& bytes) {
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8U) |
           (std::uint32_t(bytes[2]) << 16U) | (std::uint32_t(bytes[3]) << 24U);
}

float readFloat(std::istream& input) {
    std::array<unsigned char, 4> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), 4);
    return std::bit_cast<float>(littleU32(bytes));
}

struct StlSummary {
    std::uint32_t triangles = 0;
    std::array<double, 3> minimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> maximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
};

StlSummary summarizeStl(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read generated STL");
    input.seekg(80, std::ios::beg);
    std::array<unsigned char, 4> countBytes{};
    input.read(reinterpret_cast<char*>(countBytes.data()), 4);
    StlSummary summary;
    summary.triangles = littleU32(countBytes);
    for (std::uint32_t triangle = 0; triangle < summary.triangles; ++triangle) {
        for (int component = 0; component < 3; ++component) (void)readFloat(input);
        for (int vertex = 0; vertex < 3; ++vertex) {
            for (int component = 0; component < 3; ++component) {
                const double value = readFloat(input);
                summary.minimum[static_cast<std::size_t>(component)] =
                    std::min(summary.minimum[static_cast<std::size_t>(component)], value);
                summary.maximum[static_cast<std::size_t>(component)] =
                    std::max(summary.maximum[static_cast<std::size_t>(component)], value);
            }
        }
        input.ignore(2);
    }
    require(static_cast<bool>(input), "truncated generated STL");
    return summary;
}

std::vector<std::array<double, 3>> readCsv(const fs::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read generated CSV");
    std::string line;
    std::getline(input, line);
    require(line == "x,y,elevation", "unexpected DEM CSV header");
    std::vector<std::array<double, 3>> rows;
    while (std::getline(input, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parsed(line);
        std::array<double, 3> row{};
        require(static_cast<bool>(parsed >> row[0] >> row[1] >> row[2]), "invalid DEM CSV row");
        rows.push_back(row);
    }
    return rows;
}

void validateDeckAndBounds(const fs::path& deckPath, const StlSummary& stl,
                           bool terrain) {
    const luw::DeckDocument deck = luw::DeckDocument::load(qpath(deckPath));
    const auto x = deck.pair(QStringLiteral("si_x_cfd"));
    const auto y = deck.pair(QStringLiteral("si_y_cfd"));
    const auto z = deck.pair(QStringLiteral("si_z_cfd"));
    require(x && y && z, "native luwvox did not write CFD extents");
    require(std::abs(stl.minimum[0]) < 1.0e-4 && std::abs(stl.minimum[1]) < 1.0e-4,
            "STL was not translated to the origin");
    require(std::abs(stl.maximum[0] - x->second) < 0.05,
            "STL/deck x extent mismatch");
    require(std::abs(stl.maximum[1] - y->second) < 0.05,
            "STL/deck y extent mismatch");
    require(std::abs(stl.minimum[2]) < 1.0e-5, "STL z minimum is not zero");
    require(stl.maximum[2] >= 30.0, "building top was not extruded to expected height");
    require(deck.integer(QStringLiteral("utm_epsg")) == std::optional<std::int64_t>(32631),
            "unexpected UTM EPSG metadata");
    require(deck.textOr(QStringLiteral("utm_crs")) == QStringLiteral("EPSG:32631"),
            "unexpected UTM CRS metadata");
    require(z->second >= stl.maximum[2], "CFD z extent is below STL top");
    if (terrain) require(z->second > 110.0, "terrain top did not affect CFD z extent");
}

void flatCase(const fs::path& root) {
    const fs::path caseRoot = root / "flat";
    const fs::path deckPath = caseRoot / "conf.luw";
    fs::create_directories(caseRoot / "proj_temp" / "cutted_shp");
    writeDeck(deckPath, "flat_case");
    createBuildings(caseRoot / "proj_temp" / "cutted_shp" / "flat_case.shp");
    std::ostringstream output, error;
    const int status = luw::cli::run_luwvox(
        {utf8(deckPath), "--dem-source", "pkl", "--smooth-sigma", "0"}, output, error);
    require(status == 0, "flat luwvox failed: " + error.str() + output.str());
    require(output.str().find("No DEM data available, will create flat terrain") != std::string::npos,
            "flat-terrain compatibility message missing");
    const fs::path stlPath = caseRoot / "proj_temp" / "flat_case.stl";
    require(fs::is_regular_file(stlPath), "flat STL filename mismatch");
    require(!fs::exists(caseRoot / "proj_temp" / "interpolated_dem.csv"),
            "flat run unexpectedly emitted DEM CSV");
    const StlSummary summary = summarizeStl(stlPath);
    require(summary.triangles > 20U, "flat STL has too few triangles");
    validateDeckAndBounds(deckPath, summary, false);
}

void demCase(const fs::path& root) {
    const fs::path caseRoot = root / "dem";
    const fs::path deckPath = caseRoot / "conf.luw";
    fs::create_directories(caseRoot / "proj_temp" / "cutted_shp");
    writeDeck(deckPath, "dem_case");
    createBuildings(caseRoot / "proj_temp" / "cutted_shp" / "dem_case.shp");
    createDem(caseRoot / "terrain_db" / "dem.shp");
    std::ostringstream output, error;
    const int status = luw::cli::run_luwvox(
        {utf8(deckPath), "--dem-source", "shp", "--smooth-sigma", "0",
         "--terrain-resolution", "30", "--idw-neighbors", "4"},
        output, error);
    require(status == 0, "DEM luwvox failed: " + error.str() + output.str());
    const fs::path stlPath = caseRoot / "proj_temp" / "dem_case_DEM.stl";
    require(fs::is_regular_file(stlPath), "DEM STL filename mismatch");
    const StlSummary summary = summarizeStl(stlPath);
    require(summary.triangles > 100U, "DEM STL has too few triangles");
    validateDeckAndBounds(deckPath, summary, true);
    const auto all = readCsv(caseRoot / "proj_temp" / "interpolated_dem.csv");
    const auto boundary = readCsv(caseRoot / "proj_temp" / "boundary_dem.csv");
    require(!all.empty() && !boundary.empty(), "DEM CSV is empty");
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto& row : all) {
        minimum = std::min(minimum, row[2]);
        maximum = std::max(maximum, row[2]);
    }
    require(std::abs(minimum) <= 0.001, "interpolated DEM was not normalized to zero");
    require(maximum > 5.0, "interpolated DEM lost relief");
    require(boundary.size() < all.size(), "boundary DEM did not select perimeter rows");
}

void geoTiffCacheCase(const fs::path& root) {
    const fs::path caseRoot = root / "dem_geotiff";
    const fs::path deckPath = caseRoot / "conf.luw";
    fs::create_directories(caseRoot / "proj_temp" / "cutted_shp");
    writeDeck(deckPath, "dem_geotiff_case");
    createBuildings(caseRoot / "proj_temp" / "cutted_shp" /
                    "dem_geotiff_case.shp");
    createDemRaster(caseRoot / "terrain_db" / "unrestricted-surface.tiff");
    std::ostringstream output, error;
    const int status = luw::cli::run_luwvox(
        {utf8(deckPath), "--dem-source", "shp", "--smooth-sigma", "0",
         "--terrain-resolution", "30", "--idw-neighbors", "4"},
        output, error);
    require(status == 0,
            "GeoTIFF-cache luwvox failed: " + error.str() + output.str());
    require(fs::is_regular_file(caseRoot / "terrain_db" /
                                "dem_geotiff_case.shp"),
            "luwvox did not create the <casename>.shp DEM cache");
    require(output.str().find("Found DEM GeoTIFF(s) in terrain_db.") !=
                std::string::npos &&
                output.str().find("DEM GeoTIFF conversion done.") !=
                    std::string::npos,
            "luwvox GeoTIFF fallback diagnostics are missing");
    const fs::path stlPath =
        caseRoot / "proj_temp" / "dem_geotiff_case_DEM.stl";
    require(fs::is_regular_file(stlPath),
            "GeoTIFF-cache luwvox did not emit a terrain STL");
    const StlSummary summary = summarizeStl(stlPath);
    require(summary.triangles > 100U,
            "GeoTIFF-cache terrain STL has too few triangles");
    validateDeckAndBounds(deckPath, summary, true);
}

void directGeoTiffCase(const fs::path& root) {
    const fs::path caseRoot = root / "dem_geotiff_direct";
    const fs::path deckPath = caseRoot / "conf.luw";
    const fs::path rasterPath = caseRoot / "input" / "surface.tiff";
    fs::create_directories(caseRoot / "proj_temp" / "cutted_shp");
    writeDeck(deckPath, "dem_geotiff_direct_case");
    createBuildings(caseRoot / "proj_temp" / "cutted_shp" /
                    "dem_geotiff_direct_case.shp");
    createDemRaster(rasterPath);
    std::ostringstream output, error;
    const int status = luw::cli::run_luwvox(
        {utf8(deckPath), "--dem-path", utf8(rasterPath),
         "--smooth-sigma", "0", "--terrain-resolution", "30",
         "--idw-neighbors", "4"},
        output, error);
    require(status == 0,
            "direct GeoTIFF luwvox failed: " + error.str() + output.str());
    require(output.str().find("DEM path: " + utf8(rasterPath)) !=
                std::string::npos &&
                output.str().find("Loaded 36 DEM points") !=
                    std::string::npos,
            "direct GeoTIFF block-read/filter diagnostics changed: " +
                output.str());
    const auto rows =
        readCsv(caseRoot / "proj_temp" / "interpolated_dem.csv");
    require(!rows.empty(), "direct GeoTIFF interpolation CSV is empty");
    const StlSummary summary = summarizeStl(
        caseRoot / "proj_temp" / "dem_geotiff_direct_case_DEM.stl");
    validateDeckAndBounds(deckPath, summary, true);
}

void pickleCase(const fs::path& root) {
    const fs::path caseRoot = root / "pkl";
    const fs::path deckPath = caseRoot / "conf.luw";
    fs::create_directories(caseRoot / "proj_temp" / "cutted_shp");
    writeDeck(deckPath, "pkl_case");
    createBuildings(caseRoot / "proj_temp" / "cutted_shp" / "pkl_case.shp");
    createProtocol3DemPickle(caseRoot / "proj_temp" / "dem_grid.pkl");
    std::ostringstream output, error;
    const int status = luw::cli::run_luwvox(
        {utf8(deckPath), "--dem-source", "pkl", "--no-reproject",
         "--terrain-resolution", "0.0005", "--smooth-sigma", "0",
         "--idw-neighbors", "4"}, output, error);
    require(status == 0, "PKL luwvox failed: " + error.str() + output.str());
    require(output.str().find("Using DEM data from pkl file (dem_grid.pkl)") != std::string::npos,
            "protocol-3 NumPy pickle was not accepted");
    require(fs::is_regular_file(caseRoot / "proj_temp" / "pkl_case_DEM.stl"),
            "PKL DEM STL filename mismatch");
    const auto rows = readCsv(caseRoot / "proj_temp" / "interpolated_dem.csv");
    require(rows.size() == 25U, "PKL interpolation grid dimensions changed");
}

struct KrigingVariantResult {
    int status = -1;
    std::vector<std::array<double, 3>> rows;
    std::string output;
    std::string error;
};

KrigingVariantResult
runTerrainVariant(const fs::path& root, const std::string& caseName,
                  const std::string& approach, bool useGpu,
                  const std::string& smoothSigma = "0") {
    const fs::path deckPath = root / "conf.luw";
    fs::create_directories(root / "proj_temp" / "cutted_shp");
    writeDeck(deckPath, caseName);
    createBuildings(root / "proj_temp" / "cutted_shp" / (caseName + ".shp"));
    createDem(root / "terrain_db" / "dem.shp");
    std::ostringstream output, error;
    luw::PreprocessComputeOverrides computeOverrides;
    computeOverrides.useGpu = useGpu;
    computeOverrides.gpuDevice = 0;
    const luw::PreprocessComputeOverrideScope computeScope(computeOverrides);
    const int status = luw::cli::run_luwvox(
        {utf8(deckPath), "--dem-source", "shp", "--smooth-sigma", smoothSigma,
         "--terrain-resolution", "60", "--idw-neighbors", "4",
         "--terrain-approach", approach},
        output, error);
    std::vector<std::array<double, 3>> rows;
    if (status == 0) {
        rows = readCsv(root / "proj_temp" / "interpolated_dem.csv");
    }
    return {status, std::move(rows), output.str(), error.str()};
}

void gpuKrigingParityCase(const fs::path& root) {
    const KrigingVariantResult cpu =
        runTerrainVariant(root / "kriging_cpu", "kriging_cpu", "kriging", false);
    require(cpu.status == 0,
            "CPU kriging failed: " + cpu.error + cpu.output);
    const KrigingVariantResult gpu =
        runTerrainVariant(root / "kriging_opencl", "kriging_opencl", "kriging", true);
    if (gpu.status != 0) {
        require(gpu.status == 1,
                "unavailable GPU kriging returned the wrong exit status");
        require(gpu.error.find(
                    "luwvox: error: GPU preprocessing was explicitly requested") !=
                    std::string::npos,
                "unavailable GPU kriging did not emit the strict backend error: " +
                    gpu.error + gpu.output);
        require(gpu.output.find("Falling back to CPU kriging") == std::string::npos,
                "explicit GPU selection silently fell back to the CPU");
        return;
    }

    require(cpu.rows.size() == gpu.rows.size() && !cpu.rows.empty(),
            "CPU/GPU kriging grid size mismatch");
    double maximumDifference = 0.0;
    for (std::size_t index = 0; index < cpu.rows.size(); ++index) {
        require(std::abs(cpu.rows[index][0] - gpu.rows[index][0]) <= 0.001 &&
                    std::abs(cpu.rows[index][1] - gpu.rows[index][1]) <= 0.001,
                "CPU/GPU kriging grid ordering mismatch");
        maximumDifference = std::max(maximumDifference,
                                     std::abs(cpu.rows[index][2] - gpu.rows[index][2]));
    }
    require(maximumDifference <= 0.10,
            "CPU/OpenCL kriging elevation difference exceeds 0.10 m");
    require(gpu.output.find("[INFO] OpenCL kriging backend:") != std::string::npos,
            "successful GPU-selected kriging run did not identify its OpenCL backend");
    require(gpu.output.find("Falling back to CPU kriging") == std::string::npos,
            "successful GPU-selected kriging run emitted a CPU fallback message");
    require(cpu.output.find("using local ordinary kriging") != std::string::npos,
            "CPU kriging backend message missing");
}

void gpuIdwAndGaussianParityCase(const fs::path& root) {
    const KrigingVariantResult cpu = runTerrainVariant(
        root / "idw_cpu", "idw_cpu", "idw", false, "0.8");
    require(cpu.status == 0, "CPU IDW failed: " + cpu.error + cpu.output);
    const KrigingVariantResult gpu = runTerrainVariant(
        root / "idw_opencl", "idw_opencl", "idw", true, "0.8");
    if (gpu.status != 0) {
        require(gpu.status == 1,
                "unavailable GPU IDW returned the wrong exit status");
        require(gpu.error.find(
                    "luwvox: error: GPU preprocessing was explicitly requested") !=
                    std::string::npos,
                "unavailable GPU IDW did not emit the strict backend error: " +
                    gpu.error + gpu.output);
        require(gpu.output.find("Falling back to CPU") == std::string::npos,
                "explicit GPU IDW silently fell back to CPU");
        return;
    }

    require(cpu.rows.size() == gpu.rows.size() && !cpu.rows.empty(),
            "CPU/GPU IDW grid size mismatch");
    double maximumDifference = 0.0;
    for (std::size_t index = 0; index < cpu.rows.size(); ++index) {
        require(std::abs(cpu.rows[index][0] - gpu.rows[index][0]) <= 0.001 &&
                    std::abs(cpu.rows[index][1] - gpu.rows[index][1]) <= 0.001,
                "CPU/GPU IDW grid ordering mismatch");
        maximumDifference = std::max(
            maximumDifference, std::abs(cpu.rows[index][2] - gpu.rows[index][2]));
    }
    require(maximumDifference <= 0.02,
            "CPU/OpenCL IDW+Gaussian elevation difference exceeds 0.02 m");
    require(gpu.output.find("[INFO] OpenCL IDW backend:") != std::string::npos,
            "GPU-selected IDW did not identify its OpenCL backend");
    require(gpu.output.find("[INFO] OpenCL Gaussian backend:") != std::string::npos,
            "GPU-selected smoothing did not identify its OpenCL backend");
    require(gpu.output.find("Falling back to CPU") == std::string::npos,
            "GPU-selected IDW/Gaussian emitted a CPU fallback message");
}

} // namespace

int main() {
    try {
        GDALAllRegister();
        const fs::path root = retainedRunRoot();
        flatCase(root);
        demCase(root);
        geoTiffCacheCase(root);
        directGeoTiffCase(root);
        pickleCase(root);
        gpuKrigingParityCase(root);
        gpuIdwAndGaussianParityCase(root);
        std::cout << "luwvox generated fixtures retained at " << utf8(root) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "luwvox test failure: " << exception.what() << '\n';
        return 1;
    }
}
