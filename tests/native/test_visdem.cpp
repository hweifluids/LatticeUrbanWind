#include "luw/visdem_command.hpp"

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#endif

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string utf8_text(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const std::u8string text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_equal(std::string_view actual,
                   std::string_view expected,
                   const std::string& context) {
    require(actual == expected,
            context + " mismatch.\nExpected:\n" + std::string(expected)
                + "\nActual:\n" + std::string(actual));
}

#if LUW_HAS_GDAL

struct DatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) GDALClose(dataset);
    }
};

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;

void create_dem_fixture(const fs::path& path) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    require(driver != nullptr, "GTiff driver is unavailable");
    DatasetPtr dataset(driver->Create(utf8_text(path).c_str(), 6, 5, 1,
                                      GDT_Float32, nullptr));
    require(dataset != nullptr, "could not create DEM fixture");
    double transform[6]{500000.0, 100.0, 0.0, 5700000.0, 0.0, -100.0};
    require(dataset->SetGeoTransform(transform) == CE_None,
            "could not set DEM geotransform");
    OGRSpatialReference reference;
    require(reference.importFromEPSG(32630) == OGRERR_NONE,
            "could not initialize EPSG:32630");
    reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    require(dataset->SetSpatialRef(&reference) == CE_None,
            "could not set DEM CRS");
    GDALRasterBand* band = dataset->GetRasterBand(1);
    require(band != nullptr && band->SetNoDataValue(-9999.0) == CE_None,
            "could not set DEM nodata");
    std::array<float, 30> values{{
        100, 110, 120, 130, 140, 150,
         90, 100, 110, 120, 130, 140,
         80,  90, -9999, 110, 120, 130,
         70,  80,  90, 100, 110, 120,
         60,  70,  80,  90, 100, 110}};
    require(band->RasterIO(GF_Write, 0, 0, 6, 5, values.data(), 6, 5,
                           GDT_Float32, 0, 0, nullptr) == CE_None,
            "could not write DEM fixture pixels");
    dataset.reset();
}

std::vector<char> read_bytes(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "could not read fixture bytes: " + utf8_text(path));
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

struct Snapshot {
    std::vector<char> bytes;
    fs::file_time_type write_time;
};

Snapshot snapshot(const fs::path& path) {
    return {read_bytes(path), fs::last_write_time(path)};
}

void require_unchanged(const fs::path& path,
                       const Snapshot& before,
                       const std::string& context) {
    require(read_bytes(path) == before.bytes, context + " changed bytes");
    require(fs::last_write_time(path) == before.write_time,
            context + " changed modification time");
}

void require_png(const fs::path& path, int width, int height) {
    const std::vector<char> bytes = read_bytes(path);
    constexpr std::array<unsigned char, 8> signature{
        137, 80, 78, 71, 13, 10, 26, 10};
    require(bytes.size() > signature.size(), "PNG is empty: " + utf8_text(path));
    require(std::equal(signature.begin(), signature.end(), bytes.begin(),
                       [](unsigned char left, char right) {
                           return left == static_cast<unsigned char>(right);
                       }),
            "invalid PNG signature: " + utf8_text(path));
    DatasetPtr image(static_cast<GDALDataset*>(GDALOpenEx(
        utf8_text(path).c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY,
        nullptr, nullptr, nullptr)));
    require(image != nullptr, "GDAL could not decode PNG: " + utf8_text(path));
    require(image->GetRasterXSize() == width && image->GetRasterYSize() == height,
            "unexpected PNG dimensions: " + utf8_text(path));
    require(image->GetRasterCount() == 4,
            "visdem PNG must preserve Matplotlib-compatible RGBA bands");
}

void test_visdem(const fs::path& root) {
    const fs::path input = root / "dem_fixture.tif";
    const fs::path overview = root / "dem_fixture_visualization.png";
    const fs::path surface = root / "dem_fixture_3d_visualization.png";
    create_dem_fixture(input);
    const Snapshot input_before = snapshot(input);

    std::ostringstream out;
    std::ostringstream err;
    require(luw::cli::run_visdem({utf8_text(input)}, out, err) == 0,
            "visdem should render a valid DEM");
    require_equal(err.str(), "", "visdem stderr");
    const std::string expected =
        "Loading DEM file: " + utf8_text(input) + "\n"
        "\nOriginal DEM Metadata:\n"
        "  Shape: (5, 6)\n"
        "  CRS: EPSG:32630\n"
        "  Bounds: BoundingBox(left=500000.0, bottom=5699500.0, "
        "right=500600.0, top=5700000.0)\n"
        "  Resolution: (100.0, 100.0)\n"
        "  NoData value: -9999.0\n"
        "\nReprojecting to EPSG:4326 (WGS84)...\n"
        "\nReprojected DEM Metadata:\n"
        "  Shape: (4, 7)\n"
        "  CRS: EPSG:4326\n"
        "  Lat range: [51.446197, 51.451182]\n"
        "  Lon range: [-3.000000, -2.991275]\n"
        "\nElevation Statistics (after filtering -9999):\n"
        "  Min: 60.00 m\n"
        "  Max: 147.05 m\n"
        "  Mean: 103.34 m\n"
        "  Std: 22.46 m\n"
        "  Valid pixels: 27 / 28 (96.4%)\n"
        "\nVisualization saved to: " + utf8_text(overview) + "\n"
        "\nGenerating 3D surface plot...\n"
        "3D visualization saved to: " + utf8_text(surface) + "\n";
    require_equal(out.str(), expected, "Python oracle console output");
    require_png(overview, 4770, 3570);
    require_png(surface, 2955, 2498);
    require_unchanged(input, input_before, "read-only DEM input");

    const Snapshot overview_before = snapshot(overview);
    const Snapshot surface_before = snapshot(surface);
    out.str({});
    out.clear();
    err.str({});
    err.clear();
    require(luw::cli::run_visdem({utf8_text(input), "ignored"}, out, err) == 1,
            "visdem must refuse to overwrite existing output images");
    require_equal(out.str(), "Loading DEM file: " + utf8_text(input) + "\n",
                  "refusal stdout");
    require_equal(err.str(),
                  "ERROR: visdem failed: Output image already exists; refusing "
                  "to overwrite: " + utf8_text(overview) + "\n",
                  "refusal stderr");
    require_unchanged(overview, overview_before, "overview refusal");
    require_unchanged(surface, surface_before, "surface refusal");
}

void test_case_sensitive_legacy_naming(const fs::path& root) {
    const fs::path input = root / "uppercase.TIF";
    create_dem_fixture(input);
    const Snapshot before = snapshot(input);
    std::ostringstream out;
    std::ostringstream err;
    require(luw::cli::run_visdem({utf8_text(input)}, out, err) == 1,
            "uppercase .TIF must not overwrite the input after legacy replacement");
    require_equal(out.str(), "Loading DEM file: " + utf8_text(input) + "\n",
                  "uppercase naming stdout");
    require_equal(err.str(),
                  "ERROR: visdem failed: Output image already exists; refusing "
                  "to overwrite: " + utf8_text(input) + "\n",
                  "uppercase naming refusal");
    require_unchanged(input, before, "uppercase input safety");
}

#else

void test_unavailable_build() {
    std::ostringstream out;
    std::ostringstream err;
    require(luw::cli::run_visdem({"fixture.tif", "ignored"}, out, err) == 1,
            "visdem without GDAL must fail clearly");
    require_equal(out.str(), "Loading DEM file: fixture.tif\n",
                  "unavailable stdout");
    require_equal(err.str(),
                  "ERROR: Native visdem requires GDAL raster/reprojection "
                  "support; rebuild with LUW_ENABLE_GEOSPATIAL=ON and a "
                  "discoverable GDAL installation.\n",
                  "unavailable stderr");

    out.str({});
    out.clear();
    err.str({});
    err.clear();
    require(luw::cli::run_visdem({}, out, err) == 1,
            "visdem default path must remain available without argument parsing");
    require_equal(out.str(),
                  "Loading DEM file: "
                  "AST14DEM_00408232025023230_20251011022044.tif\n",
                  "legacy default filename");
}

#endif

}  // namespace

int main() {
    try {
#if LUW_HAS_GDAL
        GDALAllRegister();
        const auto nonce = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
#ifdef LUW_VISDEM_TEST_WORK_ROOT
        const fs::path base = LUW_VISDEM_TEST_WORK_ROOT;
#else
        const fs::path base = fs::temp_directory_path() / "luw-visdem-retained";
#endif
        const fs::path root = base / ("run-" + std::to_string(nonce));
        fs::create_directories(root);
        test_visdem(root);
        test_case_sensitive_legacy_naming(root);
        std::cout << "Generated fixtures retained at: " << utf8_text(root) << '\n';
#else
        test_unavailable_build();
#endif
        std::cout << "Native visdem compatibility tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Native visdem compatibility test failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
