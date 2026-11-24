#include "luw/buildingscale_command.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
#include <gdal_priv.h>
#include <gdal_version.h>
#include <ogrsf_frmts.h>
#endif

namespace {

namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double tolerance,
                 const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected)
                                 + ", actual " + std::to_string(actual));
    }
}

std::string utf8(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

void verifyDistribution() {
    const std::vector<double> values{1.0, 2.0, 3.0, 5.0, 8.0, 13.0};
    const std::vector<double> weights{1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
    const auto result = luw::cli::calculateBuildingScaleDistribution(values, weights);
    const std::array<double, 4> expectedEdges{1.0, 5.0, 9.0, 13.0};
    const std::array<double, 3> expectedCenters{3.0, 7.0, 11.0};
    const std::array<double, 3> expectedPdf{
        7.0 / 252.0, 24.0 / 252.0, 32.0 / 252.0};
    const std::array<double, 6> expectedDescending{13.0, 8.0, 5.0, 3.0, 2.0, 1.0};
    const std::array<double, 6> expectedCdf{
        32.0 / 63.0, 48.0 / 63.0, 56.0 / 63.0,
        60.0 / 63.0, 62.0 / 63.0, 1.0};
    require(result.binEdges.size() == expectedEdges.size(), "FD edge count");
    require(result.binCenters.size() == expectedCenters.size(), "FD center count");
    for (std::size_t index = 0; index < expectedEdges.size(); ++index) {
        requireNear(result.binEdges[index], expectedEdges[index], 1.0e-14,
                    "NumPy FD edge");
    }
    for (std::size_t index = 0; index < expectedCenters.size(); ++index) {
        requireNear(result.binCenters[index], expectedCenters[index], 1.0e-14,
                    "NumPy FD center");
        requireNear(result.probabilityDensity[index], expectedPdf[index], 1.0e-14,
                    "area-height weighted PDF");
    }
    for (std::size_t index = 0; index < expectedDescending.size(); ++index) {
        requireNear(result.valuesDescending[index], expectedDescending[index], 0.0,
                    "descending CDF x");
        requireNear(result.cumulativeWeights[index], expectedCdf[index], 1.0e-14,
                    "descending weighted CDF");
    }

    const auto singleton = luw::cli::calculateBuildingScaleDistribution({4.0}, {7.0});
    require(singleton.binEdges == std::vector<double>({3.5, 4.5}),
            "NumPy constant-data outer edges");
    requireNear(singleton.probabilityDensity.front(), 1.0, 0.0,
                "singleton density");

    const auto tied = luw::cli::calculateBuildingScaleDistribution(
        {2.0, 1.0, 2.0, 1.0, 2.0}, {10.0, 20.0, 30.0, 40.0, 50.0});
    const std::array<double, 5> tiedCdf{
        50.0 / 150.0, 80.0 / 150.0, 90.0 / 150.0,
        130.0 / 150.0, 1.0};
    for (std::size_t index = 0; index < tiedCdf.size(); ++index) {
        requireNear(tied.cumulativeWeights[index], tiedCdf[index], 1.0e-14,
                    "np.argsort reverse tie order");
    }
}

void verifyCliSurface() {
    std::ostringstream output, error;
    require(luw::cli::run_buildingscale({"--help"}, output, error) == 0,
            "buildingscale help succeeds");
    require(output.str().find("usage: buildingscale [-h] [--out OUT] shp_path") == 0,
            "argparse-compatible usage");
    require(output.str().find("Plot empirical CDF of building short-side lengths")
                != std::string::npos,
            "help description");

    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({}, output, error) == 2,
            "missing positional argument returns argparse status");
    require(error.str().find("the following arguments are required: shp_path")
                != std::string::npos,
            "missing positional diagnostic");

    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({"input.shp", "--out"}, output, error) == 2,
            "missing --out value returns argparse status");
    require(error.str().find("argument --out: expected one argument") != std::string::npos,
            "missing --out diagnostic");

    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({"--o"}, output, error) == 2,
            "argparse long-option abbreviation");
    require(error.str().find("argument --out: expected one argument") != std::string::npos,
            "abbreviated --out diagnostic");

    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({"--he"}, output, error) == 0,
            "argparse abbreviated help");
}

#if LUW_HAS_GDAL

struct DatasetCloser final {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) GDALClose(dataset);
    }
};
using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;

OGRSpatialReference epsg(int code) {
    OGRSpatialReference reference;
    require(reference.importFromEPSG(code) == OGRERR_NONE, "create fixture CRS");
#if GDAL_VERSION_NUM >= 3000000
    reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif
    return reference;
}

OGRPolygon rectangle(double centerX, double centerY, double longSide,
                     double shortSide, double angleDegrees) {
    const double radians = angleDegrees * 3.14159265358979323846 / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    OGRLinearRing ring;
    for (const std::array<double, 2> local : std::array<std::array<double, 2>, 5>{{
             {-longSide / 2.0, -shortSide / 2.0},
             { longSide / 2.0, -shortSide / 2.0},
             { longSide / 2.0,  shortSide / 2.0},
             {-longSide / 2.0,  shortSide / 2.0},
             {-longSide / 2.0, -shortSide / 2.0}}}) {
        ring.addPoint(centerX + local[0] * cosine - local[1] * sine,
                      centerY + local[0] * sine + local[1] * cosine);
    }
    OGRPolygon polygon;
    require(polygon.addRing(&ring) == OGRERR_NONE, "add fixture rectangle ring");
    return polygon;
}

void addFeature(OGRLayer& layer, OGRGeometry* geometry,
                std::string_view invalidHeight, double substringHeight,
                double roofHeight) {
    OGRFeature feature(layer.GetLayerDefn());
    feature.SetField("HEIGHT", std::string(invalidHeight).c_str());
    feature.SetField("misc_hgt", substringHeight);
    feature.SetField("roof_h", roofHeight);
    require(feature.SetGeometry(geometry) == OGRERR_NONE, "set fixture geometry");
    require(layer.CreateFeature(&feature) == OGRERR_NONE, "write fixture feature");
}

fs::path writeProjectedFixture(const fs::path& path) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    require(driver != nullptr, "Shapefile driver");
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), 0, 0, 0, GDT_Unknown, nullptr));
    require(static_cast<bool>(dataset), "create projected fixture");
    OGRSpatialReference reference = epsg(3857);
    OGRLayer* layer = dataset->CreateLayer("buildings", &reference, wkbPolygon, nullptr);
    require(layer != nullptr, "create projected layer");
    OGRFieldDefn invalid("HEIGHT", OFTString);
    OGRFieldDefn substring("misc_hgt", OFTReal);
    OGRFieldDefn selected("roof_h", OFTReal);
    require(layer->CreateField(&invalid) == OGRERR_NONE, "create invalid exact field");
    require(layer->CreateField(&substring) == OGRERR_NONE, "create substring height field");
    require(layer->CreateField(&selected) == OGRERR_NONE, "create selected exact field");

    OGRPolygon first = rectangle(0.0, 0.0, 10.0, 2.0, 30.0);
    addFeature(*layer, &first, "not-a-number", 99.0, 2.0);
    OGRPolygon second = rectangle(30.0, 0.0, 8.0, 4.0, 0.0);
    addFeature(*layer, &second, "bad", 99.0, 3.0);
    OGRPolygon small = rectangle(60.0, 0.0, 20.0, 1.0, 0.0);
    OGRPolygon large = rectangle(90.0, 0.0, 7.0, 6.0, 0.0);
    OGRMultiPolygon multiple;
    require(multiple.addGeometry(&small) == OGRERR_NONE, "add small multipolygon part");
    require(multiple.addGeometry(&large) == OGRERR_NONE, "add large multipolygon part");
    addFeature(*layer, &multiple, "bad", 99.0, 4.0);
    OGRPolygon filtered = rectangle(120.0, 0.0, 9.0, 7.0, 0.0);
    addFeature(*layer, &filtered, "bad", 99.0,
               std::numeric_limits<double>::quiet_NaN());
    OGRFeature nullGeometry(layer->GetLayerDefn());
    nullGeometry.SetField("roof_h", 5.0);
    require(layer->CreateFeature(&nullGeometry) == OGRERR_NONE, "write null geometry feature");
    require(layer->SyncToDisk() == OGRERR_NONE, "sync projected fixture");
    return path;
}

fs::path writeSimpleFixture(const fs::path& path, OGRSpatialReference* reference,
                            double centerX, double centerY, double width, double height) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    DatasetPtr dataset(driver->Create(utf8(path).c_str(), 0, 0, 0, GDT_Unknown, nullptr));
    require(static_cast<bool>(dataset), "create simple fixture");
    OGRLayer* layer = dataset->CreateLayer("buildings", reference, wkbPolygon, nullptr);
    require(layer != nullptr, "create simple layer");
    OGRPolygon polygon = rectangle(centerX, centerY, width, height, 0.0);
    OGRFeature feature(layer->GetLayerDefn());
    require(feature.SetGeometry(&polygon) == OGRERR_NONE, "set simple geometry");
    require(layer->CreateFeature(&feature) == OGRERR_NONE, "write simple geometry");
    require(layer->SyncToDisk() == OGRERR_NONE, "sync simple fixture");
    return path;
}

std::pair<std::uint32_t, std::uint32_t> pngDimensions(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "open generated PNG");
    std::array<unsigned char, 24> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(input.gcount() == static_cast<std::streamsize>(bytes.size())
                && bytes[0] == 0x89 && bytes[1] == 'P' && bytes[2] == 'N' && bytes[3] == 'G',
            "PNG signature");
    const auto be32 = [&](std::size_t offset) {
        return (static_cast<std::uint32_t>(bytes[offset]) << 24U)
            | (static_cast<std::uint32_t>(bytes[offset + 1U]) << 16U)
            | (static_cast<std::uint32_t>(bytes[offset + 2U]) << 8U)
            | static_cast<std::uint32_t>(bytes[offset + 3U]);
    };
    return {be32(16), be32(20)};
}

std::pair<int, int> imageDimensions(const fs::path& path) {
    DatasetPtr image(static_cast<GDALDataset*>(GDALOpenEx(
        utf8(path).c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY,
        nullptr, nullptr, nullptr)));
    require(static_cast<bool>(image), "open generated image through GDAL");
    return {image->GetRasterXSize(), image->GetRasterYSize()};
}

void verifyGdalWorkflow(const fs::path& root) {
    GDALAllRegister();
    const fs::path projected = writeProjectedFixture(root / "projected.shp");
    std::ostringstream output, error;
    require(luw::cli::run_buildingscale({utf8(projected)}, output, error) == 0,
            "projected workflow: " + error.str());
    require(output.str().find("Using existing projected CRS for metric lengths.\n") == 0,
            "projected CRS message");
    require(output.str().find("Height field: roof_h\n") != std::string::npos,
            "exact height field beats earlier substring field");
    require(output.str().find("Count: 3\n") != std::string::npos,
            "invalid height/null geometry filtering");
    require(output.str().find("Min/Median/Max (m): 2.000 / 4.000 / 6.000\n")
                != std::string::npos,
            "rotated rectangle and largest MultiPolygon short sides");
    const fs::path defaultPng = root / "projected_short_side_cdf.png";
    require(pngDimensions(defaultPng) == std::pair<std::uint32_t, std::uint32_t>{3000, 900},
            "10x3 inch 300 dpi canvas");

    const auto originalSize = fs::file_size(defaultPng);
    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({utf8(projected)}, output, error) == 1,
            "existing output is rejected");
    require(error.str().find("refusing to overwrite") != std::string::npos,
            "explicit overwrite refusal");
    require(fs::file_size(defaultPng) == originalSize, "existing output remains unchanged");

    const fs::path explicitPng = root / "explicit-output.jpg";
    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({"--out", utf8(explicitPng), utf8(projected)},
                                         output, error) == 0,
            "explicit output workflow: " + error.str());
    require(output.str().find("Output saved to: " + utf8(explicitPng)) != std::string::npos,
            "explicit output path message");
    require(imageDimensions(explicitPng) == std::pair<int, int>{3000, 900},
            "explicit JPEG dimensions");

    const fs::path suffixless = root / "suffixless-output";
    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({utf8(projected), "--o", utf8(suffixless)},
                                         output, error) == 0,
            "suffixless output workflow: " + error.str());
    require(output.str().find("Output saved to: " + utf8(suffixless)) != std::string::npos,
            "legacy suffixless console path");
    require(pngDimensions(fs::path(utf8(suffixless) + ".png"))
                == std::pair<std::uint32_t, std::uint32_t>{3000, 900},
            "Matplotlib-compatible implicit PNG suffix");

    const fs::path missingCrs = writeSimpleFixture(root / "missing-crs.shp", nullptr,
                                                    0.0, 0.0, 10.0, 2.0);
    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({utf8(missingCrs)}, output, error) == 1,
            "missing CRS fails");
    require(error.str().find("Input data has no CRS. Please set a CRS before measuring lengths.")
                != std::string::npos,
            "missing CRS diagnostic");

    OGRSpatialReference geographicReference = epsg(4326);
    const fs::path geographic = writeSimpleFixture(root / "geographic.shp", &geographicReference,
                                                   120.0, 30.0, 0.001, 0.0005);
    const fs::path geographicPng = root / "geographic-result.png";
    output.str({}); output.clear(); error.str({}); error.clear();
    require(luw::cli::run_buildingscale({utf8(geographic), "--out", utf8(geographicPng)},
                                         output, error) == 0,
            "geographic workflow: " + error.str());
    require(output.str().find("Reprojected from geographic CRS to EPSG:32651 for metric lengths.\n")
                == 0,
            "union centroid UTM selection");
    require(output.str().find("Height field: not found, using 1.0 for all features.\n")
                != std::string::npos,
            "height fallback");
}

#endif // LUW_HAS_GDAL

} // namespace

int main() {
    try {
        verifyDistribution();
        verifyCliSurface();
#if LUW_HAS_GDAL
        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const fs::path root = fs::path(LUW_BUILDINGSCALE_ARTIFACT_ROOT)
            / ("run_" + std::to_string(nonce));
        fs::create_directories(root);
        verifyGdalWorkflow(root);
        std::cout << "buildingscale native tests passed; retained artifacts: " << utf8(root) << '\n';
#else
        std::ostringstream output, error;
        require(luw::cli::run_buildingscale({"input.shp"}, output, error) == 1,
                "no-GDAL command gate");
        require(error.str().find("requires GDAL/OGR with GEOS support") != std::string::npos,
                "no-GDAL command diagnostic");
        std::cout << "buildingscale numeric/CLI tests passed; GDAL workflow gated\n";
#endif
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "buildingscale test failure: " << exception.what() << '\n';
        return 1;
    }
}
