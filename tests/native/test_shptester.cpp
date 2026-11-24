#include "luw/shptester_command.hpp"

#ifndef LUW_HAS_SHPTESTER_GIS
#define LUW_HAS_SHPTESTER_GIS 0
#endif

#if LUW_HAS_SHPTESTER_GIS
#include "shptester_diagnostics.hpp"

#include <gdal_priv.h>
#include <ogr_geometry.h>
#include <ogrsf_frmts.h>
#endif

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
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string utf8_text(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(std::string_view actual,
                   std::string_view expected,
                   const std::string& context) {
    require(actual == expected,
            context + " mismatch.\nExpected:\n" + std::string(expected)
                + "\nActual:\n" + std::string(actual));
}

void require_contains(std::string_view actual,
                      std::string_view expected,
                      const std::string& context) {
    require(actual.find(expected) != std::string_view::npos,
            context + ": expected '" + std::string(expected) + "' in:\n"
                + std::string(actual));
}

void test_usage_contract() {
    constexpr std::string_view usage =
        "Usage: python check_degenerate_shp.py <path_to_shp>\n";
    for (const std::vector<std::string>& args :
         {std::vector<std::string>{}, std::vector<std::string>{"a", "b"}}) {
        std::ostringstream out;
        std::ostringstream err;
        require(luw::cli::run_shptester(args, out, err) == 2,
                "shptester must use legacy exit code 2 for an invalid argument count");
        require_equal(out.str(), "", "usage stdout");
        require_equal(err.str(), usage, "legacy usage stderr");
    }
}

#if LUW_HAS_SHPTESTER_GIS

struct DatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) {
            GDALClose(dataset);
        }
    }
};

struct GeometryCloser {
    void operator()(OGRGeometry* geometry) const noexcept {
        OGRGeometryFactory::destroyGeometry(geometry);
    }
};

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;
using GeometryPtr = std::unique_ptr<OGRGeometry, GeometryCloser>;

GeometryPtr geometry_from_wkt(const char* wkt) {
    OGRGeometry* raw = nullptr;
    require(OGRGeometryFactory::createFromWkt(wkt, nullptr, &raw) == OGRERR_NONE
                && raw != nullptr,
            std::string("could not parse fixture WKT: ") + wkt);
    return GeometryPtr(raw);
}

void require_diagnosis(const luw::cli::detail::ShpTesterGeometryDiagnoser& diagnoser,
                       const OGRGeometry* geometry,
                       bool expected_degenerate,
                       std::string_view expected_reason,
                       const std::string& context) {
    const luw::cli::detail::ShpTesterDiagnosis result =
        diagnoser.diagnose(geometry);
    require(result.degenerate == expected_degenerate,
            context + ": unexpected degenerate flag");
    require_equal(result.reason, expected_reason, context + " reason");
}

void test_geometry_classification() {
    luw::cli::detail::ShpTesterGeometryDiagnoser diagnoser;
    require_diagnosis(diagnoser, nullptr, true, "null", "null geometry");

    OGRPolygon empty_polygon;
    require_diagnosis(diagnoser, &empty_polygon, true, "empty", "empty polygon");
    OGRMultiPolygon empty_multi;
    require_diagnosis(diagnoser, &empty_multi, true, "empty", "empty multipolygon");

    OGRLinearRing short_ring;
    short_ring.addPoint(0.0, 0.0);
    short_ring.addPoint(1.0, 0.0);
    short_ring.addPoint(0.0, 0.0);
    OGRPolygon too_few;
    require(too_few.addRing(&short_ring) == OGRERR_NONE,
            "could not create too-few-points polygon");
    require_diagnosis(diagnoser, &too_few, true, "too_few_points",
                      "too-few-points polygon");

    OGRLinearRing open_ring;
    open_ring.addPoint(0.0, 0.0);
    open_ring.addPoint(2.0, 0.0);
    open_ring.addPoint(2.0, 2.0);
    open_ring.addPoint(0.0, 2.0);
    OGRPolygon not_closed;
    require(not_closed.addRing(&open_ring) == OGRERR_NONE,
            "could not create unclosed polygon");
    require_diagnosis(diagnoser, &not_closed, true, "ring_not_closed",
                      "unclosed polygon");

    auto zero_area = geometry_from_wkt("POLYGON ((0 0, 1 0, 2 0, 0 0))");
    require_diagnosis(diagnoser, zero_area.get(), true, "zero_area",
                      "zero-area polygon");

    auto invalid_hole = geometry_from_wkt(
        "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), "
        "(5 5, 6 5, 6 6, 5 6, 5 5))");
    require_diagnosis(diagnoser, invalid_hole.get(), true,
                      "invalid: Hole lies outside shell[5 5]",
                      "invalid polygon");

    auto valid = geometry_from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))");
    require_diagnosis(diagnoser, valid.get(), false, "ok", "valid polygon");

    OGRPoint point(1.0, 2.0);
    require_diagnosis(diagnoser, &point, true, "not_polygon_type: Point",
                      "non-polygon geometry");

    auto multipart_bad = geometry_from_wkt(
        "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), "
        "((4 0, 5 0, 6 0, 4 0)))");
    require_diagnosis(diagnoser, multipart_bad.get(), true,
                      "multipart: zero_area", "degenerate multipart geometry");

    auto multipart_overlap = geometry_from_wkt(
        "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), "
        "((1 1, 3 1, 3 3, 1 3, 1 1)))");
    require_diagnosis(diagnoser, multipart_overlap.get(), true,
                      "invalid: Self-intersection[1 2]",
                      "invalid multipart geometry");
}

GDALDriver* require_driver(const char* name) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName(name);
    require(driver != nullptr, std::string(name) + " GDAL driver is unavailable");
    return driver;
}

DatasetPtr create_vector_dataset(const char* driver_name, const fs::path& path) {
    GDALDriver* driver = require_driver(driver_name);
    return DatasetPtr(driver->Create(utf8_text(path).c_str(), 0, 0, 0,
                                     GDT_Unknown, nullptr));
}

void add_feature(OGRLayer& layer, int id, const OGRGeometry* geometry) {
    OGRFeature feature(layer.GetLayerDefn());
    feature.SetField("id", id);
    if (geometry != nullptr) {
        require(feature.SetGeometry(geometry) == OGRERR_NONE,
                "could not assign fixture geometry");
    }
    require(layer.CreateFeature(&feature) == OGRERR_NONE,
            "could not write fixture feature");
}

void add_id_field(OGRLayer& layer) {
    OGRFieldDefn id_field("id", OFTInteger);
    require(layer.CreateField(&id_field) == OGRERR_NONE,
            "could not create fixture id field");
}

fs::path create_polygon_shapefile(const fs::path& root) {
    const fs::path path = root / "polygon_cases.shp";
    DatasetPtr dataset = create_vector_dataset("ESRI Shapefile", path);
    require(dataset != nullptr, "could not create polygon shapefile fixture");
    OGRLayer* layer = dataset->CreateLayer("polygon_cases", nullptr, wkbPolygon,
                                           nullptr);
    require(layer != nullptr, "could not create polygon shapefile layer");
    add_id_field(*layer);

    auto valid = geometry_from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))");
    add_feature(*layer, 1, valid.get());
    add_feature(*layer, 2, nullptr);
    auto zero_area = geometry_from_wkt("POLYGON ((3 0, 4 0, 5 0, 3 0))");
    add_feature(*layer, 3, zero_area.get());
    auto valid_multi = geometry_from_wkt(
        "MULTIPOLYGON (((10 0, 11 0, 11 1, 10 1, 10 0)), "
        "((13 0, 14 0, 14 1, 13 1, 13 0)))");
    add_feature(*layer, 4, valid_multi.get());
    dataset.reset();
    return path;
}

fs::path create_point_shapefile(const fs::path& root) {
    const fs::path path = root / "point_cases.shp";
    DatasetPtr dataset = create_vector_dataset("ESRI Shapefile", path);
    require(dataset != nullptr, "could not create point shapefile fixture");
    OGRLayer* layer = dataset->CreateLayer("point_cases", nullptr, wkbPoint,
                                           nullptr);
    require(layer != nullptr, "could not create point shapefile layer");
    add_id_field(*layer);
    OGRPoint first(0.0, 0.0);
    OGRPoint second(1.0, 1.0);
    add_feature(*layer, 1, &first);
    add_feature(*layer, 2, &second);
    dataset.reset();
    return path;
}

fs::path create_mixed_geojson(const fs::path& root) {
    const fs::path path = root / "counter_order.geojson";
    DatasetPtr dataset = create_vector_dataset("GeoJSON", path);
    require(dataset != nullptr, "could not create mixed GeoJSON fixture");
    OGRLayer* layer = dataset->CreateLayer("counter_order", nullptr, wkbUnknown,
                                           nullptr);
    require(layer != nullptr, "could not create mixed GeoJSON layer");
    add_id_field(*layer);
    OGRPoint point(0.0, 0.0);
    add_feature(*layer, 1, &point);
    add_feature(*layer, 2, nullptr);
    OGRLineString line;
    line.addPoint(0.0, 0.0);
    line.addPoint(1.0, 1.0);
    add_feature(*layer, 3, &line);
    add_feature(*layer, 4, nullptr);
    dataset.reset();
    return path;
}

std::vector<char> read_bytes(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "could not snapshot fixture: " + utf8_text(path));
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

struct FileSnapshot {
    fs::path path;
    std::vector<char> bytes;
    fs::file_time_type write_time;
};

std::vector<FileSnapshot> snapshot_shapefile(const fs::path& shp_path) {
    std::vector<FileSnapshot> snapshots;
    for (const char* extension : {".shp", ".shx", ".dbf"}) {
        fs::path path = shp_path;
        path.replace_extension(extension);
        require(fs::is_regular_file(path), "missing shapefile sidecar: " + utf8_text(path));
        snapshots.push_back({path, read_bytes(path), fs::last_write_time(path)});
    }
    return snapshots;
}

void require_unchanged(const std::vector<FileSnapshot>& snapshots) {
    for (const FileSnapshot& snapshot : snapshots) {
        require(read_bytes(snapshot.path) == snapshot.bytes,
                "read-only shptester changed bytes in " + utf8_text(snapshot.path));
        require(fs::last_write_time(snapshot.path) == snapshot.write_time,
                "read-only shptester changed timestamp of " + utf8_text(snapshot.path));
    }
}

void test_file_diagnostics(const fs::path& root) {
    const fs::path polygons = create_polygon_shapefile(root);
    const std::vector<FileSnapshot> before = snapshot_shapefile(polygons);

    std::ostringstream out;
    std::ostringstream err;
    require(luw::cli::run_shptester({utf8_text(polygons)}, out, err) == 0,
            "shptester should inspect a polygon shapefile");
    require_equal(err.str(), "", "polygon fixture stderr");
    require_equal(out.str(),
                  "Total features: 4\n"
                  "Degenerate features: 2\n"
                  "1\tnull\n"
                  "1\tzero_area\n",
                  "polygon fixture output");
    require_unchanged(before);

    const fs::path points = create_point_shapefile(root);
    out.str({});
    out.clear();
    err.str({});
    err.clear();
    require(luw::cli::run_shptester({utf8_text(points)}, out, err) == 0,
            "shptester should classify a non-polygon shapefile");
    require_equal(err.str(), "", "point fixture stderr");
    require_equal(out.str(),
                  "Total features: 2\n"
                  "Degenerate features: 2\n"
                  "2\tnot_polygon_type: Point\n",
                  "point fixture output");

    const fs::path mixed = create_mixed_geojson(root);
    out.str({});
    out.clear();
    err.str({});
    err.clear();
    require(luw::cli::run_shptester({utf8_text(mixed)}, out, err) == 0,
            "shptester should inspect a mixed vector fixture");
    require_equal(err.str(), "", "mixed fixture stderr");
    require_equal(out.str(),
                  "Total features: 4\n"
                  "Degenerate features: 4\n"
                  "2\tnull\n"
                  "1\tnot_polygon_type: Point\n"
                  "1\tnot_polygon_type: LineString\n",
                  "Counter.most_common ordering");
}

void test_read_failure(const fs::path& root) {
    const fs::path missing = root / "does_not_exist.shp";
    std::ostringstream out;
    std::ostringstream err;
    require(luw::cli::run_shptester({utf8_text(missing)}, out, err) == 1,
            "missing input must use legacy exit code 1");
    require_equal(out.str(), "", "missing input stdout");
    require_contains(err.str(), "Failed to read shapefile: ",
                     "missing input stderr prefix");
    require_contains(err.str(), "No such file or directory",
                     "missing input GDAL reason");
}

#else

void test_unavailable_build() {
    std::ostringstream out;
    std::ostringstream err;
    require(luw::cli::run_shptester({"placeholder.shp"}, out, err) == 1,
            "shptester without GIS dependencies must fail clearly");
    require_equal(out.str(), "", "unavailable stdout");
    require_equal(err.str(),
                  "Failed to read shapefile: GDAL/OGR with GEOS support is "
                  "unavailable in this build.\n",
                  "unavailable stderr");
}

#endif

}  // namespace

int main() {
    try {
        test_usage_contract();
#if LUW_HAS_SHPTESTER_GIS
        GDALAllRegister();
        const auto nonce = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
#ifdef LUW_SHPTESTER_TEST_WORK_ROOT
        const fs::path base = LUW_SHPTESTER_TEST_WORK_ROOT;
#else
        const fs::path base = fs::temp_directory_path() / "luw-shptester-retained";
#endif
        const fs::path root = base / ("run-" + std::to_string(nonce));
        fs::create_directories(root);
        test_geometry_classification();
        test_file_diagnostics(root);
        test_read_failure(root);
        std::cout << "Generated fixtures retained at: " << utf8_text(root) << '\n';
#else
        test_unavailable_build();
#endif
        std::cout << "Native shptester compatibility tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Native shptester compatibility test failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
