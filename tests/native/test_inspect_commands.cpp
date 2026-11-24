#include "luw/inspect_commands.hpp"

#include <vtk_netcdf.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#endif

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

void require_contains(const std::string& text,
                      std::string_view expected,
                      const std::string& context) {
    require(text.find(expected) != std::string::npos,
            context + ": expected output to contain '" + std::string(expected) +
                "'\nActual output:\n" + text);
}

void check_nc(int status, const std::string& operation) {
    if (status != NC_NOERR) {
        throw std::runtime_error(operation + ": " + nc_strerror(status));
    }
}

void create_netcdf_fixture(const fs::path& path) {
    int ncid = -1;
    check_nc(nc_create(utf8_text(path).c_str(), NC_CLOBBER, &ncid),
             "nc_create");
    try {
        int x_dimension = -1;
        int time_dimension = -1;
        int x_variable = -1;
        int time_variable = -1;
        int temperature_variable = -1;
        check_nc(nc_def_dim(ncid, "x", 4U, &x_dimension), "nc_def_dim");
        check_nc(nc_def_dim(ncid, "time", 1U, &time_dimension),
                 "nc_def_dim time");
        check_nc(nc_def_var(ncid, "x", NC_DOUBLE, 1, &x_dimension, &x_variable),
                 "nc_def_var x");
        check_nc(nc_def_var(ncid, "time", NC_INT, 1, &time_dimension,
                            &time_variable),
                 "nc_def_var time");
        check_nc(nc_def_var(ncid, "temperature", NC_FLOAT, 1, &x_dimension,
                            &temperature_variable),
                 "nc_def_var temperature");

        constexpr float fill = -9999.0F;
        check_nc(nc_put_att_float(ncid, temperature_variable, "_FillValue", NC_FLOAT,
                                  1U, &fill),
                 "nc_put_att _FillValue");
        constexpr char long_name[] = "air temperature";
        check_nc(nc_put_att_text(ncid, temperature_variable, "long_name",
                                 sizeof(long_name) - 1U, long_name),
                 "nc_put_att long_name");
        constexpr char units[] = "K";
        check_nc(nc_put_att_text(ncid, temperature_variable, "units",
                                 sizeof(units) - 1U, units),
                 "nc_put_att units");
        constexpr char time_units[] = "hours since 1996-01-01 12:00:00";
        check_nc(nc_put_att_text(ncid, time_variable, "units",
                                 sizeof(time_units) - 1U, time_units),
                 "nc_put_att time units");
        constexpr char source[] = "native compatibility fixture";
        check_nc(nc_put_att_text(ncid, NC_GLOBAL, "source", sizeof(source) - 1U,
                                 source),
                 "nc_put_att source");

        check_nc(nc_enddef(ncid), "nc_enddef");
        const double x_values[] = {0.0, 1.0, 2.0, 3.0};
        const int time_values[] = {0};
        const float temperature_values[] = {1.0F, 2.0F, 3.0F, fill};
        check_nc(nc_put_var_double(ncid, x_variable, x_values), "nc_put_var x");
        check_nc(nc_put_var_int(ncid, time_variable, time_values),
                 "nc_put_var time");
        check_nc(nc_put_var_float(ncid, temperature_variable,
                                  temperature_values),
                 "nc_put_var temperature");
        check_nc(nc_close(ncid), "nc_close");
        ncid = -1;
    } catch (...) {
        if (ncid >= 0) {
            (void)nc_close(ncid);
        }
        throw;
    }
}

void test_cdfinspect(const fs::path& root) {
    const fs::path project = root / "cdf_project";
    const fs::path wind = project / "wind_bc";
    fs::create_directories(wind);
    const fs::path deck = project / "conf.luw";
    std::ofstream(deck) << "[metadata]\ncasename = inspect_test\n";
    create_netcdf_fixture(wind / "b_input.nc");
    create_netcdf_fixture(wind / "a_input.nc");

    std::ostringstream out;
    std::ostringstream err;
    const int status =
        luw::cli::run_cdfinspect({utf8_text(deck)}, out, err);
    require(status == 0, "cdfinspect should accept a project deck");
    require_contains(err.str(), "Multiple .nc files found under 'wind_bc'",
                     "cdfinspect warning");
    require_contains(out.str(), "All detected NetCDF files:",
                     "cdfinspect candidate list");
    require_contains(out.str(),
                     "Using NetCDF: " + utf8_text(wind / "a_input.nc"),
                     "cdfinspect selected file");
    require_contains(out.str(), "===== Dataset Overview =====",
                     "cdfinspect overview");
    require_contains(out.str(), "Dimensions:  (x: 4, time: 1)",
                     "cdfinspect dimensions");
    require_contains(out.str(),
                     "time: min = 820497600000000000, q05 = 820497600000000000, "
                     "q95 = 820497600000000000, max = 820497600000000000",
                     "cdfinspect CF-time decoding");
    require_contains(out.str(),
                     "temperature: min = 1.0, q05 = 1.1, q95 = 2.9, max = 3.0",
                     "cdfinspect ranges");
    require_contains(out.str(), "Variable: temperature",
                     "cdfinspect variable details");
    require_contains(out.str(), "Attribute long_name: air temperature",
                     "cdfinspect attributes");
    require_contains(out.str(), "source: native compatibility fixture",
                     "cdfinspect global attributes");

    out.str({});
    out.clear();
    err.str({});
    err.clear();
    require(luw::cli::run_cdfinspect({}, out, err) == 1,
            "cdfinspect should reject a missing argument");
    require_contains(out.str(), "Usage: python cdfInspect.py <input_file_path>",
                     "cdfinspect usage");

    out.str({});
    out.clear();
    const fs::path missing = root / "missing.nc";
    require(luw::cli::run_cdfinspect({utf8_text(missing)}, out, err) == 1,
            "cdfinspect should reject a missing direct NetCDF path");
    require_contains(out.str(), "Error: Input .nc does not exist:",
                     "cdfinspect missing file");
}

#if LUW_HAS_GDAL

void create_shapefile_fixture(const fs::path& path) {
    GDALAllRegister();
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    require(driver != nullptr, "ESRI Shapefile GDAL driver is unavailable");
    GDALDataset* dataset =
        driver->Create(utf8_text(path).c_str(), 0, 0, 0, GDT_Unknown, nullptr);
    require(dataset != nullptr, "could not create shapefile fixture");
    try {
        OGRSpatialReference reference;
        require(reference.importFromEPSG(4326) == OGRERR_NONE,
                "could not create EPSG:4326 reference");
        reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        OGRLayer* layer = dataset->CreateLayer("a_buildings", &reference, wkbPoint,
                                               nullptr);
        require(layer != nullptr, "could not create shapefile layer");
        OGRFieldDefn height("height", OFTReal);
        require(layer->CreateField(&height) == OGRERR_NONE,
                "could not create height field");

        const double coordinates[][2] = {{10.0, 30.0}, {20.0, 40.0}};
        const double heights[] = {12.5, 25.0};
        for (int index = 0; index < 2; ++index) {
            OGRFeature feature(layer->GetLayerDefn());
            feature.SetField("height", heights[index]);
            OGRPoint point(coordinates[index][0], coordinates[index][1]);
            require(feature.SetGeometry(&point) == OGRERR_NONE,
                    "could not set fixture geometry");
            require(layer->CreateFeature(&feature) == OGRERR_NONE,
                    "could not write fixture feature");
        }
        GDALClose(dataset);
        dataset = nullptr;
    } catch (...) {
        if (dataset != nullptr) {
            GDALClose(dataset);
        }
        throw;
    }
}

void test_shpinspect(const fs::path& root) {
    const fs::path project = root / "shp_project";
    const fs::path buildings = project / "building_db";
    fs::create_directories(buildings);
    const fs::path deck = project / "conf.luw";
    std::ofstream(deck) << "[metadata]\ncasename = inspect_test\n";
    create_shapefile_fixture(buildings / "a_buildings.shp");

    std::ostringstream out;
    std::ostringstream err;
    const int status =
        luw::cli::run_shpinspect({utf8_text(deck)}, out, err);
    require(status == 0, "shpinspect should inspect a generated shapefile");
    require_contains(out.str(), "Using shapefile:", "shpinspect selection");
    require_contains(out.str(), "CRS full details", "shpinspect CRS");
    require_contains(out.str(), "Longitude (X): 10.000000 to 20.000000",
                     "shpinspect longitude bounds");
    require_contains(out.str(), "Latitude  (Y): 30.000000 to 40.000000",
                     "shpinspect latitude bounds");
    require_contains(out.str(), "['height', 'geometry']",
                     "shpinspect columns");
    require_contains(out.str(), "First 5 rows:", "shpinspect preview");
}

#else

void test_shpinspect(const fs::path& root) {
    const fs::path placeholder = root / "placeholder.shp";
    std::ofstream(placeholder) << "not a shapefile";
    std::ostringstream out;
    std::ostringstream err;
    const int status =
        luw::cli::run_shpinspect({utf8_text(placeholder)}, out, err);
    require(status == 1, "shpinspect without GDAL should fail clearly");
    require_contains(out.str(), "GDAL/OGR support is unavailable in this build",
                     "shpinspect unavailable message");
}

#endif

void test_shpinspect_usage() {
    std::ostringstream out;
    std::ostringstream err;
    require(luw::cli::run_shpinspect({}, out, err) == 1,
            "shpinspect should reject a missing argument");
    require_contains(out.str(), "Usage: python shpInspect.py <input_file_path>",
                     "shpinspect usage");
    require(err.str().empty(), "shpinspect usage wrote an unexpected error");
}

}  // namespace

int main() {
    try {
        const auto nonce = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        const fs::path root = fs::temp_directory_path() /
                              ("luw_native_inspect_" + std::to_string(nonce));
        fs::create_directories(root);
        test_cdfinspect(root);
        test_shpinspect(root);
        test_shpinspect_usage();
        std::cout << "Native inspect compatibility tests passed.\n";
        std::cout << "Generated fixtures retained at: " << utf8_text(root) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Native inspect compatibility test failed: " << exception.what()
                  << '\n';
        return 1;
    }
}
