#include "luw/luwbc_command.hpp"
#include "luw/luwbc_opencl.hpp"

#include <vtk_netcdf.h>

#include <QCoreApplication>

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

#if LUW_HAS_GDAL
struct GdalDatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) GDALClose(dataset);
    }
};

using GdalDatasetPtr = std::unique_ptr<GDALDataset, GdalDatasetCloser>;
#endif

void ncCheck(int status, const char* operation) {
    if (status != NC_NOERR) throw std::runtime_error(std::string(operation) + ": " + nc_strerror(status));
}

std::string pathUtf8(const fs::path& path) {
#ifdef _WIN32
    return QString::fromStdWString(path.wstring()).toUtf8().toStdString();
#else
    return path.string();
#endif
}

void writeText(const fs::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot write fixture");
    stream << text;
}

std::string readText(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void putVariable(int file, int variable, const std::vector<float>& values) {
    ncCheck(nc_put_var_float(file, variable, values.data()), "write float variable");
}

void createGeneric(const fs::path& path) {
    int file = -1;
    ncCheck(nc_create(pathUtf8(path).c_str(), NC_CLOBBER, &file), "create generic NetCDF");
    int levDim = -1, latDim = -1, lonDim = -1;
    ncCheck(nc_def_dim(file, "lev", 3, &levDim), "define lev");
    ncCheck(nc_def_dim(file, "lat", 3, &latDim), "define lat");
    ncCheck(nc_def_dim(file, "lon", 3, &lonDim), "define lon");
    int lev = -1, lat = -1, lon = -1;
    ncCheck(nc_def_var(file, "lev", NC_FLOAT, 1, &levDim, &lev), "define lev variable");
    ncCheck(nc_put_att_text(file, lev, "units", 1, "m"), "write lev units");
    ncCheck(nc_def_var(file, "lat", NC_FLOAT, 1, &latDim, &lat), "define lat variable");
    ncCheck(nc_def_var(file, "lon", NC_FLOAT, 1, &lonDim, &lon), "define lon variable");
    const int dimensions[3]{levDim, latDim, lonDim};
    int u = -1, v = -1, w = -1, tc = -1;
    ncCheck(nc_def_var(file, "u", NC_FLOAT, 3, dimensions, &u), "define u");
    ncCheck(nc_def_var(file, "v", NC_FLOAT, 3, dimensions, &v), "define v");
    ncCheck(nc_def_var(file, "w", NC_FLOAT, 3, dimensions, &w), "define w");
    ncCheck(nc_def_var(file, "TC", NC_FLOAT, 3, dimensions, &tc), "define TC");
    ncCheck(nc_enddef(file), "end generic definition");
    putVariable(file, lev, {0.0F, 50.0F, 100.0F});
    putVariable(file, lat, {31.000F, 31.001F, 31.002F});
    putVariable(file, lon, {121.000F, 121.001F, 121.002F});
    putVariable(file, u, std::vector<float>(27, 4.0F));
    putVariable(file, v, std::vector<float>(27, -2.0F));
    putVariable(file, w, std::vector<float>(27, 1.25F));
    putVariable(file, tc, std::vector<float>(27, 20.0F));
    ncCheck(nc_close(file), "close generic NetCDF");
}

void createWrf(const fs::path& path) {
    int file = -1;
    ncCheck(nc_create(pathUtf8(path).c_str(), NC_CLOBBER, &file), "create WRF NetCDF");
    int time = -1, bottom = -1, bottomStag = -1, south = -1, southStag = -1;
    int west = -1, westStag = -1;
    ncCheck(nc_def_dim(file, "Time", 1, &time), "define Time");
    ncCheck(nc_def_dim(file, "bottom_top", 2, &bottom), "define bottom_top");
    ncCheck(nc_def_dim(file, "bottom_top_stag", 3, &bottomStag), "define bottom_top_stag");
    ncCheck(nc_def_dim(file, "south_north", 2, &south), "define south_north");
    ncCheck(nc_def_dim(file, "south_north_stag", 3, &southStag), "define south_north_stag");
    ncCheck(nc_def_dim(file, "west_east", 2, &west), "define west_east");
    ncCheck(nc_def_dim(file, "west_east_stag", 3, &westStag), "define west_east_stag");
    const int mass2d[3]{time, south, west};
    const int uDims[4]{time, bottom, south, westStag};
    const int vDims[4]{time, bottom, southStag, west};
    const int wDims[4]{time, bottomStag, south, west};
    const int mass3d[4]{time, bottom, south, west};
    const int geopotentialDims[4]{time, bottomStag, south, west};
    int xlong = -1, xlat = -1, u = -1, v = -1, w = -1, t = -1;
    int ph = -1, phb = -1, hgt = -1;
    ncCheck(nc_def_var(file, "XLONG", NC_FLOAT, 3, mass2d, &xlong), "define XLONG");
    ncCheck(nc_def_var(file, "XLAT", NC_FLOAT, 3, mass2d, &xlat), "define XLAT");
    ncCheck(nc_def_var(file, "U", NC_FLOAT, 4, uDims, &u), "define U");
    ncCheck(nc_def_var(file, "V", NC_FLOAT, 4, vDims, &v), "define V");
    ncCheck(nc_def_var(file, "W", NC_FLOAT, 4, wDims, &w), "define W");
    ncCheck(nc_def_var(file, "T", NC_FLOAT, 4, mass3d, &t), "define T");
    ncCheck(nc_def_var(file, "PH", NC_FLOAT, 4, geopotentialDims, &ph), "define PH");
    ncCheck(nc_def_var(file, "PHB", NC_FLOAT, 4, geopotentialDims, &phb), "define PHB");
    ncCheck(nc_def_var(file, "HGT", NC_FLOAT, 3, mass2d, &hgt), "define HGT");
    ncCheck(nc_enddef(file), "end WRF definition");
    putVariable(file, xlong, {121.000F, 121.002F, 121.000F, 121.002F});
    putVariable(file, xlat, {31.000F, 31.000F, 31.002F, 31.002F});
    std::vector<float> uValues;
    for (int level = 0; level < 2; ++level) {
        for (int row = 0; row < 2; ++row) {
            for (float value : {2.0F, 4.0F, 6.0F}) uValues.push_back(value);
        }
    }
    putVariable(file, u, uValues);
    std::vector<float> vValues;
    for (int level = 0; level < 2; ++level) {
        for (float value : {-6.0F, -2.0F, 2.0F}) {
            for (int column = 0; column < 2; ++column) vValues.push_back(value);
        }
    }
    putVariable(file, v, vValues);
    std::vector<float> wValues;
    for (float value : {1.0F, 3.0F, 5.0F}) for (int point = 0; point < 4; ++point) wValues.push_back(value);
    putVariable(file, w, wValues);
    putVariable(file, t, std::vector<float>(8, 280.0F));
    putVariable(file, ph, std::vector<float>(12, 0.0F));
    std::vector<float> phbValues;
    for (float height : {0.0F, 100.0F, 200.0F}) {
        for (int point = 0; point < 4; ++point) phbValues.push_back(height * 9.81F);
    }
    putVariable(file, phb, phbValues);
    putVariable(file, hgt, std::vector<float>(4, 0.0F));
    ncCheck(nc_close(file), "close WRF NetCDF");
}

#if LUW_HAS_GDAL
void createDemGeoTiff(const fs::path& path) {
    GDALAllRegister();
    fs::create_directories(path.parent_path());
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (driver == nullptr) throw std::runtime_error("GTiff driver unavailable");
    GdalDatasetPtr dataset(driver->Create(pathUtf8(path).c_str(), 6, 6, 1,
                                                GDT_Float64, nullptr));
    if (!dataset) throw std::runtime_error("cannot create luwbc DEM GeoTIFF fixture");
    double transform[6]{120.998, 0.001, 0.0, 31.004, 0.0, -0.001};
    if (dataset->SetGeoTransform(transform) != CE_None) {
        throw std::runtime_error("cannot set luwbc DEM fixture transform");
    }
    OGRSpatialReference wgs84;
    if (wgs84.importFromEPSG(4326) != OGRERR_NONE) {
        throw std::runtime_error("cannot construct luwbc DEM fixture CRS");
    }
    wgs84.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    if (dataset->SetSpatialRef(&wgs84) != CE_None) {
        throw std::runtime_error("cannot set luwbc DEM fixture CRS");
    }
    std::vector<double> values(36U, -9999.0);
    values[2U * 6U + 2U] = 100.0;
    values[2U * 6U + 3U] = 110.0;
    values[3U * 6U + 2U] = 120.0;
    values[3U * 6U + 3U] = 130.0;
    GDALRasterBand* band = dataset->GetRasterBand(1);
    if (band == nullptr || band->SetNoDataValue(-9999.0) != CE_None ||
        band->RasterIO(GF_Write, 0, 0, 6, 6, values.data(), 6, 6,
                       GDT_Float64, 0, 0, nullptr) != CE_None) {
        throw std::runtime_error("cannot populate luwbc DEM GeoTIFF fixture");
    }
}

std::size_t inspectDemCache(const fs::path& path) {
    GdalDatasetPtr dataset(static_cast<GDALDataset*>(GDALOpenEx(
        pathUtf8(path).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
        nullptr, nullptr, nullptr)));
    if (!dataset || dataset->GetLayer(0) == nullptr) {
        throw std::runtime_error("cannot inspect luwbc-generated DEM cache");
    }
    OGRLayer* layer = dataset->GetLayer(0);
    OGRFeatureDefn* definition = layer->GetLayerDefn();
    if (definition == nullptr || definition->GetFieldIndex("elevation") < 0 ||
        definition->GetFieldIndex("casename") < 0) {
        throw std::runtime_error("luwbc-generated DEM cache has incompatible schema");
    }
    return static_cast<std::size_t>(layer->GetFeatureCount(TRUE));
}
#endif

std::vector<std::string> split(const std::string& line, char delimiter) {
    std::vector<std::string> values;
    std::stringstream stream(line);
    std::string value;
    while (std::getline(stream, value, delimiter)) values.push_back(value);
    return values;
}

struct CsvSummary {
    std::vector<int> transitions;
    std::array<double, 4> firstValues{}; // u,v,w,T
    double firstZ = 0.0;
    double topW = 0.0;
};

CsvSummary inspectCsv(const fs::path& path, bool temperature) {
    std::ifstream stream(path);
    std::string line;
    std::getline(stream, line);
    const std::string expected = temperature ? "X,Y,Z,u,v,w,T,patch" : "X,Y,Z,u,v,w,patch";
    if (line != expected) throw std::runtime_error("unexpected CSV header: " + line);
    CsvSummary result;
    int previous = -1;
    bool first = true;
    while (std::getline(stream, line)) {
        const std::vector<std::string> fields = split(line, ',');
        const int patch = std::stoi(fields.back());
        if (patch != previous) { result.transitions.push_back(patch); previous = patch; }
        if (first) {
            result.firstZ = std::stod(fields[2]);
            result.firstValues[0] = std::stod(fields[3]);
            result.firstValues[1] = std::stod(fields[4]);
            result.firstValues[2] = std::stod(fields[5]);
            if (temperature) result.firstValues[3] = std::stod(fields[6]);
            first = false;
        }
        if (patch == 1) result.topW = std::stod(fields[5]);
    }
    return result;
}

void expect(bool condition, const std::string& message, int& failures) {
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

fs::path makeCaseRoot(const std::string& name) {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const fs::path root = fs::path(LUW_LUWBC_TEST_WORK_ROOT)
        / (name + "_" + std::to_string(stamp));
    fs::create_directories(root / "wind_bc");
    return root;
}

std::string testGeneric(int& failures, bool useGpu) {
    const fs::path root = makeCaseRoot(useGpu ? "generic_gpu" : "generic_cpu");
    const fs::path deck = root / "conf.luw";
    const fs::path nc = root / "wind_bc" / "tiny_20260102030405.nc";
    createGeneric(nc);
    writeText(deck,
              "casename = tiny\n"
              "datetime = 20260102030405\n"
              "base_height = 10\n"
              "midmesh_basesize = 100\n"
              "preprocess_use_gpu = " + std::string(useGpu ? "true\n" : "false\n") +
              "preprocess_gpu_device = 0\n");
    std::ostringstream output, error;
    const int code = luw::cli::run_luwbc({pathUtf8(deck), "--write-vtk"}, output, error);
    expect(code == 0, "generic command succeeds: " + error.str() + output.str(), failures);
    const fs::path latest = root / "proj_temp" / "SurfData_Latest.csv";
    const fs::path dated = root / "proj_temp" / "SurfData_20260102030405.csv";
    expect(fs::is_regular_file(latest), "generic latest CSV exists", failures);
    expect(fs::is_regular_file(root / "proj_temp" / "tiny_20260102030405.vti"),
           "optional VTI output exists", failures);
    expect(readText(latest) == readText(dated), "timestamped CSV is an exact copy", failures);
    if (fs::is_regular_file(latest)) {
        const CsvSummary csv = inspectCsv(latest, true);
        expect(csv.transitions == std::vector<int>({0, 1, 2, 3, 4, 5}),
               "generic patch ordering is bottom/top/south/north/west/east", failures);
        expect(std::abs(csv.firstZ - 10.1) < 1e-6, "bottom is sampled at ground + epsilon", failures);
        expect(std::abs(csv.firstValues[0] - 4.0) < 1e-6, "generic u preserved", failures);
        expect(std::abs(csv.firstValues[1] + 2.0) < 1e-6, "generic v preserved", failures);
        expect(std::abs(csv.firstValues[2] - 1.25) < 1e-6, "generic w preserved", failures);
        expect(std::abs(csv.firstValues[3] - 293.15) < 2e-4, "TC converted to Kelvin", failures);
    }
    const std::string rewritten = readText(deck);
    expect(rewritten.find("si_z_cfd = [0.000000, 110.000000]") != std::string::npos,
           "flat top is base height plus wind AGL", failures);
    expect(rewritten.find("downstream_bc = \"+x\"") != std::string::npos,
           "downstream face is written", failures);
    expect(output.str().find("buildBC_dev finished") != std::string::npos,
           "legacy completion prompt is preserved", failures);
    if (useGpu) {
        expect(output.str().find("OpenCL fused horizontal resampling backend:") !=
                   std::string::npos,
               "GPU luwbc horizontal backend was not used", failures);
        expect(output.str().find("OpenCL fused vertical resampling backend:") !=
                   std::string::npos,
               "GPU luwbc vertical backend was not used", failures);
        expect(output.str().find("CPU fallback") == std::string::npos,
               "GPU luwbc output reported a CPU fallback", failures);
        expect(output.str().find(
                   "OpenCL vertical transfer batch: 2 point domain(s)") !=
                   std::string::npos,
               "GPU luwbc did not batch uniform/native vertical point domains",
               failures);
    }
    std::cout << "Retained generic luwbc fixture: " << pathUtf8(root) << '\n';
    return fs::is_regular_file(latest) ? readText(latest) : std::string{};
}

void testMappingPlanCache(int& failures) {
    struct RunResult {
        int code = 1;
        std::string output;
        std::string error;
        std::string csv;
    };
    const auto writeDeck = [](const fs::path& path, int meshBase) {
        writeText(path,
                  "casename = tiny\n"
                  "datetime = 20260102030405\n"
                  "base_height = 10\n"
                  "midmesh_basesize = " + std::to_string(meshBase) + "\n"
                  "preprocess_use_gpu = false\n"
                  "preprocess_gpu_device = 0\n");
    };
    const auto run = [](const fs::path& deck) {
        std::ostringstream output, error;
        RunResult result;
        result.code = luw::cli::run_luwbc({pathUtf8(deck)}, output, error);
        result.output = output.str();
        result.error = error.str();
        const fs::path csv = deck.parent_path() / "proj_temp" / "SurfData_Latest.csv";
        if (fs::is_regular_file(csv)) result.csv = readText(csv);
        return result;
    };

    const fs::path firstRoot = makeCaseRoot("mapping_cache_same_deck");
    const fs::path firstDeck = firstRoot / "conf.luw";
    createGeneric(firstRoot / "wind_bc" / "tiny_20260102030405.nc");
    writeDeck(firstDeck, 100);
    const RunResult first = run(firstDeck);
    const RunResult second = run(firstDeck);
    expect(first.code == 0 && second.code == 0,
           "same-deck cache fixture failed: " + first.error + second.error,
           failures);
    expect(first.output.find(
               "PROJ wind-grid mapping plan cache: miss") != std::string::npos
               && first.output.find(
                   "Horizontal barycentric mapping plan cache: miss") !=
                   std::string::npos
               && first.output.find(
                   "Vertical axis mapping plan cache: miss") !=
                   std::string::npos,
           "initial mapping plans were not reported as cache misses", failures);
    expect(second.output.find(
               "PROJ wind-grid mapping plan cache: hit") != std::string::npos
               && second.output.find(
                   "Horizontal barycentric mapping plan cache: hit") !=
                   std::string::npos
               && second.output.find(
                   "Vertical axis mapping plan cache: hit") !=
                   std::string::npos,
           "same-deck mapping plans were not reused", failures);
    expect(!first.csv.empty() && first.csv == second.csv,
           "cache hit changed boundary CSV bytes", failures);
    expect(second.output.find("CPU fused horizontal interpolation batch: 4 fields")
               != std::string::npos
               && second.output.find(
                   "CPU fused vertical interpolation batch: 2 point domains, 4 fields each")
                   != std::string::npos,
           "CPU multi-field/domain batch scheduling was not reported", failures);

    writeDeck(firstDeck, 50);
    const RunResult changed = run(firstDeck);
    expect(changed.code == 0, "changed-grid cache fixture failed: " + changed.error,
           failures);
    expect(changed.output.find(
               "Horizontal barycentric mapping plan cache: miss") !=
               std::string::npos
               && changed.output.find(
                   "Vertical axis mapping plan cache: miss") !=
                   std::string::npos,
           "changed target grid/vertical levels did not invalidate mapping plans",
           failures);

    const fs::path otherRoot = makeCaseRoot("mapping_cache_other_deck");
    const fs::path otherDeck = otherRoot / "conf.luw";
    createGeneric(otherRoot / "wind_bc" / "tiny_20260102030405.nc");
    writeDeck(otherDeck, 50);
    const RunResult other = run(otherDeck);
    expect(other.code == 0, "cross-deck cache fixture failed: " + other.error,
           failures);
    expect(other.output.find(
               "PROJ wind-grid mapping plan cache: miss") != std::string::npos
               && other.output.find(
                   "Horizontal barycentric mapping plan cache: miss") !=
                   std::string::npos
               && other.output.find(
                   "Vertical axis mapping plan cache: miss") !=
                   std::string::npos,
           "mapping cache leaked entries across deck scopes", failures);
    expect(!changed.csv.empty() && changed.csv == other.csv,
           "invalidated and fresh mapping plans produced different CSV bytes",
           failures);
    std::cout << "Retained luwbc mapping-cache fixtures: "
              << pathUtf8(firstRoot) << " and " << pathUtf8(otherRoot) << '\n';
}

void testWrf(int& failures) {
    const fs::path root = makeCaseRoot("wrf");
    const fs::path deck = root / "conf.luw";
    const fs::path nc = root / "wind_bc" / "wrf_20260102030406.nc";
    createWrf(nc);
    writeText(deck,
              "casename = wrf\n"
              "datetime = 20260102030406\n"
              "base_height = 5\n"
              "midmesh_basesize = 100\n");
    std::ostringstream output, error;
    const int code = luw::cli::run_luwbc({pathUtf8(deck)}, output, error);
    expect(code == 0, "WRF command succeeds: " + error.str() + output.str(), failures);
    const fs::path latest = root / "proj_temp" / "SurfData_Latest.csv";
    expect(fs::is_regular_file(latest), "WRF latest CSV exists", failures);
    if (fs::is_regular_file(latest)) {
        const CsvSummary csv = inspectCsv(latest, true);
        expect(csv.transitions == std::vector<int>({0, 1, 2, 3, 4, 5}),
               "WRF patch ordering is preserved", failures);
        // The legacy Python path synthesized incompatible mass/stagger coordinates and
        // produced NaN XLONG/XLAT values here. Non-constant inputs ensure this test locks
        // the native positional adjacent-pair averaging, not merely the success path.
        expect(std::abs(csv.firstValues[0] - 3.0) < 1e-5,
               "WRF U uses positional adjacent-pair destaggering", failures);
        expect(std::abs(csv.firstValues[1] + 4.0) < 1e-5,
               "WRF V uses positional adjacent-pair destaggering", failures);
        expect(std::abs(csv.topW - 4.0) < 2e-4, "WRF W is vertically destaggered and exported", failures);
    }
    const std::string log = output.str();
    expect(log.find("Using WRF lon/lat coords: XLONG/XLAT") != std::string::npos,
           "WRF coordinate detection is reported", failures);
    expect(log.find("Destagger along lon_stag") != std::string::npos,
           "WRF U destagger is reported", failures);
    expect(log.find("Destagger along bottom_top_stag") != std::string::npos,
           "WRF W destagger is reported", failures);
    std::cout << "Retained WRF luwbc fixture: " << pathUtf8(root) << '\n';
}

void testGpuFailFast(int& failures) {
    const fs::path root = makeCaseRoot("gpu_failfast");
    const fs::path deck = root / "conf.luw";
    const fs::path nc = root / "wind_bc" / "tiny_20260102030405.nc";
    createGeneric(nc);
    writeText(deck,
              "casename = tiny\n"
              "datetime = 20260102030405\n"
              "base_height = 10\n"
              "midmesh_basesize = 100\n"
              "preprocess_use_gpu = true\n"
              "preprocess_gpu_device = 2147483647\n");
    std::ostringstream output, error;
    const int code = luw::cli::run_luwbc({pathUtf8(deck)}, output, error);
    expect(code == 1, "invalid explicit GPU device did not fail", failures);
    expect(error.str().find("2147483647") != std::string::npos &&
               error.str().find("available") != std::string::npos,
           "GPU fail-fast diagnostic omitted device/availability details", failures);
    expect(!fs::exists(root / "proj_temp"),
           "GPU preflight failure created preprocessing outputs", failures);
    std::cout << "Retained GPU fail-fast luwbc fixture: " << pathUtf8(root)
              << '\n';
}

#if LUW_HAS_GDAL
std::string testGeoTiffCachePath(int& failures, bool useGpu) {
    const fs::path root = makeCaseRoot(
        useGpu ? "geotiff_cache_gpu" : "geotiff_cache_cpu");
    const fs::path deck = root / "conf.luw";
    const fs::path nc = root / "wind_bc" / "demcase_20260102030407.nc";
    const fs::path raster = root / "terrain_db" / "unrestricted-dem-name.tiff";
    createGeneric(nc);
    createDemGeoTiff(raster);
    writeText(deck,
              "casename = demcase\n"
              "datetime = 20260102030407\n"
              "cut_lon_manual = [121.000, 121.002]\n"
              "cut_lat_manual = [31.000, 31.002]\n"
              "base_height = 10\n"
              "midmesh_basesize = 100\n"
              "preprocess_use_gpu = " + std::string(useGpu ? "true\n" : "false\n")
              + "preprocess_gpu_device = 0\n");
    std::ostringstream output, error;
    const int code = luw::cli::run_luwbc({pathUtf8(deck)}, output, error);
    expect(code == 0,
           "GeoTIFF-backed luwbc command succeeds: " + error.str() + output.str(),
           failures);
    const fs::path cache =
        fs::absolute(root / "terrain_db" / "demcase.shp").lexically_normal();
    expect(fs::is_regular_file(cache),
           "luwbc creates the legacy <casename>.shp cache", failures);
    if (fs::is_regular_file(cache)) {
        expect(inspectDemCache(cache) == 4U,
               "luwbc cache contains the 20%-expanded DEM cells", failures);
    }
    const std::string log = output.str();
    expect(log.find("Found DEM GeoTIFF(s) in terrain_db.") != std::string::npos,
           "luwbc reports terrain_db GeoTIFF discovery", failures);
    expect(log.find("Writing shapefile:") != std::string::npos,
           "luwbc reports the compatibility cache side effect", failures);
    expect(log.find("Loading DEM from: " + pathUtf8(cache)) != std::string::npos,
           "luwbc consumes the generated shapefile cache", failures);
    expect(log.find("Loaded 4 DEM points") != std::string::npos,
           "luwbc loads the expected generated DEM features", failures);
    if (useGpu) {
        expect(log.find("OpenCL DEM IDW interpolation backend:") !=
                   std::string::npos
                   && log.find(
                       "OpenCL terrain source-column DEM IDW backend:") !=
                       std::string::npos
                   && log.find(
                       "OpenCL terrain-first MLS weight backend:") !=
                       std::string::npos,
               "integrated terrain luwbc did not execute every GPU kernel",
               failures);
        expect(log.find("fallback to target-column sampling") == std::string::npos,
               "GPU terrain initialization used the CPU target-column fallback",
               failures);
    }
    std::cout << "Retained GeoTIFF-cache luwbc fixture: " << pathUtf8(root) << '\n';
    const fs::path latest = root / "proj_temp" / "SurfData_Latest.csv";
    const std::string firstCsv = fs::is_regular_file(latest)
        ? readText(latest) : std::string{};
    if (!useGpu) {
        createGeneric(root / "wind_bc" / "demcase_20260102030408.nc");
        writeText(deck,
                  "casename = demcase\n"
                  "datetime = 20260102030408\n"
                  "cut_lon_manual = [121.000, 121.002]\n"
                  "cut_lat_manual = [31.000, 31.002]\n"
                  "base_height = 10\n"
                  "midmesh_basesize = 100\n"
                  "preprocess_use_gpu = false\n"
                  "preprocess_gpu_device = 0\n");
        std::ostringstream cachedOutput, cachedError;
        const int cachedCode = luw::cli::run_luwbc(
            {pathUtf8(deck)}, cachedOutput, cachedError);
        expect(cachedCode == 0,
               "cached terrain luwbc rerun failed: " + cachedError.str(),
               failures);
        expect(cachedOutput.str().find(
                   "DEM IDW neighbor mapping plan cache: hit") !=
                   std::string::npos
                   && cachedOutput.str().find(
                       "Terrain-first MLS mapping plan cache: hit") !=
                       std::string::npos
                   && cachedOutput.str().find(
                       "PROJ wind-grid mapping plan cache: hit") !=
                       std::string::npos
                   && cachedOutput.str().find(
                       "Horizontal barycentric mapping plan cache: hit") !=
                       std::string::npos
                   && cachedOutput.str().find(
                       "Vertical axis mapping plan cache: hit") !=
                       std::string::npos,
               "DEM/MLS mapping plans were not reused on a sequential timestamp run",
               failures);
        expect(!firstCsv.empty() && fs::is_regular_file(latest)
                   && firstCsv == readText(latest),
               "DEM/MLS cache hit changed terrain boundary CSV bytes", failures);
    }
    return firstCsv;
}
#endif

} // namespace

int main(int argc, char* argv[]) {
    QCoreApplication application(argc, argv);
    int failures = 0;
    try {
        testGpuFailFast(failures);
        const std::string cpuGeneric = testGeneric(failures, false);
        testMappingPlanCache(failures);
#if LUW_HAS_GDAL
        const std::string cpuTerrain = testGeoTiffCachePath(failures, false);
#endif
        try {
            (void)luw::cli::requireNativeOpenClLuwbcBackend(0);
            const std::string gpuGeneric = testGeneric(failures, true);
            expect(!cpuGeneric.empty() && cpuGeneric == gpuGeneric,
                   "CPU/GPU luwbc generic CSV outputs differ", failures);
#if LUW_HAS_GDAL
            const std::string gpuTerrain = testGeoTiffCachePath(failures, true);
            expect(!cpuTerrain.empty() && cpuTerrain == gpuTerrain,
                   "CPU/GPU terrain luwbc CSV outputs differ", failures);
#endif
        } catch (const luw::cli::OpenClLuwbcError& exception) {
            const std::string message = exception.what();
            if (message.find("only 0 OpenCL GPU device(s) are available") !=
                    std::string::npos ||
                message.find("OpenCL support is not enabled") !=
                    std::string::npos) {
                std::cout << "Skip integrated GPU luwbc fixture: "
                          << message << '\n';
            } else {
                throw;
            }
        }
        testWrf(failures);
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: unexpected test exception: " << exception.what() << '\n';
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
