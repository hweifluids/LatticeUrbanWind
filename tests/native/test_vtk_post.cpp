#include "luw/legacy_vtk.hpp"
#include "luw/vtk_post_commands.hpp"

#include <vtk_netcdf.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string utf8Text(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        std::ostringstream detail;
        detail << message << ": expected " << expected << ", actual " << actual;
        throw std::runtime_error(detail.str());
    }
}

void requireContains(const std::string& text, std::string_view expected, const std::string& message) {
    require(text.find(expected) != std::string::npos,
            message + ": missing '" + std::string(expected) + "' in:\n" + text);
}

void ncCheck(int status, const std::string& context) {
    if (status != NC_NOERR) throw std::runtime_error(context + ": " + nc_strerror(status));
}

struct NcReadFile final {
    explicit NcReadFile(const fs::path& path) {
        ncCheck(nc_open(utf8Text(path).c_str(), NC_NOWRITE, &id), "nc_open " + utf8Text(path));
    }
    ~NcReadFile() { if (id >= 0) nc_close(id); }
    int id = -1;
};

std::vector<float> readFloatVariable(int ncid, std::string_view name) {
    int variable = -1;
    ncCheck(nc_inq_varid(ncid, std::string(name).c_str(), &variable),
            "find " + std::string(name));
    int dimensions = 0;
    ncCheck(nc_inq_varndims(ncid, variable, &dimensions),
            "dimension count for " + std::string(name));
    std::vector<int> dimensionIds(static_cast<std::size_t>(dimensions));
    ncCheck(nc_inq_vardimid(ncid, variable, dimensionIds.data()),
            "dimensions for " + std::string(name));
    std::size_t values = 1;
    for (const int dimension : dimensionIds) {
        std::size_t length = 0;
        ncCheck(nc_inq_dimlen(ncid, dimension, &length),
                "dimension length for " + std::string(name));
        values *= length;
    }
    std::vector<float> result(values);
    ncCheck(nc_get_var_float(ncid, variable, result.data()),
            "read " + std::string(name));
    return result;
}

void requireSameFloatVariable(int left, int right, std::string_view name) {
    const std::vector<float> leftValues = readFloatVariable(left, name);
    const std::vector<float> rightValues = readFloatVariable(right, name);
    require(leftValues.size() == rightValues.size(),
            std::string(name) + " output value count is identical");
    for (std::size_t index = 0; index < leftValues.size(); ++index) {
        const bool bothNan = std::isnan(leftValues[index]) && std::isnan(rightValues[index]);
        if (!bothNan && leftValues[index] != rightValues[index]) {
            std::ostringstream detail;
            detail << name << " differs at index " << index << ": binary="
                   << leftValues[index] << ", ASCII=" << rightValues[index];
            throw std::runtime_error(detail.str());
        }
    }
}

std::vector<float> makeTupleField(std::size_t points, std::size_t components,
                                  float uBase, float vBase, float wBase) {
    std::vector<float> result(points * components);
    for (std::size_t point = 0; point < points; ++point) {
        result[point * components] = uBase + static_cast<float>(point);
        if (components > 1) result[point * components + 1] = vBase + static_cast<float>(point);
        if (components > 2) result[point * components + 2] = wBase + static_cast<float>(point);
        for (std::size_t component = 3; component < components; ++component) {
            result[point * components + component] = 400.0F + static_cast<float>(point + component);
        }
    }
    return result;
}

std::vector<float> makeScalarField(std::size_t points, float base) {
    std::vector<float> result(points);
    for (std::size_t point = 0; point < points; ++point) result[point] = base + static_cast<float>(point);
    return result;
}

void writeVtk2NcFixture(const fs::path& path, float mainBase) {
    constexpr std::array<int, 3> dimensions{2, 2, 4};
    constexpr std::size_t points = 16;
    const std::vector<float> density = makeTupleField(points, 4, 9000.0F, 9100.0F, 9200.0F);
    const std::vector<float> fallback = makeTupleField(points, 3, 5000.0F, 5100.0F, 5200.0F);
    const std::vector<float> data = makeTupleField(points, 4, mainBase, mainBase + 100.0F, mainBase + 200.0F);
    const std::array<luw::post::LegacyVtkFloatFieldView, 3> fields{{
        {"data_rho", 4, luw::post::LegacyVtkFieldKind::Scalars, density},
        {"fallback", 3, luw::post::LegacyVtkFieldKind::Vectors, fallback},
        {"data", 4, luw::post::LegacyVtkFieldKind::Scalars, data},
    }};
    luw::post::writeLegacyStructuredPointsFloat(
        path, dimensions, {500000.0, 4000000.0, 0.0}, {10.0, 10.0, 25.0}, fields,
        "vtk2nc generated compatibility fixture");
}

void writeAsciiValues(std::ofstream& output, std::span<const float> values) {
    output << std::setprecision(9);
    for (std::size_t index = 0; index < values.size(); ++index) {
        output << values[index];
        output << ((index + 1) % 7 == 0 ? '\n' : ' ');
    }
    if (values.size() % 7 != 0) output << '\n';
}

void writeAsciiVtk2NcFixture(const fs::path& path, float mainBase) {
    constexpr std::size_t points = 16;
    const std::vector<float> density = makeTupleField(points, 4, 9000.0F, 9100.0F, 9200.0F);
    const std::vector<float> fallback = makeTupleField(points, 3, 5000.0F, 5100.0F, 5200.0F);
    const std::vector<float> temperature = makeScalarField(points, -10.0F);
    const std::vector<float> data = makeTupleField(points, 4, mainBase, mainBase + 100.0F,
                                                   mainBase + 200.0F);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "create ASCII VTK fixture");
    output << "# vtk DataFile Version 3.0\n"
           << "vtk2nc ASCII retained compatibility fixture\n"
           << "ASCII\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS 2 2 4\n"
           << "ORIGIN 500000 4000000 0\n"
           << "ASPECT_RATIO 10 10 25\n\n"
           << "CELL_DATA 3\n"
           << "SCALARS cell_mask unsigned_char\n"
           << "LOOKUP_TABLE default\n"
           << "1 0 1\n"
           << "POINT_DATA 16\n"
           << "SCALARS data_rho float 4\n"
           << "LOOKUP_TABLE default\n";
    writeAsciiValues(output, density);
    output << "VECTORS fallback double\n";
    writeAsciiValues(output, fallback);
    output << "FIELD FieldData 2\n"
           << "temperature 1 16 short\n";
    writeAsciiValues(output, temperature);
    output << "data 4 16 float\n";
    writeAsciiValues(output, data);
    output.flush();
    require(static_cast<bool>(output), "finish ASCII VTK fixture");
}

void writeFallbackWindFixture(const fs::path& path) {
    constexpr std::array<int, 3> dimensions{2, 2, 4};
    constexpr std::size_t points = 16;
    const std::vector<float> density = makeTupleField(points, 4, 9000.0F, 9100.0F, 9200.0F);
    const std::vector<float> fallback = makeTupleField(points, 3, 5000.0F, 5100.0F, 5200.0F);
    const std::array<luw::post::LegacyVtkFloatFieldView, 2> fields{{
        {"data_rho", 4, luw::post::LegacyVtkFieldKind::Scalars, density},
        {"fallback", 3, luw::post::LegacyVtkFieldKind::Vectors, fallback},
    }};
    luw::post::writeLegacyStructuredPointsFloat(
        path, dimensions, {500000.0, 3000000.0, 0.0}, {10.0, 10.0, 25.0}, fields,
        "vtk2nc fallback compatibility fixture");
}

void verifySharedReader(const fs::path& path) {
    const auto vtk = luw::post::LegacyStructuredPoints::readHeader(path);
    require(vtk.dimensions == std::array<int, 3>{2, 2, 4}, "legacy reader dimensions");
    require(vtk.fields.size() == 3, "legacy reader preserves three fields");
    const auto* fallback = vtk.field("fallback");
    require(fallback != nullptr, "legacy reader discovers VECTORS");
    require(fallback->components == 3, "VECTORS has three components");
    const std::vector<float> plane = vtk.readPointComponentPlane(*fallback, 2, 1);
    require(plane.size() == 4, "component plane size");
    requireNear(plane.front(), 5108.0, 0.0, "component plane first value");
    requireNear(plane.back(), 5111.0, 0.0, "component plane last value");
}

void verifyAsciiSharedReader(const fs::path& path) {
    const auto vtk = luw::post::LegacyStructuredPoints::readHeader(path);
    require(vtk.encoding == luw::post::LegacyVtkEncoding::Ascii, "ASCII encoding is retained");
    require(vtk.dimensions == std::array<int, 3>{2, 2, 4}, "ASCII reader dimensions");
    require(vtk.pointCount == 16 && vtk.cellCount == 3, "ASCII point/cell counts");
    require(vtk.fields.size() == 5, "ASCII SCALARS/VECTORS/FIELD arrays are discovered");
    const auto* cellMask = vtk.field("cell_mask");
    require(cellMask != nullptr
                && cellMask->association == luw::post::LegacyVtkAssociation::CellData,
            "ASCII CELL_DATA association is retained");
    const auto* wind = vtk.field("data");
    require(wind != nullptr && wind->kind == luw::post::LegacyVtkFieldKind::Field,
            "ASCII FIELD data array is retained");
    require(wind->components == 4 && wind->asciiPlaneOffsets.size() == 4,
            "ASCII point field has a seek index for every z plane");
    luw::post::LegacyVtkPlaneReader reader(vtk);
    const std::vector<float> plane = reader.readPointComponentPlane(*wind, 2, 2);
    require(plane.size() == 4, "ASCII component plane size");
    requireNear(plane.front(), 308.0, 0.0, "ASCII component plane first value");
    requireNear(plane.back(), 311.0, 0.0, "ASCII component plane last value");

    const auto* fallback = vtk.field("fallback");
    require(fallback != nullptr && fallback->kind == luw::post::LegacyVtkFieldKind::Vectors,
            "ASCII VECTORS array is retained");
    const std::vector<float> vectorPlane = reader.readPointComponentPlane(*fallback, 3, 1);
    requireNear(vectorPlane.front(), 5112.0, 0.0, "ASCII double VECTORS value");

    const auto* density = vtk.field("data_rho");
    require(density != nullptr && density->kind == luw::post::LegacyVtkFieldKind::Scalars,
            "ASCII multi-component SCALARS array is retained");
    const std::vector<float> scalarPlane = reader.readPointComponentPlane(*density, 0, 3);
    requireNear(scalarPlane.front(), 403.0, 0.0, "ASCII multi-component SCALARS value");

    const auto* temperature = vtk.field("temperature");
    require(temperature != nullptr && temperature->kind == luw::post::LegacyVtkFieldKind::Field,
            "ASCII multi-array FIELD member is retained");
    const std::vector<float> fieldPlane = reader.readPointComponentPlane(*temperature, 1, 0);
    requireNear(fieldPlane.front(), -6.0, 0.0, "ASCII integer FIELD value");
}

void verifyDeflate3(int ncid, int variable) {
    int shuffle = 0;
    int deflate = 0;
    int level = 0;
    ncCheck(nc_inq_var_deflate(ncid, variable, &shuffle, &deflate, &level), "nc_inq_var_deflate");
    require(shuffle == 1 && deflate == 1 && level == 3, "NetCDF variable uses shuffle/zlib level 3");
}

void testVtk2Nc(const fs::path& root) {
    const fs::path project = root / "vtk2nc_case";
    const fs::path results = project / "RESULTS";
    fs::create_directories(results);
    const fs::path config = project / "conf.luw";
    std::ofstream(config)
        << "casename = tiny\n"
        << "datetime = 20260719T120000\n"
        << "si_x_cfd = [10, 20]\n"
        << "si_y_cfd = [30, 40]\n"
        << "cut_lon_manual = [120.0, 120.1]\n"
        << "cut_lat_manual = [30.0, 30.1]\n";

    const fs::path uvw = results / "uvw-tiny_20260719T120000.vtk";
    const fs::path uvwrho = results / "uvwrho-tiny_20260719T120000.vtk";
    writeVtk2NcFixture(uvw, 800.0F);
    writeVtk2NcFixture(uvwrho, 100.0F);
    verifySharedReader(uvwrho);

    const fs::path outputPath = project / "retained_output" / "tiny.nc";
    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_vtk2nc(
        {utf8Text(config), "--output", utf8Text(outputPath)}, output, error);
    require(status == 0, "vtk2nc returns success: " + error.str() + output.str());
    requireContains(output.str(), uvwrho.filename().string(), "uvwrho input wins over uvw");
    requireContains(output.str(), "[Info] Vertical layers before cut: 4, after cut: 2",
                    "pedestal layer message");
    requireContains(output.str(), "[OK] Saved NetCDF:", "success message");

    NcReadFile nc(outputPath);
    int dimH = -1;
    int dimY = -1;
    int dimX = -1;
    ncCheck(nc_inq_dimid(nc.id, "h", &dimH), "find h dimension");
    ncCheck(nc_inq_dimid(nc.id, "y", &dimY), "find y dimension");
    ncCheck(nc_inq_dimid(nc.id, "x", &dimX), "find x dimension");
    std::size_t length = 0;
    ncCheck(nc_inq_dimlen(nc.id, dimH, &length), "h dimension length");
    require(length == 2, "two z layers remain after 50 m cut");
    ncCheck(nc_inq_dimlen(nc.id, dimY, &length), "y dimension length");
    require(length == 2, "y dimension length");
    ncCheck(nc_inq_dimlen(nc.id, dimX, &length), "x dimension length");
    require(length == 2, "x dimension length");

    int varU = -1;
    int varX = -1;
    int varH = -1;
    int varLat = -1;
    int varLon = -1;
    ncCheck(nc_inq_varid(nc.id, "u", &varU), "find u");
    ncCheck(nc_inq_varid(nc.id, "x", &varX), "find x");
    ncCheck(nc_inq_varid(nc.id, "h", &varH), "find h");
    ncCheck(nc_inq_varid(nc.id, "lat", &varLat), "find lat");
    ncCheck(nc_inq_varid(nc.id, "lon", &varLon), "find lon");
    nc_type type = NC_NAT;
    ncCheck(nc_inq_vartype(nc.id, varU, &type), "u type");
    require(type == NC_FLOAT, "u is float32");
    verifyDeflate3(nc.id, varU);
    verifyDeflate3(nc.id, varX);

    std::array<float, 8> u{};
    std::array<float, 2> x{};
    std::array<float, 2> h{};
    std::array<float, 4> lat{};
    std::array<float, 4> lon{};
    ncCheck(nc_get_var_float(nc.id, varU, u.data()), "read u");
    ncCheck(nc_get_var_float(nc.id, varX, x.data()), "read x");
    ncCheck(nc_get_var_float(nc.id, varH, h.data()), "read h");
    ncCheck(nc_get_var_float(nc.id, varLat, lat.data()), "read lat");
    ncCheck(nc_get_var_float(nc.id, varLon, lon.data()), "read lon");
    requireNear(u.front(), 108.0, 0.0, "data array wins, first kept u");
    requireNear(u.back(), 115.0, 0.0, "data array wins, last kept u");
    requireNear(x[0], 500000.0, 0.0, "manual anchor preserves VTK x coordinate");
    requireNear(x[1], 500010.0, 0.0, "manual anchor preserves VTK x spacing");
    requireNear(h[0], 0.0, 0.0, "pedestal top becomes zero");
    requireNear(h[1], 25.0, 0.0, "height spacing retained");
    requireNear(lon[0], 120.0, 2.0e-5, "first longitude equals manual anchor");
    requireNear(lat[0], 30.0, 2.0e-5, "first latitude equals manual anchor");

    const fs::path asciiProject = root / "vtk2nc_ascii_case";
    const fs::path asciiResults = asciiProject / "RESULTS";
    fs::create_directories(asciiResults);
    const fs::path asciiConfig = asciiProject / "conf.luw";
    std::ofstream(asciiConfig)
        << "casename = tiny_ascii\n"
        << "datetime = 20260719T120000\n"
        << "si_x_cfd = [10, 20]\n"
        << "si_y_cfd = [30, 40]\n"
        << "cut_lon_manual = [120.0, 120.1]\n"
        << "cut_lat_manual = [30.0, 30.1]\n";
    const fs::path asciiVtk = asciiResults / "uvwrho-tiny_ascii_20260719T120000.vtk";
    writeAsciiVtk2NcFixture(asciiVtk, 100.0F);
    verifyAsciiSharedReader(asciiVtk);

    const fs::path asciiOutputPath = asciiProject / "retained_output" / "tiny_ascii.nc";
    output.str({});
    output.clear();
    error.str({});
    error.clear();
    require(luw::cli::run_vtk2nc(
                {utf8Text(asciiConfig), "--output", utf8Text(asciiOutputPath)}, output, error) == 0,
            "ASCII vtk2nc conversion succeeds: " + error.str() + output.str());
    requireContains(output.str(), "[OK] Saved NetCDF:", "ASCII vtk2nc success message");
    NcReadFile asciiNc(asciiOutputPath);
    for (const std::string_view variable : {"u", "v", "w", "x", "y", "h", "lat", "lon"}) {
        requireSameFloatVariable(nc.id, asciiNc.id, variable);
    }

    const fs::path rangeConfig = project / "range.luw";
    std::ofstream(rangeConfig)
        << "casename = range\n"
        << "datetime = 20260719T140000\n"
        << "si_x_cfd = [600000, 600030]\n"
        << "si_y_cfd = [3000000, 3000040]\n";
    const fs::path rangeVtk = results / "uvw-range_20260719T140000.vtk";
    writeFallbackWindFixture(rangeVtk);
    const fs::path rangeNc = project / "retained_output" / "range.nc";
    output.str({});
    output.clear();
    error.str({});
    error.clear();
    require(luw::cli::run_vtk2nc(
                {utf8Text(rangeConfig), "--output", utf8Text(rangeNc)}, output, error) == 0,
            "vtk2nc range/fallback conversion succeeds: " + error.str() + output.str());
    requireContains(output.str(),
                    "Using horizontal range from config, x=[600000.0, 600030.0], y=[3000000.0, 3000040.0]",
                    "si_x/si_y override message");
    NcReadFile rangeFile(rangeNc);
    int rangeXVar = -1;
    int rangeUVar = -1;
    ncCheck(nc_inq_varid(rangeFile.id, "x", &rangeXVar), "find range x");
    ncCheck(nc_inq_varid(rangeFile.id, "u", &rangeUVar), "find fallback u");
    std::array<float, 2> rangeX{};
    std::array<float, 8> fallbackU{};
    ncCheck(nc_get_var_float(rangeFile.id, rangeXVar, rangeX.data()), "read range x");
    ncCheck(nc_get_var_float(rangeFile.id, rangeUVar, fallbackU.data()), "read fallback u");
    requireNear(rangeX[0], 600000.0, 0.0, "si_x lower bound used without anchor");
    requireNear(rangeX[1], 600030.0, 0.0, "si_x upper bound used without anchor");
    requireNear(fallbackU[0], 5008.0, 0.0, "first non-data_rho vector is selected");
}

void writeAverageFixture(const fs::path& path) {
    constexpr std::array<int, 3> dimensions{2, 2, 4};
    constexpr std::size_t points = 16;
    const std::vector<float> vector = makeTupleField(points, 3, 1.0F, 2.0F, 3.0F);
    const std::vector<float> u = makeScalarField(points, 5.0F);
    const std::vector<float> v = makeScalarField(points, 6.0F);
    const std::vector<float> w = makeScalarField(points, 7.0F);
    const std::vector<float> tke = makeScalarField(points, 10.0F);
    const std::array<luw::post::LegacyVtkFloatFieldView, 5> fields{{
        {"u_avg", 3, luw::post::LegacyVtkFieldKind::Vectors, vector},
        {"u", 1, luw::post::LegacyVtkFieldKind::Scalars, u},
        {"v", 1, luw::post::LegacyVtkFieldKind::Scalars, v},
        {"w", 1, luw::post::LegacyVtkFieldKind::Scalars, w},
        {"tke", 1, luw::post::LegacyVtkFieldKind::Scalars, tke},
    }};
    luw::post::writeLegacyStructuredPointsFloat(
        path, dimensions, {0.0, 0.0, 0.0}, {10.0, 10.0, 25.0}, fields,
        "averaged field generated compatibility fixture");
}

void testAverageToUtmAsl(const fs::path& root) {
    const fs::path project = root / "avg_case";
    const fs::path inputDirectory = project / "RESULTS" / "crop" / "cropped_vtk_raw_assembled";
    fs::create_directories(inputDirectory);
    const fs::path config = project / "conf.luwpf";
    std::ofstream(config)
        << "casename = avg_tiny\n"
        << "datetime = 20260719T130000\n"
        << "cut_lon_manual = [116.99, 117.01]\n"
        << "cut_lat_manual = [29.99, 30.01]\n"
        << "utm_crs = EPSG:32650\n"
        << "rotate_deg = 90\n";
    const fs::path vtkPath = inputDirectory / "ANG_22p5_avg-tiny_cropped.vtk";
    writeAverageFixture(vtkPath);

    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_vtk_avg_to_utm_asl_nc(
        {"--case-dir", utf8Text(project), "--terrain-min-asl", "100", "--overwrite"},
        output, error);
    require(status == 0, "vtk_avg_to_utm_asl_nc returns success: " + error.str() + output.str());
    requireContains(output.str(), "Summary: ok=1, failed=0", "average converter summary");

    const fs::path ncPath = project / "RESULTS" / "nc_utm_asl"
        / "ANG_22p5_avg-tiny_cropped_utm_asl.nc";
    NcReadFile nc(ncPath);
    int varZ = -1;
    int varU = -1;
    int varV = -1;
    int varW = -1;
    int varTke = -1;
    int varEasting = -1;
    ncCheck(nc_inq_varid(nc.id, "z", &varZ), "find avg z");
    ncCheck(nc_inq_varid(nc.id, "u", &varU), "find avg u");
    ncCheck(nc_inq_varid(nc.id, "v", &varV), "find avg v");
    ncCheck(nc_inq_varid(nc.id, "w", &varW), "find avg w");
    ncCheck(nc_inq_varid(nc.id, "tke", &varTke), "find avg tke");
    ncCheck(nc_inq_varid(nc.id, "easting", &varEasting), "find easting");
    std::array<float, 2> z{};
    std::array<float, 8> u{};
    std::array<float, 8> v{};
    std::array<float, 8> w{};
    std::array<float, 8> tke{};
    std::array<double, 4> easting{};
    ncCheck(nc_get_var_float(nc.id, varZ, z.data()), "read avg z");
    ncCheck(nc_get_var_float(nc.id, varU, u.data()), "read avg u");
    ncCheck(nc_get_var_float(nc.id, varV, v.data()), "read avg v");
    ncCheck(nc_get_var_float(nc.id, varW, w.data()), "read avg w");
    ncCheck(nc_get_var_float(nc.id, varTke, tke.data()), "read avg tke");
    ncCheck(nc_get_var_double(nc.id, varEasting, easting.data()), "read easting");
    requireNear(z[0], 100.0, 0.0, "pedestal top maps to terrain minimum ASL");
    requireNear(z[1], 125.0, 0.0, "ASL z spacing");
    // Scalar u/v/w take priority over u_avg. At source point 8 they are
    // (13,14,15); rotate_deg=90 maps them to (14,-13,15).
    requireNear(u[0], 14.0, 1.0e-5, "scalar wind priority and eastward rotation");
    requireNear(v[0], -13.0, 1.0e-5, "scalar wind priority and northward rotation");
    requireNear(w[0], 15.0, 0.0, "vertical component preserved");
    requireNear(tke[0], 18.0, 0.0, "additional scalar retained");
    require(std::isfinite(easting[0]), "UTM easting is finite");
    verifyDeflate3(nc.id, varTke);
}

std::vector<char> readFileBytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "open retained output " + utf8Text(path));
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string readTextAttribute(int ncid, int variable, std::string_view name) {
    std::size_t length = 0;
    ncCheck(nc_inq_attlen(ncid, variable, std::string(name).c_str(), &length),
            "attribute length for " + std::string(name));
    std::string result(length, '\0');
    ncCheck(nc_get_att_text(ncid, variable, std::string(name).c_str(), result.data()),
            "read attribute " + std::string(name));
    return result;
}

std::vector<std::string> attributeNames(int ncid, int variable) {
    int count = 0;
    if (variable == NC_GLOBAL) ncCheck(nc_inq_natts(ncid, &count), "global attribute count");
    else ncCheck(nc_inq_varnatts(ncid, variable, &count), "variable attribute count");
    std::vector<std::string> result;
    for (int index = 0; index < count; ++index) {
        std::array<char, NC_MAX_NAME + 1> name{};
        ncCheck(nc_inq_attname(ncid, variable, index, name.data()), "attribute name");
        result.emplace_back(name.data());
    }
    std::sort(result.begin(), result.end());
    return result;
}

void requireAverageSchema(int ncid) {
    int dimensions = 0;
    int variables = 0;
    int attributes = 0;
    int unlimited = -2;
    ncCheck(nc_inq(ncid, &dimensions, &variables, &attributes, &unlimited), "average schema");
    require(dimensions == 3 && variables == 10 && attributes == 13 && unlimited == -1,
            "average NetCDF dimension/variable/global-attribute counts");

    const std::array<std::string_view, 3> expectedDimensionNames{"z", "y", "x"};
    const std::array<std::size_t, 3> expectedDimensionLengths{2, 2, 2};
    for (int id = 0; id < dimensions; ++id) {
        std::array<char, NC_MAX_NAME + 1> name{};
        std::size_t length = 0;
        ncCheck(nc_inq_dim(ncid, id, name.data(), &length), "average dimension");
        require(name.data() == expectedDimensionNames[static_cast<std::size_t>(id)]
                    && length == expectedDimensionLengths[static_cast<std::size_t>(id)],
                "average dimension order, names, and sizes match the legacy schema");
    }

    const std::array<std::string_view, 10> expectedNames{
        "x", "y", "z", "easting", "northing", "crs", "u", "v", "w", "tke"};
    const std::array<nc_type, 10> expectedTypes{
        NC_DOUBLE, NC_DOUBLE, NC_FLOAT, NC_DOUBLE, NC_DOUBLE,
        NC_INT, NC_FLOAT, NC_FLOAT, NC_FLOAT, NC_FLOAT};
    const std::array<int, 10> expectedRanks{1, 1, 1, 2, 2, 0, 3, 3, 3, 3};
    const std::array<std::vector<int>, 10> expectedDimensions{{
        {2}, {1}, {0}, {1, 2}, {1, 2}, {}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    for (int id = 0; id < variables; ++id) {
        std::array<char, NC_MAX_NAME + 1> name{};
        nc_type type = NC_NAT;
        int rank = 0;
        std::array<int, NC_MAX_VAR_DIMS> dimIds{};
        int variableAttributes = 0;
        ncCheck(nc_inq_var(ncid, id, name.data(), &type, &rank, dimIds.data(), &variableAttributes),
                "average variable schema");
        const std::size_t index = static_cast<std::size_t>(id);
        require(name.data() == expectedNames[index] && type == expectedTypes[index]
                    && rank == expectedRanks[index],
                "average variable order, dtype, and rank match the legacy schema");
        require(std::vector<int>(dimIds.begin(), dimIds.begin() + rank) == expectedDimensions[index],
                "average variable dimension order matches the legacy schema");

        int shuffle = 0;
        int deflate = 0;
        int level = 0;
        ncCheck(nc_inq_var_deflate(ncid, id, &shuffle, &deflate, &level),
                "average variable compression");
        if (id == 5) {
            require(shuffle == 0 && deflate == 0 && level == 0,
                    "crs remains an uncompressed scalar");
        } else {
            require(shuffle == 1 && deflate == 1 && level == 3,
                    "average variables preserve shuffle + zlib level 3");
        }
    }

    require(attributeNames(ncid, NC_GLOBAL) == std::vector<std::string>{
                "angle", "case_datetime", "casename", "history", "pedestal_height_removed_m",
                "rotate_deg", "source_vtk", "terrain_min_asl_m", "utm_crs", "vector_processing",
                "vertical_processing", "vtk_origin", "vtk_spacing"},
            "average global attribute names match the legacy schema");
    int variable = -1;
    ncCheck(nc_inq_varid(ncid, "u", &variable), "find u for schema attributes");
    require(attributeNames(ncid, variable) == std::vector<std::string>{
                "_FillValue", "coordinates", "grid_mapping", "long_name", "source_field",
                "standard_name", "units"},
            "u attribute names match the legacy schema");
    int noFill = 1;
    float fill = 0.0F;
    ncCheck(nc_inq_var_fill(ncid, variable, &noFill, &fill), "u fill value");
    require(noFill == 0 && std::isnan(fill), "u uses the legacy NaN fill value");

    int storage = NC_CONTIGUOUS;
    std::array<std::size_t, 3> chunks{};
    ncCheck(nc_inq_var_chunking(ncid, variable, &storage, chunks.data()), "u chunking");
    require(storage == NC_CHUNKED && chunks == std::array<std::size_t, 3>{1, 2, 2},
            "u uses the legacy per-plane chunks");

    int crs = -1;
    ncCheck(nc_inq_varid(ncid, "crs", &crs), "find crs for WKT format");
    const std::string spatialReference = readTextAttribute(ncid, crs, "spatial_ref");
    require(spatialReference.starts_with("PROJCRS[")
                && spatialReference.find('\n') == std::string::npos,
            "CRS WKT preserves pyproj's compact single-line format");
}

void writeAverageOracleFixture(const fs::path& path, float seed) {
    constexpr std::array<int, 3> dimensions{2, 2, 4};
    constexpr std::size_t points = 16;
    const std::vector<float> u = makeScalarField(points, seed);
    const std::vector<float> v = makeScalarField(points, seed + 20.0F);
    const std::vector<float> w = makeScalarField(points, seed + 40.0F);
    const std::vector<float> tke = makeScalarField(points, seed + 60.0F);
    const std::array<luw::post::LegacyVtkFloatFieldView, 4> fields{{
        {"u", 1, luw::post::LegacyVtkFieldKind::Scalars, u},
        {"v", 1, luw::post::LegacyVtkFieldKind::Scalars, v},
        {"w", 1, luw::post::LegacyVtkFieldKind::Scalars, w},
        {"tke", 1, luw::post::LegacyVtkFieldKind::Scalars, tke},
    }};
    luw::post::writeLegacyStructuredPointsFloat(
        path, dimensions, {1.0, 2.0, 0.0}, {10.0, 20.0, 25.0}, fields,
        "four-city average compatibility fixture");
}

void writeAverageCase(const fs::path& caseDirectory, std::string_view caseName,
                      float seed, std::optional<std::string_view> summaryValue) {
    const fs::path input = caseDirectory / "RESULTS" / "crop" / "cropped_vtk_raw_assembled";
    fs::create_directories(input);
    std::ofstream(caseDirectory / "conf.luwpf")
        << "casename = " << caseName << '\n'
        << "datetime = 20260719T230000\n"
        << "cut_lon_manual = [117.01, 116.99]\n"
        << "cut_lat_manual = [30.01, 29.99]\n"
        << "utm_crs = EPSG:32650\n"
        << "rotate_deg = 17.5\n";
    if (summaryValue) {
        std::ofstream(caseDirectory / "prep_summary.json")
            << "{\"dem_absolute_min_m\": " << *summaryValue << "}\n";
    }
    writeAverageOracleFixture(input / "ANG_20_avg-second_cropped.vtk", seed + 100.0F);
    writeAverageOracleFixture(input / "ANG_10_avg-first_cropped.vtk", seed);
}

std::string runAverage(const std::vector<std::string>& arguments) {
    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_vtk_avg_to_utm_asl_nc(arguments, output, error);
    require(status == 0, "average command succeeds: " + error.str() + output.str());
    return output.str();
}

void testAverageMultiCaseCompatibility(const fs::path& root) {
    const fs::path project = root / "avg_four_cities";
    fs::create_directories(project);
    const fs::path range = project / "Range_four_cities.txt";
    std::ofstream(range)
        << "beijing:\n"
        << "terrain_min_asl_m = 11.25\n"
        << "Chongqing total: local -99 m, terrain -22.5 m\n"
        << "shanghai:\n"
        << "33.75 m\n"
        << "Singapore terrain minimum = -4.5 m\n";
    // A conflicting default file proves that an explicit Range_*.txt path wins.
    std::ofstream(project / "Range.txt")
        << "beijing: 999 m\nchongqing: 999 m\nshanghai: 999 m\nsingapore: 999 m\n";

    constexpr std::array<std::string_view, 4> cities{
        "beijing", "chongqing", "shanghai", "singapore"};
    constexpr std::array<double, 4> terrain{11.25, -22.5, 33.75, -4.5};
    for (std::size_t index = 0; index < cities.size(); ++index) {
        const std::string summary = index == 1 ? "\"900.5\"" : std::to_string(800 + index);
        writeAverageCase(project / std::string(cities[index]), cities[index],
                         1.0F + static_cast<float>(index) * 100.0F,
                         std::string_view(summary));
    }

    const std::string multiOutput = runAverage({
        "--root", utf8Text(project), "--range-file", utf8Text(range),
        "--cases", "--terrain-min-asl", "777", "--limit", "1"});
    requireContains(multiOutput, "Summary: ok=4, failed=0", "four-city summary");
    for (std::size_t city = 0; city < cities.size(); ++city) {
        std::ostringstream expectedLog;
        expectedLog << "Case " << cities[city] << ": files=1/2, terrain_min_asl_m="
                    << std::fixed << std::setprecision(6) << terrain[city];
        requireContains(multiOutput, expectedLog.str(), "four-city discovery and Range value");

        const fs::path outputDirectory = project / std::string(cities[city])
            / "RESULTS" / "nc_utm_asl";
        const fs::path first = outputDirectory / "ANG_10_avg-first_cropped_utm_asl.nc";
        const fs::path second = outputDirectory / "ANG_20_avg-second_cropped_utm_asl.nc";
        require(fs::is_regular_file(first), "--limit 1 emits the first sorted filename");
        require(!fs::exists(second), "--limit 1 does not emit the second sorted filename");
        NcReadFile nc(first);
        if (city == 0) requireAverageSchema(nc.id);
        const std::vector<float> z = readFloatVariable(nc.id, "z");
        require(z.size() == 2, "two pedestal-trimmed z levels");
        requireNear(z[0], terrain[city], 0.0, "multi-case Range terrain minimum");
        requireNear(z[1], terrain[city] + 25.0, 0.0, "multi-case ASL z spacing");
        require(readTextAttribute(nc.id, NC_GLOBAL, "angle") == "10",
                "angle and output naming use the selected ANG_10 input");

        nc_type terrainAttributeType = NC_NAT;
        ncCheck(nc_inq_atttype(nc.id, NC_GLOBAL, "terrain_min_asl_m", &terrainAttributeType),
                "terrain attribute type");
        require(terrainAttributeType == NC_DOUBLE, "terrain minimum is a double attribute");
        double terrainAttribute = 0.0;
        ncCheck(nc_get_att_double(nc.id, NC_GLOBAL, "terrain_min_asl_m", &terrainAttribute),
                "terrain attribute value");
        requireNear(terrainAttribute, terrain[city], 0.0, "terrain attribute uses Range in multi-case mode");

        const float seed = 1.0F + static_cast<float>(city) * 100.0F;
        const float cosine = static_cast<float>(std::cos(17.5 * 0.0174532925199432957692));
        const float sine = static_cast<float>(std::sin(17.5 * 0.0174532925199432957692));
        const std::vector<float> u = readFloatVariable(nc.id, "u");
        const std::vector<float> v = readFloatVariable(nc.id, "v");
        const std::vector<float> w = readFloatVariable(nc.id, "w");
        const std::vector<float> tke = readFloatVariable(nc.id, "tke");
        for (std::size_t value = 0; value < 8; ++value) {
            const float uModel = seed + 8.0F + static_cast<float>(value);
            const float vModel = seed + 28.0F + static_cast<float>(value);
            requireNear(u[value], cosine * uModel + sine * vModel, 0.0,
                        "all eastward values match the legacy rotation");
            requireNear(v[value], -sine * uModel + cosine * vModel, 0.0,
                        "all northward values match the legacy rotation");
            requireNear(w[value], seed + 48.0F + static_cast<float>(value), 0.0,
                        "all vertical values are retained");
            requireNear(tke[value], seed + 68.0F + static_cast<float>(value), 0.0,
                        "all scalar values are retained");
        }

        if (city == 0) {
            int eastingVariable = -1;
            int northingVariable = -1;
            ncCheck(nc_inq_varid(nc.id, "easting", &eastingVariable), "find golden easting");
            ncCheck(nc_inq_varid(nc.id, "northing", &northingVariable), "find golden northing");
            std::array<double, 4> easting{};
            std::array<double, 4> northing{};
            ncCheck(nc_get_var_double(nc.id, eastingVariable, easting.data()), "read golden easting");
            ncCheck(nc_get_var_double(nc.id, northingVariable, northing.data()), "read golden northing");
            constexpr std::array<double, 4> expectedEasting{
                498401.58832921553, 498411.125498723, 498407.6024452056, 498417.13961471315};
            constexpr std::array<double, 4> expectedNorthing{
                3317879.259935158, 3317876.252877163, 3317898.334274173, 3317895.327216178};
            for (std::size_t value = 0; value < 4; ++value) {
                requireNear(easting[value], expectedEasting[value], 1.0e-6,
                            "all easting values match the old Python oracle");
                requireNear(northing[value], expectedNorthing[value], 1.0e-6,
                            "all northing values match the old Python oracle");
            }
        }
    }

    const fs::path chongqing = project / "chongqing";
    const std::string summarySubdirectory = "RESULTS/nc_summary_string";
    const std::vector<std::string> summaryArguments{
        "--root", utf8Text(project), "--case-dir", utf8Text(chongqing),
        "--range-file", utf8Text(range), "--output-subdir", summarySubdirectory, "--limit", "1"};
    const std::string summaryOutput = runAverage(summaryArguments);
    requireContains(summaryOutput, "Terrain min ASL from prep_summary.json: 900.500000000 m",
                    "numeric-string summary has legacy float(value) semantics");
    const fs::path summaryPath = chongqing / summarySubdirectory
        / "ANG_10_avg-first_cropped_utm_asl.nc";
    {
        NcReadFile summary(summaryPath);
        requireNear(readFloatVariable(summary.id, "z")[0], 900.5, 0.0,
                    "single-case summary overrides Range");
    }

    const std::vector<char> bytesBeforeSkip = readFileBytes(summaryPath);
    const fs::file_time_type timeBeforeSkip = fs::last_write_time(summaryPath);
    const std::string skipOutput = runAverage(summaryArguments);
    requireContains(skipOutput,
                    "Skip existing:",
                    "default existing-output behavior");
    requireContains(skipOutput,
                    "ANG_10_avg-first_cropped_utm_asl.nc",
                    "default skip identifies the expected output");
    require(readFileBytes(summaryPath) == bytesBeforeSkip
                && fs::last_write_time(summaryPath) == timeBeforeSkip,
            "default mode preserves existing NetCDF bytes and timestamp");

    std::vector<std::string> overwriteArguments = summaryArguments;
    overwriteArguments.insert(overwriteArguments.end(), {"--terrain-min-asl", "77.25", "--overwrite"});
    (void)runAverage(overwriteArguments);
    {
        NcReadFile overwritten(summaryPath);
        requireNear(readFloatVariable(overwritten.id, "z")[0], 77.25, 0.0,
                    "explicit terrain overrides summary when --overwrite replaces a generated fixture");
    }
    require(readFileBytes(summaryPath) != bytesBeforeSkip,
            "--overwrite replaces the controlled generated NetCDF output");

    const std::string limitZeroSubdirectory = "RESULTS/nc_limit_zero";
    (void)runAverage({
        "--root", utf8Text(project), "--case-dir", utf8Text(chongqing),
        "--range-file", utf8Text(range), "--output-subdir", limitZeroSubdirectory,
        "--limit", "0", "--terrain-min-asl", "77.25"});
    require(fs::is_regular_file(chongqing / limitZeroSubdirectory / "ANG_10_avg-first_cropped_utm_asl.nc")
                && fs::is_regular_file(chongqing / limitZeroSubdirectory / "ANG_20_avg-second_cropped_utm_asl.nc"),
            "non-positive --limit preserves the legacy all-files behavior");

    const fs::path rangeOnly = project / "range_only";
    writeAverageCase(rangeOnly, "beijing", 501.0F, std::nullopt);
    const std::string rangeOnlySubdirectory = "RESULTS/nc_range_only";
    (void)runAverage({
        "--root", utf8Text(project), "--case-dir", utf8Text(rangeOnly),
        "--range-file", utf8Text(range), "--output-subdir", rangeOnlySubdirectory, "--limit", "1"});
    NcReadFile rangeFallback(rangeOnly / rangeOnlySubdirectory / "ANG_10_avg-first_cropped_utm_asl.nc");
    requireNear(readFloatVariable(rangeFallback.id, "z")[0], 11.25, 0.0,
                "single-case mode falls back from absent summary to Range_*.txt");
}

} // namespace

int main() {
    try {
        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const fs::path root = fs::path(LUW_VTK_POST_ARTIFACT_ROOT)
            / ("run_" + std::to_string(nonce));
        fs::create_directories(root);
        testVtk2Nc(root);
        testAverageToUtmAsl(root);
        testAverageMultiCaseCompatibility(root);
        std::cout << "Native VTK post-processing compatibility tests passed.\n";
        std::cout << "Generated fixtures retained at: " << utf8Text(root) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Native VTK post-processing compatibility test failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
