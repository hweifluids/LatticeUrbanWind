#include "luw/vtk_post_commands.hpp"

#include <vtk_netcdf.h>

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
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireContains(const std::string& text, std::string_view fragment, const std::string& message) {
    require(text.find(fragment) != std::string::npos,
            message + " (missing: " + std::string(fragment) + ")");
}

void requireNear(double actual, double expected, double tolerance, const std::string& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
            message + ": actual=" + std::to_string(actual)
                + ", expected=" + std::to_string(expected));
}

std::string utf8Text(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

void ncCheck(int status, const std::string& context) {
    if (status != NC_NOERR) throw std::runtime_error(context + ": " + nc_strerror(status));
}

class NcReadFile final {
public:
    explicit NcReadFile(const fs::path& path) {
        const std::string utf8 = utf8Text(path);
        ncCheck(nc_open(utf8.c_str(), NC_NOWRITE, &id), "open " + utf8);
    }
    ~NcReadFile() {
        if (id >= 0) nc_close(id);
    }
    NcReadFile(const NcReadFile&) = delete;
    NcReadFile& operator=(const NcReadFile&) = delete;
    int id = -1;
};

struct FixtureField final {
    std::string name;
    std::string type;
    int components = 1;
    std::vector<double> values;
};

template <typename Unsigned>
void writeBigEndianInteger(std::ofstream& output, Unsigned value) {
    static_assert(std::is_unsigned_v<Unsigned>);
    std::array<unsigned char, sizeof(Unsigned)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        const unsigned int shift = static_cast<unsigned int>((bytes.size() - index - 1) * 8);
        bytes[index] = static_cast<unsigned char>((value >> shift) & static_cast<Unsigned>(0xffU));
    }
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void writeTypedValue(std::ofstream& output, std::string_view type, double raw) {
    if (type == "char") {
        const auto value = static_cast<std::int8_t>(raw);
        output.put(static_cast<char>(static_cast<std::uint8_t>(value)));
    } else if (type == "unsigned_char") {
        output.put(static_cast<char>(static_cast<std::uint8_t>(raw)));
    } else if (type == "short") {
        writeBigEndianInteger(output, static_cast<std::uint16_t>(static_cast<std::int16_t>(raw)));
    } else if (type == "unsigned_short") {
        writeBigEndianInteger(output, static_cast<std::uint16_t>(raw));
    } else if (type == "int" || type == "long") {
        writeBigEndianInteger(output, static_cast<std::uint32_t>(static_cast<std::int32_t>(raw)));
    } else if (type == "unsigned_int" || type == "unsigned_long") {
        writeBigEndianInteger(output, static_cast<std::uint32_t>(raw));
    } else if (type == "float") {
        writeBigEndianInteger(output, std::bit_cast<std::uint32_t>(static_cast<float>(raw)));
    } else if (type == "double") {
        writeBigEndianInteger(output, std::bit_cast<std::uint64_t>(raw));
    } else {
        throw std::runtime_error("unsupported fixture type: " + std::string(type));
    }
}

std::vector<double> constants(std::size_t tuples, std::span<const double> values) {
    std::vector<double> result(tuples * values.size());
    for (std::size_t tuple = 0; tuple < tuples; ++tuple) {
        for (std::size_t component = 0; component < values.size(); ++component) {
            result[tuple * values.size() + component] = values[component];
        }
    }
    return result;
}

void writeTypedFixture(const fs::path& path, bool fullFields) {
    constexpr int nx = 5;
    constexpr int ny = 4;
    constexpr int nz = 2;
    constexpr std::size_t tuples = static_cast<std::size_t>(nx * ny * nz);
    std::vector<double> velocity(tuples * 3);
    for (std::size_t tuple = 0; tuple < tuples; ++tuple) {
        const std::size_t x = tuple % nx;
        const std::size_t y = (tuple / nx) % ny;
        const std::size_t z = tuple / (nx * ny);
        const double base = 2.0 + 0.3 * static_cast<double>(x * x)
            - 0.2 * static_cast<double>(y) + 0.05 * static_cast<double>(x * y)
            + 1.5 * static_cast<double>(z);
        velocity[tuple * 3] = base;
        velocity[tuple * 3 + 1] = base + 100.0;
        velocity[tuple * 3 + 2] = base - 50.0;
    }

    std::vector<FixtureField> fields;
    fields.push_back({"Velocity", "float", 3, std::move(velocity)});
    if (fullFields) {
        const std::array<double, 1> dataValue{1.25};
        const std::array<double, 1> heightValue{-123.0};
        const std::array<double, 1> latValue{200.0};
        const std::array<double, 2> oddValues{65000.0, 1234.0};
        const std::array<double, 1> countValue{-1234567.0};
        const std::array<double, 1> bigValue{4000000000.0};
        const std::array<double, 1> signedLongValue{-123456.0};
        const std::array<double, 1> unsignedLongValue{3000000000.0};
        const std::array<double, 1> uValue{-5.0};
        const std::array<double, 3> fallbackValues{7.0, 8.0, 9.0};
        fields.push_back({"data", "double", 1, constants(tuples, dataValue)});
        fields.push_back({"height", "short", 1, constants(tuples, heightValue)});
        fields.push_back({"lat", "unsigned_char", 1, constants(tuples, latValue)});
        fields.push_back({"odd-name", "unsigned_short", 2, constants(tuples, oddValues)});
        fields.push_back({"count", "int", 1, constants(tuples, countValue)});
        fields.push_back({"big", "unsigned_int", 1, constants(tuples, bigValue)});
        fields.push_back({"signed_long", "long", 1, constants(tuples, signedLongValue)});
        fields.push_back({"unsigned_long", "unsigned_long", 1,
                          constants(tuples, unsignedLongValue)});
        fields.push_back({"u", "char", 1, constants(tuples, uValue)});
        fields.push_back({"fallback", "float", 3, constants(tuples, fallbackValues)});
    }

    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "create typed VTK fixture");
    output << "# vtk DataFile Version 3.0\n"
           << "vtk2nc_new native typed fixture\n"
           << "BINARY\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS " << nx << ' ' << ny << ' ' << nz << "\n"
           << "ORIGIN 0 0 5\n"
           << "SPACING 100 110 7.5\n"
           << "POINT_DATA " << tuples << "\n";
    for (const FixtureField& field : fields) {
        output << "SCALARS " << field.name << ' ' << field.type << ' ' << field.components << "\n"
               << "LOOKUP_TABLE default\n";
        for (const double value : field.values) writeTypedValue(output, field.type, value);
        output << '\n';
    }
    output.flush();
    require(static_cast<bool>(output), "finish typed VTK fixture");
}

void writeUnsupportedVectorFixture(const fs::path& path) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "create VECTORS rejection fixture");
    output << "# vtk DataFile Version 3.0\n"
           << "vtk2nc_new unsupported VECTORS fixture\n"
           << "BINARY\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS 2 2 1\n"
           << "ORIGIN 0 0 5\n"
           << "SPACING 100 100 10\n"
           << "POINT_DATA 4\n"
           << "VECTORS Velocity float\n";
    for (int index = 0; index < 12; ++index) writeTypedValue(output, "float", index + 1.0);
    output << '\n';
    output.flush();
    require(static_cast<bool>(output), "finish VECTORS rejection fixture");
}

void writeAsciiExtensionFixture(const fs::path& path) {
    constexpr int nx = 5;
    constexpr int ny = 4;
    constexpr int nz = 2;
    constexpr int points = nx * ny * nz;
    constexpr int cells = (nx - 1) * (ny - 1) * (nz - 1);
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "create ASCII extension fixture");
    output << "# vtk DataFile Version 3.0\n"
           << "vtk2nc_new ASCII extension fixture\n"
           << "ASCII\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS " << nx << ' ' << ny << ' ' << nz << "\n"
           << "ORIGIN 0 0 5\n"
           << "SPACING 100 110 7.5\n"
           << "CELL_DATA " << cells << "\n"
           << "SCALARS cell_mask unsigned_char\n"
           << "LOOKUP_TABLE default\n";
    for (int index = 0; index < cells; ++index) output << (index & 1) << ' ';
    output << "\nPOINT_DATA " << points << "\n"
           << "VECTORS Velocity float\n";
    for (int index = 0; index < points; ++index) output << "1 2 3\n";
    output.flush();
    require(static_cast<bool>(output), "finish ASCII extension fixture");
}

fs::path writeCase(const fs::path& project, bool projectTemp, bool fullFields,
                   std::string_view fileStem = "rho-fields_20260719T120000") {
    fs::create_directories(project / "RESULTS");
    const fs::path config = project / "conf.luw";
    std::ofstream deck(config);
    deck << "casename = native_new\n"
         << "datetime = 20260719T120000\n"
         << "cut_lon_manual = [120.0040, 120.0000]\n"
         << "cut_lat_manual = [30.0030, 30.0000]\n";
    deck.flush();
    require(static_cast<bool>(deck), "write vtk2nc_new deck");
    const fs::path vtkDirectory = projectTemp ? project / "proj_temp" / "vtk" : project / "RESULTS";
    writeTypedFixture(vtkDirectory / (std::string(fileStem) + ".vtk"), fullFields);
    return config;
}

std::size_t dimensionLength(int ncid, const char* name) {
    int dimension = -1;
    std::size_t length = 0;
    ncCheck(nc_inq_dimid(ncid, name, &dimension), "find dimension " + std::string(name));
    ncCheck(nc_inq_dimlen(ncid, dimension, &length), "read dimension " + std::string(name));
    return length;
}

int variableId(int ncid, const char* name) {
    int variable = -1;
    ncCheck(nc_inq_varid(ncid, name, &variable), "find variable " + std::string(name));
    return variable;
}

std::string textAttribute(int ncid, int variable, const char* name) {
    std::size_t length = 0;
    ncCheck(nc_inq_attlen(ncid, variable, name, &length), "attribute length " + std::string(name));
    std::string value(length, '\0');
    ncCheck(nc_get_att_text(ncid, variable, name, value.data()), "attribute " + std::string(name));
    return value;
}

void requireDeflate3(int ncid, int variable, const std::string& name) {
    int shuffle = 0;
    int deflate = 0;
    int level = 0;
    ncCheck(nc_inq_var_deflate(ncid, variable, &shuffle, &deflate, &level), "deflate " + name);
    require(shuffle == 1 && deflate == 1 && level == 3, name + " uses shuffle/zlib level 3");
}

std::vector<float> readFloatVariable(int ncid, const char* name, std::size_t count) {
    std::vector<float> values(count);
    ncCheck(nc_get_var_float(ncid, variableId(ncid, name), values.data()), "read " + std::string(name));
    return values;
}

void requireConstant(int ncid, const char* name, std::size_t count, double expected,
                     double tolerance = 1.0e-3) {
    const std::vector<float> values = readFloatVariable(ncid, name, count);
    for (const float value : values) requireNear(value, expected, tolerance, std::string(name) + " constant");
}

void verifyPrimaryNetcdf(const fs::path& path, const fs::path& sourceVtk) {
    NcReadFile nc(path);
    const std::size_t heightCount = dimensionLength(nc.id, "height");
    const std::size_t latitudeCount = dimensionLength(nc.id, "lat");
    const std::size_t longitudeCount = dimensionLength(nc.id, "lon");
    require(heightCount == 2, "height dimension preserves all levels");
    require(latitudeCount >= 2 && latitudeCount <= 16, "target latitude dimension is bounded");
    require(longitudeCount >= 2 && longitudeCount <= 20, "target longitude dimension is bounded");
    const std::size_t valuesPerVariable = heightCount * latitudeCount * longitudeCount;

    const int height = variableId(nc.id, "height");
    const int latitude = variableId(nc.id, "lat");
    const int longitude = variableId(nc.id, "lon");
    nc_type type = NC_NAT;
    ncCheck(nc_inq_vartype(nc.id, height, &type), "height type");
    require(type == NC_FLOAT, "height is float32");
    ncCheck(nc_inq_vartype(nc.id, latitude, &type), "latitude type");
    require(type == NC_DOUBLE, "latitude is float64");
    ncCheck(nc_inq_vartype(nc.id, longitude, &type), "longitude type");
    require(type == NC_DOUBLE, "longitude is float64");
    requireDeflate3(nc.id, height, "height");
    requireDeflate3(nc.id, latitude, "lat");
    requireDeflate3(nc.id, longitude, "lon");

    const std::vector<std::string> expectedVariables{
        "u", "v", "w", "rho", "height_var", "lat_var", "odd_name_c0", "odd_name_c1",
        "count", "big", "signed_long", "unsigned_long", "u_2",
        "fallback_x", "fallback_y", "fallback_z",
    };
    for (const std::string& name : expectedVariables) {
        const int variable = variableId(nc.id, name.c_str());
        requireDeflate3(nc.id, variable, name);
        int storage = 0;
        std::array<std::size_t, 3> chunks{};
        ncCheck(nc_inq_var_chunking(nc.id, variable, &storage, chunks.data()), "chunks " + name);
        require(storage == NC_CHUNKED, name + " is chunked");
        require(chunks == std::array<std::size_t, 3>{1, latitudeCount, longitudeCount},
                name + " chunks match Python contract");
    }

    const std::vector<float> heights = readFloatVariable(nc.id, "height", heightCount);
    requireNear(heights[0], 5.0, 0.0, "first height");
    requireNear(heights[1], 12.5, 0.0, "second height");
    std::vector<double> latitudes(latitudeCount);
    std::vector<double> longitudes(longitudeCount);
    ncCheck(nc_get_var_double(nc.id, latitude, latitudes.data()), "read latitude");
    ncCheck(nc_get_var_double(nc.id, longitude, longitudes.data()), "read longitude");
    require(std::is_sorted(latitudes.begin(), latitudes.end()), "latitude is ascending");
    require(std::is_sorted(longitudes.begin(), longitudes.end()), "longitude is ascending");
    constexpr std::array<double, 4> oracleLatitudes{
        30.000000015203629, 30.000991691810217, 30.001983368416809, 30.002975045023398,
    };
    constexpr std::array<double, 5> oracleLongitudes{
        119.99999999999999, 120.00103576208684, 120.00207152417372,
        120.00310728626059, 120.00414304834744,
    };
    require(std::equal(latitudes.begin(), latitudes.end(),
                       oracleLatitudes.begin(), oracleLatitudes.end()),
            "latitude coordinates match the Python oracle exactly");
    require(std::equal(longitudes.begin(), longitudes.end(),
                       oracleLongitudes.begin(), oracleLongitudes.end()),
            "longitude coordinates match the Python oracle exactly");

    const std::vector<float> u = readFloatVariable(nc.id, "u", valuesPerVariable);
    const std::vector<float> v = readFloatVariable(nc.id, "v", valuesPerVariable);
    const std::vector<float> w = readFloatVariable(nc.id, "w", valuesPerVariable);
    for (std::size_t index = 0; index < valuesPerVariable; ++index) {
        require(std::isfinite(u[index]), "cubic interpolation output is finite");
        requireNear(v[index] - u[index], 100.0, 2.0e-5, "vector component offset v-u");
        requireNear(w[index] - u[index], -50.0, 2.0e-5, "vector component offset w-u");
    }
    constexpr std::array<float, 40> oracleU{
        1.99999845F, 2.30000043F, 3.20000172F, 4.70000219F, 6.80000019F,
        1.79999983F, 2.15000749F, 3.10000205F, 4.64998007F, 6.7999773F,
        1.6000042F, 2.00001669F, 3.00000358F, 4.5999589F, 6.79995441F,
        1.40000951F, 1.85002649F, 2.9000051F, 4.5499382F, 6.79993153F,
        3.49999833F, 3.80000043F, 4.70000124F, 6.20000219F, 8.30000019F,
        3.29999971F, 3.65000749F, 4.60000181F, 6.14998007F, 8.2999773F,
        3.10000396F, 3.50001669F, 4.50000381F, 6.0999589F, 8.29995441F,
        2.90000963F, 3.35002637F, 4.40000534F, 6.0499382F, 8.29993153F,
    };
    require(std::equal(u.begin(), u.end(), oracleU.begin(), oracleU.end()),
            "cubic velocity values match the Python oracle bit-for-bit");

    requireConstant(nc.id, "rho", valuesPerVariable, 1.25, 1.0e-6);
    requireConstant(nc.id, "height_var", valuesPerVariable, -123.0, 1.0e-4);
    requireConstant(nc.id, "lat_var", valuesPerVariable, 200.0, 1.0e-4);
    requireConstant(nc.id, "odd_name_c0", valuesPerVariable, 65000.0, 2.0e-2);
    requireConstant(nc.id, "odd_name_c1", valuesPerVariable, 1234.0, 1.0e-3);
    requireConstant(nc.id, "count", valuesPerVariable, -1234567.0, 1.0);
    requireConstant(nc.id, "big", valuesPerVariable, static_cast<float>(4000000000.0), 512.0);
    requireConstant(nc.id, "signed_long", valuesPerVariable, -123456.0, 0.1);
    requireConstant(nc.id, "unsigned_long", valuesPerVariable,
                    static_cast<float>(3000000000.0), 512.0);
    requireConstant(nc.id, "u_2", valuesPerVariable, -5.0, 1.0e-5);
    requireConstant(nc.id, "fallback_x", valuesPerVariable, 7.0, 1.0e-5);
    requireConstant(nc.id, "fallback_y", valuesPerVariable, 8.0, 1.0e-5);
    requireConstant(nc.id, "fallback_z", valuesPerVariable, 9.0, 1.0e-5);

    require(textAttribute(nc.id, NC_GLOBAL, "source_vtk") == utf8Text(fs::absolute(sourceVtk)),
            "source_vtk global attribute");
    require(textAttribute(nc.id, NC_GLOBAL, "utm_crs") == "EPSG:32651", "auto-inferred UTM CRS");
    require(textAttribute(nc.id, NC_GLOBAL, "note")
                == "Converted from FluidX3D legacy VTK to strict Cartesian lon/lat grid using inverse UTM-rotation-origin-shift chain.",
            "conversion note");
    require(textAttribute(nc.id, variableId(nc.id, "u"), "long_name") == "Velocity component x",
            "main vector long_name");
    require(textAttribute(nc.id, variableId(nc.id, "u"), "units") == "m s-1",
            "main vector units");
}

void testFullConversion(const fs::path& root) {
    const fs::path project = root / "typed_case";
    const fs::path config = writeCase(project, true, true);
    const fs::path sourceVtk = project / "proj_temp" / "vtk" / "rho-fields_20260719T120000.vtk";
    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_vtk2nc_new({utf8Text(config)}, output, error);
    require(status == 0, "vtk2nc_new typed conversion succeeds: " + output.str() + error.str());
    requireContains(output.str(), "Auto-inferred CRS: EPSG:32651", "UTM inference log");
    requireContains(output.str(), "Discovered 1 VTK file(s) for processing.", "discovery log");
    requireContains(output.str(), "Target mapping attempt 1:", "mapping attempt log");
    requireContains(output.str(), "Output variables for rho-fields_20260719T120000.vtk: u, v, w, rho",
                    "field mapping log");
    requireContains(output.str(), "Conversion summary: success=1, failed=0", "success summary");
    verifyPrimaryNetcdf(project / "RESULTS" / "rho-fields_20260719T120000.nc", sourceVtk);
}

void testContinueAndFinalFailure(const fs::path& root) {
    const fs::path project = root / "continue_case";
    const fs::path config = writeCase(project, false, false, "a_fail_20260719T120000");
    writeTypedFixture(project / "RESULTS" / "z_ok_20260719T120000.vtk", false);
    fs::create_directories(project / "RESULTS" / "a_fail_20260719T120000.nc");
    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_vtk2nc_new({utf8Text(config)}, output, error);
    require(status == 2, "one per-file NetCDF failure returns final status 2");
    requireContains(output.str(), "Failed file", "per-file warning is reported");
    requireContains(output.str(), "Conversion summary: success=1, failed=1", "mixed result summary");
    requireContains(output.str(), "1 file(s) failed during conversion.", "final failure message");
    require(fs::is_regular_file(project / "RESULTS" / "z_ok_20260719T120000.nc"),
            "processing continues to later VTK file");
}

void testBinaryContractFailureIsFatal(const fs::path& root) {
    const fs::path project = root / "binary_contract_case";
    const fs::path config = writeCase(project, false, false, "z_valid_20260719T120000");
    writeUnsupportedVectorFixture(project / "RESULTS" / "a_vectors_20260719T120000.vtk");
    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_vtk2nc_new({utf8Text(config)}, output, error);
    require(status == 1, "unsupported BINARY section retains immediate fatal status 1");
    requireContains(output.str(), "Unsupported VTK data section header",
                    "unsupported BINARY section diagnostic");
    require(output.str().find("Conversion summary:") == std::string::npos,
            "fatal parser validation bypasses per-file summary like SystemExit");
    require(!fs::exists(project / "RESULTS" / "z_valid_20260719T120000.nc"),
            "fatal parser validation does not continue to later files");
}

void testAsciiExtension(const fs::path& root) {
    const fs::path project = root / "ascii_extension_case";
    const fs::path config = writeCase(project, true, false);
    const fs::path asciiPath = project / "proj_temp" / "vtk" / "ascii_20260719T120000.vtk";
    writeAsciiExtensionFixture(asciiPath);
    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_vtk2nc_new({utf8Text(config)}, output, error) == 0,
            "ASCII compatibility extension succeeds: " + output.str() + error.str());
    requireContains(output.str(), "ASCII VTK compatibility extension enabled",
                    "ASCII extension is explicitly reported");
    NcReadFile nc(project / "RESULTS" / "ascii_20260719T120000.nc");
    const std::size_t count = dimensionLength(nc.id, "height")
        * dimensionLength(nc.id, "lat") * dimensionLength(nc.id, "lon");
    requireConstant(nc.id, "u", count, 1.0, 1.0e-5);
    requireConstant(nc.id, "v", count, 2.0, 1.0e-5);
    requireConstant(nc.id, "w", count, 3.0, 1.0e-5);
    int ignored = -1;
    require(nc_inq_varid(nc.id, "cell_mask", &ignored) == NC_ENOTVAR,
            "ASCII CELL_DATA is not exposed as point output");
}

void testUsage(std::ostream& sink) {
    std::ostringstream output;
    std::ostringstream error;
    require(luw::cli::run_vtk2nc_new({}, output, error) == 1, "missing path returns status 1");
    requireContains(output.str(), "Usage: python vtk2nc_new.py <path_to_case.luw>", "legacy usage text");
    output.str({});
    output.clear();
    constexpr std::string_view missingName = "vtk2nc_new_deliberately_missing_case.luw";
    require(luw::cli::run_vtk2nc_new({std::string(missingName)}, output, error) == 1,
            "missing LUW file returns status 1");
    requireContains(output.str(), "LUW file not found: " + utf8Text(fs::absolute(missingName)),
                    "missing LUW diagnostic uses the resolved absolute path");
    require(error.str().empty(), "legacy errors remain on stdout");
    sink << output.str();
}

} // namespace

int main() {
    try {
        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const fs::path root = fs::path(LUW_VTK2NC_NEW_ARTIFACT_ROOT)
            / ("run_" + std::to_string(nonce));
        fs::create_directories(root);
        testFullConversion(root);
        testContinueAndFinalFailure(root);
        testBinaryContractFailureIsFatal(root);
        testAsciiExtension(root);
        std::ostringstream usage;
        testUsage(usage);
        std::cout << "Native vtk2nc_new compatibility tests passed.\n"
                  << "Generated fixtures retained at: " << utf8Text(root) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Native vtk2nc_new compatibility test failed: " << exception.what() << '\n';
        return 1;
    }
}
