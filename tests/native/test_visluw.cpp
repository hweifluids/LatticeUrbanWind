#include "luw/legacy_vtk.hpp"
#include "luw/vtk_post_commands.hpp"

#include <vtk_netcdf.h>
#include <vtk_zlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
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

void requireNear(double actual, double expected, double tolerance, const std::string& message) {
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

std::vector<unsigned char> readBytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "open " + utf8(path));
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::uint16_t u16(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(bytes.at(offset) | (bytes.at(offset + 1) << 8U));
}

std::uint32_t u32(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes.at(offset))
        | (static_cast<std::uint32_t>(bytes.at(offset + 1)) << 8U)
        | (static_cast<std::uint32_t>(bytes.at(offset + 2)) << 16U)
        | (static_cast<std::uint32_t>(bytes.at(offset + 3)) << 24U);
}

std::vector<unsigned char> inflateRaw(const std::vector<unsigned char>& bytes,
                                      std::size_t offset, std::size_t compressedSize,
                                      std::size_t uncompressedSize) {
    require(offset + compressedSize <= bytes.size(), "DEFLATE member bounds");
    std::vector<unsigned char> result(uncompressedSize);
    z_stream stream{};
    require(inflateInit2(&stream, -MAX_WBITS) == Z_OK, "initialize raw DEFLATE reader");
    stream.next_in = const_cast<Bytef*>(bytes.data() + offset);
    stream.avail_in = static_cast<uInt>(compressedSize);
    stream.next_out = result.data();
    stream.avail_out = static_cast<uInt>(result.size());
    const int status = inflate(&stream, Z_FINISH);
    (void)inflateEnd(&stream);
    require(status == Z_STREAM_END && stream.total_out == uncompressedSize, "inflate NPZ member");
    return result;
}

std::map<std::string, std::vector<unsigned char>> readNpzMembers(const fs::path& path) {
    const auto bytes = readBytes(path);
    std::map<std::string, std::vector<unsigned char>> result;
    std::size_t eocd = bytes.size() >= 22 ? bytes.size() - 22 : 0;
    while (eocd != 0 && u32(bytes, eocd) != 0x06054b50U) --eocd;
    require(u32(bytes, eocd) == 0x06054b50U, "ZIP end-of-central-directory");
    const std::uint16_t count = u16(bytes, eocd + 10);
    std::size_t central = u32(bytes, eocd + 16);
    for (std::uint16_t entry = 0; entry < count; ++entry) {
        require(u32(bytes, central) == 0x02014b50U, "ZIP central member");
        require(u16(bytes, central + 10) == 8, "NPZ member uses DEFLATE");
        const std::uint32_t compressed = u32(bytes, central + 20);
        const std::uint32_t uncompressed = u32(bytes, central + 24);
        const std::uint16_t nameLength = u16(bytes, central + 28);
        const std::uint16_t extraLength = u16(bytes, central + 30);
        const std::uint16_t commentLength = u16(bytes, central + 32);
        const std::uint32_t local = u32(bytes, central + 42);
        require(compressed != 0xffffffffU && uncompressed != 0xffffffffU && local != 0xffffffffU,
                "tiny fixture does not need ZIP64");
        const std::string name(bytes.begin() + static_cast<std::ptrdiff_t>(central + 46),
                               bytes.begin() + static_cast<std::ptrdiff_t>(central + 46 + nameLength));
        require(u32(bytes, local) == 0x04034b50U, "ZIP local member");
        const std::size_t dataOffset = local + 30 + u16(bytes, local + 26) + u16(bytes, local + 28);
        result[name] = inflateRaw(bytes, dataOffset, compressed, uncompressed);
        central += 46 + nameLength + extraLength + commentLength;
    }
    return result;
}

std::size_t npyDataOffset(const std::vector<unsigned char>& bytes) {
    require(bytes.size() >= 10 && bytes[0] == 0x93 && std::string(bytes.begin() + 1, bytes.begin() + 6) == "NUMPY",
            "NPY magic");
    return 10U + u16(bytes, 8);
}

float f32(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return std::bit_cast<float>(u32(bytes, offset));
}

using ScalarType = luw::post::LegacyVtkScalarType;
using FieldKind = luw::post::LegacyVtkFieldKind;

struct TypedCase final {
    ScalarType type;
    std::string_view vtkToken;
    std::string_view descriptor;
    FieldKind kind;
    std::string_view label;
};

constexpr std::array<TypedCase, 8> typedCases{{
    {ScalarType::Int8, "char", "|i1", FieldKind::Scalars, "i1_scalars"},
    {ScalarType::UInt8, "unsigned_char", "|u1", FieldKind::Vectors, "u1_vectors"},
    {ScalarType::Int16, "short", "<i2", FieldKind::Field, "i2_field"},
    {ScalarType::UInt16, "unsigned_short", "<u2", FieldKind::Scalars, "u2_scalars"},
    {ScalarType::Int32, "int", "<i4", FieldKind::Vectors, "i4_vectors"},
    {ScalarType::UInt32, "unsigned_int", "<u4", FieldKind::Field, "u4_field"},
    {ScalarType::Float32, "float", "<f4", FieldKind::Scalars, "f4_scalars"},
    {ScalarType::Float64, "double", "<f8", FieldKind::Vectors, "f8_vectors"},
}};

double typedValue(ScalarType type, std::size_t scalar) {
    switch (type) {
    case ScalarType::Int8: return static_cast<double>(-120 + static_cast<int>(scalar));
    case ScalarType::UInt8: return static_cast<double>(100U + static_cast<unsigned>(scalar));
    case ScalarType::Int16: return static_cast<double>(-30000 + static_cast<int>(scalar) * 101);
    case ScalarType::UInt16: return static_cast<double>(40000U + static_cast<unsigned>(scalar) * 101U);
    case ScalarType::Int32:
        return scalar % 2U == 0U
            ? static_cast<double>(16777217 + static_cast<std::int32_t>(scalar) * 257)
            : static_cast<double>(-16777219 - static_cast<std::int32_t>(scalar) * 257);
    case ScalarType::UInt32:
        return static_cast<double>(4000000001U - static_cast<std::uint32_t>(scalar) * 65537U);
    case ScalarType::Float32:
        return static_cast<double>(static_cast<float>(-123.25 + static_cast<double>(scalar) * 0.375));
    case ScalarType::Float64:
        return 16777217.125 + static_cast<double>(scalar) * 0.0625;
    }
    throw std::runtime_error("unhandled typed fixture scalar");
}

template <typename Value>
void appendLittleEndian(std::vector<unsigned char>& bytes, Value value) {
    using Bits = std::conditional_t<sizeof(Value) == 1, std::uint8_t,
                 std::conditional_t<sizeof(Value) == 2, std::uint16_t,
                 std::conditional_t<sizeof(Value) == 4, std::uint32_t, std::uint64_t>>>;
    const Bits bits = std::bit_cast<Bits>(value);
    for (std::size_t index = 0; index < sizeof(Value); ++index) {
        bytes.push_back(static_cast<unsigned char>(bits >> (index * 8U)));
    }
}

void appendTypedLittleEndian(std::vector<unsigned char>& bytes, ScalarType type, double value) {
    switch (type) {
    case ScalarType::Int8: appendLittleEndian(bytes, static_cast<std::int8_t>(value)); return;
    case ScalarType::UInt8: appendLittleEndian(bytes, static_cast<std::uint8_t>(value)); return;
    case ScalarType::Int16: appendLittleEndian(bytes, static_cast<std::int16_t>(value)); return;
    case ScalarType::UInt16: appendLittleEndian(bytes, static_cast<std::uint16_t>(value)); return;
    case ScalarType::Int32: appendLittleEndian(bytes, static_cast<std::int32_t>(value)); return;
    case ScalarType::UInt32: appendLittleEndian(bytes, static_cast<std::uint32_t>(value)); return;
    case ScalarType::Float32: appendLittleEndian(bytes, static_cast<float>(value)); return;
    case ScalarType::Float64: appendLittleEndian(bytes, value); return;
    }
}

std::string typedText(ScalarType type, double value) {
    switch (type) {
    case ScalarType::Int8: return std::to_string(static_cast<int>(static_cast<std::int8_t>(value)));
    case ScalarType::UInt8: return std::to_string(static_cast<unsigned>(static_cast<std::uint8_t>(value)));
    case ScalarType::Int16: return std::to_string(static_cast<std::int16_t>(value));
    case ScalarType::UInt16: return std::to_string(static_cast<std::uint16_t>(value));
    case ScalarType::Int32: return std::to_string(static_cast<std::int32_t>(value));
    case ScalarType::UInt32: return std::to_string(static_cast<std::uint32_t>(value));
    case ScalarType::Float32: {
        std::ostringstream text;
        text << std::setprecision(std::numeric_limits<float>::max_digits10)
             << static_cast<float>(value);
        return text.str();
    }
    case ScalarType::Float64: {
        std::ostringstream text;
        text << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        return text.str();
    }
    }
    throw std::runtime_error("unhandled typed fixture scalar");
}

void writeTypedVtk(const fs::path& path, const TypedCase& testCase, bool ascii) {
    constexpr std::size_t points = 36;
    std::ofstream output(path, std::ios::binary);
    output << "# vtk DataFile Version 3.0\nvisluw typed generated fixture\n"
           << (ascii ? "ASCII\n" : "BINARY\n")
           << "DATASET STRUCTURED_POINTS\nDIMENSIONS 3 3 4\nORIGIN 0 0 -10\n"
              "SPACING 100 100 25\nPOINT_DATA 36\n";
    switch (testCase.kind) {
    case FieldKind::Scalars:
        output << "SCALARS U " << testCase.vtkToken << " 3\nLOOKUP_TABLE default\n";
        break;
    case FieldKind::Vectors:
        output << "VECTORS U " << testCase.vtkToken << '\n';
        break;
    case FieldKind::Field:
        output << "FIELD FieldData 1\nU 3 36 " << testCase.vtkToken << '\n';
        break;
    }
    for (std::size_t point = 0; point < points; ++point) {
        if (ascii) {
            for (std::size_t component = 0; component < 3; ++component) {
                if (component != 0) output << ' ';
                output << typedText(testCase.type, typedValue(testCase.type, point * 3U + component));
            }
            output << '\n';
            continue;
        }
        for (std::size_t component = 0; component < 3; ++component) {
            std::vector<unsigned char> scalar;
            appendTypedLittleEndian(scalar, testCase.type,
                                    typedValue(testCase.type, point * 3U + component));
            std::reverse(scalar.begin(), scalar.end());
            output.write(reinterpret_cast<const char*>(scalar.data()),
                         static_cast<std::streamsize>(scalar.size()));
        }
    }
    if (!ascii) output << '\n';
    require(static_cast<bool>(output), "write typed VTK fixture");
}

std::vector<unsigned char> expectedTypedCrop(const TypedCase& testCase) {
    std::vector<unsigned char> expected;
    for (std::size_t x = 0; x < 2; ++x) {
        for (std::size_t y = 0; y < 2; ++y) {
            for (std::size_t z = 2; z < 4; ++z) {
                const std::size_t point = z * 9U + y * 3U + x;
                for (std::size_t component = 0; component < 3; ++component) {
                    appendTypedLittleEndian(expected, testCase.type,
                        typedValue(testCase.type, point * 3U + component));
                }
            }
        }
    }
    return expected;
}

std::vector<float> vectors(std::size_t points, float base) {
    std::vector<float> values(points * 3U);
    for (std::size_t point = 0; point < points; ++point) {
        values[point * 3U] = base + static_cast<float>(point);
        values[point * 3U + 1] = base + 100.0F + static_cast<float>(point);
        values[point * 3U + 2] = base + 200.0F + static_cast<float>(point);
    }
    return values;
}

void writeBinary(const fs::path& path, float base) {
    constexpr std::array<int, 3> dimensions{3, 3, 4};
    constexpr std::size_t points = 36;
    const auto rho = vectors(points, 9000.0F);
    const auto fallback = vectors(points, 5000.0F);
    const auto selected = vectors(points, base);
    const std::array<luw::post::LegacyVtkFloatFieldView, 3> fields{{
        {"data_rho", 3, luw::post::LegacyVtkFieldKind::Vectors, rho},
        {"fallback", 3, luw::post::LegacyVtkFieldKind::Vectors, fallback},
        {"Data", 3, luw::post::LegacyVtkFieldKind::Vectors, selected},
    }};
    luw::post::writeLegacyStructuredPointsFloat(
        path, dimensions, {0.0, 0.0, -10.0}, {100.0, 100.0, 25.0}, fields,
        "visluw binary generated fixture");
}

void writeAscii(const fs::path& path, float base) {
    constexpr std::size_t points = 36;
    const auto rho = vectors(points, 9000.0F);
    const auto fallback = vectors(points, 5000.0F);
    const auto selected = vectors(points, base);
    std::ofstream output(path, std::ios::binary);
    output << "# vtk DataFile Version 3.0\nvisluw ASCII generated fixture\nASCII\n"
              "DATASET STRUCTURED_POINTS\nDIMENSIONS 3 3 4\nORIGIN 0 0 -10\nSPACING 100 100 25\n"
              "POINT_DATA 36\n";
    auto field = [&](std::string_view name, const std::vector<float>& values) {
        output << "VECTORS " << name << " float\n";
        for (std::size_t point = 0; point < points; ++point) {
            output << values[point * 3U] << ' ' << values[point * 3U + 1] << ' '
                   << values[point * 3U + 2] << '\n';
        }
    };
    field("data_rho", rho); field("fallback", fallback); field("Data", selected);
    require(static_cast<bool>(output), "write ASCII fixture");
}

void writeConfig(const fs::path& path, std::string_view casename, std::string_view datetime) {
    std::ofstream output(path);
    output << "casename = " << casename << '\n'
           << "datetime = " << datetime << '\n'
           << "cut_lon_manual = [120.0, 120.0012]\n"
           << "cut_lat_manual = [30.0, 30.0012]\n"
           << "center_lon = 120.0\ncenter_lat = 30.0\nutm_epsg = 32651\n";
}

void ncCheck(int status, std::string_view context) {
    if (status != NC_NOERR) throw std::runtime_error(std::string(context) + ": " + nc_strerror(status));
}

void verifyPng(const fs::path& path, std::uint32_t width, std::uint32_t height) {
    const auto bytes = readBytes(path);
    require(bytes.size() > 24 && bytes[0] == 0x89 && bytes[1] == 'P' && bytes[2] == 'N' && bytes[3] == 'G',
            "PNG signature");
    auto be32 = [&](std::size_t offset) {
        return (static_cast<std::uint32_t>(bytes[offset]) << 24U)
            | (static_cast<std::uint32_t>(bytes[offset + 1]) << 16U)
            | (static_cast<std::uint32_t>(bytes[offset + 2]) << 8U)
            | static_cast<std::uint32_t>(bytes[offset + 3]);
    };
    require(be32(16) == width && be32(20) == height, "PNG dimensions");
}

void verifyNetcdf(const fs::path& path, float expectedFirst) {
    int ncid = -1; ncCheck(nc_open(utf8(path).c_str(), NC_NOWRITE, &ncid), "open NetCDF");
    int dimH = -1, dimY = -1, dimX = -1;
    ncCheck(nc_inq_dimid(ncid, "height", &dimH), "height dimension");
    ncCheck(nc_inq_dimid(ncid, "y", &dimY), "y dimension");
    ncCheck(nc_inq_dimid(ncid, "x", &dimX), "x dimension");
    std::size_t size = 0; ncCheck(nc_inq_dimlen(ncid, dimH, &size), "height length"); require(size == 2, "two heights");
    ncCheck(nc_inq_dimlen(ncid, dimY, &size), "y length"); require(size == 2, "two y points");
    ncCheck(nc_inq_dimlen(ncid, dimX, &size), "x length"); require(size == 2, "two x points");
    int varU = -1, varLon = -1, varHeight = -1;
    ncCheck(nc_inq_varid(ncid, "u_grid", &varU), "u_grid variable");
    ncCheck(nc_inq_varid(ncid, "lon", &varLon), "lon variable");
    ncCheck(nc_inq_varid(ncid, "height", &varHeight), "height variable");
    int shuffle = 0, deflate = 0, level = 0;
    ncCheck(nc_inq_var_deflate(ncid, varU, &shuffle, &deflate, &level), "u_grid deflate");
    require(shuffle == 1 && deflate == 1 && level == 4, "NetCDF compression level 4");
    int attribute = -1;
    require(nc_inq_attid(ncid, varU, "_FillValue", &attribute) == NC_ENOTATT,
            "legacy NetCDF wind variable has no explicit _FillValue");
    nc_type epsgType = NC_NAT;
    ncCheck(nc_inq_atttype(ncid, NC_GLOBAL, "crs_epsg", &epsgType), "crs_epsg attribute type");
    require(epsgType == NC_INT64, "Python integer global attribute is int64");
    float first = 0.0F; std::array<std::size_t, 3> index{0,0,0};
    ncCheck(nc_get_var1_float(ncid, varU, index.data(), &first), "first u_grid");
    requireNear(first, expectedFirst, 0.0, "selected Data array and cropped first value");
    double lon = 0.0; std::array<std::size_t, 2> index2{0,0};
    ncCheck(nc_get_var1_double(ncid, varLon, index2.data(), &lon), "first lon");
    requireNear(lon, 120.0, 2.0e-6, "mapped southwest longitude");
    std::array<float, 2> heights{}; ncCheck(nc_get_var_float(ncid, varHeight, heights.data()), "heights");
    requireNear(heights[0], 0.0, 0.0, "zmin+50 is zero height");
    requireNear(heights[1], 25.0, 0.0, "next height");
    nc_close(ncid);
}

void verifyTypedNetcdf(const fs::path& path, const TypedCase& testCase) {
    int ncid = -1;
    ncCheck(nc_open(utf8(path).c_str(), NC_NOWRITE, &ncid), "open typed NetCDF");
    const std::array<std::string_view, 3> names{"u_grid", "v_grid", "w"};
    for (std::size_t component = 0; component < names.size(); ++component) {
        int variable = -1;
        ncCheck(nc_inq_varid(ncid, std::string(names[component]).c_str(), &variable),
                "typed wind variable");
        nc_type type = NC_NAT;
        ncCheck(nc_inq_vartype(ncid, variable, &type), "typed NetCDF variable type");
        require(type == NC_FLOAT, "typed NetCDF wind remains explicit float32");
        std::array<float, 8> actual{};
        ncCheck(nc_get_var_float(ncid, variable, actual.data()), "typed NetCDF values");
        std::size_t index = 0;
        for (std::size_t z = 2; z < 4; ++z) {
            for (std::size_t y = 0; y < 2; ++y) {
                for (std::size_t x = 0; x < 2; ++x) {
                    const std::size_t point = z * 9U + y * 3U + x;
                    const float expected = static_cast<float>(
                        typedValue(testCase.type, point * 3U + component));
                    requireNear(actual[index++], expected, 0.0,
                                "typed NetCDF value is NumPy float32 conversion");
                }
            }
        }
    }
    nc_close(ncid);
}

void runCase(const fs::path& root, bool ascii) {
    const std::string name = ascii ? "ascii" : "binary";
    const fs::path project = root / name;
    const fs::path results = project / "RESULTS";
    const fs::path exported = project / "retained_output";
    fs::create_directories(results);
    const fs::path config = project / "conf.luw";
    writeConfig(config, name, "20260719120000");
    const fs::path uvw = results / (ascii
        ? "uvwrho-" + name + "_20260719120000-fallback.vtk"
        : "uvw-" + name + "_20260719120000.vtk");
    if (ascii) writeAscii(uvw, 200.0F); else writeBinary(uvw, 100.0F);
    if (!ascii) writeBinary(results / ("uvwrho-" + name + "_20260719120000.vtk"), 1000.0F);
    const fs::path nc = exported / "explicit.nc";
    std::ostringstream output, error;
    const int status = luw::cli::run_visluw({
        utf8(config), "--lon-min", "120.0", "--lon-max", "120.0012",
        "--lat-min", "30.0", "--lat-max", "30.0012", "--layers", "2",
        "--output-dir", utf8(exported), "--export-nc", "--nc-output", utf8(nc)}, output, error);
    require(status == 0, "visluw succeeds: " + error.str() + output.str());
    require(output.str().find("[INFO] Completed successfully.") != std::string::npos, "completion message");
    require(output.str().find(uvw.filename().string()) != std::string::npos, "uvw exact file has priority");
    const fs::path npz = exported / uvw.stem().concat(".npz");
    const auto members = readNpzMembers(npz);
    require(members.size() == 2 && members.contains("wind_data.npy") && members.contains("grid_info.npy"),
            "NPZ key set");
    const auto& wind = members.at("wind_data.npy");
    const std::size_t dataOffset = npyDataOffset(wind);
    const std::string header(wind.begin() + 10, wind.begin() + static_cast<std::ptrdiff_t>(dataOffset));
    require(header.find("'<f4'") != std::string::npos, "wind_data float32 dtype");
    require(header.find("(2, 2, 2, 3)") != std::string::npos, "wind_data shape");
    requireNear(f32(wind, dataOffset), (ascii ? 218.0 : 118.0), 0.0, "NPZ first selected value");
    const auto& info = members.at("grid_info.npy");
    const std::size_t infoOffset = npyDataOffset(info);
    const std::string pickle(info.begin() + static_cast<std::ptrdiff_t>(infoOffset), info.end());
    require(pickle.find("x_coords_vtk") != std::string::npos && pickle.find("vector_name") != std::string::npos,
            "grid_info object contains compatibility keys");
    require(fs::file_size(npz) < wind.size() + info.size(), "NPZ uses effective compression");
    verifyPng(exported / "sections" / "wind_0m.png", 1053, 771);
    verifyPng(exported / "sections" / "wind_speed_distribution.png", 1268, 817);
    verifyNetcdf(nc, ascii ? 218.0F : 118.0F);
}

void runTypedCase(const fs::path& root, const TypedCase& testCase, bool ascii) {
    const std::string name = std::string(ascii ? "ascii_" : "binary_")
        + std::string(testCase.label);
    const fs::path project = root / name;
    const fs::path results = project / "RESULTS";
    const fs::path exported = project / "retained_output";
    fs::create_directories(results);
    const fs::path config = project / "conf.luw";
    writeConfig(config, name, "20260719120000");
    const fs::path vtk = results / ("uvw-" + name + "_20260719120000.vtk");
    writeTypedVtk(vtk, testCase, ascii);
    const fs::path nc = exported / "explicit.nc";
    std::ostringstream output, error;
    const int status = luw::cli::run_visluw({
        utf8(config), "--lon-min", "120.0", "--lon-max", "120.0012",
        "--lat-min", "30.0", "--lat-max", "30.0012", "--layers", "1",
        "--output-dir", utf8(exported), "--export-nc", "--nc-output", utf8(nc)},
        output, error);
    require(status == 0, "typed visluw succeeds for " + name + ": " + error.str() + output.str());
    require(output.str().find("[INFO] Completed successfully.") != std::string::npos,
            "typed completion message");

    const fs::path npz = exported / vtk.stem().concat(".npz");
    const auto members = readNpzMembers(npz);
    require(members.contains("wind_data.npy"), "typed NPZ wind_data member");
    const auto& wind = members.at("wind_data.npy");
    const std::size_t dataOffset = npyDataOffset(wind);
    const std::string header(wind.begin() + 10,
                             wind.begin() + static_cast<std::ptrdiff_t>(dataOffset));
    require(header.find("'" + std::string(testCase.descriptor) + "'") != std::string::npos,
            "typed NPZ descriptor for " + name);
    require(header.find("(2, 2, 2, 3)") != std::string::npos,
            "typed NPZ C-order shape for " + name);
    const std::vector<unsigned char> expected = expectedTypedCrop(testCase);
    require(wind.size() == dataOffset + expected.size(), "typed NPZ payload size for " + name);
    require(std::equal(expected.begin(), expected.end(), wind.begin()
                       + static_cast<std::ptrdiff_t>(dataOffset)),
            "typed NPZ retains every cropped scalar for " + name);
    verifyTypedNetcdf(nc, testCase);
    require(fs::is_regular_file(exported / "sections" / "wind_0m.png"),
            "typed section plot exists for " + name);
    require(fs::is_regular_file(exported / "sections" / "wind_speed_distribution.png"),
            "typed distribution plot exists for " + name);
}

} // namespace

int main() {
    try {
        {
            std::ostringstream output, error;
            require(luw::cli::run_visluw({"--help"}, output, error) == 0,
                    "visluw --help succeeds");
            require(output.str().find("Generate visualization figures from a LUW result.") != std::string::npos,
                    "visluw help description");
        }
        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const fs::path root = fs::path(LUW_VISLUW_ARTIFACT_ROOT) / ("run_" + std::to_string(nonce));
        fs::create_directories(root);
        runCase(root, false);
        runCase(root, true);
        for (const TypedCase& testCase : typedCases) {
            runTypedCase(root, testCase, false);
            runTypedCase(root, testCase, true);
        }
        std::cout << "visluw native compatibility tests passed; retained artifacts: " << utf8(root) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "visluw test failure: " << exception.what() << '\n';
        return 1;
    }
}
