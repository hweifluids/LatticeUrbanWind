#include "luw/legacy_vtk.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace luw::post {

namespace {

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return text;
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return text;
}

std::vector<std::string> tokens(const std::string& line) {
    std::istringstream input(line);
    std::vector<std::string> result;
    for (std::string token; input >> token;) result.push_back(std::move(token));
    return result;
}

std::string readLine(std::ifstream& input) {
    std::string line;
    if (!std::getline(input, line)) return {};
    return trim(std::move(line));
}

template <typename Integer>
Integer parseInteger(const std::string& token, std::string_view context) {
    static_assert(std::is_integral_v<Integer>);
    Integer value{};
    const char* begin = token.data();
    const char* end = begin + token.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::runtime_error("Invalid integer '" + token + "' in " + std::string(context));
    }
    return value;
}

double parseDouble(const std::string& token, std::string_view context) {
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(token, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid number '" + token + "' in " + std::string(context));
    }
    if (consumed != token.size() || !std::isfinite(value)) {
        throw std::runtime_error("Invalid number '" + token + "' in " + std::string(context));
    }
    return value;
}

struct TypeInfo final {
    LegacyVtkScalarType type;
    std::size_t size;
};

std::optional<TypeInfo> vtkTypeInfo(std::string token) {
    token = lower(std::move(token));
    if (token == "char") return TypeInfo{LegacyVtkScalarType::Int8, 1};
    if (token == "unsigned_char") return TypeInfo{LegacyVtkScalarType::UInt8, 1};
    if (token == "short") return TypeInfo{LegacyVtkScalarType::Int16, 2};
    if (token == "unsigned_short") return TypeInfo{LegacyVtkScalarType::UInt16, 2};
    if (token == "int" || token == "long") return TypeInfo{LegacyVtkScalarType::Int32, 4};
    if (token == "unsigned_int" || token == "unsigned_long") {
        return TypeInfo{LegacyVtkScalarType::UInt32, 4};
    }
    if (token == "float") return TypeInfo{LegacyVtkScalarType::Float32, 4};
    if (token == "double") return TypeInfo{LegacyVtkScalarType::Float64, 8};
    return std::nullopt;
}

std::size_t checkedProduct(std::size_t left, std::size_t right, std::string_view context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::runtime_error("Size overflow while parsing " + std::string(context));
    }
    return left * right;
}

float decodeScalar(const unsigned char* source, LegacyVtkScalarType type) {
    switch (type) {
    case LegacyVtkScalarType::Int8:
        return static_cast<float>(static_cast<std::int8_t>(*source));
    case LegacyVtkScalarType::UInt8:
        return static_cast<float>(*source);
    case LegacyVtkScalarType::Int16:
        return static_cast<float>(static_cast<std::int16_t>(
            (static_cast<std::uint16_t>(source[0]) << 8U)
            | static_cast<std::uint16_t>(source[1])));
    case LegacyVtkScalarType::UInt16:
        return static_cast<float>((static_cast<std::uint16_t>(source[0]) << 8U)
                                  | static_cast<std::uint16_t>(source[1]));
    case LegacyVtkScalarType::Int32:
        return static_cast<float>(static_cast<std::int32_t>(
            (static_cast<std::uint32_t>(source[0]) << 24U)
            | (static_cast<std::uint32_t>(source[1]) << 16U)
            | (static_cast<std::uint32_t>(source[2]) << 8U)
            | static_cast<std::uint32_t>(source[3])));
    case LegacyVtkScalarType::UInt32:
        return static_cast<float>((static_cast<std::uint32_t>(source[0]) << 24U)
                                  | (static_cast<std::uint32_t>(source[1]) << 16U)
                                  | (static_cast<std::uint32_t>(source[2]) << 8U)
                                  | static_cast<std::uint32_t>(source[3]));
    case LegacyVtkScalarType::Float32: {
        const std::uint32_t bits = (static_cast<std::uint32_t>(source[0]) << 24U)
            | (static_cast<std::uint32_t>(source[1]) << 16U)
            | (static_cast<std::uint32_t>(source[2]) << 8U)
            | static_cast<std::uint32_t>(source[3]);
        return std::bit_cast<float>(bits);
    }
    case LegacyVtkScalarType::Float64: {
        std::uint64_t bits = 0;
        for (unsigned int index = 0; index < 8U; ++index) {
            bits = (bits << 8U) | static_cast<std::uint64_t>(source[index]);
        }
        return static_cast<float>(std::bit_cast<double>(bits));
    }
    }
    throw std::runtime_error("Unhandled legacy VTK scalar type");
}

template <typename Integer>
Integer parseAsciiInteger(const std::string& token,
                          std::string_view fieldName,
                          const std::filesystem::path& path) {
    static_assert(std::is_integral_v<Integer>);
    std::size_t consumed = 0;
    try {
        if constexpr (std::is_signed_v<Integer>) {
            const long long parsed = std::stoll(token, &consumed, 10);
            if (consumed != token.size()
                || parsed < static_cast<long long>(std::numeric_limits<Integer>::min())
                || parsed > static_cast<long long>(std::numeric_limits<Integer>::max())) {
                throw std::out_of_range("ASCII VTK integer is out of range");
            }
            return static_cast<Integer>(parsed);
        } else {
            if (!token.empty() && token.front() == '-') {
                throw std::out_of_range("ASCII VTK unsigned integer is negative");
            }
            const unsigned long long parsed = std::stoull(token, &consumed, 10);
            if (consumed != token.size()
                || parsed > static_cast<unsigned long long>(std::numeric_limits<Integer>::max())) {
                throw std::out_of_range("ASCII VTK integer is out of range");
            }
            return static_cast<Integer>(parsed);
        }
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid ASCII value '" + token + "' for field '"
                                 + std::string(fieldName) + "' in " + path.string());
    }
}

float decodeAsciiScalar(const std::string& token,
                        LegacyVtkScalarType type,
                        std::string_view fieldName,
                        const std::filesystem::path& path) {
    switch (type) {
    case LegacyVtkScalarType::Int8:
        return static_cast<float>(parseAsciiInteger<std::int8_t>(token, fieldName, path));
    case LegacyVtkScalarType::UInt8:
        return static_cast<float>(parseAsciiInteger<std::uint8_t>(token, fieldName, path));
    case LegacyVtkScalarType::Int16:
        return static_cast<float>(parseAsciiInteger<std::int16_t>(token, fieldName, path));
    case LegacyVtkScalarType::UInt16:
        return static_cast<float>(parseAsciiInteger<std::uint16_t>(token, fieldName, path));
    case LegacyVtkScalarType::Int32:
        return static_cast<float>(parseAsciiInteger<std::int32_t>(token, fieldName, path));
    case LegacyVtkScalarType::UInt32:
        return static_cast<float>(parseAsciiInteger<std::uint32_t>(token, fieldName, path));
    case LegacyVtkScalarType::Float32:
    case LegacyVtkScalarType::Float64: {
        std::size_t consumed = 0;
        try {
            const double value = std::stod(token, &consumed);
            if (consumed != token.size()) throw std::invalid_argument("trailing characters");
            return static_cast<float>(value);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid ASCII value '" + token + "' for field '"
                                     + std::string(fieldName) + "' in " + path.string());
        }
    }
    }
    throw std::runtime_error("Unhandled legacy VTK scalar type");
}

template <typename Value>
void encodeLittleEndianValue(unsigned char* destination, Value value) noexcept {
    using Bits = std::conditional_t<sizeof(Value) == 1, std::uint8_t,
                 std::conditional_t<sizeof(Value) == 2, std::uint16_t,
                 std::conditional_t<sizeof(Value) == 4, std::uint32_t, std::uint64_t>>>;
    const Bits bits = std::bit_cast<Bits>(value);
    for (std::size_t index = 0; index < sizeof(Value); ++index) {
        destination[index] = static_cast<unsigned char>(bits >> (index * 8U));
    }
}

void encodeAsciiScalarLittleEndian(const std::string& token,
                                   LegacyVtkScalarType type,
                                   std::string_view fieldName,
                                   const std::filesystem::path& path,
                                   unsigned char* destination) {
    switch (type) {
    case LegacyVtkScalarType::Int8:
        encodeLittleEndianValue(destination,
            parseAsciiInteger<std::int8_t>(token, fieldName, path));
        return;
    case LegacyVtkScalarType::UInt8:
        encodeLittleEndianValue(destination,
            parseAsciiInteger<std::uint8_t>(token, fieldName, path));
        return;
    case LegacyVtkScalarType::Int16:
        encodeLittleEndianValue(destination,
            parseAsciiInteger<std::int16_t>(token, fieldName, path));
        return;
    case LegacyVtkScalarType::UInt16:
        encodeLittleEndianValue(destination,
            parseAsciiInteger<std::uint16_t>(token, fieldName, path));
        return;
    case LegacyVtkScalarType::Int32:
        encodeLittleEndianValue(destination,
            parseAsciiInteger<std::int32_t>(token, fieldName, path));
        return;
    case LegacyVtkScalarType::UInt32:
        encodeLittleEndianValue(destination,
            parseAsciiInteger<std::uint32_t>(token, fieldName, path));
        return;
    case LegacyVtkScalarType::Float32:
    case LegacyVtkScalarType::Float64:
        break;
    }

    std::size_t consumed = 0;
    try {
        const double parsed = std::stod(token, &consumed);
        if (consumed != token.size()) throw std::invalid_argument("trailing characters");
        if (type == LegacyVtkScalarType::Float32) {
            encodeLittleEndianValue(destination, static_cast<float>(parsed));
        } else {
            encodeLittleEndianValue(destination, parsed);
        }
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid ASCII value '" + token + "' for field '"
                                 + std::string(fieldName) + "' in " + path.string());
    }
}

void writeBigEndianFloatBlock(std::ofstream& output, std::span<const float> values) {
    constexpr std::size_t chunkValues = 262144;
    std::vector<unsigned char> bytes;
    bytes.resize(std::min(chunkValues, values.size()) * sizeof(float));
    for (std::size_t offset = 0; offset < values.size(); offset += chunkValues) {
        const std::size_t count = std::min(chunkValues, values.size() - offset);
        bytes.resize(count * sizeof(float));
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(values[offset + index]);
            unsigned char* destination = bytes.data() + index * sizeof(float);
            destination[0] = static_cast<unsigned char>((bits >> 24U) & 0xffU);
            destination[1] = static_cast<unsigned char>((bits >> 16U) & 0xffU);
            destination[2] = static_cast<unsigned char>((bits >> 8U) & 0xffU);
            destination[3] = static_cast<unsigned char>(bits & 0xffU);
        }
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) throw std::runtime_error("Failed while writing legacy VTK binary values");
    }
}

std::vector<float> readPointPlaneFromStream(std::ifstream& input,
                                            const LegacyStructuredPoints& metadata,
                                            const LegacyVtkField& field,
                                            std::size_t zIndex) {
    if (field.association != LegacyVtkAssociation::PointData || field.tuples != metadata.pointCount) {
        throw std::runtime_error("Field '" + field.name + "' is not point data for the structured grid");
    }
    const std::size_t nz = static_cast<std::size_t>(metadata.dimensions[2]);
    if (zIndex >= nz) throw std::out_of_range("VTK z index is out of range");
    const std::size_t planeTuples = checkedProduct(static_cast<std::size_t>(metadata.dimensions[0]),
                                                   static_cast<std::size_t>(metadata.dimensions[1]), "VTK plane");
    const std::size_t planeValues = checkedProduct(planeTuples, field.components, "VTK plane");
    std::vector<float> result(planeValues);

    if (metadata.encoding == LegacyVtkEncoding::Ascii) {
        if (field.asciiPlaneOffsets.size() != nz) {
            throw std::runtime_error("Missing ASCII plane index for field '" + field.name
                                     + "' in " + metadata.path.string());
        }
        input.clear();
        input.seekg(static_cast<std::streamoff>(field.asciiPlaneOffsets[zIndex]), std::ios::beg);
        if (!input) throw std::runtime_error("Failed to seek in VTK file: " + metadata.path.string());
        for (std::size_t index = 0; index < planeValues; ++index) {
            std::string token;
            if (!(input >> token)) {
                throw std::runtime_error("Read failure for field '" + field.name
                                         + "' in " + metadata.path.string());
            }
            result[index] = decodeAsciiScalar(token, field.scalarType, field.name, metadata.path);
        }
        return result;
    }

    const std::size_t scalarBytes = field.scalarSize();
    const std::size_t planeBytes = checkedProduct(planeValues, scalarBytes, "VTK plane");
    const std::size_t precedingValues = checkedProduct(
        checkedProduct(zIndex, planeTuples, "VTK plane offset"), field.components, "VTK plane offset");
    const std::uint64_t offset = field.byteOffset
        + static_cast<std::uint64_t>(checkedProduct(precedingValues, scalarBytes, "VTK plane offset"));

    input.clear();
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) throw std::runtime_error("Failed to seek in VTK file: " + metadata.path.string());
    std::vector<unsigned char> bytes(planeBytes);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("Read failure for field '" + field.name + "' in " + metadata.path.string());
    }
    for (std::size_t index = 0; index < planeValues; ++index) {
        result[index] = decodeScalar(bytes.data() + index * scalarBytes, field.scalarType);
    }
    return result;
}

std::vector<std::uint8_t> readPointPlaneLittleEndianFromStream(
    std::ifstream& input,
    const LegacyStructuredPoints& metadata,
    const LegacyVtkField& field,
    std::size_t zIndex) {
    if (field.association != LegacyVtkAssociation::PointData || field.tuples != metadata.pointCount) {
        throw std::runtime_error("Field '" + field.name + "' is not point data for the structured grid");
    }
    const std::size_t nz = static_cast<std::size_t>(metadata.dimensions[2]);
    if (zIndex >= nz) throw std::out_of_range("VTK z index is out of range");
    const std::size_t planeTuples = checkedProduct(static_cast<std::size_t>(metadata.dimensions[0]),
                                                   static_cast<std::size_t>(metadata.dimensions[1]), "VTK plane");
    const std::size_t planeValues = checkedProduct(planeTuples, field.components, "VTK plane");
    const std::size_t scalarBytes = field.scalarSize();
    const std::size_t planeBytes = checkedProduct(planeValues, scalarBytes, "VTK plane");
    std::vector<std::uint8_t> result(planeBytes);

    if (metadata.encoding == LegacyVtkEncoding::Ascii) {
        if (field.asciiPlaneOffsets.size() != nz) {
            throw std::runtime_error("Missing ASCII plane index for field '" + field.name
                                     + "' in " + metadata.path.string());
        }
        input.clear();
        input.seekg(static_cast<std::streamoff>(field.asciiPlaneOffsets[zIndex]), std::ios::beg);
        if (!input) throw std::runtime_error("Failed to seek in VTK file: " + metadata.path.string());
        for (std::size_t index = 0; index < planeValues; ++index) {
            std::string token;
            if (!(input >> token)) {
                throw std::runtime_error("Read failure for field '" + field.name
                                         + "' in " + metadata.path.string());
            }
            encodeAsciiScalarLittleEndian(token, field.scalarType, field.name, metadata.path,
                                          result.data() + index * scalarBytes);
        }
        return result;
    }

    const std::size_t precedingValues = checkedProduct(
        checkedProduct(zIndex, planeTuples, "VTK plane offset"), field.components, "VTK plane offset");
    const std::uint64_t offset = field.byteOffset
        + static_cast<std::uint64_t>(checkedProduct(precedingValues, scalarBytes, "VTK plane offset"));
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) throw std::runtime_error("Failed to seek in VTK file: " + metadata.path.string());
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
    if (input.gcount() != static_cast<std::streamsize>(result.size())) {
        throw std::runtime_error("Read failure for field '" + field.name + "' in " + metadata.path.string());
    }
    if (scalarBytes > 1) {
        for (std::size_t index = 0; index < planeValues; ++index) {
            auto begin = result.begin() + static_cast<std::ptrdiff_t>(index * scalarBytes);
            std::reverse(begin, begin + static_cast<std::ptrdiff_t>(scalarBytes));
        }
    }
    return result;
}

} // namespace

std::size_t LegacyVtkField::scalarSize() const noexcept {
    switch (scalarType) {
    case LegacyVtkScalarType::Int8:
    case LegacyVtkScalarType::UInt8:
        return 1;
    case LegacyVtkScalarType::Int16:
    case LegacyVtkScalarType::UInt16:
        return 2;
    case LegacyVtkScalarType::Int32:
    case LegacyVtkScalarType::UInt32:
    case LegacyVtkScalarType::Float32:
        return 4;
    case LegacyVtkScalarType::Float64:
        return 8;
    }
    return 0;
}

LegacyStructuredPoints LegacyStructuredPoints::readHeader(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("VTK file not found: " + path.string());
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open VTK file: " + path.string());

    const std::string version = readLine(input);
    (void)readLine(input); // title
    const std::string encoding = upper(readLine(input));
    const std::string dataset = upper(readLine(input));
    if (lower(version).find("vtk datafile version") == std::string::npos) {
        throw std::runtime_error("Invalid VTK header in file: " + path.string());
    }
    if (encoding != "BINARY" && encoding != "ASCII") {
        throw std::runtime_error("Unsupported VTK encoding '" + encoding + "'. File: " + path.string());
    }
    if (dataset.find("DATASET STRUCTURED_POINTS") == std::string::npos) {
        throw std::runtime_error("Only DATASET STRUCTURED_POINTS is supported. File: " + path.string());
    }

    LegacyStructuredPoints result;
    result.path = std::filesystem::absolute(path).lexically_normal();
    result.encoding = encoding == "ASCII" ? LegacyVtkEncoding::Ascii : LegacyVtkEncoding::Binary;
    bool haveDimensions = false;
    bool haveOrigin = false;
    bool haveSpacing = false;
    bool haveDataAssociation = false;
    LegacyVtkAssociation activeAssociation = LegacyVtkAssociation::PointData;
    std::size_t activeTuples = 0;
    const std::uint64_t fileSize = std::filesystem::file_size(path);

    while (input) {
        const std::string line = readLine(input);
        if (!input && line.empty()) break;
        if (line.empty()) continue;
        const auto parts = tokens(line);
        if (parts.empty()) continue;
        const std::string key = upper(parts.front());

        if (key == "DIMENSIONS") {
            if (parts.size() < 4) throw std::runtime_error("Malformed DIMENSIONS in " + path.string());
            for (std::size_t axis = 0; axis < 3; ++axis) {
                result.dimensions[axis] = parseInteger<int>(parts[axis + 1], "DIMENSIONS");
                if (result.dimensions[axis] <= 0) {
                    throw std::runtime_error("Invalid dimensions in " + path.string());
                }
            }
            haveDimensions = true;
            continue;
        }
        if (key == "ORIGIN") {
            if (parts.size() < 4) throw std::runtime_error("Malformed ORIGIN in " + path.string());
            for (std::size_t axis = 0; axis < 3; ++axis) {
                result.origin[axis] = parseDouble(parts[axis + 1], "ORIGIN");
            }
            haveOrigin = true;
            continue;
        }
        if (key == "SPACING" || key == "ASPECT_RATIO") {
            if (parts.size() < 4) throw std::runtime_error("Malformed SPACING in " + path.string());
            for (std::size_t axis = 0; axis < 3; ++axis) {
                result.spacing[axis] = parseDouble(parts[axis + 1], "SPACING");
            }
            haveSpacing = true;
            continue;
        }
        if (key == "POINT_DATA" || key == "CELL_DATA") {
            if (parts.size() < 2) throw std::runtime_error("Malformed " + key + " in " + path.string());
            activeTuples = parseInteger<std::size_t>(parts[1], key);
            activeAssociation = key == "POINT_DATA"
                ? LegacyVtkAssociation::PointData
                : LegacyVtkAssociation::CellData;
            if (activeAssociation == LegacyVtkAssociation::PointData) result.pointCount = activeTuples;
            else result.cellCount = activeTuples;
            haveDataAssociation = true;
            continue;
        }
        if (!haveDataAssociation) continue;

        auto appendField = [&](std::string name,
                               std::string vtkType,
                               std::size_t components,
                               std::size_t tuplesCount,
                               LegacyVtkFieldKind kind) {
            const auto type = vtkTypeInfo(vtkType);
            if (!type) {
                throw std::runtime_error("Unsupported VTK scalar type '" + vtkType + "' in " + path.string());
            }
            if (components == 0) {
                throw std::runtime_error("Invalid component count for field '" + name + "' in " + path.string());
            }
            const auto offset = input.tellg();
            if (offset < 0) throw std::runtime_error("Failed to determine field offset in " + path.string());
            const std::size_t values = checkedProduct(tuplesCount, components, "VTK field");
            const auto byteOffset = static_cast<std::uint64_t>(offset);

            LegacyVtkField field;
            field.name = std::move(name);
            field.vtkTypeToken = lower(std::move(vtkType));
            field.scalarType = type->type;
            field.kind = kind;
            field.association = activeAssociation;
            field.components = components;
            field.tuples = tuplesCount;
            field.byteOffset = byteOffset;

            if (result.encoding == LegacyVtkEncoding::Binary) {
                const std::size_t bytes = checkedProduct(values, type->size, "VTK field");
                if (byteOffset > fileSize || bytes > fileSize - byteOffset) {
                    throw std::runtime_error("Field '" + field.name + "' exceeds file size in " + path.string());
                }
                input.seekg(static_cast<std::streamoff>(byteOffset + bytes), std::ios::beg);
                if (!input) throw std::runtime_error("Failed to seek past field in " + path.string());
            } else {
                std::size_t planeValues = 0;
                if (activeAssociation == LegacyVtkAssociation::PointData && haveDimensions) {
                    const std::size_t planeTuples = checkedProduct(
                        static_cast<std::size_t>(result.dimensions[0]),
                        static_cast<std::size_t>(result.dimensions[1]), "VTK ASCII plane");
                    planeValues = checkedProduct(planeTuples, components, "VTK ASCII plane");
                    const std::size_t expectedTuples = checkedProduct(
                        planeTuples, static_cast<std::size_t>(result.dimensions[2]),
                        "VTK ASCII point data");
                    if (tuplesCount == expectedTuples) {
                        field.asciiPlaneOffsets.reserve(static_cast<std::size_t>(result.dimensions[2]));
                    } else {
                        planeValues = 0;
                    }
                }

                for (std::size_t valueIndex = 0; valueIndex < values; ++valueIndex) {
                    if (planeValues != 0 && valueIndex % planeValues == 0) {
                        const auto planeOffset = input.tellg();
                        if (planeOffset < 0) {
                            throw std::runtime_error("Failed to index ASCII field '" + field.name
                                                     + "' in " + path.string());
                        }
                        field.asciiPlaneOffsets.push_back(static_cast<std::uint64_t>(planeOffset));
                    }
                    std::string token;
                    if (!(input >> token)) {
                        throw std::runtime_error("Field '" + field.name
                                                 + "' has too few ASCII values in " + path.string());
                    }
                    (void)decodeAsciiScalar(token, field.scalarType, field.name, path);
                }
            }
            result.fields.push_back(std::move(field));
        };

        if (key == "SCALARS") {
            if (parts.size() < 3) throw std::runtime_error("Malformed SCALARS header '" + line + "' in " + path.string());
            const std::size_t components = parts.size() >= 4
                ? parseInteger<std::size_t>(parts[3], "SCALARS") : 1;
            const std::string lookup = upper(readLine(input));
            if (!lookup.starts_with("LOOKUP_TABLE")) {
                throw std::runtime_error("Expected LOOKUP_TABLE after SCALARS '" + parts[1] + "' in " + path.string());
            }
            appendField(parts[1], parts[2], components, activeTuples, LegacyVtkFieldKind::Scalars);
            continue;
        }
        if (key == "VECTORS") {
            if (parts.size() < 3) throw std::runtime_error("Malformed VECTORS header '" + line + "' in " + path.string());
            appendField(parts[1], parts[2], 3, activeTuples, LegacyVtkFieldKind::Vectors);
            continue;
        }
        if (key == "FIELD") {
            if (parts.size() < 3) throw std::runtime_error("Malformed FIELD header '" + line + "' in " + path.string());
            const int arrayCount = parseInteger<int>(parts[2], "FIELD");
            if (arrayCount < 0) throw std::runtime_error("Negative FIELD array count in " + path.string());
            for (int arrayIndex = 0; arrayIndex < arrayCount; ++arrayIndex) {
                std::string arrayLine;
                do {
                    arrayLine = readLine(input);
                } while (input && arrayLine.empty());
                const auto arrayParts = tokens(arrayLine);
                if (arrayParts.size() < 4) {
                    throw std::runtime_error("Malformed FIELD array header '" + arrayLine + "' in " + path.string());
                }
                const std::size_t components = parseInteger<std::size_t>(arrayParts[1], "FIELD");
                const std::size_t tupleCount = parseInteger<std::size_t>(arrayParts[2], "FIELD");
                appendField(arrayParts[0], arrayParts[3], components, tupleCount, LegacyVtkFieldKind::Field);
            }
            continue;
        }

        throw std::runtime_error("Unsupported VTK data section header '" + line + "' in " + path.string());
    }

    if (!haveDimensions || !haveOrigin || !haveSpacing) {
        throw std::runtime_error("Missing DIMENSIONS/ORIGIN/SPACING metadata in " + path.string());
    }
    const std::size_t dimensionsProduct = checkedProduct(
        checkedProduct(static_cast<std::size_t>(result.dimensions[0]),
                       static_cast<std::size_t>(result.dimensions[1]), "DIMENSIONS"),
        static_cast<std::size_t>(result.dimensions[2]), "DIMENSIONS");
    if (result.pointCount == 0) result.pointCount = dimensionsProduct;
    if (result.pointCount != dimensionsProduct) {
        throw std::runtime_error("POINT_DATA mismatch in " + path.string()
                                 + ". DIMENSIONS product=" + std::to_string(dimensionsProduct)
                                 + ", POINT_DATA=" + std::to_string(result.pointCount));
    }
    if (result.fields.empty()) {
        throw std::runtime_error("No point/cell fields found in VTK file: " + path.string());
    }
    return result;
}

const LegacyVtkField* LegacyStructuredPoints::field(std::string_view exactName) const noexcept {
    const auto found = std::find_if(fields.begin(), fields.end(), [&](const LegacyVtkField& candidate) {
        return candidate.name == exactName;
    });
    return found == fields.end() ? nullptr : &*found;
}

const LegacyVtkField* LegacyStructuredPoints::fieldCaseInsensitive(std::string_view name) const noexcept {
    const std::string needle = lower(std::string(name));
    const auto found = std::find_if(fields.begin(), fields.end(), [&](const LegacyVtkField& candidate) {
        return lower(candidate.name) == needle;
    });
    return found == fields.end() ? nullptr : &*found;
}

std::vector<float> LegacyStructuredPoints::readPointPlane(const LegacyVtkField& field,
                                                           std::size_t zIndex) const {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open VTK file: " + path.string());
    return readPointPlaneFromStream(input, *this, field, zIndex);
}

std::vector<std::uint8_t> LegacyStructuredPoints::readPointPlaneLittleEndian(
    const LegacyVtkField& field,
    std::size_t zIndex) const {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open VTK file: " + path.string());
    return readPointPlaneLittleEndianFromStream(input, *this, field, zIndex);
}

std::vector<float> LegacyStructuredPoints::readPointComponentPlane(const LegacyVtkField& field,
                                                                    std::size_t zIndex,
                                                                    std::size_t component) const {
    if (component >= field.components) throw std::out_of_range("VTK field component is out of range");
    const std::vector<float> interleaved = readPointPlane(field, zIndex);
    const std::size_t tupleCount = interleaved.size() / field.components;
    std::vector<float> result(tupleCount);
    for (std::size_t index = 0; index < tupleCount; ++index) {
        result[index] = interleaved[index * field.components + component];
    }
    return result;
}

LegacyVtkPlaneReader::LegacyVtkPlaneReader(const LegacyStructuredPoints& metadata)
    : metadata_(&metadata), input_(metadata.path, std::ios::binary) {
    if (!input_) throw std::runtime_error("Failed to open VTK file: " + metadata.path.string());
}

std::vector<float> LegacyVtkPlaneReader::readPointPlane(const LegacyVtkField& field,
                                                        std::size_t zIndex) {
    return readPointPlaneFromStream(input_, *metadata_, field, zIndex);
}

std::vector<std::uint8_t> LegacyVtkPlaneReader::readPointPlaneLittleEndian(
    const LegacyVtkField& field,
    std::size_t zIndex) {
    return readPointPlaneLittleEndianFromStream(input_, *metadata_, field, zIndex);
}

std::vector<float> LegacyVtkPlaneReader::readPointComponentPlane(const LegacyVtkField& field,
                                                                 std::size_t zIndex,
                                                                 std::size_t component) {
    if (component >= field.components) throw std::out_of_range("VTK field component is out of range");
    const std::vector<float> interleaved = readPointPlane(field, zIndex);
    const std::size_t tupleCount = interleaved.size() / field.components;
    std::vector<float> result(tupleCount);
    for (std::size_t index = 0; index < tupleCount; ++index) {
        result[index] = interleaved[index * field.components + component];
    }
    return result;
}

float legacyVtkLittleEndianScalarAsFloat(const std::uint8_t* source,
                                         LegacyVtkScalarType type) noexcept {
    const auto bits16 = [&]() noexcept -> std::uint16_t {
        return static_cast<std::uint16_t>(static_cast<std::uint16_t>(source[0])
            | (static_cast<std::uint16_t>(source[1]) << 8U));
    };
    const auto bits32 = [&]() noexcept {
        return static_cast<std::uint32_t>(source[0])
            | (static_cast<std::uint32_t>(source[1]) << 8U)
            | (static_cast<std::uint32_t>(source[2]) << 16U)
            | (static_cast<std::uint32_t>(source[3]) << 24U);
    };
    switch (type) {
    case LegacyVtkScalarType::Int8:
        return static_cast<float>(std::bit_cast<std::int8_t>(source[0]));
    case LegacyVtkScalarType::UInt8:
        return static_cast<float>(source[0]);
    case LegacyVtkScalarType::Int16:
        return static_cast<float>(std::bit_cast<std::int16_t>(bits16()));
    case LegacyVtkScalarType::UInt16:
        return static_cast<float>(bits16());
    case LegacyVtkScalarType::Int32:
        return static_cast<float>(std::bit_cast<std::int32_t>(bits32()));
    case LegacyVtkScalarType::UInt32:
        return static_cast<float>(bits32());
    case LegacyVtkScalarType::Float32:
        return std::bit_cast<float>(bits32());
    case LegacyVtkScalarType::Float64: {
        std::uint64_t bits = 0;
        for (std::size_t index = 0; index < 8; ++index) {
            bits |= static_cast<std::uint64_t>(source[index]) << (index * 8U);
        }
        return static_cast<float>(std::bit_cast<double>(bits));
    }
    }
    return 0.0F;
}

void writeLegacyStructuredPointsFloat(
    const std::filesystem::path& path,
    const std::array<int, 3>& dimensions,
    const std::array<double, 3>& origin,
    const std::array<double, 3>& spacing,
    std::span<const LegacyVtkFloatFieldView> fields,
    std::string_view title) {
    for (const int size : dimensions) {
        if (size <= 0) throw std::invalid_argument("Legacy VTK dimensions must be positive");
    }
    const std::size_t pointCount = checkedProduct(
        checkedProduct(static_cast<std::size_t>(dimensions[0]),
                       static_cast<std::size_t>(dimensions[1]), "VTK output dimensions"),
        static_cast<std::size_t>(dimensions[2]), "VTK output dimensions");
    if (fields.empty()) throw std::invalid_argument("At least one VTK output field is required");
    for (const auto& field : fields) {
        if (field.name.empty() || field.components == 0) {
            throw std::invalid_argument("VTK output field name/components are invalid");
        }
        if (field.kind == LegacyVtkFieldKind::Vectors && field.components != 3) {
            throw std::invalid_argument("A VECTORS field must contain exactly three components");
        }
        if (field.values.size() != checkedProduct(pointCount, field.components, "VTK output field")) {
            throw std::invalid_argument("VTK output field '" + field.name + "' has the wrong value count");
        }
    }

    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Failed to create VTK file: " + path.string());
    output << "# vtk DataFile Version 3.0\n"
           << title << "\n"
           << "BINARY\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS " << dimensions[0] << ' ' << dimensions[1] << ' ' << dimensions[2] << "\n"
           << std::setprecision(17)
           << "ORIGIN " << origin[0] << ' ' << origin[1] << ' ' << origin[2] << "\n"
           << "SPACING " << spacing[0] << ' ' << spacing[1] << ' ' << spacing[2] << "\n\n"
           << "POINT_DATA " << pointCount << "\n";

    for (const auto& field : fields) {
        if (field.kind == LegacyVtkFieldKind::Vectors) {
            output << "VECTORS " << field.name << " float\n";
        } else if (field.kind == LegacyVtkFieldKind::Scalars) {
            output << "SCALARS " << field.name << " float";
            if (field.components != 1) output << ' ' << field.components;
            output << "\nLOOKUP_TABLE default\n";
        } else {
            output << "FIELD FieldData 1\n"
                   << field.name << ' ' << field.components << ' ' << pointCount << " float\n";
        }
        writeBigEndianFloatBlock(output, field.values);
        output << "\n";
    }
    output.flush();
    if (!output) throw std::runtime_error("Failed while finalizing VTK file: " + path.string());
}

} // namespace luw::post
