#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace luw::post {

enum class LegacyVtkScalarType {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64,
};

enum class LegacyVtkFieldKind {
    Scalars,
    Vectors,
    Field,
};

enum class LegacyVtkAssociation {
    PointData,
    CellData,
};

enum class LegacyVtkEncoding {
    Ascii,
    Binary,
};

struct LegacyVtkField final {
    std::string name;
    std::string vtkTypeToken;
    LegacyVtkScalarType scalarType = LegacyVtkScalarType::Float32;
    LegacyVtkFieldKind kind = LegacyVtkFieldKind::Scalars;
    LegacyVtkAssociation association = LegacyVtkAssociation::PointData;
    std::size_t components = 1;
    std::size_t tuples = 0;
    std::uint64_t byteOffset = 0;

    // ASCII fields have variable-width values. During the metadata scan we
    // retain one seek position per point-data z plane so plane reads remain
    // bounded and do not repeatedly tokenize all preceding planes.
    std::vector<std::uint64_t> asciiPlaneOffsets;

    [[nodiscard]] std::size_t scalarSize() const noexcept;
};

struct LegacyStructuredPoints final {
    std::filesystem::path path;
    LegacyVtkEncoding encoding = LegacyVtkEncoding::Binary;
    std::array<int, 3> dimensions{};
    std::array<double, 3> origin{};
    std::array<double, 3> spacing{};
    std::size_t pointCount = 0;
    std::size_t cellCount = 0;
    std::vector<LegacyVtkField> fields;

    [[nodiscard]] static LegacyStructuredPoints readHeader(const std::filesystem::path& path);

    [[nodiscard]] const LegacyVtkField* field(std::string_view exactName) const noexcept;
    [[nodiscard]] const LegacyVtkField* fieldCaseInsensitive(std::string_view name) const noexcept;

    // Returns one z plane in VTK tuple-major/component-minor order. The result
    // has nx*ny*field.components values and is converted to native float32.
    [[nodiscard]] std::vector<float> readPointPlane(const LegacyVtkField& field,
                                                    std::size_t zIndex) const;

    // Returns one z plane without narrowing values. Each scalar is encoded in
    // native NumPy-compatible little-endian byte order; tuple/component order
    // is identical to readPointPlane(). This keeps large reads plane-bounded.
    [[nodiscard]] std::vector<std::uint8_t> readPointPlaneLittleEndian(
        const LegacyVtkField& field,
        std::size_t zIndex) const;

    // Returns one selected component of one z plane in (y,x) row-major order.
    [[nodiscard]] std::vector<float> readPointComponentPlane(const LegacyVtkField& field,
                                                             std::size_t zIndex,
                                                             std::size_t component) const;
};

// Reuses one open stream across many planes. Large avg/season files
// contain hundreds of planes and several fields, so this avoids thousands of
// filesystem open/close cycles while retaining bounded, plane-sized memory.
class LegacyVtkPlaneReader final {
public:
    explicit LegacyVtkPlaneReader(const LegacyStructuredPoints& metadata);

    LegacyVtkPlaneReader(const LegacyVtkPlaneReader&) = delete;
    LegacyVtkPlaneReader& operator=(const LegacyVtkPlaneReader&) = delete;

    [[nodiscard]] std::vector<float> readPointPlane(const LegacyVtkField& field,
                                                    std::size_t zIndex);
    [[nodiscard]] std::vector<std::uint8_t> readPointPlaneLittleEndian(
        const LegacyVtkField& field,
        std::size_t zIndex);
    [[nodiscard]] std::vector<float> readPointComponentPlane(const LegacyVtkField& field,
                                                             std::size_t zIndex,
                                                             std::size_t component);

private:
    const LegacyStructuredPoints* metadata_ = nullptr;
    std::ifstream input_;
};

// Converts one scalar produced by readPointPlaneLittleEndian() to float for
// calculation paths that intentionally use float32 (plots and NetCDF export).
[[nodiscard]] float legacyVtkLittleEndianScalarAsFloat(
    const std::uint8_t* source,
    LegacyVtkScalarType type) noexcept;

struct LegacyVtkFloatFieldView final {
    std::string name;
    std::size_t components = 1;
    LegacyVtkFieldKind kind = LegacyVtkFieldKind::Scalars;
    std::span<const float> values;
};

// Small, deterministic binary writer used by compatibility tests and by
// future native seasonal-average synthesis. Values are supplied in legacy VTK
// point order (x fastest, then y, then z) and written as big-endian float32.
void writeLegacyStructuredPointsFloat(
    const std::filesystem::path& path,
    const std::array<int, 3>& dimensions,
    const std::array<double, 3>& origin,
    const std::array<double, 3>& spacing,
    std::span<const LegacyVtkFloatFieldView> fields,
    std::string_view title = "LatticeUrbanWind native structured points");

} // namespace luw::post
