#include "physical_output.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <cstdlib>
#endif

#if CULBM_HAS_VTKHDF
#include <hdf5.h>
#endif

namespace culbm {
namespace {

constexpr std::uint32_t k_nc_dimension = 10u;
constexpr std::uint32_t k_nc_variable = 11u;
constexpr std::uint32_t k_nc_attribute = 12u;
constexpr std::uint32_t k_nc_char = 2u;
constexpr std::uint32_t k_nc_int = 4u;
constexpr std::uint32_t k_nc_float = 5u;
constexpr std::uint32_t k_nc_double = 6u;
constexpr std::uint64_t k_io_buffer_size = 1024ull * 1024ull;

bool set_error(std::string& error, const std::string& message) {
    error = message;
    return false;
}

bool native_is_little_endian() {
    const std::uint16_t value = 1u;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1u;
}

inline std::uint32_t bswap32(const std::uint32_t v) noexcept {
#if defined(_MSC_VER)
    return _byteswap_ulong(v);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(v);
#else
    return ((v & 0x000000FFu) << 24u) | ((v & 0x0000FF00u) << 8u) |
           ((v & 0x00FF0000u) >> 8u) | ((v & 0xFF000000u) >> 24u);
#endif
}

inline std::uint64_t bswap64(const std::uint64_t v) noexcept {
#if defined(_MSC_VER)
    return _byteswap_uint64(v);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#else
    return ((v & 0x00000000000000FFull) << 56u) | ((v & 0x000000000000FF00ull) << 40u) |
           ((v & 0x0000000000FF0000ull) << 24u) | ((v & 0x00000000FF000000ull) << 8u) |
           ((v & 0x000000FF00000000ull) >> 8u) | ((v & 0x0000FF0000000000ull) >> 24u) |
           ((v & 0x00FF000000000000ull) >> 40u) | ((v & 0xFF00000000000000ull) >> 56u);
#endif
}

inline std::uint32_t float_to_be32(const float f) noexcept {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &f, sizeof(bits));
    return native_is_little_endian() ? bswap32(bits) : bits;
}

inline std::uint32_t int32_to_be32(const std::int32_t v) noexcept {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &v, sizeof(bits));
    return native_is_little_endian() ? bswap32(bits) : bits;
}

inline std::uint64_t double_to_be64(const double d) noexcept {
    std::uint64_t bits = 0ull;
    std::memcpy(&bits, &d, sizeof(bits));
    return native_is_little_endian() ? bswap64(bits) : bits;
}

template<typename Fn>
void parallel_range(const std::uint64_t count, Fn&& fn) {
    if (count == 0ull) return;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads = count >= 65536ull ? std::min(hw, 16u) : 1u;
    if (threads <= 1u) {
        fn(0ull, count);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(threads - 1u);
    const std::uint64_t chunk = (count + threads - 1ull) / threads;
    for (unsigned t = 1u; t < threads; ++t) {
        const std::uint64_t begin = std::min(count, static_cast<std::uint64_t>(t) * chunk);
        const std::uint64_t end = std::min(count, begin + chunk);
        if (begin < end) {
            workers.emplace_back([begin, end, &fn]() { fn(begin, end); });
        }
    }
    fn(0ull, std::min(count, chunk));
    for (std::thread& w : workers) w.join();
}

constexpr std::uint64_t k_parallel_chunk_points = 4ull * 1024ull * 1024ull;

template<typename T>
void append_big_endian(std::vector<char>& out, const T value) {
    static_assert(std::is_trivially_copyable<T>::value, "binary writer requires trivially-copyable values");
    std::array<char, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    if (native_is_little_endian()) {
        for (auto it = bytes.rbegin(); it != bytes.rend(); ++it) out.push_back(*it);
    }
    else {
        out.insert(out.end(), bytes.begin(), bytes.end());
    }
}

template<typename T>
void append_big_endian(std::ostream& out, const T value) {
    std::array<char, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    if (native_is_little_endian()) std::reverse(bytes.begin(), bytes.end());
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

bool checked_product(const std::uint64_t a, const std::uint64_t b, std::uint64_t& out) {
    if (a != 0ull && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool checked_align4(const std::uint64_t value, std::uint64_t& aligned) {
    if (value > std::numeric_limits<std::uint64_t>::max() - 3ull) return false;
    aligned = (value + 3ull) & ~3ull;
    return true;
}

std::string filename_only(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1u);
}

class BufferedBinaryWriter {
public:
    explicit BufferedBinaryWriter(std::ostream& out)
        : out_(out) {
        buffer_.reserve(static_cast<std::size_t>(k_io_buffer_size));
    }

    ~BufferedBinaryWriter() { flush(); }

    bool write_bytes(const void* source, const std::size_t count) {
        if (count >= static_cast<std::size_t>(k_io_buffer_size)) {
            if (!flush()) return false;
            out_.write(static_cast<const char*>(source), static_cast<std::streamsize>(count));
            if (!out_) return false;
            bytes_written_ += static_cast<std::uint64_t>(count);
            return true;
        }
        const char* data = static_cast<const char*>(source);
        std::size_t remaining = count;
        while (remaining > 0u) {
            const std::size_t available = static_cast<std::size_t>(k_io_buffer_size) - buffer_.size();
            if (available == 0u && !flush()) return false;
            const std::size_t take = std::min(remaining, static_cast<std::size_t>(k_io_buffer_size) - buffer_.size());
            buffer_.insert(buffer_.end(), data, data + take);
            data += take;
            remaining -= take;
        }
        bytes_written_ += static_cast<std::uint64_t>(count);
        return true;
    }

    template<typename T>
    bool write_be(const T value) {
        std::array<char, sizeof(T)> bytes{};
        std::memcpy(bytes.data(), &value, sizeof(T));
        if (native_is_little_endian()) std::reverse(bytes.begin(), bytes.end());
        return write_bytes(bytes.data(), bytes.size());
    }

    bool write_zeros(const std::uint64_t count) {
        static const std::array<char, 4u> zeros{};
        for (std::uint64_t i = 0ull; i < count; ++i) {
            if (!write_bytes(zeros.data(), 1u)) return false;
        }
        return true;
    }

    bool flush() {
        if (buffer_.empty()) return static_cast<bool>(out_);
        out_.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
        buffer_.clear();
        return static_cast<bool>(out_);
    }

    std::uint64_t bytes_written() const noexcept { return bytes_written_; }

private:
    std::ostream& out_;
    std::vector<char> buffer_;
    std::uint64_t bytes_written_ = 0ull;
};

bool make_parent_directory(const std::string& path, std::string& error) {
    const std::filesystem::path output_path(path);
    const std::filesystem::path parent = output_path.parent_path();
    if (parent.empty()) return true;
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) return set_error(error, "could not create output directory '" + parent.string() + "': " + ec.message());
    return true;
}

std::uint64_t plane_size(const PhysicalGridView& grid) {
    return static_cast<std::uint64_t>(grid.nx) * static_cast<std::uint64_t>(grid.ny);
}

std::uint64_t source_index(const PhysicalGridView& grid,
                           const std::uint32_t x,
                           const std::uint32_t y,
                           const std::uint32_t z) {
    return static_cast<std::uint64_t>(x) +
        (static_cast<std::uint64_t>(y) + static_cast<std::uint64_t>(z) * grid.ny) * grid.nx;
}

double x_coordinate(const PhysicalGridView& grid, const std::uint32_t x) {
    return grid.origin_x + static_cast<double>(x) * grid.spacing_x;
}

double y_coordinate(const PhysicalGridView& grid, const std::uint32_t y) {
    return grid.origin_y + static_cast<double>(y) * grid.spacing_y;
}

double z_coordinate(const PhysicalGridView& grid, const std::uint32_t z) {
    return grid.origin_z + static_cast<double>(z) * grid.spacing_z;
}

inline bool has_temperature(const PhysicalGridView& grid) noexcept {
    return grid.temperature != nullptr || static_cast<bool>(grid.temperature_at);
}

inline float raw_velocity(const PhysicalGridView& grid,
                          const std::uint64_t index,
                          const std::uint32_t component) {
    if (grid.velocity_x != nullptr) {
        if (component == 0u) return grid.velocity_x[index];
        if (component == 1u) return grid.velocity_y[index];
        return grid.velocity_z[index];
    }
    if (grid.velocity_xyz != nullptr) return grid.velocity_xyz[3ull * index + component];
    return grid.velocity_at(index, component);
}

inline float raw_density(const PhysicalGridView& grid, const std::uint64_t index) {
    return grid.density != nullptr ? grid.density[index] : grid.density_at(index);
}

inline float raw_temperature(const PhysicalGridView& grid, const std::uint64_t index) {
    return grid.temperature != nullptr ? grid.temperature[index] : grid.temperature_at(index);
}

inline std::uint8_t raw_flags(const PhysicalGridView& grid, const std::uint64_t index) {
    return grid.flags != nullptr ? grid.flags[index] : grid.flags_at(index);
}

inline float scaled_velocity(const PhysicalGridView& grid,
                             const std::uint64_t index,
                             const std::uint32_t component) {
    return static_cast<float>(static_cast<double>(raw_velocity(grid, index, component)) * grid.velocity_scale);
}

inline float scaled_density(const PhysicalGridView& grid, const std::uint64_t index) {
    return static_cast<float>(static_cast<double>(raw_density(grid, index)) * grid.density_scale);
}

inline float scaled_temperature(const PhysicalGridView& grid, const std::uint64_t index) {
    return static_cast<float>(static_cast<double>(raw_temperature(grid, index)) * grid.temperature_scale + grid.temperature_offset);
}

struct AglPointIndex {
    std::vector<std::uint32_t> x;
    std::vector<std::uint32_t> y;
    std::vector<std::uint32_t> z;
    std::vector<std::uint64_t> src;

    std::uint64_t size() const noexcept { return static_cast<std::uint64_t>(src.size()); }
};

AglPointIndex build_agl_index(const PhysicalGridView& grid, const PhysicalAglSurfaceView& surface) {
    const std::uint32_t nz = grid.output_nz();
    const std::uint64_t xy_count = plane_size(grid);
    const double tolerance = std::max(1.0e-8, std::fabs(grid.spacing_z) * 1.0e-7);
    std::vector<std::uint64_t> plane_counts(nz, 0ull);
    parallel_range(nz, [&](const std::uint64_t z0, const std::uint64_t z1) {
        for (std::uint64_t oz = z0; oz < z1; ++oz) {
            const double z_val = z_coordinate(grid, grid.z_begin + static_cast<std::uint32_t>(oz));
            std::uint64_t kept = 0ull;
            for (std::uint64_t xy = 0ull; xy < xy_count; ++xy) {
                if (z_val > static_cast<double>(surface.ground_z[xy]) + tolerance) ++kept;
            }
            plane_counts[static_cast<std::size_t>(oz)] = kept;
        }
    });
    std::vector<std::uint64_t> offsets(nz + 1u, 0ull);
    for (std::uint32_t oz = 0u; oz < nz; ++oz) offsets[oz + 1u] = offsets[oz] + plane_counts[oz];
    const std::uint64_t total = offsets[nz];
    AglPointIndex idx;
    idx.x.resize(static_cast<std::size_t>(total));
    idx.y.resize(static_cast<std::size_t>(total));
    idx.z.resize(static_cast<std::size_t>(total));
    idx.src.resize(static_cast<std::size_t>(total));
    parallel_range(nz, [&](const std::uint64_t z0, const std::uint64_t z1) {
        for (std::uint64_t oz = z0; oz < z1; ++oz) {
            const std::uint32_t gz = grid.z_begin + static_cast<std::uint32_t>(oz);
            const double z_val = z_coordinate(grid, gz);
            const std::uint64_t z_base = static_cast<std::uint64_t>(gz) * xy_count;
            std::uint64_t dst = offsets[static_cast<std::size_t>(oz)];
            for (std::uint32_t y = 0u; y < grid.ny; ++y) {
                const std::uint64_t row = static_cast<std::uint64_t>(y) * grid.nx;
                for (std::uint32_t x = 0u; x < grid.nx; ++x) {
                    const std::uint64_t xy = row + x;
                    if (z_val > static_cast<double>(surface.ground_z[xy]) + tolerance) {
                        idx.x[static_cast<std::size_t>(dst)] = x;
                        idx.y[static_cast<std::size_t>(dst)] = y;
                        idx.z[static_cast<std::size_t>(dst)] = gz;
                        idx.src[static_cast<std::size_t>(dst)] = z_base + xy;
                        ++dst;
                    }
                }
            }
        }
    });
    return idx;
}

template<typename Elem, typename FillRange>
bool write_parallel_chunks(BufferedBinaryWriter& writer,
                           const std::uint64_t total_points,
                           const std::uint64_t elems_per_point,
                           FillRange&& fill_range) {
    if (total_points == 0ull) return true;
    const std::uint64_t chunk_pts = std::min(total_points, std::max<std::uint64_t>(1ull, k_parallel_chunk_points / elems_per_point));
    std::vector<Elem> buf(static_cast<std::size_t>(chunk_pts * elems_per_point));
    for (std::uint64_t base = 0ull; base < total_points; base += chunk_pts) {
        const std::uint64_t count = std::min(chunk_pts, total_points - base);
        parallel_range(count, [&](const std::uint64_t i0, const std::uint64_t i1) {
            fill_range(base, i0, i1, buf.data());
        });
        if (!writer.write_bytes(buf.data(), static_cast<std::size_t>(count * elems_per_point) * sizeof(Elem))) {
            return false;
        }
    }
    return true;
}

bool write_structured_vtk_velocity(BufferedBinaryWriter& writer, const PhysicalGridView& grid) {
    const std::uint64_t points = grid.output_point_count();
    const std::uint64_t start_idx = static_cast<std::uint64_t>(grid.z_begin) * plane_size(grid);
    return write_parallel_chunks<std::uint32_t>(writer, points, 3ull,
        [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
            for (std::uint64_t i = i0; i < i1; ++i) {
                const std::uint64_t idx = start_idx + base + i;
                out[3ull * i] = float_to_be32(scaled_velocity(grid, idx, 0u));
                out[3ull * i + 1ull] = float_to_be32(scaled_velocity(grid, idx, 1u));
                out[3ull * i + 2ull] = float_to_be32(scaled_velocity(grid, idx, 2u));
            }
        });
}

bool write_agl_vtk_velocity(BufferedBinaryWriter& writer, const PhysicalGridView& grid, const AglPointIndex& agl) {
    return write_parallel_chunks<std::uint32_t>(writer, agl.size(), 3ull,
        [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
            for (std::uint64_t i = i0; i < i1; ++i) {
                const std::uint64_t idx = agl.src[static_cast<std::size_t>(base + i)];
                out[3ull * i] = float_to_be32(scaled_velocity(grid, idx, 0u));
                out[3ull * i + 1ull] = float_to_be32(scaled_velocity(grid, idx, 1u));
                out[3ull * i + 2ull] = float_to_be32(scaled_velocity(grid, idx, 2u));
            }
        });
}

bool validate_surface(const PhysicalGridView& grid,
                      const PhysicalAglSurfaceView& surface,
                      std::string& error) {
    if (!grid.valid(&error)) return false;
    const std::uint64_t expected = plane_size(grid);
    if (!surface.ground_z || surface.ground_count != expected) {
        return set_error(error, "AGL output requires exactly one DEM height per x/y grid column");
    }
    for (std::uint64_t xy = 0ull; xy < expected; ++xy) {
        if (!std::isfinite(static_cast<double>(surface.ground_z[xy]))) {
            return set_error(error, "AGL output received a non-finite DEM height");
        }
    }
    return true;
}

struct NcDimension {
    std::string name;
    std::uint32_t length = 0u;
};

struct NcAttribute {
    std::string name;
    std::string value;
};

struct NcVariable {
    std::string name;
    std::vector<std::uint32_t> dimensions;
    std::uint32_t type = k_nc_float;
    std::uint64_t elements = 0ull;
    std::vector<NcAttribute> attributes;
    std::uint64_t begin = 0ull;
    std::function<bool(BufferedBinaryWriter&, std::string&)> write_values;
};

std::uint64_t nc_type_size(const std::uint32_t type) {
    switch (type) {
    case k_nc_int: return 4ull;
    case k_nc_float: return 4ull;
    case k_nc_double: return 8ull;
    default: return 0ull;
    }
}

bool nc_append_name(std::vector<char>& out, const std::string& name, std::string& error) {
    if (name.size() > std::numeric_limits<std::uint32_t>::max()) {
        return set_error(error, "NetCDF name is too long");
    }
    append_big_endian<std::uint32_t>(out, static_cast<std::uint32_t>(name.size()));
    out.insert(out.end(), name.begin(), name.end());
    const std::size_t padding = (4u - name.size() % 4u) % 4u;
    out.insert(out.end(), padding, '\0');
    return true;
}

bool nc_append_attributes(std::vector<char>& out,
                          const std::vector<NcAttribute>& attributes,
                          std::string& error) {
    if (attributes.empty()) {
        append_big_endian<std::uint32_t>(out, 0u);
        append_big_endian<std::uint32_t>(out, 0u);
        return true;
    }
    if (attributes.size() > std::numeric_limits<std::uint32_t>::max()) {
        return set_error(error, "too many NetCDF attributes");
    }
    append_big_endian<std::uint32_t>(out, k_nc_attribute);
    append_big_endian<std::uint32_t>(out, static_cast<std::uint32_t>(attributes.size()));
    for (const NcAttribute& attribute : attributes) {
        if (!nc_append_name(out, attribute.name, error)) return false;
        if (attribute.value.size() > std::numeric_limits<std::uint32_t>::max()) {
            return set_error(error, "NetCDF attribute value is too long");
        }
        append_big_endian<std::uint32_t>(out, k_nc_char);
        append_big_endian<std::uint32_t>(out, static_cast<std::uint32_t>(attribute.value.size()));
        out.insert(out.end(), attribute.value.begin(), attribute.value.end());
        const std::size_t padding = (4u - attribute.value.size() % 4u) % 4u;
        out.insert(out.end(), padding, '\0');
    }
    return true;
}

bool nc_build_header(const std::vector<NcDimension>& dimensions,
                     const std::vector<NcAttribute>& global_attributes,
                     const std::vector<NcVariable>& variables,
                     std::vector<char>& header,
                     std::string& error) {
    header.clear();
    header.reserve(1024u);
    header.push_back('C');
    header.push_back('D');
    header.push_back('F');
    header.push_back(static_cast<char>(2)); // CDF-2: 64-bit file offsets
    append_big_endian<std::uint32_t>(header, 0u); // no unlimited record dimension

    if (dimensions.empty() || dimensions.size() > std::numeric_limits<std::uint32_t>::max()) {
        return set_error(error, "NetCDF output requires a valid finite dimension list");
    }
    append_big_endian<std::uint32_t>(header, k_nc_dimension);
    append_big_endian<std::uint32_t>(header, static_cast<std::uint32_t>(dimensions.size()));
    for (const NcDimension& dimension : dimensions) {
        if (!nc_append_name(header, dimension.name, error)) return false;
        append_big_endian<std::uint32_t>(header, dimension.length);
    }
    if (!nc_append_attributes(header, global_attributes, error)) return false;

    if (variables.empty() || variables.size() > std::numeric_limits<std::uint32_t>::max()) {
        return set_error(error, "NetCDF output requires at least one variable");
    }
    append_big_endian<std::uint32_t>(header, k_nc_variable);
    append_big_endian<std::uint32_t>(header, static_cast<std::uint32_t>(variables.size()));
    for (const NcVariable& variable : variables) {
        if (!nc_append_name(header, variable.name, error)) return false;
        if (variable.dimensions.size() > std::numeric_limits<std::uint32_t>::max()) {
            return set_error(error, "NetCDF variable has too many dimensions");
        }
        append_big_endian<std::uint32_t>(header, static_cast<std::uint32_t>(variable.dimensions.size()));
        for (const std::uint32_t dimension : variable.dimensions) {
            if (dimension >= dimensions.size()) return set_error(error, "NetCDF variable refers to an invalid dimension");
            append_big_endian<std::uint32_t>(header, dimension);
        }
        if (!nc_append_attributes(header, variable.attributes, error)) return false;
        const std::uint64_t item_size = nc_type_size(variable.type);
        std::uint64_t raw_bytes = 0ull, padded_bytes = 0ull;
        if (item_size == 0ull || !checked_product(variable.elements, item_size, raw_bytes) ||
            !checked_align4(raw_bytes, padded_bytes) || padded_bytes > std::numeric_limits<std::uint32_t>::max()) {
            return set_error(error, "NetCDF-3 variable exceeds its 4 GiB per-variable limit");
        }
        append_big_endian<std::uint32_t>(header, variable.type);
        append_big_endian<std::uint32_t>(header, static_cast<std::uint32_t>(padded_bytes));
        append_big_endian<std::uint64_t>(header, variable.begin);
    }
    return true;
}

bool write_netcdf_file(const std::string& path,
                       const std::vector<NcDimension>& dimensions,
                       const std::vector<NcAttribute>& global_attributes,
                       std::vector<NcVariable> variables,
                       std::string& error) {
    std::vector<char> initial_header;
    if (!nc_build_header(dimensions, global_attributes, variables, initial_header, error)) return false;
    std::uint64_t offset = static_cast<std::uint64_t>(initial_header.size());
    for (NcVariable& variable : variables) {
        const std::uint64_t item_size = nc_type_size(variable.type);
        std::uint64_t raw_bytes = 0ull, padded_bytes = 0ull;
        if (!checked_product(variable.elements, item_size, raw_bytes) || !checked_align4(raw_bytes, padded_bytes) ||
            offset > std::numeric_limits<std::uint64_t>::max() - padded_bytes) {
            return set_error(error, "NetCDF output file size overflows 64-bit offsets");
        }
        variable.begin = offset;
        offset += padded_bytes;
    }
    std::vector<char> header;
    if (!nc_build_header(dimensions, global_attributes, variables, header, error)) return false;
    if (header.size() != initial_header.size()) return set_error(error, "NetCDF header changed size while assigning offsets");
    if (!make_parent_directory(path, error)) return false;
    std::ofstream file(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!file.is_open()) return set_error(error, "could not open '" + path + "' for writing");
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    BufferedBinaryWriter writer(file);
    for (const NcVariable& variable : variables) {
        const std::uint64_t before = writer.bytes_written();
        if (!variable.write_values || !variable.write_values(writer, error)) {
            if (error.empty()) error = "failed while writing NetCDF variable '" + variable.name + "'";
            return false;
        }
        const std::uint64_t item_size = nc_type_size(variable.type);
        std::uint64_t raw_bytes = 0ull, padded_bytes = 0ull;
        if (!checked_product(variable.elements, item_size, raw_bytes) || !checked_align4(raw_bytes, padded_bytes)) {
            return set_error(error, "invalid NetCDF variable size for '" + variable.name + "'");
        }
        if (writer.bytes_written() - before != raw_bytes) {
            return set_error(error, "NetCDF variable writer emitted an unexpected byte count for '" + variable.name + "'");
        }
        if (!writer.write_zeros(padded_bytes - raw_bytes)) return set_error(error, "failed while padding NetCDF variable data");
    }
    if (!writer.flush() || !file) return set_error(error, "failed while flushing '" + path + "'");
    return true;
}

std::vector<NcAttribute> common_global_attributes(const bool agl) {
    return {
        { "Conventions", "CF-1.10" },
        { "title", agl ? "CULBM DEM-referenced AGL point output" : "CULBM global metric structured output" },
        { "coordinate_reference", agl
            ? "x and y are STL metric coordinates; z_agl is height above interpolated_dem.csv"
            : "STL metric coordinates; z=0 is the lowest non-basement STL datum" }
    };
}

std::vector<NcAttribute> coordinate_attributes(const char* axis, const char* long_name) {
    return { { "units", "m" }, { "axis", axis }, { "long_name", long_name } };
}

std::vector<NcAttribute> z_attributes(const char* long_name) {
    return { { "units", "m" }, { "axis", "Z" }, { "positive", "up" }, { "long_name", long_name } };
}

std::vector<NcAttribute> velocity_attributes(const char* long_name) {
    return { { "units", "m s-1" }, { "long_name", long_name } };
}

std::vector<NcAttribute> density_attributes() {
    return { { "units", "kg m-3" }, { "long_name", "air density" } };
}

std::vector<NcAttribute> flags_attributes() {
    return { { "long_name", "CULBM cell flags; CELL_SOLID marks solid cells" } };
}

std::vector<NcVariable> make_structured_netcdf_variables(const PhysicalGridView& grid) {
    const std::uint32_t nz = grid.output_nz();
    const std::uint64_t points = grid.output_point_count();
    const std::vector<std::uint32_t> field_dimensions = { 2u, 1u, 0u }; // z, y, x
    std::vector<NcVariable> variables;
    variables.push_back(NcVariable{
        "x", { 0u }, k_nc_double, grid.nx, coordinate_attributes("X", "global x coordinate"), 0ull,
        [&grid](BufferedBinaryWriter& writer, std::string&) {
            for (std::uint32_t x = 0u; x < grid.nx; ++x) if (!writer.write_be<double>(x_coordinate(grid, x))) return false;
            return true;
        }
    });
    variables.push_back(NcVariable{
        "y", { 1u }, k_nc_double, grid.ny, coordinate_attributes("Y", "global y coordinate"), 0ull,
        [&grid](BufferedBinaryWriter& writer, std::string&) {
            for (std::uint32_t y = 0u; y < grid.ny; ++y) if (!writer.write_be<double>(y_coordinate(grid, y))) return false;
            return true;
        }
    });
    variables.push_back(NcVariable{
        "z", { 2u }, k_nc_double, nz, z_attributes("global z coordinate; non-basement STL datum"), 0ull,
        [&grid](BufferedBinaryWriter& writer, std::string&) {
            for (std::uint32_t z = grid.z_begin; z < grid.z_begin + grid.output_nz(); ++z) {
                if (!writer.write_be<double>(z_coordinate(grid, z))) return false;
            }
            return true;
        }
    });
    const std::uint64_t start_idx = static_cast<std::uint64_t>(grid.z_begin) * plane_size(grid);
    const auto add_velocity = [&](const char* name, const std::uint32_t component, const char* long_name) {
        variables.push_back(NcVariable{
            name, field_dimensions, k_nc_float, points, velocity_attributes(long_name), 0ull,
            [&grid, component, points, start_idx](BufferedBinaryWriter& writer, std::string&) {
                return write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                    [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                        for (std::uint64_t i = i0; i < i1; ++i)
                            out[i] = float_to_be32(scaled_velocity(grid, start_idx + base + i, component));
                    });
            }
        });
    };
    add_velocity("u", 0u, "x component of velocity");
    add_velocity("v", 1u, "y component of velocity");
    add_velocity("w", 2u, "z component of velocity");
    variables.push_back(NcVariable{
        "rho", field_dimensions, k_nc_float, points, density_attributes(), 0ull,
        [&grid, points, start_idx](BufferedBinaryWriter& writer, std::string&) {
            return write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                    for (std::uint64_t i = i0; i < i1; ++i)
                        out[i] = float_to_be32(scaled_density(grid, start_idx + base + i));
                });
        }
    });
    variables.push_back(NcVariable{
        "flags", field_dimensions, k_nc_int, points, flags_attributes(), 0ull,
        [&grid, points, start_idx](BufferedBinaryWriter& writer, std::string&) {
            return write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                    for (std::uint64_t i = i0; i < i1; ++i)
                        out[i] = int32_to_be32(static_cast<std::int32_t>(raw_flags(grid, start_idx + base + i)));
                });
        }
    });
    if (has_temperature(grid)) {
        variables.push_back(NcVariable{
            "temperature", field_dimensions, k_nc_float, points,
            { { "units", "K" }, { "long_name", "air temperature" } }, 0ull,
            [&grid, points, start_idx](BufferedBinaryWriter& writer, std::string&) {
                return write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                    [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                        for (std::uint64_t i = i0; i < i1; ++i)
                            out[i] = float_to_be32(scaled_temperature(grid, start_idx + base + i));
                    });
            }
        });
    }
    return variables;
}

std::vector<NcVariable> make_agl_netcdf_variables(const PhysicalGridView& grid,
                                                    const PhysicalAglSurfaceView& surface,
                                                    const AglPointIndex& agl) {
    const std::uint64_t points = agl.size();
    std::vector<NcVariable> variables;
    const auto point_coordinates = [&](const char* name, const char* long_name, const int coordinate) {
        variables.push_back(NcVariable{
            name, { 0u }, k_nc_double, points,
            coordinate == 2 ? z_attributes(long_name) : coordinate_attributes(coordinate == 0 ? "X" : "Y", long_name),
            0ull,
            [&grid, &surface, &agl, coordinate, points](BufferedBinaryWriter& writer, std::string&) {
                return write_parallel_chunks<std::uint64_t>(writer, points, 1ull,
                    [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint64_t* out) {
                        for (std::uint64_t i = i0; i < i1; ++i) {
                            const std::size_t p = static_cast<std::size_t>(base + i);
                            double value = 0.0;
                            if (coordinate == 0) value = x_coordinate(grid, agl.x[p]);
                            else if (coordinate == 1) value = y_coordinate(grid, agl.y[p]);
                            else {
                                const std::uint64_t xy = static_cast<std::uint64_t>(agl.x[p]) +
                                    static_cast<std::uint64_t>(agl.y[p]) * grid.nx;
                                value = z_coordinate(grid, agl.z[p]) - static_cast<double>(surface.ground_z[xy]);
                            }
                            out[i] = double_to_be64(value);
                        }
                    });
            }
        });
    };
    point_coordinates("x", "global x coordinate", 0);
    point_coordinates("y", "global y coordinate", 1);
    point_coordinates("z_agl", "height above DEM terrain", 2);
    variables.push_back(NcVariable{
        "ground_z", { 0u }, k_nc_double, points, z_attributes("DEM terrain height in global datum"), 0ull,
        [&grid, &surface, &agl, points](BufferedBinaryWriter& writer, std::string&) {
            return write_parallel_chunks<std::uint64_t>(writer, points, 1ull,
                [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint64_t* out) {
                    for (std::uint64_t i = i0; i < i1; ++i) {
                        const std::size_t p = static_cast<std::size_t>(base + i);
                        const std::uint64_t xy = static_cast<std::uint64_t>(agl.x[p]) +
                            static_cast<std::uint64_t>(agl.y[p]) * grid.nx;
                        out[i] = double_to_be64(static_cast<double>(surface.ground_z[xy]));
                    }
                });
        }
    });
    const auto add_velocity = [&](const char* name, const std::uint32_t component, const char* long_name) {
        variables.push_back(NcVariable{
            name, { 0u }, k_nc_float, points, velocity_attributes(long_name), 0ull,
            [&grid, &agl, component, points](BufferedBinaryWriter& writer, std::string&) {
                return write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                    [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                        for (std::uint64_t i = i0; i < i1; ++i)
                            out[i] = float_to_be32(scaled_velocity(grid, agl.src[static_cast<std::size_t>(base + i)], component));
                    });
            }
        });
    };
    add_velocity("u", 0u, "x component of velocity");
    add_velocity("v", 1u, "y component of velocity");
    add_velocity("w", 2u, "z component of velocity");
    variables.push_back(NcVariable{
        "rho", { 0u }, k_nc_float, points, density_attributes(), 0ull,
        [&grid, &agl, points](BufferedBinaryWriter& writer, std::string&) {
            return write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                    for (std::uint64_t i = i0; i < i1; ++i)
                        out[i] = float_to_be32(scaled_density(grid, agl.src[static_cast<std::size_t>(base + i)]));
                });
        }
    });
    variables.push_back(NcVariable{
        "flags", { 0u }, k_nc_int, points, flags_attributes(), 0ull,
        [&grid, &agl, points](BufferedBinaryWriter& writer, std::string&) {
            return write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                    for (std::uint64_t i = i0; i < i1; ++i)
                        out[i] = int32_to_be32(static_cast<std::int32_t>(raw_flags(grid, agl.src[static_cast<std::size_t>(base + i)])));
                });
        }
    });
    if (has_temperature(grid)) {
        variables.push_back(NcVariable{
            "temperature", { 0u }, k_nc_float, points,
            { { "units", "K" }, { "long_name", "air temperature" } }, 0ull,
            [&grid, &agl, points](BufferedBinaryWriter& writer, std::string&) {
                return write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                    [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                        for (std::uint64_t i = i0; i < i1; ++i)
                            out[i] = float_to_be32(scaled_temperature(grid, agl.src[static_cast<std::size_t>(base + i)]));
                    });
            }
        });
    }
    return variables;
}

#if CULBM_HAS_VTKHDF

constexpr unsigned k_vtkhdf_deflate_level = 4u;
constexpr hsize_t k_vtkhdf_chunk_edge = 128u;

class Hdf5Handle {
public:
    using CloseFunction = herr_t (*)(hid_t);

    Hdf5Handle() = default;
    Hdf5Handle(const hid_t value, CloseFunction close)
        : value_(value), close_(close) {}
    Hdf5Handle(const Hdf5Handle&) = delete;
    Hdf5Handle& operator=(const Hdf5Handle&) = delete;
    Hdf5Handle(Hdf5Handle&& other) noexcept
        : value_(other.value_), close_(other.close_) {
        other.value_ = H5I_INVALID_HID;
        other.close_ = nullptr;
    }
    Hdf5Handle& operator=(Hdf5Handle&& other) noexcept {
        if (this == &other) return *this;
        reset();
        value_ = other.value_;
        close_ = other.close_;
        other.value_ = H5I_INVALID_HID;
        other.close_ = nullptr;
        return *this;
    }
    ~Hdf5Handle() { reset(); }

    bool valid() const noexcept { return value_ != H5I_INVALID_HID; }
    hid_t get() const noexcept { return value_; }
    void reset() noexcept {
        if (valid() && close_) close_(value_);
        value_ = H5I_INVALID_HID;
        close_ = nullptr;
    }

private:
    hid_t value_ = H5I_INVALID_HID;
    CloseFunction close_ = nullptr;
};

bool write_hdf5_string_attribute(const hid_t object,
                                 const char* name,
                                 const std::string& value,
                                 std::string& error) {
    if (value.empty()) return set_error(error, "VTKHDF string attribute '" + std::string(name) + "' is empty");
    Hdf5Handle dataspace(H5Screate(H5S_SCALAR), H5Sclose);
    Hdf5Handle creation(H5Pcreate(H5P_ATTRIBUTE_CREATE), H5Pclose);
    Hdf5Handle type(H5Tcreate(H5T_STRING, value.size()), H5Tclose);
    if (!dataspace.valid() || !creation.valid() || !type.valid()) {
        return set_error(error, "could not allocate HDF5 metadata for attribute '" + std::string(name) + "'");
    }
    if (H5Pset_char_encoding(creation.get(), H5T_CSET_UTF8) < 0) {
        return set_error(error, "could not set UTF-8 encoding for VTKHDF attribute '" + std::string(name) + "'");
    }
    Hdf5Handle attribute(
        H5Acreate2(object, name, type.get(), dataspace.get(), creation.get(), H5P_DEFAULT), H5Aclose);
    if (!attribute.valid() || H5Awrite(attribute.get(), type.get(), value.c_str()) < 0) {
        return set_error(error, "could not write VTKHDF attribute '" + std::string(name) + "'");
    }
    return true;
}

bool write_hdf5_vector_attribute(const hid_t object,
                                 const char* name,
                                 const hid_t storage_type,
                                 const hid_t memory_type,
                                 const hsize_t count,
                                 const void* values,
                                 std::string& error) {
    Hdf5Handle dataspace(H5Screate_simple(1, &count, nullptr), H5Sclose);
    if (!dataspace.valid()) return set_error(error, "could not allocate HDF5 metadata for attribute '" + std::string(name) + "'");
    Hdf5Handle attribute(
        H5Acreate2(object, name, storage_type, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (!attribute.valid() || H5Awrite(attribute.get(), memory_type, values) < 0) {
        return set_error(error, "could not write VTKHDF attribute '" + std::string(name) + "'");
    }
    return true;
}

bool hdf5_deflate_encoder_available() {
    if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0) return false;
    unsigned int capabilities = 0u;
    return H5Zget_filter_info(H5Z_FILTER_DEFLATE, &capabilities) >= 0 &&
        (capabilities & H5Z_FILTER_CONFIG_ENCODE_ENABLED) != 0u;
}

Hdf5Handle create_hdf5_point_dataset(const hid_t group,
                                     const char* name,
                                     const hid_t storage_type,
                                     const hsize_t* dimensions,
                                     const int rank,
                                     const bool use_deflate,
                                     std::string& error) {
    Hdf5Handle dataspace(H5Screate_simple(rank, dimensions, nullptr), H5Sclose);
    Hdf5Handle creation(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
    if (!dataspace.valid() || !creation.valid()) {
        set_error(error, "could not allocate HDF5 storage for point array '" + std::string(name) + "'");
        return {};
    }
    std::array<hsize_t, 4u> chunks = { 1u, 1u, 1u, 1u };
    chunks[0] = 1u;
    chunks[1] = std::min(dimensions[1], k_vtkhdf_chunk_edge);
    chunks[2] = std::min(dimensions[2], k_vtkhdf_chunk_edge);
    if (rank == 4) chunks[3] = dimensions[3];
    if (H5Pset_chunk(creation.get(), rank, chunks.data()) < 0) {
        set_error(error, "could not configure chunking for VTKHDF point array '" + std::string(name) + "'");
        return {};
    }
    if (use_deflate &&
        (H5Pset_shuffle(creation.get()) < 0 || H5Pset_deflate(creation.get(), k_vtkhdf_deflate_level) < 0)) {
        set_error(error, "could not configure lossless compression for VTKHDF point array '" + std::string(name) + "'");
        return {};
    }
    Hdf5Handle dataset(
        H5Dcreate2(group, name, storage_type, dataspace.get(), H5P_DEFAULT, creation.get(), H5P_DEFAULT), H5Dclose);
    if (!dataset.valid()) {
        set_error(error, "could not create VTKHDF point array '" + std::string(name) + "'");
        return {};
    }
    return dataset;
}

template<typename Value, typename Sampler>
bool write_hdf5_point_data_array(const hid_t point_group,
                                 const char* name,
                                 const hid_t storage_type,
                                 const hid_t memory_type,
                                 const PhysicalGridView& grid,
                                 const std::uint32_t components,
                                 const bool use_deflate,
                                 const char* vtk_attribute,
                                 Sampler&& sampler,
                                 std::string& error) {
    const std::uint32_t output_nz = grid.output_nz();
    const int rank = components == 1u ? 3 : 4;
    std::array<hsize_t, 4u> dimensions = {
        static_cast<hsize_t>(output_nz), static_cast<hsize_t>(grid.ny), static_cast<hsize_t>(grid.nx),
        static_cast<hsize_t>(components)
    };
    Hdf5Handle dataset = create_hdf5_point_dataset(
        point_group, name, storage_type, dimensions.data(), rank, use_deflate, error);
    if (!dataset.valid()) return false;
    if (vtk_attribute && !write_hdf5_string_attribute(dataset.get(), "Attribute", vtk_attribute, error)) return false;

    Hdf5Handle file_space(H5Dget_space(dataset.get()), H5Sclose);
    if (!file_space.valid()) return set_error(error, "could not obtain HDF5 dataspace for point array '" + std::string(name) + "'");

    const std::uint32_t tile_y_max = std::min(
        grid.ny, static_cast<std::uint32_t>(k_vtkhdf_chunk_edge));
    const std::uint32_t tile_x_max = std::min(
        grid.nx, static_cast<std::uint32_t>(k_vtkhdf_chunk_edge));
    std::vector<Value> values;
    for (std::uint32_t output_z = 0u; output_z < output_nz; ++output_z) {
        const std::uint32_t source_z = grid.z_begin + output_z;
        for (std::uint32_t y0 = 0u; y0 < grid.ny; y0 += tile_y_max) {
            const std::uint32_t tile_y = std::min(tile_y_max, grid.ny - y0);
            for (std::uint32_t x0 = 0u; x0 < grid.nx; x0 += tile_x_max) {
                const std::uint32_t tile_x = std::min(tile_x_max, grid.nx - x0);
                const std::size_t values_count = static_cast<std::size_t>(tile_y) *
                    static_cast<std::size_t>(tile_x) * static_cast<std::size_t>(components);
                values.resize(values_count);
                std::size_t destination = 0u;
                for (std::uint32_t y = y0; y < y0 + tile_y; ++y) {
                    for (std::uint32_t x = x0; x < x0 + tile_x; ++x) {
                        const std::uint64_t index = source_index(grid, x, y, source_z);
                        for (std::uint32_t component = 0u; component < components; ++component) {
                            values[destination++] = sampler(index, component);
                        }
                    }
                }

                std::array<hsize_t, 4u> offsets = {
                    static_cast<hsize_t>(output_z), static_cast<hsize_t>(y0), static_cast<hsize_t>(x0), 0u
                };
                std::array<hsize_t, 4u> counts = {
                    1u, static_cast<hsize_t>(tile_y), static_cast<hsize_t>(tile_x), static_cast<hsize_t>(components)
                };
                Hdf5Handle memory_space(H5Screate_simple(rank, counts.data(), nullptr), H5Sclose);
                if (!memory_space.valid() ||
                    H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, offsets.data(), nullptr, counts.data(), nullptr) < 0 ||
                    H5Dwrite(dataset.get(), memory_type, memory_space.get(), file_space.get(), H5P_DEFAULT, values.data()) < 0) {
                    return set_error(error, "could not write VTKHDF point array '" + std::string(name) + "'");
                }
            }
        }
    }
    return true;
}

#endif

} // namespace

std::uint32_t PhysicalGridView::output_nz() const noexcept {
    if (z_begin >= nz) return 0u;
    const std::uint32_t available = nz - z_begin;
    return z_count == 0u ? available : std::min(z_count, available);
}

std::uint64_t PhysicalGridView::source_point_count() const noexcept {
    return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
}

std::uint64_t PhysicalGridView::output_point_count() const noexcept {
    return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(output_nz());
}

bool PhysicalGridView::valid(std::string* error) const {
    const auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (nx == 0u || ny == 0u || nz == 0u || output_nz() == 0u) return fail("physical output has an empty grid");
    const bool has_vel = (velocity_x != nullptr && velocity_y != nullptr && velocity_z != nullptr) ||
        velocity_xyz != nullptr || static_cast<bool>(velocity_at);
    const bool has_rho = density != nullptr || static_cast<bool>(density_at);
    const bool has_fl = flags != nullptr || static_cast<bool>(flags_at);
    if (!has_vel || !has_rho || !has_fl) return fail("physical output is missing a required field sampler");
    if (!(std::isfinite(origin_x) && std::isfinite(origin_y) && std::isfinite(origin_z) &&
          std::isfinite(spacing_x) && std::isfinite(spacing_y) && std::isfinite(spacing_z)) ||
        !(spacing_x > 0.0 && spacing_y > 0.0 && spacing_z > 0.0)) {
        return fail("physical output has invalid metric coordinates");
    }
    std::uint64_t xy = 0ull, xyz = 0ull;
    if (!checked_product(nx, ny, xy) || !checked_product(xy, nz, xyz)) return fail("physical output grid dimensions overflow");
    return true;
}

bool count_agl_points(const PhysicalGridView& grid,
                      const PhysicalAglSurfaceView& surface,
                      std::uint64_t& point_count,
                      std::string& error) {
    point_count = 0ull;
    if (!validate_surface(grid, surface, error)) return false;
    const std::uint32_t nz = grid.output_nz();
    const std::uint64_t xy_count = plane_size(grid);
    const double tolerance = std::max(1.0e-8, std::fabs(grid.spacing_z) * 1.0e-7);
    std::vector<std::uint64_t> counts(nz, 0ull);
    parallel_range(nz, [&](const std::uint64_t z0, const std::uint64_t z1) {
        for (std::uint64_t oz = z0; oz < z1; ++oz) {
            const double z_val = z_coordinate(grid, grid.z_begin + static_cast<std::uint32_t>(oz));
            std::uint64_t kept = 0ull;
            for (std::uint64_t xy = 0ull; xy < xy_count; ++xy) {
                if (z_val > static_cast<double>(surface.ground_z[xy]) + tolerance) ++kept;
            }
            counts[static_cast<std::size_t>(oz)] = kept;
        }
    });
    for (const std::uint64_t c : counts) point_count += c;
    return true;
}

bool write_global_structured_vtk(const std::string& path,
                                 const PhysicalGridView& grid,
                                 std::string& error) {
    if (!grid.valid(&error)) return false;
    if (!make_parent_directory(path, error)) return false;
    std::ofstream file(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!file.is_open()) return set_error(error, "could not open '" + path + "' for writing");
    const std::uint64_t points = grid.output_point_count();
    const std::uint64_t start_idx = static_cast<std::uint64_t>(grid.z_begin) * plane_size(grid);
    const double output_origin_z = z_coordinate(grid, grid.z_begin);
    file << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "# vtk DataFile Version 3.0\n"
         << "CULBM global metric structured output " << filename_only(path) << "\n"
         << "BINARY\nDATASET STRUCTURED_POINTS\n"
         << "DIMENSIONS " << grid.nx << " " << grid.ny << " " << grid.output_nz() << "\n"
         << "ORIGIN " << grid.origin_x << " " << grid.origin_y << " " << output_origin_z << "\n"
         << "SPACING " << grid.spacing_x << " " << grid.spacing_y << " " << grid.spacing_z << "\n"
         << "POINT_DATA " << points << "\n"
         << "VECTORS velocity float\n";
    BufferedBinaryWriter writer(file);
    if (!write_structured_vtk_velocity(writer, grid) || !writer.flush()) return set_error(error, "failed while writing VTK velocity data");
    file << "\nSCALARS rho float 1\nLOOKUP_TABLE default\n";
    if (!write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
            [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                for (std::uint64_t i = i0; i < i1; ++i) out[i] = float_to_be32(scaled_density(grid, start_idx + base + i));
            }) || !writer.flush()) return set_error(error, "failed while writing VTK density data");
    file << "\nSCALARS flags unsigned_char 1\nLOOKUP_TABLE default\n";
    if (grid.flags != nullptr) {
        if (!writer.write_bytes(grid.flags + start_idx, static_cast<std::size_t>(points)) || !writer.flush())
            return set_error(error, "failed while writing VTK flags data");
    } else {
        if (!write_parallel_chunks<std::uint8_t>(writer, points, 1ull,
                [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint8_t* out) {
                    for (std::uint64_t i = i0; i < i1; ++i) out[i] = raw_flags(grid, start_idx + base + i);
                }) || !writer.flush()) return set_error(error, "failed while writing VTK flags data");
    }
    if (has_temperature(grid)) {
        file << "\nSCALARS temperature float 1\nLOOKUP_TABLE default\n";
        if (!write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                    for (std::uint64_t i = i0; i < i1; ++i) out[i] = float_to_be32(scaled_temperature(grid, start_idx + base + i));
                }) || !writer.flush()) return set_error(error, "failed while writing VTK temperature data");
    }
    if (!file) return set_error(error, "failed while finalizing '" + path + "'");
    return true;
}

bool write_global_structured_netcdf(const std::string& path,
                                     const PhysicalGridView& grid,
                                     std::string& error) {
    if (!grid.valid(&error)) return false;
    const std::vector<NcDimension> dimensions = {
        { "x", grid.nx }, { "y", grid.ny }, { "z", grid.output_nz() }
    };
    return write_netcdf_file(path, dimensions, common_global_attributes(false),
                             make_structured_netcdf_variables(grid), error);
}

bool write_global_structured_vtkhdf(const std::string& path,
                                    const PhysicalGridView& grid,
                                    std::string& error) {
#if !CULBM_HAS_VTKHDF
    (void)path;
    (void)grid;
    return set_error(error, "VTKHDF output is unavailable because this CULBM build has no HDF5 support");
#else
    if (!grid.valid(&error)) return false;
    if (grid.nx > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        grid.ny > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        grid.output_nz() > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return set_error(error, "VTKHDF ImageData extent exceeds the 32-bit VTK compatibility limit");
    }
    if (!make_parent_directory(path, error)) return false;

    Hdf5Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    if (!file.valid()) return set_error(error, "could not create VTKHDF file '" + path + "'");
    Hdf5Handle root(H5Gcreate2(file.get(), "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    if (!root.valid()) return set_error(error, "could not create VTKHDF root group in '" + path + "'");

    const std::int64_t version[2] = { 2, 0 };
    const int whole_extent[6] = {
        0, static_cast<int>(grid.nx) - 1,
        0, static_cast<int>(grid.ny) - 1,
        0, static_cast<int>(grid.output_nz()) - 1
    };
    const double output_origin_z = z_coordinate(grid, grid.z_begin);
    const double origin[3] = { grid.origin_x, grid.origin_y, output_origin_z };
    const double spacing[3] = { grid.spacing_x, grid.spacing_y, grid.spacing_z };
    const double direction[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    const bool use_deflate = hdf5_deflate_encoder_available();
    if (!write_hdf5_string_attribute(root.get(), "Type", "ImageData", error) ||
        !write_hdf5_vector_attribute(root.get(), "Version", H5T_STD_I64LE, H5T_NATIVE_INT64, 2u, version, error) ||
        !write_hdf5_vector_attribute(root.get(), "WholeExtent", H5T_STD_I32LE, H5T_NATIVE_INT, 6u, whole_extent, error) ||
        !write_hdf5_vector_attribute(root.get(), "Origin", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 3u, origin, error) ||
        !write_hdf5_vector_attribute(root.get(), "Spacing", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 3u, spacing, error) ||
        !write_hdf5_vector_attribute(root.get(), "Direction", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 9u, direction, error) ||
        !write_hdf5_string_attribute(root.get(), "CULBMCompression",
                                     use_deflate ? "shuffle+deflate level 4" : "chunked; deflate encoder unavailable", error)) {
        return false;
    }

    Hdf5Handle point_data(H5Gcreate2(root.get(), "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    if (!point_data.valid()) return set_error(error, "could not create VTKHDF PointData group in '" + path + "'");
    if (!write_hdf5_point_data_array<float>(
            point_data.get(), "velocity", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, grid, 3u, use_deflate, "Vectors",
            [&grid](const std::uint64_t index, const std::uint32_t component) {
                return scaled_velocity(grid, index, component);
            }, error) ||
        !write_hdf5_point_data_array<float>(
            point_data.get(), "rho", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, grid, 1u, use_deflate, "Scalars",
            [&grid](const std::uint64_t index, std::uint32_t) {
                return scaled_density(grid, index);
            }, error) ||
        !write_hdf5_point_data_array<std::uint8_t>(
            point_data.get(), "flags", H5T_STD_U8LE, H5T_NATIVE_UCHAR, grid, 1u, use_deflate, nullptr,
            [&grid](const std::uint64_t index, std::uint32_t) {
                return raw_flags(grid, index);
            }, error)) {
        return false;
    }
    if (has_temperature(grid) &&
        !write_hdf5_point_data_array<float>(
            point_data.get(), "temperature", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, grid, 1u, use_deflate, nullptr,
            [&grid](const std::uint64_t index, std::uint32_t) {
                return scaled_temperature(grid, index);
            }, error)) {
        return false;
    }
    if (H5Fflush(file.get(), H5F_SCOPE_GLOBAL) < 0) {
        return set_error(error, "could not finalize VTKHDF file '" + path + "'");
    }
    return true;
#endif
}

bool write_agl_points_vtk(const std::string& path,
                          const PhysicalGridView& grid,
                          const PhysicalAglSurfaceView& surface,
                          std::string& error) {
    if (!validate_surface(grid, surface, error)) return false;
    const AglPointIndex agl = build_agl_index(grid, surface);
    const std::uint64_t points = agl.size();
    if (points == 0ull) return set_error(error, "AGL output contains no point above the DEM terrain");
    if (points > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return set_error(error, "legacy VTK point output exceeds the 32-bit cell-index limit");
    }
    if (!make_parent_directory(path, error)) return false;
    std::ofstream file(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!file.is_open()) return set_error(error, "could not open '" + path + "' for writing");
    file << "# vtk DataFile Version 3.0\n"
         << "CULBM DEM-referenced AGL point output " << filename_only(path) << "\n"
         << "BINARY\nDATASET POLYDATA\nPOINTS " << points << " double\n";
    BufferedBinaryWriter writer(file);
    if (!write_parallel_chunks<std::uint64_t>(writer, points, 3ull,
            [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint64_t* out) {
                for (std::uint64_t i = i0; i < i1; ++i) {
                    const std::size_t p = static_cast<std::size_t>(base + i);
                    const std::uint64_t xy = static_cast<std::uint64_t>(agl.x[p]) +
                        static_cast<std::uint64_t>(agl.y[p]) * grid.nx;
                    out[3ull * i] = double_to_be64(x_coordinate(grid, agl.x[p]));
                    out[3ull * i + 1ull] = double_to_be64(y_coordinate(grid, agl.y[p]));
                    out[3ull * i + 2ull] = double_to_be64(z_coordinate(grid, agl.z[p]) - static_cast<double>(surface.ground_z[xy]));
                }
            }) || !writer.flush()) return set_error(error, "failed while writing AGL VTK coordinates");
    file << "\nVERTICES " << points << " " << 2ull * points << "\n";
    const std::uint32_t one_be = int32_to_be32(1);
    if (!write_parallel_chunks<std::uint32_t>(writer, points, 2ull,
            [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                for (std::uint64_t i = i0; i < i1; ++i) {
                    out[2ull * i] = one_be;
                    out[2ull * i + 1ull] = int32_to_be32(static_cast<std::int32_t>(base + i));
                }
            }) || !writer.flush()) return set_error(error, "failed while writing AGL VTK vertices");
    file << "\nPOINT_DATA " << points << "\nVECTORS velocity float\n";
    if (!write_agl_vtk_velocity(writer, grid, agl) || !writer.flush()) return set_error(error, "failed while writing AGL VTK velocity data");
    file << "\nSCALARS rho float 1\nLOOKUP_TABLE default\n";
    if (!write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
            [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                for (std::uint64_t i = i0; i < i1; ++i)
                    out[i] = float_to_be32(scaled_density(grid, agl.src[static_cast<std::size_t>(base + i)]));
            }) || !writer.flush()) return set_error(error, "failed while writing AGL VTK density data");
    file << "\nSCALARS flags unsigned_char 1\nLOOKUP_TABLE default\n";
    if (!write_parallel_chunks<std::uint8_t>(writer, points, 1ull,
            [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint8_t* out) {
                for (std::uint64_t i = i0; i < i1; ++i)
                    out[i] = raw_flags(grid, agl.src[static_cast<std::size_t>(base + i)]);
            }) || !writer.flush()) return set_error(error, "failed while writing AGL VTK flags data");
    file << "\nSCALARS ground_z double 1\nLOOKUP_TABLE default\n";
    if (!write_parallel_chunks<std::uint64_t>(writer, points, 1ull,
            [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint64_t* out) {
                for (std::uint64_t i = i0; i < i1; ++i) {
                    const std::size_t p = static_cast<std::size_t>(base + i);
                    const std::uint64_t xy = static_cast<std::uint64_t>(agl.x[p]) +
                        static_cast<std::uint64_t>(agl.y[p]) * grid.nx;
                    out[i] = double_to_be64(static_cast<double>(surface.ground_z[xy]));
                }
            }) || !writer.flush()) return set_error(error, "failed while writing AGL VTK terrain heights");
    if (has_temperature(grid)) {
        file << "\nSCALARS temperature float 1\nLOOKUP_TABLE default\n";
        if (!write_parallel_chunks<std::uint32_t>(writer, points, 1ull,
                [&](const std::uint64_t base, const std::uint64_t i0, const std::uint64_t i1, std::uint32_t* out) {
                    for (std::uint64_t i = i0; i < i1; ++i)
                        out[i] = float_to_be32(scaled_temperature(grid, agl.src[static_cast<std::size_t>(base + i)]));
                }) || !writer.flush()) return set_error(error, "failed while writing AGL VTK temperature data");
    }
    if (!file) return set_error(error, "failed while finalizing '" + path + "'");
    return true;
}

bool write_agl_points_netcdf(const std::string& path,
                             const PhysicalGridView& grid,
                             const PhysicalAglSurfaceView& surface,
                             std::string& error) {
    if (!validate_surface(grid, surface, error)) return false;
    const AglPointIndex agl = build_agl_index(grid, surface);
    const std::uint64_t points = agl.size();
    if (points == 0ull) return set_error(error, "AGL output contains no point above the DEM terrain");
    if (points > std::numeric_limits<std::uint32_t>::max()) {
        return set_error(error, "NetCDF-3 point dimension exceeds 32-bit length");
    }
    const std::vector<NcDimension> dimensions = { { "point", static_cast<std::uint32_t>(points) } };
    return write_netcdf_file(path, dimensions, common_global_attributes(true),
                             make_agl_netcdf_variables(grid, surface, agl), error);
}

} // namespace culbm
