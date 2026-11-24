#include "physical_output.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if CULBM_HAS_VTKHDF
#include <hdf5.h>
#endif

namespace {

constexpr std::uint64_t k_bytes_per_point = 17ull; // velocity(3*float), rho(float), flags(uint8)
constexpr std::uint32_t k_default_nx = 512u;
constexpr std::uint32_t k_default_ny = 512u;
constexpr std::uint32_t k_default_nz = 384u;

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

float signed_noise(const std::uint64_t key) {
    constexpr float scale = 2.0f / 16777215.0f;
    const std::uint32_t mantissa = static_cast<std::uint32_t>(mix64(key) >> 40u);
    return static_cast<float>(mantissa) * scale - 1.0f;
}

class PseudoTurbulenceField {
public:
    PseudoTurbulenceField(const std::uint32_t nx,
                          const std::uint32_t ny,
                          const std::uint32_t nz)
        : nx_(nx), ny_(ny), plane_(static_cast<std::uint64_t>(nx) * ny) {
        constexpr double pi = 3.1415926535897932384626433832795;
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            x_waves_[component].resize(nx);
            y_waves_[component].resize(ny);
            z_waves_[component].resize(nz);
            const double phase = 0.47 * static_cast<double>(component + 1u);
            for (std::uint32_t x = 0u; x < nx; ++x) {
                const double a = 2.0 * pi * static_cast<double>(x) / static_cast<double>(nx);
                x_waves_[component][x] = static_cast<float>(
                    0.70 * std::sin((component + 2u) * a + phase) +
                    0.30 * std::sin((component + 7u) * a - 0.5 * phase));
            }
            for (std::uint32_t y = 0u; y < ny; ++y) {
                const double a = 2.0 * pi * static_cast<double>(y) / static_cast<double>(ny);
                y_waves_[component][y] = static_cast<float>(
                    0.65 * std::sin((component + 3u) * a - phase) +
                    0.35 * std::sin((component + 9u) * a + 0.75 * phase));
            }
            for (std::uint32_t z = 0u; z < nz; ++z) {
                const double a = 2.0 * pi * static_cast<double>(z) / static_cast<double>(nz);
                z_waves_[component][z] = static_cast<float>(
                    0.55 * std::sin((component + 1u) * a + 1.5 * phase) +
                    0.45 * std::sin((component + 5u) * a - phase));
            }
        }
    }

    float velocity(const std::uint64_t index, const std::uint32_t component) const {
        if (cached_index_ != index) sample_velocity(index);
        return cached_velocity_[component];
    }

    float density(const std::uint64_t index) const {
        const float fine = signed_noise(index ^ 0xd1b54a32d192ed03ull);
        const float medium = signed_noise((index >> 5u) ^ 0x94d049bb133111ebull);
        return 1.0f + 0.018f * fine + 0.006f * medium;
    }

    std::uint8_t flags(const std::uint64_t index) const {
        const std::uint8_t occupancy = static_cast<std::uint8_t>(
            mix64(index ^ 0xa24baed4963ee407ull) & 0xffull);
        return occupancy < 5u ? 1u : (occupancy < 8u ? 2u : 0u);
    }

private:
    void sample_velocity(const std::uint64_t index) const {
        const std::uint32_t z = static_cast<std::uint32_t>(index / plane_);
        const std::uint64_t remainder = index - static_cast<std::uint64_t>(z) * plane_;
        const std::uint32_t y = static_cast<std::uint32_t>(remainder / nx_);
        const std::uint32_t x = static_cast<std::uint32_t>(
            remainder - static_cast<std::uint64_t>(y) * nx_);
        const std::uint64_t cell_key =
            static_cast<std::uint64_t>(x >> 4u) * 0x9e3779b97f4a7c15ull ^
            static_cast<std::uint64_t>(y >> 4u) * 0xbf58476d1ce4e5b9ull ^
            static_cast<std::uint64_t>(z >> 3u) * 0x94d049bb133111ebull;
        constexpr std::array<float, 3u> mean = { 7.0f, -1.5f, 0.25f };
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            const float coherent = 0.75f * x_waves_[component][x] +
                0.65f * y_waves_[component][y] + 0.50f * z_waves_[component][z];
            const float medium = signed_noise(
                cell_key ^ (0x632be59bd9b4e019ull * (component + 1u)));
            const float broadband = signed_noise(
                index ^ (0x8cb92ba72f3d8dd7ull * (component + 1u)));
            cached_velocity_[component] = mean[component] + coherent +
                0.80f * medium + 1.35f * broadband;
        }
        cached_index_ = index;
    }

    std::uint32_t nx_ = 0u;
    std::uint32_t ny_ = 0u;
    std::uint64_t plane_ = 0ull;
    std::array<std::vector<float>, 3u> x_waves_;
    std::array<std::vector<float>, 3u> y_waves_;
    std::array<std::vector<float>, 3u> z_waves_;
    mutable std::uint64_t cached_index_ = std::numeric_limits<std::uint64_t>::max();
    mutable std::array<float, 3u> cached_velocity_{};
};

bool parse_dimension(const std::string& text, std::uint32_t& output) {
    try {
        const unsigned long long parsed = std::stoull(text);
        if (parsed == 0ull || parsed > std::numeric_limits<std::uint32_t>::max()) return false;
        output = static_cast<std::uint32_t>(parsed);
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

void print_usage(const char* executable) {
    std::cout << "Usage: " << executable << " [--output DIRECTORY] [--nx N --ny N --nz N]\n"
              << "Defaults: 512 x 512 x 384, a 1.59 GiB uncompressed payload without temperature.\n";
}

struct WriteResult {
    std::string name;
    std::filesystem::path path;
    std::uint64_t bytes = 0ull;
    double seconds = 0.0;
};

template<typename Writer>
bool write_and_measure(const std::string& name,
                       const std::filesystem::path& path,
                       Writer&& writer,
                       std::vector<WriteResult>& results) {
    std::string error;
    const auto started = std::chrono::steady_clock::now();
    if (!writer(path.string(), error)) {
        std::cerr << name << " write failed: " << error << "\n";
        return false;
    }
    const auto finished = std::chrono::steady_clock::now();
    std::error_code ec;
    const std::uint64_t bytes = std::filesystem::file_size(path, ec);
    if (ec) {
        std::cerr << "Could not measure " << path.string() << ": " << ec.message() << "\n";
        return false;
    }
    const double seconds = std::chrono::duration<double>(finished - started).count();
    results.push_back({ name, path, bytes, seconds });
    std::cout << name << ": " << bytes << " bytes, " << std::fixed
              << std::setprecision(2) << seconds << " s\n";
    return true;
}

#if CULBM_HAS_VTKHDF
bool vtkhdf_has_expected_grid(const std::filesystem::path& path,
                              const culbm::PhysicalGridView& grid,
                              const PseudoTurbulenceField& field,
                              bool& deflate_present) {
    hid_t file = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t root = H5I_INVALID_HID;
    hid_t point_data = H5I_INVALID_HID;
    hid_t velocity = H5I_INVALID_HID;
    hid_t velocity_space = H5I_INVALID_HID;
    hid_t velocity_creation = H5I_INVALID_HID;
    hid_t memory_space = H5I_INVALID_HID;
    deflate_present = false;
    bool valid = false;
    do {
        if (file == H5I_INVALID_HID) break;
        root = H5Gopen2(file, "/VTKHDF", H5P_DEFAULT);
        if (root == H5I_INVALID_HID || H5Aexists(root, "Type") <= 0 ||
            H5Aexists(root, "WholeExtent") <= 0 || H5Aexists(root, "Origin") <= 0 ||
            H5Aexists(root, "Spacing") <= 0) break;
        point_data = H5Gopen2(root, "PointData", H5P_DEFAULT);
        if (point_data == H5I_INVALID_HID) break;
        velocity = H5Dopen2(point_data, "velocity", H5P_DEFAULT);
        if (velocity == H5I_INVALID_HID || H5Aexists(velocity, "Attribute") <= 0) break;
        velocity_space = H5Dget_space(velocity);
        std::array<hsize_t, 4u> dimensions{};
        if (velocity_space == H5I_INVALID_HID || H5Sget_simple_extent_ndims(velocity_space) != 4 ||
            H5Sget_simple_extent_dims(velocity_space, dimensions.data(), nullptr) < 0 ||
            dimensions[0] != grid.output_nz() || dimensions[1] != grid.ny ||
            dimensions[2] != grid.nx || dimensions[3] != 3u) break;
        velocity_creation = H5Dget_create_plist(velocity);
        if (velocity_creation == H5I_INVALID_HID) break;
        const int filters = H5Pget_nfilters(velocity_creation);
        for (int filter = 0; filter < filters; ++filter) {
            unsigned int flags = 0u;
            std::array<unsigned int, 8u> client_values{};
            std::size_t client_value_count = client_values.size();
            if (H5Pget_filter2(velocity_creation, static_cast<unsigned int>(filter), &flags,
                               &client_value_count, client_values.data(), 0u, nullptr, nullptr) == H5Z_FILTER_DEFLATE) {
                deflate_present = true;
            }
        }
        const std::uint32_t sample_x = grid.nx / 2u;
        const std::uint32_t sample_y = grid.ny / 3u;
        const std::uint32_t sample_z = grid.output_nz() / 2u;
        const std::uint64_t source_index = static_cast<std::uint64_t>(sample_x) +
            (static_cast<std::uint64_t>(sample_y) + static_cast<std::uint64_t>(sample_z) * grid.ny) * grid.nx;
        const std::array<hsize_t, 4u> offsets = { sample_z, sample_y, sample_x, 2u };
        const std::array<hsize_t, 4u> counts = { 1u, 1u, 1u, 1u };
        memory_space = H5Screate_simple(4, counts.data(), nullptr);
        float sampled_velocity = 0.0f;
        if (memory_space == H5I_INVALID_HID ||
            H5Sselect_hyperslab(velocity_space, H5S_SELECT_SET, offsets.data(), nullptr, counts.data(), nullptr) < 0 ||
            H5Dread(velocity, H5T_NATIVE_FLOAT, memory_space, velocity_space, H5P_DEFAULT, &sampled_velocity) < 0 ||
            sampled_velocity != field.velocity(source_index, 2u)) break;
        valid = true;
    } while (false);
    if (memory_space != H5I_INVALID_HID) H5Sclose(memory_space);
    if (velocity_creation != H5I_INVALID_HID) H5Pclose(velocity_creation);
    if (velocity_space != H5I_INVALID_HID) H5Sclose(velocity_space);
    if (velocity != H5I_INVALID_HID) H5Dclose(velocity);
    if (point_data != H5I_INVALID_HID) H5Gclose(point_data);
    if (root != H5I_INVALID_HID) H5Gclose(root);
    if (file != H5I_INVALID_HID) H5Fclose(file);
    return valid;
}
#endif

} // namespace

int main(int argc, char** argv) {
    std::uint32_t nx = k_default_nx;
    std::uint32_t ny = k_default_ny;
    std::uint32_t nz = k_default_nz;
    std::filesystem::path output_directory = "physical-output-gib-benchmark";
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if ((option == "--output" || option == "--nx" || option == "--ny" || option == "--nz") && argument + 1 < argc) {
            const std::string value = argv[++argument];
            if (option == "--output") output_directory = value;
            else if (option == "--nx" && !parse_dimension(value, nx)) {
                std::cerr << "Invalid --nx value: " << value << "\n";
                return 2;
            }
            else if (option == "--ny" && !parse_dimension(value, ny)) {
                std::cerr << "Invalid --ny value: " << value << "\n";
                return 2;
            }
            else if (option == "--nz" && !parse_dimension(value, nz)) {
                std::cerr << "Invalid --nz value: " << value << "\n";
                return 2;
            }
            continue;
        }
        std::cerr << "Unknown or incomplete argument: " << option << "\n";
        print_usage(argv[0]);
        return 2;
    }

    const std::uint64_t points = static_cast<std::uint64_t>(nx) * ny * nz;
    if (points == 0ull || points > std::numeric_limits<std::uint64_t>::max() / k_bytes_per_point) {
        std::cerr << "Grid dimensions overflow the benchmark payload calculation\n";
        return 2;
    }
    std::error_code ec;
    std::filesystem::create_directories(output_directory, ec);
    if (ec) {
        std::cerr << "Could not create output directory " << output_directory.string() << ": " << ec.message() << "\n";
        return 3;
    }

    PseudoTurbulenceField field(nx, ny, nz);
    culbm::PhysicalGridView grid;
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.origin_x = 0.0;
    grid.origin_y = 0.0;
    grid.origin_z = 0.0;
    grid.spacing_x = 2.0;
    grid.spacing_y = 2.0;
    grid.spacing_z = 2.0;
    grid.velocity_at = [&field](const std::uint64_t index, const std::uint32_t component) {
        return field.velocity(index, component);
    };
    grid.density_at = [&field](const std::uint64_t index) { return field.density(index); };
    grid.flags_at = [&field](const std::uint64_t index) { return field.flags(index); };

    const double raw_gib = static_cast<double>(points * k_bytes_per_point) / static_cast<double>(1ull << 30u);
    std::cout << "Deterministic pseudo-turbulence field: " << nx << " x " << ny << " x " << nz
              << " = " << points << " points\n"
              << "Uncompressed point payload (velocity, rho, flags): " << points * k_bytes_per_point
              << " bytes (" << std::fixed << std::setprecision(2) << raw_gib << " GiB)\n"
              << "Output directory: " << output_directory.string() << "\n";

    std::vector<WriteResult> results;
    const std::filesystem::path legacy_path = output_directory / "pseudo_turbulence_global.vtk";
    const std::filesystem::path netcdf_path = output_directory / "pseudo_turbulence_global.nc";
    const std::filesystem::path vtkhdf_path = output_directory / "pseudo_turbulence_global.vtkhdf";
    if (!write_and_measure("Legacy VTK", legacy_path,
                           [&grid](const std::string& path, std::string& error) {
                               return culbm::write_global_structured_vtk(path, grid, error);
                           }, results) ||
        !write_and_measure("NetCDF CDF-2", netcdf_path,
                           [&grid](const std::string& path, std::string& error) {
                               return culbm::write_global_structured_netcdf(path, grid, error);
                           }, results) ||
        !write_and_measure("VTKHDF ImageData", vtkhdf_path,
                           [&grid](const std::string& path, std::string& error) {
                               return culbm::write_global_structured_vtkhdf(path, grid, error);
                           }, results)) {
        return 4;
    }

#if CULBM_HAS_VTKHDF
    bool deflate_present = false;
    if (!vtkhdf_has_expected_grid(vtkhdf_path, grid, field, deflate_present)) {
        std::cerr << "VTKHDF metadata, dimensions, or sampled velocity validation failed\n";
        return 5;
    }
#endif

    const double baseline = static_cast<double>(results.front().bytes);
    std::cout << "\nFormat summary (legacy VTK = 1.00x):\n";
    for (const WriteResult& result : results) {
        const double gib = static_cast<double>(result.bytes) / static_cast<double>(1ull << 30u);
        const double bytes_per_point = static_cast<double>(result.bytes) / static_cast<double>(points);
        const double ratio = static_cast<double>(result.bytes) / baseline;
        std::cout << "  " << result.name << ": " << std::fixed << std::setprecision(3) << gib << " GiB, "
                  << std::setprecision(2) << bytes_per_point << " B/point, " << ratio << "x legacy, "
                  << result.seconds << " s\n";
    }
#if CULBM_HAS_VTKHDF
    std::cout << "VTKHDF HDF5 metadata and a nontrivial sampled velocity value verified; Deflate filter "
              << (deflate_present ? "is present." : "is not present (the writer used its valid uncompressed fallback).") << "\n";
#endif
    return 0;
}
