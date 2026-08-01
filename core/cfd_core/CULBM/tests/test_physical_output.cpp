#include "physical_output.hpp"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if CULBM_HAS_VTKHDF
#include <hdf5.h>
#endif

namespace {

bool file_starts_with(const std::filesystem::path& path, const std::array<char, 4u>& expected) {
    std::ifstream file(path, std::ios::binary);
    std::array<char, 4u> actual{};
    file.read(actual.data(), static_cast<std::streamsize>(actual.size()));
    return file.good() && actual == expected;
}

bool file_contains(const std::filesystem::path& path, const std::string& text) {
    std::ifstream file(path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return data.find(text) != std::string::npos;
}

#if CULBM_HAS_VTKHDF
bool vtkhdf_file_valid(const std::filesystem::path& path) {
    hid_t file = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file == H5I_INVALID_HID) return false;
    hid_t root = H5I_INVALID_HID;
    hid_t point_data = H5I_INVALID_HID;
    hid_t velocity = H5I_INVALID_HID;
    hid_t velocity_space = H5I_INVALID_HID;
    hid_t velocity_creation = H5I_INVALID_HID;
    bool valid = false;
    do {
        root = H5Gopen2(file, "/VTKHDF", H5P_DEFAULT);
        if (root == H5I_INVALID_HID || H5Aexists(root, "Version") <= 0 ||
            H5Aexists(root, "Type") <= 0 || H5Aexists(root, "WholeExtent") <= 0 ||
            H5Aexists(root, "Origin") <= 0 || H5Aexists(root, "Spacing") <= 0) break;
        point_data = H5Gopen2(root, "PointData", H5P_DEFAULT);
        if (point_data == H5I_INVALID_HID || H5Lexists(point_data, "rho", H5P_DEFAULT) <= 0 ||
            H5Lexists(point_data, "flags", H5P_DEFAULT) <= 0 ||
            H5Lexists(point_data, "temperature", H5P_DEFAULT) <= 0) break;
        velocity = H5Dopen2(point_data, "velocity", H5P_DEFAULT);
        if (velocity == H5I_INVALID_HID) break;
        velocity_space = H5Dget_space(velocity);
        hsize_t dimensions[4] = {};
        if (velocity_space == H5I_INVALID_HID || H5Aexists(velocity, "Attribute") <= 0 ||
            H5Sget_simple_extent_ndims(velocity_space) != 4 ||
            H5Sget_simple_extent_dims(velocity_space, dimensions, nullptr) < 0) break;
        velocity_creation = H5Dget_create_plist(velocity);
        if (velocity_creation == H5I_INVALID_HID) break;
        bool has_shuffle = false;
        bool has_deflate = false;
        const int filters = H5Pget_nfilters(velocity_creation);
        for (int filter = 0; filter < filters; ++filter) {
            unsigned int flags = 0u;
            std::array<unsigned int, 16u> client_value_storage{};
            std::size_t client_values = client_value_storage.size();
            H5Z_filter_t filter_id = H5Pget_filter2(
                velocity_creation, static_cast<unsigned int>(filter), &flags, &client_values,
                client_value_storage.data(), 0u, nullptr, nullptr);
            has_shuffle = has_shuffle || filter_id == H5Z_FILTER_SHUFFLE;
            has_deflate = has_deflate || filter_id == H5Z_FILTER_DEFLATE;
        }
        unsigned int deflate_capabilities = 0u;
        const bool deflate_available = H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0 &&
            H5Zget_filter_info(H5Z_FILTER_DEFLATE, &deflate_capabilities) >= 0 &&
            (deflate_capabilities & H5Z_FILTER_CONFIG_ENCODE_ENABLED) != 0u;
        if (deflate_available && (!has_shuffle || !has_deflate)) break;
        std::vector<float> values(24u, 0.0f);
        if (dimensions[0] != 2u || dimensions[1] != 2u || dimensions[2] != 2u || dimensions[3] != 3u ||
            H5Dread(velocity, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) break;
        valid = values[0] == 4.0f && values[1] == 14.0f && values[2] == 24.0f;
    } while (false);
    if (velocity_creation != H5I_INVALID_HID) H5Pclose(velocity_creation);
    if (velocity_space != H5I_INVALID_HID) H5Sclose(velocity_space);
    if (velocity != H5I_INVALID_HID) H5Dclose(velocity);
    if (point_data != H5I_INVALID_HID) H5Gclose(point_data);
    if (root != H5I_INVALID_HID) H5Gclose(root);
    H5Fclose(file);
    return valid;
}
#endif

} // namespace

int main() {
    constexpr std::uint32_t nx = 2u, ny = 2u, nz = 3u;
    constexpr std::uint64_t count = static_cast<std::uint64_t>(nx) * ny * nz;
    std::vector<float> velocity(static_cast<std::size_t>(3ull * count), 0.0f);
    std::vector<float> density(static_cast<std::size_t>(count), 1.0f);
    std::vector<float> temperature(static_cast<std::size_t>(count), 300.0f);
    std::vector<std::uint8_t> flags(static_cast<std::size_t>(count), 0u);
    for (std::uint64_t i = 0ull; i < count; ++i) {
        velocity[static_cast<std::size_t>(3ull * i)] = static_cast<float>(i);
        velocity[static_cast<std::size_t>(3ull * i + 1ull)] = static_cast<float>(i + 10ull);
        velocity[static_cast<std::size_t>(3ull * i + 2ull)] = static_cast<float>(i + 20ull);
        density[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i) * 0.01f;
        temperature[static_cast<std::size_t>(i)] = 290.0f + static_cast<float>(i);
        flags[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i % 2ull);
    }

    culbm::PhysicalGridView grid;
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.z_begin = 1u; // strip the synthetic basement plane at z=-1 m
    grid.origin_x = 10.0;
    grid.origin_y = 20.0;
    grid.origin_z = -1.0;
    grid.spacing_x = 2.0;
    grid.spacing_y = 2.0;
    grid.spacing_z = 1.0;
    grid.velocity_at = [&velocity](const std::uint64_t index, const std::uint32_t component) {
        return velocity[static_cast<std::size_t>(3ull * index + component)];
    };
    grid.density_at = [&density](const std::uint64_t index) { return density[static_cast<std::size_t>(index)]; };
    grid.temperature_at = [&temperature](const std::uint64_t index) { return temperature[static_cast<std::size_t>(index)]; };
    grid.flags_at = [&flags](const std::uint64_t index) { return flags[static_cast<std::size_t>(index)]; };

    const std::vector<float> terrain = { 0.0f, 0.25f, 0.90f, 1.20f };
    const culbm::PhysicalAglSurfaceView surface{ terrain.data(), terrain.size() };

    std::string error;
    std::uint64_t agl_points = 0ull;
    if (!culbm::count_agl_points(grid, surface, agl_points, error) || agl_points != 3ull) {
        std::cerr << "unexpected AGL point count: " << agl_points << " (" << error << ")\n";
        return 1;
    }

    const std::filesystem::path root = std::filesystem::temp_directory_path() / "culbm_physical_output_test";
    const bool keep_outputs = std::getenv("CULBM_KEEP_PHYSICAL_OUTPUT_TEST") != nullptr;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);
    if (ec) {
        std::cerr << "could not create temporary test directory\n";
        return 2;
    }

    const std::filesystem::path global_vtk = root / "global.vtk";
    const std::filesystem::path global_nc = root / "global.nc";
    const std::filesystem::path global_vtkhdf = root / "global.vtkhdf";
    const std::filesystem::path agl_vtk = root / "agl.vtk";
    const std::filesystem::path agl_nc = root / "agl.nc";
    bool written =
        culbm::write_global_structured_vtk(global_vtk.string(), grid, error) &&
        culbm::write_global_structured_netcdf(global_nc.string(), grid, error) &&
        culbm::write_agl_points_vtk(agl_vtk.string(), grid, surface, error) &&
        culbm::write_agl_points_netcdf(agl_nc.string(), grid, surface, error);
#if CULBM_HAS_VTKHDF
    written = written && culbm::write_global_structured_vtkhdf(global_vtkhdf.string(), grid, error);
#else
    error.clear();
    const bool vtkhdf_unavailable =
        !culbm::write_global_structured_vtkhdf(global_vtkhdf.string(), grid, error) &&
        error.find("no HDF5 support") != std::string::npos;
#endif
    if (!written) {
        std::cerr << "physical output writer failed: " << error << "\n";
        if (!keep_outputs) std::filesystem::remove_all(root, ec);
        return 3;
    }

    const std::array<char, 4u> cdf2 = { 'C', 'D', 'F', static_cast<char>(2) };
    bool valid =
        file_contains(global_vtk, "DATASET STRUCTURED_POINTS") &&
        file_contains(global_vtk, "DIMENSIONS 2 2 2") &&
        file_contains(global_vtk, "ORIGIN 10 20 0") &&
        file_contains(agl_vtk, "DATASET POLYDATA") &&
        file_contains(agl_vtk, "POINTS 3 double") &&
        file_starts_with(global_nc, cdf2) &&
        file_starts_with(agl_nc, cdf2);
#if CULBM_HAS_VTKHDF
    valid = valid && vtkhdf_file_valid(global_vtkhdf);
#else
    valid = valid && vtkhdf_unavailable;
#endif
    if (!keep_outputs) std::filesystem::remove_all(root, ec);
    if (!valid) {
        std::cerr << "physical output format validation failed\n";
        return 4;
    }
    std::cout << "Validated structured global and DEM-referenced AGL VTK/NetCDF outputs";
#if CULBM_HAS_VTKHDF
    std::cout << ", including VTKHDF ImageData.";
#else
    std::cout << "; verified the clear no-HDF5 VTKHDF error path.";
#endif
    if (keep_outputs) std::cout << " Files retained in " << root.string() << ".";
    std::cout << "\n";
    return 0;
}
