#pragma once

#include <cstdint>
#include <functional>
#include <string>

#ifndef CULBM_HAS_VTKHDF
#define CULBM_HAS_VTKHDF 0
#endif

namespace culbm {

// Host-side view of a uniform CULBM field. Coordinates and conversion factors
// are SI values; sampled field values remain in the solver's native units.
struct PhysicalGridView {
    std::uint32_t nx = 0u;
    std::uint32_t ny = 0u;
    std::uint32_t nz = 0u;
    std::uint32_t z_begin = 0u;
    std::uint32_t z_count = 0u; // zero means all planes from z_begin onward
    double origin_x = 0.0;
    double origin_y = 0.0;
    double origin_z = 0.0;
    double spacing_x = 0.0;
    double spacing_y = 0.0;
    double spacing_z = 0.0;

    std::function<float(std::uint64_t, std::uint32_t)> velocity_at;
    std::function<float(std::uint64_t)> density_at;
    std::function<float(std::uint64_t)> temperature_at;
    std::function<std::uint8_t(std::uint64_t)> flags_at;

    // Optional contiguous flat source arrays (size >= source_point_count()).
    // Velocity may be supplied either as SoA (velocity_x/y/z) or interleaved
    // AoS (velocity_xyz, 3 floats per cell). When present, writers bypass
    // std::function callbacks and serialize chunks in parallel.
    const float* velocity_x = nullptr;
    const float* velocity_y = nullptr;
    const float* velocity_z = nullptr;
    const float* velocity_xyz = nullptr;
    const float* density = nullptr;
    const float* temperature = nullptr;
    const std::uint8_t* flags = nullptr;

    double velocity_scale = 1.0;
    double density_scale = 1.0;
    double temperature_scale = 1.0;
    double temperature_offset = 0.0;

    std::uint32_t output_nz() const noexcept;
    std::uint64_t source_point_count() const noexcept;
    std::uint64_t output_point_count() const noexcept;
    bool valid(std::string* error = nullptr) const;
};

// Terrain height in the same metric/rebased coordinate system as
// PhysicalGridView::origin_z. One value is required for every x/y column.
struct PhysicalAglSurfaceView {
    const float* ground_z = nullptr;
    std::uint64_t ground_count = 0ull;
};

bool count_agl_points(const PhysicalGridView& grid,
                      const PhysicalAglSurfaceView& surface,
                      std::uint64_t& point_count,
                      std::string& error);

// Structured output retains every non-basement grid point. The files use the
// supplied metric coordinates directly, with z=0 already rebased by CULBM.
bool write_global_structured_vtk(const std::string& path,
                                 const PhysicalGridView& grid,
                                 std::string& error);
bool write_global_structured_netcdf(const std::string& path,
                                     const PhysicalGridView& grid,
                                     std::string& error);
// VTKHDF ImageData stores the same uniform global grid as the legacy
// STRUCTURED_POINTS output. It uses HDF5 chunking and, when its encoder is
// available, lossless shuffle+deflate compression.
bool write_global_structured_vtkhdf(const std::string& path,
                                    const PhysicalGridView& grid,
                                    std::string& error);

// AGL output drops points on or below the continuous DEM surface, emits their
// z coordinate as height above that surface, and keeps x/y in global metres.
bool write_agl_points_vtk(const std::string& path,
                          const PhysicalGridView& grid,
                          const PhysicalAglSurfaceView& surface,
                          std::string& error);
bool write_agl_points_netcdf(const std::string& path,
                             const PhysicalGridView& grid,
                             const PhysicalAglSurfaceView& surface,
                             std::string& error);

} // namespace culbm
