#pragma once

// Device memory of a case before any GPU allocation: the lattice of a grid,
// the VK inlet and particle buffers, the nested hierarchy, and the cell size
// that fits a requested per-device budget (mesh_control = gpu_memory).

#include "settings.hpp"

#include <cstdint>

namespace culbm::cases {

struct ProjectGpuMemoryEstimate {
    uint3 grid = uint3(1u);
    uint core_per_device_mb = 0u;
    uint extra_per_device_mb = 0u;
    uint total_per_device_mb = 0u;
    uint total_project_mb = 0u;
    bool top_sponge_grid_extend = false;
    uint sponge_cells = 0u;
};

uint estimate_particle_advection_extra_mb_per_device(
    const CaseSettings& deck,
    const uint3& grid,
    const culbm::particle_advection::Config* advection);

ProjectGpuMemoryEstimate estimate_project_gpu_memory_from_grid(
    const CaseSettings& deck,
    const uint3& grid,
    const culbm::particle_advection::Config* advection = nullptr);

ProjectGpuMemoryEstimate estimate_project_gpu_memory_from_cell_size(
    const CaseSettings& deck,
    float cell_size_m,
    const std::vector<float>& nesting_heights_agl_si = {},
    const culbm::particle_advection::Config* advection = nullptr,
    std::uint64_t voxelization_transient_mb = 0u);

// Budget left by a request after the runtime and allocator reserves.
uint usable_gpu_memory_request_mb(uint target_mb);

float fit_cell_size_to_gpu_memory_request(
    const CaseSettings& deck,
    uint target_mb,
    const std::vector<float>& nesting_heights_agl_si = {},
    const culbm::particle_advection::Config* advection = nullptr,
    std::uint64_t voxelization_transient_mb = 0u);

} // namespace culbm::cases
