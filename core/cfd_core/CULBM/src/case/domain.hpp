#pragma once

// Lattice of a case: the CUDA compact ground (complete base cells of the
// generated foundation slab are dropped from the lattice), the grid with the
// optional top sponge extension, the vertical nesting plan, and the device
// memory estimate before any allocation.

#include "memory_fit.hpp"
#include "../cuda_nested_grid_plan.hpp"

#include <cstdint>
#include <memory>

namespace culbm::cases {

struct CaseContext;

struct CaseDomain {
    static constexpr float lbm_ref_u = 0.10f; // 0.1731 is Ma=0.3
    static constexpr float si_rho = 1.225f;

    // CUDA compact ground. The preprocessor creates a closed slab from the
    // STL bottom to base_height; complete base cells leave the lattice and
    // the remaining fractional base is placed in z=0.
    float cuda_ground_base_height_si = 0.0f;
    uint cuda_ground_trim_cells = 0u;
    uint cuda_ground_geometry_shift_cells = 0u;
    float cuda_ground_trim_si = 0.0f;
    bool cuda_ground_compaction_active = false;
    std::string cuda_ground_compaction_reason = "base_height is not positive";
    std::string cuda_ground_probe_path;
    std::unique_ptr<Mesh> cuda_ground_probe_mesh;
    std::string cuda_ground_compaction_override; // CULBM_GROUND_COMPACTION
    bool cuda_ground_compaction_requested = true;

    // Lattice.
    uint Nx_cells = 1u;
    uint Ny_cells = 1u;
    int sponge_cells_cfg = 1;
    uint Nz_cells_uncompacted = 1u;
    uint Nz_cells_core = 1u;
    bool top_sponge_grid_extend = false;
    uint Nz_cells = 1u;
    int top_sponge_side_ref_z_cap = -1;
    uint3 lbm_N = uint3(1u);

    ProjectGpuMemoryEstimate gpu_memory_estimate;
    std::uint64_t nested_additional_gpu_memory_per_device_mb = 0u;
    std::uint64_t nested_full_height_raster_core_mb = 0u;
    culbm::VerticalNestingPlan cuda_vertical_nesting_plan;
};

// Ground compaction, lattice size, nesting plan and memory estimate.
void derive_domain(CaseContext& ctx);

} // namespace culbm::cases
