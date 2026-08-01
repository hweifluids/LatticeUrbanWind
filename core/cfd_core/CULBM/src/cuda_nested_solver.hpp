#pragma once

// Construction helpers for the CUDA vertical-nesting runtime.
//
// The ordinary setup code first builds its existing uniform lattice and
// boundary fields.  This module snapshots that host state, then creates only
// the active vertical slabs.  Geometry is deliberately *not* inherited from
// the uniform lattice: every target cell is classified directly from one
// common-finest CUDA ray-traced mask, with the required tie-to-solid rule.

#include "cuda_nested_grid_plan.hpp"
#include "cuda_nested_runtime.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

class LBM;
struct Mesh;

namespace culbm {

// True only for a ghost row facing another nested level.  The finest lower
// ghost is the global compact-ground boundary, while the coarsest upper ghost
// is the global sky boundary; neither may be pre-classified as an internal
// sealed interface.
constexpr bool is_internal_nested_ghost_row(
    const std::uint32_t z,
    const std::uint32_t lower_ghost_z,
    const std::uint32_t upper_ghost_z,
    const bool is_finest_ground_level,
    const bool is_coarsest_sky_level) noexcept {
    return
        (z == lower_ghost_z && !is_finest_ground_level) ||
        (z == upper_ghost_z && !is_coarsest_sky_level);
}

struct NestedSeedState {
    std::uint32_t nx = 0u;
    std::uint32_t ny = 0u;
    std::uint32_t nz = 0u;
    std::vector<float> rho;
    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
    std::vector<float> temperature;
    std::vector<unsigned char> flags;

    std::uint64_t cell_count() const noexcept;
    bool valid() const noexcept;
};

// Lattice parameters for the original coarse grid.  A child of ratio r uses
// dx=dx_coarse/r and dt=dt_coarse/r.  Therefore nu and alpha are multiplied
// by r, beta remains a temperature-scale conversion, and the lattice
// Coriolis angular velocity and gravity acceleration are divided by r.
struct NestedLbmPhysics {
    float coarse_nu = 0.0f;
    float coarse_alpha = 0.0f;
    float beta = 0.0f;
    float coarse_buoyancy_gx = 0.0f;
    float coarse_buoyancy_gy = 0.0f;
    float coarse_buoyancy_gz = 0.0f;
    int temperature_outlet_face = 0; // 0=off, 1=west, 2=east, 3=south, 4=north
    float coarse_omega_x = 0.0f;
    float coarse_omega_y = 0.0f;
    float coarse_omega_z = 0.0f;
    double coarse_dt_si = 0.0;
    // Setup-owned device allocations (currently one VK inlet updater per
    // level) that coexist with the complete hierarchy.
    std::uint64_t additional_gpu_memory_per_device_mb = 0u;
};

struct NestedGeometrySummary {
    std::uint32_t common_finest_ratio = 1u;
    std::uint32_t raster_nx = 0u;
    std::uint32_t raster_ny = 0u;
    std::uint32_t raster_nz = 0u;
    float stl_top_coarse_lbm_z = 0.0f;
    float terrain_top_common_finest_z = 0.0f;
    float compact_base_top_common_finest_z = std::numeric_limits<float>::quiet_NaN();
    std::uint64_t common_finest_solid_cells = 0u;
    std::uint64_t terrain_solid_cells = 0u;
    std::uint64_t compact_base_solid_cells_removed = 0u;
    std::vector<std::uint64_t> solid_cells_per_plan_level;
};

// Optional terrain complement added to the common-finest STL raster before
// any level is classified.  Each value is a common-finest lattice z
// coordinate: a cell at integer z is solid iff z < solid_below_z[x,y].  This
// is intentionally evaluated on the common-finest grid, so the direct
// majority vote below never derives terrain by recursively coarsening an
// already-coarsened level.  It is used by profile cases whose DEM is supplied
// separately from the STL.
struct NestedCommonFinestTerrain {
    std::uint32_t nx = 0u;
    std::uint32_t ny = 0u;
    std::vector<float> solid_below_z;

    bool valid() const noexcept;
    float highest_solid_z() const noexcept;
};

struct NestedOwnedLbmLevel {
    // `plan` remains in compact-ground -> sky ordering metadata even though
    // `levels` below is stored sky -> ground for scheduler recursion.
    VerticalNestingLevelPlan plan{};
    std::uint32_t plan_level_index = 0u;
    std::unique_ptr<LBM> lbm;
};

struct NestedVerticalLbm {
    VerticalNestingPlan plan{};
    NestedGeometrySummary geometry{};
    double coarse_dt_si = 0.0;
    // Coarsest sky slab first and finest ground slab last. This is the exact
    // order expected by NestedLbmScheduler.
    std::vector<NestedOwnedLbmLevel> levels;

    std::vector<NestedLbmLevel> make_scheduler_levels() const;
    bool valid() const noexcept;
};

// A physical-air-only common-finest view for output and post-processing.
// Coarser values are repeated over their logical finest children; no new
// interpolation is introduced. The compact ground/interface ghost rows are
// intentionally omitted, so `origin_z_above_ground_si` is the centre of the
// first physical-air cell above the compact-ground top.
struct NestedStitchedFields {
    std::uint32_t nx = 0u;
    std::uint32_t ny = 0u;
    std::uint32_t nz = 0u;
    float dx_si = 0.0f;
    float origin_z_above_ground_si = 0.0f;
    std::vector<float> rho;
    std::vector<float> u; // AoS, x/y/z consecutive
    std::vector<float> temperature;
    std::vector<unsigned char> flags;

    std::uint64_t cell_count() const noexcept;
    bool valid() const noexcept;
};

NestedSeedState capture_nested_seed_state(LBM& lbm);

// All nested LBM levels coexist on every selected device.  A single-level
// LBM constructor cannot see allocations owned by its siblings, so callers
// must budget the sum rather than the largest individual slab.
std::uint64_t nested_hierarchy_vram_required_mb_per_device(
    const VerticalNestingPlan& plan,
    std::uint32_t domains_x,
    std::uint32_t domains_y,
    std::uint32_t domains_z) noexcept;

// Conservative setup-time peak for the temporary common-finest geometry
// lattice when transformed STL/terrain height is not available yet.
std::uint64_t nested_full_height_raster_vram_required_mb_per_device(
    const VerticalNestingPlan& plan,
    std::uint32_t domains_x,
    std::uint32_t domains_y,
    std::uint32_t domains_z) noexcept;

// Builds the common-finest ray-traced geometry mask, direct majority masks
// for every level, and the LBM instances themselves. `transformed_mesh` must
// already be in the ordinary coarse-LBM mesh coordinate system used by
// LBM::voxelize_mesh_on_device().  No SI coordinate is passed into this
// routine; SI is used only in VerticalNestingPlan and NestedLbmPhysics.
//
// On failure this returns false, leaves `out` empty, and writes a diagnostic
// that is safe to present directly to the deck user.
bool build_nested_vertical_lbm(const VerticalNestingPlan& plan,
                               const NestedSeedState& seed,
                               const Mesh& transformed_mesh,
                               std::uint32_t domains_x,
                               std::uint32_t domains_y,
                               std::uint32_t domains_z,
                               const NestedLbmPhysics& physics,
                               NestedVerticalLbm& out,
                               std::string& error,
                               const NestedCommonFinestTerrain* terrain = nullptr,
                               float compact_base_top_coarse_lbm =
                                   std::numeric_limits<float>::quiet_NaN());

bool stitch_nested_physical_fields(NestedVerticalLbm& nested,
                                   NestedStitchedFields& out,
                                   std::string& error);

} // namespace culbm
