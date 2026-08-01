#pragma once

// Common-finest geometry rasterization for CUDA vertical nesting.
//
// A transformed Mesh remains in coarse-LBM coordinates.  This header never
// accepts metres: callers convert SI positions once, then every leaf samples
// that one mesh in a common finest logical lattice.  A target cell on a level
// with ratio r samples q=max_ratio/r points on each axis, so its solid vote is
// made directly from q^3 finest logical voxels.  It is deliberately not a
// recursive coarsening operation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(CULBM_BACKEND)
#include "cuda.hpp"
#endif

class LBM;
struct Mesh;

namespace culbm {

struct NestedGeometrySamplingPlan {
    std::uint32_t level_ratio_to_coarse = 1u;
    std::uint32_t max_ratio_to_coarse = 1u;
    // Number of common-finest logical voxels along each target-cell axis.
    std::uint32_t finest_per_target_axis = 1u;
    std::uint32_t finest_votes_per_target_cell = 1u;

    bool valid() const noexcept {
        return level_ratio_to_coarse > 0u && max_ratio_to_coarse > 0u &&
               finest_per_target_axis > 0u &&
               finest_votes_per_target_cell > 0u;
    }
};

// The planner currently admits only 1x, 2x and 4x leaf resolutions.  Keeping
// the rasterizer equally explicit avoids silently accepting a hierarchy whose
// q^3 work or interface geometry no longer has the documented meaning.
inline bool nested_geometry_supported_ratio(
    const std::uint32_t ratio_to_coarse) noexcept {
    return ratio_to_coarse == 1u || ratio_to_coarse == 2u ||
           ratio_to_coarse == 4u;
}

inline bool nested_geometry_make_sampling_plan(
    const std::uint32_t level_ratio_to_coarse,
    const std::uint32_t max_ratio_to_coarse,
    NestedGeometrySamplingPlan& result) noexcept {
    if (!nested_geometry_supported_ratio(level_ratio_to_coarse) ||
        !nested_geometry_supported_ratio(max_ratio_to_coarse) ||
        max_ratio_to_coarse < level_ratio_to_coarse ||
        max_ratio_to_coarse % level_ratio_to_coarse != 0u) {
        return false;
    }

    const std::uint32_t q = max_ratio_to_coarse / level_ratio_to_coarse;
    if (q != 1u && q != 2u && q != 4u) return false;

    result.level_ratio_to_coarse = level_ratio_to_coarse;
    result.max_ratio_to_coarse = max_ratio_to_coarse;
    result.finest_per_target_axis = q;
    result.finest_votes_per_target_cell = q * q * q;
    return true;
}

// Exact policy required by nested geometry: 2*solid_votes >= total_votes.
// The threshold form preserves that tie-to-solid rule without overflowing a
// 64-bit multiplication when this helper is also used by host validation.
inline bool nested_geometry_majority_is_solid(
    const std::uint64_t solid_votes,
    const std::uint64_t total_votes) noexcept {
    if (total_votes == 0u || solid_votes > total_votes) return false;
    return solid_votes >= total_votes / 2u + total_votes % 2u;
}

struct NestedGeometryPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct NestedGeometryTriangle {
    NestedGeometryPoint p0{};
    NestedGeometryPoint p1{};
    NestedGeometryPoint p2{};
};

struct NestedGeometryXYBounds {
    float min_x = 0.0f;
    float max_x = 0.0f;
    float min_y = 0.0f;
    float max_y = 0.0f;

    bool valid() const noexcept {
        return std::isfinite(min_x) && std::isfinite(max_x) &&
               std::isfinite(min_y) && std::isfinite(max_y) &&
               min_x <= max_x && min_y <= max_y;
    }
};

inline NestedGeometryXYBounds nested_geometry_xy_bounds(
    const NestedGeometryTriangle& triangle) noexcept {
    NestedGeometryXYBounds result;
    result.min_x = std::min(triangle.p0.x, std::min(triangle.p1.x, triangle.p2.x));
    result.max_x = std::max(triangle.p0.x, std::max(triangle.p1.x, triangle.p2.x));
    result.min_y = std::min(triangle.p0.y, std::min(triangle.p1.y, triangle.p2.y));
    result.max_y = std::max(triangle.p0.y, std::max(triangle.p1.y, triangle.p2.y));
    return result;
}

// Closed bounds deliberately keep boundary triangles in both neighbouring
// domains.  The domains only write their own interior mask cells, so this
// harmless overlap avoids a seam when a vertical ray lies on a partition.
inline bool nested_geometry_triangle_overlaps_xy(
    const NestedGeometryTriangle& triangle,
    const NestedGeometryXYBounds& bounds,
    const float padding = 0.0f) noexcept {
    if (!bounds.valid() || !std::isfinite(padding) || padding < 0.0f) return false;
    const NestedGeometryXYBounds triangle_bounds = nested_geometry_xy_bounds(triangle);
    return triangle_bounds.max_x >= bounds.min_x - padding &&
           triangle_bounds.min_x <= bounds.max_x + padding &&
           triangle_bounds.max_y >= bounds.min_y - padding &&
           triangle_bounds.min_y <= bounds.max_y + padding;
}

// Layout of one process-owned LBM domain.  `global_*0` addresses the actual
// level lattice, not SI metres and not the common finest lattice.  The latter
// is derived only at launch time from NestedGeometryRasterSpec.
struct NestedGeometryOwnedDomain {
    std::uint32_t local_domain = 0u;
    std::uint32_t global_domain = 0u;
    std::uint32_t local_nx = 0u;
    std::uint32_t local_ny = 0u;
    std::uint32_t local_nz = 0u;
    std::uint32_t inner_nx = 0u;
    std::uint32_t inner_ny = 0u;
    std::uint32_t inner_nz = 0u;
    std::uint32_t halo_x = 0u;
    std::uint32_t halo_y = 0u;
    std::uint32_t halo_z = 0u;
    std::uint32_t global_x0 = 0u;
    std::uint32_t global_y0 = 0u;
    std::uint32_t global_z0 = 0u;
    // Intersection of this domain's interior z rows with the supplied
    // physical-air interval.  Ground and artificial interface ghost rows are
    // intentionally absent from this range and remain zero in the mask.
    std::uint32_t physical_local_z_offset = 0u;
    std::uint32_t physical_nz = 0u;

    bool valid() const noexcept {
        return local_nx == inner_nx + 2u * halo_x &&
               local_ny == inner_ny + 2u * halo_y &&
               local_nz == inner_nz + 2u * halo_z &&
               inner_nx > 0u && inner_ny > 0u && inner_nz > 0u &&
               physical_local_z_offset <= inner_nz &&
               physical_nz <= inner_nz - physical_local_z_offset;
    }

    std::uint64_t local_cell_count() const noexcept {
        return static_cast<std::uint64_t>(local_nx) *
               static_cast<std::uint64_t>(local_ny) *
               static_cast<std::uint64_t>(local_nz);
    }

    std::uint64_t physical_cell_count() const noexcept {
        return static_cast<std::uint64_t>(inner_nx) *
               static_cast<std::uint64_t>(inner_ny) *
               static_cast<std::uint64_t>(physical_nz);
    }
};

// Convert a global Cartesian topology plus one MPI/local ownership id into a
// concrete local layout.  `physical_z_offset` and `physical_nz` are a global
// interval in *actual level lattice rows*.  They must be supplied explicitly:
// rasterization never assumes z=0..Nz-1 is all physical air.
inline bool nested_geometry_make_owned_domain(
    const std::uint32_t local_domain,
    const std::uint32_t global_domain,
    const std::uint32_t global_nx,
    const std::uint32_t global_ny,
    const std::uint32_t global_nz,
    const std::uint32_t domains_x,
    const std::uint32_t domains_y,
    const std::uint32_t domains_z,
    const std::uint32_t local_nx,
    const std::uint32_t local_ny,
    const std::uint32_t local_nz,
    const std::uint32_t physical_z_offset,
    const std::uint32_t physical_nz,
    NestedGeometryOwnedDomain& result) noexcept {
    if (global_nx == 0u || global_ny == 0u || global_nz == 0u ||
        domains_x == 0u || domains_y == 0u || domains_z == 0u ||
        physical_nz == 0u || global_nx % domains_x != 0u ||
        global_ny % domains_y != 0u || global_nz % domains_z != 0u) {
        return false;
    }
    const std::uint64_t domain_count =
        static_cast<std::uint64_t>(domains_x) * domains_y * domains_z;
    const std::uint64_t physical_end =
        static_cast<std::uint64_t>(physical_z_offset) + physical_nz;
    if (global_domain >= domain_count || physical_end > global_nz) return false;

    const std::uint32_t inner_nx = global_nx / domains_x;
    const std::uint32_t inner_ny = global_ny / domains_y;
    const std::uint32_t inner_nz = global_nz / domains_z;
    const std::uint32_t halo_x = domains_x > 1u ? 1u : 0u;
    const std::uint32_t halo_y = domains_y > 1u ? 1u : 0u;
    const std::uint32_t halo_z = domains_z > 1u ? 1u : 0u;
    if (local_nx != inner_nx + 2u * halo_x ||
        local_ny != inner_ny + 2u * halo_y ||
        local_nz != inner_nz + 2u * halo_z) {
        return false;
    }

    const std::uint32_t domain_x = global_domain % domains_x;
    const std::uint32_t domain_y = (global_domain / domains_x) % domains_y;
    const std::uint32_t domain_z = global_domain / (domains_x * domains_y);
    const std::uint32_t global_z0 = domain_z * inner_nz;
    const std::uint32_t global_z1 = global_z0 + inner_nz;
    const std::uint32_t physical_z0 = std::min(
        global_z1, std::max(global_z0, physical_z_offset));
    const std::uint32_t physical_z1 = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(global_z1, physical_end));

    result.local_domain = local_domain;
    result.global_domain = global_domain;
    result.local_nx = local_nx;
    result.local_ny = local_ny;
    result.local_nz = local_nz;
    result.inner_nx = inner_nx;
    result.inner_ny = inner_ny;
    result.inner_nz = inner_nz;
    result.halo_x = halo_x;
    result.halo_y = halo_y;
    result.halo_z = halo_z;
    result.global_x0 = domain_x * inner_nx;
    result.global_y0 = domain_y * inner_ny;
    result.global_z0 = global_z0;
    result.physical_local_z_offset = physical_z0 - global_z0;
    result.physical_nz = physical_z1 > physical_z0 ? physical_z1 - physical_z0 : 0u;
    return result.valid();
}

struct NestedGeometryRasterSpec {
    std::uint32_t level_ratio_to_coarse = 1u;
    std::uint32_t max_ratio_to_coarse = 1u;
    // Coarse-LBM coordinate of actual-level index (0,0,0).  For example, a
    // slab that begins 50 m above the compacted-ground plane uses its already
    // converted coarse-LBM z coordinate here.  Do not pass metres.
    float coarse_lbm_origin_x = 0.0f;
    float coarse_lbm_origin_y = 0.0f;
    float coarse_lbm_origin_z = 0.0f;
    // Global actual-level rows that represent physical air.  Rows outside the
    // interval are not voxelized; the runtime owns compact ground and any
    // artificial nested-interface ghost rows.
    std::uint32_t physical_z_offset = 0u;
    std::uint32_t physical_nz = 0u;
    bool subset_triangles_by_xy = true;
};

inline bool nested_geometry_raster_spec_is_valid(
    const NestedGeometryRasterSpec& spec) noexcept {
    NestedGeometrySamplingPlan plan;
    return nested_geometry_make_sampling_plan(
               spec.level_ratio_to_coarse, spec.max_ratio_to_coarse, plan) &&
           std::isfinite(spec.coarse_lbm_origin_x) &&
           std::isfinite(spec.coarse_lbm_origin_y) &&
           std::isfinite(spec.coarse_lbm_origin_z) &&
           spec.physical_nz > 0u;
}

#if defined(CULBM_BACKEND)

// One device-resident byte mask per process-owned LBM domain.  A mask value of
// one means the corresponding physical target cell won its common-finest
// majority vote; zero means fluid or a row intentionally left to the runtime.
// The rasterizer does not touch LBM flags, ground, or interface ghost rows.
struct NestedGeometryVolumeMask {
    NestedGeometryRasterSpec spec{};
    std::vector<NestedGeometryOwnedDomain> domains;
    std::vector<Memory<uchar>> local_masks;

    bool valid() const noexcept {
        if (!nested_geometry_raster_spec_is_valid(spec) ||
            domains.empty() || domains.size() != local_masks.size()) {
            return false;
        }
        for (std::size_t i = 0u; i < domains.size(); ++i) {
            if (!domains[i].valid() ||
                local_masks[i].length() != domains[i].local_cell_count()) {
                return false;
            }
        }
        return true;
    }

    Memory<uchar>& local_mask(const std::uint32_t local_domain) {
        return local_masks.at(local_domain);
    }
    const Memory<uchar>& local_mask(const std::uint32_t local_domain) const {
        return local_masks.at(local_domain);
    }
};

// Rasterize `transformed_mesh` into an independent volume mask.  The mesh is
// read only and is expected to be the one already transformed into coarse-LBM
// coordinates by setup.  The returned masks include local halo storage for a
// simple later hand-off, but only owned physical-air rows are written.
NestedGeometryVolumeMask rasterize_nested_geometry_volume_mask(
    LBM& lbm,
    const Mesh& transformed_mesh,
    const NestedGeometryRasterSpec& spec);

#endif // CULBM_BACKEND

} // namespace culbm
