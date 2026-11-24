#pragma once

// CUDA-only vertical nesting plan. A configured height is measured from one
// fixed horizontal reference plane: the top of the compacted ground layer.
// It is deliberately not terrain-relative AGL. Adjacent levels meet on a
// shared z plane, and each 2:1 horizontal refinement maps one coarse cell to
// exactly four fine cells.

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace culbm {

struct VerticalNestingLevelPlan {
    std::uint32_t ratio_to_coarse = 1u;
    std::uint32_t nx = 0u;
    std::uint32_t ny = 0u;
    // Physical cell rows. A slab with SI thickness L has exactly L/dx rows;
    // its physical cells cover [z_begin_si,z_end_si) with centers at
    // z_begin_si+(k+0.5)*dx. This is deliberately not an endpoint-node
    // count: CULBM lattice positions are cell centers.
    std::uint32_t physical_nz = 0u;
    // One non-physical row is reserved on each vertical side. At an internal
    // interface it is the macro-exchange ghost row; at a global outer face it
    // is the normal physical-boundary row. Ghost rows do not contribute to
    // the SI slab thickness or output/volume integration.
    std::uint32_t lower_ghost_rows = 0u;
    std::uint32_t upper_ghost_rows = 0u;
    // Total rows passed to LBM: physical_nz + lower_ghost_rows +
    // upper_ghost_rows. Runtime must use physical_first_z()/physical_end_z()
    // to distinguish donor cells from ghost targets.
    std::uint32_t allocated_nz = 0u;
    double dx_si = 0.0;
    // Legacy AGL field names are retained for callers, but their semantics are
    // fixed heights above the compacted-ground top, not local terrain AGL.
    double z_begin_agl_si = 0.0;
    double z_end_agl_si = 0.0;
    // Absolute SI planes after adding VerticalNestingPlan::ground_top_si.
    double z_begin_si = 0.0;
    double z_end_si = 0.0;
    std::string label;

    std::uint32_t physical_first_z() const noexcept {
        return lower_ghost_rows;
    }
    std::uint32_t physical_end_z() const noexcept {
        return lower_ghost_rows + physical_nz; // exclusive
    }
    std::uint32_t lower_ghost_z() const noexcept {
        return lower_ghost_rows == 0u ? 0u : lower_ghost_rows - 1u;
    }
    std::uint32_t upper_ghost_z() const noexcept {
        return physical_end_z();
    }
};

struct VerticalNestingPlan {
    bool enabled = false;
    bool valid = true;
    std::string error;
    double coarse_dx_si = 0.0;
    // Fixed SI reference plane at the top of the compacted ground layer.
    double ground_top_si = 0.0;
    // Legacy name retained for the existing deck key. This is the total height
    // above ground_top_si, not a terrain-following AGL extent.
    double domain_height_agl_si = 0.0;
    std::vector<double> interface_heights_agl_si;
    std::uint32_t domains_x = 1u;
    std::uint32_t domains_y = 1u;
    std::uint32_t domains_z = 1u;
    std::vector<VerticalNestingLevelPlan> levels; // compact ground -> sky

    bool has_three_levels() const { return levels.size() == 3u; }
};

// Restriction at a 2:1 horizontal interface. Keep this explicit rather than
// replacing it with an interpolant: one coarse value is the arithmetic mean
// of precisely its four fine children.
inline float vertical_nesting_mean4(const float a, const float b,
                                    const float c, const float d) {
    return 0.25f * (a + b + c + d);
}

inline bool vertical_nesting_nearly_integer(const double value,
                                             const double tolerance = 1.0e-5) {
    if (!std::isfinite(value)) return false;
    const double scale = std::fabs(value) > 1.0 ? std::fabs(value) : 1.0;
    return std::fabs(value - std::round(value)) <= tolerance * scale;
}

inline void vertical_nesting_set_error(VerticalNestingPlan& plan,
                                       const std::string& error) {
    plan.valid = false;
    plan.error = error;
    plan.levels.clear();
}

inline bool vertical_nesting_checked_product(const std::uint32_t left,
                                             const std::uint32_t right,
                                             std::uint32_t& product) {
    const std::uint64_t value = static_cast<std::uint64_t>(left) *
                                static_cast<std::uint64_t>(right);
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }
    product = static_cast<std::uint32_t>(value);
    return true;
}

inline VerticalNestingPlan make_vertical_nesting_plan(
    const std::vector<double>& interface_heights_above_ground_si,
    const double coarse_dx_si,
    const double domain_height_above_ground_si,
    const std::uint32_t coarse_nx,
    const std::uint32_t coarse_ny,
    const std::uint32_t domains_x,
    const std::uint32_t domains_y,
    const std::uint32_t domains_z,
    const double ground_top_si = 0.0) {
    VerticalNestingPlan result;
    result.enabled = !interface_heights_above_ground_si.empty();
    result.coarse_dx_si = coarse_dx_si;
    result.ground_top_si = ground_top_si;
    result.domain_height_agl_si = domain_height_above_ground_si;
    result.interface_heights_agl_si = interface_heights_above_ground_si;
    result.domains_x = domains_x;
    result.domains_y = domains_y;
    result.domains_z = domains_z;

    if (!result.enabled) return result;
    if (interface_heights_above_ground_si.size() > 2u) {
        vertical_nesting_set_error(
            result,
            "cuda_vertical_nesting_agl accepts at most two fixed horizontal interface heights");
        return result;
    }
    if (!(std::isfinite(coarse_dx_si) && coarse_dx_si > 0.0) ||
        !(std::isfinite(domain_height_above_ground_si) &&
          domain_height_above_ground_si > 0.0) ||
        !std::isfinite(ground_top_si) || coarse_nx == 0u || coarse_ny == 0u) {
        vertical_nesting_set_error(
            result,
            "vertical nesting needs finite positive SI dimensions and a finite compact-ground reference plane");
        return result;
    }
    if (domains_x == 0u || domains_y == 0u || domains_z == 0u) {
        vertical_nesting_set_error(
            result,
            "vertical nesting domain decomposition Dx/Dy/Dz must all be positive");
        return result;
    }

    double previous_height = 0.0;
    for (const double height : interface_heights_above_ground_si) {
        if (!(std::isfinite(height) && height > previous_height &&
              height < domain_height_above_ground_si)) {
            vertical_nesting_set_error(
                result,
                "vertical nesting heights must be finite, strictly ascending, and inside the fixed height domain above compact ground");
            return result;
        }
        previous_height = height;
    }

    const auto append_level = [&](const std::uint32_t ratio,
                                  const double z_begin,
                                  const double z_end,
                                  const char* const label) -> bool {
        const double dx = coarse_dx_si / static_cast<double>(ratio);
        const double thickness = z_end - z_begin;
        const double physical_cells_real = thickness / dx;
        if (!(std::isfinite(thickness) && thickness > 0.0) ||
            !vertical_nesting_nearly_integer(physical_cells_real)) {
            vertical_nesting_set_error(
                result,
                std::string("nested ") + label +
                    " slab thickness must be an integer number of that level's SI cells");
            return false;
        }

        const long long physical_cells_ll =
            std::llround(physical_cells_real);
        if (physical_cells_ll < 2ll ||
            physical_cells_ll > static_cast<long long>(
                std::numeric_limits<std::uint32_t>::max() - 2u)) {
            vertical_nesting_set_error(
                result,
                std::string("nested ") + label +
                    " slab needs at least two physical lattice cells");
            return false;
        }

        std::uint32_t nx = 0u;
        std::uint32_t ny = 0u;
        if (!vertical_nesting_checked_product(coarse_nx, ratio, nx) ||
            !vertical_nesting_checked_product(coarse_ny, ratio, ny)) {
            vertical_nesting_set_error(
                result,
                std::string("nested ") + label +
                    " horizontal resolution overflows uint32");
            return false;
        }

        const std::uint32_t physical_nz =
            static_cast<std::uint32_t>(physical_cells_ll);
        constexpr std::uint32_t lower_ghost_rows = 1u;
        constexpr std::uint32_t upper_ghost_rows = 1u;
        const std::uint32_t allocated_nz =
            physical_nz + lower_ghost_rows + upper_ghost_rows;
        if (nx % domains_x != 0u) {
            vertical_nesting_set_error(
                result,
                std::string("nested ") + label + " Nx=" +
                    std::to_string(nx) + " is not divisible by Dx=" +
                    std::to_string(domains_x));
            return false;
        }
        if (ny % domains_y != 0u) {
            vertical_nesting_set_error(
                result,
                std::string("nested ") + label + " Ny=" +
                    std::to_string(ny) + " is not divisible by Dy=" +
                    std::to_string(domains_y));
            return false;
        }
        if (allocated_nz % domains_z != 0u) {
            vertical_nesting_set_error(
                result,
                std::string("nested ") + label + " allocated Nz=" +
                    std::to_string(allocated_nz) + " is not divisible by Dz=" +
                    std::to_string(domains_z));
            return false;
        }
        if (allocated_nz / domains_z < 2u) {
            vertical_nesting_set_error(
                result,
                std::string("nested ") + label +
                    " needs at least two allocated z rows per Dz domain so "
                    "interface ghost flags and scalar boundaries remain local");
            return false;
        }

        VerticalNestingLevelPlan level;
        level.ratio_to_coarse = ratio;
        level.nx = nx;
        level.ny = ny;
        level.physical_nz = physical_nz;
        level.lower_ghost_rows = lower_ghost_rows;
        level.upper_ghost_rows = upper_ghost_rows;
        level.allocated_nz = allocated_nz;
        level.dx_si = dx;
        level.z_begin_agl_si = z_begin;
        level.z_end_agl_si = z_end;
        level.z_begin_si = ground_top_si + z_begin;
        level.z_end_si = ground_top_si + z_end;
        level.label = label;
        result.levels.push_back(level);
        return true;
    };

    if (interface_heights_above_ground_si.size() == 1u) {
        const double height = interface_heights_above_ground_si.front();
        if (!append_level(2u, 0.0, height, "fine") ||
            !append_level(1u, height, domain_height_above_ground_si, "coarse")) {
            return result;
        }
    }
    else {
        const double low = interface_heights_above_ground_si[0];
        const double high = interface_heights_above_ground_si[1];
        if (!append_level(4u, 0.0, low, "fine") ||
            !append_level(2u, low, high, "middle") ||
            !append_level(1u, high, domain_height_above_ground_si, "coarse")) {
            return result;
        }
    }
    return result;
}

// Preserve the original convenience form for single-domain planning. New
// CUDA code should pass Dx/Dy/Dz explicitly so LBM cannot silently truncate a
// leaf resolution during construction.
inline VerticalNestingPlan make_vertical_nesting_plan(
    const std::vector<double>& interface_heights_above_ground_si,
    const double coarse_dx_si,
    const double domain_height_above_ground_si,
    const std::uint32_t coarse_nx,
    const std::uint32_t coarse_ny) {
    return make_vertical_nesting_plan(
        interface_heights_above_ground_si,
        coarse_dx_si,
        domain_height_above_ground_si,
        coarse_nx,
        coarse_ny,
        1u,
        1u,
        1u);
}

} // namespace culbm
