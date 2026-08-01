#include "cuda_nested_solver.hpp"

#include "cuda_nested_interface.hpp"
#include "lbm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

// These setup-owned kernel definitions are captured by each LBM constructor.
// Nested levels adjust the two physically scaled controls before construction
// and restore the process defaults immediately afterwards.
extern bool buffer_nudging_active;
extern int buffer_n_cells;
extern float buffer_inv_tau_lbmu;
extern bool top_sponge_active;
vector<Device_Info> smart_device_selection(uint D);

namespace culbm {
namespace {

using std::uint32_t;
using std::uint64_t;

uint64_t index_3d(const uint32_t x, const uint32_t y, const uint32_t z,
                  const uint32_t nx, const uint32_t ny) {
    return static_cast<uint64_t>(x) +
           (static_cast<uint64_t>(y) + static_cast<uint64_t>(z) * ny) * nx;
}

bool checked_product(const uint64_t a, const uint64_t b, uint64_t& result) {
    if (a != 0u && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

bool checked_u32_product(const uint32_t a, const uint32_t b, uint32_t& result) {
    const uint64_t product = static_cast<uint64_t>(a) * b;
    if (product > std::numeric_limits<uint32_t>::max()) return false;
    result = static_cast<uint32_t>(product);
    return true;
}

bool nested_device_capacity_available(
    const std::uint64_t required_per_device_mb,
    const std::uint32_t domains_x,
    const std::uint32_t domains_y,
    const std::uint32_t domains_z,
    std::string& error) {
    std::vector<Device_Info> selected_devices;
    if (culbm::mpi_is_distributed()) {
        const std::vector<Device_Info> visible_devices = get_devices(false);
        if (visible_devices.empty()) {
            error = "nested LBM VRAM preflight found no visible CUDA device";
            return false;
        }
        const std::uint32_t ordinal = culbm::mpi_cuda_device_ordinal(
            static_cast<std::uint32_t>(visible_devices.size()));
        selected_devices.push_back(visible_devices[ordinal]);
    }
    else {
        const std::uint64_t count64 =
            static_cast<std::uint64_t>(domains_x) *
            static_cast<std::uint64_t>(domains_y) *
            static_cast<std::uint64_t>(domains_z);
        if (count64 == 0u ||
            count64 > static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max())) {
            error = "nested LBM VRAM preflight received an invalid domain count";
            return false;
        }
        selected_devices =
            smart_device_selection(static_cast<std::uint32_t>(count64));
    }
    std::uint64_t available_mb =
        std::numeric_limits<std::uint64_t>::max();
    for (const Device_Info& device : selected_devices) {
        available_mb = std::min(
            available_mb, static_cast<std::uint64_t>(device.memory));
    }
    // Match mesh fitting and Memory's runtime guard: one tranche covers
    // context/module overhead and one must remain free after allocations.
    constexpr std::uint64_t safety_reserve_mb = 512u;
    const std::uint64_t usable_mb = available_mb > safety_reserve_mb
        ? available_mb - safety_reserve_mb
        : available_mb * 9u / 10u;
    if (required_per_device_mb > usable_mb) {
        error = "nested hierarchy needs " +
            std::to_string(required_per_device_mb) +
            " MB per selected GPU, but only " +
            std::to_string(available_mb) +
            " MB is installed (" +
            std::to_string(safety_reserve_mb) +
            " MB reserved for CUDA runtime/module and allocation safety)";
        return false;
    }
    return true;
}

uint32_t round_up_multiple(const uint32_t value, const uint32_t multiple,
                           bool& ok) {
    ok = multiple > 0u;
    if (!ok) return 0u;
    const uint64_t rounded =
        (static_cast<uint64_t>(value) + multiple - 1u) / multiple * multiple;
    ok = rounded <= std::numeric_limits<uint32_t>::max();
    return ok ? static_cast<uint32_t>(rounded) : 0u;
}

float fine_horizontal_coordinate_from_coarse(const float coarse_coordinate,
                                             const uint32_t ratio) {
    // Horizontal mesh coordinates use the ordinary cell-centred domain
    // convention: the coarse boundary at -0.5 maps to fine -0.5.
    return (coarse_coordinate + 0.5f) * static_cast<float>(ratio) - 0.5f;
}

float fine_vertical_coordinate_from_coarse(const float coarse_coordinate,
                                           const uint32_t ratio) {
    // z is different after compact-ground setup. Coarse row 0 is the one
    // solid ground row and the air reference plane is its upper face at
    // coarse z=+0.5. A finest hierarchy also owns exactly one ground row,
    // so that same plane must map to fine z=+0.5, not to r-0.5. This keeps
    // generated base material below the physical fine-air slab for r=2/4.
    return (coarse_coordinate - 0.5f) * static_cast<float>(ratio) + 0.5f;
}

float3 fine_coordinate_from_coarse(const float3& coarse_coordinate,
                                   const uint32_t ratio) {
    return float3(
        fine_horizontal_coordinate_from_coarse(coarse_coordinate.x, ratio),
        fine_horizontal_coordinate_from_coarse(coarse_coordinate.y, ratio),
        fine_vertical_coordinate_from_coarse(coarse_coordinate.z, ratio));
}

std::unique_ptr<Mesh> make_common_finest_mesh(const Mesh& source,
                                               const uint32_t ratio) {
    auto result = std::make_unique<Mesh>(
        source.triangle_number, fine_coordinate_from_coarse(source.center, ratio));
    for (uint32_t triangle = 0u; triangle < source.triangle_number; ++triangle) {
        result->p0[triangle] = fine_coordinate_from_coarse(source.p0[triangle], ratio);
        result->p1[triangle] = fine_coordinate_from_coarse(source.p1[triangle], ratio);
        result->p2[triangle] = fine_coordinate_from_coarse(source.p2[triangle], ratio);
    }
    result->find_bounds();
    return result;
}

struct CommonFinestRasterLayout {
    uint32_t ratio = 1u;
    uint32_t nx = 0u;
    uint32_t ny = 0u;
    uint32_t nz = 0u;
    float stl_top_coarse_lbm_z = 0.0f;
    float terrain_top_common_finest_z =
        -std::numeric_limits<float>::max();
    std::unique_ptr<Mesh> mesh;
};

bool make_common_finest_raster_layout(
    const VerticalNestingPlan& plan,
    const Mesh& transformed_mesh,
    const uint32_t domains_x,
    const uint32_t domains_y,
    const uint32_t domains_z,
    const NestedCommonFinestTerrain* terrain,
    CommonFinestRasterLayout& layout,
    std::string& error) {
    layout = CommonFinestRasterLayout{};
    if (plan.levels.empty() || transformed_mesh.triangle_number == 0u) {
        error = "nested geometry needs a non-empty STL mesh and at least one planned slab";
        return false;
    }
    const uint32_t finest_ratio = plan.levels.front().ratio_to_coarse;
    if (finest_ratio != 2u && finest_ratio != 4u) {
        error = "nested geometry supports only a 2x or 4x finest refinement ratio";
        return false;
    }
    if (!checked_u32_product(
            plan.levels.back().nx, finest_ratio, layout.nx) ||
        !checked_u32_product(
            plan.levels.back().ny, finest_ratio, layout.ny)) {
        error = "common-finest geometry dimensions overflow uint32";
        return false;
    }
    if (terrain &&
        (terrain->nx != layout.nx || terrain->ny != layout.ny ||
         !terrain->valid())) {
        error = "common-finest terrain dimensions do not match the nested finest grid";
        return false;
    }

    layout.ratio = finest_ratio;
    layout.stl_top_coarse_lbm_z = transformed_mesh.pmax.z;
    layout.mesh = make_common_finest_mesh(
        transformed_mesh, finest_ratio);
    const float mesh_top = layout.mesh->pmax.z;
    if (!std::isfinite(mesh_top)) {
        error = "STL highest z is non-finite after common-finest coordinate conversion";
        return false;
    }
    layout.terrain_top_common_finest_z =
        terrain ? terrain->highest_solid_z()
                : -std::numeric_limits<float>::max();
    const float top = std::max(
        mesh_top, layout.terrain_top_common_finest_z);
    const double rows_real =
        std::max(1.0, std::ceil(static_cast<double>(top) + 0.5));
    const double finest_dx_si =
        plan.coarse_dx_si / static_cast<double>(finest_ratio);
    const double active_rows_real =
        1.0 + plan.domain_height_agl_si / finest_dx_si;
    if (!vertical_nesting_nearly_integer(active_rows_real) ||
        active_rows_real <= 0.0 ||
        active_rows_real >
            static_cast<double>(std::numeric_limits<uint32_t>::max()) ||
        rows_real >
            static_cast<double>(std::numeric_limits<uint32_t>::max())) {
        error = "STL highest z cannot be represented by the common-finest CUDA raster";
        return false;
    }
    const uint32_t active_rows =
        static_cast<uint32_t>(std::llround(active_rows_real));
    const uint32_t unclamped_rows =
        static_cast<uint32_t>(rows_real);
    const uint32_t requested_rows =
        std::min(unclamped_rows, active_rows);
    bool rounded_ok = false;
    layout.nz = round_up_multiple(
        requested_rows, domains_z, rounded_ok);
    if (!rounded_ok || layout.nz == 0u) {
        error = "common-finest geometry Nz cannot be padded for the configured Dz";
        return false;
    }
    if (domains_x == 0u || domains_y == 0u ||
        layout.nx % domains_x != 0u ||
        layout.ny % domains_y != 0u) {
        error = "common-finest geometry dimensions are not divisible by the configured Dx/Dy";
        return false;
    }
    return true;
}

uint32_t nearest_seed_xy(const uint32_t fine_or_level_index,
                         const uint32_t ratio_to_coarse,
                         const uint32_t coarse_extent) {
    if (coarse_extent == 0u) return 0u;
    return std::min(coarse_extent - 1u, fine_or_level_index / ratio_to_coarse);
}

uint32_t nearest_seed_z_for_height(const double height_above_ground_si,
                                   const double coarse_dx_si,
                                   const uint32_t coarse_nz) {
    if (coarse_nz == 0u) return 0u;
    // Seed row 0 is the compacted-ground boundary row. Its upper face is the
    // SI reference plane. Seed row 1 is centred at +0.5 coarse cells above
    // that plane, which gives this inverse map for physical slab cell centres.
    const double row = height_above_ground_si / coarse_dx_si + 0.5;
    const long long rounded = std::llround(row);
    if (rounded <= 0ll) return 0u;
    const uint64_t z = static_cast<uint64_t>(rounded);
    return z >= coarse_nz ? coarse_nz - 1u : static_cast<uint32_t>(z);
}

void set_solid_flag(unsigned char& flag) {
    flag = static_cast<unsigned char>((flag & ~TYPE_E) | TYPE_S);
}

void set_fluid_flag(unsigned char& flag) {
    flag = static_cast<unsigned char>(flag & ~TYPE_S);
}

bool make_common_finest_masks(
    const VerticalNestingPlan& plan,
    const CommonFinestRasterLayout& layout,
    const uint32_t domains_x,
    const uint32_t domains_y,
    const uint32_t domains_z,
    const NestedLbmPhysics& physics,
    std::vector<std::vector<unsigned char>>& masks,
    NestedGeometrySummary& summary,
    std::string& error,
    const NestedCommonFinestTerrain* terrain,
    const float compact_base_top_coarse_lbm) {
    const uint32_t finest_ratio = layout.ratio;
    const uint32_t raster_nx = layout.nx;
    const uint32_t raster_ny = layout.ny;
    const uint32_t raster_nz = layout.nz;
    const float terrain_top = layout.terrain_top_common_finest_z;
    const double finest_dx_si =
        plan.coarse_dx_si / static_cast<double>(finest_ratio);

    const float fine_nu = physics.coarse_nu * static_cast<float>(finest_ratio);
    const float fine_alpha = physics.coarse_alpha * static_cast<float>(finest_ratio);
    if (!(std::isfinite(fine_nu) && fine_nu > 0.0f &&
          std::isfinite(fine_alpha) && fine_alpha >= 0.0f)) {
        error = "common-finest geometry received invalid coarse LBM transport coefficients";
        return false;
    }

    LBM raster(uint3(raster_nx, raster_ny, raster_nz),
               domains_x, domains_y, domains_z,
               fine_nu, 0.0f, 0.0f, 0.0f, 0.0f,
               fine_alpha, physics.beta);
    raster.voxelize_mesh_on_device(layout.mesh.get(), TYPE_S);
    // voxelize_mesh_on_device() fetches flags when the lattice is not yet
    // initialized. Make the read explicit nevertheless so this contract stays
    // correct if its implementation changes.
    raster.flags.read_from_device();

    const bool remove_compact_base = std::isfinite(compact_base_top_coarse_lbm);
    const float compact_base_top_common_finest = remove_compact_base
        ? fine_vertical_coordinate_from_coarse(
            compact_base_top_coarse_lbm, finest_ratio)
        : std::numeric_limits<float>::quiet_NaN();
    uint64_t compact_base_removed_count = 0u;
    if (remove_compact_base) {
        for (uint32_t z = 0u; z < raster_nz; ++z) {
            if (static_cast<float>(z) >= compact_base_top_common_finest) continue;
            for (uint32_t y = 0u; y < raster_ny; ++y) {
                for (uint32_t x = 0u; x < raster_nx; ++x) {
                    const uint64_t n = index_3d(x, y, z, raster_nx, raster_ny);
                    if ((raster.flags[n] & TYPE_S) == 0u) continue;
                    raster.flags[n] = static_cast<unsigned char>(
                        raster.flags[n] & ~TYPE_S);
                    ++compact_base_removed_count;
                }
            }
        }
    }

    uint64_t terrain_solid_count = 0u;
    if (terrain) {
        for (uint32_t z = 0u; z < raster_nz; ++z) {
            for (uint32_t y = 0u; y < raster_ny; ++y) {
                for (uint32_t x = 0u; x < raster_nx; ++x) {
                    const uint64_t column = index_3d(x, y, 0u, raster_nx, raster_ny);
                    const uint64_t n = index_3d(x, y, z, raster_nx, raster_ny);
                    if (static_cast<float>(z) < terrain->solid_below_z[
                            static_cast<std::size_t>(column)]) {
                        if ((raster.flags[n] & TYPE_S) == 0u) ++terrain_solid_count;
                        raster.flags[n] = static_cast<unsigned char>(
                            (raster.flags[n] & ~TYPE_E) | TYPE_S);
                    }
                }
            }
        }
    }

    uint64_t finest_solid_count = 0u;
    for (uint64_t n = 0u; n < raster.get_N(); ++n) {
        if ((raster.flags[n] & TYPE_S) != 0u) ++finest_solid_count;
    }

    masks.clear();
    masks.resize(plan.levels.size());
    summary.solid_cells_per_plan_level.assign(plan.levels.size(), 0u);
    for (std::size_t level_index = 0u; level_index < plan.levels.size(); ++level_index) {
        const VerticalNestingLevelPlan& level = plan.levels[level_index];
        const uint32_t q = finest_ratio / level.ratio_to_coarse;
        if (q == 0u || finest_ratio % level.ratio_to_coarse != 0u) {
            error = "nested geometry level ratio is incompatible with the common-finest ratio";
            return false;
        }
        uint64_t cell_count = 0u;
        uint64_t xy_count = 0u;
        if (!checked_product(level.nx, level.ny, xy_count) ||
            !checked_product(xy_count, level.physical_nz, cell_count) ||
            cell_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
            error = "nested geometry target mask size overflows addressable memory";
            return false;
        }
        masks[level_index].assign(static_cast<std::size_t>(cell_count), 0u);

        const double begin_fine_cells_real = level.z_begin_agl_si / finest_dx_si;
        if (!vertical_nesting_nearly_integer(begin_fine_cells_real)) {
            error = "nested geometry interface is not aligned to the common-finest logical z lattice";
            return false;
        }
        const uint64_t begin_fine_cells =
            static_cast<uint64_t>(std::llround(begin_fine_cells_real));
        // Common-finest z=0 is the single compact ground row. Every
        // physical-air slab begins immediately above that row plus its SI
        // height measured in finest cells; it is not offset by a coarse
        // ground thickness when r=2/4.
        const uint64_t first_physical_fine_z = 1u + begin_fine_cells;
        const uint64_t votes_per_cell = static_cast<uint64_t>(q) * q * q;

        for (uint32_t z = 0u; z < level.physical_nz; ++z) {
            for (uint32_t y = 0u; y < level.ny; ++y) {
                for (uint32_t x = 0u; x < level.nx; ++x) {
                    uint64_t solid_votes = 0u;
                    for (uint32_t qz = 0u; qz < q; ++qz) {
                        const uint64_t fz = first_physical_fine_z +
                            static_cast<uint64_t>(z) * q + qz;
                        if (fz >= raster_nz) continue; // above STL top: logical fluid
                        for (uint32_t qy = 0u; qy < q; ++qy) {
                            const uint32_t fy = y * q + qy;
                            for (uint32_t qx = 0u; qx < q; ++qx) {
                                const uint32_t fx = x * q + qx;
                                const uint64_t fine_index = index_3d(
                                    fx, fy, static_cast<uint32_t>(fz), raster_nx, raster_ny);
                                if ((raster.flags[fine_index] & TYPE_S) != 0u) ++solid_votes;
                            }
                        }
                    }
                    const uint64_t target = index_3d(x, y, z, level.nx, level.ny);
                    if (nested_cell_is_solid_from_finest_count(solid_votes, votes_per_cell)) {
                        masks[level_index][static_cast<std::size_t>(target)] = 1u;
                        ++summary.solid_cells_per_plan_level[level_index];
                    }
                }
            }
        }
    }

    summary.common_finest_ratio = finest_ratio;
    summary.raster_nx = raster_nx;
    summary.raster_ny = raster_ny;
    summary.raster_nz = raster_nz;
    summary.stl_top_coarse_lbm_z = layout.stl_top_coarse_lbm_z;
    summary.terrain_top_common_finest_z = terrain_top;
    summary.compact_base_top_common_finest_z = compact_base_top_common_finest;
    summary.common_finest_solid_cells = finest_solid_count;
    summary.terrain_solid_cells = terrain_solid_count;
    summary.compact_base_solid_cells_removed = compact_base_removed_count;
    return true;
}

struct SeedEquilibriumFaces {
    bool west = false;
    bool east = false;
    bool south = false;
    bool north = false;
};

struct SeedSolidTemperatureColumns {
    std::vector<unsigned char> prescribed;
    std::vector<float> temperature;
};

SeedEquilibriumFaces inspect_seed_equilibrium_faces(const NestedSeedState& seed) {
    SeedEquilibriumFaces result;
    if (!seed.valid()) return result;
    const auto is_equilibrium = [&](const uint32_t x,
                                    const uint32_t y,
                                    const uint32_t z) {
        const uint64_t n = index_3d(x, y, z, seed.nx, seed.ny);
        return (seed.flags[static_cast<std::size_t>(n)] & TYPE_E) != 0u;
    };
    // Only inspect the four actual side surfaces.  The former helper scanned
    // the complete coarse volume once per face and again for every nested
    // level, which becomes material for production terrain grids.
    for (uint32_t z = 0u; z < seed.nz; ++z) {
        for (uint32_t y = 0u; y < seed.ny; ++y) {
            if (!result.west && is_equilibrium(0u, y, z)) result.west = true;
            if (!result.east && is_equilibrium(seed.nx - 1u, y, z)) result.east = true;
        }
    }
    for (uint32_t z = 0u; z < seed.nz; ++z) {
        for (uint32_t x = 0u; x < seed.nx; ++x) {
            if (!result.south && is_equilibrium(x, 0u, z)) result.south = true;
            if (!result.north && is_equilibrium(x, seed.ny - 1u, z)) result.north = true;
        }
    }
    return result;
}

SeedSolidTemperatureColumns inspect_seed_solid_temperature_columns(
    const NestedSeedState& seed) {
    SeedSolidTemperatureColumns result;
#ifdef TEMPERATURE
    if (!seed.valid() || seed.temperature.empty()) return result;
    const std::size_t column_count =
        static_cast<std::size_t>(seed.nx) * static_cast<std::size_t>(seed.ny);
    result.prescribed.assign(column_count, 0u);
    result.temperature.assign(column_count, 1.0f);
    for (uint32_t y = 0u; y < seed.ny; ++y) {
        for (uint32_t x = 0u; x < seed.nx; ++x) {
            const std::size_t column =
                static_cast<std::size_t>(x) +
                static_cast<std::size_t>(y) * static_cast<std::size_t>(seed.nx);
            for (uint32_t z = 0u; z < seed.nz; ++z) {
                const std::size_t n = static_cast<std::size_t>(
                    index_3d(x, y, z, seed.nx, seed.ny));
                const unsigned char flag = seed.flags[n];
                if ((flag & static_cast<unsigned char>(TYPE_S | TYPE_T)) !=
                    static_cast<unsigned char>(TYPE_S | TYPE_T) ||
                    !std::isfinite(seed.temperature[n])) {
                    continue;
                }
                // Production ground-temperature mapping deliberately assigns
                // one value to the complete solid column.  Retaining the first
                // valid prescribed value also gives a deterministic fallback
                // for hand-authored seed states.
                result.prescribed[column] = 1u;
                result.temperature[column] = seed.temperature[n];
                break;
            }
        }
    }
#else
    (void)seed;
#endif
    return result;
}

void copy_seed_cell(const NestedSeedState& seed, const uint64_t source,
                    LBM& target, const uint64_t destination) {
    target.rho[destination] = seed.rho[static_cast<std::size_t>(source)];
    target.u.x[destination] = seed.ux[static_cast<std::size_t>(source)];
    target.u.y[destination] = seed.uy[static_cast<std::size_t>(source)];
    target.u.z[destination] = seed.uz[static_cast<std::size_t>(source)];
#ifdef TEMPERATURE
    target.T[destination] = seed.temperature.empty()
        ? 1.0f : seed.temperature[static_cast<std::size_t>(source)];
#endif
    target.flags[destination] = seed.flags[static_cast<std::size_t>(source)];
}

void populate_level_from_seed(const NestedSeedState& seed,
                              const VerticalNestingPlan& complete_plan,
                              const VerticalNestingLevelPlan& level_plan,
                              const std::vector<unsigned char>& physical_mask,
                              const SeedEquilibriumFaces& seed_equilibrium_faces,
                              const SeedSolidTemperatureColumns&
                                  seed_solid_temperature_columns,
                              const bool is_finest_ground_level,
                              const bool is_coarsest_sky_level,
                              LBM& target) {
    const uint32_t ratio = level_plan.ratio_to_coarse;
    const uint32_t nx = target.get_Nx();
    const uint32_t ny = target.get_Ny();
    const uint32_t nz = target.get_Nz();
    const bool west_equilibrium = seed_equilibrium_faces.west;
    const bool east_equilibrium = seed_equilibrium_faces.east;
    const bool south_equilibrium = seed_equilibrium_faces.south;
    const bool north_equilibrium = seed_equilibrium_faces.north;
    for (uint32_t z = 0u; z < nz; ++z) {
        const bool physical = z >= level_plan.physical_first_z() &&
            z < level_plan.physical_end_z();
        const uint32_t physical_z = physical ? z - level_plan.physical_first_z() : 0u;
        uint32_t source_z = 0u;
        if (physical) {
            const double h = level_plan.z_begin_agl_si +
                (static_cast<double>(physical_z) + 0.5) * level_plan.dx_si;
            source_z = nearest_seed_z_for_height(
                h, complete_plan.coarse_dx_si, seed.nz);
        }
        else if (z == level_plan.upper_ghost_z() && is_coarsest_sky_level) {
            source_z = seed.nz - 1u;
        }
        else if (z == level_plan.lower_ghost_z() && is_finest_ground_level) {
            source_z = 0u;
        }
        else {
            // Internal interface rows are ghost targets. Give them benign
            // values first; NestedMacroPlaneBridge replaces both aperture and
            // macro state immediately after all blocks initialize.
            source_z = std::min(seed.nz - 1u, 1u);
        }

        for (uint32_t y = 0u; y < ny; ++y) {
            const uint32_t source_y = nearest_seed_xy(y, ratio, seed.ny);
            for (uint32_t x = 0u; x < nx; ++x) {
                const uint32_t source_x = nearest_seed_xy(x, ratio, seed.nx);
                const uint64_t source = index_3d(source_x, source_y, source_z,
                                                 seed.nx, seed.ny);
                const uint64_t destination = index_3d(x, y, z, nx, ny);
                copy_seed_cell(seed, source, target, destination);

                if (physical) {
                    const uint64_t mask_index = index_3d(
                        x, y, physical_z, nx, ny);
                    unsigned char& flag = target.flags[destination];
                    if (physical_mask[static_cast<std::size_t>(mask_index)] != 0u) {
#ifdef TEMPERATURE
                        // The common-finest geometry is authoritative.  Clear
                        // coarse-cell thermal classification first, then
                        // restore a prescribed solid temperature only from an
                        // actual prescribed solid in this seed column.
                        flag = static_cast<unsigned char>(flag & ~TYPE_T);
                        const std::size_t column =
                            static_cast<std::size_t>(source_x) +
                            static_cast<std::size_t>(source_y) *
                                static_cast<std::size_t>(seed.nx);
                        const unsigned char source_flag =
                            seed.flags[static_cast<std::size_t>(source)];
                        const bool source_is_prescribed_solid =
                            (source_flag &
                             static_cast<unsigned char>(TYPE_S | TYPE_T)) ==
                                static_cast<unsigned char>(TYPE_S | TYPE_T) &&
                            !seed.temperature.empty() &&
                            std::isfinite(seed.temperature[
                                static_cast<std::size_t>(source)]);
                        if (source_is_prescribed_solid) {
                            flag = static_cast<unsigned char>(flag | TYPE_T);
                        }
                        else if (
                            column < seed_solid_temperature_columns.prescribed.size() &&
                            seed_solid_temperature_columns.prescribed[column] != 0u) {
                            target.T[destination] =
                                seed_solid_temperature_columns.temperature[column];
                            flag = static_cast<unsigned char>(flag | TYPE_T);
                        }
#endif
                        set_solid_flag(flag);
                        target.u.x[destination] = 0.0f;
                        target.u.y[destination] = 0.0f;
                        target.u.z[destination] = 0.0f;
                    }
                    else {
                        // Do not recursively preserve the old uniform mesh
                        // classification. The direct common-finest vote is
                        // authoritative for STL-derived geometry.
                        set_fluid_flag(flag);
#ifdef TEMPERATURE
                        // A coarse prescribed solid may contain finer air
                        // children.  Never turn those children into internal
                        // fixed-temperature points merely by copying TYPE_T.
                        const unsigned char source_flag =
                            seed.flags[static_cast<std::size_t>(source)];
                        flag = static_cast<unsigned char>(flag & ~(TYPE_E | TYPE_T));
#else
                        flag = static_cast<unsigned char>(flag & ~TYPE_E);
#endif
                        // A coarse terrain cell can have hidden a valid
                        // side-boundary child. Restore its equilibrium type
                        // whenever that outer face exists in the seed case;
                        // the direct common-finest mask above remains the
                        // sole authority for whether this child is solid.
                        if ((x == 0u && west_equilibrium) ||
                            (x + 1u == nx && east_equilibrium) ||
                            (y == 0u && south_equilibrium) ||
                            (y + 1u == ny && north_equilibrium)) {
                            flag = static_cast<unsigned char>(flag | TYPE_E);
#ifdef TEMPERATURE
                            // Reuse a seed temperature only when it came from
                            // an actual fluid boundary, never from a coarse
                            // solid that the finer geometry resolved as air.
                            if ((source_flag & TYPE_S) == 0u &&
                                (source_flag & TYPE_T) != 0u &&
                                !seed.temperature.empty() &&
                                std::isfinite(seed.temperature[
                                    static_cast<std::size_t>(source)])) {
                                flag = static_cast<unsigned char>(flag | TYPE_T);
                            }
#endif
                        }
                    }
                }
                else if (is_internal_nested_ghost_row(
                             z,
                             level_plan.lower_ghost_z(),
                             level_plan.upper_ghost_z(),
                             is_finest_ground_level,
                             is_coarsest_sky_level)) {
                    set_solid_flag(target.flags[destination]);
                    target.u.x[destination] = 0.0f;
                    target.u.y[destination] = 0.0f;
                    target.u.z[destination] = 0.0f;
                }
            }
        }
    }

    if (is_finest_ground_level) {
        // The compact base has exactly one sealed solid boundary row at every
        // resolution. It is intentionally outside the SI physical slab.
        for (uint32_t y = 0u; y < ny; ++y) {
            for (uint32_t x = 0u; x < nx; ++x) {
                const uint64_t n = index_3d(x, y, level_plan.lower_ghost_z(), nx, ny);
                set_solid_flag(target.flags[n]);
                target.u.x[n] = 0.0f;
                target.u.y[n] = 0.0f;
                target.u.z[n] = 0.0f;
            }
        }
    }
}

} // namespace

uint64_t NestedSeedState::cell_count() const noexcept {
    return static_cast<uint64_t>(nx) * ny * nz;
}

bool NestedSeedState::valid() const noexcept {
    const uint64_t count = cell_count();
    return nx > 0u && ny > 0u && nz > 0u &&
           rho.size() == count && ux.size() == count && uy.size() == count &&
           uz.size() == count && flags.size() == count &&
           (temperature.empty() || temperature.size() == count);
}

bool NestedCommonFinestTerrain::valid() const noexcept {
    const uint64_t count = static_cast<uint64_t>(nx) * ny;
    if (nx == 0u || ny == 0u || solid_below_z.size() != count) return false;
    for (const float z : solid_below_z) {
        if (!std::isfinite(z)) return false;
    }
    return true;
}

float NestedCommonFinestTerrain::highest_solid_z() const noexcept {
    if (!valid()) return -std::numeric_limits<float>::max();
    return *std::max_element(solid_below_z.begin(), solid_below_z.end());
}

std::uint64_t nested_hierarchy_vram_required_mb_per_device(
    const VerticalNestingPlan& plan,
    const std::uint32_t domains_x,
    const std::uint32_t domains_y,
    const std::uint32_t domains_z) noexcept {
    std::uint64_t total = 0u;
    for (const VerticalNestingLevelPlan& level : plan.levels) {
        const std::uint64_t level_mb = vram_required_mb_per_device(
            level.nx, level.ny, level.allocated_nz,
            domains_x, domains_y, domains_z);
        if (level_mb > std::numeric_limits<std::uint64_t>::max() - total) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        total += level_mb;
    }
    return total;
}

std::uint64_t nested_full_height_raster_vram_required_mb_per_device(
    const VerticalNestingPlan& plan,
    const std::uint32_t domains_x,
    const std::uint32_t domains_y,
    const std::uint32_t domains_z) noexcept {
    constexpr std::uint64_t invalid =
        std::numeric_limits<std::uint64_t>::max();
    if (!plan.valid || plan.levels.empty() ||
        domains_x == 0u || domains_y == 0u || domains_z == 0u ||
        !(std::isfinite(plan.coarse_dx_si) && plan.coarse_dx_si > 0.0) ||
        !(std::isfinite(plan.domain_height_agl_si) &&
          plan.domain_height_agl_si >= 0.0)) {
        return invalid;
    }
    const std::uint32_t ratio =
        plan.levels.front().ratio_to_coarse;
    if (ratio != 2u && ratio != 4u) return invalid;
    std::uint32_t nx = 0u, ny = 0u;
    if (!checked_u32_product(
            plan.levels.back().nx, ratio, nx) ||
        !checked_u32_product(
            plan.levels.back().ny, ratio, ny) ||
        nx % domains_x != 0u || ny % domains_y != 0u) {
        return invalid;
    }
    const double finest_dx_si =
        plan.coarse_dx_si / static_cast<double>(ratio);
    const double rows_real =
        std::ceil(1.0 + plan.domain_height_agl_si / finest_dx_si);
    if (!(std::isfinite(rows_real) && rows_real > 0.0) ||
        rows_real >
            static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
        return invalid;
    }
    bool rounded_ok = false;
    const std::uint32_t nz = round_up_multiple(
        static_cast<std::uint32_t>(rows_real),
        domains_z, rounded_ok);
    if (!rounded_ok || nz == 0u) return invalid;
    return vram_required_mb_per_device(
        nx, ny, nz, domains_x, domains_y, domains_z);
}

NestedSeedState capture_nested_seed_state(LBM& lbm) {
    NestedSeedState result;
    result.nx = lbm.get_Nx();
    result.ny = lbm.get_Ny();
    result.nz = lbm.get_Nz();
    const uint64_t count = result.cell_count();
    result.rho.resize(static_cast<std::size_t>(count));
    result.ux.resize(static_cast<std::size_t>(count));
    result.uy.resize(static_cast<std::size_t>(count));
    result.uz.resize(static_cast<std::size_t>(count));
    result.flags.resize(static_cast<std::size_t>(count));
#ifdef TEMPERATURE
    result.temperature.resize(static_cast<std::size_t>(count));
#endif
    for (uint64_t n = 0u; n < count; ++n) {
        result.rho[static_cast<std::size_t>(n)] = lbm.rho[n];
        result.ux[static_cast<std::size_t>(n)] = lbm.u.x[n];
        result.uy[static_cast<std::size_t>(n)] = lbm.u.y[n];
        result.uz[static_cast<std::size_t>(n)] = lbm.u.z[n];
        result.flags[static_cast<std::size_t>(n)] = lbm.flags[n];
#ifdef TEMPERATURE
        result.temperature[static_cast<std::size_t>(n)] = lbm.T[n];
#endif
    }
    return result;
}

std::vector<NestedLbmLevel> NestedVerticalLbm::make_scheduler_levels() const {
    std::vector<NestedLbmLevel> result;
    result.reserve(levels.size());
    for (std::size_t i = 0u; i < levels.size(); ++i) {
        const NestedOwnedLbmLevel& owned = levels[i];
        const VerticalNestingLevelPlan& level = owned.plan;
        NestedLbmLevel scheduler_level;
        scheduler_level.lbm = owned.lbm.get();
        scheduler_level.refinement_to_parent = i == 0u ? 1u : 2u;
        scheduler_level.dt_si = coarse_dt_si /
            static_cast<double>(level.ratio_to_coarse);
        scheduler_level.label = level.label;
        scheduler_level.lower_ghost_z = level.lower_ghost_z();
        scheduler_level.upper_ghost_z = level.upper_ghost_z();
        scheduler_level.physical_first_z = level.physical_first_z();
        scheduler_level.physical_end_z = level.physical_end_z();
        result.push_back(std::move(scheduler_level));
    }
    return result;
}

bool NestedVerticalLbm::valid() const noexcept {
    if (!plan.valid || !plan.enabled || levels.empty()) return false;
    for (const NestedOwnedLbmLevel& level : levels) {
        if (!level.lbm || level.plan.allocated_nz != level.lbm->get_Nz()) return false;
    }
    return true;
}

uint64_t NestedStitchedFields::cell_count() const noexcept {
    return static_cast<uint64_t>(nx) * ny * nz;
}

bool NestedStitchedFields::valid() const noexcept {
    const uint64_t count = cell_count();
    return nx > 0u && ny > 0u && nz > 0u && dx_si > 0.0f &&
           rho.size() == count && u.size() == 3u * count &&
           flags.size() == count &&
           (temperature.empty() || temperature.size() == count);
}

bool build_nested_vertical_lbm(const VerticalNestingPlan& plan,
                               const NestedSeedState& seed,
                               const Mesh& transformed_mesh,
                               const uint32_t domains_x,
                               const uint32_t domains_y,
                               const uint32_t domains_z,
                               const NestedLbmPhysics& physics,
                               NestedVerticalLbm& out,
                               std::string& error,
                               const NestedCommonFinestTerrain* terrain,
                               const float compact_base_top_coarse_lbm) {
    out = NestedVerticalLbm{};
    error.clear();
    if (!plan.enabled || !plan.valid || plan.levels.size() < 2u ||
        plan.levels.size() > 3u) {
        error = "nested LBM build needs a valid enabled two- or three-level plan";
        return false;
    }
    if (!seed.valid()) {
        error = "nested LBM build received an invalid uniform seed state";
        return false;
    }
    if (seed.nx != plan.levels.back().nx || seed.ny != plan.levels.back().ny) {
        error = "nested LBM seed dimensions do not match the configured coarse horizontal grid";
        return false;
    }
    if (domains_x==0u || domains_y==0u || domains_z==0u) {
        error = "nested LBM build received a zero Cartesian domain count";
        return false;
    }
    if (!(std::isfinite(physics.coarse_dt_si) && physics.coarse_dt_si > 0.0) ||
        !(std::isfinite(physics.coarse_nu) && physics.coarse_nu > 0.0f) ||
        !(std::isfinite(physics.coarse_alpha) && physics.coarse_alpha >= 0.0f) ||
        !std::isfinite(physics.beta) ||
        !std::isfinite(physics.coarse_buoyancy_gx) ||
        !std::isfinite(physics.coarse_buoyancy_gy) ||
        !std::isfinite(physics.coarse_buoyancy_gz) ||
        physics.temperature_outlet_face<0 ||
        physics.temperature_outlet_face>4) {
        error = "nested LBM build received invalid coarse SI/LBM transport scaling";
        return false;
    }
    if (physics.temperature_outlet_face != 0) {
        for (const VerticalNestingLevelPlan& level : plan.levels) {
            const bool x_normal =
                physics.temperature_outlet_face ==
                    static_cast<int>(Temperature_Outlet_Face::West) ||
                physics.temperature_outlet_face ==
                    static_cast<int>(Temperature_Outlet_Face::East);
            const uint32_t normal_cells =
                x_normal ? level.nx / domains_x : level.ny / domains_y;
            if (normal_cells < 2u) {
                error = "nested LBM scalar outlet needs at least two physical "
                        "cells per domain along its normal";
                return false;
            }
        }
    }
    CommonFinestRasterLayout raster_layout;
    if (!make_common_finest_raster_layout(
            plan, transformed_mesh,
            domains_x, domains_y, domains_z,
            terrain, raster_layout, error)) {
        return false;
    }

    const std::uint64_t hierarchy_core_mb =
        nested_hierarchy_vram_required_mb_per_device(
            plan, domains_x, domains_y, domains_z);
    if (hierarchy_core_mb ==
            std::numeric_limits<std::uint64_t>::max() ||
        physics.additional_gpu_memory_per_device_mb >
            std::numeric_limits<std::uint64_t>::max() - hierarchy_core_mb) {
        error = "nested hierarchy VRAM estimate overflowed uint64";
        return false;
    }
    const std::uint64_t hierarchy_total_mb =
        hierarchy_core_mb +
        physics.additional_gpu_memory_per_device_mb;
    const std::uint64_t raster_core_mb =
        vram_required_mb_per_device(
            raster_layout.nx, raster_layout.ny, raster_layout.nz,
            domains_x, domains_y, domains_z);
    const std::uint64_t voxelization_mb =
        voxelization_transient_vram_mb_per_device(
            transformed_mesh.triangle_number);
    if (voxelization_mb ==
            std::numeric_limits<std::uint64_t>::max() ||
        voxelization_mb >
            std::numeric_limits<std::uint64_t>::max() - raster_core_mb) {
        error = "common-finest raster VRAM estimate overflowed uint64";
        return false;
    }
    const std::uint64_t raster_peak_mb =
        raster_core_mb + voxelization_mb;
    const std::uint64_t construction_peak_mb =
        std::max(hierarchy_total_mb, raster_peak_mb);
    if (!nested_device_capacity_available(
            construction_peak_mb,
            domains_x, domains_y, domains_z, error)) {
        return false;
    }

    std::vector<std::vector<unsigned char>> physical_masks;
    NestedGeometrySummary geometry;
    if (!make_common_finest_masks(plan, raster_layout,
                                  domains_x, domains_y, domains_z,
                                  physics, physical_masks, geometry, error,
                                  terrain, compact_base_top_coarse_lbm)) {
        return false;
    }

    out.plan = plan;
    out.geometry = std::move(geometry);
    out.coarse_dt_si = physics.coarse_dt_si;
    out.levels.reserve(plan.levels.size());
    const SeedEquilibriumFaces seed_equilibrium_faces =
        inspect_seed_equilibrium_faces(seed);
    const SeedSolidTemperatureColumns seed_solid_temperature_columns =
        inspect_seed_solid_temperature_columns(seed);
    const bool saved_buffer_nudging_active = buffer_nudging_active;
    const int saved_buffer_n_cells = buffer_n_cells;
    const float saved_buffer_inv_tau_lbmu = buffer_inv_tau_lbmu;
    const bool saved_top_sponge_active = top_sponge_active;
    // Reverse compact-ground -> sky planning order into scheduler order.
    for (std::size_t reverse = plan.levels.size(); reverse > 0u; --reverse) {
        const std::size_t plan_index = reverse - 1u;
        const VerticalNestingLevelPlan& level_plan = plan.levels[plan_index];
        const float ratio = static_cast<float>(level_plan.ratio_to_coarse);
        // x/y buffer width and relaxation time are SI quantities. Preserve
        // them under dx,dt -> dx/r,dt/r rather than making a child buffer
        // r-times thinner and r-times too strong in physical units.
        buffer_nudging_active = saved_buffer_nudging_active;
        buffer_n_cells = std::max(1, saved_buffer_n_cells *
            static_cast<int>(level_plan.ratio_to_coarse));
        buffer_inv_tau_lbmu = saved_buffer_inv_tau_lbmu / ratio;
        // Only the global sky block owns the top sponge. Its sibling-facing
        // ghost must not be mistaken for an outer sponge boundary.
        const bool is_coarsest_sky_level = plan_index + 1u == plan.levels.size();
        top_sponge_active = saved_top_sponge_active && is_coarsest_sky_level;
        std::unique_ptr<LBM> level_lbm = std::make_unique<LBM>(
            uint3(level_plan.nx, level_plan.ny, level_plan.allocated_nz),
            domains_x, domains_y, domains_z,
            physics.coarse_nu * ratio,
            0.0f, 0.0f, 0.0f, 0.0f,
            physics.coarse_alpha * ratio,
            physics.beta);
#ifdef TEMPERATURE
        level_lbm->set_buoyancy_gravity(
            physics.coarse_buoyancy_gx / ratio,
            physics.coarse_buoyancy_gy / ratio,
            physics.coarse_buoyancy_gz / ratio);
        level_lbm->set_temperature_outlet_face(
            physics.temperature_outlet_face);
#endif
        level_lbm->set_coriolis(physics.coarse_omega_x / ratio,
                                physics.coarse_omega_y / ratio,
                                physics.coarse_omega_z / ratio);

        const bool is_finest_ground_level = plan_index == 0u;
        populate_level_from_seed(seed, plan, level_plan,
                                 physical_masks[plan_index],
                                 seed_equilibrium_faces,
                                 seed_solid_temperature_columns,
                                 is_finest_ground_level,
                                 is_coarsest_sky_level,
                                 *level_lbm);

        NestedOwnedLbmLevel owned;
        owned.plan = level_plan;
        owned.plan_level_index = static_cast<uint32_t>(plan_index);
        owned.lbm = std::move(level_lbm);
        out.levels.push_back(std::move(owned));
    }
    buffer_nudging_active = saved_buffer_nudging_active;
    buffer_n_cells = saved_buffer_n_cells;
    buffer_inv_tau_lbmu = saved_buffer_inv_tau_lbmu;
    top_sponge_active = saved_top_sponge_active;
    return out.valid();
}

bool stitch_nested_physical_fields(NestedVerticalLbm& nested,
                                   NestedStitchedFields& out,
                                   std::string& error) {
    out = NestedStitchedFields{};
    error.clear();
    if (!nested.valid()) {
        error = "cannot stitch an invalid nested LBM hierarchy";
        return false;
    }
    const uint32_t finest_ratio = nested.geometry.common_finest_ratio;
    if (finest_ratio != 2u && finest_ratio != 4u) {
        error = "nested stitched output has an unsupported finest refinement ratio";
        return false;
    }
    uint32_t out_nx = 0u;
    uint32_t out_ny = 0u;
    if (!checked_u32_product(nested.plan.levels.back().nx, finest_ratio, out_nx) ||
        !checked_u32_product(nested.plan.levels.back().ny, finest_ratio, out_ny)) {
        error = "nested stitched output horizontal dimensions overflow uint32";
        return false;
    }
    uint64_t out_nz64 = 0u;
    for (const VerticalNestingLevelPlan& level : nested.plan.levels) {
        const uint32_t q = finest_ratio / level.ratio_to_coarse;
        if (q == 0u || finest_ratio % level.ratio_to_coarse != 0u) {
            error = "nested stitched output found an incompatible level ratio";
            return false;
        }
        out_nz64 += static_cast<uint64_t>(level.physical_nz) * q;
    }
    if (out_nz64 == 0u || out_nz64 > std::numeric_limits<uint32_t>::max()) {
        error = "nested stitched output vertical dimension overflows uint32";
        return false;
    }
    uint64_t cell_count = 0u;
    uint64_t xy_count = 0u;
    if (!checked_product(out_nx, out_ny, xy_count) ||
        !checked_product(xy_count, out_nz64, cell_count) ||
        cell_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max() / 3u)) {
        error = "nested stitched output exceeds addressable host memory";
        return false;
    }

    out.nx = out_nx;
    out.ny = out_ny;
    out.nz = static_cast<uint32_t>(out_nz64);
    out.dx_si = static_cast<float>(nested.plan.coarse_dx_si / finest_ratio);
    out.origin_z_above_ground_si = 0.5f * out.dx_si;
    out.rho.assign(static_cast<std::size_t>(cell_count), 1.0f);
    out.u.assign(static_cast<std::size_t>(3u * cell_count), 0.0f);
    out.flags.assign(static_cast<std::size_t>(cell_count), TYPE_S);
#ifdef TEMPERATURE
    out.temperature.assign(static_cast<std::size_t>(cell_count), 1.0f);
#endif

    for (NestedOwnedLbmLevel& owned : nested.levels) {
        LBM& source = *owned.lbm;
        const VerticalNestingLevelPlan& level = owned.plan;
        const uint32_t q = finest_ratio / level.ratio_to_coarse;
        source.rho.read_from_device();
        source.u.read_from_device();
        source.flags.read_from_device();
#ifdef TEMPERATURE
        source.T.read_from_device();
#endif
        const double begin_fine_real = level.z_begin_agl_si / out.dx_si;
        if (!vertical_nesting_nearly_integer(begin_fine_real)) {
            error = "nested stitched output found a non-aligned physical interface";
            return false;
        }
        const uint64_t begin_fine = static_cast<uint64_t>(std::llround(begin_fine_real));
        for (uint32_t z = 0u; z < level.physical_nz; ++z) {
            const uint32_t source_z = level.physical_first_z() + z;
            for (uint32_t y = 0u; y < level.ny; ++y) {
                for (uint32_t x = 0u; x < level.nx; ++x) {
                    const uint64_t source_index = index_3d(
                        x, y, source_z, source.get_Nx(), source.get_Ny());
                    for (uint32_t qz = 0u; qz < q; ++qz) {
                        const uint64_t oz = begin_fine + static_cast<uint64_t>(z) * q + qz;
                        for (uint32_t qy = 0u; qy < q; ++qy) {
                            const uint32_t oy = y * q + qy;
                            for (uint32_t qx = 0u; qx < q; ++qx) {
                                const uint32_t ox = x * q + qx;
                                const uint64_t destination = index_3d(
                                    ox, oy, static_cast<uint32_t>(oz), out.nx, out.ny);
                                out.rho[static_cast<std::size_t>(destination)] = source.rho[source_index];
                                out.u[static_cast<std::size_t>(3u * destination)] = source.u.x[source_index];
                                out.u[static_cast<std::size_t>(3u * destination + 1u)] = source.u.y[source_index];
                                out.u[static_cast<std::size_t>(3u * destination + 2u)] = source.u.z[source_index];
                                out.flags[static_cast<std::size_t>(destination)] = source.flags[source_index];
#ifdef TEMPERATURE
                                out.temperature[static_cast<std::size_t>(destination)] = source.T[source_index];
#endif
                            }
                        }
                    }
                }
            }
        }
    }
    return out.valid();
}

} // namespace culbm
