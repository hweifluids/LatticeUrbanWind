#include "domain.hpp"
#include "context.hpp"
#include "deck.hpp"
#include "geometry.hpp"
#include "../cuda_nested_solver.hpp"
#include "../solver/field_set.hpp"

#include <cstdlib>
#include <limits>

namespace culbm::cases {

void derive_domain(CaseContext& ctx) {
    CaseSettings& deck = ctx.deck;
    CaseDomain& domain = ctx.domain;

    // CUDA-only compact ground base. The preprocessor creates a closed slab
    // from the STL bottom to base_height. Drop only complete base cells from
    // the lattice, then place the remaining fractional base in z=0.
    domain.cuda_ground_base_height_si = deck.z_si_offset;
    domain.cuda_ground_trim_cells = 0u;
    domain.cuda_ground_geometry_shift_cells = 0u;
    domain.cuda_ground_trim_si = 0.0f;
    domain.cuda_ground_compaction_active = false;
    domain.cuda_ground_compaction_reason = "base_height is not positive";
    const char* const cuda_ground_compaction_env =
        std::getenv("CULBM_GROUND_COMPACTION");
    domain.cuda_ground_compaction_override =
        cuda_ground_compaction_env == nullptr
            ? std::string()
            : to_lower_ascii(trim(cuda_ground_compaction_env));
    domain.cuda_ground_compaction_requested =
        domain.cuda_ground_compaction_override.empty() ||
        !(domain.cuda_ground_compaction_override == "0" ||
          domain.cuda_ground_compaction_override == "off" ||
          domain.cuda_ground_compaction_override == "false" ||
          domain.cuda_ground_compaction_override == "no");

    domain.Nx_cells = std::max(1, (int)(deck.si_size.x / deck.cell_m + 0.5f));
    domain.Ny_cells = std::max(1, (int)(deck.si_size.y / deck.cell_m + 0.5f));
    domain.sponge_cells_cfg = std::max(1, (int)std::lround(deck.sponge_thickness_m / deck.cell_m));
    domain.Nz_cells_uncompacted = std::max(1, (int)(deck.si_size.z / deck.cell_m + 0.5f));
    if (!domain.cuda_ground_compaction_requested) {
        domain.cuda_ground_compaction_reason = "disabled by CULBM_GROUND_COMPACTION";
    }
    else if (!(std::isfinite(domain.cuda_ground_base_height_si) && domain.cuda_ground_base_height_si > 0.0f)) {
        domain.cuda_ground_compaction_reason = "base_height is not positive";
    }
    else if (!(std::isfinite(deck.cell_m) && deck.cell_m > 0.0f && deck.si_size.x > 0.0f && deck.si_size.y > 0.0f)) {
        domain.cuda_ground_compaction_reason = "invalid domain or cell scale";
    }
    else {
        const std::filesystem::path candidate =
            select_culbm_geometry_candidate(
                std::filesystem::path(deck.parent),
                deck.caseName,
                deck.profile_generation);
        if (candidate.empty()) {
            domain.cuda_ground_compaction_reason = "generated base STL was not found";
        }
        else {
            domain.cuda_ground_probe_path = candidate.string();
            if (deck.preloaded_geometry_mesh &&
                deck.preloaded_geometry_path.lexically_normal() ==
                    candidate.lexically_normal()) {
                domain.cuda_ground_probe_mesh =
                    std::move(deck.preloaded_geometry_mesh);
            }
            else {
                domain.cuda_ground_probe_mesh.reset(
                    read_stl(domain.cuda_ground_probe_path));
            }
            if (!domain.cuda_ground_probe_mesh) {
                domain.cuda_ground_compaction_reason = "generated base STL could not be read";
            }
            else {
                const float3 probe_size = domain.cuda_ground_probe_mesh->get_bounding_box_size();
                if (!(probe_size.x > 0.0f && probe_size.y > 0.0f && probe_size.z >= domain.cuda_ground_base_height_si)) {
                    domain.cuda_ground_compaction_reason = "STL is shorter than configured base_height";
                }
                else {
                    const float grid_scale = (float)domain.Ny_cells / deck.si_size.y;
                    const float geometry_scale = grid_scale * deck.si_size.x / probe_size.x;
                    const float scale_relative_error = fabsf(geometry_scale - grid_scale) / fmaxf(grid_scale, 1.0e-12f);
                    const float plane_tolerance = fmaxf(1.0e-4f, 2.0e-4f * fmaxf(domain.cuda_ground_base_height_si, probe_size.z));
                    const float base_bottom_si = domain.cuda_ground_probe_mesh->pmin.z;
                    const float base_top_si = base_bottom_si + domain.cuda_ground_base_height_si;
                    auto horizontal_plane_area = [&](const float z_plane) -> double {
                        double area = 0.0;
                        for (uint triangle = 0u; triangle < domain.cuda_ground_probe_mesh->triangle_number; ++triangle) {
                            const float3 p0 = domain.cuda_ground_probe_mesh->p0[triangle];
                            const float3 p1 = domain.cuda_ground_probe_mesh->p1[triangle];
                            const float3 p2 = domain.cuda_ground_probe_mesh->p2[triangle];
                            if (fabsf(p0.z - z_plane) > plane_tolerance ||
                                fabsf(p1.z - z_plane) > plane_tolerance ||
                                fabsf(p2.z - z_plane) > plane_tolerance) continue;
                            area += 0.5 * (double)length(cross(p1 - p0, p2 - p0));
                        }
                        return area;
                    };
                    const double minimum_plane_area = 0.005 * (double)probe_size.x * (double)probe_size.y;
                    const double base_bottom_area = horizontal_plane_area(base_bottom_si);
                    const double base_top_area = horizontal_plane_area(base_top_si);
                    const uint requested_trim = (uint)floorf(domain.cuda_ground_base_height_si * geometry_scale + 1.0e-4f);
                    const uint candidate_core_nz = requested_trim < domain.Nz_cells_uncompacted
                        ? domain.Nz_cells_uncompacted - requested_trim
                        : 0u;
                    const bool candidate_has_vertical_room = candidate_core_nz > 2u;
                    const bool candidate_sponge_extend = deck.enable_top_sponge && deck.sponge_tau_s > 0.0f &&
                        deck.sponge_ref_mode == 0 && candidate_has_vertical_room;
                    const uint candidate_total_nz = candidate_core_nz +
                        (candidate_sponge_extend ? (uint)domain.sponge_cells_cfg : 0u);
                    const float candidate_shift_si = requested_trim / grid_scale;

                    if (scale_relative_error > 0.02f) {
                        domain.cuda_ground_compaction_reason = "STL XY scale does not match the LBM grid";
                    }
                    else if (!(base_bottom_area >= minimum_plane_area && base_top_area >= minimum_plane_area)) {
                        domain.cuda_ground_compaction_reason = "STL does not have a generated base-slab signature";
                    }
                    else if (candidate_core_nz < 3u || !(deck.si_size.z - candidate_shift_si > 0.0f)) {
                        domain.cuda_ground_compaction_reason = "compaction would leave too little vertical domain";
                    }
                    else if (candidate_total_nz % deck.Dz != 0u) {
                        domain.cuda_ground_compaction_reason = "trimmed Nz is not divisible by Dz";
                    }
                    else if (candidate_shift_si > domain.cuda_ground_base_height_si + plane_tolerance) {
                        domain.cuda_ground_compaction_reason = "grid shift exceeds configured base_height";
                    }
                    else {
                        domain.cuda_ground_trim_cells = requested_trim;
                        // Compact z=0 is the first retained physical lattice
                        // row and is sealed after voxelization. Geometry must
                        // therefore move by exactly the removed-row count;
                        // an additional row would displace buildings in SI
                        // space relative to the uncompressed grid.
                        domain.cuda_ground_geometry_shift_cells = requested_trim;
                        domain.cuda_ground_trim_si = candidate_shift_si;
                        domain.cuda_ground_compaction_active = true;
                        domain.cuda_ground_compaction_reason = "enabled";
                        deck.si_size.z -= domain.cuda_ground_trim_si;
                        deck.z_si_offset -= domain.cuda_ground_trim_si;
                    }
                }
            }
        }
    }

    domain.Nz_cells_core = domain.cuda_ground_compaction_active
        ? domain.Nz_cells_uncompacted - domain.cuda_ground_trim_cells
        : domain.Nz_cells_uncompacted;
    const bool sponge_mode_supported_pre = (deck.sponge_ref_mode == 0);
    const bool sponge_has_vertical_room_pre = (domain.Nz_cells_core > 2u);
    domain.top_sponge_grid_extend = deck.enable_top_sponge && deck.sponge_tau_s > 0.0f && sponge_mode_supported_pre && sponge_has_vertical_room_pre;
    domain.Nz_cells = domain.Nz_cells_core + (domain.top_sponge_grid_extend ? (uint)domain.sponge_cells_cfg : 0u);
    domain.top_sponge_side_ref_z_cap = domain.top_sponge_grid_extend ? (int)domain.Nz_cells_core - 1 : -1;
    domain.lbm_N = uint3(domain.Nx_cells, domain.Ny_cells, domain.Nz_cells);
    domain.gpu_memory_estimate =
        estimate_project_gpu_memory_from_grid(
            deck, domain.lbm_N, &deck.particle_advection_settings);
    domain.nested_additional_gpu_memory_per_device_mb = 0u;
    domain.nested_full_height_raster_core_mb = 0u;

    // Heights in the deck are SI metres above the *top* of the compacted
    // one-cell ground boundary, not LBM z indices and not terrain-following
    // AGL. The latter would make a horizontal 4:1 interface ill-defined.
    if (!deck.cuda_vertical_nesting_agl_si.empty()) {
        if (deck.particle_advection_settings.enabled) {
            println("| ERROR: CULBM particle advection is not supported with CUDA vertical nesting. |");
            println("|        Remove cuda_vertical_nesting_agl or disable culbm_advection_enabled.   |");
            wait();
            exit(-1);
        }
        if (!domain.cuda_ground_compaction_active) {
            println("| ERROR: cuda_vertical_nesting_agl requires the sealed compact CUDA ground.     |");
            println("|        Ground compaction failed/skipped: " + domain.cuda_ground_compaction_reason + " |");
            wait();
            exit(-1);
        }
        const double coarse_dx_si = (double)deck.si_size.y / (double)domain.lbm_N.y;
        const double physical_air_height_si =
            (double)(domain.lbm_N.z - 1u) * coarse_dx_si;
        const double compact_ground_top_si = 0.5 * coarse_dx_si;
        std::vector<double> interface_heights;
        interface_heights.reserve(deck.cuda_vertical_nesting_agl_si.size());
        for (const float height : deck.cuda_vertical_nesting_agl_si) {
            interface_heights.push_back((double)height);
        }
        domain.cuda_vertical_nesting_plan = culbm::make_vertical_nesting_plan(
            interface_heights, coarse_dx_si, physical_air_height_si,
            domain.lbm_N.x, domain.lbm_N.y, deck.Dx, deck.Dy, deck.Dz, compact_ground_top_si);
        if (!domain.cuda_vertical_nesting_plan.valid) {
            println("| ERROR: invalid cuda_vertical_nesting_agl: " + domain.cuda_vertical_nesting_plan.error + " |");
            wait();
            exit(-1);
        }
    }

    if (domain.cuda_vertical_nesting_plan.enabled) {
        const std::uint64_t nested_core_mb =
            culbm::nested_hierarchy_vram_required_mb_per_device(
                domain.cuda_vertical_nesting_plan, deck.Dx, deck.Dy, deck.Dz);
        for (const culbm::VerticalNestingLevelPlan& level :
             domain.cuda_vertical_nesting_plan.levels) {
            const std::uint64_t level_extra_mb =
                bc::estimate_vk_inlet_extra_mb_per_device(
                    deck.vk_inlet_settings, level.nx, level.ny, level.allocated_nz);
            if (level_extra_mb >
                std::numeric_limits<std::uint64_t>::max() -
                    domain.nested_additional_gpu_memory_per_device_mb) {
                domain.nested_additional_gpu_memory_per_device_mb =
                    std::numeric_limits<std::uint64_t>::max();
                break;
            }
            domain.nested_additional_gpu_memory_per_device_mb += level_extra_mb;
        }
        const auto clamp_memory_mb = [](const std::uint64_t value) {
            return static_cast<uint>(std::min<std::uint64_t>(
                value, std::numeric_limits<uint>::max()));
        };
        domain.gpu_memory_estimate.core_per_device_mb =
            clamp_memory_mb(nested_core_mb);
        const std::uint64_t nested_hierarchy_total_mb =
            nested_core_mb >
                    std::numeric_limits<std::uint64_t>::max() -
                        domain.nested_additional_gpu_memory_per_device_mb
                ? std::numeric_limits<std::uint64_t>::max()
                : nested_core_mb +
                    domain.nested_additional_gpu_memory_per_device_mb;
        domain.nested_full_height_raster_core_mb =
            culbm::nested_full_height_raster_vram_required_mb_per_device(
                domain.cuda_vertical_nesting_plan, deck.Dx, deck.Dy, deck.Dz);
        const std::uint64_t nested_total_mb =
            std::max(
                nested_hierarchy_total_mb,
                domain.nested_full_height_raster_core_mb);
        domain.gpu_memory_estimate.extra_per_device_mb =
            clamp_memory_mb(
                nested_total_mb >= nested_core_mb
                    ? nested_total_mb - nested_core_mb
                    : std::numeric_limits<std::uint64_t>::max());
        domain.gpu_memory_estimate.total_per_device_mb =
            clamp_memory_mb(nested_total_mb);
        const std::uint64_t project_total_mb =
            nested_total_mb >
                    std::numeric_limits<std::uint64_t>::max() /
                        std::max<std::uint64_t>(
                            1u, static_cast<std::uint64_t>(deck.Dx) * deck.Dy * deck.Dz)
                ? std::numeric_limits<std::uint64_t>::max()
                : nested_total_mb *
                    static_cast<std::uint64_t>(deck.Dx) * deck.Dy * deck.Dz;
        domain.gpu_memory_estimate.total_project_mb =
            clamp_memory_mb(project_total_mb);
    }
    if (deck.gpu_memory_mesh_control_requested) {
        const uint usable_requested_memory_mb =
            usable_gpu_memory_request_mb(deck.memory);
        if (domain.gpu_memory_estimate.total_per_device_mb >
            usable_requested_memory_mb) {
            println("| ERROR: GPU memory fit exceeds the requested per-device budget.              |");
            println("|        Required " +
                to_string(domain.gpu_memory_estimate.total_per_device_mb) +
                " MB, usable " + to_string(usable_requested_memory_mb) +
                " MB after safety reserves (requested " + to_string(deck.memory) +
                " MB); reduce fixed GPU buffers or use a larger budget. |");
            wait();
            exit(-1);
        }
    }
}

} // namespace culbm::cases
