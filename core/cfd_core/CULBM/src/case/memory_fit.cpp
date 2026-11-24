#include "memory_fit.hpp"
#include "../particle_advection.hpp"
#include "../solver/field_set.hpp"

#include <array>
#include <limits>

namespace culbm::cases {

uint estimate_particle_advection_extra_mb_per_device(
    const CaseSettings& deck,
    const uint3& grid,
    const culbm::particle_advection::Config* const advection) {
    if (advection == nullptr || !advection->enabled) return 0u;
    const std::uint64_t local_nx =
        (std::uint64_t)(grid.x / deck.Dx + 2u * (deck.Dx > 1u));
    const std::uint64_t local_ny =
        (std::uint64_t)(grid.y / deck.Dy + 2u * (deck.Dy > 1u));
    const std::uint64_t local_nz =
        (std::uint64_t)(grid.z / deck.Dz + 2u * (deck.Dz > 1u));
    const std::uint64_t local_cells =
        local_nx * local_ny * local_nz;
    // The release schedule is not yet quantized at mesh-fit time.  One
    // release record per configured particle slot is a safe upper bound.
    const std::uint64_t release_slots =
        advection->mode == culbm::particle_advection::Mode::Automatic
            ? advection->max_particles
            : 0u;
    const std::uint64_t memory_mb =
        culbm::particle_advection_gpu_memory_mb_per_device(
            advection->max_particles, local_cells, release_slots);
    return (uint)std::min<std::uint64_t>(
        memory_mb, std::numeric_limits<uint>::max());
}

ProjectGpuMemoryEstimate estimate_project_gpu_memory_from_grid(
    const CaseSettings& deck,
    const uint3& grid,
    const culbm::particle_advection::Config* const advection) {
    ProjectGpuMemoryEstimate estimate;
    estimate.grid = grid;
    estimate.core_per_device_mb = (uint)lattice_vram_mib(grid.x, grid.y, grid.z, deck.Dx, deck.Dy, deck.Dz);
    const std::uint64_t extra_mb =
        (std::uint64_t)bc::estimate_vk_inlet_extra_mb_per_device(
            deck.vk_inlet_settings, grid.x, grid.y, grid.z) +
        (std::uint64_t)estimate_particle_advection_extra_mb_per_device(
            deck, grid, advection);
    estimate.extra_per_device_mb = (uint)std::min<std::uint64_t>(
        extra_mb, std::numeric_limits<uint>::max());
    const std::uint64_t total_mb =
        (std::uint64_t)estimate.core_per_device_mb + extra_mb;
    estimate.total_per_device_mb = (uint)std::min<std::uint64_t>(
        total_mb, std::numeric_limits<uint>::max());
    const std::uint64_t domain_count =
        (std::uint64_t)deck.Dx * (std::uint64_t)deck.Dy * (std::uint64_t)deck.Dz;
    const std::uint64_t project_mb =
        domain_count != 0u &&
            total_mb >
                std::numeric_limits<std::uint64_t>::max() / domain_count
            ? std::numeric_limits<std::uint64_t>::max()
            : total_mb * domain_count;
    estimate.total_project_mb = (uint)std::min<std::uint64_t>(
        project_mb, std::numeric_limits<uint>::max());
    return estimate;
}

ProjectGpuMemoryEstimate estimate_project_gpu_memory_from_cell_size(
    const CaseSettings& deck,
    const float cell_size_m,
    const std::vector<float>& nesting_heights_agl_si,
    const culbm::particle_advection::Config* const advection,
    const std::uint64_t voxelization_transient_mb) {
    const float safe_cell_m = fmax(cell_size_m, 1.0e-6f);
    ProjectGpuMemoryEstimate estimate;
    estimate.grid.x = std::max(1, (int)(deck.si_size.x / safe_cell_m + 0.5f));
    estimate.grid.y = std::max(1, (int)(deck.si_size.y / safe_cell_m + 0.5f));
    const uint Nz_cells_core = std::max(1, (int)(deck.si_size.z / safe_cell_m + 0.5f));
    estimate.top_sponge_grid_extend =
        deck.enable_top_sponge &&
        deck.sponge_tau_s > 0.0f &&
        deck.sponge_ref_mode == 0 &&
        Nz_cells_core > 2u;
    estimate.sponge_cells = estimate.top_sponge_grid_extend
        ? (uint)std::max(1, (int)std::lround(deck.sponge_thickness_m / safe_cell_m))
        : 0u;
    estimate.grid.z = Nz_cells_core + estimate.sponge_cells;

    ProjectGpuMemoryEstimate projected =
        estimate_project_gpu_memory_from_grid(deck, estimate.grid, advection);
    projected.top_sponge_grid_extend = estimate.top_sponge_grid_extend;
    projected.sponge_cells = estimate.sponge_cells;
    const std::uint64_t uniform_voxel_peak_mb =
        voxelization_transient_mb >
                std::numeric_limits<std::uint64_t>::max() -
                    projected.core_per_device_mb
            ? std::numeric_limits<std::uint64_t>::max()
            : (std::uint64_t)projected.core_per_device_mb +
                voxelization_transient_mb;
    if (uniform_voxel_peak_mb > projected.total_per_device_mb) {
        projected.total_per_device_mb =
            (uint)std::min<std::uint64_t>(
                uniform_voxel_peak_mb,
                std::numeric_limits<uint>::max());
        projected.extra_per_device_mb =
            (uint)std::min<std::uint64_t>(
                uniform_voxel_peak_mb -
                    projected.core_per_device_mb,
                std::numeric_limits<uint>::max());
        const std::uint64_t domain_count =
            (std::uint64_t)deck.Dx * (std::uint64_t)deck.Dy *
            (std::uint64_t)deck.Dz;
        const std::uint64_t project_peak_mb =
            domain_count != 0u &&
                    uniform_voxel_peak_mb >
                        std::numeric_limits<std::uint64_t>::max() /
                            domain_count
                ? std::numeric_limits<std::uint64_t>::max()
                : uniform_voxel_peak_mb * domain_count;
        projected.total_project_mb =
            (uint)std::min<std::uint64_t>(
                project_peak_mb, std::numeric_limits<uint>::max());
    }
    if (!nesting_heights_agl_si.empty() &&
        nesting_heights_agl_si.size() <= 2u &&
        estimate.grid.x > 0u && estimate.grid.y > 0u &&
        deck.si_size.y > 0.0f) {
        // gpu_memory mode runs before STL ground compaction.  Reproduce the
        // compacted z extent from the configured base height, then budget all
        // simultaneously resident nested slabs.  Ceil is intentional here:
        // the later planner still enforces exact SI alignment, while this
        // early fit must never under-budget a nearly aligned candidate.
        const double coarse_dx_si =
            (double)deck.si_size.y / (double)estimate.grid.y;
        const uint compact_rows = deck.z_si_offset > 0.0f
            ? (uint)std::floor(
                (double)deck.z_si_offset *
                (double)estimate.grid.y / (double)deck.si_size.y + 1.0E-4)
            : 0u;
        const uint compact_nz = estimate.grid.z > compact_rows
            ? estimate.grid.z - compact_rows : 1u;
        const double highest_interface =
            (double)nesting_heights_agl_si.back();
        const double domain_height_si = std::max(
            (double)(compact_nz > 0u ? compact_nz - 1u : 0u) *
                coarse_dx_si,
            highest_interface + 2.0 * coarse_dx_si);
        const std::array<uint, 3u> ratios =
            nesting_heights_agl_si.size() == 1u
                ? std::array<uint, 3u>{2u, 1u, 0u}
                : std::array<uint, 3u>{4u, 2u, 1u};
        const std::array<double, 4u> bounds =
            nesting_heights_agl_si.size() == 1u
                ? std::array<double, 4u>{
                    0.0,
                    (double)nesting_heights_agl_si[0],
                    domain_height_si,
                    domain_height_si}
                : std::array<double, 4u>{
                    0.0,
                    (double)nesting_heights_agl_si[0],
                    (double)nesting_heights_agl_si[1],
                    domain_height_si};
        const std::size_t level_count =
            nesting_heights_agl_si.size() + 1u;
        std::uint64_t core_mb = 0u;
        std::uint64_t extra_mb = 0u;
        for (std::size_t level = 0u; level < level_count; ++level) {
            const uint ratio = ratios[level];
            const double level_dx = coarse_dx_si / (double)ratio;
            const double thickness =
                std::max(0.0, bounds[level + 1u] - bounds[level]);
            const uint physical_nz = (uint)std::max(
                2.0, std::ceil(thickness / level_dx - 1.0E-9));
            const uint level_nx = estimate.grid.x * ratio;
            const uint level_ny = estimate.grid.y * ratio;
            const uint level_nz = physical_nz + 2u;
            core_mb += lattice_vram_mib(
                level_nx, level_ny, level_nz, deck.Dx, deck.Dy, deck.Dz);
            extra_mb += bc::estimate_vk_inlet_extra_mb_per_device(
                deck.vk_inlet_settings, level_nx, level_ny, level_nz);
        }
        // Geometry is not loaded yet in gpu_memory mode.  Budget the
        // temporary common-finest raster at the full active height; the
        // runtime builder later replaces this conservative z extent with the
        // exact STL/terrain top and adds exact triangle-buffer transients.
        std::uint64_t raster_core_mb =
            std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t finest_ratio = ratios[0];
        const std::uint64_t raster_nx64 =
            (std::uint64_t)estimate.grid.x * finest_ratio;
        const std::uint64_t raster_ny64 =
            (std::uint64_t)estimate.grid.y * finest_ratio;
        const double finest_dx_si =
            coarse_dx_si / (double)finest_ratio;
        const double raster_rows_real =
            std::ceil(1.0 + domain_height_si / finest_dx_si);
        if (raster_nx64 <= std::numeric_limits<uint>::max() &&
            raster_ny64 <= std::numeric_limits<uint>::max() &&
            std::isfinite(raster_rows_real) &&
            raster_rows_real > 0.0 &&
            raster_rows_real <=
                (double)std::numeric_limits<uint>::max()) {
            const std::uint64_t raster_rows =
                (std::uint64_t)raster_rows_real;
            const std::uint64_t raster_nz64 =
                (raster_rows + (std::uint64_t)deck.Dz - 1u) /
                    (std::uint64_t)deck.Dz * (std::uint64_t)deck.Dz;
            if (raster_nz64 <= std::numeric_limits<uint>::max()) {
                raster_core_mb = lattice_vram_mib(
                    (uint)raster_nx64, (uint)raster_ny64,
                    (uint)raster_nz64, deck.Dx, deck.Dy, deck.Dz);
            }
        }
        const std::uint64_t hierarchy_total_mb =
            extra_mb >
                    std::numeric_limits<std::uint64_t>::max() - core_mb
                ? std::numeric_limits<std::uint64_t>::max()
                : core_mb + extra_mb;
        const std::uint64_t raster_peak_mb =
            voxelization_transient_mb >
                    std::numeric_limits<std::uint64_t>::max() -
                        raster_core_mb
                ? std::numeric_limits<std::uint64_t>::max()
                : raster_core_mb + voxelization_transient_mb;
        const std::uint64_t construction_peak_mb =
            std::max(
                hierarchy_total_mb,
                std::max(raster_peak_mb, uniform_voxel_peak_mb));
        const std::uint64_t peak_extra_mb =
            construction_peak_mb >= core_mb
                ? construction_peak_mb - core_mb
                : std::numeric_limits<std::uint64_t>::max();
        const auto clamp_mb = [](const std::uint64_t value) {
            return static_cast<uint>(std::min<std::uint64_t>(
                value, std::numeric_limits<uint>::max()));
        };
        projected.core_per_device_mb = clamp_mb(core_mb);
        projected.extra_per_device_mb = clamp_mb(peak_extra_mb);
        projected.total_per_device_mb =
            clamp_mb(construction_peak_mb);
        const std::uint64_t domain_count =
            (std::uint64_t)deck.Dx * (std::uint64_t)deck.Dy * (std::uint64_t)deck.Dz;
        projected.total_project_mb = clamp_mb(
            domain_count != 0u &&
                    construction_peak_mb >
                        std::numeric_limits<std::uint64_t>::max() /
                            domain_count
                ? std::numeric_limits<std::uint64_t>::max()
                : construction_peak_mb * domain_count);
    }
    return projected;
}

uint usable_gpu_memory_request_mb(const uint target_mb) {
    // Fit from total device capacity before a CUDA context/module exists.
    // Reserve one 256 MiB tranche for that runtime footprint and a second
    // tranche for the allocator's required post-allocation safety margin.
    constexpr uint fit_reserve_mb = 512u;
    return target_mb > fit_reserve_mb
        ? target_mb - fit_reserve_mb
        : target_mb * 9u / 10u;
}

float fit_cell_size_to_gpu_memory_request(
    const CaseSettings& deck,
    const uint target_mb,
    const std::vector<float>& nesting_heights_agl_si,
    const culbm::particle_advection::Config* const advection,
    const std::uint64_t voxelization_transient_mb) {
    if (target_mb == 0u) {
        return 20.0f;
    }
    const uint usable_target_mb = usable_gpu_memory_request_mb(target_mb);

    float fit_cell_m = fmax(fmax(deck.si_size.x, deck.si_size.y), deck.si_size.z + fmax(deck.sponge_thickness_m, 0.0f));
    fit_cell_m = fmax(fit_cell_m, 1.0f);
    ProjectGpuMemoryEstimate fit_estimate =
        estimate_project_gpu_memory_from_cell_size(deck,
            fit_cell_m, nesting_heights_agl_si, advection,
            voxelization_transient_mb);
    for (int i = 0; i < 32 && fit_estimate.total_per_device_mb > usable_target_mb; ++i) {
        fit_cell_m *= 2.0f;
        fit_estimate = estimate_project_gpu_memory_from_cell_size(deck,
            fit_cell_m, nesting_heights_agl_si, advection,
            voxelization_transient_mb);
    }

    float overflow_cell_m = fit_cell_m * 0.5f;
    ProjectGpuMemoryEstimate overflow_estimate =
        estimate_project_gpu_memory_from_cell_size(deck,
            overflow_cell_m, nesting_heights_agl_si, advection,
            voxelization_transient_mb);
    for (int i = 0; i < 64 && overflow_cell_m > 1.0e-6f && overflow_estimate.total_per_device_mb <= usable_target_mb; ++i) {
        fit_cell_m = overflow_cell_m;
        fit_estimate = overflow_estimate;
        overflow_cell_m *= 0.5f;
        overflow_estimate = estimate_project_gpu_memory_from_cell_size(deck,
            overflow_cell_m, nesting_heights_agl_si, advection,
            voxelization_transient_mb);
    }

    for (int i = 0; i < 48; ++i) {
        const float mid_cell_m = 0.5f * (overflow_cell_m + fit_cell_m);
        const ProjectGpuMemoryEstimate mid_estimate =
            estimate_project_gpu_memory_from_cell_size(deck,
                mid_cell_m, nesting_heights_agl_si, advection,
                voxelization_transient_mb);
        if (mid_estimate.total_per_device_mb <= usable_target_mb) {
            fit_cell_m = mid_cell_m;
            fit_estimate = mid_estimate;
        }
        else {
            overflow_cell_m = mid_cell_m;
            overflow_estimate = mid_estimate;
        }
    }

    return fit_cell_m;
}

} // namespace culbm::cases
