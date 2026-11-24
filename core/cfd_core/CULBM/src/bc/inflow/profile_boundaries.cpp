#include "boundaries.hpp"
#include "samples.hpp"
#include "../outflow/fluxcorrection.hpp"
#include "../../case/context.hpp"
#include "../../case/report.hpp"
#include "../../run/console_log.hpp"

#include <atomic>
#include <cfloat>

namespace culbm::bc {

using namespace culbm::cases;
using run::now_str;

namespace {

void initialize_profile_velocity(const ProfileInflow& profile, culbm::solver::Solver& solver,
                                 const std::vector<float>& ground_xy,
                                 const float dir_x, const float dir_y) {
    const uint Nx = solver.grid().size(0u);
    const ulong Ntot = solver.grid().cells();
    parallel_for(Ntot, [&](ulong n) {
        uint x, y, z;
        solver.grid().coordinates(n, x, y, z);
        if ((solver.flags()[n] & CELL_SOLID) != 0u) {
            solver.velocity().x[n] = 0.0f;
            solver.velocity().y[n] = 0.0f;
            solver.velocity().z[n] = 0.0f;
            return;
        }
        const float ground_z = ground_xy[(ulong)y * (ulong)Nx + (ulong)x];
        const float u_mag = profile_speed_lbmu(profile, solver.grid().position(x, y, z).z, ground_z);
        solver.velocity().x[n] = dir_x * u_mag;
        solver.velocity().y[n] = dir_y * u_mag;
        solver.velocity().z[n] = 0.0f;
    });
}

void apply_profile_boundaries(const ProfileInflow& profile, culbm::solver::Solver& solver,
                              const std::vector<float>& ground_xy,
                              const float dir_x, const float dir_y,
                              const std::string& downstream_bc_runtime,
                              const bool downstream_open_face,
                              const int top_sponge_side_ref_z_cap) {
    const uint Nx = solver.grid().size(0u), Ny = solver.grid().size(1u), Nz = solver.grid().size(2u);
    const ulong Ntot = solver.grid().cells();
    std::atomic<ulong> mapped_bc{ 0ul };
    std::atomic<ulong> terrain_solid_bc{ 0ul };
    std::atomic<ulong> outlet_bc{ 0ul };
    parallel_for(Ntot, [&](ulong n) {
        uint x, y, z;
        solver.grid().coordinates(n, x, y, z);
        if (z == 0u) {
            solver.flags()[n] = CELL_SOLID;
            solver.velocity().x[n] = 0.0f;
            solver.velocity().y[n] = 0.0f;
            solver.velocity().z[n] = 0.0f;
            return;
        }
        const bool is_boundary =
            (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u);
        if (!is_boundary) return;
        if ((solver.flags()[n] & CELL_SOLID) != 0u) return;

        const float ground_z = ground_xy[(ulong)y * (ulong)Nx + (ulong)x];
        const float pos_z = solver.grid().position(x, y, z).z;
        if (pos_z <= ground_z) {
            solver.flags()[n] = CELL_SOLID;
            solver.velocity().x[n] = 0.0f;
            solver.velocity().y[n] = 0.0f;
            solver.velocity().z[n] = 0.0f;
            terrain_solid_bc.fetch_add(1ul, std::memory_order_relaxed);
            return;
        }
        solver.flags()[n] = (uchar)(solver.flags()[n] | CELL_EQUILIBRIUM);
        if (downstream_open_face &&
            is_downstream_boundary_cell(x, y, z, Nx, Ny, Nz, downstream_bc_runtime)) {
            outlet_bc.fetch_add(1ul, std::memory_order_relaxed);
            return;
        }
        float pos_z_eval = pos_z;
        const bool is_side_boundary = (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u);
        if (is_side_boundary && top_sponge_side_ref_z_cap >= 0 && (int)z > top_sponge_side_ref_z_cap) {
            pos_z_eval = solver.grid().position(x, y, (uint)top_sponge_side_ref_z_cap).z;
        }
        const float u_mag = profile_speed_lbmu(profile, pos_z_eval, ground_z);
        solver.velocity().x[n] = dir_x * u_mag;
        solver.velocity().y[n] = dir_y * u_mag;
        solver.velocity().z[n] = 0.0f;
        mapped_bc.fetch_add(1ul, std::memory_order_relaxed);
    });
    println("| Velocity BC     | profile boundaries mapped: " + to_string(mapped_bc.load()) + " cells                |");
    if (outlet_bc.load(std::memory_order_relaxed) > 0ul) {
        println("|                 | downstream outlet cells: " + to_string(outlet_bc.load()) + " (no fixed velocity)        |");
    }
    if (terrain_solid_bc.load(std::memory_order_relaxed) > 0ul) {
        println("|                 | boundary cells below local terrain -> solid: " +
                to_string(terrain_solid_bc.load()) + "                     |");
    }
}

} // namespace

void prepare_profile_inflow(CaseContext& ctx, ProfileInflow& profile) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const CaseGeometry& geometry = ctx.geometry;
    const CaseTerrain& terrain = ctx.terrain;
    InflowData& inflow = ctx.inflow;
    const CasePhysics& physics = ctx.physics;

    profile.origin_lbmu = float3(
        0.5f - 0.5f * (float)domain.lbm_N.x,
        0.5f - 0.5f * (float)domain.lbm_N.y,
        0.5f - 0.5f * (float)domain.lbm_N.z
    );
    profile.flat_ground_lbmu = profile.origin_lbmu.z + 0.5f +
        domain.cuda_ground_base_height_si * geometry.scale_geom -
        (float)domain.cuda_ground_geometry_shift_cells;
    profile.use_profile_dem_ground = false;
    profile.ground_z_min_lbm = profile.flat_ground_lbmu;
    profile.ground_z_max_lbm = profile.flat_ground_lbmu;

    if (terrain.profile_dem_loaded) {
        const float dem_rx = terrain.profile_dem_xmax - terrain.profile_dem_xmin;
        const float dem_ry = terrain.profile_dem_ymax - terrain.profile_dem_ymin;
        const float stl_rx = geometry.stl_size_si.x;
        const float stl_ry = geometry.stl_size_si.y;
        if (dem_rx > 1.0e-6f && dem_ry > 1.0e-6f && stl_rx > 1.0e-6f && stl_ry > 1.0e-6f) {
            const float dem_to_stl_sx = stl_rx / dem_rx;
            const float dem_to_stl_sy = stl_ry / dem_ry;
            const float scale_warn = fmaxf(fabsf(dem_to_stl_sx - 1.0f), fabsf(dem_to_stl_sy - 1.0f));
            const float off_warn_x = fabsf(terrain.profile_dem_xmin - geometry.stl_min_si.x) / stl_rx;
            const float off_warn_y = fabsf(terrain.profile_dem_ymin - geometry.stl_min_si.y) / stl_ry;
            if (scale_warn > 0.02f || off_warn_x > 0.02f || off_warn_y > 0.02f) {
                println("| Terrain DEM     | WARNING: DEM/STL XY bounds mismatch. Apply affine bounds alignment. |");
                println("|                 | DEM->STL scale x=" + to_string(dem_to_stl_sx, 6u) +
                        ", y=" + to_string(dem_to_stl_sy, 6u) + "                             |");
            }

            std::vector<SamplePoint> dem_ground_samples;
            dem_ground_samples.reserve(terrain.profile_dem_points_raw.size());
            profile.ground_z_min_lbm = +FLT_MAX;
            profile.ground_z_max_lbm = -FLT_MAX;
            for (const auto& dp : terrain.profile_dem_points_raw) {
                const float x_stl_raw = geometry.stl_min_si.x + (dp.x - terrain.profile_dem_xmin) * dem_to_stl_sx;
                const float y_stl_raw = geometry.stl_min_si.y + (dp.y - terrain.profile_dem_ymin) * dem_to_stl_sy;
                const float z_stl_raw = geometry.stl_min_si.z + domain.cuda_ground_base_height_si + dp.elevation;

                const float x_lbm = profile.origin_lbmu.x + 0.5f + (x_stl_raw - geometry.stl_min_si.x) * geometry.scale_geom;
                const float y_lbm = profile.origin_lbmu.y + 0.5f + (y_stl_raw - geometry.stl_min_si.y) * geometry.scale_geom;
                const float z_lbm = profile.origin_lbmu.z + 0.5f +
                    (z_stl_raw - geometry.stl_min_si.z) * geometry.scale_geom -
                    (float)domain.cuda_ground_geometry_shift_cells;

                if (!std::isfinite(x_lbm) || !std::isfinite(y_lbm) || !std::isfinite(z_lbm)) continue;
                SamplePoint sp;
                sp.p = float3(x_lbm, y_lbm, 0.0f);
                sp.T = z_lbm;
                sp.patch = k_patch_bottom;
                dem_ground_samples.push_back(sp);
                profile.ground_z_min_lbm = fminf(profile.ground_z_min_lbm, z_lbm);
                profile.ground_z_max_lbm = fmaxf(profile.ground_z_max_lbm, z_lbm);
            }

            if (!dem_ground_samples.empty()) {
                profile.profile_ground_plane.build_from_patch0_samples(dem_ground_samples, profile.flat_ground_lbmu);
                profile.use_profile_dem_ground = profile.profile_ground_plane.has_samples();
            }
            if (profile.use_profile_dem_ground) {
                const float gmin_si = units.si_x(profile.ground_z_min_lbm - profile.origin_lbmu.z);
                const float gmax_si = units.si_x(profile.ground_z_max_lbm - profile.origin_lbmu.z);
                println("| Terrain DEM     | profile ground enabled. z(SI) range " +
                        to_string(gmin_si, 3u) + " .. " + to_string(gmax_si, 3u) + " m |");
            }
            else {
                println("| Terrain DEM     | no valid points after mapping, fallback to flat ground     |");
            }
        }
        else {
            println("| Terrain DEM     | invalid DEM or STL XY range, fallback to flat ground       |");
        }
    }

    profile.solver_top_si = (domain.lbm_N.z > 0u) ? units.si_x((float)(domain.lbm_N.z - 1u)) : 0.0f;
    profile.core_top_si = (domain.top_sponge_side_ref_z_cap >= 0)
        ? units.si_x((float)domain.top_sponge_side_ref_z_cap)
        : profile.solver_top_si;
    profile.ground_min_si = units.si_x(profile.ground_z_min_lbm - profile.origin_lbmu.z);
    profile.ground_max_si = units.si_x(profile.ground_z_max_lbm - profile.origin_lbmu.z);
    if (!std::isfinite(profile.ground_min_si)) profile.ground_min_si = deck.z_si_offset;
    if (!std::isfinite(profile.ground_max_si)) profile.ground_max_si = profile.ground_min_si;
    profile.profile_table_top_si = profile.solver_top_si - profile.ground_min_si;
    if (!std::isfinite(profile.profile_table_top_si) || profile.profile_table_top_si <= 0.0f) {
        profile.profile_table_top_si = std::max(deck.profile_dz_si, deck.si_size.z - profile.ground_min_si);
    }
    profile.profile_table_top_si = std::max(profile.profile_table_top_si, deck.profile_dz_si);

    profile.profile_steps = static_cast<uint>(std::ceil(profile.profile_table_top_si / deck.profile_dz_si));
    inflow.profile_u_si.assign(profile.profile_steps + 1u, 0.0f);
    for (uint i = 0u; i <= profile.profile_steps; ++i) {
        const float zq = std::min(profile.profile_table_top_si, (float)i * deck.profile_dz_si);
        float u_val = interpolate_profile_cubic(inflow.profile_z_samples_si, inflow.profile_u_samples_si, zq);
        if (u_val < 0.0f) u_val = 0.0f;
        inflow.profile_u_si[i] = u_val;
    }

    profile.u_min_prof = inflow.profile_u_si.front();
    profile.u_max_prof = inflow.profile_u_si.front();
    for (const float v : inflow.profile_u_si) {
        if (v < profile.u_min_prof) profile.u_min_prof = v;
        if (v > profile.u_max_prof) profile.u_max_prof = v;
    }
    profile.profile_u_lbmu.assign(inflow.profile_u_si.size(), 0.0f);
    for (size_t i = 0; i < inflow.profile_u_si.size(); ++i) {
        profile.profile_u_lbmu[i] = inflow.profile_u_si[i] * physics.u_scale;
    }
    println("| Profile table   | local-terrain AGL top=" + to_string(profile.profile_table_top_si, 3u) +
            " m, core_top=" + to_string(profile.core_top_si, 3u) +
            " m, solver_top=" + to_string(profile.solver_top_si, 3u) + " m |");
    println("| Profile ground  | z(SI) min/max=" + to_string(profile.ground_min_si, 3u) +
            " / " + to_string(profile.ground_max_si, 3u) + " m |");
    println("| Profile U range | " + alignr(24u, to_string(fmtf(profile.u_min_prof))) +
            " to " + alignl(16u, to_string(fmtf(profile.u_max_prof))) + " m/s |");
}

float profile_speed_lbmu(const ProfileInflow& profile, const float pos_z, const float ground_z) {
    if (pos_z <= ground_z) return 0.0f;
    const float inv_dz = 1.0f / cases::CaseSettings::profile_dz_si;
    const uint last_idx = static_cast<uint>(profile.profile_u_lbmu.size() - 1u);
    float z_agl_si = units.si_x(pos_z - ground_z);
    if (z_agl_si < 0.0f) z_agl_si = 0.0f;
    long idx_l = std::lround(z_agl_si * inv_dz);
    if (idx_l < 0l) idx_l = 0l;
    uint idx = static_cast<uint>(idx_l);
    if (idx > last_idx) idx = last_idx;
    return profile.profile_u_lbmu[idx];
}

void build_profile_case_boundaries(CaseContext& ctx, const ProfileInflow& profile, culbm::solver::Solver& solver,
                                   const float dir_x, const float dir_y, const std::string& case_downstream_bc,
                                   const bool use_cuda_vertical_nesting,
                                   culbm::NestedCommonFinestTerrain& nested_profile_terrain,
                                   bool& nested_profile_terrain_active) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const uint Nx = solver.grid().size(0u), Ny = solver.grid().size(1u), Nz = solver.grid().size(2u);
    std::vector<float> ground_xy((ulong)Nx * (ulong)Ny, profile.flat_ground_lbmu);
    if (profile.use_profile_dem_ground) {
        const float zmin = solver.grid().position(0u, 0u, 0u).z;
        const float zmax = solver.grid().position(0u, 0u, Nz - 1u).z;
        parallel_for((ulong)Nx * (ulong)Ny, [&](ulong id) {
            const uint x = (uint)(id % (ulong)Nx);
            const uint y = (uint)(id / (ulong)Nx);
            const float3 pxy = solver.grid().position(x, y, 0u);
            float zg = profile.profile_ground_plane.eval_xy(pxy.x, pxy.y);
            if (!std::isfinite(zg)) zg = profile.flat_ground_lbmu;
            zg = fminf(fmaxf(zg, zmin), zmax);
            ground_xy[id] = zg;
        });
        float gmin = +FLT_MAX, gmax = -FLT_MAX;
        for (const float zg : ground_xy) {
            gmin = fminf(gmin, zg);
            gmax = fmaxf(gmax, zg);
        }
        println("| Terrain ground  | mapped z(SI) range " +
                to_string(units.si_x(gmin - profile.origin_lbmu.z), 3u) + " .. " +
                to_string(units.si_x(gmax - profile.origin_lbmu.z), 3u) + " m                     |");

        if (use_cuda_vertical_nesting) {
            const uint ratio = domain.cuda_vertical_nesting_plan.levels.front().ratio_to_coarse;
            const uint fine_nx = Nx * ratio;
            const uint fine_ny = Ny * ratio;
            nested_profile_terrain.nx = fine_nx;
            nested_profile_terrain.ny = fine_ny;
            nested_profile_terrain.solid_below_z.assign(
                (size_t)fine_nx * (size_t)fine_ny, -FLT_MAX);
            const float inv_ratio = 1.0f / (float)ratio;
            parallel_for((ulong)fine_nx * (ulong)fine_ny, [&](ulong id) {
                const uint fx = (uint)(id % (ulong)fine_nx);
                const uint fy = (uint)(id / (ulong)fine_nx);
                // Invert the shared cell-centred coarse->fine map
                // used by cuda_nested_solver.cpp, then evaluate the
                // original continuous DEM at that finest location.
                const float coarse_x = ((float)fx + 0.5f) * inv_ratio - 0.5f;
                const float coarse_y = ((float)fy + 0.5f) * inv_ratio - 0.5f;
                float zg = profile.profile_ground_plane.eval_xy(
                    profile.origin_lbmu.x + coarse_x,
                    profile.origin_lbmu.y + coarse_y);
                if (!std::isfinite(zg)) zg = profile.flat_ground_lbmu;
                zg = fminf(fmaxf(zg, zmin), zmax);
                const float coarse_grid_z = zg - profile.origin_lbmu.z;
                nested_profile_terrain.solid_below_z[(size_t)id] =
                    (coarse_grid_z - 0.5f) * (float)ratio + 0.5f;
            });
            nested_profile_terrain_active = nested_profile_terrain.valid();
            if (!nested_profile_terrain_active) {
                println("| Terrain raster  | nested DEM construction failed; aborting nested case.       |");
                wait(); exit(-1);
            }
            print_kv_row("Terrain raster", "DEM sampled directly on " +
                to_string((ulong)fine_nx) + "x" + to_string((ulong)fine_ny) +
                " common-finest cells");
        }
    }

    if (profile.use_profile_dem_ground) {
        std::atomic<ulong> terrain_clipped_to_solid{ 0ul };
        parallel_for(solver.grid().cells(), [&](ulong n) {
            if ((solver.flags()[n] & CELL_SOLID) != 0u) return;
            uint x, y, z;
            solver.grid().coordinates(n, x, y, z);
            const float pos_z = solver.grid().position(x, y, z).z;
            const float ground_z = ground_xy[(ulong)y * (ulong)Nx + (ulong)x];
            if (pos_z < ground_z) {
                solver.flags()[n] = CELL_SOLID;
                solver.velocity().x[n] = 0.0f;
                solver.velocity().y[n] = 0.0f;
                solver.velocity().z[n] = 0.0f;
                terrain_clipped_to_solid.fetch_add(1ul, std::memory_order_relaxed);
            }
        });
        if (terrain_clipped_to_solid.load(std::memory_order_relaxed) > 0ul) {
            println("| Terrain clip    | below-terrain cells forced to solid: " +
                    to_string(terrain_clipped_to_solid.load()) + "                    |");
        }
    }

    emit_progress_stage("interface_interpolation",
                        "Interface interpolation",
                        "Applying profile boundary conditions",
                        0ll,
                        1ll,
                        true);
    initialize_profile_velocity(profile, solver, ground_xy, dir_x, dir_y);
    apply_profile_boundaries(profile, solver, ground_xy, dir_x, dir_y, case_downstream_bc, deck.downstream_open_face,
                             domain.top_sponge_side_ref_z_cap);
    emit_progress_stage("interface_interpolation",
                        "Interface interpolation",
                        "Profile boundary conditions completed",
                        1ll,
                        1ll,
                        false);
    print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

    auto eval = [&](const float3& q)->float3 {
        uint x, y, z;
        solver.grid().coordinates(q, x, y, z);
        const float ground_z = ground_xy[(ulong)y * (ulong)Nx + (ulong)x];
        float pos_z_eval = q.z;
        const bool is_side_boundary = (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u);
        if (is_side_boundary && domain.top_sponge_side_ref_z_cap >= 0 && (int)z > domain.top_sponge_side_ref_z_cap) {
            pos_z_eval = solver.grid().position(x, y, (uint)domain.top_sponge_side_ref_z_cap).z;
        }
        const float u_mag = profile_speed_lbmu(profile, pos_z_eval, ground_z);
        return float3(dir_x * u_mag, dir_y * u_mag, 0.0f);
    };
    run_case_flux_correction(solver, deck.flux_correction, case_downstream_bc, eval);
}

} // namespace culbm::bc
