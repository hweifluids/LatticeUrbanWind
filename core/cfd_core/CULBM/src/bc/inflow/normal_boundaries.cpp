#include "boundaries.hpp"
#include "interpolation.hpp"
#include "interpolation_hd.hpp"
#include "../outflow/fluxcorrection.hpp"
#include "../thermal/thermal_bc.hpp"
#include "../../case/context.hpp"
#include "../../case/report.hpp"
#include "../../run/console_log.hpp"

#include <atomic>
#include <cfloat>

namespace culbm::bc {

using namespace culbm::cases;
using run::now_str;

void build_normal_boundaries(CaseContext& ctx, culbm::solver::Solver& solver, const float z0_lbmu) {
    const CaseSettings& deck = ctx.deck;
    const std::vector<SamplePoint>& samples_si = ctx.inflow.samples_si;
    std::vector<SamplePoint>& samples = ctx.inflow.samples;
    const bool csv_has_patch = ctx.inflow.csv_has_patch;
    const bool use_temperature_bc = ctx.inflow.use_temperature_bc;
    const std::string& downstream_bc = deck.downstream_bc;
    const bool downstream_open_face = deck.downstream_open_face;
    const bool use_high_order = deck.use_high_order;
    const bool flux_correction = deck.flux_correction;
    const float z_si_offset = deck.z_si_offset;
    const float z_off = ctx.physics.z_off;
    const int top_sponge_side_ref_z_cap = ctx.domain.top_sponge_side_ref_z_cap;
    println("| CDF data loaded | " + alignl(57u, to_string(samples_si.size())) + " |");

    std::vector<float3> P; P.reserve(samples.size());
    std::vector<float3> Uv; Uv.reserve(samples.size());
    for (const auto& s : samples) { P.push_back(s.p); Uv.push_back(s.u); }
    std::vector<float3> P_temp;
    std::vector<float3> Tv;
    if (use_temperature_bc) {
        P_temp.reserve(samples.size());
        Tv.reserve(samples.size());
        for (const auto& s : samples) {
            P_temp.push_back(s.p);
            Tv.push_back(float3(s.T, 0.0f, 0.0f));
        }
    }

    const bool use_patch_face_bc = csv_has_patch;
    std::vector<PatchSurfaceField2D> patch_velocity_fields(6);
    std::vector<ulong> patch_counts(6, 0ul);
    if (use_patch_face_bc) {
        for (const auto& s : samples) {
            if (s.patch >= 0 && s.patch <= 5) patch_counts[(size_t)s.patch] += 1ul;
        }
        println("| Patch samples   | " + alignr(8u, string(patch_name(k_patch_bottom))) + " = " +
                alignl(47u, to_string(patch_counts[(size_t)k_patch_bottom])) + " |");
        for (int patch = k_patch_top; patch <= k_patch_east; ++patch) {
            patch_velocity_fields[(size_t)patch].build_from_patch(
                samples, patch,
                [](const SamplePoint& s) { return s.u; },
                float3(0.0f));
            println("|                 | " + alignr(8u, string(patch_name(patch))) + " = " +
                    alignl(47u, to_string(patch_counts[(size_t)patch])) + " |");
        }
    }

    NormalTemperatureBoundary thermal_bc(ctx, solver, use_patch_face_bc);

    if (use_patch_face_bc) {
        const uint Nx = solver.grid().size(0u);
        const uint Ny = solver.grid().size(1u);
        const uint Nz = solver.grid().size(2u);
        const int downstream_patch = downstream_to_patch(downstream_bc);
        emit_progress_stage("interface_interpolation",
                            "Interface interpolation",
                            "Patch-driven 2D boundary mapping",
                            0ll,
                            1ll,
                            true);
        std::vector<uchar> voxel_flags((size_t)solver.grid().cells());
        for (ulong n = 0ul; n < solver.grid().cells(); ++n) {
            voxel_flags[(size_t)n] = solver.flags()[n];
        }

        auto set_zero_velocity = [&](const ulong n) {
            solver.velocity().x[n] = 0.0f;
            solver.velocity().y[n] = 0.0f;
            solver.velocity().z[n] = 0.0f;
        };

        // Ensure cells below terrain are always solid (robust against tiny STL/domain mismatch gaps).
        PatchSurfaceField2D ground_height_field;
        ground_height_field.build_from_patch(
            samples, k_patch_bottom,
            [](const SamplePoint& s) { return float3(s.p.z, 0.0f, 0.0f); },
            float3(z0_lbmu, 0.0f, 0.0f));
        if (ground_height_field.has_samples()) {
            std::atomic<ulong> terrain_clipped_to_solid{ 0ul };
            parallel_for(solver.grid().cells(), [&](ulong n) {
                if ((solver.flags()[n] & CELL_SOLID) != 0u) return;
                uint x, y, z;
                solver.grid().coordinates(n, x, y, z);
                const float3 pos = solver.grid().position(x, y, z);
                const float zg = ground_height_field.eval(pos.x, pos.y).x;
                if (pos.z < zg) {
                    solver.flags()[n] = CELL_SOLID;
                    set_zero_velocity(n);
                    terrain_clipped_to_solid.fetch_add(1ul, std::memory_order_relaxed);
                }
            });
            if (terrain_clipped_to_solid.load(std::memory_order_relaxed) > 0ul) {
                println("| Terrain clip    | below-terrain cells forced to solid: " +
                        to_string(terrain_clipped_to_solid.load()) + "                    |");
            }
            for (ulong n = 0ul; n < solver.grid().cells(); ++n) {
                voxel_flags[(size_t)n] = solver.flags()[n];
            }
        }

        auto voxel_is_solid = [&](const uint xi, const uint yi, const uint zi) -> bool {
            return (voxel_flags[(size_t)solver.grid().index(xi, yi, zi)] & CELL_SOLID) != 0u;
        };

        std::atomic<ulong> velocity_mapped{ 0ul };
        std::atomic<ulong> velocity_missing_patch{ 0ul };
        std::atomic<ulong> outlet_cells{ 0ul };
        std::atomic<ulong> velocity_grounded{ 0ul };
        std::atomic<ulong> velocity_below_patch_support{ 0ul };
        parallel_for(solver.grid().cells(), [&](ulong n) {
            uint x, y, z;
            solver.grid().coordinates(n, x, y, z);
            if (z == 0u) {
                solver.flags()[n] = CELL_SOLID;
                set_zero_velocity(n);
                return;
            }

            const int patch = boundary_cell_to_patch(x, y, z, Nx, Ny, Nz);
            if (patch < 0) return;

            const bool voxel_self_solid = (voxel_flags[(size_t)n] & CELL_SOLID) != 0u;
            bool side_below_ground = false;
            if (patch == k_patch_west && Nx > 1u) {
                side_below_ground = voxel_is_solid(1u, y, z);
            }
            else if (patch == k_patch_east && Nx > 1u) {
                side_below_ground = voxel_is_solid(Nx - 2u, y, z);
            }
            else if (patch == k_patch_south && Ny > 1u) {
                side_below_ground = voxel_is_solid(x, 1u, z);
            }
            else if (patch == k_patch_north && Ny > 1u) {
                side_below_ground = voxel_is_solid(x, Ny - 2u, z);
            }

            if (voxel_self_solid || side_below_ground) {
                solver.flags()[n] = CELL_SOLID;
                set_zero_velocity(n);
                velocity_grounded.fetch_add(1ul, std::memory_order_relaxed);
                return;
            }

            const PatchSurfaceField2D& face_field = patch_velocity_fields[(size_t)patch];
            if (!face_field.has_samples()) {
                velocity_missing_patch.fetch_add(1ul, std::memory_order_relaxed);
                return;
            }

            float3 pos = solver.grid().position(x, y, z);
            float a = 0.0f;
            float b = 0.0f;
            const bool is_side_patch =
                (patch == k_patch_west || patch == k_patch_east ||
                 patch == k_patch_south || patch == k_patch_north);
            if (is_side_patch && top_sponge_side_ref_z_cap >= 0 && (int)z > top_sponge_side_ref_z_cap) {
                pos.z = solver.grid().position(x, y, (uint)top_sponge_side_ref_z_cap).z;
            }
            if (!patch_surface_coordinates(patch, pos, a, b)) {
                velocity_missing_patch.fetch_add(1ul, std::memory_order_relaxed);
                return;
            }

            if (is_side_patch && face_field.below_sample_support(a, b)) {
                solver.flags()[n] = CELL_SOLID;
                set_zero_velocity(n);
                velocity_below_patch_support.fetch_add(1ul, std::memory_order_relaxed);
                return;
            }

            solver.flags()[n] = CELL_EQUILIBRIUM;
            if (downstream_open_face && patch == downstream_patch) {
                outlet_cells.fetch_add(1ul, std::memory_order_relaxed);
                return;
            }

            const float3 u = face_field.eval(a, b);
            solver.velocity().x[n] = u.x;
            solver.velocity().y[n] = u.y;
            solver.velocity().z[n] = u.z;
            velocity_mapped.fetch_add(1ul, std::memory_order_relaxed);
        });
        println("| Velocity BC     | patch-driven 2D mapping: " + to_string(velocity_mapped.load()) + " cells                 |");
        if (velocity_grounded.load(std::memory_order_relaxed) > 0ul) {
            println("|                 | underground no-slip cells: " + to_string(velocity_grounded.load()) + "                     |");
        }
        if (velocity_below_patch_support.load(std::memory_order_relaxed) > 0ul) {
            println("|                 | side cells below terrain support -> solid: " + to_string(velocity_below_patch_support.load()) + "     |");
        }
        if (outlet_cells.load(std::memory_order_relaxed) > 0ul) {
            println("|                 | downstream outlet cells: " + to_string(outlet_cells.load()) + " (no fixed velocity)        |");
        }
        if (velocity_missing_patch.load(std::memory_order_relaxed) > 0ul) {
            println("|                 | WARNING: missing patch samples for " + to_string(velocity_missing_patch.load()) + " cells         |");
        }

        thermal_bc.map_patch_faces(downstream_patch);
        emit_progress_stage("interface_interpolation",
                            "Interface interpolation",
                            "Patch-driven 2D boundary mapping completed",
                            1ll,
                            1ll,
                            false);
        thermal_bc.apply_ground_temperature_plane_bc("patch-2d");
        thermal_bc.report_temperature_bc_summary("patch-2d");
        print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

        auto eval = [&](const float3& q)->float3 {
            if (downstream_patch < k_patch_top || downstream_patch > k_patch_east) return float3(0.0f);
            const PatchSurfaceField2D& f = patch_velocity_fields[(size_t)downstream_patch];
            if (!f.has_samples()) return float3(0.0f);
            float a = 0.0f;
            float b = 0.0f;
            if (!patch_surface_coordinates(downstream_patch, q, a, b)) return float3(0.0f);
            return f.eval(a, b);
        };
        run_case_flux_correction(solver, flux_correction, downstream_bc, eval, [&]() {
            if (use_temperature_bc) thermal_bc.report_temperature_bc_summary("patch-2d/post-flux");
        });
    }
    else if (use_high_order) {
        KNNInterpolatorHD knn_hd(std::move(P), std::move(Uv));
        // Calculate z base threshold in LBM units
        const float z_base_threshold_lbmu = units.x(z_si_offset) + z0_lbmu + 0.5f;
        InletVelocityFieldHD inlet_hd(knn_hd, z_base_threshold_lbmu);
        apply_inlet_outlet_hd(solver, downstream_bc, inlet_hd, downstream_open_face, 500000ull, true, top_sponge_side_ref_z_cap);

        thermal_bc.map_high_order(std::move(P_temp), std::move(Tv), z_base_threshold_lbmu);
        thermal_bc.apply_ground_temperature_plane_bc("high-order");
        thermal_bc.report_temperature_bc_summary("high-order");
        print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

        auto eval = [&](const float3& q)->float3 { return inlet_hd(q); };
        run_case_flux_correction(solver, flux_correction, downstream_bc, eval, [&]() {
            if (use_temperature_bc) thermal_bc.report_temperature_bc_summary("high-order/post-flux");
        });
    }
    else {
        NearestNeighborInterpolator nn(std::move(P), std::move(Uv));
        InletVelocityField inlet(nn, z0_lbmu, z_off);
        apply_inlet_outlet(solver, downstream_bc, inlet, downstream_open_face, 500000ull, true, top_sponge_side_ref_z_cap);

        thermal_bc.map_low_order(std::move(P_temp), std::move(Tv), z0_lbmu, z_off);
        thermal_bc.apply_ground_temperature_plane_bc("low-order");
        thermal_bc.report_temperature_bc_summary("low-order");
        print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

        auto eval = [&](const float3& q)->float3 { return inlet(q); };
        run_case_flux_correction(solver, flux_correction, downstream_bc, eval, [&]() {
            if (use_temperature_bc) thermal_bc.report_temperature_bc_summary("low-order/post-flux");
        });
    }
}

} // namespace culbm::bc
