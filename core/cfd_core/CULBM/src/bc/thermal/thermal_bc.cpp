#include "thermal_bc.hpp"
#include "../worker_threads.hpp"
#include "../inflow/interpolation.hpp"
#include "../inflow/interpolation_hd.hpp"
#include "../../case/context.hpp"
#include "../../case/report.hpp"
#include "../../run/console_log.hpp"

#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <limits>

namespace culbm::bc {

using namespace culbm::cases;
using run::now_str;

void resolve_thermal_coupling_stage(CaseContext& ctx) {
    CaseSettings& deck = ctx.deck;
    const InflowData& inflow = ctx.inflow;
    ThermalReference& thermal = ctx.thermal;
    thermal.thermal_boundary_available =
        !deck.dataset_generation && !deck.profile_generation &&
        inflow.csv_has_temperature && inflow.csv_rows_with_temperature > 0ul;
    thermal.thermal_coupling =
        luw::cfd::resolve_thermal_coupling(
            deck.buoyancy_enabled, thermal.thermal_boundary_available);
    deck.buoyancy_enabled = thermal.thermal_coupling.buoyancy_active;
    if (thermal.thermal_coupling.fallback_applied) {
        print_kv_row("Buoyancy active",
            "false (fallback: no valid temperature boundary)");
        print_kv_row("WARNING",
            "buoyancy requested without a temperature boundary; feedback disabled");
    }
    else {
        print_kv_row("Buoyancy active", deck.buoyancy_enabled
            ? string("true (temperature boundary detected)")
            : string("false (requested off)"));
    }
}

NormalTemperatureBoundary::NormalTemperatureBoundary(const CaseContext& ctx, culbm::solver::Solver& solver, const bool use_patch_face_bc)
    : ctx_(ctx), solver(solver), use_temperature_bc(ctx.inflow.use_temperature_bc),
      T_bc_min_lbm(ctx.inflow.T_bc_min_lbm), T_bc_max_lbm(ctx.inflow.T_bc_max_lbm),
      downstream_bc(ctx.deck.downstream_bc), downstream_open_face(ctx.deck.downstream_open_face),
      patch_temperature_fields(6) {
    const std::vector<SamplePoint>& samples = ctx.inflow.samples;
    const bool csv_has_patch = ctx.inflow.csv_has_patch;
    if (use_patch_face_bc && use_temperature_bc) {
        std::vector<ulong> t_patch_counts(6, 0ul);
        std::vector<float> t_patch_min(6, +FLT_MAX);
        std::vector<float> t_patch_max(6, -FLT_MAX);
        for (const auto& s : samples) {
            if (s.patch < 0 || s.patch > 5) continue;
            const size_t pid = (size_t)s.patch;
            t_patch_counts[pid]++;
            t_patch_min[pid] = fminf(t_patch_min[pid], s.T);
            t_patch_max[pid] = fmaxf(t_patch_max[pid], s.T);
        }
        for (int patch = k_patch_top; patch <= k_patch_east; ++patch) {
            patch_temperature_fields[(size_t)patch].build_from_patch(
                samples, patch,
                [](const SamplePoint& s) { return float3(s.T, 0.0f, 0.0f); },
                float3(1.0f, 0.0f, 0.0f));
            if (t_patch_counts[(size_t)patch] > 0ul) {
                println("| T patch         | " + string(patch_name(patch)) + ": n=" + to_string(t_patch_counts[(size_t)patch]) +
                        ", SI " + to_string(fmtf(units.si_T(t_patch_min[(size_t)patch]))) + " .. " +
                        to_string(fmtf(units.si_T(t_patch_max[(size_t)patch]))) + " K                    |");
            }
            else {
                println("| T patch         | " + string(patch_name(patch)) + ": n=0                                           |");
            }
        }
    }

    if (use_temperature_bc && csv_has_patch) {
        ground_temp_plane.build_from_patch0_samples(samples, 1.0f);
        use_ground_temperature_bc = ground_temp_plane.has_samples();
        if (use_ground_temperature_bc) {
            println("| Ground T plane  | enabled from patch=0 (" + to_string(ground_temp_plane.raw_count()) +
                    " samples, grid " + to_string(ground_temp_plane.nx()) + "x" +
                    to_string(ground_temp_plane.ny()) + ", mode=" +
                    (ground_temp_plane.structured_ready() ? string("2D bilinear") : string("2D nearest")) + ") |");
        }
        else {
            println("| Ground T plane  | patch column detected, but no patch=0 samples found        |");
        }
    }
}

void NormalTemperatureBoundary::map_patch_faces(const int downstream_patch) {
    const uint Nx = solver.grid().size(0u);
    const uint Ny = solver.grid().size(1u);
    const uint Nz = solver.grid().size(2u);
    if (use_temperature_bc) {
        std::atomic<ulong> temperature_mapped{ 0ul };
        std::atomic<ulong> temperature_missing_patch{ 0ul };
        parallel_for(solver.grid().cells(), [&](ulong n) {
            uint x, y, z;
            solver.grid().coordinates(n, x, y, z);
            if (z == 0u) return;

            const int patch = boundary_cell_to_patch(x, y, z, Nx, Ny, Nz);
            if (patch < 0) return;
            if ((solver.flags()[n] & CELL_SOLID) != 0u) return; // solid temperature is handled by ground patch plane

            if (downstream_open_face && patch == downstream_patch) return;

            const PatchSurfaceField2D& face_field_t = patch_temperature_fields[(size_t)patch];
            if (!face_field_t.has_samples()) {
                temperature_missing_patch.fetch_add(1ul, std::memory_order_relaxed);
                return;
            }

            const float3 pos = solver.grid().position(x, y, z);
            float a = 0.0f;
            float b = 0.0f;
            if (!patch_surface_coordinates(patch, pos, a, b)) {
                temperature_missing_patch.fetch_add(1ul, std::memory_order_relaxed);
                return;
            }
            float Tn = face_field_t.eval(a, b).x;
            if (use_temperature_bc) {
                Tn = fminf(fmaxf(Tn, T_bc_min_lbm), T_bc_max_lbm);
            }
            solver.temperature()[n] = Tn;
            solver.flags()[n] = (uchar)(solver.flags()[n] | CELL_THERMAL);
            temperature_mapped.fetch_add(1ul, std::memory_order_relaxed);
        });
        println("| Temperature BC  | patch-driven 2D mapping: " + to_string(temperature_mapped.load()) + " cells              |");
        if (temperature_missing_patch.load(std::memory_order_relaxed) > 0ul) {
            println("|                 | WARNING: missing patch samples for " + to_string(temperature_missing_patch.load()) + " cells      |");
        }
    }
}

void NormalTemperatureBoundary::map_high_order(std::vector<float3>&& P_temp, std::vector<float3>&& Tv,
                                               const float z_base_threshold_lbmu) {
    if (use_temperature_bc) {
        KNNInterpolatorHD knn_hd_T(std::move(P_temp), std::move(Tv));
        const uint Nx = solver.grid().size(0u), Ny = solver.grid().size(1u), Nz = solver.grid().size(2u);
        std::vector<ulong> temp_boundary_indices;
        temp_boundary_indices.reserve((ulong)Nx * (ulong)Ny + (ulong)Nx * (ulong)Nz + (ulong)Ny * (ulong)Nz);
        for (ulong n = 0ul; n < solver.grid().cells(); ++n) {
            uint x, y, z;
            solver.grid().coordinates(n, x, y, z);
            if (z == 0u) continue;
            const bool on_outer = (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u);
            if (on_outer) temp_boundary_indices.push_back(n);
        }

        const ulong Nbc = (ulong)temp_boundary_indices.size();
        println("| [" + now_str() + "] temperature BC init (HD): " + to_string(Nbc) + " cells.                    |");
        emit_progress_stage("interface_interpolation",
                            "Interface interpolation",
                            "High-order temperature interpolation",
                            0ll,
                            (long long)std::max<ulong>(Nbc, 1ul),
                            Nbc == 0ul);
        if (Nbc > 0ul) {
            unsigned num_threads = available_worker_threads();
#ifdef _MSC_VER
            char* env_dup = nullptr;
            size_t env_len = 0;
            if (_dupenv_s(&env_dup, &env_len, "LBM_NUM_THREADS") == 0 && env_dup != nullptr) {
                const long v = std::strtol(env_dup, nullptr, 10);
                if (v > 0) num_threads = (unsigned)std::min<long>(v, 512l);
                std::free(env_dup);
            }
#else
            if (const char* env_p = std::getenv("LBM_NUM_THREADS")) {
                const long v = std::strtol(env_p, nullptr, 10);
                if (v > 0) num_threads = (unsigned)std::min<long>(v, 512l);
            }
#endif
            println("| Temperature BC  | worker threads = " + to_string(num_threads) + "                                   |");
            // Split temperature boundary cells into 5 faces, same priority/order as HD velocity mapping.
            std::vector<std::vector<ulong>> face_indices(5);
            for (int f = 0; f < 5; ++f) {
                face_indices[f].reserve(Nbc / 5ul + 1ul);
            }
            for (ulong i = 0ul; i < Nbc; ++i) {
                const ulong n = temp_boundary_indices[i];
                uint x, y, z;
                solver.grid().coordinates(n, x, y, z);
                int face = -1;
                if (x == 0u) face = 0;
                else if (x == Nx - 1u) face = 1;
                else if (y == 0u) face = 2;
                else if (y == Ny - 1u) face = 3;
                else if (z == Nz - 1u) face = 4;
                if (face >= 0) face_indices[face].push_back(n);
            }

            ulong mapped_total = 0ul;
            for (int f = 0; f < 5; ++f) {
                const std::vector<ulong>& fi = face_indices[f];
                const ulong Nface = (ulong)fi.size();
                if (Nface == 0ul) continue;
                if (downstream_open_face) {
                    const int patch_id =
                        (f == 0) ? k_patch_west :
                        (f == 1) ? k_patch_east :
                        (f == 2) ? k_patch_south :
                        (f == 3) ? k_patch_north :
                                   k_patch_top;
                    if (patch_id == downstream_to_patch(downstream_bc)) continue;
                }

                const char* face_name = "unknown";
                if (f == 0) face_name = "x == 0";
                else if (f == 1) face_name = "x == Nx - 1";
                else if (f == 2) face_name = "y == 0";
                else if (f == 3) face_name = "y == Ny - 1";
                else if (f == 4) face_name = "z == Nz - 1";

                println("| [" + now_str() + "] temperature BC init (HD) surface " + to_string(f + 1) +
                        "/5 (" + string(face_name) + "): " + to_string(Nface) + " cells");

                const ulong chunk = std::max<ulong>(1024ul, Nface / ((ulong)num_threads * 32ul) + 1ul);
                std::atomic<ulong> next_idx{ 0ul };
                std::atomic<ulong> processed{ 0ul };
                std::atomic<bool> done{ false };

                std::thread progress_thread([&]() {
                    while (!done.load(std::memory_order_relaxed)) {
                        const ulong current = processed.load(std::memory_order_relaxed);
                        const double pct = 100.0 * (double)current / (double)Nface;
                        if (luw_progress_gui_mode()) {
                            emit_progress_stage("interface_interpolation",
                                                "Interface interpolation",
                                                "High-order temperature surface " + to_string((ulong)f + 1ull) +
                                                    "/5 (" + string(face_name) + "): " +
                                                    to_string(current) + "/" + to_string(Nface) + " cells",
                                                (long long)current,
                                                (long long)Nface,
                                                false);
                        } else {
                            reprint("| [" + now_str() + "] temperature BC init (HD): " + to_string((float)pct, 3u) + "% |");
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    }
                    if (luw_progress_gui_mode()) {
                        emit_progress_stage("interface_interpolation",
                                            "Interface interpolation",
                                            "High-order temperature surface " + to_string((ulong)f + 1ull) +
                                                "/5 (" + string(face_name) + ") completed",
                                            (long long)Nface,
                                            (long long)Nface,
                                            false);
                    } else {
                        println("| [" + now_str() + "] temperature BC init (HD): 100.000% |");
                    }
                });

                std::vector<std::thread> workers;
                workers.reserve(num_threads);
                for (unsigned t = 0u; t < num_threads; ++t) {
                    workers.emplace_back([&]() {
                        ulong local_done = 0ul;
                        for (;;) {
                            const ulong start = next_idx.fetch_add(chunk, std::memory_order_relaxed);
                            if (start >= Nface) break;
                            const ulong end = std::min<ulong>(start + chunk, Nface);
                            for (ulong i = start; i < end; ++i) {
                                const ulong n = fi[i];
                                uint x, y, z;
                                solver.grid().coordinates(n, x, y, z);
                                const float3 pos = solver.grid().position(x, y, z);
                                float Tn = (pos.z < z_base_threshold_lbmu) ? 1.0f : knn_hd_T.eval(pos).x;
                                if (use_temperature_bc) {
                                    Tn = fminf(fmaxf(Tn, T_bc_min_lbm), T_bc_max_lbm);
                                }
                                solver.temperature()[n] = Tn;
                                solver.flags()[n] = (uchar)(solver.flags()[n] | CELL_THERMAL);
                                if (++local_done == 512ul) {
                                    processed.fetch_add(local_done, std::memory_order_relaxed);
                                    local_done = 0ul;
                                }
                            }
                        }
                        if (local_done) processed.fetch_add(local_done, std::memory_order_relaxed);
                    });
                }
                for (auto& th : workers) th.join();
                done.store(true, std::memory_order_relaxed);
                if (progress_thread.joinable()) progress_thread.join();
                mapped_total += Nface;
            }
            println("| Temperature BC  | per-face interpolation done on 5 boundary surfaces        |");
            println("| Temperature BC  | mapped " + to_string(mapped_total) + "/" + to_string(Nbc) + " cells (high-order)      |");
            emit_progress_stage("interface_interpolation",
                                "Interface interpolation",
                                "High-order temperature interpolation completed",
                                (long long)mapped_total,
                                (long long)Nbc,
                                false);
        }
    }
}

void NormalTemperatureBoundary::map_low_order(std::vector<float3>&& P_temp, std::vector<float3>&& Tv,
                                              const float z0_lbmu, const float z_off) {
    if (use_temperature_bc) {
        NearestNeighborInterpolator nn_T(std::move(P_temp), std::move(Tv));
        std::atomic<ulong> temperature_mapped{ 0ul };
        const uint Nx = solver.grid().size(0u), Ny = solver.grid().size(1u), Nz = solver.grid().size(2u);
        const float z_temperature_threshold_lbmu = z0_lbmu + z_off;
        parallel_for(solver.grid().cells(), [&](ulong n) {
            uint x, y, z;
            solver.grid().coordinates(n, x, y, z);
            if (z == 0u) return;

            const bool outlet = downstream_open_face &&
                is_downstream_boundary_cell(x, y, z, Nx, Ny, Nz, downstream_bc);

            const bool inlet_face =
                (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u) && !outlet;
            if (!inlet_face) return;

            const float3 pos = solver.grid().position(x, y, z);
            float Tn = (pos.z < z_temperature_threshold_lbmu) ? 1.0f : nn_T.eval(pos).x;
            if (use_temperature_bc) {
                Tn = fminf(fmaxf(Tn, T_bc_min_lbm), T_bc_max_lbm);
            }
            solver.temperature()[n] = Tn;
            solver.flags()[n] = (uchar)(solver.flags()[n] | CELL_THERMAL);
            temperature_mapped.fetch_add(1ul, std::memory_order_relaxed);
        });
        println("| Temperature BC  | mapped " + to_string(temperature_mapped.load()) + " cells (low-order)                     |");
    }
}

void NormalTemperatureBoundary::apply_ground_temperature_plane_bc(const string& order_tag) {
    if (!use_ground_temperature_bc) return;

    const uint Nx = solver.grid().size(0u);
    const uint Ny = solver.grid().size(1u);

    std::vector<float> xy_temp_cache((ulong)Nx * (ulong)Ny, std::numeric_limits<float>::quiet_NaN());
    ulong mapped_cells = 0ul;
    ulong unique_xy = 0ul;

    for (ulong n = 0ul; n < solver.grid().cells(); ++n) {
        if ((solver.flags()[n] & CELL_SOLID) == 0u) continue; // solid only

        uint x, y, z;
        solver.grid().coordinates(n, x, y, z);
        (void)z;

        const ulong xyid = (ulong)y * (ulong)Nx + (ulong)x;
        float Txy = xy_temp_cache[xyid];
        if (!std::isfinite(Txy)) {
            const float3 pos_xy = solver.grid().position(x, y, 0u); // ignore z by construction
            Txy = ground_temp_plane.eval_xy(pos_xy.x, pos_xy.y);
            if (use_temperature_bc) {
                Txy = fminf(fmaxf(Txy, T_bc_min_lbm), T_bc_max_lbm);
            }
            xy_temp_cache[xyid] = Txy;
            unique_xy++;
        }

        // Solid temperature uses ground patch 2D interpolation for the whole vertical column at (x,y).
        solver.temperature()[n] = Txy;
        solver.flags()[n] = (uchar)(solver.flags()[n] | CELL_SOLID | CELL_THERMAL);
        mapped_cells++;
    }

    println("| Ground T plane  | mapped " + to_string(mapped_cells) +
            " solid cells, unique (x,y)=" + to_string(unique_xy) +
            " [" + order_tag + "]                                |");
    if (mapped_cells == 0ul) {
        println("| Ground T plane  | WARNING: no solid cells were found                          |");
    }
}

void NormalTemperatureBoundary::report_temperature_bc_summary(const string& tag) {
    if (!use_temperature_bc) return;
    ulong type_t_total = 0ul;
    ulong type_t_solid = 0ul;
    ulong type_t_fluid = 0ul;
    ulong type_t_invalid = 0ul;
    float t_solid_min = +FLT_MAX, t_solid_max = -FLT_MAX;
    float t_fluid_min = +FLT_MAX, t_fluid_max = -FLT_MAX;
    for (ulong n = 0ul; n < solver.grid().cells(); ++n) {
        if ((solver.flags()[n] & CELL_THERMAL) == 0u) continue;
        type_t_total++;
        const float Tn = solver.temperature()[n];
        if (!std::isfinite(Tn)) {
            type_t_invalid++;
            continue;
        }
        if ((solver.flags()[n] & CELL_SOLID) != 0u) {
            type_t_solid++;
            t_solid_min = fminf(t_solid_min, Tn);
            t_solid_max = fmaxf(t_solid_max, Tn);
        }
        else {
            type_t_fluid++;
            t_fluid_min = fminf(t_fluid_min, Tn);
            t_fluid_max = fmaxf(t_fluid_max, Tn);
        }
    }

    println("| Temperature BC  | summary [" + tag + "]: CELL_THERMAL total=" + to_string(type_t_total) +
            ", solid=" + to_string(type_t_solid) + ", fluid=" + to_string(type_t_fluid) + "            |");
    if (type_t_solid > 0ul) {
        println("| Temperature BC  | solid CELL_THERMAL range SI: " +
                to_string(fmtf(units.si_T(t_solid_min))) + " .. " +
                to_string(fmtf(units.si_T(t_solid_max))) + " K                      |");
    }
    if (type_t_fluid > 0ul) {
        println("| Temperature BC  | fluid CELL_THERMAL range SI: " +
                to_string(fmtf(units.si_T(t_fluid_min))) + " .. " +
                to_string(fmtf(units.si_T(t_fluid_max))) + " K                      |");
    }
    if (type_t_invalid > 0ul) {
        println("| Temperature BC  | WARNING: non-finite CELL_THERMAL cells = " + to_string(type_t_invalid) + "                         |");
    }
}

} // namespace culbm::bc
