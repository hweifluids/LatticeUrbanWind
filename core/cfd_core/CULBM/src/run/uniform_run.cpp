#include "uniform_run.hpp"
#include "console_log.hpp"
#include "physical_fields.hpp"
#include "../app/progress.hpp"
#include "../app/run_report.hpp"
#include "../case/context.hpp"
#include "../case/deck.hpp"
#include "../case/memory_fit.hpp"
#include "../case/report.hpp"
#include "../physical_output.hpp"
#include "../surface_ibm.hpp"
#include "../solver/field_view.hpp"

#include <cstring>
#include <iomanip>

namespace culbm::run {

using namespace culbm::cases;

AvgTurbulenceOutput avg_turbulence_output(const CaseSettings& deck) {
    AvgTurbulenceOutput fields;
    fields.tke = deck.output_tke_in_avg_vtk;
    fields.ti = deck.output_ti_in_avg_vtk;
    fields.tls = deck.output_tls_in_avg_vtk;
    return fields;
}

UniformRun::UniformRun(CaseContext& ctx, culbm::solver::Solver& solver, const std::string& vtk_prefix, const bool extra_spacing,
                       const double preprocess_seconds, PreStepUpdate pre_step_update)
    : ctx(ctx), deck(ctx.deck), domain(ctx.domain), inflow(ctx.inflow), physics(ctx.physics),
      geometry(ctx.geometry), terrain(ctx.terrain), probes(ctx.probes), solver(solver), vtk_prefix(vtk_prefix),
      extra_spacing(extra_spacing), preprocess_seconds(preprocess_seconds),
      pre_step_update(std::move(pre_step_update)) {}

double UniformRun::run() {
    seal_compact_ground();
    attach_surface_ibm();
    case_solver_clock.start();
    print_section_title("LBM SOLVER INFORMATION");

    const RunSchedule schedule = make_run_schedule(deck, !probes.probe_requests.empty());
    base_run_steps = schedule.base_run_steps;
    extra_run_steps = schedule.extra_run_steps;
    unsteady_interval = schedule.unsteady_interval;
    enable_unsteady_u_output = unsteady_interval > 0ull;
    const char* const write_flags_env = std::getenv("CULBM_WRITE_FLAGS");
    write_diagnostic_flags = write_flags_env != nullptr && (
        std::strcmp(write_flags_env, "1") == 0 ||
        std::strcmp(write_flags_env, "true") == 0 ||
        std::strcmp(write_flags_env, "TRUE") == 0 ||
        std::strcmp(write_flags_env, "on") == 0 ||
        std::strcmp(write_flags_env, "ON") == 0);
    total_steps = schedule.total_steps;
    do_avg = schedule.do_avg;
    avg_window = schedule.avg_window;
    avg_stride = schedule.avg_stride;
    avg_start_t = schedule.avg_start_t;
    probe_window = schedule.probe_window;
    probe_start_t = schedule.probe_start_t;

    open_outputs();
    open_diagnostics();
    vtk_saved_files.reserve(16u);
    print_run_settings();
    prepare_mean_field();
    resolve_probes();

    culbm::app::run_steps(solver, 0u, total_steps); // initialize fields once before first pre-step callback
    start_particle_advection();
    run_normal_speed_benchmark(std::function<void()>());
    const double avg_sample_seconds_hint = run_avg_phase_benchmark();
    configure_eta_model(avg_sample_seconds_hint);
    print_timing_plan(avg_sample_seconds_hint);
    print_runtime_section("SOLVER START");
    emit_solver_progress(true);
    Clock iteration_clock;
    const bool sync_each_step = (particle_advection_runtime != nullptr);
    while (solver.time() < total_steps) {
        iteration_clock.start();
        if (pre_step_update) {
            pre_step_update(solver, solver.time());
            refresh_particle_advection_velocity();
        }
        solver.set_step_writes_macros(step_needs_macros(solver.time() + 1ull));
        culbm::app::run_steps(solver, 1u, total_steps, sync_each_step);
        advance_particle_advection();
        sample_surface_ibm_diagnostics();
        sample_sgs_diagnostics();
        maybe_write_unsteady_u();
        const bool avg_sampled = process_post_step_samples();
        update_runtime_eta(iteration_clock.stop(), avg_sampled);
        if (!sync_each_step && do_avg && avg_window > 0ull && solver.time() + 1ull == avg_start_t) {
            solver.synchronize();
        }
        emit_solver_progress();
    }
    solver.synchronize();
    solver.set_step_writes_macros(true);
    emit_solver_progress(true);
    finish_particle_advection();
    write_final_transient();
    maybe_write_transform_info();
    finalize_avg();
    finalize_probes();
    const double case_seconds = print_task_finished();
    solver.barrier();
    return case_seconds;
}

void UniformRun::seal_compact_ground() {
    if (domain.cuda_ground_compaction_active) {
        const uint Nx_ground = solver.grid().size(0u);
        const uint Ny_ground = solver.grid().size(1u);
        const uint Nz_ground = solver.grid().size(2u);
        if (Nz_ground == 0u) {
            println("ERROR: CUDA compact ground has no vertical lattice layer.");
            wait(); exit(-1);
        }
        ulong repaired_ground_cells = 0ul;
        for (uint y = 0u; y < Ny_ground; ++y) {
            for (uint x = 0u; x < Nx_ground; ++x) {
                const ulong n = solver.grid().index(x, y, 0u);
                const uchar previous_flag = solver.flags()[n];
                if ((previous_flag & CELL_SOLID) == 0u || (previous_flag & CELL_EQUILIBRIUM) != 0u) repaired_ground_cells++;
                solver.flags()[n] = (uchar)((previous_flag & ~(CELL_SOLID | CELL_EQUILIBRIUM)) | CELL_SOLID);
                solver.velocity().x[n] = 0.0f;
                solver.velocity().y[n] = 0.0f;
                solver.velocity().z[n] = 0.0f;
            }
        }
        println("| CUDA ground     | z=0 sealed solid layer: " +
                to_string((ulong)Nx_ground * (ulong)Ny_ground) +
                " cells, repaired=" + to_string(repaired_ground_cells) + "                 |");
    }
}

void UniformRun::attach_surface_ibm() {
    if (deck.surface_ibm_settings.enabled && !solver.has_immersed_forcing()) {
        try {
            culbm::surface_ibm::attach(solver, geometry.surface_ibm_markers);
            const culbm::surface_ibm::Statistics statistics =
                culbm::surface_ibm::statistics(solver);
            print_kv_row(
                "SurfaceIBM attach",
                to_string((ulong)statistics.markers) +
                    " markers, " + to_string((ulong)statistics.fragments) +
                    " domain fragments, " +
                    to_string((ulong)statistics.active_cells) +
                    " active cells, cleared " +
                    to_string((ulong)statistics.cleared_solid_cells) +
                    " support-band solid cells, mode=" +
                    std::string(statistics.device_resident
                        ? "device-resident" : "portable-domain") +
                    (statistics.wall_model
                        ? std::string(", log-law wall stress (one-sided, voxel core kept)")
                        : std::string("")));
        }
        catch (const std::exception& exception) {
            println("| ERROR: SurfaceIBM attachment failed: " +
                std::string(exception.what()) + " |");
            wait();
            exit(-1);
        }
    }
}

void UniformRun::open_outputs() {
    results_vtk_dir = deck.parent + "/RESULTS/vtk/";
    results_netcdf_dir = deck.parent + "/RESULTS/netcdf/";
    vtk_dir = results_vtk_dir + vtk_prefix + deck.datetime + "_raw_";
    vtk_output_Nz = (domain.top_sponge_grid_extend && domain.Nz_cells_core < solver.grid().size(2u)) ? domain.Nz_cells_core : solver.grid().size(2u);
    std::filesystem::create_directories(std::filesystem::path(vtk_dir).parent_path());
    if (deck.culbm_output_global_netcdf || deck.culbm_output_agl_netcdf) {
        std::filesystem::create_directories(std::filesystem::path(results_netcdf_dir));
    }
    if (inflow.use_temperature_bc) {
        print_kv_row("Export mode", "include temperature T field in Kelvin");
    }
    if (vtk_output_Nz < solver.grid().size(2u)) {
        print_kv_row("VTK z output", "core Nz=" + to_string((ulong)vtk_output_Nz) +
            " of solver Nz=" + to_string((ulong)solver.grid().size(2u)) + " (top sponge omitted)");
    }
}

void UniformRun::open_diagnostics() {
    sgs_diagnostics_warned = false;
    if (deck.culbm_sgs_diagnostics_interval > 0u && culbm::mpi_is_root()) {
        const std::string diagnostics_filename =
            vtk_prefix + deck.datetime + "_culbm_sgs_diagnostics.csv";
        const std::filesystem::path diagnostics_path =
            std::filesystem::path(deck.parent) / "RESULTS" /
            diagnostics_filename;
        std::filesystem::create_directories(diagnostics_path.parent_path());
        sgs_diagnostics_output.open(
            diagnostics_path.string(), std::ios::out | std::ios::trunc);
        if (!sgs_diagnostics_output.is_open()) {
            println("| ERROR: cannot create SGS diagnostics file: " +
                diagnostics_path.string() + " |");
            wait();
            exit(-1);
        }
        print_kv_row("SGS diagnostics", diagnostics_path.string());
        sgs_diagnostics_output
            << "step,time_s,mean_nu_t_lbm,max_nu_t_lbm,"
               "mean_nu_t_m2_s,max_nu_t_m2_s,"
               "mean_mu_t_pa_s,max_mu_t_pa_s,count\n";
    }
    surface_ibm_diagnostics_warned = false;
    if (deck.surface_ibm_settings.enabled &&
        deck.surface_ibm_settings.diagnostics_interval > 0u &&
        culbm::mpi_is_root()) {
        const std::string diagnostics_filename =
            vtk_prefix + deck.datetime + "_surface_ibm_diagnostics.csv";
        const std::filesystem::path diagnostics_path =
            std::filesystem::path(deck.parent) / "RESULTS" /
            diagnostics_filename;
        std::filesystem::create_directories(diagnostics_path.parent_path());
        surface_ibm_diagnostics_output.open(
            diagnostics_path.string(), std::ios::out | std::ios::trunc);
        if (!surface_ibm_diagnostics_output.is_open()) {
            println("| ERROR: cannot create SurfaceIBM diagnostics file: " +
                diagnostics_path.string() + " |");
            wait();
            exit(-1);
        }
        print_kv_row("IBM diagnostics", diagnostics_path.string());
        surface_ibm_diagnostics_output
            << "step,time_s,markers,fragments,active_cells,device_resident,cleared_solid_cells,iterations,converged,"
               "nonconverged_steps,max_slip_lu,peak_max_slip_lu,rms_slip_lu,"
               "reaction_fx_lu,reaction_fy_lu,"
               "reaction_fz_lu,reaction_tx_lu,reaction_ty_lu,reaction_tz_lu\n";
    }
}

void UniformRun::print_run_settings() {
    print_kv_row("Run steps", to_string(total_steps) + (deck.run_nstep_override > 0ull ? " (run_nstep override)" : " (default)"));
    if (extra_run_steps > 0ull) {
        print_kv_row("Extra steps", to_string(extra_run_steps) + " (legacy research_output)");
    }
    if (enable_unsteady_u_output) {
        print_kv_row("Unsteady u VTK", "every " + to_string(unsteady_interval) + " steps");
        print_kv_row("Unsteady start", to_string(deck.culbm_unsteady_start_step) + " steps");
    }
    if (write_diagnostic_flags) {
        print_kv_row("Diagnostic VTK", "raw flags enabled by CULBM_WRITE_FLAGS");
    }
}

void UniformRun::prepare_mean_field() {
    if (avg_window > 0ull) {
        print_kv_row("Avg stride", "sample every " + to_string(avg_stride) + " step(s) in purge_avg window");
    }
    Ncells = solver.grid().cells();
    include_temperature_avg = inflow.use_temperature_bc;
    deferred_particle_gpu_memory_mb =
        estimate_particle_advection_extra_mb_per_device(
        deck, uint3(solver.grid().size(0u), solver.grid().size(1u), solver.grid().size(2u)),
            &deck.particle_advection_settings);

    avg_count = 0ull;
    use_device_statistics = avg_window > 0ull &&
        solver.enable_statistics(
            include_temperature_avg,
            deferred_particle_gpu_memory_mb);
    if (avg_window > 0ull) {
        if (use_device_statistics) {
            print_kv_row("Mean-field", "CUDA-resident Welford accumulator enabled");
        }
        else {
            print_kv_row("Mean-field", "host fallback (CUDA accumulator unavailable or disabled)");
            avg_u.assign((size_t)(Ncells * 3ull), 0.0f);
            avg_rho.assign((size_t)Ncells, 0.0f);
            if (include_temperature_avg) {
                avg_T.assign((size_t)Ncells, 0.0f);
            }
            M2_u.assign((size_t)Ncells, 0.0f);
            M2_v.assign((size_t)Ncells, 0.0f);
            M2_w.assign((size_t)Ncells, 0.0f);
        }
    }
}

void UniformRun::resolve_probes() {
    dt_si = (double)deck.cell_m * ((double)domain.lbm_ref_u / (double)physics.si_ref_u);
    if (!probes.probe_requests.empty()) {
        if (!probes.probe_geo_mapping.valid) {
            print_kv_row("Probes", "disabled: geographic mapping is unavailable");
        }
        else {
            const auto is_inside_domain = [&](const double x_si, const double y_si) -> bool {
                return std::isfinite(x_si) && std::isfinite(y_si) &&
                    x_si >= 0.0 && x_si <= (double)deck.si_size.x &&
                    y_si >= 0.0 && y_si <= (double)deck.si_size.y;
            };
            const auto resolve_probe = [&](const ProbeRequest& request, ResolvedProbe& probe, std::string& warning) -> bool {
                const double base_lon = request.uses_center ? probes.probe_geo_mapping.center_lon_deg : request.lon_deg;
                const double base_lat = request.uses_center ? probes.probe_geo_mapping.center_lat_deg : request.lat_deg;
                double base_x_si = 0.0, base_y_si = 0.0;
                if (!project_probe_local_xy(base_lon, base_lat, probes.probe_geo_mapping, base_x_si, base_y_si)) {
                    warning = "projection failed";
                    return false;
                }
                if (!is_inside_domain(base_x_si, base_y_si)) {
                    warning = "base point is outside CFD domain";
                    return false;
                }

                double final_x_si = base_x_si;
                double final_y_si = base_y_si;
                if (request.offset.mode == ProbeOffsetMode::GRID_CELLS) {
                    const uint base_x = snap_probe_index(base_x_si, solver.grid().size(0u), deck.cell_m);
                    const uint base_y = snap_probe_index(base_y_si, solver.grid().size(1u), deck.cell_m);
                    final_x_si = (double)base_x * (double)deck.cell_m
                        + (double)request.offset.east_cells * (double)deck.cell_m * probes.probe_geo_mapping.east_dx
                        + (double)request.offset.north_cells * (double)deck.cell_m * probes.probe_geo_mapping.north_dx;
                    final_y_si = (double)base_y * (double)deck.cell_m
                        + (double)request.offset.east_cells * (double)deck.cell_m * probes.probe_geo_mapping.east_dy
                        + (double)request.offset.north_cells * (double)deck.cell_m * probes.probe_geo_mapping.north_dy;
                }
                else if (request.offset.mode == ProbeOffsetMode::METERS) {
                    final_x_si = base_x_si
                        + request.offset.east_m * probes.probe_geo_mapping.east_dx
                        + request.offset.north_m * probes.probe_geo_mapping.north_dx;
                    final_y_si = base_y_si
                        + request.offset.east_m * probes.probe_geo_mapping.east_dy
                        + request.offset.north_m * probes.probe_geo_mapping.north_dy;
                }

                if (!is_inside_domain(final_x_si, final_y_si)) {
                    warning = "offset point is outside CFD domain";
                    return false;
                }

                probe = ResolvedProbe{};
                probe.request = request;
                probe.x = snap_probe_index(final_x_si, solver.grid().size(0u), deck.cell_m);
                probe.y = snap_probe_index(final_y_si, solver.grid().size(1u), deck.cell_m);
                probe.snapped_x_si = (double)probe.x * (double)deck.cell_m;
                probe.snapped_y_si = (double)probe.y * (double)deck.cell_m;
                probe.label = request.raw_token;

                for (uint z = 0u; z < solver.grid().size(2u); ++z) {
                    const ulong n = solver.grid().index(probe.x, probe.y, z);
                    if ((solver.flags()[n] & CELL_SOLID) != 0u) continue;
                    probe.z_indices.push_back(z);
                }
                if (probe.z_indices.empty()) {
                    warning = "resolved column has no fluid cell";
                    return false;
                }

                const uint z0 = probe.z_indices.front();
                probe.heights_si.reserve(probe.z_indices.size());
                for (const uint z : probe.z_indices) {
                    probe.heights_si.push_back((float)(((double)z - (double)z0 + 0.5) * (double)deck.cell_m));
                }
                probe.times_si.reserve((size_t)probe_window);
                probe.velocity_series_si.reserve((size_t)probe_window * probe.z_indices.size() * 3ull);
                return true;
            };

            std::vector<std::string> used_stems;
            for (const ProbeRequest& request : probes.probe_requests) {
                ResolvedProbe probe;
                std::string warning;
                if (!resolve_probe(request, probe, warning)) {
                    println("| WARNING: probe '" + request.raw_token + "' ignored: " + warning + "                |");
                    continue;
                }
                std::string stem = make_probe_file_stem(request, probes.probe_geo_mapping, vtk_prefix);
                if (contains(used_stems, stem)) {
                    uint suffix = 2u;
                    std::string unique_stem = stem;
                    while (contains(used_stems, unique_stem)) {
                        unique_stem = stem + "_" + to_string((ulong)suffix++);
                    }
                    stem = unique_stem;
                }
                used_stems.push_back(stem);
                probe.file_stem = stem;
                resolved_probes.push_back(std::move(probe));
            }

            if (resolved_probes.empty()) {
                print_kv_row("Probes", "0 valid probe column after geometry/domain checks");
            }
            else {
                const std::string window_desc = probe_window >= total_steps
                    ? "entire run"
                    : ("last " + to_string(probe_window) + " step(s)");
                print_kv_row("Probes", to_string((ulong)resolved_probes.size()) + " active, " + window_desc);
                bool first_probe_row = true;
                for (const ResolvedProbe& probe : resolved_probes) {
                    print_kv_row(first_probe_row ? "Probe cell" : "",
                        probe.file_stem + " -> (" + to_string((ulong)probe.x) + "," + to_string((ulong)probe.y) +
                        "), levels=" + to_string((ulong)probe.z_indices.size()));
                    first_probe_row = false;
                }
            }
        }
    }
    for (const ResolvedProbe& probe : resolved_probes) {
        for (const uint z : probe.z_indices) {
            device_probe_indices.push_back(solver.grid().index(probe.x, probe.y, z));
        }
    }
    use_device_probes =
        !device_probe_indices.empty() &&
        solver.configure_probes(
            device_probe_indices,
            deferred_particle_gpu_memory_mb);
    if (!device_probe_indices.empty()) {
        print_kv_row("Probe transfer", use_device_probes
            ? "CUDA sparse gather enabled"
            : "host full-field fallback");
    }
}

void UniformRun::start_particle_advection() {
    if (deck.particle_advection_settings.enabled) {
        try {
            const double spacing_si = (double)units.si_x(1.0f);
            float3 particle_origin = (float)spacing_si * float3(
                0.5f - 0.5f * (float)solver.grid().size(0u),
                0.5f - 0.5f * (float)solver.grid().size(1u),
                0.5f - 0.5f * (float)solver.grid().size(2u)) + vtk_origin_shift;
            particle_origin.z -= terrain.physical_output_z_zero_si;
            culbm::ParticleAdvectionGeometry geometry;
            geometry.lattice_origin_m = {
                (double)particle_origin.x,
                (double)particle_origin.y,
                (double)particle_origin.z};
            geometry.spacing_m = spacing_si;
            particle_advection_runtime =
                std::make_unique<culbm::ParticleAdvection>(
                    solver,
                    deck.particle_advection_settings,
                    geometry,
                    (std::uint64_t)total_steps,
                    std::filesystem::path(deck.parent),
                    std::filesystem::path(deck.parent) / "RESULTS" / "particles",
                    vtk_prefix + deck.datetime);
            particle_advection_runtime->initialize();
            particle_advection_runtime->write_snapshot_if_due(0u, true);
            print_runtime_kv(
                "GPU advection",
                std::string(culbm::particle_advection::mode_name(
                    deck.particle_advection_settings.mode)) +
                ", " +
                culbm::particle_advection::integrator_name(
                    deck.particle_advection_settings.integrator) +
                ", particles=" +
                std::to_string(
                    particle_advection_runtime->particle_count()) +
                ", dt=" +
                to_string((double)units.si_t((ulong)1u), 9u) +
                " s");
        }
        catch (const std::exception& exception) {
            println("ERROR: CULBM particle-advection initialization failed: " +
                    std::string(exception.what()));
            wait();
            exit(-1);
        }
    }
}

void UniformRun::finish_particle_advection() {
    if (particle_advection_runtime) {
        try {
            particle_advection_runtime->write_snapshot_if_due(
                (std::uint64_t)solver.time(), true);
            print_runtime_kv(
                "Particle migration",
                std::to_string(
                    particle_advection_runtime->migrated_particles()) +
                " local GPU-domain transfer(s)");
        }
        catch (const std::exception& exception) {
            println("ERROR: final particle snapshot failed: " +
                    std::string(exception.what()));
            wait();
            exit(-1);
        }
    }
}

std::string UniformRun::format_estimated_time(const double seconds, const bool normal_hint_ready) {
    return normal_hint_ready ? print_time(seconds) : std::string("pending normal-speed sample");
}

void UniformRun::emit_solver_progress(const bool force) {
    if(!luw_progress_gui_mode()) return;
    const auto now = std::chrono::steady_clock::now();
    if(!force && last_solver_progress.time_since_epoch().count() != 0 &&
       now - last_solver_progress < std::chrono::milliseconds(120) &&
       solver.time() < total_steps) {
        return;
    }
    last_solver_progress = now;
    const double eta = culbm::app::run_progress().eta_seconds();
    emit_progress_stage(
        "solve",
        "Solving CFD",
        to_string(solver.time()) + "/" + to_string(total_steps) +
            " steps | " + to_string(culbm::app::run_progress().steps_per_second(), 3u) +
            " Steps/s | ETA " + (eta >= 0.0 ? print_time(eta) : std::string("--")),
        (long long)solver.time(),
        (long long)total_steps,
        false);
}

double UniformRun::print_task_finished() {
    const double solver_seconds = case_solver_clock.stop();
    const double case_total_seconds = preprocess_seconds + solver_seconds;
    ctx.last_normal_steps_per_second_hint = culbm::app::run_progress().plain_steps_per_second();
    print_kv_row("Task finished", "[" + now_str() + "]");
    print_kv_row("Preprocess time", print_time(preprocess_seconds) + " (recorded)");
    print_kv_row("Solver time", print_time(solver_seconds));
    print_kv_row("LBM case total", print_time(case_total_seconds) + " (preprocess + solver)");
    if (extra_spacing) {
        println("|"+string((uint)CONSOLE_WIDTH-2u, ' ')+"|");
    }
    return case_total_seconds;
}

double UniformRun::run_avg_phase_benchmark() {
    if (avg_window == 0ull) return 0.0;
    constexpr uint k_avg_benchmark_iterations = 10u;
    print_runtime_kv("Avg benchmark", "run " + to_string((ulong)k_avg_benchmark_iterations) + " iterations before solve");
    emit_progress_stage("speed_estimate",
                        "Estimating solve speed",
                        "Benchmarking mean-field post-processing",
                        0ll,
                        (long long)k_avg_benchmark_iterations,
                        false);

    reset_average_accumulator();

    Clock bench_clock;
    double total_seconds = 0.0;
    for (uint i = 0u; i < k_avg_benchmark_iterations; ++i) {
        bench_clock.start();
        if (use_device_statistics) {
            accumulate_average();
            solver.synchronize();
        }
        else {
            enqueue_read_u_rho();
            accumulate_average();
        }
        total_seconds += bench_clock.stop();
        emit_progress_stage("speed_estimate",
                            "Estimating solve speed",
                            "Benchmarking mean-field post-processing iteration " +
                                to_string((ulong)i + 1ull) + "/" + to_string((ulong)k_avg_benchmark_iterations),
                            (long long)i + 1ll,
                            (long long)k_avg_benchmark_iterations,
                            false);
    }
    const double mean_sample_seconds = total_seconds / (double)k_avg_benchmark_iterations;
    const double effective_extra_step_seconds = mean_sample_seconds / (double)avg_stride;
    print_runtime_kv("Avg benchmark",
        "sample extra = " + to_string(mean_sample_seconds, 3u) +
        " s, effective extra = " + to_string(effective_extra_step_seconds, 3u) +
        " s/step at stride " + to_string(avg_stride));
    emit_progress_stage("speed_estimate",
                        "Estimating solve speed",
                        "Mean-field sample extra = " + to_string(mean_sample_seconds, 3u) + " s",
                        1ll,
                        1ll,
                        false);

    reset_average_accumulator();
    return mean_sample_seconds;
}

void UniformRun::configure_eta_model(const double avg_sample_seconds_hint) {
    culbm::app::run_progress().plan_phases(total_steps, avg_start_t, avg_window, ctx.last_normal_steps_per_second_hint, avg_sample_seconds_hint, avg_stride);
    const string normal_sps_text = ctx.last_normal_steps_per_second_hint > 0.0
        ? to_string(ctx.last_normal_steps_per_second_hint, 3u)
        : string("dynamic");
    if (avg_window > 0ull) {
        const string avg_extra_text = avg_sample_seconds_hint > 0.0
            ? to_string(avg_sample_seconds_hint, 3u) + " s/sample"
            : string("n/a");
        print_runtime_kv("ETA model",
            "normal Steps/s = " + normal_sps_text + ", mean-field extra = " + avg_extra_text +
            ", stride = " + to_string(avg_stride));
        emit_progress_stage("speed_estimate",
                            "Estimating solve speed",
                            "ETA model ready: normal Steps/s = " + normal_sps_text + ", mean-field extra = " + avg_extra_text,
                            1ll,
                            1ll,
                            false);
    } else {
        print_runtime_kv("ETA model", "single-stage normal Steps/s = " + normal_sps_text);
        emit_progress_stage("speed_estimate",
                            "Estimating solve speed",
                            "ETA model ready: single-stage normal Steps/s = " + normal_sps_text,
                            1ll,
                            1ll,
                            false);
    }
}

void UniformRun::print_timing_plan(const double avg_sample_seconds_hint) {
    const bool normal_hint_ready = ctx.last_normal_steps_per_second_hint > 0.0;
    const double normal_sps = culbm::app::run_progress().plain_steps_per_second();
    const double normal_step_s = normal_sps > 1.0E-12 ? 1.0 / normal_sps : 0.0;
    const ulong warmup_steps = avg_window > 0ull
        ? (avg_start_t > 0ull ? std::min(avg_start_t - (ulong)1u, total_steps) : 0ull)
        : total_steps;
    const ulong statistical_steps = total_steps > warmup_steps ? total_steps - warmup_steps : 0ull;
    const ulong statistical_samples = count_avg_samples_after(warmup_steps);
    const double warmup_seconds = (double)warmup_steps * normal_step_s;
    const double statistical_seconds =
        (double)statistical_steps * normal_step_s +
        (double)statistical_samples * avg_sample_seconds_hint;
    const double solver_seconds = warmup_seconds + statistical_seconds;
    const double case_seconds = preprocess_seconds + solver_seconds;

    print_runtime_section("LBM TIMING PLAN");
    print_runtime_kv("Preprocess time", print_time(preprocess_seconds) + " (recorded)");
    print_runtime_kv("Warm-up est.",
        format_estimated_time(warmup_seconds, normal_hint_ready) +
        ", steps = " + to_string(warmup_steps));
    if (avg_window > 0ull) {
        print_runtime_kv("Statistical est.",
            format_estimated_time(statistical_seconds, normal_hint_ready) +
            ", steps = " + to_string(statistical_steps) +
            ", samples = " + to_string(statistical_samples));
    }
    else {
        print_runtime_kv("Statistical est.", "off");
    }
    print_runtime_kv("Estimated solver", format_estimated_time(solver_seconds, normal_hint_ready));
    print_runtime_kv("Estimated total",
        format_estimated_time(case_seconds, normal_hint_ready) +
        " (includes recorded preprocess)");
}

void UniformRun::update_runtime_eta(const double iteration_seconds, const bool avg_sampled) {
    const bool avg_phase = is_avg_phase();
    culbm::app::run_progress().record_iteration(avg_phase, iteration_seconds, avg_sampled);
    if (avg_phase && !avg_phase_entry_reported) {
        avg_phase_entry_reported = true;
        print_runtime_kv("Mean-field stage",
            "entered. normal Steps/s = " + to_string(culbm::app::run_progress().plain_steps_per_second(), 3u) +
            ", mean-field Steps/s = " + to_string(culbm::app::run_progress().window_steps_per_second(), 3u));
    }
}

bool UniformRun::step_needs_macros(const ulong next_step) const {
    if (particle_advection_runtime || solver.has_immersed_forcing()) return true;
    if (next_step >= total_steps) return true;
    if (enable_unsteady_u_output && next_step >= deck.culbm_unsteady_start_step &&
        (next_step % unsteady_interval) == 0ull) {
        return true;
    }
    if (do_avg && avg_window > 0ull && next_step >= avg_start_t &&
        ((next_step - avg_start_t) % avg_stride) == 0ull) {
        return true;
    }
    if (!resolved_probes.empty() && next_step >= probe_start_t) return true;
    return false;
}

void UniformRun::run_normal_speed_benchmark(const std::function<void()>& before_step) {
    if (ctx.last_normal_steps_per_second_hint > 0.0) return;
    const ulong warmup_steps = avg_window > 0ull
        ? (avg_start_t > 0ull ? std::min(avg_start_t - (ulong)1u, total_steps) : 0ull)
        : total_steps;
    if (solver.time() >= warmup_steps) return;
    const ulong remaining_warmup_steps = warmup_steps - solver.time();
    const ulong probe_steps = std::min((ulong)16u, remaining_warmup_steps);
    if (probe_steps == 0ull) return;

    print_runtime_kv("Normal benchmark", "run " + to_string(probe_steps) + " warm-up step(s) before timing plan");
    emit_progress_stage("speed_estimate",
                        "Estimating solve speed",
                        "Benchmarking normal LBM solver",
                        0ll,
                        (long long)probe_steps,
                        false);
    Clock probe_clock;
    double probe_seconds = 0.0;
    const bool sync_each_step = (particle_advection_runtime != nullptr);
    for (ulong i = 0ull; i < probe_steps; ++i) {
        probe_clock.start();
        if (before_step) before_step();
        if (pre_step_update) {
            pre_step_update(solver, solver.time());
            refresh_particle_advection_velocity();
        }
        const bool want_macros = (i + 1ull == probe_steps) || step_needs_macros(solver.time() + 1ull);
        solver.set_step_writes_macros(want_macros);
        culbm::app::run_steps(solver, 1u, total_steps, sync_each_step);
        advance_particle_advection();
        sample_surface_ibm_diagnostics();
        maybe_write_unsteady_u();
        process_post_step_samples();
        probe_seconds += probe_clock.stop();
        emit_progress_stage("speed_estimate",
                            "Estimating solve speed",
                            "Benchmarking normal LBM solver step " +
                                to_string(i + (ulong)1u) + "/" + to_string(probe_steps),
                            (long long)(i + (ulong)1u),
                            (long long)probe_steps,
                            false);
    }
    probe_clock.start();
    solver.synchronize();
    probe_seconds += probe_clock.stop();
    if (probe_seconds > 0.0) {
        ctx.last_normal_steps_per_second_hint = (double)probe_steps / probe_seconds;
        print_runtime_kv("Normal benchmark",
            "normal Steps/s = " + to_string(ctx.last_normal_steps_per_second_hint, 3u) +
            ", step = " + to_string(probe_seconds / (double)probe_steps, 3u) +
            " s, measured steps = " + to_string(probe_steps));
    }
}

void UniformRun::refresh_particle_advection_velocity() {
    if (!particle_advection_runtime) return;
    try {
        particle_advection_runtime->refresh_velocity_before_lbm_step();
    }
    catch (const std::exception& exception) {
        println("ERROR: CULBM particle-advection velocity refresh failed: " +
                std::string(exception.what()));
        wait();
        exit(-1);
    }
}

void UniformRun::advance_particle_advection() {
    if (!particle_advection_runtime) return;
    if (solver.time() == 0ull) {
        println("ERROR: internal particle-advection step ordering is invalid.");
        wait();
        exit(-1);
    }
    try {
        const std::uint64_t step_begin =
            (std::uint64_t)solver.time() - 1u;
        particle_advection_runtime->advance_after_lbm_step(step_begin);
        particle_advection_runtime->write_snapshot_if_due(
            (std::uint64_t)solver.time());
    }
    catch (const std::exception& exception) {
        println("ERROR: CULBM particle advection failed: " +
                std::string(exception.what()));
        wait();
        exit(-1);
    }
}

} // namespace culbm::run
