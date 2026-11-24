#include "nested_run.hpp"
#include "console_log.hpp"
#include "physical_fields.hpp"
#include "uniform_run.hpp"
#include "vtk_output.hpp"
#include "../app/progress.hpp"
#include "../app/run_report.hpp"
#include "../case/context.hpp"
#include "../case/deck.hpp"
#include "../case/report.hpp"
#include "../physical_output.hpp"
#include "../surface_ibm.hpp"
#include "../solver/field_view.hpp"

#include <iomanip>
#include <limits>

namespace culbm::run {

using namespace culbm::bc;
using namespace culbm::cases;

NestedRun::NestedRun(CaseContext& ctx, const culbm::NestedSeedState& seed, const std::string& vtk_prefix,
                     const bool extra_spacing, const double uniform_preprocess_seconds,
                     const culbm::NestedCommonFinestTerrain* finest_terrain, const std::string& downstream_bc_runtime)
    : ctx(ctx), deck(ctx.deck), domain(ctx.domain), inflow(ctx.inflow), physics(ctx.physics),
      geometry(ctx.geometry), terrain(ctx.terrain), probes(ctx.probes), seed(seed), vtk_prefix(vtk_prefix),
      extra_spacing(extra_spacing), uniform_preprocess_seconds(uniform_preprocess_seconds),
      finest_terrain(finest_terrain), downstream_bc_runtime(downstream_bc_runtime) {}

double NestedRun::run() {
    if (!domain.cuda_vertical_nesting_plan.enabled) {
        println("ERROR: nested run requested without an enabled vertical-nesting plan.");
        wait(); exit(-1);
    }
    nested_build_clock.start();
    coarse_dx_si = domain.cuda_vertical_nesting_plan.coarse_dx_si;
    // lbm_nu/lbm_alpha are the effective coarse values selected by the
    // common FP32 transport resolver for the current unit system.
    nested_physics.coarse_nu = physics.lbm_nu;
    nested_physics.coarse_alpha = physics.lbm_alpha;
    nested_physics.beta = physics.lbm_beta;
    nested_physics.coarse_buoyancy_gz = physics.lbm_buoyancy_gz;
    nested_physics.temperature_outlet_face = deck.downstream_open_face
        ? buffer_face_id_from_bc(downstream_bc_runtime) : 0;
    nested_physics.coarse_omega_x = coriolis_Omegax_lbmu;
    nested_physics.coarse_omega_y = coriolis_Omegay_lbmu;
    nested_physics.coarse_omega_z = coriolis_Omegaz_lbmu;
    nested_physics.coarse_dt_si = coarse_dx_si *
        ((double)domain.lbm_ref_u / (double)physics.si_ref_u);
    nested_physics.additional_gpu_memory_per_device_mb =
        domain.nested_additional_gpu_memory_per_device_mb;

    std::string nested_error;
    const float compact_base_top_coarse_lbm = domain.cuda_ground_compaction_active
        ? geometry.mesh->pmin.z + domain.cuda_ground_base_height_si * geometry.scale_geom
        : std::numeric_limits<float>::quiet_NaN();
    if (!culbm::build_nested_vertical_lbm(
            domain.cuda_vertical_nesting_plan, seed, *geometry.mesh, deck.Dx, deck.Dy, deck.Dz,
            nested_physics, nested, nested_error, finest_terrain,
            compact_base_top_coarse_lbm)) {
        println("ERROR: CUDA nested-grid construction failed: " + nested_error);
        wait(); exit(-1);
    }
    const double nested_build_seconds = nested_build_clock.stop();
    preprocess_seconds = uniform_preprocess_seconds + nested_build_seconds;

    print_section_title("CUDA VERTICAL NESTING");
    print_kv_row("Nesting order", nested.plan.has_three_levels()
        ? "coarse -> middle -> fine (1x -> 2x -> 4x)"
        : "coarse -> fine (1x -> 2x)");
    print_kv_row("Interface rule",
        "volumetric population-conserving 2:1 bridge (Rohde 2006)");
    print_kv_row("Geometry raster", "common finest " +
        to_string((ulong)nested.geometry.raster_nx) + "x" +
        to_string((ulong)nested.geometry.raster_ny) + "x" +
        to_string((ulong)nested.geometry.raster_nz) +
        ", STL top z=" + to_string(nested.geometry.stl_top_coarse_lbm_z, 4u) + " coarse LBM");
    if (finest_terrain) {
        print_kv_row("Terrain overlay", "common-finest DEM cells added=" +
            to_string(nested.geometry.terrain_solid_cells) +
            ", top z=" + to_string(nested.geometry.terrain_top_common_finest_z, 4u));
    }
    if (nested.geometry.compact_base_solid_cells_removed > 0u) {
        print_kv_row("CUDA ground", "removed " +
            to_string(nested.geometry.compact_base_solid_cells_removed) +
            " generated-base voxels below common-finest z=" +
            to_string(nested.geometry.compact_base_top_common_finest_z, 4u));
    }
    for (std::size_t i = 0u; i < nested.plan.levels.size(); ++i) {
        const auto& level = nested.plan.levels[i];
        print_kv_row("Nested solid", level.label + ": " +
            to_string(nested.geometry.solid_cells_per_plan_level[i]) +
            " direct-majority physical cells");
    }

    if (deck.surface_ibm_settings.enabled) {
        print_kv_row("Nested SurfaceIBM", "native marker spacing; half-open surface ownership; normalized physical-row support");
        print_kv_row("Nested ground", "one compact solid row; halfway bounce-back at physical z=0");
        for (const auto& owned : nested.levels) {
            const auto statistics = culbm::surface_ibm::statistics(*owned.solver);
            print_kv_row("Nested IBM layer", owned.plan.label + ": markers=" +
                to_string(statistics.markers) + ", active cells=" + to_string(statistics.active_cells) +
                ", device-resident=" + string(statistics.device_resident ? "yes" : "no"));
        }
    }

    std::vector<culbm::NestedLbmLevel> scheduler_levels =
        nested.make_scheduler_levels();
    scheduler = std::make_unique<culbm::NestedLbmScheduler>(std::move(scheduler_levels));
    scheduler->set_interface_hook(
        [](culbm::NestedLbmLevel& upper,
           culbm::NestedLbmLevel& lower,
           const culbm::NestedExchangePhase phase) {
            if (culbm::NestedVolumetricBridge::supported(upper, lower))
                culbm::NestedVolumetricBridge::exchange(upper, lower, phase);
            else
                culbm::NestedMacroPlaneBridge::exchange(upper, lower);
        });
    print_kv_row("Nested interface",
        "volumetric population-conserving 2:1 bridge (Rohde 2006)");

    if (deck.vk_inlet_settings.enable) {
        nested_vk_updaters.resize(nested.levels.size());
        uint active_vk_levels = 0u;
        for (std::size_t level_index = 0u; level_index < nested.levels.size(); ++level_index) {
            const culbm::NestedOwnedLbmLevel& owned = nested.levels[level_index];
            const float ratio = static_cast<float>(owned.plan.ratio_to_coarse);
            VkInletRuntimeConfig vk_runtime = make_vk_runtime_config(deck.vk_inlet_settings, downstream_bc_runtime);
            vk_runtime.coordinate_scale = 1.0f / ratio;
            vk_runtime.coordinate_offset = float3(
                0.5f * vk_runtime.coordinate_scale - 0.5f,
                0.5f * vk_runtime.coordinate_scale - 0.5f,
                static_cast<float>(owned.plan.z_begin_agl_si / coarse_dx_si + 1.0) -
                    static_cast<float>(owned.plan.physical_first_z()) * vk_runtime.coordinate_scale);
            // Only the sky block owns the global top boundary. Lower
            // blocks use their z faces exclusively as nesting ghosts.
            vk_runtime.allow_top_face = level_index == 0u;
            vk_runtime.has_top_coordinate_z = vk_runtime.allow_top_face;
            vk_runtime.top_coordinate_z = static_cast<float>(seed.nz - 1u);
            auto updater = std::make_unique<VonKarmanInletUpdater>(vk_runtime);
            if (updater->initialize(*owned.solver)) {
                nested_vk_updaters[level_index] = std::move(updater);
                ++active_vk_levels;
            }
        }
        if (active_vk_levels == 0u) {
            println("| VK inlet        | nested hierarchy: no valid inflow faces.                  |");
        }
        else {
            print_kv_row("VK inlet", "nested dynamic updater active on " +
                to_string((ulong)active_vk_levels) + " level(s), shared coarse-lattice realization");
            scheduler->set_level_before_step_hook(
                [&](culbm::NestedLbmLevel& level,
                    const double time_si,
                    const std::uint64_t) {
                    const double coarse_time = time_si / nested_physics.coarse_dt_si;
                    for (std::size_t i = 0u; i < nested.levels.size(); ++i) {
                        if (nested.levels[i].solver.get() != level.solver ||
                            !nested_vk_updaters[i]) continue;
                        nested_vk_updaters[i]->update_at_lattice_time(
                            *level.solver, coarse_time);
                        break;
                    }
                });
        }
    }

    const RunSchedule schedule = make_run_schedule(deck, !probes.probe_requests.empty());
    extra_run_steps = schedule.extra_run_steps;
    total_steps = schedule.total_steps;
    unsteady_interval = schedule.unsteady_interval;
    do_avg = schedule.do_avg;
    avg_window = schedule.avg_window;
    avg_stride = schedule.avg_stride;
    avg_start_t = schedule.avg_start_t;
    probe_window = schedule.probe_window;
    probe_start_t = schedule.probe_start_t;

    results_vtk_dir = deck.parent + "/RESULTS/vtk/";
    results_netcdf_dir = deck.parent + "/RESULTS/netcdf/";
    vtk_dir = results_vtk_dir + vtk_prefix + deck.datetime + "_raw_";
    std::filesystem::create_directories(std::filesystem::path(vtk_dir).parent_path());
    if (deck.culbm_output_global_netcdf || deck.culbm_output_agl_netcdf) {
        std::filesystem::create_directories(std::filesystem::path(results_netcdf_dir));
    }

    nested_output_height_si =
        (double)(domain.top_sponge_grid_extend ? domain.Nz_cells_core - 1u : domain.lbm_N.z - 1u) *
        coarse_dx_si;
    if (domain.top_sponge_grid_extend) {
        print_kv_row("VTK z output", "nested core height=" +
            to_string(nested_output_height_si, 6u) + " m; top sponge omitted");
    }

    case_solver_clock.start();
    print_section_title("LBM SOLVER INFORMATION");
    print_kv_row("Run steps", to_string(total_steps) +
        " coarse steps (all nested levels reach the same SI time)");
    print_kv_row("Coarse dt", to_string(nested_physics.coarse_dt_si, 9u) + " s");
    if (avg_window > 0u) {
        print_kv_row("Avg stride", "sample every " + to_string(avg_stride) +
            " coarse step(s) in purge_avg window");
    }
    if (unsteady_interval > 0u) {
        print_kv_row("Unsteady u VTK", "every " + to_string(unsteady_interval) + " coarse step(s)");
        print_kv_row("Unsteady start", to_string(deck.culbm_unsteady_start_step) + " coarse step(s)");
    }
    if (deck.culbm_sgs_diagnostics_interval > 0u && culbm::mpi_is_root()) {
        const std::string diagnostics_filename =
            vtk_prefix + deck.datetime + "_culbm_sgs_diagnostics.csv";
        const std::filesystem::path diagnostics_path =
            std::filesystem::path(deck.parent) / "RESULTS" /
            diagnostics_filename;
        std::filesystem::create_directories(diagnostics_path.parent_path());
        nested_sgs_diagnostics_output.open(
            diagnostics_path.string(), std::ios::out | std::ios::trunc);
        if (!nested_sgs_diagnostics_output.is_open()) {
            println("| ERROR: cannot create SGS diagnostics file: " +
                diagnostics_path.string() + " |");
            wait();
            exit(-1);
        }
        print_kv_row("SGS diagnostics", diagnostics_path.string());
        nested_sgs_diagnostics_output
            << "coarse_step,time_s,level,dx_m,dt_s,"
               "mean_nu_t_lbm,max_nu_t_lbm,"
               "mean_nu_t_m2_s,max_nu_t_m2_s,"
               "mean_mu_t_pa_s,max_mu_t_pa_s,count\n";
    }
    scheduler->initialize();
    enable_nested_device_statistics();
    if (!probes.probe_requests.empty()) {
        const culbm::NestedStitchedFields initial_fields = stitch_or_fail();
        resolve_nested_probes(initial_fields);
        configure_nested_device_probes(initial_fields);
    }
    luw_emit_progress("solve", "Solving nested CUDA CFD",
                      "nested hierarchy initialized", 0ll,
                      (long long)total_steps, false);
    auto last_progress = std::chrono::steady_clock::time_point{};
    culbm::app::begin_nested_progress(scheduler.get(), scheduler->levels(), scheduler->coarse_step_count(), total_steps);
    for (ulong step = 0u; step < total_steps; ++step) {
        scheduler->advance_one_coarse_step();
        const ulong t = scheduler->coarse_step_count();
        culbm::app::run_progress().advance(t);
        sample_nested_sgs_diagnostics(t);
        const bool sample_avg = avg_window > 0u && t >= avg_start_t &&
            ((t - avg_start_t) % avg_stride) == 0u;
        const bool sample_probes = !nested_probes.empty() && t >= probe_start_t;
        const bool write_unsteady = unsteady_interval > 0u && t >= deck.culbm_unsteady_start_step &&
            (t % unsteady_interval) == 0u;
        if (sample_avg && nested_device_stats_active) {
            ++avg_count;
            for (auto& owned : nested.levels) owned.solver->sample_statistics(avg_count);
        }
        if (sample_probes && nested_device_probes_active) {
            capture_nested_probe_sample_device(t);
        }
        const bool need_stitch = write_unsteady ||
            (sample_avg && !nested_device_stats_active) ||
            (sample_probes && !nested_device_probes_active);
        if (need_stitch) {
            const culbm::NestedStitchedFields fields = stitch_or_fail();
            if (sample_avg && !nested_device_stats_active) accumulate_average(fields);
            if (sample_probes && !nested_device_probes_active) capture_nested_probe_sample(fields, t);
            if (write_unsteady) write_raw_fields(fields, t, true);
        }
        const auto now = std::chrono::steady_clock::now();
        if (last_progress.time_since_epoch().count() == 0 ||
            now - last_progress >= std::chrono::milliseconds(250) || t == total_steps) {
            last_progress = now;
            const double elapsed = case_solver_clock.stop();
            const double steps_per_second = elapsed > 0.0 ? (double)t / elapsed : 0.0;
            const double eta = steps_per_second > 0.0
                ? (double)(total_steps - t) / steps_per_second : 0.0;
            luw_emit_progress("solve", "Solving nested CUDA CFD",
                              to_string(t) + "/" + to_string(total_steps) +
                              " coarse steps | " + to_string(steps_per_second, 3u) +
                              " coarse Steps/s | ETA " + print_time(eta),
                              (long long)t, (long long)total_steps, false);
        }
    }
    culbm::app::run_progress().end(scheduler.get());
    const culbm::NestedStitchedFields final_fields = stitch_or_fail();
    if (avg_count > 0u && nested_device_stats_active) {
        finalize_nested_device_statistics(final_fields);
    }
    write_raw_fields(final_fields, scheduler->coarse_step_count(), false);
    write_final_physical_fields(final_fields, scheduler->coarse_step_count());
    finalize_nested_probes();
    if (avg_count > 0u) {
        const std::string avg_name = vtk_prefix + deck.datetime + "_avg";
        const std::string avg_file = default_filename(
            results_vtk_dir, avg_name, ".vtk", scheduler->coarse_step_count());
        const float* avg_temperature = avg_T.empty() ? nullptr : avg_T.data();
        const float3 origin = output_origin(final_fields);
        write_avg_vtk(avg_file, final_fields.nx, final_fields.ny, final_fields.nz,
                      avg_u.data(), avg_rho.data(), avg_temperature, avg_flags.data(),
                      M2_u.data(), M2_v.data(), M2_w.data(), avg_count,
                      true, avg_turbulence_output(deck), false, 0u, final_fields.dx_si, &origin);
        print_kv_row("Avg samples", to_string(avg_count));
    }
    if (extra_run_steps > 0u && culbm::mpi_is_root()) {
        std::ofstream info_file(deck.parent + "/proj_temp/transform.info");
        if (info_file.is_open()) {
            info_file << "dt = " << std::fixed << std::setprecision(10)
                      << nested_physics.coarse_dt_si << "s\n";
        }
    }
    scheduler->finish();
    for (auto& owned : nested.levels) owned.solver->barrier();
    const double solver_seconds = case_solver_clock.stop();
    const double total_seconds = preprocess_seconds + solver_seconds;
    ctx.last_normal_steps_per_second_hint = solver_seconds > 0.0
        ? (double)total_steps / solver_seconds : 0.0;
    print_kv_row("Task finished", "[" + now_str() + "]");
    print_kv_row("Preprocess time", print_time(preprocess_seconds) + " (uniform setup + nested build)");
    print_kv_row("Solver time", print_time(solver_seconds));
    print_kv_row("Nested total", print_time(total_seconds));
    if (extra_spacing) println("|"+string((uint)CONSOLE_WIDTH-2u, ' ')+"|");
    return total_seconds;
}

void NestedRun::crop_nested_top_sponge(culbm::NestedStitchedFields& fields) {
    if (!domain.top_sponge_grid_extend) return;
    const double output_nz_real = nested_output_height_si / fields.dx_si;
    if (!culbm::vertical_nesting_nearly_integer(output_nz_real)) {
        println("ERROR: nested top-sponge output height is not aligned to finest SI cells.");
        wait(); exit(-1);
    }
    const uint output_nz = static_cast<uint>(std::llround(output_nz_real));
    if (output_nz >= fields.nz) return;
    const ulong output_cells = (ulong)fields.nx * (ulong)fields.ny * (ulong)output_nz;
    fields.rho.resize((size_t)output_cells);
    fields.u.resize((size_t)(3ull * output_cells));
    fields.flags.resize((size_t)output_cells);
    if (!fields.temperature.empty()) fields.temperature.resize((size_t)output_cells);
    fields.nz = output_nz;
}

float3 NestedRun::output_origin(const culbm::NestedStitchedFields& fields) {
    // The stitched field omits exactly one compact ground row. The
    // z origin must therefore be shifted by one finest cell, not by
    // the refinement ratio of the old coarse ground row.
    float3 origin = float3(
        fields.dx_si * (0.5f - 0.5f * (float)fields.nx),
        fields.dx_si * (0.5f - 0.5f * (float)fields.ny),
        0.0f) + vtk_origin_shift;
    // The nested vertical map anchors the ground face at +0.5
    // on both lattices. The first exported air node is half a
    // finest cell above the compacted STL datum.
    origin.z = geometry.stl_min_si.z + domain.cuda_ground_trim_si + 0.5f * fields.dx_si;
    return origin;
}

culbm::NestedStitchedFields NestedRun::stitch_or_fail() {
    culbm::NestedStitchedFields fields;
    std::string stitch_error;
    if (!culbm::stitch_nested_physical_fields(nested, fields, stitch_error)) {
        println("ERROR: CUDA nested-grid output stitching failed: " + stitch_error);
        wait(); exit(-1);
    }
    crop_nested_top_sponge(fields);
    return fields;
}

void NestedRun::write_raw_fields(const culbm::NestedStitchedFields& fields, const ulong step, const bool u_only) {
    const float3 origin = output_origin(fields);
    write_nested_raw_vtk(default_filename(vtk_dir, "u", ".vtk", step),
                         fields.nx, fields.ny, fields.nz, fields.u.data(), 3u,
                         fields.dx_si, origin, units.si_u(1.0f));
    if (u_only) return;
    write_nested_raw_vtk(default_filename(vtk_dir, "rho", ".vtk", step),
                         fields.nx, fields.ny, fields.nz, fields.rho.data(), 1u,
                         fields.dx_si, origin, units.si_rho(1.0f));
    if (inflow.use_temperature_bc && !fields.temperature.empty()) {
        write_nested_raw_vtk(default_filename(vtk_dir, "T", ".vtk", step),
                             fields.nx, fields.ny, fields.nz, fields.temperature.data(), 1u,
                             fields.dx_si, origin, units.si_dT(1.0f), units.si_T(0.0f));
    }
}

void NestedRun::write_final_physical_fields(const culbm::NestedStitchedFields& fields, const ulong step) {
    if (!deck.culbm_output_physical_requested || !culbm::mpi_is_root()) return;
    float3 global_origin = output_origin(fields);
    global_origin.z -= terrain.physical_output_z_zero_si;
    const float z_tolerance = fmaxf(1.0e-6f, fields.dx_si * 1.0e-7f);
    uint z_begin = 0u;
    while (z_begin < fields.nz && global_origin.z + (float)z_begin * fields.dx_si < -z_tolerance) {
        ++z_begin;
    }
    if (z_begin >= fields.nz) {
        println("| ERROR: physical output has no non-basement nested grid plane.               |");
        return;
    }
    culbm::PhysicalGridView view;
    view.nx = fields.nx;
    view.ny = fields.ny;
    view.nz = fields.nz;
    view.z_begin = z_begin;
    view.origin_x = global_origin.x;
    view.origin_y = global_origin.y;
    view.origin_z = global_origin.z;
    view.spacing_x = fields.dx_si;
    view.spacing_y = fields.dx_si;
    view.spacing_z = fields.dx_si;
    view.velocity_xyz = fields.u.data();
    view.density = fields.rho.data();
    view.flags = fields.flags.data();
    view.velocity_scale = units.si_u(1.0f);
    view.density_scale = units.si_rho(1.0f);
    if (inflow.use_temperature_bc && !fields.temperature.empty()) {
        view.temperature = fields.temperature.data();
        view.temperature_scale = units.si_dT(1.0f);
        view.temperature_offset = units.si_T(0.0f);
    }
    std::vector<float> ground;
    culbm::PhysicalAglSurfaceView surface;
    if (deck.culbm_output_agl_requested) {
        ground = make_physical_output_ground_grid(ctx, 
            fields.nx, fields.ny, global_origin, fields.dx_si);
        surface.ground_z = ground.data();
        surface.ground_count = (std::uint64_t)ground.size();
    }
    write_physical_outputs(deck, view, surface, results_vtk_dir, results_netcdf_dir, vtk_prefix + deck.datetime, step);
}

void NestedRun::resolve_nested_probes(const culbm::NestedStitchedFields& fields) {
    if (probes.probe_requests.empty()) return;
    if (!probes.probe_geo_mapping.valid) {
        print_kv_row("Probes", "disabled: geographic mapping is unavailable");
        return;
    }
    const auto is_inside_domain = [&](const double x_si, const double y_si) {
        return std::isfinite(x_si) && std::isfinite(y_si) &&
            x_si >= 0.0 && x_si <= (double)deck.si_size.x &&
            y_si >= 0.0 && y_si <= (double)deck.si_size.y;
    };
    const auto resolve_probe = [&](const ProbeRequest& request,
                                   ResolvedProbe& probe,
                                   std::string& warning) -> bool {
        const double base_lon = request.uses_center
            ? probes.probe_geo_mapping.center_lon_deg : request.lon_deg;
        const double base_lat = request.uses_center
            ? probes.probe_geo_mapping.center_lat_deg : request.lat_deg;
        double base_x_si = 0.0, base_y_si = 0.0;
        if (!project_probe_local_xy(base_lon, base_lat, probes.probe_geo_mapping,
                                    base_x_si, base_y_si)) {
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
            const uint base_x = snap_probe_index(base_x_si, fields.nx, fields.dx_si);
            const uint base_y = snap_probe_index(base_y_si, fields.ny, fields.dx_si);
            final_x_si = (double)base_x * (double)fields.dx_si +
                (double)request.offset.east_cells * (double)fields.dx_si * probes.probe_geo_mapping.east_dx +
                (double)request.offset.north_cells * (double)fields.dx_si * probes.probe_geo_mapping.north_dx;
            final_y_si = (double)base_y * (double)fields.dx_si +
                (double)request.offset.east_cells * (double)fields.dx_si * probes.probe_geo_mapping.east_dy +
                (double)request.offset.north_cells * (double)fields.dx_si * probes.probe_geo_mapping.north_dy;
        }
        else if (request.offset.mode == ProbeOffsetMode::METERS) {
            final_x_si = base_x_si + request.offset.east_m * probes.probe_geo_mapping.east_dx +
                request.offset.north_m * probes.probe_geo_mapping.north_dx;
            final_y_si = base_y_si + request.offset.east_m * probes.probe_geo_mapping.east_dy +
                request.offset.north_m * probes.probe_geo_mapping.north_dy;
        }
        if (!is_inside_domain(final_x_si, final_y_si)) {
            warning = "offset point is outside CFD domain";
            return false;
        }

        probe = ResolvedProbe{};
        probe.request = request;
        probe.x = snap_probe_index(final_x_si, fields.nx, fields.dx_si);
        probe.y = snap_probe_index(final_y_si, fields.ny, fields.dx_si);
        probe.snapped_x_si = (double)probe.x * (double)fields.dx_si;
        probe.snapped_y_si = (double)probe.y * (double)fields.dx_si;
        probe.label = request.raw_token;
        for (uint z = 0u; z < fields.nz; ++z) {
            const ulong n = (ulong)probe.x +
                ((ulong)probe.y + (ulong)z * (ulong)fields.ny) * (ulong)fields.nx;
            if ((fields.flags[(size_t)n] & CELL_SOLID) == 0u) probe.z_indices.push_back(z);
        }
        if (probe.z_indices.empty()) {
            warning = "resolved column has no fluid cell";
            return false;
        }
        const uint z0 = probe.z_indices.front();
        probe.heights_si.reserve(probe.z_indices.size());
        for (const uint z : probe.z_indices) {
            probe.heights_si.push_back((float)(((double)z - (double)z0 + 0.5) *
                (double)fields.dx_si));
        }
        probe.times_si.reserve((size_t)probe_window);
        probe.velocity_series_si.reserve(
            (size_t)probe_window * probe.z_indices.size() * 3ull);
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
        nested_probes.push_back(std::move(probe));
    }
    if (nested_probes.empty()) {
        print_kv_row("Probes", "0 valid probe column after nested geometry checks");
    }
    else {
        const std::string window_desc = probe_window >= total_steps
            ? "entire run" : ("last " + to_string(probe_window) + " step(s)");
        print_kv_row("Probes", to_string((ulong)nested_probes.size()) +
            " active, " + window_desc + " (common-finest output grid)");
    }
}

void NestedRun::configure_nested_device_probes(const culbm::NestedStitchedFields& fields) {
    nested_device_probes_active = false;
    probe_level_bindings.clear();
    level_probe_scratch.clear();
    if (nested_probes.empty() || nested.levels.empty()) return;
    if (nested.levels.front().solver->distributed()) return;

    const uint32_t finest_ratio = nested.geometry.common_finest_ratio;
    std::vector<uint32_t> level_q(nested.levels.size(), 1u);
    std::vector<uint64_t> level_z_begin(nested.levels.size(), 0u);
    std::vector<uint64_t> level_z_end(nested.levels.size(), 0u);
    for (std::size_t li = 0u; li < nested.levels.size(); ++li) {
        const auto& plan = nested.levels[li].plan;
        level_q[li] = finest_ratio / plan.ratio_to_coarse;
        const double begin_real = plan.z_begin_agl_si / (double)fields.dx_si;
        if (!culbm::vertical_nesting_nearly_integer(begin_real)) return;
        level_z_begin[li] = (uint64_t)std::llround(begin_real);
        level_z_end[li] = level_z_begin[li] + (uint64_t)plan.physical_nz * (uint64_t)level_q[li];
    }

    std::vector<std::vector<ulong>> level_cells(nested.levels.size());
    probe_level_bindings.resize(nested_probes.size());
    for (std::size_t pi = 0u; pi < nested_probes.size(); ++pi) {
        const ResolvedProbe& probe = nested_probes[pi];
        probe_level_bindings[pi].reserve(probe.z_indices.size());
        for (const uint z : probe.z_indices) {
            std::size_t owner = nested.levels.size();
            for (std::size_t li = 0u; li < nested.levels.size(); ++li) {
                if ((uint64_t)z >= level_z_begin[li] && (uint64_t)z < level_z_end[li]) {
                    owner = li;
                    break;
                }
            }
            if (owner >= nested.levels.size()) return;
            const auto& owned = nested.levels[owner];
            const uint32_t q = level_q[owner];
            const uint sx = probe.x / q;
            const uint sy = probe.y / q;
            const uint sz = owned.plan.physical_first_z() + (uint)(((uint64_t)z - level_z_begin[owner]) / (uint64_t)q);
            const ulong cell = owned.solver->grid().index(sx, sy, sz);
            const ulong slot = (ulong)level_cells[owner].size();
            level_cells[owner].push_back(cell);
            probe_level_bindings[pi].push_back(LevelProbeBinding{ owner, slot });
        }
    }

    bool ok = true;
    for (std::size_t li = 0u; li < nested.levels.size(); ++li) {
        if (level_cells[li].empty()) continue;
        if (!nested.levels[li].solver->configure_probes(level_cells[li], 0u)) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        for (auto& owned : nested.levels) owned.solver->disable_probes();
        probe_level_bindings.clear();
        return;
    }
    level_probe_scratch.resize(nested.levels.size());
    nested_device_probes_active = true;
    print_kv_row("Nested probes", "per-level device gather (no full-grid stitch per step)");
}

void NestedRun::capture_nested_probe_sample_device(const ulong coarse_step) {
    for (std::size_t li = 0u; li < nested.levels.size(); ++li) {
        if (nested.levels[li].solver->probes_active()) {
            nested.levels[li].solver->sample_probes(level_probe_scratch[li]);
        }
    }
    const double time_si = (double)coarse_step * nested_physics.coarse_dt_si;
    for (std::size_t pi = 0u; pi < nested_probes.size(); ++pi) {
        ResolvedProbe& probe = nested_probes[pi];
        probe.times_si.push_back(time_si);
        for (const LevelProbeBinding& b : probe_level_bindings[pi]) {
            const std::size_t i3 = (std::size_t)(3ull * b.slot);
            const auto& buf = level_probe_scratch[b.level_index];
            probe.velocity_series_si.push_back(units.si_u(buf[i3]));
            probe.velocity_series_si.push_back(units.si_u(buf[i3 + 1u]));
            probe.velocity_series_si.push_back(units.si_u(buf[i3 + 2u]));
        }
    }
}

void NestedRun::enable_nested_device_statistics() {
    nested_device_stats_active = false;
    if (avg_window == 0u || nested.levels.empty()) return;
    bool all_ok = true;
    for (auto& owned : nested.levels) {
        if (!owned.solver->enable_statistics(inflow.use_temperature_bc, 0u)) {
            all_ok = false;
            break;
        }
    }
    if (!all_ok) {
        for (auto& owned : nested.levels) owned.solver->disable_statistics();
        print_kv_row("Nested mean-field", "host stitching fallback (insufficient VRAM for device stats)");
        return;
    }
    for (auto& owned : nested.levels) owned.solver->reset_statistics();
    nested_device_stats_active = true;
    print_kv_row("Nested mean-field", "per-level device Welford (stitch once at output)");
}

void NestedRun::finalize_nested_device_statistics(const culbm::NestedStitchedFields& final_fields) {
    const ulong cells = final_fields.cell_count();
    avg_u.assign((size_t)(3ull * cells), 0.0f);
    avg_rho.assign((size_t)cells, 1.0f);
    M2_u.assign((size_t)cells, 0.0f);
    M2_v.assign((size_t)cells, 0.0f);
    M2_w.assign((size_t)cells, 0.0f);
    avg_flags = final_fields.flags;
    if (inflow.use_temperature_bc) avg_T.assign((size_t)cells, 1.0f);

    const uint32_t finest_ratio = nested.geometry.common_finest_ratio;
    const uint out_nx = final_fields.nx;
    const uint out_ny = final_fields.ny;
    const uint out_nz = final_fields.nz;
    std::vector<float> lu, lrho, lt, lm2u, lm2v, lm2w;
    for (auto& owned : nested.levels) {
        culbm::solver::Solver& solver = *owned.solver;
        const auto& level = owned.plan;
        const uint32_t q = finest_ratio / level.ratio_to_coarse;
        const uint64_t begin_fine = (uint64_t)std::llround(level.z_begin_agl_si / (double)final_fields.dx_si);
        solver.read_statistics(lu, lrho, lt, lm2u, lm2v, lm2w);
        solver.disable_statistics();
        const uint32_t src_nx = solver.grid().size(0u);
        const uint32_t src_ny = solver.grid().size(1u);
        const bool has_t = !avg_T.empty() && !lt.empty();
        parallel_for((ulong)level.physical_nz, [&](const ulong z_idx) {
            const uint32_t z = (uint32_t)z_idx;
            const uint64_t oz0 = begin_fine + (uint64_t)z * (uint64_t)q;
            if (oz0 >= (uint64_t)out_nz) return;
            const uint32_t source_z = level.physical_first_z() + z;
            for (uint32_t y = 0u; y < level.ny; ++y) {
                for (uint32_t x = 0u; x < level.nx; ++x) {
                    const uint64_t src = (uint64_t)x + ((uint64_t)y + (uint64_t)source_z * (uint64_t)src_ny) * (uint64_t)src_nx;
                    const float ux = lu[(size_t)(3ull * src)];
                    const float uy = lu[(size_t)(3ull * src + 1ull)];
                    const float uz = lu[(size_t)(3ull * src + 2ull)];
                    const float r = lrho[(size_t)src];
                    const float mu = lm2u[(size_t)src];
                    const float mv = lm2v[(size_t)src];
                    const float mw = lm2w[(size_t)src];
                    const float tv = has_t ? lt[(size_t)src] : 1.0f;
                    for (uint32_t qz = 0u; qz < q; ++qz) {
                        const uint64_t oz = oz0 + qz;
                        if (oz >= (uint64_t)out_nz) break;
                        for (uint32_t qy = 0u; qy < q; ++qy) {
                            const uint32_t oy = y * q + qy;
                            const uint64_t row = (uint64_t)(x * q) + ((uint64_t)oy + oz * (uint64_t)out_ny) * (uint64_t)out_nx;
                            for (uint32_t qx = 0u; qx < q; ++qx) {
                                const uint64_t dst = row + qx;
                                avg_u[(size_t)(3ull * dst)] = ux;
                                avg_u[(size_t)(3ull * dst + 1ull)] = uy;
                                avg_u[(size_t)(3ull * dst + 2ull)] = uz;
                                avg_rho[(size_t)dst] = r;
                                M2_u[(size_t)dst] = mu;
                                M2_v[(size_t)dst] = mv;
                                M2_w[(size_t)dst] = mw;
                                if (has_t) avg_T[(size_t)dst] = tv;
                            }
                        }
                    }
                }
            }
        });
    }
}

void NestedRun::capture_nested_probe_sample(const culbm::NestedStitchedFields& fields, const ulong coarse_step) {
    const double time_si = (double)coarse_step * nested_physics.coarse_dt_si;
    for (ResolvedProbe& probe : nested_probes) {
        probe.times_si.push_back(time_si);
        for (const uint z : probe.z_indices) {
            const ulong n = (ulong)probe.x +
                ((ulong)probe.y + (ulong)z * (ulong)fields.ny) * (ulong)fields.nx;
            const ulong n3 = 3ull * n;
            probe.velocity_series_si.push_back(units.si_u(fields.u[(size_t)n3]));
            probe.velocity_series_si.push_back(units.si_u(fields.u[(size_t)(n3 + 1ull)]));
            probe.velocity_series_si.push_back(units.si_u(fields.u[(size_t)(n3 + 2ull)]));
        }
    }
}

void NestedRun::finalize_nested_probes() {
    if (!culbm::mpi_is_root() || nested_probes.empty()) return;
    const std::filesystem::path results_dir = std::filesystem::path(deck.parent) / "RESULTS";
    std::filesystem::create_directories(results_dir);
    ulong written_files = 0ull;
    for (const ResolvedProbe& probe : nested_probes) {
        const std::filesystem::path out_path = results_dir / (probe.file_stem + ".csv");
        std::ofstream fout(out_path.string(), std::ios::out | std::ios::trunc);
        if (!fout.is_open()) {
            print_kv_row("Probe output", "failed to open " + out_path.string());
            continue;
        }
        fout << "height (m)";
        for (const double time_si : probe.times_si) {
            fout << "," << format_decimal_trimmed(time_si, 6);
        }
        fout << "\n";
        const size_t level_count = probe.z_indices.size();
        const size_t time_count = probe.times_si.size();
        for (size_t level = 0u; level < level_count; ++level) {
            fout << format_decimal_trimmed((double)probe.heights_si[level], 6);
            for (size_t ti = 0u; ti < time_count; ++ti) {
                const size_t base = (ti * level_count + level) * 3u;
                fout << ","
                     << format_decimal_trimmed((double)probe.velocity_series_si[base], 6) << ":"
                     << format_decimal_trimmed((double)probe.velocity_series_si[base + 1u], 6) << ":"
                     << format_decimal_trimmed((double)probe.velocity_series_si[base + 2u], 6);
            }
            fout << "\n";
        }
        ++written_files;
    }
    print_kv_row("Probe files", to_string(written_files) + " CSV saved to RESULTS");
}

void NestedRun::accumulate_average(const culbm::NestedStitchedFields& fields) {
    const ulong cells = fields.cell_count();
    if (avg_u.empty()) {
        avg_u.assign((size_t)(3u * cells), 0.0f);
        avg_rho.assign((size_t)cells, 0.0f);
        M2_u.assign((size_t)cells, 0.0f);
        M2_v.assign((size_t)cells, 0.0f);
        M2_w.assign((size_t)cells, 0.0f);
        avg_flags = fields.flags;
        if (inflow.use_temperature_bc) avg_T.assign((size_t)cells, 0.0f);
    }
    ++avg_count;
    const float inv_n = 1.0f / (float)avg_count;
    parallel_for(cells, [&](ulong n) {
        const ulong n3 = 3u * n;
        const float du = fields.u[(size_t)n3] - avg_u[(size_t)n3];
        avg_u[(size_t)n3] += du * inv_n;
        M2_u[(size_t)n] += du * (fields.u[(size_t)n3] - avg_u[(size_t)n3]);
        const float dv = fields.u[(size_t)(n3 + 1u)] - avg_u[(size_t)(n3 + 1u)];
        avg_u[(size_t)(n3 + 1u)] += dv * inv_n;
        M2_v[(size_t)n] += dv * (fields.u[(size_t)(n3 + 1u)] - avg_u[(size_t)(n3 + 1u)]);
        const float dw = fields.u[(size_t)(n3 + 2u)] - avg_u[(size_t)(n3 + 2u)];
        avg_u[(size_t)(n3 + 2u)] += dw * inv_n;
        M2_w[(size_t)n] += dw * (fields.u[(size_t)(n3 + 2u)] - avg_u[(size_t)(n3 + 2u)]);
        avg_rho[(size_t)n] += (fields.rho[(size_t)n] - avg_rho[(size_t)n]) * inv_n;
        if (!avg_T.empty()) {
            avg_T[(size_t)n] += (fields.temperature[(size_t)n] - avg_T[(size_t)n]) * inv_n;
        }
    });
}

void NestedRun::sample_nested_sgs_diagnostics(const ulong coarse_step) {
    if (deck.culbm_sgs_diagnostics_interval == 0u ||
        coarse_step % (ulong)deck.culbm_sgs_diagnostics_interval != 0ull) {
        return;
    }
    double nu_volume_sum = 0.0;
    double mu_volume_sum = 0.0;
    double physical_volume_sum = 0.0;
    double max_nu_si = 0.0;
    double max_mu_si = 0.0;
    std::uint64_t total_count = 0ull;
    bool has_valid_level = false;
    for (const auto& owned : nested.levels) {
        const Subgrid_Statistics statistics =
            owned.solver->subgrid_statistics();
        if (!statistics.valid || statistics.count == 0ull) continue;
        const double ratio =
            (double)owned.plan.ratio_to_coarse;
        const double level_dx_si = owned.plan.dx_si;
        const double level_dt_si =
            nested_physics.coarse_dt_si / ratio;
        const double nu_factor =
            level_dx_si * level_dx_si / level_dt_si;
        const double rho_factor = (double)units.si_rho(1.0f);
        const double mean_nu_si =
            (double)statistics.mean_nu_t * nu_factor;
        const double level_max_nu_si =
            (double)statistics.max_nu_t * nu_factor;
        const double mean_mu_si =
            (double)statistics.mean_rho_nu_t *
            rho_factor * nu_factor;
        const double level_max_mu_si =
            (double)statistics.max_rho_nu_t *
            rho_factor * nu_factor;
        const double cell_volume =
            level_dx_si * level_dx_si * level_dx_si;
        const double level_volume =
            (double)statistics.count * cell_volume;
        nu_volume_sum += mean_nu_si * level_volume;
        mu_volume_sum += mean_mu_si * level_volume;
        physical_volume_sum += level_volume;
        max_nu_si = std::max(max_nu_si, level_max_nu_si);
        max_mu_si = std::max(max_mu_si, level_max_mu_si);
        total_count += (std::uint64_t)statistics.count;
        has_valid_level = true;
        if (culbm::mpi_is_root()) {
            nested_sgs_diagnostics_output << std::setprecision(12)
                << coarse_step << ','
                << (double)coarse_step * nested_physics.coarse_dt_si
                << ',' << owned.plan.label << ','
                << level_dx_si << ',' << level_dt_si << ','
                << statistics.mean_nu_t << ','
                << statistics.max_nu_t << ','
                << mean_nu_si << ',' << level_max_nu_si << ','
                << mean_mu_si << ',' << level_max_mu_si << ','
                << statistics.count << '\n';
        }
    }
    if (!has_valid_level || !(physical_volume_sum > 0.0)) {
        if (!nested_sgs_diagnostics_warned && culbm::mpi_is_root()) {
            print_kv_row(
                "SGS diagnostics",
                "nested hierarchy has no valid fluid-cell statistics");
            nested_sgs_diagnostics_warned = true;
        }
        return;
    }
    const double mean_nu_si = nu_volume_sum / physical_volume_sum;
    const double mean_mu_si = mu_volume_sum / physical_volume_sum;
    if (culbm::mpi_is_root()) {
        nested_sgs_diagnostics_output << std::setprecision(12)
            << coarse_step << ','
            << (double)coarse_step * nested_physics.coarse_dt_si
            << ",volume_weighted_all_levels,,,,,"
            << mean_nu_si << ',' << max_nu_si << ','
            << mean_mu_si << ',' << max_mu_si << ','
            << total_count << '\n';
        nested_sgs_diagnostics_output.flush();
        print_kv_row(
            "SGS nu_t nested",
            "volume-weighted mean=" + to_string(mean_nu_si, 8u) +
            ", max=" + to_string(max_nu_si, 8u) + " m^2/s");
    }
}

} // namespace culbm::run
