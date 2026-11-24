// Per-step sampling and the outputs of UniformRun: diagnostics, host reads,
// mean-field accumulation, probes, unsteady and final fields, the mean-field
// file, the probe files, the physical outputs and transform.info.

#include "uniform_run.hpp"
#include "console_log.hpp"
#include "physical_fields.hpp"
#include "../case/context.hpp"
#include "../case/deck.hpp"
#include "../case/report.hpp"
#include "../physical_output.hpp"
#include "../surface_ibm.hpp"
#include "../solver/field_view.hpp"

#include <iomanip>

namespace culbm::run {

using namespace culbm::cases;

void UniformRun::sample_sgs_diagnostics() {
    const ulong step = solver.time();
    if (deck.culbm_sgs_diagnostics_interval == 0u ||
        step % (ulong)deck.culbm_sgs_diagnostics_interval != 0ull) {
        return;
    }
    const Subgrid_Statistics statistics = solver.subgrid_statistics();
    if (!statistics.valid) {
        if (!sgs_diagnostics_warned && culbm::mpi_is_root()) {
            print_kv_row("SGS diagnostics", "no valid fluid-cell statistics");
            sgs_diagnostics_warned = true;
        }
        return;
    }
    const double nu_factor = (double)units.si_nu(1.0f);
    const double mu_factor =
        (double)units.si_rho(1.0f) * nu_factor;
    const double mean_nu_si =
        (double)statistics.mean_nu_t * nu_factor;
    const double max_nu_si =
        (double)statistics.max_nu_t * nu_factor;
    const double mean_mu_si =
        (double)statistics.mean_rho_nu_t * mu_factor;
    const double max_mu_si =
        (double)statistics.max_rho_nu_t * mu_factor;
    if (culbm::mpi_is_root()) {
        sgs_diagnostics_output << std::setprecision(12)
            << step << ',' << (double)units.si_t(step) << ','
            << statistics.mean_nu_t << ','
            << statistics.max_nu_t << ','
            << mean_nu_si << ',' << max_nu_si << ','
            << mean_mu_si << ',' << max_mu_si << ','
            << statistics.count << '\n';
        sgs_diagnostics_output.flush();
        print_kv_row(
            "SGS nu_t",
            "mean=" + to_string(mean_nu_si, 8u) +
            ", max=" + to_string(max_nu_si, 8u) + " m^2/s");
    }
}

void UniformRun::sample_surface_ibm_diagnostics() {
    const ulong step = solver.time();
    if (!deck.surface_ibm_settings.enabled ||
        deck.surface_ibm_settings.diagnostics_interval == 0u ||
        step == 0ull ||
        step % (ulong)deck.surface_ibm_settings.diagnostics_interval != 0ull) {
        return;
    }
    const culbm::surface_ibm::Statistics statistics =
        culbm::surface_ibm::statistics(solver);
    if (!statistics.valid) {
        if (!surface_ibm_diagnostics_warned && culbm::mpi_is_root()) {
            print_runtime_kv(
                "IBM diagnostics",
                "runtime did not provide valid marker-slip statistics");
            surface_ibm_diagnostics_warned = true;
        }
        return;
    }
    if (!culbm::mpi_is_root()) return;
    surface_ibm_diagnostics_output << std::setprecision(12)
        << step << ',' << (double)units.si_t(step) << ','
        << statistics.markers << ',' << statistics.fragments << ','
        << statistics.active_cells << ','
        << (statistics.device_resident ? 1 : 0) << ','
        << statistics.cleared_solid_cells << ','
        << statistics.iterations << ',' << (statistics.converged ? 1 : 0) << ','
        << statistics.nonconverged_steps << ','
        << statistics.max_slip_lu << ',' << statistics.peak_max_slip_lu << ','
        << statistics.rms_slip_lu << ','
        << statistics.reaction_force_lu.x << ','
        << statistics.reaction_force_lu.y << ','
        << statistics.reaction_force_lu.z << ','
        << statistics.reaction_torque_lu.x << ','
        << statistics.reaction_torque_lu.y << ','
        << statistics.reaction_torque_lu.z << '\n';
    surface_ibm_diagnostics_output.flush();
    print_runtime_kv(
        "IBM slip",
        "max/rms=" + to_string(statistics.max_slip_lu, 8u) + "/" +
            to_string(statistics.rms_slip_lu, 8u) +
            " lu, iterations=" +
            to_string((ulong)statistics.iterations) +
            ", converged=" + (statistics.converged ? string("yes") : string("no")) +
            ", failed steps=" +
            to_string((ulong)statistics.nonconverged_steps));
    print_runtime_kv(
        "IBM reaction",
        "F=(" + to_string(statistics.reaction_force_lu.x, 8u) + "," +
            to_string(statistics.reaction_force_lu.y, 8u) + "," +
            to_string(statistics.reaction_force_lu.z, 8u) +
            ") lu");
}

void UniformRun::enqueue_read_u_rho() {
    solver.velocity().download();
    solver.density().download();
    if (include_temperature_avg) solver.temperature().download();
}

void UniformRun::enqueue_read_u_only() {
    solver.velocity().download();
}

void UniformRun::accumulate_from_buffers() {
    ++avg_count;
    const float inv_n = 1.0f / (float)avg_count;
    float* u_avg = avg_u.data();
    float* rho_avg = avg_rho.data();
    float* m2_u = M2_u.data();
    float* m2_v = M2_v.data();
    float* m2_w = M2_w.data();
    float* t_avg = include_temperature_avg ? avg_T.data() : nullptr;
    parallel_for(Ncells, [&](ulong n) {
        const ulong i3 = 3ull * n;
        const float ux = solver.velocity().x[n];
        const float uy = solver.velocity().y[n];
        const float uz = solver.velocity().z[n];
        float mean_u = u_avg[i3];
        float mean_v = u_avg[i3 + 1];
        float mean_w = u_avg[i3 + 2];

        const float delta_u = ux - mean_u;
        mean_u += delta_u * inv_n;
        const float delta2_u = ux - mean_u;
        m2_u[n] += delta_u * delta2_u;
        u_avg[i3] = mean_u;

        const float delta_v = uy - mean_v;
        mean_v += delta_v * inv_n;
        const float delta2_v = uy - mean_v;
        m2_v[n] += delta_v * delta2_v;
        u_avg[i3 + 1] = mean_v;

        const float delta_w = uz - mean_w;
        mean_w += delta_w * inv_n;
        const float delta2_w = uz - mean_w;
        m2_w[n] += delta_w * delta2_w;
        u_avg[i3 + 2] = mean_w;

        const float r = solver.density()[n];
        rho_avg[n]    += (r - rho_avg[n]) * inv_n;
        if (t_avg) {
            const float T = solver.temperature()[n];
            t_avg[n] += (T - t_avg[n]) * inv_n;
        }
    });
}

void UniformRun::accumulate_average() {
    if (use_device_statistics) {
        ++avg_count;
        solver.sample_statistics(avg_count);
    }
    else {
        accumulate_from_buffers();
    }
}

void UniformRun::reset_average_accumulator() {
    avg_count = 0ull;
    if (use_device_statistics) {
        solver.reset_statistics();
        return;
    }
    std::fill(avg_u.begin(), avg_u.end(), 0.0f);
    std::fill(avg_rho.begin(), avg_rho.end(), 0.0f);
    std::fill(M2_u.begin(), M2_u.end(), 0.0f);
    std::fill(M2_v.begin(), M2_v.end(), 0.0f);
    std::fill(M2_w.begin(), M2_w.end(), 0.0f);
    if (include_temperature_avg) std::fill(avg_T.begin(), avg_T.end(), 0.0f);
}

ulong UniformRun::count_avg_samples_after(const ulong current_t) {
    if (avg_window == 0ull || avg_start_t == max_ulong || current_t >= total_steps) return 0ull;
    ulong first = std::max(current_t + (ulong)1u, avg_start_t);
    const ulong offset = (first - avg_start_t) % avg_stride;
    if (offset != 0ull) first += avg_stride - offset;
    if (first > total_steps) return 0ull;
    return (ulong)1u + (total_steps - first) / avg_stride;
}

bool UniformRun::should_accumulate_avg() {
    if (avg_window == 0ull) return false;
    if (solver.time() < avg_start_t) return false;
    return ((solver.time() - avg_start_t) % avg_stride) == 0ull;
}

bool UniformRun::should_sample_probes() {
    return !resolved_probes.empty() && solver.time() >= probe_start_t;
}

void UniformRun::capture_probe_sample() {
    const double time_si = (double)solver.time() * dt_si;
    if (use_device_probes) {
        solver.sample_probes(device_probe_velocity);
        ulong sample = 0ull;
        for (ResolvedProbe& probe : resolved_probes) {
            probe.times_si.push_back(time_si);
            for (const uint z : probe.z_indices) {
                (void)z;
                const ulong i3 = 3ull * sample++;
                probe.velocity_series_si.push_back(units.si_u(device_probe_velocity[(size_t)i3]));
                probe.velocity_series_si.push_back(units.si_u(device_probe_velocity[(size_t)(i3 + 1ull)]));
                probe.velocity_series_si.push_back(units.si_u(device_probe_velocity[(size_t)(i3 + 2ull)]));
            }
        }
        return;
    }
    for (ResolvedProbe& probe : resolved_probes) {
        probe.times_si.push_back(time_si);
        for (const uint z : probe.z_indices) {
            const ulong n = solver.grid().index(probe.x, probe.y, z);
            probe.velocity_series_si.push_back(units.si_u(solver.velocity().x[n]));
            probe.velocity_series_si.push_back(units.si_u(solver.velocity().y[n]));
            probe.velocity_series_si.push_back(units.si_u(solver.velocity().z[n]));
        }
    }
}

bool UniformRun::process_post_step_samples() {
    const bool want_avg = should_accumulate_avg();
    const bool want_probe = should_sample_probes();
    if (!want_avg && !want_probe) return false;
    const ulong t_now = solver.time();

    if (want_avg) {
        if (use_device_statistics) {
            accumulate_average();
        }
        else {
            const bool need_T = include_temperature_avg;
            const bool have_rho = (last_rho_host_t == t_now);
            const bool have_T = !need_T || (last_T_host_t == t_now);
            if (last_u_host_t != t_now || !have_rho || !have_T) {
                enqueue_read_u_rho();
                last_u_host_t = t_now;
                last_rho_host_t = t_now;
                if (need_T) last_T_host_t = t_now;
            }
            accumulate_average();
        }
    }
    if (want_probe) {
        if (!use_device_probes) {
            if (last_u_host_t != t_now) {
                enqueue_read_u_only();
                last_u_host_t = t_now;
            }
        }
        capture_probe_sample();
    }
    return want_avg;
}

bool UniformRun::is_avg_phase() {
    return avg_window > 0ull && solver.time() >= avg_start_t;
}

void UniformRun::push_vtk_saved_file(const string& filename) {
    vtk_saved_files.push_back(filename);
}

void UniformRun::flush_vtk_saved_files() {
    if (vtk_saved_files.empty()) return;
    {
        const StatusPause pause;
        bool first = true;
        for (const string& filename : vtk_saved_files) {
            print_kv_row(first ? "VTK file" : "", filename + " saved");
            first = false;
        }
    }
    const string last_file = vtk_saved_files.back();
    emit_progress_stage(
        "save",
        "Saving results",
        vtk_saved_files.size() == 1u
            ? last_file
            : to_string((ulong)vtk_saved_files.size()) + " files saved; last: " + last_file,
        (long long)vtk_saved_files.size(),
        (long long)vtk_saved_files.size(),
        false);
    vtk_saved_files.clear();
}

void UniformRun::maybe_write_unsteady_u() {
    if (!enable_unsteady_u_output) return;
    const ulong t = solver.time();
    if (t == 0ull || t < deck.culbm_unsteady_start_step || (t % unsteady_interval) != 0ull) return;
    push_vtk_saved_file(default_filename(vtk_dir, "u", ".vtk", t));
    solver.velocity().write_device_to_vtk(vtk_dir, true, false, vtk_output_Nz);
    last_u_host_t = t;
    last_unsteady_u_vtk_t = t;
    flush_vtk_saved_files();
}

void UniformRun::finalize_avg() {
    if (avg_count == 0ull) {
        flush_vtk_saved_files();
        return;
    }
    if (use_device_statistics) {
        solver.read_statistics(
            avg_u, avg_rho, avg_T, M2_u, M2_v, M2_w);
    }
    const ulong t_now = solver.time();
    if (last_flags_host_t != t_now) {
        solver.flags().download();
        last_flags_host_t = t_now;
    }
    std::vector<uchar> flags_host((size_t)Ncells, (uchar)0u);
    solver.flags().copy_contiguous_component(0u, flags_host.data(), 0u, solver.grid().size(2u));
    const std::string avg_name = vtk_prefix + deck.datetime + "_avg";
    const std::string avg_file = default_filename(results_vtk_dir, avg_name, ".vtk", solver.time());
    const float* T_avg_ptr = include_temperature_avg ? avg_T.data() : nullptr;
    write_avg_vtk(avg_file, solver.grid().size(0u), solver.grid().size(1u), solver.grid().size(2u),
                  avg_u.data(), avg_rho.data(), T_avg_ptr, flags_host.data(),
                  M2_u.data(), M2_v.data(), M2_w.data(),
                  avg_count, true, avg_turbulence_output(deck), false, vtk_output_Nz);
    push_vtk_saved_file(avg_file);
    flush_vtk_saved_files();
    print_kv_row("Avg samples", to_string(avg_count));
}

void UniformRun::finalize_probes() {
    if (!culbm::mpi_is_root() || resolved_probes.empty()) return;
    const std::filesystem::path results_dir = std::filesystem::path(deck.parent) / "RESULTS";
    std::filesystem::create_directories(results_dir);
    ulong written_files = 0ull;
    for (const ResolvedProbe& probe : resolved_probes) {
        const std::filesystem::path out_path = results_dir / (probe.file_stem + ".csv");
        std::ofstream fout(out_path.string(), std::ios::out | std::ios::trunc);
        if (!fout.is_open()) {
            print_runtime_kv("Probe output", "failed to open " + out_path.string());
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
        fout.close();
        written_files++;
    }
    print_runtime_kv("Probe files", to_string(written_files) + " CSV saved to RESULTS");
    emit_progress_stage("save",
                        "Saving results",
                        to_string(written_files) + " probe CSV file(s) saved to RESULTS",
                        (long long)written_files,
                        (long long)written_files,
                        false);
}

void UniformRun::write_final_physical_fields() {
    if (!deck.culbm_output_physical_requested) return;
    // Read all fields on every rank before testing mpi_is_root(): the
    // replicated host view uses collective gathers in MPI mode.
    const ulong t_now = solver.time();
    if (last_u_host_t != t_now) {
        solver.velocity().download();
        last_u_host_t = t_now;
    }
    if (last_rho_host_t != t_now) {
        solver.density().download();
        last_rho_host_t = t_now;
    }
    if (last_flags_host_t != t_now) {
        solver.flags().download();
        last_flags_host_t = t_now;
    }
    if (inflow.use_temperature_bc && last_T_host_t != t_now) {
        solver.temperature().download();
        last_T_host_t = t_now;
    }
    if (!culbm::mpi_is_root()) return;

    const float spacing_si = units.si_x(1.0f);
    float3 global_origin = spacing_si * float3(
        0.5f - 0.5f * (float)solver.grid().size(0u),
        0.5f - 0.5f * (float)solver.grid().size(1u),
        0.5f - 0.5f * (float)solver.grid().size(2u)) + vtk_origin_shift;
    global_origin.z -= terrain.physical_output_z_zero_si;
    const float z_tolerance = fmaxf(1.0e-6f, spacing_si * 1.0e-7f);
    uint z_begin = 0u;
    while (z_begin < vtk_output_Nz && global_origin.z + (float)z_begin * spacing_si < -z_tolerance) {
        ++z_begin;
    }
    if (z_begin >= vtk_output_Nz) {
        println("| ERROR: physical output has no non-basement solver grid plane.               |");
        return;
    }

    const uint nx = solver.grid().size(0u);
    const uint ny = solver.grid().size(1u);
    const uint nz = vtk_output_Nz;
    const ulong slice_cells = (ulong)nx * (ulong)ny * (ulong)nz;
    std::vector<float> flat_ux((size_t)slice_cells), flat_uy((size_t)slice_cells), flat_uz((size_t)slice_cells);
    std::vector<float> flat_rho((size_t)slice_cells), flat_T;
    std::vector<uchar> flat_flags((size_t)slice_cells);
    solver.velocity().copy_contiguous_component(0u, flat_ux.data(), 0u, nz);
    solver.velocity().copy_contiguous_component(1u, flat_uy.data(), 0u, nz);
    solver.velocity().copy_contiguous_component(2u, flat_uz.data(), 0u, nz);
    solver.density().copy_contiguous_component(0u, flat_rho.data(), 0u, nz);
    solver.flags().copy_contiguous_component(0u, flat_flags.data(), 0u, nz);
    if (inflow.use_temperature_bc) {
        flat_T.resize((size_t)slice_cells);
        solver.temperature().copy_contiguous_component(0u, flat_T.data(), 0u, nz);
    }

    culbm::PhysicalGridView view;
    view.nx = nx;
    view.ny = ny;
    view.nz = nz;
    view.z_begin = z_begin;
    view.origin_x = global_origin.x;
    view.origin_y = global_origin.y;
    view.origin_z = global_origin.z;
    view.spacing_x = spacing_si;
    view.spacing_y = spacing_si;
    view.spacing_z = spacing_si;
    view.velocity_x = flat_ux.data();
    view.velocity_y = flat_uy.data();
    view.velocity_z = flat_uz.data();
    view.density = flat_rho.data();
    view.flags = flat_flags.data();
    view.velocity_scale = units.si_u(1.0f);
    view.density_scale = units.si_rho(1.0f);
    if (inflow.use_temperature_bc) {
        view.temperature = flat_T.data();
        view.temperature_scale = units.si_dT(1.0f);
        view.temperature_offset = units.si_T(0.0f);
    }
    std::vector<float> ground;
    culbm::PhysicalAglSurfaceView surface;
    if (deck.culbm_output_agl_requested) {
        ground = make_physical_output_ground_grid(ctx, 
            view.nx, view.ny, global_origin, spacing_si);
        surface.ground_z = ground.data();
        surface.ground_count = (std::uint64_t)ground.size();
    }
    const ulong step = solver.time();
    write_physical_outputs(deck, view, surface, results_vtk_dir, results_netcdf_dir, vtk_prefix + deck.datetime, step);
}

void UniformRun::write_final_transient() {
    const ulong vtk_t = solver.time();
    const bool need_u = (last_unsteady_u_vtk_t != vtk_t) || deck.culbm_output_physical_requested;
    const bool need_rho = true;
    const bool need_T = inflow.use_temperature_bc;
    const bool need_flags = write_diagnostic_flags || deck.culbm_output_physical_requested;
    if (need_u && last_u_host_t != vtk_t) {
        solver.velocity().download();
        last_u_host_t = vtk_t;
    }
    if (need_rho && last_rho_host_t != vtk_t) {
        solver.density().download();
        last_rho_host_t = vtk_t;
    }
    if (need_T && last_T_host_t != vtk_t) {
        solver.temperature().download();
        last_T_host_t = vtk_t;
    }
    if (need_flags && last_flags_host_t != vtk_t) {
        solver.flags().download();
        last_flags_host_t = vtk_t;
    }
    if (last_unsteady_u_vtk_t != vtk_t) {
        push_vtk_saved_file(default_filename(vtk_dir, "u", ".vtk", vtk_t));
        solver.velocity().write_host_to_vtk(vtk_dir, true, false, vtk_output_Nz);
    }
    push_vtk_saved_file(default_filename(vtk_dir, "rho", ".vtk", vtk_t));
    solver.density().write_host_to_vtk(vtk_dir, true, false, vtk_output_Nz);
    if (inflow.use_temperature_bc) {
        push_vtk_saved_file(default_filename(vtk_dir, "T", ".vtk", vtk_t));
        solver.temperature().write_host_to_vtk(vtk_dir, true, false, vtk_output_Nz);
    }
    if (write_diagnostic_flags) {
        push_vtk_saved_file(default_filename(vtk_dir, "flags", ".vtk", vtk_t));
        solver.flags().write_host_to_vtk(vtk_dir, true, false, vtk_output_Nz);
    }
    write_final_physical_fields();
}

void UniformRun::maybe_write_transform_info() {
    if (!culbm::mpi_is_root() || extra_run_steps == 0ull) return;
    println("| Writing transform.info...                                                  |");
    std::string info_path = deck.parent + "/proj_temp/transform.info";
    std::ofstream info_file(info_path);
    if (info_file.is_open()) {
        const float dt_si = deck.cell_m * (domain.lbm_ref_u / physics.si_ref_u);
        info_file << "dt = " << std::fixed << std::setprecision(10) << dt_si << "s\n";
        info_file.close();
        println("| Successfully wrote " + info_path + " |");
        emit_progress_stage("save",
                            "Saving results",
                            info_path,
                            1ll,
                            1ll,
                            false);
    }
    else {
        println("ERROR: Could not open " + info_path + " for writing.");
    }
}

} // namespace culbm::run
