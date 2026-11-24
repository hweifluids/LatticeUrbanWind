#pragma once

// Time loop of one case on a uniform lattice: compact-ground sealing and
// SurfaceIBM attachment, the speed benchmarks and the two-phase ETA model,
// the per-step VK inlet and particle updates, diagnostics, unsteady output,
// mean-field statistics (device or host Welford accumulators) and probes,
// then the final fields, the mean-field file, the probe files and the
// physical outputs.

#include "probes.hpp"
#include "schedule.hpp"
#include "vtk_output.hpp"
#include "../particle_advection.hpp"
#include "../solver/solver.hpp"

#include <memory>

namespace culbm::cases {
struct CaseContext;
struct CaseSettings;
struct CaseDomain;
struct CasePhysics;
struct CaseGeometry;
struct CaseTerrain;
}
namespace culbm::bc {
struct InflowData;
}

namespace culbm::run {

AvgTurbulenceOutput avg_turbulence_output(const cases::CaseSettings& deck);

class UniformRun {
public:
    using PreStepUpdate = std::function<void(culbm::solver::Solver&, ulong)>;

    // `lbm` carries the voxelized geometry and the boundary fields; output
    // names start with vtk_prefix.
    UniformRun(cases::CaseContext& ctx, culbm::solver::Solver& solver, const std::string& vtk_prefix, bool extra_spacing,
               double preprocess_seconds, PreStepUpdate pre_step_update);

    // Runs the case; the preprocess plus solver seconds.
    double run();

private:
    // Preparation, in run order.
    void seal_compact_ground();
    void attach_surface_ibm();
    void open_outputs();
    void open_diagnostics();
    void print_run_settings();
    void prepare_mean_field();
    void resolve_probes();
    void start_particle_advection();
    void finish_particle_advection();

    // Reporting and timing.
    std::string format_estimated_time(double seconds, bool normal_hint_ready);
    void emit_solver_progress(bool force = false);
    double print_task_finished();
    double run_avg_phase_benchmark();
    void configure_eta_model(double avg_sample_seconds_hint);
    void print_timing_plan(double avg_sample_seconds_hint);
    void update_runtime_eta(double iteration_seconds, bool avg_sampled);
    void run_normal_speed_benchmark(const std::function<void()>& before_step);

    // Per-step work after the solver step.
    void sample_sgs_diagnostics();
    void sample_surface_ibm_diagnostics();
    void enqueue_read_u_rho();
    void enqueue_read_u_only();
    void accumulate_from_buffers();
    void accumulate_average();
    void reset_average_accumulator();
    ulong count_avg_samples_after(ulong current_t);
    bool should_accumulate_avg();
    bool should_sample_probes();
    bool step_needs_macros(ulong next_step) const;
    void capture_probe_sample();
    bool process_post_step_samples();
    bool is_avg_phase();
    void refresh_particle_advection_velocity();
    void advance_particle_advection();

    // Outputs.
    void push_vtk_saved_file(const string& filename);
    void flush_vtk_saved_files();
    void maybe_write_unsteady_u();
    void finalize_avg();
    void finalize_probes();
    void write_final_physical_fields();
    void write_final_transient();
    void maybe_write_transform_info();

    cases::CaseContext& ctx;
    const cases::CaseSettings& deck;
    const cases::CaseDomain& domain;
    const bc::InflowData& inflow;
    const cases::CasePhysics& physics;
    const cases::CaseGeometry& geometry;
    const cases::CaseTerrain& terrain;
    const ProbePlan& probes;
    culbm::solver::Solver& solver;
    const std::string vtk_prefix;
    const bool extra_spacing;
    const double preprocess_seconds;
    const PreStepUpdate pre_step_update;

    Clock case_solver_clock;
    // Step schedule (RunSchedule).
    ulong base_run_steps = 0ull;
    ulong extra_run_steps = 0ull;
    ulong total_steps = 0ull;
    ulong unsteady_interval = 0ull;
    bool enable_unsteady_u_output = false;
    bool write_diagnostic_flags = false;
    bool do_avg = false;
    ulong avg_window = 0ull;
    ulong avg_stride = 1ull;
    ulong avg_start_t = max_ulong;
    ulong probe_window = 0ull;
    ulong probe_start_t = max_ulong;
    std::chrono::steady_clock::time_point last_solver_progress{};

    std::string results_vtk_dir;
    std::string results_netcdf_dir;
    std::string vtk_dir;
    uint vtk_output_Nz = 0u;
    std::ofstream sgs_diagnostics_output;
    bool sgs_diagnostics_warned = false;
    std::ofstream surface_ibm_diagnostics_output;
    bool surface_ibm_diagnostics_warned = false;
    std::vector<string> vtk_saved_files;

    // Mean field.
    ulong Ncells = 0ull;
    bool include_temperature_avg = false;
    uint deferred_particle_gpu_memory_mb = 0u;
    std::vector<float> avg_u;
    std::vector<float> avg_rho;
    std::vector<float> avg_T;
    std::vector<float> M2_u;
    std::vector<float> M2_v;
    std::vector<float> M2_w;
    ulong avg_count = 0ull;
    bool use_device_statistics = false;

    // Probes.
    double dt_si = 0.0;
    std::vector<ResolvedProbe> resolved_probes;
    std::vector<ulong> device_probe_indices;
    bool use_device_probes = false;
    std::vector<float> device_probe_velocity;

    // Step of the host copies of u, rho, T and flags.
    ulong last_u_host_t = max_ulong;
    ulong last_rho_host_t = max_ulong;
    ulong last_T_host_t = max_ulong;
    ulong last_flags_host_t = max_ulong;
    bool avg_phase_entry_reported = false;
    ulong last_unsteady_u_vtk_t = max_ulong;
    std::unique_ptr<culbm::ParticleAdvection> particle_advection_runtime;
};

} // namespace culbm::run
