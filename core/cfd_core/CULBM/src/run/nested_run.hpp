#pragma once

// Time loop of one case on a CUDA vertically nested hierarchy: the levels are
// built from the seed state of the uniform lattice and the common-finest
// geometry raster, advanced by the nested scheduler in coarse steps, and
// written as stitched common-finest fields (raw, mean field, probes,
// physical outputs).

#include "probes.hpp"
#include "schedule.hpp"
#include "../bc/inflow/vk_inlet.hpp"
#include "../cuda_nested_solver.hpp"

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

class NestedRun {
public:
    NestedRun(cases::CaseContext& ctx, const culbm::NestedSeedState& seed, const std::string& vtk_prefix,
              bool extra_spacing, double uniform_preprocess_seconds,
              const culbm::NestedCommonFinestTerrain* finest_terrain, const std::string& downstream_bc_runtime);

    // Builds and runs the hierarchy; the preprocess plus solver seconds.
    double run();

private:
    void crop_nested_top_sponge(culbm::NestedStitchedFields& fields);
    float3 output_origin(const culbm::NestedStitchedFields& fields);
    culbm::NestedStitchedFields stitch_or_fail();
    void write_raw_fields(const culbm::NestedStitchedFields& fields, ulong step, bool u_only);
    void write_final_physical_fields(const culbm::NestedStitchedFields& fields, ulong step);
    void resolve_nested_probes(const culbm::NestedStitchedFields& fields);
    void configure_nested_device_probes(const culbm::NestedStitchedFields& fields);
    void capture_nested_probe_sample_device(ulong coarse_step);
    void capture_nested_probe_sample(const culbm::NestedStitchedFields& fields, ulong coarse_step);
    void finalize_nested_probes();
    void enable_nested_device_statistics();
    void finalize_nested_device_statistics(const culbm::NestedStitchedFields& final_fields);
    void accumulate_average(const culbm::NestedStitchedFields& fields);
    void sample_nested_sgs_diagnostics(ulong coarse_step);

    cases::CaseContext& ctx;
    const cases::CaseSettings& deck;
    const cases::CaseDomain& domain;
    const bc::InflowData& inflow;
    const cases::CasePhysics& physics;
    const cases::CaseGeometry& geometry;
    const cases::CaseTerrain& terrain;
    const ProbePlan& probes;
    const culbm::NestedSeedState& seed;
    const std::string vtk_prefix;
    const bool extra_spacing;
    const double uniform_preprocess_seconds;
    const culbm::NestedCommonFinestTerrain* const finest_terrain;
    const std::string downstream_bc_runtime;

    Clock nested_build_clock;
    double coarse_dx_si = 0.0;
    culbm::NestedLbmPhysics nested_physics;
    culbm::NestedVerticalLbm nested;
    std::unique_ptr<culbm::NestedLbmScheduler> scheduler;
    std::vector<std::unique_ptr<bc::VonKarmanInletUpdater>> nested_vk_updaters;
    double preprocess_seconds = 0.0;

    // Step schedule (RunSchedule).
    ulong extra_run_steps = 0ull;
    ulong total_steps = 0ull;
    ulong unsteady_interval = 0ull;
    bool do_avg = false;
    ulong avg_window = 0ull;
    ulong avg_stride = 1ull;
    ulong avg_start_t = max_ulong;
    ulong probe_window = 0ull;
    ulong probe_start_t = max_ulong;

    std::string results_vtk_dir;
    std::string results_netcdf_dir;
    std::string vtk_dir;
    double nested_output_height_si = 0.0;
    std::vector<ResolvedProbe> nested_probes;
    struct LevelProbeBinding {
        std::size_t level_index = 0u;
        ulong slot = 0u;
    };
    bool nested_device_probes_active = false;
    std::vector<std::vector<LevelProbeBinding>> probe_level_bindings;
    std::vector<std::vector<float>> level_probe_scratch;
    bool nested_device_stats_active = false;
    std::vector<float> avg_u, avg_rho, avg_T, M2_u, M2_v, M2_w;
    std::vector<uchar> avg_flags;
    ulong avg_count = 0u;
    Clock case_solver_clock;
    std::ofstream nested_sgs_diagnostics_output;
    bool nested_sgs_diagnostics_warned = false;
};

} // namespace culbm::run
