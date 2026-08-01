#pragma once

#include "particle_advection_config.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

class LBM;

namespace culbm {

struct ParticleAdvectionGeometry {
    // Physical coordinate of global lattice sample (0,0,0).  x/y use the
    // case's metric STL coordinates; z uses the same rebased global datum as
    // culbm_output_global_* (it is not local AGL).
    std::array<double, 3u> lattice_origin_m{};
    double spacing_m = 0.0;
};

// GPU storage owned by the independent particle-advection runtime.  The MiB
// estimate follows Memory<T>'s per-allocation ceil accounting so mesh fitting
// and the allocator use the same convention.  automatic_release_slots is the
// number of release events, not the number of particles.
std::uint64_t particle_advection_gpu_memory_bytes_per_device(
    std::uint64_t particle_slots,
    std::uint64_t local_lbm_cells,
    std::uint64_t automatic_release_slots = 0u);
std::uint64_t particle_advection_gpu_memory_mb_per_device(
    std::uint64_t particle_slots,
    std::uint64_t local_lbm_cells,
    std::uint64_t automatic_release_slots = 0u);

class ParticleAdvection {
public:
    ParticleAdvection(
        LBM& lbm,
        particle_advection::Config config,
        ParticleAdvectionGeometry geometry,
        std::uint64_t total_steps,
        std::filesystem::path case_directory,
        std::filesystem::path output_directory,
        std::string output_stem);
    ~ParticleAdvection();

    ParticleAdvection(const ParticleAdvection&) = delete;
    ParticleAdvection& operator=(const ParticleAdvection&) = delete;

    void initialize();
    // Call after a solver pre-step hook changes the t_n flow state and before
    // the corresponding LBM step. This keeps RK stage-time interpolation
    // consistent with dynamic CUDA boundary forcing.
    void refresh_velocity_before_lbm_step();
    void advance_after_lbm_step(std::uint64_t step_begin);
    bool write_snapshot_if_due(std::uint64_t step, bool force = false);

    std::uint64_t particle_count() const;
    std::uint64_t migrated_particles() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace culbm
