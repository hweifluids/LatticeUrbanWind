#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace culbm::particle_advection {

enum class Mode {
    Prescribed,
    Automatic,
};

enum class Integrator {
    Euler,
    HeunRk2,
    Rk4,
    Rk6,
};

struct Config {
    bool enabled = false;
    Mode mode = Mode::Prescribed;
    Integrator integrator = Integrator::Rk4;
    std::filesystem::path particle_file = "particle.txt";

    // SI seconds. A global release time overrides per-row particle-file times.
    std::optional<double> release_time_s;
    // Missing means that particles never expire.
    std::optional<double> lifetime_s;
    // Missing means that only the initial and final particle states are output.
    std::optional<double> output_interval_s;
    std::uint64_t max_particles = 1000000u;

    // Automatic mode emits at release_time_s.value_or(0) + k*interval.
    std::optional<double> release_interval_s;
    std::optional<std::array<double, 3u>> automatic_spacing_m;
    std::optional<std::array<double, 3u>> automatic_min_m;
    std::optional<std::array<double, 3u>> automatic_max_m;
};

using RawDeck = std::unordered_map<std::string, std::string>;

// Parses only culbm_advection_* entries and ignores unrelated deck entries.
// Unknown culbm_advection_* entries and duplicate aliases are errors.
bool parse_config(const RawDeck& raw_deck, Config& config, std::string& error);

bool parse_strict_triplet(const std::string& raw,
                          std::array<double, 3u>& values,
                          std::string& error);

const char* mode_name(Mode mode);
const char* integrator_name(Integrator integrator);

struct ParticleSeed {
    std::array<double, 3u> position_m{};
    double release_time_s = 0.0;
    std::size_t source_line = 0u;
};

struct ParticleFileInfo {
    std::size_t rows_with_explicit_release_time = 0u;
    std::size_t rows_using_global_release_time = 0u;
    bool release_time_override_applied = false;
};

// Data rows are x y z [release_time_s]. Commas, semicolons, or whitespace are
// accepted separators. The first non-comment row may be an x/y/z[/t] header.
bool parse_particle_stream(std::istream& input,
                           const std::string& source_name,
                           const std::optional<double>& release_time_override_s,
                           std::vector<ParticleSeed>& particles,
                           std::string& error,
                           ParticleFileInfo* info = nullptr);

bool read_particle_file(const std::filesystem::path& path,
                        const std::optional<double>& release_time_override_s,
                        std::vector<ParticleSeed>& particles,
                        std::string& error,
                        ParticleFileInfo* info = nullptr);

bool release_step_nearest(double release_time_s,
                          double seconds_per_step,
                          std::uint64_t& release_step,
                          std::string& error);

struct LifetimeSteps {
    bool infinite = true;
    std::uint64_t steps = std::numeric_limits<std::uint64_t>::max();
};

// Finite lifetimes are floored so the represented lifetime never exceeds the
// configured SI maximum. A positive lifetime shorter than one step maps to 0.
bool lifetime_steps_floor(const std::optional<double>& lifetime_s,
                          double seconds_per_step,
                          LifetimeSteps& lifetime,
                          std::string& error);

struct QuantizedParticleSeed {
    std::array<double, 3u> position_m{};
    std::uint64_t release_step = 0u;
    std::size_t source_line = 0u;
};

bool quantize_particle_releases(const std::vector<ParticleSeed>& particles,
                                double seconds_per_step,
                                std::vector<QuantizedParticleSeed>& quantized,
                                std::string& error);

struct ReleaseBatch {
    std::uint64_t step = 0u;
    // More than one nominal SI release may map to the same lattice step.
    std::uint64_t multiplicity = 0u;
};

// Builds t_k = start_time_s + k*interval_s directly for every k. This avoids
// drift from repeatedly adding a rounded lattice-step interval.
bool build_automatic_release_schedule(
    double start_time_s,
    double interval_s,
    double seconds_per_step,
    std::uint64_t final_step,
    std::vector<ReleaseBatch>& schedule,
    std::string& error,
    std::uint64_t maximum_nominal_releases = 10000000u);

} // namespace culbm::particle_advection
