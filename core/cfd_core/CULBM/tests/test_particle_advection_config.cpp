#include "particle_advection_config.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using culbm::particle_advection::Config;
using culbm::particle_advection::Integrator;
using culbm::particle_advection::LifetimeSteps;
using culbm::particle_advection::Mode;
using culbm::particle_advection::ParticleSeed;
using culbm::particle_advection::ParticleFileInfo;
using culbm::particle_advection::QuantizedParticleSeed;
using culbm::particle_advection::RawDeck;
using culbm::particle_advection::ReleaseBatch;

struct TestState {
    int failures = 0;

    void expect(const bool condition, const std::string& message) {
        if (condition) return;
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
};

bool close(const double left, const double right) {
    return std::fabs(left - right) <= 1.0e-12;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

void test_triplets(TestState& test) {
    std::array<double, 3u> values{};
    std::string error;
    test.expect(culbm::particle_advection::parse_strict_triplet(
                    "[1.5, 2; 3e1]", values, error),
                "triplet accepts brackets and mixed documented separators");
    test.expect(close(values[0], 1.5) && close(values[1], 2.0) &&
                    close(values[2], 30.0),
                "triplet preserves all three values");
    test.expect(culbm::particle_advection::parse_strict_triplet(
                    "-1 0 2", values, error),
                "triplet accepts whitespace-only separation and signed values");
    test.expect(!culbm::particle_advection::parse_strict_triplet(
                    "[1, 2]", values, error) &&
                    contains(error, "exactly 3"),
                "triplet rejects too few items");
    test.expect(!culbm::particle_advection::parse_strict_triplet(
                    "[1, 2, 3, 4]", values, error) &&
                    contains(error, "exactly 3"),
                "triplet rejects too many items");
    test.expect(!culbm::particle_advection::parse_strict_triplet(
                    "[1,,2,3]", values, error) &&
                    contains(error, "empty item"),
                "triplet rejects empty comma-delimited items");
    test.expect(!culbm::particle_advection::parse_strict_triplet(
                    "[1, nan, 3]", values, error) &&
                    contains(error, "finite"),
                "triplet rejects non-finite numbers");
    test.expect(!culbm::particle_advection::parse_strict_triplet(
                    "[1, 2, 3", values, error) &&
                    contains(error, "unmatched"),
                "triplet rejects unmatched brackets");
}

void test_config_defaults_and_valid_modes(TestState& test) {
    Config config;
    std::string error;
    test.expect(culbm::particle_advection::parse_config(
                    {{"casename", "ignored"}, {"cell_size", "20"}},
                    config, error),
                "unrelated deck keys are ignored");
    test.expect(!config.enabled && config.mode == Mode::Prescribed &&
                    config.integrator == Integrator::Rk4 &&
                    config.particle_file == std::filesystem::path("particle.txt"),
                "disabled prescribed RK4 defaults are stable");
    test.expect(!config.release_time_s.has_value() &&
                    !config.lifetime_s.has_value() &&
                    !config.output_interval_s.has_value() &&
                    !config.release_interval_s.has_value() &&
                    config.max_particles == 1000000u,
                "optional times are absent by default");

    const RawDeck prescribed = {
        {"CULBM Advection Enable", "\"yes\""},
        {"culbm-advection-mode", "\"given\""},
        {"culbm_advection_method", "\"HEUN-RK2\""},
        {"culbm_advection_file", "\"seed particles.txt\""},
        {"culbm_advection_release_time_s", "2.25"},
        {"culbm_advection_duration_s", "9.5"},
        {"culbm_advection_output_interval_s", "0.5"},
        {"culbm_advection_max_particles", "2500000"},
    };
    test.expect(culbm::particle_advection::parse_config(
                    prescribed, config, error),
                "prescribed aliases and normalized raw keys parse");
    test.expect(config.enabled && config.mode == Mode::Prescribed &&
                    config.integrator == Integrator::HeunRk2,
                "prescribed mode and Heun-RK2 are selected");
    test.expect(config.particle_file ==
                    std::filesystem::path("seed particles.txt") &&
                    close(*config.release_time_s, 2.25) &&
                    close(*config.lifetime_s, 9.5) &&
                    close(*config.output_interval_s, 0.5) &&
                    config.max_particles == 2500000u,
                "path, times, and particle capacity are preserved");

    const std::array<std::pair<const char*, Integrator>, 8u> integrators = {{
        {"euler", Integrator::Euler},
        {"RK1", Integrator::Euler},
        {"heun", Integrator::HeunRk2},
        {"rk2", Integrator::HeunRk2},
        {"rk4", Integrator::Rk4},
        {"runge-kutta-4", Integrator::Rk4},
        {"rk6", Integrator::Rk6},
        {"runge_kutta_6", Integrator::Rk6},
    }};
    for (const auto& [token, expected] : integrators) {
        test.expect(culbm::particle_advection::parse_config(
                        {{"culbm_advection_integrator", token}},
                        config, error) &&
                        config.integrator == expected,
                    std::string("integrator token parses: ") + token);
    }

    const RawDeck automatic = {
        {"culbm_advection_enabled", "true"},
        {"culbm_advection_mode", "auto"},
        {"culbm_advection_integrator", "RK6"},
        {"culbm_advection_delta_s", "0.25"},
        {"culbm_advection_automatic_spacing_m", "[5, 10, 20]"},
        {"culbm_advection_automatic_min_m", "[0, -5, 1]"},
        {"culbm_advection_automatic_max_m", "[100, 50, 21]"},
    };
    test.expect(culbm::particle_advection::parse_config(
                    automatic, config, error),
                "automatic mode accepts interval, spacing, and optional bounds");
    test.expect(config.mode == Mode::Automatic &&
                    config.integrator == Integrator::Rk6 &&
                    close(*config.release_interval_s, 0.25),
                "automatic mode values are retained");
    test.expect((*config.automatic_spacing_m ==
                    std::array<double, 3u>{5.0, 10.0, 20.0}) &&
                    (*config.automatic_min_m ==
                    std::array<double, 3u>{0.0, -5.0, 1.0}) &&
                    (*config.automatic_max_m ==
                    std::array<double, 3u>{100.0, 50.0, 21.0}),
                "automatic triplets retain SI values");
    test.expect(std::string(culbm::particle_advection::mode_name(config.mode)) ==
                    "automatic" &&
                    std::string(culbm::particle_advection::integrator_name(
                        config.integrator)) == "rk6",
                "enum names are canonical and stable");
}

void test_config_errors(TestState& test) {
    Config config;
    std::string error;
    auto rejects = [&](const RawDeck& raw, const std::string& needle,
                       const std::string& message) {
        error.clear();
        test.expect(!culbm::particle_advection::parse_config(raw, config, error) &&
                        contains(error, needle),
                    message + " (error was: " + error + ")");
    };

    rejects({{"culbm_advection_enabled", ""}}, "bool-like",
            "blank enabled token is rejected");
    rejects({{"culbm_advection_mode", "random"}}, "prescribed or automatic",
            "unknown mode is rejected");
    rejects({{"culbm_advection_integrator", "rk3"}}, "euler",
            "unsupported integration method is rejected");
    rejects({{"culbm_advection_release_time_s", "-1"}}, ">= 0",
            "negative release override is rejected");
    rejects({{"culbm_advection_lifetime_s", "0"}}, "> 0",
            "explicit zero lifetime is rejected instead of meaning infinity");
    rejects({{"culbm_advection_output_interval_s", "0"}}, "> 0",
            "explicit zero output interval is rejected");
    rejects({{"culbm_advection_max_particles", "0"}}, "positive integer",
            "zero particle capacity is rejected");
    rejects({{"culbm_advection_max_particles", "1.5"}}, "positive integer",
            "fractional particle capacity is rejected");
    rejects({{"culbm_advection_max_particles", "-2"}}, "positive integer",
            "negative particle capacity is rejected");
    rejects({
                {"culbm_advection_max_particles",
                 "18446744073709551616"},
            }, "positive integer",
            "particle-capacity overflow is rejected");
    rejects({{"culbm_advection_release_interval_s", "inf"}}, "finite",
            "non-finite release interval is rejected");
    rejects({{"culbm_advection_auto_spacing_m", "[1, 0, 2]"}}, "> 0",
            "zero automatic spacing is rejected");
    rejects({{"culbm_advection_auto_min_m", "[0, 0, 0]"}}, "set together",
            "one-sided automatic bounds are rejected");
    rejects({
                {"culbm_advection_auto_min_m", "[0, 0, 0]"},
                {"culbm_advection_auto_max_m", "[1, 0, 2]"},
            }, "min < max",
            "non-increasing automatic bounds are rejected");
    rejects({
                {"culbm_advection_enabled", "true"},
                {"culbm_advection_mode", "automatic"},
                {"culbm_advection_auto_spacing_m", "[1, 1, 1]"},
            }, "release_interval",
            "enabled automatic mode requires an interval");
    rejects({
                {"culbm_advection_enabled", "true"},
                {"culbm_advection_mode", "automatic"},
                {"culbm_advection_release_interval_s", "1"},
            }, "auto_spacing",
            "enabled automatic mode requires a spacing triplet");
    rejects({{"culbm_advection_typo", "true"}}, "unknown",
            "unknown prefixed key is rejected");
    rejects({
                {"culbm_advection_enable", "true"},
                {"culbm_advection_enabled", "true"},
            }, "duplicate",
            "canonical and alias duplicates are rejected");

    test.expect(culbm::particle_advection::parse_config({
                    {"culbm_advection_enabled", "false"},
                    {"culbm_advection_mode", "automatic"},
                }, config, error),
                "disabled automatic settings may remain incomplete in a deck");
}

void test_particle_stream(TestState& test) {
    const std::string text =
        "\xEF\xBB\xBFx[m], y[m], z[m], release_time[s]\n"
        "# comment-only line\n"
        "1, 2, 3, 4.5 // first\n"
        "5; 6; 7; 8\n"
        "9 10 11\n";
    std::istringstream input(text);
    std::vector<ParticleSeed> particles;
    std::string error;
    test.expect(culbm::particle_advection::parse_particle_stream(
                    input, "memory-particles", std::nullopt,
                    particles, error),
                "XYZ/XYZT particle stream accepts header, BOM, delimiters, and comments");
    test.expect(particles.size() == 3u,
                "particle stream returns exactly the data rows");
    if (particles.size() == 3u) {
        test.expect(particles[0].position_m ==
                        std::array<double, 3u>{1.0, 2.0, 3.0} &&
                        close(particles[0].release_time_s, 4.5) &&
                        particles[0].source_line == 3u,
                    "XYZT row retains coordinates, release time, and source line");
        test.expect(particles[1].position_m ==
                        std::array<double, 3u>{5.0, 6.0, 7.0} &&
                        close(particles[1].release_time_s, 8.0),
                    "semicolon-separated XYZT row parses");
        test.expect(particles[2].position_m ==
                        std::array<double, 3u>{9.0, 10.0, 11.0} &&
                        close(particles[2].release_time_s, 0.0),
                    "XYZ row defaults to release time zero");
    }

    std::istringstream overridden(text);
    ParticleFileInfo info;
    test.expect(culbm::particle_advection::parse_particle_stream(
                    overridden, "override", 7.25, particles, error, &info),
                "global release time can override row times");
    bool all_overridden = particles.size() == 3u;
    for (const ParticleSeed& particle : particles) {
        all_overridden = all_overridden && close(particle.release_time_s, 7.25);
    }
    test.expect(all_overridden, "global release override applies to every row");
    test.expect(info.release_time_override_applied &&
                    info.rows_with_explicit_release_time == 2u &&
                    info.rows_using_global_release_time == 3u,
                "particle parse reports global override and explicit-time row counts");
}

void test_particle_stream_errors_and_file(TestState& test) {
    std::vector<ParticleSeed> particles;
    std::string error;
    auto rejects = [&](const std::string& text, const std::string& needle,
                       const std::string& message,
                       const std::optional<double>& override_s = std::nullopt) {
        std::istringstream input(text);
        error.clear();
        test.expect(!culbm::particle_advection::parse_particle_stream(
                        input, "bad-particles", override_s, particles, error) &&
                        contains(error, needle) && particles.empty(),
                    message + " (error was: " + error + ")");
    };

    rejects("# no data\nx y z t\n", "no particle rows",
            "header-only file is rejected");
    rejects("1 2\n", "exactly", "too few columns are rejected");
    rejects("1 2 3 4 5\n", "exactly", "too many columns are rejected");
    rejects("1,,2,3\n", "empty column",
            "empty punctuation-delimited columns are rejected");
    rejects("1 nan 3\n", "finite", "NaN coordinate is rejected");
    rejects("1 2 3 inf\n", "finite", "infinite release time is rejected");
    rejects("1 2 3 -1\n", ">= 0",
            "negative row release time is rejected even with a global override",
            5.0);
    rejects("x y z t\n1 2 3 0\nx y z t\n", "coordinate",
            "a second header in the data body is rejected");
    rejects("1 2 3\n", "override", "negative global override is rejected", -1.0);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "culbm-particle-advection-config-test-particles.txt";
    {
        std::ofstream output(path, std::ios::trunc);
        output << "x,y,z,t\n1,2,3,4\n";
    }
    test.expect(culbm::particle_advection::read_particle_file(
                    path, std::nullopt, particles, error) &&
                    particles.size() == 1u &&
                    close(particles.front().release_time_s, 4.0),
                "particle-file wrapper reads a real file");
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);

    const std::filesystem::path missing =
        path.parent_path() / "culbm-particle-advection-file-does-not-exist.txt";
    test.expect(!culbm::particle_advection::read_particle_file(
                    missing, std::nullopt, particles, error) &&
                    contains(error, "could not open") && particles.empty(),
                "particle-file wrapper reports a missing file");
}

void test_time_quantization(TestState& test) {
    std::uint64_t step = 99u;
    std::string error;
    test.expect(culbm::particle_advection::release_step_nearest(
                    0.49, 1.0, step, error) && step == 0u,
                "release time below half-step maps down");
    test.expect(culbm::particle_advection::release_step_nearest(
                    0.5, 1.0, step, error) && step == 1u,
                "release time at half-step maps to the next frame");
    test.expect(culbm::particle_advection::release_step_nearest(
                    1.5, 1.0, step, error) && step == 2u,
                "release time uses nearest-frame half-up quantization");
    test.expect(culbm::particle_advection::release_step_nearest(
                    0.15, 0.1, step, error) && step == 2u,
                "nearest-frame conversion is stable at decimal half steps");
    test.expect(!culbm::particle_advection::release_step_nearest(
                    -0.1, 1.0, step, error) && contains(error, ">= 0"),
                "negative release time is rejected");
    test.expect(!culbm::particle_advection::release_step_nearest(
                    1.0, 0.0, step, error) && contains(error, "> 0"),
                "zero lattice-step duration is rejected");
    test.expect(!culbm::particle_advection::release_step_nearest(
                    std::numeric_limits<double>::max(), 1.0, step, error) &&
                    contains(error, "uint64"),
                "release-step overflow is rejected");

    LifetimeSteps lifetime;
    test.expect(culbm::particle_advection::lifetime_steps_floor(
                    std::nullopt, 0.25, lifetime, error) &&
                    lifetime.infinite &&
                    lifetime.steps == std::numeric_limits<std::uint64_t>::max(),
                "missing lifetime remains infinite");
    test.expect(culbm::particle_advection::lifetime_steps_floor(
                    2.99, 1.0, lifetime, error) &&
                    !lifetime.infinite && lifetime.steps == 2u,
                "finite lifetime is floored and never exceeds the SI maximum");
    test.expect(culbm::particle_advection::lifetime_steps_floor(
                    0.9, 1.0, lifetime, error) &&
                    !lifetime.infinite && lifetime.steps == 0u,
                "positive sub-step lifetime maps to zero steps");
    test.expect(culbm::particle_advection::lifetime_steps_floor(
                    0.3, 0.1, lifetime, error) &&
                    !lifetime.infinite && lifetime.steps == 3u,
                "lifetime floor normalizes decimal representation roundoff");
    test.expect(!culbm::particle_advection::lifetime_steps_floor(
                    0.0, 1.0, lifetime, error) && contains(error, "> 0"),
                "explicit zero lifetime is invalid");

    const std::vector<ParticleSeed> particles = {
        {{{1.0, 2.0, 3.0}}, 0.49, 7u},
        {{{4.0, 5.0, 6.0}}, 1.5, 9u},
    };
    std::vector<QuantizedParticleSeed> quantized;
    test.expect(culbm::particle_advection::quantize_particle_releases(
                    particles, 1.0, quantized, error) &&
                    quantized.size() == 2u &&
                    quantized[0].release_step == 0u &&
                    quantized[1].release_step == 2u &&
                    quantized[1].source_line == 9u,
                "particle release quantization preserves order and metadata");
}

void test_automatic_schedule(TestState& test) {
    std::vector<ReleaseBatch> schedule;
    std::string error;
    test.expect(culbm::particle_advection::build_automatic_release_schedule(
                    0.0, 0.6, 1.0, 3u, schedule, error),
                "automatic schedule builds from absolute SI times");
    const std::vector<ReleaseBatch> expected = {
        {0u, 1u}, {1u, 2u}, {2u, 2u}, {3u, 1u},
    };
    bool matches = schedule.size() == expected.size();
    for (std::size_t i = 0u; matches && i < expected.size(); ++i) {
        matches = schedule[i].step == expected[i].step &&
            schedule[i].multiplicity == expected[i].multiplicity;
    }
    test.expect(matches,
                "absolute schedule avoids rounded-interval drift and merges equal steps");

    test.expect(culbm::particle_advection::build_automatic_release_schedule(
                    10.0, 1.0, 1.0, 3u, schedule, error) &&
                    schedule.empty(),
                "automatic start beyond the run produces an empty schedule");
    test.expect(culbm::particle_advection::build_automatic_release_schedule(
                    0.0, 1.0, 1.0, 1u, schedule, error, 2u) &&
                    schedule.size() == 2u,
                "nominal-release cap permits exactly the required event count");
    test.expect(!culbm::particle_advection::build_automatic_release_schedule(
                    0.0, 0.1, 1.0, 1u, schedule, error, 3u) &&
                    contains(error, "maximum_nominal_releases") &&
                    schedule.empty(),
                "automatic schedule cap prevents unbounded allocation");
    test.expect(!culbm::particle_advection::build_automatic_release_schedule(
                    0.0, 0.0, 1.0, 3u, schedule, error) &&
                    contains(error, "interval_s"),
                "automatic schedule rejects a zero interval");
    test.expect(!culbm::particle_advection::build_automatic_release_schedule(
                    0.0, 1.0, -1.0, 3u, schedule, error) &&
                    contains(error, "seconds_per_step"),
                "automatic schedule rejects invalid lattice time");
    test.expect(!culbm::particle_advection::build_automatic_release_schedule(
                    0.0, 1.0, 1.0, 3u, schedule, error, 0u) &&
                    contains(error, "> 0"),
                "automatic schedule rejects a zero safety cap");
}

} // namespace

int main() {
    TestState test;
    test_triplets(test);
    test_config_defaults_and_valid_modes(test);
    test_config_errors(test);
    test_particle_stream(test);
    test_particle_stream_errors_and_file(test);
    test_time_quantization(test);
    test_automatic_schedule(test);

    if (test.failures == 0) {
        std::cout << "Particle-advection config, file, and schedule tests passed.\n";
    }
    return test.failures == 0 ? 0 : 1;
}
