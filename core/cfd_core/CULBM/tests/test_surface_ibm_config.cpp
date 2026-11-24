#include "surface_ibm_config.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

using culbm::surface_ibm::Config;
using culbm::surface_ibm::DeltaKernel;
using culbm::surface_ibm::GeometryPolicy;
using culbm::surface_ibm::RawDeck;

struct TestState {
    int failures = 0;

    void expect(bool condition, const std::string& message) {
        if (condition) return;
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
};

bool close(float left, float right) {
    return std::fabs(static_cast<double>(left) - static_cast<double>(right)) <= 1.0e-7;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

void test_defaults_and_full_deck(TestState& test) {
    Config config;
    std::string error;
    test.expect(culbm::surface_ibm::parse_config({{"casename", "ignored"}}, config, error),
                "unrelated deck fields are ignored");
    test.expect(!config.enabled && config.delta_kernel == DeltaKernel::Roma3Point &&
                    config.geometry_policy == GeometryPolicy::ExteriorUnion &&
                    close(config.marker_spacing_lu, 0.75f) && config.forcing_iterations == 8 &&
                    close(config.support_thickness_lu, config.marker_spacing_lu),
                "stable disabled Roma-3/exterior-union defaults are preserved");

    const RawDeck raw = {
        {"IBM ENABLER", "yes"},
        {"culbm-surface-ibm-delta-kernel", "Roma-3-point"},
        {"culbm_surface_ibm_geometry_policy", "all_triangles"},
        {"culbm_surface_ibm_marker_spacing_lu", "0.5"},
        {"culbm_surface_ibm_forcing_iterations", "8"},
        {"culbm_surface_ibm_relaxation", "1.25"},
        {"culbm_surface_ibm_residual_tolerance", "2e-4"},
        {"culbm_surface_ibm_retraction_lu", "0.1"},
        {"culbm_surface_ibm_support_thickness_lu", "1.5"},
        {"culbm_surface_ibm_max_markers", "123456"},
        {"culbm_surface_ibm_strict_geometry", "true"},
        {"culbm_surface_ibm_diagnostics_interval", "25"},
    };
    test.expect(culbm::surface_ibm::parse_config(raw, config, error),
                "complete canonical surface-IBM deck parses");
    test.expect(config.enabled && config.delta_kernel == DeltaKernel::Roma3Point &&
                    config.geometry_policy == GeometryPolicy::AllTriangles &&
                    close(config.marker_spacing_lu, 0.5f) && config.forcing_iterations == 8 &&
                    close(config.relaxation, 1.25f) &&
                    close(config.residual_tolerance, 2.0e-4f) &&
                    close(config.retraction_lu, 0.1f) &&
                    close(config.support_thickness_lu, 1.5f) &&
                    config.max_markers == 123456 && config.strict_geometry &&
                    config.diagnostics_interval == 25,
                "all parsed settings retain their values");
    test.expect(std::string(culbm::surface_ibm::delta_kernel_name(config.delta_kernel)) ==
                    "roma3_point" &&
                    std::string(culbm::surface_ibm::geometry_policy_name(config.geometry_policy)) ==
                    "all_triangles",
                "enum names are canonical");

    test.expect(culbm::surface_ibm::parse_config({
                    {"ibm_enabler", "true"},
                    {"culbm_surface_ibm_marker_spacing_lu", "0.4"},
                }, config, error) && close(config.support_thickness_lu, 0.4f),
                "implicit support thickness follows marker spacing");

    Config legacy_false;
    test.expect(culbm::surface_ibm::parse_config(
                    {{"ibm_enabler", "n"}}, legacy_false, error) &&
                    !legacy_false.enabled,
                "legacy/common deck token n keeps SurfaceIBM disabled");
    Config legacy_true;
    test.expect(culbm::surface_ibm::parse_config(
                    {{"ibm_enabler", "enabled"}}, legacy_true, error) &&
                    legacy_true.enabled,
                "common deck token enabled activates SurfaceIBM");
}

void test_rejections(TestState& test) {
    Config config;
    std::string error;
    auto rejects = [&](const RawDeck& raw, const std::string& needle,
                       const std::string& message) {
        error.clear();
        test.expect(!culbm::surface_ibm::parse_config(raw, config, error) &&
                        contains(error, needle),
                    message + " (error was: " + error + ")");
    };

    rejects({{"culbm_surface_ibm_enabled", "true"}}, "unknown",
            "ibm_enabler remains the only master switch");
    rejects({{"culbm_surface_ibm_typo", "1"}}, "unknown",
            "unknown namespaced fields are rejected");
    rejects({{"ibm_enabler", "maybe"}}, "ibm_enabler",
            "invalid master boolean is rejected");
    rejects({{"culbm_surface_ibm_delta_kernel", "peskin4"}}, "unsupported",
            "unimplemented delta kernels are rejected");
    rejects({{"culbm_surface_ibm_geometry_policy", "guess"}}, "unsupported",
            "unknown geometry policies are rejected");
    rejects({{"culbm_surface_ibm_marker_spacing_lu", "0"}}, "> 0",
            "zero marker spacing is rejected");
    rejects({{"culbm_surface_ibm_forcing_iterations", "0"}}, "[1, 32]",
            "zero forcing iterations is rejected");
    rejects({{"culbm_surface_ibm_forcing_iterations", "33"}}, "[1, 32]",
            "forcing iterations above the supported bound are rejected");
    rejects({{"culbm_surface_ibm_relaxation", "2"}}, "(0, 2)",
            "the non-damping Richardson endpoint is rejected");
    rejects({{"culbm_surface_ibm_relaxation", "2.1"}}, "(0, 2)",
            "excess forcing relaxation is rejected");
    rejects({{"culbm_surface_ibm_residual_tolerance", "nan"}}, "invalid value",
            "non-finite tolerance is rejected");
    Config independent_geometry_and_quadrature;
    error.clear();
    test.expect(culbm::surface_ibm::parse_config({
                    {"culbm_surface_ibm_retraction_lu", "1"},
                    {"culbm_surface_ibm_support_thickness_lu", "0.5"}},
                    independent_geometry_and_quadrature, error),
                "geometric retraction is independent of quadrature thickness");
    rejects({{"culbm_surface_ibm_max_markers", "0"}}, "> 0",
            "zero marker capacity is rejected");
}

void test_published_snapshot(TestState& test) {
    Config config;
    config.enabled = true;
    config.forcing_iterations = 5;
    config.marker_spacing_lu = 0.6f;
    config.support_thickness_lu = 0.6f;
    culbm::surface_ibm::set_config(config);
    const Config copy = culbm::surface_ibm::get_config();
    test.expect(copy.enabled && copy.forcing_iterations == 5 &&
                    close(copy.marker_spacing_lu, 0.6f),
                "published configuration is returned as an independent snapshot");
}

}  // namespace

int main() {
    TestState test;
    test_defaults_and_full_deck(test);
    test_rejections(test);
    test_published_snapshot(test);
    if (test.failures != 0) {
        std::cerr << test.failures << " surface-IBM configuration test(s) failed\n";
        return 1;
    }
    std::cout << "surface-IBM configuration tests passed\n";
    return 0;
}
