#include "../src/thermal_coupling_policy.hpp"

#include <iostream>
#include <string>

namespace {

struct TestState {
    int failures = 0;

    void expect(const bool condition, const std::string& message) {
        if (condition) return;
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
};

void test_policy(TestState& test) {
    const auto active = luw::cfd::resolve_thermal_coupling(true, true);
    test.expect(active.buoyancy_requested &&
                    active.thermal_boundary_available &&
                    active.buoyancy_active && !active.fallback_applied,
                "requested buoyancy stays active with a thermal boundary");

    const auto fallback = luw::cfd::resolve_thermal_coupling(true, false);
    test.expect(fallback.buoyancy_requested &&
                    !fallback.thermal_boundary_available &&
                    !fallback.buoyancy_active && fallback.fallback_applied,
                "requested buoyancy falls back off without a thermal boundary");

    const auto passive = luw::cfd::resolve_thermal_coupling(false, true);
    test.expect(!passive.buoyancy_requested &&
                    passive.thermal_boundary_available &&
                    !passive.buoyancy_active && !passive.fallback_applied,
                "an explicit off switch preserves passive temperature transport");

    const auto inactive = luw::cfd::resolve_thermal_coupling(false, false);
    test.expect(!inactive.buoyancy_active && !inactive.fallback_applied,
                "an explicit off switch needs no fallback without temperature data");
}

} // namespace

int main() {
    TestState test;
    test_policy(test);
    if (test.failures != 0) {
        std::cerr << test.failures << " thermal-coupling policy test(s) failed\n";
        return 1;
    }
    std::cout << "thermal-coupling policy tests passed\n";
    return 0;
}
