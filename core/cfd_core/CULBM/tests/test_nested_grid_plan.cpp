#include "cuda_nested_grid_plan.hpp"
#include "cuda_nested_solver.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

bool close_enough(const double actual, const double expected) {
    return std::fabs(actual - expected) <= 1.0e-9;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

struct TestState {
    int failures = 0;

    void expect(const bool condition, const std::string& message) {
        if (condition) return;
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
};

void expect_physical_coverage(const culbm::VerticalNestingPlan& plan,
                              TestState& test,
                              const std::string& label) {
    double next_begin_si = plan.ground_top_si;
    for (const auto& level : plan.levels) {
        test.expect(close_enough(level.z_begin_si, next_begin_si),
                    label + ": adjacent physical slabs have no gap or overlap");
        test.expect(close_enough(
                        level.z_end_si - level.z_begin_si,
                        (double)level.physical_nz * level.dx_si),
                    label + ": physical_nz exactly covers its SI thickness");
        test.expect(level.lower_ghost_rows == 1u && level.upper_ghost_rows == 1u &&
                        level.allocated_nz == level.physical_nz + 2u,
                    label + ": ghost rows are excluded from physical thickness");
        test.expect(level.physical_first_z() == 1u &&
                        level.physical_end_z() == 1u + level.physical_nz &&
                        level.lower_ghost_z() == 0u &&
                        level.upper_ghost_z() == 1u + level.physical_nz,
                    label + ": physical and ghost z indices are explicit");
        next_begin_si = level.z_end_si;
    }
    test.expect(close_enough(next_begin_si,
                             plan.ground_top_si + plan.domain_height_agl_si),
                label + ": physical slabs cover the full configured SI height");
}

} // namespace

int main() {
    using culbm::VerticalNestingPlan;
    using culbm::is_internal_nested_ghost_row;
    using culbm::make_vertical_nesting_plan;
    using culbm::vertical_nesting_mean4;

    TestState test;

    const VerticalNestingPlan disabled = make_vertical_nesting_plan(
        {}, 10.0, 100.0, 5u, 7u);
    test.expect(!disabled.enabled, "empty threshold list disables nesting");
    test.expect(disabled.valid, "disabled nesting remains valid");
    test.expect(disabled.levels.empty(), "disabled nesting has no levels");

    // 5x7 intentionally is not divisible by two. It is valid because each
    // coarse cell still owns exactly a 2x2 fine patch; only GPU topology must
    // divide the actual leaf dimensions.
    const VerticalNestingPlan one = make_vertical_nesting_plan(
        {30.0}, 10.0, 100.0, 5u, 7u, 1u, 1u, 1u, 1000.0);
    test.expect(one.valid && one.enabled, "single fixed-height threshold is valid");
    test.expect(one.levels.size() == 2u, "single threshold creates two slabs");
    if (one.levels.size() == 2u) {
        const auto& fine = one.levels[0];
        const auto& coarse = one.levels[1];
        test.expect(fine.ratio_to_coarse == 2u, "lower one-threshold slab is 2x fine");
        test.expect(fine.nx == 10u && fine.ny == 14u,
                    "fine horizontal resolution is scaled without coarse divisibility rule");
        test.expect(fine.physical_nz == 6u && fine.allocated_nz == 8u,
                    "fine physical row count is thickness/dx plus two ghost rows");
        test.expect(coarse.physical_nz == 7u && coarse.allocated_nz == 9u,
                    "coarse physical row count is thickness/dx plus two ghost rows");
        test.expect(close_enough(fine.z_begin_agl_si, 0.0) &&
                    close_enough(fine.z_end_agl_si, 30.0) &&
                    close_enough(coarse.z_begin_agl_si, 30.0) &&
                    close_enough(coarse.z_end_agl_si, 100.0),
                    "adjacent slabs share exactly one fixed-height interface plane");
        test.expect(close_enough(fine.z_begin_si, 1000.0) &&
                    close_enough(fine.z_end_si, 1030.0),
                    "absolute planes include the compact-ground reference height");
    }
    expect_physical_coverage(one, test, "one-threshold plan");

    // Neither 6 m nor 18 m lies on the 8 m coarse lattice. They are still
    // valid because each slab aligns to its own dx: 2 m, 4 m and 8 m.
    const VerticalNestingPlan two = make_vertical_nesting_plan(
        {6.0, 18.0}, 8.0, 34.0, 3u, 5u, 1u, 1u, 1u);
    test.expect(two.valid && two.has_three_levels(),
                "two thresholds accept per-level, not coarse-grid, alignment");
    if (two.levels.size() == 3u) {
        const auto& fine = two.levels[0];
        const auto& middle = two.levels[1];
        const auto& coarse = two.levels[2];
        test.expect(fine.nx == 12u && fine.ny == 20u &&
                    fine.physical_nz == 3u && fine.allocated_nz == 5u,
                    "quarter-resolution slab dimensions are correct");
        test.expect(middle.nx == 6u && middle.ny == 10u &&
                    middle.physical_nz == 3u && middle.allocated_nz == 5u,
                    "half-resolution slab dimensions are correct");
        test.expect(coarse.nx == 3u && coarse.ny == 5u &&
                    coarse.physical_nz == 2u && coarse.allocated_nz == 4u,
                    "coarse slab dimensions are correct");
        test.expect(fine.physical_nz == 3u &&
                    middle.physical_nz == 3u &&
                    coarse.physical_nz == 2u,
                    "each slab validates only its own physical cell count");
    }
    expect_physical_coverage(two, test, "two-threshold plan");
    if (two.levels.size() == 3u) {
        const auto& fine = two.levels[0];
        const auto& middle = two.levels[1];
        const auto& coarse = two.levels[2];
        test.expect(
            !is_internal_nested_ghost_row(
                fine.lower_ghost_z(), fine.lower_ghost_z(),
                fine.upper_ghost_z(), true, false) &&
            is_internal_nested_ghost_row(
                fine.upper_ghost_z(), fine.lower_ghost_z(),
                fine.upper_ghost_z(), true, false),
            "finest level preserves global ground ownership and seals only its upper interface");
        test.expect(
            is_internal_nested_ghost_row(
                middle.lower_ghost_z(), middle.lower_ghost_z(),
                middle.upper_ghost_z(), false, false) &&
            is_internal_nested_ghost_row(
                middle.upper_ghost_z(), middle.lower_ghost_z(),
                middle.upper_ghost_z(), false, false),
            "middle level pre-seals both internal interface rows");
        test.expect(
            is_internal_nested_ghost_row(
                coarse.lower_ghost_z(), coarse.lower_ghost_z(),
                coarse.upper_ghost_z(), false, true) &&
            !is_internal_nested_ghost_row(
                coarse.upper_ghost_z(), coarse.lower_ghost_z(),
                coarse.upper_ghost_z(), false, true),
            "coarsest level seals only its lower interface and preserves the global sky boundary");
    }

    // User example: H is the coarsest SI grid scale.  (50,100) m is valid
    // for H=20 m because the three physical row counts are 10, 5 and 5 at
    // 5/10/20 m respectively; ghost rows must not alter those SI extents.
    const VerticalNestingPlan user_example = make_vertical_nesting_plan(
        {50.0, 100.0}, 20.0, 200.0, 4u, 6u, 1u, 1u, 1u, 123.0);
    test.expect(user_example.valid && user_example.has_three_levels(),
                "H=20 m with (50,100) m fixed interfaces is valid");
    if (user_example.levels.size() == 3u) {
        test.expect(user_example.levels[0].physical_nz == 10u &&
                    user_example.levels[1].physical_nz == 5u &&
                    user_example.levels[2].physical_nz == 5u,
                    "H=20 m example has exact 5/10/20 m physical cell counts");
        test.expect(close_enough(user_example.levels[0].z_begin_si, 123.0) &&
                    close_enough(user_example.levels[0].z_end_si, 173.0) &&
                    close_enough(user_example.levels[1].z_begin_si, 173.0) &&
                    close_enough(user_example.levels[1].z_end_si, 223.0) &&
                    close_enough(user_example.levels[2].z_begin_si, 223.0) &&
                    close_enough(user_example.levels[2].z_end_si, 323.0),
                    "H=20 m example has contiguous absolute SI slab coverage");
    }
    expect_physical_coverage(user_example, test, "H=20 m user example");

    // Arbitrary valid decomposition counts remain supported; this is not a
    // hidden one-GPU or power-of-two restriction.
    const VerticalNestingPlan many_domains = make_vertical_nesting_plan(
        {30.0}, 10.0, 100.0, 5u, 7u, 5u, 7u, 1u);
    test.expect(many_domains.valid,
                "valid non-power-of-two multi-GPU decomposition is accepted");

    const VerticalNestingPlan bad_z_decomposition = make_vertical_nesting_plan(
        {6.0, 18.0}, 8.0, 34.0, 3u, 5u, 1u, 1u, 2u);
    test.expect(!bad_z_decomposition.valid && contains(bad_z_decomposition.error, "Nz"),
                "planner rejects leaf Nz that LBM would otherwise truncate for Dz");

    const VerticalNestingPlan one_row_per_z_domain =
        make_vertical_nesting_plan(
            {10.0}, 10.0, 30.0, 4u, 4u, 1u, 1u, 4u);
    test.expect(
        !one_row_per_z_domain.valid &&
            contains(one_row_per_z_domain.error,
                     "at least two allocated z rows per Dz domain"),
        "planner rejects a z decomposition whose interface ghost and donor can land in separate domains");

    const VerticalNestingPlan bad_x_decomposition = make_vertical_nesting_plan(
        {30.0}, 10.0, 100.0, 5u, 7u, 2u, 1u, 1u);
    test.expect(!bad_x_decomposition.valid && contains(bad_x_decomposition.error, "Nx=") &&
                    contains(bad_x_decomposition.error, "Dx="),
                "planner rejects non-divisible horizontal GPU topology");

    const VerticalNestingPlan zero_domain_axis = make_vertical_nesting_plan(
        {30.0}, 10.0, 100.0, 5u, 7u, 1u, 0u, 1u);
    test.expect(!zero_domain_axis.valid && contains(zero_domain_axis.error, "positive"),
                "planner rejects a zero multi-GPU topology axis");

    const VerticalNestingPlan bad_fine_thickness = make_vertical_nesting_plan(
        {31.0}, 10.0, 100.0, 5u, 7u, 1u, 1u, 1u);
    test.expect(!bad_fine_thickness.valid && contains(bad_fine_thickness.error, "fine"),
                "fine slab must have an integer count of fine SI intervals");

    const VerticalNestingPlan bad_middle_thickness = make_vertical_nesting_plan(
        {6.0, 17.0}, 8.0, 34.0, 3u, 5u, 1u, 1u, 1u);
    test.expect(!bad_middle_thickness.valid && contains(bad_middle_thickness.error, "middle"),
                "middle slab must have an integer count of middle SI intervals");

    const VerticalNestingPlan too_thin_physical_slab = make_vertical_nesting_plan(
        {5.0}, 10.0, 15.0, 5u, 7u, 1u, 1u, 1u);
    test.expect(!too_thin_physical_slab.valid &&
                    contains(too_thin_physical_slab.error, "two physical lattice cells"),
                "a slab with one physical cell is rejected");

    const VerticalNestingPlan too_many = make_vertical_nesting_plan(
        {10.0, 20.0, 30.0}, 10.0, 100.0, 5u, 7u, 1u, 1u, 1u);
    test.expect(!too_many.valid && contains(too_many.error, "at most two"),
                "more than two thresholds is rejected");

    const VerticalNestingPlan unordered = make_vertical_nesting_plan(
        {18.0, 6.0}, 8.0, 34.0, 3u, 5u, 1u, 1u, 1u);
    test.expect(!unordered.valid && contains(unordered.error, "strictly ascending"),
                "multiple thresholds must be strictly ascending");

    test.expect(close_enough(vertical_nesting_mean4(1.0f, 2.0f, 3.0f, 4.0f), 2.5),
                "4-to-1 restriction is the arithmetic mean of four fine values");
    test.expect(close_enough(vertical_nesting_mean4(-2.0f, 2.0f, 6.0f, 10.0f), 4.0),
                "4-to-1 restriction handles signed components");

    if (test.failures == 0) {
        std::cout << "Nested vertical-grid planner tests passed.\n";
    }
    return test.failures == 0 ? 0 : 1;
}
