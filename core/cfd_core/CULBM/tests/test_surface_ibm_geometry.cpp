#include "surface_ibm_geometry.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using culbm::surface_ibm::Config;
using culbm::surface_ibm::GeometryPolicy;
using culbm::surface_ibm::GeometryReport;
using culbm::surface_ibm::Marker;

struct TestState {
    int failures = 0;

    void expect(bool condition, const std::string& message) {
        if (condition) return;
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
};

float3 point(float x, float y, float z) {
    float3 result{};
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

double dot3(const float3& a, const float3& b) {
    return static_cast<double>(a.x) * b.x +
           static_cast<double>(a.y) * b.y +
           static_cast<double>(a.z) * b.z;
}

double norm3(const float3& value) {
    return std::sqrt(dot3(value, value));
}

bool close(double left, double right, double tolerance = 1.0e-6) {
    return std::fabs(left - right) <= tolerance;
}

void assign_triangle(Mesh& mesh,
                     std::uint32_t index,
                     const float3& a,
                     const float3& b,
                     const float3& c) {
    mesh.p0[index] = a;
    mesh.p1[index] = b;
    mesh.p2[index] = c;
}

void append_cube(Mesh& mesh,
                 std::uint32_t offset,
                 float x0,
                 float y0,
                 float z0,
                 float x1,
                 float y1,
                 float z1) {
    const float3 v[8] = {
        point(x0, y0, z0), point(x1, y0, z0),
        point(x1, y1, z0), point(x0, y1, z0),
        point(x0, y0, z1), point(x1, y0, z1),
        point(x1, y1, z1), point(x0, y1, z1),
    };
    const std::uint32_t faces[12][3] = {
        {0, 2, 1}, {0, 3, 2},
        {4, 5, 6}, {4, 6, 7},
        {0, 1, 5}, {0, 5, 4},
        {3, 7, 6}, {3, 6, 2},
        {0, 4, 7}, {0, 7, 3},
        {1, 2, 6}, {1, 6, 5},
    };
    for (std::uint32_t i = 0; i < 12; ++i) {
        assign_triangle(mesh, offset + i, v[faces[i][0]], v[faces[i][1]], v[faces[i][2]]);
    }
}

Config all_triangle_config(float spacing = 0.3f) {
    Config config;
    config.enabled = true;
    config.geometry_policy = GeometryPolicy::AllTriangles;
    config.marker_spacing_lu = spacing;
    config.support_thickness_lu = spacing;
    return config;
}

void test_rotated_plane_area_normal_and_determinism(TestState& test) {
    const double angle = 0.61;
    const float cosine = static_cast<float>(std::cos(angle));
    const float sine = static_cast<float>(std::sin(angle));
    auto rotate = [&](float x, float y) {
        return point(x, y * cosine, y * sine);
    };
    Mesh mesh(2, point(0.5f, 0.5f, 0.0f));
    assign_triangle(mesh, 0, rotate(0, 0), rotate(1, 0), rotate(1, 1));
    assign_triangle(mesh, 1, rotate(0, 0), rotate(1, 1), rotate(0, 1));

    const Config config = all_triangle_config(0.3f);
    std::vector<Marker> first;
    std::vector<Marker> second;
    GeometryReport first_report;
    GeometryReport second_report;
    std::string error;
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, first, first_report, error),
                "rotated plane marker construction succeeds: " + error);
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, second, second_report, error),
                "repeat rotated-plane construction succeeds: " + error);
    test.expect(!first.empty() && first.size() == second.size() &&
                    first_report.markers == first.size(),
                "adaptive edge-bounded subdivision emits a stable non-empty marker set");
    test.expect(close(first_report.marker_area_lu2, 1.0, 2.0e-6) &&
                    close(first_report.retained_area_lu2, 1.0, 2.0e-6),
                "subtriangle areas conserve rotated square area");

    const float3 expected = point(0.0f, -sine, cosine);
    bool normals_ok = true;
    bool ids_ok = first.size() == second.size();
    for (std::size_t i = 0; i < first.size(); ++i) {
        normals_ok = normals_ok && close(norm3(first[i].normal), 1.0, 2.0e-6) &&
                     dot3(first[i].normal, expected) > 0.99999;
        if (i < second.size()) {
            ids_ok = ids_ok && first[i].global_id == i && second[i].global_id == i &&
                     first[i].source_triangle == second[i].source_triangle &&
                     first[i].position_lu.x == second[i].position_lu.x &&
                     first[i].position_lu.y == second[i].position_lu.y &&
                     first[i].position_lu.z == second[i].position_lu.z &&
                     first[i].area_lu2 == second[i].area_lu2;
        }
    }
    test.expect(normals_ok, "rotated-plane normals are unit length and match triangle winding");
    test.expect(ids_ok, "marker order, IDs, positions, sources, and areas are repeatable");
}

void test_degenerate_and_strict_open_geometry(TestState& test) {
    Mesh mesh(2, point(0, 0, 0));
    assign_triangle(mesh, 0, point(0, 0, 0), point(1, 0, 0), point(0, 1, 0));
    assign_triangle(mesh, 1, point(2, 2, 2), point(2, 2, 2), point(2, 2, 2));
    Config config = all_triangle_config(0.5f);
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error),
                "non-strict geometry skips a degenerate triangle: " + error);
    test.expect(report.degenerate_triangles == 1 && report.valid_triangles == 1 &&
                    report.open_edges == 3 && !markers.empty(),
                "degenerate and open-edge diagnostics are reported");

    config.strict_geometry = true;
    test.expect(!culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error) && markers.empty(),
                "strict geometry rejects degenerate/open input and clears output");
}

void test_closed_cube_exterior(TestState& test) {
    Mesh mesh(12, point(0.5f, 0.5f, 0.5f));
    append_cube(mesh, 0, 0, 0, 0, 1, 1, 1);
    Config config;
    config.enabled = true;
    config.marker_spacing_lu = 0.5f;
    config.support_thickness_lu = 0.5f;
    config.strict_geometry = true;
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error),
                "strict exterior-union cube succeeds: " + error);
    test.expect(report.connected_components == 1 && report.watertight_components == 1 &&
                    report.open_edges == 0 && report.nonmanifold_edges == 0 &&
                    report.ambiguous_patches == 0 && report.internal_patches == 0,
                "closed cube is recognized as one unambiguous watertight component");
    test.expect(close(report.marker_area_lu2, 6.0, 5.0e-6),
                "closed-cube marker areas conserve surface area");
    bool outward = true;
    const float3 center = point(0.5f, 0.5f, 0.5f);
    for (const Marker& marker : markers) {
        const float3 surface_position = point(
            marker.position_lu.x + marker.normal.x * config.retraction_lu,
            marker.position_lu.y + marker.normal.y * config.retraction_lu,
            marker.position_lu.z + marker.normal.z * config.retraction_lu);
        const float3 radial = point(surface_position.x - center.x,
                                    surface_position.y - center.y,
                                    surface_position.z - center.z);
        const double orientation = dot3(marker.normal, radial);
        outward = outward && orientation > 0.0;
    }
    test.expect(outward, "exterior-union classifier orients all cube normals out of the solid");
}

void test_overlapping_cube_union(TestState& test) {
    Mesh mesh(24, point(0.75f, 0.75f, 0.75f));
    append_cube(mesh, 0, 0, 0, 0, 1, 1, 1);
    append_cube(mesh, 12, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);
    Config config;
    config.enabled = true;
    config.marker_spacing_lu = 0.25f;
    config.support_thickness_lu = 0.25f;
    config.strict_geometry = true;
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    const bool built = culbm::surface_ibm::build_surface_ibm_geometry(
        mesh, config, markers, report, error);
    test.expect(built, "overlapping watertight cube union succeeds: " + error);
    test.expect(report.connected_components == 2 && report.watertight_components == 2,
                "overlapping cubes remain two independently watertight source components");
    test.expect(report.internal_patches > 0 && report.internal_triangles > 0,
                "buried portions of overlapping input surfaces are removed");
    test.expect(close(report.marker_area_lu2, 10.5, 2.0e-5),
                "exterior union preserves the analytic overlapping-cube union area");
}

void test_anisotropic_subdivision_avoids_skinny_triangle_explosion(TestState& test) {
    constexpr float length = 64.0f;
    constexpr float width = 0.125f;
    constexpr float spacing = 0.5f;
    Mesh mesh(1, point(length * 0.5f, width * 0.5f, 0.0f));
    assign_triangle(mesh, 0,
                    point(0, 0, 0), point(length, 0, 0), point(0, width, 0));
    Config config = all_triangle_config(spacing);
    // The former uniform n^2 construction needs 16,641 candidates here and
    // therefore cannot fit under this cap. The adaptive construction should.
    config.max_markers = 1024;
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    const bool built = culbm::surface_ibm::build_surface_ibm_geometry(
        mesh, config, markers, report, error);
    test.expect(built, "anisotropic skinny-triangle subdivision succeeds: " + error);

    const double longest = std::sqrt(
        static_cast<double>(length) * length + static_cast<double>(width) * width);
    const std::uint64_t old_divisions =
        static_cast<std::uint64_t>(std::ceil(longest / spacing));
    const std::uint64_t old_uniform_markers = old_divisions * old_divisions;
    std::cout << "surface-IBM skinny subdivision: adaptive=" << markers.size()
              << ", uniform_n2=" << old_uniform_markers << '\n';
    test.expect(!markers.empty() && old_uniform_markers >= markers.size() * 20ULL,
                "skinny-triangle marker count is at least 20x below uniform n^2 subdivision");
    test.expect(close(report.marker_area_lu2, 0.5 * length * width, 2.0e-6),
                "adaptive skinny-triangle patches conserve source area");

    const double maximum_area_for_bounded_edge =
        std::sqrt(3.0) * static_cast<double>(spacing) * spacing / 4.0;
    bool patch_area_bound = true;
    for (const Marker& marker : markers) {
        patch_area_bound = patch_area_bound &&
                           marker.area_lu2 <= maximum_area_for_bounded_edge + 1.0e-7;
    }
    test.expect(patch_area_bound,
                "every skinny-triangle patch respects the area bound implied by max edge <= spacing");

    Config capped = config;
    capped.max_markers = static_cast<std::uint64_t>(markers.size() - 1);
    std::vector<Marker> rejected_markers = markers;
    GeometryReport rejected_report;
    error.clear();
    test.expect(!culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, capped, rejected_markers, rejected_report, error) &&
                    rejected_markers.empty() &&
                    error.find("candidate patch count") != std::string::npos,
                "adaptive subdivision enforces max_markers and exposes no partial output");

    // A needle with the thin vertex near the middle of the longest edge has
    // two long, nearly anti-parallel barycentric directions. It exercises the
    // physical longest-axis path rather than the easier long/short-edge case.
    Mesh centered_needle(1, point(length * 0.5f, width * 0.5f, 0.0f));
    assign_triangle(centered_needle, 0,
                    point(0, 0, 0), point(length, 0, 0),
                    point(length * 0.5f, width, 0));
    markers.clear();
    report = GeometryReport{};
    error.clear();
    const bool centered_built = culbm::surface_ibm::build_surface_ibm_geometry(
        centered_needle, config, markers, report, error);
    const std::uint64_t centered_old_uniform =
        static_cast<std::uint64_t>(length / spacing) *
        static_cast<std::uint64_t>(length / spacing);
    std::cout << "surface-IBM centered-needle subdivision: adaptive=" << markers.size()
              << ", uniform_n2=" << centered_old_uniform << '\n';
    test.expect(centered_built && !markers.empty() &&
                    centered_old_uniform >= markers.size() * 20ULL,
                "centered needle avoids anti-parallel barycentric n^2 refinement: " + error);
    test.expect(close(report.marker_area_lu2, 0.5 * length * width, 2.0e-6),
                "centered-needle adaptive patches conserve source area");
}

void test_face_touching_cube_union(TestState& test) {
    Mesh mesh(24, point(1.0f, 0.5f, 0.5f));
    append_cube(mesh, 0, 0, 0, 0, 1, 1, 1);
    append_cube(mesh, 12, 1, 0, 0, 2, 1, 1);
    Config config;
    config.enabled = true;
    config.marker_spacing_lu = 0.25f;
    config.support_thickness_lu = 0.25f;
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    const bool built = culbm::surface_ibm::build_surface_ibm_geometry(
        mesh, config, markers, report, error);
    test.expect(built, "face-touching cube union succeeds: " + error);
    test.expect(report.duplicate_triangles == 2 &&
                    report.coincident_union_pairs == 2,
                "face-touching cube union cancels both shared-face facets");
    test.expect(close(report.marker_area_lu2, 10.0, 2.0e-5),
                "face-touching cube union preserves the analytic exterior area");

    config.strict_geometry = true;
    const bool strict_built = culbm::surface_ibm::build_surface_ibm_geometry(
        mesh, config, markers, report, error);
    test.expect(strict_built,
                "strict exterior union accepts a watertight touching-solid union: " + error);
}

void test_mismatched_touching_triangulation_is_rejected(TestState& test) {
    Mesh mesh(24, point(1.0f, 0.5f, 0.5f));
    append_cube(mesh, 0, 0, 0, 0, 1, 1, 1);
    append_cube(mesh, 12, 1, 0, 0, 2, 1, 1);
    // Retriangulate only the second cube's shared x=1 face with the opposite
    // diagonal. Exact coincident-facet cancellation can no longer repair this
    // common building/terrain T-junction pattern.
    assign_triangle(mesh, 20, point(1, 0, 0), point(1, 0, 1), point(1, 1, 0));
    assign_triangle(mesh, 21, point(1, 0, 1), point(1, 1, 1), point(1, 1, 0));

    Config config;
    config.enabled = true;
    config.marker_spacing_lu = 0.25f;
    config.support_thickness_lu = 0.25f;
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    test.expect(!culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error) &&
                    markers.empty() &&
                    error.find("watertight manifold") != std::string::npos,
                "exterior union rejects a mismatched touching-face triangulation instead of retaining an internal forcing sheet");

    config.geometry_policy = GeometryPolicy::AllTriangles;
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error) &&
                    report.nonmanifold_edges != 0 && !markers.empty(),
                "an intentional open/all-triangle sheet remains an explicit opt-in");
}

void test_marker_limit(TestState& test) {
    Mesh mesh(2, point(0.5f, 0.5f, 0.0f));
    assign_triangle(mesh, 0, point(0, 0, 0), point(1, 0, 0), point(1, 1, 0));
    assign_triangle(mesh, 1, point(0, 0, 0), point(1, 1, 0), point(0, 1, 0));
    Config config = all_triangle_config(0.3f);
    config.max_markers = 49;
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    test.expect(!culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error) && markers.empty(),
                "marker-capacity overflow fails without exposing partial marker output");
}

void test_duplicate_policy_and_body_identity(TestState& test) {
    Mesh mesh(2, point(0, 0, 0));
    assign_triangle(mesh, 0, point(0, 0, 0), point(1, 0, 0), point(0, 1, 0));
    assign_triangle(mesh, 1, point(0, 1, 0), point(1, 0, 0), point(0, 0, 0));
    Config config = all_triangle_config(0.5f);
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error) &&
                    report.duplicate_triangles == 1,
                "non-strict geometry deterministically collapses exact duplicate facets");
    bool neutral_body_ids = !markers.empty();
    for (const Marker& marker : markers) neutral_body_ids = neutral_body_ids && marker.body_id == 0;
    test.expect(neutral_body_ids,
                "markers use neutral body_id because Mesh carries no physical body metadata");

    config.strict_geometry = true;
    test.expect(!culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error) && markers.empty(),
                "strict geometry rejects duplicate facets instead of silently repairing them");
}

void test_interface_facet_ownership(TestState& test) {
    Mesh mesh(1, point(0, 0, 0));
    assign_triangle(mesh, 0, point(0, 0, 1), point(2, 0, 1), point(0, 1, 1));
    const Config config = all_triangle_config(0.5f);
    std::vector<Marker> markers;
    GeometryReport report;
    std::string error;
    const culbm::surface_ibm::GeometrySlab lower{0.0, 1.0, true};
    const culbm::surface_ibm::GeometrySlab upper{1.0, 2.0, true};
    const culbm::surface_ibm::GeometrySlab compact_wall{1.0, 2.0, false};
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error, &lower) && markers.empty(),
                "a roof on a slab's upper boundary is assigned to the next slab");
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error, &upper) &&
                    close(report.marker_area_lu2, 1.0, 1.0e-6),
                "the upper slab owns the full area of an interface roof");
    test.expect(culbm::surface_ibm::build_surface_ibm_geometry(
                    mesh, config, markers, report, error, &compact_wall) && markers.empty(),
                "the analytic ground excludes a coplanar IBM facet");
}

}  // namespace

int main() {
    TestState test;
    test_rotated_plane_area_normal_and_determinism(test);
    test_degenerate_and_strict_open_geometry(test);
    test_closed_cube_exterior(test);
    test_overlapping_cube_union(test);
    test_anisotropic_subdivision_avoids_skinny_triangle_explosion(test);
    test_face_touching_cube_union(test);
    test_mismatched_touching_triangulation_is_rejected(test);
    test_marker_limit(test);
    test_duplicate_policy_and_body_identity(test);
    test_interface_facet_ownership(test);
    if (test.failures != 0) {
        std::cerr << test.failures << " surface-IBM geometry test(s) failed\n";
        return 1;
    }
    std::cout << "surface-IBM geometry tests passed\n";
    return 0;
}
