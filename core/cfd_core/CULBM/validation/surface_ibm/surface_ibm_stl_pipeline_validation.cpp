#include "solver.hpp"
#include "surface_ibm.hpp"
#include "surface_ibm_geometry.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The production solver declares these setup-controlled terms externally.
// Validation keeps every optional forcing path disabled explicitly.
float coriolis_Omegax_lbmu = 0.0f;
float coriolis_Omegay_lbmu = 0.0f;
float coriolis_Omegaz_lbmu = 0.0f;
bool buffer_nudging_active = false;
int buffer_n_cells = 1;
float buffer_inv_tau_lbmu = 0.0f;
int buffer_nudge_vertical = 0;
int buffer_downstream_face_id = 0;
bool top_sponge_active = false;
int sponge_n_cells = 1;
float sponge_inv_tau_lbmu = 0.0f;
int sponge_ref_mode = 0;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr float kNu = 0.1f;
constexpr float kMarkerSpacing = 0.75f;
constexpr float kRetraction = 0.5f;
constexpr float kSupportThickness = 0.75f;
constexpr unsigned kForcingIterations = 32u;
constexpr float kBoundaryProbeSpacing = 0.25f;

#ifdef FP16C
constexpr const char* kPrecision = "FP16C";
#else
constexpr const char* kPrecision = "FP32";
#endif

enum class Shape {
    Circle,
    Square,
};

struct CaseSpec {
    std::string name;
    Shape shape = Shape::Circle;
    float half_width = 4.0f;
    float angle_deg = 0.0f;
    float phase_x = 0.23f;
    float phase_y = -0.17f;
    uint nz = 48u;
    float diffusion_times = 2.0f;
};

struct DuctGeometry {
    Shape shape = Shape::Circle;
    float half_width = 4.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
    float ex_x = 1.0f;
    float ex_y = 0.0f;
    float ey_x = 0.0f;
    float ey_y = 1.0f;

    void local(const float x, const float y, float& xi, float& eta) const {
        const float dx = x-center_x;
        const float dy = y-center_y;
        xi = dx*ex_x+dy*ex_y;
        eta = dx*ey_x+dy*ey_y;
    }

    float3 world(const float xi, const float eta, const float z) const {
        return float3(center_x+xi*ex_x+eta*ey_x,
                      center_y+xi*ex_y+eta*ey_y, z);
    }

    bool inside(const float x, const float y) const {
        float xi = 0.0f, eta = 0.0f;
        local(x, y, xi, eta);
        if(shape==Shape::Circle)
            return xi*xi+eta*eta<half_width*half_width;
        return std::fabs(xi)<half_width && std::fabs(eta)<half_width;
    }

    double signed_distance(const double x, const double y) const {
        const double dx = x-(double)center_x;
        const double dy = y-(double)center_y;
        const double xi = dx*(double)ex_x+dy*(double)ex_y;
        const double eta = dx*(double)ey_x+dy*(double)ey_y;
        if(shape==Shape::Circle)
            return std::sqrt(xi*xi+eta*eta)-(double)half_width;
        const double ax = std::fabs(xi), ay = std::fabs(eta);
        const double ox = std::max(ax-(double)half_width, 0.0);
        const double oy = std::max(ay-(double)half_width, 0.0);
        if(ox>0.0 || oy>0.0) return std::sqrt(ox*ox+oy*oy);
        return -std::min((double)half_width-ax,
                         (double)half_width-ay);
    }
};

struct AnalyticField {
    DuctGeometry geometry;
    double body_force = 0.0;
    double target_umax = 0.0;
};

struct MeshPipeline {
    std::unique_ptr<Mesh> mesh;
    std::vector<culbm::surface_ibm::Marker> forcing_markers;
    std::vector<culbm::surface_ibm::Marker> design_markers;
    culbm::surface_ibm::GeometryReport report;
    double analytic_area = 0.0;
    double mesh_area_relative_error = 0.0;
    double analytic_area_relative_error = 0.0;
    double surface_distance_rms = 0.0;
    double surface_distance_max = 0.0;
    double inward_normal_min_cosine = 1.0;
    double forcing_signed_distance_min =
        std::numeric_limits<double>::infinity();
    double forcing_signed_distance_max =
        -std::numeric_limits<double>::infinity();
    double retraction_projection_error_max = 0.0;
};

struct PhysicalMetrics {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    double flux_numeric = 0.0;
    double flux_exact_discrete = 0.0;
    double flux_relative_error = 0.0;
    double cross_velocity_rms = 0.0;
    double marker_slip_rms = 0.0;
    double marker_slip_max = 0.0;
    double marker_normal_rms = 0.0;
    double marker_normal_max = 0.0;
    double central_flux_cv = 0.0;
    double steady_flux_change = 0.0;
    double fluid_mass_relative_drift = 0.0;
    double rho_min = std::numeric_limits<double>::infinity();
    double rho_max = -std::numeric_limits<double>::infinity();
    double mach_max = 0.0;
    std::uint64_t sample_points = 0u;
};

struct RunResult {
    bool ibm = false;
    uint nx = 0u;
    uint ny = 0u;
    uint nz = 0u;
    unsigned steps = 0u;
    double runtime_seconds = 0.0;
    double mlups = 0.0;
    double checkpoint_flux = 0.0;
    double reaction_z_mean = 0.0;
    double reaction_z_cv = 0.0;
    PhysicalMetrics metrics;
    culbm::surface_ibm::Statistics statistics;
};

float3 subtract3(const float3& a, const float3& b) {
    return float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

float3 cross3(const float3& a, const float3& b) {
    return float3(a.y*b.z-a.z*b.y,
                  a.z*b.x-a.x*b.z,
                  a.x*b.y-a.y*b.x);
}

double dot3(const float3& a, const float3& b) {
    return (double)a.x*b.x+(double)a.y*b.y+(double)a.z*b.z;
}

double norm3(const float3& a) {
    return std::sqrt(dot3(a, a));
}

void assign_triangle(Mesh& mesh, const std::uint32_t index,
                     const float3& a, const float3& b, const float3& c) {
    mesh.p0[index] = a;
    mesh.p1[index] = b;
    mesh.p2[index] = c;
}

void append_oriented_quad(Mesh& mesh, std::uint32_t& index,
                          const float3& q00, const float3& q10,
                          const float3& q11, const float3& q01,
                          const float3& desired_normal) {
    const float3 normal = cross3(subtract3(q10, q00), subtract3(q11, q00));
    if(dot3(normal, desired_normal)>0.0) {
        assign_triangle(mesh, index++, q00, q10, q11);
        assign_triangle(mesh, index++, q00, q11, q01);
    } else {
        assign_triangle(mesh, index++, q00, q11, q10);
        assign_triangle(mesh, index++, q00, q01, q11);
    }
}

double square_velocity(const double xi, const double eta,
                       const double a, const double force,
                       const int maximum_odd = 63) {
    const double width = 2.0*a;
    const double x = xi+a;
    const double y = eta+a;
    double sum = 0.0;
    for(int m=1; m<=maximum_odd; m+=2) {
        const double sm = std::sin((double)m*kPi*x/width);
        for(int n=1; n<=maximum_odd; n+=2) {
            const double sn = std::sin((double)n*kPi*y/width);
            const double denominator = (double)m*(double)n*
                (((double)m/width)*((double)m/width)+
                 ((double)n/width)*((double)n/width));
            sum += sm*sn/denominator;
        }
    }
    return 16.0*force*sum/((double)kNu*std::pow(kPi, 4.0));
}

AnalyticField make_field(const CaseSpec& spec, const uint nx,
                         const uint ny) {
    AnalyticField field;
    field.geometry.shape = spec.shape;
    field.geometry.half_width = spec.half_width;
    field.geometry.center_x = 0.5f*(float)(nx-1u)+spec.phase_x;
    field.geometry.center_y = 0.5f*(float)(ny-1u)+spec.phase_y;
    const float angle = spec.angle_deg*(float)(kPi/180.0);
    field.geometry.ex_x = std::cos(angle);
    field.geometry.ex_y = std::sin(angle);
    field.geometry.ey_x = -field.geometry.ex_y;
    field.geometry.ey_y = field.geometry.ex_x;
    field.target_umax = 0.08/(double)spec.half_width;
    if(spec.shape==Shape::Circle) {
        field.body_force = 4.0*(double)kNu*field.target_umax/
            ((double)spec.half_width*(double)spec.half_width);
    } else {
        const double unit_center = square_velocity(
            0.0, 0.0, (double)spec.half_width, 1.0, 127);
        field.body_force = field.target_umax/unit_center;
    }
    return field;
}

double exact_velocity(const AnalyticField& field, const float x,
                      const float y) {
    float xi = 0.0f, eta = 0.0f;
    field.geometry.local(x, y, xi, eta);
    if(field.geometry.shape==Shape::Circle) {
        const double r2 = (double)xi*xi+(double)eta*eta;
        return field.body_force*
            ((double)field.geometry.half_width*field.geometry.half_width-r2)/
            (4.0*(double)kNu);
    }
    return square_velocity((double)xi, (double)eta,
                           (double)field.geometry.half_width,
                           field.body_force, 127);
}

std::unique_ptr<Mesh> make_triangle_mesh(const CaseSpec& spec,
                                         const AnalyticField& field) {
    constexpr float facet_span = 1.5f;
    const float z0 = 2.5f;
    const float z1 = (float)spec.nz-3.5f;
    const unsigned nz_panels = std::max(
        1u, (unsigned)std::ceil((z1-z0)/facet_span));
    const float dz = (z1-z0)/(float)nz_panels;

    if(spec.shape==Shape::Circle) {
        const unsigned circumferential_panels = std::max(
            16u, (unsigned)std::ceil(
                2.0f*(float)kPi*spec.half_width/facet_span));
        const std::uint32_t triangle_count =
            2u*circumferential_panels*nz_panels;
        auto mesh = std::make_unique<Mesh>(
            triangle_count,
            float3(field.geometry.center_x, field.geometry.center_y,
                   0.5f*(z0+z1)));
        std::vector<float3> ring(circumferential_panels+1u);
        for(unsigned i=0u; i<=circumferential_panels; ++i) {
            const float angle = 2.0f*(float)kPi*(float)i/
                                (float)circumferential_panels;
            ring[i] = float3(spec.half_width*std::cos(angle),
                             spec.half_width*std::sin(angle), 0.0f);
        }
        ring.back() = ring.front();
        std::uint32_t index = 0u;
        for(unsigned k=0u; k<nz_panels; ++k) {
            const float za = z0+(float)k*dz;
            const float zb = z0+(float)(k+1u)*dz;
            for(unsigned i=0u; i<circumferential_panels; ++i) {
                const float3 q00 = field.geometry.world(
                    ring[i].x, ring[i].y, za);
                const float3 q10 = field.geometry.world(
                    ring[i+1u].x, ring[i+1u].y, za);
                const float3 q11 = field.geometry.world(
                    ring[i+1u].x, ring[i+1u].y, zb);
                const float3 q01 = field.geometry.world(
                    ring[i].x, ring[i].y, zb);
                const float midpoint = 2.0f*(float)kPi*((float)i+0.5f)/
                                       (float)circumferential_panels;
                const float3 inward(-std::cos(midpoint),
                                    -std::sin(midpoint), 0.0f);
                append_oriented_quad(*mesh, index, q00, q10, q11, q01,
                                     inward);
            }
        }
        if(index!=triangle_count)
            throw std::runtime_error("circular STL-equivalent mesh size mismatch");
        mesh->find_bounds();
        return mesh;
    }

    const unsigned wall_panels = std::max(
        2u, (unsigned)std::ceil(2.0f*spec.half_width/facet_span));
    const std::uint32_t triangle_count = 8u*wall_panels*nz_panels;
    auto mesh = std::make_unique<Mesh>(
        triangle_count,
        float3(field.geometry.center_x, field.geometry.center_y,
               0.5f*(z0+z1)));
    const float ds = 2.0f*spec.half_width/(float)wall_panels;
    std::uint32_t index = 0u;
    const auto add_xi_wall = [&](const float fixed, const float3& inward) {
        for(unsigned k=0u; k<nz_panels; ++k) {
            const float za = z0+(float)k*dz;
            const float zb = z0+(float)(k+1u)*dz;
            for(unsigned i=0u; i<wall_panels; ++i) {
                const float sa = -spec.half_width+(float)i*ds;
                const float sb = -spec.half_width+(float)(i+1u)*ds;
                append_oriented_quad(
                    *mesh, index,
                    field.geometry.world(fixed, sa, za),
                    field.geometry.world(fixed, sb, za),
                    field.geometry.world(fixed, sb, zb),
                    field.geometry.world(fixed, sa, zb), inward);
            }
        }
    };
    const auto add_eta_wall = [&](const float fixed, const float3& inward) {
        for(unsigned k=0u; k<nz_panels; ++k) {
            const float za = z0+(float)k*dz;
            const float zb = z0+(float)(k+1u)*dz;
            for(unsigned i=0u; i<wall_panels; ++i) {
                const float sa = -spec.half_width+(float)i*ds;
                const float sb = -spec.half_width+(float)(i+1u)*ds;
                append_oriented_quad(
                    *mesh, index,
                    field.geometry.world(sa, fixed, za),
                    field.geometry.world(sb, fixed, za),
                    field.geometry.world(sb, fixed, zb),
                    field.geometry.world(sa, fixed, zb), inward);
            }
        }
    };
    add_xi_wall(spec.half_width,
                float3(-field.geometry.ex_x, -field.geometry.ex_y, 0.0f));
    add_xi_wall(-spec.half_width,
                float3(field.geometry.ex_x, field.geometry.ex_y, 0.0f));
    add_eta_wall(spec.half_width,
                 float3(-field.geometry.ey_x, -field.geometry.ey_y, 0.0f));
    add_eta_wall(-spec.half_width,
                 float3(field.geometry.ey_x, field.geometry.ey_y, 0.0f));
    if(index!=triangle_count)
        throw std::runtime_error("square STL-equivalent mesh size mismatch");
    mesh->find_bounds();
    return mesh;
}

void write_binary_stl(const std::filesystem::path& path, const Mesh& mesh) {
    std::ofstream output(path, std::ios::out|std::ios::binary|std::ios::trunc);
    if(!output) throw std::runtime_error("cannot create validation STL input");
    std::array<char, 80u> header{};
    const std::string label = "CULBM SurfaceIBM validation geometry";
    std::copy(label.begin(), label.end(), header.begin());
    output.write(header.data(), (std::streamsize)header.size());
    const std::uint32_t triangle_count = mesh.triangle_number;
    output.write(reinterpret_cast<const char*>(&triangle_count),
                 sizeof(triangle_count));
    for(std::uint32_t i=0u; i<triangle_count; ++i) {
        const float3 edge_1 = mesh.p1[i]-mesh.p0[i];
        const float3 edge_2 = mesh.p2[i]-mesh.p0[i];
        const float3 cross_value(
            edge_1.y*edge_2.z-edge_1.z*edge_2.y,
            edge_1.z*edge_2.x-edge_1.x*edge_2.z,
            edge_1.x*edge_2.y-edge_1.y*edge_2.x);
        const float magnitude = std::sqrt(
            cross_value.x*cross_value.x+cross_value.y*cross_value.y+
            cross_value.z*cross_value.z);
        const float3 normal = magnitude>0.0f
            ? cross_value/magnitude : float3(0.0f);
        const float record[12] = {
            normal.x, normal.y, normal.z,
            mesh.p0[i].x, mesh.p0[i].y, mesh.p0[i].z,
            mesh.p1[i].x, mesh.p1[i].y, mesh.p1[i].z,
            mesh.p2[i].x, mesh.p2[i].y, mesh.p2[i].z};
        const std::uint16_t attribute = 0u;
        output.write(reinterpret_cast<const char*>(record), sizeof(record));
        output.write(reinterpret_cast<const char*>(&attribute),
                     sizeof(attribute));
    }
    if(!output) throw std::runtime_error("cannot finish validation STL input");
}

culbm::surface_ibm::Config pipeline_config(const bool enabled,
                                           const double target_umax) {
    culbm::surface_ibm::Config config;
    config.enabled = enabled;
    config.geometry_policy = culbm::surface_ibm::GeometryPolicy::AllTriangles;
    config.marker_spacing_lu = kMarkerSpacing;
    config.forcing_iterations = kForcingIterations;
    config.relaxation = 1.0f;
    config.residual_tolerance = (float)std::max(1.0e-8,
                                                1.0e-2*target_umax);
    config.retraction_lu = kRetraction;
    config.support_thickness_lu = kSupportThickness;
    config.max_markers = 2000000u;
    // A duct without inlet/outlet caps is intentionally an open sheet.  This
    // is the documented AllTriangles use case, so topology is reported but
    // open edges are not promoted to a construction failure.
    config.strict_geometry = false;
    return config;
}

MeshPipeline build_pipeline(const CaseSpec& spec,
                            const AnalyticField& field,
                            const std::filesystem::path& input_directory) {
    MeshPipeline result;
    const auto generated = make_triangle_mesh(spec, field);
    std::filesystem::create_directories(input_directory);
    const std::filesystem::path stl_path = input_directory/(spec.name+".stl");
    write_binary_stl(stl_path, *generated);
    result.mesh = std::unique_ptr<Mesh>(read_stl(stl_path.string()));
    if(!result.mesh || result.mesh->triangle_number!=generated->triangle_number)
        throw std::runtime_error("binary STL round-trip changed triangle count");
    constexpr float roundtrip_tolerance = 1.0e-6f;
    const auto vertex_error = [](const float3& a, const float3& b) {
        return std::max({std::fabs(a.x-b.x), std::fabs(a.y-b.y),
                         std::fabs(a.z-b.z)});
    };
    float maximum_vertex_error = 0.0f;
    for(unsigned i=0u; i<generated->triangle_number; ++i) {
        maximum_vertex_error = std::max({
            maximum_vertex_error,
            vertex_error(result.mesh->p0[i], generated->p0[i]),
            vertex_error(result.mesh->p1[i], generated->p1[i]),
            vertex_error(result.mesh->p2[i], generated->p2[i])});
    }
    result.mesh->find_bounds();
    const float maximum_bounds_error = std::max(
        vertex_error(result.mesh->pmin, generated->pmin),
        vertex_error(result.mesh->pmax, generated->pmax));
    if(maximum_vertex_error>roundtrip_tolerance ||
       maximum_bounds_error>roundtrip_tolerance)
        throw std::runtime_error(
            "binary STL round-trip changed vertices, winding, or bounds");
    const auto config = pipeline_config(true, field.target_umax);
    std::string error;
    if(!culbm::surface_ibm::validate_config(config, error))
        throw std::runtime_error("invalid frozen pipeline configuration: "+error);
    if(!culbm::surface_ibm::build_surface_ibm_geometry(
            *result.mesh, config, result.forcing_markers,
            result.report, error))
        throw std::runtime_error("Mesh-to-marker construction failed: "+error);
    if(result.forcing_markers.empty())
        throw std::runtime_error("Mesh-to-marker construction returned no markers");

    result.design_markers = result.forcing_markers;
    double distance2 = 0.0;
    const float z_length = (float)spec.nz-6.0f;
    result.analytic_area = spec.shape==Shape::Circle
        ? 2.0*kPi*(double)spec.half_width*(double)z_length
        : 8.0*(double)spec.half_width*(double)z_length;
    for(std::size_t marker_index=0u;
        marker_index<result.design_markers.size(); ++marker_index) {
        auto& marker = result.design_markers[marker_index];
        const auto& forcing_marker = result.forcing_markers[marker_index];
        const double forcing_distance = field.geometry.signed_distance(
            forcing_marker.position_lu.x, forcing_marker.position_lu.y);
        result.forcing_signed_distance_min = std::min(
            result.forcing_signed_distance_min, forcing_distance);
        result.forcing_signed_distance_max = std::max(
            result.forcing_signed_distance_max, forcing_distance);
        marker.position_lu.x += kRetraction*marker.normal.x;
        marker.position_lu.y += kRetraction*marker.normal.y;
        marker.position_lu.z += kRetraction*marker.normal.z;
        const float3 displacement = subtract3(
            marker.position_lu, forcing_marker.position_lu);
        result.retraction_projection_error_max = std::max(
            result.retraction_projection_error_max,
            std::fabs(dot3(displacement, marker.normal)-kRetraction));
        const double distance = std::fabs(field.geometry.signed_distance(
            marker.position_lu.x, marker.position_lu.y));
        distance2 += distance*distance;
        result.surface_distance_max = std::max(
            result.surface_distance_max, distance);
        float3 expected_inward;
        if(spec.shape==Shape::Circle) {
            expected_inward = float3(
                field.geometry.center_x-marker.position_lu.x,
                field.geometry.center_y-marker.position_lu.y, 0.0f);
        } else {
            float xi = 0.0f, eta = 0.0f;
            field.geometry.local(marker.position_lu.x,
                                 marker.position_lu.y, xi, eta);
            if(std::fabs(std::fabs(xi)-spec.half_width)<=
               std::fabs(std::fabs(eta)-spec.half_width)) {
                const float sign = xi>=0.0f ? -1.0f : 1.0f;
                expected_inward = float3(sign*field.geometry.ex_x,
                                         sign*field.geometry.ex_y, 0.0f);
            } else {
                const float sign = eta>=0.0f ? -1.0f : 1.0f;
                expected_inward = float3(sign*field.geometry.ey_x,
                                         sign*field.geometry.ey_y, 0.0f);
            }
        }
        const double denominator = norm3(expected_inward)*norm3(marker.normal);
        if(denominator>0.0) {
            result.inward_normal_min_cosine = std::min(
                result.inward_normal_min_cosine,
                dot3(expected_inward, marker.normal)/denominator);
        }
    }
    result.surface_distance_rms = std::sqrt(
        distance2/(double)result.design_markers.size());
    result.mesh_area_relative_error = result.report.marker_area_lu2/
        result.report.retained_area_lu2-1.0;
    result.analytic_area_relative_error = result.report.marker_area_lu2/
        result.analytic_area-1.0;
    if(result.report.degenerate_triangles!=0u ||
       result.report.duplicate_triangles!=0u ||
       std::fabs(result.mesh_area_relative_error)>1.0e-6 ||
       result.inward_normal_min_cosine<0.95 ||
       result.forcing_signed_distance_min<=0.25 ||
       result.retraction_projection_error_max>1.0e-5)
        throw std::runtime_error("Mesh-to-marker geometry invariants failed");
    return result;
}

float roma3_phi(const float r) {
    const float a = std::fabs(r);
    if(a>=1.5f) return 0.0f;
    if(a<=0.5f)
        return (1.0f+std::sqrt(std::max(0.0f, 1.0f-3.0f*a*a)))/3.0f;
    const float q = 1.0f-a;
    return (5.0f-3.0f*a-
            std::sqrt(std::max(0.0f, 1.0f-3.0f*q*q)))/6.0f;
}

float3 interpolate_velocity(culbm::solver::Solver& lbm,
                            const culbm::surface_ibm::Marker& marker) {
    const int cx = (int)std::floor(marker.position_lu.x+0.5f);
    const int cy = (int)std::floor(marker.position_lu.y+0.5f);
    const int cz = (int)std::floor(marker.position_lu.z+0.5f);
    float3 value(0.0f);
    float weight = 0.0f;
    for(int dz=-1; dz<=1; ++dz) {
        const float wz = roma3_phi(marker.position_lu.z-(float)(cz+dz));
        for(int dy=-1; dy<=1; ++dy) {
            const float wy = roma3_phi(marker.position_lu.y-(float)(cy+dy));
            for(int dx=-1; dx<=1; ++dx) {
                const float w = roma3_phi(
                    marker.position_lu.x-(float)(cx+dx))*wy*wz;
                if(w==0.0f) continue;
                const int xx = cx+dx, yy = cy+dy, zz = cz+dz;
                if(xx<0 || yy<0 || zz<0 ||
                   xx>=(int)lbm.grid().size(0u) || yy>=(int)lbm.grid().size(1u) ||
                   zz>=(int)lbm.grid().size(2u))
                    throw std::runtime_error("design-marker support left validation domain");
                const ulong n = lbm.grid().index((uint)xx, (uint)yy, (uint)zz);
                value.x += w*lbm.velocity().x[n];
                value.y += w*lbm.velocity().y[n];
                value.z += w*lbm.velocity().z[n];
                weight += w;
            }
        }
    }
    if(!(weight>0.0f))
        throw std::runtime_error("zero design-marker interpolation weight");
    return value/weight;
}

void write_boundary_probes(
    culbm::solver::Solver& lbm, const AnalyticField& field, std::ofstream& probes,
    const CaseSpec& spec, const bool ibm) {
    std::vector<std::pair<float, float>> local_points;
    if(spec.shape==Shape::Circle) {
        const unsigned count = std::max(
            64u, (unsigned)std::ceil(
                2.0f*(float)kPi*spec.half_width/kBoundaryProbeSpacing));
        local_points.reserve(count);
        for(unsigned i=0u; i<count; ++i) {
            const float angle = 2.0f*(float)kPi*(float)i/(float)count;
            local_points.emplace_back(
                spec.half_width*std::cos(angle),
                spec.half_width*std::sin(angle));
        }
    } else {
        const unsigned segments = std::max(
            2u, (unsigned)std::ceil(
                2.0f*spec.half_width/kBoundaryProbeSpacing));
        local_points.reserve(4u*segments);
        for(unsigned i=0u; i<segments; ++i) {
            const float s = -spec.half_width+
                2.0f*spec.half_width*(float)i/(float)segments;
            local_points.emplace_back(spec.half_width, s);
            local_points.emplace_back(-s, spec.half_width);
            local_points.emplace_back(-spec.half_width, -s);
            local_points.emplace_back(s, -spec.half_width);
        }
    }

    const uint z0 = lbm.grid().size(2u)/4u;
    const uint z1 = 3u*lbm.grid().size(2u)/4u;
    for(std::size_t probe_index=0u;
        probe_index<local_points.size(); ++probe_index) {
        const float xi = local_points[probe_index].first;
        const float eta = local_points[probe_index].second;
        const float3 position = field.geometry.world(xi, eta, 0.0f);
        double ux_mean = 0.0, uy_mean = 0.0, uz_mean = 0.0;
        for(uint z=z0; z<z1; ++z) {
            culbm::surface_ibm::Marker probe;
            probe.position_lu = float3(position.x, position.y, (float)z);
            const float3 velocity = interpolate_velocity(lbm, probe);
            ux_mean += (double)velocity.x;
            uy_mean += (double)velocity.y;
            uz_mean += (double)velocity.z;
        }
        const double inverse_z = 1.0/(double)(z1-z0);
        ux_mean *= inverse_z;
        uy_mean *= inverse_z;
        uz_mean *= inverse_z;
        const double exact = exact_velocity(field, position.x, position.y);
        probes << spec.name << ','
               << (spec.shape==Shape::Circle ? "circle" : "square") << ','
               << (ibm ? "mesh_ibm" : "voxel") << ',' << kPrecision << ','
               << probe_index << ',' << position.x << ',' << position.y << ','
               << xi << ',' << eta << ',' << exact/field.target_umax << ','
               << uz_mean/field.target_umax << ','
               << ux_mean/field.target_umax << ','
               << uy_mean/field.target_umax << '\n';
    }
}

double section_flux(culbm::solver::Solver& lbm, const AnalyticField& field,
                    const uint z) {
    double flux = 0.0;
    for(uint y=0u; y<lbm.grid().size(1u); ++y)
        for(uint x=0u; x<lbm.grid().size(0u); ++x)
            if(field.geometry.inside((float)x, (float)y))
                flux += (double)lbm.velocity().z[lbm.grid().index(x, y, z)];
    return flux;
}

double central_mean_flux(culbm::solver::Solver& lbm, const AnalyticField& field) {
    const uint z0 = lbm.grid().size(2u)/4u;
    const uint z1 = 3u*lbm.grid().size(2u)/4u;
    double flux = 0.0;
    for(uint z=z0; z<z1; ++z) flux += section_flux(lbm, field, z);
    return flux/(double)(z1-z0);
}

double active_mass(culbm::solver::Solver& lbm,
                   const std::vector<unsigned char>& active) {
    double mass = 0.0;
    for(ulong n=0ull; n<lbm.grid().cells(); ++n)
        if(active[(std::size_t)n]!=0u) mass += (double)lbm.density()[n];
    return mass;
}

PhysicalMetrics compute_metrics(
    culbm::solver::Solver& lbm, const AnalyticField& field,
    const std::vector<culbm::surface_ibm::Marker>& design_markers,
    const std::vector<unsigned char>& active, const double initial_mass,
    const double checkpoint_flux, std::ofstream& profiles,
    const CaseSpec& spec, const bool ibm) {
    lbm.density().download();
    lbm.velocity().download();
    PhysicalMetrics metrics;
    const uint z0 = lbm.grid().size(2u)/4u;
    const uint z1 = 3u*lbm.grid().size(2u)/4u;
    double absolute_error = 0.0, absolute_exact = 0.0;
    double error2 = 0.0, exact2 = 0.0, cross2 = 0.0;
    std::vector<double> fluxes;
    fluxes.reserve(z1-z0);
    for(uint z=z0; z<z1; ++z) fluxes.push_back(section_flux(lbm, field, z));
    metrics.flux_numeric = std::accumulate(
        fluxes.begin(), fluxes.end(), 0.0)/(double)fluxes.size();
    double variance = 0.0;
    for(const double flux : fluxes) {
        const double delta = flux-metrics.flux_numeric;
        variance += delta*delta;
    }
    metrics.central_flux_cv = std::sqrt(variance/(double)fluxes.size())/
        std::max(std::fabs(metrics.flux_numeric), 1.0e-30);

    for(uint y=0u; y<lbm.grid().size(1u); ++y) {
        for(uint x=0u; x<lbm.grid().size(0u); ++x) {
            if(!field.geometry.inside((float)x, (float)y)) continue;
            const double exact = exact_velocity(field, (float)x, (float)y);
            metrics.flux_exact_discrete += exact;
            double ux_mean = 0.0, uy_mean = 0.0, uz_mean = 0.0;
            for(uint z=z0; z<z1; ++z) {
                const ulong n = lbm.grid().index(x, y, z);
                const double ux = (double)lbm.velocity().x[n];
                const double uy = (double)lbm.velocity().y[n];
                const double uz = (double)lbm.velocity().z[n];
                const double error = uz-exact;
                absolute_error += std::fabs(error);
                absolute_exact += std::fabs(exact);
                error2 += error*error;
                exact2 += exact*exact;
                cross2 += ux*ux+uy*uy;
                metrics.linf = std::max(
                    metrics.linf, std::fabs(error)/field.target_umax);
                metrics.mach_max = std::max(
                    metrics.mach_max,
                    std::sqrt(ux*ux+uy*uy+uz*uz)*std::sqrt(3.0));
                ux_mean += ux;
                uy_mean += uy;
                uz_mean += uz;
                ++metrics.sample_points;
            }
            const double inverse_z = 1.0/(double)(z1-z0);
            ux_mean *= inverse_z;
            uy_mean *= inverse_z;
            uz_mean *= inverse_z;
            float xi = 0.0f, eta = 0.0f;
            field.geometry.local((float)x, (float)y, xi, eta);
            profiles << spec.name << ','
                     << (spec.shape==Shape::Circle ? "circle" : "square") << ','
                     << (ibm ? "mesh_ibm" : "voxel") << ',' << kPrecision << ','
                     << x << ',' << y << ',' << xi << ',' << eta << ','
                     << exact/field.target_umax << ','
                     << uz_mean/field.target_umax << ','
                     << ux_mean/field.target_umax << ','
                     << uy_mean/field.target_umax << '\n';
        }
    }
    if(metrics.sample_points==0u || !(exact2>0.0))
        throw std::runtime_error("empty physical validation sample");
    metrics.l1 = absolute_error/absolute_exact;
    metrics.l2 = std::sqrt(error2/exact2);
    metrics.cross_velocity_rms = std::sqrt(
        cross2/(double)metrics.sample_points)/field.target_umax;
    metrics.flux_relative_error = metrics.flux_numeric/
        metrics.flux_exact_discrete-1.0;
    metrics.steady_flux_change = std::fabs(
        metrics.flux_numeric-checkpoint_flux)/
        std::max(std::fabs(metrics.flux_numeric), 1.0e-30);

    double slip2 = 0.0, normal2 = 0.0;
    std::size_t marker_count = 0u;
    const double center_z = 0.5*(double)(lbm.grid().size(2u)-1u);
    for(const auto& marker : design_markers) {
        if(std::fabs((double)marker.position_lu.z-center_z)>
           0.25*(double)lbm.grid().size(2u)) continue;
        const float3 velocity = interpolate_velocity(lbm, marker);
        const double slip = norm3(velocity);
        const double normal = dot3(velocity, marker.normal);
        slip2 += slip*slip;
        normal2 += normal*normal;
        metrics.marker_slip_max = std::max(metrics.marker_slip_max, slip);
        metrics.marker_normal_max = std::max(
            metrics.marker_normal_max, std::fabs(normal));
        ++marker_count;
    }
    if(marker_count==0u)
        throw std::runtime_error("empty central design-marker sample");
    metrics.marker_slip_rms = std::sqrt(slip2/(double)marker_count)/
        field.target_umax;
    metrics.marker_slip_max /= field.target_umax;
    metrics.marker_normal_rms = std::sqrt(normal2/(double)marker_count)/
        field.target_umax;
    metrics.marker_normal_max /= field.target_umax;

    const double final_mass = active_mass(lbm, active);
    metrics.fluid_mass_relative_drift =
        (final_mass-initial_mass)/initial_mass;
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        if(active[(std::size_t)n]==0u) continue;
        const double rho = (double)lbm.density()[n];
        metrics.rho_min = std::min(metrics.rho_min, rho);
        metrics.rho_max = std::max(metrics.rho_max, rho);
    }
    return metrics;
}

RunResult run_method(
    const CaseSpec& spec, const AnalyticField& field,
    const MeshPipeline& pipeline, const bool ibm, std::ofstream& profiles,
    std::ofstream& boundary_probes) {
    const uint nx = (uint)std::ceil(4.0f*spec.half_width+8.0f);
    const uint ny = nx;
    const auto config = pipeline_config(ibm, field.target_umax);
    std::string error;
    if(!culbm::surface_ibm::validate_config(config, error))
        throw std::runtime_error("invalid runtime SurfaceIBM config: "+error);
    culbm::surface_ibm::set_config(config);

    culbm::solver::Solver lbm(nx, ny, spec.nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(0.01f, 0.0f));
    for(uint z=0u; z<spec.nz; ++z) {
        for(uint y=0u; y<ny; ++y) {
            for(uint x=0u; x<nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                const bool fluid = field.geometry.inside((float)x, (float)y);
                lbm.density()[n] = 1.0f;
                lbm.velocity().x[n] = 0.0f;
                lbm.velocity().y[n] = 0.0f;
                lbm.velocity().z[n] = fluid
                    ? (float)exact_velocity(field, (float)x, (float)y)
                    : 0.0f;
                lbm.flags()[n] = fluid ? (uchar)0u : CELL_SOLID;
                lbm.force().x[n] = 0.0f;
                lbm.force().y[n] = 0.0f;
                lbm.force().z[n] = fluid ? (float)field.body_force : 0.0f;
                lbm.temperature()[n] = 1.0f;
            }
        }
    }
    if(ibm) culbm::surface_ibm::attach(lbm, pipeline.forcing_markers);
    std::vector<unsigned char> active((std::size_t)lbm.grid().cells(), 0u);
    double initial_mass = 0.0;
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        const bool fluid = (lbm.flags()[n]&CELL_SOLID)==0u;
        active[(std::size_t)n] = fluid ? 1u : 0u;
        if(fluid) initial_mass += (double)lbm.density()[n];
    }
    if(!(initial_mass>0.0))
        throw std::runtime_error("empty active-fluid mask");

    const unsigned steps = std::max(
        100u, (unsigned)std::ceil(
            spec.diffusion_times*(double)spec.half_width*spec.half_width/
            (double)kNu));
    const unsigned checkpoint_step = 4u*steps/5u;
    const unsigned reaction_window = std::max(20u, steps/10u);
    double checkpoint_flux = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> reactions;
    reactions.reserve(reaction_window);
    lbm.initialize_once();
    const auto begin = std::chrono::steady_clock::now();
    for(unsigned step=1u; step<=steps; ++step) {
        lbm.advance();
        if(ibm && step+reaction_window>steps)
            reactions.push_back(
                (double)culbm::surface_ibm::statistics(lbm).reaction_force_lu.z);
        if(step==checkpoint_step) {
            lbm.synchronize();
            lbm.velocity().download();
            checkpoint_flux = central_mean_flux(lbm, field);
        }
    }
    lbm.synchronize();
    const auto end = std::chrono::steady_clock::now();

    RunResult result;
    result.ibm = ibm;
    result.nx = nx;
    result.ny = ny;
    result.nz = spec.nz;
    result.steps = steps;
    result.runtime_seconds = std::chrono::duration<double>(end-begin).count();
    result.mlups = (double)lbm.grid().cells()*(double)steps/
                   std::max(result.runtime_seconds, 1.0e-30)/1.0e6;
    result.checkpoint_flux = checkpoint_flux;
    result.metrics = compute_metrics(
        lbm, field, pipeline.design_markers, active, initial_mass,
        checkpoint_flux, profiles, spec, ibm);
    write_boundary_probes(lbm, field, boundary_probes, spec, ibm);
    result.statistics = culbm::surface_ibm::statistics(lbm);
    if(!reactions.empty()) {
        result.reaction_z_mean = std::accumulate(
            reactions.begin(), reactions.end(), 0.0)/(double)reactions.size();
        double variance = 0.0;
        for(const double reaction : reactions) {
            const double delta = reaction-result.reaction_z_mean;
            variance += delta*delta;
        }
        if(reactions.size()>1u) variance /= (double)(reactions.size()-1u);
        result.reaction_z_cv = std::sqrt(variance)/
            std::max(std::fabs(result.reaction_z_mean), 1.0e-30);
    }
    return result;
}

void write_geometry_header(std::ofstream& output) {
    output << "case,shape,angle_deg,precision,nx,ny,nz,input_triangles,"
              "valid_triangles,degenerate_triangles,duplicate_triangles,"
              "retained_triangles,connected_components,watertight_components,"
              "open_edges,nonmanifold_edges,markers,input_area_lu2,"
              "retained_area_lu2,marker_area_lu2,analytic_area_lu2,"
              "marker_vs_retained_area_relative_error,"
              "marker_vs_analytic_area_relative_error,surface_distance_rms_lu,"
              "surface_distance_max_lu,inward_normal_min_cosine,"
              "forcing_signed_distance_min_lu,forcing_signed_distance_max_lu,"
              "retraction_projection_error_max_lu,warnings\n";
}

void write_geometry(std::ofstream& output, const CaseSpec& spec,
                    const uint nx, const MeshPipeline& pipeline) {
    const auto& r = pipeline.report;
    output << spec.name << ','
           << (spec.shape==Shape::Circle ? "circle" : "square") << ','
           << spec.angle_deg << ',' << kPrecision << ','
           << nx << ',' << nx << ',' << spec.nz << ','
           << r.input_triangles << ',' << r.valid_triangles << ','
           << r.degenerate_triangles << ',' << r.duplicate_triangles << ','
           << r.retained_triangles << ',' << r.connected_components << ','
           << r.watertight_components << ',' << r.open_edges << ','
           << r.nonmanifold_edges << ',' << r.markers << ','
           << r.input_area_lu2 << ',' << r.retained_area_lu2 << ','
           << r.marker_area_lu2 << ',' << pipeline.analytic_area << ','
           << pipeline.mesh_area_relative_error << ','
           << pipeline.analytic_area_relative_error << ','
           << pipeline.surface_distance_rms << ','
           << pipeline.surface_distance_max << ','
           << pipeline.inward_normal_min_cosine << ','
           << pipeline.forcing_signed_distance_min << ','
           << pipeline.forcing_signed_distance_max << ','
           << pipeline.retraction_projection_error_max << ','
           << r.warnings.size() << '\n';
}

void write_summary_header(std::ofstream& output) {
    output << "case,shape,method,precision,half_width,angle_deg,phase_x,phase_y,"
              "nx,ny,nz,steps,source_triangles,markers,active_cells,"
              "marker_spacing,retraction,support_thickness,forcing_iterations,"
              "runtime_s,mlups,l1,l2,linf,flux_numeric,flux_exact_discrete,"
              "flux_relative_error,checkpoint_flux,steady_flux_change,"
              "cross_velocity_rms,marker_slip_rms,marker_slip_max,"
              "marker_normal_rms,marker_normal_max,central_flux_cv,"
              "fluid_mass_relative_drift,rho_min,rho_max,mach_max,"
              "ibm_valid,ibm_converged,ibm_iterations,ibm_nonconverged_steps,"
              "ibm_peak_slip,reaction_z_mean,reaction_z_cv,sample_points\n";
}

void write_summary(std::ofstream& output, const CaseSpec& spec,
                   const MeshPipeline& pipeline, const RunResult& result) {
    const auto& m = result.metrics;
    output << spec.name << ','
           << (spec.shape==Shape::Circle ? "circle" : "square") << ','
           << (result.ibm ? "mesh_ibm" : "voxel") << ',' << kPrecision << ','
           << spec.half_width << ',' << spec.angle_deg << ','
           << spec.phase_x << ',' << spec.phase_y << ','
           << result.nx << ',' << result.ny << ',' << result.nz << ','
           << result.steps << ',' << pipeline.report.input_triangles << ','
           << pipeline.forcing_markers.size() << ','
           << result.statistics.active_cells << ','
           << kMarkerSpacing << ',' << kRetraction << ','
           << kSupportThickness << ',' << kForcingIterations << ','
           << result.runtime_seconds << ',' << result.mlups << ','
           << m.l1 << ',' << m.l2 << ',' << m.linf << ','
           << m.flux_numeric << ',' << m.flux_exact_discrete << ','
           << m.flux_relative_error << ',' << result.checkpoint_flux << ','
           << m.steady_flux_change << ',' << m.cross_velocity_rms << ','
           << m.marker_slip_rms << ',' << m.marker_slip_max << ','
           << m.marker_normal_rms << ',' << m.marker_normal_max << ','
           << m.central_flux_cv << ',' << m.fluid_mass_relative_drift << ','
           << m.rho_min << ',' << m.rho_max << ',' << m.mach_max << ','
           << (result.statistics.valid ? 1 : 0) << ','
           << (result.statistics.converged ? 1 : 0) << ','
           << result.statistics.iterations << ','
           << result.statistics.nonconverged_steps << ','
           << result.statistics.peak_max_slip_lu << ','
           << result.reaction_z_mean << ',' << result.reaction_z_cv << ','
           << m.sample_points << '\n';
}

void write_comparison_header(std::ofstream& output) {
    output << "case,shape,angle_deg,precision,voxel_l2,ibm_l2,"
              "l2_reduction_fraction,voxel_abs_flux_error,ibm_abs_flux_error,"
              "flux_error_reduction_fraction,voxel_runtime_s,ibm_runtime_s,"
              "runtime_ratio_ibm_over_voxel,geometry_gate,physical_gate,"
              "stability_gate,overall_gate\n";
}

bool write_comparison(std::ofstream& output, const CaseSpec& spec,
                      const MeshPipeline& pipeline, const RunResult& voxel,
                      const RunResult& ibm) {
    const double voxel_flux_error = std::fabs(
        voxel.metrics.flux_relative_error);
    const double ibm_flux_error = std::fabs(
        ibm.metrics.flux_relative_error);
    const double l2_reduction = 1.0-ibm.metrics.l2/
        std::max(voxel.metrics.l2, 1.0e-30);
    const double flux_reduction = 1.0-ibm_flux_error/
        std::max(voxel_flux_error, 1.0e-30);
    const double target_umax = 0.08/(double)spec.half_width;
    const bool geometry_gate =
        pipeline.report.markers==pipeline.forcing_markers.size() &&
        pipeline.report.degenerate_triangles==0u &&
        pipeline.report.duplicate_triangles==0u &&
        std::fabs(pipeline.mesh_area_relative_error)<=1.0e-6 &&
        pipeline.inward_normal_min_cosine>=0.95;
    const bool physical_gate =
        ibm.metrics.l2<=0.80*voxel.metrics.l2 &&
        ibm_flux_error<=voxel_flux_error &&
        ibm.metrics.l2<=0.10;
    const bool stability_gate =
        ibm.statistics.valid && ibm.statistics.converged &&
        ibm.statistics.nonconverged_steps==0ull &&
        ibm.statistics.peak_max_slip_lu<=0.012*target_umax &&
        ibm.metrics.marker_slip_rms<=0.15 &&
        ibm.metrics.marker_slip_max<=0.20 &&
        ibm.metrics.marker_normal_max<=0.01 &&
        std::fabs(ibm.metrics.fluid_mass_relative_drift)<=1.0e-5 &&
        ibm.metrics.rho_min>=0.95 && ibm.metrics.rho_max<=1.05 &&
        ibm.metrics.mach_max<0.10 &&
        ibm.metrics.central_flux_cv<0.005 &&
        ibm.metrics.steady_flux_change<0.005 &&
        ibm.reaction_z_cv<0.02;
    const bool overall = geometry_gate && physical_gate && stability_gate;
    output << spec.name << ','
           << (spec.shape==Shape::Circle ? "circle" : "square") << ','
           << spec.angle_deg << ',' << kPrecision << ','
           << voxel.metrics.l2 << ',' << ibm.metrics.l2 << ','
           << l2_reduction << ',' << voxel_flux_error << ','
           << ibm_flux_error << ',' << flux_reduction << ','
           << voxel.runtime_seconds << ',' << ibm.runtime_seconds << ','
           << ibm.runtime_seconds/std::max(voxel.runtime_seconds, 1.0e-30) << ','
           << (geometry_gate ? 1 : 0) << ','
           << (physical_gate ? 1 : 0) << ','
           << (stability_gate ? 1 : 0) << ','
           << (overall ? 1 : 0) << '\n';
    return overall;
}

std::vector<CaseSpec> suite_cases(const std::string& suite) {
    if(suite=="smoke") {
        return {
            {"circle_a4", Shape::Circle, 4.0f, 0.0f,
             0.23f, -0.17f, 48u, 2.0f},
            {"square_a4_angle45", Shape::Square, 4.0f, 45.0f,
             0.23f, -0.17f, 48u, 2.0f},
        };
    }
    if(suite=="main") {
        return {
            {"circle_a8", Shape::Circle, 8.0f, 0.0f,
             0.23f, -0.17f, 64u, 5.0f},
            {"square_a8_angle30", Shape::Square, 8.0f, 30.0f,
             0.23f, -0.17f, 64u, 5.0f},
            {"square_a8_angle45", Shape::Square, 8.0f, 45.0f,
             0.23f, -0.17f, 64u, 5.0f},
        };
    }
    throw std::runtime_error("unknown STL-pipeline validation suite: "+suite);
}

}  // namespace

int main(const int argc, const char* const argv[]) {
    try {
        if(argc!=4 || std::string(argv[1])!="--suite") {
            std::cerr << "Usage: " << argv[0]
                      << " --suite smoke|main OUTPUT_DIR\n";
            return 2;
        }
        if(cuInit(0u)!=CUDA_SUCCESS) {
            std::cout << "SKIP: CUDA driver is unavailable.\n";
            return 77;
        }
        int device_count = 0;
        if(cuDeviceGetCount(&device_count)!=CUDA_SUCCESS || device_count<=0) {
            std::cout << "SKIP: no CUDA device is available.\n";
            return 77;
        }
        Subgrid_Configuration subgrid;
        subgrid.enabled = false;
        if(!set_subgrid_configuration(subgrid))
            throw std::runtime_error(
                "cannot select SGS-off STL-pipeline validation layout");

        const std::string suite = argv[2];
        const std::filesystem::path output_dir = argv[3];
        if(std::filesystem::exists(output_dir) &&
           !std::filesystem::is_empty(output_dir))
            throw std::runtime_error(
                "validation output directory already exists and is not empty");
        std::filesystem::create_directories(output_dir);
        std::ofstream geometry(output_dir/"geometry.csv");
        std::ofstream warnings(output_dir/"geometry_warnings.csv");
        std::ofstream summary(output_dir/"summary.csv");
        std::ofstream comparison(output_dir/"comparison.csv");
        std::ofstream profiles(output_dir/"profiles.csv");
        std::ofstream boundary_probes(output_dir/"boundary_probes.csv");
        if(!geometry || !warnings || !summary || !comparison || !profiles ||
           !boundary_probes)
            throw std::runtime_error("cannot create validation CSV outputs");
        geometry << std::setprecision(10);
        warnings << std::setprecision(10);
        summary << std::setprecision(10);
        comparison << std::setprecision(10);
        profiles << std::setprecision(10);
        boundary_probes << std::setprecision(10);
        write_geometry_header(geometry);
        warnings << "case,warning_index,warning\n";
        write_summary_header(summary);
        write_comparison_header(comparison);
        profiles << "case,shape,method,precision,x,y,xi,eta,u_exact_norm,"
                    "u_numeric_norm,ux_norm,uy_norm\n";
        boundary_probes <<
            "case,shape,method,precision,probe_index,x,y,xi,eta,u_exact_norm,"
            "u_numeric_norm,ux_norm,uy_norm\n";

        const auto cases = suite_cases(suite);
        bool all_gates_pass = true;
        std::cout << "SurfaceIBM STL-pipeline validation suite=" << suite
                  << " precision=" << kPrecision
                  << " cases=" << cases.size() << '\n';
        for(std::size_t i=0u; i<cases.size(); ++i) {
            const CaseSpec& spec = cases[i];
            const uint nx = (uint)std::ceil(4.0f*spec.half_width+8.0f);
            const AnalyticField field = make_field(spec, nx, nx);
            std::cout << "GEOMETRY " << (i+1u) << '/' << cases.size()
                      << " case=" << spec.name << '\n';
            const MeshPipeline pipeline = build_pipeline(
                spec, field, output_dir/"stl_inputs");
            write_geometry(geometry, spec, nx, pipeline);
            for(std::size_t warning_index=0u;
                warning_index<pipeline.report.warnings.size();
                ++warning_index) {
                std::string warning = pipeline.report.warnings[warning_index];
                std::replace(warning.begin(), warning.end(), ',', ';');
                warnings << spec.name << ',' << warning_index << ','
                         << warning << '\n';
            }
            std::cout << "RUN case=" << spec.name << " method=voxel\n";
            const RunResult voxel = run_method(
                spec, field, pipeline, false, profiles, boundary_probes);
            write_summary(summary, spec, pipeline, voxel);
            std::cout << "RUN case=" << spec.name << " method=mesh_ibm\n";
            const RunResult ibm = run_method(
                spec, field, pipeline, true, profiles, boundary_probes);
            write_summary(summary, spec, pipeline, ibm);
            const bool gate = write_comparison(
                comparison, spec, pipeline, voxel, ibm);
            all_gates_pass = all_gates_pass && gate;
            geometry.flush();
            warnings.flush();
            summary.flush();
            comparison.flush();
            profiles.flush();
            boundary_probes.flush();
            std::cout << std::setprecision(6)
                      << "RESULT case=" << spec.name
                      << " voxel_l2=" << voxel.metrics.l2
                      << " ibm_l2=" << ibm.metrics.l2
                      << " voxel_qerr=" << voxel.metrics.flux_relative_error
                      << " ibm_qerr=" << ibm.metrics.flux_relative_error
                      << " markers=" << pipeline.forcing_markers.size()
                      << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
        }

        std::ofstream metadata(output_dir/"metadata.txt");
        metadata << "suite=" << suite << '\n'
                 << "precision=" << kPrecision << '\n'
                 << "case_count=" << cases.size() << '\n'
                 << "mesh_input=synthetic_binary_stl_roundtrip\n"
                 << "solid_flags=analytic_reference_not_ray_voxelized\n"
                 << "geometry_policy=all_triangles\n"
                 << "delta_kernel=roma3_point\n"
                 << "marker_spacing_lu=" << kMarkerSpacing << '\n'
                 << "retraction_lu=" << kRetraction << '\n'
                 << "support_thickness_lu=" << kSupportThickness << '\n'
                 << "forcing_iterations=" << kForcingIterations << '\n'
                 << "nu_lu=" << kNu << '\n'
                 << "collision=SRT\n"
                 << "lattice=D3Q19\n"
                 << "subgrid=off\n"
                 << "overall_gate=" << (all_gates_pass ? "PASS" : "FAIL")
                 << '\n';
        std::cout << "SurfaceIBM STL-pipeline validation completed: "
                  << (all_gates_pass ? "PASS" : "FAIL") << '\n';
        return all_gates_pass ? 0 : 1;
    } catch(const std::exception& error) {
        std::cerr << "SURFACE_IBM_STL_PIPELINE_VALIDATION_ERROR "
                  << error.what() << '\n';
        return 1;
    }
}
