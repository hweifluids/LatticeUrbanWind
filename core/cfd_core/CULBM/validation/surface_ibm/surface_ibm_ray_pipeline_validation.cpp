#include "lbm.hpp"
#include "surface_ibm_geometry.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Production setup owns these optional forcing terms.  This geometry-only
// validation links the production runtime with every optional term disabled.
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

constexpr uint kGridSize = 48u;
constexpr float kMarkerSpacing = 0.75f;
constexpr float kRetraction = 0.5f;
constexpr float kSupportThickness = 0.75f;
constexpr unsigned kForcingIterations = 8u;
constexpr double kSdfExclusionLu = 1.5;
constexpr double kDeepCoreDistanceLu = 3.0;
constexpr double kFarFieldDistanceLu = 3.0;
constexpr double kMaximumClearedPenetrationLu = 3.0;
constexpr double kMaximumAreaRelativeError = 5.0e-5;

#ifdef FP16C
constexpr const char* kPrecision = "FP16C";
#else
constexpr const char* kPrecision = "FP32";
#endif

struct BoxSpec {
    std::string name;
    float3 center{};
    float3 half_extent{};
    float angle_degrees = 0.0f;
};

struct ValidationResult {
    std::string name;
    bool passed = false;
    std::uint64_t triangles = 0u;
    double roundtrip_vertex_error_max = 0.0;
    double roundtrip_bounds_error_max = 0.0;
    std::size_t connected_components = 0u;
    std::size_t watertight_components = 0u;
    std::size_t open_edges = 0u;
    std::size_t nonmanifold_edges = 0u;
    std::size_t geometry_warnings = 0u;
    std::uint64_t markers = 0u;
    double marker_area = 0.0;
    double analytic_area = 0.0;
    double area_relative_error = 0.0;
    std::uint64_t pre_solid_cells = 0u;
    std::uint64_t sdf_scored_cells = 0u;
    std::uint64_t sdf_expected_solid_cells = 0u;
    std::uint64_t sdf_expected_fluid_cells = 0u;
    std::uint64_t sdf_solid_mismatches = 0u;
    std::uint64_t sdf_fluid_mismatches = 0u;
    std::uint64_t sdf_near_surface_unscored = 0u;
    std::uint64_t support_cells = 0u;
    std::uint64_t expected_cleared_solid_cells = 0u;
    std::uint64_t actual_cleared_solid_cells = 0u;
    std::uint64_t missed_support_solid_cells = 0u;
    std::uint64_t overcleared_solid_cells = 0u;
    std::uint64_t unexpected_flag_changes = 0u;
    double cleared_penetration_max = 0.0;
    std::uint64_t deep_core_checked = 0u;
    std::uint64_t deep_core_preserved = 0u;
    std::uint64_t far_field_checked = 0u;
    std::uint64_t far_field_unchanged = 0u;
    std::uint64_t post_host_device_mismatches = 0u;
    std::uint64_t post_step_flag_mismatches = 0u;
    culbm::surface_ibm::Statistics statistics;
};

float3 cross3(const float3& a, const float3& b) {
    return float3(a.y*b.z-a.z*b.y,
                  a.z*b.x-a.x*b.z,
                  a.x*b.y-a.y*b.x);
}

float length3(const float3& value) {
    return std::sqrt(value.x*value.x+value.y*value.y+value.z*value.z);
}

double component_error(const float3& a, const float3& b) {
    return std::max({std::fabs((double)a.x-(double)b.x),
                     std::fabs((double)a.y-(double)b.y),
                     std::fabs((double)a.z-(double)b.z)});
}

float3 rotate_z(const float3& point, const float angle_radians,
                const float3& center) {
    const float cosine = std::cos(angle_radians);
    const float sine = std::sin(angle_radians);
    const float x = point.x-center.x;
    const float y = point.y-center.y;
    return float3(center.x+cosine*x-sine*y,
                  center.y+sine*x+cosine*y,
                  point.z);
}

std::unique_ptr<Mesh> make_rotated_box(const BoxSpec& spec) {
    const float3 h = spec.half_extent;
    const float3 c = spec.center;
    std::array<float3, 8u> vertices = {
        float3(c.x-h.x, c.y-h.y, c.z-h.z),
        float3(c.x+h.x, c.y-h.y, c.z-h.z),
        float3(c.x+h.x, c.y+h.y, c.z-h.z),
        float3(c.x-h.x, c.y+h.y, c.z-h.z),
        float3(c.x-h.x, c.y-h.y, c.z+h.z),
        float3(c.x+h.x, c.y-h.y, c.z+h.z),
        float3(c.x+h.x, c.y+h.y, c.z+h.z),
        float3(c.x-h.x, c.y+h.y, c.z+h.z)};
    const float angle = spec.angle_degrees*3.14159265358979323846f/180.0f;
    for(float3& vertex : vertices) vertex = rotate_z(vertex, angle, c);

    // Every triangle has outward winding.  The production exterior-union
    // builder independently checks and normalizes the closed component.
    constexpr std::array<std::array<unsigned, 3u>, 12u> triangles = {{
        {{0u, 2u, 1u}}, {{0u, 3u, 2u}}, // -z
        {{4u, 5u, 6u}}, {{4u, 6u, 7u}}, // +z
        {{0u, 1u, 5u}}, {{0u, 5u, 4u}}, // -local y
        {{1u, 2u, 6u}}, {{1u, 6u, 5u}}, // +local x
        {{2u, 3u, 7u}}, {{2u, 7u, 6u}}, // +local y
        {{3u, 0u, 4u}}, {{3u, 4u, 7u}}  // -local x
    }};
    auto mesh = std::make_unique<Mesh>((uint)triangles.size(), c);
    for(std::size_t i=0u; i<triangles.size(); ++i) {
        mesh->p0[i] = vertices[triangles[i][0u]];
        mesh->p1[i] = vertices[triangles[i][1u]];
        mesh->p2[i] = vertices[triangles[i][2u]];
    }
    mesh->find_bounds();
    return mesh;
}

void write_binary_stl(const std::filesystem::path& path, const Mesh& mesh) {
    std::ofstream output(path, std::ios::out|std::ios::binary|std::ios::trunc);
    if(!output) throw std::runtime_error("cannot create synthetic binary STL");
    std::array<char, 80u> header{};
    const std::string label = "CULBM ray-pipeline validation geometry";
    std::copy(label.begin(), label.end(), header.begin());
    output.write(header.data(), (std::streamsize)header.size());
    const std::uint32_t count = mesh.triangle_number;
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for(std::uint32_t i=0u; i<count; ++i) {
        const float3 cross_value = cross3(mesh.p1[i]-mesh.p0[i],
                                          mesh.p2[i]-mesh.p0[i]);
        const float magnitude = length3(cross_value);
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
    if(!output) throw std::runtime_error("cannot finish synthetic binary STL");
}

double signed_distance(const BoxSpec& spec, const double x,
                       const double y, const double z) {
    const double angle = (double)spec.angle_degrees*
                         3.141592653589793238462643383279502884/180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double dx = x-(double)spec.center.x;
    const double dy = y-(double)spec.center.y;
    const double local_x = cosine*dx+sine*dy;
    const double local_y = -sine*dx+cosine*dy;
    const double local_z = z-(double)spec.center.z;
    const double qx = std::fabs(local_x)-(double)spec.half_extent.x;
    const double qy = std::fabs(local_y)-(double)spec.half_extent.y;
    const double qz = std::fabs(local_z)-(double)spec.half_extent.z;
    const double ox = std::max(qx, 0.0);
    const double oy = std::max(qy, 0.0);
    const double oz = std::max(qz, 0.0);
    const double outside = std::sqrt(ox*ox+oy*oy+oz*oz);
    const double inside = std::min(std::max({qx, qy, qz}), 0.0);
    return outside+inside;
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

culbm::surface_ibm::Config validation_config() {
    culbm::surface_ibm::Config config;
    config.enabled = true;
    config.geometry_policy =
        culbm::surface_ibm::GeometryPolicy::ExteriorUnion;
    config.marker_spacing_lu = kMarkerSpacing;
    config.forcing_iterations = kForcingIterations;
    config.relaxation = 1.0f;
    config.residual_tolerance = 1.0e-6f;
    config.retraction_lu = kRetraction;
    config.support_thickness_lu = kSupportThickness;
    config.max_markers = 1000000u;
    config.strict_geometry = true;
    return config;
}

void configure_solver() {
    const auto config = validation_config();
    std::string error;
    if(!culbm::surface_ibm::validate_config(config, error))
        throw std::runtime_error("invalid frozen validation config: "+error);
    culbm::surface_ibm::set_config(config);

    Subgrid_Configuration subgrid;
    subgrid.enabled = false;
    subgrid.smagorinsky_constant = 0.0f;
    subgrid.scalar_coupling = false;
    subgrid.turbulent_prandtl = 1.0f;
    subgrid.fixed_eddy_viscosity = 0.0f;
    if(!set_subgrid_configuration(subgrid))
        throw std::runtime_error("cannot disable SGS for geometry validation");
}

std::vector<unsigned char> copy_flags(LBM& lbm) {
    std::vector<unsigned char> values((std::size_t)lbm.get_N());
    for(ulong n=0ull; n<lbm.get_N(); ++n)
        values[(std::size_t)n] = lbm.flags[n];
    return values;
}

std::vector<unsigned char> build_support_mask(
    const LBM& lbm,
    const std::vector<culbm::surface_ibm::Marker>& markers) {
    std::vector<unsigned char> support((std::size_t)lbm.get_N(), 0u);
    for(const auto& marker : markers) {
        const int cx = (int)std::floor(marker.position_lu.x+0.5f);
        const int cy = (int)std::floor(marker.position_lu.y+0.5f);
        const int cz = (int)std::floor(marker.position_lu.z+0.5f);
        for(int dz=-1; dz<=1; ++dz) {
            const int z = cz+dz;
            const float wz = roma3_phi(marker.position_lu.z-(float)z);
            for(int dy=-1; dy<=1; ++dy) {
                const int y = cy+dy;
                const float wy = roma3_phi(marker.position_lu.y-(float)y);
                for(int dx=-1; dx<=1; ++dx) {
                    const int x = cx+dx;
                    const float weight =
                        roma3_phi(marker.position_lu.x-(float)x)*wy*wz;
                    if(weight==0.0f) continue;
                    if(x<0 || y<0 || z<0 || x>=(int)lbm.get_Nx() ||
                       y>=(int)lbm.get_Ny() || z>=(int)lbm.get_Nz())
                        throw std::runtime_error(
                            "marker support escaped the validation lattice");
                    support[(std::size_t)lbm.index(
                        (uint)x, (uint)y, (uint)z)] = 1u;
                }
            }
        }
    }
    return support;
}

ValidationResult validate_case(const BoxSpec& spec,
                               const std::filesystem::path& output_dir) {
    ValidationResult result;
    result.name = spec.name;
    const auto generated = make_rotated_box(spec);
    const auto stl_path = output_dir/(spec.name+".stl");
    write_binary_stl(stl_path, *generated);
    std::unique_ptr<Mesh> mesh(read_stl(stl_path.string()));
    if(!mesh || mesh->triangle_number!=generated->triangle_number)
        throw std::runtime_error("binary STL round-trip changed triangle count");
    result.triangles = mesh->triangle_number;
    for(uint i=0u; i<mesh->triangle_number; ++i) {
        result.roundtrip_vertex_error_max = std::max({
            result.roundtrip_vertex_error_max,
            component_error(mesh->p0[i], generated->p0[i]),
            component_error(mesh->p1[i], generated->p1[i]),
            component_error(mesh->p2[i], generated->p2[i])});
    }
    mesh->find_bounds();
    result.roundtrip_bounds_error_max = std::max(
        component_error(mesh->pmin, generated->pmin),
        component_error(mesh->pmax, generated->pmax));

    const auto config = validation_config();
    std::vector<culbm::surface_ibm::Marker> markers;
    culbm::surface_ibm::GeometryReport geometry;
    std::string error;
    if(!culbm::surface_ibm::build_surface_ibm_geometry(
            *mesh, config, markers, geometry, error))
        throw std::runtime_error("production mesh-to-marker build failed: "+error);
    result.connected_components = geometry.connected_components;
    result.watertight_components = geometry.watertight_components;
    result.open_edges = geometry.open_edges;
    result.nonmanifold_edges = geometry.nonmanifold_edges;
    result.geometry_warnings = geometry.warnings.size();
    result.markers = markers.size();
    result.marker_area = geometry.marker_area_lu2;
    const double hx = spec.half_extent.x;
    const double hy = spec.half_extent.y;
    const double hz = spec.half_extent.z;
    result.analytic_area = 8.0*(hx*hy+hx*hz+hy*hz);
    result.area_relative_error = std::fabs(
        result.marker_area-result.analytic_area)/result.analytic_area;

    LBM lbm(kGridSize, kGridSize, kGridSize, 0.05f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.01f, 0.0f);
    for(ulong n=0ull; n<lbm.get_N(); ++n) {
        lbm.rho[n] = 1.0f;
        lbm.u.x[n] = lbm.u.y[n] = lbm.u.z[n] = 0.0f;
        lbm.flags[n] = (uchar)0u;
        lbm.F.x[n] = lbm.F.y[n] = lbm.F.z[n] = 0.0f;
        lbm.T[n] = 1.0f;
    }
    lbm.flags.write_to_device();
    lbm.voxelize_mesh_on_device(mesh.get(), TYPE_S);
    lbm.flags.read_from_device();
    const std::vector<unsigned char> before_attach = copy_flags(lbm);

    for(uint z=0u; z<lbm.get_Nz(); ++z) {
        for(uint y=0u; y<lbm.get_Ny(); ++y) {
            for(uint x=0u; x<lbm.get_Nx(); ++x) {
                const ulong n = lbm.index(x, y, z);
                const bool solid = (before_attach[(std::size_t)n]&TYPE_S)!=0u;
                if(solid) ++result.pre_solid_cells;
                const double sdf = signed_distance(spec, x, y, z);
                if(std::fabs(sdf)<=kSdfExclusionLu) {
                    ++result.sdf_near_surface_unscored;
                    continue;
                }
                ++result.sdf_scored_cells;
                const bool expected_solid = sdf<0.0;
                if(expected_solid) {
                    ++result.sdf_expected_solid_cells;
                    if(!solid) ++result.sdf_solid_mismatches;
                } else {
                    ++result.sdf_expected_fluid_cells;
                    if(solid) ++result.sdf_fluid_mismatches;
                }
            }
        }
    }

    const std::vector<unsigned char> support = build_support_mask(lbm, markers);
    result.support_cells = (std::uint64_t)std::count(
        support.begin(), support.end(), (unsigned char)1u);
    for(std::size_t n=0u; n<support.size(); ++n) {
        if(support[n]!=0u && (before_attach[n]&TYPE_S)!=0u)
            ++result.expected_cleared_solid_cells;
    }

    lbm.attach_surface_ibm(markers);
    const auto attached_statistics = lbm.surface_ibm_statistics();
    const std::vector<unsigned char> after_attach_host = copy_flags(lbm);

    for(uint z=0u; z<lbm.get_Nz(); ++z) {
        for(uint y=0u; y<lbm.get_Ny(); ++y) {
            for(uint x=0u; x<lbm.get_Nx(); ++x) {
                const ulong index = lbm.index(x, y, z);
                const std::size_t n = (std::size_t)index;
                const uchar before = before_attach[n];
                const uchar after = after_attach_host[n];
                const bool was_solid = (before&TYPE_S)!=0u;
                const bool is_solid = (after&TYPE_S)!=0u;
                const bool in_support = support[n]!=0u;
                if(was_solid && !is_solid) {
                    ++result.actual_cleared_solid_cells;
                    const double sdf = signed_distance(spec, x, y, z);
                    result.cleared_penetration_max = std::max(
                        result.cleared_penetration_max, std::max(-sdf, 0.0));
                    if(!in_support) ++result.overcleared_solid_cells;
                }
                if(was_solid && in_support && is_solid)
                    ++result.missed_support_solid_cells;
                const uchar expected_after = was_solid && in_support
                    ? (uchar)(before&~TYPE_S) : before;
                if(after!=expected_after) ++result.unexpected_flag_changes;

                const double sdf = signed_distance(spec, x, y, z);
                if(sdf < -kDeepCoreDistanceLu) {
                    ++result.deep_core_checked;
                    if(is_solid) ++result.deep_core_preserved;
                }
                if(sdf > kFarFieldDistanceLu) {
                    ++result.far_field_checked;
                    if(after==before) ++result.far_field_unchanged;
                }
            }
        }
    }

    // Attachment is intentionally pre-initialization.  Production initialize
    // transfers the modified host flags; this explicit round-trip verifies the
    // exact payload that will be consumed by the CUDA kernels.
    lbm.flags.write_to_device();
    lbm.flags.read_from_device();
    const std::vector<unsigned char> after_attach_device = copy_flags(lbm);
    for(std::size_t n=0u; n<after_attach_host.size(); ++n) {
        if(after_attach_host[n]!=after_attach_device[n])
            ++result.post_host_device_mismatches;
    }

    // One quiescent step validates construction of the sparse device runtime
    // and makes Statistics::valid observable; it is not a flow benchmark.
    lbm.external_step();
    lbm.external_finish();
    result.statistics = lbm.surface_ibm_statistics();
    lbm.flags.read_from_device();
    const std::vector<unsigned char> after_step = copy_flags(lbm);
    for(std::size_t n=0u; n<after_attach_host.size(); ++n) {
        if(after_attach_host[n]!=after_step[n])
            ++result.post_step_flag_mismatches;
    }

    const bool topology_ok =
        result.triangles==12u && result.connected_components==1u &&
        result.watertight_components==1u && result.open_edges==0u &&
        result.nonmanifold_edges==0u;
    const bool roundtrip_ok = result.roundtrip_vertex_error_max==0.0 &&
                              result.roundtrip_bounds_error_max==0.0;
    const bool marker_ok = result.markers>0u &&
        result.area_relative_error<=kMaximumAreaRelativeError;
    const bool voxel_ok = result.pre_solid_cells>0u &&
        result.sdf_expected_solid_cells>0u &&
        result.sdf_expected_fluid_cells>0u &&
        result.sdf_solid_mismatches==0u &&
        result.sdf_fluid_mismatches==0u;
    const bool clearing_ok = result.support_cells>0u &&
        result.expected_cleared_solid_cells>0u &&
        result.actual_cleared_solid_cells==
            result.expected_cleared_solid_cells &&
        result.missed_support_solid_cells==0u &&
        result.overcleared_solid_cells==0u &&
        result.unexpected_flag_changes==0u &&
        attached_statistics.markers==result.markers &&
        attached_statistics.cleared_solid_cells==
            result.actual_cleared_solid_cells &&
        result.cleared_penetration_max<=kMaximumClearedPenetrationLu;
    const bool invariants_ok = result.deep_core_checked>0u &&
        result.deep_core_checked==result.deep_core_preserved &&
        result.far_field_checked>0u &&
        result.far_field_checked==result.far_field_unchanged &&
        result.post_host_device_mismatches==0u &&
        result.post_step_flag_mismatches==0u;
    const bool runtime_ok = result.statistics.valid &&
        result.statistics.markers==result.markers &&
        result.statistics.cleared_solid_cells==
            result.actual_cleared_solid_cells &&
        result.statistics.active_cells>0u &&
        result.statistics.device_resident &&
        result.statistics.iterations>=1u &&
        result.statistics.converged &&
        result.statistics.nonconverged_steps==0u;
    result.passed = topology_ok && roundtrip_ok && marker_ok && voxel_ok &&
                    clearing_ok && invariants_ok && runtime_ok;
    return result;
}

void write_summary(const std::filesystem::path& path,
                   const std::string& suite,
                   const std::vector<ValidationResult>& results) {
    std::ofstream output(path, std::ios::out|std::ios::trunc);
    if(!output) throw std::runtime_error("cannot create summary.csv");
    output << "suite,precision,case,passed,triangles,roundtrip_vertex_error_max,"
              "roundtrip_bounds_error_max,connected_components,watertight_components,"
              "open_edges,nonmanifold_edges,geometry_warnings,markers,marker_area,"
              "analytic_area,area_relative_error,pre_solid_cells,sdf_scored_cells,"
              "sdf_expected_solid_cells,sdf_expected_fluid_cells,"
              "sdf_solid_mismatches,sdf_fluid_mismatches,"
              "sdf_near_surface_unscored,support_cells,"
              "expected_cleared_solid_cells,actual_cleared_solid_cells,"
              "missed_support_solid_cells,overcleared_solid_cells,"
              "unexpected_flag_changes,cleared_penetration_max,deep_core_checked,"
              "deep_core_preserved,far_field_checked,far_field_unchanged,"
              "post_host_device_mismatches,post_step_flag_mismatches,"
              "stats_valid,stats_markers,stats_active_cells,"
              "stats_cleared_solid_cells,stats_device_resident,stats_iterations,"
              "stats_converged,stats_nonconverged_steps,stats_max_slip_lu\n";
    output << std::setprecision(17);
    for(const auto& r : results) {
        output << suite << ',' << kPrecision << ',' << r.name << ','
               << (r.passed ? 1 : 0) << ',' << r.triangles << ','
               << r.roundtrip_vertex_error_max << ','
               << r.roundtrip_bounds_error_max << ','
               << r.connected_components << ',' << r.watertight_components << ','
               << r.open_edges << ',' << r.nonmanifold_edges << ','
               << r.geometry_warnings << ',' << r.markers << ','
               << r.marker_area << ',' << r.analytic_area << ','
               << r.area_relative_error << ',' << r.pre_solid_cells << ','
               << r.sdf_scored_cells << ',' << r.sdf_expected_solid_cells << ','
               << r.sdf_expected_fluid_cells << ','
               << r.sdf_solid_mismatches << ',' << r.sdf_fluid_mismatches << ','
               << r.sdf_near_surface_unscored << ',' << r.support_cells << ','
               << r.expected_cleared_solid_cells << ','
               << r.actual_cleared_solid_cells << ','
               << r.missed_support_solid_cells << ','
               << r.overcleared_solid_cells << ','
               << r.unexpected_flag_changes << ','
               << r.cleared_penetration_max << ','
               << r.deep_core_checked << ',' << r.deep_core_preserved << ','
               << r.far_field_checked << ',' << r.far_field_unchanged << ','
               << r.post_host_device_mismatches << ','
               << r.post_step_flag_mismatches << ','
               << (r.statistics.valid ? 1 : 0) << ','
               << r.statistics.markers << ',' << r.statistics.active_cells << ','
               << r.statistics.cleared_solid_cells << ','
               << (r.statistics.device_resident ? 1 : 0) << ','
               << r.statistics.iterations << ','
               << (r.statistics.converged ? 1 : 0) << ','
               << r.statistics.nonconverged_steps << ','
               << r.statistics.max_slip_lu << '\n';
    }
    if(!output) throw std::runtime_error("cannot finish summary.csv");
}

void write_metadata(const std::filesystem::path& path,
                    const std::string& suite,
                    const std::size_t cases,
                    const bool passed) {
    std::ofstream output(path, std::ios::out|std::ios::trunc);
    if(!output) throw std::runtime_error("cannot create metadata.csv");
    output << "key,value\n"
           << "validation,surface_ibm_ray_pipeline\n"
           << "scope,production_binary_stl_parser_gpu_ray_voxelizer_marker_builder_and_attach_flags\n"
           << "suite," << suite << "\n"
           << "precision," << kPrecision << "\n"
           << "grid," << kGridSize << 'x' << kGridSize << 'x' << kGridSize << "\n"
           << "cases," << cases << "\n"
           << "geometry_policy,ExteriorUnion\n"
           << "strict_geometry,1\n"
           << "marker_spacing_lu," << kMarkerSpacing << "\n"
           << "retraction_lu," << kRetraction << "\n"
           << "support_thickness_lu," << kSupportThickness << "\n"
           << "sdf_exclusion_lu," << kSdfExclusionLu << "\n"
           << "deep_core_distance_lu," << kDeepCoreDistanceLu << "\n"
           << "far_field_distance_lu," << kFarFieldDistanceLu << "\n"
           << "maximum_cleared_penetration_lu,"
           << kMaximumClearedPenetrationLu << "\n"
           << "maximum_area_relative_error,"
           << kMaximumAreaRelativeError << "\n"
           << "sgs_enabled,0\n"
           << "flow_physics_claim,0\n"
           << "performance_claim,0\n"
           << "epoch_seconds," << (std::uint64_t)std::time(nullptr) << "\n"
           << "passed," << (passed ? 1 : 0) << "\n";
    if(!output) throw std::runtime_error("cannot finish metadata.csv");
}

std::vector<BoxSpec> suite_cases(const std::string& suite) {
    std::vector<BoxSpec> cases;
    cases.push_back({"rotated_box_30deg",
                     float3(24.17f, 23.81f, 24.13f),
                     float3(9.0f, 5.0f, 8.0f), 30.0f});
    if(suite=="main") {
        cases.push_back({"rotated_box_43deg_phase",
                         float3(23.73f, 24.29f, 23.66f),
                         float3(7.5f, 4.5f, 6.5f), 43.0f});
    }
    return cases;
}

void require_empty_output_directory(const std::filesystem::path& path) {
    if(std::filesystem::exists(path)) {
        if(!std::filesystem::is_directory(path))
            throw std::runtime_error("output path exists and is not a directory");
        if(std::filesystem::directory_iterator(path)!=
           std::filesystem::directory_iterator())
            throw std::runtime_error("refusing to reuse a nonempty output directory");
    } else {
        std::filesystem::create_directories(path);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        if(argc!=4 || std::string(argv[1])!="--suite") {
            std::cerr << "Usage: " << argv[0]
                      << " --suite smoke|main OUTPUT_DIR\n";
            return 2;
        }
        const std::string suite = argv[2];
        if(suite!="smoke" && suite!="main")
            throw std::runtime_error("suite must be smoke or main");
        const std::filesystem::path output_dir = argv[3];
        require_empty_output_directory(output_dir);
        const std::filesystem::path stl_dir = output_dir/"stl_inputs";
        std::filesystem::create_directories(stl_dir);

        configure_solver();
        const auto cases = suite_cases(suite);
        std::vector<ValidationResult> results;
        results.reserve(cases.size());
        bool passed = true;
        for(std::size_t i=0u; i<cases.size(); ++i) {
            std::cout << "RAY_PIPELINE_CASE " << (i+1u) << '/'
                      << cases.size() << " name=" << cases[i].name << '\n';
            results.push_back(validate_case(cases[i], stl_dir));
            passed = passed && results.back().passed;
            const auto& result = results.back();
            std::cout << std::setprecision(9)
                      << "RAY_PIPELINE_RESULT name=" << result.name
                      << " passed=" << (result.passed ? 1 : 0)
                      << " markers=" << result.markers
                      << " solid=" << result.pre_solid_cells
                      << " sdf_mismatches="
                      << result.sdf_solid_mismatches+
                         result.sdf_fluid_mismatches
                      << " cleared=" << result.actual_cleared_solid_cells
                      << " deep_preserved=" << result.deep_core_preserved
                      << '/' << result.deep_core_checked
                      << " far_unchanged=" << result.far_field_unchanged
                      << '/' << result.far_field_checked << '\n';
        }
        write_summary(output_dir/"summary.csv", suite, results);
        write_metadata(output_dir/"metadata.csv", suite, results.size(), passed);
        std::cout << "RAY_PIPELINE_VALIDATION "
                  << (passed ? "PASS" : "FAIL")
                  << " summary=" << (output_dir/"summary.csv").string()
                  << '\n';
        return passed ? 0 : 1;
    } catch(const std::exception& error) {
        std::cerr << "RAY_PIPELINE_VALIDATION ERROR: " << error.what() << '\n';
        return 2;
    }
}
