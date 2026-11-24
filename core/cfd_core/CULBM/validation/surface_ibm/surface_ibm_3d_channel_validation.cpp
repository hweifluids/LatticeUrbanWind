#include "solver.hpp"
#include "surface_ibm.hpp"

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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

constexpr float kNu = 0.1f;
constexpr float kTargetUmax = 0.025f;
constexpr float kMarkerSpacing = 0.75f;
constexpr float kSupportThickness = 0.75f;
constexpr float kRetraction = 0.5f;
constexpr unsigned kForcingIterations = 32u;
// The hardest (1,2,3) wall reaches a sub-one-percent marker residual at the
// production maximum of 32 direct-forcing iterations.  This tolerance is
// well below the grid-level errors measured by the analytic A/B comparison.
constexpr float kToleranceFraction = 1.0e-2f;
constexpr double kSubcellOffset = 0.37;
constexpr double kForcingCoordinateMinimum = 2.0;
constexpr double kForcingCoordinateMaximumInset = 3.0;
constexpr double kSampleLowerFraction = 0.25;
constexpr double kSampleUpperFraction = 0.75;
constexpr double kNearWallThickness = 3.0;

#ifdef FP16C
constexpr const char* kPrecision = "FP16C";
#else
constexpr const char* kPrecision = "FP32";
#endif

enum class Method {
    Voxel,
    SurfaceIBM,
};

struct Orientation {
    const char* name = "axis_100";
    int h = 1;
    int k = 0;
    int l = 0;
};

struct Geometry {
    uint size = 0u;
    Orientation orientation{};
    double offset = 0.0;
    double half_period_phase = 0.0;
    double normal_length = 0.0;
    double width = 0.0;
    double body_force = 0.0;
    float3 integer_normal{};
    float3 unit_normal{};
    float3 tangent_1{};
    float3 tangent_2{};
};

struct CaseSpec {
    uint size = 32u;
    Orientation orientation{};
    Method method = Method::SurfaceIBM;
};

struct MarkerSet {
    std::vector<culbm::surface_ibm::Marker> forcing;
    std::vector<culbm::surface_ibm::Marker> design;
    std::array<std::uint64_t, 2u> wall_marker_count{};
    std::array<double, 2u> wall_area{};
};

struct Metrics {
    std::uint64_t sample_points = 0u;
    std::uint64_t near_wall_points = 0u;
    std::uint64_t design_marker_samples = 0u;
    std::uint64_t design_fluid_cells = 0u;
    std::uint64_t evolved_fluid_cells = 0u;
    double vector_l2_relative = 0.0;
    double primary_l2_relative = 0.0;
    double near_wall_l2_relative = 0.0;
    double vector_linf_over_umax = 0.0;
    double sampled_flux_relative_error = 0.0;
    double normal_leakage_rms_over_umax = 0.0;
    double normal_leakage_max_over_umax = 0.0;
    double secondary_rms_over_umax = 0.0;
    double secondary_max_over_umax = 0.0;
    double design_slip_rms_over_umax = 0.0;
    double design_slip_max_over_umax = 0.0;
    double design_normal_rms_over_umax = 0.0;
    double design_net_flux_over_area_umax = 0.0;
    double design_mass_initial = 0.0;
    double design_mass_final = 0.0;
    double design_mass_relative_change = 0.0;
    double evolved_mass_initial = 0.0;
    double evolved_mass_final = 0.0;
    double evolved_mass_relative_change = 0.0;
    double global_density_mean_offset = 0.0;
    double rho_min = std::numeric_limits<double>::infinity();
    double rho_max = -std::numeric_limits<double>::infinity();
    double mach_max = 0.0;
    double sample_support_covered_fraction = 0.0;
};

struct Acceptance {
    double vector_l2_limit = 0.0;
    double near_wall_l2_limit = 0.0;
    double absolute_flux_error_limit = 0.0;
};

struct RunResult {
    CaseSpec spec{};
    Geometry geometry{};
    MarkerSet markers{};
    Metrics metrics{};
    culbm::surface_ibm::Statistics statistics{};
    unsigned steps = 0u;
    int sample_lo = 0;
    int sample_hi = 0;
    double causal_margin_lu = 0.0;
    double runtime_seconds = 0.0;
    double mlups = 0.0;
    double ibm_iterations_mean = 0.0;
    unsigned ibm_iterations_max = 0u;
    double ibm_converged_fraction = 0.0;
    Acceptance acceptance{};
    bool topology_periodic = false;
    bool finite = true;
    int physics_gate_pass = -1;
};

const char* method_name(const Method method) {
    return method == Method::SurfaceIBM ? "ibm" : "voxel";
}

double dot3d(const float3& a, const float3& b) {
    return (double)a.x*(double)b.x+
           (double)a.y*(double)b.y+
           (double)a.z*(double)b.z;
}

float3 cross3(const float3& a, const float3& b) {
    return float3(
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x);
}

double length3d(const float3& value) {
    return std::sqrt(dot3d(value, value));
}

float3 normalized(const float3& value) {
    const double magnitude = length3d(value);
    if(!(magnitude>0.0) || !std::isfinite(magnitude))
        throw std::runtime_error("cannot normalize validation direction");
    return value/(float)magnitude;
}

double positive_mod(const double value, const double period) {
    double result = std::fmod(value, period);
    if(result<0.0) result += period;
    if(result>=period) result = 0.0;
    return result;
}

Geometry make_geometry(const CaseSpec& spec) {
    if(spec.size<24u || (spec.size%2u)!=0u)
        throw std::runtime_error("periodic channel size must be even and at least 24");
    if(std::gcd(std::gcd(std::abs(spec.orientation.h),
                         std::abs(spec.orientation.k)),
                std::abs(spec.orientation.l))!=1)
        throw std::runtime_error("periodic channel normal must be primitive");

    Geometry geometry;
    geometry.size = spec.size;
    geometry.orientation = spec.orientation;
    geometry.offset = 0.25*(double)spec.size+kSubcellOffset;
    geometry.half_period_phase = 0.5*(double)spec.size;
    geometry.integer_normal = float3(
        (float)spec.orientation.h,
        (float)spec.orientation.k,
        (float)spec.orientation.l);
    geometry.normal_length = length3d(geometry.integer_normal);
    geometry.unit_normal = normalized(geometry.integer_normal);
    geometry.width = geometry.half_period_phase/geometry.normal_length;
    geometry.body_force = 8.0*(double)kNu*(double)kTargetUmax/
                          (geometry.width*geometry.width);

    const std::array<float, 3u> components = {
        std::fabs(geometry.unit_normal.x),
        std::fabs(geometry.unit_normal.y),
        std::fabs(geometry.unit_normal.z)};
    const std::size_t least_aligned = (std::size_t)std::distance(
        components.begin(),
        std::min_element(components.begin(), components.end()));
    const std::array<float3, 3u> axes = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f)};
    geometry.tangent_1 = normalized(cross3(
        axes[least_aligned], geometry.unit_normal));
    geometry.tangent_2 = normalized(cross3(
        geometry.unit_normal, geometry.tangent_1));
    return geometry;
}

double phase_at(const Geometry& geometry, const double x,
                const double y, const double z) {
    return positive_mod(
        (double)geometry.orientation.h*x+
        (double)geometry.orientation.k*y+
        (double)geometry.orientation.l*z-geometry.offset,
        (double)geometry.size);
}

bool is_design_fluid(const Geometry& geometry, const double x,
                     const double y, const double z) {
    const double phase = phase_at(geometry, x, y, z);
    return phase>0.0 && phase<geometry.half_period_phase;
}

double exact_primary_velocity(const Geometry& geometry,
                              const double phase) {
    if(!(phase>0.0 && phase<geometry.half_period_phase)) return 0.0;
    const double s = phase/geometry.normal_length;
    return geometry.body_force*s*(geometry.width-s)/(2.0*(double)kNu);
}

bool verify_periodic_topology(const Geometry& geometry) {
    const int n = (int)geometry.size;
    for(int z=0; z<n; ++z) {
        for(int y=0; y<n; ++y) {
            for(int x=0; x<n; ++x) {
                const bool reference = is_design_fluid(geometry, x, y, z);
                if(reference!=is_design_fluid(geometry, x+n, y, z) ||
                   reference!=is_design_fluid(geometry, x, y+n, z) ||
                   reference!=is_design_fluid(geometry, x, y, z+n))
                    return false;
            }
        }
    }
    return true;
}

Acceptance acceptance_for(const CaseSpec& spec) {
    // These are frozen, case-specific validation limits, not fitted values.
    // They retain at least useful margin from the FP32 development baselines
    // while tightening as the channel becomes better resolved at N=48.
    if(spec.size==32u) {
        if(std::string(spec.orientation.name)=="axis_100")
            return Acceptance{0.008, 0.018, 0.008};
        if(std::string(spec.orientation.name)=="oblique_110")
            return Acceptance{0.022, 0.035, 0.006};
        if(std::string(spec.orientation.name)=="oblique_111")
            return Acceptance{0.022, 0.035, 0.015};
        if(std::string(spec.orientation.name)=="oblique_123")
            return Acceptance{0.085, 0.085, 0.045};
    }
    if(spec.size==48u) {
        if(std::string(spec.orientation.name)=="axis_100")
            return Acceptance{0.004, 0.015, 0.005};
        if(std::string(spec.orientation.name)=="oblique_110")
            return Acceptance{0.014, 0.035, 0.004};
        if(std::string(spec.orientation.name)=="oblique_111")
            return Acceptance{0.013, 0.035, 0.008};
        if(std::string(spec.orientation.name)=="oblique_123")
            return Acceptance{0.048, 0.050, 0.020};
    }
    throw std::runtime_error("missing periodic channel acceptance limits");
}

std::array<float3, 8u> box_corners(const float lo, const float hi) {
    return {
        float3(lo, lo, lo), float3(hi, lo, lo),
        float3(lo, hi, lo), float3(hi, hi, lo),
        float3(lo, lo, hi), float3(hi, lo, hi),
        float3(lo, hi, hi), float3(hi, hi, hi)};
}

bool forcing_position_is_supported(const float3& position,
                                   const uint size) {
    const double maximum = (double)size-kForcingCoordinateMaximumInset;
    return (double)position.x>=kForcingCoordinateMinimum &&
           (double)position.y>=kForcingCoordinateMinimum &&
           (double)position.z>=kForcingCoordinateMinimum &&
           (double)position.x<=maximum &&
           (double)position.y<=maximum &&
           (double)position.z<=maximum;
}

void append_plane_markers(const Geometry& geometry, const double phase,
                          const unsigned wall_index,
                          MarkerSet& markers) {
    const float3 fluid_normal = wall_index==0u
        ? geometry.unit_normal : -geometry.unit_normal;
    const float3 center(0.5f*(float)(geometry.size-1u));
    const double center_dot = dot3d(geometry.integer_normal, center);
    const double maximum_dot =
        (double)(geometry.orientation.h+geometry.orientation.k+
                 geometry.orientation.l)*(double)(geometry.size-1u);
    const double base = geometry.offset+phase;
    const int m_min = (int)std::floor((0.0-base)/(double)geometry.size)-1;
    const int m_max = (int)std::ceil(
        (maximum_dot-base)/(double)geometry.size)+1;
    const auto corners = box_corners(0.0f, (float)(geometry.size-1u));

    for(int m=m_min; m<=m_max; ++m) {
        const double plane_constant = base+(double)m*(double)geometry.size;
        const float3 origin = center+
            (float)((plane_constant-center_dot)/
                    (geometry.normal_length*geometry.normal_length))*
                geometry.integer_normal;
        double q1_min = std::numeric_limits<double>::infinity();
        double q1_max = -std::numeric_limits<double>::infinity();
        double q2_min = std::numeric_limits<double>::infinity();
        double q2_max = -std::numeric_limits<double>::infinity();
        for(const float3& corner : corners) {
            const float3 delta = corner-origin;
            const double q1 = dot3d(delta, geometry.tangent_1);
            const double q2 = dot3d(delta, geometry.tangent_2);
            q1_min = std::min(q1_min, q1);
            q1_max = std::max(q1_max, q1);
            q2_min = std::min(q2_min, q2);
            q2_max = std::max(q2_max, q2);
        }
        q1_min -= (double)kMarkerSpacing;
        q1_max += (double)kMarkerSpacing;
        q2_min -= (double)kMarkerSpacing;
        q2_max += (double)kMarkerSpacing;
        const unsigned bins_1 = std::max(
            1u, (unsigned)std::ceil((q1_max-q1_min)/kMarkerSpacing));
        const unsigned bins_2 = std::max(
            1u, (unsigned)std::ceil((q2_max-q2_min)/kMarkerSpacing));
        const double spacing_1 = (q1_max-q1_min)/(double)bins_1;
        const double spacing_2 = (q2_max-q2_min)/(double)bins_2;
        const float area = (float)(spacing_1*spacing_2);
        for(unsigned j=0u; j<bins_2; ++j) {
            const float q2 = (float)(q2_min+((double)j+0.5)*spacing_2);
            for(unsigned i=0u; i<bins_1; ++i) {
                const float q1 = (float)(q1_min+((double)i+0.5)*spacing_1);
                const float3 design_position = origin+
                    q1*geometry.tangent_1+q2*geometry.tangent_2;
                const float3 forcing_position = design_position-
                    kRetraction*fluid_normal;
                if(!forcing_position_is_supported(
                       forcing_position, geometry.size))
                    continue;

                culbm::surface_ibm::Marker design;
                design.position_lu = design_position;
                design.wall_velocity_lu = float3(0.0f);
                design.normal = fluid_normal;
                design.area_lu2 = area;
                design.source_triangle = wall_index;
                design.global_id = (std::uint64_t)markers.design.size();
                culbm::surface_ibm::Marker forcing = design;
                forcing.position_lu = forcing_position;
                markers.design.push_back(design);
                markers.forcing.push_back(forcing);
                ++markers.wall_marker_count[wall_index];
                markers.wall_area[wall_index] += (double)area;
            }
        }
    }
}

MarkerSet make_markers(const Geometry& geometry) {
    MarkerSet markers;
    append_plane_markers(geometry, 0.0, 0u, markers);
    append_plane_markers(
        geometry, geometry.half_period_phase, 1u, markers);
    if(markers.forcing.empty() ||
       markers.forcing.size()!=markers.design.size())
        throw std::runtime_error("periodic channel marker generation failed");
    for(std::size_t i=0u; i<markers.forcing.size(); ++i) {
        markers.forcing[i].global_id = (std::uint64_t)i;
        markers.design[i].global_id = (std::uint64_t)i;
    }
    return markers;
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

float3 interpolate_velocity(
    culbm::solver::Solver& lbm, const culbm::surface_ibm::Marker& marker) {
    const int cx = (int)std::floor(marker.position_lu.x+0.5f);
    const int cy = (int)std::floor(marker.position_lu.y+0.5f);
    const int cz = (int)std::floor(marker.position_lu.z+0.5f);
    float3 value(0.0f);
    float weight = 0.0f;
    for(int dz=-1; dz<=1; ++dz) {
        const int z = cz+dz;
        const float wz = roma3_phi(marker.position_lu.z-(float)z);
        for(int dy=-1; dy<=1; ++dy) {
            const int y = cy+dy;
            const float wy = roma3_phi(marker.position_lu.y-(float)y);
            for(int dx=-1; dx<=1; ++dx) {
                const int x = cx+dx;
                if(x<0 || y<0 || z<0 ||
                   x>=(int)lbm.grid().size(0u) || y>=(int)lbm.grid().size(1u) ||
                   z>=(int)lbm.grid().size(2u))
                    throw std::runtime_error(
                        "diagnostic marker support leaves lattice");
                const float w = roma3_phi(
                    marker.position_lu.x-(float)x)*wy*wz;
                if(w==0.0f) continue;
                const ulong n = lbm.grid().index((uint)x, (uint)y, (uint)z);
                value.x += w*lbm.velocity().x[n];
                value.y += w*lbm.velocity().y[n];
                value.z += w*lbm.velocity().z[n];
                weight += w;
            }
        }
    }
    if(!(weight>0.0f))
        throw std::runtime_error("zero diagnostic interpolation weight");
    return value/weight;
}

void configure_surface_ibm(const bool enabled) {
    culbm::surface_ibm::Config configuration;
    configuration.enabled = enabled;
    configuration.geometry_policy =
        culbm::surface_ibm::GeometryPolicy::AllTriangles;
    configuration.marker_spacing_lu = kMarkerSpacing;
    configuration.forcing_iterations = kForcingIterations;
    configuration.relaxation = 1.0f;
    configuration.residual_tolerance =
        kToleranceFraction*kTargetUmax;
    configuration.retraction_lu = kRetraction;
    configuration.support_thickness_lu = kSupportThickness;
    configuration.max_markers = 1000000u;
    configuration.strict_geometry = true;
    std::string error;
    if(!culbm::surface_ibm::validate_config(configuration, error))
        throw std::runtime_error(
            "invalid periodic channel SurfaceIBM config: "+error);
    culbm::surface_ibm::set_config(configuration);
}

bool in_sample_box(const RunResult& result, const float3& point) {
    return point.x>=(float)result.sample_lo &&
           point.y>=(float)result.sample_lo &&
           point.z>=(float)result.sample_lo &&
           point.x<=(float)result.sample_hi &&
           point.y<=(float)result.sample_hi &&
           point.z<=(float)result.sample_hi;
}

bool projected_forcing_support_is_complete(
    const Geometry& geometry, const float3& point) {
    const double phase = phase_at(geometry, point.x, point.y, point.z);
    const bool lower = phase<=0.5*geometry.half_period_phase;
    const double distance = lower
        ? phase/geometry.normal_length
        : (geometry.half_period_phase-phase)/geometry.normal_length;
    const float3 fluid_normal = lower
        ? geometry.unit_normal : -geometry.unit_normal;
    const float3 design = lower
        ? point-(float)distance*geometry.unit_normal
        : point+(float)distance*geometry.unit_normal;
    const float3 forcing = design-kRetraction*fluid_normal;
    return forcing_position_is_supported(forcing, geometry.size);
}

bool finite_metrics(const Metrics& metrics) {
    const std::array<double, 28u> values = {
        metrics.vector_l2_relative,
        metrics.primary_l2_relative,
        metrics.near_wall_l2_relative,
        metrics.vector_linf_over_umax,
        metrics.sampled_flux_relative_error,
        metrics.normal_leakage_rms_over_umax,
        metrics.normal_leakage_max_over_umax,
        metrics.secondary_rms_over_umax,
        metrics.secondary_max_over_umax,
        metrics.design_slip_rms_over_umax,
        metrics.design_slip_max_over_umax,
        metrics.design_normal_rms_over_umax,
        metrics.design_net_flux_over_area_umax,
        metrics.design_mass_initial,
        metrics.design_mass_final,
        metrics.design_mass_relative_change,
        metrics.evolved_mass_initial,
        metrics.evolved_mass_final,
        metrics.evolved_mass_relative_change,
        metrics.global_density_mean_offset,
        metrics.rho_min,
        metrics.rho_max,
        metrics.mach_max,
        metrics.sample_support_covered_fraction,
        (double)metrics.sample_points,
        (double)metrics.near_wall_points,
        (double)metrics.design_fluid_cells,
        (double)metrics.evolved_fluid_cells};
    return std::all_of(values.begin(), values.end(), [](const double value) {
        return std::isfinite(value);
    });
}

void compute_metrics(RunResult& result, culbm::solver::Solver& lbm,
                     const std::vector<unsigned char>& design_fluid,
                     const std::vector<unsigned char>& evolved_fluid,
                     std::ofstream& profiles) {
    double exact2 = 0.0;
    double vector_error2 = 0.0;
    double primary_error2 = 0.0;
    double near_exact2 = 0.0;
    double near_error2 = 0.0;
    double normal2 = 0.0;
    double secondary2 = 0.0;
    double flux_numeric = 0.0;
    double flux_exact = 0.0;
    std::uint64_t support_covered = 0u;

    for(uint z=0u; z<result.spec.size; ++z) {
        for(uint y=0u; y<result.spec.size; ++y) {
            for(uint x=0u; x<result.spec.size; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                const double rho = (double)lbm.density()[n];
                result.metrics.global_density_mean_offset += rho-1.0;
                if(design_fluid[(std::size_t)n]!=0u)
                    result.metrics.design_mass_final += rho;
                if(evolved_fluid[(std::size_t)n]!=0u) {
                    result.metrics.evolved_mass_final += rho;
                    result.metrics.rho_min = std::min(
                        result.metrics.rho_min, rho);
                    result.metrics.rho_max = std::max(
                        result.metrics.rho_max, rho);
                    const float3 velocity(
                        lbm.velocity().x[n], lbm.velocity().y[n], lbm.velocity().z[n]);
                    result.metrics.mach_max = std::max(
                        result.metrics.mach_max,
                        std::sqrt(3.0)*length3d(velocity));
                }
                const float3 point((float)x, (float)y, (float)z);
                if(design_fluid[(std::size_t)n]==0u ||
                   !in_sample_box(result, point))
                    continue;

                const double phase = phase_at(
                    result.geometry, point.x, point.y, point.z);
                const double s = phase/result.geometry.normal_length;
                const double wall_distance = std::min(
                    s, result.geometry.width-s);
                const double exact = exact_primary_velocity(
                    result.geometry, phase);
                const float3 velocity(
                    lbm.velocity().x[n], lbm.velocity().y[n], lbm.velocity().z[n]);
                const double primary = dot3d(
                    velocity, result.geometry.tangent_1);
                const double normal = dot3d(
                    velocity, result.geometry.unit_normal);
                const double secondary = dot3d(
                    velocity, result.geometry.tangent_2);
                const double primary_error = primary-exact;
                const double local_error2 = primary_error*primary_error+
                    normal*normal+secondary*secondary;
                exact2 += exact*exact;
                vector_error2 += local_error2;
                primary_error2 += primary_error*primary_error;
                normal2 += normal*normal;
                secondary2 += secondary*secondary;
                flux_numeric += primary;
                flux_exact += exact;
                result.metrics.vector_linf_over_umax = std::max(
                    result.metrics.vector_linf_over_umax,
                    std::sqrt(local_error2)/(double)kTargetUmax);
                result.metrics.normal_leakage_max_over_umax = std::max(
                    result.metrics.normal_leakage_max_over_umax,
                    std::fabs(normal)/(double)kTargetUmax);
                result.metrics.secondary_max_over_umax = std::max(
                    result.metrics.secondary_max_over_umax,
                    std::fabs(secondary)/(double)kTargetUmax);
                ++result.metrics.sample_points;
                if(projected_forcing_support_is_complete(
                       result.geometry, point))
                    ++support_covered;
                if(wall_distance<=kNearWallThickness) {
                    near_exact2 += exact*exact;
                    near_error2 += local_error2;
                    ++result.metrics.near_wall_points;
                }
                profiles << result.spec.size << ','
                         << result.spec.orientation.name << ','
                         << method_name(result.spec.method) << ','
                         << kPrecision << ',' << x << ',' << y << ',' << z
                         << ',' << phase << ',' << s << ','
                         << s/result.geometry.width << ',' << wall_distance
                         << ',' << exact/(double)kTargetUmax << ','
                         << primary/(double)kTargetUmax << ','
                         << normal/(double)kTargetUmax << ','
                         << secondary/(double)kTargetUmax << ','
                         << (wall_distance<=kNearWallThickness ? 1 : 0)
                         << '\n';
            }
        }
    }
    if(result.metrics.sample_points==0u || !(exact2>0.0) ||
       result.metrics.near_wall_points==0u || !(near_exact2>0.0) ||
       !(std::fabs(flux_exact)>0.0))
        throw std::runtime_error("empty periodic channel sample region");
    const double count = (double)result.metrics.sample_points;
    result.metrics.vector_l2_relative = std::sqrt(vector_error2/exact2);
    result.metrics.primary_l2_relative = std::sqrt(primary_error2/exact2);
    result.metrics.near_wall_l2_relative =
        std::sqrt(near_error2/near_exact2);
    result.metrics.sampled_flux_relative_error =
        flux_numeric/flux_exact-1.0;
    result.metrics.normal_leakage_rms_over_umax =
        std::sqrt(normal2/count)/(double)kTargetUmax;
    result.metrics.secondary_rms_over_umax =
        std::sqrt(secondary2/count)/(double)kTargetUmax;
    result.metrics.sample_support_covered_fraction =
        (double)support_covered/count;

    double slip2_area = 0.0;
    double normal2_area = 0.0;
    double net_flux = 0.0;
    double diagnostic_area = 0.0;
    for(const auto& marker : result.markers.design) {
        if(!in_sample_box(result, marker.position_lu)) continue;
        const float3 velocity = interpolate_velocity(lbm, marker);
        const double slip = length3d(velocity);
        const double normal = dot3d(velocity, marker.normal);
        const double area = (double)marker.area_lu2;
        slip2_area += area*slip*slip;
        normal2_area += area*normal*normal;
        net_flux += area*normal;
        diagnostic_area += area;
        result.metrics.design_slip_max_over_umax = std::max(
            result.metrics.design_slip_max_over_umax,
            slip/(double)kTargetUmax);
        ++result.metrics.design_marker_samples;
    }
    if(!(diagnostic_area>0.0))
        throw std::runtime_error("empty design-wall marker diagnostic region");
    result.metrics.design_slip_rms_over_umax =
        std::sqrt(slip2_area/diagnostic_area)/(double)kTargetUmax;
    result.metrics.design_normal_rms_over_umax =
        std::sqrt(normal2_area/diagnostic_area)/(double)kTargetUmax;
    result.metrics.design_net_flux_over_area_umax =
        net_flux/(diagnostic_area*(double)kTargetUmax);

    const double cells = (double)lbm.grid().cells();
    result.metrics.global_density_mean_offset /= cells;
    result.metrics.design_mass_relative_change =
        (result.metrics.design_mass_final-result.metrics.design_mass_initial)/
        result.metrics.design_mass_initial;
    result.metrics.evolved_mass_relative_change =
        (result.metrics.evolved_mass_final-result.metrics.evolved_mass_initial)/
        result.metrics.evolved_mass_initial;
}

RunResult run_case(const CaseSpec& spec, std::ofstream& profiles) {
    RunResult result;
    result.spec = spec;
    result.acceptance = acceptance_for(spec);
    result.geometry = make_geometry(spec);
    result.topology_periodic = verify_periodic_topology(result.geometry);
    if(!result.topology_periodic)
        throw std::runtime_error("periodic channel topology self-check failed");
    result.markers = make_markers(result.geometry);
    result.steps = std::max(3u, spec.size/8u);
    result.sample_lo = (int)std::ceil(
        kSampleLowerFraction*(double)spec.size);
    result.sample_hi = (int)std::floor(
        kSampleUpperFraction*(double)spec.size)-1;
    const double transition_guard =
        kForcingCoordinateMinimum+kRetraction;
    const double low_clearance =
        (double)result.sample_lo-transition_guard;
    const double high_clearance =
        ((double)spec.size-1.0-transition_guard)-
        (double)result.sample_hi;
    result.causal_margin_lu =
        std::min(low_clearance, high_clearance)-(double)result.steps;
    if(result.causal_margin_lu<1.0)
        throw std::runtime_error(
            "short-time sample gate has insufficient causal margin");

    const bool ibm = spec.method==Method::SurfaceIBM;
    configure_surface_ibm(ibm);
    culbm::solver::Solver lbm(spec.size, spec.size, spec.size, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(0.01f, 0.0f));
    const std::size_t cell_count =
        (std::size_t)spec.size*(std::size_t)spec.size*(std::size_t)spec.size;
    std::vector<unsigned char> design_fluid(cell_count, 0u);
    for(uint z=0u; z<spec.size; ++z) {
        for(uint y=0u; y<spec.size; ++y) {
            for(uint x=0u; x<spec.size; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                const double phase = phase_at(result.geometry, x, y, z);
                const bool fluid = phase>0.0 &&
                                   phase<result.geometry.half_period_phase;
                const double exact = exact_primary_velocity(
                    result.geometry, phase);
                design_fluid[(std::size_t)n] = fluid ? 1u : 0u;
                lbm.density()[n] = 1.0f;
                lbm.velocity().x[n] = fluid
                    ? (float)exact*result.geometry.tangent_1.x : 0.0f;
                lbm.velocity().y[n] = fluid
                    ? (float)exact*result.geometry.tangent_1.y : 0.0f;
                lbm.velocity().z[n] = fluid
                    ? (float)exact*result.geometry.tangent_1.z : 0.0f;
                lbm.flags()[n] = fluid ? (uchar)0u : CELL_SOLID;
                // This is deliberately a field force rather than a global
                // constructor force: cells cleared solely for the diffuse IBM
                // support remain unforced fictitious fluid.
                lbm.force().x[n] = fluid
                    ? (float)result.geometry.body_force*
                          result.geometry.tangent_1.x : 0.0f;
                lbm.force().y[n] = fluid
                    ? (float)result.geometry.body_force*
                          result.geometry.tangent_1.y : 0.0f;
                lbm.force().z[n] = fluid
                    ? (float)result.geometry.body_force*
                          result.geometry.tangent_1.z : 0.0f;
                lbm.temperature()[n] = 1.0f;
            }
        }
    }
    if(ibm) culbm::surface_ibm::attach(lbm, result.markers.forcing);

    std::vector<unsigned char> evolved_fluid(cell_count, 0u);
    for(std::size_t i=0u; i<cell_count; ++i) {
        const double rho = (double)lbm.density()[(ulong)i];
        if(design_fluid[i]!=0u) {
            ++result.metrics.design_fluid_cells;
            result.metrics.design_mass_initial += rho;
        }
        if((lbm.flags()[(ulong)i]&CELL_SOLID)==0u) {
            evolved_fluid[i] = 1u;
            ++result.metrics.evolved_fluid_cells;
            result.metrics.evolved_mass_initial += rho;
        }
    }
    if(result.metrics.design_fluid_cells==0u ||
       result.metrics.evolved_fluid_cells==0u)
        throw std::runtime_error("periodic channel has no fluid cells");

    std::uint64_t iteration_sum = 0u;
    unsigned converged_steps = 0u;
    lbm.initialize_once();
    const auto begin = std::chrono::steady_clock::now();
    for(unsigned step=0u; step<result.steps; ++step) {
        lbm.advance();
        if(ibm) {
            const auto statistics = culbm::surface_ibm::statistics(lbm);
            iteration_sum += statistics.iterations;
            result.ibm_iterations_max = std::max(
                result.ibm_iterations_max, statistics.iterations);
            if(statistics.converged) ++converged_steps;
        }
    }
    lbm.synchronize();
    const auto end = std::chrono::steady_clock::now();
    result.runtime_seconds =
        std::chrono::duration<double>(end-begin).count();
    result.mlups = (double)cell_count*(double)result.steps/
        std::max(result.runtime_seconds, 1.0e-30)/1.0e6;
    result.statistics = culbm::surface_ibm::statistics(lbm);
    if(ibm) {
        result.ibm_iterations_mean =
            (double)iteration_sum/(double)result.steps;
        result.ibm_converged_fraction =
            (double)converged_steps/(double)result.steps;
    }

    lbm.density().download();
    lbm.velocity().download();
    compute_metrics(
        result, lbm, design_fluid, evolved_fluid, profiles);
    result.finite = finite_metrics(result.metrics) &&
        std::isfinite(result.runtime_seconds) &&
        std::isfinite(result.mlups) &&
        std::isfinite(result.causal_margin_lu);
    if(ibm) {
        result.physics_gate_pass =
            result.finite && result.topology_periodic &&
            result.causal_margin_lu>=1.0 &&
            result.metrics.sample_support_covered_fraction>=0.999999 &&
            result.statistics.valid &&
            result.metrics.vector_l2_relative<=
                result.acceptance.vector_l2_limit &&
            result.metrics.near_wall_l2_relative<=
                result.acceptance.near_wall_l2_limit &&
            std::fabs(result.metrics.sampled_flux_relative_error)<=
                result.acceptance.absolute_flux_error_limit &&
            result.metrics.normal_leakage_rms_over_umax<=0.02 &&
            result.metrics.secondary_rms_over_umax<=0.02 &&
            result.metrics.design_normal_rms_over_umax<=0.005 &&
            std::fabs(result.metrics.evolved_mass_relative_change)<=1.0e-4 &&
            result.metrics.mach_max<=0.08 &&
            result.ibm_converged_fraction>=0.95 ? 1 : 0;
    }
    return result;
}

void write_summary_header(std::ofstream& output) {
    output
        << "size,orientation,h,k,l,normal_x,normal_y,normal_z,method,precision,"
           "nu,target_umax,offset_phase,phase_period,channel_width,body_force,"
           "retraction,marker_spacing,support_thickness,forcing_iterations,"
           "residual_tolerance,steps,sample_lo,sample_hi,causal_margin_lu,"
           "topology_periodic,forcing_markers,wall0_markers,wall1_markers,"
           "wall0_area,wall1_area,wall0_area_fraction,wall1_area_fraction,"
           "active_cells,cleared_solid_cells,runtime_s,mlups,sample_points,"
           "near_wall_points,sample_support_covered_fraction,vector_l2_rel,"
           "vector_l2_limit,primary_l2_rel,near_wall_l2_rel,"
           "near_wall_l2_limit,vector_linf_over_umax,"
           "sampled_flux_rel_error,normal_leakage_rms_over_umax,"
           "absolute_flux_error_limit,"
           "normal_leakage_max_over_umax,secondary_rms_over_umax,"
           "secondary_max_over_umax,design_marker_samples,"
           "design_slip_rms_over_umax,design_slip_max_over_umax,"
           "design_normal_rms_over_umax,design_net_flux_over_area_umax,"
           "design_fluid_cells,evolved_fluid_cells,design_mass_initial,"
           "design_mass_final,design_mass_relative_change,evolved_mass_initial,"
           "evolved_mass_final,evolved_mass_relative_change,"
           "global_density_mean_offset,rho_min,rho_max,mach_max,ibm_valid,"
           "ibm_converged_final,ibm_iterations_final,ibm_iterations_mean,"
           "ibm_iterations_max,ibm_converged_fraction,ibm_nonconverged_steps,"
           "ibm_max_slip_over_umax,ibm_rms_slip_over_umax,"
           "ibm_peak_slip_over_umax,reaction_tangent,reaction_normal,finite,"
           "physics_gate_pass\n";
}

void write_summary(std::ofstream& output, const RunResult& result) {
    const auto& g = result.geometry;
    const auto& m = result.metrics;
    const auto& s = result.statistics;
    const double full_wall_area =
        (double)result.spec.size*(double)result.spec.size*g.normal_length;
    const double reaction_tangent = dot3d(
        s.reaction_force_lu, g.tangent_1);
    const double reaction_normal = dot3d(
        s.reaction_force_lu, g.unit_normal);
    output << result.spec.size << ',' << result.spec.orientation.name << ','
           << result.spec.orientation.h << ',' << result.spec.orientation.k
           << ',' << result.spec.orientation.l << ',' << g.unit_normal.x
           << ',' << g.unit_normal.y << ',' << g.unit_normal.z << ','
           << method_name(result.spec.method) << ',' << kPrecision << ','
           << kNu << ',' << kTargetUmax << ',' << g.offset << ','
           << result.spec.size << ',' << g.width << ',' << g.body_force << ','
           << kRetraction << ',' << kMarkerSpacing << ',' << kSupportThickness
           << ',' << kForcingIterations << ','
           << kToleranceFraction*kTargetUmax << ',' << result.steps << ','
           << result.sample_lo << ',' << result.sample_hi << ','
           << result.causal_margin_lu << ','
           << (result.topology_periodic ? 1 : 0) << ','
           << result.markers.forcing.size() << ','
           << result.markers.wall_marker_count[0] << ','
           << result.markers.wall_marker_count[1] << ','
           << result.markers.wall_area[0] << ','
           << result.markers.wall_area[1] << ','
           << result.markers.wall_area[0]/full_wall_area << ','
           << result.markers.wall_area[1]/full_wall_area << ','
           << s.active_cells << ',' << s.cleared_solid_cells << ','
           << result.runtime_seconds << ',' << result.mlups << ','
           << m.sample_points << ',' << m.near_wall_points << ','
           << m.sample_support_covered_fraction << ','
           << m.vector_l2_relative << ','
           << result.acceptance.vector_l2_limit << ','
           << m.primary_l2_relative << ',' << m.near_wall_l2_relative << ','
           << result.acceptance.near_wall_l2_limit << ','
           << m.vector_linf_over_umax << ','
           << m.sampled_flux_relative_error << ','
           << m.normal_leakage_rms_over_umax << ','
           << result.acceptance.absolute_flux_error_limit << ','
           << m.normal_leakage_max_over_umax << ','
           << m.secondary_rms_over_umax << ','
           << m.secondary_max_over_umax << ','
           << m.design_marker_samples << ','
           << m.design_slip_rms_over_umax << ','
           << m.design_slip_max_over_umax << ','
           << m.design_normal_rms_over_umax << ','
           << m.design_net_flux_over_area_umax << ','
           << m.design_fluid_cells << ',' << m.evolved_fluid_cells << ','
           << m.design_mass_initial << ',' << m.design_mass_final << ','
           << m.design_mass_relative_change << ','
           << m.evolved_mass_initial << ',' << m.evolved_mass_final << ','
           << m.evolved_mass_relative_change << ','
           << m.global_density_mean_offset << ',' << m.rho_min << ','
           << m.rho_max << ',' << m.mach_max << ','
           << (s.valid ? 1 : 0) << ',' << (s.converged ? 1 : 0) << ','
           << s.iterations << ',' << result.ibm_iterations_mean << ','
           << result.ibm_iterations_max << ','
           << result.ibm_converged_fraction << ',' << s.nonconverged_steps
           << ',' << s.max_slip_lu/(double)kTargetUmax << ','
           << s.rms_slip_lu/(double)kTargetUmax << ','
           << s.peak_max_slip_lu/(double)kTargetUmax << ','
           << reaction_tangent << ',' << reaction_normal << ','
           << (result.finite ? 1 : 0) << ',' << result.physics_gate_pass
           << '\n';
}

const RunResult* find_result(const std::vector<RunResult>& results,
                             const uint size, const std::string& orientation,
                             const Method method) {
    const auto found = std::find_if(
        results.begin(), results.end(), [&](const RunResult& result) {
            return result.spec.size==size &&
                   orientation==result.spec.orientation.name &&
                   result.spec.method==method;
        });
    return found==results.end() ? nullptr : &*found;
}

void write_comparisons(const std::filesystem::path& path,
                       const std::vector<RunResult>& results) {
    std::ofstream output(path, std::ios::out|std::ios::trunc);
    if(!output) throw std::runtime_error("cannot create comparison CSV");
    output << std::setprecision(10)
           << "size,orientation,precision,voxel_l2,ibm_l2,"
              "l2_reduction_fraction,voxel_near_wall_l2,ibm_near_wall_l2,"
              "near_wall_l2_reduction_fraction,voxel_abs_flux_error,"
              "ibm_abs_flux_error,flux_error_reduction_fraction,"
              "voxel_design_slip,ibm_design_slip,slip_reduction_fraction,"
              "voxel_runtime_s,ibm_runtime_s,runtime_ratio_ibm_over_voxel,"
              "ibm_l2_better,ibm_near_wall_better,ibm_flux_better,"
              "minimum_l2_reduction,minimum_near_wall_l2_reduction,"
              "minimum_flux_error_reduction,pair_gate_pass,ibm_gate_pass\n";
    for(const auto& ibm : results) {
        if(ibm.spec.method!=Method::SurfaceIBM) continue;
        const RunResult* voxel = find_result(
            results, ibm.spec.size, ibm.spec.orientation.name, Method::Voxel);
        if(voxel==nullptr) continue;
        const double voxel_flux = std::fabs(
            voxel->metrics.sampled_flux_relative_error);
        const double ibm_flux = std::fabs(
            ibm.metrics.sampled_flux_relative_error);
        constexpr double minimum_l2_reduction = 0.05;
        constexpr double minimum_near_wall_reduction = 0.05;
        constexpr double minimum_flux_reduction = 0.10;
        const double l2_reduction = 1.0-ibm.metrics.vector_l2_relative/
            std::max(voxel->metrics.vector_l2_relative, 1.0e-30);
        const double near_wall_reduction =
            1.0-ibm.metrics.near_wall_l2_relative/
                std::max(voxel->metrics.near_wall_l2_relative, 1.0e-30);
        const double flux_reduction =
            1.0-ibm_flux/std::max(voxel_flux, 1.0e-30);
        const bool pair_gate = l2_reduction>=minimum_l2_reduction &&
            near_wall_reduction>=minimum_near_wall_reduction &&
            flux_reduction>=minimum_flux_reduction;
        output << ibm.spec.size << ',' << ibm.spec.orientation.name << ','
               << kPrecision << ',' << voxel->metrics.vector_l2_relative << ','
               << ibm.metrics.vector_l2_relative << ','
               << l2_reduction << ','
               << voxel->metrics.near_wall_l2_relative << ','
               << ibm.metrics.near_wall_l2_relative << ','
               << near_wall_reduction
               << ',' << voxel_flux << ',' << ibm_flux << ','
               << flux_reduction << ','
               << voxel->metrics.design_slip_rms_over_umax << ','
               << ibm.metrics.design_slip_rms_over_umax << ','
               << 1.0-ibm.metrics.design_slip_rms_over_umax/
                    std::max(voxel->metrics.design_slip_rms_over_umax, 1.0e-30)
               << ',' << voxel->runtime_seconds << ',' << ibm.runtime_seconds
               << ',' << ibm.runtime_seconds/
                    std::max(voxel->runtime_seconds, 1.0e-30) << ','
               << (ibm.metrics.vector_l2_relative<
                       voxel->metrics.vector_l2_relative ? 1 : 0) << ','
               << (ibm.metrics.near_wall_l2_relative<
                       voxel->metrics.near_wall_l2_relative ? 1 : 0) << ','
               << (ibm_flux<voxel_flux ? 1 : 0) << ','
               << minimum_l2_reduction << ','
               << minimum_near_wall_reduction << ','
               << minimum_flux_reduction << ','
               << (pair_gate ? 1 : 0) << ','
               << ibm.physics_gate_pass << '\n';
    }
}

std::vector<CaseSpec> make_cases(const std::string& suite) {
    const std::array<Orientation, 4u> orientations = {
        Orientation{"axis_100", 1, 0, 0},
        Orientation{"oblique_110", 1, 1, 0},
        Orientation{"oblique_111", 1, 1, 1},
        Orientation{"oblique_123", 1, 2, 3}};
    std::vector<uint> sizes;
    if(suite=="smoke") sizes = {32u};
    else if(suite=="main") sizes = {32u, 48u};
    else throw std::runtime_error("unknown validation suite: "+suite);
    std::vector<CaseSpec> cases;
    for(const uint size : sizes) {
        for(const Orientation& orientation : orientations) {
            cases.push_back(CaseSpec{size, orientation, Method::Voxel});
            cases.push_back(CaseSpec{size, orientation, Method::SurfaceIBM});
        }
    }
    return cases;
}

void write_metadata(const std::filesystem::path& path,
                    const std::string& suite,
                    const std::size_t case_count) {
    std::ofstream output(path, std::ios::out|std::ios::trunc);
    if(!output) throw std::runtime_error("cannot create metadata file");
    output << std::setprecision(10)
           << "suite=" << suite << '\n'
           << "precision=" << kPrecision << '\n'
           << "case_count=" << case_count << '\n'
           << "geometry=periodic_two-wall_slab_on_three-torus\n"
           << "phase_definition=(h*x+k*y+l*z-offset)_mod_N\n"
           << "fluid_phase_interval=(0,N/2)\n"
           << "primitive_normals=(1,0,0);(1,1,0);(1,1,1);(1,2,3)\n"
           << "channel_width=(N/2)/sqrt(h^2+k^2+l^2)\n"
           << "analytic_solution=u_t=g*s*(W-s)/(2*nu)\n"
           << "body_force=8*nu*target_umax/W^2\n"
           << "initial_condition=analytic_steady_profile\n"
           << "force_application=design_fluid_only\n"
           << "ibm_support_cells_cleared_from_solid_are_unforced=1\n"
           << "outer_topology=fully_periodic_voxel_slab\n"
           << "ibm_patch=all_periodic_plane_copies_with_complete_non-shell_Roma3_support\n"
           << "patch_transition=outer_incomplete_support_band_uses_voxel_wall\n"
           << "sample_region=central_coordinate_box_[ceil(N/4),floor(3N/4)-1]^3\n"
           << "short_time_gate=D3Q19_information_speed_at_most_one_coordinate_cell_per_step\n"
           << "steps=max(3,N/8)\n"
           << "required_causal_margin_lu=1\n"
           << "note_size_pairs_are_resolution_sensitivity_not_asymptotic_order\n"
           << "note_short_run_runtime_is_diagnostic_not_a_performance_benchmark\n"
           << "design_surface_slip_is_regularized_interpolation_diagnostic_not_gate=1\n"
           << "no_slip_gate=computational_marker_convergence_plus_analytic_field_error\n"
           << "nu_lu=" << kNu << '\n'
           << "target_umax_lu=" << kTargetUmax << '\n'
           << "phase_offset=N/4+" << kSubcellOffset << '\n'
           << "retraction_lu=" << kRetraction << '\n'
           << "marker_spacing_lu=" << kMarkerSpacing << '\n'
           << "support_thickness_lu=" << kSupportThickness << '\n'
           << "forcing_iterations=" << kForcingIterations << '\n'
           << "residual_tolerance_lu="
           << kToleranceFraction*kTargetUmax << '\n'
           << "ibm_case_limits=stored_per_row_in_summary.csv\n"
           << "pair_gate_minimum_l2_reduction=0.05\n"
           << "pair_gate_minimum_near_wall_l2_reduction=0.05\n"
           << "pair_gate_minimum_flux_error_reduction=0.10\n"
           << "ibm_gate_normal_leakage_rms_over_umax=0.02\n"
           << "ibm_gate_secondary_rms_over_umax=0.02\n"
           << "ibm_gate_design_normal_rms_over_umax=0.005\n"
           << "ibm_gate_abs_evolved_mass_relative_change=1e-4\n"
           << "ibm_gate_mach_max=0.08\n"
           << "ibm_gate_converged_step_fraction=0.95\n";
}

} // namespace

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
                "cannot select SGS-off periodic channel validation layout");

        const std::string suite = argv[2];
        const std::filesystem::path output_directory = argv[3];
        if(std::filesystem::exists(output_directory) &&
           !std::filesystem::is_empty(output_directory))
            throw std::runtime_error(
                "validation output directory already exists and is not empty");
        std::filesystem::create_directories(output_directory);
        std::ofstream summary(
            output_directory/"summary.csv", std::ios::out|std::ios::trunc);
        std::ofstream profiles(
            output_directory/"profiles.csv", std::ios::out|std::ios::trunc);
        if(!summary || !profiles)
            throw std::runtime_error("cannot create validation CSV files");
        summary << std::setprecision(10);
        profiles << std::setprecision(10);
        write_summary_header(summary);
        profiles
            << "size,orientation,method,precision,x,y,z,phase,s,s_over_W,"
               "wall_distance,u_exact_over_umax,u_primary_over_umax,"
               "u_normal_over_umax,u_secondary_over_umax,near_wall\n";

        const std::vector<CaseSpec> cases = make_cases(suite);
        std::vector<RunResult> results;
        results.reserve(cases.size());
        std::cout << "SurfaceIBM periodic 3-D channel validation suite="
                  << suite << " precision=" << kPrecision
                  << " cases=" << cases.size() << '\n';
        for(std::size_t i=0u; i<cases.size(); ++i) {
            const CaseSpec& spec = cases[i];
            std::cout << "CASE " << (i+1u) << '/' << cases.size()
                      << " N=" << spec.size
                      << " orientation=" << spec.orientation.name
                      << " method=" << method_name(spec.method) << '\n';
            RunResult result = run_case(spec, profiles);
            write_summary(summary, result);
            summary.flush();
            profiles.flush();
            std::cout << std::setprecision(6)
                      << "RESULT l2=" << result.metrics.vector_l2_relative
                      << " near_wall_l2="
                      << result.metrics.near_wall_l2_relative
                      << " flux="
                      << result.metrics.sampled_flux_relative_error
                      << " normal_rms="
                      << result.metrics.normal_leakage_rms_over_umax
                      << " design_slip="
                      << result.metrics.design_slip_rms_over_umax
                      << " mass="
                      << result.metrics.evolved_mass_relative_change
                      << " Ma=" << result.metrics.mach_max
                      << " causal_margin=" << result.causal_margin_lu
                      << " runtime_s=" << result.runtime_seconds
                      << " gate=" << result.physics_gate_pass << '\n';
            results.push_back(std::move(result));
        }
        write_comparisons(output_directory/"comparisons.csv", results);
        write_metadata(output_directory/"metadata.txt", suite, results.size());
        std::cout << "SurfaceIBM periodic 3-D channel validation completed.\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "SURFACE_IBM_3D_CHANNEL_VALIDATION_ERROR "
                  << error.what() << '\n';
        return 1;
    }
}
