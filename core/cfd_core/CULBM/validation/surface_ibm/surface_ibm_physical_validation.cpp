#include "lbm.hpp"

#include <algorithm>
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

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr float kNu = 0.1f;
constexpr uint kNz = 96u;

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
    Shape shape = Shape::Circle;
    bool ibm = false;
    bool cleared_only = false;
    float half_width = 4.0f;
    float angle_deg = 0.0f;
    float phase_x = 0.23f;
    float phase_y = -0.17f;
    float marker_spacing = 0.75f;
    float support_thickness = 0.75f;
    // A single, pre-registered half-lattice solid-side offset is used for all
    // validation geometries.  It compensates the hydrodynamic wall shift of
    // the symmetric three-point regularized delta without per-case fitting.
    float retraction = 0.5f;
    unsigned forcing_iterations = 32u;
    // The frozen verification profile uses a 0.5% Umax residual ceiling.  A
    // dedicated sensitivity suite exercises looser and tighter values without
    // changing the reported main profile.
    float tolerance_fraction = 5.0e-3f;
    float speed_scale = 1.0f;
    float diffusion_times = 5.0f;
    bool cold_start = false;
    std::string group = "main";
};

const char* method_name(const CaseSpec& spec) {
    if(spec.ibm) return "ibm";
    if(spec.cleared_only) return "cleared_only";
    return "voxel";
}

struct Geometry {
    Shape shape = Shape::Circle;
    float half_width = 4.0f;
    float angle_rad = 0.0f;
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

    bool inside(const float x, const float y) const {
        float xi = 0.0f, eta = 0.0f;
        local(x, y, xi, eta);
        if(shape==Shape::Circle)
            return xi*xi+eta*eta < half_width*half_width;
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
        return -std::min((double)half_width-ax, (double)half_width-ay);
    }
};

struct AnalyticField {
    Geometry geometry;
    double body_force = 0.0;
    double target_umax = 0.0;
    double continuous_flux = 0.0;
};

struct Metrics {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    double cross_rms = 0.0;
    double q_numeric = 0.0;
    double q_exact_discrete = 0.0;
    double q_exact_continuous = 0.0;
    double q_error_discrete = 0.0;
    double q_error_continuous = 0.0;
    double hydraulic_size = 0.0;
    double fitted_radius = std::numeric_limits<double>::quiet_NaN();
    double z_profile_change = 0.0;
    double central_flux_cv = 0.0;
    double central_flux_max_deviation = 0.0;
    double all_z_flux_cv = 0.0;
    double all_z_flux_max_deviation = 0.0;
    double marker_slip_rms = 0.0;
    double marker_slip_max = 0.0;
    double marker_normal_rms = 0.0;
    double marker_normal_max = 0.0;
    double marker_net_flux = 0.0;
    double global_density_mean_offset = 0.0;
    double fluid_mass_relative_drift = 0.0;
    double rho_min = std::numeric_limits<double>::infinity();
    double rho_max = -std::numeric_limits<double>::infinity();
    double mach_max = 0.0;
    double voxel_link_rms = std::numeric_limits<double>::quiet_NaN();
    double voxel_link_p95 = std::numeric_limits<double>::quiet_NaN();
    double voxel_link_max = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t sample_points = 0u;
};

struct RunResult {
    CaseSpec spec;
    uint nx = 0u;
    uint ny = 0u;
    uint nz = 0u;
    unsigned steps = 0u;
    std::size_t markers = 0u;
    std::uint64_t cleared_support_cells = 0u;
    culbm::surface_ibm::Statistics statistics;
    Metrics metrics;
    double checkpoint_flux = 0.0;
    double initial_fluid_mass = 0.0;
    double checkpoint_fluid_mass = 0.0;
    double steady_flux_change = 0.0;
    double runtime_seconds = 0.0;
    double reaction_z_mean = 0.0;
    double reaction_z_cv = 0.0;
};

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

double square_flux(const double a, const double force) {
    const double width = 2.0*a;
    double sum = 0.0;
    for(int m=1; m<=255; m+=2) {
        for(int n=1; n<=255; n+=2) {
            const double denominator = (double)m*(double)m*(double)n*(double)n*
                (((double)m/width)*((double)m/width)+
                 ((double)n/width)*((double)n/width));
            sum += 1.0/denominator;
        }
    }
    return 64.0*force*width*width*sum/
        ((double)kNu*std::pow(kPi, 6.0));
}

double exact_velocity(const AnalyticField& field, const float x,
                      const float y) {
    float xi = 0.0f, eta = 0.0f;
    field.geometry.local(x, y, xi, eta);
    if(field.geometry.shape==Shape::Circle) {
        const double r2 = (double)xi*(double)xi+(double)eta*(double)eta;
        return field.body_force*
            ((double)field.geometry.half_width*field.geometry.half_width-r2)/
            (4.0*(double)kNu);
    }
    return square_velocity((double)xi, (double)eta,
                           (double)field.geometry.half_width,
                           field.body_force);
}

AnalyticField make_field(const CaseSpec& spec, const uint nx,
                         const uint ny) {
    AnalyticField field;
    field.geometry.shape = spec.shape;
    field.geometry.half_width = spec.half_width;
    field.geometry.angle_rad = spec.angle_deg*(float)(kPi/180.0);
    field.geometry.center_x = 0.5f*(float)(nx-1u)+spec.phase_x;
    field.geometry.center_y = 0.5f*(float)(ny-1u)+spec.phase_y;
    field.geometry.ex_x = std::cos(field.geometry.angle_rad);
    field.geometry.ex_y = std::sin(field.geometry.angle_rad);
    field.geometry.ey_x = -field.geometry.ex_y;
    field.geometry.ey_y = field.geometry.ex_x;
    field.target_umax = 0.08*spec.speed_scale/(double)spec.half_width;
    if(spec.shape==Shape::Circle) {
        field.body_force = 4.0*(double)kNu*field.target_umax/
            ((double)spec.half_width*(double)spec.half_width);
        field.continuous_flux = kPi*field.body_force*
            std::pow((double)spec.half_width, 4.0)/(8.0*(double)kNu);
    } else {
        const double unit_center = square_velocity(
            0.0, 0.0, (double)spec.half_width, 1.0, 127);
        field.body_force = field.target_umax/unit_center;
        field.continuous_flux = square_flux(
            (double)spec.half_width, field.body_force);
    }
    return field;
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

std::vector<culbm::surface_ibm::Marker> make_markers(
    const AnalyticField& field, const CaseSpec& spec, const uint nz) {
    const float spacing = spec.marker_spacing;
    const float z0 = 2.0f;
    const float z1 = (float)nz-3.0f;
    const unsigned nz_bins = std::max(
        1u, (unsigned)std::ceil((z1-z0)/spacing));
    const float dz = (z1-z0)/(float)nz_bins;
    std::vector<culbm::surface_ibm::Marker> markers;
    if(spec.shape==Shape::Circle) {
        const unsigned nt = std::max(8u, (unsigned)std::ceil(
            2.0f*(float)kPi*spec.half_width/spacing));
        const float dt = 2.0f*(float)kPi/(float)nt;
        markers.reserve((std::size_t)nt*(std::size_t)nz_bins);
        for(unsigned k=0u; k<nz_bins; ++k) {
            const float z = z0+((float)k+0.5f)*dz;
            for(unsigned i=0u; i<nt; ++i) {
                const float angle = ((float)i+0.5f)*dt;
                const float nx = std::cos(angle);
                const float ny = std::sin(angle);
                culbm::surface_ibm::Marker marker;
                const float marker_radius =
                    spec.half_width+spec.retraction;
                marker.position_lu = float3(
                    field.geometry.center_x+marker_radius*nx,
                    field.geometry.center_y+marker_radius*ny, z);
                marker.wall_velocity_lu = float3(0.0f);
                marker.normal = float3(-nx, -ny, 0.0f);
                marker.area_lu2 = spec.half_width*dt*dz;
                marker.source_triangle = i;
                marker.global_id = (std::uint64_t)markers.size();
                markers.push_back(marker);
            }
        }
    } else {
        const unsigned ns = std::max(2u, (unsigned)std::ceil(
            2.0f*spec.half_width/spacing));
        const float ds = 2.0f*spec.half_width/(float)ns;
        markers.reserve(4u*(std::size_t)ns*(std::size_t)nz_bins);
        const auto add_wall = [&](const float sign, const bool xi_wall,
                                  const float normal_sign) {
            for(unsigned k=0u; k<nz_bins; ++k) {
                const float z = z0+((float)k+0.5f)*dz;
                for(unsigned i=0u; i<ns; ++i) {
                    const float varying = -spec.half_width+
                        ((float)i+0.5f)*ds;
                    const float fixed = sign*
                        (spec.half_width+spec.retraction);
                    const float xi = xi_wall ? fixed : varying;
                    const float eta = xi_wall ? varying : fixed;
                    culbm::surface_ibm::Marker marker;
                    marker.position_lu = float3(
                        field.geometry.center_x+
                            xi*field.geometry.ex_x+eta*field.geometry.ey_x,
                        field.geometry.center_y+
                            xi*field.geometry.ex_y+eta*field.geometry.ey_y,
                        z);
                    marker.wall_velocity_lu = float3(0.0f);
                    marker.normal = xi_wall
                        ? float3(normal_sign*field.geometry.ex_x,
                                 normal_sign*field.geometry.ex_y, 0.0f)
                        : float3(normal_sign*field.geometry.ey_x,
                                 normal_sign*field.geometry.ey_y, 0.0f);
                    marker.area_lu2 = ds*dz;
                    marker.source_triangle = (std::uint32_t)markers.size();
                    marker.global_id = (std::uint64_t)markers.size();
                    markers.push_back(marker);
                }
            }
        };
        add_wall(1.0f, true, -1.0f);
        add_wall(-1.0f, true, 1.0f);
        add_wall(1.0f, false, -1.0f);
        add_wall(-1.0f, false, 1.0f);
    }
    return markers;
}

void configure_runtime(const CaseSpec& spec, const double target_umax) {
    if(spec.ibm && spec.cleared_only)
        throw std::runtime_error(
            "validation case cannot enable IBM and cleared-only together");
    culbm::surface_ibm::Config config;
    config.enabled = spec.ibm;
    config.geometry_policy = culbm::surface_ibm::GeometryPolicy::AllTriangles;
    config.marker_spacing_lu = spec.marker_spacing;
    config.forcing_iterations = spec.forcing_iterations;
    config.relaxation = 1.0f;
    config.residual_tolerance = (float)std::max(
        1.0e-8, spec.tolerance_fraction*target_umax);
    config.retraction_lu = spec.retraction;
    config.support_thickness_lu = spec.support_thickness;
    config.max_markers = 1000000u;
    config.strict_geometry = true;
    std::string error;
    if(!culbm::surface_ibm::validate_config(config, error))
        throw std::runtime_error("invalid validation SurfaceIBM config: "+error);
    culbm::surface_ibm::set_config(config);
}

std::uint64_t clear_marker_support_solids(
    LBM& lbm,
    const std::vector<culbm::surface_ibm::Marker>& markers) {
    std::uint64_t cleared = 0u;
    for(const auto& marker : markers) {
        const int cx = (int)std::floor((double)marker.position_lu.x+0.5);
        const int cy = (int)std::floor((double)marker.position_lu.y+0.5);
        const int cz = (int)std::floor((double)marker.position_lu.z+0.5);
        for(int dz=-1; dz<=1; ++dz) {
            const int z = cz+dz;
            const float wz = roma3_phi(marker.position_lu.z-(float)z);
            for(int dy=-1; dy<=1; ++dy) {
                const int y = cy+dy;
                const float wy = roma3_phi(marker.position_lu.y-(float)y);
                for(int dx=-1; dx<=1; ++dx) {
                    const int x = cx+dx;
                    const float w = roma3_phi(
                        marker.position_lu.x-(float)x)*wy*wz;
                    if(w==0.0f) continue;
                    if(x<0 || y<0 || z<0 || x>=(int)lbm.get_Nx() ||
                       y>=(int)lbm.get_Ny() || z>=(int)lbm.get_Nz())
                        throw std::runtime_error(
                            "cleared-only marker support leaves lattice");
                    uchar& flag = lbm.flags[lbm.index(
                        (uint)x, (uint)y, (uint)z)];
                    if((flag&TYPE_S)!=0u) {
                        flag = (uchar)(flag&~TYPE_S);
                        ++cleared;
                    }
                }
            }
        }
    }
    return cleared;
}

double section_flux_from_host(LBM& lbm, const AnalyticField& field,
                              const uint z) {
    double flux = 0.0;
    for(uint y=0u; y<lbm.get_Ny(); ++y) {
        for(uint x=0u; x<lbm.get_Nx(); ++x) {
            if(!field.geometry.inside((float)x, (float)y)) continue;
            flux += (double)lbm.u.z[lbm.index(x, y, z)];
        }
    }
    return flux;
}

double mean_flux_from_host(LBM& lbm, const AnalyticField& field,
                           const uint z0, const uint z1) {
    if(z0>=z1 || z1>lbm.get_Nz())
        throw std::runtime_error("invalid validation flux interval");
    double flux = 0.0;
    for(uint z=z0; z<z1; ++z)
        flux += section_flux_from_host(lbm, field, z);
    return flux/(double)(z1-z0);
}

double active_fluid_mass_from_host(
    LBM& lbm, const std::vector<unsigned char>& active_fluid) {
    if(active_fluid.size()!=(std::size_t)lbm.get_N())
        throw std::runtime_error("invalid validation fluid mask");
    double mass = 0.0;
    for(ulong n=0ull; n<lbm.get_N(); ++n) {
        if(active_fluid[(std::size_t)n]!=0u) mass += (double)lbm.rho[n];
    }
    return mass;
}

float3 interpolate_velocity(const LBM& lbm,
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
                const ulong n = lbm.index(
                    (uint)(cx+dx), (uint)(cy+dy), (uint)(cz+dz));
                value.x += w*lbm.u.x[n];
                value.y += w*lbm.u.y[n];
                value.z += w*lbm.u.z[n];
                weight += w;
            }
        }
    }
    if(!(weight>0.0f)) throw std::runtime_error("zero marker interpolation weight");
    return value/weight;
}

void compute_link_geometry(Metrics& metrics, const AnalyticField& field,
                           const uint nx, const uint ny) {
    static constexpr int directions[4][2] = {
        {1, 0}, {0, 1}, {1, 1}, {1, -1}
    };
    metrics.voxel_link_rms = 0.0;
    metrics.voxel_link_p95 = 0.0;
    metrics.voxel_link_max = 0.0;
    std::vector<double> errors;
    for(uint y=1u; y+1u<ny; ++y) {
        for(uint x=1u; x+1u<nx; ++x) {
            const bool a = field.geometry.inside((float)x, (float)y);
            for(const auto& direction : directions) {
                const int xx = (int)x+direction[0];
                const int yy = (int)y+direction[1];
                const bool b = field.geometry.inside((float)xx, (float)yy);
                if(a==b) continue;
                const double mx = 0.5*((double)x+(double)xx);
                const double my = 0.5*((double)y+(double)yy);
                errors.push_back(std::fabs(
                    field.geometry.signed_distance(mx, my)));
            }
        }
    }
    if(errors.empty()) throw std::runtime_error("no voxel boundary links found");
    double sum2 = 0.0;
    for(const double error : errors) {
        sum2 += error*error;
        metrics.voxel_link_max = std::max(metrics.voxel_link_max, error);
    }
    metrics.voxel_link_rms = std::sqrt(sum2/(double)errors.size());
    std::sort(errors.begin(), errors.end());
    const std::size_t p95 = std::min(
        errors.size()-1u, (std::size_t)std::floor(0.95*(errors.size()-1u)));
    metrics.voxel_link_p95 = errors[p95];
}

Metrics compute_metrics(LBM& lbm, const AnalyticField& field,
                        const std::vector<culbm::surface_ibm::Marker>& markers,
                        const std::vector<unsigned char>& active_fluid,
                        const double initial_fluid_mass,
                        std::ofstream& profile, const CaseSpec& spec) {
    lbm.rho.read_from_device();
    lbm.u.read_from_device();
    Metrics metrics;
    const uint z0 = lbm.get_Nz()/4u;
    const uint z1 = 3u*lbm.get_Nz()/4u;
    const uint zmid = lbm.get_Nz()/2u;
    double abs_error = 0.0, abs_exact = 0.0;
    double error2 = 0.0, exact2 = 0.0, cross2 = 0.0;
    double profile_delta2 = 0.0, profile_exact2 = 0.0;
    double fit_s0 = 0.0, fit_sx = 0.0, fit_sxx = 0.0;
    double fit_sy = 0.0, fit_sxy = 0.0;
    const double center_z = 0.5*(double)(lbm.get_Nz()-1u);

    for(ulong n=0ull; n<lbm.get_N(); ++n) {
        const double rho = (double)lbm.rho[n];
        metrics.rho_min = std::min(metrics.rho_min, rho);
        metrics.rho_max = std::max(metrics.rho_max, rho);
        metrics.global_density_mean_offset += rho-1.0;
    }
    metrics.global_density_mean_offset /= (double)lbm.get_N();
    const double final_fluid_mass = active_fluid_mass_from_host(
        lbm, active_fluid);
    metrics.fluid_mass_relative_drift =
        (final_fluid_mass-initial_fluid_mass)/initial_fluid_mass;

    std::vector<double> all_z_fluxes;
    std::vector<double> central_fluxes;
    all_z_fluxes.reserve(lbm.get_Nz());
    central_fluxes.reserve(z1-z0);
    for(uint z=0u; z<lbm.get_Nz(); ++z) {
        const double flux = section_flux_from_host(lbm, field, z);
        all_z_fluxes.push_back(flux);
        if(z>=z0 && z<z1) central_fluxes.push_back(flux);
    }
    const auto flux_variation = [](const std::vector<double>& values,
                                   double& cv, double& maximum_deviation) {
        const double mean = std::accumulate(
            values.begin(), values.end(), 0.0)/(double)values.size();
        double variance = 0.0;
        maximum_deviation = 0.0;
        for(const double value : values) {
            const double delta = value-mean;
            variance += delta*delta;
            maximum_deviation = std::max(
                maximum_deviation, std::fabs(delta));
        }
        variance /= (double)values.size();
        const double denominator = std::max(std::fabs(mean), 1.0e-30);
        cv = std::sqrt(variance)/denominator;
        maximum_deviation /= denominator;
    };
    flux_variation(central_fluxes, metrics.central_flux_cv,
                   metrics.central_flux_max_deviation);
    flux_variation(all_z_fluxes, metrics.all_z_flux_cv,
                   metrics.all_z_flux_max_deviation);
    metrics.q_numeric = std::accumulate(
        central_fluxes.begin(), central_fluxes.end(), 0.0)/
        (double)central_fluxes.size();

    for(uint y=0u; y<lbm.get_Ny(); ++y) {
        for(uint x=0u; x<lbm.get_Nx(); ++x) {
            if(!field.geometry.inside((float)x, (float)y)) continue;
            const double exact = exact_velocity(field, (float)x, (float)y);
            metrics.q_exact_discrete += exact;
            double ux = 0.0, uy = 0.0, uz = 0.0;
            for(uint z=z0; z<z1; ++z) {
                const ulong n = lbm.index(x, y, z);
                const double ux_z = (double)lbm.u.x[n];
                const double uy_z = (double)lbm.u.y[n];
                const double uz_z = (double)lbm.u.z[n];
                const double error = uz_z-exact;
                abs_error += std::fabs(error);
                abs_exact += std::fabs(exact);
                error2 += error*error;
                exact2 += exact*exact;
                cross2 += ux_z*ux_z+uy_z*uy_z;
                metrics.linf = std::max(
                    metrics.linf, std::fabs(error)/field.target_umax);
                metrics.mach_max = std::max(
                    metrics.mach_max,
                    std::sqrt(ux_z*ux_z+uy_z*uy_z+uz_z*uz_z)*
                        std::sqrt(3.0));
                ++metrics.sample_points;
                ux += ux_z;
                uy += uy_z;
                uz += uz_z;
            }
            const double inverse_z = 1.0/(double)(z1-z0);
            ux *= inverse_z;
            uy *= inverse_z;
            uz *= inverse_z;
            const ulong n_quarter = lbm.index(x, y, z0);
            const ulong n_mid = lbm.index(x, y, zmid);
            const double delta = (double)lbm.u.z[n_quarter]-
                                 (double)lbm.u.z[n_mid];
            profile_delta2 += delta*delta;
            profile_exact2 += exact*exact;

            float xi = 0.0f, eta = 0.0f;
            field.geometry.local((float)x, (float)y, xi, eta);
            profile << spec.group << ','
                    << (spec.shape==Shape::Circle ? "circle" : "square") << ','
                    << method_name(spec) << ',' << kPrecision << ','
                    << spec.half_width << ',' << spec.angle_deg << ','
                    << spec.phase_x << ',' << spec.phase_y << ','
                    << (spec.cold_start ? 1 : 0) << ','
                    << x << ',' << y << ',' << xi << ',' << eta << ','
                    << exact/field.target_umax << ','
                    << uz/field.target_umax << ','
                    << ux/field.target_umax << ','
                    << uy/field.target_umax << '\n';

            if(spec.shape==Shape::Circle) {
                const double r2 = (double)xi*(double)xi+
                                  (double)eta*(double)eta;
                fit_s0 += 1.0;
                fit_sx += r2;
                fit_sxx += r2*r2;
                fit_sy += uz;
                fit_sxy += r2*uz;
            }
        }
    }
    if(metrics.sample_points==0u || !(exact2>0.0))
        throw std::runtime_error("empty validation sample");
    metrics.l1 = abs_error/abs_exact;
    metrics.l2 = std::sqrt(error2/exact2);
    metrics.cross_rms = std::sqrt(
        cross2/(double)metrics.sample_points)/field.target_umax;
    metrics.z_profile_change = std::sqrt(profile_delta2/profile_exact2);
    metrics.q_exact_continuous = field.continuous_flux;
    metrics.q_error_discrete = metrics.q_numeric/
        metrics.q_exact_discrete-1.0;
    metrics.q_error_continuous = metrics.q_numeric/
        metrics.q_exact_continuous-1.0;
    if(metrics.q_numeric>0.0 && metrics.q_exact_discrete>0.0)
        metrics.hydraulic_size = (double)spec.half_width*std::pow(
            metrics.q_numeric/metrics.q_exact_discrete, 0.25);
    if(spec.shape==Shape::Circle) {
        const double determinant = fit_s0*fit_sxx-fit_sx*fit_sx;
        if(std::fabs(determinant)>1.0e-20) {
            const double intercept =
                (fit_sy*fit_sxx-fit_sx*fit_sxy)/determinant;
            const double slope =
                (fit_s0*fit_sxy-fit_sx*fit_sy)/determinant;
            if(intercept>0.0 && slope<0.0)
                metrics.fitted_radius = std::sqrt(-intercept/slope);
        }
    }

    double slip2 = 0.0, normal2 = 0.0, area = 0.0;
    std::size_t marker_samples = 0u;
    for(const auto& marker : markers) {
        if(std::fabs((double)marker.position_lu.z-center_z)>
           0.25*(double)lbm.get_Nz()) continue;
        const float3 velocity = interpolate_velocity(lbm, marker);
        const double slip = std::sqrt(
            (double)velocity.x*velocity.x+
            (double)velocity.y*velocity.y+
            (double)velocity.z*velocity.z);
        const double normal =
            (double)velocity.x*marker.normal.x+
            (double)velocity.y*marker.normal.y+
            (double)velocity.z*marker.normal.z;
        slip2 += slip*slip;
        normal2 += normal*normal;
        metrics.marker_slip_max = std::max(metrics.marker_slip_max, slip);
        metrics.marker_normal_max = std::max(
            metrics.marker_normal_max, std::fabs(normal));
        metrics.marker_net_flux += normal*(double)marker.area_lu2;
        area += (double)marker.area_lu2;
        ++marker_samples;
    }
    if(marker_samples==0u) throw std::runtime_error("empty marker sample");
    metrics.marker_slip_rms = std::sqrt(slip2/(double)marker_samples)/
        field.target_umax;
    metrics.marker_slip_max /= field.target_umax;
    metrics.marker_normal_rms = std::sqrt(normal2/(double)marker_samples)/
        field.target_umax;
    metrics.marker_normal_max /= field.target_umax;
    metrics.marker_net_flux /= std::max(
        area*field.target_umax, std::numeric_limits<double>::min());
    // For cleared-only this remains the geometry error of the original voxel
    // mask, which is intentionally shared with its paired voxel control.
    if(!spec.ibm)
        compute_link_geometry(metrics, field, lbm.get_Nx(), lbm.get_Ny());
    return metrics;
}

RunResult run_case(const CaseSpec& spec, std::ofstream& profile) {
    const uint nx = (uint)std::ceil(4.0f*spec.half_width+8.0f);
    const uint ny = nx;
    const AnalyticField field = make_field(spec, nx, ny);
    configure_runtime(spec, field.target_umax);
    const auto forcing_markers = make_markers(field, spec, kNz);
    CaseSpec design_surface_spec = spec;
    design_surface_spec.retraction = 0.0f;
    const auto design_surface_markers = make_markers(
        field, design_surface_spec, kNz);
    // Apply the pressure-gradient surrogate only in the continuous design
    // fluid.  SurfaceIBM deliberately clears a narrow solid-side support band;
    // forcing those fictitious support cells would add a non-physical load.
    LBM lbm(nx, ny, kNz, kNu, 0.0f, 0.0f,
            0.0f, 0.0f, 0.01f, 0.0f);
    for(uint z=0u; z<kNz; ++z) {
        for(uint y=0u; y<ny; ++y) {
            for(uint x=0u; x<nx; ++x) {
                const ulong n = lbm.index(x, y, z);
                const bool fluid = field.geometry.inside((float)x, (float)y);
                lbm.rho[n] = 1.0f;
                lbm.u.x[n] = 0.0f;
                lbm.u.y[n] = 0.0f;
                lbm.u.z[n] = fluid && !spec.cold_start
                    ? (float)exact_velocity(field, (float)x, (float)y)
                    : 0.0f;
                lbm.flags[n] = fluid ? (uchar)0u : TYPE_S;
                lbm.F.x[n] = 0.0f;
                lbm.F.y[n] = 0.0f;
                lbm.F.z[n] = fluid ? (float)field.body_force : 0.0f;
                lbm.T[n] = 1.0f;
            }
        }
    }
    std::uint64_t cleared_support_cells = 0u;
    if(spec.cleared_only)
        cleared_support_cells = clear_marker_support_solids(
            lbm, forcing_markers);
    if(spec.ibm) lbm.attach_surface_ibm(forcing_markers);
    std::vector<unsigned char> active_fluid((std::size_t)lbm.get_N(), 0u);
    double initial_fluid_mass = 0.0;
    for(ulong n=0ull; n<lbm.get_N(); ++n) {
        const bool active = (lbm.flags[n]&TYPE_S)==0u;
        active_fluid[(std::size_t)n] = active ? 1u : 0u;
        if(active) initial_fluid_mass += (double)lbm.rho[n];
    }
    if(!(initial_fluid_mass>0.0))
        throw std::runtime_error("empty active-fluid validation mask");
    const unsigned steps = std::max(100u, (unsigned)std::ceil(
        spec.diffusion_times*(double)spec.half_width*spec.half_width/
        (double)kNu));
    const unsigned checkpoint = 4u*steps/5u;
    const unsigned reaction_window = std::max(50u, steps/10u);
    double checkpoint_flux = std::numeric_limits<double>::quiet_NaN();
    double checkpoint_fluid_mass = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> reactions;
    reactions.reserve(reaction_window);
    lbm.external_initialize();
    const auto begin = std::chrono::steady_clock::now();
    for(unsigned step=1u; step<=steps; ++step) {
        lbm.external_step();
        if(spec.ibm && step+reaction_window>steps) {
            const auto statistics = lbm.surface_ibm_statistics();
            reactions.push_back((double)statistics.reaction_force_lu.z);
        }
        if(step==checkpoint) {
            lbm.external_finish();
            lbm.u.read_from_device();
            lbm.rho.read_from_device();
            checkpoint_flux = mean_flux_from_host(
                lbm, field, kNz/4u, 3u*kNz/4u);
            checkpoint_fluid_mass = active_fluid_mass_from_host(
                lbm, active_fluid);
        }
    }
    lbm.external_finish();
    const auto end = std::chrono::steady_clock::now();
    lbm.u.read_from_device();
    RunResult result;
    result.spec = spec;
    result.nx = nx;
    result.ny = ny;
    result.nz = kNz;
    result.steps = steps;
    result.markers = forcing_markers.size();
    result.metrics = compute_metrics(
        lbm, field, design_surface_markers, active_fluid,
        initial_fluid_mass, profile, spec);
    result.checkpoint_flux = checkpoint_flux;
    result.initial_fluid_mass = initial_fluid_mass;
    result.checkpoint_fluid_mass = checkpoint_fluid_mass;
    result.steady_flux_change = std::fabs(
        result.metrics.q_numeric-checkpoint_flux)/std::max(
            std::fabs(result.metrics.q_numeric), 1.0e-30);
    result.runtime_seconds = std::chrono::duration<double>(end-begin).count();
    result.statistics = lbm.surface_ibm_statistics();
    result.cleared_support_cells = spec.ibm
        ? result.statistics.cleared_solid_cells
        : cleared_support_cells;
    if(!reactions.empty()) {
        result.reaction_z_mean = std::accumulate(
            reactions.begin(), reactions.end(), 0.0)/(double)reactions.size();
        double variance = 0.0;
        for(const double reaction : reactions) {
            const double delta = reaction-result.reaction_z_mean;
            variance += delta*delta;
        }
        if(reactions.size()>1u) variance /= (double)(reactions.size()-1u);
        result.reaction_z_cv = std::sqrt(variance)/std::max(
            std::fabs(result.reaction_z_mean), 1.0e-30);
    }
    return result;
}

void write_header(std::ofstream& summary) {
    summary
        << "group,shape,method,precision,half_width,angle_deg,phase_x,phase_y,"
           "marker_spacing,support_thickness,retraction,forcing_iterations,tolerance_fraction,"
           "speed_scale,diffusion_times,cold_start,nx,ny,nz,steps,markers,active_cells,"
           "cleared_solid_cells,"
           "runtime_s,target_umax,l1,l2,linf,q_numeric,q_exact_discrete,"
           "q_exact_continuous,q_error_discrete,q_error_continuous,checkpoint_flux,"
           "initial_fluid_mass,checkpoint_fluid_mass,"
           "steady_flux_change,hydraulic_size,"
           "fitted_radius,cross_rms,z_profile_change,central_flux_cv,"
           "central_flux_max_deviation,all_z_flux_cv,all_z_flux_max_deviation,"
           "marker_slip_rms,"
           "marker_slip_max,marker_normal_rms,marker_normal_max,marker_net_flux,"
           "global_density_mean_offset,fluid_mass_relative_drift,"
           "rho_min,rho_max,mach_max,voxel_link_rms,voxel_link_p95,"
           "voxel_link_max,ibm_valid,ibm_converged,ibm_iterations,"
           "ibm_nonconverged_steps,ibm_peak_slip,reaction_z_mean,reaction_z_cv,"
           "sample_points\n";
}

void write_result(std::ofstream& summary, const RunResult& result) {
    const auto& s = result.spec;
    const auto& m = result.metrics;
    const double target_umax = 0.08*(double)s.speed_scale/(double)s.half_width;
    summary << s.group << ','
            << (s.shape==Shape::Circle ? "circle" : "square") << ','
            << method_name(s) << ',' << kPrecision << ','
            << s.half_width << ',' << s.angle_deg << ','
            << s.phase_x << ',' << s.phase_y << ','
            << s.marker_spacing << ',' << s.support_thickness << ','
            << s.retraction << ','
            << s.forcing_iterations << ',' << s.tolerance_fraction << ','
            << s.speed_scale << ',' << s.diffusion_times << ','
            << (s.cold_start ? 1 : 0) << ','
            << result.nx << ',' << result.ny << ',' << result.nz << ','
            << result.steps << ',' << result.markers << ','
            << result.statistics.active_cells << ','
            << result.cleared_support_cells << ','
            << result.runtime_seconds << ','
            << target_umax << ',' << m.l1 << ',' << m.l2 << ',' << m.linf << ','
            << m.q_numeric << ',' << m.q_exact_discrete << ','
            << m.q_exact_continuous << ',' << m.q_error_discrete << ','
            << m.q_error_continuous << ',' << result.checkpoint_flux << ','
            << result.initial_fluid_mass << ','
            << result.checkpoint_fluid_mass << ','
            << result.steady_flux_change << ',' << m.hydraulic_size << ','
            << m.fitted_radius << ',' << m.cross_rms << ','
            << m.z_profile_change << ',' << m.central_flux_cv << ','
            << m.central_flux_max_deviation << ',' << m.all_z_flux_cv << ','
            << m.all_z_flux_max_deviation << ',' << m.marker_slip_rms << ','
            << m.marker_slip_max << ',' << m.marker_normal_rms << ','
            << m.marker_normal_max << ',' << m.marker_net_flux << ','
            << m.global_density_mean_offset << ','
            << m.fluid_mass_relative_drift << ','
            << m.rho_min << ',' << m.rho_max << ','
            << m.mach_max << ',' << m.voxel_link_rms << ','
            << m.voxel_link_p95 << ',' << m.voxel_link_max << ','
            << (result.statistics.valid ? 1 : 0) << ','
            << (result.statistics.converged ? 1 : 0) << ','
            << result.statistics.iterations << ','
            << result.statistics.nonconverged_steps << ','
            << result.statistics.peak_max_slip_lu << ','
            << result.reaction_z_mean << ',' << result.reaction_z_cv << ','
            << m.sample_points << '\n';
}

std::vector<CaseSpec> suite_cases(const std::string& suite) {
    std::vector<CaseSpec> cases;
    const auto add_pair = [&](CaseSpec spec) {
        spec.group = suite;
        spec.ibm = false;
        cases.push_back(spec);
        spec.ibm = true;
        cases.push_back(spec);
    };
    if(suite=="pilot") {
        CaseSpec circle;
        circle.shape = Shape::Circle;
        circle.half_width = 4.0f;
        add_pair(circle);
        CaseSpec square;
        square.shape = Shape::Square;
        square.half_width = 4.0f;
        square.angle_deg = 45.0f;
        add_pair(square);
        return cases;
    }
    if(suite=="main") {
        for(const float width : {4.0f, 8.0f, 16.0f}) {
            CaseSpec circle;
            circle.shape = Shape::Circle;
            circle.half_width = width;
            add_pair(circle);
        }
        for(const float width : {4.0f, 8.0f, 16.0f}) {
            for(const float angle : {0.0f, 15.0f, 30.0f, 45.0f}) {
                CaseSpec square;
                square.shape = Shape::Square;
                square.half_width = width;
                square.angle_deg = angle;
                add_pair(square);
            }
        }
        for(const auto phase : std::vector<std::pair<float, float>>{
                {0.0f, 0.0f}, {0.25f, 0.25f}, {0.5f, 0.0f}, {0.5f, 0.5f}}) {
            CaseSpec circle;
            circle.shape = Shape::Circle;
            circle.half_width = 8.0f;
            circle.phase_x = phase.first;
            circle.phase_y = phase.second;
            circle.group = "phase";
            circle.ibm = false;
            cases.push_back(circle);
            circle.ibm = true;
            cases.push_back(circle);
        }
        return cases;
    }
    if(suite=="production") {
        for(const float width : {8.0f, 16.0f}) {
            CaseSpec circle;
            circle.shape = Shape::Circle;
            circle.half_width = width;
            add_pair(circle);
            for(const float angle : {0.0f, 30.0f, 45.0f}) {
                CaseSpec square;
                square.shape = Shape::Square;
                square.half_width = width;
                square.angle_deg = angle;
                add_pair(square);
            }
        }
        return cases;
    }
    if(suite=="convergence") {
        for(const float width : {8.0f, 16.0f, 24.0f}) {
            CaseSpec circle;
            circle.shape = Shape::Circle;
            circle.half_width = width;
            add_pair(circle);
            for(const float angle : {30.0f, 45.0f}) {
                CaseSpec square;
                square.shape = Shape::Square;
                square.half_width = width;
                square.angle_deg = angle;
                add_pair(square);
            }
        }
        return cases;
    }
    if(suite=="coldstart") {
        for(const Shape shape : {Shape::Circle, Shape::Square}) {
            for(const bool cold : {false, true}) {
                CaseSpec spec;
                spec.shape = shape;
                spec.half_width = 8.0f;
                spec.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
                spec.ibm = true;
                spec.cold_start = cold;
                spec.group = "coldstart";
                cases.push_back(spec);
            }
        }
        return cases;
    }
    if(suite=="controls") {
        for(const Shape shape : {Shape::Circle, Shape::Square}) {
            CaseSpec spec;
            spec.shape = shape;
            spec.half_width = 8.0f;
            spec.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
            spec.cold_start = true;
            spec.group = "controls";
            cases.push_back(spec);
            spec.cleared_only = true;
            cases.push_back(spec);
            spec.cleared_only = false;
            spec.ibm = true;
            cases.push_back(spec);
        }
        return cases;
    }
    if(suite=="tolerance") {
        for(const Shape shape : {Shape::Circle, Shape::Square}) {
            for(const float tolerance : {
                    1.0e-2f, 5.0e-3f, 1.0e-3f, 1.0e-4f}) {
                CaseSpec spec;
                spec.shape = shape;
                spec.half_width = 8.0f;
                spec.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
                spec.ibm = true;
                spec.tolerance_fraction = tolerance;
                spec.group = "tolerance";
                cases.push_back(spec);
            }
        }
        return cases;
    }
    if(suite=="calibration") {
        for(const Shape shape : {Shape::Circle, Shape::Square}) {
            CaseSpec voxel;
            voxel.shape = shape;
            voxel.half_width = 4.0f;
            voxel.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
            voxel.group = "calibration";
            voxel.ibm = false;
            cases.push_back(voxel);
            for(const float retraction : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
                CaseSpec spec = voxel;
                spec.ibm = true;
                spec.retraction = retraction;
                cases.push_back(spec);
            }
        }
        return cases;
    }
    if(suite=="sensitivity") {
        for(const Shape shape : {Shape::Circle, Shape::Square}) {
            for(const float spacing : {0.5f, 0.75f, 1.0f, 1.25f}) {
                CaseSpec spec;
                spec.shape = shape;
                spec.half_width = 8.0f;
                spec.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
                spec.marker_spacing = spacing;
                spec.ibm = true;
                spec.group = "spacing";
                cases.push_back(spec);
            }
            for(const unsigned iterations : {8u, 16u, 32u}) {
                CaseSpec spec;
                spec.shape = shape;
                spec.half_width = 8.0f;
                spec.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
                spec.forcing_iterations = iterations;
                spec.ibm = true;
                spec.group = "iterations";
                cases.push_back(spec);
            }
            for(const float thickness : {0.5f, 0.75f, 1.0f}) {
                CaseSpec spec;
                spec.shape = shape;
                spec.half_width = 8.0f;
                spec.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
                spec.support_thickness = thickness;
                spec.ibm = true;
                spec.group = "thickness";
                cases.push_back(spec);
            }
            for(const float retraction : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
                CaseSpec spec;
                spec.shape = shape;
                spec.half_width = 8.0f;
                spec.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
                spec.retraction = retraction;
                spec.ibm = true;
                spec.group = "retraction";
                cases.push_back(spec);
            }
            for(const float speed : {0.5f, 1.0f}) {
                CaseSpec spec;
                spec.shape = shape;
                spec.half_width = 8.0f;
                spec.angle_deg = shape==Shape::Square ? 45.0f : 0.0f;
                spec.speed_scale = speed;
                spec.ibm = true;
                spec.group = "mach";
                cases.push_back(spec);
            }
        }
        return cases;
    }
    throw std::runtime_error("unknown validation suite: "+suite);
}

} // namespace

int main(const int argc, const char* const argv[]) {
    try {
        if(argc!=4 || std::string(argv[1])!="--suite") {
            std::cerr << "Usage: " << argv[0]
                      << " --suite pilot|calibration|main|production|convergence|"
                         "coldstart|controls|tolerance|sensitivity OUTPUT_DIR\n";
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
            throw std::runtime_error("cannot select SGS-off validation layout");

        const std::string suite = argv[2];
        const std::filesystem::path output = argv[3];
        if(std::filesystem::exists(output) &&
           !std::filesystem::is_empty(output))
            throw std::runtime_error(
                "validation output directory already exists and is not empty");
        std::filesystem::create_directories(output);
        std::ofstream summary(output/"summary.csv", std::ios::out|std::ios::trunc);
        std::ofstream profile(output/"profiles.csv", std::ios::out|std::ios::trunc);
        if(!summary || !profile)
            throw std::runtime_error("cannot create validation output files");
        summary << std::setprecision(10);
        profile << std::setprecision(10);
        write_header(summary);
        profile << "group,shape,method,precision,half_width,angle_deg,phase_x,"
                   "phase_y,cold_start,x,y,xi,eta,u_exact_norm,u_numeric_norm,"
                   "ux_norm,uy_norm\n";
        const auto cases = suite_cases(suite);
        std::cout << "SurfaceIBM physical validation suite=" << suite
                  << " precision=" << kPrecision
                  << " cases=" << cases.size() << '\n';
        for(std::size_t i=0u; i<cases.size(); ++i) {
            const auto& spec = cases[i];
            std::cout << "CASE " << (i+1u) << '/' << cases.size()
                      << " shape="
                      << (spec.shape==Shape::Circle ? "circle" : "square")
                      << " method=" << method_name(spec)
                      << " a=" << spec.half_width
                      << " angle=" << spec.angle_deg
                      << " spacing=" << spec.marker_spacing
                      << " iterations=" << spec.forcing_iterations << '\n';
            const RunResult result = run_case(spec, profile);
            write_result(summary, result);
            summary.flush();
            profile.flush();
            std::cout << std::setprecision(6)
                      << "RESULT l2=" << result.metrics.l2
                      << " qerr=" << result.metrics.q_error_discrete
                      << " slip_rms=" << result.metrics.marker_slip_rms
                      << " fluid_mass_drift="
                      << result.metrics.fluid_mass_relative_drift
                      << " Ma=" << result.metrics.mach_max
                      << " nonconverged="
                      << result.statistics.nonconverged_steps
                      << " runtime_s=" << result.runtime_seconds << '\n';
        }
        std::ofstream metadata(output/"metadata.txt", std::ios::out|std::ios::trunc);
        metadata << "suite=" << suite << '\n'
                 << "precision=" << kPrecision << '\n'
                 << "case_count=" << cases.size() << '\n'
                 << "nu_lu=" << kNu << '\n'
                 << "nz=" << kNz << '\n'
                 << "lattice=D3Q19\n"
                 << "collision=SRT\n"
                 << "subgrid=off\n"
                 << "surface_ibm_retraction_lu=0.5\n"
                 << "surface_ibm_default_tolerance_fraction=0.005\n"
                 << "cleared_only=Roma3_nonzero_support_TYPE_S_clear_without_IBM_force\n";
        std::cout << "SurfaceIBM physical validation completed.\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "SURFACE_IBM_PHYSICAL_VALIDATION_ERROR "
                  << error.what() << '\n';
        return 1;
    }
}
