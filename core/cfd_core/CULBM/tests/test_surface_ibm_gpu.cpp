#include "solver.hpp"
#include "surface_ibm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
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

constexpr uint kNx = 24u;
constexpr uint kNy = 24u;
constexpr uint kNz = 24u;
constexpr float kInitialSpeed = 0.02f;
constexpr float kInverseSqrtTwo = 0.7071067811865475244f;

struct TestState {
    int failures = 0;

    void expect(const bool condition, const std::string& message) {
        if(condition) return;
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }

    void expect_close(const double actual, const double expected,
                      const double tolerance, const std::string& message) {
        if(std::isfinite(actual) &&
           std::fabs(actual-expected)<=tolerance) return;
        ++failures;
        std::cerr << "FAIL: " << message << " (actual=" << actual
                  << ", expected=" << expected
                  << ", tolerance=" << tolerance << ")\n";
    }
};

struct Snapshot {
    culbm::surface_ibm::Statistics statistics;
    std::vector<float> density;
    std::vector<float> velocity;
    bool requested_topology = false;
    bool distributed = false;
    bool distinct_devices = true;
    bool force_field_preserved = false;
    bool center_solid_cleared = false;
    bool fields_finite = true;
    float minimum_density = std::numeric_limits<float>::infinity();
    float maximum_density = -std::numeric_limits<float>::infinity();
    float cpu_max_slip = 0.0f;
};

float dot3(const float3& a, const float3& b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

float roma3_phi(const float r) {
    const float a = std::fabs(r);
    if(a>=1.5f) return 0.0f;
    if(a<=0.5f) {
        return (1.0f+std::sqrt(std::max(0.0f, 1.0f-3.0f*a*a)))/3.0f;
    }
    const float q = 1.0f-a;
    return (5.0f-3.0f*a-
            std::sqrt(std::max(0.0f, 1.0f-3.0f*q*q)))/6.0f;
}

std::vector<culbm::surface_ibm::Marker> make_oblique_markers() {
    const float3 normal(kInverseSqrtTwo, kInverseSqrtTwo, 0.0f);
    const float3 tangent_1(kInverseSqrtTwo, -kInverseSqrtTwo, 0.0f);
    const float3 tangent_2(0.0f, 0.0f, 1.0f);
    const float3 center(12.0f, 12.0f, 12.0f);
    std::vector<culbm::surface_ibm::Marker> markers;
    markers.reserve(25u);
    for(int j=-2; j<=2; ++j) {
        for(int i=-2; i<=2; ++i) {
            culbm::surface_ibm::Marker marker;
            marker.position_lu = center+(float)i*tangent_1+(float)j*tangent_2;
            marker.wall_velocity_lu = float3(0.0f);
            marker.normal = normal;
            marker.area_lu2 = 1.0f;
            marker.source_triangle = 0u;
            marker.global_id = (std::uint64_t)markers.size();
            markers.push_back(marker);
        }
    }
    return markers;
}

std::vector<culbm::surface_ibm::Marker> make_sphere_markers() {
    constexpr std::uint32_t count = 128u;
    constexpr float radius = 3.0f;
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float golden_angle = 2.39996322972865332223f;
    const float3 center(12.0f, 12.0f, 12.0f);
    std::vector<culbm::surface_ibm::Marker> markers;
    markers.reserve(count);
    for(std::uint32_t i=0u; i<count; ++i) {
        const float z = 1.0f-2.0f*((float)i+0.5f)/(float)count;
        const float radial = std::sqrt(std::max(0.0f, 1.0f-z*z));
        const float angle = golden_angle*(float)i;
        const float3 normal(radial*std::cos(angle),
                            radial*std::sin(angle), z);
        culbm::surface_ibm::Marker marker;
        marker.position_lu = center+radius*normal;
        marker.wall_velocity_lu = float3(0.0f);
        marker.normal = normal;
        marker.area_lu2 = 4.0f*kPi*radius*radius/(float)count;
        marker.source_triangle = i;
        marker.global_id = i;
        markers.push_back(marker);
    }
    return markers;
}

std::vector<culbm::surface_ibm::Marker> make_reduction_tail_markers(
    const std::size_t count) {
    std::vector<culbm::surface_ibm::Marker> markers;
    markers.reserve(count);
    for(std::size_t i=0u; i<count; ++i) {
        const float x = 11.0f+0.25f*(float)(i%9u);
        const float y = 11.0f+0.25f*(float)((i/9u)%8u);
        culbm::surface_ibm::Marker marker;
        marker.position_lu = float3(x, y, 12.25f);
        marker.wall_velocity_lu = float3(0.0f);
        marker.normal = float3(0.0f, 0.0f, 1.0f);
        marker.area_lu2 = 0.0625f;
        marker.source_triangle = (std::uint32_t)i;
        marker.global_id = (std::uint64_t)i;
        markers.push_back(marker);
    }
    return markers;
}

float cpu_max_marker_slip(
    culbm::solver::Solver& lbm,
    const std::vector<culbm::surface_ibm::Marker>& markers) {
    float maximum = 0.0f;
    for(const auto& marker : markers) {
        const int cx = (int)std::floor(marker.position_lu.x+0.5f);
        const int cy = (int)std::floor(marker.position_lu.y+0.5f);
        const int cz = (int)std::floor(marker.position_lu.z+0.5f);
        float3 interpolated(0.0f);
        float weight = 0.0f;
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
                    const ulong n = lbm.grid().index((uint)x, (uint)y, (uint)z);
                    interpolated.x += w*lbm.velocity().x[n];
                    interpolated.y += w*lbm.velocity().y[n];
                    interpolated.z += w*lbm.velocity().z[n];
                    weight += w;
                }
            }
        }
        interpolated = interpolated/weight;
        const float3 residual = marker.wall_velocity_lu-interpolated;
        maximum = std::max(maximum, std::sqrt(dot3(residual, residual)));
    }
    return maximum;
}

void configure_surface_ibm() {
    culbm::surface_ibm::Config configuration;
    configuration.enabled = true;
    configuration.marker_spacing_lu = 1.0f;
    configuration.forcing_iterations = 8u;
    configuration.relaxation = 1.0f;
    configuration.residual_tolerance = 1.0e-3f;
    configuration.support_thickness_lu = 0.75f;
    configuration.max_markers = 1000u;
    std::string error;
    if(!culbm::surface_ibm::validate_config(configuration, error)) {
        throw std::runtime_error("Invalid SurfaceIBM test config: "+error);
    }
    culbm::surface_ibm::set_config(configuration);

    // Keep the production SUBGRID parameter layout active while making its
    // physical contribution exactly zero. This exercises SurfaceIBM's required
    // position after all runtime SGS kernel parameters.
    Subgrid_Configuration subgrid;
    subgrid.enabled = true;
    subgrid.smagorinsky_constant = 0.0f;
    subgrid.scalar_coupling = false;
    subgrid.turbulent_prandtl = 1.0f;
    subgrid.fixed_eddy_viscosity = 0.0f;
    if(!set_subgrid_configuration(subgrid)) {
        throw std::runtime_error("Cannot select the SurfaceIBM test SGS layout");
    }
}

void configure_surface_ibm_benchmark(
    const bool enabled, const unsigned forcing_iterations) {
    culbm::surface_ibm::Config configuration;
    configuration.enabled = enabled;
    configuration.marker_spacing_lu = 1.0f;
    configuration.forcing_iterations = forcing_iterations;
    configuration.relaxation = 1.0f;
    // Keep the benchmark at the iteration ceiling.  Production cases should
    // use their validated tolerance and will often terminate earlier.
    configuration.residual_tolerance = 1.0e-8f;
    configuration.support_thickness_lu = 0.75f;
    configuration.max_markers = 10000u;
    std::string error;
    if(!culbm::surface_ibm::validate_config(configuration, error)) {
        throw std::runtime_error("Invalid SurfaceIBM benchmark config: "+error);
    }
    culbm::surface_ibm::set_config(configuration);

    Subgrid_Configuration subgrid;
    subgrid.enabled = true;
    subgrid.smagorinsky_constant = 0.0f;
    subgrid.scalar_coupling = false;
    subgrid.turbulent_prandtl = 1.0f;
    subgrid.fixed_eddy_viscosity = 0.0f;
    if(!set_subgrid_configuration(subgrid)) {
        throw std::runtime_error(
            "Cannot select the SurfaceIBM benchmark SGS layout");
    }
}

std::vector<culbm::surface_ibm::Marker> make_benchmark_markers(
    const uint grid_size) {
    const int half_span = std::min(40, (int)grid_size/2-8);
    const float3 normal(kInverseSqrtTwo, kInverseSqrtTwo, 0.0f);
    const float3 tangent_1(kInverseSqrtTwo, -kInverseSqrtTwo, 0.0f);
    const float3 tangent_2(0.0f, 0.0f, 1.0f);
    const float center_coordinate = 0.5f*(float)grid_size;
    const float3 center(center_coordinate, center_coordinate,
                        center_coordinate);
    std::vector<culbm::surface_ibm::Marker> markers;
    markers.reserve((std::size_t)(2*half_span+1)*(2*half_span+1));
    for(int j=-half_span; j<=half_span; ++j) {
        for(int i=-half_span; i<=half_span; ++i) {
            culbm::surface_ibm::Marker marker;
            marker.position_lu =
                center+(float)i*tangent_1+(float)j*tangent_2;
            marker.wall_velocity_lu = float3(0.0f);
            marker.normal = normal;
            marker.area_lu2 = 1.0f;
            marker.source_triangle = 0u;
            marker.global_id = (std::uint64_t)markers.size();
            markers.push_back(marker);
        }
    }
    return markers;
}

void run_surface_ibm_benchmark(const bool enabled,
                               const unsigned timed_steps,
                               const uint grid_size) {
    constexpr unsigned warmup_steps = 10u;
    const float3 normal(kInverseSqrtTwo, kInverseSqrtTwo, 0.0f);
    const auto markers = make_benchmark_markers(grid_size);
    culbm::solver::Solver lbm(grid_size, grid_size, grid_size, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(0.05f).thermal(0.01f, 0.0f));
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        lbm.density()[n] = 1.0f;
        lbm.velocity().x[n] = 0.01f*normal.x;
        lbm.velocity().y[n] = 0.01f*normal.y;
        lbm.velocity().z[n] = 0.0f;
        lbm.flags()[n] = (uchar)0u;
        lbm.force().x[n] = lbm.force().y[n] = lbm.force().z[n] = 0.0f;
        lbm.temperature()[n] = 1.0f;
    }
    if(enabled) culbm::surface_ibm::attach(lbm, markers);
    lbm.initialize_once();
    for(unsigned step=0u; step<warmup_steps; ++step) lbm.advance();
    lbm.synchronize();

    const auto begin = std::chrono::steady_clock::now();
    for(unsigned step=0u; step<timed_steps; ++step) lbm.advance();
    lbm.synchronize();
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end-begin).count();
    const double steps_per_second = (double)timed_steps/seconds;
    const double mlups = (double)lbm.grid().cells()*steps_per_second/1.0e6;
    const auto statistics = culbm::surface_ibm::statistics(lbm);
    if(!(seconds>0.0) || !std::isfinite(mlups) ||
       (enabled && (!statistics.valid || !statistics.device_resident))) {
        throw std::runtime_error("SurfaceIBM benchmark produced invalid results");
    }
    std::cout << std::fixed << std::setprecision(6)
              << "SURFACE_IBM_BENCHMARK mode=" << (enabled ? "on" : "off")
              << " cells=" << lbm.grid().cells()
              << " markers=" << (enabled ? statistics.markers : 0u)
              << " active=" << (enabled ? statistics.active_cells : 0u)
              << " forcing_iterations="
              << (enabled ? statistics.iterations : 0u)
              << " warmup_steps=" << warmup_steps
              << " timed_steps=" << timed_steps
              << " seconds=" << seconds
              << " steps_per_second=" << steps_per_second
              << " mlups=" << mlups << '\n';
}

Snapshot solve(const uint dx) {
    const std::vector<culbm::surface_ibm::Marker> markers =
        make_oblique_markers();
    culbm::solver::Solver lbm(kNx, kNy, kNz, dx, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(0.05f).thermal(0.01f, 0.0f));

    Snapshot result;
    result.requested_topology =
        lbm.grid().split(0u)==dx && lbm.grid().split(1u)==1u && lbm.grid().split(2u)==1u;
    result.distributed = lbm.distributed();
    if(dx>1u && !result.distributed) {
        result.distinct_devices =
            lbm.local(0u).device().ordinal() !=
            lbm.local(1u).device().ordinal();
    }

    const float3 normal(kInverseSqrtTwo, kInverseSqrtTwo, 0.0f);
    for(uint z=0u; z<kNz; ++z) {
        for(uint y=0u; y<kNy; ++y) {
            for(uint x=0u; x<kNx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                lbm.density()[n] = 1.0f;
                lbm.velocity().x[n] = kInitialSpeed*normal.x;
                lbm.velocity().y[n] = kInitialSpeed*normal.y;
                lbm.velocity().z[n] = 0.0f;
                lbm.flags()[n] = (uchar)0u;
                lbm.force().x[n] = 0.0f;
                lbm.force().y[n] = 0.0f;
                lbm.force().z[n] = 0.0f;
                lbm.temperature()[n] = 1.0f;
            }
        }
    }

    const ulong center = lbm.grid().index(12u, 12u, 12u);
    lbm.flags()[center] = CELL_SOLID;
    // Place the independent FORCE_FIELD probe inside the IBM support band so
    // the regression also catches accidental clearing of a co-located field.
    const ulong force_probe = center;
    constexpr float probe_fx = 1.0e-7f;
    constexpr float probe_fy = -2.0e-7f;
    constexpr float probe_fz = 3.0e-7f;
    lbm.force().x[force_probe] = probe_fx;
    lbm.force().y[force_probe] = probe_fy;
    lbm.force().z[force_probe] = probe_fz;

    culbm::surface_ibm::attach(lbm, markers);
    const auto attached = culbm::surface_ibm::statistics(lbm);
    std::uint64_t center_cleared = 0u;
    const uint center_domain = 1u; // x=12 belongs to domain 1 for 24/2.
    if(!result.distributed || lbm.owns(center_domain)) {
        center_cleared = (lbm.flags()[center]&CELL_SOLID)==0u ? 1u : 0u;
    }
    if(result.distributed) {
        culbm::mpi_allreduce_sum_uint64(&center_cleared, 1u);
    }
    result.center_solid_cleared = center_cleared==1u;
    if(attached.markers!=markers.size() || attached.valid ||
       attached.cleared_solid_cells!=1u) {
        throw std::runtime_error(
            "SurfaceIBM attach statistics or support clearing are inconsistent");
    }

    lbm.advance();
    lbm.advance();
    lbm.synchronize();
    result.statistics = culbm::surface_ibm::statistics(lbm);
    lbm.density().download();
    lbm.velocity().download();
    lbm.force().download();

    result.force_field_preserved =
        lbm.force().x[force_probe]==probe_fx &&
        lbm.force().y[force_probe]==probe_fy &&
        lbm.force().z[force_probe]==probe_fz;
    result.density.resize((std::size_t)lbm.grid().cells());
    result.velocity.resize(3u*(std::size_t)lbm.grid().cells());
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        const float density = lbm.density()[n];
        const float ux = lbm.velocity().x[n];
        const float uy = lbm.velocity().y[n];
        const float uz = lbm.velocity().z[n];
        result.density[(std::size_t)n] = density;
        result.velocity[(std::size_t)n] = ux;
        result.velocity[(std::size_t)lbm.grid().cells()+(std::size_t)n] = uy;
        result.velocity[2u*(std::size_t)lbm.grid().cells()+(std::size_t)n] = uz;
        result.fields_finite = result.fields_finite &&
            std::isfinite(density) && std::isfinite(ux) &&
            std::isfinite(uy) && std::isfinite(uz);
        result.minimum_density = std::min(result.minimum_density, density);
        result.maximum_density = std::max(result.maximum_density, density);
    }
    result.cpu_max_slip = cpu_max_marker_slip(lbm, markers);
    return result;
}

void check_thermal_wall_conflict(TestState& test, const uint dx) {
    const auto markers = make_oblique_markers();
    culbm::solver::Solver lbm(kNx, kNy, kNz, dx, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(0.05f).thermal(0.01f, 0.0f));
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        lbm.density()[n] = 1.0f;
        lbm.velocity().x[n] = 0.0f;
        lbm.velocity().y[n] = 0.0f;
        lbm.velocity().z[n] = 0.0f;
        lbm.flags()[n] = (uchar)0u;
        lbm.temperature()[n] = 1.0f;
    }
    const ulong center = lbm.grid().index(12u, 12u, 12u);
    lbm.flags()[center] = (uchar)(CELL_SOLID|CELL_THERMAL);
    bool rejected = false;
    try {
        culbm::surface_ibm::attach(lbm, markers);
    } catch(const std::runtime_error& error) {
        rejected = std::string(error.what()).find("CELL_SOLID|CELL_THERMAL") !=
            std::string::npos;
    }
    test.expect(rejected,
                "SurfaceIBM rejects a solid-temperature wall instead of changing its scalar semantics");
    test.expect(lbm.flags()[center]==(uchar)(CELL_SOLID|CELL_THERMAL),
                "failed thermal-wall attachment leaves the original flags unchanged");

    lbm.flags()[center] = CELL_EQUILIBRIUM;
    rejected = false;
    try {
        culbm::surface_ibm::attach(lbm, markers);
    } catch(const std::runtime_error& error) {
        rejected = std::string(error.what()).find("CELL_EQUILIBRIUM") !=
            std::string::npos;
    }
    test.expect(rejected,
                "SurfaceIBM rejects overlap with an equilibrium boundary");
    test.expect(lbm.flags()[center]==CELL_EQUILIBRIUM,
                "failed equilibrium-boundary attachment leaves flags unchanged");
}

void check_mpi_configuration_identity(TestState& test) {
    const auto markers = make_oblique_markers();
    culbm::solver::Solver lbm(kNx, kNy, kNz, 2u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(0.05f).thermal(0.01f, 0.0f));
    const culbm::surface_ibm::Config shared_config =
        culbm::surface_ibm::get_config();
    culbm::surface_ibm::Config rank_config = shared_config;
    if(culbm::mpi_rank()==1) rank_config.enabled = false;
    culbm::surface_ibm::set_config(rank_config);
    bool rejected = false;
    try {
        culbm::surface_ibm::attach(lbm, markers);
    } catch(const std::runtime_error& error) {
        rejected = std::string(error.what()).find("same complete Config") !=
            std::string::npos;
    }
    culbm::surface_ibm::set_config(shared_config);
    test.expect(rejected,
                "SurfaceIBM collectively rejects rank-divergent enablement");

    rank_config = shared_config;
    if(culbm::mpi_rank()==1) rank_config.forcing_iterations = 7u;
    culbm::surface_ibm::set_config(rank_config);
    rejected = false;
    try {
        culbm::surface_ibm::attach(lbm, markers);
    } catch(const std::runtime_error& error) {
        rejected = std::string(error.what()).find("same complete Config") !=
            std::string::npos;
    }
    // Restore one identical snapshot before any later LBM construction.
    culbm::surface_ibm::set_config(shared_config);
    test.expect(rejected,
                "SurfaceIBM collectively rejects a rank-divergent MPI Config");

    rejected = false;
    try {
        culbm::surface_ibm::verify_attach_preflight(
            shared_config, markers, culbm::mpi_rank()==1,
            false, true);
    } catch(const std::runtime_error& error) {
        rejected = std::string(error.what()).find("same LBM attachment state") !=
            std::string::npos;
    }
    test.expect(rejected,
                "SurfaceIBM collectively rejects rank-divergent lifecycle state");

    std::vector<culbm::surface_ibm::Marker> reordered = markers;
    if(culbm::mpi_rank()==1)
        std::reverse(reordered.begin(), reordered.end());
    bool accepted = true;
    try {
        culbm::surface_ibm::verify_attach_preflight(
            shared_config, reordered, false, false, true);
    } catch(const std::exception&) {
        accepted = false;
    }
    test.expect(accepted,
                "SurfaceIBM marker identity is independent of input ordering");

    std::vector<culbm::surface_ibm::Marker> changed = markers;
    if(culbm::mpi_rank()==1) changed.front().position_lu.x += 0.125f;
    rejected = false;
    try {
        culbm::surface_ibm::verify_attach_preflight(
            shared_config, changed, false, false, true);
    } catch(const std::runtime_error& error) {
        rejected = std::string(error.what()).find("same deterministic global marker set") !=
            std::string::npos;
    }
    test.expect(rejected,
                "SurfaceIBM collectively rejects rank-divergent marker payloads");
}

void check_curved_sphere(TestState& test) {
    const auto markers = make_sphere_markers();
    culbm::solver::Solver lbm(kNx, kNy, kNz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(0.05f).thermal(0.01f, 0.0f));
    const float3 center(12.0f, 12.0f, 12.0f);
    constexpr float incident_speed = 0.015f;
    for(uint z=0u; z<kNz; ++z) {
        for(uint y=0u; y<kNy; ++y) {
            for(uint x=0u; x<kNx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                const float3 offset((float)x-center.x, (float)y-center.y,
                                    (float)z-center.z);
                const bool solid = dot3(offset, offset)<=4.0f;
                lbm.density()[n] = 1.0f;
                lbm.velocity().x[n] = solid ? 0.0f : incident_speed;
                lbm.velocity().y[n] = 0.0f;
                lbm.velocity().z[n] = 0.0f;
                lbm.flags()[n] = solid ? CELL_SOLID : (uchar)0u;
                lbm.force().x[n] = 0.0f;
                lbm.force().y[n] = 0.0f;
                lbm.force().z[n] = 0.0f;
                lbm.temperature()[n] = 1.0f;
            }
        }
    }
    const ulong center_index = lbm.grid().index(12u, 12u, 12u);
    culbm::surface_ibm::attach(lbm, markers);
    const auto attached = culbm::surface_ibm::statistics(lbm);
    test.expect((lbm.flags()[center_index]&CELL_SOLID)!=0u,
                "curved SurfaceIBM preserves the deep voxel solid core");
    test.expect(attached.cleared_solid_cells>0u,
                "curved SurfaceIBM clears the overlapping force-support band");
    lbm.advance();
    lbm.synchronize();
    const auto statistics = culbm::surface_ibm::statistics(lbm);
    const float reaction_x = statistics.reaction_force_lu.x;
    const float transverse = std::sqrt(
        statistics.reaction_force_lu.y*statistics.reaction_force_lu.y+
        statistics.reaction_force_lu.z*statistics.reaction_force_lu.z);
    std::cout << "Curved-sphere SurfaceIBM: markers=" << statistics.markers
              << ", active=" << statistics.active_cells
              << ", cleared=" << attached.cleared_solid_cells
              << ", max/rms slip=" << statistics.max_slip_lu << '/'
              << statistics.rms_slip_lu << ", reaction="
              << statistics.reaction_force_lu.x << '/'
              << statistics.reaction_force_lu.y << '/'
              << statistics.reaction_force_lu.z << '\n';
    test.expect(statistics.valid && statistics.markers==markers.size() &&
                statistics.active_cells>0u &&
                statistics.active_cells<lbm.grid().cells() &&
                statistics.device_resident &&
                std::isfinite(statistics.max_slip_lu) &&
                statistics.max_slip_lu<incident_speed,
                "closed curved SurfaceIBM reduces marker slip from uniform flow");
    test.expect(std::isfinite(reaction_x) && reaction_x>0.0f &&
                std::isfinite(transverse) &&
                transverse<=0.05f*std::max(reaction_x, 1.0e-8f),
                "symmetric curved SurfaceIBM produces an opposing axial reaction");
}

void check_half_integer_support(TestState& test) {
    culbm::surface_ibm::Marker marker;
    marker.position_lu = float3(12.5f, 12.5f, 12.5f);
    marker.wall_velocity_lu = float3(0.0f);
    marker.normal = float3(1.0f, 0.0f, 0.0f);
    marker.area_lu2 = 1.0f;
    marker.global_id = 0u;
    std::vector<culbm::surface_ibm::Marker> markers{marker};
    culbm::solver::Solver lbm(kNx, kNy, kNz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(0.05f).thermal(0.01f, 0.0f));
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        lbm.density()[n] = 1.0f;
        lbm.velocity().x[n] = lbm.velocity().y[n] = lbm.velocity().z[n] = 0.0f;
        lbm.flags()[n] = (uchar)0u;
        lbm.force().x[n] = lbm.force().y[n] = lbm.force().z[n] = 0.0f;
        lbm.temperature()[n] = 1.0f;
    }
    for(uint z=12u; z<=14u; ++z)
        for(uint y=12u; y<=14u; ++y)
            for(uint x=12u; x<=14u; ++x)
                lbm.flags()[lbm.grid().index(x, y, z)] = CELL_SOLID;
    culbm::surface_ibm::attach(lbm, markers);
    std::uint64_t remaining = 0u;
    for(uint z=12u; z<=14u; ++z)
        for(uint y=12u; y<=14u; ++y)
            for(uint x=12u; x<=14u; ++x)
                if((lbm.flags()[lbm.grid().index(x, y, z)]&CELL_SOLID)!=0u) ++remaining;
    const auto statistics = culbm::surface_ibm::statistics(lbm);
    test.expect(statistics.cleared_solid_cells==8u && remaining==19u,
                "SurfaceIBM clears only nonzero Roma support at half-integer markers");
}

void check_reduction_tails(TestState& test) {
    constexpr std::size_t counts[] = {1u, 63u, 64u, 65u};
    for(const std::size_t count : counts) {
        const auto markers = make_reduction_tail_markers(count);
        culbm::solver::Solver lbm(kNx, kNy, kNz, 1u, 1u, 1u,
            culbm::solver::CaseCoefficients().viscosity(0.05f).thermal(0.01f, 0.0f));
        for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
            lbm.density()[n] = 1.0f;
            lbm.velocity().x[n] = 0.0f;
            lbm.velocity().y[n] = 0.0f;
            lbm.velocity().z[n] = 5.0e-4f;
            lbm.flags()[n] = (uchar)0u;
            lbm.force().x[n] = lbm.force().y[n] = 0.0f;
            lbm.force().z[n] = 1.0e-5f;
            lbm.temperature()[n] = 1.0f;
        }
        culbm::surface_ibm::attach(lbm, markers);
        for(unsigned step=0u; step<2u; ++step) {
            lbm.advance();
            lbm.synchronize();
            const auto statistics = culbm::surface_ibm::statistics(lbm);
            test.expect(statistics.valid && statistics.device_resident &&
                        statistics.markers==count &&
                        statistics.active_cells>0u &&
                        statistics.iterations==1u && statistics.converged &&
                        statistics.nonconverged_steps==0ull &&
                        std::isfinite(statistics.max_slip_lu) &&
                        statistics.max_slip_lu<=1.0e-3f &&
                        std::isfinite(statistics.reaction_force_lu.z) &&
                        statistics.reaction_force_lu.z>0.0f,
                        "SurfaceIBM reduction tail keeps a converged, nonzero "
                        "two-step reaction for marker count "+
                            std::to_string(count)+", step "+
                            std::to_string(step+1u));
        }
    }
}

void check_physics(TestState& test, const Snapshot& result,
                   const uint dx) {
    const float3 normal(kInverseSqrtTwo, kInverseSqrtTwo, 0.0f);
    const float3 tangent_1(kInverseSqrtTwo, -kInverseSqrtTwo, 0.0f);
    const float3 tangent_2(0.0f, 0.0f, 1.0f);
    const auto& statistics = result.statistics;
    const float normal_reaction = dot3(statistics.reaction_force_lu, normal);
    const float tangent_reaction_1 =
        dot3(statistics.reaction_force_lu, tangent_1);
    const float tangent_reaction_2 =
        dot3(statistics.reaction_force_lu, tangent_2);

    test.expect(result.requested_topology,
                "SurfaceIBM uses the requested domain topology");
    test.expect(result.center_solid_cleared,
                "SurfaceIBM clears CELL_SOLID inside marker support before initialization");
    test.expect(result.force_field_preserved,
                "SurfaceIBM keeps the existing FORCE_FIELD payload independent");
    test.expect(statistics.valid,
                "SurfaceIBM reports valid post-step diagnostics");
    test.expect(statistics.markers==25u,
                "SurfaceIBM reports every oblique-plane marker");
    test.expect(statistics.fragments>=statistics.markers,
                "SurfaceIBM creates at least one owned fragment per marker");
    test.expect(statistics.active_cells>0u &&
                statistics.active_cells<result.density.size(),
                "SurfaceIBM direct forcing uses a strict support-cell subset");
    test.expect(statistics.device_resident==(dx==1u),
                "SurfaceIBM selects device-resident mode only for one local GPU");
    if(dx>1u) {
        test.expect(statistics.fragments>statistics.markers,
                    "markers crossing the GPU partition create multiple fragments");
    }
    test.expect(statistics.iterations>=1u && statistics.iterations<=8u,
                "SurfaceIBM converges within the configured direct-forcing limit");
    test.expect(statistics.converged && statistics.max_slip_lu<=1.0e-3f,
                "coupling-normalized SurfaceIBM meets the requested residual tolerance");
    test.expect(statistics.peak_max_slip_lu>=statistics.max_slip_lu &&
                statistics.nonconverged_steps<=2u,
                "SurfaceIBM retains bounded cumulative non-convergence diagnostics");
    test.expect(std::isfinite(statistics.minimum_coupling) &&
                statistics.minimum_coupling>0.0f &&
                statistics.maximum_coupling>=statistics.minimum_coupling,
                "SurfaceIBM reports a finite positive marker-coupling normalization");
    test.expect(std::isfinite(statistics.max_slip_lu) &&
                statistics.max_slip_lu>0.0f &&
                statistics.max_slip_lu<0.0195f,
                "SurfaceIBM reduces maximum marker slip from uniform inflow");
    test.expect(std::isfinite(statistics.rms_slip_lu) &&
                statistics.rms_slip_lu<0.0190f,
                "SurfaceIBM reduces RMS marker slip from uniform inflow");
    test.expect(std::isfinite(normal_reaction) && normal_reaction>0.0f,
                "wall reaction opposes the incident normal flow");
    const float reaction_scale = std::max(std::fabs(normal_reaction), 1.0e-8f);
    test.expect(std::fabs(tangent_reaction_1)<=0.05f*reaction_scale &&
                std::fabs(tangent_reaction_2)<=0.05f*reaction_scale,
                "symmetric oblique wall has negligible tangential reaction");
    test.expect(result.fields_finite && result.minimum_density>0.5f &&
                result.maximum_density<1.5f,
                "SurfaceIBM keeps density and velocity finite and bounded");
    test.expect(result.cpu_max_slip<kInitialSpeed,
                "independent CPU Roma3 interpolation confirms slip reduction");
    test.expect_close(result.cpu_max_slip, statistics.max_slip_lu, 2.0e-4,
                      "runtime slip diagnostic matches CPU Roma3 interpolation");

    std::cout << dx << " GPU-domain SurfaceIBM: fragments="
              << statistics.fragments << ", active="
              << statistics.active_cells << ", iterations="
              << statistics.iterations << ", max/rms slip="
              << statistics.max_slip_lu << '/' << statistics.rms_slip_lu
              << ", CPU max slip=" << result.cpu_max_slip
              << ", reaction normal/tangent=" << normal_reaction << '/'
              << tangent_reaction_1 << '/' << tangent_reaction_2
              << ", rho range=" << result.minimum_density << '/'
              << result.maximum_density << '\n';
}

void compare_snapshots(TestState& test, const Snapshot& one,
                       const Snapshot& two) {
    test.expect(two.distinct_devices,
                "two-domain SurfaceIBM run uses two distinct CUDA devices");
    const auto compare = [&](const std::vector<float>& expected,
                             const std::vector<float>& actual,
                             const char* name) {
        test.expect(expected.size()==actual.size() && !expected.empty(),
                    std::string(name)+" snapshots have equal nonzero extent");
        if(expected.size()!=actual.size() || expected.empty()) return;
        double sum_squared = 0.0;
        double maximum = 0.0;
        for(std::size_t i=0u; i<expected.size(); ++i) {
            const double error =
                std::fabs((double)actual[i]-(double)expected[i]);
            maximum = std::max(maximum, error);
            sum_squared += error*error;
        }
        const double rms = std::sqrt(sum_squared/(double)expected.size());
        std::cout << "SurfaceIBM two-GPU " << name << " max/RMS="
                  << maximum << '/' << rms << '\n';
        test.expect(maximum<=2.0e-4,
                    std::string(name)+" two-GPU maximum error is bounded");
        test.expect(rms<=5.0e-5,
                    std::string(name)+" two-GPU RMS error is bounded");
    };
    compare(one.density, two.density, "density");
    compare(one.velocity, two.velocity, "velocity");
    test.expect_close(two.statistics.max_slip_lu,
                      one.statistics.max_slip_lu, 2.0e-4,
                      "one/two-GPU maximum marker slip agrees");
    test.expect_close(two.statistics.rms_slip_lu,
                      one.statistics.rms_slip_lu, 2.0e-4,
                      "one/two-GPU RMS marker slip agrees");
    const float3 normal(kInverseSqrtTwo, kInverseSqrtTwo, 0.0f);
    const double one_reaction = dot3(one.statistics.reaction_force_lu, normal);
    const double two_reaction = dot3(two.statistics.reaction_force_lu, normal);
    test.expect_close(two_reaction, one_reaction,
                      1.0e-5+1.0e-3*std::fabs(one_reaction),
                      "one/two-GPU normal reaction agrees");
}

} // namespace

int main(const int argc, const char* const argv[]) {
    try {
        if(cuInit(0u)!=CUDA_SUCCESS) {
            std::cout << "SKIP: CUDA driver is unavailable.\n";
            return 77;
        }
        int device_count = 0;
        if(cuDeviceGetCount(&device_count)!=CUDA_SUCCESS || device_count<=0) {
            std::cout << "SKIP: no CUDA device is available.\n";
            return 77;
        }
        const std::string mode = argc>=2 ? std::string(argv[1]) : "";
        const bool multi_gpu = argc==2 && mode=="--multi-gpu";
        const bool mpi_gpu = argc==2 && mode=="--mpi";
        const bool benchmark_off =
            (argc>=2 && argc<=5) && mode=="--benchmark-off";
        const bool benchmark_on =
            (argc>=2 && argc<=5) && mode=="--benchmark-on";
        if(argc>5 || (argc>=2 && !multi_gpu && !mpi_gpu &&
                     !benchmark_off && !benchmark_on)) {
            std::cerr << "Usage: " << argv[0]
                      << " [--multi-gpu|--mpi|--benchmark-off [steps] "
                         "[forcing_iterations] [grid_size]|--benchmark-on "
                         "[steps] [forcing_iterations] [grid_size]]\n";
            return 2;
        }
        if(multi_gpu && device_count<2) {
            std::cout << "SKIP: SurfaceIBM multi-GPU test requires two devices.\n";
            return 77;
        }

        if(benchmark_off || benchmark_on) {
            unsigned timed_steps = 50u;
            unsigned forcing_iterations = 8u;
            uint grid_size = 96u;
            if(argc>=3) {
                const unsigned long parsed = std::stoul(argv[2]);
                if(parsed==0ul ||
                   parsed>(unsigned long)std::numeric_limits<unsigned>::max())
                    throw std::runtime_error(
                        "SurfaceIBM benchmark steps must be in [1, UINT_MAX]");
                timed_steps = (unsigned)parsed;
            }
            if(argc>=4) {
                const unsigned long parsed = std::stoul(argv[3]);
                if(parsed==0ul || parsed>32ul)
                    throw std::runtime_error(
                        "SurfaceIBM benchmark forcing iterations must be in [1, 32]");
                forcing_iterations = (unsigned)parsed;
            }
            if(argc==5) {
                const unsigned long parsed = std::stoul(argv[4]);
                if(parsed<96ul || parsed>256ul)
                    throw std::runtime_error(
                        "SurfaceIBM benchmark grid size must be in [96, 256]");
                grid_size = (uint)parsed;
            }
            configure_surface_ibm_benchmark(
                benchmark_on, forcing_iterations);
            run_surface_ibm_benchmark(
                benchmark_on, timed_steps, grid_size);
            return 0;
        }

        configure_surface_ibm();
        TestState test;
        check_thermal_wall_conflict(test, mpi_gpu ? 2u : 1u);
        if(mpi_gpu) {
            check_mpi_configuration_identity(test);
            const Snapshot distributed = solve(2u);
            test.expect(distributed.distributed,
                        "SurfaceIBM MPI test entered one-domain-per-rank mode");
            check_physics(test, distributed, 2u);
            if(test.failures==0 && culbm::mpi_is_root()) {
                std::cout << "SurfaceIBM two-rank MPI GPU integration test passed.\n";
            }
            return test.failures==0 ? 0 : 1;
        }
        check_half_integer_support(test);
        check_reduction_tails(test);
        check_curved_sphere(test);
        const Snapshot one = solve(1u);
        check_physics(test, one, 1u);
        if(multi_gpu) {
            const Snapshot two = solve(2u);
            check_physics(test, two, 2u);
            compare_snapshots(test, one, two);
        }
        if(test.failures==0) {
            std::cout << "SurfaceIBM GPU integration tests passed.\n";
        }
        return test.failures==0 ? 0 : 1;
    } catch(const std::exception& error) {
        std::cerr << "SURFACE_IBM_TEST_ERROR " << error.what() << '\n';
        return 2;
    }
}
