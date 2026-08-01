#include "lbm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

    void expect_relative(const double actual, const double expected,
                         const double tolerance, const std::string& message) {
        const double scale = std::max(std::fabs(expected), 1.0E-30);
        expect_close(actual, expected, tolerance*scale, message);
    }
};

const char* lattice_name(const Temperature_Lattice lattice) {
    return lattice==Temperature_Lattice::Q19 ? "D3Q19" : "D3Q7";
}

void initialize_fourier_temperature(LBM& lbm, const double amplitude) {
    const double kx = 2.0*kPi/(double)lbm.get_Nx();
    const double ky = 2.0*kPi/(double)lbm.get_Ny();
    const double kz = 2.0*kPi/(double)lbm.get_Nz();
    for(uint z=0u; z<lbm.get_Nz(); ++z) {
        for(uint y=0u; y<lbm.get_Ny(); ++y) {
            for(uint x=0u; x<lbm.get_Nx(); ++x) {
                const ulong n = lbm.index(x, y, z);
                lbm.rho[n] = 1.0f;
                lbm.u.x[n] = 0.0f;
                lbm.u.y[n] = 0.0f;
                lbm.u.z[n] = 0.0f;
                lbm.flags[n] = (uchar)0u;
                lbm.T[n] = (float)(1.0+amplitude*std::cos(
                    kx*(double)x+ky*(double)y+kz*(double)z));
            }
        }
    }
}

double fitted_fourier_diffusivity(
    LBM& lbm, const double initial_amplitude, const uint steps) {
    lbm.T.read_from_device();
    const double kx = 2.0*kPi/(double)lbm.get_Nx();
    const double ky = 2.0*kPi/(double)lbm.get_Ny();
    const double kz = 2.0*kPi/(double)lbm.get_Nz();
    double cosine = 0.0;
    double sine = 0.0;
    for(uint z=0u; z<lbm.get_Nz(); ++z) {
        for(uint y=0u; y<lbm.get_Ny(); ++y) {
            for(uint x=0u; x<lbm.get_Nx(); ++x) {
                const double phase =
                    kx*(double)x+ky*(double)y+kz*(double)z;
                const double theta =
                    (double)lbm.T[lbm.index(x, y, z)]-1.0;
                cosine += theta*std::cos(phase);
                sine += theta*std::sin(phase);
            }
        }
    }
    const double count = (double)lbm.get_N();
    const double amplitude =
        std::hypot(2.0*cosine/count, 2.0*sine/count);
    const double k_squared = kx*kx+ky*ky+kz*kz;
    // UPDATE_FIELDS stores the macro scalar reconstructed at the beginning of
    // stream_collide.  After N calls it therefore represents N-1 completed
    // scalar collisions unless update_fields is explicitly refreshed.
    const uint completed_scalar_collisions = steps>0u ? steps-1u : 0u;
    if(completed_scalar_collisions==0u) {
        throw std::invalid_argument(
            "Fourier fit requires at least two stream-collide steps");
    }
    return -std::log(amplitude/initial_amplitude)/
        (k_squared*(double)completed_scalar_collisions);
}

void test_fixed_eddy_viscosity_scalar(
    TestState& test, const Temperature_Lattice lattice) {
    // Keep the Fourier mode in the low-wave-number regime so this checks the
    // requested diffusivity rather than finite-k lattice dispersion.
    constexpr uint n = 48u;
    constexpr uint steps = 160u;
    constexpr double amplitude = 0.10;
    constexpr float base_nu = 0.05f;
    constexpr float base_alpha = 0.01f;
    constexpr float fixed_nu_t = 0.015f;
    constexpr float turbulent_prandtl = 0.75f;
    constexpr double expected_alpha =
        (double)base_alpha+(double)fixed_nu_t/(double)turbulent_prandtl;

    test.expect(set_temperature_lattice(lattice),
                std::string(lattice_name(lattice))+" lattice can be selected");
    Subgrid_Configuration configuration;
    configuration.enabled = true;
    configuration.smagorinsky_constant =
        (float)std::sqrt(0.453989945);
    configuration.scalar_coupling = true;
    configuration.turbulent_prandtl = turbulent_prandtl;
    configuration.fixed_eddy_viscosity = fixed_nu_t;
    test.expect(set_subgrid_configuration(configuration),
                std::string(lattice_name(lattice))+" fixed SGS config is accepted");

    LBM lbm(n, n, n, base_nu,
            0.0f, 0.0f, 0.0f, 0.0f, base_alpha, 0.0f);
    initialize_fourier_temperature(lbm, amplitude);
    lbm.external_initialize();
    for(uint step=0u; step<steps; ++step) lbm.external_step();
    lbm.external_finish();

    const double measured_alpha =
        fitted_fourier_diffusivity(lbm, amplitude, steps);
    const Subgrid_Statistics statistics = lbm.subgrid_statistics();
    std::cout << lattice_name(lattice)
              << " fixed-nu_t Fourier alpha=" << measured_alpha
              << ", expected=" << expected_alpha
              << ", nu_t mean/max=" << statistics.mean_nu_t
              << '/' << statistics.max_nu_t
              << ", cells=" << statistics.count << '\n';
    test.expect_relative(
        measured_alpha, expected_alpha, 0.05,
        std::string(lattice_name(lattice))+
            " Fourier decay recovers alpha+nu_t/Pr_t");
    test.expect(statistics.valid,
                std::string(lattice_name(lattice))+" SGS statistics are valid");
    test.expect(statistics.count==(std::uint64_t)n*n*n,
                std::string(lattice_name(lattice))+
                    " SGS statistics count every physical periodic cell");
    test.expect_close(statistics.mean_nu_t, fixed_nu_t, 2.0E-7,
                      std::string(lattice_name(lattice))+" mean nu_t is fixed value");
    test.expect_close(statistics.max_nu_t, fixed_nu_t, 2.0E-7,
                      std::string(lattice_name(lattice))+" max nu_t is fixed value");
    test.expect_relative(
        statistics.mean_rho_nu_t, fixed_nu_t, 2.0E-4,
        std::string(lattice_name(lattice))+" rho*nu_t mean retains rho=1");
}

void test_base_scalar_diffusivity_without_subgrid(
    TestState& test, const Temperature_Lattice lattice) {
    // Use the same effective scalar diffusivity as the fixed-nu_t test.  This
    // keeps the finite-wave-number error identical while independently proving
    // that the runtime SGS-off kernel uses alpha0 and exposes no SGS statistics.
    constexpr uint n = 48u;
    constexpr uint steps = 160u;
    constexpr double amplitude = 0.10;
    constexpr float base_nu = 0.05f;
    constexpr float base_alpha = 0.03f;

    test.expect(set_temperature_lattice(lattice),
                std::string(lattice_name(lattice))+
                    " SGS-off lattice can be selected");
    Subgrid_Configuration configuration;
    configuration.enabled = false;
    configuration.smagorinsky_constant =
        (float)std::sqrt(0.453989945);
    configuration.scalar_coupling = false;
    configuration.turbulent_prandtl = 1.0f;
    configuration.fixed_eddy_viscosity = -1.0f;
    test.expect(set_subgrid_configuration(configuration),
                std::string(lattice_name(lattice))+
                    " SGS-off configuration is accepted");

    LBM lbm(n, n, n, base_nu,
            0.0f, 0.0f, 0.0f, 0.0f, base_alpha, 0.0f);
    initialize_fourier_temperature(lbm, amplitude);
    lbm.external_initialize();
    for(uint step=0u; step<steps; ++step) lbm.external_step();
    lbm.external_finish();

    const double measured_alpha =
        fitted_fourier_diffusivity(lbm, amplitude, steps);
    const Subgrid_Statistics statistics = lbm.subgrid_statistics();
    std::cout << lattice_name(lattice)
              << " SGS-off Fourier alpha=" << measured_alpha
              << ", expected=" << base_alpha
              << ", stats valid/count=" << statistics.valid
              << '/' << statistics.count << '\n';
    test.expect_relative(
        measured_alpha, base_alpha, 0.05,
        std::string(lattice_name(lattice))+
            " SGS-off Fourier decay recovers base alpha");
    test.expect(!statistics.valid,
                std::string(lattice_name(lattice))+
                    " SGS-off statistics are invalid");
    test.expect(statistics.count==0u,
                std::string(lattice_name(lattice))+
                    " SGS-off statistics count is zero");
}

void test_shifted_equilibrium_statistics(TestState& test) {
    constexpr uint n = 16u;
    test.expect(set_temperature_lattice(Temperature_Lattice::Q19),
                "D3Q19 can be selected for shifted-equilibrium SGS test");
    Subgrid_Configuration configuration;
    configuration.enabled = true;
    configuration.smagorinsky_constant =
        (float)std::sqrt(0.453989945);
    configuration.scalar_coupling = false;
    configuration.turbulent_prandtl = 1.0f;
    configuration.fixed_eddy_viscosity = -1.0f;
    test.expect(set_subgrid_configuration(configuration),
                "stress-derived SGS config is accepted for shifted-equilibrium test");
    LBM lbm(n, n, n, 0.05f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.01f, 0.0f);
    for(ulong i=0ull; i<lbm.get_N(); ++i) {
        lbm.rho[i] = 1.2f;
        lbm.u.x[i] = 0.031f;
        lbm.u.y[i] = -0.019f;
        lbm.u.z[i] = 0.011f;
        lbm.T[i] = 1.0f;
        lbm.flags[i] = (uchar)0u;
    }
    lbm.external_initialize();
    lbm.external_finish();
    const Subgrid_Statistics statistics = lbm.subgrid_statistics();
    std::cout << "Shifted equilibrium SGS nu_t mean/max="
              << statistics.mean_nu_t << '/' << statistics.max_nu_t
              << ", cells=" << statistics.count << '\n';
    test.expect(statistics.valid,
                "shifted-equilibrium SGS statistics are valid");
    test.expect(statistics.count==(std::uint64_t)n*n*n,
                "shifted-equilibrium statistics count all cells");
    // FP16C compression can leave a small residual stress, but omitting the
    // (rho-1)/3 pressure shift yields O(1e-1) nu_t for this rho=1.2 state.
    test.expect(statistics.max_nu_t<5.0E-3f,
                "shifted rho=1.2 equilibrium does not create spurious SGS viscosity");
}

void test_forced_nonequilibrium_statistics(TestState& test) {
    // At t=0 the shifted DDFs are equilibrium at raw velocity u0.  Guo
    // reconstruction gives physical velocity u0+F/(2*rho).  After adding the
    // required (uF+Fu)/2 source moment, the O(u0*F) term cancels and only the
    // known O(F^2) start-up residual Hxx=F^2/(4*rho) remains.  Omitting the
    // force correction makes this test overpredict nu_t by more than an order
    // of magnitude for the selected u0.
    constexpr uint n = 16u;
    constexpr float base_nu = 0.05f;
    constexpr float force_x = 0.04f;
    constexpr float raw_velocity_x = 0.08f;
    constexpr double cs_squared = 0.453989945;
    test.expect(set_temperature_lattice(Temperature_Lattice::Q19),
                "D3Q19 can be selected for forced SGS formula test");
    Subgrid_Configuration configuration;
    configuration.enabled = true;
    configuration.smagorinsky_constant =
        (float)std::sqrt(cs_squared);
    configuration.scalar_coupling = false;
    configuration.turbulent_prandtl = 1.0f;
    configuration.fixed_eddy_viscosity = -1.0f;
    test.expect(set_subgrid_configuration(configuration),
                "stress-derived SGS config is accepted for forced formula test");

    LBM lbm(n, n, n, base_nu,
            force_x, 0.0f, 0.0f, 0.0f, 0.01f, 0.0f);
    for(ulong i=0ull; i<lbm.get_N(); ++i) {
        lbm.rho[i] = 1.0f;
        lbm.u.x[i] = raw_velocity_x;
        lbm.u.y[i] = 0.0f;
        lbm.u.z[i] = 0.0f;
        lbm.T[i] = 1.0f;
        lbm.flags[i] = (uchar)0u;
    }
    lbm.external_initialize();
    lbm.external_finish();
    const Subgrid_Statistics statistics = lbm.subgrid_statistics();

    const double tau0 = 0.5+3.0*(double)base_nu;
    const double force_residual =
        0.25*(double)force_x*(double)force_x;
    const double prefactor = 18.0*std::sqrt(2.0)*
        (double)configuration.smagorinsky_constant*
        (double)configuration.smagorinsky_constant;
    const double tau_eff = 0.5*(tau0+std::sqrt(
        tau0*tau0+prefactor*force_residual));
    const double expected_nu_t = (tau_eff-tau0)/3.0;
    const double uncorrected_stress =
        (double)raw_velocity_x*(double)force_x+force_residual;
    const double uncorrected_tau_eff = 0.5*(tau0+std::sqrt(
        tau0*tau0+prefactor*uncorrected_stress));
    const double uncorrected_nu_t =
        (uncorrected_tau_eff-tau0)/3.0;
    std::cout << "Forced non-equilibrium SGS nu_t mean/max="
              << statistics.mean_nu_t << '/' << statistics.max_nu_t
              << ", expected=" << expected_nu_t
              << ", cells=" << statistics.count << '\n';
    test.expect(statistics.valid,
                "forced non-equilibrium SGS statistics are valid");
    test.expect(statistics.count==(std::uint64_t)n*n*n,
                "forced non-equilibrium statistics count all cells");
    test.expect(expected_nu_t>0.0,
                "forced non-equilibrium reference viscosity is positive");
    test.expect_relative(statistics.mean_nu_t, expected_nu_t, 0.05,
                         "production SGS mean includes Guo force correction");
    test.expect_relative(statistics.max_nu_t, expected_nu_t, 0.05,
                         "production SGS max includes Guo force correction");
    test.expect_relative(statistics.mean_rho_nu_t, expected_nu_t, 0.05,
                         "production rho*nu_t matches rho=1 reference");
    test.expect(statistics.max_nu_t<0.25*uncorrected_nu_t,
                "uniform force is not interpreted as leading-order shear");
}

struct FieldSnapshot {
    std::vector<float> rho;
    std::vector<float> velocity;
    std::vector<float> temperature;
    Subgrid_Statistics statistics;
    bool requested_topology = false;
    bool distinct_devices = true;
};

constexpr uint kMultiNx = 40u;
constexpr uint kMultiNy = 20u;
constexpr uint kMultiNz = 20u;
constexpr std::uint64_t kMultiCells =
    (std::uint64_t)kMultiNx*kMultiNy*kMultiNz;

FieldSnapshot solve_subgrid_snapshot(
    const uint dx, const Temperature_Lattice lattice,
    const bool stress_derived_subgrid) {
    constexpr uint steps = 48u;
    if(!set_temperature_lattice(lattice)) {
        throw std::runtime_error("Cannot select scalar lattice for multi-GPU SGS test");
    }
    Subgrid_Configuration configuration;
    configuration.enabled = true;
    configuration.smagorinsky_constant =
        (float)std::sqrt(0.453989945);
    configuration.scalar_coupling = true;
    configuration.turbulent_prandtl = 1.0f;
    configuration.fixed_eddy_viscosity =
        stress_derived_subgrid ? -1.0f : 0.012f;
    if(!set_subgrid_configuration(configuration)) {
        throw std::runtime_error("Cannot select SGS configuration for multi-GPU test");
    }
    LBM lbm(kMultiNx, kMultiNy, kMultiNz, dx, 1u, 1u, 0.04f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.015f, 0.0f);
    FieldSnapshot result;
    result.requested_topology =
        lbm.get_Dx()==dx && lbm.get_Dy()==1u && lbm.get_Dz()==1u;
    if(dx>1u) {
        result.distinct_devices =
            lbm.get_local_domain(0u).get_device().info.id !=
            lbm.get_local_domain(1u).get_device().info.id;
    }
    const double kx = 2.0*kPi/(double)kMultiNx;
    const double ky = 2.0*kPi/(double)kMultiNy;
    const double kz = 2.0*kPi/(double)kMultiNz;
    for(uint z=0u; z<kMultiNz; ++z) {
        for(uint y=0u; y<kMultiNy; ++y) {
            for(uint x=0u; x<kMultiNx; ++x) {
                const ulong index = lbm.index(x, y, z);
                const double px = kx*(double)x;
                const double py = ky*(double)y;
                const double pz = kz*(double)z;
                lbm.rho[index] = 1.0f;
                lbm.u.x[index] = (float)(
                    0.025*std::sin(px)*std::cos(py)*std::cos(pz));
                lbm.u.y[index] = (float)(
                    -0.025*std::cos(px)*std::sin(py)*std::cos(pz));
                lbm.u.z[index] = 0.0f;
                lbm.T[index] = (float)(
                    1.0+0.05*std::cos(px+py+pz));
                lbm.flags[index] = (uchar)0u;
            }
        }
    }
    lbm.external_initialize();
    for(uint step=0u; step<steps; ++step) lbm.external_step();
    lbm.external_finish();
    lbm.rho.read_from_device();
    lbm.u.read_from_device();
    lbm.T.read_from_device();
    result.statistics = lbm.subgrid_statistics();
    result.rho.resize((std::size_t)lbm.get_N());
    result.temperature.resize((std::size_t)lbm.get_N());
    result.velocity.resize(3u*(std::size_t)lbm.get_N());
    for(ulong i=0ull; i<lbm.get_N(); ++i) {
        result.rho[(std::size_t)i] = lbm.rho[i];
        result.temperature[(std::size_t)i] = lbm.T[i];
        result.velocity[(std::size_t)i] = lbm.u.x[i];
        result.velocity[(std::size_t)lbm.get_N()+(std::size_t)i] = lbm.u.y[i];
        result.velocity[2u*(std::size_t)lbm.get_N()+(std::size_t)i] = lbm.u.z[i];
    }
    return result;
}

void test_multi_gpu_case(
    TestState& test, const Temperature_Lattice lattice,
    const bool stress_derived_subgrid) {
    const std::string case_name = std::string(lattice_name(lattice))+
        (stress_derived_subgrid ? " stress-derived" : " fixed");
    const FieldSnapshot one =
        solve_subgrid_snapshot(1u, lattice, stress_derived_subgrid);
    const FieldSnapshot two =
        solve_subgrid_snapshot(2u, lattice, stress_derived_subgrid);
    test.expect(one.requested_topology && two.requested_topology,
                case_name+" SGS comparison uses requested one/two-domain topology");
    test.expect(two.distinct_devices,
                case_name+" SGS two-domain comparison uses distinct GPUs");
    const auto compare = [&](const std::vector<float>& expected,
                             const std::vector<float>& actual,
                             const char* name) {
        test.expect(expected.size()==actual.size() && !expected.empty(),
                    case_name+' '+name+" snapshots have equal nonzero extent");
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
        std::cout << case_name << " SGS two-GPU " << name << " max/RMS="
                  << maximum << '/' << rms << '\n';
        test.expect(maximum<=2.0E-4,
                    case_name+' '+name+" two-GPU max error is bounded");
        test.expect(rms<=5.0E-5,
                    case_name+' '+name+" two-GPU RMS error is bounded");
    };
    compare(one.rho, two.rho, "rho");
    compare(one.velocity, two.velocity, "velocity");
    compare(one.temperature, two.temperature, "temperature");
    test.expect(one.statistics.valid && two.statistics.valid,
                case_name+" one/two-GPU SGS statistics are valid");
    test.expect(one.statistics.count==kMultiCells,
                case_name+" one-GPU SGS count equals the global cell count");
    test.expect(two.statistics.count==kMultiCells,
                case_name+" two-GPU SGS count excludes halos exactly");
    test.expect_relative(two.statistics.mean_nu_t,
                         one.statistics.mean_nu_t, 2.0E-5,
                         case_name+" one/two-GPU mean nu_t agrees");
    test.expect_close(two.statistics.max_nu_t,
                      one.statistics.max_nu_t, 2.0E-7,
                      case_name+" one/two-GPU max nu_t agrees");
    test.expect_relative(two.statistics.mean_rho_nu_t,
                         one.statistics.mean_rho_nu_t, 2.0E-5,
                         case_name+" one/two-GPU mean rho*nu_t agrees");
    if(stress_derived_subgrid) {
        test.expect(one.statistics.mean_nu_t>0.0 &&
                    one.statistics.max_nu_t>0.0f,
                    case_name+" SGS is active across the halo case");
    } else {
        test.expect_close(one.statistics.mean_nu_t, 0.012, 2.0E-7,
                          case_name+" fixed SGS retains prescribed nu_t");
    }
}

void test_multi_gpu_consistency(TestState& test) {
    test_multi_gpu_case(test, Temperature_Lattice::Q19, false);
    test_multi_gpu_case(test, Temperature_Lattice::Q19, true);
    test_multi_gpu_case(test, Temperature_Lattice::Q7, true);
}

enum class PerformanceOrder {
    OffOn,
    OnOff
};

const char* performance_order_name(const PerformanceOrder order) {
    return order==PerformanceOrder::OnOff ? "on-off" : "off-on";
}

struct PerformanceOptions {
    uint nx = 512u;
    uint ny = 64u;
    uint nz = 64u;
    uint dx = 1u;
    uint warmup = 200u;
    uint steps = 1000u;
    Temperature_Lattice lattice = Temperature_Lattice::Q19;
    PerformanceOrder order = PerformanceOrder::OffOn;
    std::string output;
};

uint parse_uint(const std::string& text, const std::string& option) {
    std::size_t consumed = 0u;
    const unsigned long value = std::stoul(text, &consumed);
    if(consumed!=text.size() || value==0ul ||
       value>(unsigned long)std::numeric_limits<uint>::max()) {
        throw std::invalid_argument("Invalid positive integer for "+option);
    }
    return (uint)value;
}

PerformanceOptions parse_performance_options(
    const int argc, const char* const argv[]) {
    PerformanceOptions options;
    for(int i=2; i<argc; ++i) {
        const std::string option = argv[i];
        const auto value = [&]() -> std::string {
            if(i+1>=argc) throw std::invalid_argument("Missing value after "+option);
            return argv[++i];
        };
        if(option=="--nx") options.nx = parse_uint(value(), option);
        else if(option=="--ny") options.ny = parse_uint(value(), option);
        else if(option=="--nz") options.nz = parse_uint(value(), option);
        else if(option=="--dx") options.dx = parse_uint(value(), option);
        else if(option=="--warmup") options.warmup = parse_uint(value(), option);
        else if(option=="--steps") options.steps = parse_uint(value(), option);
        else if(option=="--output") options.output = value();
        else if(option=="--order") {
            const std::string order = value();
            if(order=="off-on") options.order = PerformanceOrder::OffOn;
            else if(order=="on-off") options.order = PerformanceOrder::OnOff;
            else throw std::invalid_argument(
                "--order must be off-on or on-off");
        }
        else if(option=="--thermal") {
            const std::string lattice = value();
            if(lattice=="D3Q7" || lattice=="d3q7")
                options.lattice = Temperature_Lattice::Q7;
            else if(lattice=="D3Q19" || lattice=="d3q19")
                options.lattice = Temperature_Lattice::Q19;
            else throw std::invalid_argument("--thermal must be D3Q7 or D3Q19");
        } else {
            throw std::invalid_argument("Unknown performance option: "+option);
        }
    }
    if(options.nx%options.dx!=0u)
        throw std::invalid_argument("nx must be divisible by dx");
    return options;
}

struct PerformanceResult {
    double steps_per_second = 0.0;
    double statistics_seconds = 0.0;
    double checksum = 0.0;
    bool finite = true;
    Subgrid_Statistics statistics;
    std::vector<float> field;
};

PerformanceResult run_performance_case(
    const PerformanceOptions& options, const bool subgrid_enabled) {
    if(!set_temperature_lattice(options.lattice))
        throw std::runtime_error("Cannot select performance scalar lattice");
    Subgrid_Configuration configuration;
    configuration.enabled = subgrid_enabled;
    configuration.smagorinsky_constant =
        (float)std::sqrt(0.453989945);
    configuration.scalar_coupling = subgrid_enabled;
    configuration.turbulent_prandtl = 1.0f;
    configuration.fixed_eddy_viscosity = -1.0f;
    if(!set_subgrid_configuration(configuration))
        throw std::runtime_error("Cannot select performance SGS configuration");

    LBM lbm(options.nx, options.ny, options.nz,
            options.dx, 1u, 1u, 0.025f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.015f, 0.0f);
    const double kx = 2.0*kPi/(double)options.nx;
    const double ky = 2.0*kPi/(double)options.ny;
    const double kz = 2.0*kPi/(double)options.nz;
    for(uint z=0u; z<options.nz; ++z) {
        for(uint y=0u; y<options.ny; ++y) {
            for(uint x=0u; x<options.nx; ++x) {
                const ulong n = lbm.index(x, y, z);
                const double px = kx*(double)x;
                const double py = ky*(double)y;
                const double pz = kz*(double)z;
                lbm.rho[n] = 1.0f;
                lbm.u.x[n] = (float)(
                    0.035*std::sin(px)*std::cos(py)*std::cos(pz));
                lbm.u.y[n] = (float)(
                    -0.035*std::cos(px)*std::sin(py)*std::cos(pz));
                lbm.u.z[n] = 0.0f;
                lbm.T[n] = (float)(1.0+0.04*std::cos(px+py+pz));
                lbm.flags[n] = (uchar)0u;
            }
        }
    }
    lbm.external_initialize();
    for(uint step=0u; step<options.warmup; ++step) lbm.external_step();
    lbm.external_finish();
    const auto start = std::chrono::steady_clock::now();
    for(uint step=0u; step<options.steps; ++step) lbm.external_step();
    lbm.external_finish();
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();

    PerformanceResult result;
    result.steps_per_second = (double)options.steps/elapsed;
    const auto statistics_start = std::chrono::steady_clock::now();
    result.statistics = lbm.subgrid_statistics();
    result.statistics_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-statistics_start).count();
    lbm.rho.read_from_device();
    lbm.u.read_from_device();
    lbm.T.read_from_device();
    result.field.resize(5u*(std::size_t)lbm.get_N());
    for(ulong n=0ull; n<lbm.get_N(); ++n) {
        const std::size_t i = (std::size_t)n;
        const double weight = 1.0+(double)(n%17ull)/17.0;
        const float values[5] = {
            lbm.rho[n], lbm.u.x[n], lbm.u.y[n], lbm.u.z[n], lbm.T[n]};
        for(std::size_t component=0u; component<5u; ++component) {
            result.field[component*(std::size_t)lbm.get_N()+i] =
                values[component];
            result.checksum += weight*(double)values[component];
            result.finite = result.finite && std::isfinite(values[component]);
        }
    }
    return result;
}

int run_performance(const PerformanceOptions& options) {
    PerformanceResult off;
    PerformanceResult on;
    if(options.order==PerformanceOrder::OnOff) {
        on = run_performance_case(options, true);
        off = run_performance_case(options, false);
    } else {
        off = run_performance_case(options, false);
        on = run_performance_case(options, true);
    }
    double difference_squared = 0.0;
    double reference_squared = 0.0;
    double maximum_difference = 0.0;
    const bool fields_comparable =
        !off.field.empty() && off.field.size()==on.field.size();
    if(fields_comparable) {
        for(std::size_t i=0u; i<off.field.size(); ++i) {
            const double difference = (double)on.field[i]-(double)off.field[i];
            difference_squared += difference*difference;
            reference_squared += (double)off.field[i]*(double)off.field[i];
            maximum_difference =
                std::max(maximum_difference, std::fabs(difference));
        }
    }
    const double relative_l2 =
        reference_squared>0.0 ? std::sqrt(difference_squared/reference_squared) : 0.0;
    const double ratio = on.steps_per_second/off.steps_per_second;
    const std::uint64_t cells =
        (std::uint64_t)options.nx*options.ny*options.nz;
    const bool throughput_finite =
        std::isfinite(off.steps_per_second) && off.steps_per_second>0.0 &&
        std::isfinite(on.steps_per_second) && on.steps_per_second>0.0 &&
        std::isfinite(ratio);
    const bool off_statistics_disabled =
        !off.statistics.valid && off.statistics.count==0u;
    const bool subgrid_statistics_complete =
        on.statistics.valid && on.statistics.count==cells;
    const bool eddy_viscosity_positive =
        std::isfinite(on.statistics.mean_nu_t) &&
        std::isfinite(on.statistics.max_nu_t) &&
        on.statistics.mean_nu_t>0.0 && on.statistics.max_nu_t>0.0f;
    const bool fields_differ = fields_comparable &&
        difference_squared>0.0 && maximum_difference>0.0;
    const bool passes_half_speed_gate = throughput_finite && ratio>=0.5;
    const bool passes_acceptance_gate =
        off.finite && on.finite && off_statistics_disabled &&
        subgrid_statistics_complete && eddy_viscosity_positive &&
        fields_differ && passes_half_speed_gate;
#if defined(FP16C)
    const char* precision = "FP16C";
#elif defined(FP16S)
    const char* precision = "FP16S";
#else
    const char* precision = "FP32";
#endif
    std::ostringstream json;
    json << std::setprecision(12)
         << "{\n"
         << "  \"precision\": \"" << precision << "\",\n"
         << "  \"thermal_lattice\": \""
         << lattice_name(options.lattice) << "\",\n"
         << "  \"execution_order\": \""
         << performance_order_name(options.order) << "\",\n"
         << "  \"smagorinsky_cs\": "
         << std::sqrt(0.453989945) << ",\n"
         << "  \"smagorinsky_cs_squared\": 0.453989945,\n"
         << "  \"turbulent_prandtl\": 1.0,\n"
         << "  \"grid\": [" << options.nx << ',' << options.ny
         << ',' << options.nz << "],\n"
         << "  \"decomposition\": [" << options.dx << ",1,1],\n"
         << "  \"cells\": "
         << cells << ",\n"
         << "  \"warmup_steps\": " << options.warmup << ",\n"
         << "  \"timed_steps\": " << options.steps << ",\n"
         << "  \"off_steps_per_s\": " << off.steps_per_second << ",\n"
         << "  \"on_steps_per_s\": " << on.steps_per_second << ",\n"
         << "  \"throughput_ratio\": " << ratio << ",\n"
         << "  \"throughput_loss_percent\": " << 100.0*(1.0-ratio) << ",\n"
         << "  \"off_checksum\": " << off.checksum << ",\n"
         << "  \"on_checksum\": " << on.checksum << ",\n"
         << "  \"field_relative_l2\": " << relative_l2 << ",\n"
         << "  \"field_max_abs_difference\": " << maximum_difference << ",\n"
         << "  \"subgrid_statistics_seconds\": "
         << on.statistics_seconds << ",\n"
         << "  \"subgrid_statistics_step_equivalents\": "
         << on.statistics_seconds*on.steps_per_second << ",\n"
         << "  \"nu_t_mean_lbm\": " << on.statistics.mean_nu_t << ",\n"
         << "  \"nu_t_max_lbm\": " << on.statistics.max_nu_t << ",\n"
         << "  \"subgrid_count\": " << on.statistics.count << ",\n"
         << "  \"off_subgrid_statistics_disabled\": "
         << (off_statistics_disabled ? "true" : "false") << ",\n"
         << "  \"subgrid_statistics_complete\": "
         << (subgrid_statistics_complete ? "true" : "false") << ",\n"
         << "  \"eddy_viscosity_positive\": "
         << (eddy_viscosity_positive ? "true" : "false") << ",\n"
         << "  \"fields_differ\": "
         << (fields_differ ? "true" : "false") << ",\n"
         << "  \"throughput_finite\": "
         << (throughput_finite ? "true" : "false") << ",\n"
         << "  \"finite\": "
         << (off.finite && on.finite ? "true" : "false") << ",\n"
         << "  \"passes_half_speed_gate\": "
         << (passes_half_speed_gate ? "true" : "false") << ",\n"
         << "  \"passes_acceptance_gate\": "
         << (passes_acceptance_gate ? "true" : "false") << "\n"
         << "}\n";
    std::cout << json.str();
    if(!options.output.empty()) {
        std::ofstream output(options.output);
        if(!output) throw std::runtime_error("Cannot write performance JSON");
        output << json.str();
    }
    return passes_acceptance_gate ? 0 : 1;
}

int run_analytic_tests() {
    TestState test;
    test_shifted_equilibrium_statistics(test);
    test_forced_nonequilibrium_statistics(test);
    test_base_scalar_diffusivity_without_subgrid(
        test, Temperature_Lattice::Q7);
    test_base_scalar_diffusivity_without_subgrid(
        test, Temperature_Lattice::Q19);
    test_fixed_eddy_viscosity_scalar(test, Temperature_Lattice::Q7);
    test_fixed_eddy_viscosity_scalar(test, Temperature_Lattice::Q19);
    if(test.failures==0) {
        std::cout << "GPU SGS off/on momentum and scalar tests passed for "
                     "D3Q7/D3Q19.\n";
    }
    return test.failures==0 ? 0 : 1;
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
        if(argc>=2 && std::string(argv[1])=="--performance") {
            return run_performance(parse_performance_options(argc, argv));
        }
        if(argc==2 && std::string(argv[1])=="--multi-gpu") {
            if(device_count<2) {
                std::cout << "SKIP: SGS multi-GPU test requires two devices.\n";
                return 77;
            }
            TestState test;
            test_multi_gpu_consistency(test);
            return test.failures==0 ? 0 : 1;
        }
        if(argc!=1) {
            std::cerr << "Usage: " << argv[0]
                      << " [--multi-gpu|--performance [--thermal D3Q7|D3Q19]"
                         " [--order off-on|on-off] [options]]\n";
            return 2;
        }
        return run_analytic_tests();
    } catch(const std::exception& error) {
        std::cerr << "SGS_TEST_ERROR " << error.what() << '\n';
        return 2;
    }
}
