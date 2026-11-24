#include "solver.hpp"
#include "surface_ibm.hpp"
#include "cuda_nested_solver.hpp"
#include "cuda_nested_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// The production executable defines these run-time controls in
// bc/outflow/damping_layers.cpp and case/physics.cpp.
// This focused solver test deliberately does not link the case/deck frontend,
// so provide neutral definitions while exercising the production LBM and
// production NVRTC kernel unchanged.
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
constexpr float kNu = 0.10f;
constexpr float kAlpha = 0.05f;

struct TestState {
    int failures = 0;

    void expect(const bool condition, const std::string& message) {
        if(condition) return;
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }

    void expect_close(const double actual,
                      const double expected,
                      const double absolute_tolerance,
                      const std::string& message) {
        if(std::isfinite(actual) &&
           std::fabs(actual-expected)<=absolute_tolerance) return;
        std::cerr << "FAIL: " << message
                  << " (actual=" << actual
                  << ", expected=" << expected
                  << ", tolerance=" << absolute_tolerance << ")\n";
        ++failures;
    }

    void expect_relative(const double actual,
                         const double expected,
                         const double relative_tolerance,
                         const std::string& message) {
        const double scale = std::max(std::fabs(expected), 1.0E-30);
        if(std::isfinite(actual) &&
           std::fabs(actual-expected)<=relative_tolerance*scale) return;
        std::cerr << "FAIL: " << message
                  << " (actual=" << actual
                  << ", expected=" << expected
                  << ", relative tolerance=" << relative_tolerance << ")\n";
        ++failures;
    }
};

const char* lattice_name(const Temperature_Lattice lattice) {
    return lattice==Temperature_Lattice::Q19 ? "D3Q19" : "D3Q7";
}

void initialize_quiescent_fields(culbm::solver::Solver& lbm,
                                 const float temperature,
                                 const uchar flag = (uchar)0u) {
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        lbm.density()[n] = 1.0f;
        lbm.velocity().x[n] = 0.0f;
        lbm.velocity().y[n] = 0.0f;
        lbm.velocity().z[n] = 0.0f;
        lbm.flags()[n] = flag;
        lbm.temperature()[n] = temperature;
    }
}

void advance(culbm::solver::Solver& lbm, const uint steps) {
    lbm.initialize_once();
    for(uint step=0u; step<steps; ++step) lbm.advance();
    lbm.synchronize();
}

std::array<double, 3u> mean_velocity(culbm::solver::Solver& lbm) {
    lbm.velocity().download();
    std::array<double, 3u> mean = {0.0, 0.0, 0.0};
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        mean[0] += (double)lbm.velocity().x[n];
        mean[1] += (double)lbm.velocity().y[n];
        mean[2] += (double)lbm.velocity().z[n];
    }
    const double inverse_count = 1.0/(double)lbm.grid().cells();
    for(double& component : mean) component *= inverse_count;
    return mean;
}

void test_lattice_contract(TestState& test,
                           const Temperature_Lattice lattice) {
    const bool selected = set_temperature_lattice(lattice);
    const double expected_cs2 =
        lattice==Temperature_Lattice::Q19 ? 1.0/3.0 : 1.0/4.0;
    const double expected_tau = 0.5+(double)kAlpha/expected_cs2;
    test.expect(selected,
                std::string(lattice_name(lattice))+
                " can be selected in the production binary");
    test.expect(get_temperature_lattice()==lattice,
                std::string(lattice_name(lattice))+
                " selection is retained");
    test.expect(get_temperature_set()==(uint)lattice,
                std::string(lattice_name(lattice))+
                " exposes the expected population count");
    test.expect(
        get_temperature_transfers()==
            (lattice==Temperature_Lattice::Q19 ? 5u : 1u),
        std::string(lattice_name(lattice))+
        " exposes the expected per-face halo population count");
    test.expect_close(
        get_temperature_lattice_cs2(), expected_cs2, 1.0E-7,
        std::string(lattice_name(lattice))+
        " exposes the expected scalar lattice sound speed");
    test.expect_close(
        get_temperature_tau(kAlpha), expected_tau, 1.0E-7,
        std::string(lattice_name(lattice))+
        " maps diffusivity to the expected relaxation time");
}

void test_relaxation_diagnostics(TestState& test) {
    const TRT_Relaxation_Diagnostic ordinary =
        get_trt_relaxation_diagnostic(0.75f);
    test.expect_close(
        ordinary.magic_parameter, 3.0/16.0, 1.0E-8,
        "TRT host diagnostics use the standard Lambda=3/16");
    test.expect_close(
        ordinary.minus.tau, 1.25, 1.0E-7,
        "TRT tau-minus satisfies the standard magic-parameter relation");
    test.expect_close(
        ((double)ordinary.plus.tau-0.5)*
            ((double)ordinary.minus.tau-0.5),
        3.0/16.0, 1.0E-8,
        "TRT plus/minus relaxation times preserve Lambda");
    test.expect(
        ordinary.plus.valid && ordinary.minus.valid &&
            !ordinary.plus.near_marginal &&
            !ordinary.minus.near_marginal,
        "well-damped TRT branches do not trigger the marginal warning");

    const TRT_Relaxation_Diagnostic marginal =
        get_trt_relaxation_diagnostic(0.50025f);
    test.expect(
        marginal.plus.near_marginal && marginal.minus.near_marginal,
        "both weakly damped TRT eigenbranches trigger the unit-circle warning");
    test.expect(
        marginal.plus.eigenvalue_magnitude>
            collision_eigenvalue_warning_threshold &&
        marginal.minus.eigenvalue_magnitude>
            collision_eigenvalue_warning_threshold,
        "TRT warning state is tied to the documented |1-omega| threshold");

    set_temperature_lattice(Temperature_Lattice::Q7);
    const Relaxation_Branch_Diagnostic ordinary_temperature =
        get_relaxation_branch_diagnostic(get_temperature_tau(kAlpha));
    const Relaxation_Branch_Diagnostic marginal_temperature =
        get_relaxation_branch_diagnostic(get_temperature_tau(1.0E-5f));
    test.expect(
        ordinary_temperature.valid &&
            !ordinary_temperature.near_marginal,
        "an ordinarily damped temperature SRT branch is not warned");
    test.expect(
        marginal_temperature.near_marginal,
        "a weakly damped temperature SRT branch triggers the same diagnostic");
}

void test_temperature_memory_accounting(TestState& test) {
    set_temperature_lattice(Temperature_Lattice::Q7);
    test.expect(
        !set_temperature_lattice(static_cast<Temperature_Lattice>(42u)) &&
            get_temperature_lattice()==Temperature_Lattice::Q7,
        "invalid scalar lattice selection is rejected without changing the active lattice");
    const uint q7_bytes_per_cell = culbm::solver::cell_cost(lattice_storage()).device;
    const uint q7_bandwidth_per_cell = culbm::solver::cell_cost(lattice_storage()).traffic;
    set_temperature_lattice(Temperature_Lattice::Q19);
    const uint q19_bytes_per_cell = culbm::solver::cell_cost(lattice_storage()).device;
    const uint q19_bandwidth_per_cell = culbm::solver::cell_cost(lattice_storage()).traffic;
    const uint population_delta = 19u-7u;
    test.expect(
        q19_bytes_per_cell-q7_bytes_per_cell==
            population_delta*get_ddf_bytes(),
        "D3Q19 scalar VRAM accounting includes all twelve additional DDFs");
    test.expect(
        q19_bandwidth_per_cell-q7_bandwidth_per_cell==
            population_delta*(2u*get_ddf_bytes()+1u),
        "D3Q19 scalar bandwidth accounting includes DDF traffic and additional neighbor flags");
}

void test_boussinesq_unit_conversion(TestState& test) {
    constexpr float si_gravity = -9.80665f;
    constexpr float si_reference_temperature = 293.15f;
    constexpr float si_temperature_delta = 12.0f;
    constexpr float temperature_scale = 37.0f;
    Units local_units;
    local_units.set_m_kg_s_K(2.5f, 1.0f, 0.10f, temperature_scale);
    local_units.set_temperature_reference(1.0f, si_reference_temperature);

    const double converted_acceleration =
        (double)local_units.g(si_gravity)*
        (double)local_units.beta(1.0f/si_reference_temperature)*
        (double)local_units.dT(si_temperature_delta);
    const double directly_scaled_acceleration =
        (double)local_units.g(
            si_gravity*si_temperature_delta/si_reference_temperature);
    test.expect_relative(
        converted_acceleration, directly_scaled_acceleration, 2.0E-6,
        "SI-to-LBM Boussinesq g*beta*dT conversion is independent of the"
        " affine temperature scale and offset");
    test.expect_close(
        local_units.si_T(local_units.T(311.25f)), 311.25, 8.0E-5,
        "absolute Kelvin temperature survives the affine SI/LBM round trip");
    test.expect_close(
        local_units.si_dT(local_units.dT(si_temperature_delta)),
        si_temperature_delta, 1.0E-5,
        "temperature differences use the offset-free SI/LBM conversion");
}

void test_peak_memory_accounting(TestState& test) {
    test.expect(
        voxelization_transient_vram_mb_per_device(1u)==4u,
        "one STL triangle rounds to three 1 MiB vertex buffers plus one 1 MiB parameter buffer");
    test.expect(
        voxelization_transient_vram_mb_per_device(87381u)==4u,
        "87381 float3 values still fit in each 1 MiB rounded vertex allocation");
    test.expect(
        voxelization_transient_vram_mb_per_device(87382u)==7u,
        "crossing 1 MiB rounds each of the three vertex allocations independently");

    const culbm::VerticalNestingPlan padded_plan =
        culbm::make_vertical_nesting_plan(
            {3.0}, 1.0, 9.0, 512u, 512u, 1u, 1u, 4u);
    test.expect(
        padded_plan.valid && padded_plan.enabled,
        "full-height raster padding regression uses a valid nested plan");
    const std::uint64_t padded_raster_mb =
        culbm::nested_full_height_raster_vram_required_mb_per_device(
            padded_plan, 1u, 1u, 4u);
    const std::uint64_t expected_padded_raster_mb =
        lattice_vram_mib(
            1024u, 1024u, 20u, 1u, 1u, 4u);
    test.expect(
        padded_raster_mb==expected_padded_raster_mb,
        "full-height common-finest raster rounds active Nz up to the configured Dz");

    const culbm::VerticalNestingPlan raster_dominant_plan =
        culbm::make_vertical_nesting_plan(
            {3.0}, 1.0, 105.0, 256u, 256u, 1u, 1u, 4u);
    const std::uint64_t hierarchy_mb =
        culbm::nested_hierarchy_vram_required_mb_per_device(
            raster_dominant_plan, 1u, 1u, 4u);
    const std::uint64_t full_raster_mb =
        culbm::nested_full_height_raster_vram_required_mb_per_device(
            raster_dominant_plan, 1u, 1u, 4u);
    test.expect(
        raster_dominant_plan.valid,
        "raster-dominant peak regression uses a valid nested plan");
    test.expect(
        full_raster_mb>hierarchy_mb,
        "temporary common-finest geometry raster can dominate the final nested hierarchy peak"
        " (raster="+std::to_string(full_raster_mb)+
        " MB, hierarchy="+std::to_string(hierarchy_mb)+" MB)");
}

void test_periodic_fourier_diffusion(TestState& test,
                                     const Temperature_Lattice lattice) {
    constexpr uint N = 40u;
    constexpr uint steps = 80u;
    constexpr double amplitude_initial = 0.10;
    const double wave_number = 2.0*kPi/(double)N;
    const double wave_number_squared = 3.0*wave_number*wave_number;

    test.expect(set_temperature_lattice(lattice),
                "temperature lattice selection succeeds before Fourier test");
    culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    const Temperature_Lattice other_lattice =
        lattice==Temperature_Lattice::Q7
            ? Temperature_Lattice::Q19
            : Temperature_Lattice::Q7;
    test.expect(
        !set_temperature_lattice(other_lattice) &&
            get_temperature_lattice()==lattice,
        std::string(lattice_name(lattice))+
        " cannot be changed while an LBM instance owns lattice-sized storage");
    initialize_quiescent_fields(lbm, 1.0f);
    for(uint z=0u; z<N; ++z) {
        for(uint y=0u; y<N; ++y) {
            for(uint x=0u; x<N; ++x) {
                const double phase =
                    wave_number*(double)(x+y+z);
                lbm.temperature()[lbm.grid().index(x, y, z)] =
                    (float)(1.0+amplitude_initial*std::cos(phase));
            }
        }
    }

    advance(lbm, steps);
    lbm.temperature().download();

    double mean = 0.0;
    double cosine_projection = 0.0;
    double sine_projection = 0.0;
    for(uint z=0u; z<N; ++z) {
        for(uint y=0u; y<N; ++y) {
            for(uint x=0u; x<N; ++x) {
                const double phase =
                    wave_number*(double)(x+y+z);
                const double perturbation =
                    (double)lbm.temperature()[lbm.grid().index(x, y, z)]-1.0;
                mean += (double)lbm.temperature()[lbm.grid().index(x, y, z)];
                cosine_projection += perturbation*std::cos(phase);
                sine_projection += perturbation*std::sin(phase);
            }
        }
    }
    const double count = (double)lbm.grid().cells();
    mean /= count;
    cosine_projection *= 2.0/count;
    sine_projection *= 2.0/count;
    const double amplitude_measured =
        std::hypot(cosine_projection, sine_projection);
    const double amplitude_expected = amplitude_initial*std::exp(
        -(double)kAlpha*wave_number_squared*(double)steps);
    const double diffusivity_measured = -std::log(
        amplitude_measured/amplitude_initial)/
        (wave_number_squared*(double)steps);

    std::cout << lattice_name(lattice)
              << " Fourier: amplitude=" << amplitude_measured
              << ", expected=" << amplitude_expected
              << ", fitted alpha=" << diffusivity_measured
              << ", mean=" << mean << '\n';
    test.expect_relative(
        diffusivity_measured, kAlpha, 0.05,
        std::string(lattice_name(lattice))+
        " periodic Fourier decay recovers the requested diffusivity");
    test.expect_close(
        mean, 1.0, 5.0E-4,
        std::string(lattice_name(lattice))+
        " periodic Fourier diffusion conserves mean temperature");
    test.expect(
        std::fabs(sine_projection)<=0.02*amplitude_measured,
        std::string(lattice_name(lattice))+
        " periodic pure diffusion does not create a material phase shift");
}

void test_periodic_fourier_advection(TestState& test,
                                     const Temperature_Lattice lattice) {
    constexpr uint N = 40u;
    constexpr uint steps = 40u;
    constexpr double amplitude_initial = 0.05;
    constexpr float ux = 0.020f;
    constexpr float uy = -0.010f;
    constexpr float uz = 0.015f;
    const double wave_number = 2.0*kPi/(double)N;
    const double phase_expected =
        wave_number*(double)(ux+uy+uz)*(double)steps;

    test.expect(set_temperature_lattice(lattice),
                "temperature lattice selection succeeds before advection test");
    culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    initialize_quiescent_fields(lbm, 1.0f);
    for(uint z=0u; z<N; ++z) {
        for(uint y=0u; y<N; ++y) {
            for(uint x=0u; x<N; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                const double phase = wave_number*(double)(x+y+z);
                lbm.velocity().x[n] = ux;
                lbm.velocity().y[n] = uy;
                lbm.velocity().z[n] = uz;
                lbm.temperature()[n] =
                    (float)(1.0+amplitude_initial*std::cos(phase));
            }
        }
    }

    advance(lbm, steps);
    lbm.temperature().download();

    double cosine_projection = 0.0;
    double sine_projection = 0.0;
    for(uint z=0u; z<N; ++z) {
        for(uint y=0u; y<N; ++y) {
            for(uint x=0u; x<N; ++x) {
                const double phase = wave_number*(double)(x+y+z);
                const double perturbation =
                    (double)lbm.temperature()[lbm.grid().index(x, y, z)]-1.0;
                cosine_projection += perturbation*std::cos(phase);
                sine_projection += perturbation*std::sin(phase);
            }
        }
    }
    const double projection_scale = 2.0/(double)lbm.grid().cells();
    cosine_projection *= projection_scale;
    sine_projection *= projection_scale;
    const double phase_measured =
        std::atan2(sine_projection, cosine_projection);

    std::cout << lattice_name(lattice)
              << " Fourier advection: phase=" << phase_measured
              << ", expected=" << phase_expected << '\n';
    test.expect_close(
        phase_measured, phase_expected, 1.0E-2,
        std::string(lattice_name(lattice))+
        " periodic Fourier mode is advected at the imposed fluid velocity");
}

struct FourierField {
    std::vector<float> temperature;
    bool expected_topology = false;
    bool distinct_devices = true;
};

FourierField solve_fourier_field(const Temperature_Lattice lattice,
                                 const uint Dx,
                                 const uint Dy,
                                 const uint Dz) {
    constexpr uint Nx = 40u;
    constexpr uint Ny = 20u;
    constexpr uint Nz = 20u;
    constexpr uint steps = 48u;
    constexpr double amplitude = 0.08;
    const double wave_number_x = 2.0*kPi/(double)Nx;
    const double wave_number_y = 2.0*kPi/(double)Ny;
    const double wave_number_z = 2.0*kPi/(double)Nz;

    set_temperature_lattice(lattice);
    culbm::solver::Solver lbm(Nx, Ny, Nz, Dx, Dy, Dz,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    const uint domain_count = Dx*Dy*Dz;
    FourierField result;
    result.expected_topology =
        !lbm.distributed() &&
        lbm.grid().split(0u)==Dx &&
        lbm.grid().split(1u)==Dy &&
        lbm.grid().split(2u)==Dz &&
        lbm.local_count()==domain_count &&
        lbm.grid().count()==domain_count;
    if(domain_count>1u) {
        result.distinct_devices =
            lbm.local(0u).device().ordinal() !=
            lbm.local(1u).device().ordinal();
    }
    initialize_quiescent_fields(lbm, 1.0f);
    for(uint z=0u; z<Nz; ++z) {
        for(uint y=0u; y<Ny; ++y) {
            for(uint x=0u; x<Nx; ++x) {
                const double phase =
                    wave_number_x*(double)x+
                    wave_number_y*(double)y+
                    wave_number_z*(double)z;
                lbm.temperature()[lbm.grid().index(x, y, z)] =
                    (float)(1.0+amplitude*std::cos(phase));
            }
        }
    }

    advance(lbm, steps);
    lbm.temperature().download();
    result.temperature.resize((std::size_t)lbm.grid().cells());
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        result.temperature[(std::size_t)n] = lbm.temperature()[n];
    }
    return result;
}

void test_multigpu_scalar_halo(TestState& test,
                               const Temperature_Lattice lattice) {
    constexpr uint Nx = 40u;
    constexpr uint Ny = 20u;
    constexpr uint Nz = 20u;
    const FourierField single_gpu =
        solve_fourier_field(lattice, 1u, 1u, 1u);
    const std::string prefix = lattice_name(lattice);

    test.expect(
        single_gpu.expected_topology,
        prefix+" reference Fourier case uses one local CUDA domain");
    struct AxisDecomposition {
        uint dx;
        uint dy;
        uint dz;
        uint axis;
        const char* label;
    };
    for(const AxisDecomposition decomposition : {
        AxisDecomposition{2u, 1u, 1u, 0u, "Dx=2"},
        AxisDecomposition{1u, 2u, 1u, 1u, "Dy=2"},
        AxisDecomposition{1u, 1u, 2u, 2u, "Dz=2"}}) {
        const FourierField two_gpu = solve_fourier_field(
            lattice, decomposition.dx, decomposition.dy, decomposition.dz);
        const std::string decomposition_name = decomposition.label;
        test.expect(
            two_gpu.expected_topology,
            prefix+" halo Fourier case uses a local "+
                decomposition_name+" decomposition");
        test.expect(
            two_gpu.distinct_devices,
            prefix+" "+decomposition_name+
                " decomposition assigns distinct CUDA devices");
        double squared_error_sum = 0.0;
        double maximum_error = 0.0;
        double interface_maximum_error = 0.0;
        for(uint z=0u; z<Nz; ++z) {
            for(uint y=0u; y<Ny; ++y) {
                for(uint x=0u; x<Nx; ++x) {
                    const std::size_t n =
                        (std::size_t)x+
                        ((std::size_t)y+(std::size_t)z*(std::size_t)Ny)*
                        (std::size_t)Nx;
                    const double error = std::fabs(
                        (double)two_gpu.temperature[n]-
                        (double)single_gpu.temperature[n]);
                    squared_error_sum += error*error;
                    maximum_error = std::max(maximum_error, error);
                    const uint coordinate =
                        decomposition.axis==0u ? x :
                        decomposition.axis==1u ? y : z;
                    const uint extent =
                        decomposition.axis==0u ? Nx :
                        decomposition.axis==1u ? Ny : Nz;
                    if(coordinate==0u || coordinate==extent-1u ||
                       coordinate==extent/2u-1u ||
                       coordinate==extent/2u) {
                        interface_maximum_error =
                            std::max(interface_maximum_error, error);
                    }
                }
            }
        }
        const double rms_error = std::sqrt(
            squared_error_sum/(double)single_gpu.temperature.size());

        std::cout
            << prefix << " scalar halo " << decomposition_name
            << " versus one GPU: max error=" << maximum_error
            << ", interface max error=" << interface_maximum_error
            << ", RMS error=" << rms_error << '\n';
        test.expect(
            maximum_error<=2.0E-4,
            prefix+" "+decomposition_name+
                " scalar halo agrees with the single-GPU Fourier field");
        test.expect(
            interface_maximum_error<=2.0E-4,
            prefix+" "+decomposition_name+
                " scalar halo agrees at periodic and decomposition interfaces");
        test.expect(
            rms_error<=5.0E-5,
            prefix+" "+decomposition_name+
                " scalar halo has a small global RMS difference");
    }
}

void test_fixed_temperature_solid_walls(
    TestState& test, const Temperature_Lattice lattice) {
    constexpr uint Nx = 34u;
    constexpr uint Ny = 8u;
    constexpr uint Nz = 8u;
    constexpr uint odd_steps = 127u;
    constexpr float left_temperature = 0.85f;
    constexpr float right_temperature = 1.15f;
    const double wall_separation = (double)(Nx-2u);

    test.expect(set_temperature_lattice(lattice),
                "temperature lattice selection succeeds before fixed-wall test");
    culbm::solver::Solver lbm(Nx, Ny, Nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    initialize_quiescent_fields(lbm, 1.0f);
    for(uint z=0u; z<Nz; ++z) {
        for(uint y=0u; y<Ny; ++y) {
            for(uint x=0u; x<Nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                if(x==0u) {
                    lbm.flags()[n] = (uchar)(CELL_SOLID|CELL_THERMAL);
                    lbm.temperature()[n] = left_temperature;
                } else if(x==Nx-1u) {
                    lbm.flags()[n] = (uchar)(CELL_SOLID|CELL_THERMAL);
                    lbm.temperature()[n] = right_temperature;
                } else {
                    const double fraction =
                        ((double)x-0.5)/wall_separation;
                    lbm.temperature()[n] = (float)(
                        (double)left_temperature+
                        ((double)right_temperature-
                         (double)left_temperature)*fraction);
                }
            }
        }
    }

    lbm.initialize_once();
    for(uint step=0u; step<odd_steps; ++step) lbm.advance();
    lbm.temperature().download();

    const auto profile_maximum_error = [&]() {
        double maximum_error = 0.0;
        for(uint z=0u; z<Nz; ++z) {
            for(uint y=0u; y<Ny; ++y) {
                for(uint x=1u; x<Nx-1u; ++x) {
                    const double fraction =
                        ((double)x-0.5)/wall_separation;
                    const double expected =
                        (double)left_temperature+
                        ((double)right_temperature-
                         (double)left_temperature)*fraction;
                    maximum_error = std::max(
                        maximum_error,
                        std::fabs((double)lbm.temperature()[lbm.grid().index(x, y, z)]-
                                  expected));
                }
            }
        }
        return maximum_error;
    };
    const double odd_maximum_error = profile_maximum_error();
    lbm.advance();
    lbm.temperature().download();
    const double even_maximum_error = profile_maximum_error();
    lbm.synchronize();

    std::cout << lattice_name(lattice)
              << " fixed-temperature solid walls: odd max error="
              << odd_maximum_error
              << ", even max error=" << even_maximum_error << '\n';
    test.expect(
        odd_maximum_error<=3.0E-3,
        std::string(lattice_name(lattice))+
        " fixed-temperature walls preserve the profile on odd pull parity");
    test.expect(
        even_maximum_error<=3.0E-3,
        std::string(lattice_name(lattice))+
        " fixed-temperature walls preserve the profile on even pull parity");
    test.expect_close(
        lbm.temperature()[lbm.grid().index(0u, 0u, 0u)], left_temperature, 1.0E-7,
        std::string(lattice_name(lattice))+
        " left solid wall retains its prescribed temperature");
    test.expect_close(
        lbm.temperature()[lbm.grid().index(Nx-1u, 0u, 0u)], right_temperature, 1.0E-7,
        std::string(lattice_name(lattice))+
        " right solid wall retains its prescribed temperature");
}

void test_adiabatic_solid_walls(TestState& test,
                                const Temperature_Lattice lattice) {
    constexpr uint Nx = 18u;
    constexpr uint Ny = 8u;
    constexpr uint Nz = 8u;
    constexpr uint odd_steps = 63u;
    constexpr float fluid_temperature = 1.12f;
    constexpr float unrelated_solid_temperature = 0.60f;

    test.expect(set_temperature_lattice(lattice),
                "temperature lattice selection succeeds before adiabatic-wall test");
    culbm::solver::Solver lbm(Nx, Ny, Nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    initialize_quiescent_fields(lbm, fluid_temperature);
    for(uint z=0u; z<Nz; ++z) {
        for(uint y=0u; y<Ny; ++y) {
            for(const uint x : {0u, Nx-1u}) {
                const ulong n = lbm.grid().index(x, y, z);
                lbm.flags()[n] = CELL_SOLID;
                // A truly adiabatic wall must not impose this unrelated
                // solid-node value on the neighboring fluid.
                lbm.temperature()[n] = unrelated_solid_temperature;
            }
        }
    }

    lbm.initialize_once();
    for(uint step=0u; step<odd_steps; ++step) lbm.advance();
    lbm.temperature().download();

    const auto uniform_fluid_error = [&]() {
        std::array<double, 2u> result = {0.0, 0.0};
        std::uint64_t fluid_count = 0u;
        for(uint z=0u; z<Nz; ++z) {
            for(uint y=0u; y<Ny; ++y) {
                for(uint x=1u; x<Nx-1u; ++x) {
                    const double value =
                        (double)lbm.temperature()[lbm.grid().index(x, y, z)];
                    result[0] += value;
                    result[1] = std::max(
                        result[1],
                        std::fabs(value-(double)fluid_temperature));
                    ++fluid_count;
                }
            }
        }
        result[0] /= (double)fluid_count;
        return result;
    };
    const std::array<double, 2u> odd_error = uniform_fluid_error();
    lbm.advance();
    lbm.temperature().download();
    const std::array<double, 2u> even_error = uniform_fluid_error();
    lbm.synchronize();

    std::cout << lattice_name(lattice)
              << " adiabatic solid walls: odd/even max error="
              << odd_error[1] << "/" << even_error[1] << '\n';
    test.expect_close(
        odd_error[0], fluid_temperature, 2.0E-3,
        std::string(lattice_name(lattice))+
        " adiabatic walls conserve uniform temperature on odd pull parity");
    test.expect_close(
        even_error[0], fluid_temperature, 2.0E-3,
        std::string(lattice_name(lattice))+
        " adiabatic walls conserve uniform temperature on even pull parity");
    test.expect(
        odd_error[1]<=4.0E-3 && even_error[1]<=4.0E-3,
        std::string(lattice_name(lattice))+
        " adiabatic walls do not leak solid temperature on either pull parity");
}

struct BuoyancySamples {
    std::array<double, 3u> early = {0.0, 0.0, 0.0};
    std::array<double, 3u> late = {0.0, 0.0, 0.0};
};

BuoyancySamples uniform_buoyancy_samples(
    const Temperature_Lattice lattice,
    const float temperature,
    const float beta,
    const float gravity_z) {
    constexpr uint N = 8u;
    constexpr uint early_steps = 4u;
    constexpr uint additional_steps = 8u;
    set_temperature_lattice(lattice);
    culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, beta));
    initialize_quiescent_fields(lbm, temperature);
    lbm.set_gravity(0.0f, 0.0f, gravity_z);
    advance(lbm, early_steps);
    BuoyancySamples samples;
    samples.early = mean_velocity(lbm);
    advance(lbm, additional_steps);
    samples.late = mean_velocity(lbm);
    return samples;
}

void test_independent_buoyancy(TestState& test,
                               const Temperature_Lattice lattice) {
    constexpr float beta = 0.20f;
    constexpr float temperature_delta = 0.10f;
    constexpr float gravity_z = -1.0E-4f;
    constexpr double sample_step_delta = 8.0;
    const double acceleration_expected =
        -(double)gravity_z*(double)beta*(double)temperature_delta;

    const BuoyancySamples hot = uniform_buoyancy_samples(
        lattice, 1.0f+temperature_delta, beta, gravity_z);
    const BuoyancySamples neutral = uniform_buoyancy_samples(
        lattice, 1.0f, beta, gravity_z);
    const BuoyancySamples cold = uniform_buoyancy_samples(
        lattice, 1.0f-temperature_delta, beta, gravity_z);

    std::array<double, 3u> hot_slope = {0.0, 0.0, 0.0};
    std::array<double, 3u> neutral_slope = {0.0, 0.0, 0.0};
    std::array<double, 3u> cold_slope = {0.0, 0.0, 0.0};
    for(std::size_t component=0u; component<3u; ++component) {
        hot_slope[component] =
            (hot.late[component]-hot.early[component])/sample_step_delta;
        neutral_slope[component] =
            (neutral.late[component]-neutral.early[component])/
            sample_step_delta;
        cold_slope[component] =
            (cold.late[component]-cold.early[component])/sample_step_delta;
    }

    std::cout << lattice_name(lattice)
              << " buoyancy: hot az=" << hot_slope[2]
              << ", neutral az=" << neutral_slope[2]
              << ", cold az=" << cold_slope[2]
              << ", expected |az|=" << acceleration_expected << '\n';
    test.expect_relative(
        hot_slope[2], acceleration_expected, 0.05,
        std::string(lattice_name(lattice))+
        " hot fluid accelerates opposite downward gravity with the configured magnitude");
    test.expect_relative(
        cold_slope[2], -acceleration_expected, 0.05,
        std::string(lattice_name(lattice))+
        " cold fluid accelerates along downward gravity with the configured magnitude");
    test.expect_close(
        neutral_slope[2], 0.0, 2.0E-8,
        std::string(lattice_name(lattice))+
        " reference-temperature fluid has no buoyancy acceleration");
    for(std::size_t component=0u; component<2u; ++component) {
        test.expect_close(
            hot_slope[component], 0.0, 2.0E-8,
            std::string(lattice_name(lattice))+
            " independent z gravity creates no transverse hot-fluid acceleration");
        test.expect_close(
            cold_slope[component], 0.0, 2.0E-8,
            std::string(lattice_name(lattice))+
            " independent z gravity creates no transverse cold-fluid acceleration");
    }
}

void test_buoyant_scalar_uses_physical_velocity(
    TestState& test, const Temperature_Lattice lattice) {
    constexpr uint Nx = 32u;
    constexpr uint Ny = 6u;
    constexpr uint Nz = 6u;
    constexpr float mean_temperature = 1.05f;
    constexpr float temperature_amplitude = 0.02f;
    constexpr float beta = 0.50f;
    constexpr float gravity_x = -2.00f;
    const std::string prefix = lattice_name(lattice);

    test.expect(set_temperature_lattice(lattice),
                prefix+
                    " temperature lattice selection succeeds before the"
                    " force-coupled scalar test");
    culbm::solver::Solver lbm(Nx, Ny, Nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, beta));
    initialize_quiescent_fields(lbm, mean_temperature, CELL_THERMAL);

    std::array<double, Nx> initial_temperature{};
    std::array<double, Nx> half_step_velocity{};
    for(uint x=0u; x<Nx; ++x) {
        const double temperature =
            (double)mean_temperature+
            (double)temperature_amplitude*
                std::sin(2.0*kPi*(double)x/(double)Nx);
        initial_temperature[x] = temperature;
        half_step_velocity[x] =
            -0.5*(double)gravity_x*(double)beta*(temperature-1.0);
        for(uint z=0u; z<Nz; ++z) {
            for(uint y=0u; y<Ny; ++y) {
                lbm.temperature()[lbm.grid().index(x, y, z)] = (float)temperature;
            }
        }
    }
    lbm.set_gravity(gravity_x, 0.0f, 0.0f);

    // The first production collision must rebuild each fixed-temperature DDF
    // with the Guo physical velocity u_raw+F/(2 rho).  Releasing CELL_THERMAL and
    // gravity exposes those DDFs through the next streamed temperature field.
    lbm.initialize_once();
    lbm.advance();
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) lbm.flags()[n] = (uchar)0u;
    lbm.flags().upload();
    lbm.set_gravity(0.0f, 0.0f, 0.0f);
    lbm.advance();
    lbm.temperature().download();

    const double center_weight =
        lattice==Temperature_Lattice::Q19 ? 2.0/3.0 : 3.0/4.0;
    const double neighbor_weight =
        lattice==Temperature_Lattice::Q19 ? 1.0/6.0 : 1.0/8.0;
    double physical_velocity_maximum_error = 0.0;
    double raw_velocity_maximum_error = 0.0;
    for(uint x=0u; x<Nx; ++x) {
        const uint xm = (x+Nx-1u)%Nx;
        const uint xp = (x+1u)%Nx;
        const double raw_velocity_prediction =
            center_weight*initial_temperature[x]+
            neighbor_weight*(initial_temperature[xm]+
                             initial_temperature[xp]);
        const double physical_velocity_prediction =
            raw_velocity_prediction+
            0.5*(initial_temperature[xm]*half_step_velocity[xm]-
                 initial_temperature[xp]*half_step_velocity[xp]);
        for(uint z=0u; z<Nz; ++z) {
            for(uint y=0u; y<Ny; ++y) {
                const double actual =
                    (double)lbm.temperature()[lbm.grid().index(x, y, z)];
                physical_velocity_maximum_error = std::max(
                    physical_velocity_maximum_error,
                    std::fabs(actual-physical_velocity_prediction));
                raw_velocity_maximum_error = std::max(
                    raw_velocity_maximum_error,
                    std::fabs(actual-raw_velocity_prediction));
            }
        }
    }
    lbm.synchronize();

    std::cout << prefix
              << " force-coupled scalar equilibrium: physical-u max error="
              << physical_velocity_maximum_error
              << ", raw-u max error=" << raw_velocity_maximum_error << '\n';
    test.expect(
        physical_velocity_maximum_error<=6.0E-4,
        prefix+
            " scalar equilibrium uses u_raw+F/(2 rho), not the pre-force"
            " momentum velocity");
    test.expect(
        physical_velocity_maximum_error<0.20*raw_velocity_maximum_error,
        prefix+
            " nonuniform buoyant scalar transport is decisively closer to"
            " the Guo physical-velocity prediction");
}

void test_force_aware_equilibrium_velocity_boundary(
    TestState& test, const Temperature_Lattice lattice) {
    constexpr uint N = 8u;
    constexpr float prescribed_x = 0.030f;
    constexpr float prescribed_y = -0.020f;
    constexpr float prescribed_z = 0.010f;
    constexpr float fixed_temperature = 1.05f;
    constexpr float beta = 0.50f;
    constexpr float gravity_x = -0.80f;
    constexpr float gravity_y = 0.40f;
    constexpr float gravity_z = -0.60f;
    const std::string prefix = lattice_name(lattice);

    test.expect(set_temperature_lattice(lattice),
                prefix+
                    " temperature lattice selection succeeds before the"
                    " force-aware Dirichlet test");
    culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, beta));
    initialize_quiescent_fields(
        lbm, fixed_temperature, (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL));
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        lbm.velocity().x[n] = prescribed_x;
        lbm.velocity().y[n] = prescribed_y;
        lbm.velocity().z[n] = prescribed_z;
    }
    lbm.set_gravity(gravity_x, gravity_y, gravity_z);

    lbm.initialize_once();
    lbm.advance();
    lbm.velocity().download();
    double prescribed_macro_maximum_error = 0.0;
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        prescribed_macro_maximum_error = std::max(
            prescribed_macro_maximum_error,
            std::fabs((double)lbm.velocity().x[n]-(double)prescribed_x));
        prescribed_macro_maximum_error = std::max(
            prescribed_macro_maximum_error,
            std::fabs((double)lbm.velocity().y[n]-(double)prescribed_y));
        prescribed_macro_maximum_error = std::max(
            prescribed_macro_maximum_error,
            std::fabs((double)lbm.velocity().z[n]-(double)prescribed_z));
    }

    // Turn the periodic test box into ordinary fluid while retaining the same
    // uniform temperature and force.  Its next physical velocity is therefore
    // the first moment of the populations emitted by CELL_EQUILIBRIUM plus the ordinary
    // Guo F/(2 rho) correction.  A strict force-aware boundary emits
    // feq(u_bc)-S(u_bc,F)/2, so this reconstruction must recover u_bc.
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) lbm.flags()[n] = (uchar)0u;
    lbm.flags().upload();
    lbm.advance();
    lbm.velocity().download();

    double emitted_population_maximum_error = 0.0;
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        emitted_population_maximum_error = std::max(
            emitted_population_maximum_error,
            std::fabs((double)lbm.velocity().x[n]-(double)prescribed_x));
        emitted_population_maximum_error = std::max(
            emitted_population_maximum_error,
            std::fabs((double)lbm.velocity().y[n]-(double)prescribed_y));
        emitted_population_maximum_error = std::max(
            emitted_population_maximum_error,
            std::fabs((double)lbm.velocity().z[n]-(double)prescribed_z));
    }
    lbm.synchronize();

    std::cout << prefix
              << " force-aware CELL_EQUILIBRIUM|CELL_THERMAL: prescribed macro max error="
              << prescribed_macro_maximum_error
              << ", emitted-population max error="
              << emitted_population_maximum_error << '\n';
    test.expect(
        prescribed_macro_maximum_error<=1.0E-7,
        prefix+
            " CELL_EQUILIBRIUM|CELL_THERMAL retains the user-prescribed physical velocity"
            " under nonzero buoyancy");
    test.expect(
        emitted_population_maximum_error<=6.0E-4,
        prefix+
            " CELL_EQUILIBRIUM|CELL_THERMAL emits equilibrium populations at the prescribed"
            " physical velocity without an extra F/(2 rho) offset");
}

void test_force_aware_equilibrium_velocity_boundary_initialization(
    TestState& test, const Temperature_Lattice lattice) {
    constexpr uint N = 8u;
    constexpr float prescribed_x = 0.030f;
    constexpr float prescribed_y = -0.020f;
    constexpr float prescribed_z = 0.010f;
    constexpr float fixed_temperature = 1.05f;
    constexpr float beta = 0.50f;
    constexpr float gravity_x = -0.80f;
    constexpr float gravity_y = 0.40f;
    constexpr float gravity_z = -0.60f;
    constexpr float global_force_x = 0.006f;
    constexpr float global_force_y = -0.004f;
    constexpr float global_force_z = 0.002f;
    const std::string prefix = lattice_name(lattice);

    test.expect(set_temperature_lattice(lattice),
                prefix+
                    " temperature lattice selection succeeds before the"
                    " force-aware Dirichlet initialization test");
    culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).volume_force(global_force_x, global_force_y, global_force_z).thermal(kAlpha, beta));
    initialize_quiescent_fields(
        lbm, fixed_temperature, (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL));
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        lbm.velocity().x[n] = prescribed_x;
        lbm.velocity().y[n] = prescribed_y;
        lbm.velocity().z[n] = prescribed_z;
    }
    lbm.set_gravity(gravity_x, gravity_y, gravity_z);

    // Do not execute a CELL_EQUILIBRIUM collision.  Releasing the boundary immediately
    // after production initialization isolates the initial populations.  They
    // must already encode raw momentum rho*u_bc-F_total/2, including both the
    // global body force and the temperature-dependent Boussinesq force.
    lbm.initialize_once();
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) lbm.flags()[n] = (uchar)0u;
    lbm.flags().upload();
    lbm.advance();
    lbm.velocity().download();

    double initialized_population_maximum_error = 0.0;
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        initialized_population_maximum_error = std::max(
            initialized_population_maximum_error,
            std::fabs((double)lbm.velocity().x[n]-(double)prescribed_x));
        initialized_population_maximum_error = std::max(
            initialized_population_maximum_error,
            std::fabs((double)lbm.velocity().y[n]-(double)prescribed_y));
        initialized_population_maximum_error = std::max(
            initialized_population_maximum_error,
            std::fabs((double)lbm.velocity().z[n]-(double)prescribed_z));
    }
    lbm.synchronize();

    std::cout << prefix
              << " force-aware CELL_EQUILIBRIUM|CELL_THERMAL initialization: emitted-"
                 "population max error="
              << initialized_population_maximum_error << '\n';
    test.expect(
        initialized_population_maximum_error<=6.0E-4,
        prefix+
            " CELL_EQUILIBRIUM|CELL_THERMAL initialization emits populations at raw momentum"
            " rho*u_bc-F_total/2 before the first collision");
}

void test_dynamic_free_slip_boundaries(
    TestState& test, const Temperature_Lattice lattice,
    const uint steps = 160u) {
    constexpr uint Nx = 36u;
    constexpr uint Ny = 8u;
    constexpr uint Nz = 20u;
    constexpr float tangential_velocity = 0.025f;
    constexpr uint z_face_mask = 16u | 32u;
    const std::string prefix = lattice_name(lattice);

    test.expect(set_temperature_lattice(lattice),
                prefix+
                " temperature lattice selection succeeds before free-slip test");
    culbm::solver::Solver lbm(Nx, Ny, Nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    initialize_quiescent_fields(lbm, 1.0f);
    for(uint z=0u; z<Nz; ++z) {
        for(uint y=0u; y<Ny; ++y) {
            for(uint x=0u; x<Nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                lbm.velocity().x[n] = tangential_velocity;
                if(z==0u || z==Nz-1u) {
                    lbm.flags()[n] = (uchar)(CELL_SOLID|CELL_AUX_X);
                    // Deliberately inconsistent ghost macros verify that
                    // specular closure uses the fluid-node outgoing DDFs.
                    lbm.velocity().x[n] = -0.05f;
                    lbm.velocity().z[n] = 0.03f;
                    lbm.temperature()[n] = 0.75f;
                }
            }
        }
    }

    lbm.initialize_once();
    lbm.initialize_once(); lbm.free_slip(z_face_mask);
    lbm.velocity().download();
    lbm.temperature().download();
    for(uint y=0u; y<Ny; ++y) {
        for(uint x=0u; x<Nx; ++x) {
            for(const uint z : {0u, Nz-1u}) {
                const ulong n = lbm.grid().index(x, y, z);
                test.expect_close(
                    lbm.velocity().x[n], tangential_velocity, 1.0E-7,
                    prefix+
                    " free-slip wall copies adjacent tangential velocity");
                test.expect_close(
                    lbm.velocity().z[n], 0.0, 1.0E-7,
                    prefix+
                    " free-slip wall removes wall-normal velocity");
                test.expect_close(
                    lbm.temperature()[n], 1.0, 1.0E-7,
                    prefix+
                    " free-slip wall applies zero-normal-gradient temperature");
            }
        }
    }

    for(uint step=0u; step<steps; ++step) {
        lbm.initialize_once(); lbm.free_slip(z_face_mask);
        lbm.advance();
    }
    lbm.synchronize();
    lbm.velocity().download();
    lbm.temperature().download();

    double tangential_mean = 0.0;
    double maximum_normal_velocity = 0.0;
    double maximum_temperature_error = 0.0;
    uint maximum_normal_x = 0u;
    uint maximum_normal_y = 0u;
    uint maximum_normal_z = 0u;
    std::uint64_t interior_count = 0u;
    for(uint z=1u; z<Nz-1u; ++z) {
        for(uint y=0u; y<Ny; ++y) {
            for(uint x=0u; x<Nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                tangential_mean += (double)lbm.velocity().x[n];
                const double normal_velocity =
                    std::fabs((double)lbm.velocity().z[n]);
                if(normal_velocity>maximum_normal_velocity) {
                    maximum_normal_velocity = normal_velocity;
                    maximum_normal_x = x;
                    maximum_normal_y = y;
                    maximum_normal_z = z;
                }
                maximum_temperature_error = std::max(
                    maximum_temperature_error,
                    std::fabs((double)lbm.temperature()[n]-1.0));
                ++interior_count;
            }
        }
    }
    tangential_mean /= (double)interior_count;
    std::cout << prefix
              << " dynamic free-slip: mean tangential velocity="
              << tangential_mean
              << ", max normal velocity=" << maximum_normal_velocity
              << " at (" << maximum_normal_x << ',' << maximum_normal_y
              << ',' << maximum_normal_z << ')'
              << ", max temperature error=" << maximum_temperature_error
              << '\n';
    if(steps<=4u) {
        for(uint z=1u; z<Nz-1u; ++z) {
            double layer_ux = 0.0;
            double layer_uz = 0.0;
            double layer_T = 0.0;
            for(uint y=0u; y<Ny; ++y) {
                for(uint x=0u; x<Nx; ++x) {
                    const ulong n = lbm.grid().index(x, y, z);
                    layer_ux += (double)lbm.velocity().x[n];
                    layer_uz += (double)lbm.velocity().z[n];
                    layer_T += (double)lbm.temperature()[n];
                }
            }
            const double inverse_layer =
                1.0/(double)((std::uint64_t)Nx*(std::uint64_t)Ny);
            std::cout << prefix << " free-slip z=" << z
                      << ": mean ux=" << layer_ux*inverse_layer
                      << ", mean uz=" << layer_uz*inverse_layer
                      << ", mean T=" << layer_T*inverse_layer << '\n';
        }
    }
    test.expect_close(
        tangential_mean, tangential_velocity, 2.0E-4,
        prefix+
        " free-slip walls preserve a uniform tangential flow");
    test.expect(
        maximum_normal_velocity<=2.0E-5,
        prefix+
        " free-slip walls do not create appreciable normal flow");
    test.expect(
        maximum_temperature_error<=5.0E-4,
        prefix+
        " adiabatic walls preserve a uniform scalar");
}

void test_free_slip_xz_corners(
    TestState& test, const Temperature_Lattice lattice) {
    constexpr uint Nx = 24u;
    constexpr uint Ny = 8u;
    constexpr uint Nz = 18u;
    constexpr uint steps = 240u;
    constexpr float tangential_velocity = 0.02f;
    constexpr uint xz_face_mask = 1u | 2u | 16u | 32u;
    const std::string prefix = lattice_name(lattice);

    test.expect(set_temperature_lattice(lattice),
                prefix+
                " temperature lattice selection succeeds before corner test");
    culbm::solver::Solver lbm(Nx, Ny, Nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    initialize_quiescent_fields(lbm, 1.0f);
    for(uint z=0u; z<Nz; ++z) {
        for(uint y=0u; y<Ny; ++y) {
            for(uint x=0u; x<Nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                lbm.velocity().y[n] = tangential_velocity;
                if(x==0u || x==Nx-1u || z==0u || z==Nz-1u) {
                    lbm.flags()[n] = (uchar)(CELL_SOLID|CELL_AUX_X);
                    lbm.velocity().x[n] = 0.03f;
                    lbm.velocity().y[n] = -0.04f;
                    lbm.velocity().z[n] = -0.02f;
                    lbm.temperature()[n] = 0.8f;
                }
            }
        }
    }

    lbm.initialize_once();
    for(uint step=0u; step<steps; ++step) {
        lbm.initialize_once(); lbm.free_slip(xz_face_mask);
        lbm.advance();
    }
    lbm.synchronize();
    lbm.velocity().download();
    lbm.temperature().download();

    double tangential_mean = 0.0;
    double maximum_normal_velocity = 0.0;
    double maximum_temperature_error = 0.0;
    std::uint64_t interior_count = 0u;
    for(uint z=1u; z<Nz-1u; ++z) {
        for(uint y=0u; y<Ny; ++y) {
            for(uint x=1u; x<Nx-1u; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                tangential_mean += (double)lbm.velocity().y[n];
                maximum_normal_velocity = std::max(
                    maximum_normal_velocity,
                    std::max(std::fabs((double)lbm.velocity().x[n]),
                             std::fabs((double)lbm.velocity().z[n])));
                maximum_temperature_error = std::max(
                    maximum_temperature_error,
                    std::fabs((double)lbm.temperature()[n]-1.0));
                ++interior_count;
            }
        }
    }
    tangential_mean /= (double)interior_count;
    std::cout << prefix
              << " four-wall free-slip: mean tangential velocity="
              << tangential_mean
              << ", max wall-normal velocity="
              << maximum_normal_velocity
              << ", max temperature error="
              << maximum_temperature_error << '\n';
    test.expect_close(
        tangential_mean, tangential_velocity, 2.0E-4,
        prefix+
        " x/z free-slip walls preserve corner-tangential flow");
    test.expect(
        maximum_normal_velocity<=2.0E-5,
        prefix+
        " x/z free-slip walls and corners remain impermeable");
    test.expect(
        maximum_temperature_error<=5.0E-4,
        prefix+
        " x/z adiabatic walls and corners preserve uniform scalar");
}

void test_nested_temperature_bridge(TestState& test,
                                    const Temperature_Lattice lattice) {
    constexpr uint upper_nx = 8u;
    constexpr uint upper_ny = 8u;
    constexpr uint lower_nx = 2u*upper_nx;
    constexpr uint lower_ny = 2u*upper_ny;
    constexpr uint nz = 4u;
    constexpr float upper_donor_temperature = 1.18f;
    constexpr float lower_donor_temperature = 0.82f;
    const std::string prefix = lattice_name(lattice);

    test.expect(set_temperature_lattice(lattice),
                prefix+" temperature lattice selection succeeds before nested bridge test");
    culbm::solver::Solver upper(upper_nx, upper_ny, nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    culbm::solver::Solver lower(lower_nx, lower_ny, nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    initialize_quiescent_fields(upper, 1.0f);
    initialize_quiescent_fields(lower, 1.0f);

    // z=1 in the upper block and z=Nz-2 in the lower block are the donor
    // rows.  Seed CELL_THERMAL on both ghost rows deliberately verifies that closed
    // aperture slots are reset to adiabatic solids instead of inheriting it.
    for(uint y=0u; y<upper_ny; ++y) {
        for(uint x=0u; x<upper_nx; ++x) {
            upper.temperature()[upper.grid().index(x, y, 1u)] = upper_donor_temperature;
            upper.flags()[upper.grid().index(x, y, 0u)] = CELL_THERMAL;
        }
    }
    for(uint y=0u; y<lower_ny; ++y) {
        for(uint x=0u; x<lower_nx; ++x) {
            lower.temperature()[lower.grid().index(x, y, nz-2u)] = lower_donor_temperature;
            lower.flags()[lower.grid().index(x, y, nz-1u)] = CELL_THERMAL;
        }
    }
    // The first 2x2 fine donor group closes one coarse aperture; all other
    // groups remain open.
    for(uint y=0u; y<2u; ++y) {
        for(uint x=0u; x<2u; ++x) {
            lower.flags()[lower.grid().index(x, y, nz-2u)] = CELL_SOLID;
        }
    }

    upper.initialize_once();
    lower.initialize_once();
    culbm::NestedLbmLevel upper_level;
    upper_level.solver = &upper;
    upper_level.refinement_to_parent = 1u;
    upper_level.dt_si = 1.0;
    upper_level.lower_ghost_z = 0u;
    upper_level.upper_ghost_z = nz-1u;
    upper_level.physical_first_z = 1u;
    upper_level.physical_end_z = nz-1u;
    culbm::NestedLbmLevel lower_level;
    lower_level.solver = &lower;
    lower_level.refinement_to_parent = 2u;
    lower_level.dt_si = 0.5;
    lower_level.lower_ghost_z = 0u;
    lower_level.upper_ghost_z = nz-1u;
    lower_level.physical_first_z = 0u;
    lower_level.physical_end_z = nz-1u;

    culbm::NestedMacroPlaneBridge::exchange(upper_level, lower_level);
    const auto upper_boundary = culbm::NestedMacroPlaneBridge::read(
        upper_level, culbm::NestedPlaneSide::Bottom,
        culbm::NestedPlaneLocation::Boundary);
    const auto lower_boundary = culbm::NestedMacroPlaneBridge::read(
        lower_level, culbm::NestedPlaneSide::Top,
        culbm::NestedPlaneLocation::Boundary);
    const auto plane_index = [](const uint x, const uint y, const uint nx) {
        return (std::size_t)x+(std::size_t)y*(std::size_t)nx;
    };
    const std::size_t closed_upper = plane_index(0u, 0u, upper_nx);
    test.expect(
        (upper_boundary.flags[closed_upper] & CELL_SOLID) != 0u &&
        (upper_boundary.flags[closed_upper] & CELL_THERMAL) == 0u,
        prefix+" nested closed coarse aperture is an adiabatic solid");
    for(uint y=0u; y<2u; ++y) {
        for(uint x=0u; x<2u; ++x) {
            const std::size_t closed_lower = plane_index(x, y, lower_nx);
            test.expect(
                (lower_boundary.flags[closed_lower] & CELL_SOLID) != 0u &&
                (lower_boundary.flags[closed_lower] & CELL_THERMAL) == 0u,
                prefix+" nested closed fine aperture slot is an adiabatic solid");
        }
    }

    const std::size_t open_upper = plane_index(2u, 2u, upper_nx);
    const std::size_t open_lower = plane_index(4u, 4u, lower_nx);
    test.expect(
        (upper_boundary.flags[open_upper] &
         (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL)) == (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL),
        prefix+" nested open coarse interface consumes the bridged temperature");
    test.expect(
        (lower_boundary.flags[open_lower] &
         (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL)) == (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL),
        prefix+" nested open fine interface consumes the bridged temperature");
    test.expect_close(
        upper_boundary.temperature[open_upper],
        lower_donor_temperature, 1.0E-6,
        prefix+" nested fine-to-coarse restriction writes the donor temperature");
    test.expect_close(
        lower_boundary.temperature[open_lower],
        upper_donor_temperature, 1.0E-6,
        prefix+" nested coarse-to-fine prolongation writes the donor temperature");

    // One real production-kernel step is essential here: without CELL_THERMAL the
    // newly written macro T would be discarded and reconstructed from stale
    // ghost gi during this step.
    upper.advance();
    lower.advance();
    const auto upper_after_step = culbm::NestedMacroPlaneBridge::read(
        upper_level, culbm::NestedPlaneSide::Bottom,
        culbm::NestedPlaneLocation::Boundary);
    const auto lower_after_step = culbm::NestedMacroPlaneBridge::read(
        lower_level, culbm::NestedPlaneSide::Top,
        culbm::NestedPlaneLocation::Boundary);
    test.expect_close(
        upper_after_step.temperature[open_upper],
        lower_donor_temperature, 1.0E-6,
        prefix+" nested coarse ghost keeps the exchanged temperature through collision");
    test.expect_close(
        lower_after_step.temperature[open_lower],
        upper_donor_temperature, 1.0E-6,
        prefix+" nested fine ghost keeps the exchanged temperature through collision");
    upper.synchronize();
    lower.synchronize();
}

constexpr uint kNestedSeamUpperNx = 8u;
constexpr uint kNestedSeamUpperNy = 8u;
constexpr uint kNestedSeamLowerNx = 2u*kNestedSeamUpperNx;
constexpr uint kNestedSeamLowerNy = 2u*kNestedSeamUpperNy;
constexpr uint kNestedSeamNz = 4u;
constexpr uint kNestedSeamSteps = 2u;

bool is_nested_seam_closed_cell(const bool fine,
                                const uint x,
                                const uint y) {
    const uint nx = fine ? kNestedSeamLowerNx : kNestedSeamUpperNx;
    const uint ny = fine ? kNestedSeamLowerNy : kNestedSeamUpperNy;
    const uint half_width = fine ? 2u : 1u;
    return x>=nx/2u-half_width && x<nx/2u+half_width &&
           y>=ny/2u-half_width && y<ny/2u+half_width;
}

struct NestedSeamComparisonField {
    std::array<std::vector<float>, kNestedSeamSteps> upper_temperature;
    std::array<std::vector<float>, kNestedSeamSteps> lower_temperature;
    bool lattice_selected = false;
    bool expected_topology = false;
    bool distinct_devices = true;
    bool aperture_geometry_valid = false;
    bool closed_aperture_straddles_seam = true;
};

std::vector<float> capture_temperature_field(culbm::solver::Solver& lbm) {
    lbm.temperature().download();
    std::vector<float> result((std::size_t)lbm.grid().cells());
    for(ulong n=0ull; n<lbm.grid().cells(); ++n)
        result[(std::size_t)n] = lbm.temperature()[n];
    return result;
}

NestedSeamComparisonField solve_nested_seam_comparison(
    const uint Dx, const uint Dy) {
    NestedSeamComparisonField result;
    result.lattice_selected =
        set_temperature_lattice(Temperature_Lattice::Q19);
    culbm::solver::Solver upper(kNestedSeamUpperNx, kNestedSeamUpperNy, kNestedSeamNz, Dx, Dy, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    culbm::solver::Solver lower(kNestedSeamLowerNx, kNestedSeamLowerNy, kNestedSeamNz, Dx, Dy, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    const uint domain_count = Dx*Dy;
    const auto has_expected_topology =
        [Dx, Dy, domain_count](const culbm::solver::Solver& lbm) {
            return !lbm.distributed() &&
                   lbm.grid().split(0u)==Dx &&
                   lbm.grid().split(1u)==Dy &&
                   lbm.grid().split(2u)==1u &&
                   lbm.local_count()==domain_count &&
                   lbm.grid().count()==domain_count;
        };
    result.expected_topology =
        has_expected_topology(upper) &&
        has_expected_topology(lower);
    if(domain_count>1u) {
        result.distinct_devices =
            upper.local(0u).device().ordinal() !=
                upper.local(1u).device().ordinal() &&
            lower.local(0u).device().ordinal() !=
                lower.local(1u).device().ordinal();
    }

    initialize_quiescent_fields(upper, 1.0f);
    initialize_quiescent_fields(lower, 1.0f);
    for(uint z=0u; z<kNestedSeamNz; ++z) {
        for(uint y=0u; y<kNestedSeamUpperNy; ++y) {
            for(uint x=0u; x<kNestedSeamUpperNx; ++x) {
                const ulong n = upper.grid().index(x, y, z);
                const double phase = 2.0*kPi*(
                    (double)x/(double)kNestedSeamUpperNx+
                    0.31*(double)y/(double)kNestedSeamUpperNy+
                    0.17*(double)z/(double)kNestedSeamNz);
                upper.temperature()[n] =
                    (float)(0.96+0.055*std::sin(phase));
            }
        }
        for(uint y=0u; y<kNestedSeamLowerNy; ++y) {
            for(uint x=0u; x<kNestedSeamLowerNx; ++x) {
                const ulong n = lower.grid().index(x, y, z);
                const double phase = 2.0*kPi*(
                    0.43*(double)x/(double)kNestedSeamLowerNx-
                    0.29*(double)y/(double)kNestedSeamLowerNy+
                    0.23*(double)z/(double)kNestedSeamNz);
                lower.temperature()[n] =
                    (float)(1.04+0.047*std::cos(phase));
            }
        }
    }

    // These deliberately stale interface values initialize gi before the
    // bridge classifies the aperture.  If the bridge does not publish its
    // new CELL_SOLID flags to neighboring-domain halos, D3Q19 diagonal links
    // stream these values through the closed patch instead of bouncing back.
    for(uint y=0u; y<kNestedSeamUpperNy; ++y) {
        for(uint x=0u; x<kNestedSeamUpperNx; ++x) {
            const ulong n = upper.grid().index(x, y, 0u);
            upper.flags()[n] = CELL_THERMAL;
            upper.temperature()[n] =
                0.56f+0.003f*(float)x+0.002f*(float)y;
        }
    }
    for(uint y=0u; y<kNestedSeamLowerNy; ++y) {
        for(uint x=0u; x<kNestedSeamLowerNx; ++x) {
            const ulong n =
                lower.grid().index(x, y, kNestedSeamNz-1u);
            lower.flags()[n] = CELL_THERMAL;
            lower.temperature()[n] =
                1.44f-0.002f*(float)x+0.001f*(float)y;
            if(is_nested_seam_closed_cell(true, x, y)) {
                lower.flags()[
                    lower.grid().index(x, y, kNestedSeamNz-2u)] = CELL_SOLID;
            }
        }
    }

    upper.initialize_once();
    lower.initialize_once();
    culbm::NestedLbmLevel upper_level;
    upper_level.solver = &upper;
    upper_level.refinement_to_parent = 1u;
    upper_level.dt_si = 1.0;
    upper_level.lower_ghost_z = 0u;
    upper_level.upper_ghost_z = kNestedSeamNz-1u;
    upper_level.physical_first_z = 1u;
    upper_level.physical_end_z = kNestedSeamNz-1u;
    culbm::NestedLbmLevel lower_level;
    lower_level.solver = &lower;
    lower_level.refinement_to_parent = 2u;
    lower_level.dt_si = 0.5;
    lower_level.lower_ghost_z = 0u;
    lower_level.upper_ghost_z = kNestedSeamNz-1u;
    lower_level.physical_first_z = 0u;
    lower_level.physical_end_z = kNestedSeamNz-1u;

    for(uint step=0u; step<kNestedSeamSteps; ++step) {
        culbm::NestedMacroPlaneBridge::exchange(
            upper_level, lower_level);
        if(step==0u) {
            const culbm::NestedMacroPlane upper_boundary =
                culbm::NestedMacroPlaneBridge::read(
                    upper_level, culbm::NestedPlaneSide::Bottom,
                    culbm::NestedPlaneLocation::Boundary);
            const culbm::NestedMacroPlane lower_boundary =
                culbm::NestedMacroPlaneBridge::read(
                    lower_level, culbm::NestedPlaneSide::Top,
                    culbm::NestedPlaneLocation::Boundary);
            result.aperture_geometry_valid =
                upper_boundary.valid() && lower_boundary.valid();
            const auto check_aperture =
                [&result](const culbm::NestedMacroPlane& plane,
                          const bool fine) {
                    for(uint y=0u; y<plane.ny; ++y) {
                        for(uint x=0u; x<plane.nx; ++x) {
                            const std::size_t n =
                                (std::size_t)x+
                                (std::size_t)y*(std::size_t)plane.nx;
                            const uchar flag = plane.flags[n];
                            const bool closed =
                                is_nested_seam_closed_cell(fine, x, y);
                            const bool valid = closed
                                ? (flag&CELL_SOLID)!=0u &&
                                  (flag&CELL_THERMAL)==0u
                                : (flag&(uchar)(CELL_EQUILIBRIUM|CELL_THERMAL))==
                                      (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL) &&
                                  (flag&CELL_SOLID)==0u;
                            result.aperture_geometry_valid =
                                result.aperture_geometry_valid && valid;
                        }
                    }
                };
            check_aperture(upper_boundary, false);
            check_aperture(lower_boundary, true);

            const auto solid_at =
                [](const culbm::NestedMacroPlane& plane,
                   const uint x, const uint y) {
                    const std::size_t n =
                        (std::size_t)x+
                        (std::size_t)y*(std::size_t)plane.nx;
                    return (plane.flags[n]&CELL_SOLID)!=0u;
                };
            if(Dx==2u) {
                result.closed_aperture_straddles_seam =
                    solid_at(
                        upper_boundary,
                        kNestedSeamUpperNx/2u-1u,
                        kNestedSeamUpperNy/2u-1u) &&
                    solid_at(
                        upper_boundary,
                        kNestedSeamUpperNx/2u,
                        kNestedSeamUpperNy/2u-1u) &&
                    solid_at(
                        lower_boundary,
                        kNestedSeamLowerNx/2u-1u,
                        kNestedSeamLowerNy/2u-1u) &&
                    solid_at(
                        lower_boundary,
                        kNestedSeamLowerNx/2u,
                        kNestedSeamLowerNy/2u-1u);
            } else if(Dy==2u) {
                result.closed_aperture_straddles_seam =
                    solid_at(
                        upper_boundary,
                        kNestedSeamUpperNx/2u-1u,
                        kNestedSeamUpperNy/2u-1u) &&
                    solid_at(
                        upper_boundary,
                        kNestedSeamUpperNx/2u-1u,
                        kNestedSeamUpperNy/2u) &&
                    solid_at(
                        lower_boundary,
                        kNestedSeamLowerNx/2u-1u,
                        kNestedSeamLowerNy/2u-1u) &&
                    solid_at(
                        lower_boundary,
                        kNestedSeamLowerNx/2u-1u,
                        kNestedSeamLowerNy/2u);
            }
        }

        // Consecutive t=0/t=1 production steps exercise both Esoteric-Pull
        // parities after the bridge has synchronized the new interface flags.
        upper.advance();
        lower.advance();
        result.upper_temperature[step] =
            capture_temperature_field(upper);
        result.lower_temperature[step] =
            capture_temperature_field(lower);
    }
    upper.synchronize();
    lower.synchronize();
    return result;
}

void test_multigpu_nested_temperature_bridge(TestState& test) {
    const NestedSeamComparisonField reference =
        solve_nested_seam_comparison(1u, 1u);
    test.expect(
        reference.lattice_selected,
        "D3Q19 can be selected before the nested seam reference");
    test.expect(
        reference.expected_topology,
        "D3Q19 nested seam reference uses one local CUDA domain per level");
    test.expect(
        reference.aperture_geometry_valid,
        "D3Q19 nested seam reference creates the exact centered closed aperture");

    struct Decomposition {
        uint dx;
        uint dy;
        uint axis;
        const char* label;
    };
    for(const Decomposition decomposition : {
        Decomposition{2u, 1u, 0u, "Dx=2"},
        Decomposition{1u, 2u, 1u, "Dy=2"}}) {
        const NestedSeamComparisonField two_gpu =
            solve_nested_seam_comparison(
                decomposition.dx, decomposition.dy);
        const std::string case_name = decomposition.label;
        test.expect(
            two_gpu.lattice_selected,
            "D3Q19 can be selected before nested seam "+case_name);
        test.expect(
            two_gpu.expected_topology,
            "D3Q19 nested seam "+case_name+
                " uses the requested two-domain topology on both levels");
        test.expect(
            two_gpu.distinct_devices,
            "D3Q19 nested seam "+case_name+
                " assigns distinct CUDA devices on both levels");
        test.expect(
            two_gpu.aperture_geometry_valid,
            "D3Q19 nested seam "+case_name+
                " preserves the exact open/closed interface flags");
        test.expect(
            two_gpu.closed_aperture_straddles_seam,
            "D3Q19 nested seam "+case_name+
                " places closed interface cells on both sides of the partition");

        const auto compare_level =
            [&](const std::vector<float>& expected,
                const std::vector<float>& actual,
                const uint nx,
                const uint ny,
                const bool upper_level,
                const uint step) {
                const std::string level_name =
                    upper_level ? "upper" : "lower";
                test.expect(
                    expected.size()==actual.size() && !expected.empty(),
                    "D3Q19 nested seam "+case_name+" "+level_name+
                        " step "+std::to_string(step+1u)+
                        " has the same non-empty global extent as one GPU");
                if(expected.size()!=actual.size() || expected.empty())
                    return;

                double squared_error_sum = 0.0;
                double maximum_error = 0.0;
                double seam_maximum_error = 0.0;
                double interface_seam_maximum_error = 0.0;
                for(uint z=0u; z<kNestedSeamNz; ++z) {
                    for(uint y=0u; y<ny; ++y) {
                        for(uint x=0u; x<nx; ++x) {
                            const std::size_t n =
                                (std::size_t)x+
                                ((std::size_t)y+
                                 (std::size_t)z*(std::size_t)ny)*
                                (std::size_t)nx;
                            const double error = std::fabs(
                                (double)actual[n]-
                                (double)expected[n]);
                            squared_error_sum += error*error;
                            maximum_error =
                                std::max(maximum_error, error);
                            const uint coordinate =
                                decomposition.axis==0u ? x : y;
                            const uint extent =
                                decomposition.axis==0u ? nx : ny;
                            const bool on_seam =
                                coordinate==extent/2u-1u ||
                                coordinate==extent/2u;
                            if(on_seam) {
                                seam_maximum_error =
                                    std::max(
                                        seam_maximum_error, error);
                            }
                            const bool on_nested_interface =
                                upper_level ? z<=1u
                                            : z>=kNestedSeamNz-2u;
                            if(on_seam && on_nested_interface) {
                                interface_seam_maximum_error =
                                    std::max(
                                        interface_seam_maximum_error,
                                        error);
                            }
                        }
                    }
                }
                const double rms_error = std::sqrt(
                    squared_error_sum/(double)expected.size());
                std::cout
                    << "D3Q19 nested bridge " << case_name
                    << " " << level_name
                    << " after step " << step+1u
                    << " versus one GPU: max error="
                    << maximum_error
                    << ", seam max error="
                    << seam_maximum_error
                    << ", interface/seam max error="
                    << interface_seam_maximum_error
                    << ", RMS error=" << rms_error << '\n';
                test.expect(
                    maximum_error<=2.0E-4,
                    "D3Q19 nested seam "+case_name+" "+level_name+
                        " step "+std::to_string(step+1u)+
                        " full temperature field agrees pointwise with one GPU");
                test.expect(
                    seam_maximum_error<=2.0E-4,
                    "D3Q19 nested seam "+case_name+" "+level_name+
                        " step "+std::to_string(step+1u)+
                        " agrees across the decomposition seam");
                test.expect(
                    interface_seam_maximum_error<=2.0E-4,
                    "D3Q19 nested seam "+case_name+" "+level_name+
                        " step "+std::to_string(step+1u)+
                        " agrees where the closed aperture crosses the seam");
                test.expect(
                    rms_error<=5.0E-5,
                    "D3Q19 nested seam "+case_name+" "+level_name+
                        " step "+std::to_string(step+1u)+
                        " has a small global RMS difference");
            };
        for(uint step=0u; step<kNestedSeamSteps; ++step) {
            compare_level(
                reference.upper_temperature[step],
                two_gpu.upper_temperature[step],
                kNestedSeamUpperNx, kNestedSeamUpperNy,
                true, step);
            compare_level(
                reference.lower_temperature[step],
                two_gpu.lower_temperature[step],
                kNestedSeamLowerNx, kNestedSeamLowerNy,
                false, step);
        }
    }
}

void test_temperature_zero_gradient_outlet(
    TestState& test, const Temperature_Lattice lattice) {
    constexpr uint Nx = 18u;
    constexpr uint Ny = 16u;
    constexpr uint Nz = 8u;
    const std::string prefix = lattice_name(lattice);
    struct OutletCase {
        Temperature_Outlet_Face face;
        const char* name;
    };
    for(const OutletCase outlet : {
        OutletCase{Temperature_Outlet_Face::West, "west"},
        OutletCase{Temperature_Outlet_Face::East, "east"},
        OutletCase{Temperature_Outlet_Face::South, "south"},
        OutletCase{Temperature_Outlet_Face::North, "north"}}) {
        test.expect(set_temperature_lattice(lattice),
                    prefix+" lattice selection succeeds before scalar outlet test");
        culbm::solver::Solver lbm(Nx, Ny, Nz, 1u, 1u, 1u,
            culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
        test.expect(
            !lbm.set_temperature_outlet_face(99) &&
            lbm.physics().temperature_outlet_face==0,
            prefix+" rejects an invalid scalar outlet face without changing state");
        test.expect(
            lbm.set_temperature_outlet_face(outlet.face),
            prefix+" accepts the "+outlet.name+" scalar outlet face");
        initialize_quiescent_fields(lbm, 1.0f);
        for(uint z=0u; z<Nz; ++z) {
            for(uint y=0u; y<Ny; ++y) {
                for(uint x=0u; x<Nx; ++x) {
                    const double phase =
                        2.0*kPi*((double)x/(double)Nx+
                                 0.5*(double)y/(double)Ny+
                                 0.25*(double)z/(double)Nz);
                    lbm.temperature()[lbm.grid().index(x, y, z)] =
                        (float)(1.0+0.04*std::sin(phase));
                }
            }
        }

        uint boundary_x = Nx/2u;
        uint boundary_y = Ny/2u;
        uint inner_x = boundary_x;
        uint inner_y = boundary_y;
        if(outlet.face==Temperature_Outlet_Face::West) {
            boundary_x = 0u;
            inner_x = 1u;
        } else if(outlet.face==Temperature_Outlet_Face::East) {
            boundary_x = Nx-1u;
            inner_x = Nx-2u;
        } else if(outlet.face==Temperature_Outlet_Face::South) {
            boundary_y = 0u;
            inner_y = 1u;
        } else {
            boundary_y = Ny-1u;
            inner_y = Ny-2u;
        }
        const uint sample_z = Nz/2u;
        for(uint z=1u; z<Nz-1u; ++z) {
            if(outlet.face==Temperature_Outlet_Face::West ||
               outlet.face==Temperature_Outlet_Face::East) {
                for(uint y=0u; y<Ny; ++y)
                    lbm.flags()[lbm.grid().index(boundary_x, y, z)] = CELL_EQUILIBRIUM;
            } else {
                for(uint x=1u; x<Nx-1u; ++x)
                    lbm.flags()[lbm.grid().index(x, boundary_y, z)] = CELL_EQUILIBRIUM;
            }
        }

        // The top edge owns its independent fixed-temperature condition and
        // must not be overwritten by the side outlet pre-kernel.
        const uint protected_x =
            outlet.face==Temperature_Outlet_Face::South ||
            outlet.face==Temperature_Outlet_Face::North
                ? 0u : boundary_x;
        const uint protected_y =
            outlet.face==Temperature_Outlet_Face::South ||
            outlet.face==Temperature_Outlet_Face::North
                ? boundary_y : Ny/2u;
        const ulong protected_node =
            lbm.grid().index(protected_x, protected_y, Nz-1u);
        constexpr float protected_temperature = 1.31f;
        lbm.flags()[protected_node] = (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL);
        lbm.temperature()[protected_node] = protected_temperature;

        lbm.initialize_once();
        const ulong boundary_node =
            lbm.grid().index(boundary_x, boundary_y, sample_z);
        const ulong inner_node =
            lbm.grid().index(inner_x, inner_y, sample_z);
        // Make the macro array stale without touching gi.  A correct outlet
        // reconstructs from the interior DDF state instead of retaining this.
        lbm.temperature()[boundary_node] = 0.40f;
        lbm.temperature().upload();
        for(uint step=0u; step<3u; ++step) {
            lbm.advance();
            lbm.temperature().download();
            lbm.flags().download();
            const double boundary_temperature =
                (double)lbm.temperature()[boundary_node];
            const double inner_temperature =
                (double)lbm.temperature()[inner_node];
            std::cout << prefix << " " << outlet.name
                      << " scalar outlet step " << step+1u
                      << ": boundary=" << boundary_temperature
                      << ", inner=" << inner_temperature << '\n';
            test.expect_close(
                boundary_temperature, inner_temperature, 2.0E-5,
                prefix+" "+outlet.name+
                    " scalar outlet has zero normal gradient on both pull parities");
            test.expect(
                (lbm.flags()[boundary_node] &
                 (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL)) == (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL),
                prefix+" "+outlet.name+
                    " scalar outlet is promoted to a thermal equilibrium boundary");
            test.expect_close(
                lbm.temperature()[protected_node], protected_temperature, 1.0E-7,
                prefix+" "+outlet.name+
                    " scalar outlet preserves the higher-priority top/edge temperature");
        }
        test.expect(
            !lbm.set_temperature_outlet_face(Temperature_Outlet_Face::None) &&
            lbm.physics().temperature_outlet_face==(int)outlet.face,
            prefix+" scalar outlet face cannot change after initialization");
        lbm.synchronize();
        lbm.reset();
        test.expect(
            !lbm.set_temperature_outlet_face(Temperature_Outlet_Face::None) &&
            lbm.physics().temperature_outlet_face==(int)outlet.face,
            prefix+" scalar outlet face remains immutable after reset");
    }
}

constexpr uint kOutletComparisonNx = 24u;
constexpr uint kOutletComparisonNy = 20u;
constexpr uint kOutletComparisonNz = 8u;
constexpr uint kOutletComparisonOddSteps = 5u;
constexpr uint kOutletComparisonEvenSteps = 6u;

bool is_outlet_comparison_cell(
    const Temperature_Outlet_Face face,
    const uint x, const uint y, const uint z) {
    if(z==0u || z==kOutletComparisonNz-1u) return false;
    switch(face) {
    case Temperature_Outlet_Face::West:
        return x==0u;
    case Temperature_Outlet_Face::East:
        return x==kOutletComparisonNx-1u;
    case Temperature_Outlet_Face::South:
        return y==0u && x>0u && x<kOutletComparisonNx-1u;
    case Temperature_Outlet_Face::North:
        return y==kOutletComparisonNy-1u &&
               x>0u && x<kOutletComparisonNx-1u;
    default:
        return false;
    }
}

struct OutletComparisonField {
    std::vector<float> odd_temperature;
    std::vector<float> even_temperature;
    std::array<double, 2u> pull_parity_maximum_error = {0.0, 0.0};
    bool expected_topology = false;
    bool distinct_devices = true;
    bool outlet_configured = false;
    bool outlet_flags_valid = false;
    bool outlet_cells_present = false;
};

OutletComparisonField solve_outlet_comparison_field(
    const Temperature_Lattice lattice,
    const Temperature_Outlet_Face face,
    const uint Dx,
    const uint Dy) {
    set_temperature_lattice(lattice);
    culbm::solver::Solver lbm(kOutletComparisonNx, kOutletComparisonNy, kOutletComparisonNz, Dx, Dy, 1u,
        culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
    OutletComparisonField result;
    const uint domain_count = Dx*Dy;
    result.expected_topology =
        !lbm.distributed() &&
        lbm.grid().split(0u)==Dx &&
        lbm.grid().split(1u)==Dy &&
        lbm.grid().split(2u)==1u &&
        lbm.local_count()==domain_count &&
        lbm.grid().count()==domain_count;
    if(domain_count>1u) {
        result.distinct_devices =
            lbm.local(0u).device().ordinal() !=
            lbm.local(1u).device().ordinal();
    }
    result.outlet_configured = lbm.set_temperature_outlet_face(face);

    initialize_quiescent_fields(lbm, 1.0f);
    std::vector<std::array<ulong, 2u>> outlet_nodes;
    outlet_nodes.reserve(
        (std::size_t)std::max(kOutletComparisonNx, kOutletComparisonNy)*
        (std::size_t)kOutletComparisonNz);
    for(uint z=0u; z<kOutletComparisonNz; ++z) {
        for(uint y=0u; y<kOutletComparisonNy; ++y) {
            for(uint x=0u; x<kOutletComparisonNx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                const double phase_a = 2.0*kPi*(
                    (double)x/(double)kOutletComparisonNx+
                    0.37*(double)y/(double)kOutletComparisonNy+
                    0.19*(double)z/(double)kOutletComparisonNz);
                const double phase_b = 2.0*kPi*(
                    0.23*(double)x/(double)kOutletComparisonNx-
                    0.41*(double)y/(double)kOutletComparisonNy+
                    0.31*(double)z/(double)kOutletComparisonNz);
                lbm.temperature()[n] = (float)(
                    1.0+0.045*std::sin(phase_a)+
                    0.017*std::cos(phase_b));
                lbm.velocity().x[n] = 0.014f;
                lbm.velocity().y[n] = -0.009f;
                lbm.velocity().z[n] = 0.004f;
                if(!is_outlet_comparison_cell(face, x, y, z)) continue;

                uint inner_x = x;
                uint inner_y = y;
                if(face==Temperature_Outlet_Face::West) {
                    ++inner_x;
                } else if(face==Temperature_Outlet_Face::East) {
                    --inner_x;
                } else if(face==Temperature_Outlet_Face::South) {
                    ++inner_y;
                } else if(face==Temperature_Outlet_Face::North) {
                    --inner_y;
                }
                lbm.flags()[n] = CELL_EQUILIBRIUM;
                outlet_nodes.push_back(
                    {n, lbm.grid().index(inner_x, inner_y, z)});
            }
        }
    }
    result.outlet_cells_present = !outlet_nodes.empty();

    lbm.initialize_once();
    // Deliberately make only the macro outlet values stale.  Both the
    // single- and multi-domain runs must reconstruct them from the current
    // adjacent-interior gi state on every pull parity.
    for(const std::array<ulong, 2u>& nodes : outlet_nodes)
        lbm.temperature()[nodes[0]] = 0.40f;
    lbm.temperature().upload();

    for(uint step=0u; step<kOutletComparisonEvenSteps; ++step) {
        lbm.advance();
        lbm.temperature().download();
        double& parity_error =
            result.pull_parity_maximum_error[(std::size_t)(step%2u)];
        for(std::size_t i=0u; i<outlet_nodes.size(); ++i) {
            // stream_collide writes the interior macro reconstructed from the
            // same completed DDF state read by the outlet pre-kernel.  The
            // prior host T value is intentionally not used here: after a
            // collision it describes the previous pre-collision macro state.
            parity_error = std::max(
                parity_error,
                std::fabs(
                    (double)lbm.temperature()[outlet_nodes[i][0]]-
                    (double)lbm.temperature()[outlet_nodes[i][1]]));
        }
        if(step+1u==kOutletComparisonOddSteps) {
            result.odd_temperature.resize((std::size_t)lbm.grid().cells());
            for(ulong n=0ull; n<lbm.grid().cells(); ++n)
                result.odd_temperature[(std::size_t)n] = lbm.temperature()[n];
        }
        if(step+1u==kOutletComparisonEvenSteps) {
            result.even_temperature.resize((std::size_t)lbm.grid().cells());
            for(ulong n=0ull; n<lbm.grid().cells(); ++n)
                result.even_temperature[(std::size_t)n] = lbm.temperature()[n];
        }
    }
    lbm.flags().download();
    result.outlet_flags_valid = result.outlet_cells_present;
    for(const std::array<ulong, 2u>& nodes : outlet_nodes) {
        if((lbm.flags()[nodes[0]] & (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL)) !=
           (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL)) {
            result.outlet_flags_valid = false;
            break;
        }
    }
    lbm.synchronize();
    return result;
}

void test_multigpu_temperature_outlet(
    TestState& test, const Temperature_Lattice lattice) {
    const std::string prefix = lattice_name(lattice);
    const OutletComparisonField east_reference =
        solve_outlet_comparison_field(
            lattice, Temperature_Outlet_Face::East, 1u, 1u);
    const OutletComparisonField north_reference =
        solve_outlet_comparison_field(
            lattice, Temperature_Outlet_Face::North, 1u, 1u);

    for(const std::pair<const OutletComparisonField*, const char*> reference : {
        std::make_pair(&east_reference, "east"),
        std::make_pair(&north_reference, "north")}) {
        test.expect(
            reference.first->expected_topology,
            prefix+" "+reference.second+
                " outlet reference uses one local CUDA domain");
        test.expect(
            reference.first->outlet_configured &&
            reference.first->outlet_cells_present &&
            reference.first->outlet_flags_valid,
            prefix+" "+reference.second+
                " outlet reference configures every dynamic scalar boundary cell");
        test.expect(
            reference.first->pull_parity_maximum_error[0] <= 3.0E-4 &&
            reference.first->pull_parity_maximum_error[1] <= 3.0E-4,
            prefix+" "+reference.second+
                " outlet reference satisfies zero normal gradient on both pull parities");
    }

    struct OutletDecomposition {
        uint dx;
        uint dy;
        uint decomposition_axis;
        Temperature_Outlet_Face face;
        const char* label;
    };
    for(const OutletDecomposition decomposition : {
        OutletDecomposition{
            2u, 1u, 0u, Temperature_Outlet_Face::East,
            "Dx=2 east outlet normal"},
        OutletDecomposition{
            2u, 1u, 0u, Temperature_Outlet_Face::North,
            "Dx=2 north outlet tangential"},
        OutletDecomposition{
            1u, 2u, 1u, Temperature_Outlet_Face::North,
            "Dy=2 north outlet normal"},
        OutletDecomposition{
            1u, 2u, 1u, Temperature_Outlet_Face::East,
            "Dy=2 east outlet tangential"}}) {
        const OutletComparisonField& reference =
            decomposition.face==Temperature_Outlet_Face::East
                ? east_reference : north_reference;
        const OutletComparisonField two_gpu =
            solve_outlet_comparison_field(
                lattice, decomposition.face,
                decomposition.dx, decomposition.dy);
        const std::string case_name = decomposition.label;
        test.expect(
            two_gpu.expected_topology,
            prefix+" "+case_name+" uses the requested local decomposition");
        test.expect(
            two_gpu.distinct_devices,
            prefix+" "+case_name+" assigns distinct CUDA devices");
        test.expect(
            two_gpu.outlet_configured &&
            two_gpu.outlet_cells_present &&
            two_gpu.outlet_flags_valid,
            prefix+" "+case_name+
                " promotes the entire distributed outlet to CELL_EQUILIBRIUM|CELL_THERMAL");
        test.expect(
            two_gpu.pull_parity_maximum_error[0] <= 3.0E-4 &&
            two_gpu.pull_parity_maximum_error[1] <= 3.0E-4,
            prefix+" "+case_name+
                " reconstructs its adjacent-interior scalar on both pull parities");

        const auto compare_snapshot = [&](
            const std::vector<float>& expected,
            const std::vector<float>& actual,
            const char* const snapshot_name) {
            test.expect(
                expected.size()==actual.size() && !expected.empty(),
                prefix+" "+case_name+" "+snapshot_name+
                    " snapshot has the same non-empty global extent as one GPU");
            if(expected.size()!=actual.size() || expected.empty()) return;

            double squared_error_sum = 0.0;
            double maximum_error = 0.0;
            double interface_maximum_error = 0.0;
            double outlet_maximum_error = 0.0;
            for(uint z=0u; z<kOutletComparisonNz; ++z) {
                for(uint y=0u; y<kOutletComparisonNy; ++y) {
                    for(uint x=0u; x<kOutletComparisonNx; ++x) {
                        const std::size_t n =
                            (std::size_t)x+
                            ((std::size_t)y+
                             (std::size_t)z*
                             (std::size_t)kOutletComparisonNy)*
                            (std::size_t)kOutletComparisonNx;
                        const double error = std::fabs(
                            (double)actual[n]-(double)expected[n]);
                        squared_error_sum += error*error;
                        maximum_error = std::max(maximum_error, error);
                        const uint coordinate =
                            decomposition.decomposition_axis==0u ? x : y;
                        const uint extent =
                            decomposition.decomposition_axis==0u
                                ? kOutletComparisonNx
                                : kOutletComparisonNy;
                        if(coordinate==0u || coordinate==extent-1u ||
                           coordinate==extent/2u-1u ||
                           coordinate==extent/2u) {
                            interface_maximum_error =
                                std::max(interface_maximum_error, error);
                        }
                        if(is_outlet_comparison_cell(
                               decomposition.face, x, y, z)) {
                            outlet_maximum_error =
                                std::max(outlet_maximum_error, error);
                        }
                    }
                }
            }
            const double rms_error = std::sqrt(
                squared_error_sum/(double)expected.size());
            std::cout
                << prefix << " " << case_name << " " << snapshot_name
                << " versus one GPU: max error=" << maximum_error
                << ", interface max error=" << interface_maximum_error
                << ", outlet max error=" << outlet_maximum_error
                << ", RMS error=" << rms_error << '\n';
            test.expect(
                maximum_error<=2.0E-4,
                prefix+" "+case_name+" "+snapshot_name+
                    " full scalar field agrees pointwise with one GPU");
            test.expect(
                interface_maximum_error<=2.0E-4,
                prefix+" "+case_name+" "+snapshot_name+
                    " agrees at periodic and decomposition interfaces");
            test.expect(
                outlet_maximum_error<=2.0E-4,
                prefix+" "+case_name+" "+snapshot_name+
                    " distributed outlet agrees pointwise with one GPU");
            test.expect(
                rms_error<=5.0E-5,
                prefix+" "+case_name+" "+snapshot_name+
                    " has a small global RMS difference");
        };
        compare_snapshot(
            reference.odd_temperature,
            two_gpu.odd_temperature,
            "after 5 steps (odd)");
        compare_snapshot(
            reference.even_temperature,
            two_gpu.even_temperature,
            "after 6 steps (even)");
    }
}

int run_manual_update_fields_forcing_test() {
#ifdef UPDATE_FIELDS
    std::cerr
        << "FAIL: manual-field forcing regression must be compiled with "
           "CULBM_DISABLE_UPDATE_FIELDS.\n";
    return 2;
#else
    TestState test;
    constexpr uint N = 10u;
    constexpr float local_x = 0.010f;
    constexpr float local_y = 0.020f;
    constexpr float local_z = -0.010f;
    constexpr float reference_x = 0.050f;
    constexpr float reference_y = -0.020f;
    constexpr float reference_z = 0.030f;
    constexpr double tolerance = 8.0E-4;
    test.expect(
        set_temperature_lattice(Temperature_Lattice::Q19),
        "manual-field regression selects the production D3Q19 scalar lattice");

    buffer_n_cells = 2;
    buffer_inv_tau_lbmu = 0.20f;
    buffer_nudge_vertical = 1;
    buffer_downstream_face_id = 2;
    top_sponge_active = false;
    constexpr float buffer_fixed_x = 0.017f;
    constexpr float buffer_fixed_y = -0.011f;
    constexpr float buffer_fixed_z = 0.006f;
    const auto prepare_buffer_case = [&](culbm::solver::Solver& lbm) {
        initialize_quiescent_fields(lbm, 1.0f);
        for(uint z=0u; z<N; ++z) {
            for(uint y=0u; y<N; ++y) {
                for(uint x=0u; x<N; ++x) {
                    const ulong n = lbm.grid().index(x, y, z);
                    lbm.velocity().x[n] = x==0u ? reference_x : local_x;
                    lbm.velocity().y[n] = x==0u ? reference_y : local_y;
                    lbm.velocity().z[n] = x==0u ? reference_z : local_z;
                }
            }
        }
        const ulong fixed = lbm.grid().index(1u, 6u, 5u);
        lbm.flags()[fixed] = CELL_EQUILIBRIUM;
        lbm.velocity().x[fixed] = buffer_fixed_x;
        lbm.velocity().y[fixed] = buffer_fixed_y;
        lbm.velocity().z[fixed] = buffer_fixed_z;
    };
    std::array<double, 3u> buffer_raw_velocity{};
    buffer_nudging_active = false;
    {
        culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
            culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
        prepare_buffer_case(lbm);
        const ulong sample = lbm.grid().index(1u, 5u, 5u);
        lbm.initialize_once();
        lbm.refresh_macros();
        lbm.velocity().download();
        buffer_raw_velocity = {
            (double)lbm.velocity().x[sample],
            (double)lbm.velocity().y[sample],
            (double)lbm.velocity().z[sample]};
        lbm.synchronize();
    }
    buffer_nudging_active = true;
    {
        culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
            culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, 0.0f));
        prepare_buffer_case(lbm);
        const ulong sample = lbm.grid().index(1u, 5u, 5u);
        const ulong fixed = lbm.grid().index(1u, 6u, 5u);

        lbm.initialize_once();
        lbm.refresh_macros();
        lbm.velocity().download();
        const double buffer_weight = 0.5;
        const double half_force_weight =
            0.5*buffer_weight*(double)buffer_inv_tau_lbmu;
        test.expect_close(
            lbm.velocity().x[sample],
            buffer_raw_velocity[0]+half_force_weight*
                (reference_x-buffer_raw_velocity[0]), tolerance,
            "manual update_fields includes west-buffer x momentum forcing");
        test.expect_close(
            lbm.velocity().y[sample],
            buffer_raw_velocity[1]+half_force_weight*
                (reference_y-buffer_raw_velocity[1]), tolerance,
            "manual update_fields includes west-buffer y momentum forcing");
        test.expect_close(
            lbm.velocity().z[sample],
            buffer_raw_velocity[2]+half_force_weight*
                (reference_z-buffer_raw_velocity[2]), tolerance,
            "manual update_fields includes optional vertical buffer forcing");
        test.expect_close(
            lbm.velocity().x[fixed], buffer_fixed_x, tolerance,
            "manual update_fields preserves CELL_EQUILIBRIUM prescribed x velocity in a buffer");
        test.expect_close(
            lbm.velocity().y[fixed], buffer_fixed_y, tolerance,
            "manual update_fields preserves CELL_EQUILIBRIUM prescribed y velocity in a buffer");
        test.expect_close(
            lbm.velocity().z[fixed], buffer_fixed_z, tolerance,
            "manual update_fields preserves CELL_EQUILIBRIUM prescribed z velocity in a buffer");
        lbm.synchronize();
    }

    buffer_nudging_active = false;
    sponge_n_cells = 3;
    sponge_inv_tau_lbmu = 0.20f;
    sponge_ref_mode = 0;
    constexpr float beta = 0.50f;
    constexpr float gravity_z = -0.50f;
    constexpr float local_temperature = 1.00f;
    constexpr float reference_temperature = 1.20f;
    constexpr float top_fixed_temperature = 1.10f;
    constexpr float top_fixed_x = -0.014f;
    constexpr float top_fixed_y = 0.009f;
    constexpr float top_fixed_z = 0.012f;
    const auto prepare_top_case = [&](culbm::solver::Solver& lbm) {
        initialize_quiescent_fields(lbm, local_temperature);
        for(uint z=0u; z<N; ++z) {
            for(uint y=0u; y<N; ++y) {
                for(uint x=0u; x<N; ++x) {
                    const ulong n = lbm.grid().index(x, y, z);
                    const bool reference_plane = z==N-1u;
                    lbm.velocity().x[n] = reference_plane ? reference_x : local_x;
                    lbm.velocity().y[n] = reference_plane ? reference_y : local_y;
                    lbm.velocity().z[n] = reference_plane ? reference_z : local_z;
                    lbm.temperature()[n] = reference_plane
                        ? reference_temperature : local_temperature;
                    if(reference_plane)
                        lbm.flags()[n] = (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL);
                }
            }
        }
        const ulong fixed = lbm.grid().index(6u, 5u, N-2u);
        lbm.flags()[fixed] = (uchar)(CELL_EQUILIBRIUM|CELL_THERMAL);
        lbm.velocity().x[fixed] = top_fixed_x;
        lbm.velocity().y[fixed] = top_fixed_y;
        lbm.velocity().z[fixed] = top_fixed_z;
        lbm.temperature()[fixed] = top_fixed_temperature;
    };
    std::array<double, 3u> top_raw_velocity{};
    double top_raw_temperature = 0.0;
    top_sponge_active = false;
    {
        culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
            culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, beta));
        prepare_top_case(lbm);
        const ulong sample = lbm.grid().index(5u, 5u, N-2u);
        lbm.set_gravity(0.0f, 0.0f, gravity_z);
        lbm.initialize_once();
        lbm.refresh_macros();
        lbm.velocity().download();
        lbm.temperature().download();
        top_raw_velocity = {
            (double)lbm.velocity().x[sample],
            (double)lbm.velocity().y[sample],
            (double)lbm.velocity().z[sample]};
        top_raw_temperature = (double)lbm.temperature()[sample];
        top_raw_velocity[2] -=
            0.5*(-(double)gravity_z*(double)beta*
                 (top_raw_temperature-1.0));
        lbm.synchronize();
    }
    top_sponge_active = true;
    {
        culbm::solver::Solver lbm(N, N, N, 1u, 1u, 1u,
            culbm::solver::CaseCoefficients().viscosity(kNu).thermal(kAlpha, beta));
        prepare_top_case(lbm);
        const ulong sample = lbm.grid().index(5u, 5u, N-2u);
        const ulong fixed = lbm.grid().index(6u, 5u, N-2u);
        lbm.set_gravity(0.0f, 0.0f, gravity_z);

        lbm.initialize_once();
        lbm.refresh_macros();
        lbm.velocity().download();
        lbm.temperature().download();
        const double sigma = (double)sponge_inv_tau_lbmu;
        const double expected_temperature =
            top_raw_temperature+
            sigma*(reference_temperature-top_raw_temperature);
        const double half_sponge = 0.5*sigma;
        const double expected_buoyancy_force =
            -(double)gravity_z*(double)beta*
            (expected_temperature-1.0);
        test.expect_close(
            lbm.temperature()[sample], expected_temperature, tolerance,
            "manual update_fields applies top scalar relaxation before buoyancy");
        test.expect_close(
            lbm.velocity().x[sample],
            top_raw_velocity[0]+half_sponge*
                (reference_x-top_raw_velocity[0]), tolerance,
            "manual update_fields includes top-sponge x momentum forcing");
        test.expect_close(
            lbm.velocity().y[sample],
            top_raw_velocity[1]+half_sponge*
                (reference_y-top_raw_velocity[1]), tolerance,
            "manual update_fields includes top-sponge y momentum forcing");
        test.expect_close(
            lbm.velocity().z[sample],
            top_raw_velocity[2]+half_sponge*
                (reference_z-top_raw_velocity[2])+
                0.5*expected_buoyancy_force,
            tolerance,
            "manual update_fields uses top-relaxed temperature in Guo buoyancy velocity");
        test.expect_close(
            lbm.velocity().x[fixed], top_fixed_x, tolerance,
            "top sponge does not half-force a CELL_EQUILIBRIUM x velocity");
        test.expect_close(
            lbm.velocity().y[fixed], top_fixed_y, tolerance,
            "top sponge does not half-force a CELL_EQUILIBRIUM y velocity");
        test.expect_close(
            lbm.velocity().z[fixed], top_fixed_z, tolerance,
            "top sponge and buoyancy do not half-force a CELL_EQUILIBRIUM z velocity");
        test.expect_close(
            lbm.temperature()[fixed], top_fixed_temperature, tolerance,
            "top scalar sponge preserves CELL_EQUILIBRIUM|CELL_THERMAL temperature");
        lbm.synchronize();
    }

    buffer_nudging_active = false;
    top_sponge_active = false;
    if(test.failures==0) {
        std::cout
            << "Manual update_fields Guo forcing regression passed for buffer "
               "nudging, top sponge, buoyancy ordering, and CELL_EQUILIBRIUM.\n";
    }
    return test.failures==0 ? 0 : 1;
#endif
}

int run_tests() {
    TestState test;
    test_relaxation_diagnostics(test);
    test_temperature_memory_accounting(test);
    test_boussinesq_unit_conversion(test);
    test_peak_memory_accounting(test);
    for(const Temperature_Lattice lattice :
        {Temperature_Lattice::Q7, Temperature_Lattice::Q19}) {
        test_lattice_contract(test, lattice);
        test_periodic_fourier_diffusion(test, lattice);
        test_periodic_fourier_advection(test, lattice);
        test_fixed_temperature_solid_walls(test, lattice);
        test_adiabatic_solid_walls(test, lattice);
        test_independent_buoyancy(test, lattice);
        test_buoyant_scalar_uses_physical_velocity(test, lattice);
        test_force_aware_equilibrium_velocity_boundary(test, lattice);
        test_force_aware_equilibrium_velocity_boundary_initialization(
            test, lattice);
        test_dynamic_free_slip_boundaries(test, lattice);
        test_free_slip_xz_corners(test, lattice);
        test_nested_temperature_bridge(test, lattice);
        test_temperature_zero_gradient_outlet(test, lattice);
    }
    if(test.failures==0) {
        std::cout
            << "Production GPU temperature analytic tests passed for "
               "D3Q7 and D3Q19.\n";
    }
    return test.failures==0 ? 0 : 1;
}

int run_multigpu_halo_test() {
    TestState test;
    test_multigpu_scalar_halo(test, Temperature_Lattice::Q7);
    test_multigpu_scalar_halo(test, Temperature_Lattice::Q19);
    test_multigpu_temperature_outlet(test, Temperature_Lattice::Q7);
    test_multigpu_temperature_outlet(test, Temperature_Lattice::Q19);
    test_multigpu_nested_temperature_bridge(test);
    if(test.failures==0) {
        std::cout
            << "Production D3Q7/D3Q19 two-GPU scalar halo, outlet, and "
               "nested bridge tests passed.\n";
    }
    return test.failures==0 ? 0 : 1;
}

int run_free_slip_test(const uint steps) {
    TestState test;
    for(const Temperature_Lattice lattice :
        {Temperature_Lattice::Q7, Temperature_Lattice::Q19}) {
        test_dynamic_free_slip_boundaries(test, lattice, steps);
        test_free_slip_xz_corners(test, lattice);
    }
    return test.failures==0 ? 0 : 1;
}

} // namespace

int main(const int argc, const char* const argv[]) {
    const CUresult initialization = cuInit(0u);
    if(initialization!=CUDA_SUCCESS) {
        std::cout << "SKIP: CUDA driver is unavailable.\n";
        return 77;
    }
    int device_count = 0;
    if(cuDeviceGetCount(&device_count)!=CUDA_SUCCESS || device_count<=0) {
        std::cout << "SKIP: no CUDA device is available.\n";
        return 77;
    }
    if(argc==2 && std::string(argv[1])=="--multi-gpu-halo") {
        if(device_count<2) {
            std::cout
                << "SKIP: the scalar halo regression requires "
                   "two CUDA devices.\n";
            return 77;
        }
        return run_multigpu_halo_test();
    }
    if((argc==2 || argc==3) &&
       std::string(argv[1])=="--free-slip-only") {
        const uint steps =
            argc==3 ? (uint)std::stoul(argv[2]) : 1u;
        return run_free_slip_test(steps);
    }
    if(argc==2 &&
       std::string(argv[1])=="--manual-update-fields-forcing") {
        return run_manual_update_fields_forcing_test();
    }
    if(argc!=1) {
        std::cerr
            << "Usage: " << argv[0]
            << " [--multi-gpu-halo|--free-slip-only [steps]|"
               "--manual-update-fields-forcing]\n";
        return 2;
    }
    return run_tests();
}
