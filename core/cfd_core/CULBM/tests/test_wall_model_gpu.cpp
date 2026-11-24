// GPU checks and research drivers for the CULBM bounce-back wall model.
//
// Default (CTest) mode runs short deterministic checks:
//   * laminar body-force channel with rough wall-modelled walls: the steady
//     first-node velocity must satisfy the explicit log-law stress balance;
//   * mass conservation and wall-normal impermeability of the correction;
//   * wall model disabled keeps the plain no-slip bounce-back solution;
//   * optional one/two-GPU agreement (--multi-gpu).
// Manual modes (not registered with CTest):
//   --channel   turbulent body-force channel (WMLES), writes mean profiles;
//   --benchmark ground + cube-array throughput, wall model off/on.
#include "solver.hpp"
#include "surface_ibm.hpp"

#include <algorithm>
#include <cstdlib>
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
};

// Deterministic, platform-independent pseudo random numbers in [-1,1).
struct Lcg {
    std::uint64_t state;
    explicit Lcg(const std::uint64_t seed) : state(seed*2862933555777941757ull+3037000493ull) {}
    double next() {
        state = state*6364136223846793005ull+1442695040888963407ull;
        return (double)((state>>11)&((1ull<<53)-1ull))/(double)(1ull<<52)-1.0;
    }
};

void configure(const bool wall_model, const float z0, const float kappa,
               const bool sgs, const float cs, const float molecular_nu,
               const uint sgs_layers = 3u) {
    if(!set_temperature_lattice(Temperature_Lattice::Q7)) {
        throw std::runtime_error("cannot select D3Q7 scalar lattice");
    }
    Subgrid_Configuration subgrid;
    subgrid.enabled = sgs;
    subgrid.smagorinsky_constant = cs;
    subgrid.scalar_coupling = false;
    subgrid.turbulent_prandtl = 1.0f;
    subgrid.fixed_eddy_viscosity = -1.0f;
    if(!set_subgrid_configuration(subgrid)) {
        throw std::runtime_error("cannot set SGS configuration");
    }
    Wall_Model_Configuration configuration;
    configuration.enabled = wall_model;
    configuration.kappa = kappa;
    configuration.z0_ground_m = z0;
    configuration.z0_building_m = z0;
    configuration.molecular_nu_m2_s = molecular_nu;
    configuration.sgs_layers = sgs_layers;
    if(!set_wall_model_configuration(configuration)) {
        throw std::runtime_error("cannot set wall model configuration");
    }
}

// Channel of 2h fluid rows between solid rows z=0 and z=2h+1; periodic in x,y.
// The halfway walls lie at z=1/2 and z=2h+1/2, so cell z has wall distance
// z-1/2 (lower half) in lattice units.
struct ChannelField {
    uint nx = 0u, ny = 0u, nz = 0u;
    std::vector<double> mean_u; // per z
    double mass = 0.0;
    double max_abs_uz_wall = 0.0;
    bool finite = true;
};

// Open channel (--open): bottom wall, free-slip lid (CELL_SOLID|CELL_AUX_X ghost row,
// requires a FREE_SLIP_BOUNDARIES build) at height h.
bool g_open_channel = false;

double wall_distance(const uint z, const uint h) {
    if(g_open_channel) return (double)z-0.5;
    return z<=h ? (double)z-0.5 : (double)(2u*h+1u-z)-0.5;
}

void fill_channel(culbm::solver::Solver& lbm, const uint h, const double utau, const double z0,
                  const double kappa, const double noise, const std::uint64_t seed) {
    const uint nx = lbm.grid().size(0u), ny = lbm.grid().size(1u), nz = lbm.grid().size(2u);
    Lcg random(seed);
    for(uint z=0u; z<nz; ++z) {
        const bool solid = z==0u || z==nz-1u;
        const double zw = solid ? 0.0 : wall_distance(z, h);
        const double mean = solid ? 0.0 : utau/kappa*std::log1p(zw/z0);
        for(uint y=0u; y<ny; ++y) {
            for(uint x=0u; x<nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                lbm.density()[n] = 1.0f;
                lbm.flags()[n] = solid ? (uchar)((g_open_channel && z==nz-1u) ? (CELL_SOLID|CELL_AUX_X) : CELL_SOLID) : (uchar)0u;
                double ux = mean, uy = 0.0, uz = 0.0;
                if(!solid && noise>0.0) {
                    // Large-scale streamwise rolls plus small random noise.
                    const double px = 2.0*kPi*(double)x/(double)nx;
                    const double py = 2.0*kPi*(double)y/(double)ny;
                    const double pz = kPi*zw/(double)h;
                    ux += noise*mean*(0.5*std::sin(2.0*py)*std::sin(pz)+0.25*random.next());
                    uy += noise*mean*(0.25*std::sin(px)*std::cos(pz)+0.25*random.next());
                    uz += noise*mean*(0.25*std::sin(2.0*py)*std::sin(px)*std::sin(pz)+0.25*random.next());
                }
                lbm.velocity().x[n] = solid ? 0.0f : (float)ux;
                lbm.velocity().y[n] = solid ? 0.0f : (float)uy;
                lbm.velocity().z[n] = solid ? 0.0f : (float)uz;
                lbm.temperature()[n] = 1.0f;
            }
        }
    }
}

ChannelField read_channel(culbm::solver::Solver& lbm) {
    ChannelField field;
    field.nx = lbm.grid().size(0u);
    field.ny = lbm.grid().size(1u);
    field.nz = lbm.grid().size(2u);
    lbm.density().download();
    lbm.velocity().download();
    field.mean_u.assign(field.nz, 0.0);
    for(uint z=1u; z+1u<field.nz; ++z) {
        double sum = 0.0;
        for(uint y=0u; y<field.ny; ++y) {
            for(uint x=0u; x<field.nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                const double ux = lbm.velocity().x[n], uz = lbm.velocity().z[n], rho = lbm.density()[n];
                field.finite = field.finite && std::isfinite(ux) && std::isfinite(uz) && std::isfinite(rho);
                sum += ux;
                field.mass += rho;
                if(z==1u || z+2u==field.nz)
                    field.max_abs_uz_wall = std::max(field.max_abs_uz_wall, std::fabs(uz));
            }
        }
        field.mean_u[z] = sum/(double)(field.nx*field.ny);
    }
    return field;
}

// Laminar body-force channel with steady first-node log-law balance.  With
// constant drag Cd the wall-modelled stress is rho*Cd*u1^2 per face, which in
// steady state balances the body force g*h per unit wall area.
void test_laminar_rough_balance(TestState& test) {
    constexpr uint h = 8u;
    constexpr float nu = 0.02f;
    constexpr float kappa = 0.41f;
    constexpr float z0 = 0.05f; // lattice units (default Units: 1 cell = 1 m)
    const double cd = std::pow((double)kappa/std::log1p(0.5/(double)z0), 2.0);
    const double u1_target = 0.02;
    const double g = cd*u1_target*u1_target/(double)h;
    configure(true, z0, kappa, false, 0.0f, nu);
    culbm::solver::Solver lbm(4u, 4u, 2u*h+2u, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(nu).volume_force((float)g, 0.0f, 0.0f).thermal(nu, 0.0f));
    fill_channel(lbm, h, 0.0, z0, kappa, 0.0, 1u);
    lbm.initialize_once();
    const double mass0 = (double)(4u*4u*2u*h);
    uint laminar_steps = 120000u;
    if(const char* env = std::getenv("CULBM_WM_TEST_LAMINAR_STEPS")) laminar_steps = (uint)std::strtoul(env, nullptr, 10);
    for(uint step=0u; step<laminar_steps; ++step) lbm.advance();
    lbm.synchronize();
    // Esoteric-Pull reflects wall populations after two steps, so odd and
    // even steps settle to slightly different wall states; average a pair.
    const ChannelField field_a = read_channel(lbm);
    lbm.advance();
    lbm.synchronize();
    ChannelField field = read_channel(lbm);
    for(std::size_t z=0u; z<field.mean_u.size(); ++z)
        field.mean_u[z] = 0.5*(field.mean_u[z]+field_a.mean_u[z]);
    field.finite = field.finite && field_a.finite;
    std::cout << "laminar rough wall model: u1 odd/even=" << std::setprecision(9)
              << field_a.mean_u[1] << '/' << read_channel(lbm).mean_u[1] << '\n';
    test.expect(field.finite, "laminar wall-model channel stays finite");
    const double u1_bottom = field.mean_u[1];
    const double u1_top = field.mean_u[2u*h];
    // The wall function sees the free-slip node velocity; the stored Guo
    // velocity additionally contains -T (stress removed this step) and +g/2.
    const double rho1 = lbm.density()[lbm.grid().index(0u, 0u, 1u)];
    const double rho_mid = lbm.density()[lbm.grid().index(0u, 0u, h)];
    std::cout << "laminar rough wall model: rho(z=1)=" << std::setprecision(9) << rho1
              << " rho(centre)=" << rho_mid << '\n';
    const double u_free_slip = u1_bottom+cd*u1_target*u1_target-0.5*g;
    const double error = std::fabs(u_free_slip-u1_target)/u1_target;
    std::cout << "laminar rough wall model: u1=" << std::setprecision(8) << u1_bottom
              << " target=" << u1_target << " rel.err=" << error
              << " top/bottom=" << u1_top/u1_bottom << '\n';
    test.expect(error<3.0E-3, "steady first-node velocity follows the log-law stress balance");
    test.expect(std::fabs(u1_top/u1_bottom-1.0)<1.0E-4, "upper and lower walls are symmetric (FP32 rounding)");
    test.expect(std::fabs(field.mass/mass0-1.0)<2.0E-6, "wall model conserves mass");
    test.expect(field.max_abs_uz_wall<1.0E-7, "wall model keeps the wall impermeable");
    // Interior Poiseuille curvature: u(zw)=u1+g/(2nu)*((zw-1/2)*(2h-1/2-zw))...
    // check the centreline excess against the analytic parabola over the slip.
    const double zc = (double)h-0.5;
    const double analytic_excess = g/(2.0*(double)nu)*((zc*(2.0*h-zc))-(0.5*(2.0*h-0.5)));
    const double measured_excess = field.mean_u[h]-u1_bottom;
    std::cout << "laminar rough wall model: centre excess=" << measured_excess
              << " analytic=" << analytic_excess << '\n';
    test.expect(std::fabs(measured_excess/analytic_excess-1.0)<2.0E-2,
                "bulk viscous profile above the modelled wall is Poiseuille");
}

// Regression for concave edges: a one-cell well cut into a block must stay
// nearly stagnant.  Superposing the tangential exchanges of two faces on a
// shared diagonal link once pumped jets faster than the free stream there.
void test_concave_pocket(TestState& test) {
    constexpr uint n = 24u, nz = 14u;
    configure(true, 0.03f, 0.41f, true, 0.173265956f, 1.0E-4f);
    culbm::solver::Solver lbm(n, n, nz, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(1.0E-4f).volume_force(2.0E-6f, 0.0f, 0.0f).thermal(1.0E-4f, 0.0f));
    const auto block = [&](const uint x, const uint y, const uint z) {
        const bool in_block = x>=6u && x<18u && y>=6u && y<18u && z>=1u && z<=5u;
        const bool well = x==11u && y==11u;          // one-cell vertical well
        const bool slot = y==14u && x>=8u && x<16u && z>=3u; // one-cell slot from the roof
        return in_block && !well && !slot;
    };
    for(uint z=0u; z<nz; ++z) for(uint y=0u; y<n; ++y) for(uint x=0u; x<n; ++x) {
        const ulong i = lbm.grid().index(x, y, z);
        const bool solid = z==0u || z==nz-1u || block(x, y, z);
        lbm.flags()[i] = solid ? (uchar)CELL_SOLID : (uchar)0u;
        lbm.density()[i] = 1.0f;
        const bool inside_footprint = x>=6u && x<18u && y>=6u && y<18u && z<=5u;
        lbm.velocity().x[i] = (solid || inside_footprint) ? 0.0f : 0.05f;
        lbm.velocity().y[i] = 0.0f;
        lbm.velocity().z[i] = 0.0f;
        lbm.temperature()[i] = 1.0f;
    }
    lbm.initialize_once();
    double well_max = 0.0;
    double top_mean = 0.0;
    for(uint s=0u; s<6000u; ++s) {
        lbm.advance();
        if(s<3000u || s%500u!=499u) continue;
        lbm.synchronize();
        lbm.velocity().download();
        double top = 0.0;
        for(uint y=0u; y<n; ++y) for(uint x=0u; x<n; ++x) top += lbm.velocity().x[lbm.grid().index(x, y, 9u)];
        top_mean = top/(double)(n*n);
        for(uint z=1u; z<=4u; ++z) {
            const ulong i = lbm.grid().index(11u, 11u, z);
            well_max = std::max(well_max, std::sqrt((double)lbm.velocity().x[i]*lbm.velocity().x[i]+(double)lbm.velocity().y[i]*lbm.velocity().y[i]+
                                                    (double)lbm.velocity().z[i]*lbm.velocity().z[i]));
        }
    }
    std::cout << "concave pocket: max |u| in one-cell well=" << well_max << " free-stream mean u=" << top_mean << '\n';
    test.expect(std::isfinite(well_max) && std::isfinite(top_mean), "concave pocket run stays finite");
    test.expect(well_max < 0.3*top_mean, "one-cell well stays nearly stagnant with the wall model");
}

// Diagnostic: momentum input of the uniform body force in a periodic box.
void probe_force_input() {
    const double g = 1.0E-6;
    configure(false, 0.05f, 0.41f, false, 0.0f, 0.02f);
    culbm::solver::Solver lbm(4u, 4u, 4u, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(0.02f).volume_force((float)g, 0.0f, 0.0f).thermal(0.02f, 0.0f));
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        lbm.density()[n] = 1.0f; lbm.velocity().x[n] = 0.0f; lbm.velocity().y[n] = 0.0f; lbm.velocity().z[n] = 0.0f;
        lbm.flags()[n] = 0u; lbm.temperature()[n] = 1.0f;
    }
    lbm.initialize_once();
    for(uint s=0u; s<1000u; ++s) lbm.advance();
    lbm.synchronize();
    lbm.velocity().download();
    const double u0 = lbm.velocity().x[0];
    for(uint s=0u; s<1000u; ++s) lbm.advance();
    lbm.synchronize();
    lbm.velocity().download();
    const double u1 = lbm.velocity().x[0];
    std::cout << "force probe: du/dt/g=" << std::setprecision(9) << (u1-u0)/(1000.0*g)
              << " u(1000)/(g*1000)=" << u0/(1000.0*g) << '\n';
}

// Smooth wall (z0=0) at low wall Reynolds number uses the viscous sublayer
// branch tau/rho = nu*u/d (d=1/2), so the steady free-slip node velocity is
// u_s = g*h/(2*nu).
void test_smooth_viscous_sublayer(TestState& test) {
    constexpr uint h = 8u;
    constexpr float nu = 0.02f;
    const double u_target = 0.01; // Re_d = 0.5*u/nu = 0.25, far below the log-law intersection
    const double g = 2.0*(double)nu*u_target/(double)h;
    configure(true, 0.0f, 0.41f, false, 0.0f, nu);
    culbm::solver::Solver lbm(4u, 4u, 2u*h+2u, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(nu).volume_force((float)g, 0.0f, 0.0f).thermal(nu, 0.0f));
    fill_channel(lbm, h, 0.0, 0.05, 0.41, 0.0, 1u);
    lbm.initialize_once();
    for(uint step=0u; step<40000u; ++step) lbm.advance();
    lbm.synchronize();
    const ChannelField field = read_channel(lbm);
    const double T = 2.0*(double)nu*u_target;
    const double u_free_slip = field.mean_u[1]+T-0.5*g;
    const double error = std::fabs(u_free_slip/u_target-1.0);
    std::cout << "smooth viscous sublayer: u_s=" << u_free_slip << " target=" << u_target
              << " rel.err=" << error << '\n';
    test.expect(field.finite && error<3.0E-3, "smooth-wall viscous sublayer balances the body force");
}

// Disabled wall model must keep the original no-slip solution.
void test_disabled_is_no_slip(TestState& test) {
    constexpr uint h = 8u;
    constexpr float nu = 0.05f;
    const double g = 1.0E-6;
    configure(false, 0.05f, 0.41f, false, 0.0f, nu);
    culbm::solver::Solver lbm(4u, 4u, 2u*h+2u, 1u, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(nu).volume_force((float)g, 0.0f, 0.0f).thermal(nu, 0.0f));
    fill_channel(lbm, h, 0.0, 0.05, 0.41, 0.0, 1u);
    lbm.initialize_once();
    for(uint step=0u; step<20000u; ++step) lbm.advance();
    lbm.synchronize();
    const ChannelField field = read_channel(lbm);
    // Halfway bounce-back Poiseuille: u(zw)=g/(2nu)*zw*(2h-zw), SRT slip O(1e-3).
    const double zc = (double)h-0.5;
    const double analytic = g/(2.0*(double)nu)*zc*(2.0*h-zc);
    const double error = std::fabs(field.mean_u[h]/analytic-1.0);
    std::cout << "disabled wall model: centre u=" << field.mean_u[h] << " analytic=" << analytic
              << " rel.err=" << error << '\n';
    test.expect(error<1.0E-2, "disabled wall model recovers no-slip Poiseuille flow");
}

// ----------------------------------------------------------------------------
// Turbulent channel (manual research mode)

struct ChannelOptions {
    uint h = 16u;
    uint nx = 0u, ny = 0u;
    uint domains_x = 1u;
    ulong steps = 120000ull;
    ulong sample_start = 60000ull;
    uint sample_every = 25u;
    bool wall_model = true;
    float z0 = 0.02f; // lattice units of the h=16 reference; scaled with h
    float z0_over_h = -1.0f;
    float utau = 0.004f;
    float cs = 0.173265956f;
    float nu = 1.0E-5f;
    float kappa = 0.41f;
    uint sgs_layers = 3u;
    std::string output;
};

ChannelOptions parse_channel(const int argc, const char* const argv[]) {
    ChannelOptions options;
    for(int i=2; i<argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if(i+1>=argc) throw std::invalid_argument("missing value for "+key);
            return argv[++i];
        };
        if(key=="--h") options.h = (uint)std::stoul(value());
        else if(key=="--nx") options.nx = (uint)std::stoul(value());
        else if(key=="--ny") options.ny = (uint)std::stoul(value());
        else if(key=="--dx") options.domains_x = (uint)std::stoul(value());
        else if(key=="--steps") options.steps = std::stoull(value());
        else if(key=="--sample-start") options.sample_start = std::stoull(value());
        else if(key=="--sample-every") options.sample_every = (uint)std::stoul(value());
        else if(key=="--wm") options.wall_model = std::stoi(value())!=0;
        else if(key=="--z0-over-h") options.z0_over_h = std::stof(value());
        else if(key=="--utau") options.utau = std::stof(value());
        else if(key=="--cs") options.cs = std::stof(value());
        else if(key=="--nu") options.nu = std::stof(value());
        else if(key=="--out") options.output = value();
        else if(key=="--sgs-layers") options.sgs_layers = (uint)std::stoul(value());
        else if(key=="--open") g_open_channel = std::stoi(value())!=0;
        else throw std::invalid_argument("unknown option "+key);
    }
    if(options.nx==0u) options.nx = 8u*options.h;
    if(options.ny==0u) options.ny = 4u*options.h;
    if(options.z0_over_h<=0.0f) options.z0_over_h = 0.02f/16.0f;
    options.z0 = options.z0_over_h*(float)options.h;
    return options;
}

int run_channel(const ChannelOptions& o) {
    configure(o.wall_model, o.z0, o.kappa, true, o.cs, o.nu, o.sgs_layers);
    const uint nz = g_open_channel ? o.h+2u : 2u*o.h+2u;
    const uint fluid_rows = nz-2u;
    const double g = (double)o.utau*(double)o.utau/(double)o.h;
    culbm::solver::Solver lbm(o.nx, o.ny, nz, o.domains_x, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(o.nu).volume_force((float)g, 0.0f, 0.0f).thermal(o.nu, 0.0f));
    fill_channel(lbm, o.h, o.utau, o.z0, o.kappa, 0.3, 12345u);
    lbm.initialize_once();
    std::vector<double> sum_u(nz, 0.0), sum_uu(nz, 0.0), sum_vv(nz, 0.0), sum_ww(nz, 0.0), sum_uw(nz, 0.0);
    ulong samples = 0ull;
    std::vector<std::pair<ulong, double>> bulk_history;
    const auto t0 = std::chrono::steady_clock::now();
    for(ulong step=1ull; step<=o.steps; ++step) {
        lbm.advance();
        const bool sample = step>o.sample_start && step%o.sample_every==0ull;
        const bool monitor = step%2000ull==0ull;
        static const ulong trace_from = std::getenv("CULBM_WM_TRACE_FROM") ?
            std::strtoull(std::getenv("CULBM_WM_TRACE_FROM"), nullptr, 10) : ~0ull;
        static const ulong trace_every = std::getenv("CULBM_WM_TRACE_EVERY") ?
            std::strtoull(std::getenv("CULBM_WM_TRACE_EVERY"), nullptr, 10) : 50ull;
        if(step>=trace_from && step%trace_every==0ull) {
            // Diagnostic trace: extreme |u|, |u_z| and rho over all fluid rows.
            lbm.density().download();
            lbm.velocity().download();
            double best = 0.0, best_uz = 0.0, rmin = 1e30, rmax = -1e30;
            uint bx = 0u, by = 0u, bz = 0u, zx = 0u, zy = 0u, zz = 0u;
            for(uint z=1u; z+1u<nz; ++z) for(uint y=0u; y<o.ny; ++y) for(uint x=0u; x<o.nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                const double mag = std::sqrt((double)lbm.velocity().x[n]*lbm.velocity().x[n]+(double)lbm.velocity().y[n]*lbm.velocity().y[n]+
                                             (double)lbm.velocity().z[n]*lbm.velocity().z[n]);
                if(mag>best) { best = mag; bx = x; by = y; bz = z; }
                if(std::fabs(lbm.velocity().z[n])>best_uz) { best_uz = std::fabs(lbm.velocity().z[n]); zx = x; zy = y; zz = z; }
                rmin = std::min(rmin, (double)lbm.density()[n]);
                rmax = std::max(rmax, (double)lbm.density()[n]);
            }
            const ulong nb = lbm.grid().index(bx, by, bz);
            std::cout << "trace " << step << " max|u|=" << best << " @(" << bx << ',' << by << ',' << bz
                      << ") u=(" << lbm.velocity().x[nb] << ',' << lbm.velocity().y[nb] << ',' << lbm.velocity().z[nb] << ") rho=" << lbm.density()[nb]
                      << " max|uz|=" << best_uz << " @(" << zx << ',' << zy << ',' << zz << ") rho[min,max]=["
                      << rmin << ',' << rmax << "]\n";
            if(best>0.5) return 1;
        }
        if(!sample && !monitor) continue;
        lbm.velocity().download();
        double bulk = 0.0;
        bool finite = true;
        double umax = 0.0;
        uint umax_z = 0u;
        std::vector<double> plane_u(nz, 0.0);
        for(uint z=1u; z+1u<nz; ++z) {
            double su = 0.0;
            for(uint y=0u; y<o.ny; ++y) for(uint x=0u; x<o.nx; ++x) {
                const ulong n = lbm.grid().index(x, y, z);
                su += lbm.velocity().x[n];
                finite = finite && std::isfinite(lbm.velocity().x[n]);
                const double mag = std::sqrt((double)lbm.velocity().x[n]*lbm.velocity().x[n]+(double)lbm.velocity().y[n]*lbm.velocity().y[n]+
                                             (double)lbm.velocity().z[n]*lbm.velocity().z[n]);
                if(mag>umax) { umax = mag; umax_z = z; }
            }
            plane_u[z] = su/(double)(o.nx*o.ny);
            bulk += plane_u[z];
        }
        bulk /= (double)fluid_rows;
        if(!finite) {
            std::cerr << "non-finite velocity at step " << step << '\n';
            return 1;
        }
        if(monitor) {
            bulk_history.emplace_back(step, bulk);
            std::cout << "step " << step << " Ub/u*=" << bulk/(double)o.utau
                      << " U1/u*=" << plane_u[1]/(double)o.utau
                      << " max|u|=" << umax << " at z=" << umax_z << '\n';
        }
        if(sample) {
            for(uint z=1u; z+1u<nz; ++z) {
                const double um = plane_u[z];
                double suu = 0.0, svv = 0.0, sww = 0.0, suw = 0.0;
                for(uint y=0u; y<o.ny; ++y) for(uint x=0u; x<o.nx; ++x) {
                    const ulong n = lbm.grid().index(x, y, z);
                    const double up = lbm.velocity().x[n]-um, vp = lbm.velocity().y[n], wp = lbm.velocity().z[n];
                    suu += up*up; svv += vp*vp; sww += wp*wp; suw += up*wp;
                }
                const double cells = (double)(o.nx*o.ny);
                sum_u[z] += um; sum_uu[z] += suu/cells; sum_vv[z] += svv/cells;
                sum_ww[z] += sww/cells; sum_uw[z] += suw/cells;
            }
            ++samples;
        }
    }
    lbm.synchronize();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    if(samples==0ull) throw std::runtime_error("no samples collected");
    // Fold both halves onto wall distance.
    std::ostringstream csv;
    csv << "zw_lu,zw_over_h,U_over_utau,loglaw_over_utau,urms_over_utau,vrms_over_utau,wrms_over_utau,"
           "minus_uw_resolved_over_utau2\n";
    double bulk = 0.0;
    for(uint z=1u; z<=o.h; ++z) {
        const uint zt = g_open_channel ? z : nz-1u-z;
        const double zw = (double)z-0.5;
        const double U = 0.5*(sum_u[z]+sum_u[zt])/(double)samples;
        const double uu = 0.5*(sum_uu[z]+sum_uu[zt])/(double)samples;
        const double vv = 0.5*(sum_vv[z]+sum_vv[zt])/(double)samples;
        const double ww = 0.5*(sum_ww[z]+sum_ww[zt])/(double)samples;
        const double uw = g_open_channel ? sum_uw[z]/(double)samples
                                         : 0.5*(sum_uw[z]-sum_uw[zt])/(double)samples;
        bulk += U;
        const double ut = (double)o.utau;
        csv << std::setprecision(8) << zw << ',' << zw/(double)o.h << ',' << U/ut << ','
            << std::log1p(zw/(double)o.z0)/(double)o.kappa << ',' << std::sqrt(uu)/ut << ','
            << std::sqrt(vv)/ut << ',' << std::sqrt(ww)/ut << ',' << -uw/(ut*ut) << '\n';
    }
    bulk /= (double)o.h;
    const double kappa = (double)o.kappa, z0 = (double)o.z0, h = (double)o.h;
    // Rough log law integrated over the half channel (no wake).
    const double bulk_log = (1.0/kappa)*((1.0+h/z0)*std::log1p(h/z0)/(h/z0)-1.0);
    const double mlups = (double)o.nx*(double)o.ny*(double)nz*(double)o.steps/seconds*1.0E-6;
    std::ostringstream summary;
    summary << std::setprecision(6)
            << "{\n  \"wall_model\": " << (o.wall_model ? "true" : "false")
            << ",\n  \"h\": " << o.h << ",\n  \"nx\": " << o.nx << ",\n  \"ny\": " << o.ny
            << ",\n  \"z0_lu\": " << o.z0 << ",\n  \"z0_over_h\": " << o.z0_over_h
            << ",\n  \"utau\": " << o.utau << ",\n  \"cs\": " << o.cs << ",\n  \"nu\": " << o.nu
            << ",\n  \"steps\": " << o.steps << ",\n  \"samples\": " << samples
            << ",\n  \"eddy_turnovers_h_over_utau\": " << (double)o.steps*(double)o.utau/h
            << ",\n  \"bulk_over_utau\": " << bulk/(double)o.utau
            << ",\n  \"bulk_loglaw_over_utau\": " << bulk_log
            << ",\n  \"bulk_relative_error\": " << bulk/(double)o.utau/bulk_log-1.0
            << ",\n  \"mlups\": " << mlups << "\n}\n";
    std::cout << summary.str();
    if(!o.output.empty()) {
        std::ofstream(o.output+"_profile.csv") << csv.str();
        std::ofstream(o.output+"_summary.json") << summary.str();
        std::ofstream history(o.output+"_bulk_history.csv");
        history << "step,Ub_over_utau\n";
        for(const auto& entry : bulk_history)
            history << entry.first << ',' << entry.second/(double)o.utau << '\n';
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Throughput benchmark: ground plus an aligned cube array.

int run_benchmark(const int argc, const char* const argv[]) {
    uint nx = 256u, ny = 256u, nz = 96u, steps = 2000u, warmup = 200u, repeats = 3u;
    uint period = 32u, size = 12u, height = 12u, sgs_layers = 3u;
    std::string only;
    bool cubes = true;
    for(int i=2; i<argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if(i+1>=argc) throw std::invalid_argument("missing value for "+key);
            return argv[++i];
        };
        if(key=="--nx") nx = (uint)std::stoul(value());
        else if(key=="--ny") ny = (uint)std::stoul(value());
        else if(key=="--nz") nz = (uint)std::stoul(value());
        else if(key=="--steps") steps = (uint)std::stoul(value());
        else if(key=="--warmup") warmup = (uint)std::stoul(value());
        else if(key=="--repeats") repeats = (uint)std::stoul(value());
        else if(key=="--no-cubes") cubes = false;
        else if(key=="--period") period = (uint)std::stoul(value());
        else if(key=="--size") size = (uint)std::stoul(value());
        else if(key=="--height") height = (uint)std::stoul(value());
        else if(key=="--sgs-layers") sgs_layers = (uint)std::stoul(value());
        else if(key=="--only") only = value();
        else throw std::invalid_argument("unknown option "+key);
    }
    auto run = [&](const bool wall_model) {
        configure(wall_model, 0.03f, 0.41f, true, 0.173265956f, 1.0E-5f, sgs_layers);
        culbm::solver::Solver lbm(nx, ny, nz, 1u, 1u, 1u,
            culbm::solver::CaseCoefficients().viscosity(1.0E-5f).thermal(1.0E-5f, 0.0f));
        ulong wall_cells = 0ull;
        for(uint z=0u; z<nz; ++z) for(uint y=0u; y<ny; ++y) for(uint x=0u; x<nx; ++x) {
            const ulong n = lbm.grid().index(x, y, z);
            bool solid = z==0u;
            if(cubes && z<=height) {
                const uint cx = x%period, cy = y%period, o = (period-size)/2u;
                solid = solid || (cx>=o && cx<o+size && cy>=o && cy<o+size);
            }
            lbm.flags()[n] = solid ? (uchar)CELL_SOLID : (uchar)0u;
            lbm.density()[n] = 1.0f;
            lbm.velocity().x[n] = solid ? 0.0f : 0.05f;
            lbm.velocity().y[n] = 0.0f;
            lbm.velocity().z[n] = 0.0f;
            lbm.temperature()[n] = 1.0f;
        }
        for(uint z=1u; z+1u<nz; ++z) for(uint y=1u; y+1u<ny; ++y) for(uint x=1u; x+1u<nx; ++x) {
            const ulong n = lbm.grid().index(x, y, z);
            if(lbm.flags()[n]&CELL_SOLID) continue;
            const bool adjacent = (lbm.flags()[lbm.grid().index(x+1u, y, z)]|lbm.flags()[lbm.grid().index(x-1u, y, z)]|
                                   lbm.flags()[lbm.grid().index(x, y+1u, z)]|lbm.flags()[lbm.grid().index(x, y-1u, z)]|
                                   lbm.flags()[lbm.grid().index(x, y, z+1u)]|lbm.flags()[lbm.grid().index(x, y, z-1u)])&CELL_SOLID;
            wall_cells += adjacent ? 1ull : 0ull;
        }
        lbm.initialize_once();
        for(uint s=0u; s<warmup; ++s) lbm.advance();
        lbm.synchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for(uint s=0u; s<steps; ++s) lbm.advance();
        lbm.synchronize();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        const double mlups = (double)nx*(double)ny*(double)nz*(double)steps/seconds*1.0E-6;
        lbm.velocity().download();
        bool finite = true;
        for(ulong n=0ull; n<lbm.grid().cells(); ++n) finite = finite && std::isfinite(lbm.velocity().x[n]);
        std::cout << "benchmark wm=" << (wall_model ? 1 : 0) << " cells=" << lbm.grid().cells()
                  << " wall_cells=" << wall_cells << " (" << 100.0*(double)wall_cells/(double)lbm.grid().cells()
                  << "%) MLUPS=" << std::setprecision(6) << mlups << " finite=" << (finite ? 1 : 0) << '\n';
        return mlups;
    };
    std::vector<double> off, on;
    if(only=="off" || only=="on") {
        run(only=="on");
        return 0;
    }
    for(uint r=0u; r<repeats; ++r) {
        if(r%2u==0u) { off.push_back(run(false)); on.push_back(run(true)); }
        else { on.push_back(run(true)); off.push_back(run(false)); }
    }
    std::sort(off.begin(), off.end());
    std::sort(on.begin(), on.end());
    const double m_off = off[off.size()/2u], m_on = on[on.size()/2u];
    std::cout << "median MLUPS off=" << m_off << " on=" << m_on << " on/off=" << m_on/m_off << '\n';
    return 0;
}

// ----------------------------------------------------------------------------
// One/two-GPU agreement of a short turbulent wall-modelled channel.

std::vector<float> channel_snapshot(const uint domains_x, const ulong steps) {
    ChannelOptions o;
    o.h = 8u; o.nx = 64u; o.ny = 32u;
    o.z0 = 0.01f;
    configure(true, o.z0, o.kappa, true, o.cs, o.nu);
    const uint nz = 2u*o.h+2u;
    const double g = (double)o.utau*(double)o.utau/(double)o.h;
    culbm::solver::Solver lbm(o.nx, o.ny, nz, domains_x, 1u, 1u,
        culbm::solver::CaseCoefficients().viscosity(o.nu).volume_force((float)g, 0.0f, 0.0f).thermal(o.nu, 0.0f));
    fill_channel(lbm, o.h, o.utau, o.z0, o.kappa, 0.3, 777u);
    lbm.initialize_once();
    for(ulong s=0ull; s<steps; ++s) lbm.advance();
    lbm.synchronize();
    lbm.velocity().download();
    std::vector<float> velocity(3u*(std::size_t)lbm.grid().cells());
    for(ulong n=0ull; n<lbm.grid().cells(); ++n) {
        velocity[(std::size_t)n] = lbm.velocity().x[n];
        velocity[(std::size_t)(lbm.grid().cells()+n)] = lbm.velocity().y[n];
        velocity[(std::size_t)(2ull*lbm.grid().cells()+n)] = lbm.velocity().z[n];
    }
    return velocity;
}

void test_multi_gpu(TestState& test) {
    const std::vector<float> one = channel_snapshot(1u, 400ull);
    const std::vector<float> two = channel_snapshot(2u, 400ull);
    test.expect(one.size()==two.size() && !one.empty(), "snapshots have equal extent");
    if(one.size()!=two.size()) return;
    double max_error = 0.0;
    std::size_t identical = 0u;
    for(std::size_t i=0u; i<one.size(); ++i) {
        max_error = std::max(max_error, (double)std::fabs(one[i]-two[i]));
        identical += one[i]==two[i] ? 1u : 0u;
    }
    std::cout << "wall model one/two-GPU max |du|=" << max_error << " identical="
              << identical << '/' << one.size() << '\n';
    test.expect(max_error<=2.0E-4, "wall-modelled channel agrees between one and two GPUs");
}

// ----------------------------------------------------------------------------
// Wall-modelled SurfaceIBM surfaces.
//
// Laminar body-force slab between two horizontal IBM planes, z_b=4.3 (normal
// +z) and z_t=nz-1-4.3 (normal -z), each backed by a voxel solid core.  The
// IBM cannot place markers whose Roma3 support touches the periodic seams, so
// the planes cover x,y in [2, n-3]; the laterally uniform centre is sampled.
// The wall-modelled core faces reflect specularly, so the 14 fluid rows of a
// centre column (z=5..18) must be carried by the two modelled wall stresses in
// steady state: tau_w = 7*rho*g = rho*Cd(h)*u_p^2 with the exterior probe at
// h=5-z_b=0.7.

void configure_surface_ibm(const bool enabled) {
    culbm::surface_ibm::Config configuration;
    configuration.enabled = enabled;
    configuration.marker_spacing_lu = 0.75f;
    configuration.forcing_iterations = 8u;
    configuration.relaxation = 1.0f;
    configuration.residual_tolerance = 1.0E-5f;
    configuration.retraction_lu = 0.5f;
    configuration.support_thickness_lu = 0.75f;
    configuration.max_markers = 1000000u;
    std::string error;
    if(!culbm::surface_ibm::validate_config(configuration, error))
        throw std::runtime_error("invalid SurfaceIBM test configuration: "+error);
    culbm::surface_ibm::set_config(configuration);
}

void append_plane_markers(std::vector<culbm::surface_ibm::Marker>& markers,
                          const uint n, const float z_surface, const float normal_z) {
    const float lo = 2.0f, length = (float)n-5.0f;
    const uint count = (uint)std::lround(length/0.75f);
    const float spacing = length/(float)count;
    for(uint j=0u; j<count; ++j) {
        for(uint i=0u; i<count; ++i) {
            culbm::surface_ibm::Marker marker;
            marker.position_lu = float3(lo+((float)i+0.5f)*spacing, lo+((float)j+0.5f)*spacing,
                                        z_surface-0.5f*normal_z); // retracted into the solid
            marker.wall_velocity_lu = float3(0.0f);
            marker.normal = float3(0.0f, 0.0f, normal_z);
            marker.area_lu2 = spacing*spacing;
            marker.global_id = (std::uint64_t)markers.size();
            markers.push_back(marker);
        }
    }
}

struct IbmSlabResult {
    bool finite = true;
    double u_probe = 0.0, u_mid = 0.0, max_uz_probe = 0.0;
    double mass_ratio = 0.0;
    double fluid_cells = 0.0;
    double probe_u2_sum = 0.0; // sum of ux^2 over both exterior probe rows of the marker patches
    culbm::surface_ibm::Statistics statistics;
    std::vector<float> velocity;
};

enum class IbmSlabMode { normal, padded_core, padded_voxel_only, z_partition };

IbmSlabResult ibm_slab(const bool wall_model_ibm, const uint domains, const ulong steps,
                       const double g, const float nu, const float z0, const float zb=4.3f,
                       const IbmSlabMode mode=IbmSlabMode::normal) {
    constexpr uint n = 64u, nz = 24u;
    const bool split_z = mode==IbmSlabMode::z_partition;
    const bool attach_markers = mode!=IbmSlabMode::padded_voxel_only;
    const float core_padding = mode==IbmSlabMode::padded_core || !attach_markers ? 0.2f : 0.0f;
    const float zt = split_z ? 18.7f : (float)(nz-1u)-zb;
    configure(true, z0, 0.41f, false, 0.0f, nu);
    {
        Wall_Model_Configuration configuration = get_wall_model_configuration();
        configuration.surface_ibm = wall_model_ibm;
        if(!set_wall_model_configuration(configuration))
            throw std::runtime_error("cannot select the IBM wall-model mode");
    }
    configure_surface_ibm(attach_markers);
    IbmSlabResult result;
    {
        culbm::solver::Solver lbm(n, n, nz, split_z ? 1u : domains, 1u, split_z ? domains : 1u,
            culbm::solver::CaseCoefficients().viscosity(nu).volume_force((float)g, 0.0f, 0.0f).thermal(nu, 0.0f));
        for(uint z=0u; z<nz; ++z) for(uint y=0u; y<n; ++y) for(uint x=0u; x<n; ++x) {
            const ulong c = lbm.grid().index(x, y, z);
            const bool solid = (float)z<zb+core_padding || (float)z>zt-core_padding;
            lbm.density()[c] = 1.0f;
            lbm.velocity().x[c] = solid ? 0.0f : 0.02f;
            lbm.velocity().y[c] = 0.0f;
            lbm.velocity().z[c] = 0.0f;
            lbm.flags()[c] = solid ? CELL_SOLID : (uchar)0u;
            lbm.temperature()[c] = 1.0f;
        }
        std::vector<culbm::surface_ibm::Marker> markers;
        append_plane_markers(markers, n, zb, 1.0f);
        append_plane_markers(markers, n, zt, -1.0f);
        if(attach_markers) culbm::surface_ibm::attach(lbm, markers);
        double mass0 = 0.0;
        for(ulong c=0ull; c<lbm.grid().cells(); ++c) if((lbm.flags()[c]&CELL_SOLID)==0u) mass0 += 1.0;
        lbm.initialize_once();
        for(ulong s=0ull; s<steps; ++s) lbm.advance();
        lbm.synchronize();
        result.statistics = culbm::surface_ibm::statistics(lbm);
        lbm.density().download();
        lbm.velocity().download();
        double mass = 0.0;
        std::vector<double> mean(nz, 0.0);
        uint samples = 0u;
        for(uint z=0u; z<nz; ++z) {
            uint count = 0u;
            for(uint y=24u; y<40u; ++y) for(uint x=24u; x<40u; ++x) {
                const ulong c = lbm.grid().index(x, y, z);
                mean[z] += lbm.velocity().x[c];
                ++count;
                if(z==5u) result.max_uz_probe = std::max(result.max_uz_probe, (double)std::fabs(lbm.velocity().z[c]));
            }
            mean[z] /= (double)count;
            samples = count;
        }
        (void)samples;
        for(ulong c=0ull; c<lbm.grid().cells(); ++c) {
            result.finite = result.finite && std::isfinite(lbm.velocity().x[c]) && std::isfinite(lbm.velocity().z[c]) &&
                            std::isfinite(lbm.density()[c]);
            if((lbm.flags()[c]&CELL_SOLID)==0u) mass += lbm.density()[c];
        }
        result.mass_ratio = mass/mass0;
        result.fluid_cells = mass0;
        // Marker patches cover x,y in [2, n-3]; each marker area is spacing^2,
        // so the patch integral of ux^2 over the probe rows z=5 and z=nz-6
        // approximates sum(area*u_probe^2) of all markers.
        for(const uint z : {5u, nz-6u}) {
            for(uint y=2u; y<=n-3u; ++y) for(uint x=2u; x<=n-3u; ++x) {
                const double ux = lbm.velocity().x[lbm.grid().index(x, y, z)];
                const double wx = (x==2u || x==n-3u) ? 0.5 : 1.0;
                const double wy = (y==2u || y==n-3u) ? 0.5 : 1.0;
                result.probe_u2_sum += wx*wy*ux*ux;
            }
        }
        if(std::getenv("CULBM_WM_IBM_DEBUG")) {
            std::cout << "IBM slab profile (wm_ibm=" << wall_model_ibm << ", resident="
                      << result.statistics.device_resident << ", iterations=" << result.statistics.iterations
                      << ", max slip=" << result.statistics.max_slip_lu << ", reaction="
                      << result.statistics.reaction_force_lu.x << ',' << result.statistics.reaction_force_lu.y
                      << ',' << result.statistics.reaction_force_lu.z << "):";
            for(uint z=0u; z<nz; ++z) std::cout << ' ' << mean[z];
            std::cout << '\n';
        }
        const uint first_fluid = (uint)std::ceil(zb+core_padding);
        result.u_probe = 0.5*(mean[first_fluid]+mean[(uint)std::floor(zt-core_padding)]);
        result.u_mid = 0.5*(mean[nz/2u-1u]+mean[nz/2u]);
        result.velocity.resize(3u*(std::size_t)lbm.grid().cells());
        for(ulong c=0ull; c<lbm.grid().cells(); ++c) {
            result.velocity[(std::size_t)c] = lbm.velocity().x[c];
            result.velocity[(std::size_t)(lbm.grid().cells()+c)] = lbm.velocity().y[c];
            result.velocity[(std::size_t)(2ull*lbm.grid().cells()+c)] = lbm.velocity().z[c];
        }
    }
    configure_surface_ibm(false);
    return result;
}

void test_ibm_laminar_balance(TestState& test) {
    constexpr float nu = 0.02f, z0 = 0.05f;
    // Exterior probe at h=0.7; 14 fluid rows per centre column, two walls.
    const double cd = std::pow(0.41/std::log1p(0.7/(double)z0), 2.0);
    const double u_target = 0.02;
    const double g = cd*u_target*u_target/7.0;
    const IbmSlabResult on = ibm_slab(true, 1u, 60000ull, g, nu, z0);
    // Global steady momentum balance: the free-slip IBM cores take no stress;
    // the seam strips without markers keep the bounce-back wall model (6% of
    // the plane area), so the IBM wall force carries slightly less than the
    // whole body force.
    const double reaction = on.statistics.reaction_force_lu.x;
    const double body = g*on.fluid_cells;
    // Wall law: total marker force = Cd(h=0.7)*sum(area*u_probe^2).
    const double law = cd*on.probe_u2_sum;
    std::cout << "IBM wall model laminar slab: centre u_probe=" << std::setprecision(7) << on.u_probe
              << " (local target " << u_target << ") reaction/body-force=" << reaction/body
              << " reaction/wall-law=" << reaction/law << " u_mid=" << on.u_mid
              << " mass ratio-1=" << on.mass_ratio-1.0
              << " max|uz(probe)|=" << on.max_uz_probe << '\n';
    test.expect(on.finite, "IBM wall-model slab stays finite");
    test.expect(on.statistics.wall_model && on.statistics.cleared_solid_cells==0u,
                "IBM runtime reports the wall model and keeps the voxel core");
    test.expect(reaction/body>0.88 && reaction/body<1.0,
                "IBM wall force carries the body force apart from the wall-modelled seams");
    test.expect(std::fabs(reaction/law-1.0)<2.0E-2,
                "IBM wall force follows the log law of the exterior probe velocity");
    // Seam strips without markers are frictionless, so the sampled centre
    // carries a little more than its own column (patch covers 85% of the area).
    test.expect(on.u_probe>0.99*u_target && on.u_probe<1.10*u_target,
                "centre probe velocity is close to the local log-law balance");
    test.expect(std::fabs(on.mass_ratio-1.0)<1.0E-5, "IBM wall model conserves mass");
    test.expect(on.max_uz_probe<1.0E-3*on.u_probe, "IBM wall model keeps the surface impermeable");
    // No-slip IBM at the same forcing: the first exterior row is much slower.
    const IbmSlabResult off = ibm_slab(false, 1u, 20000ull, g, nu, z0);
    std::cout << "IBM no-slip slab (same g): u_probe=" << off.u_probe << '\n';
    test.expect(off.finite && off.u_probe<0.5*on.u_probe,
                "culbm_wall_model_ibm=false keeps the no-slip immersed surface");
}

void test_ibm_subcell_wall_distance(TestState& test, const uint domains_x) {
    // Both geometries have the same voxel core and sample the same exterior
    // fluid row. Moving the immersed surface through that row's volume must
    // not turn its bounded first-cell wall stress into a singular impulse.
    for(const float z0 : {0.016f, 0.0f}) {
        const auto halfway = ibm_slab(true, domains_x, 4u, 0.0, 0.02f, z0, 4.5f);
        const auto cut = ibm_slab(true, domains_x, 4u, 0.0, 0.02f, z0, 4.9999f);
        const double reference = halfway.statistics.reaction_force_lu.x;
        const double reaction = cut.statistics.reaction_force_lu.x;
        std::cout << "IBM subcell wall z0=" << z0 << " domains=" << domains_x
                  << " halfway reaction=" << reference << " cut reaction=" << reaction
                  << " cut probe=" << cut.u_probe << '\n';
        test.expect(halfway.finite && cut.finite && reference>0.0 &&
                    std::fabs(reaction-reference) < 0.002*reference,
                    "a subcell IBM wall matches the resolved first-cell shear");
        test.expect(std::fabs(cut.u_probe-halfway.u_probe)<2.0E-5 &&
                    std::fabs(cut.mass_ratio-1.0)<1.0E-5,
                    "subcell IBM wall conserves mass and keeps the probe velocity bounded");
    }
    // A conservatively voxelized roof can cover the last cell in a retracted
    // marker's support. With no exterior stress to spread, the first fluid
    // row must retain the voxel wall law on that face.
    const auto voxel = ibm_slab(true, domains_x, 24u, 0.0, 0.02f, 0.05f, 4.9f, IbmSlabMode::padded_voxel_only);
    const auto covered = ibm_slab(true, domains_x, 24u, 0.0, 0.02f, 0.05f, 4.9f, IbmSlabMode::padded_core);
    std::cout << "Buried IBM support, voxel probe=" << voxel.u_probe
              << " with markers=" << covered.u_probe
              << " marker reaction=" << covered.statistics.reaction_force_lu.x << '\n';
    test.expect(covered.finite && voxel.finite && voxel.u_probe<0.01999 &&
                std::fabs(covered.u_probe-voxel.u_probe)<2.0E-6 &&
                std::fabs(covered.statistics.reaction_force_lu.x)<1.0E-8,
                "a wall without an exterior marker sample retains the voxel shear");
}

void test_ibm_multi_gpu(TestState& test) {
    // At z=11.3 the solid neighbour and its fluid recipient at z=12 belong
    // to different devices. Coverage is owned by the fluid cell.
    const auto local_wall = ibm_slab(true, 1u, 24u, 0.0, 0.02f, 0.05f, 11.3f, IbmSlabMode::z_partition);
    const auto split_wall = ibm_slab(true, 2u, 24u, 0.0, 0.02f, 0.05f, 11.3f, IbmSlabMode::z_partition);
    const std::size_t recipient = 32u+64u*(32u+64u*12u);
    test.expect(local_wall.finite && split_wall.finite &&
                std::fabs(local_wall.velocity[recipient]-split_wall.velocity[recipient])<2.0E-6,
                "a marker-supported wall across a GPU boundary keeps single-device shear");
    constexpr float nu = 0.02f, z0 = 0.05f;
    const double cd = std::pow(0.41/std::log1p(0.7/(double)z0), 2.0);
    const double g = cd*0.02*0.02/7.0;
    const IbmSlabResult one = ibm_slab(true, 1u, 2000ull, g, nu, z0);
    const IbmSlabResult two = ibm_slab(true, 2u, 2000ull, g, nu, z0);
    test.expect(one.velocity.size()==two.velocity.size() && !one.velocity.empty(),
                "IBM snapshots have equal extent");
    if(one.velocity.size()!=two.velocity.size()) return;
    double max_error = 0.0;
    for(std::size_t i=0u; i<one.velocity.size(); ++i)
        max_error = std::max(max_error, (double)std::fabs(one.velocity[i]-two.velocity[i]));
    std::cout << "IBM wall model device/portable (1/2 GPU) max |du|=" << max_error
              << " reaction_x " << one.statistics.reaction_force_lu.x << '/' << two.statistics.reaction_force_lu.x
              << " resident " << one.statistics.device_resident << '/' << two.statistics.device_resident << '\n';
    test.expect(one.statistics.device_resident && !two.statistics.device_resident,
                "single- and two-GPU runs use the device and portable IBM paths");
    // IBM force spreading uses atomics: the reaction varies by ~1e-4 between
    // runs of the same path in FP16C. On D3Q27 the eight corner links feed the
    // same atomic sums and the run-to-run spread reaches ~2e-3 (reaction) and
    // ~1e-4 (velocity), so that lattice gets a tenfold wider band.
    const bool corner_lattice = get_momentum_lattice()==Momentum_Lattice::Q27;
    const double reaction_tolerance = corner_lattice ? 5.0E-3 : 5.0E-4;
    const float velocity_tolerance = corner_lattice ? 5.0E-4f : 5.0E-5f;
    test.expect(std::fabs(one.statistics.reaction_force_lu.x/two.statistics.reaction_force_lu.x-1.0)<reaction_tolerance,
                "IBM wall-model reaction is decomposition independent");
    // The portable path evaluates the wall function on the host; FP16C storage
    // amplifies the last-bit differences to a few 1e-6 after 2000 steps.
    test.expect(max_error<=velocity_tolerance, "IBM wall model agrees between the device and portable paths");
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
        if(const char* lattice = std::getenv("CULBM_TEST_VELOCITY_SET")) {
            const std::string name(lattice);
            if(name!="D3Q19" && name!="D3Q27") {
                std::cerr << "CULBM_TEST_VELOCITY_SET must be D3Q19 or D3Q27\n";
                return 2;
            }
            set_momentum_lattice(name=="D3Q27" ? Momentum_Lattice::Q27 : Momentum_Lattice::Q19);
            std::cout << "momentum lattice " << get_momentum_lattice_name() << "\n";
        }
        if(argc>=2 && std::string(argv[1])=="--channel") return run_channel(parse_channel(argc, argv));
        if(argc>=2 && std::string(argv[1])=="--benchmark") return run_benchmark(argc, argv);
        if(argc==2 && std::string(argv[1])=="--multi-gpu") {
            if(device_count<2) {
                std::cout << "SKIP: wall-model multi-GPU test requires two devices.\n";
                return 77;
            }
            TestState test;
            test_multi_gpu(test);
            test_ibm_multi_gpu(test);
            test_ibm_subcell_wall_distance(test, 2u);
            return test.failures==0 ? 0 : 1;
        }
        if(argc==2 && std::string(argv[1])=="--ibm-cut-cell") {
            TestState test;
            test_ibm_subcell_wall_distance(test, 1u);
            return test.failures==0 ? 0 : 1;
        }
        if(argc==2 && std::string(argv[1])=="--ibm") {
            TestState test;
            test_ibm_laminar_balance(test);
            return test.failures==0 ? 0 : 1;
        }
        if(argc!=1) {
            std::cerr << "Usage: " << argv[0] << " [--ibm|--ibm-cut-cell|--multi-gpu|--channel [options]|--benchmark [options]]\n";
            return 2;
        }
        TestState test;
        if(std::getenv("CULBM_WM_TEST_FORCE_PROBE")) probe_force_input();
        test_laminar_rough_balance(test);
        if(!std::getenv("CULBM_WM_TEST_ONLY_LAMINAR")) {
            test_disabled_is_no_slip(test);
            test_smooth_viscous_sublayer(test);
            test_concave_pocket(test);
            test_ibm_laminar_balance(test);
            test_ibm_subcell_wall_distance(test, 1u);
        }
        if(test.failures==0) std::cout << "GPU wall-model tests passed.\n";
        return test.failures==0 ? 0 : 1;
    } catch(const std::exception& error) {
        std::cerr << "WALL_MODEL_TEST_ERROR " << error.what() << '\n';
        return 2;
    }
}
