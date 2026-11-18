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
#include "lbm.hpp"

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

// Open channel (--open): bottom wall, free-slip lid (TYPE_S|TYPE_X ghost row,
// requires a FREE_SLIP_BOUNDARIES build) at height h.
bool g_open_channel = false;

double wall_distance(const uint z, const uint h) {
    if(g_open_channel) return (double)z-0.5;
    return z<=h ? (double)z-0.5 : (double)(2u*h+1u-z)-0.5;
}

void fill_channel(LBM& lbm, const uint h, const double utau, const double z0,
                  const double kappa, const double noise, const std::uint64_t seed) {
    const uint nx = lbm.get_Nx(), ny = lbm.get_Ny(), nz = lbm.get_Nz();
    Lcg random(seed);
    for(uint z=0u; z<nz; ++z) {
        const bool solid = z==0u || z==nz-1u;
        const double zw = solid ? 0.0 : wall_distance(z, h);
        const double mean = solid ? 0.0 : utau/kappa*std::log1p(zw/z0);
        for(uint y=0u; y<ny; ++y) {
            for(uint x=0u; x<nx; ++x) {
                const ulong n = lbm.index(x, y, z);
                lbm.rho[n] = 1.0f;
                lbm.flags[n] = solid ? (uchar)((g_open_channel && z==nz-1u) ? (TYPE_S|TYPE_X) : TYPE_S) : (uchar)0u;
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
                lbm.u.x[n] = solid ? 0.0f : (float)ux;
                lbm.u.y[n] = solid ? 0.0f : (float)uy;
                lbm.u.z[n] = solid ? 0.0f : (float)uz;
                lbm.T[n] = 1.0f;
            }
        }
    }
}

ChannelField read_channel(LBM& lbm) {
    ChannelField field;
    field.nx = lbm.get_Nx();
    field.ny = lbm.get_Ny();
    field.nz = lbm.get_Nz();
    lbm.rho.read_from_device();
    lbm.u.read_from_device();
    field.mean_u.assign(field.nz, 0.0);
    for(uint z=1u; z+1u<field.nz; ++z) {
        double sum = 0.0;
        for(uint y=0u; y<field.ny; ++y) {
            for(uint x=0u; x<field.nx; ++x) {
                const ulong n = lbm.index(x, y, z);
                const double ux = lbm.u.x[n], uz = lbm.u.z[n], rho = lbm.rho[n];
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
    LBM lbm(4u, 4u, 2u*h+2u, nu, (float)g, 0.0f, 0.0f, 0.0f, nu, 0.0f);
    fill_channel(lbm, h, 0.0, z0, kappa, 0.0, 1u);
    lbm.external_initialize();
    const double mass0 = (double)(4u*4u*2u*h);
    uint laminar_steps = 120000u;
    if(const char* env = std::getenv("CULBM_WM_TEST_LAMINAR_STEPS")) laminar_steps = (uint)std::strtoul(env, nullptr, 10);
    for(uint step=0u; step<laminar_steps; ++step) lbm.external_step();
    lbm.external_finish();
    // Esoteric-Pull reflects wall populations after two steps, so odd and
    // even steps settle to slightly different wall states; average a pair.
    const ChannelField field_a = read_channel(lbm);
    lbm.external_step();
    lbm.external_finish();
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
    const double rho1 = lbm.rho[lbm.index(0u, 0u, 1u)];
    const double rho_mid = lbm.rho[lbm.index(0u, 0u, h)];
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
    LBM lbm(n, n, nz, 1.0E-4f, 2.0E-6f, 0.0f, 0.0f, 0.0f, 1.0E-4f, 0.0f);
    const auto block = [&](const uint x, const uint y, const uint z) {
        const bool in_block = x>=6u && x<18u && y>=6u && y<18u && z>=1u && z<=5u;
        const bool well = x==11u && y==11u;          // one-cell vertical well
        const bool slot = y==14u && x>=8u && x<16u && z>=3u; // one-cell slot from the roof
        return in_block && !well && !slot;
    };
    for(uint z=0u; z<nz; ++z) for(uint y=0u; y<n; ++y) for(uint x=0u; x<n; ++x) {
        const ulong i = lbm.index(x, y, z);
        const bool solid = z==0u || z==nz-1u || block(x, y, z);
        lbm.flags[i] = solid ? (uchar)TYPE_S : (uchar)0u;
        lbm.rho[i] = 1.0f;
        const bool inside_footprint = x>=6u && x<18u && y>=6u && y<18u && z<=5u;
        lbm.u.x[i] = (solid || inside_footprint) ? 0.0f : 0.05f;
        lbm.u.y[i] = 0.0f;
        lbm.u.z[i] = 0.0f;
        lbm.T[i] = 1.0f;
    }
    lbm.external_initialize();
    double well_max = 0.0;
    double top_mean = 0.0;
    for(uint s=0u; s<6000u; ++s) {
        lbm.external_step();
        if(s<3000u || s%500u!=499u) continue;
        lbm.external_finish();
        lbm.u.read_from_device();
        double top = 0.0;
        for(uint y=0u; y<n; ++y) for(uint x=0u; x<n; ++x) top += lbm.u.x[lbm.index(x, y, 9u)];
        top_mean = top/(double)(n*n);
        for(uint z=1u; z<=4u; ++z) {
            const ulong i = lbm.index(11u, 11u, z);
            well_max = std::max(well_max, std::sqrt((double)lbm.u.x[i]*lbm.u.x[i]+(double)lbm.u.y[i]*lbm.u.y[i]+
                                                    (double)lbm.u.z[i]*lbm.u.z[i]));
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
    LBM lbm(4u, 4u, 4u, 0.02f, (float)g, 0.0f, 0.0f, 0.0f, 0.02f, 0.0f);
    for(ulong n=0ull; n<lbm.get_N(); ++n) {
        lbm.rho[n] = 1.0f; lbm.u.x[n] = 0.0f; lbm.u.y[n] = 0.0f; lbm.u.z[n] = 0.0f;
        lbm.flags[n] = 0u; lbm.T[n] = 1.0f;
    }
    lbm.external_initialize();
    for(uint s=0u; s<1000u; ++s) lbm.external_step();
    lbm.external_finish();
    lbm.u.read_from_device();
    const double u0 = lbm.u.x[0];
    for(uint s=0u; s<1000u; ++s) lbm.external_step();
    lbm.external_finish();
    lbm.u.read_from_device();
    const double u1 = lbm.u.x[0];
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
    LBM lbm(4u, 4u, 2u*h+2u, nu, (float)g, 0.0f, 0.0f, 0.0f, nu, 0.0f);
    fill_channel(lbm, h, 0.0, 0.05, 0.41, 0.0, 1u);
    lbm.external_initialize();
    for(uint step=0u; step<40000u; ++step) lbm.external_step();
    lbm.external_finish();
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
    LBM lbm(4u, 4u, 2u*h+2u, nu, (float)g, 0.0f, 0.0f, 0.0f, nu, 0.0f);
    fill_channel(lbm, h, 0.0, 0.05, 0.41, 0.0, 1u);
    lbm.external_initialize();
    for(uint step=0u; step<20000u; ++step) lbm.external_step();
    lbm.external_finish();
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
    LBM lbm(o.nx, o.ny, nz, o.domains_x, 1u, 1u, o.nu, (float)g, 0.0f, 0.0f, 0.0f, o.nu, 0.0f);
    fill_channel(lbm, o.h, o.utau, o.z0, o.kappa, 0.3, 12345u);
    lbm.external_initialize();
    std::vector<double> sum_u(nz, 0.0), sum_uu(nz, 0.0), sum_vv(nz, 0.0), sum_ww(nz, 0.0), sum_uw(nz, 0.0);
    ulong samples = 0ull;
    std::vector<std::pair<ulong, double>> bulk_history;
    const auto t0 = std::chrono::steady_clock::now();
    for(ulong step=1ull; step<=o.steps; ++step) {
        lbm.external_step();
        const bool sample = step>o.sample_start && step%o.sample_every==0ull;
        const bool monitor = step%2000ull==0ull;
        static const ulong trace_from = std::getenv("CULBM_WM_TRACE_FROM") ?
            std::strtoull(std::getenv("CULBM_WM_TRACE_FROM"), nullptr, 10) : ~0ull;
        static const ulong trace_every = std::getenv("CULBM_WM_TRACE_EVERY") ?
            std::strtoull(std::getenv("CULBM_WM_TRACE_EVERY"), nullptr, 10) : 50ull;
        if(step>=trace_from && step%trace_every==0ull) {
            // Diagnostic trace: extreme |u|, |u_z| and rho over all fluid rows.
            lbm.rho.read_from_device();
            lbm.u.read_from_device();
            double best = 0.0, best_uz = 0.0, rmin = 1e30, rmax = -1e30;
            uint bx = 0u, by = 0u, bz = 0u, zx = 0u, zy = 0u, zz = 0u;
            for(uint z=1u; z+1u<nz; ++z) for(uint y=0u; y<o.ny; ++y) for(uint x=0u; x<o.nx; ++x) {
                const ulong n = lbm.index(x, y, z);
                const double mag = std::sqrt((double)lbm.u.x[n]*lbm.u.x[n]+(double)lbm.u.y[n]*lbm.u.y[n]+
                                             (double)lbm.u.z[n]*lbm.u.z[n]);
                if(mag>best) { best = mag; bx = x; by = y; bz = z; }
                if(std::fabs(lbm.u.z[n])>best_uz) { best_uz = std::fabs(lbm.u.z[n]); zx = x; zy = y; zz = z; }
                rmin = std::min(rmin, (double)lbm.rho[n]);
                rmax = std::max(rmax, (double)lbm.rho[n]);
            }
            const ulong nb = lbm.index(bx, by, bz);
            std::cout << "trace " << step << " max|u|=" << best << " @(" << bx << ',' << by << ',' << bz
                      << ") u=(" << lbm.u.x[nb] << ',' << lbm.u.y[nb] << ',' << lbm.u.z[nb] << ") rho=" << lbm.rho[nb]
                      << " max|uz|=" << best_uz << " @(" << zx << ',' << zy << ',' << zz << ") rho[min,max]=["
                      << rmin << ',' << rmax << "]\n";
            if(best>0.5) return 1;
        }
        if(!sample && !monitor) continue;
        lbm.u.read_from_device();
        double bulk = 0.0;
        bool finite = true;
        double umax = 0.0;
        uint umax_z = 0u;
        std::vector<double> plane_u(nz, 0.0);
        for(uint z=1u; z+1u<nz; ++z) {
            double su = 0.0;
            for(uint y=0u; y<o.ny; ++y) for(uint x=0u; x<o.nx; ++x) {
                const ulong n = lbm.index(x, y, z);
                su += lbm.u.x[n];
                finite = finite && std::isfinite(lbm.u.x[n]);
                const double mag = std::sqrt((double)lbm.u.x[n]*lbm.u.x[n]+(double)lbm.u.y[n]*lbm.u.y[n]+
                                             (double)lbm.u.z[n]*lbm.u.z[n]);
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
                    const ulong n = lbm.index(x, y, z);
                    const double up = lbm.u.x[n]-um, vp = lbm.u.y[n], wp = lbm.u.z[n];
                    suu += up*up; svv += vp*vp; sww += wp*wp; suw += up*wp;
                }
                const double cells = (double)(o.nx*o.ny);
                sum_u[z] += um; sum_uu[z] += suu/cells; sum_vv[z] += svv/cells;
                sum_ww[z] += sww/cells; sum_uw[z] += suw/cells;
            }
            ++samples;
        }
    }
    lbm.external_finish();
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
        LBM lbm(nx, ny, nz, 1.0E-5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0E-5f, 0.0f);
        ulong wall_cells = 0ull;
        for(uint z=0u; z<nz; ++z) for(uint y=0u; y<ny; ++y) for(uint x=0u; x<nx; ++x) {
            const ulong n = lbm.index(x, y, z);
            bool solid = z==0u;
            if(cubes && z<=height) {
                const uint cx = x%period, cy = y%period, o = (period-size)/2u;
                solid = solid || (cx>=o && cx<o+size && cy>=o && cy<o+size);
            }
            lbm.flags[n] = solid ? (uchar)TYPE_S : (uchar)0u;
            lbm.rho[n] = 1.0f;
            lbm.u.x[n] = solid ? 0.0f : 0.05f;
            lbm.u.y[n] = 0.0f;
            lbm.u.z[n] = 0.0f;
            lbm.T[n] = 1.0f;
        }
        for(uint z=1u; z+1u<nz; ++z) for(uint y=1u; y+1u<ny; ++y) for(uint x=1u; x+1u<nx; ++x) {
            const ulong n = lbm.index(x, y, z);
            if(lbm.flags[n]&TYPE_S) continue;
            const bool adjacent = (lbm.flags[lbm.index(x+1u, y, z)]|lbm.flags[lbm.index(x-1u, y, z)]|
                                   lbm.flags[lbm.index(x, y+1u, z)]|lbm.flags[lbm.index(x, y-1u, z)]|
                                   lbm.flags[lbm.index(x, y, z+1u)]|lbm.flags[lbm.index(x, y, z-1u)])&TYPE_S;
            wall_cells += adjacent ? 1ull : 0ull;
        }
        lbm.external_initialize();
        for(uint s=0u; s<warmup; ++s) lbm.external_step();
        lbm.external_finish();
        const auto t0 = std::chrono::steady_clock::now();
        for(uint s=0u; s<steps; ++s) lbm.external_step();
        lbm.external_finish();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        const double mlups = (double)nx*(double)ny*(double)nz*(double)steps/seconds*1.0E-6;
        lbm.u.read_from_device();
        bool finite = true;
        for(ulong n=0ull; n<lbm.get_N(); ++n) finite = finite && std::isfinite(lbm.u.x[n]);
        std::cout << "benchmark wm=" << (wall_model ? 1 : 0) << " cells=" << lbm.get_N()
                  << " wall_cells=" << wall_cells << " (" << 100.0*(double)wall_cells/(double)lbm.get_N()
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
    LBM lbm(o.nx, o.ny, nz, domains_x, 1u, 1u, o.nu, (float)g, 0.0f, 0.0f, 0.0f, o.nu, 0.0f);
    fill_channel(lbm, o.h, o.utau, o.z0, o.kappa, 0.3, 777u);
    lbm.external_initialize();
    for(ulong s=0ull; s<steps; ++s) lbm.external_step();
    lbm.external_finish();
    lbm.u.read_from_device();
    std::vector<float> velocity(3u*(std::size_t)lbm.get_N());
    for(ulong n=0ull; n<lbm.get_N(); ++n) {
        velocity[(std::size_t)n] = lbm.u.x[n];
        velocity[(std::size_t)(lbm.get_N()+n)] = lbm.u.y[n];
        velocity[(std::size_t)(2ull*lbm.get_N()+n)] = lbm.u.z[n];
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
        if(argc>=2 && std::string(argv[1])=="--channel") return run_channel(parse_channel(argc, argv));
        if(argc>=2 && std::string(argv[1])=="--benchmark") return run_benchmark(argc, argv);
        if(argc==2 && std::string(argv[1])=="--multi-gpu") {
            if(device_count<2) {
                std::cout << "SKIP: wall-model multi-GPU test requires two devices.\n";
                return 77;
            }
            TestState test;
            test_multi_gpu(test);
            return test.failures==0 ? 0 : 1;
        }
        if(argc!=1) {
            std::cerr << "Usage: " << argv[0] << " [--multi-gpu|--channel [options]|--benchmark [options]]\n";
            return 2;
        }
        TestState test;
        if(std::getenv("CULBM_WM_TEST_FORCE_PROBE")) probe_force_input();
        test_laminar_rough_balance(test);
        if(!std::getenv("CULBM_WM_TEST_ONLY_LAMINAR")) {
            test_disabled_is_no_slip(test);
            test_smooth_viscous_sublayer(test);
            test_concave_pocket(test);
        }
        if(test.failures==0) std::cout << "GPU wall-model tests passed.\n";
        return test.failures==0 ? 0 : 1;
    } catch(const std::exception& error) {
        std::cerr << "WALL_MODEL_TEST_ERROR " << error.what() << '\n';
        return 2;
    }
}
