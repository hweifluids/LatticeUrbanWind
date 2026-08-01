#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

namespace {

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

constexpr std::array<int, 19> kCx = {
    0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0};
constexpr std::array<int, 19> kCy = {
    0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1};
constexpr std::array<int, 19> kCz = {
    0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1};
constexpr std::array<double, 19> kWeight = {
    1.0/3.0,
    1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,
    1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,
    1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,
    1.0/36.0,1.0/36.0};

std::array<double, 19> shifted_equilibrium(
    const double rho, const double ux, const double uy, const double uz) {
    std::array<double, 19> f{};
    const double u2 = ux*ux+uy*uy+uz*uz;
    for(std::size_t i=0; i<f.size(); ++i) {
        const double cu =
            (double)kCx[i]*ux+(double)kCy[i]*uy+(double)kCz[i]*uz;
        f[i] = kWeight[i]*(
            rho*(1.0+3.0*cu+4.5*cu*cu-1.5*u2)-1.0);
    }
    return f;
}

double shifted_nonequilibrium_q(
    const std::array<double, 19>& f,
    const double rho, const double ux, const double uy, const double uz,
    const double fx = 0.0, const double fy = 0.0, const double fz = 0.0) {
    const double shifted_pressure = (rho-1.0)/3.0;
    const double hxx =
        f[1]+f[2]+f[7]+f[8]+f[9]+f[10]+f[13]+f[14]+f[15]+f[16]
        -(shifted_pressure+rho*ux*ux)+ux*fx;
    const double hyy =
        f[3]+f[4]+f[7]+f[8]+f[11]+f[12]+f[13]+f[14]+f[17]+f[18]
        -(shifted_pressure+rho*uy*uy)+uy*fy;
    const double hzz =
        f[5]+f[6]+f[9]+f[10]+f[11]+f[12]+f[15]+f[16]+f[17]+f[18]
        -(shifted_pressure+rho*uz*uz)+uz*fz;
    const double hxy = f[7]+f[8]-f[13]-f[14]-rho*ux*uy+
        0.5*(ux*fy+uy*fx);
    const double hxz = f[9]+f[10]-f[15]-f[16]-rho*ux*uz+
        0.5*(ux*fz+uz*fx);
    const double hyz = f[11]+f[12]-f[17]-f[18]-rho*uy*uz+
        0.5*(uy*fz+uz*fy);
    return hxx*hxx+hyy*hyy+hzz*hzz+
        2.0*(hxy*hxy+hxz*hxz+hyz*hyz);
}

void test_guo_force_correction(TestState& test) {
    constexpr double rho = 1.1;
    constexpr double raw_ux = 0.08;
    constexpr double raw_uy = -0.04;
    constexpr double raw_uz = 0.02;
    constexpr double fx = 0.010;
    constexpr double fy = -0.006;
    constexpr double fz = 0.004;
    const std::array<double, 19> equilibrium =
        shifted_equilibrium(rho, raw_ux, raw_uy, raw_uz);
    const double ux = raw_ux+0.5*fx/rho;
    const double uy = raw_uy+0.5*fy/rho;
    const double uz = raw_uz+0.5*fz/rho;
    const double corrected_q = shifted_nonequilibrium_q(
        equilibrium, rho, ux, uy, uz, fx, fy, fz);
    const double force_squared = fx*fx+fy*fy+fz*fz;
    const double expected_q =
        force_squared*force_squared/(16.0*rho*rho);
    test.expect_close(
        corrected_q, expected_q, 2.0E-19,
        "Guo correction removes the leading uniform-force stress");

    const double uncorrected_q = shifted_nonequilibrium_q(
        equilibrium, rho, ux, uy, uz);
    test.expect(
        corrected_q<0.01*uncorrected_q,
        "Guo correction does not interpret uniform acceleration as shear");
}

double subgrid_tau(const double tau0, const double rho,
                   const double q, const double cs_squared) {
    const double prefactor = 18.0*std::sqrt(2.0)*cs_squared;
    return 0.5*(tau0+
        std::sqrt(tau0*tau0+prefactor*std::sqrt(q)/rho));
}

void test_shifted_equilibrium_moment(TestState& test) {
    constexpr double rho = 1.2;
    constexpr double ux = 0.041;
    constexpr double uy = -0.027;
    constexpr double uz = 0.013;
    const std::array<double, 19> equilibrium =
        shifted_equilibrium(rho, ux, uy, uz);
    test.expect_close(
        shifted_nonequilibrium_q(equilibrium, rho, ux, uy, uz),
        0.0, 2.0E-31,
        "shifted D3Q19 equilibrium has zero non-equilibrium stress");

    constexpr double delta = 0.0025;
    std::array<double, 19> perturbed = equilibrium;
    // The opposite xy pair preserves momentum; compensating f0 preserves
    // shifted density.  It creates Hxx=Hyy=Hxy=2*delta and Q=16*delta^2.
    perturbed[7] += delta;
    perturbed[8] += delta;
    perturbed[0] -= 2.0*delta;
    test.expect_close(
        shifted_nonequilibrium_q(perturbed, rho, ux, uy, uz),
        16.0*delta*delta, 2.0E-17,
        "shifted moment contraction preserves the known stress perturbation");
}

void test_smagorinsky_formula(TestState& test) {
    constexpr double tau0 = 0.53;
    constexpr double rho = 1.0;
    constexpr double q = 1.0E-4;
    constexpr double cs_squared = 0.454*0.454;
    const double tau = subgrid_tau(tau0, rho, q, cs_squared);
    const double nu_t = (tau-tau0)/3.0;
    test.expect_close(tau, 0.5536903911078341, 2.0E-15,
                      "Smagorinsky effective tau matches reference vector");
    test.expect_close(nu_t, 0.007896797035944703, 2.0E-15,
                      "Smagorinsky eddy viscosity matches reference vector");

    constexpr double alpha0 = 0.01;
    constexpr double turbulent_prandtl = 0.7;
    const double alpha_eff = alpha0+nu_t/turbulent_prandtl;
    test.expect_close(alpha_eff, 0.021281138622778147, 2.0E-15,
                      "scalar SGS uses nu_t/Pr_t");
    test.expect_close(0.5+3.0*alpha_eff, 0.5638434158683344, 2.0E-15,
                      "D3Q19 scalar relaxation follows cs^2=1/3");
    test.expect_close(0.5+4.0*alpha_eff, 0.5851245544911126, 2.0E-15,
                      "D3Q7 scalar relaxation follows cs^2=1/4");
    test.expect_close(subgrid_tau(tau0, rho, 0.0, cs_squared), tau0,
                      0.0, "zero stress produces zero eddy viscosity");
    test.expect_close(subgrid_tau(tau0, rho, q, 0.0), tau0,
                      0.0, "zero Smagorinsky coefficient disables SGS");
    test.expect(
        subgrid_tau(tau0, rho, 1.0E-2, cs_squared)>tau,
        "effective tau is monotone in stress magnitude");

    constexpr double gea_ck = 0.6;
    constexpr double gea_ce = 1.048;
    const double gea_cs_squared =
        gea_ck*std::sqrt(gea_ck/gea_ce);
    const double gea_cs = std::sqrt(gea_cs_squared);
    const double gea_prefactor =
        18.0*std::sqrt(2.0)*gea_cs_squared;
    test.expect_close(
        gea_cs_squared, 0.453989945074786, 1.0E-15,
        "GEA Ck/Ce mapping produces the documented Cs squared");
    test.expect_close(
        gea_cs, 0.673787759665302, 1.0E-15,
        "GEA Cs squared is converted to the traditional Cs parameter");
    test.expect_close(
        gea_prefactor, 11.556697275104, 2.0E-12,
        "GEA coefficient produces the kernel stress prefactor");
}

} // namespace

int main() {
    TestState test;
    test_shifted_equilibrium_moment(test);
    test_guo_force_correction(test);
    test_smagorinsky_formula(test);
    if(test.failures==0) {
        std::cout << "SGS shifted-moment and closure formula tests passed.\n";
    }
    return test.failures==0 ? 0 : 1;
}
