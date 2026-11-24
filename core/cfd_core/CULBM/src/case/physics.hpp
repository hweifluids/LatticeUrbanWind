#pragma once

// Lattice units and transport of a case: the reference velocity, the global
// Units scale, the base kinematic viscosity and thermal diffusivity resolved
// to FP32 relaxation times above 0.5, buoyancy, and the Coriolis rotation.

#include "../core/core.hpp"

// Coriolis rotation in lattice units; the kernel definitions read them.
extern float coriolis_Omegax_lbmu;
extern float coriolis_Omegay_lbmu;
extern float coriolis_Omegaz_lbmu;
extern float coriolis_f_lbmu;

namespace culbm::cases {

struct CaseContext;

// Requested and effective transport of one unit system. A common factor
// >= 1 scales nu and alpha (keeping the base Prandtl number) until both FP32
// relaxation times lie strictly above 0.5.
struct BaseTransportResolution {
    float common_scale = 1.0f;
    float requested_nu_lbm = 0.0f;
    float requested_alpha_lbm = 0.0f;
    float requested_momentum_tau = 0.5f;
    float requested_thermal_tau = 0.5f;
    float effective_nu_si = 0.0f;
    float effective_alpha_si = 0.0f;
    float effective_nu_lbm = 0.0f;
    float effective_alpha_lbm = 0.0f;
    float effective_momentum_tau = 0.5f;
    float effective_thermal_tau = 0.5f;
};

struct CasePhysics {
    float si_ref_u = 10.0f; // largest inflow speed of the case
    float u_scale = 0.0f;   // lattice velocity per m/s
    float z_off = 0.0f;     // lattice height of the base top plus 0.5
    float si_beta_air = 0.0f; // volumetric thermal expansion [1/K]
    static constexpr float si_gravity = 9.80665f; // standard gravitational acceleration [m/s^2]
    float lbm_nu = 0.0f;
    float lbm_alpha = 0.0f;
    BaseTransportResolution base_transport;
    float lbm_beta = 0.0f;
    float lbm_buoyancy_gz = 0.0f;
};

// Units from the lattice width and the reference velocity and temperature,
// then u_scale, z_off and the expansion coefficient.
void set_initial_units(CaseContext& ctx);
// Effective base transport for the current Units scale (lbm_nu, lbm_alpha);
// reports the resolution (and the wall-model constants) when show_info.
void resolve_base_transport(CaseContext& ctx, const std::string& scope, bool show_info);
void print_base_transport(const CaseContext& ctx, const std::string& scope,
                          const BaseTransportResolution& resolved);
// Lattice buoyancy coefficient and gravity from the current Units scale.
void set_buoyancy_coefficients(CaseContext& ctx);
// Coriolis rotation of the lon/lat window centre, in lattice units.
void update_coriolis(CaseContext& ctx, bool show_info);

} // namespace culbm::cases
