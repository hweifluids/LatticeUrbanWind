#include "physics.hpp"
#include "context.hpp"
#include "report.hpp"

#include <limits>

// for Coriolis source term
float coriolis_Omegax_lbmu = 0.0f;
float coriolis_Omegay_lbmu = 0.0f;
float coriolis_Omegaz_lbmu = 0.0f;

namespace culbm::cases {

namespace {

// 1: both FP32 relaxation times above 0.5, 0: not, -1: not representable.
int evaluate_base_transport(const CaseContext& ctx, const float common_scale,
                            BaseTransportResolution& resolved) {
    const CaseSettings& deck = ctx.deck;
    if (!(common_scale >= 1.0f) || !std::isfinite(common_scale)) {
        return -1;
    }
    const double effective_nu_si =
        deck.culbm_base_kinematic_viscosity_m2_s * (double)common_scale;
    const double effective_alpha_si =
        deck.culbm_base_thermal_diffusivity_m2_s * (double)common_scale;
    if (!(effective_nu_si > 0.0) || !(effective_alpha_si > 0.0) ||
        !std::isfinite(effective_nu_si) || !std::isfinite(effective_alpha_si) ||
        effective_nu_si > (double)std::numeric_limits<float>::max() ||
        effective_alpha_si > (double)std::numeric_limits<float>::max()) {
        return -1;
    }

    resolved.common_scale = common_scale;
    resolved.requested_nu_lbm =
        units.nu((float)deck.culbm_base_kinematic_viscosity_m2_s);
    resolved.requested_alpha_lbm =
        units.alpha((float)deck.culbm_base_thermal_diffusivity_m2_s);
    resolved.requested_momentum_tau =
        0.5f + 3.0f * resolved.requested_nu_lbm;
    resolved.requested_thermal_tau =
        get_temperature_tau(resolved.requested_alpha_lbm);
    resolved.effective_nu_si = (float)effective_nu_si;
    resolved.effective_alpha_si = (float)effective_alpha_si;
    resolved.effective_nu_lbm = units.nu(resolved.effective_nu_si);
    resolved.effective_alpha_lbm = units.alpha(resolved.effective_alpha_si);
    resolved.effective_momentum_tau =
        0.5f + 3.0f * resolved.effective_nu_lbm;
    resolved.effective_thermal_tau =
        get_temperature_tau(resolved.effective_alpha_lbm);

    if (!std::isfinite(resolved.requested_nu_lbm) ||
        !std::isfinite(resolved.requested_alpha_lbm) ||
        !std::isfinite(resolved.requested_momentum_tau) ||
        !std::isfinite(resolved.requested_thermal_tau) ||
        !std::isfinite(resolved.effective_nu_lbm) ||
        !std::isfinite(resolved.effective_alpha_lbm) ||
        !std::isfinite(resolved.effective_momentum_tau) ||
        !std::isfinite(resolved.effective_thermal_tau)) {
        return -1;
    }
    return resolved.effective_momentum_tau > 0.5f &&
           resolved.effective_thermal_tau > 0.5f ? 1 : 0;
}

[[noreturn]] void fail_base_transport(const CaseContext& ctx, const std::string& scope) {
    const CaseSettings& deck = ctx.deck;
    println("| ERROR: cannot represent positive base momentum/thermal relaxation in FP32. |");
    print_kv_row("Transport units", scope);
    print_kv_row("Requested SI", "nu=" +
        to_string(deck.culbm_base_kinematic_viscosity_m2_s) +
        " m^2/s, alpha=" +
        to_string(deck.culbm_base_thermal_diffusivity_m2_s) + " m^2/s");
    wait();
    exit(-1);
}

} // namespace

void print_base_transport(const CaseContext& ctx, const std::string& scope,
                          const BaseTransportResolution& resolved) {
    const CaseSettings& deck = ctx.deck;
    print_kv_row("Transport units", scope);
    print_kv_row("Requested SI", "nu=" +
        to_string(deck.culbm_base_kinematic_viscosity_m2_s) +
        " m^2/s, alpha=" +
        to_string(deck.culbm_base_thermal_diffusivity_m2_s) + " m^2/s");
    print_kv_row("Requested LBM", "nu=" +
        to_string(resolved.requested_nu_lbm) + ", alpha=" +
        to_string(resolved.requested_alpha_lbm));
    print_kv_row("Requested tau", "momentum=" +
        to_string(resolved.requested_momentum_tau) + ", thermal=" +
        to_string(resolved.requested_thermal_tau));
    print_kv_row("Transport scale", to_string(resolved.common_scale) +
        " (common nu/alpha factor; base Pr preserved)");
    print_kv_row("Effective SI", "nu=" +
        to_string(resolved.effective_nu_si) + " m^2/s, alpha=" +
        to_string(resolved.effective_alpha_si) + " m^2/s");
    print_kv_row("Effective LBM", "nu=" +
        to_string(resolved.effective_nu_lbm) + ", alpha=" +
        to_string(resolved.effective_alpha_lbm));
    print_kv_row("Effective tau", "momentum=" +
        to_string(resolved.effective_momentum_tau) + ", thermal=" +
        to_string(resolved.effective_thermal_tau));
    if (resolved.common_scale > 1.0f) {
        print_kv_row("WARNING",
            "requested FP32 tau <= 0.5; effective base transport was increased");
    }
}

void set_initial_units(CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const bc::ThermalReference& thermal = ctx.thermal;
    CasePhysics& physics = ctx.physics;
    units.set_m_kg_s_K((float)domain.lbm_N.y, domain.lbm_ref_u, 1.0f, 1.0f, deck.si_size.y, physics.si_ref_u, domain.si_rho, thermal.temperature_scale_kelvin);
    units.set_temperature_reference(1.0f, thermal.temperature_ref_kelvin); // T_lbm=1.0 maps to the selected auto/explicit T_ref

    physics.u_scale = domain.lbm_ref_u / physics.si_ref_u;
    physics.z_off = units.x(deck.z_si_offset) + 0.5f;

    physics.si_beta_air = 1.0f / thermal.temperature_ref_kelvin; // volumetric thermal expansion [1/K]
    physics.lbm_nu = 0.0f;
    physics.lbm_alpha = 0.0f;
}

void resolve_base_transport(CaseContext& ctx, const std::string& scope, const bool show_info) {
    const CaseSettings& deck = ctx.deck;
    CasePhysics& physics = ctx.physics;
    BaseTransportResolution resolved;
    int status = evaluate_base_transport(ctx, 1.0f, resolved);
    if (status < 0) {
        fail_base_transport(ctx, scope);
    }
    if (status == 0) {
        float lower = 1.0f;
        float upper = 1.0f;
        for (;;) {
            if (upper > 0.5f * std::numeric_limits<float>::max()) {
                fail_base_transport(ctx, scope);
            }
            upper *= 2.0f;
            BaseTransportResolution candidate;
            status = evaluate_base_transport(ctx, upper, candidate);
            if (status < 0) {
                fail_base_transport(ctx, scope);
            }
            if (status > 0) {
                resolved = candidate;
                break;
            }
            lower = upper;
        }

        // The predicate is monotonic for positive coefficients.  Search adjacent
        // FP32 scale values so the selected common factor is the smallest one
        // that produces both actual FP32 relaxation times strictly above 0.5.
        const float infinity = std::numeric_limits<float>::infinity();
        for (uint iteration = 0u; iteration < 512u; ++iteration) {
            const float adjacent = std::nextafter(lower, infinity);
            if (!(adjacent < upper)) {
                break;
            }
            const float midpoint = 0.5f * lower + 0.5f * upper;
            if (!(midpoint > lower) || !(midpoint < upper)) {
                break;
            }
            BaseTransportResolution candidate;
            status = evaluate_base_transport(ctx, midpoint, candidate);
            if (status < 0) {
                fail_base_transport(ctx, scope);
            }
            if (status > 0) {
                upper = midpoint;
                resolved = candidate;
            }
            else {
                lower = midpoint;
            }
        }
        status = evaluate_base_transport(ctx, upper, resolved);
        if (status != 1) {
            fail_base_transport(ctx, scope);
        }
    }

    physics.base_transport = resolved;
    physics.lbm_nu = resolved.effective_nu_lbm;
    physics.lbm_alpha = resolved.effective_alpha_lbm;
    if (show_info) {
        print_base_transport(ctx, scope, resolved);
        if (deck.wall_model_settings.enabled) {
            const Wall_Model_Lattice wm = resolve_wall_model_lattice(
                get_wall_model_configuration(), units.si_x(1.0f), units.si_t(1ull),
                deck.surface_ibm_settings.enabled);
            const float d_m = 0.5f * wm.cell_size_m;
            auto cd_text = [&](const bool smooth, const float cd) {
                return smooth ? string("smooth log law")
                              : "Cd=" + to_string(fmtf(cd, 6)) +
                                ", u*/U1=" + to_string(fmtf(std::sqrt(cd), 5));
            };
            print_kv_row("WM first node", "d=" + to_string(fmtf(d_m, 4)) +
                " m (halfway bounce-back)");
            print_kv_row("WM ground", cd_text(wm.smooth_h, wm.cd_h));
            print_kv_row("WM building", cd_text(wm.smooth_v, wm.cd_v));
            const float z0_max = std::max(deck.wall_model_settings.z0_ground_m,
                                          deck.wall_model_settings.z0_building_m);
            if (z0_max > 0.2f * d_m) {
                print_kv_row("WARNING", "z0 > 0.2*d: first node lies in the roughness sublayer");
            }
        }
    }
}

void set_buoyancy_coefficients(CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    CasePhysics& physics = ctx.physics;
    physics.lbm_beta = deck.buoyancy_enabled ? units.beta(physics.si_beta_air) : 0.0f;
    physics.lbm_buoyancy_gz = deck.buoyancy_enabled ? units.g(-physics.si_gravity) : 0.0f;
}

void update_coriolis(CaseContext& ctx, const bool show_info) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const CasePhysics& physics = ctx.physics;
    if (deck.enable_coriolis) {
        // 1. center in degrees
        const float center_lon_deg = 0.5f * (deck.cut_lon_min_deg + deck.cut_lon_max_deg);
        const float center_lat_deg = 0.5f * (deck.cut_lat_min_deg + deck.cut_lat_max_deg);

        // 2. Earth rotation in SI
        const float Omega_earth_si = 7.292115e-5f; // [1/s]
        const float deg2rad = 3.14159265358979323846f / 180.0f;
        const float lat_rad = center_lat_deg * deg2rad;

        // local ENU frame: x east, y north, z up
        const float Omegax_si = 0.0f;
        const float Omegay_si = Omega_earth_si * cosf(lat_rad);
        const float Omegaz_si = Omega_earth_si * sinf(lat_rad);

        // 3. convert to lattice units
        // dx_SI is your cell size in meters, dt_SI = dx_SI * (u_lbm / u_SI)
        const float dx_si = deck.cell_m;
        const float dt_si = dx_si * (domain.lbm_ref_u / physics.si_ref_u);

        coriolis_Omegax_lbmu = Omegax_si * dt_si;
        coriolis_Omegay_lbmu = Omegay_si * dt_si;
        coriolis_Omegaz_lbmu = Omegaz_si * dt_si;

        if (show_info) {
            print_kv_row("Coriolis", "enabled. center(lon,lat)=(" + to_string(center_lon_deg, 6u) + ", " + to_string(center_lat_deg, 6u) + ") deg");
            print_kv_row("", "Omega(lbmu)=(" + to_string(coriolis_Omegax_lbmu, 8u) + ", " + to_string(coriolis_Omegay_lbmu, 8u) + ", " + to_string(coriolis_Omegaz_lbmu, 8u) + ") per step");
        }

        const float omega_si = 7.292e-5f;           // Earth rotation [1/s]
        float f_si = 0.0f;
        if (deck.cut_lat_min_deg != 0.0f || deck.cut_lat_max_deg != 0.0f) {
            f_si = 2.0f * omega_si * std::sin(center_lat_deg * deg2rad);
        }
        coriolis_f_lbmu = f_si * dt_si;
        if (show_info) {
            print_kv_row("", "f_SI=" + to_string(f_si, 8u) + " 1/s, f_LBM=" + to_string(coriolis_f_lbmu, 8u));
        }
    }
    else if (show_info) {
        print_kv_row("Coriolis", "disabled by 'coriolis_term' setting in .luw");
    }
}

} // namespace culbm::cases
