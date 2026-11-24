#include "features.hpp"
#include "settings.hpp"

namespace culbm::cases {

void apply_lattice_selection(CaseSettings& deck) {
    if (!set_momentum_lattice(deck.momentum_lattice_setting)) {
        println("| ERROR: culbm_velocity_set cannot change after a solver instance exists.      |");
        wait();
        exit(-1);
    }
    if (!set_ddf_storage(deck.ddf_storage_setting)) {
        println("| ERROR: culbm_ddf_storage cannot change after a solver instance exists.       |");
        wait();
        exit(-1);
    }
    if (!set_temperature_lattice(deck.temperature_lattice_setting)) {
        println("| ERROR: requested temperature lattice is unsupported by this momentum build. |");
        wait();
        exit(-1);
    }
    deck.temperature_lattice_applied = true;
}

void apply_model_selection(CaseSettings& deck) {
    Subgrid_Configuration subgrid_configuration;
    subgrid_configuration.enabled = deck.culbm_sgs_enabled;
    subgrid_configuration.smagorinsky_constant = (float)deck.culbm_sgs_cs;
    subgrid_configuration.scalar_coupling = deck.culbm_sgs_scalar_coupling;
    subgrid_configuration.turbulent_prandtl =
        (float)deck.culbm_sgs_turbulent_prandtl;
    subgrid_configuration.fixed_eddy_viscosity = -1.0f;
    if (!set_subgrid_configuration(subgrid_configuration)) {
        println("| ERROR: SGS configuration is invalid or was changed after constructing an LBM instance. |");
        wait();
        exit(-1);
    }
    // The smooth-wall law uses the requested molecular viscosity, never the
    // FP32 tau floor amplification applied to the collision viscosity.
    deck.wall_model_settings.molecular_nu_m2_s = (float)deck.culbm_base_kinematic_viscosity_m2_s;
    if (!set_wall_model_configuration(deck.wall_model_settings)) {
        println("| ERROR: wall model configuration is invalid or was changed after constructing an LBM instance. |");
        wait();
        exit(-1);
    }
}

} // namespace culbm::cases
