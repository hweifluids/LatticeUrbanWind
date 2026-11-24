#pragma once

// Inflow data of a case before the solver exists: the SurfData boundary
// samples (normal decks), the inflow list (dataset decks) or the profile
// table (profile decks), and the reference velocity derived from them.

#include "samples.hpp"

namespace culbm::cases {
struct CaseContext;
}

namespace culbm::bc {

struct InflowData {
    // SurfData_<datetime>.csv in SI units and its columns.
    std::vector<SamplePoint> samples_si;
    bool csv_has_temperature = false;
    ulong csv_rows_with_temperature = 0ul;
    float csv_temp_min_si = k_temperature_ref_kelvin;
    float csv_temp_max_si = k_temperature_ref_kelvin;
    bool csv_has_patch = false;
    ulong csv_rows_with_patch = 0ul;
    bool csv_samples_valid = true;
    // wind_bc/profile.dat (profile decks) and the tabulated profile.
    std::vector<float> profile_z_samples_si;
    std::vector<float> profile_u_samples_si;
    std::vector<float> profile_u_si;
    // The samples in lattice units (normal decks).
    std::vector<SamplePoint> samples;
    bool use_temperature_bc = false;
    float T_bc_min_lbm = 1.0f;
    float T_bc_max_lbm = 1.0f;
};

// Reads the inflow data of the deck mode, adapts the Boussinesq temperature
// reference and scale to the sample temperatures and sets the SI reference
// velocity (ctx.physics.si_ref_u).
void load_inflow(cases::CaseContext& ctx);
// Normal decks: the samples in lattice units and the temperature bounds.
void convert_samples_to_lattice(cases::CaseContext& ctx);

// Dataset case velocity of an inflow speed and a wind direction (degrees,
// the direction the wind comes from).
float3 wind_velocity_lbmu(float inflow_si, float angle_deg, float u_scale);

} // namespace culbm::bc
