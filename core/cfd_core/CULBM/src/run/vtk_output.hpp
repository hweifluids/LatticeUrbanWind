#pragma once

// Legacy binary VTK (structured points, big-endian float) files of the case
// outputs that the solver field views do not write: the mean-field file with
// the turbulence fields (TKE, TI, TLS) and the stitched raw fields of a
// nested hierarchy.

#include "../core/core.hpp"

namespace culbm::run {

// Turbulence fields of the mean-field file (deck output_tke_ti_tls).
struct AvgTurbulenceOutput {
    bool tke = true;
    bool ti = true;
    bool tls = true;
};

// Mean velocity, density and temperature (T_avg may be null), the fluid mask
// and the turbulence fields from the Welford second moments M2_*. With
// convert_to_si_units the values, spacing and origin are SI (origin shifted
// by vtk_origin_shift unless origin_override is given); Nz_write > 0 limits
// the output to the lowest Nz_write planes.
void write_avg_vtk(const string& filename, uint Nx, uint Ny, uint Nz,
                   const float* u_avg, const float* rho_avg, const float* T_avg, const uchar* flags,
                   const float* M2_u, const float* M2_v, const float* M2_w,
                   ulong avg_count, bool convert_to_si_units, const AvgTurbulenceOutput& turbulence,
                   bool print_saved_message = true, uint Nz_write = 0u,
                   float spacing_override = 0.0f, const float3* origin_override = nullptr);

// One field of `components` floats per point, value*factor+offset.
void write_nested_raw_vtk(const string& filename, uint Nx, uint Ny, uint Nz,
                          const float* data, uint components,
                          float spacing_si, const float3& origin_si,
                          float factor, float offset = 0.0f);

} // namespace culbm::run
