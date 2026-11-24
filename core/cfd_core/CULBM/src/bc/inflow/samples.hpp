#pragma once

// Boundary samples of a LUW case: SurfData_<datetime>.csv (X,Y,Z,u,v,w with
// optional T and patch columns, SI units) and wind_bc/profile.dat (z,U).

#include "../thermal/temperature_limits.hpp"
#include "../../core/core.hpp"

namespace culbm::bc {

struct SamplePoint {
    float3 p;
    float3 u;
    float T = k_temperature_ref_kelvin;
    int patch = -1; // -1 means unspecified/legacy CSV
};
struct ProfileSample { float z; float u; };

// Missing files yield no samples (with a console message). Invalid
// temperatures clear *samples_valid and yield no samples.
std::vector<SamplePoint> read_samples(const string& csv_path,
                                      bool* has_temperature_column = nullptr,
                                      ulong* rows_with_temperature = nullptr,
                                      float* temperature_min_si = nullptr,
                                      float* temperature_max_si = nullptr,
                                      bool* has_patch_column = nullptr,
                                      ulong* rows_with_patch = nullptr,
                                      bool* samples_valid = nullptr);

// profile.dat: columns z,U in SI units, with header.
std::vector<ProfileSample> read_profile_dat(const string& path);

// Cubic Hermite interpolation of u(z) with finite-difference slopes; the end
// values hold outside the samples.
float interpolate_profile_cubic(const std::vector<float>& z,
                                const std::vector<float>& u,
                                float zq);

} // namespace culbm::bc
