#pragma once

// Terrain of a case: proj_temp/interpolated_dem.csv (x, y, elevation in SI
// units, relative to the lowest terrain point), used as the profile-mode
// ground and as the ground of the terrain-relative (AGL) physical output.

#include "../bc/patch_fields.hpp"

namespace culbm::cases {

struct CaseContext;

struct DemCsvPoint { float x; float y; float elevation; };

// Points with finite coordinates; the bounds of the returned points.
std::vector<DemCsvPoint> read_interpolated_dem_csv(
    const string& path,
    float* x_min = nullptr, float* x_max = nullptr,
    float* y_min = nullptr, float* y_max = nullptr,
    float* elev_min = nullptr, float* elev_max = nullptr);

struct CaseTerrain {
    std::vector<DemCsvPoint> profile_dem_points_raw;
    float profile_dem_xmin = 0.0f, profile_dem_xmax = 0.0f;
    float profile_dem_ymin = 0.0f, profile_dem_ymax = 0.0f;
    float profile_dem_emin = 0.0f, profile_dem_emax = 0.0f;
    bool profile_dem_loaded = false;
    // Physical output datum: the top of the generated foundation.
    float physical_output_z_zero_si = 0.0f;
    bc::GroundPlane2D physical_output_ground_plane;
    bool physical_output_dem_ready = false;
};

// Reads the DEM (profile decks and AGL output) and maps it onto the STL frame
// for the AGL output; reports the physical outputs.
void load_terrain(CaseContext& ctx);

// Terrain height (output datum) at every (x, y) of an output grid; empty
// without AGL output.
std::vector<float> make_physical_output_ground_grid(const CaseContext& ctx, uint nx, uint ny,
                                                    const float3& global_origin, float spacing_si);

} // namespace culbm::cases
