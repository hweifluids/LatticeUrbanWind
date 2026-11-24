#pragma once

// Velocity boundaries of the three deck modes on a solver whose geometry is
// voxelized: normal decks from the SurfData samples (patch-driven 2D fields,
// high-order k-nearest-neighbour or nearest-neighbour interpolation),
// dataset cases from a uniform inflow, profile cases from the tabulated
// profile above the local terrain.

#include "../patch_fields.hpp"
#include "../../cuda_nested_solver.hpp"
#include "../../solver/solver.hpp"

namespace culbm::cases {
struct CaseContext;
}

namespace culbm::bc {

// Normal decks: velocity and temperature boundary cells and the optional
// flux correction. z0_lbmu is the lattice z of the first cell centre.
void build_normal_boundaries(cases::CaseContext& ctx, culbm::solver::Solver& solver, float z0_lbmu);

// Dataset cases.
void initialize_uniform_velocity(culbm::solver::Solver& solver, const float3& u);
void apply_velocity_boundaries(culbm::solver::Solver& solver, const float3& u, const std::string& downstream_bc_runtime,
                               bool downstream_open_face);

// Profile cases: the profile table in lattice units over the terrain
// (DEM, else flat ground at the top of the compact base).
struct ProfileInflow {
    std::vector<float> profile_u_lbmu;
    float3 origin_lbmu = float3(0.0f);
    float flat_ground_lbmu = 0.0f;
    GroundPlane2D profile_ground_plane;
    bool use_profile_dem_ground = false;
    float ground_z_min_lbm = 0.0f;
    float ground_z_max_lbm = 0.0f;
    float solver_top_si = 0.0f;
    float core_top_si = 0.0f;
    float ground_min_si = 0.0f;
    float ground_max_si = 0.0f;
    float profile_table_top_si = 0.0f;
    uint profile_steps = 0u;
    float u_min_prof = 0.0f;
    float u_max_prof = 0.0f;
};
void prepare_profile_inflow(cases::CaseContext& ctx, ProfileInflow& profile);
// Profile speed at lattice height pos_z over a ground at ground_z.
float profile_speed_lbmu(const ProfileInflow& profile, float pos_z, float ground_z);
// Ground of every (x, y) column, terrain clipping, initial velocity,
// boundaries and flux correction of one profile case; with nesting also the
// common-finest terrain raster of the nested hierarchy.
void build_profile_case_boundaries(cases::CaseContext& ctx, const ProfileInflow& profile, culbm::solver::Solver& solver,
                                   float dir_x, float dir_y, const std::string& case_downstream_bc,
                                   bool use_cuda_vertical_nesting,
                                   culbm::NestedCommonFinestTerrain& nested_profile_terrain,
                                   bool& nested_profile_terrain_active);

} // namespace culbm::bc
