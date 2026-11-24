#pragma once

// Temperature side of the boundaries: the Boussinesq reference adapted to the
// sample temperatures, the buoyancy coupling decision, and the temperature
// boundary cells of normal decks (outer faces from the SurfData samples,
// solid ground from the patch-0 plane).

#include "coupling_policy.hpp"
#include "temperature_limits.hpp"
#include "../patch_fields.hpp"
#include "../../solver/solver.hpp"

namespace culbm::cases {
struct CaseContext;
}

namespace culbm::bc {

struct ThermalReference {
    float temperature_ref_kelvin = k_temperature_ref_kelvin;
    float temperature_scale_kelvin = k_temperature_ref_kelvin;
    bool temperature_ref_adaptive = false;
    bool temperature_scale_adaptive = false;
    bool boussinesq_temperature_range_checked = false;
    float boussinesq_max_relative_temperature_difference = 0.0f;
    bool thermal_boundary_available = false;
    luw::cfd::ThermalCouplingDecision thermal_coupling;
};

// Buoyancy is active only with a temperature boundary (deck.buoyancy_enabled
// becomes the active coupling).
void resolve_thermal_coupling_stage(cases::CaseContext& ctx);

// Temperature boundary cells of a normal deck.
class NormalTemperatureBoundary {
public:
    // Patch temperature fields and the ground plane from the lattice samples.
    NormalTemperatureBoundary(const cases::CaseContext& ctx, culbm::solver::Solver& solver, bool use_patch_face_bc);

    // Outer faces from the patch fields (patch-driven decks).
    void map_patch_faces(int downstream_patch);
    // Outer faces by k-nearest-neighbour interpolation (high_order decks).
    void map_high_order(std::vector<float3>&& P_temp, std::vector<float3>&& Tv, float z_base_threshold_lbmu);
    // Outer faces by nearest-neighbour interpolation.
    void map_low_order(std::vector<float3>&& P_temp, std::vector<float3>&& Tv, float z0_lbmu, float z_off);
    // Solid cells take the patch-0 ground temperature of their column.
    void apply_ground_temperature_plane_bc(const string& order_tag);
    void report_temperature_bc_summary(const string& tag);

private:
    const cases::CaseContext& ctx_;
    culbm::solver::Solver& solver;
    const bool use_temperature_bc;
    const float T_bc_min_lbm;
    const float T_bc_max_lbm;
    const std::string& downstream_bc;
    const bool downstream_open_face;
    std::vector<PatchSurfaceField2D> patch_temperature_fields;
    GroundPlane2D ground_temp_plane;
    bool use_ground_temperature_bc = false;
};

} // namespace culbm::bc
