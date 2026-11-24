#pragma once

// Damping layers of the outer boundaries: lateral buffer nudging towards the
// inflow state and the top sponge, plus the scalar outlet face. The layer
// controls are globals that the kernel definitions and the nested-grid
// builder read when a solver is constructed.

#include "../../solver/solver.hpp"

extern bool buffer_nudging_active;
extern int buffer_n_cells;
extern int buffer_downstream_face_id; // 1 west, 2 east, 3 south, 4 north, 0 none
extern float buffer_inv_tau_lbmu;
extern int buffer_nudge_vertical;
extern bool top_sponge_active;
extern int sponge_n_cells;
extern float sponge_inv_tau_lbmu;
extern int sponge_ref_mode;

namespace culbm::cases {
struct CaseContext;
struct CaseSettings;
}

namespace culbm::bc {

// Buffer width and rate for the current lattice and outlet.
void update_buffer_nudging(cases::CaseContext& ctx, bool show_info, const std::string& downstream_bc_runtime);
// Sponge cells and rate for the current lattice.
void update_top_sponge(cases::CaseContext& ctx, bool show_info);
// Profile decks report the buffer once; the outlet follows each angle.
void print_profile_buffer_nudging(const cases::CaseContext& ctx);
// Scalar outlet on the downstream face when downstream_open_face is set.
void configure_temperature_outlet(culbm::solver::Solver& solver, const cases::CaseSettings& deck, const std::string& bc);

} // namespace culbm::bc
