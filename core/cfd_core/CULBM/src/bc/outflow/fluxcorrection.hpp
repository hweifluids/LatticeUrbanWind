#pragma once

// Mass-flux correction of the boundary velocities: scales the inflow so that
// the net flux through the outer faces vanishes, optionally filling the
// outlet from an inflow estimate.

#include "../../solver/solver.hpp"

namespace culbm::bc {

void apply_flux_correction(culbm::solver::Solver& solver,
                           const std::string& downstream_bc,
                           const std::function<float3(const float3&)>& inlet_eval,
                           bool show_report,
                           double* avg_delta_mps,
                           double* net_before,
                           double* net_after);

// Flux-correction stage of a case (deck flux_correction): balances the
// boundary mass flux with `eval` as the outlet estimate and reports it, then
// runs `after`; reports the stage as skipped when it is off.
void run_case_flux_correction(culbm::solver::Solver& solver, bool enabled, const std::string& downstream_bc,
                              const std::function<float3(const float3&)>& eval,
                              const std::function<void()>& after = std::function<void()>());

} // namespace culbm::bc
