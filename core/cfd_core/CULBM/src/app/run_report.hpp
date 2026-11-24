#pragma once

// Console reports of a CULBM run: the banner at start-up, the run report
// printed once a solver is initialized (lattice, decomposition, GPUs, memory,
// momentum lattice and DDF storage, enabled features, transport coefficients),
// and the workload a solver or a nested hierarchy advances per reported step,
// which is the basis of the MLUPS and GB/s figures of the run progress.

#include "progress.hpp"

#include <cstdint>
#include <vector>

namespace culbm::solver {
class Solver;
}
namespace culbm {
struct NestedLbmLevel;
}

namespace culbm::app {

void print_banner();
// `last_step` is the step the run ends at (open_ended when unknown).
void print_run_report(const culbm::solver::Solver& solver, std::uint64_t last_step);

// One step of the lattice: every cell once.
Workload workload_of(const culbm::solver::Solver& solver);
// One coarse step of a nested hierarchy: every level as often as it subcycles.
Workload workload_of(const std::vector<culbm::NestedLbmLevel>& levels);

// Advances `solver` by `steps` time steps with run progress. The first call
// initializes the solver, prints the run report and starts the progress over
// steps 0..total_steps (or over this call's steps when total_steps is
// open_ended); the progress closes when total_steps is reached.
void run_steps(culbm::solver::Solver& solver, std::uint64_t steps = open_ended,
	std::uint64_t total_steps = open_ended, bool synchronize_after = true);

// Prints the workload of a nested run and starts its progress (coarse steps
// `step` to `last_step`).
void begin_nested_progress(const void* owner, const std::vector<culbm::NestedLbmLevel>& levels,
	std::uint64_t step, std::uint64_t last_step);

} // namespace culbm::app
