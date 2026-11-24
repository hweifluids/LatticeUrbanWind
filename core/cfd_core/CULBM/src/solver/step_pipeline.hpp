#pragma once

// Ordered stages of one lattice time step. The solver builds the stage list
// once from its configuration and runs it every step; a nested-grid scheduler
// runs the same per-level step as its sub-schedule (Solver::step).
//
//   thermal_outlet   zero-gradient scalar outlet from the interior DDFs
//   immersed_forcing SurfaceIBM forcing iterations (when attached)
//   stream_collide   streaming and collision, including the wall model, the
//                    SGS model and the Guo forcing of all body forces
//   population_halo  DDF halo exchange between subdomains
//   device_barrier   wait for the device (one process-local subdomain)
//   advance_clock    step counter of every subdomain
//
// Device statistics, probes and output are scheduled between steps by the
// caller and are not stages.

#include <vector>

namespace culbm::solver {

enum class Stage : unsigned char {
	thermal_outlet,
	immersed_forcing,
	stream_collide,
	population_halo,
	device_barrier,
	advance_clock
};

const char* stage_name(Stage stage);

struct PipelineConfig {
	unsigned local_subdomains = 1u; // subdomains of this process
	bool split = false;             // more than one subdomain globally
	bool immersed_forcing = false;  // SurfaceIBM runtime attached
};

class StepPipeline {
public:
	StepPipeline() = default;
	explicit StepPipeline(const PipelineConfig& config) { build(config); }
	void build(const PipelineConfig& config);
	const std::vector<Stage>& stages() const { return stages_; }

private:
	std::vector<Stage> stages_;
};

} // namespace culbm::solver
