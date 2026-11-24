#include "step_pipeline.hpp"

namespace culbm::solver {

const char* stage_name(const Stage stage) {
	switch(stage) {
	case Stage::thermal_outlet: return "thermal_outlet";
	case Stage::immersed_forcing: return "immersed_forcing";
	case Stage::stream_collide: return "stream_collide";
	case Stage::population_halo: return "population_halo";
	case Stage::device_barrier: return "device_barrier";
	case Stage::advance_clock: return "advance_clock";
	}
	return "unknown";
}

void StepPipeline::build(const PipelineConfig& config) {
	stages_.clear();
	// The outlet reconstructs T from the DDFs of the finished step, and the
	// IBM forcing needs the base velocity before the collision applies it.
	stages_.push_back(Stage::thermal_outlet);
	if(config.immersed_forcing) stages_.push_back(Stage::immersed_forcing);
	stages_.push_back(Stage::stream_collide);
	if(config.split) stages_.push_back(Stage::population_halo);
	// Exchanges already order multi-GPU steps; a lone subdomain is fenced
	// explicitly so the host never runs more than a step ahead.
	if(config.local_subdomains==1u) stages_.push_back(Stage::device_barrier);
	stages_.push_back(Stage::advance_clock);
}

} // namespace culbm::solver
