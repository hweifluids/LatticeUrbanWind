#pragma once

#include "utilities.hpp"
#include <mutex>

class LBM;
struct Info { // contains redundant information for console printing
	LBM* lbm = nullptr;
	double runtime_lbm=0.0, runtime_total=0.0f, runtime_total_last=0.0; // lbm (compute) and total (compute + rendering + data evaluation) runtime
	double runtime_lbm_timestep_last=1.0, runtime_lbm_timestep_smooth=1.0; // for printing simulation info
	ulong runtime_lbm_samples=0ull;
	Clock clock; // for measuring total runtime
	ulong steps=max_ulong, steps_last=0ull; // runtime_total_last and steps_last are there if multiple run() commands are executed consecutively
	bool phase_eta_enabled=false; // enables two-phase ETA model (normal stage + mean-field stage)
	ulong phase_total_steps=0ull, phase_avg_start_t=max_ulong, phase_avg_steps=0ull;
	double phase_normal_step_s=0.0, phase_avg_step_s=0.0; // averaged wall-time per solver step
	double phase_normal_runtime_sum=0.0, phase_avg_runtime_sum=0.0;
	ulong phase_normal_samples=0ull, phase_avg_samples=0ull;
	uint cpu_mem_required=0u, gpu_mem_required=0u; // all in MB
	string collision = "";
	std::mutex allow_printing; // to prevent threading conflicts when continuously printing updates to console
	void clear_two_phase_eta();
	void configure_two_phase_eta(const ulong total_steps, const ulong avg_start_t, const ulong avg_steps, const double normal_steps_per_s_hint, const double avg_steps_per_s_hint);
	void update_two_phase_eta_step(const bool avg_phase, const double step_seconds);
	double steps_per_second() const;
	double normal_steps_per_second() const;
	double avg_steps_per_second() const;
	void append(const ulong steps, const ulong total_steps, const ulong t);
	void update(const double dt);
	double time() const; // returns either elapsed time or remaining time
	void print_logo() const;
	void print_initialize(LBM* lbm); // enables interactive rendering
	void print_update() const;
	void print_finalize(); // disables interactive rendering
};
extern Info info; // declared in info.cpp
