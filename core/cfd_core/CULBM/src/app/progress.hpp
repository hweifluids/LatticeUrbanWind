#pragma once

// Run progress of the CULBM time loop.
//
// The compute thread announces a run (begin), publishes the step counter after
// every step (advance: one relaxed atomic store, no synchronization with the
// devices or with the console) and closes the run (end). The console thread of
// the executable polls the reporter about every 50 ms; it samples the counter,
// derives the throughput from the measured step rate and draws the status line
// or writes the JSON progress records.
//
// Every figure derives from one measured rate, so the columns agree for one
// GPU, several GPUs and nested lattices:
//   steps_per_second  steps of the counter per wall-clock second, measured over
//                     the last >= 1 s and >= 20 steps (at most ~60 s);
//   mlups             cell updates per step x steps_per_second / 1e6;
//   bandwidth_gbs     device bytes per step x steps_per_second / 1e9.
// A uniform lattice updates each of its cells once per step. A nested run
// counts coarse steps; a coarse step updates every level as often as it
// subcycles, and cell updates are summed over the levels.
//
// Output (command line --progress=MODE, else the environment CULBM_PROGRESS):
//   line  a status line rewritten in place four times per second (default,
//         except under the LUW GUI, LUW_PROGRESS_MODE=gui, where it is off)
//   json  one "[[CULBM_PROGRESS]]{...}" record per second instead of the line
//   off   neither
// The run summary row at the end of a run is printed in every mode.

#include "../core/types.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace culbm::app {

inline constexpr std::uint64_t open_ended = std::numeric_limits<std::uint64_t>::max();

enum class ProgressOutput { automatic, line, json, off };

// "line", "json" or "off" (also "none", "0"); false for anything else.
bool parse_progress_output(const std::string& text, ProgressOutput& output);
const char* progress_output_name(ProgressOutput output);

// What one reported step costs.
struct Workload {
	std::uint64_t cell_updates = 0u; // lattice cell updates per step
	std::uint64_t bytes = 0u;        // device memory traffic per step
	unsigned levels = 1u;            // lattices advanced per step
};

// Step rate from (time, step) samples of a monotonic step counter.
class StepRate {
public:
	void clear() { samples_.clear(); }
	void add(double seconds, std::uint64_t step);
	// Steps per second up to the newest sample; 0 while unknown or stalled.
	double rate() const;

private:
	std::deque<std::pair<double, std::uint64_t>> samples_;
};

// ETA model of a LUW case: plain solver steps, then the mean-field window in
// which every stride-th step also takes a statistics sample. The plain step
// time and the extra time of a sample are running means of the measured loop
// iterations, each seeded with its benchmark hint.
class PhaseEta {
public:
	void clear() { *this = PhaseEta(); }
	void plan(std::uint64_t total_steps, std::uint64_t window_start, std::uint64_t window_steps,
		double plain_rate_hint, double sample_seconds_hint, std::uint64_t stride);
	// One iteration of the time loop. `fallback` is the measured step time,
	// used while no plain iteration has been timed.
	void record(bool in_window, double seconds, bool sampled, double fallback);
	bool planned() const { return planned_; }
	bool timed() const { return step_count_>0u; } // a plain step time is known
	double plain_step_seconds(double fallback) const;
	double sample_seconds() const { return sample_count_ ? sample_sum_/(double)sample_count_ : 0.0; }
	double window_step_seconds(double fallback) const { return plain_step_seconds(fallback)+sample_seconds()/(double)stride_; }
	// Seconds from step t to the end; negative while no step time is known.
	double remaining(std::uint64_t t, double fallback) const;
	// Statistics samples still due after step t.
	std::uint64_t samples_after(std::uint64_t t) const;

private:
	bool planned_ = false;
	bool window_ = false;
	std::uint64_t total_ = 0u, start_ = 0u, stride_ = 1u;
	double step_sum_ = 0.0, sample_sum_ = 0.0;
	std::uint64_t step_count_ = 0u, sample_count_ = 0u;
};

// State of a run at one instant.
struct ProgressSnapshot {
	std::uint64_t step = 0u;
	std::uint64_t first = 0u;          // step at 0 %
	std::uint64_t last = open_ended;   // step at 100 %
	double elapsed = 0.0;              // seconds since begin()
	double steps_per_second = 0.0;     // 0 while unknown
	double eta = -1.0;                 // seconds; negative while unknown
	Workload work;
	double mlups() const { return (double)work.cell_updates*steps_per_second*1.0E-6; }
	double bandwidth_gbs() const { return (double)work.bytes*steps_per_second*1.0E-9; }
};

// Console texts of a snapshot.
std::string progress_status_row(const ProgressSnapshot& s);  // "| Progress | ... |"
std::string progress_summary_row(const ProgressSnapshot& s); // "| Run summary | ... |"
std::string progress_record(const ProgressSnapshot& s, const char* event); // JSON line

class RunProgress {
public:
	using ClockFunction = double (*)();

	void select_output(ProgressOutput output) { requested_.store(output); }
	ProgressOutput output() const; // automatic resolved
	void use_clock(ClockFunction clock) { clock_ = clock; } // tests

	// Compute thread. A run covers steps first..last (last = open_ended when
	// unknown); `step` is the current step. begin() replaces a run of another
	// owner, end() only closes the caller's own run.
	void begin(const void* owner, const Workload& work, std::uint64_t step, std::uint64_t first, std::uint64_t last);
	bool runs(const void* owner) const;
	// Moves the 0 % and 100 % steps of the caller's run; false when it has none.
	bool retarget(const void* owner, std::uint64_t first, std::uint64_t last);
	void advance(const std::uint64_t step) noexcept { step_.store(step, std::memory_order_relaxed); }
	void end(const void* owner);
	// Closes the line and reports where the run stopped (interrupt).
	void abandon(const std::string& reason);

	// ETA model of the LUW time loop (see PhaseEta).
	void plan_phases(std::uint64_t total_steps, std::uint64_t window_start, std::uint64_t window_steps,
		double plain_rate_hint, double sample_seconds_hint, std::uint64_t stride);
	void record_iteration(bool in_window, double seconds, bool sampled);
	double plain_steps_per_second() const;  // ETA model, plain steps
	double window_steps_per_second() const; // ETA model, mean-field window

	// Measured figures, as shown on the console.
	double steps_per_second() const;
	double eta_seconds() const; // negative while unknown
	ProgressSnapshot snapshot() const;

	// Console thread.
	void poll();

private:
	double now() const;
	void sample_locked(double now) const;
	double measured_step_seconds_locked() const;
	double eta_locked(double now, std::uint64_t step) const;
	ProgressSnapshot snapshot_locked(double now) const;
	void publish(unsigned session, const std::string& row, const std::string& record);

	std::atomic<ProgressOutput> requested_{ProgressOutput::automatic};
	ClockFunction clock_ = nullptr;
	std::atomic<std::uint64_t> step_{0u};
	std::atomic<unsigned> session_{0u}; // bumped by begin/end; stale drawings are dropped

	mutable std::mutex mutex_; // everything below
	mutable StepRate rate_;
	bool active_ = false;
	const void* owner_ = nullptr;
	Workload work_;
	std::uint64_t first_ = 0u, last_ = open_ended, begin_step_ = 0u;
	double begin_time_ = 0.0, line_time_ = 0.0, record_time_ = 0.0;
	PhaseEta eta_;
};

RunProgress& run_progress();

} // namespace culbm::app
