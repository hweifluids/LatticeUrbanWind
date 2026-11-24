#include "progress.hpp"

#include "../core/console.hpp"
#include "../core/progress.hpp"
#include "../core/status_line.hpp"
#include "../core/text.hpp"
#include "../cuda_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace culbm::app {

namespace {

constexpr double line_interval = 0.25;   // s between status line drawings
constexpr double record_interval = 1.0;  // s between JSON records
constexpr double rate_span = 1.0;        // s, shortest rate window ...
constexpr std::uint64_t rate_steps = 20u; // ... that also holds this many steps
constexpr double rate_history = 60.0;    // s of samples kept
constexpr double sample_spacing = 0.02;  // s; denser samples replace the newest

double steady_seconds() {
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string fixed(const double value, const int decimals) {
	char text[64];
	std::snprintf(text, sizeof(text), "%.*f", decimals, value);
	return text;
}

// Three significant digits below 100, whole numbers above.
std::string rate_text(const double value) {
	return value>=99.95 ? fixed(value, 0) : value>=9.995 ? fixed(value, 1) : fixed(value, 2);
}

std::string seconds_text(const double seconds) {
	return seconds<59.95 ? fixed(seconds, 1)+" s" : core::print_time(seconds);
}

std::string json_number(const double value, const int decimals) {
	return std::isfinite(value) ? fixed(value, decimals) : std::string("null");
}

bool known(const std::uint64_t last) { return last!=open_ended; }

} // namespace

bool parse_progress_output(const std::string& text, ProgressOutput& output) {
	if(text=="line") output = ProgressOutput::line;
	else if(text=="json") output = ProgressOutput::json;
	else if(text=="off"||text=="none"||text=="0") output = ProgressOutput::off;
	else return false;
	return true;
}

const char* progress_output_name(const ProgressOutput output) {
	switch(output) {
	case ProgressOutput::line: return "line";
	case ProgressOutput::json: return "json";
	case ProgressOutput::off: return "off";
	default: return "automatic";
	}
}

// ---- StepRate ------------------------------------------------------------------------

void StepRate::add(const double seconds, const std::uint64_t step) {
	// Samples stay at least sample_spacing apart: a denser one replaces the
	// newest, so the newest is always current.
	const size_t n = samples_.size();
	if(n>=2u && seconds-samples_[n-2u].first<sample_spacing) {
		samples_.back() = {seconds, step};
	} else {
		samples_.emplace_back(seconds, step);
	}
	// Keep one sample older than the history so a slow run still has a window.
	while(samples_.size()>2u && seconds-samples_[1].first>=rate_history) samples_.pop_front();
}

double StepRate::rate() const {
	if(samples_.size()<2u) return 0.0;
	const auto& newest = samples_.back();
	const std::pair<double, std::uint64_t>* reference = &samples_.front();
	for(size_t i=samples_.size()-1u; i-->0u; ) {
		const auto& s = samples_[i];
		if(newest.first-s.first>=rate_span && newest.second>=s.second+rate_steps) {
			reference = &s;
			break;
		}
	}
	const double span = newest.first-reference->first;
	if(span<line_interval || newest.second<=reference->second) return 0.0;
	return (double)(newest.second-reference->second)/span;
}

// ---- PhaseEta ------------------------------------------------------------------------

void PhaseEta::plan(const std::uint64_t total_steps, const std::uint64_t window_start, const std::uint64_t window_steps,
	const double plain_rate_hint, const double sample_seconds_hint, const std::uint64_t stride) {
	clear();
	planned_ = total_steps!=open_ended;
	total_ = total_steps;
	window_ = window_steps>0u;
	start_ = window_start;
	stride_ = std::max<std::uint64_t>(stride, 1u);
	if(plain_rate_hint>0.0) {
		step_sum_ = 1.0/plain_rate_hint;
		step_count_ = 1u;
	}
	if(sample_seconds_hint>0.0) {
		sample_sum_ = sample_seconds_hint;
		sample_count_ = 1u;
	}
}

void PhaseEta::record(const bool in_window, const double seconds, const bool sampled, const double fallback) {
	if(!planned_ || !(seconds>0.0)) return;
	if(in_window && sampled) {
		sample_sum_ += std::max(seconds-plain_step_seconds(fallback), 0.0);
		++sample_count_;
	} else {
		step_sum_ += seconds;
		++step_count_;
	}
}

double PhaseEta::plain_step_seconds(const double fallback) const {
	return step_count_ ? step_sum_/(double)step_count_ : fallback;
}

std::uint64_t PhaseEta::samples_after(const std::uint64_t t) const {
	if(!window_ || t>=total_) return 0u;
	std::uint64_t next = std::max(t+1u, start_);
	const std::uint64_t offset = (next-start_)%stride_;
	if(offset!=0u) next += stride_-offset;
	return next>total_ ? 0u : 1u+(total_-next)/stride_;
}

double PhaseEta::remaining(const std::uint64_t t, const double fallback) const {
	if(!planned_) return -1.0;
	const double step = plain_step_seconds(fallback);
	if(!(step>0.0)) return -1.0;
	const std::uint64_t now = std::min(t, total_);
	return (double)(total_-now)*step+(double)samples_after(now)*sample_seconds();
}

// ---- texts ---------------------------------------------------------------------------

std::string progress_status_row(const ProgressSnapshot& s) {
	std::string value;
	if(known(s.last)) {
		const double span = s.last>s.first ? (double)(s.last-s.first) : 1.0;
		const double done = s.step>s.first ? std::min((double)(s.step-s.first)/span, 1.0) : 0.0;
		value = core::to_string(s.step)+"/"+core::to_string(s.last)+" ("+core::to_string((unsigned)(100.0*done))+"%)";
	} else {
		value = "step "+core::to_string(s.step);
	}
	if(s.steps_per_second>0.0) {
		value += "  "+rate_text(s.steps_per_second)+" steps/s  "+fixed(s.mlups(), 0)+" MLUPS  "+
			fixed(s.bandwidth_gbs(), s.bandwidth_gbs()<9.95 ? 1 : 0)+" GB/s";
	} else {
		value += "  -- steps/s  -- MLUPS  -- GB/s";
	}
	if(!known(s.last)) value += "  elapsed "+core::print_time(s.elapsed);
	else value += s.eta>=0.0 ? "  ETA "+core::print_time(s.eta) : std::string("  ETA --");
	return std::string(s.work.levels>1u ? "| Nested progress | " : "| Progress | ")+value+" |";
}

std::string progress_summary_row(const ProgressSnapshot& s) {
	const std::uint64_t steps = s.step>s.first ? s.step-s.first : 0u;
	std::string value = core::to_string(steps)+(s.work.levels>1u ? " coarse steps in " : " steps in ")+seconds_text(s.elapsed);
	if(s.steps_per_second>0.0) {
		value += ": "+rate_text(s.steps_per_second)+" steps/s, "+fixed(s.mlups(), 0)+" MLUPS, "+
			fixed(s.bandwidth_gbs(), s.bandwidth_gbs()<9.95 ? 1 : 0)+" GB/s (mean)";
	}
	return std::string(s.work.levels>1u ? "| Nested summary | " : "| Run summary | ")+value+" |";
}

std::string progress_record(const ProgressSnapshot& s, const char* event) {
	const bool rate = s.steps_per_second>0.0;
	std::string r = "[[CULBM_PROGRESS]]{\"event\":\""+std::string(event)+"\"";
	r += ",\"step\":"+core::to_string(s.step);
	r += ",\"total_steps\":"+(known(s.last) ? core::to_string(s.last) : std::string("null"));
	r += ",\"elapsed_s\":"+json_number(s.elapsed, 3);
	r += ",\"steps_per_second\":"+json_number(s.steps_per_second, 3);
	r += ",\"mlups\":"+(rate ? json_number(s.mlups(), 1) : std::string("0.0"));
	r += ",\"bandwidth_gbs\":"+(rate ? json_number(s.bandwidth_gbs(), 2) : std::string("0.00"));
	r += ",\"eta_s\":"+(s.eta>=0.0 ? json_number(s.eta, 1) : std::string("null"));
	r += ",\"cells\":"+core::to_string(s.work.cell_updates);
	r += ",\"levels\":"+core::to_string(s.work.levels)+"}";
	return r;
}

// ---- RunProgress ---------------------------------------------------------------------

RunProgress& run_progress() {
	static RunProgress progress;
	return progress;
}

ProgressOutput RunProgress::output() const {
	const ProgressOutput requested = requested_.load();
	if(requested!=ProgressOutput::automatic) return requested;
	return core::luw_progress_gui_mode() ? ProgressOutput::off : ProgressOutput::line;
}

double RunProgress::now() const {
	return clock_ ? clock_() : steady_seconds();
}

void RunProgress::sample_locked(const double now) const {
	rate_.add(now, step_.load(std::memory_order_relaxed));
}

double RunProgress::measured_step_seconds_locked() const {
	const double rate = rate_.rate();
	return rate>0.0 ? 1.0/rate : 0.0;
}

double RunProgress::eta_locked(const double now, const std::uint64_t step) const {
	if(!known(last_)) return -1.0;
	const std::uint64_t t = std::min(step, last_);
	double measured = measured_step_seconds_locked();
	if(eta_.planned()) {
		const double model = eta_.remaining(t, measured);
		if(model>=0.0) return model;
	}
	if(t==last_) return 0.0;
	if(!(measured>0.0) && step>begin_step_ && now>begin_time_) measured = (now-begin_time_)/(double)(step-begin_step_);
	return measured>0.0 ? (double)(last_-t)*measured : -1.0;
}

ProgressSnapshot RunProgress::snapshot_locked(const double now) const {
	ProgressSnapshot s;
	s.step = step_.load(std::memory_order_relaxed);
	s.first = first_;
	s.last = last_;
	s.elapsed = std::max(now-begin_time_, 0.0);
	s.steps_per_second = rate_.rate();
	s.eta = eta_locked(now, s.step);
	s.work = work_;
	return s;
}

void RunProgress::begin(const void* owner, const Workload& work, const std::uint64_t step, const std::uint64_t first,
	const std::uint64_t last) {
	const void* previous = nullptr;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if(active_ && owner_!=owner) previous = owner_;
	}
	if(previous!=nullptr) end(previous);
	const double t = now();
	std::lock_guard<std::mutex> lock(mutex_);
	owner_ = owner;
	work_ = work;
	first_ = first;
	last_ = last;
	begin_step_ = step;
	begin_time_ = t;
	line_time_ = t;
	record_time_ = t;
	step_.store(step, std::memory_order_relaxed);
	rate_.clear();
	rate_.add(t, step);
	eta_.clear();
	active_ = true;
	session_.fetch_add(1u);
}

bool RunProgress::runs(const void* owner) const {
	std::lock_guard<std::mutex> lock(mutex_);
	return active_ && owner_==owner;
}

bool RunProgress::retarget(const void* owner, const std::uint64_t first, const std::uint64_t last) {
	std::lock_guard<std::mutex> lock(mutex_);
	if(!active_ || owner_!=owner) return false;
	first_ = first;
	last_ = last;
	return true;
}

void RunProgress::end(const void* owner) {
	std::string row, record, summary;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if(!active_ || owner_!=owner) return;
		const double t = now();
		sample_locked(t);
		ProgressSnapshot s = snapshot_locked(t);
		row = progress_status_row(s);
		// The summary averages over the whole run.
		const std::uint64_t steps = s.step>begin_step_ ? s.step-begin_step_ : 0u;
		s.first = begin_step_;
		s.steps_per_second = steps>0u && s.elapsed>0.0 ? (double)steps/s.elapsed : 0.0;
		if(known(s.last) && s.step>=s.last) s.eta = 0.0;
		summary = progress_summary_row(s);
		if(output()==ProgressOutput::json) record = progress_record(s, "end");
		active_ = false;
		session_.fetch_add(1u);
	}
	std::lock_guard<std::mutex> hold(core::status_line().gate());
	core::StatusLine& line = core::status_line();
	if(line.shown_locked()) line.draw_locked(row); // final state, then a fresh line
	line.close_locked();
	if(!record.empty() && culbm::mpi_is_root()) {
		std::lock_guard<std::mutex> lock(core::console_io_mutex());
		std::cout << record << '\n';
		std::cout.flush();
	}
	core::println(summary);
}

void RunProgress::abandon(const std::string& reason) {
	std::string where;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if(active_) {
			const std::uint64_t t = step_.load(std::memory_order_relaxed);
			where = " at step "+core::to_string(t)+(known(last_) ? " of "+core::to_string(last_) : std::string());
		}
		active_ = false;
		session_.fetch_add(1u);
	}
	std::lock_guard<std::mutex> hold(core::status_line().gate());
	core::status_line().close_locked();
	core::println("| Stopped | "+reason+where+" |");
}

void RunProgress::plan_phases(const std::uint64_t total_steps, const std::uint64_t window_start, const std::uint64_t window_steps,
	const double plain_rate_hint, const double sample_seconds_hint, const std::uint64_t stride) {
	std::lock_guard<std::mutex> lock(mutex_);
	eta_.plan(total_steps, window_start, window_steps, plain_rate_hint, sample_seconds_hint, stride);
}

void RunProgress::record_iteration(const bool in_window, const double seconds, const bool sampled) {
	std::lock_guard<std::mutex> lock(mutex_);
	if(!eta_.planned()) return;
	double fallback = 0.0; // only needed until a plain step was timed
	if(!eta_.timed()) {
		sample_locked(now());
		fallback = measured_step_seconds_locked();
	}
	eta_.record(in_window, seconds, sampled, fallback);
}

double RunProgress::plain_steps_per_second() const {
	std::lock_guard<std::mutex> lock(mutex_);
	sample_locked(now());
	const double step = eta_.plain_step_seconds(measured_step_seconds_locked());
	return step>0.0 ? 1.0/step : 0.0;
}

double RunProgress::window_steps_per_second() const {
	std::lock_guard<std::mutex> lock(mutex_);
	sample_locked(now());
	const double step = eta_.window_step_seconds(measured_step_seconds_locked());
	return step>0.0 ? 1.0/step : 0.0;
}

double RunProgress::steps_per_second() const {
	std::lock_guard<std::mutex> lock(mutex_);
	sample_locked(now());
	return rate_.rate();
}

double RunProgress::eta_seconds() const {
	std::lock_guard<std::mutex> lock(mutex_);
	const double t = now();
	sample_locked(t);
	return eta_locked(t, step_.load(std::memory_order_relaxed));
}

ProgressSnapshot RunProgress::snapshot() const {
	std::lock_guard<std::mutex> lock(mutex_);
	const double t = now();
	sample_locked(t);
	return snapshot_locked(t);
}

void RunProgress::poll() {
	const ProgressOutput out = output();
	std::string row, record;
	unsigned session = 0u;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if(!active_) return;
		const double t = now();
		sample_locked(t);
		if(out==ProgressOutput::line && t-line_time_>=line_interval) {
			row = progress_status_row(snapshot_locked(t));
			line_time_ = t;
		}
		if(out==ProgressOutput::json && t-record_time_>=record_interval) {
			record = progress_record(snapshot_locked(t), "progress");
			record_time_ = t;
		}
		session = session_.load();
	}
	if(!row.empty() || !record.empty()) publish(session, row, record);
}

void RunProgress::publish(const unsigned session, const std::string& row, const std::string& record) {
	std::lock_guard<std::mutex> hold(core::status_line().gate());
	if(session_.load()!=session) return; // the run ended meanwhile
	if(!row.empty()) core::status_line().draw_locked(row);
	if(!record.empty() && culbm::mpi_is_root()) {
		std::lock_guard<std::mutex> lock(core::console_io_mutex());
		std::cout << record << '\n';
		std::cout.flush();
	}
}

} // namespace culbm::app
