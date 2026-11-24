// Run progress without a GPU: the measured step rate, the two-phase ETA model
// of the LUW time loop, throughput consistency (MLUPS = cells x steps/s,
// GB/s = bytes x steps/s), the console and JSON texts, and the command line.
#include "app/launch.hpp"
#include "app/progress.hpp"
#include "core/console_table.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace culbm::app;

namespace {

int failures = 0;

void expect(const bool condition, const std::string& message) {
	if(condition) return;
	++failures;
	std::cerr << "FAIL: " << message << '\n';
}

bool near(const double a, const double b, const double relative) {
	return std::fabs(a-b)<=relative*std::fmax(std::fabs(b), 1.0E-12);
}

// Number after "key": in a JSON record; NaN when absent or null.
double field(const std::string& record, const std::string& key) {
	const std::string tag = "\""+key+"\":";
	const size_t at = record.find(tag);
	if(at==std::string::npos || record.compare(at+tag.size(), 4u, "null")==0) return std::nan("");
	return std::stod(record.substr(at+tag.size()));
}

double fake_now = 0.0;
double fake_clock() { return fake_now; }

void check_step_rate() {
	StepRate rate;
	expect(rate.rate()==0.0, "rate: empty history is unknown");
	// 250 steps/s, sampled every 50 ms for 5 s.
	for(int i=0; i<=100; ++i) rate.add(0.05*i, (std::uint64_t)(12.5*i));
	expect(near(rate.rate(), 250.0, 1.0E-3), "rate: steady 250 steps/s, got "+std::to_string(rate.rate()));
	// A stall: the rate falls once the window holds no steps ...
	for(int i=1; i<=100; ++i) rate.add(5.0+0.05*i, 1250u);
	expect(rate.rate()<50.0, "rate: falls during a stall");
	// ... and reads 0 when nothing moved for the whole history.
	for(int i=1; i<=1300; ++i) rate.add(10.0+0.05*i, 1250u);
	expect(rate.rate()==0.0, "rate: stalled for the whole history, got "+std::to_string(rate.rate()));

	// Dense samples (every 4 ms, as from per-iteration calls) after a 3 s
	// stall: the window still spans the last second only.
	StepRate dense;
	dense.add(0.0, 0u);
	for(int i=1; i<=1250; ++i) dense.add(3.0+0.004*i, (std::uint64_t)i);
	expect(near(dense.rate(), 250.0, 1.0E-2), "rate: dense samples, got "+std::to_string(dense.rate()));

	// Slow steps (0.8 steps/s): the window grows until it holds 20 steps.
	StepRate slow;
	for(int i=0; i<=1200; ++i) slow.add(0.05*i, (std::uint64_t)(0.04*i));
	expect(near(slow.rate(), 0.8, 0.06), "rate: slow steps, got "+std::to_string(slow.rate()));

	// Shorter than one drawing interval: unknown.
	StepRate start;
	start.add(0.0, 0u);
	start.add(0.1, 30u);
	expect(start.rate()==0.0, "rate: unknown in the first 0.25 s");
}

void check_phase_eta() {
	PhaseEta eta;
	expect(eta.remaining(0u, 1.0)<0.0, "eta: unplanned model has no estimate");
	// 100 steps, mean-field window from step 51 for 50 steps, a sample every
	// second step; hints 100 steps/s and 0.02 s per sample.
	eta.plan(100u, 51u, 50u, 100.0, 0.02, 2u);
	expect(eta.samples_after(0u)==25u, "eta: samples after 0");
	expect(near(eta.remaining(0u, 0.0), 100*0.01+25*0.02, 1.0E-12), "eta: remaining from the hints");
	eta.record(false, 0.03, false, 0.0); // plain mean (0.01+0.03)/2
	expect(near(eta.plain_step_seconds(0.0), 0.02, 1.0E-12), "eta: plain step mean");
	eta.record(true, 0.05, true, 0.0);   // sample extra 0.05-0.02, mean (0.02+0.03)/2
	expect(near(eta.sample_seconds(), 0.025, 1.0E-12), "eta: sample extra mean");
	eta.record(true, 0.02, false, 0.0);  // unsampled window step counts as a plain step
	expect(near(eta.plain_step_seconds(0.0), 0.06/3.0, 1.0E-12), "eta: unsampled window step is plain");
	expect(near(eta.window_step_seconds(0.0), 0.02+0.0125, 1.0E-12), "eta: window step");
	expect(eta.samples_after(60u)==20u, "eta: samples after 60");
	expect(near(eta.remaining(60u, 0.0), 40*0.02+20*0.025, 1.0E-12), "eta: remaining from 60");
	expect(eta.remaining(150u, 0.0)==0.0, "eta: nothing remains past the end");
	// Stride offsets: samples at 13, 16, 19.
	PhaseEta offset;
	offset.plan(20u, 10u, 11u, 10.0, 0.0, 3u);
	expect(offset.samples_after(10u)==3u, "eta: stride offset");
	// No window, no hint: the fallback (measured) step time is used.
	PhaseEta plain;
	plain.plan(10u, 0u, 0u, 0.0, 0.0, 1u);
	expect(plain.samples_after(0u)==0u, "eta: no window");
	expect(near(plain.remaining(4u, 0.5), 3.0, 1.0E-12), "eta: fallback step time");
	expect(plain.remaining(4u, 0.0)<0.0, "eta: unknown without any step time");
}

void check_texts() {
	ProgressSnapshot s;
	s.step = 1200u;
	s.first = 0u;
	s.last = 3000u;
	s.elapsed = 4.75;
	s.steps_per_second = 253.0;
	s.eta = 7.1;
	s.work.cell_updates = 14668836u;
	s.work.bytes = 14668836ull*143u;
	expect(near(s.mlups(), 14668836.0*253.0*1.0E-6, 1.0E-12), "throughput: MLUPS = cells x steps/s");
	expect(near(s.bandwidth_gbs(), 14668836.0*143.0*253.0*1.0E-9, 1.0E-12), "throughput: GB/s = bytes x steps/s");

	const std::string row = progress_status_row(s);
	expect(row.find("1200/3000 (40%)")!=std::string::npos, "row: step and percentage: "+row);
	expect(row.find("253 steps/s  3711 MLUPS  531 GB/s  ETA 7s")!=std::string::npos, "row: figures: "+row);
	expect(culbm::core::format_console_table_line(row, false).size()==(size_t)culbm::core::CONSOLE_WIDTH,
		"row: one console line");
	expect(row.find("MLUPs")==std::string::npos && row.find("Current Step")==std::string::npos, "row: no table header words");

	const std::string record = progress_record(s, "progress");
	expect(record.rfind("[[CULBM_PROGRESS]]{\"event\":\"progress\"", 0)==0, "record: prefix: "+record);
	expect(field(record, "step")==1200.0 && field(record, "total_steps")==3000.0, "record: steps");
	expect(field(record, "cells")==14668836.0 && field(record, "levels")==1.0, "record: workload");
	// Every figure agrees with steps/s x cells within the printed precision.
	const double sps = field(record, "steps_per_second");
	expect(near(field(record, "mlups"), sps*field(record, "cells")*1.0E-6, 2.0E-5), "record: mlups = steps/s x cells");
	expect(near(field(record, "bandwidth_gbs"), sps*14668836.0*143.0*1.0E-9, 2.0E-5), "record: bandwidth = steps/s x bytes");
	expect(near(field(record, "eta_s"), 7.1, 1.0E-9), "record: eta");

	ProgressSnapshot open = s;
	open.last = open_ended;
	open.eta = -1.0;
	open.steps_per_second = 0.0;
	const std::string open_record = progress_record(open, "progress");
	expect(std::isnan(field(open_record, "total_steps")) && std::isnan(field(open_record, "eta_s")), "record: unknown values are null");
	const std::string open_row = progress_status_row(open);
	expect(open_row.find("step 1200  -- steps/s")!=std::string::npos && open_row.find("elapsed 5s")!=std::string::npos,
		"row: open-ended run: "+open_row);

	ProgressSnapshot nested = s;
	nested.work.levels = 2u;
	expect(progress_status_row(nested).rfind("| Nested progress |", 0)==0, "row: nested key");
	expect(progress_summary_row(nested).find("1200 coarse steps in 4.8 s: 253 steps/s")!=std::string::npos,
		"summary: nested: "+progress_summary_row(nested));
}

void check_reporter() {
	fake_now = 100.0;
	RunProgress progress;
	progress.use_clock(fake_clock);
	progress.select_output(ProgressOutput::json);
	Workload work;
	work.cell_updates = 1000000u;
	work.bytes = 143000000u;
	int owner = 0;
	std::ostringstream captured;
	std::streambuf* const console = std::cout.rdbuf(captured.rdbuf());
	progress.begin(&owner, work, 0u, 0u, 2000u);
	// 200 steps/s for 5 s, polled every 50 ms like the console loop.
	for(int i=1; i<=100; ++i) {
		fake_now = 100.0+0.05*i;
		progress.advance((std::uint64_t)(10*i));
		progress.poll();
	}
	const ProgressSnapshot s = progress.snapshot();
	expect(near(s.steps_per_second, 200.0, 1.0E-3), "reporter: measured rate");
	expect(near(s.mlups(), 200.0, 1.0E-3), "reporter: MLUPS = cells x steps/s");
	expect(near(s.eta, 1000.0/200.0, 1.0E-3), "reporter: ETA from the measured rate");
	expect(near(progress.plain_steps_per_second(), 200.0, 1.0E-3), "reporter: plain rate falls back to the measured one");
	// The ETA model takes over once planned.
	progress.plan_phases(2000u, 1001u, 1000u, 100.0, 0.0, 1u);
	expect(near(progress.eta_seconds(), 1000.0/100.0, 1.0E-9), "reporter: ETA from the model hint");
	progress.record_iteration(false, 0.01, false);
	expect(near(progress.plain_steps_per_second(), 100.0, 1.0E-9), "reporter: model plain rate");
	expect(progress.runs(&owner), "reporter: runs");
	int other = 0;
	expect(!progress.retarget(&other, 0u, 10u) && progress.retarget(&owner, 0u, 2000u), "reporter: retarget own run only");
	progress.end(&owner);
	expect(!progress.runs(&owner), "reporter: closed");
	progress.end(&owner); // idempotent
	std::cout.rdbuf(console);

	const std::string out = captured.str();
	size_t records = 0u, at = 0u;
	while((at = out.find("[[CULBM_PROGRESS]]{\"event\":\"progress\"", at))!=std::string::npos) { ++records; ++at; }
	expect(records==4u || records==5u, "reporter: one record per second, got "+std::to_string(records));
	const size_t end_at = out.find("[[CULBM_PROGRESS]]{\"event\":\"end\"");
	expect(end_at!=std::string::npos, "reporter: end record");
	if(end_at!=std::string::npos) {
		const std::string end_record = out.substr(end_at, out.find('\n', end_at)-end_at);
		expect(field(end_record, "step")==1000.0 && near(field(end_record, "steps_per_second"), 200.0, 1.0E-3),
			"reporter: end record is the run mean: "+end_record);
	}
	expect(out.find("| Run summary     | 1000 steps in 5.0 s: 200 steps/s, 200 MLUPS, 29 GB/s (mean)")!=std::string::npos,
		"reporter: summary row: "+out.substr(out.find("| Run summary")==std::string::npos ? 0u : out.find("| Run summary"), 120u));
}

void check_launch() {
	LaunchOptions a = parse_launch_options({"--progress=json", "C:\\case\\conf.luw", "extra"});
	expect(a.progress==ProgressOutput::json && a.error.empty(), "launch: --progress=json");
	expect(a.case_arguments==std::vector<std::string>({"C:\\case\\conf.luw", "extra"}), "launch: case arguments keep their order");
	LaunchOptions b = parse_launch_options({"deck.luw", "--progress-json"});
	expect(b.progress==ProgressOutput::json && b.case_arguments.size()==1u, "launch: --progress-json after the deck");
	LaunchOptions c = parse_launch_options({"--progress=fast"});
	expect(!c.error.empty(), "launch: invalid progress output");
	LaunchOptions d = parse_launch_options({"--", "--help"});
	expect(!d.help && d.case_arguments==std::vector<std::string>({"--help"}), "launch: -- ends the options");
	LaunchOptions e = parse_launch_options({"-h"});
	expect(e.help, "launch: -h");
	LaunchOptions f = parse_launch_options({"deck.luw"});
	expect(f.progress==ProgressOutput::automatic, "launch: no option leaves the environment in charge");
	LaunchOptions g = parse_launch_options({"--dump-config", "out.txt", "deck.luw"});
	expect(g.dump_config=="out.txt" && g.case_arguments==std::vector<std::string>({"deck.luw"}) && g.error.empty(),
		"launch: --dump-config takes the next word");
	LaunchOptions h = parse_launch_options({"deck.luw", "--dump-config=D:\\dumps\\a.txt"});
	expect(h.dump_config=="D:\\dumps\\a.txt" && h.case_arguments.size()==1u, "launch: --dump-config=<file>");
	LaunchOptions i = parse_launch_options({"deck.luw", "--dump-config"});
	expect(!i.error.empty(), "launch: --dump-config without a file name");
	ProgressOutput o = ProgressOutput::automatic;
	expect(parse_progress_output("off", o) && o==ProgressOutput::off && !parse_progress_output("JSON", o), "launch: output names");
	std::string invalid;
	expect(progress_output_from_environment(invalid)==ProgressOutput::automatic && invalid.empty(), "launch: CULBM_PROGRESS unset");
	set_case_arguments({"x.luw"});
	expect(case_arguments().size()==1u && case_arguments()[0]=="x.luw", "launch: stored case arguments");
	RunProgress fresh;
	expect(fresh.output()==ProgressOutput::line, "launch: default output is the status line");
}

} // namespace

int main() {
	check_step_rate();
	check_phase_eta();
	check_texts();
	check_reporter();
	check_launch();
	if(failures==0) std::cout << "CULBM.RunProgress: all checks passed\n";
	return failures==0 ? 0 : 1;
}
