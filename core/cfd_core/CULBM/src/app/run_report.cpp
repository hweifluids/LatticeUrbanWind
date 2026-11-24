#include "run_report.hpp"

#include "../cuda_nested_runtime.hpp"
#include "../solver/field_set.hpp"
#include "../solver/field_table.hpp"
#include "../solver/solver.hpp"
#include "../cuda_nested_runtime.hpp"
#include "../solver/solver.hpp"

#include <algorithm>
#include <cmath>

namespace culbm::app {

namespace {

using culbm::solver::Solver;

std::string rule() {
	return "|"+string((size_t)CONSOLE_WIDTH-2u, '-')+"|";
}

void section(const string& title) {
	println(rule());
	println("|"+alignc((uint)CONSOLE_WIDTH-2u, title)+"|");
	println(rule());
}

void row(const string& key, const string& value) {
	println("| "+key+" | "+value+" |");
}

string join(const std::vector<string>& items, const string& separator) {
	string text;
	for(const string& item : items) text += (text.empty() ? "" : separator)+item;
	return text;
}

string triple(const float a, const float b, const float c) {
	return to_string(a, 8u)+", "+to_string(b, 8u)+", "+to_string(c, 8u);
}

const char* storage_name(const DDF_Storage storage) {
	switch(storage) {
	case DDF_Storage::FP16S: return "FP16S";
	case DDF_Storage::FP32: return "FP32";
	default: return "FP16C";
	}
}

const char* collision_name() {
#if defined(TRT)
	return "TRT";
#else
	return "SRT";
#endif
}

string thermal_outlet_name(const int face) {
	switch((Temperature_Outlet_Face)face) {
	case Temperature_Outlet_Face::West: return "west (-x), zero-gradient";
	case Temperature_Outlet_Face::East: return "east (+x), zero-gradient";
	case Temperature_Outlet_Face::South: return "south (-y), zero-gradient";
	case Temperature_Outlet_Face::North: return "north (+y), zero-gradient";
	default: return "off";
	}
}

// Local GPUs, grouped by model: "2 x NVIDIA ..." or a list of models.
string gpu_list(const Solver& solver) {
	std::vector<string> names;
	for(uint d=0u; d<solver.local_count(); ++d) names.push_back(solver.local(d).device().properties().name);
	const bool same = std::all_of(names.begin(), names.end(), [&](const string& n) { return n==names.front(); });
	if(names.empty()) return "none";
	return same ? to_string((uint)names.size())+" x "+names.front() : join(names, ", ");
}

string device_memory(const Solver& solver, const culbm::solver::LatticeStorage& storage) {
	std::vector<std::uint64_t> booked;
	for(uint d=0u; d<solver.local_count(); ++d) booked.push_back(solver.local(d).device().booked_mib());
	const bool same = std::all_of(booked.begin(), booked.end(), [&](const std::uint64_t b) { return b==booked.front(); });
	string text;
	if(same) {
		text = to_string((ulong)booked.front())+" MiB booked per GPU";
	} else {
		std::vector<string> each;
		for(const std::uint64_t b : booked) each.push_back(to_string((ulong)b));
		text = join(each, " / ")+" MiB booked per GPU";
	}
	// The largest single allocation of a subdomain (halo layers included).
	unsigned widest = 0u;
	for(const culbm::solver::CellField& field : culbm::solver::cell_fields(storage)) widest = std::max(widest, field.bytes());
	const std::uint64_t bytes = (std::uint64_t)widest*solver.local(0u).cells();
	return text+", largest buffer "+to_string((ulong)((bytes+1048575u)/1048576u))+" MiB";
}

string feature_list(const Solver& solver) {
	const culbm::solver::FeatureSet& f = solver.features();
	const culbm::solver::BodyForces& forces = solver.physics().forces;
	std::vector<string> on;
#ifdef TEMPERATURE
	on.push_back("D3Q"+to_string(f.temperature_set)+" temperature");
	if(forces.gx!=0.0f || forces.gy!=0.0f || forces.gz!=0.0f) on.push_back("buoyancy");
#endif
	if(forces.omega_x!=0.0f || forces.omega_y!=0.0f || forces.omega_z!=0.0f) on.push_back("Coriolis");
	if(forces.fx!=0.0f || forces.fy!=0.0f || forces.fz!=0.0f) on.push_back("volume force");
	if(f.subgrid.enabled) on.push_back("Smagorinsky SGS (Cs "+to_string(f.subgrid.smagorinsky_constant, 4u)+")");
	if(f.wall_model.enabled) on.push_back(f.wall_model.ground_only ? "wall model (voxel walls)" : "wall model");
	if(f.surface_ibm.enabled) on.push_back("SurfaceIBM");
	if(solver.statistics_active()) on.push_back("device statistics");
	if(solver.probes_active()) on.push_back("velocity probes");
	return on.empty() ? string("none") : join(on, ", ");
}

} // namespace

void print_banner() {
	// The LatticeUrbanWind mark: L, U and W in a converging V.
	static const char* const mark[] = {
		" ______________   ______________ ",
		"\\   ________  | |  ________   /",
		" \\  \\       | | | |       /  /",
		"  \\  \\   L  | | | |  U   /  /",
		"   \\  \\     | | | |     /  /",
		"    \\  \\_.-\"  | |  \"-._/  /",
		"     \\    _.-\" _ \"-._    /",
		"      \\.-\" _.-\" \"-._ \"-./",
		"       .-\"  .-\"-.  \"-.",
		"       \\  v\"     \"v  /",
		"        \\  \\  W  /  /",
		"         \\  \\   /  /",
		"          \\  \\ /  /",
		"           \\  '  /",
		"            \\   /",
		"             \\ /",
		"              '      CULBM Version 4.0.0",
		"       CUDA lattice-Boltzmann solver for LatticeUrbanWind"};
	const uint width = (uint)CONSOLE_WIDTH-2u;
	string text = rule()+"\n";
	for(const char* line : mark) text += "|"+alignc(width, string(line))+"|\n";
	print(text+rule()+"\n"); // verbatim: the mark contains '|'
}

void print_run_report(const Solver& solver, const std::uint64_t last_step) {
	const culbm::solver::Decomposition& grid = solver.grid();
	const culbm::solver::FeatureSet& f = solver.features();
	const culbm::solver::Physics& physics = solver.physics();
	const culbm::solver::LatticeStorage storage = culbm::solver::storage_of(f);
	const culbm::solver::CellCost cost = culbm::solver::cell_cost(storage);
	const ulong cells = grid.cells();
	const auto box = [](const uint a, const uint b, const uint c) {
		return to_string(a)+" x "+to_string(b)+" x "+to_string(c);
	};

	section("RUN REPORT");
	row("Lattice", box(grid.size(0u), grid.size(1u), grid.size(2u))+" = "+to_string(cells)+" cells");
	row("Subdomains", box(grid.split(0u), grid.split(1u), grid.split(2u))+" = "+to_string(grid.count())+", "+
		box(grid.size(0u)/grid.split(0u), grid.size(1u)/grid.split(1u), grid.size(2u)/grid.split(2u))+" cells each"+
		(solver.distributed() ? ", one per MPI rank" : ""));
	row("GPUs", gpu_list(solver));
	row("Momentum", "D3Q"+to_string(f.velocity_set)+" "+collision_name()+", DDFs stored as "+storage_name(f.storage)+
		" ("+to_string(f.ddf_bytes)+" B), FP32 arithmetic");
	row("Device memory", device_memory(solver, storage));
	row("Host memory", to_string(cells*(ulong)cost.host/1048576u)+" MiB per rank ("+to_string(cost.host)+" B/cell)");
	row("Cell traffic", to_string(cost.traffic)+" B per cell and step (basis of GB/s)");
	row("Features", feature_list(solver));
	row("Steps", last_step==open_ended ? string("open-ended") : to_string((ulong)last_step));

	const culbm::solver::Transport& transport = physics.transport;
	row("Viscosity", "nu = "+to_string(transport.nu, 8u)+", tau = "+to_string(transport.tau(), 8u));
#ifdef TRT
	const TRT_Relaxation_Diagnostic trt = get_trt_relaxation_diagnostic(transport.tau());
	row("TRT", "Lambda = "+to_string(trt.magic_parameter, 8u)+" (3/16), tau +/- = "+to_string(trt.plus.tau, 8u)+
		" / "+to_string(trt.minus.tau, 8u));
	row("TRT omega", "+/- = "+to_string(trt.plus.omega, 8u)+" / "+to_string(trt.minus.omega, 8u)+
		", abs(1-omega) +/- = "+to_string(trt.plus.eigenvalue_magnitude, 8u)+" / "+to_string(trt.minus.eigenvalue_magnitude, 8u));
#endif
	// Lattice speed of sound times the box diagonal over nu.
	const double diagonal = std::sqrt((double)grid.size(0u)*grid.size(0u)+(double)grid.size(1u)*grid.size(1u)+
		(double)grid.size(2u)*grid.size(2u));
	const float reynolds = (float)(0.57735027*diagonal/(double)transport.nu);
	row("Reynolds bound", "Re < "+(reynolds>=100.0f ? to_string(to_uint(reynolds)) : to_string(reynolds, 6u)));
	const culbm::solver::BodyForces& forces = physics.forces;
	row("Coriolis omega", triple(forces.omega_x, forces.omega_y, forces.omega_z));
#ifdef TEMPERATURE
	const float thermal_tau = 0.5f+transport.alpha/f.temperature_cs2;
	const Relaxation_Branch_Diagnostic thermal = get_relaxation_branch_diagnostic(thermal_tau);
	row("Thermal", "D3Q"+to_string(f.temperature_set)+", alpha = "+to_string(transport.alpha, 8u)+
		", beta = "+to_string(transport.beta, 8u));
	row("Thermal relax.", "tau = "+to_string(thermal_tau, 8u)+", omega = "+to_string(thermal.omega, 8u)+
		", abs(1-omega) = "+to_string(thermal.eigenvalue_magnitude, 8u));
	row("Buoyancy g", triple(forces.gx, forces.gy, forces.gz));
	row("Thermal outlet", thermal_outlet_name(physics.temperature_outlet_face));
#endif
	const string threshold = to_string(collision_eigenvalue_warning_threshold, 2u);
#ifdef TRT
	if(trt.plus.near_marginal || trt.minus.near_marginal) {
		log_warning("A TRT collision branch damps less than 2% per step (abs(1-omega) +/- = "+
			to_string(trt.plus.eigenvalue_magnitude, 8u)+" / "+to_string(trt.minus.eigenvalue_magnitude, 8u)+
			", threshold "+threshold+"). Stability depends on the case; Lambda and the relaxation times are kept.");
	}
#endif
#ifdef TEMPERATURE
	if(thermal.near_marginal) {
		log_warning("The temperature collision damps less than 2% per step (abs(1-omega_T) = "+
			to_string(thermal.eigenvalue_magnitude, 8u)+" > "+threshold+"). Stability depends on the case, "
			"including the cell Peclet number and the boundaries; tau_T is kept.");
	}
#endif
}

Workload workload_of(const Solver& solver) {
	Workload work;
	work.cell_updates = solver.grid().cells();
	work.bytes = work.cell_updates*(std::uint64_t)culbm::solver::cell_cost(culbm::solver::storage_of(solver.features())).traffic;
	work.levels = 1u;
	return work;
}

void run_steps(Solver& solver, const std::uint64_t steps, const std::uint64_t total_steps,
	const bool synchronize_after) {
	RunProgress& progress = run_progress();
	const ulong t = solver.time();
	const ulong first = total_steps!=open_ended ? 0u : t;
	const ulong last = total_steps!=open_ended ? total_steps : steps==open_ended ? open_ended : t+steps;
	if(!solver.initialized()) {
		solver.initialize();
		print_run_report(solver, last);
		progress.begin(&solver, workload_of(solver), t, first, last);
	} else if(!progress.retarget(&solver, first, last)) {
		progress.begin(&solver, workload_of(solver), t, first, last);
	}
	if(synchronize_after) {
		solver.run(steps);
	} else {
		for(ulong done=0u; done!=steps; ++done) solver.step();
		if(steps>0u && solver.local_count()>1u) solver.pace();
	}
	progress.advance(solver.time()); // the devices finished or paced the steps of this call
	if(total_steps!=open_ended && steps>0u && solver.time()>=total_steps) {
		if(!synchronize_after) solver.synchronize();
		progress.end(&solver);
	}
}

Workload workload_of(const std::vector<culbm::NestedLbmLevel>& levels) {
	Workload work;
	std::uint64_t substeps = 1u; // steps of a level per coarse step
	for(const culbm::NestedLbmLevel& level : levels) {
		substeps *= std::max<std::uint64_t>(level.refinement_to_parent, 1u);
		const Workload one = workload_of(*level.solver);
		work.cell_updates += one.cell_updates*substeps;
		work.bytes += one.bytes*substeps;
	}
	work.levels = (unsigned)std::max<size_t>(levels.size(), 1u);
	return work;
}

void begin_nested_progress(const void* owner, const std::vector<culbm::NestedLbmLevel>& levels,
	const std::uint64_t step, const std::uint64_t last_step) {
	const Workload work = workload_of(levels);
	std::vector<string> terms;
	std::uint64_t substeps = 1u;
	for(const culbm::NestedLbmLevel& level : levels) {
		substeps *= std::max<std::uint64_t>(level.refinement_to_parent, 1u);
		terms.push_back(to_string((ulong)level.solver->grid().cells())+" x "+to_string((ulong)substeps));
	}
	row("Progress basis", "coarse steps; "+join(terms, " + ")+" = "+to_string((ulong)work.cell_updates)+
		" cell updates per coarse step");
	run_progress().begin(owner, work, step, 0u, last_step);
}

} // namespace culbm::app
