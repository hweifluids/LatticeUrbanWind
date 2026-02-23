#include "info.hpp"
#include "lbm.hpp"

namespace {
constexpr uint k_progress_col_mlups = 9u;
constexpr uint k_progress_col_bandwidth = 13u;
constexpr uint k_progress_col_steps = 11u;
constexpr uint k_progress_col_current = 19u;
constexpr uint k_progress_col_time = 36u;

inline string progress_separator_top() {
	return "|---------.-------'-----.-----------.-------------------.------------------------------------|";
}
inline string progress_separator_bottom() {
	return "|---------'-------------'-----------'-------------------'------------------------------------|";
}
inline string progress_header_row(const bool elapsed_time) {
	return
		"|"+alignc(k_progress_col_mlups, "MLUPs")+"|"+
		alignc(k_progress_col_bandwidth, "Bandwidth")+"|"+
		alignc(k_progress_col_steps, "Steps/s")+"|"+
		alignc(k_progress_col_current, "Current Step")+"|"+
		alignc(k_progress_col_time, elapsed_time ? "Elapsed Time" : "Time Remaining")+"|";
}
inline string progress_value_row(const Info& info) {
	if(info.lbm==nullptr) return "";
	const double dt_smooth = info.runtime_lbm_timestep_smooth>1.0E-9 ? info.runtime_lbm_timestep_smooth : 1.0E-9;
	float progress_ratio = 1.0f;
	if(info.steps>0ull) {
		const float ratio_raw = (float)(info.lbm->get_t()-info.steps_last)/(float)info.steps;
		progress_ratio = ratio_raw<0.0f ? 0.0f : (ratio_raw>1.0f ? 1.0f : ratio_raw);
	}
	const string current = info.steps==max_ulong
		? to_string(info.lbm->get_t())
		: to_string(info.lbm->get_t())+" "+print_percentage(progress_ratio);
	return
		"|"+alignc(k_progress_col_mlups, to_uint((double)info.lbm->get_N()*1E-6/dt_smooth))+"|"+
		alignc(k_progress_col_bandwidth, to_string(to_uint((double)info.lbm->get_N()*(double)bandwidth_bytes_per_cell_device()*1E-9/dt_smooth))+" GB/s")+"|"+
		alignc(k_progress_col_steps, to_uint(1.0/dt_smooth))+"|"+
		alignc(k_progress_col_current, current)+"|"+
		alignc(k_progress_col_time, print_time(info.time()))+"|";
}
}

Info info;

void Info::append(const ulong steps, const ulong total_steps, const ulong t) {
	if(total_steps==max_ulong) { // total_steps is not provided/used
		this->steps = steps; // has to be executed before info.print_initialize()
		this->steps_last = t; // reset last step count if multiple run() commands are executed consecutively
		this->runtime_total_last = this->runtime_total; // reset last runtime if multiple run() commands are executed consecutively
		this->runtime_total = clock.stop();
	} else { // total_steps has been specified
		this->steps = total_steps; // has to be executed before info.print_initialize()
	}
}
void Info::update(const double dt) {
	this->runtime_lbm_timestep_last = dt; // exact dt
	this->runtime_lbm_timestep_smooth = (dt+0.3)/(0.3/runtime_lbm_timestep_smooth+1.0); // smoothed dt
	this->runtime_lbm += dt; // skip first step since it is likely slower than average
	this->runtime_total = clock.stop();
}
double Info::time() const { // returns either elapsed time or remaining time
	if(lbm==nullptr) return 0.0;
	return steps==max_ulong ? runtime_total : ((double)steps/(double)max(lbm->get_t()-steps_last, 1ull)-1.0)*(runtime_total-runtime_total_last); // time estimation on average so far
	//return steps==max_ulong ? runtime_lbm : ((double)steps-(double)(lbm->get_t()-steps_last))*runtime_lbm_timestep_smooth; // instantaneous time estimation
}
void Info::print_logo() const {
	const uint inner = (uint)CONSOLE_WIDTH-2u;
	const vector<string> logo = {
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
		"              '      FluidX3D Version 3.3",
		"                     Copyright (c) Dr. Moritz Lehmann"
	};
	print("|"+string(inner, '-')+"|\n");
	for(const string& line : logo) print("|"+alignc(inner, line)+"|\n");
	print("|"+string(inner, '-')+"|\n");
}
void Info::print_initialize(LBM* lbm) {
	info.allow_printing.lock(); // disable print_update() until print_initialize() has finished
	this->lbm = lbm;
#if defined(SRT)
	collision = "SRT";
#elif defined(TRT)
	collision = "TRT";
#endif // TRT
#if defined(FP16S)
	collision += " (FP32/FP16S)";
#elif defined(FP16C)
	collision += " (FP32/FP16C)";
#else // FP32
	collision += " (FP32/FP32)";
#endif // FP32
	bool all_domains_use_ram = true; // reset cpu/gpu_mem_required to get valid values for consecutive simulations
	for(uint d=0u; d<lbm->get_D(); d++) {
		all_domains_use_ram = all_domains_use_ram&&lbm->lbm_domain[d]->get_device().info.uses_ram;
	}
	if(all_domains_use_ram) {
		cpu_mem_required = lbm->get_D()*lbm->lbm_domain[0]->get_device().info.memory_used;
		gpu_mem_required = 0u;
	} else {
		cpu_mem_required = (uint)(lbm->get_N()*(ulong)bytes_per_cell_host()/1048576ull);
		gpu_mem_required = lbm->lbm_domain[0]->get_device().info.memory_used;
	}
	const float Re = lbm->get_Re_max();
	println("|-----------------.-----------------------------------------------------------|");
	println("| Grid Resolution | "+alignr(57u, to_string(lbm->get_Nx())+" x "+to_string(lbm->get_Ny())+" x "+to_string(lbm->get_Nz())+" = "+to_string(lbm->get_N()))+" |");
	println("| Grid Domains    | "+alignr(57u, to_string(lbm->get_Dx())+" x "+to_string(lbm->get_Dy())+" x "+to_string(lbm->get_Dz())+" = "+to_string(lbm->get_D()))+" |");
	println("| LBM Type        | "+alignr(57u, /***************/ "D"+to_string(lbm->get_velocity_set()==9?2:3)+"Q"+to_string(lbm->get_velocity_set())+" "+collision)+" |");
	println("| Memory Usage    | "+alignr(54u, /*******/ "CPU "+to_string(cpu_mem_required)+" MB, GPU "+to_string(lbm->get_D())+"x "+to_string(gpu_mem_required))+" MB |");
	println("| Max Alloc Size  | "+alignr(54u, /*************/ (uint)(lbm->get_N()/(ulong)lbm->get_D()*(ulong)(lbm->get_velocity_set()*sizeof(fpxx))/1048576ull))+" MB |");
	println("| Time Steps      | "+alignr(57u, /***************************************************************/ (steps==max_ulong ? "infinite" : to_string(steps)))+" |");
	println("| Kin. Viscosity  | "+alignr(57u, /*************************************************************************************/ to_string(lbm->get_nu(), 8u))+" |");
	println("| Relaxation Time | "+alignr(57u, /************************************************************************************/ to_string(lbm->get_tau(), 8u))+" |");
	println("| Reynolds Number | "+alignr(57u, /******************************************/ "Re < "+string(Re>=100.0f ? to_string(to_uint(Re)) : to_string(Re, 6u)))+" |");
	println("| Coriolis Omega  | "+alignr(57u, alignr(15u, to_string(lbm->get_omega_x(), 8u))+","+alignr(15u, to_string(lbm->get_omega_y(), 8u))+","+alignr(15u, to_string(lbm->get_omega_z(), 8u)))+" |");
#ifdef SURFACE
	println("| Surface Tension | "+alignr(57u, /**********************************************************************************/ to_string(lbm->get_sigma(), 8u))+" |");
#endif // SURFACE
#ifdef TEMPERATURE
	println("| Thermal Diff.   | "+alignr(57u, /**********************************************************************************/ to_string(lbm->get_alpha(), 8u))+" |");
	println("| Thermal Exp.    | "+alignr(57u, /***********************************************************************************/ to_string(lbm->get_beta(), 8u))+" |");
#endif // TEMPERATURE
#ifndef INTERACTIVE_GRAPHICS_ASCII
	println(progress_separator_top());
	println(progress_header_row(steps==max_ulong));
	println("|"+string((uint)CONSOLE_WIDTH-2u, ' ')+"|");
#else // INTERACTIVE_GRAPHICS_ASCII
	println("'-----------------'-----------------------------------------------------------'");
#endif // INTERACTIVE_GRAPHICS_ASCII
	clock.start();
	info.allow_printing.unlock();
}
void Info::print_update() const {
	if(lbm==nullptr) return;
	info.allow_printing.lock();
	reprint(progress_value_row(*this));
#ifdef GRAPHICS
	if(key_G) { // print camera settings
		const string camera_position = "float3("+alignr(9u, to_string(camera.pos.x/(float)lbm->get_Nx(), 6u))+"f*(float)Nx, "+alignr(9u, to_string(camera.pos.y/(float)lbm->get_Ny(), 6u))+"f*(float)Ny, "+alignr(9u, to_string(camera.pos.z/(float)lbm->get_Nz(), 6u))+"f*(float)Nz)";
		const string camera_rx_ry_fov = alignr(6u, to_string(degrees(camera.rx)-90.0, 1u))+"f, "+alignr(5u, to_string(180.0-degrees(camera.ry), 1u))+"f, "+alignr(5u, to_string(camera.fov, 1u))+"f";
		const string camera_zoom = alignr(8u, to_string(camera.zoom*(float)fmax(fmax(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz())/(float)min(camera.width, camera.height), 6u))+"f";
		if(camera.free) println("\rlbm.graphics.set_camera_free("+camera_position+", "+camera_rx_ry_fov+");");
		else println("\rlbm.graphics.set_camera_centered("+camera_rx_ry_fov+", "+camera_zoom+");          ");
		key_G = false;
	}
#endif // GRAPHICS
	info.allow_printing.unlock();
}
void Info::print_finalize() {
	if(lbm!=nullptr) {
		println();
	}
	lbm = nullptr;
	println(progress_separator_bottom());
}
