#include "lbm.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
extern float coriolis_Omegax_lbmu;
extern float coriolis_Omegay_lbmu;
extern float coriolis_Omegaz_lbmu;
extern bool buffer_nudging_active;
extern int buffer_n_cells;
extern float buffer_inv_tau_lbmu;
extern int buffer_nudge_vertical;
extern int buffer_downstream_face_id;
extern bool top_sponge_active;
extern int sponge_n_cells;
extern float sponge_inv_tau_lbmu;
extern int sponge_ref_mode;



Units units; // for unit conversion
float coriolis_f_lbmu = 0.0f;
float3 vtk_origin_shift = float3(0.0f); // SI translation applied to VTK origins

#if defined(D2Q9)
const uint velocity_set = 9u;
const uint dimensions = 2u;
const uint transfers = 3u;
#elif defined(D3Q15)
const uint velocity_set = 15u;
const uint dimensions = 3u;
const uint transfers = 5u;
#elif defined(D3Q19)
const uint velocity_set = 19u;
const uint dimensions = 3u;
const uint transfers = 5u;
#elif defined(D3Q27)
const uint velocity_set = 27u;
const uint dimensions = 3u;
const uint transfers = 9u;
#endif // D3Q27

namespace {
Temperature_Lattice temperature_lattice = Temperature_Lattice::Q7;
Subgrid_Configuration subgrid_configuration;
std::atomic_uint active_lbm_instances{0u};

constexpr std::uint64_t device_mib = 1048576u;

std::uint64_t tracked_device_allocation_mb(
	const std::uint64_t elements, const std::uint64_t bytes_per_element) {
	constexpr std::uint64_t limit =
		std::numeric_limits<std::uint64_t>::max();
	if(elements==0u) return 0u;
	if(bytes_per_element > limit/elements) return limit;
	const std::uint64_t bytes = elements*bytes_per_element;
	return bytes/device_mib+(bytes%device_mib!=0u);
}

bool add_device_allocation_mb(
	std::uint64_t& total_mb, const std::uint64_t allocation_mb) {
	constexpr std::uint64_t limit =
		std::numeric_limits<std::uint64_t>::max();
	if(allocation_mb > limit-total_mb) {
		total_mb = limit;
		return false;
	}
	total_mb += allocation_mb;
	return true;
}

bool optional_device_memory_fits(
	const Device& device,
	const std::uint64_t required_mb,
	const std::uint64_t future_device_memory_mb) {
	constexpr std::uint64_t limit =
		std::numeric_limits<std::uint64_t>::max();
	if(required_mb > limit/device_mib ||
	   future_device_memory_mb > limit/device_mib) return false;
	const std::uint64_t required_bytes = required_mb*device_mib;
	const std::uint64_t future_bytes = future_device_memory_mb*device_mib;
	const std::uint64_t total_bytes =
		(std::uint64_t)device.get_total_memory_bytes();
	const std::uint64_t free_bytes =
		(std::uint64_t)device.get_free_memory_bytes();
	const std::uint64_t reserve_bytes =
		std::min<std::uint64_t>(256u*device_mib, total_bytes/10u);
	if(required_bytes > free_bytes) return false;
	const std::uint64_t after_required = free_bytes-required_bytes;
	if(future_bytes > after_required) return false;
	return after_required-future_bytes >= reserve_bytes;
}
}

bool set_subgrid_configuration(
	const Subgrid_Configuration& configuration) {
	const bool valid_fixed =
		configuration.fixed_eddy_viscosity==-1.0f ||
		(std::isfinite(configuration.fixed_eddy_viscosity) &&
		 configuration.fixed_eddy_viscosity>=0.0f);
	if(!std::isfinite(configuration.smagorinsky_constant) ||
	   configuration.smagorinsky_constant<0.0f ||
	   !std::isfinite(configuration.turbulent_prandtl) ||
	   configuration.turbulent_prandtl<=0.0f || !valid_fixed) return false;
	const float prefactor = 18.0f*std::sqrt(2.0f)*
		configuration.smagorinsky_constant*
		configuration.smagorinsky_constant;
	const float scalar_tau_factor = 1.0f/
		(3.0f*configuration.turbulent_prandtl*(1.0f/4.0f));
	if(!std::isfinite(prefactor) || !std::isfinite(scalar_tau_factor) ||
	   (configuration.fixed_eddy_viscosity>=0.0f &&
	    !std::isfinite(3.0f*configuration.fixed_eddy_viscosity))) return false;
	const bool changes =
		configuration.enabled!=subgrid_configuration.enabled ||
		configuration.smagorinsky_constant!=
			subgrid_configuration.smagorinsky_constant ||
		configuration.scalar_coupling!=
			subgrid_configuration.scalar_coupling ||
		configuration.turbulent_prandtl!=
			subgrid_configuration.turbulent_prandtl ||
		configuration.fixed_eddy_viscosity!=
			subgrid_configuration.fixed_eddy_viscosity;
	if(!changes) return true;
	if(active_lbm_instances.load(std::memory_order_acquire)>0u) return false;
	subgrid_configuration = configuration;
	return true;
}

Subgrid_Configuration get_subgrid_configuration() {
	return subgrid_configuration;
}

bool set_temperature_lattice(const Temperature_Lattice lattice) {
	if(lattice!=Temperature_Lattice::Q7 &&
	   lattice!=Temperature_Lattice::Q19) return false;
#if !defined(D3Q19)
	if(lattice==Temperature_Lattice::Q19) return false;
#endif
	if(active_lbm_instances.load(std::memory_order_acquire)>0u &&
	   lattice!=temperature_lattice) return false;
	temperature_lattice = lattice;
	return true;
}
Temperature_Lattice get_temperature_lattice() {
	return temperature_lattice;
}
uint get_temperature_set() {
	return static_cast<uint>(temperature_lattice);
}
uint get_temperature_transfers() {
	return temperature_lattice==Temperature_Lattice::Q19 ? 5u : 1u;
}
float get_temperature_lattice_cs2() {
	return temperature_lattice==Temperature_Lattice::Q19
		? 1.0f/3.0f : 1.0f/4.0f;
}
float get_temperature_tau(const float alpha) {
	return 0.5f+alpha/get_temperature_lattice_cs2();
}
const char* get_temperature_lattice_name() {
	return temperature_lattice==Temperature_Lattice::Q19
		? "D3Q19" : "D3Q7";
}

Relaxation_Branch_Diagnostic get_relaxation_branch_diagnostic(
	const float tau) {
	Relaxation_Branch_Diagnostic diagnostic{};
	diagnostic.tau = tau;
	diagnostic.valid = std::isfinite(tau)&&tau!=0.0f;
	if(!diagnostic.valid) {
		diagnostic.omega = std::numeric_limits<float>::quiet_NaN();
		diagnostic.eigenvalue_magnitude =
			std::numeric_limits<float>::quiet_NaN();
		return diagnostic;
	}
	diagnostic.omega = 1.0f/tau;
	diagnostic.eigenvalue_magnitude =
		std::fabs(1.0f-diagnostic.omega);
	diagnostic.valid = std::isfinite(diagnostic.omega)&&
		std::isfinite(diagnostic.eigenvalue_magnitude);
	diagnostic.near_marginal = diagnostic.valid&&
		diagnostic.eigenvalue_magnitude>
			collision_eigenvalue_warning_threshold;
	return diagnostic;
}

TRT_Relaxation_Diagnostic get_trt_relaxation_diagnostic(
	const float tau_plus) {
	TRT_Relaxation_Diagnostic diagnostic{};
	diagnostic.magic_parameter = trt_magic_parameter;
	diagnostic.plus = get_relaxation_branch_diagnostic(tau_plus);
	const float denominator = tau_plus-0.5f;
	const float tau_minus = std::isfinite(denominator)&&denominator>0.0f
		? 0.5f+trt_magic_parameter/denominator
		: std::numeric_limits<float>::quiet_NaN();
	diagnostic.minus = get_relaxation_branch_diagnostic(tau_minus);
	return diagnostic;
}

namespace {
struct TriangleAABB {
	float3 pmin;
	float3 pmax;
};
struct DomainProjectedBounds {
	float min_a;
	float max_a;
	float min_b;
	float max_b;
};

inline TriangleAABB triangle_aabb(const float3& p0, const float3& p1, const float3& p2) {
	TriangleAABB aabb;
	aabb.pmin = float3(
		fmin(fmin(p0.x, p1.x), p2.x),
		fmin(fmin(p0.y, p1.y), p2.y),
		fmin(fmin(p0.z, p1.z), p2.z)
	);
	aabb.pmax = float3(
		fmax(fmax(p0.x, p1.x), p2.x),
		fmax(fmax(p0.y, p1.y), p2.y),
		fmax(fmax(p0.z, p1.z), p2.z)
	);
	return aabb;
}
inline DomainProjectedBounds make_projected_bounds(const uint direction, const float x_min, const float x_max, const float y_min, const float y_max, const float z_min, const float z_max) {
	if(direction==0u) return DomainProjectedBounds{ y_min, y_max, z_min, z_max }; // yz projection
	if(direction==1u) return DomainProjectedBounds{ x_min, x_max, z_min, z_max }; // xz projection
	return DomainProjectedBounds{ x_min, x_max, y_min, y_max }; // xy projection
}
inline bool overlap_1d(const float min0, const float max0, const float min1, const float max1, const float pad, const float eps) {
	return max0>=min1-pad-eps && min0<=max1+pad+eps;
}
inline bool overlaps_projected(const TriangleAABB& tri, const DomainProjectedBounds& dom, const uint direction, const float pad, const float eps) {
	float tri_min_a = 0.0f, tri_max_a = 0.0f, tri_min_b = 0.0f, tri_max_b = 0.0f;
	if(direction==0u) { // yz
		tri_min_a = tri.pmin.y; tri_max_a = tri.pmax.y;
		tri_min_b = tri.pmin.z; tri_max_b = tri.pmax.z;
	} else if(direction==1u) { // xz
		tri_min_a = tri.pmin.x; tri_max_a = tri.pmax.x;
		tri_min_b = tri.pmin.z; tri_max_b = tri.pmax.z;
	} else { // xy
		tri_min_a = tri.pmin.x; tri_max_a = tri.pmax.x;
		tri_min_b = tri.pmin.y; tri_max_b = tri.pmax.y;
	}
	return overlap_1d(tri_min_a, tri_max_a, dom.min_a, dom.max_a, pad, eps) &&
		overlap_1d(tri_min_b, tri_max_b, dom.min_b, dom.max_b, pad, eps);
}

bool concurrent_graphics_enabled() {
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
	return true;
#else
	return false;
#endif
}

#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
bool interactive_simulation_may_advance() {
	struct Simulation_Control {
		uint32_t running;
		uint32_t advance;
	};
	while(true) {
		Simulation_Control control = {};
		const bool control_root =
			!culbm::mpi_is_distributed() || culbm::mpi_is_root();
		if(control_root) {
			control.running =
				running.load(std::memory_order_acquire) &&
				!stop_requested.load(std::memory_order_acquire)
					? 1u : 0u;
			control.advance =
				key_P.load(std::memory_order_acquire) ? 1u : 0u;
		}
		if(culbm::mpi_is_distributed())
			culbm::mpi_broadcast_simulation_bytes(
				&control, sizeof(control));
		if(control.running==0u)
			stop_requested.store(true, std::memory_order_release);
		if(control.running==0u)
			running.store(false, std::memory_order_release);
		if(!control_root)
			key_P.store(control.advance!=0u, std::memory_order_release);
		if(control.running==0u) return false;
		if(control.advance!=0u) return true;
		sleep(0.016);
	}
}
#endif
} // namespace

inline uint tracked_allocation_mb(const ulong bytes) {
	const ulong whole_mb = bytes/1048576ull;
	const ulong rounded_mb = whole_mb+(bytes%1048576ull!=0ull);
	return (uint)min((ulong)max_uint, rounded_mb);
}

std::uint64_t voxelization_transient_vram_mb_per_device(
	const std::uint64_t triangle_count) {
	std::uint64_t total_mb = 0u;
	const std::uint64_t vertices_mb =
		tracked_device_allocation_mb(triangle_count, sizeof(float3));
	bool valid = add_device_allocation_mb(total_mb, vertices_mb);
	valid &= add_device_allocation_mb(total_mb, vertices_mb);
	valid &= add_device_allocation_mb(total_mb, vertices_mb);
	valid &= add_device_allocation_mb(
		total_mb, tracked_device_allocation_mb(16u, sizeof(float)));
	return valid ? total_mb : std::numeric_limits<std::uint64_t>::max();
}

uint bytes_per_cell_host() { // returns the number of Bytes per cell allocated in host memory
	uint bytes_per_cell = 17u; // rho, u, flags
#ifdef FORCE_FIELD
	bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#ifdef SURFACE
	bytes_per_cell += 4u; // phi
#endif // SURFACE
#ifdef TEMPERATURE
	bytes_per_cell += 4u; // T
#endif // TEMPERATURE
	return bytes_per_cell;
}
uint bytes_per_cell_device() { // returns the number of Bytes per cell allocated in device memory
	uint bytes_per_cell = velocity_set*sizeof(fpxx)+17u; // fi, rho, u, flags
#ifdef FORCE_FIELD
	bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#ifdef SURFACE
	bytes_per_cell += 12u; // phi, mass, flags
#endif // SURFACE
#ifdef TEMPERATURE
	bytes_per_cell += get_temperature_set()*sizeof(fpxx)+4u; // gi, T
#endif // TEMPERATURE
	return bytes_per_cell;
}
uint bandwidth_bytes_per_cell_device() { // returns the bandwidth in Bytes per cell per time step from/to device memory
	uint bandwidth_bytes_per_cell = velocity_set*2u*sizeof(fpxx)+1u; // lattice.set()*2*fi, flags
#ifdef UPDATE_FIELDS
	bandwidth_bytes_per_cell += 16u; // rho, u
#ifdef TEMPERATURE
	bandwidth_bytes_per_cell += 4u; // T
#endif // TEMPERATURE
#endif // UPDATE_FIELDS
#ifdef FORCE_FIELD
	bandwidth_bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#if defined(MOVING_BOUNDARIES)||defined(SURFACE)
	bandwidth_bytes_per_cell += (velocity_set-1u)*1u; // neighbor flags have to be loaded
#elif defined(TEMPERATURE)
	bandwidth_bytes_per_cell += get_temperature_set()-1u; // scalar-neighbor flags
#endif // MOVING_BOUNDARIES or SURFACE; otherwise TEMPERATURE
#ifdef SURFACE
	bandwidth_bytes_per_cell += (1u+(2u*velocity_set-1u)*sizeof(fpxx)+8u+(velocity_set-1u)*4u) + 1u + 1u + (4u+velocity_set+4u+4u+4u); // surface_0 (flags, fi, mass, massex), surface_1 (flags), surface_2 (flags), surface_3 (rho, flags, mass, massex, phi)
#endif // SURFACE
#ifdef TEMPERATURE
	bandwidth_bytes_per_cell += get_temperature_set()*2u*sizeof(fpxx); // 2*gi
#endif // TEMPERATURE
	return bandwidth_bytes_per_cell;
}
uint3 resolution(const float3 box_aspect_ratio, const uint memory) { // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	// This helper fits from total capacity before the CUDA module is resident:
	// 256 MiB covers runtime/module overhead and another 256 MiB is the
	// allocator's post-allocation guard.
	constexpr uint fit_reserve_mb = 512u;
	const uint usable_memory = memory>fit_reserve_mb
		? memory-fit_reserve_mb
		: memory*9u/10u;
	const auto make_resolution = [&](const float scaling) {
#ifndef D2Q9
		return uint3(
			max(1u, to_uint(scaling*box_aspect_ratio.x)),
			max(1u, to_uint(scaling*box_aspect_ratio.y)),
			max(1u, to_uint(scaling*box_aspect_ratio.z))
		);
#else // D2Q9
		return uint3(
			max(1u, to_uint(scaling*box_aspect_ratio.x)),
			max(1u, to_uint(scaling*box_aspect_ratio.y)),
			1u
		);
#endif // D2Q9
	};
#ifndef D2Q9
	float memory_required = (box_aspect_ratio.x*box_aspect_ratio.y*box_aspect_ratio.z)*(float)bytes_per_cell_device()/1048576.0f; // in MB
	float fit_scaling = cbrt((float)usable_memory/memory_required);
#else // D2Q9
	float memory_required = (box_aspect_ratio.x*box_aspect_ratio.y)*(float)bytes_per_cell_device()/1048576.0f; // in MB
	float fit_scaling = sqrt((float)memory/memory_required);
#endif // D2Q9
	fit_scaling = fmax(fit_scaling, 1.0e-6f);
	while(vram_required_mb_per_device(make_resolution(fit_scaling).x, make_resolution(fit_scaling).y, make_resolution(fit_scaling).z, 1u, 1u, 1u)>usable_memory&&fit_scaling>1.0e-6f) fit_scaling *= 0.5f;
	float overflow_scaling = fit_scaling*2.0f;
	while(vram_required_mb_per_device(make_resolution(overflow_scaling).x, make_resolution(overflow_scaling).y, make_resolution(overflow_scaling).z, 1u, 1u, 1u)<=usable_memory&&overflow_scaling<1.0e8f) {
		fit_scaling = overflow_scaling;
		overflow_scaling *= 2.0f;
	}
	uint3 best = make_resolution(fit_scaling);
	for(uint i=0u; i<48u; i++) {
		const float mid_scaling = 0.5f*(fit_scaling+overflow_scaling);
		const uint3 candidate = make_resolution(mid_scaling);
		if(vram_required_mb_per_device(candidate.x, candidate.y, candidate.z, 1u, 1u, 1u)<=usable_memory) {
			fit_scaling = mid_scaling;
			best = candidate;
		} else {
			overflow_scaling = mid_scaling;
		}
	}
	return best;
}

// required VRAM in MB for a given resolution
uint vram_required_mb_per_device(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz) {
	const uint Hx = Dx>1u, Hy = Dy>1u, Hz = Dz>1u;
	const ulong local_Nx = (ulong)(Nx/Dx+2u*Hx);
	const ulong local_Ny = (ulong)(Ny/Dy+2u*Hy);
	const ulong local_Nz = (ulong)(Nz/Dz+2u*Hz);
	const ulong local_N = local_Nx*local_Ny*local_Nz;
	uint memory_mb = 0u;

	memory_mb += tracked_allocation_mb(local_N*(ulong)velocity_set*sizeof(fpxx)); // fi
	memory_mb += tracked_allocation_mb(local_N*sizeof(float)); // rho
	memory_mb += tracked_allocation_mb(local_N*3ull*sizeof(float)); // u
	memory_mb += tracked_allocation_mb(local_N*sizeof(uchar)); // flags
#ifdef FORCE_FIELD
	memory_mb += tracked_allocation_mb(local_N*3ull*sizeof(float)); // F
	memory_mb += tracked_allocation_mb(4ull*sizeof(float)); // object_sum
#endif // FORCE_FIELD
#ifdef SURFACE
	memory_mb += tracked_allocation_mb(local_N*sizeof(float)); // phi
	memory_mb += tracked_allocation_mb(local_N*sizeof(float)); // mass
	memory_mb += tracked_allocation_mb(local_N*sizeof(float)); // massex
#endif // SURFACE
#ifdef TEMPERATURE
	memory_mb += tracked_allocation_mb(
		local_N*(ulong)get_temperature_set()*sizeof(fpxx)); // gi
	memory_mb += tracked_allocation_mb(local_N*sizeof(float)); // T
#endif // TEMPERATURE
#ifdef GRAPHICS
	const ulong pixels = (ulong)GRAPHICS_FRAME_WIDTH*(ulong)GRAPHICS_FRAME_HEIGHT;
	memory_mb += tracked_allocation_mb(pixels*sizeof(int)); // bitmap
	memory_mb += tracked_allocation_mb(pixels*sizeof(int)); // zbuffer
	memory_mb += tracked_allocation_mb(15ull*sizeof(float)); // camera_parameters
#endif // GRAPHICS
	if(Dx*Dy*Dz>1u) {
		ulong Amax = 0ull;
		if(Dx>1u) Amax = max(Amax, local_Ny*local_Nz);
		if(Dy>1u) Amax = max(Amax, local_Nz*local_Nx);
		if(Dz>1u) Amax = max(Amax, local_Nx*local_Ny);
		uint transfer_cell_bytes = max(transfers*(uint)sizeof(fpxx), 17u);
#ifdef TEMPERATURE
		transfer_cell_bytes =
			max(transfer_cell_bytes,
				(transfers+get_temperature_transfers())*(uint)sizeof(fpxx));
#endif // TEMPERATURE
		const ulong transfer_bytes = Amax*(ulong)transfer_cell_bytes;
		memory_mb += 4u*tracked_allocation_mb(transfer_bytes); // simulation send/receive +/- buffers
#ifdef GRAPHICS
		if(concurrent_graphics_enabled()) {
			memory_mb += tracked_allocation_mb(local_N*sizeof(float)); // graphics rho snapshot
			memory_mb += tracked_allocation_mb(local_N*3ull*sizeof(float)); // graphics u snapshot
			memory_mb += tracked_allocation_mb(local_N*sizeof(uchar)); // graphics flags snapshot
#ifdef FORCE_FIELD
			memory_mb += tracked_allocation_mb(local_N*3ull*sizeof(float)); // graphics F snapshot
#endif // FORCE_FIELD
#ifdef SURFACE
			memory_mb += tracked_allocation_mb(local_N*sizeof(float)); // graphics phi snapshot
#endif // SURFACE
#ifdef TEMPERATURE
			memory_mb += tracked_allocation_mb(local_N*sizeof(float)); // graphics T snapshot
#endif // TEMPERATURE
			const ulong graphics_transfer_bytes = Amax*17ull;
			memory_mb += 4u*tracked_allocation_mb(graphics_transfer_bytes); // graphics send/receive +/- buffers
		}
#endif // GRAPHICS
	}
	return memory_mb;
}
uint vram_required_mb_total(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz) {
	const uint per_dev = vram_required_mb_per_device(Nx, Ny, Nz, Dx, Dy, Dz);
	return per_dev * (Dx * Dy * Dz);
}


string default_filename(const string& path, const string& name, const string& extension, const ulong t) { // generate a default filename with timestamp
	string time = "00000000"+to_string(t);
	time = substring(time, length(time)-9u, 9u);
	return (path=="" ? get_exe_path()+"export/" : path)+create_file_extension((name=="" ? "file" : name)+"-"+time, extension);
}
string default_filename(const string& name, const string& extension, const ulong t) { // generate a default filename with timestamp at exe_path/export/
	return default_filename("", name, extension, t);
}



LBM_Domain::LBM_Domain(const Device_Info& device_info, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const int Ox, const int Oy, const int Oz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // constructor with manual device selection and domain offset
	this->Nx = Nx; this->Ny = Ny; this->Nz = Nz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	this->Ox = Ox; this->Oy = Oy; this->Oz = Oz;
	this->nu = nu;
	this->fx = fx; this->fy = fy; this->fz = fz;
	this->sigma = sigma;
	this->alpha = alpha; this->beta = beta;
	this->particles_N = particles_N;
	this->particles_rho = particles_rho;
	string cuda_kernel_source;
#ifdef GRAPHICS
	graphics = Graphics(this);
	cuda_kernel_source = device_defines()+graphics.device_defines()+get_cuda_kernel_source();
#else // GRAPHICS
	cuda_kernel_source = device_defines()+get_cuda_kernel_source();
#endif // GRAPHICS
	this->device = Device(device_info, cuda_kernel_source);
	print_info("Allocating memory. This may take a few seconds.");
	luw_emit_progress("gpu_memory",
					  "Configuring GPU memory",
					  "Allocating CFD buffers on device "+to_string(device_info.id),
					  0ll,
					  1ll,
					  false);
	allocate(device); // lbm first
#ifdef GRAPHICS
	graphics.allocate(device); // graphics after lbm
#endif // GRAPHICS
	luw_emit_progress("gpu_memory",
					  "Configuring GPU memory",
					  "Device "+to_string(device_info.id)+" buffers ready",
					  1ll,
					  1ll,
					  false);
}

void LBM_Domain::allocate(Device& device) {
	const ulong N = get_N();
	fi = Memory<fpxx>(device, N, velocity_set, false);
	rho = Memory<float>(device, N, 1u, true, true, 1.0f);
	u = Memory<float>(device, N, 3u);
	flags = Memory<uchar>(device, N);
	kernel_initialize = Kernel(device, N, "initialize", fi, rho, u, flags
#ifdef VOLUME_FORCE
		, fx, fy, fz, omega_x, omega_y, omega_z
#ifdef TEMPERATURE
		, buoyancy_gx, buoyancy_gy, buoyancy_gz
#endif // TEMPERATURE
#endif // VOLUME_FORCE
	);
	kernel_stream_collide = Kernel(device, N, "stream_collide", fi, rho, u, flags, get_t(), fx, fy, fz, omega_x, omega_y, omega_z
#ifdef TEMPERATURE
		, buoyancy_gx, buoyancy_gy, buoyancy_gz
#endif // TEMPERATURE
	);
	kernel_update_fields = Kernel(device, N, "update_fields", fi, rho, u, flags, get_t(), fx, fy, fz, omega_x, omega_y, omega_z
#ifdef TEMPERATURE
		, buoyancy_gx, buoyancy_gy, buoyancy_gz
#endif // TEMPERATURE
	);

#ifdef FORCE_FIELD
	F = Memory<float>(device, N, 3u);
	object_sum = Memory<float>(device, 1u, 4u); // x, y, z, cell count
	kernel_initialize.add_parameters(F);
	kernel_stream_collide.add_parameters(F);
	kernel_update_fields.add_parameters(F);
	kernel_update_force_field = Kernel(device, N, "update_force_field", fi, flags, get_t(), F);
	kernel_reset_force_field = Kernel(device, N, "reset_force_field", F);
	kernel_object_center_of_mass = Kernel(device, N, "object_center_of_mass", flags, (uchar)0u, object_sum);
	kernel_object_force = Kernel(device, N, "object_force", F, flags, (uchar)0u, object_sum);
	kernel_object_torque = Kernel(device, N, "object_torque", F, flags, (uchar)0u, 0.0f, 0.0f, 0.0f, object_sum);
#endif // FORCE_FIELD

#ifdef MOVING_BOUNDARIES
	kernel_update_moving_boundaries = Kernel(device, N, "update_moving_boundaries", u, flags);
#endif // MOVING_BOUNDARIES

#ifdef SURFACE
	phi = Memory<float>(device, N);
	mass = Memory<float>(device, N, 1u, false);
	massex = Memory<float>(device, N, 1u, false);
	kernel_initialize.add_parameters(mass, massex, phi);
	kernel_stream_collide.add_parameters(mass);
	kernel_surface_0 = Kernel(device, N, "surface_0", fi, rho, u, flags, mass, massex, phi, get_t(), fx, fy, fz);
	kernel_surface_1 = Kernel(device, N, "surface_1", flags);
	kernel_surface_2 = Kernel(device, N, "surface_2", fi, rho, u, flags, get_t());
	kernel_surface_3 = Kernel(device, N, "surface_3", rho, flags, mass, massex, phi);
#endif // SURFACE

#ifdef TEMPERATURE
	gi = Memory<fpxx>(device, N, get_temperature_set(), false);
	T = Memory<float>(device, N, 1u, true, true, 1.0f);
	kernel_initialize.add_parameters(gi, T);
	kernel_stream_collide.add_parameters(gi, T);
	kernel_update_fields.add_parameters(gi, T);
	kernel_update_temperature_outlet = Kernel(
		device, 0ull, "update_temperature_outlet",
		gi, T, flags, get_t(), temperature_outlet_face);
	kernel_update_free_slip_boundaries = Kernel(
		device, N, "culbm_update_free_slip_boundaries",
		rho, u, T, flags, (uint)0u);
#endif // TEMPERATURE

	if(get_subgrid_configuration().enabled) {
		update_fields_control_parameter =
			kernel_update_fields.get_number_of_parameters();
		kernel_update_fields.add_parameters((uint)1u); // write rho/u/T
		// These buffers are tiny and sampled infrequently.  Pinning their host
		// mirrors lets all local GPU reductions overlap before the final read.
		subgrid_sum = Memory<double>(
			device, 1ull, 2u, true, true, 0.0, false);
		subgrid_max = Memory<float>(
			device, 1ull, 2u, true, true, 0.0f, false);
		subgrid_count = Memory<ulong>(
			device, 1ull, 1u, true, true, (ulong)0u, false);
		update_fields_subgrid_parameter =
			kernel_update_fields.get_number_of_parameters();
		kernel_update_fields.add_parameters(
			(uint)0u, subgrid_sum, subgrid_max, subgrid_count);
	}

#ifdef PARTICLES
	particles = Memory<float>(device, (ulong)particles_N, 3u);
	kernel_integrate_particles = Kernel(device, (ulong)particles_N, "integrate_particles", particles, u, flags, 1.0f);
#ifdef FORCE_FIELD
	kernel_integrate_particles.add_parameters(F, fx, fy, fz);
#endif // FORCE_FIELD
#endif // PARTICLES

	if(get_D()>1u) allocate_transfer(device);
}

void LBM_Domain::enqueue_initialize() { // call kernel_initialize
#ifdef VOLUME_FORCE
	kernel_initialize.set_parameters(4u, fx, fy, fz, omega_x, omega_y, omega_z
#ifdef TEMPERATURE
		, buoyancy_gx, buoyancy_gy, buoyancy_gz
#endif // TEMPERATURE
	);
#endif // VOLUME_FORCE
	kernel_initialize.enqueue_run();
}

void LBM_Domain::enqueue_update_temperature_outlet() {
#ifdef TEMPERATURE
	if(temperature_outlet_face==0) return;
	const ulong face_area =
		temperature_outlet_face<=(int)Temperature_Outlet_Face::East
		? (ulong)Ny*(ulong)Nz
		: (ulong)Nz*(ulong)Nx;
	kernel_update_temperature_outlet
		.set_ranges(face_area)
		.set_parameters(3u, get_t(), temperature_outlet_face)
		.enqueue_run();
#endif // TEMPERATURE
}

void LBM_Domain::enqueue_update_free_slip_boundaries(
	const uint face_mask) {
#ifdef TEMPERATURE
	if(face_mask==0u) return;
	kernel_update_free_slip_boundaries
		.set_parameters(4u, face_mask)
		.enqueue_run();
#ifdef MOVING_BOUNDARIES
	// Wall velocities changed after initialization, so refresh TYPE_MS on
	// adjacent fluid cells before stream/collide consumes them.
	enqueue_update_moving_boundaries();
#endif // MOVING_BOUNDARIES
#else
	(void)face_mask;
#endif // TEMPERATURE
}

void LBM_Domain::enqueue_stream_collide() { // call kernel_stream_collide to perform one LBM time step
	kernel_stream_collide.set_parameters(4u, get_t(), fx, fy, fz, omega_x, omega_y, omega_z
#ifdef TEMPERATURE
		, buoyancy_gx, buoyancy_gy, buoyancy_gz
#endif // TEMPERATURE
	).enqueue_run();
}

void LBM_Domain::enqueue_update_fields() { // update fields (rho, u, T) manually
#ifndef UPDATE_FIELDS
	const ulong current_t = get_t();
	if(current_t!=t_last_update_fields) { // only run kernel_update_fields if the time step has changed since last update
#ifdef TEMPERATURE
		enqueue_update_temperature_outlet();
#endif // TEMPERATURE
		kernel_update_fields.set_parameters(4u, current_t, fx, fy, fz, omega_x, omega_y, omega_z
#ifdef TEMPERATURE
			, buoyancy_gx, buoyancy_gy, buoyancy_gz
#endif // TEMPERATURE
		);
		if(get_subgrid_configuration().enabled) {
			kernel_update_fields.set_parameters(
				update_fields_control_parameter, (uint)1u);
			kernel_update_fields.set_parameters(
				update_fields_subgrid_parameter, (uint)0u);
		}
		kernel_update_fields.enqueue_run();
		t_last_update_fields = current_t;
	}
#endif // UPDATE_FIELDS
}

void LBM_Domain::enqueue_subgrid_statistics() {
	if(!get_subgrid_configuration().enabled) return;
	subgrid_sum.x[0] = 0.0;
	subgrid_sum.y[0] = 0.0;
	subgrid_max.x[0] = 0.0f;
	subgrid_max.y[0] = 0.0f;
	subgrid_count.x[0] = (ulong)0u;
	subgrid_sum.enqueue_write_to_device();
	subgrid_max.enqueue_write_to_device();
	subgrid_count.enqueue_write_to_device();
	kernel_update_fields.set_parameters(
		4u, get_t(), fx, fy, fz, omega_x, omega_y, omega_z
#ifdef TEMPERATURE
		, buoyancy_gx, buoyancy_gy, buoyancy_gz
#endif // TEMPERATURE
	);
	kernel_update_fields.set_parameters(
		update_fields_control_parameter, (uint)0u);
	kernel_update_fields.set_parameters(
		update_fields_subgrid_parameter, (uint)1u);
	kernel_update_fields.enqueue_run();
	subgrid_sum.enqueue_read_from_device();
	subgrid_max.enqueue_read_from_device();
	subgrid_count.enqueue_read_from_device();
}

Subgrid_Statistics LBM_Domain::read_subgrid_statistics() {
	Subgrid_Statistics statistics;
	if(!get_subgrid_configuration().enabled) return statistics;
	device.finish_queue();
	statistics.count = (std::uint64_t)subgrid_count.x[0];
	statistics.valid = statistics.count>0u;
	if(!statistics.valid) return statistics;
	statistics.mean_nu_t =
		subgrid_sum.x[0]/(double)statistics.count;
	statistics.max_nu_t = subgrid_max.x[0];
	statistics.mean_rho_nu_t =
		subgrid_sum.y[0]/(double)statistics.count;
	statistics.max_rho_nu_t = subgrid_max.y[0];
	return statistics;
}

bool LBM_Domain::enable_device_statistics(
	const bool include_temperature,
	const uint future_device_memory_mb) {
	disable_device_statistics();
	const ulong N = get_N();
	std::uint64_t required_mb = 0u;
	bool estimate_valid = true;
	estimate_valid &= add_device_allocation_mb(
		required_mb, tracked_device_allocation_mb(N, 3u*sizeof(float)));
	estimate_valid &= add_device_allocation_mb(
		required_mb, tracked_device_allocation_mb(N, sizeof(float)));
	estimate_valid &= add_device_allocation_mb(
		required_mb, tracked_device_allocation_mb(N, 3u*sizeof(float)));
#ifdef TEMPERATURE
	if(include_temperature) {
		estimate_valid &= add_device_allocation_mb(
			required_mb, tracked_device_allocation_mb(N, sizeof(float)));
	}
#endif // TEMPERATURE
	// Statistics are optional.  Use current driver-reported free VRAM and
	// protect both the allocator's reserve and any deferred required feature
	// (notably independent particle advection) before choosing the GPU path.
	if(!estimate_valid ||
	   !optional_device_memory_fits(
		   device, required_mb, future_device_memory_mb)) return false;

	statistics_mean_u = Memory<float>(device, N, 3u, true, true, 0.0f);
	statistics_mean_rho = Memory<float>(device, N, 1u, true, true, 0.0f);
	statistics_M2_u = Memory<float>(device, N, 3u, true, true, 0.0f);
#ifdef TEMPERATURE
	device_statistics_temperature = include_temperature;
	if(include_temperature)
		statistics_mean_T = Memory<float>(device, N, 1u, true, true, 0.0f);
	if(include_temperature) {
		kernel_accumulate_statistics = Kernel(
			device, N, "culbm_accumulate_statistics",
			rho, u, T, statistics_mean_u, statistics_mean_rho,
			statistics_M2_u, statistics_mean_T, 0.0f);
	} else {
		kernel_accumulate_statistics = Kernel(
			device, N, "culbm_accumulate_statistics",
			rho, u, T, statistics_mean_u, statistics_mean_rho,
			statistics_M2_u, static_cast<float*>(nullptr), 0.0f);
	}
#else
	(void)include_temperature;
	kernel_accumulate_statistics = Kernel(
		device, N, "culbm_accumulate_statistics",
		rho, u, statistics_mean_u, statistics_mean_rho,
		statistics_M2_u, 0.0f);
#endif // TEMPERATURE
	device_statistics_active = true;
	return true;
}

void LBM_Domain::disable_device_statistics() {
	device_statistics_active = false;
	device_statistics_temperature = false;
	kernel_accumulate_statistics = Kernel();
	statistics_mean_u = Memory<float>();
	statistics_mean_rho = Memory<float>();
	statistics_M2_u = Memory<float>();
#ifdef TEMPERATURE
	statistics_mean_T = Memory<float>();
#endif // TEMPERATURE
}

void LBM_Domain::reset_device_statistics() {
	if(!device_statistics_active) return;
	statistics_mean_u.reset(0.0f);
	statistics_mean_rho.reset(0.0f);
	statistics_M2_u.reset(0.0f);
#ifdef TEMPERATURE
	if(device_statistics_temperature) statistics_mean_T.reset(0.0f);
#endif // TEMPERATURE
}

void LBM_Domain::enqueue_device_statistics_sample(
	const float inverse_sample_count) {
	if(!device_statistics_active) return;
	if(!(inverse_sample_count > 0.0f) || !std::isfinite(inverse_sample_count))
		print_error("CUDA statistics received an invalid sample count.");
#ifndef UPDATE_FIELDS
	enqueue_update_fields();
#endif // UPDATE_FIELDS
#ifdef TEMPERATURE
	kernel_accumulate_statistics.set_parameters(7u, inverse_sample_count).enqueue_run();
#else
	kernel_accumulate_statistics.set_parameters(5u, inverse_sample_count).enqueue_run();
#endif // TEMPERATURE
}

bool LBM_Domain::configure_device_probe_sampler(
	const vector<ulong>& local_indices,
	const vector<ulong>& output_slots,
	const uint future_device_memory_mb) {
	disable_device_probe_sampler();
	if(local_indices.empty()) return true;
	if(local_indices.size()!=output_slots.size())
		print_error("CUDA probe index/output-slot sizes differ.");
	for(const ulong index : local_indices) {
		if(index>=get_N()) print_error("CUDA probe index is outside its local domain.");
	}
	std::uint64_t required_mb = 0u;
	bool estimate_valid = add_device_allocation_mb(
		required_mb, tracked_device_allocation_mb(
			(std::uint64_t)local_indices.size(), sizeof(ulong)));
	estimate_valid &= add_device_allocation_mb(
		required_mb, tracked_device_allocation_mb(
			(std::uint64_t)local_indices.size(), 3u*sizeof(float)));
	if(!estimate_valid ||
	   !optional_device_memory_fits(
		   device, required_mb, future_device_memory_mb)) return false;
	probe_indices = Memory<ulong>(
		device, (ulong)local_indices.size(), 1u, true, true, 0ull);
	for(ulong i=0ull; i<(ulong)local_indices.size(); ++i)
		probe_indices[i] = local_indices[(size_t)i];
	probe_indices.write_to_device();
	probe_velocity = Memory<float>(
		device, (ulong)local_indices.size(), 3u, true, true, 0.0f);
	probe_output_slots = output_slots;
	kernel_gather_probe_velocity = Kernel(
		device, (ulong)local_indices.size(), "culbm_gather_probe_velocity",
		u, probe_indices, probe_velocity, (ulong)local_indices.size());
	return true;
}

void LBM_Domain::disable_device_probe_sampler() {
	kernel_gather_probe_velocity = Kernel();
	probe_indices = Memory<ulong>();
	probe_velocity = Memory<float>();
	probe_output_slots.clear();
}

void LBM_Domain::enqueue_device_probe_sample() {
	if(probe_output_slots.empty()) return;
#ifndef UPDATE_FIELDS
	enqueue_update_fields();
#endif // UPDATE_FIELDS
	kernel_gather_probe_velocity.enqueue_run();
}
#ifdef SURFACE
void LBM_Domain::enqueue_surface_0() {
	kernel_surface_0.set_parameters(7u, get_t(), fx, fy, fz).enqueue_run();
}
void LBM_Domain::enqueue_surface_1() {
	kernel_surface_1.enqueue_run();
}
void LBM_Domain::enqueue_surface_2() {
	kernel_surface_2.set_parameters(4u, get_t()).enqueue_run();
}
void LBM_Domain::enqueue_surface_3() {
	kernel_surface_3.enqueue_run();
}
#endif // SURFACE
#ifdef FORCE_FIELD
void LBM_Domain::enqueue_update_force_field() { // calculate forces from fluid on TYPE_S cells
	const ulong current_t = get_t();
	if(current_t!=t_last_force_field) { // only run kernel_update_force_field if the time step has changed since last update
		kernel_update_force_field.set_parameters(2u, current_t).enqueue_run();
		t_last_force_field = current_t;
	}
}
void LBM_Domain::enqueue_object_center_of_mass(const uchar flag_marker) { // calculate center of mass of all cells flagged with flag_marker
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_center_of_mass.set_parameters(1u, flag_marker).enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_force(const uchar flag_marker) { // add up force for all cells flagged with flag_marker
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_force.set_parameters(2u, flag_marker).enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_torque(const float3& rotation_center, const uchar flag_marker) { // add up torque around specified rotation_center for all cells flagged with flag_marker
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_torque.set_parameters(2u, flag_marker, rotation_center.x, rotation_center.y, rotation_center.z).enqueue_run();
	object_sum.enqueue_read_from_device();
}
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
void LBM_Domain::enqueue_update_moving_boundaries() { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	kernel_update_moving_boundaries.enqueue_run();
}
#endif // MOVING_BOUNDARIES
#ifdef PARTICLES
void LBM_Domain::enqueue_integrate_particles(const uint time_step_multiplicator) { // intgegrate particles forward in time and couple particles to fluid
#ifdef FORCE_FIELD
	if(particles_rho!=1.0f) kernel_reset_force_field.enqueue_run(); // only reset force field if particles have buoyancy and apply forces on fluid
	kernel_integrate_particles.set_parameters(5u, fx, fy, fz);
#endif // FORCE_FIELD
	kernel_integrate_particles.set_parameters(3u, (float)time_step_multiplicator).enqueue_run();
}
#endif // PARTICLES

void LBM_Domain::increment_time_step(const uint steps) {
	const ulong current_t =
		t.fetch_add((ulong)steps, std::memory_order_acq_rel)+(ulong)steps;
#ifdef UPDATE_FIELDS
	t_last_update_fields = current_t;
#endif // UPDATE_FIELDS
}
void LBM_Domain::reset_time_step() {
	t.store(0ull, std::memory_order_release);
#ifdef UPDATE_FIELDS
	t_last_update_fields = 0ull;
#endif // UPDATE_FIELDS
}
void LBM_Domain::finish_queue() {
	device.finish_queue();
}

uint LBM_Domain::get_velocity_set() const {
	return velocity_set;
}

//void LBM_Domain::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) { // voxelize triangle mesh
//	Memory<float3> p0(device, mesh->triangle_number, 1u, mesh->p0);
//	Memory<float3> p1(device, mesh->triangle_number, 1u, mesh->p1);
//	Memory<float3> p2(device, mesh->triangle_number, 1u, mesh->p2);
//	Memory<float> bounding_box_and_velocity(device, 16u);
//	const float x0=mesh->pmin.x-2.0f, y0=mesh->pmin.y-2.0f, z0=mesh->pmin.z-2.0f, x1=mesh->pmax.x+2.0f, y1=mesh->pmax.y+2.0f, z1=mesh->pmax.z+2.0f; // use bounding box of mesh to speed up voxelization; add tolerance of 2 cells for re-voxelization of moving objects
//	bounding_box_and_velocity[ 0] = as_float(mesh->triangle_number);
//	bounding_box_and_velocity[ 1] = x0;
//	bounding_box_and_velocity[ 2] = y0;
//	bounding_box_and_velocity[ 3] = z0;
//	bounding_box_and_velocity[ 4] = x1;
//	bounding_box_and_velocity[ 5] = y1;
//	bounding_box_and_velocity[ 6] = z1;
//	bounding_box_and_velocity[ 7] = rotation_center.x;
//	bounding_box_and_velocity[ 8] = rotation_center.y;
//	bounding_box_and_velocity[ 9] = rotation_center.z;
//	bounding_box_and_velocity[10] = linear_velocity.x;
//	bounding_box_and_velocity[11] = linear_velocity.y;
//	bounding_box_and_velocity[12] = linear_velocity.z;
//	bounding_box_and_velocity[13] = rotational_velocity.x;
//	bounding_box_and_velocity[14] = rotational_velocity.y;
//	bounding_box_and_velocity[15] = rotational_velocity.z;
//	uint direction = 0u;
//	if(length(rotational_velocity)==0.0f) { // choose direction of minimum bounding-box cross-section area
//		float v[3] = { (y1-y0)*(z1-z0), (z1-z0)*(x1-x0), (x1-x0)*(y1-y0) };
//		float vmin = v[0];
//		for(uint i=1u; i<3u; i++) {
//			if(v[i]<vmin) {
//				vmin = v[i];
//				direction = i;
//			}
//		}
//	} else { // choose direction closest to rotation axis
//		float v[3] = { fabsf(rotational_velocity.x), fabsf(rotational_velocity.y), fabsf(rotational_velocity.z) };
//		float vmax = v[0];
//		for(uint i=1u; i<3u; i++) {
//			if(v[i]>vmax) {
//				vmax = v[i];
//				direction = i; // find direction of minimum bounding-box cross-section area
//			}
//		}
//	}
//	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
//	Kernel kernel_voxelize_mesh(device, A[direction], "voxelize_mesh", direction, fi, u, flags, t+1ull, flag, p0, p1, p2, bounding_box_and_velocity);
//#ifdef SURFACE
//	kernel_voxelize_mesh.add_parameters(mass, massex);
//#endif // SURFACE
//	p0.write_to_device();
//	p1.write_to_device();
//	p2.write_to_device();
//	bounding_box_and_velocity.write_to_device();
//	kernel_voxelize_mesh.run();
//}

// voxelize triangle mesh
void LBM_Domain::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity, const uint direction_override_0, const uint direction_override_1, const vector<uint>* triangle_ids_0, const vector<uint>* triangle_ids_1) {
	(void)direction_override_1;
	(void)triangle_ids_1;
	// use bounding box of mesh to speed up voxelization; add tolerance of 2 cells for re-voxelization of moving objects
	const float x0 = mesh->pmin.x - 2.0f, y0 = mesh->pmin.y - 2.0f, z0 = mesh->pmin.z - 2.0f, x1 = mesh->pmax.x + 2.0f, y1 = mesh->pmax.y + 2.0f, z1 = mesh->pmax.z + 2.0f;
	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
	auto run_voxelize_pass = [&](const uint direction, const uchar pass_flag, const vector<uint>* triangle_ids) {
		if(direction>2u) return;
		if(triangle_ids!=nullptr&&triangle_ids->empty()) return;
		const uint triangle_number = triangle_ids==nullptr ? mesh->triangle_number : (uint)triangle_ids->size();
		if(triangle_number==0u) return;

		vector<float3> p0_subset, p1_subset, p2_subset;
		float3* p0_host = mesh->p0;
		float3* p1_host = mesh->p1;
		float3* p2_host = mesh->p2;
		if(triangle_ids!=nullptr) {
			p0_subset.resize(triangle_number);
			p1_subset.resize(triangle_number);
			p2_subset.resize(triangle_number);
			for(uint i=0u; i<triangle_number; i++) {
				const uint triangle_id = (*triangle_ids)[i];
				p0_subset[i] = mesh->p0[triangle_id];
				p1_subset[i] = mesh->p1[triangle_id];
				p2_subset[i] = mesh->p2[triangle_id];
			}
			p0_host = p0_subset.data();
			p1_host = p1_subset.data();
			p2_host = p2_subset.data();
		}

		Memory<float3> p0(device, triangle_number, 1u, p0_host);
		Memory<float3> p1(device, triangle_number, 1u, p1_host);
		Memory<float3> p2(device, triangle_number, 1u, p2_host);
		Memory<float> bounding_box_and_velocity(device, 16u);
		{
			float triangle_number_as_float;
			std::memcpy(&triangle_number_as_float, &triangle_number, sizeof(uint));
			bounding_box_and_velocity[0] = triangle_number_as_float;
		}
		bounding_box_and_velocity[1] = x0;
		bounding_box_and_velocity[2] = y0;
		bounding_box_and_velocity[3] = z0;
		bounding_box_and_velocity[4] = x1;
		bounding_box_and_velocity[5] = y1;
		bounding_box_and_velocity[6] = z1;
		bounding_box_and_velocity[7] = rotation_center.x;
		bounding_box_and_velocity[8] = rotation_center.y;
		bounding_box_and_velocity[9] = rotation_center.z;
		bounding_box_and_velocity[10] = linear_velocity.x;
		bounding_box_and_velocity[11] = linear_velocity.y;
		bounding_box_and_velocity[12] = linear_velocity.z;
		bounding_box_and_velocity[13] = rotational_velocity.x;
		bounding_box_and_velocity[14] = rotational_velocity.y;
		bounding_box_and_velocity[15] = rotational_velocity.z;
		bounding_box_and_velocity.write_to_device();

		Kernel kernel_voxelize_mesh(device, A[direction], "voxelize_mesh", direction, fi, u, flags, t+1ull, pass_flag, p0, p1, p2, bounding_box_and_velocity);
#ifdef SURFACE
		kernel_voxelize_mesh.add_parameters(mass, massex);
#endif // SURFACE
		kernel_voxelize_mesh.run();
	};

	auto auto_voxelize_direction = [&]() {
		if(length(rotational_velocity)==0.0f) { // choose direction of minimum bounding-box cross-section area
			float v[3] = { (y1-y0)*(z1-z0), (z1-z0)*(x1-x0), (x1-x0)*(y1-y0) };
			float vmin = v[0];
			uint direction = 0u;
			for(uint i=1u; i<3u; i++) {
				if(v[i]<vmin) {
					vmin = v[i];
					direction = i;
				}
			}
			return direction;
		} else { // choose direction closest to rotation axis
			float v[3] = { fabsf(rotational_velocity.x), fabsf(rotational_velocity.y), fabsf(rotational_velocity.z) };
			float vmax = v[0];
			uint direction = 0u;
			for(uint i=1u; i<3u; i++) {
				if(v[i]>vmax) {
					vmax = v[i];
					direction = i;
				}
			}
			return direction;
		}
	};

	if(flag==TYPE_S) {
		const uint direction_0 = direction_override_0==max_uint ? 2u : direction_override_0;
		run_voxelize_pass(direction_0, (uchar)TYPE_S, triangle_ids_0);
		// echo counts after voxelization
		flags.read_from_device();
		ulong solid_cells = 0ull, fluid_cells = 0ull;
		const ulong N_local = flags.length();
		for(ulong n=0ull; n<N_local; n++) {
			const uchar f = flags[n];
			if(f&TYPE_S) solid_cells++;
			else         fluid_cells++;
		}
		print_info(
			"Voxelized cells (single-GPU local domain, includes halos, device "+to_string(device.info.id)+
			"): solid = "+to_string(solid_cells)+", fluid = "+to_string(fluid_cells)+", total = "+to_string(N_local)+"."
		);
	} else {
		uint direction = direction_override_0;
		if(direction==max_uint) direction = auto_voxelize_direction();
		run_voxelize_pass(direction, flag, triangle_ids_0);
	}
}
void LBM_Domain::enqueue_unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag) { // remove voxelized triangle mesh from LBM grid
	const float x0 = mesh->pmin.x, y0 = mesh->pmin.y, z0 = mesh->pmin.z, x1 = mesh->pmax.x, y1 = mesh->pmax.y, z1 = mesh->pmax.z; // remove all flags in bounding box of mesh
	Kernel kernel_unvoxelize_mesh(device, get_N(), "unvoxelize_mesh", flags, flag, x0, y0, z0, x1, y1, z1);
	kernel_unvoxelize_mesh.run();
}

string LBM_Domain::device_defines() const {
	const Subgrid_Configuration subgrid = get_subgrid_configuration();
	const float subgrid_prefactor = 18.0f*sqrt(2.0f)*
		sq(subgrid.smagorinsky_constant);
	const float subgrid_scalar_tau_factor = 1.0f/
		(3.0f*subgrid.turbulent_prandtl*get_temperature_lattice_cs2());
	const uint Nx_global = (Nx-2u*(Dx>1u))*Dx;
	const uint Ny_global = (Ny-2u*(Dy>1u))*Dy;
	const uint Nz_global = (Nz-2u*(Dz>1u))*Dz;
	const int west_local_x = -Ox;
	const int east_local_x = (int)Nx_global-1-Ox;
	const int south_local_y = -Oy;
	const int north_local_y = (int)Ny_global-1-Oy;
	const int top_local_z = (int)Nz_global-1-Oz;
	const int has_west_face = west_local_x>=0&&west_local_x<(int)Nx ? 1 : 0;
	const int has_east_face = east_local_x>=0&&east_local_x<(int)Nx ? 1 : 0;
	const int has_south_face = south_local_y>=0&&south_local_y<(int)Ny ? 1 : 0;
	const int has_north_face = north_local_y>=0&&north_local_y<(int)Ny ? 1 : 0;
	const int has_top_face = top_local_z>=0&&top_local_z<(int)Nz ? 1 : 0;
	return
	"\n	#define def_Nx "+to_string(Nx)+"u"
	"\n	#define def_Ny "+to_string(Ny)+"u"
	"\n	#define def_Nz "+to_string(Nz)+"u"
	"\n	#define def_N "+to_string(get_N())+"ull"
	"\n	#define uxx "+(get_N()<=(ulong)max_uint ? "uint" : "ulong")+""

	"\n	#define def_Dx "+to_string(Dx)+"u"
	"\n	#define def_Dy "+to_string(Dy)+"u"
	"\n	#define def_Dz "+to_string(Dz)+"u"

	"\n	#define def_Ox "+to_string(Ox)+""
	"\n	#define def_Oy "+to_string(Oy)+""
	"\n	#define def_Oz "+to_string(Oz)+""
	"\n	#define def_Nx_global "+to_string(Nx_global)+"u"
	"\n	#define def_Ny_global "+to_string(Ny_global)+"u"
	"\n	#define def_Nz_global "+to_string(Nz_global)+"u"
	"\n	#define def_west_local_x "+to_string(west_local_x)+""
	"\n	#define def_east_local_x "+to_string(east_local_x)+""
	"\n	#define def_south_local_y "+to_string(south_local_y)+""
	"\n	#define def_north_local_y "+to_string(north_local_y)+""
	"\n	#define def_top_local_z "+to_string(top_local_z)+""
	"\n	#define def_has_west_face "+to_string(has_west_face)+""
	"\n	#define def_has_east_face "+to_string(has_east_face)+""
	"\n	#define def_has_south_face "+to_string(has_south_face)+""
	"\n	#define def_has_north_face "+to_string(has_north_face)+""
	"\n	#define def_has_top_face "+to_string(has_top_face)+""

	"\n	#define def_Ax "+to_string(Ny*Nz)+"u"
	"\n	#define def_Ay "+to_string(Nz*Nx)+"u"
	"\n	#define def_Az "+to_string(Nx*Ny)+"u"

	"\n	#define D"+to_string(dimensions)+"Q"+to_string(velocity_set)+""
	"\n	#define def_velocity_set "+to_string(velocity_set)+"u"
	"\n	#define def_dimensions "+to_string(dimensions)+"u"
	"\n	#define def_transfers "+to_string(transfers)+"u"

	"\n	#define def_c 0.57735027f"
	"\n	#define def_tau "+to_string(get_tau())+"f"
	"\n	#define def_w "+to_string(1.0f/get_tau())+"f"
#if defined(TRT)
	"\n	#define def_trt_magic "+to_string(trt_magic_parameter)+"f"
#endif
#if defined(D2Q9)
	"\n	#define def_w0 (1.0f/2.25f)"
	"\n	#define def_ws (1.0f/9.0f)"
	"\n	#define def_we (1.0f/36.0f)"
#elif defined(D3Q15)
	"\n	#define def_w0 (1.0f/4.5f)"
	"\n	#define def_ws (1.0f/9.0f)"
	"\n	#define def_wc (1.0f/72.0f)"
#elif defined(D3Q19)
	"\n	#define def_w0 (1.0f/3.0f)"
	"\n	#define def_ws (1.0f/18.0f)"
	"\n	#define def_we (1.0f/36.0f)"
#elif defined(D3Q27)
	"\n	#define def_w0 (1.0f/3.375f)"
	"\n	#define def_ws (1.0f/13.5f)"
	"\n	#define def_we (1.0f/54.0f)"
	"\n	#define def_wc (1.0f/216.0f)"
#endif

#if defined(SRT)
	"\n	#define SRT"
#elif defined(TRT)
	"\n	#define TRT"
#endif

	"\n	#define TYPE_S 0x01"
	"\n	#define TYPE_E 0x02"
	"\n	#define TYPE_T 0x04"
	"\n	#define TYPE_F 0x08"
	"\n	#define TYPE_I 0x10"
	"\n	#define TYPE_G 0x20"
	"\n	#define TYPE_X 0x40"
	"\n	#define TYPE_Y 0x80"

	"\n	#define TYPE_MS 0x03"
	"\n	#define TYPE_BO 0x03"
	"\n	#define TYPE_IF 0x18"
	"\n	#define TYPE_IG 0x30"
	"\n	#define TYPE_GI 0x38"
	"\n	#define TYPE_SU 0x38"

#if defined(FP16S)
	"\n	#define fpxx half"
	"\n	#define fpxx_copy ushort"
	"\n	#define load(p,o) cuda_vload_half(o,p)*3.0517578E-5f"
	"\n	#define store(p,o,x) cuda_vstore_half_rn((x)*32768.0f,o,p)"
#elif defined(FP16C)
	"\n	#define fpxx ushort"
	"\n	#define fpxx_copy ushort"
	"\n	#define load(p,o) half_to_float_custom(p[o])"
	"\n	#define store(p,o,x) p[o]=float_to_half_custom(x)"
#else
	"\n	#define fpxx float"
	"\n	#define fpxx_copy float"
	"\n	#define load(p,o) p[o]"
	"\n	#define store(p,o,x) p[o]=x"
#endif

#ifdef UPDATE_FIELDS
	"\n	#define UPDATE_FIELDS"
#endif

#ifdef VOLUME_FORCE
	"\n	#define VOLUME_FORCE"
#endif

#ifdef MOVING_BOUNDARIES
	"\n	#define MOVING_BOUNDARIES"
#endif

#ifdef FREE_SLIP_BOUNDARIES
	"\n	#define FREE_SLIP_BOUNDARIES"
#endif

#ifdef EQUILIBRIUM_BOUNDARIES
	"\n	#define EQUILIBRIUM_BOUNDARIES"
#endif

#ifdef FORCE_FIELD
	"\n	#define FORCE_FIELD"
#endif

#ifdef SURFACE
	"\n	#define SURFACE"
	"\n	#define def_6_sigma "+to_string(6.0f*sigma)+"f"
#endif

#ifdef TEMPERATURE
	"\n	#define TEMPERATURE"
	"\n	#define def_temperature_set "+to_string(get_temperature_set())+"u"
	"\n	#define def_temperature_transfers "+to_string(get_temperature_transfers())+"u"
	+ string(get_temperature_lattice()==Temperature_Lattice::Q19
		? "\n	#define TEMPERATURE_D3Q19"
		: "\n	#define TEMPERATURE_D3Q7")
	+ "\n	#define def_w_T "+to_string(1.0f/get_temperature_tau(alpha))+"f"
	"\n	#define def_beta "+to_string(beta)+"f"
	"\n	#define def_T_avg "+to_string(T_avg)+"f"
#endif

	+ (subgrid.enabled
		? "\n	#define SUBGRID"
		  "\n	#define def_subgrid_prefactor "+
			to_string(subgrid_prefactor)+"f"+
		  (subgrid.fixed_eddy_viscosity>=0.0f
			? "\n	#define SUBGRID_FIXED_EDDY_VISCOSITY"
			  "\n	#define def_subgrid_fixed_nu_t "+
				to_string(subgrid.fixed_eddy_viscosity)+"f"
			: string(""))+
		  (subgrid.scalar_coupling
			? "\n	#define SUBGRID_SCALAR"
			  "\n	#define def_subgrid_scalar_tau_factor "+
				to_string(subgrid_scalar_tau_factor)+"f"
			: string(""))
		: string("")) +

#ifdef PARTICLES
	"\n\t#define PARTICLES"
	"\n\t#define def_particles_N " + to_string(particles_N) + "ull"
	"\n\t#define def_particles_rho " + to_string(particles_rho) + "f"
#endif

	"\n\t#define CORIOLIS"
	"\n\t#define def_Omegax " + to_string(coriolis_Omegax_lbmu) + "f"
	"\n\t#define def_Omegay " + to_string(coriolis_Omegay_lbmu) + "f"
	"\n\t#define def_Omegaz " + to_string(coriolis_Omegaz_lbmu) + "f"
	"\n\t#define def_coriolis_f " + to_string(coriolis_f_lbmu) + "f"
	"\n\t#define def_downstream_face " + to_string(buffer_downstream_face_id) + ""
	+ (buffer_nudging_active
		? "\n\t#define BUFFER_NUDGING"
		  "\n\t#define def_buffer_N " + to_string(buffer_n_cells) + "u"
		  "\n\t#define def_buffer_inv_tau " + to_string(buffer_inv_tau_lbmu) + "f"
		  "\n\t#define def_buffer_nudge_vertical " + to_string(buffer_nudge_vertical) + ""
		: string(""))
	+ (top_sponge_active
		? "\n\t#define TOP_SPONGE"
		  "\n\t#define def_sponge_N " + to_string(sponge_n_cells) + "u"
		  "\n\t#define def_sponge_inv_tau " + to_string(sponge_inv_tau_lbmu) + "f"
		  "\n\t#define def_sponge_ref_mode " + to_string(sponge_ref_mode) + ""
		: string(""))
;}

string LBM_Domain::full_cuda_source() const {
#ifdef GRAPHICS
	return device_defines() + graphics.device_defines() + get_cuda_kernel_source();
#else
	return device_defines() + get_cuda_kernel_source();
#endif
}


#ifdef GRAPHICS
void LBM_Domain::Graphics::allocate(Device& device) {
	bitmap = Memory<int>(device, camera.width*camera.height);
	zbuffer = Memory<int>(device, camera.width*camera.height, 1u, lbm->get_D()>1u); // if there are multiple domains, allocate zbuffer also on host side
	camera_parameters = Memory<float>(device, 15u);
	kernel_clear = Kernel(device, bitmap.length(), "graphics_clear", bitmap, zbuffer);

	const bool use_snapshot =
		lbm->get_D()>1u && concurrent_graphics_enabled();
	Memory<float>& render_rho =
		use_snapshot ? lbm->graphics_rho : lbm->rho;
	Memory<float>& render_u =
		use_snapshot ? lbm->graphics_u : lbm->u;
	Memory<uchar>& render_flags =
		use_snapshot ? lbm->graphics_flags : lbm->flags;
	kernel_graphics_flags = Kernel(device, lbm->get_N(), "graphics_flags", camera_parameters, bitmap, zbuffer, render_flags);
	kernel_graphics_flags_mc = Kernel(device, lbm->get_N(), "graphics_flags_mc", camera_parameters, bitmap, zbuffer, render_flags);
	kernel_graphics_field = Kernel(device, lbm->get_D()==1u ? camera.width*camera.height : lbm->get_N(), lbm->get_D()==1u ? "graphics_field_rt" : "graphics_field", camera_parameters, bitmap, zbuffer, 0, render_rho, render_u, render_flags); // raytraced field visualization only works for single-GPU
	kernel_graphics_field_slice = Kernel(device, lbm->get_N(), "graphics_field_slice", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, render_rho, render_u, render_flags);
#ifndef D2Q9
	kernel_graphics_streamline = Kernel(device, (lbm->get_Nx()/GRAPHICS_STREAMLINE_SPARSE)*(lbm->get_Ny()/GRAPHICS_STREAMLINE_SPARSE)*(lbm->get_Nz()/GRAPHICS_STREAMLINE_SPARSE), "graphics_streamline", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, render_rho, render_u, render_flags); // 3D
#else // D2Q9
	kernel_graphics_streamline = Kernel(device, (lbm->get_Nx()/GRAPHICS_STREAMLINE_SPARSE)*(lbm->get_Ny()/GRAPHICS_STREAMLINE_SPARSE), "graphics_streamline", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, render_rho, render_u, render_flags); // 2D
#endif // D2Q9
	kernel_graphics_q = Kernel(device, lbm->get_N(), "graphics_q", camera_parameters, bitmap, zbuffer, 0, render_rho, render_u);

#ifdef FORCE_FIELD
	Memory<float>& render_F =
		use_snapshot ? lbm->graphics_F : lbm->F;
	kernel_graphics_flags.add_parameters(render_F);
	kernel_graphics_flags_mc.add_parameters(render_F);
#endif // FORCE_FIELD

#ifdef SURFACE
	Memory<float>& render_phi =
		use_snapshot ? lbm->graphics_phi : lbm->phi;
	skybox = Memory<int>(device, skybox_image->width()*skybox_image->height(), 1u, skybox_image->data());
	kernel_graphics_rasterize_phi = Kernel(device, lbm->get_N(), "graphics_rasterize_phi", camera_parameters, bitmap, zbuffer, render_phi);
	kernel_graphics_raytrace_phi = Kernel(device, bitmap.length(), "graphics_raytrace_phi", camera_parameters, bitmap, skybox, render_phi, render_flags);
	kernel_graphics_q.add_parameters(render_flags);
#endif // SURFACE

#ifdef TEMPERATURE
	Memory<float>& render_T =
		use_snapshot ? lbm->graphics_T : lbm->T;
	kernel_graphics_field.add_parameters(render_T);
	kernel_graphics_field_slice.add_parameters(render_T);
	kernel_graphics_streamline.add_parameters(render_T);
	kernel_graphics_q.add_parameters(render_T);
#endif // TEMPERATURE

#ifdef PARTICLES
	Memory<float>& render_particles =
		use_snapshot ? lbm->graphics_particles : lbm->particles;
	kernel_graphics_particles = Kernel(device, render_particles.length(), "graphics_particles", camera_parameters, bitmap, zbuffer, render_particles);
#endif // PARTICLES
}

bool LBM_Domain::Graphics::update_camera(
	const float* const camera_override) {
	if(camera_override==nullptr) camera.update_matrix();
	bool change = false;
	for(uint i=0u; i<15u; i++) {
		const float data =
			camera_override==nullptr ? camera.data(i) : camera_override[i];
		change |= (camera_parameters[i]!=data);
		camera_parameters[i] = data;
	}
	return change; // return false if camera parameters remain unchanged
}
bool LBM_Domain::Graphics::enqueue_draw_frame(
	const int visualization_modes, const int field_mode,
	const int slice_mode, const int slice_x, const int slice_y,
	const int slice_z, const bool visualization_change,
	const float* const camera_override) {
	const bool camera_update = update_camera(camera_override);
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
	if(!visualization_change&&!camera_update&&lbm->get_t()==t_last_rendered_frame) return false; // don't render a new frame if the scene hasn't changed since last frame
#endif // INTERACTIVE_GRAPHICS||INTERACTIVE_GRAPHICS_ASCII
	t_last_rendered_frame = lbm->get_t();
	if(camera_update) camera_parameters.enqueue_write_to_device(); // camera_parameters PCIe transfer and kernel_clear execution can happen simulataneously
	kernel_clear.enqueue_run();
	const int sx=slice_x-lbm->Ox, sy=slice_y-lbm->Oy, sz=slice_z-lbm->Oz; // subtract domain offsets
#ifdef SURFACE
	if((visualization_modes&VIS_PHI_RAYTRACE)&&lbm->get_D()==1u) kernel_graphics_raytrace_phi.enqueue_run(); // disable raytracing for multi-GPU (domain decomposition rendering doesn't work for raytracing)
	if(visualization_modes&VIS_PHI_RASTERIZE) kernel_graphics_rasterize_phi.enqueue_run();
#endif // SURFACE
	if(visualization_modes&VIS_FLAG_LATTICE) kernel_graphics_flags.enqueue_run();
	if(visualization_modes&VIS_FLAG_SURFACE) kernel_graphics_flags_mc.enqueue_run();
	if(visualization_modes&VIS_STREAMLINES) kernel_graphics_streamline.set_parameters(3u, field_mode, slice_mode, sx, sy, sz).enqueue_run();
	if(visualization_modes&VIS_Q_CRITERION) kernel_graphics_q.set_parameters(3u, field_mode).enqueue_run();
#ifdef PARTICLES
	if(visualization_modes&VIS_PARTICLES) kernel_graphics_particles.enqueue_run();
#endif // PARTICLES
	if(visualization_modes&VIS_FIELD) {
		switch(slice_mode) { // 0 (no slice), 1 (x), 2 (y), 3 (z), 4 (xz), 5 (xyz), 6 (yz), 7 (xy)
			case 0: // no slice
				kernel_graphics_field.set_parameters(3u, field_mode).enqueue_run();
				break;
			case 1: case 2: case 3: // x/y/z
				kernel_graphics_field_slice.set_ranges(lbm->get_area((uint)clamp(slice_mode-1, 0, 2))).set_parameters(3u, field_mode, slice_mode, sx, sy, sz).enqueue_run();
				break;
			case 4: // xz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 5: // xyz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 6: // yz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 7: // xy
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				break;
		}
	}
	bitmap.enqueue_read_from_device();
	if(lbm->get_D()>1u) zbuffer.enqueue_read_from_device();
	return true; // new frame has been rendered
}
int* LBM_Domain::Graphics::get_bitmap() { // returns pointer to zbuffer
	return bitmap.data();
}
int* LBM_Domain::Graphics::get_zbuffer() { // returns pointer to zbuffer
	return zbuffer.data();
}

string LBM_Domain::Graphics::device_defines() const { return
	"\n	#define GRAPHICS"
	"\n	#define def_background_color " +to_string(GRAPHICS_BACKGROUND_COLOR)+""
	"\n	#define def_screen_width "     +to_string(camera.width)+"u"
	"\n	#define def_screen_height "    +to_string(camera.height)+"u"
	"\n	#define def_scale_u "          +to_string(1.0f/(0.57735027f*(GRAPHICS_U_MAX)))+"f"
	"\n	#define def_scale_rho "        +to_string(0.5f/(GRAPHICS_RHO_DELTA))+"f"
	"\n	#define def_scale_T "          +to_string(0.5f/(GRAPHICS_T_DELTA))+"f"
	"\n	#define def_scale_F "          +to_string(0.5f/(GRAPHICS_F_MAX))+"f"
	"\n	#define def_scale_Q_min "      +to_string(GRAPHICS_Q_CRITERION)+"f"
	"\n	#define def_streamline_sparse "+to_string(GRAPHICS_STREAMLINE_SPARSE)+"u"
	"\n	#define def_streamline_length "+to_string(GRAPHICS_STREAMLINE_LENGTH)+"u"
	"\n	#define def_n "                +to_string(1.333f)+"f" // refractive index of water for raytracing graphics
	"\n	#define def_attenuation "      +to_string(ln(clamp(GRAPHICS_RAYTRACING_TRANSMITTANCE, 1E-9f, 1.0f))/(float)max(max(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz()))+"f" // (negative) attenuation parameter for raytracing graphics
	"\n	#define def_absorption_color " +to_string(GRAPHICS_RAYTRACING_COLOR)+"" // absorption color of fluid for raytracing graphics

	"\n	#define COLOR_S (127<<16|127<<8|127)" // (stationary or moving) solid boundary
	"\n	#define COLOR_E (  0<<16|255<<8|  0)" // equilibrium boundary (inflow/outflow)
	"\n	#define COLOR_M (255<<16|  0<<8|255)" // cells next to moving solid boundary
	"\n	#define COLOR_T (255<<16|  0<<8|  0)" // temperature boundary
	"\n	#define COLOR_F (  0<<16|  0<<8|255)" // fluid
	"\n	#define COLOR_I (  0<<16|255<<8|255)" // interface
	"\n	#define COLOR_0 (127<<16|127<<8|127)" // regular cell or gas
	"\n	#define COLOR_X (255<<16|127<<8|  0)" // reserved type X
	"\n	#define COLOR_Y (255<<16|255<<8|  0)" // reserved type Y
	"\n	#define COLOR_P (255<<16|255<<8|191)" // particles

#ifdef GRAPHICS_TRANSPARENCY
	"\n	#define GRAPHICS_TRANSPARENCY "+to_string(GRAPHICS_TRANSPARENCY)+"f"
#endif // GRAPHICS_TRANSPARENCY

#ifndef SURFACE
	"\n	#define def_skybox_width 1u"
	"\n	#define def_skybox_height 1u"
#else // SURFACE
	"\n	#define def_skybox_width " +to_string(skybox_image->width() )+"u"
	"\n	#define def_skybox_height "+to_string(skybox_image->height())+"u"
#endif // SURFACE
;}
#endif // GRAPHICS



bool peer_topology_ordering_disabled() {
	const char* const p2p_disabled = std::getenv("CULBM_DISABLE_P2P");
	if(p2p_disabled!=nullptr && std::strcmp(p2p_disabled, "0")!=0)
		return true;
	const char* const value = std::getenv("CULBM_P2P_TOPOLOGY");
	return value!=nullptr && (
		std::strcmp(value, "0")==0 ||
		std::strcmp(value, "off")==0 ||
		std::strcmp(value, "OFF")==0 ||
		std::strcmp(value, "false")==0 ||
		std::strcmp(value, "FALSE")==0);
}

void optimize_device_mapping_for_peer_topology(
	vector<Device_Info>& device_infos,
	const uint Nx, const uint Ny, const uint Nz,
	const uint Dx, const uint Dy, const uint Dz) {
	const uint device_count = (uint)device_infos.size();
	if(device_count<2u || peer_topology_ordering_disabled()) return;
	vector<unsigned char> bidirectional_peer(
		(size_t)device_count*(size_t)device_count, 0u);
	for(uint source=0u; source<device_count; ++source) {
		for(uint destination=0u; destination<device_count; ++destination) {
			if(source==destination) {
				bidirectional_peer[(size_t)source*device_count+destination] = 1u;
				continue;
			}
			int forward = 0;
			int reverse = 0;
			const bool accessible =
				cuDeviceCanAccessPeer(
					&forward, device_infos[source].cu_device,
					device_infos[destination].cu_device)==CUDA_SUCCESS &&
				cuDeviceCanAccessPeer(
					&reverse, device_infos[destination].cu_device,
					device_infos[source].cu_device)==CUDA_SUCCESS &&
				forward!=0 && reverse!=0;
			bidirectional_peer[(size_t)source*device_count+destination] =
				accessible ? 1u : 0u;
		}
	}
	vector<uint> domain_to_device(device_count);
	for(uint domain=0u; domain<device_count; ++domain)
		domain_to_device[domain] = domain;
	const std::array<std::uint64_t, 3u> face_weights = {
		(std::uint64_t)(Ny/Dy)*(std::uint64_t)(Nz/Dz),
		(std::uint64_t)(Nx/Dx)*(std::uint64_t)(Nz/Dz),
		(std::uint64_t)(Nx/Dx)*(std::uint64_t)(Ny/Dy)
	};
	const std::uint64_t initial_score = culbm::peer_assignment_score(
		bidirectional_peer, device_count, domain_to_device,
		Dx, Dy, Dz, face_weights);
	std::uint64_t optimized_score = initial_score;
	if(!culbm::optimize_peer_device_assignment(
		   bidirectional_peer, device_count, domain_to_device,
		   Dx, Dy, Dz, face_weights, &optimized_score)) return;

	vector<Device_Info> reordered(device_count);
	string mapping = "";
	for(uint domain=0u; domain<device_count; ++domain) {
		reordered[domain] = device_infos[domain_to_device[domain]];
		if(domain>0u) mapping += ", ";
		mapping += "d"+to_string(domain)+"->GPU"+
			to_string(reordered[domain].id);
	}
	device_infos.swap(reordered);
	print_info(
		"CUDA P2P topology reordered selected devices ("+mapping+
		"), weighted direct-peer halo score "+to_string(initial_score)+
		" -> "+to_string(optimized_score)+".");
}

vector<Device_Info> smart_device_selection(const uint D) {
	const vector<Device_Info>& devices = get_devices(false); // all available CUDA devices
	vector<Device_Info> device_infos(D);
	if(devices.empty()) print_error("No CUDA devices found.");
	vector<Device_Info> candidate_devices;
	candidate_devices.reserve(devices.size());
	for(uint i=0u; i<(uint)devices.size(); i++) {
		if(devices[i].is_gpu) candidate_devices.push_back(devices[i]);
	}
	if(candidate_devices.empty()) {
		print_warning("No GPU CUDA devices found. Falling back to all CUDA devices for domain assignment.");
		candidate_devices = devices;
	}
	const uint available_cards = (uint)candidate_devices.size();
	if(D>available_cards) {
		print_info(
			"Requested domain partitions: "+to_string(D)+
			", available physical cards: "+to_string(available_cards)+"."
		);
		for(uint i=0u; i<available_cards; i++) {
			const Device_Info& di = candidate_devices[i];
			print_info(
				"Available card "+to_string(i)+": id="+to_string(di.id)+
				", model=\""+di.name+"\", vendor=\""+di.vendor+
				"\", memory="+to_string(di.memory)+" MB."
			);
		}
		print_error(
			"Domain partition count ("+to_string(D)+
			") exceeds available physical cards ("+to_string(available_cards)+
			"). Reduce Dx*Dy*Dz to <= "+to_string(available_cards)+"."
		);
	}
	// Device ID selection via CLI is disabled; always auto-select devices.
	vector<vector<Device_Info>> device_type_ids; // a vector of all different devices, containing vectors of their device IDs
	for(uint i=0u; i<(uint)candidate_devices.size(); i++) {
		const string name_i = candidate_devices[i].name;
		bool already_exists = false;
		for(uint j=0u; j<(uint)device_type_ids.size(); j++) {
			const string name_j = device_type_ids[j][0].name;
			if(name_i==name_j) {
				device_type_ids[j].push_back(candidate_devices[i]);
				already_exists = true;
			}
		}
		if(!already_exists) device_type_ids.push_back(vector<Device_Info>(1, candidate_devices[i]));
	}
	float best_value = -1.0f;
	int best_j = -1;
	for(uint j=0u; j<(uint)device_type_ids.size(); j++) {
		const float value = device_type_ids[j][0].tflops;
		if((uint)device_type_ids[j].size()>=D && value>best_value) {
			best_value = value;
			best_j = j;
		}
	}
	if(best_j>=0) { // select all devices of fastest device type with at least D devices of the same type
		for(uint d=0; d<D; d++) device_infos[d] = device_type_ids[best_j][d];
	} else {
		print_warning(
			"Not enough identical devices for "+to_string(D)+
			" domains. Using mixed models without oversubscription."
		);
		vector<char> used(candidate_devices.size(), false);
		for(uint d=0u; d<D; d++) {
			int best_i = -1;
			for(uint i=0u; i<(uint)candidate_devices.size(); i++) {
				if(used[i]) continue;
				if(best_i<0) {
					best_i = (int)i;
					continue;
				}
				const Device_Info& a = candidate_devices[i];
				const Device_Info& b = candidate_devices[(uint)best_i];
				if(a.tflops>b.tflops ||
				   (a.tflops==b.tflops && a.memory>b.memory) ||
				   (a.tflops==b.tflops && a.memory==b.memory && a.id<b.id)) {
					best_i = (int)i;
				}
			}
			if(best_i<0) print_error("Internal device selection error.");
			used[(uint)best_i] = true;
			device_infos[d] = candidate_devices[(uint)best_i];
		}
	}
	//for(uint j=0u; j<(uint)device_type_ids.size(); j++) print_info("Device Type "+to_string(j)+" ("+device_type_ids[j][0].name+"): "+to_string((uint)device_type_ids[j].size())+"x");
	return device_infos;
}

LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) // single device
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho)
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, fx, fy, fz, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const uint particles_N, const float particles_rho)
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, Dx, Dy, Dz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) // single device
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, fx, fy, fz, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // local multi-GPU or one MPI domain per rank
#ifdef GRAPHICS
	static std::atomic<std::uint64_t> next_graphics_session = 1ull;
	graphics_session_id = next_graphics_session.fetch_add(
		1ull, std::memory_order_acq_rel);
	graphics_active_session.store(
		graphics_session_id, std::memory_order_release);
#endif // GRAPHICS
	const uint NDx=(Nx/Dx)*Dx, NDy=(Ny/Dy)*Dy, NDz=(Nz/Dz)*Dz; // make resolution equally divisible by domains
	if(NDx!=Nx||NDy!=Ny||NDz!=Nz) print_warning("LBM grid ("+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+") is not equally divisible in domains ("+to_string(Dx)+"x"+to_string(Dy)+"x"+to_string(Dz)+"). Changing resolution to ("+to_string(NDx)+"x"+to_string(NDy)+"x"+to_string(NDz)+").");
	this->Nx = NDx; this->Ny = NDy; this->Nz = NDz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	const uint mem_per_dev_mb = vram_required_mb_per_device(this->Nx, this->Ny, this->Nz, Dx, Dy, Dz);
	const uint mem_total_mb = vram_required_mb_total(this->Nx, this->Ny, this->Nz, Dx, Dy, Dz);
	print_info("Required VRAM: " + to_string(Dx * Dy * Dz) + "x " + to_string(mem_per_dev_mb) + " MB = " + to_string(mem_total_mb) + " MB total for grid " + to_string(this->Nx) + "x" + to_string(this->Ny) + "x" + to_string(this->Nz) + ".");
	const uint global_D = Dx*Dy*Dz;
	const uint Hx=Dx>1u, Hy=Dy>1u, Hz=Dz>1u; // halo offsets
	culbm::mpi_validate_domain_count(global_D);
	const bool distributed = culbm::mpi_is_distributed();
	distributed_mode = distributed;
	vector<Device_Info> device_infos;
	if(distributed) {
		const vector<Device_Info>& visible_devices = get_devices(false);
		const uint ordinal = culbm::mpi_cuda_device_ordinal((uint)visible_devices.size());
		device_infos.push_back(visible_devices[ordinal]);
		local_D = 1u;
		global_domain_id = (uint)culbm::mpi_rank();
		if(culbm::mpi_is_root()) print_info(
			"MPI distributed mode: "+to_string((uint)culbm::mpi_size())+
			" rank(s), one Cartesian domain and one CUDA GPU per rank.");
	} else {
		device_infos = smart_device_selection(global_D);
		optimize_device_mapping_for_peer_topology(
			device_infos, this->Nx, this->Ny, this->Nz, Dx, Dy, Dz);
		local_D = global_D;
		global_domain_id = 0u;
	}
	sanity_checks_constructor(device_infos, this->Nx, this->Ny, this->Nz, Dx, Dy, Dz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	lbm_domain = new LBM_Domain*[local_D];
	for(uint local_domain=0u; local_domain<local_D; local_domain++) {
		const uint d = distributed ? global_domain_id : local_domain;
		const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy); // d = x+(y+z*Dy)*Dx
		lbm_domain[local_domain] = new LBM_Domain(device_infos[local_domain], this->Nx/Dx+2u*Hx, this->Ny/Dy+2u*Hy, this->Nz/Dz+2u*Hz, Dx, Dy, Dz, (int)(x*this->Nx/Dx)-(int)Hx, (int)(y*this->Ny/Dy)-(int)Hy, (int)(z*this->Nz/Dz)-(int)Hz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	}
	{
		Memory<float>** buffers_rho = new Memory<float>*[local_D];
		for(uint d=0u; d<local_D; d++) buffers_rho[d] = &(lbm_domain[d]->rho);
		rho = Memory_Container(this, buffers_rho, "rho");
	} {
		Memory<float>** buffers_u = new Memory<float>*[local_D];
		for(uint d=0u; d<local_D; d++) buffers_u[d] = &(lbm_domain[d]->u);
		u = Memory_Container(this, buffers_u, "u");
	} {
		Memory<uchar>** buffers_flags = new Memory<uchar>*[local_D];
		for(uint d=0u; d<local_D; d++) buffers_flags[d] = &(lbm_domain[d]->flags);
		flags = Memory_Container(this, buffers_flags, "flags");
	} {
#ifdef FORCE_FIELD
		Memory<float>** buffers_F = new Memory<float>*[local_D];
		for(uint d=0u; d<local_D; d++) buffers_F[d] = &(lbm_domain[d]->F);
		F = Memory_Container(this, buffers_F, "F");
#endif // FORCE_FIELD
	} {
#ifdef SURFACE
		Memory<float>** buffers_phi = new Memory<float>*[local_D];
		for(uint d=0u; d<local_D; d++) buffers_phi[d] = &(lbm_domain[d]->phi);
		phi = Memory_Container(this, buffers_phi, "phi");
#endif // SURFACE
	} {
#ifdef TEMPERATURE
		Memory<float>** buffers_T = new Memory<float>*[local_D];
		for(uint d=0u; d<local_D; d++) buffers_T[d] = &(lbm_domain[d]->T);
		T = Memory_Container(this, buffers_T, "T");
#endif // TEMPERATURE
	} {
#ifdef PARTICLES
		particles = &(lbm_domain[0]->particles);
#endif // PARTICLES
	}
#ifdef GRAPHICS
	graphics = Graphics(this);
#endif // GRAPHICS
	active_lbm_instances.fetch_add(1u, std::memory_order_acq_rel);
}
LBM::~LBM() {
#ifdef GRAPHICS
	std::lock_guard<std::mutex> frame_lock(camera.rendring_frame);
	camera.allow_rendering.store(false, std::memory_order_release);
	if(graphics_active_session.load(std::memory_order_acquire)==
	   graphics_session_id)
		graphics_active_session.store(0ull, std::memory_order_release);
#endif // GRAPHICS
	info.print_finalize();
	for(uint d=0u; d<get_D(); d++) delete lbm_domain[d];
	delete[] lbm_domain;
	active_lbm_instances.fetch_sub(1u, std::memory_order_acq_rel);
}

void LBM::sanity_checks_constructor(const vector<Device_Info>& device_infos, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // sanity checks on grid resolution and extension support
	if((ulong)Nx*(ulong)Ny*(ulong)Nz==0ull) print_error("Grid point number is 0: "+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+" = 0.");
	if(Dx*Dy*Dz==0u) print_error("You specified 0 LBM grid domains ("+to_string(Dx)+"x"+to_string(Dy)+"x"+to_string(Dz)+"). There has to be at least 1 domain in every direction. Check your input in LBM constructor.");
	const uint local_Nx=Nx/Dx+2u*(Dx>1u), local_Ny=Ny/Dy+2u*(Dy>1u), local_Nz=Nz/Dz+2u*(Dz>1u);
	uint memory_available = max_uint; // in MB
	for(Device_Info device_info : device_infos) memory_available = min(memory_available, device_info.memory);
	uint memory_required = vram_required_mb_per_device(Nx, Ny, Nz, Dx, Dy, Dz);
	if(memory_required>memory_available) {
		float factor = cbrt((float)memory_available/(float)memory_required);
		const uint maxNx=(uint)(factor*(float)Nx), maxNy=(uint)(factor*(float)Ny), maxNz=(uint)(factor*(float)Nz);
		string message = "Grid resolution ("+to_string(Nx)+", "+to_string(Ny)+", "+to_string(Nz)+") is too large: "+to_string(Dx*Dy*Dz)+"x "+to_string(memory_required)+" MB required, "+to_string(Dx*Dy*Dz)+"x "+to_string(memory_available)+" MB available. Largest possible resolution is ("+to_string(maxNx)+", "+to_string(maxNy)+", "+to_string(maxNz)+"). Restart the simulation with lower resolution or on different device(s) with more memory.";
#if !defined(FP16S)&&!defined(FP16C)
		uint memory_required_fp16 = (uint)((ulong)Nx*(ulong)Ny*(ulong)Nz/((ulong)(Dx*Dy*Dz))*(ulong)(bytes_per_cell_device()-velocity_set*2u)/1048576ull); // in MB
		float factor_fp16 = cbrt((float)memory_available/(float)memory_required_fp16);
		const uint maxNx_fp16=(uint)(factor_fp16*(float)Nx), maxNy_fp16=(uint)(factor_fp16*(float)Ny), maxNz_fp16=(uint)(factor_fp16*(float)Nz);
		message += " Consider using FP16S/FP16C memory compression to double maximum grid resolution to a maximum of ("+to_string(maxNx_fp16)+", "+to_string(maxNy_fp16)+", "+to_string(maxNz_fp16)+"); for this, uncomment \"#define FP16S\" or \"#define FP16C\" in defines.hpp.";
#endif // !FP16S&&!FP16C
		print_error(message);
	}
	if(nu==0.0f) print_error("Viscosity cannot be 0. Change it in setup.cpp."); // sanity checks for viscosity
	else if(nu<0.0f) print_error("Viscosity cannot be negative. Remove the \"-\" in setup.cpp.");
#ifdef D2Q9
	if(Nz!=1u) print_error("D2Q9 is the 2D velocity set. You have to set Nz=1u in the LBM constructor! Currently you have set Nz="+to_string(Nz)+"u.");
#endif // D2Q9
#if !defined(SRT)&&!defined(TRT)
	print_error("No LBM collision operator selected. Uncomment either \"#define SRT\" or \"#define TRT\" in defines.hpp");
#elif defined(SRT)&&defined(TRT)
	print_error("Too many LBM collision operators selected. Comment out either \"#define SRT\" or \"#define TRT\" in defines.hpp");
#endif // SRT && TRT
#ifndef VOLUME_FORCE
	if(fx!=0.0f||fy!=0.0f||fz!=0.0f) print_error("Volume force is set in LBM constructor in main_setup(), but VOLUME_FORCE is not enabled. Uncomment \"#define VOLUME_FORCE\" in defines.hpp.");
#else // VOLUME_FORCE
#ifndef FORCE_FIELD
	if(fx==0.0f&&fy==0.0f&&fz==0.0f) print_warning("The VOLUME_FORCE extension is enabled but the volume force in LBM constructor is set to zero. You may disable the extension by commenting out \"#define VOLUME_FORCE\" in defines.hpp.");
#endif // FORCE_FIELD
#endif // VOLUME_FORCE
#ifndef SURFACE
	if(sigma!=0.0f) print_error("Surface tension is set in LBM constructor in main_setup(), but SURFACE is not enabled. Uncomment \"#define SURFACE\" in defines.hpp.");
#endif // SURFACE
#ifndef TEMPERATURE
	if(alpha!=0.0f||beta!=0.0f) print_error("Thermal diffusion/expansion coefficients are set in LBM constructor in main_setup(), but TEMPERATURE is not enabled. Uncomment \"#define TEMPERATURE\" in defines.hpp.");
#else // TEMPERATURE
	if(alpha<0.0f) print_error("Thermal diffusion coefficient alpha cannot be negative.");
	if(alpha==0.0f)
		print_warning("Thermal diffusion coefficient alpha is zero, so tau_T is at the marginal value 0.5.");
	if(alpha==0.0f&&beta==0.0f) print_warning("The TEMPERATURE extension is enabled but the thermal diffusion/expansion coefficients alpha/beta in the LBM constructor are both set to zero. You may disable the extension by commenting out \"#define TEMPERATURE\" in defines.hpp.");
#endif // TEMPERATURE
#ifdef PARTICLES
	if(particles_N==0u) print_error("The PARTICLES extension is enabled but the number of particles is set to 0. Comment out \"#define PARTICLES\" in defines.hpp.");
	if(Dx*Dy*Dz>1u) print_error("The PARTICLES extension is not supported in multi-GPU/MPI mode.");
#if !defined(VOLUME_FORCE)||!defined(FORCE_FIELD)
	if(particles_rho!=1.0f) print_error("Particle density is set unequal to 1, but particle-fluid 2-way-coupling is not enabled. Uncomment both \"#define VOLUME_FORCE\" and \"#define FORCE_FIELD\" in defines.hpp.");
#endif // !VOLUME_FORCE||!FORCE_FIELD
#ifdef FORCE_FIELD
	if(particles_rho==1.0f) print_warning("Particle density is set to 1, so particles behave as passive tracers without acting a force on the fluid, but particle-fluid 2-way-coupling is enabled. You may comment out \"#define FORCE_FIELD\" in defines.hpp.");
#endif // FORCE_FIELD
#else // PARTICLES
	if(particles_N>0u) print_error("The PARTICLES extension is disabled but the number of particles is set to "+to_string(particles_N)+">0. Uncomment \"#define PARTICLES\" in defines.hpp.");
#endif // PARTICLES
}

void LBM::sanity_checks_initialization() { // sanity checks during initialization on used extensions based on used flags
	uchar flags_used = 0u;
	bool moving_boundaries_used=false, equilibrium_boundaries_used=false, surface_used=false, temperature_used=false; // identify used extensions based used flags
	const uint threads = thread::hardware_concurrency();
	vector<uchar> t_flags_used(threads, 0u);
	vector<char> t_moving_boundaries_used(threads, false); // don't use vector<bool> as it uses bit-packing which is broken for multithreading
	vector<char> t_equilibrium_boundaries_used(threads, false); // don't use vector<bool> as it uses bit-packing which is broken for multithreading
	parallel_for(get_N(), threads, [&](ulong n, uint t) {
		const uchar flagsn = flags[n];
		const uchar flagsn_bo = flagsn&(TYPE_S|TYPE_E);
		t_flags_used[t] = t_flags_used[t]|flagsn;
		if(flagsn_bo&TYPE_S) t_moving_boundaries_used[t] = t_moving_boundaries_used[t] || (((flagsn_bo==TYPE_S)&&(u.x[n]!=0.0f||u.y[n]!=0.0f||u.z[n]!=0.0f))||(flagsn_bo==(TYPE_S|TYPE_E)));
		t_equilibrium_boundaries_used[t] = t_equilibrium_boundaries_used[t] || flagsn_bo==TYPE_E;
	});
	for(uint t=0u; t<threads; t++) {
		flags_used = flags_used|t_flags_used[t];
		moving_boundaries_used = moving_boundaries_used || t_moving_boundaries_used[t];
		equilibrium_boundaries_used = equilibrium_boundaries_used || t_equilibrium_boundaries_used[t];
	}
	surface_used = (bool)(flags_used&(TYPE_F|TYPE_I|TYPE_G));
	temperature_used = (bool)(flags_used&TYPE_T);
#ifndef MOVING_BOUNDARIES
	if(moving_boundaries_used) print_warning("Some boundary cells have non-zero velocity, but MOVING_BOUNDARIES is not enabled. If you intend to use moving boundaries, uncomment \"#define MOVING_BOUNDARIES\" in defines.hpp.");
#else // MOVING_BOUNDARIES
	if(!moving_boundaries_used) print_warning("The MOVING_BOUNDARIES extension is enabled but no moving boundary cells (TYPE_S flag and velocity unequal to zero) are placed in the simulation box. You may disable the extension by commenting out \"#define MOVING_BOUNDARIES\" in defines.hpp.");
#endif // MOVING_BOUNDARIES
#ifndef EQUILIBRIUM_BOUNDARIES
	if(equilibrium_boundaries_used) print_error("Some cells are set as equilibrium boundaries with the TYPE_E flag, but EQUILIBRIUM_BOUNDARIES is not enabled. Uncomment \"#define EQUILIBRIUM_BOUNDARIES\" in defines.hpp.");
#else // EQUILIBRIUM_BOUNDARIES
	if(!equilibrium_boundaries_used) print_warning("The EQUILIBRIUM_BOUNDARIES extension is enabled but no equilibrium boundary cells (TYPE_E flag) are placed in the simulation box. You may disable the extension by commenting out \"#define EQUILIBRIUM_BOUNDARIES\" in defines.hpp.");
#endif // EQUILIBRIUM_BOUNDARIES
#ifndef SURFACE
	if(surface_used) print_error("Some cells are set as fluid/interface/gas with the TYPE_F/TYPE_I/TYPE_G flags, but SURFACE is not enabled. Uncomment \"#define SURFACE\" in defines.hpp.");
#else // SURFACE
	if(!surface_used) print_error("The SURFACE extension is enabled but no fluid/interface/gas cells (TYPE_F/TYPE_I/TYPE_G flags) are placed in the simulation box. Disable the extension by commenting out \"#define SURFACE\" in defines.hpp.");
#endif // SURFACE
#ifndef TEMPERATURE
	if(temperature_used) print_error("Some cells are set as temperature boundary with the TYPE_T flag, but TEMPERATURE is not enabled. Uncomment \"#define TEMPERATURE\" in defines.hpp.");
#endif // TEMPERATURE
}

void LBM::initialize() { // write all data fields to device and call kernel_initialize
#ifndef BENCHMARK
	sanity_checks_initialization();
#endif // BENCHMARK

	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->rho.enqueue_write_to_device();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->u.enqueue_write_to_device();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->flags.enqueue_write_to_device();
#ifdef FORCE_FIELD
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->F.enqueue_write_to_device();
#endif // FORCE_FIELD
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->phi.enqueue_write_to_device();
#endif // SURFACE
#ifdef TEMPERATURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->T.enqueue_write_to_device();
#endif // TEMPERATURE
#ifdef PARTICLES
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->particles.enqueue_write_to_device();
#endif // PARTICLES

	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step(); // the communicate calls at initialization need an odd time step
	communicate_rho_u_flags();
#ifdef SURFACE
	communicate_phi_massex_flags();
#endif // SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_initialize(); // odd time step is baked-in the kernel
	communicate_rho_u_flags();
#ifdef SURFACE
	communicate_phi_massex_flags();
#endif // SURFACE
#ifdef TEMPERATURE
	communicate_T(); // T halo data is required for field_slice rendering
	communicate_fi_gi(); // fi/gi time step must be odd here
#else // TEMPERATURE
	communicate_fi(); // time step must be odd here
#endif // TEMPERATURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->reset_time_step(); // set time step to 0 again
	initialized = true;
	ever_initialized = true;
}

void LBM::do_time_step() { // call kernel_stream_collide to perform one LBM time step
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_0();
#endif // SURFACE
#ifdef TEMPERATURE
	for(uint d=0u; d<get_D(); d++)
		lbm_domain[d]->enqueue_update_temperature_outlet();
#endif // TEMPERATURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_stream_collide(); // run LBM stream_collide kernel after domain communication
#ifdef SURFACE
	communicate_rho_u_flags(); // rho/u/flags halo data is required for SURFACE extension
#endif // SURFACE
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_1();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_2();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_3();
	communicate_phi_massex_flags();
#endif // SURFACE
#ifdef TEMPERATURE
	communicate_fi_gi();
#else // TEMPERATURE
	communicate_fi();
#endif // TEMPERATURE
#ifdef PARTICLES
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_integrate_particles(); // intgegrate particles forward in time and couple particles to fluid
#endif // PARTICLES
	if(get_D()==1u) for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // this additional domain synchronization barrier is only required in single-GPU, as communication calls already provide all necessary synchronization barriers in multi-GPU
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step();
}

void LBM::run(const ulong steps, const ulong total_steps) { // initializes the LBM simulation (copies data to device and runs initialize kernel), then runs LBM
	info.append(steps, total_steps, get_t()); // total_steps parameter is just for runtime estimation
	if(!initialized) {
		initialize();
		info.print_initialize(this); // only print setup info if the setup is new (run() was not called before)
#ifdef GRAPHICS
		camera.allow_rendering = true;
#endif // GRAPHICS
	}
	Clock clock;
	for(ulong i=1ull; i<=steps; i++) {
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
		if(!interactive_simulation_may_advance())
			throw Interactive_Simulation_Stop();
#endif // INTERACTIVE_GRAPHICS_ASCII || INTERACTIVE_GRAPHICS
		clock.start();
		do_time_step();
		info.update(clock.stop());
	}
	if(get_D()>1u) for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // wait for everything to finish (multi-GPU only)
}

void LBM::finish_case() {
#ifdef GRAPHICS
	{
		// Do not hold this mutex across the MPI barrier: another rank's
		// graphics thread may need it to finish the current frame.
		std::lock_guard<std::mutex> frame_lock(camera.rendring_frame);
		camera.allow_rendering.store(false, std::memory_order_release);
	}
#endif // GRAPHICS
	if(is_distributed()) culbm::mpi_barrier();
}

void LBM::external_initialize() {
	if(!initialized) initialize();
}

void LBM::external_update_free_slip_boundaries(const uint face_mask) {
	external_initialize();
	for(uint d=0u; d<get_D(); ++d)
		lbm_domain[d]->enqueue_update_free_slip_boundaries(face_mask);
}

void LBM::external_step() {
	external_initialize();
	do_time_step();
}

void LBM::external_finish() {
	for(uint d=0u; d<get_D(); ++d) lbm_domain[d]->finish_queue();
}

void LBM::external_synchronize_flags() {
	communicate_flags();
	for(uint d=0u; d<get_D(); ++d) lbm_domain[d]->finish_queue();
}

void LBM::update_fields() { // update fields (rho, u, T) manually
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_fields();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}

Subgrid_Statistics LBM::subgrid_statistics() {
	Subgrid_Statistics statistics;
	if(!initialized || !get_subgrid_configuration().enabled) return statistics;
	for(uint d=0u; d<get_D(); ++d)
		lbm_domain[d]->enqueue_subgrid_statistics();

	double sums[2] = { 0.0, 0.0 };
	float maxima[2] = { 0.0f, 0.0f };
	std::uint64_t count = 0u;
	for(uint d=0u; d<get_D(); ++d) {
		const Subgrid_Statistics local =
			lbm_domain[d]->read_subgrid_statistics();
		if(!local.valid) continue;
		sums[0] += local.mean_nu_t*(double)local.count;
		sums[1] += local.mean_rho_nu_t*(double)local.count;
		maxima[0] = std::max(maxima[0], local.max_nu_t);
		maxima[1] = std::max(maxima[1], local.max_rho_nu_t);
		count += local.count;
	}
	if(is_distributed()) {
		culbm::mpi_allreduce_sum_double(sums, 2u);
		culbm::mpi_allreduce_max_float(maxima, 2u);
		culbm::mpi_allreduce_sum_uint64(&count, 1u);
	}
	statistics.count = count;
	statistics.valid = count>0u;
	if(!statistics.valid) return statistics;
	statistics.mean_nu_t = sums[0]/(double)count;
	statistics.max_nu_t = maxima[0];
	statistics.mean_rho_nu_t = sums[1]/(double)count;
	statistics.max_rho_nu_t = maxima[1];
	return statistics;
}

void LBM::prepare_advection_fields() {
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_fields();
	if(get_global_D()>1u) communicate_rho_u_flags();
	else for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}

bool LBM::enable_device_statistics(
	const bool include_temperature,
	const uint future_device_memory_mb) {
	disable_device_statistics();
	const char* const override_value = std::getenv("CULBM_DEVICE_STATISTICS");
	if(override_value!=nullptr && (
		std::strcmp(override_value, "0")==0 ||
		std::strcmp(override_value, "off")==0 ||
		std::strcmp(override_value, "OFF")==0 ||
		std::strcmp(override_value, "false")==0 ||
		std::strcmp(override_value, "FALSE")==0)) return false;
	for(uint d=0u; d<get_D(); ++d) {
		if(lbm_domain[d]->enable_device_statistics(
			   include_temperature, future_device_memory_mb)) continue;
		for(uint rollback=0u; rollback<=d; ++rollback)
			lbm_domain[rollback]->disable_device_statistics();
		return false;
	}
	return get_D()>0u;
}

void LBM::disable_device_statistics() {
	for(uint d=0u; d<get_D(); ++d)
		lbm_domain[d]->disable_device_statistics();
}

void LBM::reset_device_statistics() {
	for(uint d=0u; d<get_D(); ++d)
		lbm_domain[d]->reset_device_statistics();
}

void LBM::enqueue_device_statistics_sample(const ulong sample_count) {
	if(!uses_device_statistics()) return;
	if(sample_count==0ull)
		print_error("CUDA statistics sample count must be positive.");
	const float inverse_sample_count = 1.0f/(float)sample_count;
	for(uint d=0u; d<get_D(); ++d)
		lbm_domain[d]->enqueue_device_statistics_sample(inverse_sample_count);
}

void LBM::finish_device_statistics() {
	for(uint d=0u; d<get_D(); ++d) lbm_domain[d]->finish_queue();
}

void LBM::read_device_statistics(
	vector<float>& mean_u,
	vector<float>& mean_rho,
	vector<float>& mean_T,
	vector<float>& M2_u,
	vector<float>& M2_v,
	vector<float>& M2_w) {
	if(!uses_device_statistics()) {
		print_error("CUDA statistics were requested before being enabled.");
	}
	vector<Memory<float>*> mean_u_buffers(get_D());
	vector<Memory<float>*> mean_rho_buffers(get_D());
	vector<Memory<float>*> M2_u_buffers(get_D());
	for(uint d=0u; d<get_D(); ++d) {
		mean_u_buffers[d] = &lbm_domain[d]->get_statistics_mean_u();
		mean_rho_buffers[d] = &lbm_domain[d]->get_statistics_mean_rho();
		M2_u_buffers[d] = &lbm_domain[d]->get_statistics_M2_u();
	}
	Memory_Container<float> mean_u_view(this, mean_u_buffers.data(), "statistics_mean_u");
	Memory_Container<float> mean_rho_view(this, mean_rho_buffers.data(), "statistics_mean_rho");
	Memory_Container<float> M2_u_view(this, M2_u_buffers.data(), "statistics_M2_u");
	mean_u_view.read_from_device();
	mean_rho_view.read_from_device();
	M2_u_view.read_from_device();

	const ulong N = get_N();
	mean_u.resize((size_t)(3ull*N));
	mean_rho.resize((size_t)N);
	M2_u.resize((size_t)N);
	M2_v.resize((size_t)N);
	M2_w.resize((size_t)N);
	parallel_for(N, [&](const ulong n) {
		const ulong n3 = 3ull*n;
		mean_u[(size_t)n3] = mean_u_view.x[n];
		mean_u[(size_t)(n3+1ull)] = mean_u_view.y[n];
		mean_u[(size_t)(n3+2ull)] = mean_u_view.z[n];
		mean_rho[(size_t)n] = mean_rho_view[n];
		M2_u[(size_t)n] = M2_u_view.x[n];
		M2_v[(size_t)n] = M2_u_view.y[n];
		M2_w[(size_t)n] = M2_u_view.z[n];
	});

	mean_T.clear();
#ifdef TEMPERATURE
	if(lbm_domain[0]->has_device_statistics_temperature()) {
		vector<Memory<float>*> mean_T_buffers(get_D());
		for(uint d=0u; d<get_D(); ++d)
			mean_T_buffers[d] = &lbm_domain[d]->get_statistics_mean_T();
		Memory_Container<float> mean_T_view(this, mean_T_buffers.data(), "statistics_mean_T");
		mean_T_view.read_from_device();
		mean_T.resize((size_t)N);
		parallel_for(N, [&](const ulong n) {
			mean_T[(size_t)n] = mean_T_view[n];
		});
	}
#endif // TEMPERATURE
}

bool LBM::configure_device_velocity_probes(
	const vector<ulong>& global_indices,
	const uint future_device_memory_mb) {
	disable_device_velocity_probes();
	if(global_indices.empty() || is_distributed()) return false;
	const uint Nx_inner = Nx/Dx, Ny_inner = Ny/Dy, Nz_inner = Nz/Dz;
	const uint Hx = Dx>1u ? 1u : 0u;
	const uint Hy = Dy>1u ? 1u : 0u;
	const uint Hz = Dz>1u ? 1u : 0u;
	vector<vector<ulong>> local_indices(get_D());
	vector<vector<ulong>> output_slots(get_D());
	for(ulong slot=0ull; slot<(ulong)global_indices.size(); ++slot) {
		const ulong global = global_indices[(size_t)slot];
		if(global>=get_N()) {
			disable_device_velocity_probes();
			return false;
		}
		const ulong xy = (ulong)Nx*(ulong)Ny;
		const uint x = (uint)(global%(ulong)Nx);
		const uint y = (uint)((global/(ulong)Nx)%(ulong)Ny);
		const uint z = (uint)(global/xy);
		const uint domain_x = x/Nx_inner;
		const uint domain_y = y/Ny_inner;
		const uint domain_z = z/Nz_inner;
		const uint domain = domain_x+(domain_y+domain_z*Dy)*Dx;
		const ulong local = (ulong)(x%Nx_inner+Hx)+
			((ulong)(y%Ny_inner+Hy)+(ulong)(z%Nz_inner+Hz)*
			 (ulong)(Ny_inner+2u*Hy))*(ulong)(Nx_inner+2u*Hx);
		local_indices[domain].push_back(local);
		output_slots[domain].push_back(slot);
	}
	for(uint d=0u; d<get_D(); ++d) {
		if(lbm_domain[d]->configure_device_probe_sampler(
			   local_indices[d], output_slots[d],
			   future_device_memory_mb)) continue;
		disable_device_velocity_probes();
		return false;
	}
	return uses_device_velocity_probes();
}

void LBM::disable_device_velocity_probes() {
	for(uint d=0u; d<get_D(); ++d)
		lbm_domain[d]->disable_device_probe_sampler();
}

bool LBM::uses_device_velocity_probes() const {
	for(uint d=0u; d<get_D(); ++d) {
		if(lbm_domain[d]->has_device_probe_sampler()) return true;
	}
	return false;
}

void LBM::sample_device_velocity_probes(vector<float>& samples) {
	if(!uses_device_velocity_probes()) {
		print_error("CUDA probe sampler was requested before being configured.");
	}
	ulong sample_count = 0ull;
	for(uint d=0u; d<get_D(); ++d) {
		for(const ulong slot : lbm_domain[d]->get_device_probe_output_slots())
			sample_count = max(sample_count, slot+(ulong)1u);
		lbm_domain[d]->enqueue_device_probe_sample();
	}
	samples.assign((size_t)(3ull*sample_count), 0.0f);
	for(uint d=0u; d<get_D(); ++d) {
		if(!lbm_domain[d]->has_device_probe_sampler()) continue;
		Memory<float>& values = lbm_domain[d]->get_device_probe_velocity();
		values.read_from_device();
		const ulong local_count = values.length();
		const vector<ulong>& slots = lbm_domain[d]->get_device_probe_output_slots();
		for(ulong i=0ull; i<local_count; ++i) {
			const ulong slot = slots[(size_t)i];
			const ulong output = 3ull*slot;
			samples[(size_t)output] = values.x[i];
			samples[(size_t)(output+1ull)] = values.y[i];
			samples[(size_t)(output+2ull)] = values.z[i];
		}
	}
}

void LBM::reset() { // reset simulation (takes effect in following run() call)
	initialized = false;
}

#ifdef FORCE_FIELD
void LBM::update_force_field() { // calculate forces from fluid on TYPE_S cells
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_force_field();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
float3 LBM::object_center_of_mass(const uchar flag_marker) { // calculate center of mass of all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_center_of_mass(flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_com = float3(0.0f, 0.0f, 0.0f);
	ulong object_cells = 0ull;
	for(uint d=0u; d<get_D(); d++) {
		object_com += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
		object_cells += (ulong)as_uint(lbm_domain[d]->object_sum.w[0]);
	}
	float object_values[3] = { object_com.x, object_com.y, object_com.z };
	uint64_t object_cells_global = (uint64_t)object_cells;
	culbm::mpi_allreduce_sum_float(object_values, 3u);
	culbm::mpi_allreduce_sum_uint64(&object_cells_global, 1u);
	object_com = float3(object_values[0], object_values[1], object_values[2]);
	return object_com/(float)object_cells_global;
}
float3 LBM::object_force(const uchar flag_marker) { // add up force for all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_force(flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_force = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_force += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	float object_values[3] = { object_force.x, object_force.y, object_force.z };
	culbm::mpi_allreduce_sum_float(object_values, 3u);
	return float3(object_values[0], object_values[1], object_values[2]);
}
float3 LBM::object_torque(const float3& rotation_center, const uchar flag_marker) { // add up torque around specified rotation center for all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_torque(rotation_center, flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_torque = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_torque += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	float object_values[3] = { object_torque.x, object_torque.y, object_torque.z };
	culbm::mpi_allreduce_sum_float(object_values, 3u);
	return float3(object_values[0], object_values[1], object_values[2]);
}
#endif // FORCE_FIELD

#ifdef MOVING_BOUNDARIES
void LBM::update_moving_boundaries() { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_moving_boundaries();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
#ifdef GRAPHICS
	camera.key_update = true; // to prevent flickering of flags in interactive graphics when camera is not moved
#endif // GRAPHICS
}
#endif // MOVING_BOUNDARIES

#if defined(PARTICLES)&&!defined(FORCE_FIELD)
void LBM::integrate_particles(const ulong steps, const ulong total_steps, const uint time_step_multiplicator) { // intgegrate passive tracer particles forward in time in stationary flow field
	info.append(steps, total_steps, get_t());
	Clock clock;
	for(ulong i=1ull; i<=steps; i+=(ulong)time_step_multiplicator) {
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
		if(!interactive_simulation_may_advance())
			throw Interactive_Simulation_Stop();
#endif // INTERACTIVE_GRAPHICS_ASCII || INTERACTIVE_GRAPHICS
		clock.start();
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_integrate_particles(time_step_multiplicator);
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step(time_step_multiplicator);
		info.update(clock.stop());
	}
}
#endif // PARTICLES&&!FORCE_FIELD

void LBM::write_status(const string& path) { // write LBM status report to a .txt file
	if(!culbm::mpi_is_root()) return;
	string status = "";
	status += "Grid Resolution = "+to_string(Nx)+" x "+to_string(Ny)+" x "+to_string(Nz)+" = "+to_string(get_N())+"\n";
	status += "Grid Domains = "+to_string(Dx)+" x "+to_string(Dy)+" x "+to_string(Dz)+" = "+to_string(get_global_D())+"\n";
	status += "LBM Type = D"+string(get_velocity_set()==9 ? "2" : "3")+"Q"+to_string(get_velocity_set())+" "+info.collision+"\n";
	status += "Memory Usage = CPU "+to_string(info.cpu_mem_required)+" MB/rank, GPU "+to_string(get_global_D())+"x "+to_string(info.gpu_mem_required)+" MB\n";
	status += "Maximum Allocation Size = "+to_string((uint)(get_N()/(ulong)get_global_D()*(ulong)(get_velocity_set()*sizeof(fpxx))/1048576ull))+" MB\n";
	status += "Time Steps = "+to_string(get_t())+" / "+(info.steps==max_ulong ? "infinite" : to_string(info.steps))+"\n";
	status += "Runtime = "+print_time(info.runtime_total)+" (total) = "+print_time(info.runtime_lbm)+" (LBM) + "+print_time(info.runtime_total-info.runtime_lbm)+" (rendering and data evaluation)\n";
	status += "Average MLUPs/s = "+to_string(to_uint(1E-6*(double)get_N()*(double)get_t()/info.runtime_lbm))+"\n";
	status += "Kinematic Viscosity = "+to_string(get_nu())+"\n";
	status += "Relaxation Time = "+to_string(get_tau())+"\n";
	status += "Maximum Reynolds Number = "+to_string(get_Re_max())+"\n";
#ifdef VOLUME_FORCE
	status += "Volume Force = ("+to_string(get_fx())+", "+to_string(get_fy())+", "+to_string(get_fz())+")\n";
#endif // VOLUME_FORCE
#ifdef SURFACE
	status += "Surface Tension Coefficient = "+to_string(get_sigma())+"\n";
#endif // SURFACE
#ifdef TEMPERATURE
	status += "Thermal Lattice = "+string(get_temperature_lattice_name())+"\n";
	status += "Thermal Diffusion Coefficient = "+to_string(get_alpha())+"\n";
	status += "Thermal Relaxation Time = "+to_string(get_temperature_tau(get_alpha()))+"\n";
	status += "Thermal Expansion Coefficient = "+to_string(get_beta())+"\n";
	const float3 gravity = get_buoyancy_gravity();
	status += "Buoyancy Gravity = ("+to_string(gravity.x)+", "+
		to_string(gravity.y)+", "+to_string(gravity.z)+")\n";
#endif // TEMPERATURE
	const string filename = default_filename(path, "status", ".txt", get_t());
	write_file(filename, status);
}

void LBM::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) { // voxelize triangle mesh
	const uint domain_count = get_D();
	luw_emit_progress("voxelization",
					  "Voxelizing geometry",
					  to_string(mesh->triangle_number)+" triangles across "+to_string(domain_count)+" domain(s)",
					  0ll,
					  (long long)domain_count,
					  false);
	vector<TriangleAABB> triangle_bounds;
	if(domain_count>1u) {
		triangle_bounds.resize(mesh->triangle_number);
		parallel_for(mesh->triangle_number, [&](uint i) {
			triangle_bounds[i] = triangle_aabb(mesh->p0[i], mesh->p1[i], mesh->p2[i]);
		});
	}

	uint direction_0 = 0u;
	if(flag==TYPE_S) {
		// Keep TYPE_S on z for consistency and deterministic behavior.
		direction_0 = 2u;
	} else if(length(rotational_velocity)==0.0f) { // choose direction of minimum bounding-box cross-section area
		const float x0 = mesh->pmin.x-2.0f, y0 = mesh->pmin.y-2.0f, z0 = mesh->pmin.z-2.0f;
		const float x1 = mesh->pmax.x+2.0f, y1 = mesh->pmax.y+2.0f, z1 = mesh->pmax.z+2.0f;
		float v[3] = { (y1-y0)*(z1-z0), (z1-z0)*(x1-x0), (x1-x0)*(y1-y0) };
		float vmin = v[0];
		for(uint i=1u; i<3u; i++) {
			if(v[i]<vmin) {
				vmin = v[i];
				direction_0 = i;
			}
		}
	} else { // choose direction closest to rotation axis
		float v[3] = { fabsf(rotational_velocity.x), fabsf(rotational_velocity.y), fabsf(rotational_velocity.z) };
		float vmax = v[0];
		for(uint i=1u; i<3u; i++) {
			if(v[i]>vmax) {
				vmax = v[i];
				direction_0 = i;
			}
		}
	}
	print_info("Voxelize direction selected="+to_string(direction_0)+".");

	const float overlap_pad = 1.0f;
	const float overlap_eps = 1.0e-4f;
	const bool print_subset_summary = domain_count>1u && length(linear_velocity)==0.0f && length(rotational_velocity)==0.0f;
	vector<uint> subset_triangles_pass0(domain_count, 0u);
	vector<std::uint64_t> voxelize_add_mb_per_domain(domain_count, 0u);
	vector<vector<uint>> triangle_ids_per_domain(domain_count);
	const uint NxDx = Nx/Dx, NyDy = Ny/Dy, NzDz = Nz/Dz;
	const uint Hx = Dx>1u, Hy = Dy>1u, Hz = Dz>1u;
	const uint local_Nx = NxDx+2u*Hx, local_Ny = NyDy+2u*Hy, local_Nz = NzDz+2u*Hz;
	auto collect_domain_triangle_ids = [&](const uint d) {
		const uint domain_x = (d%(Dx*Dy))%Dx;
		const uint domain_y = (d%(Dx*Dy))/Dx;
		const uint domain_z = d/(Dx*Dy);
		const int Ox = (int)(domain_x*NxDx)-(int)Hx;
		const int Oy = (int)(domain_y*NyDy)-(int)Hy;
		const int Oz = (int)(domain_z*NzDz)-(int)Hz;
		const float x_min = (float)Ox;
		const float y_min = (float)Oy;
		const float z_min = (float)Oz;
		const float x_max = (float)(Ox+(int)local_Nx-1);
		const float y_max = (float)(Oy+(int)local_Ny-1);
		const float z_max = (float)(Oz+(int)local_Nz-1);
		const DomainProjectedBounds bounds_0 = make_projected_bounds(direction_0, x_min, x_max, y_min, y_max, z_min, z_max);

		vector<uint> triangle_ids_0;
		triangle_ids_0.reserve(mesh->triangle_number/domain_count+1u);
		for(uint i=0u; i<mesh->triangle_number; i++) {
			if(overlaps_projected(triangle_bounds[i], bounds_0, direction_0, overlap_pad, overlap_eps)) triangle_ids_0.push_back(i);
		}
		subset_triangles_pass0[d] = (uint)triangle_ids_0.size();
		triangle_ids_per_domain[d].swap(triangle_ids_0);
	};

	if(domain_count==1u) {
		// Single domain: skip subset collection/copy and use original mesh buffers directly.
		subset_triangles_pass0[0u] = mesh->triangle_number;
		triangle_ids_per_domain[0u].clear();
	} else {
		parallel_for((ulong)domain_count, domain_count, [&](ulong d) {
			collect_domain_triangle_ids((uint)d);
		});
	}

	// Preflight the four temporary CUDA allocations using the same
	// per-allocation MiB rounding as Memory<T>.
	for(uint d=0u; d<domain_count; d++) {
		const Device_Info& di = lbm_domain[d]->get_device().info;
		const uint triangle_count = subset_triangles_pass0[d];
		const std::uint64_t add_mb =
			voxelization_transient_vram_mb_per_device(triangle_count);
		voxelize_add_mb_per_domain[d] = add_mb;
		const double predicted_total_mb =
			(double)di.memory_used+(double)add_mb;
		print_info(
			"Voxelize preflight d="+to_string(d)+
			" (gpu "+to_string(di.id)+"): triangles="+to_string(triangle_count)+
			", intent alloc="+to_string((double)add_mb/1024.0, 3u)+" GB"+
			", predicted total="+to_string(predicted_total_mb/1024.0, 3u)+" / "+to_string((double)di.memory/1024.0, 3u)+" GB"
		);
		const double p0_mb =
			(double)triangle_count*(double)sizeof(float3)/1048576.0;
		if(p0_mb>(double)di.max_global_buffer) {
			print_warning(
				"Voxelize preflight d="+to_string(d)+": single triangle buffer intends "+to_string(p0_mb/1024.0, 3u)+
				" GB, exceeds CL max alloc "+to_string((double)di.max_global_buffer/1024.0, 3u)+" GB."
			);
		}
		if(predicted_total_mb>(double)di.memory) {
			print_warning(
				"Voxelize preflight d="+to_string(d)+": predicted total allocation exceeds device memory ("+
				to_string(predicted_total_mb/1024.0, 3u)+" / "+to_string((double)di.memory/1024.0, 3u)+" GB)."
			);
		} else if(predicted_total_mb>0.95*(double)di.memory) {
			print_warning(
				"Voxelize preflight d="+to_string(d)+": predicted total allocation is above 95% memory ("+
				to_string(predicted_total_mb/1024.0, 3u)+" / "+to_string((double)di.memory/1024.0, 3u)+" GB)."
			);
		}
	}
	vector<uint> gpu_ids;
	vector<vector<uint>> domains_per_gpu;
	for(uint d=0u; d<domain_count; d++) {
		const uint gpu_id = lbm_domain[d]->get_device().info.id;
		uint g = 0u;
		for(; g<(uint)gpu_ids.size(); g++) {
			if(gpu_ids[g]==gpu_id) break;
		}
		if(g==(uint)gpu_ids.size()) {
			gpu_ids.push_back(gpu_id);
			domains_per_gpu.push_back(vector<uint>());
		}
		domains_per_gpu[g].push_back(d);
	}
	for(uint g=0u; g<(uint)gpu_ids.size(); g++) {
		const vector<uint>& domains = domains_per_gpu[g];
		if(domains.empty()) continue;
		const Device_Info& di = lbm_domain[domains[0]]->get_device().info;
		std::uint64_t base_sum_mb = 0u;
		std::uint64_t temp_sum_mb = 0u;
		std::uint64_t temp_max_mb = 0u;
		string domain_list = "";
		for(uint i=0u; i<(uint)domains.size(); i++) {
			const uint d = domains[i];
			base_sum_mb +=
				(std::uint64_t)lbm_domain[d]->get_device().info.memory_used;
			temp_sum_mb += voxelize_add_mb_per_domain[d];
			temp_max_mb = std::max(
				temp_max_mb, voxelize_add_mb_per_domain[d]);
			domain_list += (i==0u ? "" : ",")+to_string(d);
		}
		const std::uint64_t predicted_seq_mb =
			base_sum_mb+temp_max_mb; // domains sharing a GPU run sequentially
		const std::uint64_t predicted_parallel_mb =
			base_sum_mb+temp_sum_mb;
		print_info(
			"Voxelize preflight GPU "+to_string(di.id)+": domains={"+domain_list+"}, intent total="+
			to_string((double)predicted_seq_mb/1024.0, 3u)+" / "+to_string((double)di.memory/1024.0, 3u)+
			" GB (base="+to_string((double)base_sum_mb/1024.0, 3u)+" GB, temp seq="+to_string((double)temp_max_mb/1024.0, 3u)+
			" GB, temp parallel="+to_string((double)temp_sum_mb/1024.0, 3u)+" GB)."
		);
		if(predicted_seq_mb>(std::uint64_t)di.memory) {
			print_warning(
				"Voxelize preflight GPU "+to_string(di.id)+": predicted sequential peak exceeds memory ("+
				to_string((double)predicted_seq_mb/1024.0, 3u)+" / "+to_string((double)di.memory/1024.0, 3u)+" GB)."
			);
		} else if((double)predicted_seq_mb>0.90*(double)di.memory) {
			print_warning(
				"Voxelize preflight GPU "+to_string(di.id)+": predicted sequential peak is above 90% ("+
				to_string((double)predicted_seq_mb/1024.0, 3u)+" / "+to_string((double)di.memory/1024.0, 3u)+" GB)."
			);
		}
		const Device& selected_device =
			lbm_domain[domains[0]]->get_device();
		const std::uint64_t free_bytes =
			(std::uint64_t)selected_device.get_free_memory_bytes();
		const std::uint64_t total_bytes =
			(std::uint64_t)selected_device.get_total_memory_bytes();
		const std::uint64_t reserve_bytes =
			std::min<std::uint64_t>(256u*device_mib, total_bytes/10u);
		const std::uint64_t temp_bytes =
			temp_max_mb > std::numeric_limits<std::uint64_t>::max()/device_mib
				? std::numeric_limits<std::uint64_t>::max()
				: temp_max_mb*device_mib;
		if(temp_bytes>free_bytes ||
		   free_bytes-temp_bytes<reserve_bytes) {
			print_error(
				"Voxelization on GPU "+to_string(di.id)+" needs "+
				to_string((ulong)temp_max_mb)+
				" MB of temporary triangle buffers, but only "+
				to_string((ulong)(free_bytes/device_mib))+
				" MB is free and "+
				to_string((ulong)(reserve_bytes/device_mib))+
				" MB must remain reserved. Reduce the mesh/grid size.");
		}
	}

	auto launch_voxelize_domain = [&](const uint d) {
		const vector<uint>* triangle_ids_0 = nullptr;
		if(domain_count>1u) {
			triangle_ids_0 = &triangle_ids_per_domain[d];
			if(triangle_ids_per_domain[d].size()==mesh->triangle_number) triangle_ids_0 = nullptr; // avoid full-size host subset copy
		}
		lbm_domain[d]->voxelize_mesh_on_device(mesh, flag, rotation_center, linear_velocity, rotational_velocity, direction_0, max_uint, triangle_ids_0, nullptr);
	};
	std::atomic_uint completed_voxel_domains{0u};
	auto launch_and_report_voxelize_domain = [&](const uint d) {
		launch_voxelize_domain(d);
		const uint completed = completed_voxel_domains.fetch_add(1u)+1u;
		luw_emit_progress("voxelization",
						  "Voxelizing geometry",
						  "Finished domain "+to_string((ulong)completed)+"/"+to_string((ulong)domain_count),
						  (long long)completed,
						  (long long)domain_count,
						  false);
	};
	if(domain_count==1u) {
		launch_and_report_voxelize_domain(0u);
	} else if(gpu_ids.size()<=1u) { // single physical GPU with multiple domains: launch sequentially to minimize transient peak
		for(uint d=0u; d<domain_count; d++) launch_and_report_voxelize_domain(d);
	} else {
		parallel_for((ulong)gpu_ids.size(), gpu_ids.size(), [&](ulong g) {
			const vector<uint>& domains = domains_per_gpu[(uint)g];
			for(uint i=0u; i<(uint)domains.size(); i++) launch_and_report_voxelize_domain(domains[i]);
		});
	}

	if(print_subset_summary) {
		ulong pass0_sum = 0ull;
		uint pass0_min = subset_triangles_pass0[0], pass0_max = subset_triangles_pass0[0];
		for(uint d=0u; d<domain_count; d++) {
			const uint n0 = subset_triangles_pass0[d];
			pass0_sum += (ulong)n0;
			pass0_min = min(pass0_min, n0);
			pass0_max = max(pass0_max, n0);
		}
		const float mb_factor = 36.0f/1048576.0f; // 3 arrays * float3
		print_info(
			"Voxelize pass "+to_string(direction_0)+" triangle subsets: min="+to_string(pass0_min)+", max="+to_string(pass0_max)+", sum="+to_string(pass0_sum)+", d0="+to_string(subset_triangles_pass0[0])+
			", est peak MB/GPU="+to_string((float)pass0_max*mb_factor, 2u)
		);
	}
	if(flag==TYPE_S) {
		ulong solid_global = 0ull, fluid_global = 0ull;
		const ulong N_global = flags.length();
		for(ulong n=0ull; n<N_global; n++) {
			if(flags[n]&TYPE_S) solid_global++;
			else                fluid_global++;
		}
		print_info(
			"Voxelized cells (whole domain global, no halos): solid = "+to_string(solid_global)+
			", fluid = "+to_string(fluid_global)+", total = "+to_string(N_global)+"."
		);
	}
#ifdef MOVING_BOUNDARIES
	if((flag&(TYPE_S|TYPE_E))==TYPE_S&&(length(linear_velocity)>0.0f||length(rotational_velocity)>0.0f)) update_moving_boundaries();
#endif // MOVING_BOUNDARIES
	if(!initialized) {
		flags.read_from_device();
		u.read_from_device();
	}
}
void LBM::unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag) { // remove voxelized triangle mesh from LBM grid by removing all flags in mesh bounding box (only required when bounding box size changes during re-voxelization)
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_unvoxelize_mesh_on_device(mesh, flag);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
void LBM::write_mesh_to_vtk(const Mesh* mesh, const string& path, const bool convert_to_si_units) const { // write mesh to binary .vtk file
	if(!culbm::mpi_is_root()) return;
	const string filename = default_filename(path, "mesh", ".vtk", get_t());
	const string header_1 = "# vtk DataFile Version 3.0\nCULBM "+filename.substr(filename.rfind('/')+1)+"\nBINARY\nDATASET POLYDATA\nPOINTS "+to_string(3u*mesh->triangle_number)+" float\n";
	const string header_2 = "POLYGONS "+to_string(mesh->triangle_number)+" "+to_string(4u*mesh->triangle_number)+"\n";
	float* points = new float[9u*mesh->triangle_number];
	int* triangles = new int[4u*mesh->triangle_number];
	const float spacing = convert_to_si_units ? units.si_x(1.0f) : 1.0f;
	const float3 offset = center();
	parallel_for(mesh->triangle_number, [&](uint i) {
		points[9u*i   ] = reverse_bytes(spacing*(mesh->p0[i].x-offset.x));
		points[9u*i+1u] = reverse_bytes(spacing*(mesh->p0[i].y-offset.y));
		points[9u*i+2u] = reverse_bytes(spacing*(mesh->p0[i].z-offset.z));
		points[9u*i+3u] = reverse_bytes(spacing*(mesh->p1[i].x-offset.x));
		points[9u*i+4u] = reverse_bytes(spacing*(mesh->p1[i].y-offset.y));
		points[9u*i+5u] = reverse_bytes(spacing*(mesh->p1[i].z-offset.z));
		points[9u*i+6u] = reverse_bytes(spacing*(mesh->p2[i].x-offset.x));
		points[9u*i+7u] = reverse_bytes(spacing*(mesh->p2[i].y-offset.y));
		points[9u*i+8u] = reverse_bytes(spacing*(mesh->p2[i].z-offset.z));
		triangles[4u*i   ] = reverse_bytes(3); // 3 vertices per triangle
		triangles[4u*i+1u] = reverse_bytes(3*(int)i  ); // vertex 0
		triangles[4u*i+2u] = reverse_bytes(3*(int)i+1); // vertex 1
		triangles[4u*i+3u] = reverse_bytes(3*(int)i+2); // vertex 2
	});
	create_folder(filename);
	std::ofstream file(filename, std::ios::out|std::ios::binary);
	file.write(header_1.c_str(), header_1.length()); // write non-binary file header
	file.write((char*)points, 4u*9u*mesh->triangle_number); // write binary data
	file.write(header_2.c_str(), header_2.length()); // write non-binary file header
	file.write((char*)triangles, 4u*4u*mesh->triangle_number); // write binary data
	file.close();
	delete[] points;
	delete[] triangles;
	info.allow_printing.lock();
	print_info("File \""+filename+"\" saved.");
	info.allow_printing.unlock();
}
void LBM::voxelize_stl(const string& path, const float3& center, const float3x3& rotation, const float size, const uchar flag) { // voxelize triangle mesh
	const Mesh* mesh = read_stl(path, this->size(), center, rotation, size);
	flags.write_to_device();
	voxelize_mesh_on_device(mesh, flag);
	delete mesh;
	flags.read_from_device();
}
void LBM::voxelize_stl(const string& path, const float3x3& rotation, const float size, const uchar flag) { // read and voxelize binary .stl file (place in box center)
	voxelize_stl(path, center(), rotation, size, flag);
}
void LBM::voxelize_stl(const string& path, const float3& center, const float size, const uchar flag) { // read and voxelize binary .stl file (no rotation)
	voxelize_stl(path, center, float3x3(1.0f), size, flag);
}
void LBM::voxelize_stl(const string& path, const float size, const uchar flag) { // read and voxelize binary .stl file (place in box center, no rotation)
	voxelize_stl(path, center(), float3x3(1.0f), size, flag);
}

#ifdef GRAPHICS
int* LBM::Graphics::draw_frame() {
	if(key_1) { visualization_modes = (visualization_modes&~0b11)|(((visualization_modes&0b11)+1)%4); key_1 = false; }
	if(key_2) { visualization_modes ^= VIS_FIELD        ; key_2 = false; }
	if(key_3) { visualization_modes ^= VIS_STREAMLINES  ; key_3 = false; }
	if(key_4) { visualization_modes ^= VIS_Q_CRITERION  ; key_4 = false; }
	if(key_5) { visualization_modes ^= VIS_PHI_RASTERIZE; key_5 = false; }
	if(key_6) { visualization_modes ^= VIS_PHI_RAYTRACE ; key_6 = false; }
	if(key_7) { visualization_modes ^= VIS_PARTICLES    ; key_7 = false; }
	if(key_T) {
		slice_mode = (slice_mode+1)%8; key_T = false;
	}
	if(key_Z) {
#ifndef TEMPERATURE
		field_mode = (field_mode+1)%2; key_Z = false; // field_mode = { 0 (u), 1 (rho) }
#else // TEMPERATURE
		field_mode = (field_mode+1)%3; key_Z = false; // field_mode = { 0 (u), 1 (rho), 2 (T) }
#endif // TEMPERATURE
	}
	if(slice_mode==1u) {
		if(key_Q) { slice_x = clamp(slice_x-1, 0, (int)lbm->get_Nx()-1); key_Q = false; }
		if(key_E) { slice_x = clamp(slice_x+1, 0, (int)lbm->get_Nx()-1); key_E = false; }
	}
	if(slice_mode==2u) {
		if(key_Q) { slice_y = clamp(slice_y-1, 0, (int)lbm->get_Ny()-1); key_Q = false; }
		if(key_E) { slice_y = clamp(slice_y+1, 0, (int)lbm->get_Ny()-1); key_E = false; }
	}
	if(slice_mode==3u) {
		if(key_Q) { slice_z = clamp(slice_z-1, 0, (int)lbm->get_Nz()-1); key_Q = false; }
		if(key_E) { slice_z = clamp(slice_z+1, 0, (int)lbm->get_Nz()-1); key_E = false; }
	}

	struct Distributed_Graphics_Frame {
		uint64_t frame_id;
		uint64_t session_id;
		uint32_t width, height;
		int visualization_modes, field_mode, slice_mode;
		int slice_x, slice_y, slice_z;
		uint32_t visualization_change;
		uint32_t refresh_snapshot;
		uint32_t refresh_rho_u_halo;
		uint32_t refresh_T_halo;
		float camera_data[15];
	};

	const ulong current_time = lbm->get_t();
	bool visualization_change =
		camera.key_update||
		last_visualization_modes!=visualization_modes||
		last_field_mode!=field_mode||
		last_slice_mode!=slice_mode||
		last_slice_x!=slice_x||
		last_slice_y!=slice_y||
		last_slice_z!=slice_z;
	const bool use_snapshot =
		lbm->get_global_D()>1u && concurrent_graphics_enabled();
	bool refresh_snapshot =
		lbm->get_global_D()>1u &&
		(last_snapshot_time!=current_time || visualization_change);
	const bool needs_rho_u_halo =
		slice_mode!=0 ||
		(visualization_modes&
			(VIS_FIELD|VIS_STREAMLINES|VIS_Q_CRITERION))!=0;
	bool refresh_rho_u_halo =
		refresh_snapshot && needs_rho_u_halo;
#ifdef TEMPERATURE
	bool refresh_T_halo =
		refresh_snapshot && field_mode==2 && needs_rho_u_halo;
#else // TEMPERATURE
	bool refresh_T_halo = false;
#endif // TEMPERATURE
	Distributed_Graphics_Frame distributed_frame = {};
	const float* camera_override = nullptr;
	if(culbm::mpi_is_distributed()) {
		if(culbm::mpi_is_root()) {
			camera.update_matrix();
			distributed_frame.frame_id = ++distributed_frame_id;
			distributed_frame.session_id =
				lbm->get_graphics_session_id();
			distributed_frame.width = camera.width;
			distributed_frame.height = camera.height;
			distributed_frame.visualization_modes = visualization_modes;
			distributed_frame.field_mode = field_mode;
			distributed_frame.slice_mode = slice_mode;
			distributed_frame.slice_x = slice_x;
			distributed_frame.slice_y = slice_y;
			distributed_frame.slice_z = slice_z;
			distributed_frame.visualization_change =
				(uint32_t)visualization_change;
			distributed_frame.refresh_snapshot =
				(uint32_t)refresh_snapshot;
			distributed_frame.refresh_rho_u_halo =
				(uint32_t)refresh_rho_u_halo;
			distributed_frame.refresh_T_halo =
				(uint32_t)refresh_T_halo;
			for(uint i=0u; i<15u; ++i)
				distributed_frame.camera_data[i] = camera.data(i);
		}
		culbm::mpi_broadcast_graphics_bytes(
			&distributed_frame, sizeof(distributed_frame));
		if(distributed_frame.session_id!=
		   lbm->get_graphics_session_id()) {
			culbm::mpi_abort(
				"Distributed graphics ranks entered different "
				"solver sessions.");
		}
		if(distributed_frame.width!=camera.width ||
		   distributed_frame.height!=camera.height) {
			culbm::mpi_abort(
				"All distributed graphics ranks must use the root "
				"framebuffer dimensions.");
		}
		distributed_frame_id = distributed_frame.frame_id;
		visualization_modes = distributed_frame.visualization_modes;
		field_mode = distributed_frame.field_mode;
		slice_mode = distributed_frame.slice_mode;
		slice_x = distributed_frame.slice_x;
		slice_y = distributed_frame.slice_y;
		slice_z = distributed_frame.slice_z;
		visualization_change =
			distributed_frame.visualization_change!=0u;
		refresh_snapshot = distributed_frame.refresh_snapshot!=0u;
		refresh_rho_u_halo =
			distributed_frame.refresh_rho_u_halo!=0u;
		refresh_T_halo = distributed_frame.refresh_T_halo!=0u;
		camera_override = distributed_frame.camera_data;
	}

#ifndef UPDATE_FIELDS
	if(visualization_modes&(VIS_FIELD|VIS_STREAMLINES|VIS_Q_CRITERION)) {
		for(uint d=0u; d<lbm->get_D(); d++)
			lbm->lbm_domain[d]->enqueue_update_fields();
	}
#endif // UPDATE_FIELDS
	if(refresh_snapshot) {
		if(use_snapshot) {
			for(uint d=0u; d<lbm->get_D(); ++d)
				lbm->lbm_domain[d]->enqueue_graphics_snapshot();
		}
		if(refresh_rho_u_halo) {
			if(use_snapshot)
				lbm->communicate_graphics_rho_u_flags();
			else
				lbm->communicate_rho_u_flags();
			last_rho_halo_time = current_time;
		}
#ifdef TEMPERATURE
		if(refresh_T_halo) {
			if(use_snapshot)
				lbm->communicate_graphics_T();
			else
				lbm->communicate_T();
			last_T_halo_time = current_time;
		}
#endif // TEMPERATURE
		last_snapshot_time = current_time;
	}
	camera.key_update = false;
	last_visualization_modes = visualization_modes;
	last_field_mode = field_mode;
	last_slice_mode = slice_mode;
	last_slice_x = slice_x;
	last_slice_y = slice_y;
	last_slice_z = slice_z;
	bool new_frame = true;
	for(uint d=0u; d<lbm->get_D(); d++)
		new_frame = new_frame &&
			lbm->lbm_domain[d]->graphics.enqueue_draw_frame(
				visualization_modes, field_mode, slice_mode,
				slice_x, slice_y, slice_z, visualization_change,
				camera_override);
	for(uint d=0u; d<lbm->get_D(); d++) lbm->lbm_domain[d]->finish_queue();
	int* bitmap = lbm->lbm_domain[0]->graphics.get_bitmap();
	int* zbuffer = lbm->lbm_domain[0]->graphics.get_zbuffer();
	for(uint d=1u; d<lbm->get_D()&&new_frame; d++) {
		const int* bitmap_d = lbm->lbm_domain[d]->graphics.get_bitmap(); // each domain renders its own frame
		const int* zbuffer_d = lbm->lbm_domain[d]->graphics.get_zbuffer();
		for(uint i=0u; i<camera.width*camera.height; i++) {
#ifndef GRAPHICS_TRANSPARENCY
			const int zdi = zbuffer_d[i];
			if(zdi>zbuffer[i]) {
				bitmap[i] = bitmap_d[i]; // overlay frames using their z-buffers
				zbuffer[i] = zdi;
			}
#else // GRAPHICS_TRANSPARENCY
			bitmap[i] = color_add(bitmap[i], bitmap_d[i]);
#endif // GRAPHICS_TRANSPARENCY
		}
	}
		if(culbm::mpi_is_distributed()) {
		const ulong pixels=(ulong)camera.width*(ulong)camera.height;
		vector<int> gathered_bitmap;
		vector<int> gathered_zbuffer;
		if(culbm::mpi_is_root()) {
			gathered_bitmap.resize(pixels*(ulong)culbm::mpi_size());
			gathered_zbuffer.resize(pixels*(ulong)culbm::mpi_size());
		}
		culbm::mpi_gather_equal_bytes(
			bitmap, (uint64_t)(pixels*sizeof(int)),
			culbm::mpi_is_root() ? gathered_bitmap.data() : nullptr);
		culbm::mpi_gather_equal_bytes(
			zbuffer, (uint64_t)(pixels*sizeof(int)),
			culbm::mpi_is_root() ? gathered_zbuffer.data() : nullptr);
		if(culbm::mpi_is_root()) {
			for(int rank=1; rank<culbm::mpi_size(); rank++) {
				const int* bitmap_rank=gathered_bitmap.data()+(ulong)rank*pixels;
				const int* zbuffer_rank=gathered_zbuffer.data()+(ulong)rank*pixels;
				for(ulong i=0ull; i<pixels; i++) {
#ifndef GRAPHICS_TRANSPARENCY
					if(zbuffer_rank[i]>zbuffer[i]) {
						bitmap[i]=bitmap_rank[i];
						zbuffer[i]=zbuffer_rank[i];
					}
#else // GRAPHICS_TRANSPARENCY
					bitmap[i]=color_add(bitmap[i], bitmap_rank[i]);
#endif // GRAPHICS_TRANSPARENCY
				}
			}
		}
	}
	camera.allow_labeling = new_frame; // only print new label on frame if a new frame has been rendered

	return bitmap;
}

void LBM::Graphics::set_camera_centered(const float rx, const float ry, const float fov, const float zoom) {
	camera.free = false;
	camera.rx = 0.5*pi+((double)rx*pi/180.0);
	camera.ry = pi-((double)ry*pi/180.0);
	camera.fov = clamp((float)fov, 1E-6f, 179.0f);
	camera.set_zoom(0.5f*(float)fmax(fmax(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz())/zoom);
}
void LBM::Graphics::set_camera_free(const float3& p, const float rx, const float ry, const float fov) {
	camera.free = true;
	camera.rx = 0.5*pi+((double)rx*pi/180.0);
	camera.ry = pi-((double)ry*pi/180.0);
	camera.fov = clamp((float)fov, 1E-6f, 179.0f);
	camera.zoom = 1E16f;
	camera.pos = p;
}
void LBM::Graphics::print_frame() { // preview current frame in console
#ifndef INTERACTIVE_GRAPHICS_ASCII
	camera.rendring_frame.lock(); // block rendering for other threads until finished
	camera.key_update = true; // force rendering new frame
	int* image_data = draw_frame(); // make sure the frame is fully rendered
	Image* image = new Image(camera.width, camera.height, image_data);
	info.allow_printing.lock();
	println();
	print_image(image);
	info.allow_printing.unlock();
	delete image;
	camera.rendring_frame.unlock();
#endif // INTERACTIVE_GRAPHICS_ASCII
}
#endif // GRAPHICS



void LBM_Domain::allocate_transfer(Device& device) { // allocate all memory for multi-device trqansfer
	ulong Amax = 0ull; // maximum domain side area of communicated directions
	if(Dx>1u) Amax = max(Amax, (ulong)Ny*(ulong)Nz); // Ax
	if(Dy>1u) Amax = max(Amax, (ulong)Nz*(ulong)Nx); // Ay
	if(Dz>1u) Amax = max(Amax, (ulong)Nx*(ulong)Ny); // Az

	uint transfer_cell_bytes = max(transfers*(uint)sizeof(fpxx), 17u);
#ifdef TEMPERATURE
	transfer_cell_bytes =
		max(transfer_cell_bytes,
			(transfers+get_temperature_transfers())*(uint)sizeof(fpxx));
#endif // TEMPERATURE
	transfer_buffer_p = Memory<char>(device, Amax, transfer_cell_bytes, true, true, 0, false); // only allocate one set of transfer buffers in plus/minus directions, for all x/y/z transfers
	transfer_buffer_m = Memory<char>(device, Amax, transfer_cell_bytes, true, true, 0, false);
	transfer_buffer_recv_p = Memory<char>(device, Amax, transfer_cell_bytes, true, true, 0, false);
	transfer_buffer_recv_m = Memory<char>(device, Amax, transfer_cell_bytes, true, true, 0, false); // these transfer buffers must not be zero-copy!
	const bool pinned_halo_staging =
		transfer_buffer_p.is_host_buffer_pinned() &&
		transfer_buffer_m.is_host_buffer_pinned() &&
		transfer_buffer_recv_p.is_host_buffer_pinned() &&
		transfer_buffer_recv_m.is_host_buffer_pinned();
	if(pinned_halo_staging) {
		print_info(
			"CUDA halo host staging uses pinned asynchronous transfers.");
	} else {
		print_warning(
			"CUDA halo host staging allocation fell back to pageable memory; "
			"inter-device transfers will synchronize.");
	}

#ifdef GRAPHICS
	if(concurrent_graphics_enabled()) {
		print_info(
			"Concurrent graphics: isolated field snapshot and halo "
			"transfer channel enabled.");
		graphics_rho = Memory<float>(device, get_N(), 1u, false);
		graphics_u = Memory<float>(device, get_N(), 3u, false);
		graphics_flags = Memory<uchar>(device, get_N(), 1u, false);
#ifdef FORCE_FIELD
		graphics_F = Memory<float>(device, get_N(), 3u, false);
#endif // FORCE_FIELD
#ifdef SURFACE
		graphics_phi = Memory<float>(device, get_N(), 1u, false);
#endif // SURFACE
#ifdef TEMPERATURE
		graphics_T = Memory<float>(device, get_N(), 1u, false);
#endif // TEMPERATURE
#ifdef PARTICLES
		graphics_particles =
			Memory<float>(device, (ulong)particles_N, 3u, false);
#endif // PARTICLES
		const ulong graphics_snapshot_range =
#ifdef PARTICLES
			max(get_N(), (ulong)particles_N);
#else
			get_N();
#endif // PARTICLES
		kernel_graphics_snapshot = Kernel(
			device, graphics_snapshot_range, "graphics_snapshot_fields",
			graphics_rho, graphics_u, graphics_flags);
#ifdef FORCE_FIELD
		kernel_graphics_snapshot.add_parameters(graphics_F);
#endif // FORCE_FIELD
#ifdef SURFACE
		kernel_graphics_snapshot.add_parameters(graphics_phi);
#endif // SURFACE
#ifdef TEMPERATURE
		kernel_graphics_snapshot.add_parameters(graphics_T);
#endif // TEMPERATURE
#ifdef PARTICLES
		kernel_graphics_snapshot.add_parameters(graphics_particles);
#endif // PARTICLES
		kernel_graphics_snapshot.add_parameters(rho, u, flags);
#ifdef FORCE_FIELD
		kernel_graphics_snapshot.add_parameters(F);
#endif // FORCE_FIELD
#ifdef SURFACE
		kernel_graphics_snapshot.add_parameters(phi);
#endif // SURFACE
#ifdef TEMPERATURE
		kernel_graphics_snapshot.add_parameters(T);
#endif // TEMPERATURE
#ifdef PARTICLES
		kernel_graphics_snapshot.add_parameters(particles);
#endif // PARTICLES

		graphics_transfer_buffer_p =
			Memory<char>(device, Amax, 17u, true, true, 0, false);
		graphics_transfer_buffer_m =
			Memory<char>(device, Amax, 17u, true, true, 0, false);
		graphics_transfer_buffer_recv_p =
			Memory<char>(device, Amax, 17u, true, true, 0, false);
		graphics_transfer_buffer_recv_m =
			Memory<char>(device, Amax, 17u, true, true, 0, false);
	}
#endif // GRAPHICS

	kernel_transfer[enum_transfer_field::fi              ][0] = Kernel(device, 0u, "transfer_extract_fi"              , 0u, get_t(), transfer_buffer_p, transfer_buffer_m, fi);
	kernel_transfer[enum_transfer_field::fi              ][1] = Kernel(device, 0u, "transfer__insert_fi"              , 0u, get_t(), transfer_buffer_recv_p, transfer_buffer_recv_m, fi);
	kernel_transfer[enum_transfer_field::rho_u_flags     ][0] = Kernel(device, 0u, "transfer_extract_rho_u_flags"     , 0u, get_t(), transfer_buffer_p, transfer_buffer_m, rho, u, flags);
	kernel_transfer[enum_transfer_field::rho_u_flags     ][1] = Kernel(device, 0u, "transfer__insert_rho_u_flags"     , 0u, get_t(), transfer_buffer_recv_p, transfer_buffer_recv_m, rho, u, flags);
	kernel_transfer[enum_transfer_field::flags           ][0] = Kernel(device, 0u, "transfer_extract_flags"           , 0u, get_t(), transfer_buffer_p, transfer_buffer_m, flags);
	kernel_transfer[enum_transfer_field::flags           ][1] = Kernel(device, 0u, "transfer__insert_flags"           , 0u, get_t(), transfer_buffer_recv_p, transfer_buffer_recv_m, flags);
#ifdef SURFACE
	kernel_transfer[enum_transfer_field::phi_massex_flags][0] = Kernel(device, 0u, "transfer_extract_phi_massex_flags", 0u, get_t(), transfer_buffer_p, transfer_buffer_m, phi, massex, flags);
	kernel_transfer[enum_transfer_field::phi_massex_flags][1] = Kernel(device, 0u, "transfer__insert_phi_massex_flags", 0u, get_t(), transfer_buffer_recv_p, transfer_buffer_recv_m, phi, massex, flags);
#endif // SURFACE
#ifdef TEMPERATURE
	kernel_transfer[enum_transfer_field::gi              ][0] = Kernel(device, 0u, "transfer_extract_gi"              , 0u, get_t(), transfer_buffer_p, transfer_buffer_m, gi);
	kernel_transfer[enum_transfer_field::gi              ][1] = Kernel(device, 0u, "transfer__insert_gi"              , 0u, get_t(), transfer_buffer_recv_p, transfer_buffer_recv_m, gi);
	kernel_transfer[enum_transfer_field::T               ][0] = Kernel(device, 0u, "transfer_extract_T"               , 0u, get_t(), transfer_buffer_p, transfer_buffer_m, T);
	kernel_transfer[enum_transfer_field::T               ][1] = Kernel(device, 0u, "transfer__insert_T"               , 0u, get_t(), transfer_buffer_recv_p, transfer_buffer_recv_m, T);
	kernel_transfer[enum_transfer_field::fi_gi           ][0] = Kernel(device, 0u, "transfer_extract_fi_gi"           , 0u, get_t(), transfer_buffer_p, transfer_buffer_m, fi, gi);
	kernel_transfer[enum_transfer_field::fi_gi           ][1] = Kernel(device, 0u, "transfer__insert_fi_gi"           , 0u, get_t(), transfer_buffer_recv_p, transfer_buffer_recv_m, fi, gi);
#endif // TEMPERATURE

#ifdef GRAPHICS
	if(concurrent_graphics_enabled()) {
		graphics_kernel_transfer[enum_transfer_field::rho_u_flags][0] =
			Kernel(device, 0u, "transfer_extract_rho_u_flags", 0u, get_t(),
				graphics_transfer_buffer_p, graphics_transfer_buffer_m,
				graphics_rho, graphics_u, graphics_flags);
		graphics_kernel_transfer[enum_transfer_field::rho_u_flags][1] =
			Kernel(device, 0u, "transfer__insert_rho_u_flags", 0u, get_t(),
				graphics_transfer_buffer_recv_p,
				graphics_transfer_buffer_recv_m,
				graphics_rho, graphics_u, graphics_flags);
#ifdef TEMPERATURE
		graphics_kernel_transfer[enum_transfer_field::T][0] =
			Kernel(device, 0u, "transfer_extract_T", 0u, get_t(),
				graphics_transfer_buffer_p, graphics_transfer_buffer_m,
				graphics_T);
		graphics_kernel_transfer[enum_transfer_field::T][1] =
			Kernel(device, 0u, "transfer__insert_T", 0u, get_t(),
				graphics_transfer_buffer_recv_p,
				graphics_transfer_buffer_recv_m,
				graphics_T);
#endif // TEMPERATURE
	}
#endif // GRAPHICS
}

#ifdef GRAPHICS
void LBM_Domain::enqueue_graphics_snapshot() {
	kernel_graphics_snapshot.enqueue_run();
}
#endif // GRAPHICS

ulong LBM_Domain::get_area(const uint direction) {
	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
	return A[direction];
}
void LBM_Domain::enqueue_transfer_extract_field(Kernel& kernel_transfer_extract_field, const uint direction, const uint bytes_per_cell) {
	kernel_transfer_extract_field.set_ranges(get_area(direction)); // direction: x=0, y=1, z=2
	kernel_transfer_extract_field.set_parameters(0u, direction, get_t()).enqueue_run(); // selective in-VRAM copy
}
void LBM_Domain::enqueue_transfer_insert_field(Kernel& kernel_transfer_insert_field, const uint direction, const uint bytes_per_cell) {
	kernel_transfer_insert_field.set_ranges(get_area(direction)); // direction: x=0, y=1, z=2
	kernel_transfer_insert_field.set_parameters(0u, direction, get_t()).enqueue_run(); // selective in-VRAM copy
}
void LBM::communicate_field(
	const enum_transfer_field field, const uint bytes_per_cell,
	const CudaTransferChannel channel) {
	const uint domain_counts[3] = { Dx, Dy, Dz };
	const bool graphics_channel =
		channel==CudaTransferChannel::graphics;
#ifndef GRAPHICS
	if(graphics_channel)
		print_error(
			"Graphics halo communication requires the GRAPHICS extension.");
#endif // GRAPHICS
	if(graphics_channel && field!=enum_transfer_field::rho_u_flags
#ifdef TEMPERATURE
	   && field!=enum_transfer_field::T
#endif // TEMPERATURE
	) {
		print_error(
			"Unsupported field requested on the graphics transfer channel.");
	}
	if(is_distributed()) {
		LBM_Domain* const domain = lbm_domain[0];
		const uint global_domain = get_global_domain_id();
		Memory<char>& transfer_buffer_p =
			domain->get_transfer_buffer_p(channel);
		Memory<char>& transfer_buffer_m =
			domain->get_transfer_buffer_m(channel);
		Memory<char>& transfer_buffer_recv_p =
			domain->get_transfer_buffer_recv_p(channel);
		Memory<char>& transfer_buffer_recv_m =
			domain->get_transfer_buffer_recv_m(channel);
		for(uint direction=0u; direction<3u; ++direction) {
			if(domain_counts[direction]<=1u) continue;
			const ulong length=domain->get_area(direction)*(ulong)bytes_per_cell;
			domain->enqueue_transfer_extract_field(
				domain->get_transfer_kernel(field, 0u, channel),
				direction, bytes_per_cell);
			transfer_buffer_p.enqueue_read_from_device(0ull, length);
			transfer_buffer_m.enqueue_read_from_device(0ull, length);
			domain->finish_queue();
			const uint positive=culbm::positive_domain_neighbor(
				global_domain, Dx, Dy, Dz, direction);
			const uint negative=culbm::negative_domain_neighbor(
				global_domain, Dx, Dy, Dz, direction);
			const int tag_base=100+(int)field*8+(int)direction*2;
			if(graphics_channel) {
				culbm::mpi_exchange_graphics_halo_pair_bytes(
					transfer_buffer_p.data(), transfer_buffer_m.data(), (uint64_t)length,
					(int)positive, (int)negative,
					transfer_buffer_recv_p.data(), transfer_buffer_recv_m.data(),
					tag_base, tag_base+1);
			}
			else {
				culbm::mpi_exchange_halo_pair_bytes(
					transfer_buffer_p.data(), transfer_buffer_m.data(), (uint64_t)length,
					(int)positive, (int)negative,
					transfer_buffer_recv_p.data(), transfer_buffer_recv_m.data(),
					tag_base, tag_base+1);
			}
			transfer_buffer_recv_p.enqueue_write_to_device(0ull, length);
			transfer_buffer_recv_m.enqueue_write_to_device(0ull, length);
			domain->enqueue_transfer_insert_field(
				domain->get_transfer_kernel(field, 1u, channel),
				direction, bytes_per_cell);
		}
		return;
	}
	for(uint direction=0u; direction<3u; ++direction) {
		if(domain_counts[direction]<=1u) continue;
		for(uint d=0u; d<get_D(); ++d) {
			lbm_domain[d]->enqueue_transfer_extract_field(
				lbm_domain[d]->get_transfer_kernel(field, 0u, channel),
				direction, bytes_per_cell);
			lbm_domain[d]->get_device().record_transfer_ready(channel);
		}
		for(uint d=0u; d<get_D(); ++d)
			lbm_domain[d]->get_device().begin_transfer_batch(channel);
		for(uint d=0u; d<get_D(); ++d) {
			const uint positive =
				culbm::positive_domain_neighbor(
					d, Dx, Dy, Dz, direction);
			const ulong length=
				lbm_domain[d]->get_area(direction)*(ulong)bytes_per_cell;
			lbm_domain[positive]->get_transfer_buffer_recv_m(channel)
				.enqueue_copy_from_device_buffer(
					lbm_domain[d]->get_transfer_buffer_p(channel),
					0ull, length, channel);
			lbm_domain[d]->get_transfer_buffer_recv_p(channel)
				.enqueue_copy_from_device_buffer(
					lbm_domain[positive]->get_transfer_buffer_m(channel),
					0ull, length, channel);
		}
		for(uint d=0u; d<get_D(); ++d)
			lbm_domain[d]->get_device().complete_transfer_batch(channel);
		for(uint d=0u; d<get_D(); ++d)
			lbm_domain[d]->enqueue_transfer_insert_field(
				lbm_domain[d]->get_transfer_kernel(field, 1u, channel),
				direction, bytes_per_cell);
		for(uint d=0u; d<get_D(); ++d)
			lbm_domain[d]->get_device().record_transfer_consumed(channel);
		for(uint d=0u; d<get_D(); ++d) {
			const uint positive =
				culbm::positive_domain_neighbor(
					d, Dx, Dy, Dz, direction);
			const uint negative =
				culbm::negative_domain_neighbor(
					d, Dx, Dy, Dz, direction);
			lbm_domain[d]->get_device().wait_for_transfer_batch(
				lbm_domain[positive]->get_device(), channel);
			if(negative!=positive)
				lbm_domain[d]->get_device().wait_for_transfer_batch(
					lbm_domain[negative]->get_device(), channel);
		}
	}
}

void LBM::communicate_fi() {
	communicate_field(enum_transfer_field::fi, transfers*sizeof(fpxx));
}
void LBM::communicate_rho_u_flags() {
	communicate_field(enum_transfer_field::rho_u_flags, 17u);
}
#ifdef GRAPHICS
void LBM::communicate_graphics_rho_u_flags() {
	communicate_field(
		enum_transfer_field::rho_u_flags, 17u,
		CudaTransferChannel::graphics);
}
#endif // GRAPHICS
void LBM::communicate_flags() {
	communicate_field(enum_transfer_field::flags, 1u);
}
#ifdef SURFACE
void LBM::communicate_phi_massex_flags() {
	communicate_field(enum_transfer_field::phi_massex_flags, 9u);
}
#endif // SURFACE
#ifdef TEMPERATURE
void LBM::communicate_fi_gi() {
	communicate_field(
		enum_transfer_field::fi_gi,
		(transfers+get_temperature_transfers())*sizeof(fpxx));
}
void LBM::communicate_gi() {
	communicate_field(
		enum_transfer_field::gi,
		get_temperature_transfers()*sizeof(fpxx));
}
void LBM::communicate_T() {
	communicate_field(enum_transfer_field::T, 4u);
}
#ifdef GRAPHICS
void LBM::communicate_graphics_T() {
	communicate_field(
		enum_transfer_field::T, 4u,
		CudaTransferChannel::graphics);
}
#endif // GRAPHICS
#endif // TEMPERATURE
