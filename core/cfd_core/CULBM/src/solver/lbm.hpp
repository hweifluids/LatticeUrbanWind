#pragma once

#include "defines.hpp"
#include "cuda.hpp"
#include "graphics.hpp"
#include "units.hpp"
#include "info.hpp"

extern float coriolis_f_lbmu;
extern float3 vtk_origin_shift; // VTK origin shift in SI units for field outputs

enum class Temperature_Lattice : uint {
	Q7 = 7u,
	Q19 = 19u
};

// Runtime scalar outlet orientation.  Values deliberately match the existing
// downstream-face convention used by the setup/buffer code.
enum class Temperature_Outlet_Face : int {
	None = 0,
	West = 1,
	East = 2,
	South = 3,
	North = 4
};

bool set_temperature_lattice(Temperature_Lattice lattice);
Temperature_Lattice get_temperature_lattice();
uint get_temperature_set();
uint get_temperature_transfers();
float get_temperature_lattice_cs2();
float get_temperature_tau(const float alpha);
const char* get_temperature_lattice_name();

struct Subgrid_Configuration {
#ifdef SUBGRID
	bool enabled = true;
#else
	bool enabled = false;
#endif
	float smagorinsky_constant = 0.173265956f;
	bool scalar_coupling = true;
	float turbulent_prandtl = 1.0f;
	// -1 selects the local stress-derived Smagorinsky value.  A non-negative
	// value is an LBM-unit constant eddy viscosity for deterministic tests.
	float fixed_eddy_viscosity = -1.0f;
};

struct Subgrid_Statistics {
	double mean_nu_t = 0.0;
	float max_nu_t = 0.0f;
	double mean_rho_nu_t = 0.0;
	float max_rho_nu_t = 0.0f;
	std::uint64_t count = 0u;
	bool valid = false;
};

bool set_subgrid_configuration(const Subgrid_Configuration& configuration);
Subgrid_Configuration get_subgrid_configuration();

// For one locally relaxed non-equilibrium collision mode, 1-omega is the
// per-collision eigenvalue. The warning threshold is deliberately diagnostic:
// it marks less than 2% damping per collision, but is not a stability proof.
constexpr float collision_eigenvalue_warning_threshold = 0.98f;
constexpr float trt_magic_parameter = 3.0f/16.0f;

struct Relaxation_Branch_Diagnostic {
	float tau;
	float omega;
	float eigenvalue_magnitude;
	bool valid;
	bool near_marginal;
};

struct TRT_Relaxation_Diagnostic {
	float magic_parameter;
	Relaxation_Branch_Diagnostic plus;
	Relaxation_Branch_Diagnostic minus;
};

Relaxation_Branch_Diagnostic get_relaxation_branch_diagnostic(
	const float tau);
TRT_Relaxation_Diagnostic get_trt_relaxation_diagnostic(
	const float tau_plus);

uint bytes_per_cell_host(); // returns the number of Bytes per cell allocated in host memory
uint bytes_per_cell_device(); // returns the number of Bytes per cell allocated in device memory
uint bandwidth_bytes_per_cell_device(); // returns the bandwidth in Bytes per cell per time step from/to device memory
uint3 resolution(const float3 box_aspect_ratio, const uint memory); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
uint vram_required_mb_per_device(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz);
uint vram_required_mb_total(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz);
std::uint64_t voxelization_transient_vram_mb_per_device(
	const std::uint64_t triangle_count);

string default_filename(const string& path, const string& name, const string& extension, const ulong t); // generate a default filename with timestamp
string default_filename(const string& name, const string& extension, const ulong t); // generate a default filename with timestamp at exe_path/export/

#pragma warning(disable:26812)
enum enum_transfer_field {
	fi,
	rho_u_flags,
	flags,
	phi_massex_flags,
	gi,
	T,
	fi_gi,
	enum_transfer_field_length
};

class LBM_Domain {
private:
	friend class LBM;
	uint Nx=1u, Ny=1u, Nz=1u; // (local) lattice dimensions
	uint Dx=1u, Dy=1u, Dz=1u; // lattice domains
	int Ox=0, Oy=0, Oz=0; // lattice domain offset
	std::atomic<ulong> t = 0ull; // discrete time step in LBM units; read by the graphics thread

	float nu = 1.0f/6.0f; // kinematic shear viscosity
	float fx=0.0f, fy=0.0f, fz=0.0f; // global force per volume
	float omega_x = 0.0f, omega_y = 0.0f, omega_z = 0.0f; // global rotation for Coriolis
	float sigma=0.0f; // surface tension coefficient
	float alpha=1.0f, beta=1.0f, T_avg=1.0f; // alpha = thermal diffusion coefficient, beta = (volumetric) thermal expansion coefficient, T_avg = 1 = average temperature
#ifdef TEMPERATURE
	float buoyancy_gx=0.0f, buoyancy_gy=0.0f, buoyancy_gz=0.0f; // gravity acceleration used only by the Boussinesq temperature force
	int temperature_outlet_face=0; // 0=off, 1=west, 2=east, 3=south, 4=north
#endif // TEMPERATURE
	uint particles_N = 0u;
	float particles_rho = 1.0f;

	Device device; // CUDA device associated with this LBM domain
	Kernel kernel_initialize; // initialization kernel
	Kernel kernel_stream_collide; // main LBM kernel
	Kernel kernel_update_fields; // reads DDFs and updates (rho, u, T) in device memory
	uint update_fields_control_parameter = 0u;
	uint update_fields_subgrid_parameter = 0u;
	Memory<double> subgrid_sum;
	Memory<float> subgrid_max;
	Memory<ulong> subgrid_count;
#ifdef TEMPERATURE
	Kernel kernel_update_temperature_outlet; // first-order zero-normal-gradient scalar outlet
	Kernel kernel_update_free_slip_boundaries; // mirror macro fields on validation-only free-slip outer walls
#endif // TEMPERATURE
	Kernel kernel_accumulate_statistics; // updates per-cell Welford statistics on the CUDA device
	Kernel kernel_gather_probe_velocity; // gathers sparse probe velocities without a full-field host transfer
	Memory<fpxx> fi; // LBM density distribution functions (DDFs); only exist in device memory
	ulong t_last_update_fields = max_ulong; // optimization to not call kernel_update_fields multiple times if (rho, u, T) are already up-to-date
	Memory<float> statistics_mean_u, statistics_mean_rho, statistics_M2_u;
#ifdef TEMPERATURE
	Memory<float> statistics_mean_T;
#endif // TEMPERATURE
	Memory<ulong> probe_indices;
	Memory<float> probe_velocity;
	vector<ulong> probe_output_slots;
	bool device_statistics_active = false;
	bool device_statistics_temperature = false;
#ifdef FORCE_FIELD
	Kernel kernel_update_force_field; // calculate forces from fluid on TYPE_S cells
	Kernel kernel_reset_force_field; // reset force field (also on TYPE_S cells)
	Kernel kernel_object_center_of_mass; // calculate center of mass of all cells flagged with flag_marker
	Kernel kernel_object_force; // add up force for all cells flagged with flag_marker
	Kernel kernel_object_torque; // add up torque around specified rotation_center for all cells flagged with flag_marker
	ulong t_last_force_field = max_ulong; // optimization to not call kernel_update_force_field multiple times if F is already up-to-date
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	Kernel kernel_update_moving_boundaries; // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#ifdef SURFACE
	Kernel kernel_surface_0; // additional kernel for computing mass conservation and mass flux computation
	Kernel kernel_surface_1; // additional kernel for flag handling
	Kernel kernel_surface_2; // additional kernel for flag handling
	Kernel kernel_surface_3; // additional kernel for flag handling and mass conservation
	Memory<float> mass; // fluid mass; phi=mass/rho
	Memory<float> massex; // excess mass; used for mass conservation
#endif // SURFACE
#ifdef TEMPERATURE
	Memory<fpxx> gi; // thermal DDFs
#endif // TEMPERATURE
#ifdef PARTICLES
	Kernel kernel_integrate_particles; // intgegrates particles forward in time and couples particles to fluid
#endif // PARTICLES

	void allocate(Device& device); // allocate all memory for data fields on host and device and set up kernels
	string device_defines() const; // returns per-domain CUDA preprocessor constants
	string full_cuda_source() const; // returns device_defines + possible graphics defines + kernel source
#ifdef TEMPERATURE
	void set_temperature_outlet_face(const int face) {
		this->temperature_outlet_face = face;
	}
#endif // TEMPERATURE

public:
	Memory<float> rho;
	Memory<float> u; // velocity of every cell
	Memory<uchar> flags; // flags of every cell
#ifdef FORCE_FIELD
	Memory<float> F; // individual force for every cell
	Memory<float> object_sum; // sum of individual cell data for an object
#endif // FORCE_FIELD
#ifdef SURFACE
	Memory<float> phi; // fill level of every cell
#endif // SURFACE
#ifdef TEMPERATURE
	Memory<float> T; // temperature of every cell
#endif // TEMPERATURE
#ifdef PARTICLES
	Memory<float> particles; // particle positions
#endif // PARTICLES

	Memory<char> transfer_buffer_p, transfer_buffer_m, transfer_buffer_recv_p, transfer_buffer_recv_m; // transfer buffers for multi-device domain communication, only allocate one set of transfer buffers in plus/minus directions, for all x/y/z transfers
	Kernel kernel_transfer[enum_transfer_field::enum_transfer_field_length][2]; // for each field one extract and one insert kernel
	Memory<char> graphics_transfer_buffer_p, graphics_transfer_buffer_m, graphics_transfer_buffer_recv_p, graphics_transfer_buffer_recv_m; // independent snapshot buffers for graphics-thread halo exchange
	Kernel graphics_kernel_transfer[enum_transfer_field::enum_transfer_field_length][2]; // graphics has separate kernel argument storage from the simulation thread
#ifdef GRAPHICS
	Memory<float> graphics_rho, graphics_u;
	Memory<uchar> graphics_flags;
#ifdef FORCE_FIELD
	Memory<float> graphics_F;
#endif // FORCE_FIELD
#ifdef SURFACE
	Memory<float> graphics_phi;
#endif // SURFACE
#ifdef TEMPERATURE
	Memory<float> graphics_T;
#endif // TEMPERATURE
#ifdef PARTICLES
	Memory<float> graphics_particles;
#endif // PARTICLES
	Kernel kernel_graphics_snapshot;
	void enqueue_graphics_snapshot();
#endif // GRAPHICS
	void allocate_transfer(Device& device); // allocate all memory for multi-device transfer
	Memory<char>& get_transfer_buffer_p(const CudaTransferChannel channel) {
		return channel==CudaTransferChannel::graphics
			? graphics_transfer_buffer_p : transfer_buffer_p;
	}
	Memory<char>& get_transfer_buffer_m(const CudaTransferChannel channel) {
		return channel==CudaTransferChannel::graphics
			? graphics_transfer_buffer_m : transfer_buffer_m;
	}
	Memory<char>& get_transfer_buffer_recv_p(
		const CudaTransferChannel channel) {
		return channel==CudaTransferChannel::graphics
			? graphics_transfer_buffer_recv_p : transfer_buffer_recv_p;
	}
	Memory<char>& get_transfer_buffer_recv_m(
		const CudaTransferChannel channel) {
		return channel==CudaTransferChannel::graphics
			? graphics_transfer_buffer_recv_m : transfer_buffer_recv_m;
	}
	Kernel& get_transfer_kernel(
		const enum_transfer_field field, const uint operation,
		const CudaTransferChannel channel) {
		return channel==CudaTransferChannel::graphics
			? graphics_kernel_transfer[field][operation]
			: kernel_transfer[field][operation];
	}
	ulong get_area(const uint direction);
	void enqueue_transfer_extract_field(Kernel& kernel_transfer_extract_field, const uint direction, const uint bytes_per_cell);
	void enqueue_transfer_insert_field(Kernel& kernel_transfer_insert_field, const uint direction, const uint bytes_per_cell);

	LBM_Domain(const Device_Info& device_info, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const int Ox, const int Oy, const int Oz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho); // compiles CUDA source and allocates memory

	void enqueue_initialize(); // write all data fields to device and call kernel_initialize
	void enqueue_update_temperature_outlet(); // update scalar outlet T from the adjacent interior DDF state
	void enqueue_update_free_slip_boundaries(const uint face_mask); // mirror outer-wall macro fields from adjacent interior cells
	void enqueue_stream_collide(); // call kernel_stream_collide to perform one LBM time step
	void enqueue_update_fields(); // update fields (rho, u, T) manually
	void enqueue_subgrid_statistics();
	Subgrid_Statistics read_subgrid_statistics();
	bool enable_device_statistics(
		const bool include_temperature,
		const uint future_device_memory_mb=0u);
	void disable_device_statistics();
	void reset_device_statistics();
	void enqueue_device_statistics_sample(const float inverse_sample_count);
	bool has_device_statistics() const { return device_statistics_active; }
	bool has_device_statistics_temperature() const { return device_statistics_temperature; }
	Memory<float>& get_statistics_mean_u() { return statistics_mean_u; }
	Memory<float>& get_statistics_mean_rho() { return statistics_mean_rho; }
	Memory<float>& get_statistics_M2_u() { return statistics_M2_u; }
#ifdef TEMPERATURE
	Memory<float>& get_statistics_mean_T() { return statistics_mean_T; }
#endif // TEMPERATURE
	bool configure_device_probe_sampler(
		const vector<ulong>& local_indices,
		const vector<ulong>& output_slots,
		const uint future_device_memory_mb=0u);
	void disable_device_probe_sampler();
	bool has_device_probe_sampler() const { return !probe_output_slots.empty(); }
	void enqueue_device_probe_sample();
	const vector<ulong>& get_device_probe_output_slots() const {
		return probe_output_slots;
	}
	Memory<float>& get_device_probe_velocity() { return probe_velocity; }
#ifdef SURFACE
	void enqueue_surface_0();
	void enqueue_surface_1();
	void enqueue_surface_2();
	void enqueue_surface_3();
#endif // SURFACE
#ifdef FORCE_FIELD
	void enqueue_update_force_field(); // calculate forces from fluid on TYPE_S cells
	void enqueue_object_center_of_mass(const uchar flag_marker=TYPE_S); // calculate center of mass of all cells flagged with flag_marker
	void enqueue_object_force(const uchar flag_marker=TYPE_S); // add up force for all cells flagged with flag_marker
	void enqueue_object_torque(const float3& rotation_center, const uchar flag_marker=TYPE_S); // add up torque around specified rotation_center for all cells flagged with flag_marker
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	void enqueue_update_moving_boundaries(); // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#ifdef PARTICLES
	void enqueue_integrate_particles(const uint time_step_multiplicator=1u); // intgegrates particles forward in time and couples particles to fluid
#endif // PARTICLES

	void increment_time_step(const uint steps=1u); // increment time step
	void reset_time_step(); // reset time step
	void finish_queue();

	Device& get_device() { return device; }
	const Device& get_device() const { return device; }
	uint get_Nx() const { return Nx; } // get (local) lattice dimensions in x-direction
	uint get_Ny() const { return Ny; } // get (local) lattice dimensions in y-direction
	uint get_Nz() const { return Nz; } // get (local) lattice dimensions in z-direction
	ulong get_N() const { return (ulong)Nx*(ulong)Ny*(ulong)Nz; } // get (local) number of lattice points
	uint get_Dx() const { return Dx; } // get lattice domains in x-direction
	uint get_Dy() const { return Dy; } // get lattice domains in y-direction
	uint get_Dz() const { return Dz; } // get lattice domains in z-direction
	uint get_D() const { return Dx*Dy*Dz; } // get number of lattice domains
	int get_Ox() const { return Ox; } // global offset of the local x=0 cell (including halo)
	int get_Oy() const { return Oy; } // global offset of the local y=0 cell (including halo)
	int get_Oz() const { return Oz; } // global offset of the local z=0 cell (including halo)
	float get_nu() const { return nu; } // get kinematic shear viscosity
	float get_tau() const { return 3.0f*get_nu()+0.5f; } // get LBM relaxation time
	float get_fx() const { return fx; } // get global froce per volume
	float get_fy() const { return fy; } // get global froce per volume
	float get_fz() const { return fz; } // get global froce per volume
	float get_omega_x() const { return omega_x; } // get Coriolis rotation x-component
	float get_omega_y() const { return omega_y; } // get Coriolis rotation y-component
	float get_omega_z() const { return omega_z; } // get Coriolis rotation z-component
	float get_sigma() const { return sigma; } // get surface tension coefficient
	float get_alpha() const { return alpha; } // get thermal diffusion coefficient
	float get_beta() const { return beta; } // get thermal expansion coefficient
#ifdef TEMPERATURE
	float get_buoyancy_gx() const { return buoyancy_gx; }
	float get_buoyancy_gy() const { return buoyancy_gy; }
	float get_buoyancy_gz() const { return buoyancy_gz; }
	int get_temperature_outlet_face() const { return temperature_outlet_face; }
#endif // TEMPERATURE
	ulong get_t() const {
		return t.load(std::memory_order_acquire);
	} // get discrete time step in LBM units
	uint get_velocity_set() const; // get LBM velocity set
	void set_fx(const float fx) { this->fx = fx; } // set global froce per volume
	void set_fy(const float fy) { this->fy = fy; } // set global froce per volume
	void set_fz(const float fz) { this->fz = fz; } // set global froce per volume
	void set_f(const float fx, const float fy, const float fz) { set_fx(fx); set_fy(fy); set_fz(fz); } // set global froce per volume
	void set_coriolis(const float omega_x, const float omega_y, const float omega_z) {
		this->omega_x = omega_x;
		this->omega_y = omega_y;
		this->omega_z = omega_z;
	}
#ifdef TEMPERATURE
	void set_buoyancy_gravity(const float gx, const float gy, const float gz) {
		this->buoyancy_gx = gx;
		this->buoyancy_gy = gy;
		this->buoyancy_gz = gz;
	}
#endif // TEMPERATURE
	void voxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S, const float3& rotation_center=float3(0.0f), const float3& linear_velocity=float3(0.0f), const float3& rotational_velocity=float3(0.0f), const uint direction_override_0=max_uint, const uint direction_override_1=max_uint, const vector<uint>* triangle_ids_0=nullptr, const vector<uint>* triangle_ids_1=nullptr); // voxelize mesh
	void enqueue_unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S); // remove voxelized triangle mesh from LBM grid

#ifdef GRAPHICS
	class Graphics {
	private:
		Kernel kernel_clear; // reset bitmap and zbuffer
		Memory<int> bitmap; // bitmap for rendering
		Memory<int> zbuffer; // z-buffer for rendering
		Memory<float> camera_parameters; // contains camera position, rotation, field of view etc.

		LBM_Domain* lbm = nullptr;
		Kernel kernel_graphics_flags; // render flag lattice with wireframe
		Kernel kernel_graphics_flags_mc; // render flag lattice with marching-cubes
		Kernel kernel_graphics_field; // render a colored velocity vector for each cell
		Kernel kernel_graphics_field_slice; // render one slice of velocity field according to slics settings
		Kernel kernel_graphics_streamline; // render streamlines
		Kernel kernel_graphics_q; // render vorticity (Q-criterion)

#ifdef SURFACE
		Image* skybox_image = nullptr;
		Memory<int> skybox; // skybox for free surface raytracing
		Kernel kernel_graphics_rasterize_phi; // rasterize free surface
		Kernel kernel_graphics_raytrace_phi; // raytrace free surface
		Image* get_skybox_image() const { return skybox_image; }
		static Image* flat_skybox_image() {
			static Image image(1u, 1u);
			static const int initialized = (image.set_color(0u, GRAPHICS_BACKGROUND_COLOR), 0);
			(void)initialized;
			return &image;
		}
#endif // SURFACE

#ifdef PARTICLES
		Kernel kernel_graphics_particles;
#endif // PARTICLES

		ulong t_last_rendered_frame = max_ulong; // optimization to not call draw_frame() multiple times if camera_parameters and LBM time step are unchanged
		bool update_camera(
			const float* camera_override = nullptr); // update camera_parameters and return if they are changed from their previous state

	public:
		Graphics() {} // default constructor
		Graphics(LBM_Domain* lbm) {
			this->lbm = lbm;
#ifdef SURFACE
			skybox_image = flat_skybox_image(); // no external texture decoder in CULBM
#endif // SURFACE
		}
		Graphics& operator=(const Graphics& graphics) { // copy assignment
			lbm = graphics.lbm;
#ifdef SURFACE
			skybox_image = graphics.get_skybox_image();
#endif // SURFACE
			return *this;
		}
		void allocate(Device& device); // allocate memory for bitmap and zbuffer
		bool enqueue_draw_frame(const int visualization_modes, const int field_mode=0, const int slice_mode=0, const int slice_x=0, const int slice_y=0, const int slice_z=0, const bool visualization_change=true, const float* camera_override=nullptr); // main rendering function, calls rendering kernels, returns true if new frame is rendered, false if old frame is returned when camera has not moved
		int* get_bitmap(); // returns pointer to bitmap
		int* get_zbuffer(); // returns pointer to zbuffer
		string device_defines() const; // returns per-domain CUDA preprocessor constants
	}; // Graphics
	Graphics graphics;
#endif // GRAPHICS
}; // LBM_Domain



class LBM {
private:
	uint Nx=1u, Ny=1u, Nz=1u; // (global) lattice dimensions
	uint Dx=1u, Dy=1u, Dz=1u; // lattice domains
	uint local_D=1u, global_domain_id=0u; // process-local domains and MPI-owned global domain
	bool distributed_mode = false; // immutable after construction; avoids MPI checks in per-cell host access
	bool initialized = false; // becomes true after LBM::initialize() has been called
	bool ever_initialized = false; // reset() may request a reinitialize, but immutable runtime topology must stay locked
#ifdef GRAPHICS
	std::uint64_t graphics_session_id = 0ull;
#endif // GRAPHICS

	void sanity_checks_constructor(const vector<Device_Info>& device_infos, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho); // sanity checks on grid resolution and extension support
	void sanity_checks_initialization(); // sanity checks during initialization on used extensions based on used flags
	void initialize(); // write all data fields to device and call kernel_initialize
	void do_time_step(); // call kernel_stream_collide to perform one LBM time step

	void communicate_field(
		const enum_transfer_field field, const uint bytes_per_cell,
		const CudaTransferChannel channel =
			CudaTransferChannel::simulation);

	void communicate_fi();
	void communicate_rho_u_flags();
#ifdef GRAPHICS
	void communicate_graphics_rho_u_flags();
#endif // GRAPHICS
	void communicate_flags();
#ifdef SURFACE
	void communicate_phi_massex_flags();
#endif // SURFACE
#ifdef TEMPERATURE
	void communicate_fi_gi();
	void communicate_gi();
	void communicate_T();
#ifdef GRAPHICS
	void communicate_graphics_T();
#endif // GRAPHICS
#endif // TEMPERATURE

public:
	template<typename T> class Memory_Container { // does not hold any data itsef, just links to LBM_Domain data
	private:
		ulong N = 0ull; // buffer length
		uint d = 1u; // buffer dimensions
		LBM* lbm = nullptr;
		Memory<T>** buffers = nullptr; // process-local host buffers
		string name = "";
		vector<T> distributed_data; // replicated host view used for deterministic setup and root I/O

		uint Nx=1u, Ny=1u, Nz=1u, Dx=1u, Dy=1u, Dz=1u, D=1u, LD=1u; // auxiliary variables: global topology and process-local domain count
		uint NxDx=1u, NyDy=1u, NzDz=1u, Hx=0u, Hy=0u, Hz=0u; // auxiliary variables: number of domains, shortcuts for N_/D_, halo offsets
		ulong NxNy=1ull, local_Nx=1ull, local_Ny=1ull, local_Nz=1ull, local_N=1ull; // auxiliary variables: shortcut for Nx*Ny, size of each domain, number of cells in each domain
		inline void initialize_auxiliary_variables() { // these variables are frequently used in reference() functions, so pre-compute them only once here
			Nx = lbm->get_Nx(); Ny = lbm->get_Ny(); Nz = lbm->get_Nz();
			Dx = lbm->get_Dx(); Dy = lbm->get_Dy(); Dz = lbm->get_Dz();
			D = Dx*Dy*Dz; // global number of domains
			LD = lbm->get_D(); // process-local number of domains
			NxNy = (ulong)Nx*(ulong)Ny; // shortcut for Nx*Ny
			NxDx=Nx/Dx; NyDy=Ny/Dy; NzDz=Nz/Dz; // shortcuts for N_/D_
			Hx=Dx>1u; Hy=Dy>1u; Hz=Dz>1u; // halo offsets
			local_Nx=(ulong)(NxDx+2u*Hx); local_Ny=(ulong)(NyDy+2u*Hy); local_Nz=(ulong)(NzDz+2u*Hz); // size of each domain
			local_N = local_Nx*local_Ny*local_Nz; // number of cells in each domain
		}
		inline void initialize_auxiliary_pointers() {
			/********/ x = Pointer(this, 0x0u);
			if(d>0x1u) y = Pointer(this, 0x1u);
			if(d>0x2u) z = Pointer(this, 0x2u);
		}
		inline T& reference(const ulong i) { // stitch together domain buffers and make them appear as one single large buffer
			if(D==1u) { // take shortcut for single domain
				return buffers[0]->data()[i]; // array of structures
			} else { // decompose index for multiple domains
				const ulong global_i=i%N, t=global_i%NxNy;
				const uint x=(uint)(t%(ulong)Nx), y=(uint)(t/(ulong)Nx), z=(uint)(global_i/NxNy); // n = x+(y+z*Ny)*Nx
				const uint px=x%NxDx, py=y%NyDy, pz=z%NzDz, dx=x/NxDx, dy=y/NyDy, dz=z/NzDz, domain=dx+(dy+dz*Dy)*Dx; // 3D position within domain and which domain
				const ulong local_i = (ulong)(px+Hx)+((ulong)(py+Hy)+(ulong)(pz+Hz)*local_Ny)*local_Nx; // add halo offsets
				const ulong local_dimension = i/N;
				if(lbm->is_distributed()) {
					if(lbm->owns_domain(domain)) return buffers[0]->data()[local_i+local_dimension*local_N];
					return distributed_data[global_i+local_dimension*N];
				}
				return buffers[domain]->data()[local_i+local_dimension*local_N]; // array of structures
			}
		}
		inline T& reference(const ulong i, const uint dimension) { // stitch together domain buffers and make them appear as one single large buffer
			if(D==1u) { // take shortcut for single domain
				return buffers[0]->data()[i+(ulong)dimension*N]; // array of structures
			} else { // decompose index for multiple domains
				const ulong global_i=i%N, t=global_i%NxNy;
				const uint x=(uint)(t%(ulong)Nx), y=(uint)(t/(ulong)Nx), z=(uint)(global_i/NxNy); // n = x+(y+z*Ny)*Nx
				const uint px=x%NxDx, py=y%NyDy, pz=z%NzDz, dx=x/NxDx, dy=y/NyDy, dz=z/NzDz, domain=dx+(dy+dz*Dy)*Dx; // 3D position within domain and which domain
				const ulong local_i = (ulong)(px+Hx)+((ulong)(py+Hy)+(ulong)(pz+Hz)*local_Ny)*local_Nx; // add halo offsets
				const ulong local_dimension = max(i/N, (ulong)dimension);
				if(lbm->is_distributed()) {
					if(lbm->owns_domain(domain)) return buffers[0]->data()[local_i+local_dimension*local_N];
					return distributed_data[global_i+local_dimension*N];
				}
				return buffers[domain]->data()[local_i+local_dimension*local_N]; // array of structures
			}
		}
		inline void gather_distributed_data() {
			if(!lbm->is_distributed()) return;
			const ulong inner_N=(ulong)NxDx*(ulong)NyDy*(ulong)NzDz;
			vector<T> local(inner_N*(ulong)d);
			for(uint dimension=0u; dimension<d; dimension++) {
				for(uint z=0u; z<NzDz; z++) for(uint y=0u; y<NyDy; y++) for(uint x=0u; x<NxDx; x++) {
					const ulong inner_i=(ulong)x+((ulong)y+(ulong)z*(ulong)NyDy)*(ulong)NxDx;
					const ulong local_i=(ulong)(x+Hx)+((ulong)(y+Hy)+(ulong)(z+Hz)*local_Ny)*local_Nx;
					local[inner_i+(ulong)dimension*inner_N]=buffers[0]->data()[local_i+(ulong)dimension*local_N];
				}
			}
			vector<T> gathered(inner_N*(ulong)d*(ulong)D);
			culbm::mpi_allgather_equal_bytes(local.data(), (uint64_t)(local.size()*sizeof(T)), gathered.data());
			for(uint domain=0u; domain<D; domain++) {
				const uint domain_x=(domain%(Dx*Dy))%Dx, domain_y=(domain%(Dx*Dy))/Dx, domain_z=domain/(Dx*Dy);
				for(uint dimension=0u; dimension<d; dimension++) {
					for(uint z=0u; z<NzDz; z++) for(uint y=0u; y<NyDy; y++) for(uint x=0u; x<NxDx; x++) {
						const ulong inner_i=(ulong)x+((ulong)y+(ulong)z*(ulong)NyDy)*(ulong)NxDx;
						const ulong global_x=(ulong)domain_x*(ulong)NxDx+(ulong)x;
						const ulong global_y=(ulong)domain_y*(ulong)NyDy+(ulong)y;
						const ulong global_z=(ulong)domain_z*(ulong)NzDz+(ulong)z;
						const ulong global_i=global_x+(global_y+global_z*(ulong)Ny)*(ulong)Nx;
						distributed_data[global_i+(ulong)dimension*N]=gathered[(ulong)domain*inner_N*(ulong)d+(ulong)dimension*inner_N+inner_i];
					}
				}
			}
		}
		inline string vtk_type() const {
			/**/ if constexpr(std::is_same<T, char >::value) return "char" ; else if constexpr(std::is_same<T, uchar >::value) return "unsigned_char" ;
			else if constexpr(std::is_same<T, short>::value) return "short"; else if constexpr(std::is_same<T, ushort>::value) return "unsigned_short";
			else if constexpr(std::is_same<T, int  >::value) return "int"  ; else if constexpr(std::is_same<T, uint  >::value) return "unsigned_int"  ;
			else if constexpr(std::is_same<T, slong>::value) return "long" ; else if constexpr(std::is_same<T, ulong >::value) return "unsigned_long" ;
			else if constexpr(std::is_same<T, float>::value) return "float"; else if constexpr(std::is_same<T, double>::value) return "double"        ;
			else print_error("Error in vtk_type(): Type not supported.");
			return "";
		}
		inline void write_vtk(const string& path, const bool convert_to_si_units=true, const bool print_saved_message=true, const uint Nz_write=0u) { // write binary .vtk file
			if(!culbm::mpi_is_root()) return;
			float spacing = 1.0f;
			T unit_conversion_factor = (T)1;
			if(convert_to_si_units) {
				spacing = units.si_x(1.0f);
				if(name=="rho") unit_conversion_factor = (T)units.si_rho(1.0f);
				if(name=="u"  ) unit_conversion_factor = (T)units.si_u  (1.0f);
				if(name=="F"  ) unit_conversion_factor = (T)units.si_F  (1.0f);
				if(name=="T"  ) unit_conversion_factor = (T)units.si_dT (1.0f);
			}
			const string filename = create_file_extension(path, ".vtk");
			const uint Nz_out = (Nz_write>0u&&Nz_write<Nz) ? Nz_write : Nz;
			const ulong points = (ulong)Nx*(ulong)Ny*(ulong)Nz_out;
			float3 origin = spacing*float3(0.5f-0.5f*(float)Nx, 0.5f-0.5f*(float)Ny, 0.5f-0.5f*(float)Nz);
			if(convert_to_si_units) origin += vtk_origin_shift;
			const string header =
				"# vtk DataFile Version 3.0\nCULBM "+filename.substr(filename.rfind('/')+1)+"\nBINARY\nDATASET STRUCTURED_POINTS\n"
				"DIMENSIONS "+to_string(Nx)+" "+to_string(Ny)+" "+to_string(Nz_out)+"\n"
				"ORIGIN "+to_string(origin.x)+" "+to_string(origin.y)+" "+to_string(origin.z)+"\n"
				"SPACING "+to_string(spacing)+" "+to_string(spacing)+" "+to_string(spacing)+"\n"
				"POINT_DATA "+to_string(points)+"\n"
				"SCALARS data "+vtk_type()+" "+to_string(dimensions())+"\nLOOKUP_TABLE default\n"
			;
			const uint chunk_size_MB = 4u*thread::hardware_concurrency(); // in MB; convert and write data in chunks, to reduce memory footprint and time for large memory allocation
			const ulong chunk_elements = (1048576ull*(ulong)chunk_size_MB)/((ulong)dimensions()*sizeof(T));
			const ulong chunks=points/chunk_elements, chunk_remainder=points%chunk_elements;
			T* data = new T[chunk_elements*(ulong)dimensions()];
			create_folder(filename);
			std::ofstream file(filename, std::ios::out|std::ios::binary);
			file.write(header.c_str(), header.length()); // write non-binary file header
			for(ulong c=0u; c<chunks+1ull; c++) { // iterate over all full chunks + last chunk_remainder chunk
				const ulong N = c<chunks ? chunk_elements : chunk_remainder;
				if(N==0ull) break; // chunk_remainder may be 0, then skip last iteration
				parallel_for(N, [&](ulong i) {
					for(uint d=0u; d<dimensions(); d++) { // LBM to SI units, LittleEndian to BigEndian, AoS to SoA
						const T value = reference(c*chunk_elements+i, d);
						const T si_value = convert_to_si_units&&name=="T" ? (T)units.si_T((float)value) : (T)(unit_conversion_factor*value);
						data[i*(ulong)dimensions()+(ulong)d] = reverse_bytes(si_value);
					}
				});
				file.write((char*)data, N*(ulong)dimensions()*sizeof(T)); // write binary data
			}
			file.close();
			delete[] data;
			if(print_saved_message) {
				info.allow_printing.lock();
				print_info("File \""+filename+"\" saved.");
				info.allow_printing.unlock();
			}
		}

	public:
		class Pointer {
		private:
			Memory_Container* memory = nullptr;
			uint dimension = 0u;
		public:
			inline Pointer() {}; // default constructor
			inline Pointer(Memory_Container* memory, const uint dimension) {
				this->memory = memory;
				this->dimension = dimension;
			}
			inline T& operator[](const ulong i) { return memory->reference(i, dimension); }
			inline const T& operator[](const ulong i) const { return memory->reference(i, dimension); }
		};
		Pointer x, y, z; // host buffer auxiliary pointers for multi-dimensional array access (array of structures)

		inline Memory_Container(LBM* lbm, Memory<T>** buffers, const string& name) {
			this->N = lbm->get_N();
			this->d = buffers[0]->dimensions();
			if(this->N*(ulong)this->d==0ull) print_error("Memory size must be larger than 0.");
			this->lbm = lbm;
			this->buffers = buffers;
			this->name = name;
			initialize_auxiliary_variables();
			if(lbm->is_distributed()) {
				distributed_data.resize(N*(ulong)d);
				for(uint dimension=0u; dimension<d; dimension++) std::fill(distributed_data.begin()+(ulong)dimension*N, distributed_data.begin()+(ulong)(dimension+1u)*N, buffers[0]->data()[(ulong)dimension*local_N]);
			}
			initialize_auxiliary_pointers();
		}
		inline Memory_Container() {} // default constructor
		inline Memory_Container& operator=(Memory_Container&& memory) noexcept { // move assignment
			this->N = memory.N;
			this->d = memory.d;
			this->lbm = memory.lbm;
			this->buffers = memory.buffers;
			this->name = memory.name;
			this->distributed_data = std::move(memory.distributed_data);
			initialize_auxiliary_variables();
			initialize_auxiliary_pointers();
			return *this;
		}
		inline void reset(const T value=(T)0) {
			if(lbm->is_distributed()) std::fill(distributed_data.begin(), distributed_data.end(), value);
			for(uint domain=0u; domain<LD; domain++) buffers[domain]->reset(value);
		}
		inline const ulong length() const { return N; }
		inline const uint dimensions() const { return d; }
		inline const ulong range() const { return N*(ulong)d; }
		inline const ulong capacity() const { return N*(ulong)d*sizeof(T); } // returns capacity of the buffer in Byte
		inline T& operator[](const ulong i) { return reference(i); }
		inline const T& operator[](const ulong i) const { return reference(i); }
		inline const T operator()(const ulong i) const { return reference(i); }
		inline const T operator()(const ulong i, const uint dimension) const { return reference(i, dimension); } // array of structures
		inline void read_from_device() {
#ifndef UPDATE_FIELDS
			for(uint domain=0u; domain<LD; domain++) lbm->lbm_domain[domain]->enqueue_update_fields(); // make sure data in device memory is up-to-date
#endif // UPDATE_FIELDS
			for(uint domain=0u; domain<LD; domain++) buffers[domain]->enqueue_read_from_device();
			for(uint domain=0u; domain<LD; domain++) buffers[domain]->finish_queue();
			gather_distributed_data();
		}
		inline void write_to_device() {
			for(uint domain=0u; domain<LD; domain++) buffers[domain]->enqueue_write_to_device();
			for(uint domain=0u; domain<LD; domain++) buffers[domain]->finish_queue();
		}
		inline void write_host_to_vtk(const string& path="", const bool convert_to_si_units=true, const bool print_saved_message=true, const uint Nz_write=0u) { // write binary .vtk file
			write_vtk(default_filename(path, name, ".vtk", lbm->get_t()), convert_to_si_units, print_saved_message, Nz_write);
		}
		inline void write_device_to_vtk(const string& path="", const bool convert_to_si_units=true, const bool print_saved_message=true, const uint Nz_write=0u) { // write binary .vtk file
			read_from_device();
			write_host_to_vtk(path, convert_to_si_units, print_saved_message, Nz_write);
		}
	};

	LBM_Domain** lbm_domain; // one LBM domain per GPU

	Memory_Container<float> rho; // density of every cell
	Memory_Container<float> u; // velocity of every cell
	Memory_Container<uchar> flags; // flags of every cell
#ifdef FORCE_FIELD
	Memory_Container<float> F; // individual force for every cell
#endif // FORCE_FIELD
#ifdef SURFACE
	Memory_Container<float> phi; // fill level of every cell
#endif // SURFACE
#ifdef TEMPERATURE
	Memory_Container<float> T; // temperature of every cell
#endif // TEMPERATURE
#ifdef PARTICLES
	Memory<float>* particles; // particle positions
#endif // PARTICLES

	LBM(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=0.0f); // compiles CUDA source and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=1.0f); // compiles CUDA source and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const uint particles_N, const float particles_rho=1.0f); // compiles CUDA source and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho=1.0f); // compiles CUDA source and allocates memory
	LBM(const uint3 N, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=0.0f); // compiles CUDA source and allocates memory
	LBM(const uint3 N, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=1.0f); // compiles CUDA source and allocates memory
	LBM(const uint3 N, const float nu, const uint particles_N, const float particles_rho=1.0f); // compiles CUDA source and allocates memory
	LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho=1.0f); // compiles CUDA source and allocates memory
	~LBM();

	// CUDA nested-grid runtime entry points.  They expose initialization and a
	// single native step without changing LBM::run() or the ordinary solver path.
	void external_initialize();
	void external_update_free_slip_boundaries(const uint face_mask);
	void external_step();
	void external_finish();
	void external_synchronize_flags(); // nested bridge: publish device-side interface flags into x/y/z halos
	bool external_initialized() const { return initialized; }

	void run(const ulong steps=max_ulong, const ulong total_steps=max_ulong); // initializes the LBM simulation (copies data to device and runs initialize kernel), then runs LBM
	void finish_case(); // drain graphics and align distributed ranks before this solver is destroyed
#ifdef GRAPHICS
	std::uint64_t get_graphics_session_id() const {
		return graphics_session_id;
	}
#endif // GRAPHICS
	void update_fields(); // update fields (rho, u, T) manually
	// Samples the current pre-collision DDF state on the GPU without advancing
	// time or modifying rho/u/T.  The extra full-field work occurs only here.
	Subgrid_Statistics subgrid_statistics();
	// Make the current velocity and flag fields, including inter-domain halo
	// cells, available to a following CUDA advection kernel.
	void prepare_advection_fields();
	bool enable_device_statistics(
		const bool include_temperature,
		const uint future_device_memory_mb=0u);
	void disable_device_statistics();
	void reset_device_statistics();
	void enqueue_device_statistics_sample(const ulong sample_count);
	void finish_device_statistics();
	bool uses_device_statistics() const {
		return get_D() > 0u && lbm_domain[0]->has_device_statistics();
	}
	void read_device_statistics(vector<float>& mean_u,
	                            vector<float>& mean_rho,
	                            vector<float>& mean_T,
	                            vector<float>& M2_u,
	                            vector<float>& M2_v,
	                            vector<float>& M2_w);
	bool configure_device_velocity_probes(
		const vector<ulong>& global_indices,
		const uint future_device_memory_mb=0u);
	void disable_device_velocity_probes();
	bool uses_device_velocity_probes() const;
	void sample_device_velocity_probes(vector<float>& samples);
	void reset(); // reset simulation (takes effect in following run() call)
#ifdef FORCE_FIELD
	void update_force_field(); // calculate forces from fluid on TYPE_S cells
	float3 object_center_of_mass(const uchar flag_marker=TYPE_S); // calculate center of mass of all cells flagged with flag_marker
	float3 object_force(const uchar flag_marker=TYPE_S); // add up force for all cells flagged with flag_marker
	float3 object_torque(const float3& rotation_center, const uchar flag_marker=TYPE_S); // add up torque around specified rotation_center for all cells flagged with flag_marker
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	void update_moving_boundaries(); // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#if defined(PARTICLES)&&!defined(FORCE_FIELD)
	void integrate_particles(const ulong steps=max_ulong, const ulong total_steps=max_ulong, const uint time_step_multiplicator=1u); // intgegrate passive tracer particles forward in time in stationary flow field
#endif // PARTICLES&&!FORCE_FIELD

	uint get_Nx() const { return Nx; } // get (global) lattice dimensions in x-direction
	uint get_Ny() const { return Ny; } // get (global) lattice dimensions in y-direction
	uint get_Nz() const { return Nz; } // get (global) lattice dimensions in z-direction
	ulong get_N() const { return (ulong)Nx*(ulong)Ny*(ulong)Nz; } // get (global) number of lattice points
	uint get_Dx() const { return Dx; } // get lattice domains in x-direction
	uint get_Dy() const { return Dy; } // get lattice domains in y-direction
	uint get_Dz() const { return Dz; } // get lattice domains in z-direction
	uint get_D() const { return local_D; } // get process-local number of lattice domains
	uint get_global_D() const { return Dx*Dy*Dz; } // get global Cartesian domain count
	bool is_distributed() const { return distributed_mode; }
	uint get_global_domain_id() const { return global_domain_id; }
	bool owns_domain(const uint domain) const { return !is_distributed() || domain==global_domain_id; }
	uint local_domain_index(const uint domain) const { return is_distributed() ? 0u : domain; }
	LBM_Domain& get_local_domain(const uint domain) { return *lbm_domain[domain]; }
	const LBM_Domain& get_local_domain(const uint domain) const { return *lbm_domain[domain]; }
	float get_nu() const { return lbm_domain[0]->get_nu(); } // get kinematic shear viscosity
	float get_tau() const { return 3.0f*get_nu()+0.5f; } // get LBM relaxation time
	float get_Re_max() const { return 0.57735027f*sqrt((float)(sq(Nx)+sq(Ny)+sq(Nz)))/get_nu(); } // Re < Re_max = c*L_max/nu
	float get_fx() const { return lbm_domain[0]->get_fx(); } // get global froce per volume
	float get_fy() const { return lbm_domain[0]->get_fy(); } // get global froce per volume
	float get_fz() const { return lbm_domain[0]->get_fz(); } // get global froce per volume
	float get_omega_x() const { return lbm_domain[0]->get_omega_x(); } // get Coriolis rotation x-component
	float get_omega_y() const { return lbm_domain[0]->get_omega_y(); } // get Coriolis rotation y-component
	float get_omega_z() const { return lbm_domain[0]->get_omega_z(); } // get Coriolis rotation z-component
	float get_sigma() const { return lbm_domain[0]->get_sigma(); } // get surface tension coefficient
	float get_alpha() const { return lbm_domain[0]->get_alpha(); } // get thermal diffusion coefficient
	float get_beta() const { return lbm_domain[0]->get_beta(); } // get thermal expansion coefficient
#ifdef TEMPERATURE
	float3 get_buoyancy_gravity() const {
		return float3(lbm_domain[0]->get_buoyancy_gx(),
		              lbm_domain[0]->get_buoyancy_gy(),
		              lbm_domain[0]->get_buoyancy_gz());
	}
	int get_temperature_outlet_face() const {
		return lbm_domain[0]->get_temperature_outlet_face();
	}
#endif // TEMPERATURE
	ulong get_t() const { return lbm_domain[0]->get_t(); } // get discrete time step in LBM units
	uint get_velocity_set() const { return lbm_domain[0]->get_velocity_set(); }
	void set_fx(const float fx) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fx(fx); } // set global froce per volume
	void set_fy(const float fy) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fy(fy); } // set global froce per volume
	void set_fz(const float fz) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fz(fz); } // set global froce per volume
	void set_f(const float fx, const float fy, const float fz) { set_fx(fx); set_fy(fy); set_fz(fz); } // set global froce per volume
	void set_coriolis(const float omega_x, const float omega_y, const float omega_z) {
		for (uint d = 0u; d < get_D(); d++) lbm_domain[d]->set_coriolis(omega_x, omega_y, omega_z);
	}
#ifdef TEMPERATURE
	void set_buoyancy_gravity(const float gx, const float gy, const float gz) {
		for(uint d=0u; d<get_D(); d++)
			lbm_domain[d]->set_buoyancy_gravity(gx, gy, gz);
	}
	bool set_temperature_outlet_face(const int face) {
		if(face<(int)Temperature_Outlet_Face::None ||
		   face>(int)Temperature_Outlet_Face::North) return false;
		if(ever_initialized) return false; // changing face would leave TYPE_T on an old host/device face, even after reset()
		if((face==(int)Temperature_Outlet_Face::West ||
		    face==(int)Temperature_Outlet_Face::East) && Nx/Dx<2u) return false;
		if((face==(int)Temperature_Outlet_Face::South ||
		    face==(int)Temperature_Outlet_Face::North) && Ny/Dy<2u) return false;
		for(uint d=0u; d<get_D(); d++)
			lbm_domain[d]->set_temperature_outlet_face(face);
		return true;
	}
	bool set_temperature_outlet_face(const Temperature_Outlet_Face face) {
		return set_temperature_outlet_face((int)face);
	}
#endif // TEMPERATURE

	void coordinates(const ulong n, uint& x, uint& y, uint& z) const { // disassemble 1D linear index to 3D coordinates (n -> x,y,z)
		const ulong t = n%((ulong)Nx*(ulong)Ny); // n = x+(y+z*Ny)*Nx
		x = (uint)(t%(ulong)Nx);
		y = (uint)(t/(ulong)Nx);
		z = (uint)(n/((ulong)Nx*(ulong)Ny));
	}
	void coordinates(const float3& p, uint& x, uint& y, uint& z) const { // turn 3D position into closest 3D grid coordinates
		const float3 mp = mirror_position(p);
		x = (uint)(mp.x+1.5f*(float)Nx)%Nx;
		y = (uint)(mp.y+1.5f*(float)Ny)%Ny;
		z = (uint)(mp.z+1.5f*(float)Nz)%Nz;
	}
	ulong index(const uint x, const uint y, const uint z) const { // turn 3D coordinates into 1D linear index
		return (ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx;
	}
	ulong index(const uint3 xyz) const { // turn 3D coordinates into 1D linear index
		return index(xyz.x, xyz.y, xyz.z);
	}
	ulong index(const float3& p) const { // turn 3D position into closest 1D linear index
		uint x=0u, y=0u, z=0u;
		coordinates(p, x, y, z);
		return index(x, y, z);
	}
	float3 position(const uint x, const uint y, const uint z) const { // returns position in box [-Nx/2, Nx/2] x [-Ny/2, Ny/2] x [-Nz/2, Nz/2]
		return float3((float)x-0.5f*(float)Nx+0.5f, (float)y-0.5f*(float)Ny+0.5f, (float)z-0.5f*(float)Nz+0.5f);
	}
	float3 position(const ulong n) const { // returns position in box [-Nx/2, Nx/2] x [-Ny/2, Ny/2] x [-Nz/2, Nz/2]
		uint x, y, z;
		coordinates(n, x, y, z);
		return position(x, y, z);
	}
	float3 mirror_position(const float3& p) const { // mirror position into periodic boundaries
		float3 r;
		r.x = sign(p.x)*(fmod(fabs(p.x)+0.5f*(float)Nx, (float)Nx)-0.5f*(float)Nx);
		r.y = sign(p.y)*(fmod(fabs(p.y)+0.5f*(float)Ny, (float)Ny)-0.5f*(float)Ny);
		r.z = sign(p.z)*(fmod(fabs(p.z)+0.5f*(float)Nz, (float)Nz)-0.5f*(float)Nz);
		return r;
	}
	float3 size() const { // returns size of box
		return float3((float)Nx, (float)Ny, (float)Nz);
	}
	float3 center() const { // returns center of box
		return float3(0.5f*(float)Nx-0.5f, 0.5f*(float)Ny-0.5f, 0.5f*(float)Nz-0.5f);
	}
	uint smallest_side_length() const {
		return min(min(Nx, Ny), Nz);
	}
	uint largest_side_length() const {
		return max(max(Nx, Ny), Nz);
	}
	float3 relative_position(const uint x, const uint y, const uint z) const { // returns relative position in box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
		return float3(((float)x+0.5f)/(float)Nx-0.5f, ((float)y+0.5f)/(float)Ny-0.5f, ((float)z+0.5f)/(float)Nz-0.5f);
	}
	float3 relative_position(const ulong n) const { // returns relative position in box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
		uint x, y, z;
		coordinates(n, x, y, z);
		return relative_position(x, y, z);
	}
	void write_status(const string& path=""); // write LBM status report to a .txt file

	void voxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S, const float3& rotation_center=float3(0.0f), const float3& linear_velocity=float3(0.0f), const float3& rotational_velocity=float3(0.0f)); // voxelize mesh
	void unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S); // remove voxelized triangle mesh from LBM grid
	void write_mesh_to_vtk(const Mesh* mesh, const string& path="", const bool convert_to_si_units=true) const; // write mesh to binary .vtk file
	void voxelize_stl(const string& path, const float3& center, const float3x3& rotation, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file
	void voxelize_stl(const string& path, const float3x3& rotation, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (place in box center)
	void voxelize_stl(const string& path, const float3& center, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (no rotation)
	void voxelize_stl(const string& path, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (place in box center, no rotation)

#ifdef GRAPHICS
	class Graphics {
	private:
		LBM* lbm = nullptr;
		ulong last_snapshot_time = max_ulong;
		ulong last_rho_halo_time = max_ulong, last_T_halo_time = max_ulong; // graphics halo refresh tracking
		ulong distributed_frame_id = 0ull;
		int last_visualization_modes=0, last_field_mode=0, last_slice_mode=0, last_slice_x=0, last_slice_y=0, last_slice_z=0; // don't render a new frame if the scene hasn't changed since last frame
		void default_settings() {
			visualization_modes |= VIS_FLAG_LATTICE;
#ifdef PARTICLES
			visualization_modes |= VIS_PARTICLES;
#endif // PARTICLES
		}

	public:
		int visualization_modes=0, field_mode=0, slice_mode=0, slice_x=0, slice_y=0, slice_z=0; // field_mode = { 0 (u), 1 (rho), 2 (T) }, slice_mode = { 0 (no slice), 1 (x), 2 (y), 3 (z), 4 (xz), 5 (xyz), 6 (yz), 7 (xy) }, slice_{xyz} = position of slices

		Graphics() {} // default constructor
		Graphics(LBM* lbm) {
			this->lbm = lbm;
			camera.set_zoom(0.5f*(float)fmax(fmax(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz()));
			slice_x = (int)lbm->get_Nx()/2;
			slice_y = (int)lbm->get_Ny()/2;
			slice_z = (int)lbm->get_Nz()/2;
			default_settings();
		}
		Graphics& operator=(const Graphics& graphics) { // copy assignment
			lbm = graphics.lbm;
			visualization_modes = graphics.visualization_modes;
			field_mode = graphics.field_mode;
			slice_mode = graphics.slice_mode;
			slice_x = graphics.slice_x;
			slice_y = graphics.slice_y;
			slice_z = graphics.slice_z;
			return *this;
		}

		int* draw_frame(); // main rendering function, calls rendering kernels

		void set_camera_centered(const float rx=0.0f, const float ry=0.0f, const float fov=100.0f, const float zoom=1.0f); // set camera centered
		void set_camera_free(const float3& p=float3(0.0f), const float rx=0.0f, const float ry=0.0f, const float fov=100.0f); // set camera free
		void print_frame(); // preview preview of current frame in console
	}; // Graphics
	Graphics graphics;
#endif // GRAPHICS
}; // LBM
