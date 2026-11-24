#include "kernel_set.hpp"
#include "kernel.hpp"

#include <algorithm>
#include <cstdlib>

// Inflow, nudging and sponge constants owned by the case setup.
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

float coriolis_f_lbmu = 0.0f;

namespace culbm::solver {
namespace {

// Emits preprocessor feature flags and typed C++17 device configuration constants.
class DeviceConfigBuilder {
public:
	DeviceConfigBuilder& flag(const string& name) {
		flags_ += "\n#define ";
		flags_ += name;
		return *this;
	}
	DeviceConfigBuilder& alias(const string& name, const string& target) {
		aliases_ += "\nusing ";
		aliases_ += name;
		aliases_ += " = ";
		aliases_ += target;
		aliases_ += ';';
		return *this;
	}
	template<typename V> DeviceConfigBuilder& u32(const string& name, const V value) {
		return constant("uint", name, to_string(value)+"u");
	}
	DeviceConfigBuilder& u64(const string& name, const ulong value) {
		return constant("ulong", name, to_string(value)+"ull");
	}
	template<typename V> DeviceConfigBuilder& i32(const string& name, const V value) {
		return constant("int", name, to_string(value));
	}
	DeviceConfigBuilder& f32(const string& name, const float value) {
		return constant("float", name, to_string(value)+"f");
	}
	DeviceConfigBuilder& f32_expr(const string& name, const string& expr) {
		return constant("float", name, expr);
	}
	DeviceConfigBuilder& boolean(const string& name, const bool value) {
		return constant("bool", name, value ? "true" : "false");
	}
	string build() const {
		return flags_+aliases_+"\nnamespace kcfg {"+constants_+"\n}\n";
	}

private:
	DeviceConfigBuilder& constant(const char* type, const string& name, const string& value) {
		constants_ += "\n\tconstexpr ";
		constants_ += type;
		constants_ += ' ';
		constants_ += name;
		constants_ += " = ";
		constants_ += value;
		constants_ += ';';
		return *this;
	}
	string flags_;
	string aliases_;
	string constants_;
};

void add_geometry(DeviceConfigBuilder& d, const Placement& p, const Decomposition& grid) {
	const uint nx = p.extent[0], ny = p.extent[1], nz = p.extent[2];
	d.alias("cell_idx_t", p.cells()<=(ulong)max_uint ? "uint" : "ulong");
	d.u32("nx", nx).u32("ny", ny).u32("nz", nz);
	d.u64("cells", p.cells()).u64("plane_xy", (ulong)nx*(ulong)ny);
	d.u32("split_x", grid.split(0u)).u32("split_y", grid.split(1u)).u32("split_z", grid.split(2u));
	d.i32("origin_x", p.origin[0]).i32("origin_y", p.origin[1]).i32("origin_z", p.origin[2]);
	d.u32("global_nx", grid.size(0u)).u32("global_ny", grid.size(1u)).u32("global_nz", grid.size(2u));
	// Local coordinates of the global outer faces; a face lies in this
	// subdomain when its coordinate is inside the local extent.
	const int west = -p.origin[0];
	const int east = (int)grid.size(0u)-1-p.origin[0];
	const int south = -p.origin[1];
	const int north = (int)grid.size(1u)-1-p.origin[1];
	const int top = (int)grid.size(2u)-1-p.origin[2];
	d.i32("west_x", west).i32("east_x", east);
	d.i32("south_y", south).i32("north_y", north);
	d.i32("top_z", top);
	const auto inside = [](const int c, const uint n) { return c>=0 && c<(int)n; };
	d.boolean("has_west", inside(west, nx)).boolean("has_east", inside(east, nx));
	d.boolean("has_south", inside(south, ny)).boolean("has_north", inside(north, ny));
	d.boolean("has_top", inside(top, nz));
	d.u32("area_x", ny*nz).u32("area_y", nz*nx).u32("area_z", nx*ny);
}

void add_lattice(DeviceConfigBuilder& d, const FeatureSet& f, const Transport& transport) {
	d.flag("D3Q"+to_string(f.velocity_set));
	d.u32("q_mom", f.velocity_set).u32("dims", 3u).u32("q_mom_face", f.face_ddfs);
	d.f32_expr("cs", "0.57735027f");
	d.f32("tau", transport.tau()).f32("omega", 1.0f/transport.tau());
#if defined(TRT)
	d.f32("trt_magic", trt_magic_parameter);
#endif
	if(f.velocity_set==27u) {
		d.f32_expr("w0", "(1.0f/3.375f)").f32_expr("ws", "(1.0f/13.5f)")
			.f32_expr("we", "(1.0f/54.0f)").f32_expr("wc", "(1.0f/216.0f)");
	} else {
		d.f32_expr("w0", "(1.0f/3.0f)").f32_expr("ws", "(1.0f/18.0f)").f32_expr("we", "(1.0f/36.0f)");
	}
#if defined(SRT)
	d.flag("SRT");
#elif defined(TRT)
	d.flag("TRT");
#endif
	switch(f.storage) {
	case DDF_Storage::FP16S:
		d.flag("DDF_FP16S").alias("ddf_elem_t", "half").alias("ddf_raw_t", "ushort");
		break;
	case DDF_Storage::FP16C:
		d.flag("DDF_FP16C").alias("ddf_elem_t", "ushort").alias("ddf_raw_t", "ushort");
		break;
	default:
		d.flag("DDF_FP32").alias("ddf_elem_t", "float").alias("ddf_raw_t", "float");
		break;
	}
}

void add_physics(DeviceConfigBuilder& d, const FeatureSet& f, const Transport& transport) {
#ifdef UPDATE_FIELDS
	d.flag("UPDATE_FIELDS");
#endif
	d.flag("VOLUME_FORCE");
#ifdef FREE_SLIP_BOUNDARIES
	d.flag("FREE_SLIP_BOUNDARIES"); // validation builds: mirrored outer walls
#endif
	d.flag("EQUILIBRIUM_BOUNDARIES");
	d.flag("FORCE_FIELD");
	if(f.temperature_enabled) {
		d.flag("TEMPERATURE");
		d.u32("q_temp", f.temperature_set).u32("q_temp_face", f.temperature_face_ddfs);
		d.flag(f.temperature==Temperature_Lattice::Q19 ? "TEMPERATURE_D3Q19" : "TEMPERATURE_D3Q7");
		d.f32("omega_T", 1.0f/(0.5f+transport.alpha/f.temperature_cs2));
		d.f32("beta", transport.beta).f32("T_ref", 1.0f);
	}

	const Subgrid_Configuration& sgs = f.subgrid;
	if(sgs.enabled) {
		d.flag("SUBGRID").f32("sgs_prefactor", 18.0f*sqrt(2.0f)*sq(sgs.smagorinsky_constant));
		if(sgs.fixed_eddy_viscosity>=0.0f)
			d.flag("SUBGRID_FIXED_EDDY_VISCOSITY").f32("sgs_fixed_nu_t", sgs.fixed_eddy_viscosity);
		if(sgs.scalar_coupling && f.temperature_enabled)
			d.flag("SUBGRID_SCALAR").f32("sgs_scalar_tau_factor", 1.0f/(3.0f*sgs.turbulent_prandtl*f.temperature_cs2));
	}

	const Wall_Model_Lattice& wm = f.wall_model;
	if(wm.enabled) {
		d.flag("WALL_MODEL").f32("wm_cd_h", wm.cd_h).f32("wm_cd_v", wm.cd_v);
		d.boolean("wm_smooth_h", wm.smooth_h).boolean("wm_smooth_v", wm.smooth_v);
		d.f32("wm_nu", wm.nu_lbm).f32("wm_kappa", wm.kappa).f32("wm_log_E", wm.log_E).f32("wm_re_laminar", wm.re_laminar);
		if(wm.sgs) d.flag("WALL_MODEL_SGS").f32("wm_sgs_layers", wm.sgs_layers);
		if(wm.ibm) {
			d.flag("WALL_MODEL_IBM").f32("wm_z0_h", wm.z0_h_lu).f32("wm_z0_v", wm.z0_v_lu);
			d.f32("wm_ibm_retraction", wm.ibm_retraction_lu)
				.f32("wm_ibm_first_cell_dist", Wall_Model_Lattice::first_cell_distance_lu);
			d.f32_expr("wm_ibm_min_weight", "0.01f").f32_expr("wm_ibm_cos_h", "0.70710678f");
		}
	}

	if(f.surface_ibm.enabled) {
		d.flag("SURFACE_IBM");
		if(f.surface_ibm.support_z_min>=0)
			d.flag("SURFACE_IBM_BOUNDED_Z").i32("ibm_z_min", f.surface_ibm.support_z_min)
				.i32("ibm_z_max", f.surface_ibm.support_z_max);
	}

	// Rotation, inflow nudging and top sponge come from the case setup.
	d.flag("CORIOLIS").f32("coriolis_omega_x", coriolis_Omegax_lbmu)
		.f32("coriolis_omega_y", coriolis_Omegay_lbmu).f32("coriolis_omega_z", coriolis_Omegaz_lbmu);
	d.f32("coriolis_f", coriolis_f_lbmu).i32("downstream_face", buffer_downstream_face_id);
	if(buffer_nudging_active) {
		d.flag("BUFFER_NUDGING").u32("buffer_cells", buffer_n_cells).f32("buffer_inv_tau", buffer_inv_tau_lbmu);
		d.boolean("buffer_nudge_z", buffer_nudge_vertical==1);
	}
	if(top_sponge_active) {
		d.flag("TOP_SPONGE").u32("sponge_cells", sponge_n_cells).f32("sponge_inv_tau", sponge_inv_tau_lbmu);
		d.i32("sponge_ref_mode", sponge_ref_mode);
	}
}

// Threads per block of stream_collide; CULBM_STREAM_BLOCK_SIZE overrides the
// default for tuning experiments.
uint stream_collide_threads() {
	uint threads = gpu::default_block_size;
	if(const char* const text = std::getenv("CULBM_STREAM_BLOCK_SIZE")) {
		char* end = nullptr;
		const unsigned long requested = std::strtoul(text, &end, 10);
		const bool valid = end!=text && *end=='\0' && requested>=32ul && requested<=1024ul && requested%32ul==0ul;
		if(!valid) fatal_error("CULBM_STREAM_BLOCK_SIZE must be a multiple of 32 in [32, 1024].");
		threads = (uint)requested;
	}
	log_info("stream_collide runs with "+to_string(threads)+" threads per block.");
	return threads;
}

// Slots of the body forces follow the time step (or the four fields for the
// initialization kernel) in every collision kernel.
gpu::Launch& bind_forces(gpu::Launch& kernel, const uint slot, const BodyForces& b, const bool temperature) {
	if(temperature) return kernel.bind(slot, b.fx, b.fy, b.fz, b.omega_x, b.omega_y, b.omega_z, b.gx, b.gy, b.gz);
	return kernel.bind(slot, b.fx, b.fy, b.fz, b.omega_x, b.omega_y, b.omega_z);
}
gpu::Launch& push_forces(gpu::Launch& kernel, const BodyForces& b, const bool temperature) {
	if(temperature) return kernel.push(b.fx, b.fy, b.fz, b.omega_x, b.omega_y, b.omega_z, b.gx, b.gy, b.gz);
	return kernel.push(b.fx, b.fy, b.fz, b.omega_x, b.omega_y, b.omega_z);
}

} // namespace

string kernel_definitions(const Placement& placement, const Decomposition& grid,
	const FeatureSet& features, const Transport& transport) {
	DeviceConfigBuilder d;
	add_geometry(d, placement, grid);
	add_lattice(d, features, transport);
	add_physics(d, features, transport);
	return d.build();
}

string kernel_source(const Placement& placement, const Decomposition& grid,
	const FeatureSet& features, const Transport& transport) {
	return kernel_definitions(placement, grid, features, transport)+get_cuda_kernel_source();
}

KernelSet::KernelSet(const gpu::Device& device, FieldSet& fields, const FeatureSet& features,
	const BodyForces& forces, const int outlet_face, const ulong t)
	: cells_(fields.cells()), temperature_(features.temperature_enabled), subgrid_(features.subgrid.enabled),
	  surface_ibm_(features.surface_ibm.enabled) {
	const Wall_Model_Lattice& wm = features.wall_model;
	has_damping_tables_ = buffer_nudging_active || top_sponge_active;
	if(has_damping_tables_) {
		const ulong table_len = (ulong)std::max(
			buffer_nudging_active ? (buffer_n_cells + 1) : 1,
			top_sponge_active ? std::max(sponge_n_cells, 1) : 1);
		damping_init_ = gpu::Launch(device, "culbm_init_damping_tables", table_len);
	}
	// Argument lists must match the kernel signatures in lbm_kernels.cu,
	// including the order of the optional trailing groups.
	initialize_ = gpu::Launch(device, "initialize", cells_);
	initialize_.push(fields.ddf, fields.density, fields.velocity, fields.flags);
	push_forces(initialize_, forces, temperature_);
	initialize_.push(fields.force);
	init_force_flag_slot_ = initialize_.slot_count();
	initialize_.push((uint)1u);
	if(temperature_) initialize_.push(fields.thermal_ddf, fields.temperature);

	stream_collide_ = gpu::Launch(device, "stream_collide", cells_);
	stream_collide_.block(stream_collide_threads());
	stream_collide_.push(fields.ddf, fields.density, fields.velocity, fields.flags, t);
	push_forces(stream_collide_, forces, temperature_);
	stream_collide_.push(fields.force);
	stream_force_flag_slot_ = stream_collide_.slot_count();
	stream_collide_.push((uint)1u);
	stream_write_flag_slot_ = stream_collide_.slot_count();
	stream_collide_.push((uint)1u);
	if(temperature_) stream_collide_.push(fields.thermal_ddf, fields.temperature);

	update_fields_ = gpu::Launch(device, "update_fields", cells_);
	update_fields_.push(fields.ddf, fields.density, fields.velocity, fields.flags, t);
	push_forces(update_fields_, forces, temperature_);
	update_fields_.push(fields.force);
	update_force_flag_slot_ = update_fields_.slot_count();
	update_fields_.push((uint)1u);
	if(temperature_) update_fields_.push(fields.thermal_ddf, fields.temperature);

	if(temperature_) {
		outlet_ = gpu::Launch(device, "update_temperature_outlet");
		outlet_.push(fields.thermal_ddf, fields.temperature, fields.flags, t, outlet_face);
		free_slip_ = gpu::Launch(device, "culbm_update_free_slip_boundaries", cells_);
		free_slip_.push(fields.density, fields.velocity, fields.temperature, fields.flags, (uint)0u);
	}

	if(subgrid_) {
		write_slot_ = update_fields_.slot_count();
		update_fields_.push((uint)1u);
		sample_slot_ = update_fields_.slot_count();
		update_fields_.push((uint)0u, fields.sgs_sum, fields.sgs_max, fields.sgs_count);
	}
	if(surface_ibm_) {
		stream_collide_.push(fields.ibm_force, fields.ibm_mask);
		// The support list pointer slot holds the force pointer (same size)
		// until SurfaceIBM binds its immutable support list.
		update_fields_.push(fields.ibm_force);
		support_slot_ = update_fields_.slot_count();
		update_fields_.push(fields.ibm_force, (ulong)0ull, (uint)0u);
	}
	if(wm.sgs) {
		stream_collide_.push(fields.wall_columns);
		update_fields_.push(fields.wall_columns);
	}
	// update_fields needs the mask for the free-slip faces of wall-modelled bodies.
	if(wm.ibm && surface_ibm_) update_fields_.push(fields.ibm_mask);
}

void KernelSet::init_damping_tables() {
	if(has_damping_tables_) damping_init_.submit();
}

void KernelSet::initialize(const BodyForces& forces, const uint read_force_field) {
	bind_forces(initialize_, 4u, forces, temperature_).bind(init_force_flag_slot_, read_force_field).submit();
}

void KernelSet::stream_collide(const ulong t, const BodyForces& forces, const uint write_macros,
	const uint read_force_field) {
	stream_collide_.bind(4u, t);
	bind_forces(stream_collide_, 5u, forces, temperature_);
	stream_collide_.bind(stream_force_flag_slot_, read_force_field);
	stream_collide_.bind(stream_write_flag_slot_, write_macros).submit();
}

void KernelSet::prepare_macros(const ulong t, const BodyForces& forces, const uint read_force_field,
	const uint write_macros, const uint sample_sgs) {
	update_fields_.bind(4u, t);
	bind_forces(update_fields_, 5u, forces, temperature_);
	update_fields_.bind(update_force_flag_slot_, read_force_field);
	if(subgrid_) {
		update_fields_.bind(write_slot_, write_macros);
		update_fields_.bind(sample_slot_, sample_sgs);
	}
}

void KernelSet::update_macros(const ulong t, const BodyForces& forces, const uint read_force_field) {
	prepare_macros(t, forces, read_force_field, 1u, 0u);
	if(surface_ibm_) update_fields_.over(cells_).bind(support_slot_+1u, (ulong)0ull, (uint)0u);
	update_fields_.submit();
}

void KernelSet::update_ibm_support(const ulong t, const BodyForces& forces, const uint read_force_field,
	gpu::MirroredBuffer<ulong>& cells, const ulong count) {
	prepare_macros(t, forces, read_force_field, 1u, 0u);
	update_fields_.over(count).bind(support_slot_, cells, count, (uint)1u).submit();
}

void KernelSet::sample_subgrid(const ulong t, const BodyForces& forces, const uint read_force_field) {
	prepare_macros(t, forces, read_force_field, 0u, 1u);
	if(surface_ibm_) update_fields_.over(cells_).bind(support_slot_+1u, (ulong)0ull, (uint)0u);
	update_fields_.submit();
}

void KernelSet::temperature_outlet(const ulong t, const int face, const ulong face_cells) {
	if(!temperature_) return;
	outlet_.over(face_cells).bind(3u, t, face).submit();
}

void KernelSet::free_slip(const uint face_mask) {
	if(!temperature_) return;
	free_slip_.bind(4u, face_mask).submit();
}

} // namespace culbm::solver
