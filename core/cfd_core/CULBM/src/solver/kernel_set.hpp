#pragma once

// Per-subdomain NVRTC specialisation and the configured launches of the
// lattice kernels. The CUDA source of a subdomain is a block of preprocessor
// definitions (lattice size and placement, velocity sets, DDF storage, enabled
// physics and their constants) followed by the common kernel source.

#include "decomposition.hpp"
#include "field_set.hpp"
#include "lattice_config.hpp"
#include "../gpu/gpu.hpp"

extern float coriolis_f_lbmu; // Coriolis parameter in lattice units (setup)

namespace culbm::solver {

// Body forces bound to the collision kernels on every launch.
struct BodyForces {
	float fx = 0.0f, fy = 0.0f, fz = 0.0f; // volume force
	float omega_x = 0.0f, omega_y = 0.0f, omega_z = 0.0f; // Coriolis rotation
	float gx = 0.0f, gy = 0.0f, gz = 0.0f; // Boussinesq gravity
};

// Transport coefficients compiled into the kernels.
struct Transport {
	float nu = 1.0f/6.0f; // kinematic viscosity
	float alpha = 1.0f;   // thermal diffusivity
	float beta = 1.0f;    // thermal expansion coefficient
	float tau() const { return 3.0f*nu+0.5f; }
};

// The NVRTC definitions of one subdomain (prepended to the kernel source).
string kernel_definitions(const Placement& placement, const Decomposition& grid,
	const FeatureSet& features, const Transport& transport);
// Definitions followed by the common kernel source.
string kernel_source(const Placement& placement, const Decomposition& grid,
	const FeatureSet& features, const Transport& transport);

class KernelSet {
public:
	KernelSet() = default;
	KernelSet(const gpu::Device& device, FieldSet& fields, const FeatureSet& features,
		const BodyForces& forces, int outlet_face, ulong t);

	void init_damping_tables();
	void initialize(const BodyForces& forces, uint read_force_field);
	void stream_collide(ulong t, const BodyForces& forces, uint write_macros, uint read_force_field);
	// Recompute rho, u and T from the DDFs on every cell.
	void update_macros(ulong t, const BodyForces& forces, uint read_force_field);
	// Recompute rho, u and T only on the given SurfaceIBM support cells.
	void update_ibm_support(ulong t, const BodyForces& forces, uint read_force_field, gpu::MirroredBuffer<ulong>& cells, ulong count);
	// Accumulate the subgrid diagnostics instead of writing rho, u and T.
	void sample_subgrid(ulong t, const BodyForces& forces, uint read_force_field);
	void temperature_outlet(ulong t, int face, ulong face_cells);
	void free_slip(uint face_mask);

private:
	// update_fields with its optional SGS control and SurfaceIBM support slots.
	void prepare_macros(ulong t, const BodyForces& forces, uint read_force_field, uint write_macros, uint sample_sgs);

	gpu::Launch damping_init_, initialize_, stream_collide_, update_fields_, outlet_, free_slip_;
	ulong cells_ = 0u;
	bool temperature_ = true, subgrid_ = false, surface_ibm_ = false, has_damping_tables_ = false;
	uint init_force_flag_slot_ = 0u;
	uint stream_write_flag_slot_ = 0u;
	uint stream_force_flag_slot_ = 0u;
	uint update_force_flag_slot_ = 0u;
	uint write_slot_ = 0u;   // update_fields: 1 writes rho/u/T
	uint sample_slot_ = 0u;  // update_fields: 1 accumulates the SGS sums
	uint support_slot_ = 0u; // update_fields: support list, its length, list mode
};

} // namespace culbm::solver
