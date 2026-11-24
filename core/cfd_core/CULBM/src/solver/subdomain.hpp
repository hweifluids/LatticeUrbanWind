#pragma once

// One subdomain of a solver on one GPU: its placement in the global lattice,
// the device with the NVRTC module specialised for it, its fields, kernels and
// halo port, its time step, and the optional samplers (device statistics,
// velocity probes) and the device part of the voxelization.

#include "decomposition.hpp"
#include "field_set.hpp"
#include "halo_exchanger.hpp"
#include "kernel_set.hpp"
#include "lattice_config.hpp"
#include "../gpu/gpu.hpp"

#include <atomic>
#include <vector>

namespace culbm::solver {

// Physics shared by all subdomains of a solver; the solver owns it and the
// subdomains read it whenever they launch.
struct Physics {
	Transport transport;
	BodyForces forces;
	int temperature_outlet_face = 0; // Temperature_Outlet_Face value; 0 = off
};

// Running mean and second moment of rho, u (and T) on the device.
struct DeviceStatistics {
	gpu::MirroredBuffer<float> mean_u, mean_rho, m2_u, mean_T;
	gpu::Launch accumulate;
	uint inverse_slot = 7u;
	bool active = false;
	bool temperature = false;
};

// Velocities of a sparse set of cells gathered on the device.
struct ProbeSampler {
	gpu::MirroredBuffer<ulong> cells;    // local cell indices
	gpu::MirroredBuffer<float> velocity; // three components per probe
	std::vector<ulong> slots;            // output slot of each probe
	gpu::Launch gather;
};

class Subdomain {
public:
	Subdomain(const Decomposition& grid, unsigned id, const gpu::DeviceProperties& gpu,
		const FeatureSet& features, const Physics& physics);
	Subdomain(const Subdomain&) = delete;
	Subdomain& operator=(const Subdomain&) = delete;

	const Placement& placement() const { return placement_; }
	uint extent(const uint axis) const { return placement_.extent[axis]; }
	int origin(const uint axis) const { return placement_.origin[axis]; }
	index_t cells() const { return placement_.cells(); }
	gpu::Device& device() { return device_; }
	const gpu::Device& device() const { return device_; }
	FieldSet& fields() { return fields_; }
	const FieldSet& fields() const { return fields_; }
	KernelSet& kernels() { return kernels_; }
	HaloPort& halo() { return halo_; }
	const Wall_Model_Lattice& wall_model() const { return features_.wall_model; }
	bool has_ibm_storage() const { return features_.surface_ibm.enabled; }

	// The console thread reads the step while the solver advances it.
	ulong time() const { return t_.load(std::memory_order_acquire); }
	void advance_clock(uint steps=1u);
	void rewind_clock();

	// Work queued on the device's compute stream.
	void initialize_fields();
	void set_step_writes_macros(const bool write) { step_writes_macros_ = write; }
	void stream_collide();
	void refresh_macros(); // rho, u, T from the DDFs; skipped when already current
	void temperature_outlet();
	void free_slip(uint face_mask);
	void ibm_base_predictor(gpu::MirroredBuffer<ulong>& support, ulong count);
	void start_subgrid_sample();
	Subgrid_Statistics finish_subgrid_sample(); // waits for the device
	void synchronize() { device_.synchronize(); }

	bool enable_statistics(bool include_temperature, uint later_mib);
	void disable_statistics();
	void reset_statistics();
	void sample_statistics(float inverse_samples);
	DeviceStatistics& statistics() { return statistics_; }

	bool configure_probes(const std::vector<ulong>& local_cells, const std::vector<ulong>& slots, uint later_mib);
	void disable_probes();
	bool has_probes() const { return !probes_.slots.empty(); }
	void sample_probes();
	ProbeSampler& probes() { return probes_; }

	// One voxelization pass of `mesh` (or of the triangles `subset`) along
	// `axis` that marks cells with `flag`.
	void voxelize(const Mesh& mesh, uint axis, uchar flag, const std::vector<uint>* subset,
		const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity);

private:
	Placement placement_;
	FeatureSet features_;
	const Physics& physics_;
	gpu::Device device_;
	FieldSet fields_;
	KernelSet kernels_;
	HaloPort halo_;
	std::atomic<ulong> t_{0u};
	ulong macros_time_ = ~0ull; // step at which rho/u/T were last recomputed
	bool step_writes_macros_ = true;
	bool last_step_wrote_macros_ = true;
	DeviceStatistics statistics_;
	ProbeSampler probes_;
};

} // namespace culbm::solver
