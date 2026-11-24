#pragma once

// CULBM solver: the global lattice, its decomposition into subdomains (one per
// GPU, or one per MPI rank), the frozen feature set, the halo exchanger, the
// per-step pipeline, initialization and time stepping, and the host views of
// the cell fields.

#include "case_validator.hpp"
#include "decomposition.hpp"
#include "field_view.hpp"
#include "halo_exchanger.hpp"
#include "immersed_forcing.hpp"
#include "lattice_config.hpp"
#include "step_pipeline.hpp"
#include "subdomain.hpp"

#include <memory>
#include <vector>

namespace culbm::solver {

class Solver {
public:
	// The lattice is shrunk to a multiple of the subdomain counts.
	Solver(uint nx, uint ny, uint nz, uint dx, uint dy, uint dz, const CaseCoefficients& coefficients);
	~Solver();
	Solver(const Solver&) = delete;
	Solver& operator=(const Solver&) = delete;

	const Decomposition& grid() const { return grid_; }
	const FeatureSet& features() const { return features_; }
	bool distributed() const { return distributed_; }
	uint owned_subdomain() const { return owned_; } // MPI: this rank's subdomain
	uint local_count() const { return (uint)locals_.size(); }
	Subdomain& local(const uint d) { return *locals_[d]; }
	const Subdomain& local(const uint d) const { return *locals_[d]; }
	bool owns(const uint id) const { return !distributed_ || id==owned_; }
	ulong time() const { return locals_.front()->time(); }

	Physics& physics() { return physics_; }
	const Physics& physics() const { return physics_; }

	FieldView<float>& density() { return density_; }
	FieldView<float>& velocity() { return velocity_; }
	FieldView<uchar>& flags() { return flags_; }
	FieldView<float, HostCopy::on_demand>& force() { return force_; }
	FieldView<float>& temperature() { return temperature_; }

	// SurfaceIBM: the forcing stage runs before every collision once attached.
	bool has_ibm_storage() const { return features_.surface_ibm.enabled; }
	void attach_immersed_forcing(std::unique_ptr<ImmersedForcing> forcing);
	void detach_immersed_forcing();
	bool has_immersed_forcing() const { return forcing_!=nullptr; }
	ImmersedForcing* immersed_forcing() const { return forcing_.get(); }

	// Body forces compiled into every launch: the Coriolis rotation and the
	// Boussinesq gravity, both in lattice units.
	void set_rotation(float omega_x, float omega_y, float omega_z);
	void set_gravity(float gx, float gy, float gz);
	// Scalar outlet face (Temperature_Outlet_Face). False for an invalid face,
	// after the first initialization (its CELL_THERMAL cells would stay on the old
	// face) and when a subdomain has fewer than two cells along the normal.
	bool set_temperature_outlet_face(int face);
	bool set_temperature_outlet_face(const Temperature_Outlet_Face face) { return set_temperature_outlet_face((int)face); }

	// Lifecycle. initialize() uploads the host fields and builds the
	// equilibrium DDFs; reset() makes the next run initialize again.
	bool initialized() const { return initialized_; }
	bool ever_initialized() const { return ever_initialized_; }
	void initialize();
	void initialize_once() { if(!initialized_) initialize(); }
	void reset() { initialized_ = false; }

	void step(); // one time step, see StepPipeline
	// Controls whether stream_collide writes macroscopic rho/u/T for interior
	// cells on subsequent steps (damping reference planes still stay updated).
	void set_step_writes_macros(bool write);
	// One time step of a step-wise driver (nested scheduler, tests); the first
	// call initializes.
	void advance() { initialize_once(); step(); }
	// `steps` time steps; the devices are finished when it returns.
	void run(ulong steps);
	const StepPipeline& pipeline() const { return pipeline_; }

	void synchronize();
	void pace();
	// Distributed runs: every rank waits for the others.
	void barrier() const;
	void refresh_macros();      // rho, u, T current on the devices
	void publish_flags();       // device flags into all halo layers
	void free_slip(uint face_mask);
	void prepare_advection();   // velocity and flags with halos for particle kernels
	Subgrid_Statistics subgrid_statistics();
	HaloExchanger& halos() { return halos_; }

	// Device statistics (running means and second moments).
	bool enable_statistics(bool include_temperature, uint later_mib);
	void disable_statistics();
	void reset_statistics();
	void sample_statistics(ulong samples);
	bool statistics_active() const { return !locals_.empty() && locals_.front()->statistics().active; }
	void read_statistics(vector<float>& mean_u, vector<float>& mean_rho, vector<float>& mean_T,
		vector<float>& m2_u, vector<float>& m2_v, vector<float>& m2_w);

	// Velocity probes gathered on the devices (not in MPI mode).
	bool configure_probes(const vector<ulong>& global_cells, uint later_mib);
	void disable_probes();
	bool probes_active() const;
	void sample_probes(vector<float>& samples);

	// Marks the cells inside `mesh` (lattice coordinates) with `flag`; a
	// moving body also sets the wall velocity of its cells.
	void voxelize(const Mesh& mesh, uchar flag=CELL_SOLID, const float3& rotation_center=float3(0.0f),
		const float3& linear_velocity=float3(0.0f), const float3& rotational_velocity=float3(0.0f));

private:
	void build_wall_columns();

	Decomposition grid_;
	FeatureSet features_;
	Physics physics_;
	SelectionLock lock_;
	bool distributed_ = false;
	uint owned_ = 0u;
	std::vector<std::unique_ptr<Subdomain>> locals_;
	HaloExchanger halos_;
	StepPipeline pipeline_;
	std::unique_ptr<ImmersedForcing> forcing_;
	bool initialized_ = false;
	bool ever_initialized_ = false;

	FieldView<float> density_;
	FieldView<float> velocity_;
	FieldView<uchar> flags_;
	FieldView<float, HostCopy::on_demand> force_;
	FieldView<float> temperature_;
};

} // namespace culbm::solver
