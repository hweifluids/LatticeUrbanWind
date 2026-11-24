#pragma once

// Runtime lattice selections of the CULBM solver (deck options) and the
// constants derived from them. They are process-wide settings: a solver
// freezes them at construction, and they cannot change while one exists.
//
//   momentum lattice    D3Q19 / D3Q27        (culbm_velocity_set)
//   DDF storage         FP16C / FP16S / FP32 (culbm_ddf_storage)
//   temperature lattice D3Q7 / D3Q19
//   subgrid model       Smagorinsky-Lilly configuration
//   wall model          equilibrium wall-stress law, per nesting level

#include "../core/build_features.hpp"
#include "../core/core.hpp"
#include "../core/lattice_units.hpp"
#include "../surface_ibm_config.hpp"

#include <atomic>
#include <cstdint>

enum class Temperature_Lattice : uint {
	Q7 = 7u,
	Q19 = 19u
};

// Runtime scalar outlet orientation. Values deliberately match the
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
bool set_thermal_lattice_enabled(bool enabled);
bool get_thermal_lattice_enabled();
// Scoped override for whether newly constructed solvers allocate and step the
// thermal lattice (default true for direct solver callers and unit tests).
class Thermal_Lattice_Scope {
public:
	explicit Thermal_Lattice_Scope(bool enabled);
	~Thermal_Lattice_Scope();
	Thermal_Lattice_Scope(const Thermal_Lattice_Scope&) = delete;
	Thermal_Lattice_Scope& operator=(const Thermal_Lattice_Scope&) = delete;
private:
	bool previous_enabled = true;
};
uint get_temperature_set();
uint get_temperature_transfers();
float get_temperature_lattice_cs2();
float get_temperature_tau(const float alpha);
const char* get_temperature_lattice_name();

// Momentum velocity set and DDF storage format, selected from the deck before
// the first solver is created (culbm_velocity_set, culbm_ddf_storage).
enum class Momentum_Lattice : uint {
	Q19 = 19u,
	Q27 = 27u
};
enum class DDF_Storage : uint {
	FP16C = 0u, // custom 16-bit format, software conversion
	FP16S = 1u, // range-shifted IEEE half, hardware conversion
	FP32 = 2u
};
bool set_momentum_lattice(Momentum_Lattice lattice);
Momentum_Lattice get_momentum_lattice();
const char* get_momentum_lattice_name();
bool set_ddf_storage(DDF_Storage storage);
DDF_Storage get_ddf_storage();
const char* get_ddf_storage_name();
uint get_ddf_bytes(); // bytes per stored DDF

struct Subgrid_Configuration {
#ifdef SUBGRID
	bool enabled = true;
#else
	bool enabled = false;
#endif
	float smagorinsky_constant = 0.173265956f;
	bool scalar_coupling = true;
	float turbulent_prandtl = 1.0f;
	// -1 selects the local stress-derived Smagorinsky value. A non-negative
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

// Equilibrium wall-stress model for link-wise CELL_SOLID bounce-back walls. It is
// frozen into the NVRTC source of every subdomain at construction, so it must
// be set before constructing a solver. Roughness lengths are SI metres; a zero
// roughness selects the smooth-wall log law with the molecular viscosity. The
// library default is off so drivers that construct solvers directly (tests,
// validation) keep plain no-slip walls; the production deck defaults it on.
struct Wall_Model_Configuration {
	bool enabled = false;
	float kappa = 0.41f;
	float smooth_wall_B = 5.2f; // smooth log-law intercept, E = exp(kappa*B)
	float z0_ground_m = 0.03f; // upward-facing horizontal faces (ground, roofs)
	float z0_building_m = 0.01f; // vertical and downward-facing faces
	float molecular_nu_m2_s = 1.48E-5f; // smooth-wall law only
	// Lattice rows above upward-facing walls in which the Smagorinsky length
	// is raised to kappa*z_w (near-wall SGS layer); 0 disables the layer.
	uint sgs_layers = 3u;
	// With SurfaceIBM: also wall-model the immersed surfaces (one-sided
	// log-law stress, voxel core kept with free-slip faces); false keeps them
	// no-slip and wall-models only the voxel walls (ground slab, terrain).
	bool surface_ibm = true;
};

// Lattice constants of one solver; derived from the configuration, the global
// Units scale and the active nesting level scope.
struct Wall_Model_Lattice {
	// The retained voxel represents a full lattice control volume. Its velocity
	// can match a wall law no closer than the first halfway-wall sample.
	static constexpr float first_cell_distance_lu = 0.5f;
	bool enabled = false;
	bool ground_only = false; // SurfaceIBM solver (immersed surfaces handled by the runtime)
	bool smooth_h = false, smooth_v = false;
	float cd_h = 0.0f, cd_v = 0.0f; // (kappa/ln(1+d/z0))^2, d = dx/2
	float nu_lbm = 0.0f; // molecular viscosity in lattice units
	float kappa = 0.41f;
	float log_E = 0.0f;
	float re_laminar = 0.0f; // (y+ at the laminar/log intersection)^2
	float cell_size_m = 1.0f;
	float time_step_s = 1.0f;
	bool sgs = false; // near-wall SGS layer active (requires the SGS model)
	float sgs_layers = 0.0f;
	bool ibm = false; // SurfaceIBM surfaces wall-modelled (ground_only solvers)
	float z0_h_lu = 0.0f, z0_v_lu = 0.0f; // roughness lengths in lattice units
	float ibm_retraction_lu = 0.5f; // marker retraction behind the input surface
};

bool set_wall_model_configuration(const Wall_Model_Configuration& configuration);
Wall_Model_Configuration get_wall_model_configuration();
// Resolve lattice constants for a lattice with cell size dx_m and step dt_s.
Wall_Model_Lattice resolve_wall_model_lattice(const Wall_Model_Configuration& configuration,
	const float dx_m, const float dt_s, const bool ground_only);

// Nested CUDA levels refine dx and dt by ratio_to_coarse relative to the global
// Units scale; solvers constructed inside this scope use the level scale.
// Only the global ground slab owns a physical bottom wall; upper slabs set
// has_ground_wall=false so their internal lower nesting ghost is not mistaken
// for a solid ground base by the near-wall SGS column map.
class Wall_Model_Level_Scope {
public:
	explicit Wall_Model_Level_Scope(const float ratio_to_coarse, const bool has_ground_wall=true);
	~Wall_Model_Level_Scope();
	Wall_Model_Level_Scope(const Wall_Model_Level_Scope&) = delete;
	Wall_Model_Level_Scope& operator=(const Wall_Model_Level_Scope&) = delete;
private:
	float previous_ratio = 1.0f;
	bool previous_has_ground_wall = true;
};
float get_wall_model_level_ratio();
bool get_wall_model_level_has_ground();

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

namespace culbm::solver {

// Snapshot of every runtime selection a solver compiles into its kernels and
// sizes its fields with.
struct FeatureSet {
	uint velocity_set = 19u;     // momentum DDFs per cell
	uint face_ddfs = 5u;         // momentum DDFs crossing one subdomain face
	DDF_Storage storage = DDF_Storage::FP16C;
	uint ddf_bytes = 2u;         // bytes per stored DDF
	bool temperature_enabled = true;
	Temperature_Lattice temperature = Temperature_Lattice::Q7;
	uint temperature_set = 7u;   // thermal DDFs per cell
	uint temperature_face_ddfs = 1u;
	float temperature_cs2 = 0.25f;
	Subgrid_Configuration subgrid;
	surface_ibm::Config surface_ibm;
	Wall_Model_Lattice wall_model;

	// The current process-wide selections, with the wall model resolved for
	// the active nesting level and the global Units scale.
	static FeatureSet current();
};

// While a solver exists the lattice selections above are locked.
class SelectionLock {
public:
	SelectionLock();
	~SelectionLock();
	SelectionLock(const SelectionLock&) = delete;
	SelectionLock& operator=(const SelectionLock&) = delete;
	static bool held();
};

} // namespace culbm::solver
