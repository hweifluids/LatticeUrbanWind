#include "lattice_config.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

Units units; // global solver scale (SI <-> lattice)

namespace {

Temperature_Lattice temperature_lattice = Temperature_Lattice::Q7;
bool thermal_lattice_enabled = true;
Momentum_Lattice momentum_lattice = Momentum_Lattice::Q19;
#ifdef CULBM_DISABLE_FP16C
DDF_Storage ddf_storage = DDF_Storage::FP32;
#else
DDF_Storage ddf_storage = DDF_Storage::FP16C;
#endif
Subgrid_Configuration subgrid_configuration;
Wall_Model_Configuration wall_model_configuration;
float wall_model_level_ratio = 1.0f;
bool wall_model_level_has_ground = true;
std::atomic_uint live_solvers{0u};

bool selections_locked() {
	return live_solvers.load(std::memory_order_acquire)>0u;
}

} // namespace

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
	if(selections_locked()) return false;
	subgrid_configuration = configuration;
	return true;
}

Subgrid_Configuration get_subgrid_configuration() {
	return subgrid_configuration;
}

bool set_wall_model_configuration(
	const Wall_Model_Configuration& configuration) {
	if(!std::isfinite(configuration.kappa) || configuration.kappa<=0.0f ||
	   configuration.kappa>1.0f ||
	   !std::isfinite(configuration.smooth_wall_B) ||
	   !std::isfinite(configuration.z0_ground_m) ||
	   configuration.z0_ground_m<0.0f ||
	   !std::isfinite(configuration.z0_building_m) ||
	   configuration.z0_building_m<0.0f ||
	   !std::isfinite(configuration.molecular_nu_m2_s) ||
	   configuration.molecular_nu_m2_s<=0.0f ||
	   configuration.sgs_layers>64u) return false;
	const bool changes =
		configuration.enabled!=wall_model_configuration.enabled ||
		configuration.kappa!=wall_model_configuration.kappa ||
		configuration.smooth_wall_B!=wall_model_configuration.smooth_wall_B ||
		configuration.z0_ground_m!=wall_model_configuration.z0_ground_m ||
		configuration.z0_building_m!=wall_model_configuration.z0_building_m ||
		configuration.molecular_nu_m2_s!=
			wall_model_configuration.molecular_nu_m2_s ||
		configuration.sgs_layers!=wall_model_configuration.sgs_layers ||
		configuration.surface_ibm!=wall_model_configuration.surface_ibm;
	if(!changes) return true;
	if(selections_locked()) return false;
	wall_model_configuration = configuration;
	return true;
}

Wall_Model_Configuration get_wall_model_configuration() {
	return wall_model_configuration;
}

Wall_Model_Lattice resolve_wall_model_lattice(
	const Wall_Model_Configuration& configuration,
	const float dx_m, const float dt_s, const bool ground_only) {
	Wall_Model_Lattice lattice;
	lattice.enabled = configuration.enabled;
	lattice.ground_only = ground_only;
	lattice.kappa = configuration.kappa;
	lattice.cell_size_m = dx_m;
	lattice.time_step_s = dt_s;
	if(!configuration.enabled) return lattice;
	if(!(std::isfinite(dx_m) && dx_m>0.0f && std::isfinite(dt_s) && dt_s>0.0f)) {
		throw std::invalid_argument(
			"wall model requires a positive finite lattice cell size and time step");
	}
	// Halfway bounce-back places the wall half a cell below the first node.
	const double d = Wall_Model_Lattice::first_cell_distance_lu*(double)dx_m;
	const double kappa = (double)configuration.kappa;
	const auto drag = [&](const double z0) {
		const double l = std::log1p(d/z0);
		return (float)((kappa/l)*(kappa/l));
	};
	lattice.smooth_h = !(configuration.z0_ground_m>0.0f);
	lattice.smooth_v = !(configuration.z0_building_m>0.0f);
	lattice.cd_h = lattice.smooth_h ? 0.0f : drag((double)configuration.z0_ground_m);
	lattice.cd_v = lattice.smooth_v ? 0.0f : drag((double)configuration.z0_building_m);
	lattice.ibm = ground_only && configuration.surface_ibm;
	lattice.z0_h_lu = lattice.smooth_h ? 0.0f : (float)((double)configuration.z0_ground_m/(double)dx_m);
	lattice.z0_v_lu = lattice.smooth_v ? 0.0f : (float)((double)configuration.z0_building_m/(double)dx_m);
	lattice.nu_lbm = (float)((double)configuration.molecular_nu_m2_s*
		(double)dt_s/((double)dx_m*(double)dx_m));
	lattice.log_E = (float)(kappa*(double)configuration.smooth_wall_B);
	// y+ where u+=y+ meets u+=ln(E*y+)/kappa; Re_d=u*d/nu equals y+^2 there.
	double y = 11.0;
	for(int iteration=0; iteration<200; ++iteration)
		y = ((double)lattice.log_E+std::log(y))/kappa;
	lattice.re_laminar = (float)(y*y);
	const Subgrid_Configuration subgrid = get_subgrid_configuration();
	lattice.sgs = configuration.sgs_layers>0u && subgrid.enabled &&
		subgrid.fixed_eddy_viscosity<0.0f;
	lattice.sgs_layers = lattice.sgs ? (float)configuration.sgs_layers : 0.0f;
	if(!(std::isfinite(lattice.cd_h) && std::isfinite(lattice.cd_v) &&
	     std::isfinite(lattice.nu_lbm) && lattice.nu_lbm>0.0f &&
	     std::isfinite(lattice.re_laminar) && lattice.re_laminar>1.0f)) {
		throw std::invalid_argument("wall model lattice constants are not finite");
	}
	return lattice;
}

Wall_Model_Level_Scope::Wall_Model_Level_Scope(const float ratio_to_coarse, const bool has_ground_wall) {
	if(!(std::isfinite(ratio_to_coarse) && ratio_to_coarse>0.0f)) {
		throw std::invalid_argument("wall model level ratio must be positive");
	}
	previous_ratio = wall_model_level_ratio;
	previous_has_ground_wall = wall_model_level_has_ground;
	wall_model_level_ratio = ratio_to_coarse;
	wall_model_level_has_ground = has_ground_wall;
}

Wall_Model_Level_Scope::~Wall_Model_Level_Scope() {
	wall_model_level_ratio = previous_ratio;
	wall_model_level_has_ground = previous_has_ground_wall;
}

float get_wall_model_level_ratio() {
	return wall_model_level_ratio;
}

bool get_wall_model_level_has_ground() {
	return wall_model_level_has_ground;
}

bool set_temperature_lattice(const Temperature_Lattice lattice) {
	if(lattice!=Temperature_Lattice::Q7 &&
	   lattice!=Temperature_Lattice::Q19) return false;
	if(selections_locked() && lattice!=temperature_lattice) return false;
	temperature_lattice = lattice;
	return true;
}
Temperature_Lattice get_temperature_lattice() {
	return temperature_lattice;
}
bool set_thermal_lattice_enabled(const bool enabled) {
	if(selections_locked() && enabled!=thermal_lattice_enabled) return false;
	thermal_lattice_enabled = enabled;
	return true;
}
bool get_thermal_lattice_enabled() {
	return thermal_lattice_enabled;
}
Thermal_Lattice_Scope::Thermal_Lattice_Scope(const bool enabled) {
	previous_enabled = thermal_lattice_enabled;
	thermal_lattice_enabled = enabled;
}
Thermal_Lattice_Scope::~Thermal_Lattice_Scope() {
	thermal_lattice_enabled = previous_enabled;
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
bool set_momentum_lattice(const Momentum_Lattice lattice) {
	if(lattice!=Momentum_Lattice::Q19 && lattice!=Momentum_Lattice::Q27) return false;
	if(selections_locked() && lattice!=momentum_lattice) return false;
	momentum_lattice = lattice;
	return true;
}
Momentum_Lattice get_momentum_lattice() {
	return momentum_lattice;
}
const char* get_momentum_lattice_name() {
	return momentum_lattice==Momentum_Lattice::Q27 ? "D3Q27" : "D3Q19";
}
bool set_ddf_storage(const DDF_Storage storage) {
	if(storage!=DDF_Storage::FP16C && storage!=DDF_Storage::FP16S && storage!=DDF_Storage::FP32) return false;
	if(selections_locked() && storage!=ddf_storage) return false;
	ddf_storage = storage;
	return true;
}
DDF_Storage get_ddf_storage() {
	return ddf_storage;
}
const char* get_ddf_storage_name() {
	return ddf_storage==DDF_Storage::FP16S ? "FP16S" : ddf_storage==DDF_Storage::FP16C ? "FP16C" : "FP32";
}
uint get_ddf_bytes() {
	return ddf_storage==DDF_Storage::FP32 ? 4u : 2u;
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

namespace culbm::solver {

FeatureSet FeatureSet::current() {
	FeatureSet f;
	f.velocity_set = static_cast<uint>(momentum_lattice);
	f.face_ddfs = momentum_lattice==Momentum_Lattice::Q27 ? 9u : 5u;
	f.storage = ddf_storage;
	f.ddf_bytes = get_ddf_bytes();
	f.temperature_enabled = get_thermal_lattice_enabled();
	f.temperature = temperature_lattice;
	f.temperature_set = get_temperature_set();
	f.temperature_face_ddfs = get_temperature_transfers();
	f.temperature_cs2 = get_temperature_lattice_cs2();
	f.subgrid = subgrid_configuration;
	f.surface_ibm = surface_ibm::get_config();
	// The wall model scales with the nesting level the solver is built for.
	const float level = get_wall_model_level_ratio();
	f.wall_model = resolve_wall_model_lattice(wall_model_configuration,
		units.si_x(1.0f)/level, units.si_t(1ull)/level, f.surface_ibm.enabled);
	if(!get_wall_model_level_has_ground()) {
		f.wall_model.sgs = false;
		f.wall_model.sgs_layers = 0.0f;
	}
	f.wall_model.ibm_retraction_lu = f.surface_ibm.retraction_lu;
	return f;
}

SelectionLock::SelectionLock() {
	live_solvers.fetch_add(1u, std::memory_order_acq_rel);
}

SelectionLock::~SelectionLock() {
	live_solvers.fetch_sub(1u, std::memory_order_acq_rel);
}

bool SelectionLock::held() {
	return selections_locked();
}

} // namespace culbm::solver
