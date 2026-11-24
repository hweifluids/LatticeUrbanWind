#include "case_validator.hpp"
#include "field_set.hpp"

#include <cmath>
#include <thread>

#if defined(SRT)==defined(TRT)
#error "core/build_features.hpp must select exactly one collision operator (SRT or TRT)."
#endif

namespace culbm::solver {
namespace {

string dims(const uint x, const uint y, const uint z, const char* separator) {
	return to_string(x)+separator+to_string(y)+separator+to_string(z);
}

// The largest lattice of the same shape that `available` MiB would hold when
// `required` MiB hold nx x ny x nz.
string scaled_lattice(const Decomposition& grid, const uint available, const uint required) {
	const float scale = cbrt((float)available/(float)required);
	return "("+dims((uint)(scale*(float)grid.size(0u)), (uint)(scale*(float)grid.size(1u)),
		(uint)(scale*(float)grid.size(2u)), ", ")+")";
}

} // namespace

void CaseValidator::check_decomposition(const uint nx, const uint ny, const uint nz,
	const uint dx, const uint dy, const uint dz) {
	if(dx==0u || dy==0u || dz==0u)
		fatal_error("The lattice must be split into at least one subdomain per axis, not "+dims(dx, dy, dz, "x")+".");
	if(nx<dx || ny<dy || nz<dz)
		fatal_error("The lattice "+dims(nx, ny, nz, "x")+" has fewer cells than subdomains ("+dims(dx, dy, dz, "x")+").");
}

void CaseValidator::check_construction(const Decomposition& grid, const std::vector<gpu::DeviceProperties>& gpus,
	const CaseCoefficients& c) {
	if(surface_ibm::get_config().enabled) {
#ifndef UPDATE_FIELDS
		fatal_error("SurfaceIBM needs UPDATE_FIELDS: every forcing iteration rebuilds the physical macro velocity.");
#endif
	}
	if(grid.cells()==0u) fatal_error("The lattice "+dims(grid.size(0u), grid.size(1u), grid.size(2u), "x")+" has no cells.");

	// Every subdomain needs its share of the smallest GPU.
	uint available = max_uint;
	for(const gpu::DeviceProperties& gpu : gpus) available = min(available, gpu.memory_mb);
	const uint required = (uint)min((std::uint64_t)max_uint,
		lattice_vram_mib(grid.size(0u), grid.size(1u), grid.size(2u), grid.split(0u), grid.split(1u), grid.split(2u)));
	if(required>available) {
		const string count = to_string(grid.count());
		string message = "The lattice "+dims(grid.size(0u), grid.size(1u), grid.size(2u), " x ")+" needs "+count+" x "+
			to_string(required)+" MiB of GPU memory, but only "+count+" x "+to_string(available)+" MiB are installed. "
			"The largest lattice of this shape that fits is "+scaled_lattice(grid, available, required)+".";
		if(get_ddf_bytes()==4u) {
			// FP32 DDFs: the same lattice with 16-bit DDF storage.
			const ulong per_cell = (ulong)(cell_cost(lattice_storage()).device-(uint)get_momentum_lattice()*2u);
			const uint compressed = (uint)(grid.cells()/(ulong)grid.count()*per_cell/1048576ull);
			message += " With culbm_ddf_storage = FP16C or FP16S it would be "+scaled_lattice(grid, available, compressed)+".";
		}
		fatal_error(message);
	}

	if(c.nu==0.0f) fatal_error("The kinematic viscosity is zero.");
	if(c.nu<0.0f) fatal_error("The kinematic viscosity is negative.");
	if(c.sigma!=0.0f) fatal_error("A surface tension was given, but free surfaces are not supported.");
	if(c.alpha<0.0f) fatal_error("The thermal diffusion coefficient alpha is negative.");
	if(c.alpha==0.0f)
		log_warning("Thermal diffusion coefficient alpha is zero, so tau_T is at the marginal value 0.5.");
	if(c.alpha==0.0f && c.beta==0.0f)
		log_warning("Thermal diffusion and expansion coefficients are both zero; the temperature field is passive.");
	if(c.particles>0u)
		fatal_error(to_string(c.particles)+" lattice particles were requested, but they are not supported "
			"(use the particle-advection runtime).");
}

void CaseValidator::check_immersed_boundary(const bool storage, const bool attached) {
	if(storage && !attached)
		fatal_error("SurfaceIBM is enabled, but no marker set was attached before the first initialization.");
}

void CaseValidator::check_cell_types(FieldView<uchar>& flags, FieldView<float>& velocity) {
	// Per-thread findings of a parallel scan over the host flags.
	struct Findings {
		uchar types = 0u;        // union of all flag bits
		bool moving = false;     // solid cells with a wall velocity
		bool equilibrium = false; // CELL_EQUILIBRIUM cells
	};
	const uint threads = std::max(1u, std::thread::hardware_concurrency());
	std::vector<Findings> found(threads);
	parallel_for(flags.length(), threads, [&](const ulong n, const uint t) {
		Findings& f = found[t];
		const uchar type = flags[n];
		const uchar boundary = type&(CELL_SOLID|CELL_EQUILIBRIUM);
		f.types |= type;
		if(boundary==(CELL_SOLID|CELL_EQUILIBRIUM)) f.moving = true;
		else if(boundary==CELL_SOLID && (velocity.cell(n, 0u)!=0.0f || velocity.cell(n, 1u)!=0.0f || velocity.cell(n, 2u)!=0.0f)) f.moving = true;
		if(boundary==CELL_EQUILIBRIUM) f.equilibrium = true;
	});
	Findings all;
	for(const Findings& f : found) {
		all.types |= f.types;
		all.moving = all.moving || f.moving;
		all.equilibrium = all.equilibrium || f.equilibrium;
	}
	if(all.moving)
		log_warning("Some solid cells carry a wall velocity, but moving boundaries are not supported; they act as resting walls.");
	if(!all.equilibrium)
		log_warning("No equilibrium boundary cells (CELL_EQUILIBRIUM) are set; the lattice has no inflow or outflow.");
	if(all.types&(CELL_FLUID|CELL_INTERFACE|CELL_GAS))
		fatal_error("Some cells carry the free-surface types CELL_FLUID/CELL_INTERFACE/CELL_GAS, but free surfaces are not supported.");
}

} // namespace culbm::solver
