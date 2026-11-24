// VRAM accounting against real allocations: the cell field table that plans
// device memory (culbm::solver::domain_footprint) must describe the buffers a
// subdomain actually allocates. For several lattice configurations an LBM is
// constructed and every domain's allocation counters are compared with the
// plan; both the exact bytes and the booked MiB must agree within 1%.
#include "solver.hpp"
#include "surface_ibm.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

float coriolis_Omegax_lbmu = 0.0f;
float coriolis_Omegay_lbmu = 0.0f;
float coriolis_Omegaz_lbmu = 0.0f;
bool buffer_nudging_active = false;
int buffer_n_cells = 1;
float buffer_inv_tau_lbmu = 0.0f;
int buffer_nudge_vertical = 0;
int buffer_downstream_face_id = 0;
bool top_sponge_active = false;
int sponge_n_cells = 1;
float sponge_inv_tau_lbmu = 0.0f;
int sponge_ref_mode = 0;

namespace {

constexpr double tolerance = 0.01;

struct Case {
	const char* name;
	Momentum_Lattice lattice;
	DDF_Storage storage;
	Temperature_Lattice temperature;
	bool surface_ibm;
	uint nx, ny, nz, dx;
	// Expected per-cell costs of the field table: device bytes as in the
	// hand-written accounting it replaced; the host counts rho, u, flags and T
	// (the F mirror is created on first host use).
	unsigned device_bytes, host_bytes, traffic_bytes;
};

int failures = 0;

void expect(const bool condition, const std::string& message) {
	if(condition) return;
	++failures;
	std::cerr << "FAIL: " << message << '\n';
}

double relative(const double actual, const double planned) {
	return actual>0.0 ? std::fabs(actual-planned)/actual : (planned==0.0 ? 0.0 : 1.0);
}

void configure_surface_ibm(const bool enabled) {
	culbm::surface_ibm::Config config;
	config.enabled = enabled;
	if(enabled) {
		config.marker_spacing_lu = 1.0f;
		config.forcing_iterations = 4u;
		config.relaxation = 1.0f;
		config.residual_tolerance = 1.0e-3f;
		config.support_thickness_lu = 0.75f;
		config.max_markers = 1000u;
	}
	std::string error;
	if(!culbm::surface_ibm::validate_config(config, error))
		throw std::runtime_error("invalid SurfaceIBM test config: "+error);
	culbm::surface_ibm::set_config(config);
}

void run(const Case& c) {
	if(!set_momentum_lattice(c.lattice) || !set_ddf_storage(c.storage) ||
	   !set_temperature_lattice(c.temperature))
		throw std::runtime_error(string(c.name)+": lattice selection was rejected");
	configure_surface_ibm(c.surface_ibm);

	const culbm::solver::LatticeStorage storage = lattice_storage();
	const culbm::solver::CellCost cost = culbm::solver::cell_cost(storage);
	expect(cost.device==c.device_bytes && cost.host==c.host_bytes && cost.traffic==c.traffic_bytes,
		string(c.name)+": per-cell cost "+std::to_string(cost.device)+"/"+std::to_string(cost.host)+"/"+
		std::to_string(cost.traffic)+" B, expected "+std::to_string(c.device_bytes)+"/"+
		std::to_string(c.host_bytes)+"/"+std::to_string(c.traffic_bytes)+" B (device/host/step)");
	const culbm::solver::DomainFootprint plan = culbm::solver::domain_footprint(storage, c.nx, c.ny, c.nz, c.dx, 1u, 1u);
	expect(plan.valid && plan.mib==lattice_vram_mib(c.nx, c.ny, c.nz, c.dx, 1u, 1u),
		string(c.name)+": lattice_vram_mib differs from the storage table");

	culbm::solver::Solver lbm(c.nx, c.ny, c.nz, c.dx, 1u, 1u,
	    culbm::solver::CaseCoefficients().viscosity(0.05f).thermal(0.01f, 0.0f));
	for(uint d=0u; d<lbm.local_count(); ++d) {
		const gpu::Device& device = lbm.local(d).device();
		const std::uint64_t bytes = device.allocated_bytes();
		const std::uint64_t mib = device.booked_mib();
		const double byte_error = relative((double)bytes, (double)plan.bytes);
		const double mib_error = relative((double)mib, (double)plan.mib);
		std::cout << std::left << std::setw(26) << c.name << " domain " << d << " (GPU " << device.ordinal()
			<< "): allocated " << bytes << " B / " << mib << " MiB, planned " << plan.bytes << " B / "
			<< plan.mib << " MiB, error " << std::setprecision(3) << 100.0*byte_error << "% / "
			<< 100.0*mib_error << "%\n";
		expect(byte_error<=tolerance, string(c.name)+": allocated bytes differ from the plan by more than 1%");
		expect(mib_error<=tolerance, string(c.name)+": booked MiB differ from the plan by more than 1%");
		// The plan never misses a cell field: what is left over are the
		// constant-size reduction buffers of the subgrid statistics.
		expect(bytes>=plan.bytes && bytes-plan.bytes<=4096u,
			string(c.name)+": allocations outside the storage table exceed 4 KiB");
	}
}

} // namespace

int main(const int argc, const char* const argv[]) {
	(void)argv;
	try {
		if(cuInit(0u)!=CUDA_SUCCESS) {
			std::cout << "SKIP: CUDA driver is unavailable.\n";
			return 77;
		}
		int device_count = 0;
		if(cuDeviceGetCount(&device_count)!=CUDA_SUCCESS || device_count<=0) {
			std::cout << "SKIP: no CUDA device is available.\n";
			return 77;
		}
		if(argc>1) {
			std::cerr << "usage: CULBM-vram-accounting-test\n";
			return 2;
		}
		// Grids are a few hundred MiB per domain so the constant-size buffers
		// (three subgrid reductions of 1 MiB booking each) stay below 1%.
		const Case cases[] = {
			{"D3Q19 FP16C, T D3Q7", Momentum_Lattice::Q19, DDF_Storage::FP16C, Temperature_Lattice::Q7, false,
				256u, 192u, 128u, 1u, 85u, 21u, 143u},
			{"D3Q27 FP32, T D3Q19", Momentum_Lattice::Q27, DDF_Storage::FP32, Temperature_Lattice::Q19, false,
				256u, 192u, 128u, 1u, 217u, 21u, 419u},
			{"D3Q19 FP16S, SurfaceIBM", Momentum_Lattice::Q19, DDF_Storage::FP16S, Temperature_Lattice::Q7, true,
				256u, 192u, 128u, 1u, 98u, 21u, 144u},
			{"D3Q19 FP16C, 2 domains", Momentum_Lattice::Q19, DDF_Storage::FP16C, Temperature_Lattice::Q7, false,
				384u, 256u, 128u, 2u, 85u, 21u, 143u},
		};
		for(const Case& c : cases) {
			if(c.dx>(uint)device_count) {
				std::cout << "skipped " << c.name << ": needs " << c.dx << " GPUs\n";
				continue;
			}
			run(c);
		}
		// The first case is a hand-checked example of the old accounting:
		// 6 Mi cells x (38+4+12+1+12+14+4) B, each field rounded on its own.
		set_momentum_lattice(Momentum_Lattice::Q19);
		set_ddf_storage(DDF_Storage::FP16C);
		set_temperature_lattice(Temperature_Lattice::Q7);
		configure_surface_ibm(false);
		expect(lattice_vram_mib(256u, 192u, 128u, 1u, 1u, 1u)==510u,
			"256x192x128 D3Q19/FP16C/D3Q7 plans 510 MiB");
		if(failures==0) std::cout << "VRAM accounting matches the allocated buffers.\n";
		return failures==0 ? 0 : 1;
	} catch(const std::exception& error) {
		std::cerr << "VRAM_ACCOUNTING_TEST_ERROR " << error.what() << '\n';
		return 2;
	}
}
