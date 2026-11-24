#include "solver.hpp"
#include "voxelizer.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace culbm::solver {

namespace detail {

void before_host_read(Solver& solver) {
	if(!solver.initialized()) return;
	for(uint d=0u; d<solver.local_count(); ++d) solver.local(d).refresh_macros();
}

ulong solver_time(const Solver& solver) {
	return solver.time();
}

} // namespace detail

namespace {

void report_wall_model(const Wall_Model_Lattice& wm) {
	if(!wm.enabled) return;
	const string ground = wm.smooth_h ? string("smooth") : to_string(wm.cd_h, 6u);
	const string building = wm.smooth_v ? string("smooth") : to_string(wm.cd_v, 6u);
	const string layer = wm.sgs ? to_string((uint)wm.sgs_layers)+" row(s)" : string("off");
	const string surfaces = !wm.ground_only ? string()
		: wm.ibm ? string(", SurfaceIBM surfaces and voxel walls") : string(", voxel walls only (SurfaceIBM surfaces no-slip)");
	log_info("Wall model: dx="+to_string(wm.cell_size_m, 4u)+" m, Cd ground="+ground+", Cd building="+building+
		", SGS layer="+layer+surfaces);
}

// CULBM_DEVICE_STATISTICS=0/off/false keeps statistics on the host.
bool device_statistics_vetoed() {
	const char* const value = std::getenv("CULBM_DEVICE_STATISTICS");
	if(value==nullptr) return false;
	for(const char* off : {"0", "off", "OFF", "false", "FALSE"})
		if(std::strcmp(value, off)==0) return true;
	return false;
}

} // namespace

Solver::Solver(const uint nx, const uint ny, const uint nz, const uint dx, const uint dy, const uint dz,
	const CaseCoefficients& coefficients) {
	CaseValidator::check_decomposition(nx, ny, nz, dx, dy, dz);
	grid_ = Decomposition::fitted(nx, ny, nz, dx, dy, dz);
	const string requested = to_string(nx)+"x"+to_string(ny)+"x"+to_string(nz);
	const string fitted = to_string(grid_.size(0u))+"x"+to_string(grid_.size(1u))+"x"+to_string(grid_.size(2u));
	if(requested!=fitted)
		log_warning("The lattice "+requested+" is not divisible into "+to_string(dx)+"x"+to_string(dy)+"x"+to_string(dz)+
			" subdomains; it is reduced to "+fitted+".");
	features_ = FeatureSet::current();
	report_wall_model(features_.wall_model);
	physics_.transport.nu = coefficients.nu;
	physics_.transport.alpha = coefficients.alpha;
	physics_.transport.beta = coefficients.beta;
	physics_.forces.fx = coefficients.fx;
	physics_.forces.fy = coefficients.fy;
	physics_.forces.fz = coefficients.fz;

	const uint count = grid_.count();
	const CellCost cost = cell_cost(storage_of(features_));
	const std::uint64_t plan = lattice_vram_mib(grid_.size(0u), grid_.size(1u), grid_.size(2u), dx, dy, dz);
	log_info("VRAM plan: "+to_string((ulong)plan)+" MiB per GPU x "+to_string(count)+" = "+to_string((ulong)plan*(ulong)count)+
		" MiB for "+fitted+" cells ("+to_string(cost.device)+" B/cell, host mirrors "+to_string(cost.host)+" B/cell).");

	culbm::mpi_validate_domain_count(count);
	distributed_ = culbm::mpi_is_distributed();
	std::vector<gpu::DeviceProperties> gpus;
	if(distributed_) {
		gpus.push_back(gpu::rank_device());
		owned_ = (uint)culbm::mpi_rank();
		if(culbm::mpi_is_root())
			log_info("MPI distributed mode: "+to_string((uint)culbm::mpi_size())+" rank(s), one subdomain and one GPU per rank.");
	} else {
		gpus = gpu::choose_devices(count);
		gpu::arrange_for_peer_halos(gpus, grid_.size(0u), grid_.size(1u), grid_.size(2u), dx, dy, dz);
	}
	CaseValidator::check_construction(grid_, gpus, coefficients);

	for(uint d=0u; d<(uint)gpus.size(); ++d)
		locals_.push_back(std::make_unique<Subdomain>(grid_, distributed_ ? owned_ : d, gpus[d], features_, physics_));
	string booked;
	for(const auto& s : locals_) booked += (booked.empty() ? "" : ", ")+to_string((ulong)s->device().booked_mib());
	log_info("VRAM booked by the lattice: "+booked+" MiB per GPU (plan "+to_string((ulong)plan)+" MiB).");

	std::vector<Subdomain*> parts;
	for(const auto& s : locals_) parts.push_back(s.get());
	halos_ = HaloExchanger(grid_, parts, distributed_, owned_, features_);
	pipeline_.build(PipelineConfig{local_count(), grid_.any_split(), false});

	const auto collect = [&](auto member) {
		using Buffer = std::remove_reference_t<decltype(locals_.front()->fields().*member)>;
		std::vector<Buffer*> buffers;
		for(const auto& s : locals_) buffers.push_back(&(s->fields().*member));
		return buffers;
	};
	density_.bind(*this, grid_, distributed_, owned_, collect(&FieldSet::density), "rho");
	velocity_.bind(*this, grid_, distributed_, owned_, collect(&FieldSet::velocity), "u");
	flags_.bind(*this, grid_, distributed_, owned_, collect(&FieldSet::flags), "flags");
	force_.bind(*this, grid_, distributed_, owned_, collect(&FieldSet::force), "F");
	if(features_.temperature_enabled)
		temperature_.bind(*this, grid_, distributed_, owned_, collect(&FieldSet::temperature), "T");
}

Solver::~Solver() {
	forcing_.reset(); // its device buffers go before the subdomains
}

void Solver::attach_immersed_forcing(std::unique_ptr<ImmersedForcing> forcing) {
	forcing_ = std::move(forcing);
	pipeline_.build(PipelineConfig{local_count(), grid_.any_split(), has_immersed_forcing()});
}

void Solver::detach_immersed_forcing() {
	forcing_.reset();
	pipeline_.build(PipelineConfig{local_count(), grid_.any_split(), false});
}

void Solver::set_rotation(const float omega_x, const float omega_y, const float omega_z) {
	physics_.forces.omega_x = omega_x;
	physics_.forces.omega_y = omega_y;
	physics_.forces.omega_z = omega_z;
}

void Solver::set_gravity(const float gx, const float gy, const float gz) {
	physics_.forces.gx = gx;
	physics_.forces.gy = gy;
	physics_.forces.gz = gz;
}

bool Solver::set_temperature_outlet_face(const int face) {
	if(face<(int)Temperature_Outlet_Face::None || face>(int)Temperature_Outlet_Face::North) return false;
	if(ever_initialized_) return false;
	if(!features_.temperature_enabled) return true;
	const bool normal_x = face==(int)Temperature_Outlet_Face::West || face==(int)Temperature_Outlet_Face::East;
	const bool normal_y = face==(int)Temperature_Outlet_Face::South || face==(int)Temperature_Outlet_Face::North;
	if(normal_x && grid_.inner(0u)<2u) return false;
	if(normal_y && grid_.inner(1u)<2u) return false;
	physics_.temperature_outlet_face = face;
	return true;
}

void Solver::barrier() const {
	if(distributed_) culbm::mpi_barrier();
}

// Near-wall SGS layer map: for every local column (halos included) the global
// z of the first fluid row above the column's solid base, taken from the
// global host flags so every decomposition sees identical values. Columns
// whose global z=0 cell is not a wall carry 0xFFFF. SurfaceIBM solvers only
// model the compacted ground row, so their map ignores solids above it.
void Solver::build_wall_columns() {
	const long long nx = grid_.size(0u), ny = grid_.size(1u);
	const ulong nz = grid_.size(2u);
	const auto wall = [&](const ulong x, const ulong y, const ulong z) {
		return (flags_[x+(y+z*(ulong)ny)*(ulong)nx]&(CELL_SOLID|CELL_EQUILIBRIUM|CELL_AUX_X))==CELL_SOLID;
	};
	for(const auto& s : locals_) {
		if(!s->wall_model().sgs) continue;
		gpu::MirroredBuffer<ushort>& columns = s->fields().wall_columns;
		for(uint y=0u; y<s->extent(1u); ++y) {
			for(uint x=0u; x<s->extent(0u); ++x) {
				// Halo columns wrap around the periodic global lattice.
				const ulong gx = (ulong)(((long long)x+s->origin(0u)+nx)%nx);
				const ulong gy = (ulong)(((long long)y+s->origin(1u)+ny)%ny);
				ushort surface = (ushort)0xFFFFu;
				if(wall(gx, gy, 0u)) {
					// Solid stack from the bottom: compacted ground slab of any
					// thickness, terrain or voxel roofs. With no-slip SurfaceIBM
					// the scan runs after the support band was cleared and stops
					// below the skin of immersed bodies (no layer above roofs).
					ulong z = 1u;
					while(z<nz && wall(gx, gy, z)) ++z;
					if(z<nz && z<0xFFFFu) surface = (ushort)z;
				}
				columns[(ulong)x+(ulong)y*(ulong)s->extent(0u)] = surface;
			}
		}
		columns.upload_async();
	}
}

void Solver::initialize() {
	CaseValidator::check_immersed_boundary(has_ibm_storage(), has_immersed_forcing());
	CaseValidator::check_cell_types(flags_, velocity_);
	for(const auto& s : locals_) s->fields().density.upload_async();
	for(const auto& s : locals_) s->fields().velocity.upload_async();
	for(const auto& s : locals_) s->fields().flags.upload_async();
	build_wall_columns();
	// F has a host copy only once the host wrote or read it; until then the
	// device keeps the zeros it was created with.
	for(const auto& s : locals_) if(s->fields().force.has_host()) s->fields().force.upload_async();
	if(features_.temperature_enabled)
		for(const auto& s : locals_) s->fields().temperature.upload_async();
	// The initial exchanges run on an odd step so the pack kernels address the
	// DDFs the equilibrium initialization writes.
	for(const auto& s : locals_) s->advance_clock();
	halos_.exchange(HaloField::macros);
	for(const auto& s : locals_) s->initialize_fields();
	halos_.exchange(HaloField::macros);
	if(features_.temperature_enabled) halos_.exchange(HaloField::temperature);
	halos_.exchange(HaloField::populations);
	synchronize();
	for(const auto& s : locals_) s->rewind_clock();
	initialized_ = true;
	ever_initialized_ = true;
}

void Solver::step() {
	for(const Stage stage : pipeline_.stages()) {
		switch(stage) {
		case Stage::thermal_outlet:
			for(const auto& s : locals_) s->temperature_outlet();
			break;
		case Stage::immersed_forcing:
			forcing_->prepare_step();
			break;
		case Stage::stream_collide:
			for(const auto& s : locals_) s->stream_collide();
			break;
		case Stage::population_halo:
			halos_.exchange(HaloField::populations);
			break;
		case Stage::device_barrier:
			pace();
			break;
		case Stage::advance_clock:
			for(const auto& s : locals_) s->advance_clock();
			break;
		}
	}
}

void Solver::set_step_writes_macros(const bool write) {
	for(const auto& s : locals_) s->set_step_writes_macros(write);
}

void Solver::run(const ulong steps) {
	for(ulong done=0u; done!=steps; ++done) step();
	if(steps>0u) synchronize();
}

void Solver::synchronize() {
	for(const auto& s : locals_) s->synchronize();
}

void Solver::pace() {
	for(const auto& s : locals_) s->device().record_pace();
	for(const auto& s : locals_) s->device().wait_previous_pace();
}

void Solver::refresh_macros() {
	for(const auto& s : locals_) s->refresh_macros();
	synchronize();
}

void Solver::publish_flags() {
	halos_.exchange(HaloField::flags);
	synchronize();
}

void Solver::free_slip(const uint face_mask) {
	for(const auto& s : locals_) s->free_slip(face_mask);
}

void Solver::prepare_advection() {
	for(const auto& s : locals_) s->refresh_macros();
	if(grid_.any_split()) halos_.exchange(HaloField::macros);
	else synchronize();
}

Subgrid_Statistics Solver::subgrid_statistics() {
	Subgrid_Statistics result;
	if(!initialized_ || !features_.subgrid.enabled) return result;
	for(const auto& s : locals_) s->start_subgrid_sample();
	double sums[2] = {0.0, 0.0};
	float maxima[2] = {0.0f, 0.0f};
	std::uint64_t cells = 0u;
	for(const auto& s : locals_) {
		const Subgrid_Statistics part = s->finish_subgrid_sample();
		if(!part.valid) continue;
		sums[0] += part.mean_nu_t*(double)part.count;
		sums[1] += part.mean_rho_nu_t*(double)part.count;
		maxima[0] = std::max(maxima[0], part.max_nu_t);
		maxima[1] = std::max(maxima[1], part.max_rho_nu_t);
		cells += part.count;
	}
	if(distributed_) {
		culbm::mpi_allreduce_sum_double(sums, 2u);
		culbm::mpi_allreduce_max_float(maxima, 2u);
		culbm::mpi_allreduce_sum_uint64(&cells, 1u);
	}
	result.count = cells;
	result.valid = cells>0u;
	if(!result.valid) return result;
	result.mean_nu_t = sums[0]/(double)cells;
	result.max_nu_t = maxima[0];
	result.mean_rho_nu_t = sums[1]/(double)cells;
	result.max_rho_nu_t = maxima[1];
	return result;
}

bool Solver::enable_statistics(const bool include_temperature, const uint later_mib) {
	disable_statistics();
	if(device_statistics_vetoed()) return false;
	for(uint d=0u; d<local_count(); ++d) {
		if(locals_[d]->enable_statistics(include_temperature, later_mib)) continue;
		for(uint undo=0u; undo<=d; ++undo) locals_[undo]->disable_statistics();
		return false;
	}
	return local_count()>0u;
}

void Solver::disable_statistics() {
	for(const auto& s : locals_) s->disable_statistics();
}

void Solver::reset_statistics() {
	for(const auto& s : locals_) s->reset_statistics();
}

void Solver::sample_statistics(const ulong samples) {
	if(!statistics_active()) return;
	if(samples==0u) fatal_error("Device statistics need a positive sample count.");
	const float inverse = 1.0f/(float)samples;
	for(const auto& s : locals_) s->sample_statistics(inverse);
}

void Solver::read_statistics(vector<float>& mean_u, vector<float>& mean_rho, vector<float>& mean_T,
	vector<float>& m2_u, vector<float>& m2_v, vector<float>& m2_w) {
	if(!statistics_active()) fatal_error("Device statistics were read before they were enabled.");
	const auto view_of = [&](FieldView<float>& view, gpu::MirroredBuffer<float> DeviceStatistics::*member, const char* name) {
		std::vector<gpu::MirroredBuffer<float>*> parts;
		for(const auto& s : locals_) {
			gpu::MirroredBuffer<float>& buf = s->statistics().*member;
			buf.attach_host();
			parts.push_back(&buf);
		}
		view.bind(*this, grid_, distributed_, owned_, parts, name);
		view.download();
	};
	const auto release_of = [&](gpu::MirroredBuffer<float> DeviceStatistics::*member) {
		for(const auto& s : locals_) (s->statistics().*member).release_host();
	};
	const ulong n = grid_.cells();
	const uint nz = grid_.size(2u);
	{
		FieldView<float> u_mean;
		view_of(u_mean, &DeviceStatistics::mean_u, "statistics_mean_u");
		vector<float> ux((size_t)n), uy((size_t)n), uz((size_t)n);
		u_mean.copy_contiguous_component(0u, ux.data(), 0u, nz);
		u_mean.copy_contiguous_component(1u, uy.data(), 0u, nz);
		u_mean.copy_contiguous_component(2u, uz.data(), 0u, nz);
		release_of(&DeviceStatistics::mean_u);
		mean_u.resize((size_t)(3u*n));
		parallel_for(n, [&](const ulong i) {
			mean_u[(size_t)(3u*i+0u)] = ux[(size_t)i];
			mean_u[(size_t)(3u*i+1u)] = uy[(size_t)i];
			mean_u[(size_t)(3u*i+2u)] = uz[(size_t)i];
		});
	}
	{
		FieldView<float> rho_mean;
		view_of(rho_mean, &DeviceStatistics::mean_rho, "statistics_mean_rho");
		mean_rho.resize((size_t)n);
		rho_mean.copy_contiguous_component(0u, mean_rho.data(), 0u, nz);
		release_of(&DeviceStatistics::mean_rho);
	}
	{
		FieldView<float> u_m2;
		view_of(u_m2, &DeviceStatistics::m2_u, "statistics_M2_u");
		m2_u.resize((size_t)n);
		m2_v.resize((size_t)n);
		m2_w.resize((size_t)n);
		u_m2.copy_contiguous_component(0u, m2_u.data(), 0u, nz);
		u_m2.copy_contiguous_component(1u, m2_v.data(), 0u, nz);
		u_m2.copy_contiguous_component(2u, m2_w.data(), 0u, nz);
		release_of(&DeviceStatistics::m2_u);
	}
	mean_T.clear();
	if(locals_.front()->statistics().temperature) {
		FieldView<float> t_mean;
		view_of(t_mean, &DeviceStatistics::mean_T, "statistics_mean_T");
		mean_T.resize((size_t)n);
		t_mean.copy_contiguous_component(0u, mean_T.data(), 0u, nz);
		release_of(&DeviceStatistics::mean_T);
	}
}

bool Solver::configure_probes(const vector<ulong>& global_cells, const uint later_mib) {
	disable_probes();
	if(global_cells.empty() || distributed_) return false;
	std::vector<std::vector<ulong>> cells(local_count()), slots(local_count());
	const ulong layer = (ulong)grid_.size(0u)*grid_.size(1u);
	for(ulong slot=0u; slot<(ulong)global_cells.size(); ++slot) {
		const ulong g = global_cells[(size_t)slot];
		if(g>=grid_.cells()) {
			disable_probes();
			return false;
		}
		const Decomposition::Owner owner = grid_.owner((uint)(g%grid_.size(0u)), (uint)((g/grid_.size(0u))%grid_.size(1u)), (uint)(g/layer));
		cells[owner.id].push_back(owner.local);
		slots[owner.id].push_back(slot);
	}
	for(uint d=0u; d<local_count(); ++d) {
		if(locals_[d]->configure_probes(cells[d], slots[d], later_mib)) continue;
		disable_probes();
		return false;
	}
	return probes_active();
}

void Solver::disable_probes() {
	for(const auto& s : locals_) s->disable_probes();
}

bool Solver::probes_active() const {
	for(const auto& s : locals_) if(s->has_probes()) return true;
	return false;
}

void Solver::sample_probes(vector<float>& samples) {
	if(!probes_active()) fatal_error("Velocity probes were sampled before they were configured.");
	ulong count = 0u;
	for(const auto& s : locals_) {
		for(const ulong slot : s->probes().slots) count = std::max(count, slot+1u);
		s->sample_probes();
		if(s->has_probes()) s->probes().velocity.download_async();
	}
	samples.assign((size_t)(3u*count), 0.0f);
	for(const auto& s : locals_) {
		if(!s->has_probes()) continue;
		s->synchronize();
		ProbeSampler& p = s->probes();
		for(ulong i=0u; i<p.velocity.count(); ++i) {
			const size_t out = (size_t)(3u*p.slots[(size_t)i]);
			for(uint c=0u; c<3u; ++c) samples[out+c] = p.velocity.component(c)[i];
		}
	}
}

void Solver::voxelize(const Mesh& mesh, const uchar flag, const float3& rotation_center,
	const float3& linear_velocity, const float3& rotational_velocity) {
	voxelize_mesh(*this, mesh, flag, rotation_center, linear_velocity, rotational_velocity);
}

} // namespace culbm::solver
