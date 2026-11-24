#include "subdomain.hpp"

#include <cmath>
#include <cstring>

namespace culbm::solver {

Subdomain::Subdomain(const Decomposition& grid, const unsigned id, const gpu::DeviceProperties& gpu,
	const FeatureSet& features, const Physics& physics)
	: placement_(grid.place(id)), features_(features), physics_(physics) {
	device_ = gpu::Device(gpu, kernel_source(placement_, grid, features_, physics_.transport));
	const string ordinal = to_string(gpu.ordinal);
	log_info("Allocating the lattice fields on GPU "+ordinal+".");
	luw_emit_progress("gpu_memory", "Configuring GPU memory", "Allocating CFD buffers on device "+ordinal, 0ll, 1ll, false);
	fields_ = FieldSet(device_, features_, cells(), (index_t)extent(0u)*extent(1u));
	kernels_ = KernelSet(device_, fields_, features_, physics_.forces, physics_.temperature_outlet_face, time());
	kernels_.init_damping_tables();
	if(grid.any_split())
		halo_ = HaloPort(device_, fields_, grid.halo_capacity(), halo_cell_bytes(storage_of(features_)), time(),
			features_.temperature_enabled);
	luw_emit_progress("gpu_memory", "Configuring GPU memory", "Device "+ordinal+" buffers ready", 1ll, 1ll, false);
}

void Subdomain::advance_clock(const uint steps) {
	const ulong now = t_.fetch_add((ulong)steps, std::memory_order_acq_rel)+(ulong)steps;
#ifdef UPDATE_FIELDS
	if(last_step_wrote_macros_) macros_time_ = now;
#else
	(void)now;
#endif
}

void Subdomain::rewind_clock() {
	t_.store(0u, std::memory_order_release);
#ifdef UPDATE_FIELDS
	macros_time_ = 0u;
#endif
}

void Subdomain::initialize_fields() {
	kernels_.initialize(physics_.forces, fields_.force.has_host() ? 1u : 0u);
}

void Subdomain::stream_collide() {
	const uint write_macros = step_writes_macros_ ? 1u : 0u;
	const uint read_force = fields_.force.has_host() ? 1u : 0u;
	kernels_.stream_collide(time(), physics_.forces, write_macros, read_force);
	last_step_wrote_macros_ = step_writes_macros_;
}

void Subdomain::refresh_macros() {
	if(macros_time_==time()) return;
	temperature_outlet();
	kernels_.update_macros(time(), physics_.forces, fields_.force.has_host() ? 1u : 0u);
	macros_time_ = time();
}

void Subdomain::temperature_outlet() {
	if(!features_.temperature_enabled) return;
	const int face = physics_.temperature_outlet_face;
	if(face==0) return;
	const bool x_normal = face<=(int)Temperature_Outlet_Face::East;
	const index_t face_cells = x_normal ? (index_t)extent(1u)*extent(2u) : (index_t)extent(2u)*extent(0u);
	kernels_.temperature_outlet(time(), face, face_cells);
}

void Subdomain::free_slip(const uint face_mask) {
	if(face_mask!=0u) kernels_.free_slip(face_mask);
}

void Subdomain::ibm_base_predictor(gpu::MirroredBuffer<ulong>& support, const ulong count) {
	// Unlike refresh_macros() this always runs: SurfaceIBM needs the base
	// velocity of its support cells after it has reset its own force field.
	if(!features_.surface_ibm.enabled || count==0u) return;
	kernels_.update_ibm_support(time(), physics_.forces, fields_.force.has_host() ? 1u : 0u, support, count);
}

void Subdomain::start_subgrid_sample() {
	if(!features_.subgrid.enabled) return;
	gpu::MirroredBuffer<double>& sum = fields_.sgs_sum;
	gpu::MirroredBuffer<float>& maximum = fields_.sgs_max;
	gpu::MirroredBuffer<ulong>& count = fields_.sgs_count;
	sum[0] = sum[1] = 0.0;           // nu_t, rho*nu_t
	maximum[0] = maximum[1] = 0.0f;
	count[0] = 0u;
	sum.upload_async();
	maximum.upload_async();
	count.upload_async();
	kernels_.sample_subgrid(time(), physics_.forces, fields_.force.has_host() ? 1u : 0u);
	sum.download_async();
	maximum.download_async();
	count.download_async();
}

Subgrid_Statistics Subdomain::finish_subgrid_sample() {
	Subgrid_Statistics s;
	if(!features_.subgrid.enabled) return s;
	device_.synchronize();
	s.count = (std::uint64_t)fields_.sgs_count[0];
	s.valid = s.count>0u;
	if(!s.valid) return s;
	s.mean_nu_t = fields_.sgs_sum[0]/(double)s.count;
	s.max_nu_t = fields_.sgs_max[0];
	s.mean_rho_nu_t = fields_.sgs_sum[1]/(double)s.count;
	s.max_rho_nu_t = fields_.sgs_max[1];
	return s;
}

bool Subdomain::enable_statistics(const bool include_temperature, const uint later_mib) {
	disable_statistics();
	const bool with_temperature = include_temperature && features_.temperature_enabled;
	const index_t n = cells();
	gpu::AllocationTally needed;
	needed.add(n, 3u*sizeof(float)).add(n, sizeof(float)).add(n, 3u*sizeof(float)); // mean u, mean rho, M2 u
	if(with_temperature) needed.add(n, sizeof(float)); // mean T
	// Optional: use the free VRAM the driver reports now, keeping the runtime
	// reserve and memory promised to later required features (particles).
	if(!needed.valid() || !device_.optional_allocation_fits(needed.mib(), later_mib)) return false;
	DeviceStatistics& s = statistics_;
	s.mean_u = gpu::MirroredBuffer<float>(device_, n, 3u, gpu::HostOnDemand{});
	s.mean_rho = gpu::MirroredBuffer<float>(device_, n, 1u, gpu::HostOnDemand{});
	s.m2_u = gpu::MirroredBuffer<float>(device_, n, 3u, gpu::HostOnDemand{});
	s.accumulate = gpu::Launch(device_, "culbm_accumulate_statistics", n);
	s.temperature = with_temperature;
	FieldSet& f = fields_;
	if(features_.temperature_enabled) {
		if(with_temperature) {
			s.mean_T = gpu::MirroredBuffer<float>(device_, n, 1u, gpu::HostOnDemand{});
			s.accumulate.push(f.density, f.velocity, f.temperature, s.mean_u, s.mean_rho, s.m2_u, s.mean_T, 0.0f);
		} else {
			s.accumulate.push(f.density, f.velocity, f.temperature, s.mean_u, s.mean_rho, s.m2_u,
				static_cast<float*>(nullptr), 0.0f);
		}
		s.inverse_slot = 7u;
	} else {
		s.accumulate.push(f.density, f.velocity, s.mean_u, s.mean_rho, s.m2_u, 0.0f);
		s.inverse_slot = 5u;
	}
	s.active = true;
	return true;
}

void Subdomain::disable_statistics() {
	DeviceStatistics& s = statistics_;
	s.active = false;
	s.temperature = false;
	s.accumulate = gpu::Launch();
	s.mean_u = gpu::MirroredBuffer<float>();
	s.mean_rho = gpu::MirroredBuffer<float>();
	s.m2_u = gpu::MirroredBuffer<float>();
	s.mean_T = gpu::MirroredBuffer<float>();
}

void Subdomain::reset_statistics() {
	DeviceStatistics& s = statistics_;
	if(!s.active) return;
	s.mean_u.zero_device_async();
	s.mean_rho.zero_device_async();
	s.m2_u.zero_device_async();
	if(s.temperature) s.mean_T.zero_device_async();
}

void Subdomain::sample_statistics(const float inverse_samples) {
	if(!statistics_.active) return;
	if(!(inverse_samples>0.0f) || !std::isfinite(inverse_samples))
		fatal_error("Device statistics received an invalid sample count.");
	refresh_macros();
	statistics_.accumulate.bind(statistics_.inverse_slot, inverse_samples).submit();
}

bool Subdomain::configure_probes(const std::vector<ulong>& local_cells, const std::vector<ulong>& slots,
	const uint later_mib) {
	disable_probes();
	if(local_cells.empty()) return true;
	if(local_cells.size()!=slots.size()) fatal_error("Probe cells and output slots differ in number.");
	for(const ulong cell : local_cells)
		if(cell>=cells()) fatal_error("A velocity probe lies outside its subdomain.");
	const ulong count = (ulong)local_cells.size();
	gpu::AllocationTally needed;
	needed.add(count, sizeof(ulong)).add(count, 3u*sizeof(float)); // cell indices, velocities
	if(!needed.valid() || !device_.optional_allocation_fits(needed.mib(), later_mib)) return false;
	ProbeSampler& p = probes_;
	p.cells = gpu::MirroredBuffer<ulong>(device_, count);
	std::copy(local_cells.begin(), local_cells.end(), p.cells.data());
	p.cells.upload();
	p.velocity = gpu::MirroredBuffer<float>(device_, count, 3u, gpu::HostMemory::pinned, 0.0f);
	p.slots = slots;
	p.gather = gpu::Launch(device_, "culbm_gather_probe_velocity", count);
	p.gather.push(fields_.velocity, p.cells, p.velocity, count);
	return true;
}

void Subdomain::disable_probes() {
	ProbeSampler& p = probes_;
	p.gather = gpu::Launch();
	p.cells = gpu::MirroredBuffer<ulong>();
	p.velocity = gpu::MirroredBuffer<float>();
	p.slots.clear();
}

void Subdomain::sample_probes() {
	if(probes_.slots.empty()) return;
	refresh_macros();
	probes_.gather.submit();
}

void Subdomain::voxelize(const Mesh& mesh, const uint axis, const uchar flag, const std::vector<uint>* subset,
	const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) {
	const uint triangles = subset!=nullptr ? (uint)subset->size() : mesh.triangle_number;
	if(axis<=2u && triangles>0u) {
		// Vertices of the subset, or the mesh arrays themselves.
		std::vector<float3> picked[3];
		const float3* vertices[3] = {mesh.p0, mesh.p1, mesh.p2};
		if(subset!=nullptr) {
			const float3* const source[3] = {mesh.p0, mesh.p1, mesh.p2};
			for(uint k=0u; k<3u; ++k) {
				picked[k].reserve(triangles);
				for(const uint triangle : *subset) picked[k].push_back(source[k][triangle]);
				vertices[k] = picked[k].data();
			}
		}
		gpu::Buffer<float3> corner[3];
		for(uint k=0u; k<3u; ++k) {
			corner[k] = gpu::Buffer<float3>(device_, triangles, 1u, gpu::Init::none);
			corner[k].copy_from(vertices[k]);
		}
		// Parameter block: triangle count (bit pattern), the mesh bounding box
		// padded by two cells (for re-voxelized moving objects), rotation centre,
		// linear and angular velocity.
		const float bounds[6] = {
			mesh.pmin.x-2.0f, mesh.pmin.y-2.0f, mesh.pmin.z-2.0f,
			mesh.pmax.x+2.0f, mesh.pmax.y+2.0f, mesh.pmax.z+2.0f};
		float block[16];
		std::memcpy(&block[0], &triangles, sizeof(uint));
		std::memcpy(&block[1], bounds, sizeof(bounds));
		const float3 motion[3] = {rotation_center, linear_velocity, rotational_velocity};
		for(uint k=0u; k<3u; ++k) {
			block[7u+3u*k] = motion[k].x;
			block[8u+3u*k] = motion[k].y;
			block[9u+3u*k] = motion[k].z;
		}
		gpu::Buffer<float> parameters(device_, 16u, 1u, gpu::Init::none);
		parameters.copy_from(block);

		// 2D tile binning along the two transverse axes reduces ray-triangle
		// tests by orders of magnitude on dense meshes while preserving exact
		// ascending triangle order and bit-identical voxelized flags.
		bool launched_binned = false;
		if(triangles>=2048u) {
			const uint axis_u = axis==0u ? 1u : 0u;
			const uint axis_v = axis==2u ? 1u : 2u;
			const uint ext_u = extent(axis_u);
			const uint ext_v = extent(axis_v);
			const int orig_u = origin(axis_u);
			const int orig_v = origin(axis_v);
			const auto coord = [](const float3& p, const uint a) -> float {
				return a==0u ? p.x : (a==1u ? p.y : p.z);
			};
			// Precompute clamped local cell bounds [u0,u1] x [v0,v1] per triangle.
			struct TriSpan { int u0, u1, v0, v1; };
			std::vector<TriSpan> spans(triangles);
			parallel_for(triangles, [&](const uint i) {
				const float3 a = vertices[0][i], b = vertices[1][i], c = vertices[2][i];
				const float lo_u = std::fmin(std::fmin(coord(a, axis_u), coord(b, axis_u)), coord(c, axis_u));
				const float hi_u = std::fmax(std::fmax(coord(a, axis_u), coord(b, axis_u)), coord(c, axis_u));
				const float lo_v = std::fmin(std::fmin(coord(a, axis_v), coord(b, axis_v)), coord(c, axis_v));
				const float hi_v = std::fmax(std::fmax(coord(a, axis_v), coord(b, axis_v)), coord(c, axis_v));
				const int u0 = std::max(0, (int)std::floor(lo_u-1.0e-3f)-orig_u);
				const int u1 = std::min((int)ext_u-1, (int)std::ceil(hi_u+1.0e-3f)-orig_u);
				const int v0 = std::max(0, (int)std::floor(lo_v-1.0e-3f)-orig_v);
				const int v1 = std::min((int)ext_v-1, (int)std::ceil(hi_v+1.0e-3f)-orig_v);
				spans[i] = TriSpan{u0, u1, v0, v1};
			});
			const uint candidate_sizes[3] = {4u, 8u, 16u};
			for(const uint bin_size : candidate_sizes) {
				const uint nbins_u = (ext_u+bin_size-1u)/bin_size;
				const uint nbins_v = (ext_v+bin_size-1u)/bin_size;
				const ulong nbins = (ulong)nbins_u*(ulong)nbins_v;
				if(nbins==0ull || nbins>2097152ull) continue;
				std::vector<uint> offsets((size_t)(nbins+1ull), 0u);
				ulong total_entries = 0ull;
				constexpr ulong max_entries = 67108864ull; // 256 MiB CSR cap
				for(uint i=0u; i<triangles; ++i) {
					const TriSpan& s = spans[i];
					if(s.u0>s.u1 || s.v0>s.v1) continue;
					const uint bu0 = (uint)s.u0/bin_size, bu1 = (uint)s.u1/bin_size;
					const uint bv0 = (uint)s.v0/bin_size, bv1 = (uint)s.v1/bin_size;
					total_entries += (ulong)(bu1-bu0+1u)*(ulong)(bv1-bv0+1u);
					if(total_entries>max_entries) break;
					for(uint bv=bv0; bv<=bv1; ++bv) {
						const ulong row = (ulong)bv*(ulong)nbins_u;
						for(uint bu=bu0; bu<=bu1; ++bu) ++offsets[(size_t)(row+bu+1u)];
					}
				}
				if(total_entries>max_entries || total_entries>0xFFFFFFFFull) continue;
				gpu::AllocationTally bin_tally;
				bin_tally.add(nbins+1ull, sizeof(uint)).add(std::max<ulong>(total_entries, 1ull), sizeof(uint));
				if(!bin_tally.valid() || !device_.optional_allocation_fits(bin_tally.mib(), 0u)) continue;
				for(ulong b=0ull; b<nbins; ++b) offsets[(size_t)(b+1ull)] += offsets[(size_t)b];
				std::vector<uint> cursor(offsets.begin(), offsets.begin()+(std::ptrdiff_t)nbins);
				std::vector<uint> bin_tris((size_t)std::max<ulong>(total_entries, 1ull), 0u);
				for(uint i=0u; i<triangles; ++i) {
					const TriSpan& s = spans[i];
					if(s.u0>s.u1 || s.v0>s.v1) continue;
					const uint bu0 = (uint)s.u0/bin_size, bu1 = (uint)s.u1/bin_size;
					const uint bv0 = (uint)s.v0/bin_size, bv1 = (uint)s.v1/bin_size;
					for(uint bv=bv0; bv<=bv1; ++bv) {
						const ulong row = (ulong)bv*(ulong)nbins_u;
						for(uint bu=bu0; bu<=bu1; ++bu)
							bin_tris[(size_t)(cursor[(size_t)(row+bu)]++)] = i;
					}
				}
				gpu::Buffer<uint> d_offsets(device_, (index_t)(nbins+1ull), 1u, gpu::Init::none);
				gpu::Buffer<uint> d_tris(device_, (index_t)bin_tris.size(), 1u, gpu::Init::none);
				d_offsets.copy_from(offsets.data());
				d_tris.copy_from(bin_tris.data());
				gpu::Launch(device_, "voxelize_mesh_binned", placement_.face_cells(axis))
					.push(axis, fields_.ddf, fields_.velocity, fields_.flags, time()+1ull, flag,
						corner[0], corner[1], corner[2], parameters, bin_size, nbins_u, d_offsets, d_tris)
					.submit_and_wait();
				launched_binned = true;
				break;
			}
		}
		if(!launched_binned) {
			gpu::Launch(device_, "voxelize_mesh", placement_.face_cells(axis))
				.push(axis, fields_.ddf, fields_.velocity, fields_.flags, time()+1ull, flag,
					corner[0], corner[1], corner[2], parameters)
				.submit_and_wait();
		}
	}
	if(flag!=CELL_SOLID) return;
	// Report the solid and fluid cells of this subdomain, halo layers included.
	gpu::MirroredBuffer<uchar>& flags = fields_.flags;
	flags.download();
	ulong solid = 0u;
	for(ulong n=0u; n<flags.count(); ++n) solid += (flags[n]&CELL_SOLID) ? 1u : 0u;
	log_info("Voxelized cells (single-GPU local domain, includes halos, device "+to_string(device_.ordinal())+
		"): solid = "+to_string(solid)+", fluid = "+to_string(flags.count()-solid)+", total = "+to_string(flags.count())+".");
}

} // namespace culbm::solver
