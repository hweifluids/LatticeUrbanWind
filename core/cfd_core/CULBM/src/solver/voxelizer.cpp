#include "voxelizer.hpp"
#include "field_set.hpp"
#include "solver.hpp"

#include <atomic>
#include <cmath>
#include <limits>

namespace culbm::solver {
namespace {

struct Box {
	float lo[3];
	float hi[3];
};

Box triangle_box(const float3& a, const float3& b, const float3& c) {
	return Box{{std::fmin(std::fmin(a.x, b.x), c.x), std::fmin(std::fmin(a.y, b.y), c.y), std::fmin(std::fmin(a.z, b.z), c.z)},
		{std::fmax(std::fmax(a.x, b.x), c.x), std::fmax(std::fmax(a.y, b.y), c.y), std::fmax(std::fmax(a.z, b.z), c.z)}};
}

// Cells of a subdomain (halo layers included) as a box in global coordinates.
Box subdomain_box(const Placement& p) {
	Box box;
	for(uint a=0u; a<3u; ++a) {
		box.lo[a] = (float)p.origin[a];
		box.hi[a] = (float)(p.origin[a]+(int)p.extent[a]-1);
	}
	return box;
}

// A ray along `axis` can hit the triangle inside the subdomain when their
// boxes overlap on the two other axes (with one cell of margin).
bool rays_may_hit(const Box& triangle, const Box& region, const uint axis) {
	constexpr float margin = 1.0f, slack = 1.0e-4f;
	const uint across[2] = {axis==0u ? 1u : 0u, axis==2u ? 1u : 2u};
	for(const uint a : across) {
		if(!(triangle.hi[a]>=region.lo[a]-margin-slack && triangle.lo[a]<=region.hi[a]+margin+slack)) return false;
	}
	return true;
}

string gib(const double mib) {
	return to_string(mib/1024.0, 3u);
}

} // namespace

uint voxelization_axis(const Mesh& mesh, const uchar flag, const float3& rotational_velocity) {
	if(flag==CELL_SOLID) return 2u; // solids always cast along z, for a deterministic result
	float score[3];
	bool smallest = true;
	if(length(rotational_velocity)==0.0f) {
		// Cross-sections of the padded bounding box normal to each axis.
		const float wx = (mesh.pmax.x+2.0f)-(mesh.pmin.x-2.0f);
		const float wy = (mesh.pmax.y+2.0f)-(mesh.pmin.y-2.0f);
		const float wz = (mesh.pmax.z+2.0f)-(mesh.pmin.z-2.0f);
		score[0] = wy*wz;
		score[1] = wz*wx;
		score[2] = wx*wy;
	} else {
		score[0] = fabsf(rotational_velocity.x);
		score[1] = fabsf(rotational_velocity.y);
		score[2] = fabsf(rotational_velocity.z);
		smallest = false;
	}
	uint best = 0u;
	for(uint a=1u; a<3u; ++a) {
		if(smallest ? score[a]<score[best] : score[a]>score[best]) best = a;
	}
	return best;
}

void voxelize_mesh(Solver& solver, const Mesh& mesh, const uchar flag, const float3& rotation_center,
	const float3& linear_velocity, const float3& rotational_velocity) {
	const uint count = solver.local_count();
	const uint triangles = mesh.triangle_number;
	luw_emit_progress("voxelization", "Voxelizing geometry",
		to_string(triangles)+" triangles across "+to_string(count)+" domain(s)", 0ll, (long long)count, false);
	const uint axis = voxelization_axis(mesh, flag, rotational_velocity);
	log_info("Voxelize direction selected="+to_string(axis)+".");

	// Triangles each subdomain has to see; one subdomain takes the whole mesh.
	std::vector<std::vector<uint>> subsets(count);
	std::vector<uint> subset_size(count, triangles);
	if(count>1u) {
		std::vector<Box> boxes(triangles);
		parallel_for(triangles, [&](const uint i) { boxes[i] = triangle_box(mesh.p0[i], mesh.p1[i], mesh.p2[i]); });
		parallel_for((ulong)count, count, [&](const ulong d) {
			const Box region = subdomain_box(solver.local((uint)d).placement());
			std::vector<uint>& subset = subsets[d];
			subset.reserve(triangles/count+1u);
			for(uint i=0u; i<triangles; ++i)
				if(rays_may_hit(boxes[i], region, axis)) subset.push_back(i);
			subset_size[d] = (uint)subset.size();
		});
	}

	// VRAM preflight of the temporary triangle buffers, per subdomain ...
	std::vector<std::uint64_t> temporary_mib(count, 0u);
	for(uint d=0u; d<count; ++d) {
		const gpu::Device& gpu = solver.local(d).device();
		const double installed = (double)gpu.properties().memory_mb;
		temporary_mib[d] = voxelization_transient_vram_mb_per_device(subset_size[d]);
		const double total = (double)gpu.booked_mib()+(double)temporary_mib[d];
		const string where = "Voxelize preflight d="+to_string(d);
		log_info(where+" (gpu "+to_string(gpu.ordinal())+"): triangles="+to_string(subset_size[d])+
			", intent alloc="+gib((double)temporary_mib[d])+" GB, predicted total="+gib(total)+" / "+gib(installed)+" GB");
		const double vertex_array = (double)subset_size[d]*(double)sizeof(float3)/1048576.0;
		if(vertex_array>installed)
			log_warning(where+": one vertex array alone needs "+gib(vertex_array)+" GB of the "+gib(installed)+" GB installed.");
		if(total>installed)
			log_warning(where+": the predicted total exceeds the installed memory ("+gib(total)+" / "+gib(installed)+" GB).");
		else if(total>0.95*installed)
			log_warning(where+": the predicted total is above 95% of the installed memory ("+gib(total)+" / "+gib(installed)+" GB).");
	}

	// ... and per GPU: subdomains on the same GPU voxelize one after another.
	std::vector<uint> gpu_ids;
	std::vector<std::vector<uint>> on_gpu;
	for(uint d=0u; d<count; ++d) {
		const uint id = solver.local(d).device().ordinal();
		const auto known = std::find(gpu_ids.begin(), gpu_ids.end(), id);
		const std::size_t g = (std::size_t)(known-gpu_ids.begin());
		if(known==gpu_ids.end()) {
			gpu_ids.push_back(id);
			on_gpu.emplace_back();
		}
		on_gpu[g].push_back(d);
	}
	for(std::size_t g=0u; g<gpu_ids.size(); ++g) {
		const gpu::Device& gpu = solver.local(on_gpu[g].front()).device();
		const double installed = (double)gpu.properties().memory_mb;
		std::uint64_t booked = 0u, temporary_sum = 0u, temporary_peak = 0u;
		string members;
		for(const uint d : on_gpu[g]) {
			booked += solver.local(d).device().booked_mib();
			temporary_sum += temporary_mib[d];
			temporary_peak = std::max(temporary_peak, temporary_mib[d]);
			members += (members.empty() ? "" : ",")+to_string(d);
		}
		const std::uint64_t peak = booked+temporary_peak;
		const string where = "Voxelize preflight GPU "+to_string(gpu_ids[g]);
		log_info(where+": domains={"+members+"}, intent total="+gib((double)peak)+" / "+gib(installed)+
			" GB (base="+gib((double)booked)+" GB, temp seq="+gib((double)temporary_peak)+
			" GB, temp parallel="+gib((double)temporary_sum)+" GB).");
		if((double)peak>installed)
			log_warning(where+": the sequential peak exceeds the installed memory ("+gib((double)peak)+" / "+gib(installed)+" GB).");
		else if((double)peak>0.90*installed)
			log_warning(where+": the sequential peak is above 90% of the installed memory ("+gib((double)peak)+" / "+gib(installed)+" GB).");
		const std::uint64_t free = (std::uint64_t)gpu.free_memory_bytes();
		const std::uint64_t reserve = gpu::runtime_reserve_bytes((std::uint64_t)gpu.total_memory_bytes());
		const std::uint64_t needed = temporary_peak>std::numeric_limits<std::uint64_t>::max()/gpu::mib_bytes
			? std::numeric_limits<std::uint64_t>::max() : temporary_peak*gpu::mib_bytes;
		if(needed>free || free-needed<reserve) {
			fatal_error("Voxelization on GPU "+to_string(gpu_ids[g])+" needs "+to_string((ulong)temporary_peak)+
				" MiB of temporary triangle buffers, but only "+to_string((ulong)(free/gpu::mib_bytes))+" MiB are free and "+
				to_string((ulong)(reserve/gpu::mib_bytes))+" MiB must stay reserved. Reduce the mesh or the lattice.");
		}
	}

	std::atomic_uint finished{0u};
	const auto run = [&](const uint d) {
		// A subset that holds every triangle is passed as the whole mesh.
		const std::vector<uint>* subset = count>1u && subsets[d].size()!=triangles ? &subsets[d] : nullptr;
		solver.local(d).voxelize(mesh, axis, flag, subset, rotation_center, linear_velocity, rotational_velocity);
		const uint done = finished.fetch_add(1u)+1u;
		luw_emit_progress("voxelization", "Voxelizing geometry",
			"Finished domain "+to_string((ulong)done)+"/"+to_string((ulong)count), (long long)done, (long long)count, false);
	};
	if(gpu_ids.size()<=1u) {
		for(uint d=0u; d<count; ++d) run(d); // one GPU: lowest transient peak
	} else {
		parallel_for((ulong)gpu_ids.size(), (uint)gpu_ids.size(), [&](const ulong g) {
			for(const uint d : on_gpu[g]) run(d);
		});
	}

	if(count>1u && length(linear_velocity)==0.0f && length(rotational_velocity)==0.0f) {
		const auto range = std::minmax_element(subset_size.begin(), subset_size.end());
		ulong sum = 0u;
		for(const uint n : subset_size) sum += n;
		log_info("Voxelize pass "+to_string(axis)+" triangle subsets: min="+to_string(*range.first)+", max="+
			to_string(*range.second)+", sum="+to_string(sum)+", d0="+to_string(subset_size[0])+
			", est peak MB/GPU="+to_string((float)*range.second*(36.0f/1048576.0f), 2u));
	}
	FieldView<uchar>& flags = solver.flags();
	if(flag==CELL_SOLID) {
		ulong solid = 0u;
		for(ulong n=0u; n<flags.length(); ++n) solid += (flags[n]&CELL_SOLID) ? 1u : 0u;
		log_info("Voxelized cells (whole domain global, no halos): solid = "+to_string(solid)+
			", fluid = "+to_string(flags.length()-solid)+", total = "+to_string(flags.length())+".");
	}
	if(!solver.initialized()) {
		flags.download();
		solver.velocity().download();
	}
}

} // namespace culbm::solver
