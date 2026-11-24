#include "field_set.hpp"

namespace culbm::solver {
namespace {

LatticeStorage make_storage(const uint velocity_set, const uint face_ddfs, const uint ddf_bytes,
	const bool temperature_enabled, const uint temperature_set, const uint temperature_face_ddfs,
	const bool surface_ibm) {
	LatticeStorage s;
	s.velocity_set = velocity_set;
	s.face_ddfs = face_ddfs;
	s.ddf_bytes = ddf_bytes;
	// FORCE_FIELD is part of every build; UPDATE_FIELDS is on unless
	// CULBM_DISABLE_UPDATE_FIELDS; TEMPERATURE is on-demand per solver.
#ifdef UPDATE_FIELDS
	s.update_fields = true;
#else
	s.update_fields = false;
#endif
	s.force_field = true;
	s.temperature = temperature_enabled;
	s.temperature_set = temperature_set;
	s.temperature_face_ddfs = temperature_face_ddfs;
	s.surface_ibm = surface_ibm;
	return s;
}

} // namespace

LatticeStorage storage_of(const FeatureSet& f) {
	return make_storage(f.velocity_set, f.face_ddfs, f.ddf_bytes, f.temperature_enabled, f.temperature_set,
		f.temperature_face_ddfs, f.surface_ibm.enabled);
}

LatticeStorage current_storage() {
	return make_storage((uint)get_momentum_lattice(), get_momentum_lattice()==Momentum_Lattice::Q27 ? 9u : 5u,
		get_ddf_bytes(), get_thermal_lattice_enabled(), get_temperature_set(), get_temperature_transfers(),
		surface_ibm::get_config().enabled);
}

FieldSet::FieldSet(const gpu::Device& device, const FeatureSet& features, const index_t cells, const index_t columns)
	: cells_(cells) {
	// Device allocation order: the lattice fields, the subgrid reductions, the
	// SurfaceIBM fields, the SGS column map. The order fixes the device
	// addresses, and with them the order in which the SurfaceIBM atomics of a
	// run accumulate; keeping it keeps IBM runs reproducible across builds.
	const std::vector<CellField> table = cell_fields(storage_of(features));
	const auto is_ibm = [](const CellField& f) { return f.id==FieldId::ibm_force || f.id==FieldId::ibm_mask; };
	for(const CellField& field : table) if(!is_ibm(field)) allocate(device, field);
	if(features.subgrid.enabled) {
		// Tiny and sampled infrequently; pinned so the reductions of all local
		// GPUs overlap before the final read.
		sgs_sum = gpu::MirroredBuffer<double>(device, 1u, 2u, gpu::HostMemory::pinned, 0.0);
		sgs_max = gpu::MirroredBuffer<float>(device, 1u, 2u, gpu::HostMemory::pinned, 0.0f);
		sgs_count = gpu::MirroredBuffer<ulong>(device, 1u, 1u, gpu::HostMemory::pinned, (ulong)0u);
	}
	for(const CellField& field : table) if(is_ibm(field)) allocate(device, field);
	if(features.wall_model.sgs)
		wall_columns = gpu::MirroredBuffer<ushort>(device, columns, 1u, gpu::HostMemory::pageable, (ushort)0xFFFFu);
}

void FieldSet::allocate(const gpu::Device& device, const CellField& field) {
	const uint parts = field.components;
	switch(field.id) {
	case FieldId::ddf: ddf = gpu::Buffer<ushort>(device, cells_, parts); break;
	case FieldId::density: density = gpu::MirroredBuffer<float>(device, cells_, parts, gpu::HostMemory::pageable, 1.0f); break;
	case FieldId::velocity: velocity = gpu::MirroredBuffer<float>(device, cells_, parts); break;
	case FieldId::flags: flags = gpu::MirroredBuffer<uchar>(device, cells_, parts); break;
	case FieldId::force: force = gpu::MirroredBuffer<float>(device, cells_, parts, gpu::HostOnDemand{}); break;
	case FieldId::thermal_ddf: thermal_ddf = gpu::Buffer<ushort>(device, cells_, parts); break;
	case FieldId::temperature: temperature = gpu::MirroredBuffer<float>(device, cells_, parts, gpu::HostMemory::pageable, 1.0f); break;
	case FieldId::ibm_force: ibm_force = gpu::Buffer<float>(device, cells_, parts); break;
	case FieldId::ibm_mask: ibm_mask = gpu::Buffer<uchar>(device, cells_, parts); break;
	}
}

} // namespace culbm::solver

culbm::solver::LatticeStorage lattice_storage() {
	return culbm::solver::current_storage();
}

std::uint64_t lattice_vram_mib(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz) {
	return culbm::solver::domain_footprint(lattice_storage(), Nx, Ny, Nz, Dx, Dy, Dz).mib;
}

std::uint64_t voxelization_transient_vram_mb_per_device(const std::uint64_t triangle_count) {
	// Three vertex arrays and the 16-float parameter block.
	gpu::AllocationTally tally;
	tally.add(triangle_count, sizeof(float3), 3u).add(16u, sizeof(float));
	return tally.valid() ? tally.mib() : ~std::uint64_t(0u);
}
