#include "halo_exchanger.hpp"
#include "subdomain.hpp"

namespace culbm::solver {
namespace {

unsigned slot_of(const HaloField field) {
	switch(field) {
	case HaloField::macros: return 0u;
	case HaloField::flags: return 1u;
	case HaloField::temperature: return 2u;
	default: return 3u;
	}
}

const char* kernel_suffix(const HaloField field, const bool temperature_enabled) {
	switch(field) {
	case HaloField::macros: return "rho_u_flags";
	case HaloField::flags: return "flags";
	case HaloField::temperature: return "T";
	default: return temperature_enabled ? "fi_gi" : "fi";
	}
}

} // namespace

uint halo_cell_bytes(const HaloField field, const FeatureSet& f) {
	switch(field) {
	case HaloField::macros: return 17u; // rho, three velocity components, flags
	case HaloField::flags: return 1u;
	case HaloField::temperature: return f.temperature_enabled ? 4u : 0u;
	default: return (f.face_ddfs+(f.temperature_enabled ? f.temperature_face_ddfs : 0u))*f.ddf_bytes;
	}
}

HaloPort::HaloPort(const gpu::Device& device, FieldSet& fields, const index_t face_capacity,
	const uint cell_bytes, const ulong t, const bool temperature_enabled) {
	// One set of face buffers serves all split axes. Their pinned host mirrors
	// stage MPI exchanges and the copies between GPUs without peer access.
	const auto face_buffer = [&]() {
		return gpu::MirroredBuffer<char>(device, face_capacity, cell_bytes, gpu::HostMemory::pinned, (char)0);
	};
	to_upper = face_buffer();
	to_lower = face_buffer();
	from_upper = face_buffer();
	from_lower = face_buffer();
	if(to_upper.host_pinned() && to_lower.host_pinned() && from_upper.host_pinned() && from_lower.host_pinned())
		log_info("Halo faces are staged through pinned host memory (asynchronous).");
	else
		log_warning("Halo faces could not be pinned in host memory; copies between GPUs will synchronize.");

	// Slots 0 and 1 (axis, time step) are rebound on every exchange.
	const auto connect = [&](const HaloField field, const auto&... cell_fields) {
		const unsigned s = slot_of(field);
		const string suffix = kernel_suffix(field, temperature_enabled);
		pack_[s] = gpu::Launch(device, "culbm_halo_pack_"+suffix).block(128u);
		pack_[s].push(0u, t, to_upper, to_lower, cell_fields...);
		unpack_[s] = gpu::Launch(device, "culbm_halo_unpack_"+suffix).block(128u);
		unpack_[s].push(0u, t, from_upper, from_lower, cell_fields...);
	};
	connect(HaloField::macros, fields.density, fields.velocity, fields.flags);
	connect(HaloField::flags, fields.flags);
	if(temperature_enabled) {
		connect(HaloField::temperature, fields.temperature);
		connect(HaloField::populations, fields.ddf, fields.thermal_ddf);
	} else {
		connect(HaloField::populations, fields.ddf);
	}
}

void HaloPort::pack(const HaloField field, const uint axis, const ulong t, const index_t face_cells) {
	pack_[slot_of(field)].over(face_cells).bind(0u, axis, t).submit();
}

void HaloPort::unpack(const HaloField field, const uint axis, const ulong t, const index_t face_cells) {
	unpack_[slot_of(field)].over(face_cells).bind(0u, axis, t).submit();
}

HaloExchanger::HaloExchanger(const Decomposition& grid, std::vector<Subdomain*> locals, const bool distributed,
	const unsigned owned, const FeatureSet& features)
	: grid_(grid), locals_(std::move(locals)), distributed_(distributed), owned_(owned), features_(features) {}

void HaloExchanger::exchange(const HaloField field) {
	if(field==HaloField::temperature && !features_.temperature_enabled) return;
	const uint bytes = halo_cell_bytes(field, features_);
	for(uint axis=0u; axis<3u; ++axis) {
		if(!grid_.split_axis(axis)) continue;
		if(distributed_) exchange_ranks(field, axis, bytes);
		else exchange_devices(field, axis, bytes);
	}
}

void HaloExchanger::exchange_devices(const HaloField field, const uint axis, const uint cell_bytes) {
	// Pack every face, then copy all of them, then unpack: the copy streams of
	// all GPUs run while every compute stream waits only for its own inputs.
	for(Subdomain* s : locals_) {
		s->halo().pack(field, axis, s->time(), s->placement().face_cells(axis));
		s->device().halo_sources_ready();
	}
	for(Subdomain* s : locals_) s->device().halo_copies_begin();
	for(uint id=0u; id<(uint)locals_.size(); ++id) {
		Subdomain& self = *locals_[id];
		Subdomain& upper = *locals_[grid_.upper_neighbour(id, axis)];
		const index_t bytes = self.placement().face_cells(axis)*cell_bytes;
		gpu::copy_halo(upper.halo().from_lower.buffer(), self.halo().to_upper, 0u, bytes);
		gpu::copy_halo(self.halo().from_upper.buffer(), upper.halo().to_lower, 0u, bytes);
	}
	for(Subdomain* s : locals_) s->device().halo_copies_end();
	for(Subdomain* s : locals_) s->halo().unpack(field, axis, s->time(), s->placement().face_cells(axis));
	for(Subdomain* s : locals_) s->device().halo_buffers_consumed();
	for(uint id=0u; id<(uint)locals_.size(); ++id) {
		// The send buffers of this GPU may still be read by its neighbours.
		const uint upper = grid_.upper_neighbour(id, axis), lower = grid_.lower_neighbour(id, axis);
		locals_[id]->device().wait_for_peer_copies(locals_[upper]->device());
		if(lower!=upper) locals_[id]->device().wait_for_peer_copies(locals_[lower]->device());
	}
}

void HaloExchanger::exchange_ranks(const HaloField field, const uint axis, const uint cell_bytes) {
	Subdomain& self = *locals_.front();
	HaloPort& port = self.halo();
	const index_t face = self.placement().face_cells(axis);
	const index_t bytes = face*cell_bytes;
	port.pack(field, axis, self.time(), face);
	port.to_upper.download_async(0u, bytes);
	port.to_lower.download_async(0u, bytes);
	self.synchronize();
	const int tag = 100+(int)field*8+(int)axis*2;
	culbm::mpi_exchange_halo_pair_bytes(port.to_upper.data(), port.to_lower.data(), (std::uint64_t)bytes,
		(int)grid_.upper_neighbour(owned_, axis), (int)grid_.lower_neighbour(owned_, axis),
		port.from_upper.data(), port.from_lower.data(), tag, tag+1);
	port.from_upper.upload_async(0u, bytes);
	port.from_lower.upload_async(0u, bytes);
	port.unpack(field, axis, self.time(), face);
}

} // namespace culbm::solver
