#pragma once

// Exchange of the one-cell halo layers between subdomains. Every subdomain
// owns a HaloPort: four face buffers (send towards the upper and the lower
// neighbour, receive from them) and a pack and an unpack kernel per exchanged
// field. Local GPUs copy the packed faces device to device (peer access or
// pinned host staging, see gpu::copy_halo); MPI ranks swap them through their
// pinned host mirrors.

#include "decomposition.hpp"
#include "field_set.hpp"
#include "lattice_config.hpp"
#include "../gpu/gpu.hpp"

#include <vector>

namespace culbm::solver {

class Subdomain;

// Fields that cross subdomain faces; the value also selects the MPI tag group.
enum class HaloField : unsigned {
	macros = 1u,      // rho, u and flags
	flags = 2u,       // flags only
	temperature = 5u, // T
	populations = 6u  // momentum and thermal DDFs
};
constexpr unsigned halo_field_count = 4u;

class HaloPort {
public:
	HaloPort() = default;
	// `face_capacity` cells of up to `cell_bytes` bytes per face buffer.
	HaloPort(const gpu::Device& device, FieldSet& fields, index_t face_capacity, uint cell_bytes, ulong t,
		bool temperature_enabled);
	HaloPort(HaloPort&&) = default;
	HaloPort& operator=(HaloPort&&) = default;

	bool open() const { return to_upper.size()>0u; }
	// Fill the send buffers from the faces normal to `axis` / scatter the
	// receive buffers into the halo layers.
	void pack(HaloField field, uint axis, ulong t, index_t face_cells);
	void unpack(HaloField field, uint axis, ulong t, index_t face_cells);

	gpu::MirroredBuffer<char> to_upper, to_lower, from_upper, from_lower;

private:
	gpu::Launch pack_[halo_field_count], unpack_[halo_field_count];
};

// Bytes per face cell that `field` packs.
uint halo_cell_bytes(HaloField field, const FeatureSet& features);

class HaloExchanger {
public:
	HaloExchanger() = default;
	// `locals` are the process-local subdomains; in MPI mode the single one is
	// subdomain `owned` of the decomposition.
	HaloExchanger(const Decomposition& grid, std::vector<Subdomain*> locals, bool distributed,
		unsigned owned, const FeatureSet& features);

	// Update the halo layers of `field` on every subdomain, axis by axis.
	void exchange(HaloField field);

private:
	void exchange_devices(HaloField field, uint axis, uint cell_bytes);
	void exchange_ranks(HaloField field, uint axis, uint cell_bytes);

	Decomposition grid_;
	std::vector<Subdomain*> locals_;
	bool distributed_ = false;
	unsigned owned_ = 0u;
	FeatureSet features_;
};

} // namespace culbm::solver
