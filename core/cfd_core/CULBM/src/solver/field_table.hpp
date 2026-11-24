#pragma once

// Descriptor table of the per-cell fields of a CULBM subdomain. The same table
// allocates the fields (FieldSet), plans device memory (VRAM estimate and GPU
// selection), sizes the halo buffers and reports the per-step traffic, so the
// plan and the allocations cannot drift apart (CULBM.VramAccountingGPU and
// CULBM.SolverLayout check it).

#include <cstdint>
#include <vector>

namespace culbm::solver {

// What one lattice cell stores, given the runtime lattice configuration.
struct LatticeStorage {
	unsigned velocity_set = 19u;         // momentum DDFs per cell
	unsigned ddf_bytes = 2u;             // bytes per stored DDF
	unsigned face_ddfs = 5u;             // momentum DDFs crossing one subdomain face
	bool update_fields = true;           // rho, u (and T) rewritten every step
	bool force_field = true;             // per-cell force F
	bool temperature = true;             // thermal DDFs gi and temperature T
	unsigned temperature_set = 7u;       // thermal DDFs per cell
	unsigned temperature_face_ddfs = 1u; // thermal DDFs crossing one subdomain face
	bool surface_ibm = false;            // SurfaceIBM force and active-cell mask
};

enum class FieldId : unsigned char {
	ddf,         // momentum DDFs (ushort words)
	density,     // rho
	velocity,    // u, three components
	flags,       // cell type bits
	force,       // per-cell body force F, three components
	thermal_ddf, // thermal DDFs gi (ushort words)
	temperature, // T
	ibm_force,   // SurfaceIBM Eulerian force, three components
	ibm_mask     // SurfaceIBM active-cell mask
};

// Where the host copy of a field lives.
enum class HostCopy : unsigned char {
	none,      // device only
	on_demand, // created on first host access (not part of the plan)
	resident   // allocated with the field
};

// One per-cell field; each is a single device allocation of
// cells x components x word_bytes bytes.
struct CellField {
	FieldId id;
	const char* name;
	unsigned word_bytes;  // bytes of one stored element
	unsigned components;  // elements per cell
	HostCopy host;
	unsigned traffic;     // device bytes moved per cell and time step
	unsigned bytes() const { return word_bytes*components; }
};

// Active fields in allocation order.
std::vector<CellField> cell_fields(const LatticeStorage& storage);

// Sums over cell_fields(); host counts resident copies only.
struct CellCost {
	unsigned device = 0u;
	unsigned host = 0u;
	unsigned traffic = 0u;
};
CellCost cell_cost(const LatticeStorage& storage);

// Bytes per face cell of the largest halo exchange; sizes the four halo buffers.
unsigned halo_cell_bytes(const LatticeStorage& storage);

// Device memory of one subdomain of an Nx x Ny x Nz lattice cut into
// Dx x Dy x Dz subdomains (halo layers and halo buffers included): booked MiB
// and exact bytes.
struct DomainFootprint {
	std::uint64_t mib = 0u;
	std::uint64_t bytes = 0u;
	bool valid = true;
};
DomainFootprint domain_footprint(const LatticeStorage& storage,
	unsigned Nx, unsigned Ny, unsigned Nz, unsigned Dx, unsigned Dy, unsigned Dz);

} // namespace culbm::solver
