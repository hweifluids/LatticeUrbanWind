#include "field_table.hpp"

#include "../gpu/vram.hpp"

#include <algorithm>

namespace culbm::solver {

std::vector<CellField> cell_fields(const LatticeStorage& s) {
	// Macroscopic fields are only streamed every step when UPDATE_FIELDS is on.
	const unsigned per_step = s.update_fields ? 1u : 0u;
	// DDFs are kept in 16-bit words; an FP32 DDF occupies two of them.
	const unsigned ddf_words = s.ddf_bytes/2u;
	std::vector<CellField> table;
	table.push_back({FieldId::ddf, "fi", 2u, s.velocity_set*ddf_words, HostCopy::none,
		2u*s.velocity_set*s.ddf_bytes});
	table.push_back({FieldId::density, "rho", 4u, 1u, HostCopy::resident, 4u*per_step});
	table.push_back({FieldId::velocity, "u", 4u, 3u, HostCopy::resident, 12u*per_step});
	// The scalar collision reads the flags of its temperature_set-1 neighbours.
	table.push_back({FieldId::flags, "flags", 1u, 1u, HostCopy::resident,
		1u+(s.temperature ? s.temperature_set-1u : 0u)});
	// F is an input of the Guo forcing that only tests and validations write
	// from the host; its mirror exists once something touches it there.
	if(s.force_field) table.push_back({FieldId::force, "F", 4u, 3u, HostCopy::on_demand, 12u});
	if(s.temperature) {
		table.push_back({FieldId::thermal_ddf, "gi", 2u, s.temperature_set*ddf_words, HostCopy::none,
			2u*s.temperature_set*s.ddf_bytes});
		table.push_back({FieldId::temperature, "T", 4u, 1u, HostCopy::resident, 4u*per_step});
	}
	if(s.surface_ibm) {
		// Force updates only touch the marker-support lists, not every cell.
		table.push_back({FieldId::ibm_force, "surface_ibm_force", 4u, 3u, HostCopy::none, 0u});
		table.push_back({FieldId::ibm_mask, "surface_ibm_mask", 1u, 1u, HostCopy::none, 1u});
	}
	return table;
}

CellCost cell_cost(const LatticeStorage& storage) {
	CellCost cost;
	for(const CellField& field : cell_fields(storage)) {
		cost.device += field.bytes();
		if(field.host==HostCopy::resident) cost.host += field.bytes();
		cost.traffic += field.traffic;
	}
	return cost;
}

unsigned halo_cell_bytes(const LatticeStorage& s) {
	constexpr unsigned macro_bytes = 17u; // rho, u and flags of one face cell
	const unsigned momentum = s.face_ddfs*s.ddf_bytes;
	const unsigned coupled = s.temperature ? (s.face_ddfs+s.temperature_face_ddfs)*s.ddf_bytes : 0u;
	return std::max({momentum, macro_bytes, coupled});
}

DomainFootprint domain_footprint(const LatticeStorage& storage,
	const unsigned Nx, const unsigned Ny, const unsigned Nz,
	const unsigned Dx, const unsigned Dy, const unsigned Dz) {
	// A split axis carries one halo layer on each side.
	const std::uint64_t ex = Nx/Dx+(Dx>1u ? 2u : 0u);
	const std::uint64_t ey = Ny/Dy+(Dy>1u ? 2u : 0u);
	const std::uint64_t ez = Nz/Dz+(Dz>1u ? 2u : 0u);
	gpu::AllocationTally tally;
	for(const CellField& field : cell_fields(storage)) tally.add(ex*ey*ez, field.bytes());
	std::uint64_t face = 0u; // largest face of a split axis
	if(Dx>1u) face = std::max(face, ey*ez);
	if(Dy>1u) face = std::max(face, ez*ex);
	if(Dz>1u) face = std::max(face, ex*ey);
	// One send and one receive buffer per side, shared by all split axes.
	tally.add(face, halo_cell_bytes(storage), 4u);
	return DomainFootprint{tally.mib(), tally.bytes(), tally.valid()};
}

} // namespace culbm::solver
