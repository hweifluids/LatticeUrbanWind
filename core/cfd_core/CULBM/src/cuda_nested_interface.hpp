#pragma once

// Host-side rules for a single horizontal 2:1 CUDA nesting interface.
//
// Geometry is classified from one common finest logical voxel field. A coarse
// cell is solid when at least half of its finest logical voxels are solid;
// ties deliberately resolve to solid. The interface itself uses the same
// conservative rule over its four fine face slots: two or more blocked slots
// close the entire face.
//
// The coupling contract is deliberately simple: an open face is always
// restricted with an arithmetic mean of its four fixed slots. A closed fine
// slot contributes a caller-supplied canonical solid macro (normally rho=1,
// u=0, T=the local wall temperature), rather than being dropped and
// re-normalized. Prolongation copies the coarse macro unchanged only into
// open slots. These helpers do not implement flux correction or higher-order
// interpolation.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace culbm {

constexpr std::uint8_t nested_interface_slot_count = 4u;
constexpr std::uint8_t nested_interface_solid_flag = 0x01u;

struct NestedInterfaceMacro {
	double rho = 1.0;
	std::array<double, 3u> u = { 0.0, 0.0, 0.0 };
	double temperature = 1.0;
};

// `solid_voxels` and `total_voxels` refer directly to the common finest
// logical voxel field. Do not feed this function recursively coarsened flags:
// recursive majority votes can differ from one direct vote over the finest
// voxels.
inline bool nested_cell_is_solid_from_finest_count(
	const std::uint64_t solid_voxels,
	const std::uint64_t total_voxels) noexcept {
	if(total_voxels == 0u || solid_voxels > total_voxels) return false;
	// ceil(total/2), without a potentially overflowing 2*solid comparison.
	const std::uint64_t solid_threshold =
		total_voxels/2u + total_voxels%2u;
	return solid_voxels >= solid_threshold;
}

inline NestedInterfaceMacro nested_canonical_solid_macro(
	const double wall_temperature) noexcept {
	NestedInterfaceMacro macro;
	macro.rho = 1.0;
	macro.u = { 0.0, 0.0, 0.0 };
	macro.temperature = wall_temperature;
	return macro;
}

inline bool nested_interface_macro_is_valid(
	const NestedInterfaceMacro& macro) noexcept {
	return std::isfinite(macro.rho) && macro.rho > 0.0 &&
		std::isfinite(macro.u[0]) && std::isfinite(macro.u[1]) &&
		std::isfinite(macro.u[2]) && std::isfinite(macro.temperature);
}

struct NestedInterfaceAperture {
	std::uint8_t open_mask = 0u;
	std::uint8_t raw_open_count = 0u;
	bool coarse_cell_solid = false;
	bool closed_by_tie_solid = false;

	bool is_open() const noexcept { return open_mask != 0u; }
	bool slot_is_open(const std::uint8_t slot) const noexcept {
		return slot < nested_interface_slot_count &&
			(open_mask & (std::uint8_t)(1u << slot)) != 0u;
	}
	std::uint8_t open_count() const noexcept {
		std::uint8_t count = 0u;
		for(std::uint8_t slot = 0u;
			slot < nested_interface_slot_count; ++slot) {
			if(slot_is_open(slot)) ++count;
		}
		return count;
	}
};

// `fine_child_is_fluid` must be based on the four fine, boundary-adjacent
// interior cells after their solid/covered status has been decided. It must
// not use ghost or TYPE_E cells. One shared aperture object is used for both
// directions, so a closed slot cannot leak in only one direction.
inline NestedInterfaceAperture nested_make_interface_aperture(
	const bool coarse_cell_solid,
	const std::array<bool, nested_interface_slot_count>& fine_child_is_fluid)
	noexcept {
	NestedInterfaceAperture aperture;
	aperture.coarse_cell_solid = coarse_cell_solid;
	for(std::uint8_t slot = 0u;
		slot < nested_interface_slot_count; ++slot) {
		if(fine_child_is_fluid[slot]) {
			aperture.open_mask |= (std::uint8_t)(1u << slot);
			++aperture.raw_open_count;
		}
	}

	// A coarse solid always blocks the face. Otherwise, 2/4 blocked slots is
	// the face-level tie case and is intentionally closed as solid.
	if(coarse_cell_solid || aperture.raw_open_count <= 2u) {
		aperture.closed_by_tie_solid = !coarse_cell_solid;
		aperture.open_mask = 0u;
	}
	return aperture;
}

struct NestedInterfaceRestriction {
	bool valid = false;
	NestedInterfaceMacro coarse_macro{};
};

// Restrict through an open aperture using the required strict four-slot mean.
// Each closed slot uses the corresponding `solid_slots[slot]`, allowing a
// caller to supply a spatially varying wall temperature. No open-only
// renormalization, flux weighting, or velocity amplification occurs here.
inline NestedInterfaceRestriction nested_restrict_mean4(
	const std::array<NestedInterfaceMacro, nested_interface_slot_count>&
		fine_slots,
	const NestedInterfaceAperture& aperture,
	const std::array<NestedInterfaceMacro, nested_interface_slot_count>&
		solid_slots) noexcept {
	NestedInterfaceRestriction result;
	if(!aperture.is_open()) return result;

	NestedInterfaceMacro sum{};
	sum.rho = 0.0;
	sum.u = { 0.0, 0.0, 0.0 };
	sum.temperature = 0.0;
	for(std::uint8_t slot = 0u;
		slot < nested_interface_slot_count; ++slot) {
		const NestedInterfaceMacro& source = aperture.slot_is_open(slot)
			? fine_slots[slot]
			: solid_slots[slot];
		if(!nested_interface_macro_is_valid(source)) return result;
		sum.rho += source.rho;
		sum.u[0] += source.u[0];
		sum.u[1] += source.u[1];
		sum.u[2] += source.u[2];
		sum.temperature += source.temperature;
	}

	result.coarse_macro.rho = 0.25*sum.rho;
	result.coarse_macro.u = {
		0.25*sum.u[0], 0.25*sum.u[1], 0.25*sum.u[2]
	};
	result.coarse_macro.temperature = 0.25*sum.temperature;
	result.valid = nested_interface_macro_is_valid(result.coarse_macro);
	return result;
}

inline NestedInterfaceRestriction nested_restrict_mean4(
	const std::array<NestedInterfaceMacro, nested_interface_slot_count>&
		fine_slots,
	const NestedInterfaceAperture& aperture,
	const NestedInterfaceMacro& solid_slot) noexcept {
	std::array<NestedInterfaceMacro, nested_interface_slot_count> solid_slots;
	solid_slots.fill(solid_slot);
	return nested_restrict_mean4(fine_slots, aperture, solid_slots);
}

struct NestedInterfaceProlongation {
	bool valid = false;
	std::uint8_t write_mask = 0u;
	std::array<NestedInterfaceMacro, nested_interface_slot_count> fine_macros{};

	bool writes_slot(const std::uint8_t slot) const noexcept {
		return slot < nested_interface_slot_count &&
			(write_mask & (std::uint8_t)(1u << slot)) != 0u;
	}
};

// Coarse-to-fine transfer is a constant copy into open fluid slots only.
// Closed slots are deliberately not initialized here: their owner must retain
// the solid/bounce-back state and must never be switched to TYPE_E.
inline NestedInterfaceProlongation nested_prolongate_constant(
	const NestedInterfaceMacro& coarse_macro,
	const NestedInterfaceAperture& aperture) noexcept {
	NestedInterfaceProlongation result;
	if(!aperture.is_open() || !nested_interface_macro_is_valid(coarse_macro)) {
		return result;
	}
	for(std::uint8_t slot = 0u;
		slot < nested_interface_slot_count; ++slot) {
		if(!aperture.slot_is_open(slot)) continue;
		result.fine_macros[slot] = coarse_macro;
		result.write_mask |= (std::uint8_t)(1u << slot);
	}
	result.valid = result.write_mask != 0u;
	return result;
}

// Flat arrays used by the scheduler bridge. `flags` uses the caller-supplied
// `solid_flag_mask` (TYPE_S is 0x01 in CULBM); flags are intentionally not
// rewritten by restriction/prolongation.
struct NestedInterfacePlane {
	std::uint32_t nx = 0u;
	std::uint32_t ny = 0u;
	std::vector<NestedInterfaceMacro> macros;
	std::vector<std::uint8_t> flags;

	std::size_t cell_count() const noexcept {
		return (std::size_t)nx*(std::size_t)ny;
	}
	bool has_storage() const noexcept {
		return nx > 0u && ny > 0u && macros.size() == cell_count() &&
			flags.size() == cell_count();
	}
	std::size_t index(const std::uint32_t x, const std::uint32_t y) const
		noexcept {
		return (std::size_t)x + (std::size_t)y*(std::size_t)nx;
	}
};

inline bool nested_interface_plane_is_fluid(
	const NestedInterfacePlane& plane,
	const std::uint32_t x,
	const std::uint32_t y,
	const std::uint8_t solid_flag_mask = nested_interface_solid_flag) noexcept {
	return plane.has_storage() && x < plane.nx && y < plane.ny &&
		(plane.flags[plane.index(x, y)] & solid_flag_mask) == 0u;
}

inline bool nested_interface_planes_are_2_to_1(
	const NestedInterfacePlane& coarse,
	const NestedInterfacePlane& fine) noexcept {
	if(!coarse.has_storage() || !fine.has_storage()) return false;
	if(coarse.nx > std::numeric_limits<std::uint32_t>::max()/2u ||
		coarse.ny > std::numeric_limits<std::uint32_t>::max()/2u) return false;
	return fine.nx == 2u*coarse.nx && fine.ny == 2u*coarse.ny;
}

// Child slot order is stable across GPUs/ranks:
//   0=(2x,2y), 1=(2x+1,2y), 2=(2x,2y+1), 3=(2x+1,2y+1).
inline bool nested_build_plane_apertures(
	const NestedInterfacePlane& coarse,
	const NestedInterfacePlane& fine,
	std::vector<NestedInterfaceAperture>& apertures,
	const std::uint8_t solid_flag_mask = nested_interface_solid_flag) {
	if(!nested_interface_planes_are_2_to_1(coarse, fine)) return false;
	apertures.resize(coarse.cell_count());
	for(std::uint32_t y = 0u; y < coarse.ny; ++y) {
		for(std::uint32_t x = 0u; x < coarse.nx; ++x) {
			const bool coarse_solid = !nested_interface_plane_is_fluid(
				coarse, x, y, solid_flag_mask);
			const std::uint32_t fx = 2u*x;
			const std::uint32_t fy = 2u*y;
			const std::array<bool, nested_interface_slot_count> fine_fluid = {{
				nested_interface_plane_is_fluid(fine, fx, fy, solid_flag_mask),
				nested_interface_plane_is_fluid(fine, fx+1u, fy, solid_flag_mask),
				nested_interface_plane_is_fluid(fine, fx, fy+1u, solid_flag_mask),
				nested_interface_plane_is_fluid(fine, fx+1u, fy+1u, solid_flag_mask)
			}};
			apertures[coarse.index(x, y)] = nested_make_interface_aperture(
				coarse_solid, fine_fluid);
		}
	}
	return true;
}

struct NestedInterfacePlaneTransfer {
	bool valid = false;
	std::size_t transferred_parent_cells = 0u;
};

inline NestedInterfacePlaneTransfer nested_restrict_plane_mean4(
	const NestedInterfacePlane& fine,
	const std::vector<NestedInterfaceAperture>& apertures,
	const NestedInterfaceMacro& solid_slot,
	NestedInterfacePlane& coarse) {
	NestedInterfacePlaneTransfer result;
	if(!nested_interface_planes_are_2_to_1(coarse, fine) ||
		apertures.size() != coarse.cell_count() ||
		!nested_interface_macro_is_valid(solid_slot)) return result;
	for(std::uint32_t y = 0u; y < coarse.ny; ++y) {
		for(std::uint32_t x = 0u; x < coarse.nx; ++x) {
			const std::size_t parent = coarse.index(x, y);
			const NestedInterfaceAperture& aperture = apertures[parent];
			if(!aperture.is_open()) continue;
			const std::uint32_t fx = 2u*x;
			const std::uint32_t fy = 2u*y;
			const std::array<NestedInterfaceMacro, nested_interface_slot_count>
				fine_slots = {{
					fine.macros[fine.index(fx, fy)],
					fine.macros[fine.index(fx+1u, fy)],
					fine.macros[fine.index(fx, fy+1u)],
					fine.macros[fine.index(fx+1u, fy+1u)]
				}};
			const NestedInterfaceRestriction restricted = nested_restrict_mean4(
				fine_slots, aperture, solid_slot);
			if(!restricted.valid) return result;
			coarse.macros[parent] = restricted.coarse_macro;
			++result.transferred_parent_cells;
		}
	}
	result.valid = true;
	return result;
}

inline NestedInterfacePlaneTransfer nested_prolongate_plane_constant(
	const NestedInterfacePlane& coarse,
	const std::vector<NestedInterfaceAperture>& apertures,
	NestedInterfacePlane& fine) {
	NestedInterfacePlaneTransfer result;
	if(!nested_interface_planes_are_2_to_1(coarse, fine) ||
		apertures.size() != coarse.cell_count()) return result;
	for(std::uint32_t y = 0u; y < coarse.ny; ++y) {
		for(std::uint32_t x = 0u; x < coarse.nx; ++x) {
			const std::size_t parent = coarse.index(x, y);
			const NestedInterfaceProlongation prolongated =
				nested_prolongate_constant(coarse.macros[parent], apertures[parent]);
			if(!apertures[parent].is_open()) continue;
			if(!prolongated.valid) return result;
			const std::uint32_t fx = 2u*x;
			const std::uint32_t fy = 2u*y;
			const std::array<std::size_t, nested_interface_slot_count> children = {{
				fine.index(fx, fy), fine.index(fx+1u, fy),
				fine.index(fx, fy+1u), fine.index(fx+1u, fy+1u)
			}};
			for(std::uint8_t slot = 0u;
				slot < nested_interface_slot_count; ++slot) {
				if(!prolongated.writes_slot(slot)) continue;
				fine.macros[children[slot]] = prolongated.fine_macros[slot];
			}
			++result.transferred_parent_cells;
		}
	}
	result.valid = true;
	return result;
}

} // namespace culbm
