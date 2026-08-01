#include "cuda_nested_interface.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

bool nearly_equal(const double a, const double b) {
	return std::fabs(a-b) < 1.0e-12;
}

bool expect(const bool value, const char* const message) {
	if(value) return true;
	std::cerr << "FAILED: " << message << '\n';
	return false;
}

} // namespace

int main() {
	using culbm::NestedInterfaceMacro;
	using culbm::NestedInterfacePlane;
	using culbm::nested_build_plane_apertures;
	using culbm::nested_cell_is_solid_from_finest_count;
	using culbm::nested_canonical_solid_macro;
	using culbm::nested_make_interface_aperture;
	using culbm::nested_prolongate_constant;
	using culbm::nested_prolongate_plane_constant;
	using culbm::nested_restrict_mean4;
	using culbm::nested_restrict_plane_mean4;

	// Direct finest-voxel majority: ties intentionally go to solid.
	if(!expect(!nested_cell_is_solid_from_finest_count(3u, 8u),
		"3/8 finest solid voxels must remain fluid")) return 1;
	if(!expect(nested_cell_is_solid_from_finest_count(4u, 8u),
		"4/8 finest solid voxels must resolve to solid")) return 1;
	if(!expect(!nested_cell_is_solid_from_finest_count(2u, 5u),
		"2/5 finest solid voxels must remain fluid")) return 1;
	if(!expect(nested_cell_is_solid_from_finest_count(3u, 5u),
		"3/5 finest solid voxels must resolve to solid")) return 1;

	const auto all_open = nested_make_interface_aperture(
		false, { true, true, true, true });
	if(!expect(all_open.is_open() && all_open.open_count() == 4u &&
		all_open.open_mask == 0x0fu, "all four fluid slots must open")) return 1;

	const auto three_open = nested_make_interface_aperture(
		false, { true, true, true, false });
	if(!expect(three_open.is_open() && three_open.raw_open_count == 3u &&
		three_open.open_mask == 0x07u,
		"one blocked child keeps the other three slots open")) return 1;

	const auto tie_closed = nested_make_interface_aperture(
		false, { true, true, false, false });
	if(!expect(!tie_closed.is_open() && tie_closed.closed_by_tie_solid,
		"two open/two blocked slots must close by tie-to-solid")) return 1;

	const auto coarse_solid = nested_make_interface_aperture(
		true, { true, true, true, true });
	if(!expect(!coarse_solid.is_open() && coarse_solid.coarse_cell_solid,
		"a coarse solid cell must close every fine slot")) return 1;

	std::array<NestedInterfaceMacro, 4u> fine_slots = {{
		{ 2.0, { 4.0, 1.0, -1.0 }, 10.0 },
		{ 4.0, { 8.0, 2.0, -2.0 }, 20.0 },
		{ 6.0, { 12.0, 3.0, -3.0 }, 30.0 },
		{ 99.0, { 99.0, 99.0, 99.0 }, 99.0 }
	}};
	const NestedInterfaceMacro solid = nested_canonical_solid_macro(40.0);
	const auto restricted = nested_restrict_mean4(fine_slots, three_open, solid);
	if(!expect(restricted.valid, "three-slot aperture must produce a macro")) return 1;
	if(!expect(nearly_equal(restricted.coarse_macro.rho, 3.25) &&
		nearly_equal(restricted.coarse_macro.u[0], 6.0) &&
		nearly_equal(restricted.coarse_macro.u[1], 1.5) &&
		nearly_equal(restricted.coarse_macro.u[2], -1.5) &&
		nearly_equal(restricted.coarse_macro.temperature, 25.0),
		"restriction must use the strict fixed four-slot mean with canonical solid slot")) return 1;

	const auto prolongated = nested_prolongate_constant(
		restricted.coarse_macro, three_open);
	if(!expect(prolongated.valid && prolongated.write_mask == 0x07u,
		"prolongation must write precisely the open slots")) return 1;
	for(std::uint8_t slot = 0u; slot < 3u; ++slot) {
		if(!expect(prolongated.writes_slot(slot) &&
			nearly_equal(prolongated.fine_macros[slot].rho,
				restricted.coarse_macro.rho) &&
			nearly_equal(prolongated.fine_macros[slot].u[0],
				restricted.coarse_macro.u[0]) &&
			nearly_equal(prolongated.fine_macros[slot].temperature,
				restricted.coarse_macro.temperature),
			"every open slot must receive an unchanged coarse macro")) return 1;
	}
	if(!expect(!prolongated.writes_slot(3u),
		"closed fine slot must never be overwritten as an equilibrium cell")) return 1;

	const auto blocked_restriction = nested_restrict_mean4(fine_slots, tie_closed, solid);
	const auto blocked_prolongation = nested_prolongate_constant(solid, tie_closed);
	if(!expect(!blocked_restriction.valid && !blocked_prolongation.valid &&
		blocked_prolongation.write_mask == 0u,
		"a closed face must perform no cross-level macro transfer")) return 1;

	// Plane helpers use the same slot order and never modify flags. The fourth
	// fine cell is solid, so the strict mean remains the one tested above.
	NestedInterfacePlane coarse;
	coarse.nx = 1u;
	coarse.ny = 1u;
	coarse.macros.resize(1u);
	coarse.flags = { 0u };
	NestedInterfacePlane fine;
	fine.nx = 2u;
	fine.ny = 2u;
	fine.macros = { fine_slots[0], fine_slots[1], fine_slots[2], fine_slots[3] };
	fine.flags = { 0u, 0u, 0u, 1u };
	std::vector<culbm::NestedInterfaceAperture> apertures;
	if(!expect(nested_build_plane_apertures(coarse, fine, apertures) &&
		apertures.size() == 1u && apertures[0].open_mask == 0x07u,
		"plane helper must retain the documented four-slot order")) return 1;
	const auto plane_restriction = nested_restrict_plane_mean4(
		fine, apertures, solid, coarse);
	if(!expect(plane_restriction.valid &&
		nearly_equal(coarse.macros[0].rho, 3.25),
		"plane restriction must use strict mean4")) return 1;
	fine.macros[3] = { 77.0, { 77.0, 77.0, 77.0 }, 77.0 };
	const auto plane_prolongation = nested_prolongate_plane_constant(
		coarse, apertures, fine);
	if(!expect(plane_prolongation.valid &&
		nearly_equal(fine.macros[0].rho, coarse.macros[0].rho) &&
		nearly_equal(fine.macros[1].rho, coarse.macros[0].rho) &&
		nearly_equal(fine.macros[2].rho, coarse.macros[0].rho) &&
		nearly_equal(fine.macros[3].rho, 77.0),
		"plane prolongation must leave closed slots untouched")) return 1;

	std::cout << "Validated nested interface majority, aperture, and strict mean4 rules.\n";
	return 0;
}
