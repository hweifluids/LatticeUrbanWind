#pragma once

// Bit flags stored in the per-cell flag byte (host and device share the same
// bit assignments).
enum CellFlag : unsigned char {
	CELL_SOLID         = 0x01u, // solid wall / bounce-back boundary
	CELL_EQUILIBRIUM   = 0x02u, // equilibrium boundary (prescribed velocity / density)
	CELL_THERMAL       = 0x04u, // Dirichlet temperature boundary
	CELL_FLUID         = 0x08u, // fluid phase bit
	CELL_INTERFACE     = 0x10u, // interface phase bit
	CELL_GAS           = 0x20u, // gas phase bit
	CELL_AUX_X         = 0x40u, // auxiliary flag bit A (free-slip / voxelization AND)
	CELL_AUX_Y         = 0x80u, // auxiliary flag bit B (voxelization AND)
	CELL_BOUNDARY_MASK = 0x03u, // CELL_SOLID | CELL_EQUILIBRIUM
	CELL_PHASE_MASK    = 0x38u  // CELL_FLUID | CELL_INTERFACE | CELL_GAS
};
