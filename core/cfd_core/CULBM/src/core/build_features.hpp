#pragma once

// Compile-time feature switches of the CULBM build. The momentum velocity set (D3Q19/D3Q27) and
// the DDF storage format (FP16C/FP16S/FP32) are runtime settings selected from the deck, see
// set_momentum_lattice() and set_ddf_storage() in solver/lattice_config.hpp.

#ifdef CULBM_USE_TRT
#define TRT // two-relaxation-time collision (opt-in)
#else
#define SRT // single-relaxation-time collision (default)
#endif

#ifndef CULBM_DISABLE_UPDATE_FIELDS
#define UPDATE_FIELDS // store rho and u every time step
#endif
#define VOLUME_FORCE // global body force with Guo forcing
#define FORCE_FIELD // per-cell force field (SurfaceIBM, buoyancy, sponge and nudging forces)
#define EQUILIBRIUM_BOUNDARIES // cells with fixed velocity/density (CELL_EQUILIBRIUM)
#define TEMPERATURE // runtime-selectable D3Q7/D3Q19 temperature transport
#ifndef CULBM_DISABLE_SUBGRID
#define SUBGRID // Smagorinsky-Lilly subgrid model
#endif

#include "cell_flags.hpp"
