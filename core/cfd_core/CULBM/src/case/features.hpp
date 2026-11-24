#pragma once

// Solver features selected by the deck: the momentum and temperature
// lattices and the DDF storage, the Smagorinsky subgrid model and the wall
// model. They are process-wide settings that every solver freezes at
// construction; SurfaceIBM and particle advection publish their own
// configurations (surface_ibm_config, particle_advection_config).

namespace culbm::cases {

struct CaseSettings;

// Momentum lattice, DDF storage and temperature lattice; deck errors end the
// process.
void apply_lattice_selection(CaseSettings& deck);
// Subgrid model and wall model (molecular viscosity of the smooth-wall law).
void apply_model_selection(CaseSettings& deck);

} // namespace culbm::cases
