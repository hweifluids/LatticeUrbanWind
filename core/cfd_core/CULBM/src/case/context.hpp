#pragma once

// Everything the case setup knows about one deck before and between solver
// runs: the deck settings, probes, lattice domain, inflow data, thermal
// reference, lattice units and transport, geometry and terrain. The stages of
// run/case_runner fill it in order; each module reads the parts it needs.

#include "domain.hpp"
#include "geometry.hpp"
#include "physics.hpp"
#include "settings.hpp"
#include "terrain.hpp"
#include "../bc/inflow/inflow_data.hpp"
#include "../bc/thermal/thermal_bc.hpp"
#include "../run/probes.hpp"

namespace culbm::cases {

struct CaseContext {
    CaseSettings deck;
    run::ProbePlan probes;
    CaseDomain domain;
    bc::InflowData inflow;
    bc::ThermalReference thermal;
    CasePhysics physics;
    CaseGeometry geometry;
    CaseTerrain terrain;
    // Measured plain steps/s of the previous case, the ETA hint of the next.
    double last_normal_steps_per_second_hint = 0.0;
};

} // namespace culbm::cases
