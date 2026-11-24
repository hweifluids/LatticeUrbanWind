#pragma once

// Entry of the case setup: reads the deck named on the command line (or asked
// for on the console), derives the case in stages (settings, probes, domain,
// inflow, units and transport, damping layers, geometry, terrain) and runs
// its cases: one case for *.luw, one per inflow and angle for *.luwdg, one
// per angle for *.luwpf. With --dump-config it stops before allocating a
// solver (case/config_dump.hpp).

namespace culbm::run {

void run_deck_cases();

} // namespace culbm::run
