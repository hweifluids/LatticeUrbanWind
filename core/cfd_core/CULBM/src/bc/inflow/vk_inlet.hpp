#pragma once

// Von Karman synthetic turbulence inlet: a random-Fourier-mode velocity
// perturbation on the inflow faces, evaluated on the GPU before every step.

#include "vk_inlet_settings.hpp"
#include "../../solver/solver.hpp"

#include <memory>

namespace culbm::bc {

class VonKarmanInletUpdater {
public:
    explicit VonKarmanInletUpdater(const VkInletRuntimeConfig& cfg);
    ~VonKarmanInletUpdater();
    VonKarmanInletUpdater(const VonKarmanInletUpdater&) = delete;
    VonKarmanInletUpdater& operator=(const VonKarmanInletUpdater&) = delete;

    // Collects the inflow faces from the boundary flags and velocities of
    // `lbm` and builds the GPU runtime; false when nothing is active.
    bool initialize(culbm::solver::Solver& solver);
    bool active() const;
    // Applies the perturbation at lattice step t (or a fractional coarse
    // lattice time for nested slabs).
    bool update(culbm::solver::Solver& solver, ulong t);
    bool update_at_lattice_time(culbm::solver::Solver& solver, double lattice_time);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace culbm::bc
