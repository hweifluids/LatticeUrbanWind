#pragma once

// Immersed-boundary forcing of a solver (SurfaceIBM): prepare_step() runs
// before every collision. The solver owns the attached forcing and destroys
// it before its subdomains.

namespace culbm::solver {

class ImmersedForcing {
public:
	virtual ~ImmersedForcing() = default;
	virtual void prepare_step() = 0;
};

} // namespace culbm::solver
