#pragma once

// Consistency checks of a case: before GPUs are opened (decomposition),
// once the GPUs are known (memory, coefficients, unsupported features) and
// before the first initialization (cell types in the host flags). Fatal
// problems end the run with a message, doubtful settings print a warning.

#include "decomposition.hpp"
#include "field_view.hpp"
#include "lattice_config.hpp"
#include "../gpu/gpu.hpp"

#include <vector>

namespace culbm::solver {

// The coefficients a solver is constructed with.
struct CaseCoefficients {
	float nu = 1.0f/6.0f;
	float fx = 0.0f, fy = 0.0f, fz = 0.0f;
	float sigma = 0.0f;        // surface tension (unsupported)
	float alpha = 0.0f, beta = 0.0f;
	uint particles = 0u;       // Lagrangian particles of the lattice (unsupported)
	float particles_rho = 0.0f;

	CaseCoefficients& viscosity(const float kinematic) { nu = kinematic; return *this; }
	CaseCoefficients& volume_force(const float x, const float y, const float z) { fx = x; fy = y; fz = z; return *this; }
	CaseCoefficients& thermal(const float diffusivity, const float expansion) { alpha = diffusivity; beta = expansion; return *this; }
};

class CaseValidator {
public:
	// Subdomain counts and lattice size, before the lattice is fitted to them.
	static void check_decomposition(uint nx, uint ny, uint nz, uint dx, uint dy, uint dz);
	// Device memory, coefficients and compile-time requirements.
	static void check_construction(const Decomposition& grid, const std::vector<gpu::DeviceProperties>& gpus,
		const CaseCoefficients& coefficients);
	// SurfaceIBM storage needs an attached marker set before initialization.
	static void check_immersed_boundary(bool storage, bool attached);
	// Cell types the host flags use.
	static void check_cell_types(FieldView<uchar>& flags, FieldView<float>& velocity);
};

} // namespace culbm::solver
