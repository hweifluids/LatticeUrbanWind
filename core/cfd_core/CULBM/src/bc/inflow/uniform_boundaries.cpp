#include "boundaries.hpp"

namespace culbm::bc {

void initialize_uniform_velocity(culbm::solver::Solver& solver, const float3& u) {
    const ulong Ntot = solver.grid().cells();
    for (ulong n = 0ull; n < Ntot; ++n) {
        solver.velocity().x[n] = u.x;
        solver.velocity().y[n] = u.y;
        solver.velocity().z[n] = u.z;
    }
}

void apply_velocity_boundaries(culbm::solver::Solver& solver, const float3& u, const std::string& downstream_bc_runtime,
                               const bool downstream_open_face) {
    const uint Nx = solver.grid().size(0u), Ny = solver.grid().size(1u), Nz = solver.grid().size(2u);
    const bool has_ground = (Nz > 1u);
    const ulong Ntot = solver.grid().cells();
    for (ulong n = 0ull; n < Ntot; ++n) {
        uint x, y, z;
        solver.grid().coordinates(n, x, y, z);
        if (has_ground && z == 0u) {
            solver.flags()[n] = CELL_SOLID;
            continue;
        }
        const bool is_boundary =
            (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || (has_ground && z == Nz - 1u));
        if (is_boundary) {
            solver.flags()[n] = CELL_EQUILIBRIUM;
            if (downstream_open_face &&
                is_downstream_boundary_cell(x, y, z, Nx, Ny, Nz, downstream_bc_runtime)) {
                continue;
            }
            solver.velocity().x[n] = u.x;
            solver.velocity().y[n] = u.y;
            solver.velocity().z[n] = u.z;
        }
    }
}

} // namespace culbm::bc
