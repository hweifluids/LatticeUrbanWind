#pragma once

// Host side of the triangle-mesh voxelization: choice of the ray axis,
// partition of the triangles over the subdomains, VRAM preflight of the
// temporary triangle buffers per GPU, and the launch of the per-subdomain
// passes (concurrently on different GPUs, in sequence on a shared GPU).

#include "../core/core.hpp"

namespace culbm::solver {

class Solver;

// Ray axis of a voxelization pass: z for solids; otherwise the axis of the
// smallest bounding-box cross-section, or the axis closest to the rotation.
uint voxelization_axis(const Mesh& mesh, uchar flag, const float3& rotational_velocity);

void voxelize_mesh(Solver& solver, const Mesh& mesh, uchar flag, const float3& rotation_center,
	const float3& linear_velocity, const float3& rotational_velocity);

} // namespace culbm::solver
