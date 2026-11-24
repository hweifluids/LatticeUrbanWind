#pragma once

// Which GPUs a run uses and which lattice domain each GPU computes.

#include "device.hpp"

#include <vector>

namespace culbm::gpu {

// GPUs for `domains` process-local lattice domains, one per domain. The fastest
// model with at least `domains` installed units wins (first units in driver
// order); otherwise the fastest distinct GPUs are combined. Aborts when fewer
// GPUs than domains are visible.
std::vector<DeviceProperties> choose_devices(uint domains);

// The GPU an MPI rank drives (one Cartesian domain per rank).
DeviceProperties rank_device();

// Reorder `devices` (index = domain) so that neighbouring domains of the
// Dx x Dy x Dz decomposition of an Nx x Ny x Nz lattice share direct peer
// links, weighted by halo face size. Only strictly better layouts are taken.
// CULBM_DISABLE_P2P or CULBM_P2P_TOPOLOGY=0 keep the given order.
void arrange_for_peer_halos(std::vector<DeviceProperties>& devices,
	uint Nx, uint Ny, uint Nz, uint Dx, uint Dy, uint Dz);

} // namespace culbm::gpu
