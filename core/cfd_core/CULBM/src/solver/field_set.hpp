#pragma once

// The device fields of one subdomain, allocated from the descriptor table in
// field_table.hpp, plus the small auxiliary buffers some features add.

#include "decomposition.hpp"
#include "field_table.hpp"
#include "lattice_config.hpp"
#include "../gpu/gpu.hpp"

namespace culbm::solver {

// Storage summary of a feature set, the input of the descriptor table.
LatticeStorage storage_of(const FeatureSet& features);
// Storage summary of the current process-wide lattice selections.
LatticeStorage current_storage();

class FieldSet {
public:
	FieldSet() = default;
	// `columns` is the number of (x, y) columns of the subdomain, the size of
	// the near-wall SGS column map.
	FieldSet(const gpu::Device& device, const FeatureSet& features, index_t cells, index_t columns);

	index_t cells() const { return cells_; }

	// Cell fields (see FieldId). Fields a configuration does not use stay empty.
	gpu::Buffer<ushort> ddf;
	gpu::MirroredBuffer<float> density;
	gpu::MirroredBuffer<float> velocity;
	gpu::MirroredBuffer<uchar> flags;
	gpu::MirroredBuffer<float> force; // host mirror attached on first host use
	gpu::Buffer<ushort> thermal_ddf;
	gpu::MirroredBuffer<float> temperature;
	gpu::Buffer<float> ibm_force;
	gpu::Buffer<uchar> ibm_mask;

	// Subgrid diagnostics: sums of nu_t and rho*nu_t, their maxima and the
	// sampled cell count, reduced on the device (pinned, read asynchronously).
	gpu::MirroredBuffer<double> sgs_sum;
	gpu::MirroredBuffer<float> sgs_max;
	gpu::MirroredBuffer<ulong> sgs_count;
	// Near-wall SGS layer: per column the global z of the first fluid row above
	// the solid base, 0xFFFF where the column has no wall at z=0.
	gpu::MirroredBuffer<ushort> wall_columns;

private:
	void allocate(const gpu::Device& device, const CellField& field);

	index_t cells_ = 0u;
};

} // namespace culbm::solver

// Per-cell storage of the current runtime configuration (velocity set, DDF
// format, temperature lattice, SurfaceIBM); plans VRAM and bandwidth reports.
culbm::solver::LatticeStorage lattice_storage();
// Booked device memory (MiB) of one subdomain of an Nx x Ny x Nz lattice split
// into Dx x Dy x Dz subdomains, for the current runtime configuration.
std::uint64_t lattice_vram_mib(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz);
// Temporary device memory (MiB) the voxelization of `triangle_count`
// triangles needs on one GPU.
std::uint64_t voxelization_transient_vram_mb_per_device(const std::uint64_t triangle_count);
