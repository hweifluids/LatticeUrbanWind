#pragma once

// Cartesian decomposition of the global lattice into subdomains: placement of
// each subdomain (extent with halo layers, global origin), the mapping of a
// global cell to its owner, and the halo faces exchanged between neighbours.
// Pure index arithmetic, shared by the solver and its unit test.

#include "../cuda_topology.hpp"
#include "../core/scalar.hpp"
#include "../core/vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace culbm::solver {

using index_t = std::uint64_t;

// One subdomain of a Decomposition.
struct Placement {
	unsigned id = 0u;            // subdomain number, id = x+(y+z*Dy)*Dx
	unsigned cell[3] = {0u, 0u, 0u}; // position in the subdomain grid
	unsigned extent[3] = {1u, 1u, 1u}; // local cells per axis, halo layers included
	int origin[3] = {0, 0, 0};   // global coordinate of local cell 0 (may be -1)
	unsigned halo[3] = {0u, 0u, 0u}; // halo layers on each side of an axis (0 or 1)

	index_t cells() const { return (index_t)extent[0]*extent[1]*extent[2]; }
	// Cells of the local plane normal to `axis` (x=0, y=1, z=2).
	index_t face_cells(const unsigned axis) const {
		return (index_t)extent[(axis+1u)%3u]*extent[(axis+2u)%3u];
	}
	index_t local_index(const unsigned x, const unsigned y, const unsigned z) const {
		return (index_t)x+((index_t)y+(index_t)z*extent[1])*extent[0];
	}
};

class Decomposition {
public:
	Decomposition() = default;
	// Cell counts must be multiples of the subdomain counts (see fitted()).
	Decomposition(const unsigned nx, const unsigned ny, const unsigned nz,
		const unsigned dx, const unsigned dy, const unsigned dz)
		: size_{nx, ny, nz}, split_{dx, dy, dz} {}

	// The largest lattice not above nx x ny x nz that the split divides evenly.
	static Decomposition fitted(const unsigned nx, const unsigned ny, const unsigned nz,
		const unsigned dx, const unsigned dy, const unsigned dz) {
		const auto fit = [](const unsigned n, const unsigned d) { return d==0u ? n : n-n%d; };
		return Decomposition(fit(nx, dx), fit(ny, dy), fit(nz, dz), dx, dy, dz);
	}

	unsigned size(const unsigned axis) const { return size_[axis]; }
	// Global cell index n = x+(y+z*Ny)*Nx and its coordinates.
	index_t index(const unsigned x, const unsigned y, const unsigned z) const {
		return (index_t)x+(index_t)size_[0]*((index_t)y+(index_t)size_[1]*(index_t)z);
	}
	void coordinates(const index_t n, unsigned& x, unsigned& y, unsigned& z) const {
		const index_t row = (index_t)size_[0], layer = row*(index_t)size_[1];
		x = (unsigned)(n%row);
		y = (unsigned)((n%layer)/row);
		z = (unsigned)(n/layer);
	}
	// Cell that holds the periodic image of position p, in a box of the
	// lattice size centred on the origin.
	void coordinates(const core::float3& p, unsigned& x, unsigned& y, unsigned& z) const {
		const auto wrap = [](const float c, const unsigned n) {
			const float half = 0.5f*(float)n;
			const float image = core::sign(c)*(std::fmod(std::fabs(c)+half, (float)n)-half);
			return (unsigned)(image+1.5f*(float)n)%n;
		};
		x = wrap(p.x, size_[0]);
		y = wrap(p.y, size_[1]);
		z = wrap(p.z, size_[2]);
	}
	// Centre of cell (x, y, z) in a box of the lattice size centred on the origin.
	core::float3 position(const unsigned x, const unsigned y, const unsigned z) const {
		return core::float3((float)x-0.5f*(float)size_[0]+0.5f, (float)y-0.5f*(float)size_[1]+0.5f,
			(float)z-0.5f*(float)size_[2]+0.5f);
	}
	unsigned split(const unsigned axis) const { return split_[axis]; }
	unsigned count() const { return split_[0]*split_[1]*split_[2]; }
	index_t cells() const { return (index_t)size_[0]*size_[1]*size_[2]; }
	bool split_axis(const unsigned axis) const { return split_[axis]>1u; }
	bool any_split() const { return count()>1u; }
	// Interior cells of every subdomain along `axis`.
	unsigned inner(const unsigned axis) const { return size_[axis]/split_[axis]; }
	unsigned halo(const unsigned axis) const { return split_axis(axis) ? 1u : 0u; }

	Placement place(const unsigned id) const {
		Placement p;
		p.id = id;
		const unsigned layer = split_[0]*split_[1];
		p.cell[0] = (id%layer)%split_[0];
		p.cell[1] = (id%layer)/split_[0];
		p.cell[2] = id/layer;
		for(unsigned a=0u; a<3u; ++a) {
			p.halo[a] = halo(a);
			p.extent[a] = inner(a)+2u*p.halo[a];
			p.origin[a] = (int)(p.cell[a]*inner(a))-(int)p.halo[a];
		}
		return p;
	}

	// Subdomain that owns global cell (x, y, z) and the cell's local index there.
	struct Owner {
		unsigned id;
		index_t local;
	};
	Owner owner(const unsigned x, const unsigned y, const unsigned z) const {
		const unsigned ix = x/inner(0u), iy = y/inner(1u), iz = z/inner(2u);
		const index_t ex = inner(0u)+2u*halo(0u), ey = inner(1u)+2u*halo(1u);
		const index_t lx = x-ix*inner(0u)+halo(0u);
		const index_t ly = y-iy*inner(1u)+halo(1u);
		const index_t lz = z-iz*inner(2u)+halo(2u);
		return Owner{ix+(iy+iz*split_[1])*split_[0], lx+(ly+lz*ey)*ex};
	}

	// Neighbouring subdomain across the upper / lower face of `axis` (periodic).
	unsigned upper_neighbour(const unsigned id, const unsigned axis) const {
		return positive_domain_neighbor(id, split_[0], split_[1], split_[2], axis);
	}
	unsigned lower_neighbour(const unsigned id, const unsigned axis) const {
		return negative_domain_neighbor(id, split_[0], split_[1], split_[2], axis);
	}

	// Cells of the largest exchanged face of a subdomain; sizes its halo buffers.
	index_t halo_capacity() const {
		const Placement p = place(0u);
		index_t largest = 0u;
		for(unsigned a=0u; a<3u; ++a)
			if(split_axis(a)) largest = std::max(largest, p.face_cells(a));
		return largest;
	}

private:
	unsigned size_[3] = {1u, 1u, 1u};
	unsigned split_[3] = {1u, 1u, 1u};
};

} // namespace culbm::solver
