#pragma once

// Host view of one field across all subdomains of a solver. It addresses the
// field by global cell index, as if the host mirrors of the subdomains were a
// single array of N cells per component (component c of cell n at n+c*N),
// moves the mirrors to and from the devices and writes the legacy raw VTK
// files. In MPI mode the cells of the other ranks come from a replicated copy
// that download() refreshes.

#include "decomposition.hpp"
#include "field_table.hpp"
#include "../core/core.hpp"
#include "../gpu/gpu.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

extern float3 vtk_origin_shift; // SI translation applied to VTK origins

// <path or exe_path/export/><name>-<t as 9 digits><extension>
string default_filename(const string& path, const string& name, const string& extension, const ulong t);
string default_filename(const string& name, const string& extension, const ulong t);

namespace culbm::solver {

class Solver;

namespace detail {

// Hooks into the owning solver (solver.cpp).
void before_host_read(Solver& solver); // refreshes rho/u/T in builds without UPDATE_FIELDS
ulong solver_time(const Solver& solver);

// Raw VTK output (field_view.cpp).
struct RawVtkScale {
	float factor = 1.0f;    // multiplies every value
	bool temperature = false; // absolute temperature conversion instead of a factor
};
RawVtkScale raw_vtk_scale(const string& field, bool si_units);
void write_raw_vtk(const string& filename, const Decomposition& grid, unsigned nz_written, bool si_units,
	const char* type, unsigned components, std::size_t value_bytes,
	const std::function<void(index_t first, index_t count, unsigned char* out)>& encode);
void report_saved(const string& filename);
float si_temperature(float lattice_temperature);

template<typename T> constexpr const char* vtk_type_name() {
	if constexpr(std::is_same<T, char>::value) return "char";
	else if constexpr(std::is_same<T, uchar>::value) return "unsigned_char";
	else if constexpr(std::is_same<T, short>::value) return "short";
	else if constexpr(std::is_same<T, ushort>::value) return "unsigned_short";
	else if constexpr(std::is_same<T, int>::value) return "int";
	else if constexpr(std::is_same<T, uint>::value) return "unsigned_int";
	else if constexpr(std::is_same<T, slong>::value) return "long";
	else if constexpr(std::is_same<T, ulong>::value) return "unsigned_long";
	else if constexpr(std::is_same<T, float>::value) return "float";
	else if constexpr(std::is_same<T, double>::value) return "double";
	else return nullptr;
}

} // namespace detail

template<typename T, HostCopy Copy = HostCopy::resident>
class FieldView {
public:
	// Component accessor: view.y[n] is component 1 of cell n.
	class Component {
	public:
		T& operator[](const index_t n) const { return view_->cell(n, axis_); }
	private:
		friend class FieldView;
		Component(FieldView* view, const unsigned axis) : view_(view), axis_(axis) {}
		FieldView* view_;
		unsigned axis_;
	};
	Component x{this, 0u}, y{this, 1u}, z{this, 2u};

	FieldView() = default;
	FieldView(const FieldView&) = delete;
	FieldView& operator=(const FieldView&) = delete;

	// `parts` are the process-local subdomain buffers in subdomain order; in MPI
	// mode there is one, the subdomain `owned`.
	void bind(Solver& solver, const Decomposition& grid, const bool distributed, const unsigned owned,
		std::vector<gpu::MirroredBuffer<T>*> parts, string name) {
		solver_ = &solver;
		grid_ = grid;
		parts_ = std::move(parts);
		name_ = std::move(name);
		distributed_ = distributed;
		owned_ = owned;
		cells_ = grid.cells();
		components_ = parts_.empty() ? 1u : parts_[0]->components();
		if(cells_*components_==0u) fatal_error("A field view needs at least one cell.");
		single_ = grid.count()==1u;
		const Placement p = grid.place(0u);
		for(unsigned a=0u; a<3u; ++a) {
			inner_[a] = grid.inner(a);
			halo_[a] = p.halo[a];
		}
		extent_x_ = p.extent[0];
		extent_y_ = p.extent[1];
		local_cells_ = p.cells();
		hosts_.assign(parts_.size(), nullptr);
		if constexpr(Copy!=HostCopy::on_demand) cache_hosts();
		if(distributed_) {
			// Until the first download the other ranks' cells read as the
			// initial value of the field.
			replica_.resize((std::size_t)(cells_*components_));
			for(unsigned c=0u; c<components_; ++c) {
				const T initial = Copy==HostCopy::on_demand ? T() : hosts_[0][(index_t)c*local_cells_];
				std::fill(replica_.begin()+(std::ptrdiff_t)(c*cells_), replica_.begin()+(std::ptrdiff_t)((c+1u)*cells_), initial);
			}
		}
	}

	index_t length() const { return cells_; }
	unsigned dimensions() const { return components_; }

	T& operator[](const index_t i) {
		if constexpr(Copy==HostCopy::on_demand) ensure_hosts();
		if(single_) return hosts_[0][i];
		return stitched(i%cells_, (unsigned)(i/cells_));
	}
	const T& operator[](const index_t i) const { return const_cast<FieldView&>(*this)[i]; }

	// Component c of cell n (n < N).
	T& cell(const index_t n, const unsigned c) {
		if constexpr(Copy==HostCopy::on_demand) ensure_hosts();
		if(single_) return hosts_[0][n+(index_t)c*cells_];
		return stitched(n%cells_, std::max((unsigned)(n/cells_), c));
	}

	// Copy every subdomain to its host mirror and wait; then refresh the
	// replicated cells of the other MPI ranks.
	void download() {
		if constexpr(Copy==HostCopy::on_demand) ensure_hosts();
		detail::before_host_read(*solver_);
		for(gpu::MirroredBuffer<T>* part : parts_) part->download_async();
		for(gpu::MirroredBuffer<T>* part : parts_) part->synchronize();
		if(distributed_) gather();
	}
	// Copy every host mirror to its subdomain and wait.
	void upload() {
		if constexpr(Copy==HostCopy::on_demand) ensure_hosts();
		for(gpu::MirroredBuffer<T>* part : parts_) part->upload_async();
		for(gpu::MirroredBuffer<T>* part : parts_) part->synchronize();
	}

	// Copy layers [z_begin, z_begin+nz_count) of component `c` into a flat
	// contiguous global array `dst` (size nx*ny*nz_count) without per-cell
	// division/modulo overhead.
	void copy_contiguous_component(const unsigned c, T* const dst, const unsigned z_begin, const unsigned nz_count) const {
		if(nz_count==0u || dst==nullptr) return;
		const unsigned nx = grid_.size(0u), ny = grid_.size(1u);
		const index_t plane = (index_t)nx*(index_t)ny;
		if(single_) {
			const T* const src = hosts_[0]+(index_t)c*cells_+(index_t)z_begin*plane;
			std::memcpy(dst, src, (std::size_t)(plane*(index_t)nz_count)*sizeof(T));
			return;
		}
		const unsigned sx_count = grid_.split(0u), sy_count = grid_.split(1u);
		const unsigned ix = inner_[0], iy = inner_[1], iz = inner_[2];
		const unsigned hx = halo_[0], hy = halo_[1], hz = halo_[2];
		parallel_for((ulong)nz_count, [&](const ulong dz) {
			const unsigned gz = z_begin+(unsigned)dz;
			const unsigned sz = gz/iz;
			const unsigned lz = gz-sz*iz+hz;
			T* const out_plane = dst+(index_t)dz*plane;
			for(unsigned sy=0u; sy<sy_count; ++sy) {
				const unsigned y0 = sy*iy;
				for(unsigned sx=0u; sx<sx_count; ++sx) {
					const unsigned x0 = sx*ix;
					const unsigned id = sx+(sy+sz*sy_count)*sx_count;
					if(distributed_ && id!=owned_) {
						const T* const rep = replica_.data()+(std::size_t)((index_t)c*cells_+(index_t)gz*plane);
						for(unsigned ly=0u; ly<iy; ++ly)
							std::memcpy(out_plane+(index_t)(y0+ly)*nx+x0, rep+(index_t)(y0+ly)*nx+x0, (std::size_t)ix*sizeof(T));
					} else {
						const T* const h = distributed_ ? hosts_[0] : hosts_[id];
						const T* const base = h+(index_t)c*local_cells_+((index_t)lz*extent_y_+(index_t)hy)*extent_x_+(index_t)hx;
						for(unsigned ly=0u; ly<iy; ++ly)
							std::memcpy(out_plane+(index_t)(y0+ly)*nx+x0, base+(index_t)ly*extent_x_, (std::size_t)ix*sizeof(T));
					}
				}
			}
		});
	}

	// Raw big-endian VTK file of the host mirrors (MPI root only); nz_write>0
	// limits the output to the lowest nz_write layers.
	void write_host_to_vtk(const string& path="", const bool si_units=true, const bool report=true, const uint nz_write=0u) {
		if(!culbm::mpi_is_root()) return;
		if constexpr(Copy==HostCopy::on_demand) ensure_hosts();
		const string filename = default_filename(path, name_, ".vtk", detail::solver_time(*solver_));
		const unsigned nz = nz_write>0u && nz_write<grid_.size(2u) ? nz_write : grid_.size(2u);
		const detail::RawVtkScale scale = detail::raw_vtk_scale(name_, si_units);
		const T factor = (T)scale.factor;
		const unsigned parts = components_;
		const index_t plane = (index_t)grid_.size(0u)*(index_t)grid_.size(1u);
		std::vector<T> flat;
		if(!single_) {
			flat.resize((std::size_t)(plane*(index_t)nz*(index_t)parts));
			for(unsigned c=0u; c<parts; ++c) copy_contiguous_component(c, flat.data()+(std::size_t)((index_t)c*plane*(index_t)nz), 0u, nz);
		}
		const T* const base = single_ ? hosts_[0] : flat.data();
		const index_t stride = single_ ? cells_ : plane*(index_t)nz;
		detail::write_raw_vtk(filename, grid_, nz, si_units, detail::vtk_type_name<T>(), parts, sizeof(T),
			[&](const index_t first, const index_t count, unsigned char* const out) {
				T* const values = reinterpret_cast<T*>(out);
				parallel_for(count, [&](const ulong i) {
					const index_t g = first+i;
					for(unsigned c=0u; c<parts; ++c) {
						const T value = base[g+(index_t)c*stride];
						const T converted = scale.temperature ? (T)detail::si_temperature((float)value) : (T)(factor*value);
						values[i*parts+c] = reverse_bytes(converted);
					}
				});
			});
		if(report) detail::report_saved(filename);
	}
	void write_device_to_vtk(const string& path="", const bool si_units=true, const bool report=true, const uint nz_write=0u) {
		download();
		write_host_to_vtk(path, si_units, report, nz_write);
	}

private:
	T& stitched(const index_t g, const unsigned c) {
		const index_t layer = (index_t)grid_.size(0u)*grid_.size(1u);
		const index_t in_layer = g%layer;
		const unsigned gx = (unsigned)(in_layer%grid_.size(0u));
		const unsigned gy = (unsigned)(in_layer/grid_.size(0u));
		const unsigned gz = (unsigned)(g/layer);
		const unsigned sx = gx/inner_[0], sy = gy/inner_[1], sz = gz/inner_[2];
		const unsigned id = sx+(sy+sz*grid_.split(1u))*grid_.split(0u);
		const index_t local = (index_t)(gx-sx*inner_[0]+halo_[0])+((index_t)(gy-sy*inner_[1]+halo_[1])+
			(index_t)(gz-sz*inner_[2]+halo_[2])*extent_y_)*extent_x_;
		if(distributed_) {
			if(id==owned_) return hosts_[0][local+(index_t)c*local_cells_];
			return replica_[(std::size_t)(g+(index_t)c*cells_)];
		}
		return hosts_[id][local+(index_t)c*local_cells_];
	}

	void cache_hosts() {
		for(std::size_t d=0u; d<parts_.size(); ++d) hosts_[d] = parts_[d]->data();
	}
	void ensure_hosts() {
		if(!hosts_.empty() && hosts_[0]!=nullptr) return;
		for(gpu::MirroredBuffer<T>* part : parts_) part->attach_host();
		cache_hosts();
	}

	// MPI: every rank contributes its interior cells; all ranks rebuild the
	// replicated global field from the gathered blocks.
	void gather() {
		const Placement mine = grid_.place(owned_);
		const index_t block = (index_t)inner_[0]*inner_[1]*inner_[2];
		std::vector<T> local((std::size_t)(block*components_));
		const auto for_interior = [&](const auto& visit) {
			for(unsigned c=0u; c<components_; ++c)
				for(unsigned z=0u; z<inner_[2]; ++z)
					for(unsigned y=0u; y<inner_[1]; ++y)
						for(unsigned x=0u; x<inner_[0]; ++x)
							visit(c, x, y, z, (index_t)x+((index_t)y+(index_t)z*inner_[1])*inner_[0]);
		};
		for_interior([&](const unsigned c, const unsigned x, const unsigned y, const unsigned z, const index_t i) {
			local[(std::size_t)(i+(index_t)c*block)] =
				hosts_[0][mine.local_index(x+halo_[0], y+halo_[1], z+halo_[2])+(index_t)c*local_cells_];
		});
		std::vector<T> all((std::size_t)(block*components_*grid_.count()));
		culbm::mpi_allgather_equal_bytes(local.data(), (std::uint64_t)(local.size()*sizeof(T)), all.data());
		for(unsigned id=0u; id<grid_.count(); ++id) {
			const Placement p = grid_.place(id);
			const T* const source = all.data()+(std::size_t)((index_t)id*block*components_);
			for_interior([&](const unsigned c, const unsigned x, const unsigned y, const unsigned z, const index_t i) {
				const index_t gx = (index_t)p.cell[0]*inner_[0]+x;
				const index_t gy = (index_t)p.cell[1]*inner_[1]+y;
				const index_t gz = (index_t)p.cell[2]*inner_[2]+z;
				const index_t g = gx+(gy+gz*grid_.size(1u))*grid_.size(0u);
				replica_[(std::size_t)(g+(index_t)c*cells_)] = source[(std::size_t)(i+(index_t)c*block)];
			});
		}
	}

	Solver* solver_ = nullptr;
	Decomposition grid_;
	std::vector<gpu::MirroredBuffer<T>*> parts_;
	std::vector<T*> hosts_; // cached host mirror addresses of parts_
	string name_;
	bool distributed_ = false;
	bool single_ = true;
	unsigned owned_ = 0u;
	index_t cells_ = 0u;
	unsigned components_ = 1u;
	unsigned inner_[3] = {1u, 1u, 1u};
	unsigned halo_[3] = {0u, 0u, 0u};
	index_t extent_x_ = 1u, extent_y_ = 1u, local_cells_ = 1u;
	std::vector<T> replica_; // MPI: the whole field, refreshed by download()
};

} // namespace culbm::solver
