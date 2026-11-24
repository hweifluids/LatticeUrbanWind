#pragma once

// Triangle meshes and the binary STL reader.

#include "console.hpp"
#include "progress.hpp"
#include "system.hpp"
#include "text.hpp"
#include "vec.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace culbm {
namespace core {
namespace detail {
// C fmin/fmax semantics (a NaN argument is ignored) with a fixed tie rule:
// of two equal values (+0 and -0) the second argument is returned. GCC
// treats the fmin/fmax builtins as commutative, so the former code got its
// signed-zero ties from operand order chosen by the compiler; in the former
// solver build that was this rule.
inline double lower_of(const double a, const double b) { return (a<b||b!=b) ? a : b; }
inline double upper_of(const double a, const double b) { return (a>b||b!=b) ? a : b; }
} // namespace detail
} // namespace core
} // namespace culbm

// Triangle soup with separate corner arrays and an axis-aligned bounding box.
// It is declared in the global namespace because solver headers forward-declare
// `struct Mesh;` there. Bounds are only updated by find_bounds() and by the
// transforms below.
struct Mesh {
	culbm::core::uint triangle_number = 0u;
	culbm::core::float3 center, pmin, pmax;
	culbm::core::float3* p0;
	culbm::core::float3* p1;
	culbm::core::float3* p2;

	Mesh(const culbm::core::uint triangle_number, const culbm::core::float3& center)
		: triangle_number(triangle_number), center(center), pmin(center), pmax(center),
		  p0(new culbm::core::float3[triangle_number]), p1(new culbm::core::float3[triangle_number]), p2(new culbm::core::float3[triangle_number]) {}
	~Mesh() {
		delete[] p0;
		delete[] p1;
		delete[] p2;
	}
	Mesh(const Mesh&) = delete; // owns its corner arrays
	Mesh& operator=(const Mesh&) = delete;

	// Axis-aligned box of the corners p0[0] and p0/p1/p2 of triangles 1..n-1
	// (the second and third corner of triangle 0 are not included, as in
	// every former mesh box). The defining rule is the per-axis fold
	//   lo = lower_of(lower_of(lower_of(a, b), c), lo)   (triangle order),
	// likewise with upper_of. Both ignore NaN and otherwise return the
	// smaller (larger) value, so a plain comparison scan yields the same bits
	// whenever the extreme is finite and non-zero; a zero (+0/-0 ties), an
	// infinity or an all-NaN axis is resolved by the fold itself.
	// An empty mesh keeps its box.
	void find_bounds() {
		if(triangle_number==0u) return;
		float lo[3], hi[3];
		scan_extremes(lo, hi);
		for(int axis=0; axis<3; axis++) {
			if(!is_plain_extreme(lo[axis])) lo[axis] = fold_extreme(axis, false);
			if(!is_plain_extreme(hi[axis])) hi[axis] = fold_extreme(axis, true);
		}
		pmin = culbm::core::float3(lo[0], lo[1], lo[2]);
		pmax = culbm::core::float3(hi[0], hi[1], hi[2]);
	}
	// Scales corners and box about the mesh center.
	void scale(const float factor) {
		for(culbm::core::uint i=0u; i<triangle_number; i++) {
			p0[i] = factor*(p0[i]-center)+center;
			p1[i] = factor*(p1[i]-center)+center;
			p2[i] = factor*(p2[i]-center)+center;
		}
		pmin = factor*(pmin-center)+center;
		pmax = factor*(pmax-center)+center;
	}
	// Shifts corners, center and box.
	void translate(const culbm::core::float3& shift) {
		for(culbm::core::uint i=0u; i<triangle_number; i++) {
			p0[i] += shift;
			p1[i] += shift;
			p2[i] += shift;
		}
		center += shift;
		pmin += shift;
		pmax += shift;
	}
	culbm::core::float3 get_bounding_box_size() const { return pmax-pmin; }

private:
	static float axis_value(const culbm::core::float3& p, const int axis) { return axis==0 ? p.x : axis==1 ? p.y : p.z; }
	static bool is_plain_extreme(const float v) { return v!=0.0f&&std::isfinite(v); }
	// Comparison scan; NaN never replaces a value, infinities start the search.
	void scan_extremes(float lo[3], float hi[3]) const {
		constexpr float infinity = std::numeric_limits<float>::infinity();
		float lx = infinity, ly = infinity, lz = infinity, hx = -infinity, hy = -infinity, hz = -infinity;
		const auto take = [&](const culbm::core::float3& p) {
			lx = p.x<lx ? p.x : lx; ly = p.y<ly ? p.y : ly; lz = p.z<lz ? p.z : lz;
			hx = p.x>hx ? p.x : hx; hy = p.y>hy ? p.y : hy; hz = p.z>hz ? p.z : hz;
		};
		take(p0[0]);
		for(culbm::core::uint i=1u; i<triangle_number; i++) {
			take(p0[i]);
			take(p1[i]);
			take(p2[i]);
		}
		lo[0] = lx; lo[1] = ly; lo[2] = lz;
		hi[0] = hx; hi[1] = hy; hi[2] = hz;
	}
	// The defining fold for one axis (upper=false: minimum, true: maximum).
	float fold_extreme(const int axis, const bool upper) const {
		float bound = axis_value(p0[0], axis);
		for(culbm::core::uint i=1u; i<triangle_number; i++) {
			const double a = (double)axis_value(p0[i], axis), b = (double)axis_value(p1[i], axis), c = (double)axis_value(p2[i], axis);
			using culbm::core::detail::lower_of;
			using culbm::core::detail::upper_of;
			bound = upper ? (float)upper_of(upper_of(upper_of(a, b), c), (double)bound)
			              : (float)lower_of(lower_of(lower_of(a, b), c), (double)bound);
		}
		return bound;
	}
};

namespace culbm {
namespace core {

namespace detail {

// Runs body(i) for i in [0, count); large meshes are split over all cores.
// Each index is processed independently, so the result does not depend on
// the thread count.
template<typename Body> inline void for_each_triangle(const uint count, const Body& body) {
	constexpr uint serial_limit = 65536u;
	const uint threads = default_thread_count();
	if(count<serial_limit||threads<2u) {
		for(uint i=0u; i<count; i++) body(i);
		return;
	}
	run_static_blocks(count, threads, [&body](const ulong i, const uint) { body((uint)i); });
}

// Binary STL: 80-byte header, uint32 triangle count, then 50 bytes per
// triangle (normal, three corners as float32 xyz, uint16 attribute).
// Every corner is rotated by `rotation`, then the mesh is scaled about the
// origin and placed at `center`:
//   size == 0: scaled to fit box_size,
//   size  > 0: longest box side scaled to `size`,
//   size  < 0: scaled by -size.
// With `reposition` the bounding-box center is moved to `center` first.
inline ::Mesh* load_binary_stl(const std::string& path, const bool reposition, const float3& box_size, const float3& center, const float3x3& rotation, const float size) {
	const std::string filename = create_file_extension(path, ".stl");
	const std::string quoted = "File \""+filename+"\"";
	std::unique_ptr<char[]> bytes;
	ulong file_size = 0ull;
	{
		std::ifstream input(filename, std::ios::binary|std::ios::ate); // opened at the end to learn the size
		if(!input.is_open()) fatal_error(quoted+" does not exist!");
		const std::streamoff end_position = input.tellg();
		file_size = end_position>0 ? (ulong)end_position : 0ull;
		bytes.reset(new char[file_size>0ull ? file_size : 1ull]);
		input.seekg(0);
		input.read(bytes.get(), (std::streamsize)file_size);
	}
	if(file_size==0ull) fatal_error(quoted+" is corrupt!");
	uint triangles = 0u;
	if(file_size>=84ull) std::memcpy(&triangles, bytes.get()+80, sizeof(uint));
	if(triangles>0u&&file_size==84ull+50ull*(ulong)triangles) {
		log_info("Loading \""+filename+"\" with "+to_string(triangles)+" triangles.");
	} else {
		fatal_error(quoted+" is corrupt or unsupported! Only binary .stl files are supported.");
	}
	luw_emit_progress("load_stl", "Loading STL", filename+" ("+to_string(triangles)+" triangles)", 0ll, 1ll, false);

	::Mesh* mesh = new ::Mesh(triangles, center);
	const char* const records = bytes.get()+84;
	for_each_triangle(triangles, [&](const uint i) {
		float values[12]; // normal followed by the three corners
		std::memcpy(values, records+50ull*(ulong)i, sizeof(values));
		mesh->p0[i] = rotation*float3(values[3], values[4], values[5]);
		mesh->p1[i] = rotation*float3(values[6], values[7], values[8]);
		mesh->p2[i] = rotation*float3(values[9], values[10], values[11]);
	});
	bytes.reset();
	mesh->find_bounds();

	const float3 extent = mesh->pmax-mesh->pmin;
	float factor = 1.0f;
	if(size==0.0f) {
		factor = (float)lower_of(lower_of((double)(box_size.x/extent.x), (double)(box_size.y/extent.y)), (double)(box_size.z/extent.z));
	} else if(size>0.0f) {
		factor = size/(float)upper_of(upper_of((double)extent.x, (double)extent.y), (double)extent.z);
	} else {
		factor = -size;
	}
	const float3 offset = reposition ? -0.5f*(mesh->pmin+mesh->pmax) : float3(0.0f);
	for_each_triangle(triangles, [&](const uint i) {
		mesh->p0[i] = center+factor*(offset+mesh->p0[i]);
		mesh->p1[i] = center+factor*(offset+mesh->p1[i]);
		mesh->p2[i] = center+factor*(offset+mesh->p2[i]);
	});
	mesh->find_bounds();
	luw_emit_progress("load_stl", "Loading STL", filename+" loaded", 1ll, 1ll, false);
	return mesh;
}

} // namespace detail

// Reads a binary STL, fits it into box_size and centers it at `center`
// (see detail::load_binary_stl for the meaning of `size`).
inline ::Mesh* read_stl(const std::string& path, const float3& box_size, const float3& center, const float3x3& rotation, const float size) {
	return detail::load_binary_stl(path, true, box_size, center, rotation, size);
}
// Reads a binary STL without re-centering: corners are rotated, scaled by
// |scale| about the origin and shifted by `offset`.
inline ::Mesh* read_stl(const std::string& path, const float scale=1.0f, const float3x3& rotation=float3x3(1.0f), const float3& offset=float3(0.0f)) {
	return detail::load_binary_stl(path, false, float3(1.0f), offset, rotation, (float)-std::fabs((double)scale));
}

} // namespace core
} // namespace culbm
