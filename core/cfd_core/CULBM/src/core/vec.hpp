#pragma once

// Small host vector types. float3 and uint3 are plain 12-byte aggregates of
// three components (same layout as the device side), with implicit
// construction from scalars and from the other arithmetic types.
//
// Each operator evaluates exactly one IEEE operation per component and the
// sums inside dot products and matrix rows run left to right (x, y, z). The
// voxelizer and the STL transforms depend on these results bit for bit.

#include "scalar.hpp"

#include <cmath>

namespace culbm {
namespace core {

struct uint3 {
	uint x, y, z;
	uint3() = default;
	uint3(const uint vx, const uint vy, const uint vz) : x(vx), y(vy), z(vz) {}
	uint3(const int vx, const int vy, const int vz) : x((uint)vx), y((uint)vy), z((uint)vz) {}
	uint3(const float vx, const float vy, const float vz) : x(to_uint(vx)), y(to_uint(vy)), z(to_uint(vz)) {}
	uint3(const double vx, const double vy, const double vz) : x(to_uint(vx)), y(to_uint(vy)), z(to_uint(vz)) {}
	uint3(const uint v) : x(v), y(v), z(v) {}
	uint3(const int v) : x((uint)v), y((uint)v), z((uint)v) {}
	uint3(const float v) : uint3(to_uint(v)) {}
	uint3(const double v) : uint3(to_uint(v)) {}
	bool operator==(const uint3& o) const { return x==o.x&&y==o.y&&z==o.z; }
	bool operator!=(const uint3& o) const { return !(*this==o); }
};

struct float3 {
	float x, y, z;
	float3() = default;
	float3(const float vx, const float vy, const float vz) : x(vx), y(vy), z(vz) {}
	float3(const double vx, const double vy, const double vz) : x((float)vx), y((float)vy), z((float)vz) {}
	float3(const int vx, const int vy, const int vz) : x((float)vx), y((float)vy), z((float)vz) {}
	float3(const uint vx, const uint vy, const uint vz) : x((float)vx), y((float)vy), z((float)vz) {}
	float3(const float v) : x(v), y(v), z(v) {}
	float3(const double v) : float3((float)v) {}
	float3(const int v) : float3((float)v) {}
	float3(const uint v) : float3((float)v) {}

	float3& operator+=(const float3& o) { x += o.x; y += o.y; z += o.z; return *this; }
	float3& operator-=(const float3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
	float3& operator+=(const float s) { x += s; y += s; z += s; return *this; }
	float3& operator-=(const float s) { x -= s; y -= s; z -= s; return *this; }
	float3& operator*=(const float s) { x *= s; y *= s; z *= s; return *this; }
	float3& operator/=(const float s) { x /= s; y /= s; z /= s; return *this; }

	const float3& operator+() const { return *this; }
	float3 operator-() const { return float3(-x, -y, -z); }
	float3 operator+(const float3& o) const { return float3(x+o.x, y+o.y, z+o.z); }
	float3 operator-(const float3& o) const { return float3(x-o.x, y-o.y, z-o.z); }
	float operator*(const float3& o) const { return x*o.x+y*o.y+z*o.z; } // dot product
	bool operator==(const float3& o) const { return x==o.x&&y==o.y&&z==o.z; }
	bool operator!=(const float3& o) const { return !(x==o.x&&y==o.y&&z==o.z); }
};

// Vector-scalar arithmetic. The scalar is always the right operand of the
// component operation (v.x op s), whichever side it was written on, and a
// subtraction of a scalar is an addition of its negation.
inline float3 operator+(const float3& v, const float s) { return float3(v.x+s, v.y+s, v.z+s); }
inline float3 operator+(const float s, const float3& v) { return v+s; }
inline float3 operator-(const float3& v, const float s) { return v+(-s); }
inline float3 operator-(const float s, const float3& v) { return float3(s-v.x, s-v.y, s-v.z); }
inline float3 operator*(const float3& v, const float s) { return float3(v.x*s, v.y*s, v.z*s); }
inline float3 operator*(const float s, const float3& v) { return v*s; }
inline float3 operator/(const float3& v, const float s) { return float3(v.x/s, v.y/s, v.z/s); }

inline float dot(const float3& a, const float3& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline float3 cross(const float3& a, const float3& b) {
	return float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
// Euclidean norm; the square root is taken in double and rounded to float.
inline float length(const float3& p) {
	const float squared = sq(p.x)+sq(p.y)+sq(p.z);
	return (float)std::sqrt((double)squared);
}

// Row-major 3x3 matrix (xx xy xz / yx yy yz / zx zy zz).
struct float3x3 {
	float xx, xy, xz; // first row
	float yx, yy, yz;
	float zx, zy, zz;
	float3x3() = default;
	float3x3(const float diagonal) // diagonal matrix, float3x3(1.0f) is the identity
		: xx(diagonal), xy(0.0f), xz(0.0f), yx(0.0f), yy(diagonal), yz(0.0f), zx(0.0f), zy(0.0f), zz(diagonal) {}
	float3x3(const float m_xx, const float m_xy, const float m_xz, const float m_yx, const float m_yy, const float m_yz, const float m_zx, const float m_zy, const float m_zz)
		: xx(m_xx), xy(m_xy), xz(m_xz), yx(m_yx), yy(m_yy), yz(m_yz), zx(m_zx), zy(m_zy), zz(m_zz) {}
};
inline float3 operator*(const float3x3& m, const float3& v) { // matrix times column vector
	const float rx = m.xx*v.x+m.xy*v.y+m.xz*v.z;
	const float ry = m.yx*v.x+m.yy*v.y+m.yz*v.z;
	const float rz = m.zx*v.x+m.zy*v.y+m.zz*v.z;
	return float3(rx, ry, rz);
}

} // namespace core
} // namespace culbm
