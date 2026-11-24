#pragma once

// Scalar helpers used by the host code: small powers, sign, integer min/max,
// rounding conversions, byte-order reversal and cubic Hermite blending.
//
// The overload families deliberately cover the same argument types as the
// former helpers so that unqualified calls in the solver resolve to the same
// arithmetic. Float rounding helpers evaluate through the C double functions,
// because that is the precision the solver results were produced with.

#include "types.hpp"

#include <cmath>
#include <cstring>

namespace culbm {
namespace core {

// ---- products ------------------------------------------------------------------
inline float  sq(const float  v) { return v*v; }
inline double sq(const double v) { return v*v; }
inline int    sq(const int    v) { return v*v; }
inline uint   sq(const uint   v) { return v*v; }
inline slong  sq(const slong  v) { return v*v; }
inline ulong  sq(const ulong  v) { return v*v; }

inline float  cb(const float  v) { return v*v*v; }
inline double cb(const double v) { return v*v*v; }
inline int    cb(const int    v) { return v*v*v; }
inline uint   cb(const uint   v) { return v*v*v; }
inline slong  cb(const slong  v) { return v*v*v; }
inline ulong  cb(const ulong  v) { return v*v*v; }

// ---- sign: +1 for zero and positive values, -1 otherwise (NaN gives -1) --------
inline float  sign(const float  v) { return v>=0.0f ? 1.0f : -1.0f; }
inline double sign(const double v) { return v>=0.0 ? 1.0 : -1.0; }
inline int    sign(const int    v) { return v<0 ? -1 : 1; }
inline slong  sign(const slong  v) { return v<0ll ? -1ll : 1ll; }

// ---- integer min/max (floating point code uses fmin/fmax) -----------------------
inline int   min(const int   a, const int   b) { return b<a ? b : a; }
inline uint  min(const uint  a, const uint  b) { return b<a ? b : a; }
inline slong min(const slong a, const slong b) { return b<a ? b : a; }
inline ulong min(const ulong a, const ulong b) { return b<a ? b : a; }
inline int   max(const int   a, const int   b) { return a<b ? b : a; }
inline uint  max(const uint  a, const uint  b) { return a<b ? b : a; }
inline slong max(const slong a, const slong b) { return a<b ? b : a; }
inline ulong max(const ulong a, const ulong b) { return a<b ? b : a; }

// ---- round half up to unsigned integers; negative input and NaN give 0 ----------
// The comparison runs in double (C fmax), the addition in the argument type.
inline uint to_uint(const float v) {
	return static_cast<uint>(std::fmax(static_cast<double>(v+0.5f), 0.5));
}
inline uint to_uint(const double v) {
	return static_cast<uint>(std::fmax(v+0.5, 0.5));
}
inline ulong to_ulong(const float v) {
	return static_cast<ulong>(std::fmax(static_cast<double>(v+0.5f), 0.5));
}
inline ulong to_ulong(const double v) {
	return static_cast<ulong>(std::fmax(v+0.5, 0.5));
}

// ---- byte order reversal (little endian <-> big endian file data) ---------------
namespace detail {
template<typename T> inline T swap_byte_order(const T value) {
	unsigned char bytes[sizeof(T)];
	std::memcpy(bytes, &value, sizeof(T));
	for(unsigned lo=0u, hi=(unsigned)sizeof(T)-1u; lo<hi; lo++, hi--) {
		const unsigned char keep = bytes[lo];
		bytes[lo] = bytes[hi];
		bytes[hi] = keep;
	}
	T result;
	std::memcpy(&result, bytes, sizeof(T));
	return result;
}
} // namespace detail
inline float  reverse_bytes(const float  v) { return detail::swap_byte_order(v); }
inline double reverse_bytes(const double v) { return detail::swap_byte_order(v); }
inline char   reverse_bytes(const char   v) { return v; }
inline uchar  reverse_bytes(const uchar  v) { return v; }
inline short  reverse_bytes(const short  v) { return detail::swap_byte_order(v); }
inline ushort reverse_bytes(const ushort v) { return detail::swap_byte_order(v); }
inline int    reverse_bytes(const int    v) { return detail::swap_byte_order(v); }
inline uint   reverse_bytes(const uint   v) { return detail::swap_byte_order(v); }
inline slong  reverse_bytes(const slong  v) { return detail::swap_byte_order(v); }
inline ulong  reverse_bytes(const ulong  v) { return detail::swap_byte_order(v); }

// ---- cubic Hermite blend between a (t=0) and b (t=1) with end slopes va, vb ----
// Weights are formed in float; the four products are summed left to right.
template<typename T> inline T hermite_spline(const T& p0, const T& p1, const T& slope0, const T& slope1, const float s) {
	const float s3 = cb(s), s2 = sq(s);
	const float w_p0 = 2.0f*s3-3.0f*s2+1.0f;
	const float w_p1 = -2.0f*s3+3.0f*s2;
	const float w_slope0 = s3-2.0f*s2+s;
	const float w_slope1 = s3-s2;
	return w_p0*p0+w_p1*p1+w_slope0*slope0+w_slope1*slope1;
}

} // namespace core
} // namespace culbm
