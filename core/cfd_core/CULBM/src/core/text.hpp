#pragma once

// Text helpers of the host code: number formatting, alignment and a few
// string predicates.
//
// Number formatting is part of the solver's observable output: the strings
// are pasted into NVRTC kernel constants, VTK headers and console tables.
// The floating-point formats below are therefore fixed:
//  * to_string(float)/to_string(double): "d.dddddddd[E<exp>]" with 8 (float)
//    or 16 (double) fraction digits. The mantissa is found by multiplying
//    with a fixed sequence of decimal powers in the argument's own precision,
//    so the last digits carry that (slightly lossy) rounding.
//  * to_string(value, decimals): fixed-point with at most 8 (float) or 16
//    (double) fraction digits, rounded by adding half a unit of the last
//    digit before truncation.
//  * NaN prints as "NaN" (never signed), infinities as "Inf"/"-Inf", and the
//    fixed-point forms print "Inf" for values beyond the 64-bit range.

#include "scalar.hpp"

#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

namespace culbm {
namespace core {

namespace detail {

// Decimal digits of v, most significant first.
inline std::string decimal_digits(ulong v) {
	char buffer[24];
	char* first = buffer+sizeof(buffer);
	do {
		*--first = (char)('0'+(int)(v%10ull));
		v /= 10ull;
	} while(v!=0ull);
	return std::string(first, buffer+sizeof(buffer));
}
inline std::string signed_decimal_digits(const slong v) {
	return v<0ll ? "-"+decimal_digits(0ull-(ulong)v) : decimal_digits((ulong)v);
}
// The lowest `count` decimal digits of v, zero-padded on the left; nothing for count<=0.
inline std::string fixed_width_digits(ulong v, const int count) {
	if(count<=0) return std::string();
	std::string digits((size_t)count, '0');
	for(int i=count-1; i>=0; i--) {
		digits[(size_t)i] = (char)('0'+(int)(v%10ull));
		v /= 10ull;
	}
	return digits;
}

// One step of the decimal normalisation: when the test holds, the value is
// multiplied by `factor` and the decimal exponent moves by `shift`.
template<typename Real> struct DecimalStep {
	Real bound;
	Real factor;
	int shift;
};
// Large values are reduced (applied while value >= bound) ...
inline constexpr DecimalStep<float> float_reduce[] = {
	{1E32f, 1E-32f, 32}, {1E16f, 1E-16f, 16}, {1E8f, 1E-8f, 8}, {1E4f, 1E-4f, 4}, {1E2f, 1E-2f, 2}, {1E1f, 1E-1f, 1}
};
// ... and values in (0,1] are enlarged (applied while value < bound).
inline constexpr DecimalStep<float> float_enlarge[] = {
	{1E-31f, 1E32f, 32}, {1E-15f, 1E16f, 16}, {1E-7f, 1E8f, 8}, {1E-3f, 1E4f, 4}, {1E-1f, 1E2f, 2}, {1E0f, 1E1f, 1}
};
inline constexpr DecimalStep<double> double_reduce[] = {
	{1E256, 1E-256, 256}, {1E128, 1E-128, 128}, {1E64, 1E-64, 64}, {1E32, 1E-32, 32}, {1E16, 1E-16, 16},
	{1E8, 1E-8, 8}, {1E4, 1E-4, 4}, {1E2, 1E-2, 2}, {1E1, 1E-1, 1}
};
inline constexpr DecimalStep<double> double_enlarge[] = {
	{1E-255, 1E256, 256}, {1E-127, 1E128, 128}, {1E-63, 1E64, 64}, {1E-31, 1E32, 32}, {1E-15, 1E16, 16},
	{1E-7, 1E8, 8}, {1E-3, 1E4, 4}, {1E-1, 1E2, 2}, {1E0, 1E1, 1}
};

// Decomposes a non-negative finite value into leading digit(s), a fraction
// with `fraction_digits` decimal digits (rounded half up) and an exponent.
template<typename Real, typename Fraction, size_t Steps>
inline void decimal_scientific(Real value, const DecimalStep<Real> (&reduce)[Steps], const DecimalStep<Real> (&enlarge)[Steps],
	const Real fraction_scale, const Fraction fraction_limit, uint& leading, Fraction& fraction, int& exponent) {
	exponent = 0;
	if(value>=reduce[Steps-1u].bound) {
		for(const DecimalStep<Real>& step : reduce) {
			if(value>=step.bound) { value *= step.factor; exponent += step.shift; }
		}
	}
	if(value>(Real)0&&value<=(Real)1) {
		for(const DecimalStep<Real>& step : enlarge) {
			if(value<step.bound) { value *= step.factor; exponent -= step.shift; }
		}
	}
	leading = (uint)value;
	const Real scaled_rest = (value-(Real)leading)*fraction_scale;
	fraction = (Fraction)scaled_rest;
	if(scaled_rest-(Real)fraction>=(Real)0.5) { // round the last fraction digit half up
		fraction++;
		if(fraction>=fraction_limit) { // carry into the leading digit
			fraction = (Fraction)0;
			leading++;
			if(leading>=10u) {
				leading = 1u;
				exponent++;
			}
		}
	}
}

inline constexpr float float_decimal_scale[9] = { 1.0f, 1E1f, 1E2f, 1E3f, 1E4f, 1E5f, 1E6f, 1E7f, 1E8f };
inline constexpr double double_decimal_scale[17] = {
	1.0, 1E1, 1E2, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9, 1E10, 1E11, 1E12, 1E13, 1E14, 1E15, 1E16
};

template<typename T> inline constexpr bool is_other_u64_spelling = std::is_same<T, unsigned long long>::value&&!std::is_same<T, ulong>::value;
template<typename T> inline constexpr bool is_other_s64_spelling = std::is_same<T, long long>::value&&!std::is_same<T, slong>::value;

} // namespace detail

// ---- to_string ------------------------------------------------------------------
inline std::string to_string(const std::string& s) { return s; }
inline std::string to_string(const char c) { return std::string(1u, c); }
inline std::string to_string(const uchar c) { return std::string(1u, (char)c); }
inline std::string to_string(const uint v) { return detail::decimal_digits(v); }
inline std::string to_string(const int v) { return detail::signed_decimal_digits(v); }
inline std::string to_string(const ulong v) { return detail::decimal_digits(v); }
inline std::string to_string(const slong v) { return detail::signed_decimal_digits(v); }
// Platforms where the 64-bit aliases are `long` still print `long long` arguments.
template<typename T, typename std::enable_if<detail::is_other_u64_spelling<T>, int>::type = 0>
inline std::string to_string(const T v) { return detail::decimal_digits((ulong)v); }
template<typename T, typename std::enable_if<detail::is_other_s64_spelling<T>, int>::type = 0>
inline std::string to_string(const T v) { return detail::signed_decimal_digits((slong)v); }

inline std::string to_string(float v) { // full precision, "d.dddddddd[E<exp>]"
	const bool negative = v<0.0f;
	if(negative) v = -v;
	const std::string sign_text = negative ? "-" : "";
	if(std::isnan(v)) return sign_text+"NaN";
	if(std::isinf(v)) return sign_text+"Inf";
	uint leading = 0u, fraction = 0u;
	int exponent = 0;
	detail::decimal_scientific(v, detail::float_reduce, detail::float_enlarge, 1E8f, 100000000u, leading, fraction, exponent);
	std::string text = sign_text+detail::decimal_digits(leading)+"."+detail::fixed_width_digits(fraction, 8);
	if(exponent!=0) text += "E"+detail::signed_decimal_digits(exponent);
	return text;
}
inline std::string to_string(double v) { // full precision, "d.dddddddddddddddd[E<exp>]"
	const bool negative = v<0.0;
	if(negative) v = -v;
	const std::string sign_text = negative ? "-" : "";
	if(std::isnan(v)) return sign_text+"NaN";
	if(std::isinf(v)) return sign_text+"Inf";
	uint leading = 0u;
	ulong fraction = 0ull;
	int exponent = 0;
	detail::decimal_scientific(v, detail::double_reduce, detail::double_enlarge, 1E16, 10000000000000000ull, leading, fraction, exponent);
	std::string text = sign_text+detail::decimal_digits(leading)+"."+detail::fixed_width_digits(fraction, 16);
	if(exponent!=0) text += "E"+detail::signed_decimal_digits(exponent);
	return text;
}
inline std::string to_string(float v, const uint decimals) { // fixed point, at most 8 fraction digits
	const bool negative = v<0.0f;
	if(negative) v = -v;
	const std::string sign_text = negative ? "-" : "";
	if(std::isnan(v)) return sign_text+"NaN";
	if(std::isinf(v)||v>(float)max_ulong) return sign_text+"Inf";
	const float scale = detail::float_decimal_scale[decimals<8u ? decimals : 8u];
	v += 0.5f/scale;
	const ulong whole = (ulong)v;
	const uint fraction = (uint)((v-(float)whole)*scale);
	std::string text = sign_text+detail::decimal_digits(whole);
	if(decimals!=0u) text += "."+detail::fixed_width_digits(fraction, (int)decimals<8 ? (int)decimals : 8);
	return text;
}
inline std::string to_string(double v, const uint decimals) { // fixed point, at most 16 fraction digits
	const bool negative = v<0.0;
	if(negative) v = -v;
	const std::string sign_text = negative ? "-" : "";
	if(std::isnan(v)) return sign_text+"NaN";
	if(std::isinf(v)||v>(double)max_ulong) return sign_text+"Inf";
	const double scale = detail::double_decimal_scale[decimals<16u ? decimals : 16u];
	v += 0.5/scale;
	const ulong whole = (ulong)v;
	const ulong fraction = (ulong)((v-(double)whole)*scale);
	std::string text = sign_text+detail::decimal_digits(whole);
	if(decimals!=0u) text += "."+detail::fixed_width_digits(fraction, (int)decimals<16 ? (int)decimals : 16);
	return text;
}

// ---- string predicates and slicing ---------------------------------------------------
inline uint length(const std::string& s) { return (uint)s.length(); }
inline bool contains(const std::string& s, const std::string& match) { return s.find(match)!=std::string::npos; }
template<class T> inline bool contains(const std::vector<T>& v, const T& match) {
	for(const T& element : v) if(element==match) return true;
	return false;
}
inline bool begins_with(const std::string& s, const std::string& prefix) {
	return prefix.size()<=s.size()&&s.compare(0u, prefix.size(), prefix)==0;
}
// Characters removed by trim(): blank, tab, line feed, vertical tab, form feed, carriage return and NUL.
inline bool is_trim_character(const char c) {
	return c==' '||c=='\t'||c=='\n'||c=='\v'||c=='\f'||c=='\r'||c=='\0';
}
inline std::string trim(const std::string& s) {
	size_t first = 0u, last = s.size();
	while(first<last&&is_trim_character(s[first])) first++;
	while(last>first&&is_trim_character(s[last-1u])) last--;
	return s.substr(first, last-first);
}
// Up to `count` characters from `start`; start beyond the end throws like std::string::substr.
inline std::string substring(const std::string& s, const uint start, const uint count=max_uint) {
	const uint available = (uint)s.length()-start;
	return s.substr(start, count<available ? count : available);
}

// ---- column alignment ---------------------------------------------------------------
// Text that is already at least n characters wide is returned unchanged by
// alignl/alignr; shorter text is padded with blanks to width n.
inline std::string alignl(const uint n, const std::string& x="") {
	return x.length()>=n ? x : x+std::string(n-(uint)x.length(), ' ');
}
inline std::string alignr(const uint n, const std::string& x="") {
	return x.length()>=n ? x : std::string(n-(uint)x.length(), ' ')+x;
}
inline std::string alignc(const uint n, const std::string& x="") { // converts x to centered string with spaces on both sides such that length is n if x is not longer than n
	if((uint)x.length()>=n) return x.substr(0u, n);
	const uint left = (n-(uint)x.length())/2u;
	const uint right = n-(uint)x.length()-left;
	return std::string(left, ' ')+x+std::string(right, ' ');
}
template<typename T> inline std::string alignl(const uint n, const T x) { return alignl(n, to_string(x)); }
template<typename T> inline std::string alignr(const uint n, const T x) { return alignr(n, to_string(x)); }
template<typename T> inline std::string alignc(const uint n, const T x) { return alignc(n, to_string(x)); }

// ---- durations ------------------------------------------------------------------------
// Seconds as "[Y y][DDD d][HH h][MM m]SS s": a unit appears once the input
// exceeds it, and inner units are zero-padded once an outer unit is shown.
// A year is 31556925 s.
inline std::string print_time(const double t) {
	uint rest = (uint)(t+0.5);
	const uint years = rest/31556925u; rest %= 31556925u;
	const uint days = rest/86400u; rest %= 86400u;
	const uint hours = rest/3600u; rest %= 3600u;
	const uint minutes = rest/60u; rest %= 60u;
	const bool show_years = t>31556924.5, show_days = t>86399.5, show_hours = t>3599.5, show_minutes = t>59.5;
	std::string text;
	if(show_years) text += detail::decimal_digits(years)+"y";
	if(show_days) {
		if(show_years) text += days<10u ? "00" : days<100u ? "0" : "";
		text += detail::decimal_digits(days)+"d";
	}
	if(show_hours) text += (show_days&&hours<10u ? "0" : "")+detail::decimal_digits(hours)+"h";
	if(show_minutes) text += (show_hours&&minutes<10u ? "0" : "")+detail::decimal_digits(minutes)+"m";
	text += (show_minutes&&rest<10u ? "0" : "")+detail::decimal_digits(rest)+"s";
	return text;
}

} // namespace core
} // namespace culbm
