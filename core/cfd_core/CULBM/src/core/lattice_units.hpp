#pragma once

// Conversion between SI quantities and lattice quantities.
//
// The lattice is described by four SI scale factors: the metres per cell
// (dx), the seconds per time step (dt), the kilograms per lattice mass unit
// and the kelvins per lattice temperature unit, plus an absolute temperature
// offset (T_SI = T_lattice*kelvin + offset). Every conversion is one product
// or quotient of these factors, evaluated in float in the order written.

#include "core.hpp"

namespace culbm {
namespace core {

class Units {
public:
	// Derives the scales from one length x, velocity u, density rho and
	// temperature T given both in lattice units and in SI units.
	void set_m_kg_s_K(const float x, const float u, const float rho, const float T, const float si_x, const float si_u, const float si_rho, const float si_T) {
		metre = si_x/x;
		kilogram = si_rho/rho*cb(metre);
		second = u/si_u*metre;
		kelvin = si_T/T;
		kelvin_offset = 0.0f;
		report();
	}
	// Sets the scales directly.
	void set_m_kg_s_K(const float m, const float kg, const float s, const float K) {
		metre = m;
		kilogram = kg;
		second = s;
		kelvin = K;
		kelvin_offset = 0.0f;
		report();
	}
	// Keeps the temperature scale and shifts the offset so that lattice T_ref maps to si_T_ref.
	void set_temperature_reference(const float T_ref, const float si_T_ref) {
		kelvin_offset = si_T_ref-T_ref*kelvin;
	}

	// SI -> lattice
	float x(const float si_x) const { return si_x/metre; } // length [m]
	ulong t(const float si_t) const { return to_ulong(si_t/second); } // time [s], rounded to whole steps
	float u(const float si_u) const { return si_u*second/metre; } // velocity [m/s]
	float nu(const float si_nu) const { return si_nu*second/sq(metre); } // kinematic viscosity [m^2/s]
	float g(const float si_g) const { return si_g/metre*sq(second); } // acceleration [m/s^2]
	float T(const float si_T) const { return (si_T-kelvin_offset)/kelvin; } // absolute temperature [K]
	float dT(const float si_dT) const { return si_dT/kelvin; } // temperature difference [K]
	float alpha(const float si_alpha) const { return si_alpha*second/sq(metre); } // thermal diffusivity [m^2/s]
	float beta(const float si_beta) const { return si_beta*kelvin; } // thermal expansion coefficient [1/K]

	// lattice -> SI
	float si_x(const uint x) const { return (float)x*metre; }
	float si_x(const float x) const { return x*metre; }
	float si_t(const ulong t) const { return (float)t*second; }
	float si_u(const float u) const { return u*metre/second; }
	float si_rho(const float rho) const { return rho*kilogram/cb(metre); }
	float si_nu(const float nu) const { return nu*sq(metre)/second; }
	float si_F(const float F) const { return F*kilogram*metre/sq(second); } // force [kg*m/s^2]
	float si_T(const float T) const { return T*kelvin+kelvin_offset; }
	float si_dT(const float dT) const { return dT*kelvin; }

private:
	float metre = 1.0f, kilogram = 1.0f, second = 1.0f, kelvin = 1.0f, kelvin_offset = 0.0f;
	void report() const {
		log_info("Unit Conversion: 1 cell = "+to_string(1000.0f*si_x(1.0f), 3u)+" mm, 1 s = "+to_string(t(1.0f))+" time steps");
	}
};

} // namespace core
} // namespace culbm

using culbm::core::Units;
extern Units units; // global solver scale, defined in solver/lattice_config.cpp
