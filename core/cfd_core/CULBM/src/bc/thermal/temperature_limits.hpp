#pragma once

// Temperature constants of the LUW boundary data (Kelvin).

namespace culbm::bc {

constexpr float k_temperature_ref_kelvin = 293.15f; // 20 C -> T_lbm = 1.0
constexpr float k_temperature_min_kelvin = 223.15f; // -50 C
constexpr float k_temperature_max_kelvin = 343.15f; // 70 C
constexpr float k_boussinesq_relative_temperature_warning = 0.05f;

} // namespace culbm::bc
