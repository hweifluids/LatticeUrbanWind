#pragma once

namespace luw::cfd {

struct ThermalCouplingDecision {
    bool buoyancy_requested = false;
    bool thermal_boundary_available = false;
    bool buoyancy_active = false;
    bool fallback_applied = false;
};

// A Boussinesq force is meaningful only when setup can actually prescribe a
// temperature boundary. Temperature transport remains a separate concern:
// callers may still advect a supplied temperature field with buoyancy off.
constexpr ThermalCouplingDecision resolve_thermal_coupling(
    const bool buoyancy_requested,
    const bool thermal_boundary_available) noexcept {
    return ThermalCouplingDecision{
        buoyancy_requested,
        thermal_boundary_available,
        buoyancy_requested && thermal_boundary_available,
        buoyancy_requested && !thermal_boundary_available,
    };
}

} // namespace luw::cfd
