#pragma once

#include "surface_ibm_config.hpp"
#include "surface_ibm_geometry.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class LBM;

namespace culbm::surface_ibm {

// Collective guard used by LBM::attach_surface_ibm before any rank-local
// rejection or runtime allocation.  In distributed mode every rank must call
// attachment together, with one identical configuration, attachment state and
// deterministic marker set.
void verify_attach_preflight(
    const Config& config,
    const std::vector<Marker>& markers,
    bool initialized,
    bool already_attached,
    bool storage_enabled);

struct Statistics {
    bool valid = false;
    std::uint64_t markers = 0u;
    std::uint64_t fragments = 0u;
    std::uint64_t active_cells = 0u;
    std::uint64_t cleared_solid_cells = 0u;
    bool device_resident = false;
    unsigned iterations = 0u;
    bool converged = false;
    std::uint64_t nonconverged_steps = 0u;
    float minimum_coupling = 0.0f;
    float maximum_coupling = 0.0f;
    float max_slip_lu = 0.0f;
    float peak_max_slip_lu = 0.0f;
    float rms_slip_lu = 0.0f;
    float3 reaction_force_lu{};
    float3 reaction_torque_lu{};
};

// Static surface IBM runtime.  One local GPU uses an active-cell sparse,
// device-resident forcing loop.  Multi-domain/MPI runs partition every marker
// into owned fragments and reduce interpolation partials across host/ranks,
// then spread only to owned Eulerian cells.  This avoids assuming that the
// ordinary one-cell LBM halo can perform additive force communication.
class Runtime {
public:
    Runtime(LBM& lbm, const std::vector<Marker>& markers,
            const Config& config);
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void prepare_step();
    Statistics statistics() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace culbm::surface_ibm
