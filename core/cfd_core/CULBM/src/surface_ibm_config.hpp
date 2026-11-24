#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace culbm::surface_ibm {

enum class DeltaKernel {
    Roma3Point,
};

enum class GeometryPolicy {
    ExteriorUnion,
    AllTriangles,
};

struct Config {
    bool enabled = false;
    DeltaKernel delta_kernel = DeltaKernel::Roma3Point;
    GeometryPolicy geometry_policy = GeometryPolicy::ExteriorUnion;
    float marker_spacing_lu = 0.75f;
    unsigned forcing_iterations = 8;
    float relaxation = 1.0f;
    float residual_tolerance = 1.0e-3f;
    float retraction_lu = 0.5f;
    float support_thickness_lu = 0.75f;
    std::uint64_t max_markers = 5000000ULL;
    bool strict_geometry = false;
    unsigned diagnostics_interval = 0;
    // Internal slab bounds, inclusive, in global native lattice coordinates.
    // -1/-1 retains the ordinary full Roma support. Nested levels set these
    // before construction so interpolation and spreading never force ghosts.
    int support_z_min = -1;
    int support_z_max = -1;
};

using RawDeck = std::unordered_map<std::string, std::string>;

bool parse_config(const RawDeck& raw, Config& config, std::string& error);
bool validate_config(const Config& config, std::string& error);

const char* delta_kernel_name(DeltaKernel kernel);
const char* geometry_policy_name(GeometryPolicy policy);

// Setup publishes a validated immutable snapshot before solver construction.
void set_config(const Config& config);
Config get_config();

// A geometry-only raster or a native slab captures its own options at LBM
// construction; concurrent callers keep their published deck configuration.
class ScopedConfig {
public:
    explicit ScopedConfig(const Config& config);
    ~ScopedConfig();
    ScopedConfig(const ScopedConfig&) = delete;
    ScopedConfig& operator=(const ScopedConfig&) = delete;
private:
    Config local_;
    const Config* previous_;
};

}  // namespace culbm::surface_ibm
