#pragma once

#include "surface_ibm_config.hpp"
#include "core/core.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace culbm::surface_ibm {

struct Marker {
    float3 position_lu{};
    float3 wall_velocity_lu{};
    float3 normal{};
    float area_lu2 = 0.0f;
    // Reserved for future loader-provided physical body metadata. Mesh does
    // not preserve STL solid/attribute IDs, so generated markers currently use 0.
    std::uint32_t body_id = 0;
    std::uint32_t source_triangle = 0;
    std::uint64_t global_id = 0;
};

struct GeometryReport {
    std::size_t input_triangles = 0;
    std::size_t valid_triangles = 0;
    std::size_t degenerate_triangles = 0;
    std::size_t duplicate_triangles = 0;
    std::size_t coincident_union_pairs = 0;
    std::size_t internal_triangles = 0;
    std::size_t ambiguous_triangles = 0;
    std::size_t retained_triangles = 0;
    std::size_t internal_patches = 0;
    std::size_t ambiguous_patches = 0;
    std::size_t connected_components = 0;
    std::size_t watertight_components = 0;
    std::size_t open_edges = 0;
    std::size_t nonmanifold_edges = 0;
    std::size_t markers = 0;
    double input_area_lu2 = 0.0;
    double valid_area_lu2 = 0.0;
    double retained_area_lu2 = 0.0;
    double marker_area_lu2 = 0.0;
    std::vector<std::string> warnings;
};

// Clip quadrature, not the solid used for exterior-union classification.
// Half-open ownership prevents a horizontal facet on an interface from
// being represented by two levels. The compact ground can exclude its
// lower plane because that outer wall retains halfway bounce-back.
struct GeometrySlab {
    double lower_z_lu = 0.0;
    double upper_z_lu = 0.0;
    bool include_lower_plane = true;
};

bool build_surface_ibm_geometry(const Mesh& mesh,
                                const Config& config,
                                std::vector<Marker>& markers,
                                GeometryReport& report,
                                std::string& error,
                                const GeometrySlab* slab = nullptr);

}  // namespace culbm::surface_ibm
