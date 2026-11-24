#pragma once

// Geometry of a case: the STL chosen from proj_temp, its memory peak during
// voxelization, its transform into lattice coordinates (and the inverse SI
// origin of every output), and the SurfaceIBM markers built from it.

#include "../core/core.hpp"
#include "../surface_ibm_geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace culbm::cases {

struct CaseContext;

// proj_temp/<case>_DEM_PF.stl (profile decks), <case>_DG.stl, then any
// *_DEM_PF.stl (profile decks), *_DG.stl and *.stl; empty when none exists.
std::filesystem::path select_culbm_geometry_candidate(
    const std::filesystem::path& project_directory,
    const std::string& case_name,
    bool profile_generation);

struct CaseGeometry {
    std::string stl_path;
    std::unique_ptr<Mesh> mesh; // lattice coordinates once loaded
    std::uint64_t voxelization_transient_mb = 0u;
    std::uint64_t uniform_core_mb = 0u;
    std::uint64_t uniform_voxel_peak_mb = 0u;
    std::uint64_t nested_voxel_peak_mb = 0u;
    std::uint64_t exact_mesh_peak_mb = 0u;
    float3 stl_size_si = float3(0.0f);
    float3 stl_min_si = float3(0.0f);
    float3 stl_max_si = float3(0.0f);
    float3 domain_min_si = float3(0.0f);
    float target_lbm_x = 0.0f;
    float scale_geom = 1.0f;

    std::vector<culbm::surface_ibm::Marker> surface_ibm_markers;
    culbm::surface_ibm::GeometryReport surface_ibm_geometry_report;
    std::size_t surface_ibm_dropped_external_support = 0u;
    double surface_ibm_active_marker_area_lu2 = 0.0;
};

// Physical output requests, STL selection and loading, memory peak checks,
// SI output origin (vtk_origin_shift) and the transform into the lattice.
void load_case_geometry(CaseContext& ctx);
// SurfaceIBM markers of the transformed STL with complete in-domain support
// (uniform lattices; nested levels build their own).
void build_surface_ibm_markers(CaseContext& ctx);
// Geometry rows of the console report.
void print_geometry_summary(const CaseContext& ctx);

} // namespace culbm::cases
