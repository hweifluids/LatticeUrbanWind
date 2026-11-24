#include "geometry.hpp"
#include "context.hpp"
#include "memory_fit.hpp"
#include "report.hpp"
#include "../run/console_log.hpp"
#include "../solver/field_set.hpp"
#include "../solver/field_view.hpp"

#include <limits>

namespace culbm::cases {

using run::now_str;

namespace {

std::uint64_t saturated_add_mb(const std::uint64_t left, const std::uint64_t right) {
    return right > std::numeric_limits<std::uint64_t>::max() - left
        ? std::numeric_limits<std::uint64_t>::max()
        : left + right;
}

} // namespace

std::filesystem::path select_culbm_geometry_candidate(
    const std::filesystem::path& project_directory,
    const std::string& case_name,
    const bool profile_generation) {
    const std::filesystem::path geometry_dir =
        project_directory / "proj_temp";
    if (!std::filesystem::exists(geometry_dir)) {
        return std::filesystem::path();
    }
    const std::filesystem::path dem_pf_candidate =
        geometry_dir / (case_name + "_DEM_PF.stl");
    const std::filesystem::path dg_candidate =
        geometry_dir / (case_name + "_DG.stl");
    if (profile_generation &&
        std::filesystem::exists(dem_pf_candidate) &&
        std::filesystem::is_regular_file(dem_pf_candidate)) {
        return dem_pf_candidate;
    }
    if (std::filesystem::exists(dg_candidate) &&
        std::filesystem::is_regular_file(dg_candidate)) {
        return dg_candidate;
    }
    if (profile_generation) {
        const std::string dem_pf_suffix = "_DEM_PF.stl";
        for (const auto& entry :
             std::filesystem::directory_iterator(geometry_dir)) {
            if (!entry.is_regular_file()) continue;
            const std::string filename =
                entry.path().filename().string();
            if (filename.size() >= dem_pf_suffix.size() &&
                filename.substr(
                    filename.size() - dem_pf_suffix.size()) ==
                    dem_pf_suffix) {
                return entry.path();
            }
        }
    }
    const std::string dg_suffix = "_DG.stl";
    for (const auto& entry :
         std::filesystem::directory_iterator(geometry_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string filename =
            entry.path().filename().string();
        if (filename.size() >= dg_suffix.size() &&
            filename.substr(filename.size() - dg_suffix.size()) ==
                dg_suffix) {
            return entry.path();
        }
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(geometry_dir)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".stl") {
            return entry.path();
        }
    }
    return std::filesystem::path();
}

void load_case_geometry(CaseContext& ctx) {
    CaseSettings& deck = ctx.deck;
    CaseDomain& domain = ctx.domain;
    CaseGeometry& geometry = ctx.geometry;
    print_section_title("LOADING GEOMETRY AND VOXELIZE");
    print_kv_row("Geometry", "Loading buildings as geometry, meshing...");

    deck.culbm_output_global_requested =
        deck.culbm_output_global_vtk || deck.culbm_output_global_netcdf || deck.culbm_output_global_vtkhdf;
    deck.culbm_output_agl_requested =
        deck.culbm_output_agl_vtk || deck.culbm_output_agl_netcdf;
    deck.culbm_output_physical_requested =
        deck.culbm_output_global_requested || deck.culbm_output_agl_requested;
#if !CULBM_HAS_VTKHDF
    if (deck.culbm_output_global_vtkhdf) {
        println("| ERROR: VTKHDF output was requested but this CULBM build has no HDF5 support. |");
        println("|        Reconfigure with CULBM_ENABLE_VTKHDF=ON and an HDF5 C package.        |");
        wait(); exit(-1);
    }
#endif

    {
        const std::filesystem::path project_directory(deck.parent);
        const std::filesystem::path dir = project_directory / "proj_temp";
        if (!std::filesystem::exists(dir)) {
            println("ERROR: directory not found: " + dir.string());
            wait(); exit(-1);
        }
        const std::filesystem::path selected_geometry =
            select_culbm_geometry_candidate(
                project_directory,
                deck.caseName,
                deck.profile_generation);
        if (selected_geometry.empty()) {
            const std::filesystem::path dem_pf_candidate =
                dir / (deck.caseName + "_DEM_PF.stl");
            const std::filesystem::path dg_candidate =
                dir / (deck.caseName + "_DG.stl");
            if (deck.profile_generation) {
                println("ERROR: no STL file. Tried " + dem_pf_candidate.string() + ", *_DEM_PF.stl, " +
                        dg_candidate.string() + ", *_DG.stl, then *.stl under " + dir.string());
            }
            else {
                println("ERROR: no STL file. Tried " + dg_candidate.string() + ", *_DG.stl, then *.stl under " + dir.string());
            }
            wait(); exit(-1);
        }
        geometry.stl_path = selected_geometry.string();
    }
    if (domain.cuda_ground_probe_mesh &&
        std::filesystem::path(domain.cuda_ground_probe_path).lexically_normal() ==
            std::filesystem::path(geometry.stl_path).lexically_normal()) {
        geometry.mesh.reset(domain.cuda_ground_probe_mesh.release());
    }
    else if (deck.preloaded_geometry_mesh &&
             deck.preloaded_geometry_path.lexically_normal() ==
                 std::filesystem::path(geometry.stl_path).lexically_normal()) {
        geometry.mesh.reset(deck.preloaded_geometry_mesh.release());
    }
    else {
        geometry.mesh.reset(read_stl(geometry.stl_path));
    }

    print_kv_row("Time code", "[" + now_str() + "]");

    if (!geometry.mesh) { println("ERROR: failed to load STL"); wait(); exit(-1); }
    geometry.voxelization_transient_mb =
        voxelization_transient_vram_mb_per_device(
            geometry.mesh->triangle_number);
    geometry.uniform_core_mb =
        lattice_vram_mib(
            domain.lbm_N.x, domain.lbm_N.y, domain.lbm_N.z, deck.Dx, deck.Dy, deck.Dz);
    geometry.uniform_voxel_peak_mb =
        saturated_add_mb(geometry.uniform_core_mb, geometry.voxelization_transient_mb);
    geometry.nested_voxel_peak_mb =
        domain.cuda_vertical_nesting_plan.enabled
            ? saturated_add_mb(
                domain.nested_full_height_raster_core_mb,
                geometry.voxelization_transient_mb)
            : 0u;
    geometry.exact_mesh_peak_mb = std::max(
        (std::uint64_t)domain.gpu_memory_estimate.total_per_device_mb,
        std::max(geometry.uniform_voxel_peak_mb, geometry.nested_voxel_peak_mb));
    if (geometry.exact_mesh_peak_mb >
        (std::uint64_t)domain.gpu_memory_estimate.total_per_device_mb) {
        domain.gpu_memory_estimate.total_per_device_mb =
            (uint)std::min<std::uint64_t>(
                geometry.exact_mesh_peak_mb, std::numeric_limits<uint>::max());
        domain.gpu_memory_estimate.extra_per_device_mb =
            geometry.exact_mesh_peak_mb >= domain.gpu_memory_estimate.core_per_device_mb
                ? (uint)std::min<std::uint64_t>(
                    geometry.exact_mesh_peak_mb -
                        domain.gpu_memory_estimate.core_per_device_mb,
                    std::numeric_limits<uint>::max())
                : std::numeric_limits<uint>::max();
        const std::uint64_t domain_count =
            (std::uint64_t)deck.Dx * (std::uint64_t)deck.Dy * (std::uint64_t)deck.Dz;
        const std::uint64_t project_peak_mb =
            domain_count != 0u &&
                    geometry.exact_mesh_peak_mb >
                        std::numeric_limits<std::uint64_t>::max() /
                            domain_count
                ? std::numeric_limits<std::uint64_t>::max()
                : geometry.exact_mesh_peak_mb * domain_count;
        domain.gpu_memory_estimate.total_project_mb =
            (uint)std::min<std::uint64_t>(
                project_peak_mb, std::numeric_limits<uint>::max());
    }
    print_kv_row(
        "GPU STL peak",
        to_string((ulong)geometry.exact_mesh_peak_mb) +
        " MB/device (voxel transient " +
        to_string((ulong)geometry.voxelization_transient_mb) + " MB)");
    if (deck.gpu_memory_mesh_control_requested &&
        geometry.exact_mesh_peak_mb >
            (std::uint64_t)usable_gpu_memory_request_mb(deck.memory)) {
        println("| ERROR: STL-aware GPU peak exceeds the requested per-device budget.          |");
        println("|        Required " + to_string((ulong)geometry.exact_mesh_peak_mb) +
            " MB including voxelization, usable " +
            to_string((ulong)usable_gpu_memory_request_mb(deck.memory)) +
            " MB after safety reserves; simplify STL or increase gpu_memory. |");
        wait();
        exit(-1);
    }
    geometry.stl_size_si = geometry.mesh->get_bounding_box_size();
    geometry.stl_min_si = geometry.mesh->pmin;
    geometry.stl_max_si = geometry.mesh->pmax;
    geometry.domain_min_si = float3(
        units.si_x(0.5f - 0.5f * (float)domain.lbm_N.x),
        units.si_x(0.5f - 0.5f * (float)domain.lbm_N.y),
        units.si_x(0.5f - 0.5f * (float)domain.lbm_N.z));
    // Integer lattice indices are node centres; the STL minimum lies
    // halfway between the boundary node and the first interior node.
    // Use the inverse of the geometry transform for every SI output.
    vtk_origin_shift = geometry.stl_min_si - geometry.domain_min_si - float3(units.si_x(0.5f));
    if (domain.cuda_ground_compaction_active) vtk_origin_shift.z += domain.cuda_ground_trim_si;
    geometry.target_lbm_x = units.x(deck.si_size.x);
    geometry.scale_geom = geometry.target_lbm_x / geometry.stl_size_si.x;
    geometry.mesh->scale(geometry.scale_geom);
    geometry.mesh->translate(float3(0.5f - geometry.mesh->pmin.x, 0.5f - geometry.mesh->pmin.y, 0.5f - geometry.mesh->pmin.z));
    if (domain.cuda_ground_compaction_active) {
        geometry.mesh->translate(float3(0.0f, 0.0f, -(float)domain.cuda_ground_geometry_shift_cells));
        print_kv_row("CUDA ground", "geometry shifted down by " + to_string((ulong)domain.cuda_ground_geometry_shift_cells) + " lattice cells");
    }
}

void build_surface_ibm_markers(CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    CaseDomain& domain = ctx.domain;
    CaseGeometry& geometry = ctx.geometry;
    if (deck.surface_ibm_settings.enabled && !domain.cuda_vertical_nesting_plan.enabled) {
        std::string surface_ibm_geometry_error;
        if (!culbm::surface_ibm::build_surface_ibm_geometry(
                *geometry.mesh,
                deck.surface_ibm_settings,
                geometry.surface_ibm_markers,
                geometry.surface_ibm_geometry_report,
                surface_ibm_geometry_error)) {
            println("| ERROR: SurfaceIBM marker construction failed: " +
                    surface_ibm_geometry_error + " |");
            wait();
            exit(-1);
        }
        if (geometry.surface_ibm_markers.empty()) {
            println("| ERROR: SurfaceIBM is enabled, but the transformed STL produced no markers.  |");
            println("|        Check geometry policy, STL topology, and marker-spacing settings.     |");
            wait();
            exit(-1);
        }

        // Roma's three-point kernel needs one neighbor on either side of the
        // nearest cell. Keep that complete support away from the global outer
        // cell, whose existing inlet/outlet/side boundary conditions remain in
        // charge. This also prevents double application at STL base/envelope
        // faces that were scaled exactly onto the computational box.
        const auto ibm_support_fits = [](const float coordinate,
                                         const uint extent) -> bool {
            if (!std::isfinite(coordinate) || extent < 5u) return false;
            const long long center =
                (long long)std::floor((double)coordinate + 0.5);
            return center >= 2ll &&
                   center <= (long long)extent - 3ll;
        };
        const auto first_dropped_marker = std::remove_if(
            geometry.surface_ibm_markers.begin(),
            geometry.surface_ibm_markers.end(),
            [&](const culbm::surface_ibm::Marker& marker) {
                // Geometry retraction moves a retained marker into the solid.
                // Test both the original surface centroid and the retracted
                // location so retraction cannot pull an artificial outer-box
                // facet just far enough inward to bypass ownership filtering.
                const float3 surface_position =
                    marker.position_lu +
                    deck.surface_ibm_settings.retraction_lu * marker.normal;
                return !ibm_support_fits(surface_position.x, domain.lbm_N.x) ||
                       !ibm_support_fits(surface_position.y, domain.lbm_N.y) ||
                       !ibm_support_fits(surface_position.z, domain.lbm_N.z) ||
                       !ibm_support_fits(marker.position_lu.x, domain.lbm_N.x) ||
                       !ibm_support_fits(marker.position_lu.y, domain.lbm_N.y) ||
                       !ibm_support_fits(marker.position_lu.z, domain.lbm_N.z);
            });
        geometry.surface_ibm_dropped_external_support = (std::size_t)(
            geometry.surface_ibm_markers.end() - first_dropped_marker);
        geometry.surface_ibm_markers.erase(first_dropped_marker, geometry.surface_ibm_markers.end());
        for (const culbm::surface_ibm::Marker& marker : geometry.surface_ibm_markers) {
            geometry.surface_ibm_active_marker_area_lu2 += (double)marker.area_lu2;
        }
        if (geometry.surface_ibm_markers.empty()) {
            println("| ERROR: SurfaceIBM has no marker with complete in-domain Roma support.       |");
            println("|        Outer-box STL faces remain owned by the existing domain boundaries.  |");
            println("|        Move/trim the immersed surface away from the global outer cells.      |");
            wait();
            exit(-1);
        }

        print_kv_row(
            "SurfaceIBM",
            "direct-forcing / " +
                string(culbm::surface_ibm::delta_kernel_name(
                    deck.surface_ibm_settings.delta_kernel)) +
                ", spacing=" +
                to_string(deck.surface_ibm_settings.marker_spacing_lu, 4u) +
                " lu, iterations=" +
                to_string((ulong)deck.surface_ibm_settings.forcing_iterations));
        print_kv_row(
            "IBM markers",
            "active/generated=" +
                to_string((ulong)geometry.surface_ibm_markers.size()) + "/" +
                to_string((ulong)geometry.surface_ibm_geometry_report.markers) +
                ", dropped outer support=" +
                to_string((ulong)geometry.surface_ibm_dropped_external_support));
        print_kv_row(
            "IBM triangles",
            "input=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.input_triangles) +
                ", valid=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.valid_triangles) +
                ", retained=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.retained_triangles));
        print_kv_row(
            "IBM filtered",
            "degenerate=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.degenerate_triangles) +
                ", duplicate=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.duplicate_triangles) +
                ", coincident union pairs=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.coincident_union_pairs) +
                ", internal=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.internal_triangles) +
                ", ambiguous=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.ambiguous_triangles));
        print_kv_row(
            "IBM patches",
            "internal=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.internal_patches) +
                ", ambiguous=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.ambiguous_patches));
        print_kv_row(
            "IBM topology",
            "components=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.connected_components) +
                ", watertight=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.watertight_components) +
                ", open/nonmanifold edges=" +
                to_string((ulong)geometry.surface_ibm_geometry_report.open_edges) + "/" +
                to_string((ulong)geometry.surface_ibm_geometry_report.nonmanifold_edges));
        print_kv_row(
            "IBM area check",
            "input/retained/generated/active=" +
                to_string(geometry.surface_ibm_geometry_report.input_area_lu2, 6u) + "/" +
                to_string(geometry.surface_ibm_geometry_report.retained_area_lu2, 6u) + "/" +
                to_string(geometry.surface_ibm_geometry_report.marker_area_lu2, 6u) +
                "/" + to_string(geometry.surface_ibm_active_marker_area_lu2, 6u) +
                " lu^2");
        for (const std::string& warning : geometry.surface_ibm_geometry_report.warnings) {
            print_kv_row("IBM warning", warning);
        }

        // Portable Runtime owns position(3), interpolation partial(6), and
        // force(3) buffers.  The single-GPU device-resident path adds wall
        // velocity(3), volume(1), and coupling(1).  Budget the larger 17-float
        // layout here.  Runtime repeats the check with exact fragment and
        // active-cell counts plus live free memory at attach.
        constexpr std::uint64_t ibm_fragment_bytes_per_marker =
            17ull * sizeof(float);
        const std::uint64_t marker_count =
            (std::uint64_t)geometry.surface_ibm_markers.size();
        if (marker_count >
            std::numeric_limits<std::uint64_t>::max() /
                ibm_fragment_bytes_per_marker) {
            println("| ERROR: SurfaceIBM marker-buffer size overflow.                              |");
            wait();
            exit(-1);
        }
        const std::uint64_t marker_buffer_bytes =
            marker_count * ibm_fragment_bytes_per_marker;
        const std::uint64_t marker_buffer_mb =
            marker_buffer_bytes / 1048576ull +
            (marker_buffer_bytes % 1048576ull != 0ull ? 1ull : 0ull);
        const std::uint64_t ibm_runtime_peak_mb =
            saturated_add_mb(geometry.uniform_core_mb, marker_buffer_mb);
        print_kv_row(
            "IBM GPU budget",
            "<= " + to_string((ulong)marker_buffer_mb) +
                " MB marker buffers/device, runtime peak <= " +
                to_string((ulong)ibm_runtime_peak_mb) + " MB/device");
        if (ibm_runtime_peak_mb >
            (std::uint64_t)domain.gpu_memory_estimate.total_per_device_mb) {
            domain.gpu_memory_estimate.total_per_device_mb =
                (uint)std::min<std::uint64_t>(
                    ibm_runtime_peak_mb, std::numeric_limits<uint>::max());
            domain.gpu_memory_estimate.extra_per_device_mb =
                ibm_runtime_peak_mb >= domain.gpu_memory_estimate.core_per_device_mb
                    ? (uint)std::min<std::uint64_t>(
                        ibm_runtime_peak_mb-
                            domain.gpu_memory_estimate.core_per_device_mb,
                        std::numeric_limits<uint>::max())
                    : 0u;
            const std::uint64_t domain_count =
                (std::uint64_t)deck.Dx*(std::uint64_t)deck.Dy*(std::uint64_t)deck.Dz;
            const std::uint64_t project_peak_mb =
                domain_count!=0u && ibm_runtime_peak_mb>
                    std::numeric_limits<std::uint64_t>::max()/domain_count
                    ? std::numeric_limits<std::uint64_t>::max()
                    : ibm_runtime_peak_mb*domain_count;
            domain.gpu_memory_estimate.total_project_mb =
                (uint)std::min<std::uint64_t>(
                    project_peak_mb, std::numeric_limits<uint>::max());
        }
        if (deck.gpu_memory_mesh_control_requested &&
            ibm_runtime_peak_mb >
                (std::uint64_t)usable_gpu_memory_request_mb(deck.memory)) {
            println("| ERROR: SurfaceIBM runtime buffers exceed the requested per-device budget.  |");
            println("|        Reduce marker density/grid size or increase gpu_memory.              |");
            wait();
            exit(-1);
        }
    }
}

void print_geometry_summary(const CaseContext& ctx) {
    const CaseGeometry& geometry = ctx.geometry;
    print_kv_row("Geometry STL", geometry.stl_path);
    print_kv_row("STL bounds SI", "x=[" + to_string(geometry.stl_min_si.x, 3u) + ", " + to_string(geometry.stl_max_si.x, 3u) +
            "], y=[" + to_string(geometry.stl_min_si.y, 3u) + ", " + to_string(geometry.stl_max_si.y, 3u) +
            "], z=[" + to_string(geometry.stl_min_si.z, 3u) + ", " + to_string(geometry.stl_max_si.z, 3u) + "]");
    print_kv_row("Geometry", "scaled by " + to_string(geometry.scale_geom, 4u) + ", ready for voxelization");
}

} // namespace culbm::cases
