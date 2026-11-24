#include "surface_ibm.hpp"

#include "cuda_mpi.hpp"
#include "solver/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace culbm::surface_ibm {
namespace {

constexpr std::uint64_t reduction_marker_chunk = 1u << 18u;
constexpr float minimum_density = 1.0e-12f;
constexpr float partition_tolerance = 2.0e-4f;
static_assert(gpu::default_block_size == 64u,
              "SurfaceIBM block reductions require 64 CUDA threads.");

bool finite3(const float3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

std::uint32_t float_bits(const float value) {
    std::uint32_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void hash_word(std::uint64_t& hash, const std::uint64_t value) {
    for(unsigned byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (8u*byte)) & 0xFFu;
        hash *= 1099511628211ull;
    }
}

std::uint64_t marker_payload_hash(const Marker& marker) {
    std::uint64_t hash = 1469598103934665603ull;
    hash_word(hash, marker.global_id);
    hash_word(hash, (std::uint64_t)marker.body_id);
    hash_word(hash, (std::uint64_t)marker.source_triangle);
    hash_word(hash, (std::uint64_t)float_bits(marker.position_lu.x));
    hash_word(hash, (std::uint64_t)float_bits(marker.position_lu.y));
    hash_word(hash, (std::uint64_t)float_bits(marker.position_lu.z));
    hash_word(hash, (std::uint64_t)float_bits(marker.wall_velocity_lu.x));
    hash_word(hash, (std::uint64_t)float_bits(marker.wall_velocity_lu.y));
    hash_word(hash, (std::uint64_t)float_bits(marker.wall_velocity_lu.z));
    hash_word(hash, (std::uint64_t)float_bits(marker.normal.x));
    hash_word(hash, (std::uint64_t)float_bits(marker.normal.y));
    hash_word(hash, (std::uint64_t)float_bits(marker.normal.z));
    hash_word(hash, (std::uint64_t)float_bits(marker.area_lu2));
    return hash;
}

struct MarkerSignature {
    std::uint64_t count = 0u;
    std::uint64_t xor_hash = 0u;
    std::uint64_t sum_hash = 0u;
};

MarkerSignature marker_signature(const std::vector<Marker>& markers) {
    MarkerSignature signature;
    signature.count = (std::uint64_t)markers.size();
    for(const Marker& marker : markers) {
        const std::uint64_t hash = marker_payload_hash(marker);
        // Both reductions are commutative so rank-local input ordering does
        // not matter.  The canonical global_id sort still occurs before use.
        signature.xor_hash ^= hash;
        signature.sum_hash +=
            hash*0x9E3779B97F4A7C15ull+0xD1B54A32D192ED03ull;
    }
    return signature;
}

std::array<std::uint64_t, 14> configuration_words(const Config& config) {
    return {{
        config.enabled ? 1u : 0u,
        (std::uint64_t)config.delta_kernel,
        (std::uint64_t)config.geometry_policy,
        (std::uint64_t)float_bits(config.marker_spacing_lu),
        (std::uint64_t)config.forcing_iterations,
        (std::uint64_t)float_bits(config.relaxation),
        (std::uint64_t)float_bits(config.residual_tolerance),
        (std::uint64_t)float_bits(config.retraction_lu),
        (std::uint64_t)float_bits(config.support_thickness_lu),
        config.max_markers,
        config.strict_geometry ? 1u : 0u,
        (std::uint64_t)config.diagnostics_interval,
        (std::uint64_t)(std::int64_t)config.support_z_min,
        (std::uint64_t)(std::int64_t)config.support_z_max,
    }};
}

int nearest_grid_coordinate(const float value) {
    return (int)std::floor((double)value + 0.5);
}

// Wall-modelled SurfaceIBM (culbm_wall_model_ibm).  Must match the device
// defines kcfg::wm_ibm_min_weight/kcfg::wm_ibm_cos_h and surface_ibm_wm_factor().
constexpr float wall_model_min_exterior_weight = 0.01f;
constexpr float wall_model_upward_cosine = 0.70710678f;

// Use the resolved first-cell distance for subcell cuts, after combining
// all fragments so that the matching height is independent of decomposition.
float wall_model_factor(const Wall_Model_Lattice& wm, const float ut,
                        const float mean_distance, const bool upward) {
    const float h = std::max(mean_distance, Wall_Model_Lattice::first_cell_distance_lu);
    const bool smooth = upward ? wm.smooth_h : wm.smooth_v;
    if(!smooth) {
        const float z0 = upward ? wm.z0_h_lu : wm.z0_v_lu;
        const float r = wm.kappa/std::log1p(h/z0);
        return r*r*ut;
    }
    const float nu = wm.nu_lbm;
    if(h*ut <= wm.re_laminar*nu) return nu/h;
    float utau = std::sqrt(nu*ut/h);
    for(int k = 0; k < 6; ++k) {
        const float L = std::log(h*utau/nu)+wm.log_E;
        utau = std::max(utau-(utau*L-wm.kappa*ut)/(L+1.0f), 0.5f*utau);
    }
    return utau*utau/ut;
}

float roma3_phi_host(const float r) {
    const float a = std::fabs(r);
    if(a >= 1.5f) return 0.0f;
    if(a <= 0.5f)
        return (1.0f+std::sqrt(std::max(0.0f, 1.0f-3.0f*a*a)))/3.0f;
    const float q = 1.0f-a;
    return (5.0f-3.0f*a-
            std::sqrt(std::max(0.0f, 1.0f-3.0f*q*q)))/6.0f;
}

struct OwnedBox {
    int x0 = 0, x1 = -1;
    int y0 = 0, y1 = -1;
    int z0 = 0, z1 = -1;

    bool intersects_support(const Marker& marker) const {
        const int cx = nearest_grid_coordinate(marker.position_lu.x);
        const int cy = nearest_grid_coordinate(marker.position_lu.y);
        const int cz = nearest_grid_coordinate(marker.position_lu.z);
        return cx + 1 >= x0 && cx - 1 <= x1 &&
               cy + 1 >= y0 && cy - 1 <= y1 &&
               cz + 1 >= z0 && cz - 1 <= z1;
    }
};

OwnedBox owned_box(const culbm::solver::Subdomain& domain, const Config& config) {
    const int hx = (int)domain.placement().halo[0];
    const int hy = (int)domain.placement().halo[1];
    const int hz = (int)domain.placement().halo[2];
    OwnedBox box;
    box.x0 = domain.origin(0u) + hx;
    box.x1 = domain.origin(0u) + (int)domain.extent(0u) - hx - 1;
    box.y0 = domain.origin(1u) + hy;
    box.y1 = domain.origin(1u) + (int)domain.extent(1u) - hy - 1;
    box.z0 = domain.origin(2u) + hz;
    box.z1 = domain.origin(2u) + (int)domain.extent(2u) - hz - 1;
    if(config.support_z_min >= 0) {
        box.z0 = std::max(box.z0, config.support_z_min);
        box.z1 = std::min(box.z1, config.support_z_max);
    }
    return box;
}

std::uint64_t global_cell_index(
    const int x, const int y, const int z,
    const uint Nx, const uint Ny) {
    return (std::uint64_t)x+
           ((std::uint64_t)y+(std::uint64_t)z*(std::uint64_t)Ny)*
               (std::uint64_t)Nx;
}

uint global_domain_for_cell(
    const int x, const int y, const int z,
    const culbm::solver::Solver& solver) {
    const uint nx = solver.grid().size(0u)/solver.grid().split(0u);
    const uint ny = solver.grid().size(1u)/solver.grid().split(1u);
    const uint nz = solver.grid().size(2u)/solver.grid().split(2u);
    const uint dx = std::min((uint)x/nx, solver.grid().split(0u)-1u);
    const uint dy = std::min((uint)y/ny, solver.grid().split(1u)-1u);
    const uint dz = std::min((uint)z/nz, solver.grid().split(2u)-1u);
    return dx+(dy+dz*solver.grid().split(1u))*solver.grid().split(0u);
}

std::vector<ulong> build_active_cells(
    const culbm::solver::Subdomain& domain,
    const std::vector<Marker>& markers,
    const std::vector<std::uint64_t>& marker_indices,
    const Config& config) {
    const OwnedBox box = owned_box(domain, config);
    const std::size_t local_cell_count = (std::size_t)domain.cells();
    const std::size_t maximum_support_count =
        marker_indices.size() <=
                std::numeric_limits<std::size_t>::max()/27u
            ? 27u*marker_indices.size()
            : std::numeric_limits<std::size_t>::max();
    // Sorting a sparse list is faster and smaller for small surfaces.  Once
    // its worst-case storage exceeds a one-bit-per-cell bitmap, use the
    // bitmap instead so preprocessing memory cannot grow as 27*markers.
    const bool use_bitmap = maximum_support_count > local_cell_count/64u;
    std::vector<ulong> cells;
    std::vector<std::uint64_t> bitmap;
    if(use_bitmap) bitmap.assign((local_cell_count+63u)/64u, 0ull);
    else cells.reserve(maximum_support_count);

    const auto add_cell = [&](const ulong cell) {
        if(use_bitmap) {
            bitmap[(std::size_t)cell/64u] |=
                1ull<<((std::size_t)cell%64u);
        } else {
            cells.push_back(cell);
        }
    };
    for(const std::uint64_t marker_index : marker_indices) {
        const Marker& marker = markers[(std::size_t)marker_index];
        const int cx = nearest_grid_coordinate(marker.position_lu.x);
        const int cy = nearest_grid_coordinate(marker.position_lu.y);
        const int cz = nearest_grid_coordinate(marker.position_lu.z);
        for(int dz = -1; dz <= 1; ++dz) {
            const int gz = cz+dz;
            const float wz = roma3_phi_host(
                marker.position_lu.z-(float)gz);
            if(gz < box.z0 || gz > box.z1) continue;
            for(int dy = -1; dy <= 1; ++dy) {
                const int gy = cy+dy;
                const float wy = roma3_phi_host(
                    marker.position_lu.y-(float)gy);
                if(gy < box.y0 || gy > box.y1) continue;
                for(int dx = -1; dx <= 1; ++dx) {
                    const int gx = cx+dx;
                    const float w = roma3_phi_host(
                        marker.position_lu.x-(float)gx)*wy*wz;
                    if(w == 0.0f) continue;
                    if(gx < box.x0 || gx > box.x1) continue;
                    const ulong lx = (ulong)(gx-domain.origin(0u));
                    const ulong ly = (ulong)(gy-domain.origin(1u));
                    const ulong lz = (ulong)(gz-domain.origin(2u));
                    add_cell(lx+(ly+lz*(ulong)domain.extent(1u))*
                                 (ulong)domain.extent(0u));
                }
            }
        }
    }
    if(use_bitmap) {
        for(std::size_t word_index = 0u; word_index < bitmap.size();
            ++word_index) {
            const std::uint64_t word = bitmap[word_index];
            if(word==0ull) continue;
            for(std::size_t bit = 0u; bit < 64u; ++bit) {
                const std::size_t cell = 64u*word_index+bit;
                if(cell>=local_cell_count) break;
                if((word&(1ull<<bit))!=0ull) cells.push_back((ulong)cell);
            }
        }
    } else {
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    }
    return cells;
}

} // namespace

void verify_attach_preflight(
    const Config& config,
    const std::vector<Marker>& markers,
    const bool initialized,
    const bool already_attached,
    const bool storage_enabled) {
    constexpr std::size_t config_word_count = 14u;
    constexpr std::size_t state_word = 14u;
    constexpr std::size_t marker_count_word = 15u;
    constexpr std::size_t marker_xor_word = 16u;
    constexpr std::size_t marker_sum_word = 17u;
    std::array<std::uint64_t, 18> local{};
    const auto config_words = configuration_words(config);
    std::copy(config_words.begin(), config_words.end(), local.begin());
    local[state_word] =
        (initialized ? 1ull : 0ull) |
        (already_attached ? 2ull : 0ull) |
        (storage_enabled ? 4ull : 0ull);
    const MarkerSignature markers_signature = marker_signature(markers);
    local[marker_count_word] = markers_signature.count;
    local[marker_xor_word] = markers_signature.xor_hash;
    local[marker_sum_word] = markers_signature.sum_hash;

    if(culbm::mpi_is_distributed()) {
        std::vector<std::array<std::uint64_t, 18>> all;
        try {
            all.resize((std::size_t)culbm::mpi_size());
        } catch(const std::bad_alloc&) {
            // A rank-local failure before the first attachment collective
            // cannot be converted into a coordinated exception.  Follow the
            // solver's existing distributed allocation-failure contract.
            culbm::mpi_abort(
                "SurfaceIBM attach preflight host allocation failed");
        }
        culbm::mpi_allgather_equal_bytes(
            local.data(), sizeof(local), all.data());
        const auto& reference = all.front();
        for(std::size_t rank = 1u; rank < all.size(); ++rank) {
            if(!std::equal(
                    reference.begin(), reference.begin()+config_word_count,
                    all[rank].begin())) {
                throw std::runtime_error(
                    "SurfaceIBM requires the same complete Config on every "
                    "MPI rank; rank "+std::to_string(rank)+
                    " differs from rank 0.");
            }
        }
        for(std::size_t rank = 1u; rank < all.size(); ++rank) {
            if(all[rank][state_word] != reference[state_word]) {
                throw std::runtime_error(
                    "SurfaceIBM requires every MPI rank to have the same "
                    "LBM attachment state; rank "+std::to_string(rank)+
                    " differs from rank 0.");
            }
        }
        for(std::size_t rank = 1u; rank < all.size(); ++rank) {
            if(all[rank][marker_count_word] !=
                    reference[marker_count_word] ||
               all[rank][marker_xor_word] != reference[marker_xor_word] ||
               all[rank][marker_sum_word] != reference[marker_sum_word]) {
                throw std::runtime_error(
                    "SurfaceIBM requires the same deterministic global marker "
                    "set on every MPI rank; rank "+std::to_string(rank)+
                    " differs from rank 0.");
            }
        }
    }

    if(initialized)
        throw std::logic_error(
            "SurfaceIBM markers must be attached before the first LBM "
            "initialization.");
    if(already_attached)
        throw std::logic_error(
            "SurfaceIBM markers may be attached only once per LBM instance.");
    if(!config.enabled)
        throw std::logic_error(
            "SurfaceIBM must be enabled before LBM construction.");
    if(!storage_enabled)
        throw std::logic_error(
            "SurfaceIBM configuration changed after LBM construction; "
            "construct a new LBM instance after enabling it.");
    std::string config_error;
    if(!validate_config(config, config_error))
        throw std::runtime_error(
            "Invalid SurfaceIBM configuration: "+config_error);
    if(markers.empty())
        throw std::runtime_error(
            "SurfaceIBM is enabled but attach_surface_ibm received no markers.");
    if((std::uint64_t)markers.size() > config.max_markers)
        throw std::runtime_error(
            "SurfaceIBM marker count exceeds "
            "culbm_surface_ibm_max_markers.");
}

class Runtime::Impl {
private:
    struct DomainState {
        culbm::solver::Subdomain* domain = nullptr;
        std::vector<std::uint64_t> marker_indices;
        std::unique_ptr<gpu::MirroredBuffer<ulong>> active_cells;
        std::unique_ptr<gpu::Buffer<float>> applied_force; // device only
        std::unique_ptr<gpu::MirroredBuffer<float>> positions;
        std::unique_ptr<gpu::MirroredBuffer<float>> partial;
        std::unique_ptr<gpu::MirroredBuffer<float>> fragment_force;
        std::unique_ptr<gpu::MirroredBuffer<float>> wall_velocity;
        std::unique_ptr<gpu::MirroredBuffer<float>> marker_volume;
        std::unique_ptr<gpu::MirroredBuffer<float>> marker_coupling;
        std::unique_ptr<gpu::MirroredBuffer<float>> metrics;
        std::unique_ptr<gpu::MirroredBuffer<float>> reaction;
        std::unique_ptr<gpu::MirroredBuffer<uint>> iteration_control;
        // Wall model: fragment normals, exterior partial sums and spread
        // values of the tangential compensation force.
        bool wall_model = false;
        ulong wm_interior_count = 0ull;
        std::vector<std::uint64_t> seam_indices;
        std::unique_ptr<gpu::MirroredBuffer<float>> seam_positions;
        std::unique_ptr<gpu::MirroredBuffer<float>> seam_normals;
        std::unique_ptr<gpu::MirroredBuffer<float>> normals;
        std::unique_ptr<gpu::MirroredBuffer<float>> wm_partial;
        std::unique_ptr<gpu::MirroredBuffer<float>> wm_force;
        gpu::Launch wm_partial_kernel;
        gpu::Launch wm_spread_kernel;
        gpu::Launch wm_device;
        gpu::Launch wm_reduce_reaction;
        gpu::Launch reset_force;
        gpu::Launch mark_active;
        gpu::Launch interpolate_partial;
        gpu::Launch interpolate_coupling;
        gpu::Launch spread;
        gpu::Launch apply_force_delta;
        gpu::Launch direct_forcing;
        gpu::Launch begin_device_step;
        gpu::Launch reset_device_measure;
        gpu::Launch reduce_device_measure;
        gpu::Launch finish_device_iteration;

        DomainState(culbm::solver::Subdomain& linked_domain,
                    const std::vector<Marker>& markers,
                    std::vector<std::uint64_t> indices,
                    const std::vector<ulong>& active_indices,
                    const bool wall_model_enabled)
            : domain(&linked_domain), marker_indices(std::move(indices)),
              wall_model(wall_model_enabled) {
            const gpu::Device& device = domain->device();
            const ulong active_count = (ulong)active_indices.size();
            if(active_count != 0ull) {
                active_cells = std::make_unique<gpu::MirroredBuffer<ulong>>(
                    device, active_count, 1u,
                    gpu::HostMemory::pinned, 0ull);
                applied_force = std::make_unique<gpu::Buffer<float>>(
                    device, active_count, 3u);
                iteration_control = std::make_unique<gpu::MirroredBuffer<uint>>(
                    device, 3ull, 1u,
                    gpu::HostMemory::pinned, 0u);
                for(ulong i = 0ull; i < active_count; ++i)
                    (*active_cells)[i] = active_indices[(std::size_t)i];
                active_cells->upload_async();
                mark_active = gpu::Launch(device, "surface_ibm_mark_active_cells", active_count).push(
                    *active_cells, domain->fields().ibm_mask,
                    active_count);
                mark_active.submit();
                reset_force = gpu::Launch(device, "surface_ibm_reset_active_force", active_count).push(
                    *active_cells, domain->fields().ibm_force,
                    *applied_force, active_count);
                apply_force_delta = gpu::Launch(device, "surface_ibm_apply_active_force_delta", active_count).push(
                    *active_cells, domain->fields().ibm_force,
                    *applied_force, domain->fields().density, domain->fields().velocity, active_count,
                    *iteration_control, (uint)0u);
            }

            const ulong count = (ulong)marker_indices.size();
            if(count == 0ull) return;
            positions = std::make_unique<gpu::MirroredBuffer<float>>(
                device, count, 3u,
                gpu::HostMemory::pinned, 0.0f);
            partial = std::make_unique<gpu::MirroredBuffer<float>>(
                device, count, 6u,
                gpu::HostMemory::pinned, 0.0f);
            fragment_force = std::make_unique<gpu::MirroredBuffer<float>>(
                device, count, 3u,
                gpu::HostMemory::pinned, 0.0f);
            for(ulong fragment = 0ull; fragment < count; ++fragment) {
                const Marker& marker =
                    markers[(std::size_t)marker_indices[(std::size_t)fragment]];
                (*positions)[fragment] = marker.position_lu.x;
                (*positions)[count+fragment] = marker.position_lu.y;
                (*positions)[2ull*count+fragment] = marker.position_lu.z;
            }
            positions->upload_async();
            interpolate_partial = gpu::Launch(device, "surface_ibm_interpolate_partial", count).push(
                *positions, domain->fields().velocity, domain->fields().density, *partial, count);
            interpolate_coupling = gpu::Launch(device, "surface_ibm_interpolate_coupling", count).push(
                *positions, domain->fields().ibm_force, *partial, count);
            spread = gpu::Launch(device, "surface_ibm_spread", count).push(
                *positions, *fragment_force,
                domain->fields().ibm_force, count);
            if(wall_model) {
                normals = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, count, 3u,
                    gpu::HostMemory::pinned, 0.0f);
                wm_partial = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, count, 12u,
                    gpu::HostMemory::pinned, 0.0f);
                wm_force = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, count, 4u,
                    gpu::HostMemory::pinned, 0.0f);
                for(ulong fragment = 0ull; fragment < count; ++fragment) {
                    const Marker& marker = markers[
                        (std::size_t)marker_indices[(std::size_t)fragment]];
                    (*normals)[fragment] = marker.normal.x;
                    (*normals)[count+fragment] = marker.normal.y;
                    (*normals)[2ull*count+fragment] = marker.normal.z;
                }
                normals->upload_async();
                wm_partial_kernel = gpu::Launch(device, "surface_ibm_wm_partial", count).push(
                    *positions, *normals, domain->fields().velocity, domain->fields().density,
                    domain->fields().flags, *wm_partial, count);
                wm_spread_kernel = gpu::Launch(device, "surface_ibm_wm_spread", count).push(
                    *positions, *normals,
                    domain->fields().flags, *wm_force,
                    domain->fields().ibm_force, count);
            }
        }

        ulong fragment_count() const {
            return (ulong)marker_indices.size();
        }

        ulong active_cell_count() const {
            return active_cells ? active_cells->count() : 0ull;
        }

        void enqueue_reset_force() {
            if(active_cell_count() != 0ull) reset_force.submit();
        }

        void enqueue_apply_force_delta() {
            if(active_cell_count() != 0ull) apply_force_delta.submit();
        }

        void enable_device_direct(
            const std::vector<Marker>& markers,
            const std::vector<float>& coupling,
            const Config& config) {
            const ulong count = fragment_count();
            if(count == 0ull || count != (ulong)markers.size() ||
               count != (ulong)coupling.size())
                throw std::logic_error(
                    "SurfaceIBM device-direct path requires one fragment per marker.");
            for(ulong i = 0ull; i < count; ++i) {
                if(marker_indices[(std::size_t)i] != (std::uint64_t)i)
                    throw std::logic_error(
                        "SurfaceIBM device-direct marker mapping is not identity.");
            }
            const gpu::Device& device = domain->device();
            wall_velocity = std::make_unique<gpu::MirroredBuffer<float>>(
                device, count, 3u,
                gpu::HostMemory::pinned, 0.0f);
            marker_volume = std::make_unique<gpu::MirroredBuffer<float>>(
                device, count, 1u,
                gpu::HostMemory::pinned, 0.0f);
            marker_coupling = std::make_unique<gpu::MirroredBuffer<float>>(
                device, count, 1u,
                gpu::HostMemory::pinned, 0.0f);
            metrics = std::make_unique<gpu::MirroredBuffer<float>>(
                device, 4ull, 1u,
                gpu::HostMemory::pinned, 0.0f);
            reaction = std::make_unique<gpu::MirroredBuffer<float>>(
                device, 6ull, 1u,
                gpu::HostMemory::pinned, 0.0f);
            for(ulong i = 0ull; i < count; ++i) {
                const Marker& marker = markers[(std::size_t)i];
                (*wall_velocity)[i] = marker.wall_velocity_lu.x;
                (*wall_velocity)[count+i] = marker.wall_velocity_lu.y;
                (*wall_velocity)[2ull*count+i] = marker.wall_velocity_lu.z;
                (*marker_volume)[i] =
                    marker.area_lu2*config.support_thickness_lu;
                (*marker_coupling)[i] = coupling[(std::size_t)i];
            }
            wall_velocity->upload_async();
            marker_volume->upload_async();
            marker_coupling->upload_async();
            metrics->upload_async();
            reaction->upload_async();
            begin_device_step = gpu::Launch(device, "surface_ibm_begin_device_step", count).push(
                *metrics, *reaction, *iteration_control,
                *fragment_force, count);
            reset_device_measure = gpu::Launch(device, "surface_ibm_reset_device_measure", 1ull).push(
                *metrics, *iteration_control, (uint)0u);
            // The block reduction is written for exactly 64 threads.
            reduce_device_measure = gpu::Launch(
                device, "surface_ibm_reduce_device_measure", count)
                .block(gpu::default_block_size).push(
                *positions, *partial, *fragment_force, *metrics, *reaction,
                count, *iteration_control, (uint)0u);
            finish_device_iteration = gpu::Launch(device, "surface_ibm_finish_device_iteration", 1ull).push(
                *metrics, *iteration_control, config.residual_tolerance);
            direct_forcing = gpu::Launch(device, "surface_ibm_direct_forcing_step", count).push(
                *positions, *wall_velocity, *marker_volume, *marker_coupling,
                domain->fields().velocity, domain->fields().density, domain->fields().ibm_force,
                *partial, *fragment_force, *iteration_control, count,
                config.relaxation, (uint)1u);
            if(wall_model) {
                wm_interior_count = count;
                wm_device = gpu::Launch(device, "surface_ibm_wm_device", count).push(
                    *positions, *normals, *wall_velocity, *marker_volume,
                    domain->fields().velocity, domain->fields().density, domain->fields().flags,
                    domain->fields().ibm_force, *fragment_force,
                    count, 1.0f/config.support_thickness_lu);
                wm_reduce_reaction = gpu::Launch(
                    device, "surface_ibm_reduce_reaction", count)
                    .block(gpu::default_block_size).push(
                    *positions, *fragment_force, *reaction, count);
            }
            apply_force_delta.bind(7u, (uint)1u);
        }

        void enable_multi_gpu_wall_model(
            const std::vector<Marker>& markers,
            const std::vector<Marker>& seam_markers,
            const std::vector<std::uint64_t>& interior_indices,
            std::vector<std::uint64_t> seam_marker_indices,
            const Config& config) {
            const gpu::Device& device = domain->device();
            // Release full-size portable fragment buffers before allocating
            // partitioned interior + seam buffers.
            positions.reset();
            partial.reset();
            fragment_force.reset();
            normals.reset();
            wm_partial.reset();
            wm_force.reset();
            std::vector<std::uint64_t>().swap(marker_indices);

            wm_interior_count = (ulong)interior_indices.size();
            if(wm_interior_count != 0ull) {
                positions = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, wm_interior_count, 3u, gpu::HostMemory::pinned, 0.0f);
                normals = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, wm_interior_count, 3u, gpu::HostMemory::pinned, 0.0f);
                wall_velocity = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, wm_interior_count, 3u, gpu::HostMemory::pinned, 0.0f);
                marker_volume = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, wm_interior_count, 1u, gpu::HostMemory::pinned, 0.0f);
                fragment_force = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, wm_interior_count, 3u, gpu::HostOnDemand{});
                reaction = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, 6ull, 1u, gpu::HostMemory::pinned, 0.0f);
                for(ulong i = 0ull; i < wm_interior_count; ++i) {
                    const Marker& marker =
                        markers[(std::size_t)interior_indices[(std::size_t)i]];
                    (*positions)[i] = marker.position_lu.x;
                    (*positions)[wm_interior_count+i] = marker.position_lu.y;
                    (*positions)[2ull*wm_interior_count+i] = marker.position_lu.z;
                    (*normals)[i] = marker.normal.x;
                    (*normals)[wm_interior_count+i] = marker.normal.y;
                    (*normals)[2ull*wm_interior_count+i] = marker.normal.z;
                    (*wall_velocity)[i] = marker.wall_velocity_lu.x;
                    (*wall_velocity)[wm_interior_count+i] = marker.wall_velocity_lu.y;
                    (*wall_velocity)[2ull*wm_interior_count+i] = marker.wall_velocity_lu.z;
                    (*marker_volume)[i] =
                        marker.area_lu2*config.support_thickness_lu;
                }
                positions->upload_async();
                normals->upload_async();
                wall_velocity->upload_async();
                marker_volume->upload_async();
                reaction->upload_async();
                wm_device = gpu::Launch(device, "surface_ibm_wm_device", wm_interior_count).push(
                    *positions, *normals, *wall_velocity, *marker_volume,
                    domain->fields().velocity, domain->fields().density, domain->fields().flags,
                    domain->fields().ibm_force, *fragment_force,
                    wm_interior_count, 1.0f/config.support_thickness_lu);
                wm_reduce_reaction = gpu::Launch(
                    device, "surface_ibm_reduce_reaction", wm_interior_count)
                    .block(gpu::default_block_size).push(
                    *positions, *fragment_force, *reaction, wm_interior_count);
            }

            seam_indices = std::move(seam_marker_indices);
            const ulong sc = (ulong)seam_indices.size();
            if(sc != 0ull) {
                seam_positions = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, sc, 3u, gpu::HostMemory::pinned, 0.0f);
                seam_normals = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, sc, 3u, gpu::HostMemory::pinned, 0.0f);
                wm_partial = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, sc, 12u, gpu::HostMemory::pinned, 0.0f);
                wm_force = std::make_unique<gpu::MirroredBuffer<float>>(
                    device, sc, 4u, gpu::HostMemory::pinned, 0.0f);
                for(ulong i = 0ull; i < sc; ++i) {
                    const Marker& marker =
                        seam_markers[(std::size_t)seam_indices[(std::size_t)i]];
                    (*seam_positions)[i] = marker.position_lu.x;
                    (*seam_positions)[sc+i] = marker.position_lu.y;
                    (*seam_positions)[2ull*sc+i] = marker.position_lu.z;
                    (*seam_normals)[i] = marker.normal.x;
                    (*seam_normals)[sc+i] = marker.normal.y;
                    (*seam_normals)[2ull*sc+i] = marker.normal.z;
                }
                seam_positions->upload_async();
                seam_normals->upload_async();
                wm_partial_kernel = gpu::Launch(device, "surface_ibm_wm_partial", sc).push(
                    *seam_positions, *seam_normals, domain->fields().velocity, domain->fields().density,
                    domain->fields().flags, *wm_partial, sc);
                wm_spread_kernel = gpu::Launch(device, "surface_ibm_wm_spread", sc).push(
                    *seam_positions, *seam_normals,
                    domain->fields().flags, *wm_force,
                    domain->fields().ibm_force, sc);
            }
        }

        void release_multi_gpu_wm_host_storage() {
            if(active_cells) active_cells->release_host();
            if(positions) positions->release_host();
            if(normals) normals->release_host();
            if(wall_velocity) wall_velocity->release_host();
            if(marker_volume) marker_volume->release_host();
            if(seam_positions) seam_positions->release_host();
            if(seam_normals) seam_normals->release_host();
        }

        ulong seam_count() const {
            return (ulong)seam_indices.size();
        }

        bool device_direct_enabled() const {
            return direct_forcing.threads() != 0ull;
        }

        void release_device_direct_host_storage() {
            if(!device_direct_enabled()) return;
            // Kernel arguments retain the CUDA allocations.  Once their
            // initialization copies have completed, the single-device path
            // only reads metrics, reaction and iteration_control on the host.
            // Release the remaining O(markers + active cells) staging mirrors
            // instead of pinning them for the lifetime of the simulation.
            if(active_cells) active_cells->release_host();
            if(positions) positions->release_host();
            if(partial) partial->release_host();
            if(fragment_force) fragment_force->release_host();
            if(wall_velocity) wall_velocity->release_host();
            if(marker_volume) marker_volume->release_host();
            if(marker_coupling) marker_coupling->release_host();
            if(normals) normals->release_host();
            if(wm_partial) wm_partial->release_host();
            if(wm_force) wm_force->release_host();
            std::vector<std::uint64_t>().swap(marker_indices);
        }

        void enqueue_begin_device_direct_step() {
            begin_device_step.submit();
        }

        // Wall-modelled surfaces: one exterior wall-stress pass replaces the
        // no-slip forcing iterations; then reduce the reaction.
        void enqueue_device_wall_model_step() {
            if(wm_interior_count == 0ull) return;
            reaction->zero_device_async();
            wm_device.submit();
            wm_reduce_reaction.submit();
            reaction->download_async();
        }

        void enqueue_device_iterations(const unsigned count) {
            for(unsigned iteration = 0u; iteration < count; ++iteration) {
                // Build and apply one correction unconditionally while this
                // iteration is active.  Convergence must be decided from the
                // post-application velocity, not the stale pre-force slip.
                direct_forcing.bind(12u, (uint)1u);
                direct_forcing.submit();
                enqueue_apply_force_delta();

                reset_device_measure.bind(2u, (uint)0u);
                reset_device_measure.submit();
                direct_forcing.bind(12u, (uint)2u);
                direct_forcing.submit();
                reduce_device_measure.bind(7u, (uint)0u);
                reduce_device_measure.submit();
                finish_device_iteration.submit();
            }
            // The last active post-application pass already left final marker
            // diagnostics in partial.  Re-reduce those values unconditionally
            // to obtain final metrics and cumulative reaction without another
            // marker interpolation stencil traversal.
            reset_device_measure.bind(2u, (uint)1u);
            reset_device_measure.submit();
            reduce_device_measure.bind(7u, (uint)1u);
            reduce_device_measure.submit();
            metrics->download_async();
            reaction->download_async();
            iteration_control->download_async();
        }
    };

    struct Measurement {
        float max_slip = 0.0f;
        double weighted_slip_squared = 0.0;
        double volume = 0.0;
    };

    culbm::solver::Solver& solver_;
    Config config_;
    std::vector<Marker> markers_;
    std::vector<float3> marker_force_density_;
    std::vector<float3> marker_force_delta_;
    std::vector<float> marker_coupling_;
    std::vector<float> reduction_values_;
    std::vector<std::unique_ptr<DomainState>> domains_;
    Statistics statistics_;
    bool device_direct_ = false;
    float3 device_reaction_force_{};
    float3 device_reaction_torque_{};
    bool nonconvergence_warned_ = false;
    // Wall-modelled immersed surfaces (portable path keeps per-marker state;
    // multi-GPU local runs keep interior markers on device and only reduce seam
    // markers on host).
    bool wall_model_ = false;
    bool multi_gpu_wm_device_ = false;
    Wall_Model_Lattice wm_lattice_{};
    std::vector<Marker> seam_markers_;
    std::vector<float3> wm_spread_value_;
    std::vector<float> wm_side_;

    void validate_markers() const {
        if(markers_.empty())
            throw std::runtime_error(
                "SurfaceIBM is enabled but attach_surface_ibm received no markers.");
        if((std::uint64_t)markers_.size() > config_.max_markers)
            throw std::runtime_error(
                "SurfaceIBM marker count exceeds "
                "culbm_surface_ibm_max_markers.");
        if(config_.delta_kernel != DeltaKernel::Roma3Point)
            throw std::runtime_error(
                "SurfaceIBM runtime currently supports only the Roma3Point delta kernel.");

        std::uint64_t previous_id = 0u;
        bool first = true;
        for(const Marker& marker : markers_) {
            if(!finite3(marker.position_lu) ||
               !finite3(marker.wall_velocity_lu) ||
               !finite3(marker.normal) ||
               !std::isfinite(marker.area_lu2) || marker.area_lu2 <= 0.0f) {
                throw std::runtime_error(
                    "SurfaceIBM marker payload contains a non-finite value or "
                    "a non-positive quadrature area (global_id="+
                    std::to_string(marker.global_id)+").");
            }
            if(!first && marker.global_id == previous_id)
                throw std::runtime_error(
                    "SurfaceIBM marker global_id values must be unique.");
            first = false;
            previous_id = marker.global_id;

            const int cx = nearest_grid_coordinate(marker.position_lu.x);
            const int cy = nearest_grid_coordinate(marker.position_lu.y);
            const int cz = nearest_grid_coordinate(marker.position_lu.z);
            const bool bounded = config_.support_z_min >= 0;
            const bool valid_z = bounded
                ? config_.support_z_max <= (int)solver_.grid().size(2u)-2 &&
                  marker.position_lu.z > (float)config_.support_z_min-1.5f &&
                  marker.position_lu.z < (float)config_.support_z_max+1.5f
                : cz-1 >= 1 && cz+1 <= (int)solver_.grid().size(2u)-2;
            if(cx-1 < 1 || cy-1 < 1 || !valid_z ||
               cx+1 > (int)solver_.grid().size(0u)-2 ||
               cy+1 > (int)solver_.grid().size(1u)-2) {
                throw std::runtime_error(
                    "SurfaceIBM marker Roma3 support must lie completely "
                    "inside the global non-shell lattice (global_id="+
                    std::to_string(marker.global_id)+").");
            }
        }
    }

    std::uint64_t clear_marker_support_solids() {
        const auto visit_support = [&](const auto& visitor) {
            for(const Marker& marker : markers_) {
                const int cx = nearest_grid_coordinate(marker.position_lu.x);
                const int cy = nearest_grid_coordinate(marker.position_lu.y);
                const int cz = nearest_grid_coordinate(marker.position_lu.z);
                for(int dz = -1; dz <= 1; ++dz) {
                    const int z = cz+dz;
                    if(config_.support_z_min >= 0 &&
                       (z < config_.support_z_min || z > config_.support_z_max)) continue;
                    const float wz = roma3_phi_host(
                        marker.position_lu.z-(float)z);
                    for(int dy = -1; dy <= 1; ++dy) {
                        const int y = cy+dy;
                        const float wy = roma3_phi_host(
                            marker.position_lu.y-(float)y);
                        for(int dx = -1; dx <= 1; ++dx) {
                            const int x = cx+dx;
                            const float w = roma3_phi_host(
                                marker.position_lu.x-(float)x)*wy*wz;
                            if(w == 0.0f) continue;
                            const uint domain =
                                global_domain_for_cell(x, y, z, solver_);
                            if(!solver_.owns(domain)) continue;
                            const std::uint64_t n = global_cell_index(
                                x, y, z, solver_.grid().size(0u), solver_.grid().size(1u));
                            visitor((ulong)n, solver_.flags()[(ulong)n]);
                        }
                    }
                }
            }
        };

        std::uint64_t thermal_conflicts = 0u;
        std::uint64_t equilibrium_conflicts = 0u;
        visit_support([&](const ulong, const uchar flag) {
            if((flag&(CELL_SOLID|CELL_THERMAL))==(CELL_SOLID|CELL_THERMAL)) {
                ++thermal_conflicts;
            }
            if((flag&CELL_EQUILIBRIUM)!=0u) ++equilibrium_conflicts;
        });
        if(culbm::mpi_is_distributed()) {
            culbm::mpi_allreduce_sum_uint64(&thermal_conflicts, 1u);
            culbm::mpi_allreduce_sum_uint64(&equilibrium_conflicts, 1u);
        }
        if(thermal_conflicts!=0u) {
            throw std::runtime_error(
                "SurfaceIBM is hydrodynamic-only and cannot clear CELL_SOLID on "
                "a CELL_SOLID|CELL_THERMAL solid-temperature wall. Disable the solid "
                "temperature boundary or use the legacy voxel wall until a "
                "separate thermal immersed-boundary mask is implemented.");
        }
        if(equilibrium_conflicts!=0u) {
            throw std::runtime_error(
                "SurfaceIBM marker support overlaps a CELL_EQUILIBRIUM/equilibrium "
                "boundary. Move the immersed surface away from that boundary.");
        }

        std::uint64_t cleared = 0u;
        visit_support([&](const ulong, uchar& flag) {
            if((flag&CELL_SOLID)!=0u) {
                flag = (uchar)(flag&~CELL_SOLID);
                ++cleared;
            }
        });
        if(culbm::mpi_is_distributed())
            culbm::mpi_allreduce_sum_uint64(&cleared, 1u);
        return cleared;
    }

    // Mark core cells. A fluid face also needs an exterior marker stencil
    // in the same direction before its voxel shear can be replaced.
    std::uint64_t mark_core_cells() {
        std::uint64_t marked = 0u;
        for(const auto& state : domains_) {
            const ulong active_count = state->active_cell_count();
            if(active_count == 0ull) continue;
            culbm::solver::Subdomain& domain = *state->domain;
            const ulong nx = (ulong)domain.extent(0u);
            const ulong ny = (ulong)domain.extent(1u);
            std::vector<ulong> cells;
            for(ulong i = 0ull; i < active_count; ++i) {
                const ulong n = (*(state->active_cells))[i];
                const int gx = (int)(n%nx)+domain.origin(0u);
                const int gy = (int)((n/nx)%ny)+domain.origin(1u);
                const int gz = (int)(n/(nx*ny))+domain.origin(2u);
                const std::uint64_t g = global_cell_index(
                    gx, gy, gz, solver_.grid().size(0u), solver_.grid().size(1u));
                if((solver_.flags()[(ulong)g]&CELL_SOLID) != 0u) cells.push_back(n);
            }
            if(cells.empty()) continue;
            const gpu::Device& device = domain.device();
            gpu::MirroredBuffer<ulong> list(device, (ulong)cells.size(), 1u,
                                            gpu::HostMemory::pinned, 0ull);
            for(std::size_t i = 0u; i < cells.size(); ++i)
                list[(ulong)i] = cells[i];
            list.upload_async();
            gpu::Launch(device, "surface_ibm_mark_core_cells", (ulong)cells.size())
                .push(list, domain.fields().ibm_mask, (ulong)cells.size())
                .submit();
            domain.synchronize();
            marked += (std::uint64_t)cells.size();
        }
        if(culbm::mpi_is_distributed())
            culbm::mpi_allreduce_sum_uint64(&marked, 1u);
        return marked;
    }

    void mark_wall_model_faces() {
        std::vector<float> weights(2u*markers_.size(), 0.0f);
        for(const auto& state : domains_) {
            const ulong count = state->fragment_count();
            if(count==0ull) continue;
            culbm::solver::Subdomain& domain = *state->domain;
            // Attach precedes Solver::initialize(), which normally uploads flags.
            domain.fields().flags.upload_async();
            gpu::Launch(domain.device(), "surface_ibm_wm_support_weights", count)
                .push(*state->positions, *state->normals, domain.fields().flags, *state->wm_force, count)
                .submit();
            state->wm_force->download_async(0ull, 2ull*count);
            domain.synchronize();
            for(ulong f=0ull; f<count; ++f) {
                const std::size_t m = (std::size_t)state->marker_indices[(std::size_t)f];
                weights[2u*m] += (*state->wm_force)[f];
                weights[2u*m+1u] += (*state->wm_force)[count+f];
            }
        }
        if(culbm::mpi_is_distributed()) {
            for(std::size_t start=0u; start<weights.size(); start+=reduction_marker_chunk) {
                const auto length = (std::uint32_t)std::min<std::size_t>(
                    reduction_marker_chunk, weights.size()-start);
                culbm::mpi_allreduce_sum_float(weights.data()+start, length);
            }
        }
        for(const auto& state : domains_) {
            const ulong count = state->fragment_count();
            if(count==0ull) continue;
            for(ulong f=0ull; f<count; ++f) {
                const std::size_t m = (std::size_t)state->marker_indices[(std::size_t)f];
                (*state->wm_force)[f] = weights[2u*m]>wall_model_min_exterior_weight ? 1.0f
                    : weights[2u*m+1u]>wall_model_min_exterior_weight ? -1.0f : 0.0f;
            }
            state->wm_force->upload_async(0ull, count);
            culbm::solver::Subdomain& domain = *state->domain;
            // The first force lane is aligned 32-bit scratch during attach.
            // The commit clears it before the first physical predictor.
            state->enqueue_reset_force();
            gpu::Launch(domain.device(), "surface_ibm_wm_cover_faces", count)
                .push(*state->positions, *state->normals, domain.fields().flags, domain.fields().ibm_mask,
                      *state->wm_force, domain.fields().ibm_force, count)
                .submit();
            const ulong active = state->active_cell_count();
            if(active!=0ull) {
                gpu::Launch(domain.device(), "surface_ibm_wm_commit_faces", active)
                    .push(*state->active_cells, domain.fields().ibm_force,
                          domain.fields().ibm_mask, active)
                    .submit();
            }
            domain.synchronize();
        }
    }

    // Portable (multi-domain/MPI) wall-model step: reduce the exterior probe
    // partials per marker, evaluate the wall function on the host and spread
    // the wall force over each fragment's exterior cells.
    void apply_wall_model_portable() {
        for(const auto& state : domains_) {
            if(state->fragment_count() == 0ull) continue;
            state->wm_partial_kernel.submit();
            state->wm_partial->download_async();
        }
        for(const auto& state : domains_) state->domain->synchronize();
        const std::uint64_t marker_count = (std::uint64_t)markers_.size();
        const std::uint64_t chunk_capacity = std::min<std::uint64_t>(
            reduction_marker_chunk/2u,
            (std::uint64_t)std::numeric_limits<std::uint32_t>::max()/12u);
        for(std::uint64_t begin = 0u; begin < marker_count;
            begin += chunk_capacity) {
            const std::uint64_t count =
                std::min(chunk_capacity, marker_count-begin);
            const std::size_t value_count = (std::size_t)(12u*count);
            std::fill(reduction_values_.begin(),
                      reduction_values_.begin()+value_count, 0.0f);
            float* const values = reduction_values_.data();
            for(const auto& state : domains_) {
                const auto first = std::lower_bound(
                    state->marker_indices.begin(), state->marker_indices.end(), begin);
                const auto last = std::lower_bound(
                    first, state->marker_indices.end(), begin+count);
                const ulong fragments = state->fragment_count();
                for(auto it = first; it != last; ++it) {
                    const std::size_t fragment =
                        (std::size_t)(it-state->marker_indices.begin());
                    const std::size_t local = (std::size_t)(*it-begin);
                    for(std::size_t component = 0u; component < 12u; ++component)
                        values[component*(std::size_t)count+local] +=
                            (*(state->wm_partial))[
                                (ulong)fragment+(ulong)component*fragments];
                }
            }
            if(culbm::mpi_is_distributed())
                culbm::mpi_allreduce_sum_float(
                    values, (std::uint32_t)(12u*count));
            for(std::uint64_t local = 0u; local < count; ++local) {
                const std::size_t m = (std::size_t)(begin+local);
                const Marker& marker = markers_[m];
                wm_spread_value_[m] = float3(0.0f);
                wm_side_[m] = 1.0f;
                // Trusted normal first, then the flipped side (inward STL normal).
                std::size_t o = 0u;
                float weight = values[(std::size_t)(4u*count+local)];
                if(!(weight > wall_model_min_exterior_weight)) {
                    o = (std::size_t)(6u*count);
                    weight = values[o+(std::size_t)(4u*count+local)];
                    wm_side_[m] = -1.0f;
                }
                if(!(weight > wall_model_min_exterior_weight)) continue;
                const float inv = 1.0f/weight;
                const float rho = values[o+(std::size_t)(3u*count+local)]*inv;
                const float h = values[o+(std::size_t)(5u*count+local)]*inv;
                const float rx = values[o+(std::size_t)local]*inv-
                                 marker.wall_velocity_lu.x;
                const float ry = values[o+(std::size_t)(count+local)]*inv-
                                 marker.wall_velocity_lu.y;
                const float rz = values[o+(std::size_t)(2u*count+local)]*inv-
                                 marker.wall_velocity_lu.z;
                const float3& n = marker.normal;
                const float rn = rx*n.x+ry*n.y+rz*n.z;
                const float tx = rx-rn*n.x, ty = ry-rn*n.y, tz = rz-rn*n.z;
                const float ut = std::sqrt(tx*tx+ty*ty+tz*tz);
                float3 wall(0.0f);
                if(ut > 0.0f && std::isfinite(ut) && rho > minimum_density) {
                    const float k = wall_model_factor(
                        wm_lattice_, ut, h,
                        wm_side_[m]*n.z >= wall_model_upward_cosine);
                    const float scale = -marker.area_lu2*rho*k;
                    wall = float3(scale*tx, scale*ty, scale*tz);
                }
                const float volume =
                    marker.area_lu2*config_.support_thickness_lu;
                float3& density = marker_force_density_[m];
                const float3 force = wall;
                if(!finite3(force)) continue;
                density.x += force.x/volume;
                density.y += force.y/volume;
                density.z += force.z/volume;
                wm_spread_value_[m] = float3(force.x*inv, force.y*inv,
                                             force.z*inv);
            }
        }
        for(const auto& state : domains_) {
            const ulong fragments = state->fragment_count();
            if(fragments == 0ull) continue;
            for(ulong fragment = 0ull; fragment < fragments; ++fragment) {
                const std::size_t marker =
                    (std::size_t)state->marker_indices[(std::size_t)fragment];
                const float3 value = wm_spread_value_[marker];
                (*(state->wm_force))[fragment] = value.x;
                (*(state->wm_force))[fragments+fragment] = value.y;
                (*(state->wm_force))[2ull*fragments+fragment] = value.z;
                (*(state->wm_force))[3ull*fragments+fragment] = wm_side_[marker];
            }
            state->wm_force->upload_async();
            state->wm_spread_kernel.submit();
        }
    }

    // Partition multi-GPU wall-modelled markers into domain-interior markers
    // (100% support owned by one GPU -> run directly on device via
    // surface_ibm_wm_device) and seam markers (straddling >=2 domains -> run
    // the compact two-pass portable reduction over seam markers only).
    void enable_multi_gpu_wall_model_path() {
        std::vector<uint> owner_count(markers_.size(), 0u);
        for(const auto& state : domains_) {
            for(const std::uint64_t m : state->marker_indices)
                ++owner_count[(std::size_t)m];
        }
        constexpr std::size_t k_none = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> global_to_seam(markers_.size(), k_none);
        seam_markers_.clear();
        for(std::size_t m = 0u; m < markers_.size(); ++m) {
            if(owner_count[m] >= 2u) {
                global_to_seam[m] = seam_markers_.size();
                seam_markers_.push_back(markers_[m]);
            }
        }
        for(const auto& state : domains_) {
            std::vector<std::uint64_t> interior_indices;
            std::vector<std::uint64_t> seam_indices;
            interior_indices.reserve(state->marker_indices.size());
            for(const std::uint64_t m : state->marker_indices) {
                if(owner_count[(std::size_t)m] == 1u) {
                    interior_indices.push_back(m);
                } else {
                    seam_indices.push_back((std::uint64_t)global_to_seam[(std::size_t)m]);
                }
            }
            state->enable_multi_gpu_wall_model(
                markers_, seam_markers_, interior_indices, std::move(seam_indices), config_);
        }
        for(const auto& state : domains_) state->domain->synchronize();
        for(const auto& state : domains_) state->release_multi_gpu_wm_host_storage();

        std::vector<Marker>().swap(markers_);
        std::vector<float3>().swap(marker_force_density_);
        std::vector<float3>().swap(marker_force_delta_);
        std::vector<float>().swap(marker_coupling_);
        wm_spread_value_.assign(seam_markers_.size(), float3(0.0f));
        wm_side_.assign(seam_markers_.size(), 1.0f);
        reduction_values_.resize(12u*std::min<std::size_t>(
            seam_markers_.size(), (std::size_t)reduction_marker_chunk/2u));
        multi_gpu_wm_device_ = true;
    }

    void apply_wall_model_multi_gpu_device() {
        for(const auto& state : domains_) {
            state->enqueue_device_wall_model_step();
            if(state->seam_count() != 0ull) {
                state->wm_partial_kernel.submit();
                state->wm_partial->download_async();
            }
        }
        for(const auto& state : domains_) state->domain->synchronize();

        double force_x = 0.0, force_y = 0.0, force_z = 0.0;
        double torque_x = 0.0, torque_y = 0.0, torque_z = 0.0;
        for(const auto& state : domains_) {
            if(state->wm_interior_count == 0ull) continue;
            force_x += (double)(*(state->reaction))[0];
            force_y += (double)(*(state->reaction))[1];
            force_z += (double)(*(state->reaction))[2];
            torque_x += (double)(*(state->reaction))[3];
            torque_y += (double)(*(state->reaction))[4];
            torque_z += (double)(*(state->reaction))[5];
        }

        const std::uint64_t seam_count = (std::uint64_t)seam_markers_.size();
        if(seam_count != 0u) {
            const std::uint64_t chunk_capacity = std::min<std::uint64_t>(
                reduction_marker_chunk/2u,
                (std::uint64_t)std::numeric_limits<std::uint32_t>::max()/12u);
            for(std::uint64_t begin = 0u; begin < seam_count; begin += chunk_capacity) {
                const std::uint64_t count = std::min(chunk_capacity, seam_count-begin);
                const std::size_t value_count = (std::size_t)(12u*count);
                std::fill(reduction_values_.begin(),
                          reduction_values_.begin()+value_count, 0.0f);
                float* const values = reduction_values_.data();
                for(const auto& state : domains_) {
                    const auto first = std::lower_bound(
                        state->seam_indices.begin(), state->seam_indices.end(), begin);
                    const auto last = std::lower_bound(
                        first, state->seam_indices.end(), begin+count);
                    const ulong fragments = state->seam_count();
                    for(auto it = first; it != last; ++it) {
                        const std::size_t fragment =
                            (std::size_t)(it-state->seam_indices.begin());
                        const std::size_t local = (std::size_t)(*it-begin);
                        for(std::size_t component = 0u; component < 12u; ++component)
                            values[component*(std::size_t)count+local] +=
                                (*(state->wm_partial))[
                                    (ulong)fragment+(ulong)component*fragments];
                    }
                }
                for(std::uint64_t local = 0u; local < count; ++local) {
                    const std::size_t m = (std::size_t)(begin+local);
                    const Marker& marker = seam_markers_[m];
                    wm_spread_value_[m] = float3(0.0f);
                    wm_side_[m] = 1.0f;
                    std::size_t o = 0u;
                    float weight = values[(std::size_t)(4u*count+local)];
                    if(!(weight > wall_model_min_exterior_weight)) {
                        o = (std::size_t)(6u*count);
                        weight = values[o+(std::size_t)(4u*count+local)];
                        wm_side_[m] = -1.0f;
                    }
                    if(!(weight > wall_model_min_exterior_weight)) continue;
                    const float inv = 1.0f/weight;
                    const float rho = values[o+(std::size_t)(3u*count+local)]*inv;
                    const float h = values[o+(std::size_t)(5u*count+local)]*inv;
                    const float rx = values[o+(std::size_t)local]*inv-
                                     marker.wall_velocity_lu.x;
                    const float ry = values[o+(std::size_t)(count+local)]*inv-
                                     marker.wall_velocity_lu.y;
                    const float rz = values[o+(std::size_t)(2u*count+local)]*inv-
                                     marker.wall_velocity_lu.z;
                    const float3& n = marker.normal;
                    const float rn = rx*n.x+ry*n.y+rz*n.z;
                    const float tx = rx-rn*n.x, ty = ry-rn*n.y, tz = rz-rn*n.z;
                    const float ut = std::sqrt(tx*tx+ty*ty+tz*tz);
                    float3 force(0.0f);
                    if(ut > 0.0f && std::isfinite(ut) && rho > minimum_density) {
                        const float k = wall_model_factor(
                            wm_lattice_, ut, h,
                            wm_side_[m]*n.z >= wall_model_upward_cosine);
                        const float scale = -marker.area_lu2*rho*k;
                        force = float3(scale*tx, scale*ty, scale*tz);
                    }
                    if(!finite3(force)) continue;
                    const double rxn = -(double)force.x;
                    const double ryn = -(double)force.y;
                    const double rzn = -(double)force.z;
                    force_x += rxn;
                    force_y += ryn;
                    force_z += rzn;
                    torque_x += (double)marker.position_lu.y*rzn-
                                (double)marker.position_lu.z*ryn;
                    torque_y += (double)marker.position_lu.z*rxn-
                                (double)marker.position_lu.x*rzn;
                    torque_z += (double)marker.position_lu.x*ryn-
                                (double)marker.position_lu.y*rxn;
                    wm_spread_value_[m] = float3(force.x*inv, force.y*inv, force.z*inv);
                }
            }
            for(const auto& state : domains_) {
                const ulong fragments = state->seam_count();
                if(fragments == 0ull) continue;
                for(ulong fragment = 0ull; fragment < fragments; ++fragment) {
                    const std::size_t marker =
                        (std::size_t)state->seam_indices[(std::size_t)fragment];
                    const float3 value = wm_spread_value_[marker];
                    (*(state->wm_force))[fragment] = value.x;
                    (*(state->wm_force))[fragments+fragment] = value.y;
                    (*(state->wm_force))[2ull*fragments+fragment] = value.z;
                    (*(state->wm_force))[3ull*fragments+fragment] = wm_side_[marker];
                }
                state->wm_force->upload_async();
                state->wm_spread_kernel.submit();
            }
        }
        device_reaction_force_ = float3((float)force_x, (float)force_y, (float)force_z);
        device_reaction_torque_ = float3((float)torque_x, (float)torque_y, (float)torque_z);
        if(!finite3(device_reaction_force_) || !finite3(device_reaction_torque_))
            throw std::runtime_error(
                "SurfaceIBM wall-model reaction became non-finite.");
    }

    void build_domain_fragments() {
        struct PendingDomain {
            culbm::solver::Subdomain* domain = nullptr;
            std::vector<std::uint64_t> indices;
            std::vector<ulong> active_cells;
        };
        const bool device_direct_candidate =
            !culbm::mpi_is_distributed() && solver_.local_count()==1u;
        const std::uint64_t bytes_per_fragment =
            (12u+(device_direct_candidate ? 5u : 0u)+(wall_model_ ? 19u : 0u))*
            sizeof(float);
        constexpr std::uint64_t bytes_per_active_cell =
            sizeof(ulong)+3u*sizeof(float);
        std::uint64_t local_fragments = 0u;
        std::uint64_t local_active_cells = 0u;
        std::vector<PendingDomain> pending;
        pending.reserve(solver_.local_count());
        bool local_preflight_ok = true;
        std::string local_preflight_error;
        for(uint local_domain = 0u; local_domain < solver_.local_count(); ++local_domain) {
            culbm::solver::Subdomain& domain = solver_.local(local_domain);
            const OwnedBox box = owned_box(domain, config_);
            std::vector<std::uint64_t> indices;
            for(std::uint64_t marker = 0u;
                marker < (std::uint64_t)markers_.size(); ++marker) {
                if(box.intersects_support(markers_[(std::size_t)marker]))
                    indices.push_back(marker);
            }
            local_fragments += (std::uint64_t)indices.size();
            std::vector<ulong> active_cells =
                build_active_cells(domain, markers_, indices, config_);
            local_active_cells += (std::uint64_t)active_cells.size();
            const std::uint64_t count = (std::uint64_t)indices.size();
            const std::uint64_t active_count =
                (std::uint64_t)active_cells.size();
            if(count > std::numeric_limits<std::uint64_t>::max()/bytes_per_fragment ||
               active_count >
                   std::numeric_limits<std::uint64_t>::max()/bytes_per_active_cell ||
               count*bytes_per_fragment >
                   std::numeric_limits<std::uint64_t>::max()-
                       active_count*bytes_per_active_cell) {
                local_preflight_ok = false;
                local_preflight_error =
                    "SurfaceIBM fragment-buffer size overflow.";
            } else if(local_preflight_ok) {
                const gpu::Device& device = domain.device();
                const std::uint64_t required =
                    count*bytes_per_fragment+
                    active_count*bytes_per_active_cell;
                const std::uint64_t free =
                    (std::uint64_t)device.free_memory_bytes();
                const std::uint64_t total =
                    (std::uint64_t)device.total_memory_bytes();
                const std::uint64_t reserve = std::min<std::uint64_t>(
                    256ull*1024ull*1024ull, total/10ull);
                if(required > free || free-required < reserve) {
                    local_preflight_ok = false;
                    local_preflight_error =
                        "SurfaceIBM marker fragments require "+
                        std::to_string((required+1048575ull)/1048576ull)+
                        " MiB on CUDA device "+
                        std::to_string(device.ordinal())+
                        " in addition to the Eulerian fields; reduce "
                        "culbm_surface_ibm_max_markers/mesh density or the grid size.";
                }
            }
            pending.push_back(PendingDomain{
                &domain, std::move(indices), std::move(active_cells)});
        }

        std::uint64_t preflight_success = local_preflight_ok ? 1u : 0u;
        if(culbm::mpi_is_distributed()) {
            culbm::mpi_allreduce_sum_uint64(&preflight_success, 1u);
            if(preflight_success != (std::uint64_t)culbm::mpi_size()) {
                throw std::runtime_error(
                    local_preflight_ok
                        ? "SurfaceIBM fragment-buffer preflight failed on another MPI rank."
                        : local_preflight_error);
            }
        } else if(!local_preflight_ok) {
            throw std::runtime_error(local_preflight_error);
        }

        bool local_allocation_ok = true;
        std::string local_allocation_error;
        try {
            domains_.reserve(pending.size());
            for(PendingDomain& item : pending) {
                domains_.push_back(std::make_unique<DomainState>(
                    *item.domain, markers_, std::move(item.indices),
                    item.active_cells, wall_model_));
            }
        } catch(const std::exception& exception) {
            local_allocation_ok = false;
            local_allocation_error = exception.what();
        } catch(...) {
            local_allocation_ok = false;
            local_allocation_error = "unknown host/CUDA allocation exception";
        }
        std::uint64_t allocation_success = local_allocation_ok ? 1u : 0u;
        if(culbm::mpi_is_distributed()) {
            culbm::mpi_allreduce_sum_uint64(&allocation_success, 1u);
            if(allocation_success != (std::uint64_t)culbm::mpi_size()) {
                throw std::runtime_error(
                    local_allocation_ok
                        ? "SurfaceIBM fragment-buffer allocation failed on another MPI rank."
                        : "SurfaceIBM fragment-buffer allocation failed: "+
                            local_allocation_error);
            }
        } else if(!local_allocation_ok) {
            throw std::runtime_error(
                "SurfaceIBM fragment-buffer allocation failed: "+
                local_allocation_error);
        }
        bool local_initialization_ok = true;
        std::string local_initialization_error;
        try {
            for(const auto& state : domains_) state->domain->synchronize();
        } catch(const std::exception& exception) {
            local_initialization_ok = false;
            local_initialization_error = exception.what();
        } catch(...) {
            local_initialization_ok = false;
            local_initialization_error = "unknown CUDA initialization exception";
        }
        std::uint64_t initialization_success =
            local_initialization_ok ? 1u : 0u;
        if(culbm::mpi_is_distributed()) {
            culbm::mpi_allreduce_sum_uint64(&initialization_success, 1u);
            if(initialization_success != (std::uint64_t)culbm::mpi_size()) {
                throw std::runtime_error(
                    local_initialization_ok
                        ? "SurfaceIBM fragment initialization failed on another MPI rank."
                        : "SurfaceIBM fragment initialization failed: "+
                            local_initialization_error);
            }
        } else if(!local_initialization_ok) {
            throw std::runtime_error(
                "SurfaceIBM fragment initialization failed: "+
                local_initialization_error);
        }
        statistics_.fragments = local_fragments;
        statistics_.active_cells = local_active_cells;
        if(culbm::mpi_is_distributed()) {
            culbm::mpi_allreduce_sum_uint64(&statistics_.fragments, 1u);
            culbm::mpi_allreduce_sum_uint64(&statistics_.active_cells, 1u);
        }
    }

    void enqueue_interpolation_and_read() {
        for(const auto& state : domains_) {
            if(state->fragment_count() == 0ull) continue;
            state->interpolate_partial.submit();
            state->partial->download_async();
        }
        for(const auto& state : domains_) state->domain->synchronize();
    }

    void build_marker_coupling() {
        const std::size_t marker_count = markers_.size();
        marker_coupling_.assign(marker_count, 0.0f);
        std::vector<float> partition_weight(marker_count, 0.0f);

        for(const auto& state : domains_) state->enqueue_reset_force();
        for(const auto& state : domains_) {
            const ulong fragments = state->fragment_count();
            if(fragments==0ull) continue;
            for(ulong fragment=0ull; fragment<fragments; ++fragment) {
                const std::size_t marker_index =
                    (std::size_t)state->marker_indices[(std::size_t)fragment];
                const float volume = markers_[marker_index].area_lu2*
                    config_.support_thickness_lu;
                (*(state->fragment_force))[fragment] = volume;
                (*(state->fragment_force))[fragments+fragment] = 0.0f;
                (*(state->fragment_force))[2ull*fragments+fragment] = 0.0f;
            }
            state->fragment_force->upload_async();
            state->spread.submit();
        }
        for(const auto& state : domains_) state->domain->synchronize();

        for(const auto& state : domains_) {
            if(state->fragment_count()==0ull) continue;
            state->interpolate_coupling.submit();
            state->partial->download_async();
        }
        for(const auto& state : domains_) state->domain->synchronize();
        for(const auto& state : domains_) {
            const ulong fragments = state->fragment_count();
            for(ulong fragment=0ull; fragment<fragments; ++fragment) {
                const std::size_t marker_index =
                    (std::size_t)state->marker_indices[(std::size_t)fragment];
                marker_coupling_[marker_index] +=
                    (*(state->partial))[fragment];
                partition_weight[marker_index] +=
                    (*(state->partial))[fragments+fragment];
            }
        }

        if(culbm::mpi_is_distributed()) {
            const std::size_t chunk = (std::size_t)reduction_marker_chunk;
            for(std::size_t begin=0u; begin<marker_count; begin+=chunk) {
                const std::size_t count = std::min(chunk, marker_count-begin);
                culbm::mpi_allreduce_sum_float(
                    marker_coupling_.data()+begin, (std::uint32_t)count);
                culbm::mpi_allreduce_sum_float(
                    partition_weight.data()+begin, (std::uint32_t)count);
            }
        }

        float minimum = std::numeric_limits<float>::infinity();
        float maximum = 0.0f;
        for(std::size_t marker=0u; marker<marker_count; ++marker) {
            const float coupling = marker_coupling_[marker];
            const float weight = partition_weight[marker];
            if(!std::isfinite(coupling) || coupling<=1.0e-8f ||
               !std::isfinite(weight) ||
               std::fabs(weight-1.0f)>partition_tolerance) {
                throw std::runtime_error(
                    "SurfaceIBM could not construct a finite marker-coupling "
                    "normalization (global_id="+
                    std::to_string(markers_[marker].global_id)+
                    ", coupling="+std::to_string(coupling)+
                    ", weight="+std::to_string(weight)+").");
            }
            minimum = std::min(minimum, coupling);
            maximum = std::max(maximum, coupling);
        }
        statistics_.minimum_coupling = minimum;
        statistics_.maximum_coupling = maximum;
        for(const auto& state : domains_) state->enqueue_reset_force();
        for(const auto& state : domains_) state->domain->synchronize();
    }

    void enable_device_direct_path() {
        if(culbm::mpi_is_distributed() || domains_.size()!=1u) return;
        DomainState& state = *domains_.front();
        if(state.fragment_count() != (ulong)markers_.size()) return;
        state.enable_device_direct(markers_, marker_coupling_, config_);
        state.domain->synchronize();
        device_direct_ = state.device_direct_enabled();
        statistics_.device_resident = device_direct_;
        if(device_direct_) {
            // The single-device loop never enters the host/MPI reduction
            // path; release its O(markers) host work arrays after attach.
            std::vector<float3>().swap(marker_force_density_);
            std::vector<float3>().swap(marker_force_delta_);
            std::vector<float>().swap(marker_coupling_);
            std::vector<float>().swap(reduction_values_);
        }
    }

    Measurement reduce_partials(const bool update_marker_force) {
        Measurement measurement;
        const std::uint64_t marker_count = (std::uint64_t)markers_.size();
        const std::uint64_t max_chunk_by_mpi =
            (std::uint64_t)std::numeric_limits<std::uint32_t>::max()/6u;
        const std::uint64_t chunk_capacity =
            std::min(reduction_marker_chunk, max_chunk_by_mpi);
        if(update_marker_force)
            std::fill(marker_force_delta_.begin(), marker_force_delta_.end(),
                      float3(0.0f));

        for(std::uint64_t begin = 0u; begin < marker_count;
            begin += chunk_capacity) {
            const std::uint64_t count =
                std::min(chunk_capacity, marker_count-begin);
            const std::size_t value_count = (std::size_t)(6u*count);
            std::fill(reduction_values_.begin(),
                      reduction_values_.begin()+value_count, 0.0f);
            float* const values = reduction_values_.data();
            for(const auto& state : domains_) {
                const auto first = std::lower_bound(
                    state->marker_indices.begin(), state->marker_indices.end(), begin);
                const auto last = std::lower_bound(
                    first, state->marker_indices.end(), begin+count);
                const ulong fragments = state->fragment_count();
                for(auto it = first; it != last; ++it) {
                    const std::size_t fragment =
                        (std::size_t)(it-state->marker_indices.begin());
                    const std::size_t local = (std::size_t)(*it-begin);
                    for(std::size_t component = 0u; component < 6u; ++component)
                        values[component*(std::size_t)count+local] +=
                            (*(state->partial))[
                                (ulong)fragment+(ulong)component*fragments];
                }
            }
            if(culbm::mpi_is_distributed())
                culbm::mpi_allreduce_sum_float(
                    values, (std::uint32_t)(6u*count));

            for(std::uint64_t local = 0u; local < count; ++local) {
                const std::size_t marker_index = (std::size_t)(begin+local);
                const Marker& marker = markers_[marker_index];
                const float weight = values[(std::size_t)(4u*count+local)];
                const float invalid = values[(std::size_t)(5u*count+local)];
                if(invalid != 0.0f || !std::isfinite(weight) ||
                   std::fabs(weight-1.0f) > partition_tolerance) {
                    throw std::runtime_error(
                        "SurfaceIBM interpolation support is incomplete or "
                        "contains a non-finite Eulerian sample (global_id="+
                        std::to_string(marker.global_id)+
                        ", weight="+std::to_string(weight)+").");
                }
                const float inv_weight = 1.0f/weight;
                const float ux = values[(std::size_t)local]*inv_weight;
                const float uy = values[(std::size_t)(count+local)]*inv_weight;
                const float uz = values[(std::size_t)(2u*count+local)]*inv_weight;
                const float rho =
                    values[(std::size_t)(3u*count+local)]*inv_weight;
                if(!(rho > minimum_density) || !std::isfinite(rho) ||
                   !std::isfinite(ux) || !std::isfinite(uy) ||
                   !std::isfinite(uz)) {
                    throw std::runtime_error(
                        "SurfaceIBM interpolation produced a non-finite or "
                        "non-positive marker density (global_id="+
                        std::to_string(marker.global_id)+").");
                }

                const float rx = marker.wall_velocity_lu.x-ux;
                const float ry = marker.wall_velocity_lu.y-uy;
                const float rz = marker.wall_velocity_lu.z-uz;
                const float slip = std::sqrt(rx*rx+ry*ry+rz*rz);
                if(!std::isfinite(slip))
                    throw std::runtime_error(
                        "SurfaceIBM marker slip became non-finite.");
                measurement.max_slip = std::max(measurement.max_slip, slip);
                const double volume = (double)marker.area_lu2*
                    (double)config_.support_thickness_lu;
                measurement.weighted_slip_squared +=
                    volume*(double)slip*(double)slip;
                measurement.volume += volume;

                if(update_marker_force) {
                    const float factor =
                        2.0f*rho*config_.relaxation/
                        marker_coupling_[marker_index];
                    const float3 delta(factor*rx, factor*ry, factor*rz);
                    marker_force_delta_[marker_index] = delta;
                    float3& force = marker_force_density_[marker_index];
                    force.x += delta.x;
                    force.y += delta.y;
                    force.z += delta.z;
                    if(!finite3(force))
                        throw std::runtime_error(
                            "SurfaceIBM marker force became non-finite.");
                }
            }
        }
        return measurement;
    }

    Measurement measure(const bool update_marker_force) {
        enqueue_interpolation_and_read();
        return reduce_partials(update_marker_force);
    }

    Measurement run_device_iterations(unsigned& applied_iterations) {
        DomainState& state = *domains_.front();
        state.enqueue_device_iterations(config_.forcing_iterations);
        state.domain->synchronize();
        const float max_slip = (*(state.metrics))[0];
        const float weighted = (*(state.metrics))[1];
        const float volume = (*(state.metrics))[2];
        const float invalid = (*(state.metrics))[3];
        if(invalid != 0.0f || !std::isfinite(max_slip) ||
           !std::isfinite(weighted) || weighted < 0.0f ||
           !std::isfinite(volume) || !(volume > 0.0f)) {
            throw std::runtime_error(
                "SurfaceIBM device-resident interpolation produced an "
                "invalid support, density, velocity, coupling or force sample.");
        }
        device_reaction_force_ = float3(
            (*(state.reaction))[0], (*(state.reaction))[1],
            (*(state.reaction))[2]);
        device_reaction_torque_ = float3(
            (*(state.reaction))[3], (*(state.reaction))[4],
            (*(state.reaction))[5]);
        applied_iterations = (*(state.iteration_control))[1];
        if((*(state.iteration_control))[2] != 0u)
            throw std::runtime_error(
                "SurfaceIBM device-resident force update became non-finite.");
        if(applied_iterations > config_.forcing_iterations)
            throw std::runtime_error(
                "SurfaceIBM device-resident iteration counter is invalid.");
        if(!finite3(device_reaction_force_) ||
           !finite3(device_reaction_torque_))
            throw std::runtime_error(
                "SurfaceIBM device-resident reaction became non-finite.");
        Measurement measurement;
        measurement.max_slip = max_slip;
        measurement.weighted_slip_squared = (double)weighted;
        measurement.volume = (double)volume;
        return measurement;
    }

    void spread_marker_force_delta() {
        for(const auto& state : domains_) {
            const ulong fragments = state->fragment_count();
            if(fragments == 0ull) continue;
            for(ulong fragment = 0ull; fragment < fragments; ++fragment) {
                const std::size_t marker_index =
                    (std::size_t)state->marker_indices[(std::size_t)fragment];
                const float volume = markers_[marker_index].area_lu2*
                    config_.support_thickness_lu;
                const float3 force = marker_force_delta_[marker_index];
                (*(state->fragment_force))[fragment] = volume*force.x;
                (*(state->fragment_force))[fragments+fragment] = volume*force.y;
                (*(state->fragment_force))[2ull*fragments+fragment] = volume*force.z;
            }
            state->fragment_force->upload_async();
            state->spread.submit();
        }
        for(const auto& state : domains_) state->enqueue_apply_force_delta();
        // The next interpolation read is ordered in each domain queue and is
        // the only synchronization required for an iteration.  Avoiding a
        // second finish lets independent domain queues overlap their spread.
    }

    void finalize_statistics(const Measurement& measurement,
                             const unsigned iterations) {
        double force_x = (double)device_reaction_force_.x;
        double force_y = (double)device_reaction_force_.y;
        double force_z = (double)device_reaction_force_.z;
        double torque_x = (double)device_reaction_torque_.x;
        double torque_y = (double)device_reaction_torque_.y;
        double torque_z = (double)device_reaction_torque_.z;
        if(!device_direct_ && !multi_gpu_wm_device_) {
            force_x = force_y = force_z = 0.0;
            torque_x = torque_y = torque_z = 0.0;
            for(std::size_t i = 0u; i < markers_.size(); ++i) {
                const Marker& marker = markers_[i];
                const double volume = (double)marker.area_lu2*
                    (double)config_.support_thickness_lu;
                const double rxn = -volume*(double)marker_force_density_[i].x;
                const double ryn = -volume*(double)marker_force_density_[i].y;
                const double rzn = -volume*(double)marker_force_density_[i].z;
                force_x += rxn;
                force_y += ryn;
                force_z += rzn;
                torque_x += (double)marker.position_lu.y*rzn-
                            (double)marker.position_lu.z*ryn;
                torque_y += (double)marker.position_lu.z*rxn-
                            (double)marker.position_lu.x*rzn;
                torque_z += (double)marker.position_lu.x*ryn-
                            (double)marker.position_lu.y*rxn;
            }
        }
        statistics_.valid = true;
        statistics_.iterations = iterations;
        statistics_.converged =
            measurement.max_slip <= config_.residual_tolerance;
        if(!statistics_.converged) ++statistics_.nonconverged_steps;
        statistics_.max_slip_lu = measurement.max_slip;
        statistics_.peak_max_slip_lu = std::max(
            statistics_.peak_max_slip_lu, measurement.max_slip);
        statistics_.rms_slip_lu = measurement.volume > 0.0
            ? (float)std::sqrt(
                measurement.weighted_slip_squared/measurement.volume)
            : 0.0f;
        statistics_.reaction_force_lu = float3(
            (float)force_x, (float)force_y, (float)force_z);
        statistics_.reaction_torque_lu = float3(
            (float)torque_x, (float)torque_y, (float)torque_z);
        if(!statistics_.converged &&
           iterations>=config_.forcing_iterations &&
           !nonconvergence_warned_ && culbm::mpi_is_root()) {
            log_warning(
                "SurfaceIBM reached its forcing-iteration limit without "
                "meeting culbm_surface_ibm_residual_tolerance (max slip="+
                std::to_string(statistics_.max_slip_lu)+
                ", tolerance="+
                std::to_string(config_.residual_tolerance)+
                "). Refine marker spacing or adjust the validated forcing "
                "parameters; the step continues with explicit diagnostics.");
            nonconvergence_warned_ = true;
        }
    }

public:
    Impl(culbm::solver::Solver& solver, const std::vector<Marker>& markers,
         const Config& config)
        : solver_(solver), config_(config), markers_(markers) {
        std::sort(markers_.begin(), markers_.end(),
            [](const Marker& a, const Marker& b) {
                return a.global_id < b.global_id;
            });
        validate_markers();
        if(solver_.local_count() > 0u) {
            wm_lattice_ = solver_.local(0u).wall_model();
            wall_model_ = wm_lattice_.enabled && wm_lattice_.ibm;
        }
        statistics_.wall_model = wall_model_;
        marker_force_density_.assign(markers_.size(), float3(0.0f));
        marker_force_delta_.assign(markers_.size(), float3(0.0f));
        const std::size_t reduction_markers = std::min<std::size_t>(
            markers_.size(), (std::size_t)reduction_marker_chunk);
        reduction_values_.resize(6u*reduction_markers);
        statistics_.markers = (std::uint64_t)markers_.size();
        build_domain_fragments();
        build_marker_coupling();
        enable_device_direct_path();
        if(wall_model_) {
            // Replace voxel shear only on core faces served by a marker.
            statistics_.wall_model_core_cells = mark_core_cells();
            mark_wall_model_faces();
            if(!device_direct_) {
                if(!culbm::mpi_is_distributed() && domains_.size() > 1u) {
                    enable_multi_gpu_wall_model_path();
                } else {
                    wm_spread_value_.assign(markers_.size(), float3(0.0f));
                    wm_side_.assign(markers_.size(), 1.0f);
                    // 12 reduction lanes per marker for the two-sided probe.
                    reduction_values_.resize(12u*std::min<std::size_t>(
                        markers_.size(), (std::size_t)reduction_marker_chunk/2u));
                }
            }
        } else {
            statistics_.cleared_solid_cells = clear_marker_support_solids();
        }
        if(device_direct_) {
            domains_.front()->release_device_direct_host_storage();
            std::vector<Marker>().swap(markers_);
        }
    }

    void prepare_step() {
        // Distributions contain the previous step's IBM response.  Clear the
        // explicit field, then force the existing update_fields kernel to
        // reconstruct the base physical velocity with every non-IBM Guo term.
        if(!device_direct_ && !multi_gpu_wm_device_) {
            std::fill(marker_force_density_.begin(),
                      marker_force_density_.end(), float3(0.0f));
            std::fill(marker_force_delta_.begin(), marker_force_delta_.end(),
                      float3(0.0f));
        }
        for(const auto& state : domains_) state->enqueue_reset_force();
        if(device_direct_ && !wall_model_) domains_.front()->enqueue_begin_device_direct_step();
        for(const auto& state : domains_) {
            if(state->active_cell_count()==0ull) continue;
            state->domain->ibm_base_predictor(
                *state->active_cells, state->active_cell_count());
        }

        if(wall_model_) {
            if(device_direct_) {
                DomainState& state = *domains_.front();
                state.enqueue_device_wall_model_step();
                state.domain->synchronize();
                device_reaction_force_ = float3(
                    (*(state.reaction))[0], (*(state.reaction))[1],
                    (*(state.reaction))[2]);
                device_reaction_torque_ = float3(
                    (*(state.reaction))[3], (*(state.reaction))[4],
                    (*(state.reaction))[5]);
                if(!finite3(device_reaction_force_) ||
                   !finite3(device_reaction_torque_))
                    throw std::runtime_error(
                        "SurfaceIBM wall-model reaction became non-finite.");
            } else if(multi_gpu_wm_device_) {
                apply_wall_model_multi_gpu_device();
            } else {
                apply_wall_model_portable();
            }
            finalize_statistics(Measurement{}, 0u);
            return;
        }
        if(device_direct_) {
            unsigned applied_iterations = 0u;
            const Measurement final_measurement =
                run_device_iterations(applied_iterations);
            finalize_statistics(final_measurement, applied_iterations);
            return;
        }

        unsigned applied_iterations = 0u;
        Measurement final_measurement;
        bool have_final_measurement = false;
        for(unsigned iteration = 0u;
            iteration < config_.forcing_iterations; ++iteration) {
            measure(true);
            spread_marker_force_delta();
            ++applied_iterations;
            final_measurement = measure(false);
            have_final_measurement = true;
            if(final_measurement.max_slip <= config_.residual_tolerance) break;
        }
        if(!have_final_measurement)
            final_measurement = measure(false);
        finalize_statistics(final_measurement, applied_iterations);
    }

    Statistics statistics() const { return statistics_; }
};

Runtime::Runtime(culbm::solver::Solver& solver, const std::vector<Marker>& markers,
                 const Config& config) {
    try {
        impl_ = std::make_unique<Impl>(solver, markers, config);
    } catch(const std::bad_alloc&) {
        // Rank-local host allocation failure while peers are entering later
        // collectives cannot safely unwind as an ordinary exception.
        if(culbm::mpi_is_distributed())
            culbm::mpi_abort(
                "SurfaceIBM host allocation failed during collective attach");
        throw;
    }
}

Runtime::~Runtime() = default;

void Runtime::prepare_step() { impl_->prepare_step(); }

Statistics Runtime::statistics() const { return impl_->statistics(); }

void attach(culbm::solver::Solver& solver, const std::vector<Marker>& markers) {
    const Config& config = solver.features().surface_ibm;
    // The collective preflight is the first check that can reject: every
    // rank takes part even when its local state differs.
    verify_attach_preflight(config, markers,
        solver.initialized() || solver.ever_initialized(),
        solver.has_immersed_forcing(), solver.has_ibm_storage());
    std::unique_ptr<Runtime> runtime;
    try {
        runtime = std::make_unique<Runtime>(solver, markers, config);
    } catch(const std::bad_alloc&) {
        // Also covers the allocation of the runtime object itself.
        if(culbm::mpi_is_distributed())
            culbm::mpi_abort("SurfaceIBM runtime allocation failed during collective attach");
        throw;
    }
    solver.attach_immersed_forcing(std::move(runtime));
}

Statistics statistics(const culbm::solver::Solver& solver) {
    const Runtime* const runtime = dynamic_cast<const Runtime*>(solver.immersed_forcing());
    return runtime != nullptr ? runtime->statistics() : Statistics{};
}

} // namespace culbm::surface_ibm
