#include "vk_inlet.hpp"
#include "../patch_fields.hpp"

#include <array>
#include <cfloat>
#include <limits>
#include <random>

namespace culbm::bc {

namespace {

static uint tracked_buffer_mb_setup(const ulong bytes) {
    const ulong whole_mb = bytes / 1048576ull;
    const ulong rounded_mb = whole_mb + (bytes % 1048576ull != 0ull);
    return (uint)std::min<ulong>((ulong)max_uint, rounded_mb);
}

} // namespace

const char* vk_face_mode_name(const VkInletFaceMode mode) {
    if (mode == VkInletFaceMode::AUTO_SIDES) return "AUTO_SIDES";
    if (mode == VkInletFaceMode::EXCLUDE_DOWNSTREAM) return "EXCLUDE_DOWNSTREAM";
    if (mode == VkInletFaceMode::EXCLUDE_DOWNSTREAM_SIDES) return "EXCLUDE_DOWNSTREAM_SIDES";
    if (mode == VkInletFaceMode::ALL_SIDES) return "ALL_SIDES";
    if (mode == VkInletFaceMode::ALL_SELECTED) return "ALL_SELECTED";
    return "TARGET_INFLOW";
}
VkInletFaceMode resolve_vk_face_mode(const VkInletFaceMode mode, const bool inflow_only) {
    if (mode != VkInletFaceMode::AUTO_SIDES) return mode;
    return inflow_only ? VkInletFaceMode::EXCLUDE_DOWNSTREAM_SIDES : VkInletFaceMode::ALL_SIDES;
}
const char* vk_uc_mode_name(const VkInletUcMode mode) {
    return (mode == VkInletUcMode::NORM_MEAN) ? "NORM_MEAN" : "NORMAL_COMPONENT";
}
uint estimate_vk_inlet_extra_mb_per_device(const VkInletSettings& vk_inlet_settings,
                                           const uint Nx, const uint Ny, const uint Nz) {
    if (!vk_inlet_settings.enable || Nx < 2u || Ny < 2u || Nz < 2u) {
        return 0u;
    }

    // Conservative upper bound before boundary flags are known exactly:
    // 4 side planes (excluding bottom/top overlap) plus the full top plane.
    const ulong nz_side = Nz > 2u ? (ulong)(Nz - 2u) : 0ull;
    const ulong nx_inner = Nx > 2u ? (ulong)(Nx - 2u) : 0ull;
    const ulong point_count_upper =
        2ull * (ulong)Ny * nz_side +
        2ull * nx_inner * nz_side +
        (ulong)Nx * (ulong)Ny;
    const ulong mode_stride = 5ull * (ulong)std::max(1, vk_inlet_settings.nmodes);

    uint memory_mb = 0u;
    memory_mb += tracked_buffer_mb_setup(point_count_upper * sizeof(ulong)); // point_cell
    memory_mb += tracked_buffer_mb_setup(point_count_upper * sizeof(uchar)); // point_face
    memory_mb += tracked_buffer_mb_setup(point_count_upper * 7ull * sizeof(float)); // point_data
    memory_mb += tracked_buffer_mb_setup(mode_stride * 10ull * sizeof(float)); // mode_data
    return memory_mb;
}

VkInletRuntimeConfig make_vk_runtime_config(const VkInletSettings& vk_inlet_settings,
                                            const std::string& downstream_bc_runtime) {
    VkInletRuntimeConfig cfg;
    cfg.enable = vk_inlet_settings.enable;
    cfg.ti = vk_inlet_settings.ti;
    cfg.sigma_lbm = units.u(vk_inlet_settings.sigma_si);
    cfg.L_lbm = units.x(vk_inlet_settings.L_si);
    cfg.nmodes = vk_inlet_settings.nmodes;
    cfg.seed = vk_inlet_settings.seed;
    cfg.update_stride = vk_inlet_settings.update_stride;
    cfg.uc_mode = vk_inlet_settings.uc_mode;
    cfg.same_realization_all_faces = vk_inlet_settings.same_realization_all_faces;
    cfg.stride_interpolation = vk_inlet_settings.stride_interpolation;
    cfg.inflow_only = vk_inlet_settings.inflow_only;
    cfg.face_mode = resolve_vk_face_mode(vk_inlet_settings.face_mode, vk_inlet_settings.inflow_only);
    cfg.anisotropy_scale = vk_inlet_settings.anisotropy_scale;
    // Keep parsed for compatibility with existing config/deck fields.
    cfg.downstream_face_id = downstream_face_id_from_bc(downstream_bc_runtime);
    if (cfg.enable && !(cfg.L_lbm > 0.0f)) {
        println("| WARNING: vk_inlet_l converts to non-positive LBM value. Disabled.            |");
        cfg.enable = false;
    }
    if (cfg.enable && !(cfg.ti > 0.0f || cfg.sigma_lbm > 0.0f)) {
        println("| WARNING: vk_inlet_ti/sigma converts to non-positive LBM value. Disabled.     |");
        cfg.enable = false;
    }
    if (cfg.enable && cfg.ti > 0.0f && cfg.sigma_lbm > 0.0f) {
        println("| VK inlet note   | TI mode is active; vk_inlet_sigma acts only as fallback.   |");
    }
    if (!(std::isfinite(cfg.anisotropy_scale.x) && std::isfinite(cfg.anisotropy_scale.y) && std::isfinite(cfg.anisotropy_scale.z) &&
          cfg.anisotropy_scale.x >= 0.0f && cfg.anisotropy_scale.y >= 0.0f && cfg.anisotropy_scale.z >= 0.0f)) {
        println("| WARNING: vk_inlet_anisotropy has invalid component(s). Reset to [1,1,1].     |");
        cfg.anisotropy_scale = float3(1.0f, 1.0f, 1.0f);
    }
    if (cfg.nmodes > k_vk_nmodes_max) cfg.nmodes = k_vk_nmodes_max;
    if (cfg.nmodes <= 0) cfg.nmodes = k_vk_nmodes_default;
    if (cfg.update_stride <= 0) cfg.update_stride = 1;
    return cfg;
}

class VonKarmanInletUpdater::Impl {
public:
    explicit Impl(const VkInletRuntimeConfig& cfg) : cfg_(cfg) {}

    bool initialize(culbm::solver::Solver& solver) {
        active_ = false;
        domains_.clear();
        last_applied_time_ = -std::numeric_limits<double>::infinity();
        for (auto& modes : face_modes_) modes.clear();
        if (!cfg_.enable) return false;
        if (!(cfg_.L_lbm > 0.0f) || cfg_.nmodes <= 0) return false;

        const uint Nx = solver.grid().size(0u);
        const uint Ny = solver.grid().size(1u);
        const uint Nz = solver.grid().size(2u);
        if (Nx < 2u || Ny < 2u || Nz < 2u) {
            println("| VK inlet        | disabled: grid too small for turbulent inlet faces         |");
            return false;
        }

        setup_face_geometry_();
        const int target_face = opposite_side_face_(cfg_.downstream_face_id);
        if (cfg_.face_mode == VkInletFaceMode::EXCLUDE_DOWNSTREAM) {
            if (cfg_.downstream_face_id >= 0) {
                println("| VK inlet target | all except downstream " + string(face_name_(cfg_.downstream_face_id)) +
                        "                                    |");
            } else {
                println("| VK inlet target | all selected faces; downstream unknown                     |");
            }
        } else if (cfg_.face_mode == VkInletFaceMode::EXCLUDE_DOWNSTREAM_SIDES) {
            if (cfg_.downstream_face_id >= 0) {
                println("| VK inlet target | side faces except downstream " + string(face_name_(cfg_.downstream_face_id)) +
                        "                               |");
            } else {
                println("| VK inlet target | side faces; downstream unknown; top excluded               |");
            }
        } else if (cfg_.face_mode == VkInletFaceMode::ALL_SIDES) {
            println("| VK inlet target | all side faces; top excluded                             |");
        } else if (cfg_.face_mode == VkInletFaceMode::TARGET_INFLOW && target_face >= 0) {
            println("| VK inlet target | " + string(face_name_(target_face)) +
                    " (opposite downstream " + string(face_name_(cfg_.downstream_face_id)) + ")                         |");
        } else if (cfg_.inflow_only) {
            println("| VK inlet target | velocity-normal side faces; top excluded                  |");
        } else {
            println("| VK inlet warn   | all boundary-face mode also includes outlet/side/top cells |");
        }
        collect_face_points_(solver);

        uint face_enabled_count = 0u;
        float3 mean_u_all = float3(0.0f);
        double sum_u_mag = 0.0;
        ulong count_u = 0ull;
        for (size_t i = 0u; i < faces_.size(); ++i) {
            VkFaceData& face = faces_[i];
            if (!face.enabled || face.points.empty()) continue;
            face_enabled_count++;
            for (const VkPoint& p : face.points) {
                mean_u_all += p.base_u;
                sum_u_mag += (double)length(p.base_u);
                count_u++;
            }
            println("| VK inlet face   | " + string(face_name_(face.id)) +
                    ": points=" + to_string((ulong)face.points.size()) +
                    ", Uc=" + to_string(face.uc, 6u) + "                                  |");
        }

        if (face_enabled_count == 0u) {
            println("| VK inlet        | enabled in config, but no valid inflow faces found         |");
            return false;
        }

        if (count_u == 0ull) return false;
        const float u_ref = (float)(sum_u_mag / (double)count_u);
        const float3 mean_u = mean_u_all / (float)count_u;
        float3 conv_dir = mean_u;
        const float conv_len = length(conv_dir);
        if (conv_len > 1.0e-7f) conv_dir /= conv_len;
        else conv_dir = float3(1.0f, 0.0f, 0.0f);

        if (cfg_.same_realization_all_faces) {
            println("| VK inlet        | same realization on all active faces                        |");
        } else {
            println("| VK inlet        | independent realization per active face                     |");
        }
        string filter_desc;
        if (cfg_.face_mode == VkInletFaceMode::EXCLUDE_DOWNSTREAM) {
            filter_desc = (cfg_.downstream_face_id >= 0) ? "all cells except downstream face" : "all selected boundary face cells";
        } else if (cfg_.face_mode == VkInletFaceMode::EXCLUDE_DOWNSTREAM_SIDES) {
            filter_desc = (cfg_.downstream_face_id >= 0) ? "side faces except downstream" : "side faces with top excluded";
        } else if (cfg_.face_mode == VkInletFaceMode::ALL_SIDES) {
            filter_desc = "all side faces";
        } else if (cfg_.face_mode == VkInletFaceMode::TARGET_INFLOW && target_face >= 0) {
            filter_desc = "target face cells from downstream";
        } else {
            filter_desc = "all selected boundary face cells";
        }
        println("| VK inlet filter | " + filter_desc + "                  |");
        if (!build_face_modes_(u_ref, conv_dir, cfg_.seed)) {
            println("| VK inlet        | failed to build VK spectrum modes                           |");
            return false;
        }

        if (!build_gpu_runtime_(solver)) {
            println("| VK inlet        | no valid GPU runtime mapping for inflow faces              |");
            return false;
        }

        active_ = true;
        println("| VK inlet        | active: L_lbm=" + to_string(cfg_.L_lbm, 6u) +
                ", TI=" + to_string(cfg_.ti, 4u) +
                ", sigma_lbm(fallback)=" + to_string(cfg_.sigma_lbm, 6u) +
                ", stride=" + to_string((ulong)cfg_.update_stride) +
                ", uc_mode=" + string(vk_uc_mode_name(cfg_.uc_mode)) +
                ", face_mode=" + string(vk_face_mode_name(cfg_.face_mode)) +
                ", aniso=[" + to_string(cfg_.anisotropy_scale.x, 3u) + "," +
                             to_string(cfg_.anisotropy_scale.y, 3u) + "," +
                             to_string(cfg_.anisotropy_scale.z, 3u) + "]" +
                ", inflow_only=" + string(cfg_.inflow_only ? "true" : "false") +
                ", interp=" + string(cfg_.stride_interpolation ? "true" : "false") + " |");
        println("| VK inlet modes  | per-face modes=" + to_string((ulong)cfg_.nmodes) + "                                  |");
        return true;
    }

    bool active() const { return active_; }

    bool update(culbm::solver::Solver& solver, const ulong t) {
        return update_at_lattice_time(solver, static_cast<double>(t));
    }

    bool update_at_lattice_time(culbm::solver::Solver& solver, const double lattice_time) {
        (void)solver;
        if (!active_) return false;
        if (!(std::isfinite(lattice_time) && lattice_time >= 0.0)) return false;
        if (lattice_time == last_applied_time_) return false;
        last_applied_time_ = lattice_time;

        uint use_interp = 0u;
        float t0 = 0.0f;
        float t1 = 0.0f;
        float alpha = 0.0f;
        compute_time_params_(lattice_time, use_interp, t0, t1, alpha);

        bool updated = false;
        for (const auto& rt_ptr : domains_) {
            if (!rt_ptr || !rt_ptr->active) continue;
            DomainRuntime& rt = *rt_ptr;
            rt.kernel_apply.bind(0u, use_interp, t0, t1, alpha).submit();
            updated = true;
        }
        return updated;
    }

private:
    enum VkFaceId : int {
        WEST = 0,
        EAST = 1,
        SOUTH = 2,
        NORTH = 3,
        TOP = 4
    };
    static constexpr int VK_FACE_COUNT = 5;

    struct VkMode {
        float kx = 0.0f;
        float ky = 0.0f;
        float kz = 0.0f;
        float omega = 0.0f;
        float Ax = 0.0f;
        float Ay = 0.0f;
        float Az = 0.0f;
        float phix = 0.0f;
        float phiy = 0.0f;
        float phiz = 0.0f;
    };

    struct VkPoint {
        ulong n = 0ull;
        uint x = 0u;
        uint y = 0u;
        uint z = 0u;
        float3 base_u = float3(0.0f);
    };

    struct VkFaceData {
        int id = WEST;
        float3 n = float3(0.0f);
        float3 t1 = float3(0.0f);
        float3 t2 = float3(0.0f);
        float uc = 0.0f;
        bool enabled = false;
        std::vector<VkPoint> points;
    };

    struct DomainPoint {
        ulong local_n = 0ull;
        uchar face_id = 0u;
        float px = 0.0f;
        float py = 0.0f;
        float pz = 0.0f;
        float3 base_u = float3(0.0f);
        float sigma = 0.0f;
    };

    struct DomainRuntime {
        bool active = false;
        ulong point_count = 0ull;
        ulong mode_count = 0ull;
        ulong mode_stride = 0ull;
        gpu::MirroredBuffer<ulong> point_cell;
        gpu::MirroredBuffer<uchar> point_face;
        gpu::MirroredBuffer<float> point_data; // SoA: px, py, pz, base_u.x, base_u.y, base_u.z, sigma
        gpu::MirroredBuffer<float> mode_data;  // SoA over 4 faces: kx, ky, kz, omega, Ax, Ay, Az, phix, phiy, phiz
        gpu::Launch kernel_apply;
    };

    static const char* face_name_(const int id) {
        if (id == WEST) return "west";
        if (id == EAST) return "east";
        if (id == SOUTH) return "south";
        if (id == NORTH) return "north";
        if (id == TOP) return "top";
        return "unknown";
    }

    static int opposite_side_face_(const int id) {
        if (id == WEST) return EAST;
        if (id == EAST) return WEST;
        if (id == SOUTH) return NORTH;
        if (id == NORTH) return SOUTH;
        return -1;
    }

    void setup_face_geometry_() {
        faces_[0].id = WEST;
        faces_[0].n = float3(+1.0f, 0.0f, 0.0f);
        faces_[0].t1 = float3(0.0f, +1.0f, 0.0f);
        faces_[0].t2 = float3(0.0f, 0.0f, +1.0f);

        faces_[1].id = EAST;
        faces_[1].n = float3(-1.0f, 0.0f, 0.0f);
        faces_[1].t1 = float3(0.0f, +1.0f, 0.0f);
        faces_[1].t2 = float3(0.0f, 0.0f, +1.0f);

        faces_[2].id = SOUTH;
        faces_[2].n = float3(0.0f, +1.0f, 0.0f);
        faces_[2].t1 = float3(+1.0f, 0.0f, 0.0f);
        faces_[2].t2 = float3(0.0f, 0.0f, +1.0f);

        faces_[3].id = NORTH;
        faces_[3].n = float3(0.0f, -1.0f, 0.0f);
        faces_[3].t1 = float3(+1.0f, 0.0f, 0.0f);
        faces_[3].t2 = float3(0.0f, 0.0f, +1.0f);

        faces_[4].id = TOP;
        faces_[4].n = float3(0.0f, 0.0f, -1.0f);
        faces_[4].t1 = float3(+1.0f, 0.0f, 0.0f);
        faces_[4].t2 = float3(0.0f, +1.0f, 0.0f);
    }

    bool cell_is_valid_inlet_(culbm::solver::Solver& solver, const ulong n, const VkFaceData& face, const uint x, const uint y, const uint z) {
        (void)x;
        (void)y;
        if (z == 0u) return false; // keep bottom excluded
        if ((solver.flags()[n] & CELL_SOLID) != 0u) return false;
        if ((solver.flags()[n] & CELL_EQUILIBRIUM) == 0u) return false;
        if (face.id == TOP && !cfg_.allow_top_face) return false;

        const int target_face = opposite_side_face_(cfg_.downstream_face_id);
        if (cfg_.face_mode == VkInletFaceMode::TARGET_INFLOW) {
            if (target_face >= 0 && face.id != target_face) return false;
            if (target_face < 0 && face.id == TOP && cfg_.inflow_only) return false;
        } else if (cfg_.face_mode == VkInletFaceMode::EXCLUDE_DOWNSTREAM) {
            if (cfg_.downstream_face_id >= 0 && face.id == cfg_.downstream_face_id) return false;
        } else if (cfg_.face_mode == VkInletFaceMode::EXCLUDE_DOWNSTREAM_SIDES) {
            if (face.id == TOP) return false;
            if (cfg_.downstream_face_id >= 0 && face.id == cfg_.downstream_face_id) return false;
        } else if (cfg_.face_mode == VkInletFaceMode::ALL_SIDES) {
            if (face.id == TOP) return false;
        } else if (face.id == TOP && cfg_.inflow_only) {
            return false;
        }
        return true;
    }

    void add_point_(VkFaceData& face, const ulong n, const uint x, const uint y, const uint z, culbm::solver::Solver& solver) {
        VkPoint p;
        p.n = n;
        p.x = x;
        p.y = y;
        p.z = z;
        p.base_u = float3(solver.velocity().x[n], solver.velocity().y[n], solver.velocity().z[n]);
        face.points.push_back(p);
    }

    void collect_face_points_(culbm::solver::Solver& solver) {
        for (auto& face : faces_) {
            face.points.clear();
            face.uc = 0.0f;
            face.enabled = false;
        }

        const uint Nx = solver.grid().size(0u);
        const uint Ny = solver.grid().size(1u);
        const uint Nz = solver.grid().size(2u);
        if (Nx < 2u || Ny < 2u || Nz < 2u) return;

        // West and east include corners; south and north skip x corners to keep a deterministic exclusive ownership.
        for (uint z = 1u; z + 1u < Nz; ++z) {
            for (uint y = 0u; y < Ny; ++y) {
                const ulong n_w = solver.grid().index(0u, y, z);
                if (cell_is_valid_inlet_(solver, n_w, faces_[WEST], 0u, y, z)) {
                    add_point_(faces_[WEST], n_w, 0u, y, z, solver);
                }
                const ulong n_e = solver.grid().index(Nx - 1u, y, z);
                if (cell_is_valid_inlet_(solver, n_e, faces_[EAST], Nx - 1u, y, z)) {
                    add_point_(faces_[EAST], n_e, Nx - 1u, y, z, solver);
                }
            }
            if (Nx > 2u) {
                for (uint x = 1u; x + 1u < Nx; ++x) {
                    const ulong n_s = solver.grid().index(x, 0u, z);
                    if (cell_is_valid_inlet_(solver, n_s, faces_[SOUTH], x, 0u, z)) {
                        add_point_(faces_[SOUTH], n_s, x, 0u, z, solver);
                    }
                    const ulong n_n = solver.grid().index(x, Ny - 1u, z);
                    if (cell_is_valid_inlet_(solver, n_n, faces_[NORTH], x, Ny - 1u, z)) {
                        add_point_(faces_[NORTH], n_n, x, Ny - 1u, z, solver);
                    }
                }
            }
        }

        // Top face includes full plane (z = Nz-1). No overlap with side loops above.
        const uint z_top = Nz - 1u;
        for (uint y = 0u; y < Ny; ++y) {
            for (uint x = 0u; x < Nx; ++x) {
                const ulong n_t = solver.grid().index(x, y, z_top);
                if (cell_is_valid_inlet_(solver, n_t, faces_[TOP], x, y, z_top)) {
                    add_point_(faces_[TOP], n_t, x, y, z_top, solver);
                }
            }
        }

        for (auto& face : faces_) {
            if (face.points.empty()) continue;
            float3 mean_u = float3(0.0f);
            for (const VkPoint& p : face.points) mean_u += p.base_u;
            mean_u /= (float)face.points.size();
            face.uc = (cfg_.uc_mode == VkInletUcMode::NORM_MEAN)
                ? length(mean_u)
                : fabsf(dot(mean_u, face.n));

            if (!(face.uc > 1.0e-7f)) {
                println("| VK inlet face   | " + string(face_name_(face.id)) + ": disabled (Uc is too small)               |");
                continue;
            }
            face.enabled = true;
        }
    }

    static ulong mix_seed_(const ulong seed, const uint face_id) {
        ulong x = seed ^ (0x9E3779B97F4A7C15ull * (ulong)(face_id + 1u));
        x ^= (x >> 33u);
        x *= 0xff51afd7ed558ccdull;
        x ^= (x >> 33u);
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= (x >> 33u);
        return x;
    }

    bool build_modes_for_seed_(const float u_ref, const float3& conv_dir, const ulong seed, std::vector<VkMode>& out_modes) {
        out_modes.clear();
        const float L = cfg_.L_lbm;
        if (!(L > 0.0f) || cfg_.nmodes <= 0) return false;

        const float delta_min = 1.0f; // lattice spacing on side planes
        const float k_max = pif / delta_min;
        float k_min = 2.0f * pif / (10.0f * L);
        if (!(k_min > 0.0f) || !std::isfinite(k_min)) k_min = 1.0e-4f;
        if (k_min >= 0.99f * k_max) {
            k_min = 0.1f * k_max;
            println("| VK inlet        | warning: adjusted k_min to keep k-band valid               |");
        }
        const float log_k_min = logf(k_min);
        const float log_k_max = logf(k_max);
        const float log_k_span = fmaxf(log_k_max - log_k_min, 1.0e-6f);

        std::mt19937_64 rng((unsigned long long)seed);
        std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

        std::vector<float> a_raw((size_t)cfg_.nmodes, 0.0f);
        out_modes.resize((size_t)cfg_.nmodes);
        double sum_a2 = 0.0;

        for (int m = 0; m < cfg_.nmodes; ++m) {
            const float xi = ((float)m + uni01(rng)) / (float)cfg_.nmodes;
            const float k = expf(log_k_min + xi * log_k_span);

            // Isotropic direction in local (t1, t2, n) coordinates.
            const float zeta = 2.0f * uni01(rng) - 1.0f;
            const float az = 2.0f * pif * uni01(rng);
            const float r = sqrtf(fmaxf(0.0f, 1.0f - zeta * zeta));
            const float dir_x = r * cosf(az);
            const float dir_y = r * sinf(az);
            const float dir_z = zeta;
            const float kx = k * dir_x;
            const float ky = k * dir_y;
            const float kz = k * dir_z;

            const float kL = k * L;
            const float denom = powf(1.0f + kL * kL, 17.0f / 6.0f);
            const float W = (denom > 0.0f) ? (powf(k, 4.0f) / denom) : 0.0f;
            const float a = sqrtf(fmaxf(W, 0.0f));
            a_raw[(size_t)m] = a;
            sum_a2 += (double)a * (double)a;

            VkMode mode;
            mode.kx = kx;
            mode.ky = ky;
            mode.kz = kz;
            mode.omega = u_ref * (kx * conv_dir.x + ky * conv_dir.y + kz * conv_dir.z);
            mode.phix = 2.0f * pif * uni01(rng);
            mode.phiy = 2.0f * pif * uni01(rng);
            mode.phiz = 2.0f * pif * uni01(rng);
            out_modes[(size_t)m] = mode;
        }

        const double variance_raw = 0.5 * sum_a2;
        if (!(variance_raw > 0.0)) {
            out_modes.clear();
            return false;
        }
        const float scale = 1.0f / (float)sqrt(variance_raw); // unit-RMS basis, point-wise sigma applies later
        const float ax = cfg_.anisotropy_scale.x;
        const float ay = cfg_.anisotropy_scale.y;
        const float az = cfg_.anisotropy_scale.z;
        for (size_t m = 0u; m < out_modes.size(); ++m) {
            const float A = a_raw[m] * scale;
            out_modes[m].Ax = A * ax;
            out_modes[m].Ay = A * ay;
            out_modes[m].Az = A * az;
        }
        return true;
    }

    bool build_face_modes_(const float u_ref, const float3& conv_dir, const ulong seed) {
        for (auto& modes : face_modes_) modes.clear();

        bool any_active = false;
        for (const auto& face : faces_) {
            if (face.enabled && !face.points.empty()) {
                any_active = true;
                break;
            }
        }
        if (!any_active) return false;

        if (cfg_.same_realization_all_faces) {
            std::vector<VkMode> shared;
            if (!build_modes_for_seed_(u_ref, conv_dir, seed, shared)) return false;
            for (int fid = 0; fid < VK_FACE_COUNT; ++fid) {
                if (faces_[(size_t)fid].enabled && !faces_[(size_t)fid].points.empty()) {
                    face_modes_[(size_t)fid] = shared;
                }
            }
            return true;
        }

        bool built_any = false;
        for (int fid = 0; fid < VK_FACE_COUNT; ++fid) {
            if (!faces_[(size_t)fid].enabled || faces_[(size_t)fid].points.empty()) continue;
            if (!build_modes_for_seed_(u_ref, conv_dir, mix_seed_(seed, (uint)fid), face_modes_[(size_t)fid])) {
                return false;
            }
            built_any = true;
        }
        return built_any;
    }

    bool build_gpu_runtime_(culbm::solver::Solver& solver) {
        const ulong mode_count = (ulong)cfg_.nmodes;
        if (mode_count == 0ull) return false;
        for (int fid = 0; fid < VK_FACE_COUNT; ++fid) {
            if (!faces_[(size_t)fid].enabled || faces_[(size_t)fid].points.empty()) continue;
            if (face_modes_[(size_t)fid].size() != (size_t)mode_count) return false;
        }

        const ulong mode_stride = (ulong)VK_FACE_COUNT * mode_count;
        std::vector<float> mode_data((size_t)(10ull * mode_stride), 0.0f);
        auto set_mode = [&](const ulong face_id, const ulong m, const VkMode& mode) {
            const ulong idx = face_id * mode_count + m;
            mode_data[(size_t)idx] = mode.kx;
            mode_data[(size_t)(mode_stride + idx)] = mode.ky;
            mode_data[(size_t)(2ull * mode_stride + idx)] = mode.kz;
            mode_data[(size_t)(3ull * mode_stride + idx)] = mode.omega;
            mode_data[(size_t)(4ull * mode_stride + idx)] = mode.Ax * cosf(mode.phix);
            mode_data[(size_t)(5ull * mode_stride + idx)] = -mode.Ax * sinf(mode.phix);
            mode_data[(size_t)(6ull * mode_stride + idx)] = mode.Ay * cosf(mode.phiy);
            mode_data[(size_t)(7ull * mode_stride + idx)] = -mode.Ay * sinf(mode.phiy);
            mode_data[(size_t)(8ull * mode_stride + idx)] = mode.Az * cosf(mode.phiz);
            mode_data[(size_t)(9ull * mode_stride + idx)] = -mode.Az * sinf(mode.phiz);
        };
        for (int fid = 0; fid < VK_FACE_COUNT; ++fid) {
            const auto& modes = face_modes_[(size_t)fid];
            for (ulong m = 0ull; m < (ulong)modes.size(); ++m) {
                set_mode((ulong)fid, m, modes[(size_t)m]);
            }
        }

        const uint D = solver.grid().count();
        const uint Dx = solver.grid().split(0u);
        const uint Dy = solver.grid().split(1u);
        const uint Dz = solver.grid().split(2u);
        const uint nx_inner = solver.grid().size(0u) / Dx;
        const uint ny_inner = solver.grid().size(1u) / Dy;
        const uint nz_inner = solver.grid().size(2u) / Dz;
        const uint Hx = Dx > 1u ? 1u : 0u;
        const uint Hy = Dy > 1u ? 1u : 0u;
        const uint Hz = Dz > 1u ? 1u : 0u;

        std::vector<std::vector<DomainPoint>> domain_points(D);
        double sigma_sum_global = 0.0;
        ulong sigma_count_global = 0ull;
        float sigma_min_global = FLT_MAX;
        float sigma_max_global = 0.0f;
        std::array<double, VK_FACE_COUNT> sigma_sum_face = {};
        std::array<ulong, VK_FACE_COUNT> sigma_count_face = {};
        std::array<float, VK_FACE_COUNT> sigma_min_face = {};
        std::array<float, VK_FACE_COUNT> sigma_max_face = {};
        sigma_min_face.fill(FLT_MAX);
        sigma_max_face.fill(0.0f);
        ulong map_mismatch_count = 0ull;
        ulong face_mismatch_count = 0ull;
        ulong local_flag_mismatch_count = 0ull;
        for (const VkFaceData& face : faces_) {
            if (!face.enabled || face.points.empty()) continue;
            for (const VkPoint& p : face.points) {
                const uint x = p.x;
                const uint y = p.y;
                const uint z = p.z;
                const uint dx = x / nx_inner;
                const uint dy = y / ny_inner;
                const uint dz = z / nz_inner;
                const uint domain = dx + (dy + dz * Dy) * Dx;
                if (domain >= D || !solver.owns(domain)) continue;
                const uint local_domain = (solver.distributed() ? 0u : domain);

                const ulong lNx = solver.local(local_domain).extent(0u);
                const ulong lNy = solver.local(local_domain).extent(1u);
                const uint px = x % nx_inner;
                const uint py = y % ny_inner;
                const uint pz = z % nz_inner;
                const ulong lx = (ulong)px + (ulong)Hx;
                const ulong ly = (ulong)py + (ulong)Hy;
                const ulong lz = (ulong)pz + (ulong)Hz;

                const float u_char = (cfg_.uc_mode == VkInletUcMode::NORM_MEAN)
                    ? length(p.base_u)
                    : fabsf(dot(p.base_u, face.n));
                const float sigma_local = cfg_.ti > 0.0f ? (cfg_.ti * u_char) : cfg_.sigma_lbm;
                if (!(sigma_local > 0.0f)) continue;

                DomainPoint dp;
                dp.local_n = lx + (ly + lz * lNy) * lNx;
                dp.face_id = (uchar)face.id;
                dp.px = cfg_.coordinate_scale * (float)x + cfg_.coordinate_offset.x;
                dp.py = cfg_.coordinate_scale * (float)y + cfg_.coordinate_offset.y;
                dp.pz = (face.id == TOP && cfg_.has_top_coordinate_z)
                    ? cfg_.top_coordinate_z
                    : cfg_.coordinate_scale * (float)z + cfg_.coordinate_offset.z;
                dp.base_u = p.base_u;
                dp.sigma = sigma_local;

                // Debug safety check: ensure global->local mapping is exact.
                const uint dom_x = domain % Dx;
                const uint dom_y = (domain / Dx) % Dy;
                const uint dom_z = domain / (Dx * Dy);
                const uint rx = (uint)(lx - (ulong)Hx) + dom_x * nx_inner;
                const uint ry = (uint)(ly - (ulong)Hy) + dom_y * ny_inner;
                const uint rz = (uint)(lz - (ulong)Hz) + dom_z * nz_inner;
                if (rx != x || ry != y || rz != z) {
                    map_mismatch_count++;
                    if (map_mismatch_count <= 3ull) {
                        println("| VK inlet warn   | map mismatch: g=(" + to_string((ulong)x) + "," + to_string((ulong)y) + "," + to_string((ulong)z) +
                                "), r=(" + to_string((ulong)rx) + "," + to_string((ulong)ry) + "," + to_string((ulong)rz) + ")         |");
                    }
                }
                const bool on_face =
                    (face.id == WEST  && rx == 0u) ||
                    (face.id == EAST  && rx + 1u == solver.grid().size(0u)) ||
                    (face.id == SOUTH && ry == 0u) ||
                    (face.id == NORTH && ry + 1u == solver.grid().size(1u)) ||
                    (face.id == TOP   && rz + 1u == solver.grid().size(2u));
                if (!on_face) {
                    face_mismatch_count++;
                    if (face_mismatch_count <= 3ull) {
                        println("| VK inlet warn   | face mismatch: face=" + string(face_name_(face.id)) +
                                ", g=(" + to_string((ulong)x) + "," + to_string((ulong)y) + "," + to_string((ulong)z) + ")             |");
                    }
                }
                const uchar lf = solver.local(local_domain).fields().flags[dp.local_n];
                if (((lf & CELL_EQUILIBRIUM) == 0u) || ((lf & CELL_SOLID) != 0u)) {
                    local_flag_mismatch_count++;
                    if (local_flag_mismatch_count <= 3ull) {
                        println("| VK inlet warn   | local flag mismatch: flag=" + to_string((ulong)lf) +
                                ", face=" + string(face_name_(face.id)) + "                          |");
                    }
                }
                domain_points[domain].push_back(dp);

                sigma_sum_global += (double)sigma_local;
                sigma_count_global++;
                sigma_min_global = fminf(sigma_min_global, sigma_local);
                sigma_max_global = fmaxf(sigma_max_global, sigma_local);
                if (face.id >= 0 && face.id < VK_FACE_COUNT) {
                    const size_t fid = (size_t)face.id;
                    sigma_sum_face[fid] += (double)sigma_local;
                    sigma_count_face[fid]++;
                    sigma_min_face[fid] = fminf(sigma_min_face[fid], sigma_local);
                    sigma_max_face[fid] = fmaxf(sigma_max_face[fid], sigma_local);
                }
            }
        }

        domains_.clear();
        domains_.resize(D);
        uint active_domains = 0u;
        for (uint d = 0u; d < D; ++d) {
            if (domain_points[d].empty()) continue;
            const uint local_domain = (solver.distributed() ? 0u : d);

            const gpu::Device& device = solver.local(local_domain).device();
            auto rt = std::make_unique<DomainRuntime>();
            rt->active = true;
            rt->point_count = (ulong)domain_points[d].size();
            rt->mode_count = mode_count;
            rt->mode_stride = mode_stride;

            // Pinned mirrors, written once before the first launch.
            rt->point_cell = gpu::MirroredBuffer<ulong>(device, rt->point_count, 1u, gpu::HostMemory::pinned, 0ull);
            rt->point_face = gpu::MirroredBuffer<uchar>(device, rt->point_count, 1u, gpu::HostMemory::pinned, (uchar)0u);
            rt->point_data = gpu::MirroredBuffer<float>(device, rt->point_count, 7u, gpu::HostMemory::pinned, 0.0f);
            rt->mode_data = gpu::MirroredBuffer<float>(device, rt->mode_stride, 10u, gpu::HostMemory::pinned, 0.0f);

            for (ulong i = 0ull; i < rt->point_count; ++i) {
                const DomainPoint& dp = domain_points[d][(size_t)i];
                rt->point_cell[i] = dp.local_n;
                rt->point_face[i] = dp.face_id;
                rt->point_data[i] = dp.px;
                rt->point_data[rt->point_count + i] = dp.py;
                rt->point_data[2ull * rt->point_count + i] = dp.pz;
                rt->point_data[3ull * rt->point_count + i] = dp.base_u.x;
                rt->point_data[4ull * rt->point_count + i] = dp.base_u.y;
                rt->point_data[5ull * rt->point_count + i] = dp.base_u.z;
                rt->point_data[6ull * rt->point_count + i] = dp.sigma;
            }
            rt->point_cell.upload();
            rt->point_face.upload();
            rt->point_data.upload();
            rt->point_cell.release_host();
            rt->point_face.release_host();
            rt->point_data.release_host();

            for (ulong m = 0ull; m < rt->mode_stride; ++m) {
                rt->mode_data[m] = mode_data[(size_t)m];
                rt->mode_data[rt->mode_stride + m] = mode_data[(size_t)(mode_stride + m)];
                rt->mode_data[2ull * rt->mode_stride + m] = mode_data[(size_t)(2ull * mode_stride + m)];
                rt->mode_data[3ull * rt->mode_stride + m] = mode_data[(size_t)(3ull * mode_stride + m)];
                rt->mode_data[4ull * rt->mode_stride + m] = mode_data[(size_t)(4ull * mode_stride + m)];
                rt->mode_data[5ull * rt->mode_stride + m] = mode_data[(size_t)(5ull * mode_stride + m)];
                rt->mode_data[6ull * rt->mode_stride + m] = mode_data[(size_t)(6ull * mode_stride + m)];
                rt->mode_data[7ull * rt->mode_stride + m] = mode_data[(size_t)(7ull * mode_stride + m)];
                rt->mode_data[8ull * rt->mode_stride + m] = mode_data[(size_t)(8ull * mode_stride + m)];
                rt->mode_data[9ull * rt->mode_stride + m] = mode_data[(size_t)(9ull * mode_stride + m)];
            }
            rt->mode_data.upload();
            rt->mode_data.release_host();

            rt->kernel_apply = gpu::Launch(device, "vk_inlet_apply", rt->point_count).push(
                (uint)0u, 0.0f, 0.0f, 0.0f,
                rt->point_count,
                rt->mode_count,
                rt->mode_stride,
                rt->point_cell, rt->point_face, rt->point_data, rt->mode_data,
                solver.local(local_domain).fields().velocity
            );

            domains_[d] = std::move(rt);
            active_domains++;
        }

        if (active_domains == 0u) return false;
        println("| VK inlet backend| GPU on-the-fly kernels on " + to_string((ulong)active_domains) + " domain(s)            |");
        if (map_mismatch_count > 0ull || face_mismatch_count > 0ull || local_flag_mismatch_count > 0ull) {
            println("| VK inlet warn   | mapping checks: map=" + to_string(map_mismatch_count) +
                    ", face=" + to_string(face_mismatch_count) +
                    ", local_flag=" + to_string(local_flag_mismatch_count) + "                        |");
        }
        if (sigma_count_global > 0ull) {
            const float sigma_mean = (float)(sigma_sum_global / (double)sigma_count_global);
            println("| VK inlet sigma  | global min/mean/max = " +
                    to_string(sigma_min_global, 6u) + " / " +
                    to_string(sigma_mean, 6u) + " / " +
                    to_string(sigma_max_global, 6u) + " (LBM)                    |");
            for (int fid = 0; fid < VK_FACE_COUNT; ++fid) {
                const ulong cnt = sigma_count_face[(size_t)fid];
                if (cnt == 0ull) continue;
                const float mean_f = (float)(sigma_sum_face[(size_t)fid] / (double)cnt);
                println("| VK sigma face   | " + string(face_name_(fid)) + ": n=" + to_string(cnt) +
                        ", min/mean/max=" + to_string(sigma_min_face[(size_t)fid], 6u) + "/" +
                        to_string(mean_f, 6u) + "/" +
                        to_string(sigma_max_face[(size_t)fid], 6u) + "          |");
            }
        }
        return true;
    }

    void compute_time_params_(const double t, uint& use_interp, float& t0, float& t1, float& alpha) const {
        const ulong stride = cfg_.update_stride > 1 ? (ulong)cfg_.update_stride : 1ull;
        if (stride <= 1ull) {
            use_interp = 0u;
            t0 = (float)t;
            t1 = t0;
            alpha = 0.0f;
            return;
        }
        if (cfg_.stride_interpolation) {
            const double stride_d = static_cast<double>(stride);
            const double anchor = floor(t / stride_d) * stride_d;
            use_interp = 1u;
            t0 = (float)anchor;
            t1 = (float)(anchor + stride_d);
            alpha = (float)((t - anchor) / stride_d);
        } else {
            const double stride_d = static_cast<double>(stride);
            const double hold_t = floor(t / stride_d) * stride_d;
            use_interp = 0u;
            t0 = (float)hold_t;
            t1 = t0;
            alpha = 0.0f;
        }
    }

private:
    VkInletRuntimeConfig cfg_;
    std::array<VkFaceData, VK_FACE_COUNT> faces_;
    std::array<std::vector<VkMode>, VK_FACE_COUNT> face_modes_;
    std::vector<std::unique_ptr<DomainRuntime>> domains_;
    bool active_ = false;
    double last_applied_time_ = -std::numeric_limits<double>::infinity();
};

VonKarmanInletUpdater::VonKarmanInletUpdater(const VkInletRuntimeConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

VonKarmanInletUpdater::~VonKarmanInletUpdater() = default;

bool VonKarmanInletUpdater::initialize(culbm::solver::Solver& solver) {
    return impl_->initialize(solver);
}

bool VonKarmanInletUpdater::active() const {
    return impl_->active();
}

bool VonKarmanInletUpdater::update(culbm::solver::Solver& solver, const ulong t) {
    return impl_->update(solver, t);
}

bool VonKarmanInletUpdater::update_at_lattice_time(culbm::solver::Solver& solver, const double lattice_time) {
    return impl_->update_at_lattice_time(solver, lattice_time);
}

} // namespace culbm::bc
