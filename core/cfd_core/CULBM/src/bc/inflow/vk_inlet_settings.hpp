#pragma once

// Deck settings of the von Karman synthetic turbulence inlet and the runtime
// configuration derived from them in lattice units.

#include "../../core/core.hpp"

namespace culbm::bc {

constexpr int k_vk_nmodes_default = 256;
constexpr int k_vk_nmodes_max = 512;

enum class VkInletUcMode {
    NORM_MEAN = 0,
    NORMAL_COMPONENT = 1
};

enum class VkInletFaceMode {
    AUTO_SIDES = 0,
    TARGET_INFLOW = 1,
    EXCLUDE_DOWNSTREAM = 2,
    EXCLUDE_DOWNSTREAM_SIDES = 3,
    ALL_SIDES = 4,
    ALL_SELECTED = 5
};

const char* vk_face_mode_name(VkInletFaceMode mode);
const char* vk_uc_mode_name(VkInletUcMode mode);
VkInletFaceMode resolve_vk_face_mode(VkInletFaceMode mode, bool inflow_only);

struct VkInletSettings {
    bool enable = true;
    // turbulence intensity (fraction); sigma_local uses uc_mode:
    // NORM_MEAN -> ti*|U_base|, NORMAL_COMPONENT -> ti*|U_base·n_face|
    float ti = 0.05f;
    float sigma_si = 0.0f;
    float L_si = 100.0f;
    int nmodes = k_vk_nmodes_default;
    ulong seed = 100ull;
    int update_stride = 1;
    VkInletUcMode uc_mode = VkInletUcMode::NORM_MEAN;
    bool same_realization_all_faces = true;
    bool stride_interpolation = false; // optional double-buffer interpolation for stride > 1
    bool inflow_only = false; // true: side faces except outlet, false: all four side faces; top is excluded in AUTO_SIDES
    VkInletFaceMode face_mode = VkInletFaceMode::AUTO_SIDES;
    float3 anisotropy_scale = float3(1.0f, 1.0f, 1.0f); // per-component perturbation gain [ax, ay, az]
};

struct VkInletRuntimeConfig {
    bool enable = true;
    float ti = 0.05f;
    float sigma_lbm = 0.0f;
    float L_lbm = 100.0f;
    int nmodes = k_vk_nmodes_default;
    ulong seed = 100ull;
    int update_stride = 1;
    VkInletUcMode uc_mode = VkInletUcMode::NORM_MEAN;
    bool same_realization_all_faces = true;
    bool stride_interpolation = false;
    bool inflow_only = false;
    VkInletFaceMode face_mode = VkInletFaceMode::AUTO_SIDES;
    float3 anisotropy_scale = float3(1.0f, 1.0f, 1.0f);
    int downstream_face_id = -1; // {0:west,1:east,2:south,3:north}, -1 means unknown/none
    // Nested slabs express their point coordinates and time on the original
    // coarse lattice. This retains one physically continuous VK realization
    // across 2:1 interfaces instead of generating a new field per slab.
    float coordinate_scale = 1.0f;
    float3 coordinate_offset = float3(0.0f);
    bool allow_top_face = true;
    bool has_top_coordinate_z = false;
    float top_coordinate_z = 0.0f;
};

// Runtime configuration for a case whose outlet is `downstream_bc_runtime`
// (the global Units scale converts sigma and L); invalid values disable the
// inlet with a console warning.
VkInletRuntimeConfig make_vk_runtime_config(const VkInletSettings& vk_inlet_settings,
                                            const std::string& downstream_bc_runtime);

// Conservative device memory of the inlet runtime on one subdomain.
uint estimate_vk_inlet_extra_mb_per_device(const VkInletSettings& vk_inlet_settings,
                                           uint Nx, uint Ny, uint Nz);

} // namespace culbm::bc
