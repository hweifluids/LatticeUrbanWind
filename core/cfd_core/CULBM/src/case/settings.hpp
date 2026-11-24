#pragma once

// Settings of one LUW deck (*.luw, *.luwdg, *.luwpf) after reading,
// defaulting and validation: case identity, domain and grid controls,
// boundary and inflow controls, run length and outputs, and the requested
// lattice and solver features. Each field keeps the name of the deck
// handling it came from; read_case_settings() fills the structure.

#include "../core/core.hpp"
#include "../bc/inflow/vk_inlet_settings.hpp"
#include "../bc/thermal/temperature_limits.hpp"
#include "../particle_advection_config.hpp"
#include "../solver/lattice_config.hpp"
#include "../surface_ibm_config.hpp"

#include <memory>

namespace culbm::cases {

struct CaseSettings {
    // Deck file and mode (*.luwdg: dataset batch, *.luwpf: profile forcing).
    bool dataset_generation = false;
    bool profile_generation = false;
    std::string conf_used_path = "Integrated defaults (*.luw not found)";
    std::string parent; // project directory that contains the deck

    // Case identity.
    std::string caseName = "example";
    std::string datetime = "20990101120000";
    bool datetime_from_config = false;

    // Domain and grid. base_height (z_si_offset) and the z extent of si_size
    // are reduced by the CUDA ground compaction.
    float z_si_offset = 50.0f;
    float z_limit_override = 0.0f;
    float3 si_size = float3(0);
    uint memory = 20000u;
    float cell_m = 20.0f;
    bool gpu_memory_mesh_control_requested = false;
    uint Dx = 1u, Dy = 1u, Dz = 1u;
    std::vector<float> cuda_vertical_nesting_agl_si;
    static constexpr float profile_dz_si = 0.1f;

    // Geographic frame (probes, Coriolis).
    float cut_lon_min_deg = 0.0f; // longitude/latitude given in *.luw, degrees
    float cut_lon_max_deg = 0.0f;
    float cut_lat_min_deg = 0.0f;
    float cut_lat_max_deg = 0.0f;
    bool has_cut_lon_manual = false;
    bool has_cut_lat_manual = false;
    bool has_rotate_deg = false;
    double rotate_deg_from_config = 0.0;
    std::string utm_crs_from_config;

    // Boundaries and inflow.
    std::string downstream_bc = "+y";
    std::string downstream_bc_yaw = "INTERNAL ERROR";
    bool downstream_open_face = false;
    bool use_high_order = false;
    bool flux_correction = false;
    bool enable_coriolis = false;
    bool enable_buffer_nudging = true;
    float buffer_thickness_m = 160.0f;
    float buffer_tau_s = 300.0f;
    int buffer_nudge_vertical = 0;
    bool enable_top_sponge = true;
    float sponge_thickness_m = 200.0f;
    float sponge_tau_s = 120.0f;
    int sponge_ref_mode = 0;
    bc::VkInletSettings vk_inlet_settings;
    std::vector<float> inflow_list; // dataset batch: inflow speeds
    std::vector<float> angle_list;  // dataset and profile batches: wind directions

    // Run length and outputs.
    uint research_output_steps = 0u;
    uint unsteady_output_interval = 0u;
    ulong culbm_unsteady_start_step = 0u;
    uint purge_avg_steps = 0u;
    uint purge_avg_stride = 1u;
    ulong run_nstep_override = 0ull;
    std::string validation_status = "fail";
    bool output_tke_in_avg_vtk = true;
    bool output_ti_in_avg_vtk = true;
    bool output_tls_in_avg_vtk = true;
    bool culbm_output_global_vtk = false;
    bool culbm_output_global_netcdf = false;
    bool culbm_output_global_vtkhdf = false;
    bool culbm_output_agl_vtk = false;
    bool culbm_output_agl_netcdf = false;
    // Derived when the geometry is loaded.
    bool culbm_output_global_requested = false;
    bool culbm_output_agl_requested = false;
    bool culbm_output_physical_requested = false;
    bool probes_defined = false;
    std::string probes_raw_text;
    bool probes_output_defined = false;
    uint probes_output_steps = 0u;

    // Thermal coupling. buoyancy_enabled holds the request until the
    // temperature boundaries are known, then the active coupling.
    bool buoyancy_enabled = true; // default-on unless explicitly false/False/0
    bool buoyancy_explicit = false;
    bool boussinesq_reference_auto = true;
    float boussinesq_reference_setting_kelvin = bc::k_temperature_ref_kelvin;

    // Lattice selections and solver models.
    Temperature_Lattice temperature_lattice_setting = Temperature_Lattice::Q7;
    Momentum_Lattice momentum_lattice_setting = Momentum_Lattice::Q19;
    DDF_Storage ddf_storage_setting = DDF_Storage::FP16C;
    bool temperature_lattice_applied = false;
    bool culbm_sgs_enabled = false; // set from the library default by the reader
    double culbm_sgs_cs = 0.173265956;
    bool culbm_sgs_cs_explicit = false;
    bool culbm_sgs_cs_squared_explicit = false;
    bool culbm_sgs_scalar_coupling = true;
    double culbm_sgs_turbulent_prandtl = 1.0;
    double culbm_base_kinematic_viscosity_m2_s = 1.48E-5;
    double culbm_base_thermal_diffusivity_m2_s = 2.10E-5;
    uint culbm_sgs_diagnostics_interval = 0u;
    Wall_Model_Configuration wall_model_settings;
    culbm::particle_advection::Config particle_advection_settings;
    culbm::surface_ibm::Config surface_ibm_settings;

    // gpu_memory fitting reads the STL early; the geometry loader reuses it.
    std::filesystem::path preloaded_geometry_path;
    std::unique_ptr<Mesh> preloaded_geometry_mesh;
};

// Reads the deck named by the first case argument (or asked for on the
// console), applies its defaults and validation, and publishes the lattice
// selections and the SurfaceIBM configuration. Deck errors end the process.
void read_case_settings(CaseSettings& settings);

// Profile decks: datetime fallback notice, Coriolis needs the lon/lat window.
void finish_profile_settings(CaseSettings& deck);

// Validation status of the preprocessing; asks before continuing when it did
// not pass (normal decks only).
void confirm_validation(const CaseSettings& deck);

} // namespace culbm::cases
