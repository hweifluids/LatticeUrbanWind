#include "case_dump.hpp"
#include "config_dump.hpp"
#include "context.hpp"
#include "../bc/inflow/boundaries.hpp"
#include "../bc/inflow/vk_inlet_settings.hpp"
#include "../bc/outflow/damping_layers.hpp"
#include "../run/schedule.hpp"

#include <array>
#include <optional>

namespace culbm::cases {

using namespace culbm::bc;
using namespace culbm::run;

void dump_settings_stage(const CaseContext& ctx) {
    if (!ConfigDump::enabled()) return;
    const CaseSettings& deck = ctx.deck;
    const ProbePlan& probes = ctx.probes;
    ConfigDump& dump = ConfigDump::get();
    const auto opt_text = [](const std::optional<double>& v) {
        return v ? ConfigDump::text(*v) : std::string("none");
    };
    const auto opt3_text = [](const std::optional<std::array<double, 3u>>& v) {
        if (!v) return std::string("none");
        return ConfigDump::text((*v)[0]) + ", " +
            ConfigDump::text((*v)[1]) + ", " +
            ConfigDump::text((*v)[2]);
    };
    dump.section("deck");
    dump.put("mode", deck.dataset_generation ? "dataset" : deck.profile_generation ? "profile" : "normal");
    dump.put("conf_used_path", deck.conf_used_path);
    dump.put("project_directory", deck.parent);
    dump.put("casename", deck.caseName);
    dump.put("datetime", deck.datetime);
    dump.put("datetime_from_deck", deck.datetime_from_config);
    dump.put("downstream_bc", deck.downstream_bc);
    dump.put("downstream_bc_yaw", deck.downstream_bc_yaw);
    dump.put("downstream_open_face", deck.downstream_open_face);
    dump.put("base_height_m", deck.z_si_offset);
    dump.put("z_limit_m", deck.z_limit_override);
    dump.put("profile_dz_m", deck.profile_dz_si);
    dump.put("memory_mb", deck.memory);
    dump.put("cell_size_m", deck.cell_m);
    dump.put("gpu_memory_fit", deck.gpu_memory_mesh_control_requested);
    dump.put("fit_geometry_path", deck.preloaded_geometry_path.string());
    dump.put("fit_geometry_loaded", deck.preloaded_geometry_mesh != nullptr);
    dump.put("si_size_m", deck.si_size);
    dump.put("subdomains", uint3(deck.Dx, deck.Dy, deck.Dz));
    dump.put("high_order", deck.use_high_order);
    dump.put("flux_correction", deck.flux_correction);
    dump.put("research_output_steps", deck.research_output_steps);
    dump.put("unsteady_output_interval", deck.unsteady_output_interval);
    dump.put("unsteady_start_step", deck.culbm_unsteady_start_step);
    dump.put("purge_avg_steps", deck.purge_avg_steps);
    dump.put("purge_avg_stride", deck.purge_avg_stride);
    dump.put("run_nstep", deck.run_nstep_override);
    dump.put("validation", deck.validation_status);
    dump.put("avg_output_tke", deck.output_tke_in_avg_vtk);
    dump.put("avg_output_ti", deck.output_ti_in_avg_vtk);
    dump.put("avg_output_tls", deck.output_tls_in_avg_vtk);
    dump.put("output_global_vtk", deck.culbm_output_global_vtk);
    dump.put("output_global_netcdf", deck.culbm_output_global_netcdf);
    dump.put("output_global_vtkhdf", deck.culbm_output_global_vtkhdf);
    dump.put("output_agl_vtk", deck.culbm_output_agl_vtk);
    dump.put("output_agl_netcdf", deck.culbm_output_agl_netcdf);
    dump.put("coriolis", deck.enable_coriolis);
    dump.put("buffer_nudging", deck.enable_buffer_nudging);
    dump.put("buffer_thickness_m", deck.buffer_thickness_m);
    dump.put("buffer_tau_s", deck.buffer_tau_s);
    dump.put("buffer_nudge_vertical", deck.buffer_nudge_vertical);
    dump.put("top_sponge", deck.enable_top_sponge);
    dump.put("sponge_thickness_m", deck.sponge_thickness_m);
    dump.put("sponge_tau_s", deck.sponge_tau_s);
    dump.put("sponge_ref_mode", deck.sponge_ref_mode);
    dump.put("nesting_heights_m", deck.cuda_vertical_nesting_agl_si);
    dump.put("inflow_list", deck.inflow_list);
    dump.put("angle_list", deck.angle_list);
    dump.put("cut_lon_min_deg", deck.cut_lon_min_deg);
    dump.put("cut_lon_max_deg", deck.cut_lon_max_deg);
    dump.put("cut_lat_min_deg", deck.cut_lat_min_deg);
    dump.put("cut_lat_max_deg", deck.cut_lat_max_deg);
    dump.put("cut_lon_from_deck", deck.has_cut_lon_manual);
    dump.put("cut_lat_from_deck", deck.has_cut_lat_manual);
    dump.put("utm_crs", deck.utm_crs_from_config);
    dump.put("rotate_deg_from_deck", deck.has_rotate_deg);
    dump.put("rotate_deg", deck.rotate_deg_from_config);
    dump.put("probes_defined", deck.probes_defined);
    dump.put("probes_text", deck.probes_raw_text);
    dump.put("probes_output_defined", deck.probes_output_defined);
    dump.put("probes_output_steps", deck.probes_output_steps);
    dump.put("buoyancy_requested", deck.buoyancy_enabled);
    dump.put("buoyancy_explicit", deck.buoyancy_explicit);
    dump.put("boussinesq_reference_auto", deck.boussinesq_reference_auto);
    dump.put("boussinesq_reference_k", deck.boussinesq_reference_setting_kelvin);
    dump.section("vk_inlet");
    dump.put("enable", deck.vk_inlet_settings.enable);
    dump.put("ti", deck.vk_inlet_settings.ti);
    dump.put("sigma_m_s", deck.vk_inlet_settings.sigma_si);
    dump.put("length_m", deck.vk_inlet_settings.L_si);
    dump.put("modes", deck.vk_inlet_settings.nmodes);
    dump.put("seed", deck.vk_inlet_settings.seed);
    dump.put("update_stride", deck.vk_inlet_settings.update_stride);
    dump.put("uc_mode", vk_uc_mode_name(deck.vk_inlet_settings.uc_mode));
    dump.put("same_realization_all_faces", deck.vk_inlet_settings.same_realization_all_faces);
    dump.put("stride_interpolation", deck.vk_inlet_settings.stride_interpolation);
    dump.put("inflow_only", deck.vk_inlet_settings.inflow_only);
    dump.put("face_mode", vk_face_mode_name(deck.vk_inlet_settings.face_mode));
    dump.put("anisotropy", deck.vk_inlet_settings.anisotropy_scale);
    dump.section("lattice");
    dump.put("velocity_set_requested", (uint)deck.momentum_lattice_setting);
    dump.put("ddf_storage_requested", (uint)deck.ddf_storage_setting);
    dump.put("temperature_lattice_requested", (uint)deck.temperature_lattice_setting);
    dump.put("velocity_set", get_momentum_lattice_name());
    dump.put("ddf_storage", get_ddf_storage_name());
    dump.put("ddf_bytes", get_ddf_bytes());
    dump.put("temperature_lattice", get_temperature_lattice_name());
    dump.put("temperature_set", get_temperature_set());
    dump.section("subgrid");
    dump.put("enable", deck.culbm_sgs_enabled);
    dump.put("cs", deck.culbm_sgs_cs);
    dump.put("cs_explicit", deck.culbm_sgs_cs_explicit);
    dump.put("cs_squared_explicit", deck.culbm_sgs_cs_squared_explicit);
    dump.put("scalar_coupling", deck.culbm_sgs_scalar_coupling);
    dump.put("turbulent_prandtl", deck.culbm_sgs_turbulent_prandtl);
    dump.put("base_nu_m2_s", deck.culbm_base_kinematic_viscosity_m2_s);
    dump.put("base_alpha_m2_s", deck.culbm_base_thermal_diffusivity_m2_s);
    dump.put("diagnostics_interval", deck.culbm_sgs_diagnostics_interval);
    {
        const Subgrid_Configuration applied = get_subgrid_configuration();
        dump.put("applied.enabled", applied.enabled);
        dump.put("applied.smagorinsky_constant", applied.smagorinsky_constant);
        dump.put("applied.scalar_coupling", applied.scalar_coupling);
        dump.put("applied.turbulent_prandtl", applied.turbulent_prandtl);
        dump.put("applied.fixed_eddy_viscosity", applied.fixed_eddy_viscosity);
    }
    const auto dump_wall_model = [&](const std::string& prefix, const Wall_Model_Configuration& w) {
        dump.put(prefix + "enabled", w.enabled);
        dump.put(prefix + "kappa", w.kappa);
        dump.put(prefix + "smooth_wall_b", w.smooth_wall_B);
        dump.put(prefix + "z0_ground_m", w.z0_ground_m);
        dump.put(prefix + "z0_building_m", w.z0_building_m);
        dump.put(prefix + "molecular_nu_m2_s", w.molecular_nu_m2_s);
        dump.put(prefix + "sgs_layers", w.sgs_layers);
        dump.put(prefix + "surface_ibm", w.surface_ibm);
    };
    dump.section("wall_model");
    dump_wall_model("", deck.wall_model_settings);
    dump_wall_model("applied.", get_wall_model_configuration());
    const auto dump_ibm = [&](const std::string& prefix, const culbm::surface_ibm::Config& c) {
        dump.put(prefix + "enabled", c.enabled);
        dump.put(prefix + "delta_kernel", culbm::surface_ibm::delta_kernel_name(c.delta_kernel));
        dump.put(prefix + "geometry_policy", culbm::surface_ibm::geometry_policy_name(c.geometry_policy));
        dump.put(prefix + "marker_spacing_lu", c.marker_spacing_lu);
        dump.put(prefix + "forcing_iterations", c.forcing_iterations);
        dump.put(prefix + "relaxation", c.relaxation);
        dump.put(prefix + "residual_tolerance", c.residual_tolerance);
        dump.put(prefix + "retraction_lu", c.retraction_lu);
        dump.put(prefix + "support_thickness_lu", c.support_thickness_lu);
        dump.put(prefix + "max_markers", c.max_markers);
        dump.put(prefix + "strict_geometry", c.strict_geometry);
        dump.put(prefix + "diagnostics_interval", c.diagnostics_interval);
        dump.put(prefix + "support_z_min", c.support_z_min);
        dump.put(prefix + "support_z_max", c.support_z_max);
    };
    dump.section("surface_ibm");
    dump_ibm("", deck.surface_ibm_settings);
    dump_ibm("applied.", culbm::surface_ibm::get_config());
    dump.section("particle_advection");
    dump.put("enabled", deck.particle_advection_settings.enabled);
    dump.put("mode", culbm::particle_advection::mode_name(deck.particle_advection_settings.mode));
    dump.put("integrator", culbm::particle_advection::integrator_name(deck.particle_advection_settings.integrator));
    dump.put("particle_file", deck.particle_advection_settings.particle_file.string());
    dump.put("release_time_s", opt_text(deck.particle_advection_settings.release_time_s));
    dump.put("lifetime_s", opt_text(deck.particle_advection_settings.lifetime_s));
    dump.put("output_interval_s", opt_text(deck.particle_advection_settings.output_interval_s));
    dump.put("max_particles", deck.particle_advection_settings.max_particles);
    dump.put("release_interval_s", opt_text(deck.particle_advection_settings.release_interval_s));
    dump.put("automatic_spacing_m", opt3_text(deck.particle_advection_settings.automatic_spacing_m));
    dump.put("automatic_min_m", opt3_text(deck.particle_advection_settings.automatic_min_m));
    dump.put("automatic_max_m", opt3_text(deck.particle_advection_settings.automatic_max_m));
    dump.section("probes");
    dump.put("requests", probes.probe_requests.size());
    for (size_t i = 0u; i < probes.probe_requests.size(); ++i) {
        const ProbeRequest& r = probes.probe_requests[i];
        const std::string k = "request." + std::to_string(i) + ".";
        dump.put(k + "token", r.raw_token);
        dump.put(k + "lon_deg", r.lon_deg);
        dump.put(k + "lat_deg", r.lat_deg);
        dump.put(k + "center", r.uses_center);
        dump.put(k + "offset_mode", (int)r.offset.mode);
        dump.put(k + "offset_cells", std::to_string(r.offset.north_cells) + " N, " + std::to_string(r.offset.east_cells) + " E");
        dump.put(k + "offset_north_m", r.offset.north_m);
        dump.put(k + "offset_east_m", r.offset.east_m);
        dump.put(k + "offset_label", r.offset.label);
    }
    dump.put("mapping.valid", probes.probe_geo_mapping.valid);
    dump.put("mapping.utm_zone", probes.probe_geo_mapping.utm_zone);
    dump.put("mapping.utm_north", probes.probe_geo_mapping.utm_north);
    dump.put("mapping.rotate_deg", probes.probe_geo_mapping.rotate_deg);
    dump.put("mapping.pivot_x", probes.probe_geo_mapping.pivot_x);
    dump.put("mapping.pivot_y", probes.probe_geo_mapping.pivot_y);
    dump.put("mapping.x_min_rot", probes.probe_geo_mapping.x_min_rot);
    dump.put("mapping.y_min_rot", probes.probe_geo_mapping.y_min_rot);
    dump.put("mapping.center_lon_deg", probes.probe_geo_mapping.center_lon_deg);
    dump.put("mapping.center_lat_deg", probes.probe_geo_mapping.center_lat_deg);
    dump.put("mapping.east", ConfigDump::text(probes.probe_geo_mapping.east_dx) + ", " + ConfigDump::text(probes.probe_geo_mapping.east_dy));
    dump.put("mapping.north", ConfigDump::text(probes.probe_geo_mapping.north_dx) + ", " + ConfigDump::text(probes.probe_geo_mapping.north_dy));
}

void dump_domain_stage(const CaseContext& ctx) {
    if (!ConfigDump::enabled()) return;
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const CasePhysics& physics = ctx.physics;
    ConfigDump& dump = ConfigDump::get();
    dump.section("ground_compaction");
    dump.put("override", domain.cuda_ground_compaction_override);
    dump.put("requested", domain.cuda_ground_compaction_requested);
    dump.put("base_height_m", domain.cuda_ground_base_height_si);
    dump.put("probe_path", domain.cuda_ground_probe_path);
    dump.put("probe_loaded", domain.cuda_ground_probe_mesh != nullptr);
    dump.put("active", domain.cuda_ground_compaction_active);
    dump.put("reason", domain.cuda_ground_compaction_reason);
    dump.put("trim_cells", domain.cuda_ground_trim_cells);
    dump.put("geometry_shift_cells", domain.cuda_ground_geometry_shift_cells);
    dump.put("trim_m", domain.cuda_ground_trim_si);
    dump.section("grid");
    dump.put("nx_cells", domain.Nx_cells);
    dump.put("ny_cells", domain.Ny_cells);
    dump.put("sponge_cells", domain.sponge_cells_cfg);
    dump.put("nz_cells_uncompacted", domain.Nz_cells_uncompacted);
    dump.put("nz_cells_core", domain.Nz_cells_core);
    dump.put("top_sponge_grid_extend", domain.top_sponge_grid_extend);
    dump.put("nz_cells", domain.Nz_cells);
    dump.put("side_reference_z_cap", domain.top_sponge_side_ref_z_cap);
    dump.put("lattice", domain.lbm_N);
    dump.put("si_size_m", deck.si_size);
    dump.put("base_height_m", deck.z_si_offset);
    dump.put("lbm_reference_u", domain.lbm_ref_u);
    dump.put("si_reference_u_initial", physics.si_ref_u);
    dump.put("si_density", domain.si_rho);
    dump.section("gpu_estimate");
    dump.put("grid", domain.gpu_memory_estimate.grid);
    dump.put("core_per_device_mb", domain.gpu_memory_estimate.core_per_device_mb);
    dump.put("extra_per_device_mb", domain.gpu_memory_estimate.extra_per_device_mb);
    dump.put("total_per_device_mb", domain.gpu_memory_estimate.total_per_device_mb);
    dump.put("total_project_mb", domain.gpu_memory_estimate.total_project_mb);
    dump.put("top_sponge_grid_extend", domain.gpu_memory_estimate.top_sponge_grid_extend);
    dump.put("sponge_cells", domain.gpu_memory_estimate.sponge_cells);
    dump.put("nested_additional_mb", domain.nested_additional_gpu_memory_per_device_mb);
    dump.put("nested_raster_core_mb", domain.nested_full_height_raster_core_mb);
    dump.section("nesting");
    dump.put("enabled", domain.cuda_vertical_nesting_plan.enabled);
    dump.put("valid", domain.cuda_vertical_nesting_plan.valid);
    dump.put("error", domain.cuda_vertical_nesting_plan.error);
    dump.put("coarse_dx_m", domain.cuda_vertical_nesting_plan.coarse_dx_si);
    dump.put("ground_top_m", domain.cuda_vertical_nesting_plan.ground_top_si);
    dump.put("domain_height_m", domain.cuda_vertical_nesting_plan.domain_height_agl_si);
    dump.put("interface_heights_m", domain.cuda_vertical_nesting_plan.interface_heights_agl_si);
    dump.put("domains", uint3(domain.cuda_vertical_nesting_plan.domains_x, domain.cuda_vertical_nesting_plan.domains_y, domain.cuda_vertical_nesting_plan.domains_z));
    for (size_t i = 0u; i < domain.cuda_vertical_nesting_plan.levels.size(); ++i) {
        const culbm::VerticalNestingLevelPlan& level = domain.cuda_vertical_nesting_plan.levels[i];
        const std::string k = "level." + std::to_string(i) + ".";
        dump.put(k + "label", level.label);
        dump.put(k + "ratio", level.ratio_to_coarse);
        dump.put(k + "size", uint3(level.nx, level.ny, level.allocated_nz));
        dump.put(k + "physical_nz", level.physical_nz);
        dump.put(k + "ghost_rows", std::to_string(level.lower_ghost_rows) + ", " + std::to_string(level.upper_ghost_rows));
        dump.put(k + "dx_m", level.dx_si);
        dump.put(k + "z_agl_m", ConfigDump::text(level.z_begin_agl_si) + ", " + ConfigDump::text(level.z_end_agl_si));
        dump.put(k + "z_m", ConfigDump::text(level.z_begin_si) + ", " + ConfigDump::text(level.z_end_si));
    }
}

void dump_inflow_stage(const CaseContext& ctx) {
    if (!ConfigDump::enabled()) return;
    const InflowData& inflow = ctx.inflow;
    const ThermalReference& thermal = ctx.thermal;
    const CasePhysics& physics = ctx.physics;
    const CaseSettings& deck = ctx.deck;
    ConfigDump& dump = ConfigDump::get();
    const auto sample_checksum = [](const std::vector<SamplePoint>& v) {
        Checksum sum;
        for (const SamplePoint& p : v) sum.value(p.p).value(p.u).value(p.T).value(p.patch);
        return sum;
    };
    dump.section("inflow");
    dump.put_checksum("samples_si", inflow.samples_si.size(), sample_checksum(inflow.samples_si));
    dump.put("csv_has_temperature", inflow.csv_has_temperature);
    dump.put("csv_rows_with_temperature", inflow.csv_rows_with_temperature);
    dump.put("csv_temperature_min_k", inflow.csv_temp_min_si);
    dump.put("csv_temperature_max_k", inflow.csv_temp_max_si);
    dump.put("csv_has_patch", inflow.csv_has_patch);
    dump.put("csv_rows_with_patch", inflow.csv_rows_with_patch);
    dump.put("csv_valid", inflow.csv_samples_valid);
    dump.put("profile_z_m", inflow.profile_z_samples_si);
    dump.put("profile_u_m_s", inflow.profile_u_samples_si);
    dump.put("si_reference_u", physics.si_ref_u);
    dump.section("thermal");
    dump.put("temperature_reference_k", thermal.temperature_ref_kelvin);
    dump.put("temperature_scale_k", thermal.temperature_scale_kelvin);
    dump.put("temperature_reference_adaptive", thermal.temperature_ref_adaptive);
    dump.put("temperature_scale_adaptive", thermal.temperature_scale_adaptive);
    dump.put("boussinesq_range_checked", thermal.boussinesq_temperature_range_checked);
    dump.put("boussinesq_max_relative_difference", thermal.boussinesq_max_relative_temperature_difference);
    dump.put("boundary_available", thermal.thermal_boundary_available);
    dump.put("coupling.requested", thermal.thermal_coupling.buoyancy_requested);
    dump.put("coupling.boundary_available", thermal.thermal_coupling.thermal_boundary_available);
    dump.put("coupling.active", thermal.thermal_coupling.buoyancy_active);
    dump.put("coupling.fallback", thermal.thermal_coupling.fallback_applied);
    dump.put("buoyancy", deck.buoyancy_enabled);
    dump.put("temperature_bc", inflow.use_temperature_bc);
    dump.put("temperature_bc_min_lbm", inflow.T_bc_min_lbm);
    dump.put("temperature_bc_max_lbm", inflow.T_bc_max_lbm);
    dump.section("units");
    dump.put_units("", units);
    dump.put("u_scale", physics.u_scale);
    dump.put("z_offset_lbm", physics.z_off);
    dump.put("beta_air_si", physics.si_beta_air);
    dump.put("gravity_si", physics.si_gravity);
    dump.section("transport");
    dump.put("common_scale", physics.base_transport.common_scale);
    dump.put("requested_nu_lbm", physics.base_transport.requested_nu_lbm);
    dump.put("requested_alpha_lbm", physics.base_transport.requested_alpha_lbm);
    dump.put("requested_momentum_tau", physics.base_transport.requested_momentum_tau);
    dump.put("requested_thermal_tau", physics.base_transport.requested_thermal_tau);
    dump.put("effective_nu_si", physics.base_transport.effective_nu_si);
    dump.put("effective_alpha_si", physics.base_transport.effective_alpha_si);
    dump.put("effective_nu_lbm", physics.base_transport.effective_nu_lbm);
    dump.put("effective_alpha_lbm", physics.base_transport.effective_alpha_lbm);
    dump.put("effective_momentum_tau", physics.base_transport.effective_momentum_tau);
    dump.put("effective_thermal_tau", physics.base_transport.effective_thermal_tau);
    dump.put("nu_lbm", physics.lbm_nu);
    dump.put("alpha_lbm", physics.lbm_alpha);
    dump.put("beta_lbm", physics.lbm_beta);
    dump.put("gravity_z_lbm", physics.lbm_buoyancy_gz);
    dump.section("forces");
    dump.put("coriolis_omega_lbm", float3(coriolis_Omegax_lbmu, coriolis_Omegay_lbmu, coriolis_Omegaz_lbmu));
    dump.put("coriolis_f_lbm", coriolis_f_lbmu);
    dump.section("outflow");
    dump.put("buffer_face", buffer_downstream_face_id);
    dump.put("buffer_cells", buffer_n_cells);
    dump.put("buffer_inv_tau_lbm", buffer_inv_tau_lbmu);
    dump.put("buffer_active", buffer_nudging_active);
    dump.put("sponge_cells", sponge_n_cells);
    dump.put("sponge_inv_tau_lbm", sponge_inv_tau_lbmu);
    dump.put("sponge_active", top_sponge_active);
    dump.section("samples");
    dump.put_checksum("samples_lbm", inflow.samples.size(), sample_checksum(inflow.samples));
}

void dump_geometry_stage(const CaseContext& ctx) {
    if (!ConfigDump::enabled()) return;
    const CaseSettings& deck = ctx.deck;
    const CaseGeometry& geometry = ctx.geometry;
    const CaseTerrain& terrain = ctx.terrain;
    const CaseDomain& domain = ctx.domain;
    ConfigDump& dump = ConfigDump::get();
    dump.section("geometry");
    dump.put("output_global", deck.culbm_output_global_requested);
    dump.put("output_agl", deck.culbm_output_agl_requested);
    dump.put("output_physical", deck.culbm_output_physical_requested);
    dump.put("stl_path", geometry.stl_path);
    dump.put("voxelization_transient_mb", geometry.voxelization_transient_mb);
    dump.put("uniform_core_mb", geometry.uniform_core_mb);
    dump.put("uniform_voxel_peak_mb", geometry.uniform_voxel_peak_mb);
    dump.put("nested_voxel_peak_mb", geometry.nested_voxel_peak_mb);
    dump.put("exact_mesh_peak_mb", geometry.exact_mesh_peak_mb);
    dump.put("gpu.core_per_device_mb", domain.gpu_memory_estimate.core_per_device_mb);
    dump.put("gpu.extra_per_device_mb", domain.gpu_memory_estimate.extra_per_device_mb);
    dump.put("gpu.total_per_device_mb", domain.gpu_memory_estimate.total_per_device_mb);
    dump.put("gpu.total_project_mb", domain.gpu_memory_estimate.total_project_mb);
    dump.put("stl_size_m", geometry.stl_size_si);
    dump.put("stl_min_m", geometry.stl_min_si);
    dump.put("stl_max_m", geometry.stl_max_si);
    dump.put("domain_min_m", geometry.domain_min_si);
    dump.put("vtk_origin_shift_m", vtk_origin_shift);
    dump.put("target_lbm_x", geometry.target_lbm_x);
    dump.put("scale", geometry.scale_geom);
    dump.put_mesh("mesh", geometry.mesh.get());
    dump.section("surface_ibm_markers");
    {
        Checksum sum;
        for (const culbm::surface_ibm::Marker& m : geometry.surface_ibm_markers) {
            sum.value(m.position_lu).value(m.wall_velocity_lu).value(m.normal).value(m.area_lu2)
                .value(m.body_id).value(m.source_triangle).value(m.global_id);
        }
        dump.put_checksum("markers", geometry.surface_ibm_markers.size(), sum);
    }
    dump.put("dropped_outer_support", geometry.surface_ibm_dropped_external_support);
    dump.put("active_area_lu2", geometry.surface_ibm_active_marker_area_lu2);
    dump.put("report.input_triangles", geometry.surface_ibm_geometry_report.input_triangles);
    dump.put("report.valid_triangles", geometry.surface_ibm_geometry_report.valid_triangles);
    dump.put("report.degenerate_triangles", geometry.surface_ibm_geometry_report.degenerate_triangles);
    dump.put("report.duplicate_triangles", geometry.surface_ibm_geometry_report.duplicate_triangles);
    dump.put("report.coincident_union_pairs", geometry.surface_ibm_geometry_report.coincident_union_pairs);
    dump.put("report.internal_triangles", geometry.surface_ibm_geometry_report.internal_triangles);
    dump.put("report.ambiguous_triangles", geometry.surface_ibm_geometry_report.ambiguous_triangles);
    dump.put("report.retained_triangles", geometry.surface_ibm_geometry_report.retained_triangles);
    dump.put("report.internal_patches", geometry.surface_ibm_geometry_report.internal_patches);
    dump.put("report.ambiguous_patches", geometry.surface_ibm_geometry_report.ambiguous_patches);
    dump.put("report.connected_components", geometry.surface_ibm_geometry_report.connected_components);
    dump.put("report.watertight_components", geometry.surface_ibm_geometry_report.watertight_components);
    dump.put("report.open_edges", geometry.surface_ibm_geometry_report.open_edges);
    dump.put("report.nonmanifold_edges", geometry.surface_ibm_geometry_report.nonmanifold_edges);
    dump.put("report.markers", geometry.surface_ibm_geometry_report.markers);
    dump.put("report.input_area_lu2", geometry.surface_ibm_geometry_report.input_area_lu2);
    dump.put("report.valid_area_lu2", geometry.surface_ibm_geometry_report.valid_area_lu2);
    dump.put("report.retained_area_lu2", geometry.surface_ibm_geometry_report.retained_area_lu2);
    dump.put("report.marker_area_lu2", geometry.surface_ibm_geometry_report.marker_area_lu2);
    dump.put("report.warnings", geometry.surface_ibm_geometry_report.warnings.size());
    dump.section("terrain");
    {
        Checksum sum;
        for (const DemCsvPoint& p : terrain.profile_dem_points_raw) sum.value(p.x).value(p.y).value(p.elevation);
        dump.put_checksum("dem_points", terrain.profile_dem_points_raw.size(), sum);
    }
    dump.put("dem_loaded", terrain.profile_dem_loaded);
    dump.put("dem_x_m", float3(terrain.profile_dem_xmin, terrain.profile_dem_xmax, 0.0f));
    dump.put("dem_y_m", float3(terrain.profile_dem_ymin, terrain.profile_dem_ymax, 0.0f));
    dump.put("dem_elevation_m", float3(terrain.profile_dem_emin, terrain.profile_dem_emax, 0.0f));
    dump.put("physical_z_zero_m", terrain.physical_output_z_zero_si);
    dump.put("physical_dem_ready", terrain.physical_output_dem_ready);
    dump.put("physical_ground.samples", terrain.physical_output_ground_plane.raw_count());
    dump.put("physical_ground.grid", std::to_string(terrain.physical_output_ground_plane.nx()) + " x " + std::to_string(terrain.physical_output_ground_plane.ny()));
    dump.put("physical_ground.structured", terrain.physical_output_ground_plane.structured_ready());
    if (terrain.physical_output_ground_plane.has_samples()) {
        dump.put("physical_ground.at_min", terrain.physical_output_ground_plane.eval_xy(geometry.stl_min_si.x, geometry.stl_min_si.y));
        dump.put("physical_ground.at_max", terrain.physical_output_ground_plane.eval_xy(geometry.stl_max_si.x, geometry.stl_max_si.y));
        dump.put("physical_ground.at_center", terrain.physical_output_ground_plane.eval_xy(0.5f * (geometry.stl_min_si.x + geometry.stl_max_si.x), 0.5f * (geometry.stl_min_si.y + geometry.stl_max_si.y)));
    }
}

void dump_case_physics(const CaseContext& ctx, ConfigDump& dump) {
    const CasePhysics& physics = ctx.physics;
    dump.put("si_reference_u", physics.si_ref_u);
    dump.put("u_scale", physics.u_scale);
    dump.put_units("units.", units);
    dump.put("z_offset_lbm", physics.z_off);
    dump.put("nu_lbm", physics.lbm_nu);
    dump.put("alpha_lbm", physics.lbm_alpha);
    dump.put("beta_lbm", physics.lbm_beta);
    dump.put("gravity_z_lbm", physics.lbm_buoyancy_gz);
    dump.put("transport.common_scale", physics.base_transport.common_scale);
    dump.put("transport.effective_momentum_tau", physics.base_transport.effective_momentum_tau);
    dump.put("transport.effective_thermal_tau", physics.base_transport.effective_thermal_tau);
    dump.put("coriolis_omega_lbm", float3(coriolis_Omegax_lbmu, coriolis_Omegay_lbmu, coriolis_Omegaz_lbmu));
    dump.put("coriolis_f_lbm", coriolis_f_lbmu);
    dump.put("buffer_face", buffer_downstream_face_id);
    dump.put("buffer_cells", buffer_n_cells);
    dump.put("buffer_inv_tau_lbm", buffer_inv_tau_lbmu);
    dump.put("buffer_active", buffer_nudging_active);
    dump.put("sponge_cells", sponge_n_cells);
    dump.put("sponge_inv_tau_lbm", sponge_inv_tau_lbmu);
    dump.put("sponge_active", top_sponge_active);
}

void dump_case_schedule(const CaseContext& ctx, ConfigDump& dump, const std::string& case_downstream_bc) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    const CasePhysics& physics = ctx.physics;
    const ProbePlan& probes = ctx.probes;
    const RunSchedule sc = make_run_schedule(deck, !probes.probe_requests.empty());
    dump.put("schedule.base_steps", sc.base_run_steps);
    dump.put("schedule.extra_steps", sc.extra_run_steps);
    dump.put("schedule.total_steps", sc.total_steps);
    dump.put("schedule.unsteady_interval", sc.unsteady_interval);
    dump.put("schedule.avg_window", sc.avg_window);
    dump.put("schedule.avg_stride", sc.avg_stride);
    dump.put("schedule.avg_start", sc.avg_start_t);
    dump.put("schedule.probe_window", sc.probe_window);
    dump.put("schedule.probe_start", sc.probe_start_t);
    dump.put("schedule.dt_s", (double)deck.cell_m * ((double)domain.lbm_ref_u / (double)physics.si_ref_u));
    dump.put("schedule.transform_dt_s", deck.cell_m * (domain.lbm_ref_u / physics.si_ref_u));
    if (domain.cuda_vertical_nesting_plan.enabled) {
        dump.put("schedule.coarse_dt_s", domain.cuda_vertical_nesting_plan.coarse_dx_si * ((double)domain.lbm_ref_u / (double)physics.si_ref_u));
        dump.put("schedule.nested_output_height_m", (double)(domain.top_sponge_grid_extend ? domain.Nz_cells_core - 1u : domain.lbm_N.z - 1u) * domain.cuda_vertical_nesting_plan.coarse_dx_si);
    }
    dump.put("temperature_outlet_face", deck.downstream_open_face ? buffer_face_id_from_bc(case_downstream_bc) : 0);
    const VkInletRuntimeConfig vk = make_vk_runtime_config(deck.vk_inlet_settings, case_downstream_bc);
    dump.put("vk.enable", vk.enable);
    dump.put("vk.ti", vk.ti);
    dump.put("vk.sigma_lbm", vk.sigma_lbm);
    dump.put("vk.length_lbm", vk.L_lbm);
    dump.put("vk.modes", vk.nmodes);
    dump.put("vk.seed", vk.seed);
    dump.put("vk.update_stride", vk.update_stride);
    dump.put("vk.uc_mode", vk_uc_mode_name(vk.uc_mode));
    dump.put("vk.same_realization_all_faces", vk.same_realization_all_faces);
    dump.put("vk.stride_interpolation", vk.stride_interpolation);
    dump.put("vk.inflow_only", vk.inflow_only);
    dump.put("vk.face_mode", vk_face_mode_name(vk.face_mode));
    dump.put("vk.anisotropy", vk.anisotropy_scale);
    dump.put("vk.downstream_face", vk.downstream_face_id);
}

void dump_profile_stage(const bc::ProfileInflow& profile, const std::vector<float>& profile_u_si,
                        const uint total_cases, const bool profile_single_case_standard_output) {
    if (!ConfigDump::enabled()) return;
    ConfigDump& dump = ConfigDump::get();
    dump.section("profile");
    dump.put("origin_lbm", profile.origin_lbmu);
    dump.put("flat_ground_lbm", profile.flat_ground_lbmu);
    dump.put("dem_ground", profile.use_profile_dem_ground);
    dump.put("ground_z_min_lbm", profile.ground_z_min_lbm);
    dump.put("ground_z_max_lbm", profile.ground_z_max_lbm);
    dump.put("ground.samples", profile.profile_ground_plane.raw_count());
    dump.put("ground.grid", std::to_string(profile.profile_ground_plane.nx()) + " x " + std::to_string(profile.profile_ground_plane.ny()));
    dump.put("ground.structured", profile.profile_ground_plane.structured_ready());
    if (profile.profile_ground_plane.has_samples()) {
        dump.put("ground.at_origin", profile.profile_ground_plane.eval_xy(profile.origin_lbmu.x, profile.origin_lbmu.y));
        dump.put("ground.at_center", profile.profile_ground_plane.eval_xy(0.0f, 0.0f));
        dump.put("ground.at_far", profile.profile_ground_plane.eval_xy(-profile.origin_lbmu.x, -profile.origin_lbmu.y));
    }
    dump.put("solver_top_m", profile.solver_top_si);
    dump.put("core_top_m", profile.core_top_si);
    dump.put("ground_min_m", profile.ground_min_si);
    dump.put("ground_max_m", profile.ground_max_si);
    dump.put("table_top_m", profile.profile_table_top_si);
    dump.put("table_steps", profile.profile_steps);
    dump.put("table_u_m_s", profile_u_si);
    dump.put("table_u_lbm", profile.profile_u_lbmu);
    dump.put("table_u_min", profile.u_min_prof);
    dump.put("table_u_max", profile.u_max_prof);
    dump.put("cases", total_cases);
    dump.put("standard_output_names", profile_single_case_standard_output);
}

} // namespace culbm::cases
