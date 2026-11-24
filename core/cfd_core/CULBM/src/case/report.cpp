#include "report.hpp"
#include "context.hpp"
#include "../app/run_report.hpp"
#include "../bc/outflow/damping_layers.hpp"

#include <iomanip>
#include <sstream>

namespace culbm::cases {

using namespace culbm::bc;

string hr_plain() {
    return "|"+string((uint)CONSOLE_WIDTH-2u, '-')+"|";
}
void print_section_title(const string& title) {
    print("\r");
    println(hr_plain());
    println("|"+alignc((uint)CONSOLE_WIDTH-2u, title)+"|");
    println(hr_plain());
}
void print_kv_row(const string& key, const string& value) {
    print("\r");
    println("| "+key+" | "+value+" |");
}
string avg_turbulence_output_desc(const CaseSettings& deck) {
    string out = "[";
    bool first = true;
    auto append_item = [&](const char* name) {
        if (!first) out += ",";
        out += name;
        first = false;
    };
    if (deck.output_tke_in_avg_vtk) append_item("tke");
    if (deck.output_ti_in_avg_vtk) append_item("ti");
    if (deck.output_tls_in_avg_vtk) append_item("tls");
    out += "]";
    return out;
}

void print_runtime_kv(const string& key, const string& value) {
    const StatusPause pause;
    print_kv_row(key, value);
}

void print_runtime_section(const string& title) {
    const StatusPause pause;
    print_section_title(title);
}

std::string fmtf(const float v, const int prec) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

std::string format_tag(const float v) {
    std::string s = to_string(v, 3u);
    size_t dot = s.find('.');
    if (dot != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s.empty() ? std::string("0") : s;
}

void emit_progress_stage(const std::string& stage, const std::string& label, const std::string& detail,
                         const long long current, const long long total, const bool indeterminate) {
    luw_emit_progress(stage, label, detail, current, total, indeterminate);
}

void print_setup_banner() {
    println(hr_plain());
    println("|"+alignc((uint)CONSOLE_WIDTH-2u, "LatticeUrbanWind LUW: Towards Micrometeorology Fastest Simulation")+"|");
    println("|"+alignc((uint)CONSOLE_WIDTH-2u, "Developed by Huanxia Wei's Team")+"|");
    println("|"+alignc((uint)CONSOLE_WIDTH-2u, "Version - CULBM-v4.0.0-261123")+"|");
    println(hr_plain());
    culbm::app::print_banner();
}

void print_parameter_information(const CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    const run::ProbePlan& probes = ctx.probes;
    println("|"+string((uint)CONSOLE_WIDTH-2u, ' ')+"|");
    print_section_title("PARAMETER INFORMATION");
 // println("| Grid Domains    | " + alignr(57u,to_string()) + " |");
    println("| Configure deck  | " + alignr(57u, to_string(deck.conf_used_path)) + " |");
    println("| Casename / Time | " + alignr(40u, to_string(deck.caseName)) + alignr(17u, to_string(deck.datetime)) + " |");
    println("| Basement Height | " + alignr(55u, to_string(fmtf(deck.z_si_offset))) + " m |");
    println("| SI Size (m)     | " + alignr(12u, " X:") + alignl(11u, to_string(fmtf(deck.si_size.x))) + "   Y: " + alignl(11u, to_string(fmtf(deck.si_size.y))) + "   Z: " + alignl(11u, to_string(fmtf(deck.si_size.z))) + " | ");
    if (deck.profile_generation) {
        println("| Downstream BC   | " + alignr(57u, "auto by angle (dominant axis)") + " |");
        println("| Normal Yaw      | " + alignr(57u, "auto by angle list") + " |");
    }
    else {
        println("| Downstream BC   | " + alignr(57u, to_string(deck.downstream_bc)) + " |");
        println("| Normal Yaw      | " + alignr(53u, to_string(deck.downstream_bc_yaw)) + " deg |");
    }
    println("| Downstream Open | " + alignr(57u, deck.downstream_open_face ? string("true") : string("false")) + " |");
    println("| GPU Decompose   | " + alignr(49u, to_string(deck.Dx)) + ", " + alignr(2u, to_string(deck.Dy)) + ", " + alignr(2u, to_string(deck.Dz)) + " |");
    println("| VRAM Request    | " + alignr(54u, to_string(deck.memory)) + " MB |");
    println("| Run Steps       | " + alignr(57u, deck.run_nstep_override > 0ull ? to_string(deck.run_nstep_override) + " (run_nstep)" : string("20001 (default)")) + " |");
    println("| Purge Avg Steps | " + alignr(57u, deck.purge_avg_steps > 0u ? to_string(deck.purge_avg_steps) : string("off")) + " |");
    println("| Purge Avg Stride | " + alignr(57u, deck.purge_avg_steps > 0u ? ("every " + to_string((ulong)deck.purge_avg_stride) + " step(s)") : string("n/a")) + " |");
    println("| Avg Turb Output | " + alignr(57u, avg_turbulence_output_desc(deck)) + " |");
    println("| Unsteady U VTK | " + alignr(57u, deck.unsteady_output_interval > 0u ? ("every " + to_string((ulong)deck.unsteady_output_interval) + " step(s)") : string("off")) + " |");
    {
        std::string probes_desc = "off";
        if (!probes.probe_requests.empty()) {
            probes_desc = to_string((ulong)probes.probe_requests.size()) + " request(s)";
            if (!probes.probe_geo_mapping.valid) probes_desc += " (mapping unavailable)";
        }
        println("| Probes         | " + alignr(57u, probes_desc) + " |");
        std::string probe_window_desc = "n/a";
        if (!probes.probe_requests.empty()) {
            if (deck.probes_output_defined && deck.probes_output_steps > 0u) {
                probe_window_desc = "last " + to_string((ulong)deck.probes_output_steps) + " step(s) via probes_output";
            }
            else if (deck.purge_avg_steps > 0u || deck.research_output_steps > 0u) {
                const ulong fallback_steps = (ulong)std::max(deck.purge_avg_steps, deck.research_output_steps);
                probe_window_desc = "fallback last " + to_string(fallback_steps) + " step(s)";
            }
            else {
                probe_window_desc = "entire simulation";
            }
        }
        println("| Probes Window  | " + alignr(57u, probe_window_desc) + " |");
    }
    println("| Buoyancy request| " + alignr(57u, deck.buoyancy_enabled ? (deck.buoyancy_explicit ? string("true") : string("true (default)")) : string("false")) + " |");
    println("| Boussinesq Tref | " + alignr(57u, deck.boussinesq_reference_auto
            ? string("auto (input Tmin/Tmax center)")
            : to_string(fmtf(deck.boussinesq_reference_setting_kelvin)) + " K (explicit)") + " |");
    println("| Thermal Lattice | " + alignr(57u, string(get_temperature_lattice_name())) + " |");
    println("| SGS LES         | " + alignr(57u, deck.culbm_sgs_enabled
            ? string("Smagorinsky-Lilly enabled") : string("disabled")) + " |");
    println("| SGS Cs          | " + alignr(57u,
            to_string(fmtf((float)deck.culbm_sgs_cs, 9))) + " |");
    println("| SGS scalar      | " + alignr(57u,
            deck.culbm_sgs_enabled && deck.culbm_sgs_scalar_coupling
                ? string("enabled") : string("disabled")) + " |");
    println("| SGS Pr_t        | " + alignr(57u,
            to_string(fmtf((float)deck.culbm_sgs_turbulent_prandtl, 6))) + " |");
    println("| SGS diagnostics | " + alignr(57u,
            deck.culbm_sgs_diagnostics_interval > 0u
                ? "every " + to_string((ulong)deck.culbm_sgs_diagnostics_interval) +
                    " step(s)"
                : string("off")) + " |");
    println("| Wall model      | " + alignr(57u, deck.wall_model_settings.enabled
            ? string(deck.surface_ibm_settings.enabled
                ? (deck.wall_model_settings.surface_ibm
                    ? "log-law wall stress (voxel walls and IBM surfaces)"
                    : "log-law wall stress (voxel walls; IBM surfaces no-slip)")
                : "log-law wall stress on bounce-back walls")
            : string("disabled (plain no-slip bounce-back)")) + " |");
    if (deck.wall_model_settings.enabled) {
        auto z0_text = [&](const float z0) {
            return z0 > 0.0f ? to_string(fmtf(z0, 6)) + " m" : string("smooth (z0=0)");
        };
        println("| WM z0 ground    | " + alignr(57u, z0_text(deck.wall_model_settings.z0_ground_m) +
                " (upward-facing faces)") + " |");
        println("| WM z0 building  | " + alignr(57u, z0_text(deck.wall_model_settings.z0_building_m) +
                " (vertical/downward faces)") + " |");
        println("| WM kappa        | " + alignr(57u, to_string(fmtf(deck.wall_model_settings.kappa, 4))) + " |");
        println("| WM SGS layer    | " + alignr(57u, deck.wall_model_settings.sgs_layers > 0u && deck.culbm_sgs_enabled
                ? to_string((ulong)deck.wall_model_settings.sgs_layers) + " row(s): Smagorinsky l=max(Cs*dx,kappa*z_w)"
                : string("off")) + " |");
    }
    println("| SurfaceIBM      | " + alignr(57u,
            deck.surface_ibm_settings.enabled
                ? string("enabled (direct forcing)")
                : string("disabled (voxel wall path)")) + " |");
    if (deck.surface_ibm_settings.enabled) {
        println("| IBM delta       | " + alignr(57u,
                string(culbm::surface_ibm::delta_kernel_name(
                    deck.surface_ibm_settings.delta_kernel))) + " |");
        println("| IBM geometry    | " + alignr(57u,
                string(culbm::surface_ibm::geometry_policy_name(
                    deck.surface_ibm_settings.geometry_policy))) + " |");
        println("| IBM spacing     | " + alignr(53u,
                to_string(fmtf(deck.surface_ibm_settings.marker_spacing_lu, 4))) +
                " lu |" );
        println("| IBM iterations  | " + alignr(57u,
                to_string((ulong)deck.surface_ibm_settings.forcing_iterations)) +
                " |");
    }
    println("| Requested nu SI | " + alignr(51u,
            to_string(fmtf((float)deck.culbm_base_kinematic_viscosity_m2_s, 9))) +
            " m^2/s |" );
    println("| Req. alpha SI   | " + alignr(51u,
            to_string(fmtf((float)deck.culbm_base_thermal_diffusivity_m2_s, 9))) +
            " m^2/s |" );
    println("| VK inlet switch | " + alignr(57u, deck.vk_inlet_settings.enable ? string("true") : string("false")) + " |");
    if (deck.vk_inlet_settings.enable) {
        println("| VK TI (frac)    | " + alignr(57u, to_string(fmtf(deck.vk_inlet_settings.ti))) + " |");
        println("| VK sigma (SI)   | " + alignr(57u, to_string(fmtf(deck.vk_inlet_settings.sigma_si))) + " m/s |");
        println("| VK L (SI)       | " + alignr(57u, to_string(fmtf(deck.vk_inlet_settings.L_si))) + " m |");
        println("| VK modes        | " + alignr(57u, to_string((ulong)deck.vk_inlet_settings.nmodes)) + " |");
        println("| VK stride       | " + alignr(57u, to_string((ulong)deck.vk_inlet_settings.update_stride)) + " |");
        println("| VK Uc mode      | " + alignr(57u, string(vk_uc_mode_name(deck.vk_inlet_settings.uc_mode))) + " |");
        println("| VK aniso scale  | " + alignr(38u, "[") +
                alignr(6u, to_string(fmtf(deck.vk_inlet_settings.anisotropy_scale.x))) + ", " +
                alignr(6u, to_string(fmtf(deck.vk_inlet_settings.anisotropy_scale.y))) + ", " +
                alignr(6u, to_string(fmtf(deck.vk_inlet_settings.anisotropy_scale.z))) + "] |");
        println("| VK same realiz. | " + alignr(57u, deck.vk_inlet_settings.same_realization_all_faces ? string("true") : string("false")) + " |");
        println("| VK inflow only  | " + alignr(57u, deck.vk_inlet_settings.inflow_only ? string("true") : string("false")) + " |");
        println("| VK face mode    | " + alignr(57u, string(vk_face_mode_name(deck.vk_inlet_settings.face_mode))) + " |");
    }
}

void print_domain_report(const CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    print_section_title("DOMAIN AND TRANSFORMATION");
    if (domain.cuda_ground_compaction_active) {
        print_kv_row("CUDA ground", "base -> one z=0 solid layer; trim=" + to_string((ulong)domain.cuda_ground_trim_cells) +
                " complete cells (" + to_string(fmtf(domain.cuda_ground_trim_si)) + " m)");
    }
    else {
        print_kv_row("CUDA ground", "base compaction skipped: " + domain.cuda_ground_compaction_reason);
    }
    const ulong n_cells_total = (ulong)domain.lbm_N.x*(ulong)domain.lbm_N.y*(ulong)domain.lbm_N.z;
    println("| Grid Resolution | " + alignr(45u, to_string(domain.lbm_N.x)) + "," + alignr(5u, to_string(domain.lbm_N.y)) + "," + alignr(5u, to_string(domain.lbm_N.z)) + " (nCell = " + to_string(n_cells_total) + ") |");
    if (domain.cuda_vertical_nesting_plan.enabled) {
        const char* nesting_mode = domain.cuda_vertical_nesting_plan.has_three_levels()
            ? "three levels: 4x/2x/1x" : "two levels: 2x/1x";
        print_kv_row("CUDA nesting", string(nesting_mode) +
            ", heights above compact ground = " +
            to_string(deck.cuda_vertical_nesting_agl_si.size() == 1u
                ? deck.cuda_vertical_nesting_agl_si[0] : deck.cuda_vertical_nesting_agl_si[0]) +
            (deck.cuda_vertical_nesting_agl_si.size() == 2u
                ? ", " + to_string(deck.cuda_vertical_nesting_agl_si[1]) : "") + " m");
        for (const auto& level : domain.cuda_vertical_nesting_plan.levels) {
            print_kv_row("", level.label + ": dx=" +
                to_string(level.dx_si, 6u) + " m, N=" +
                to_string((ulong)level.nx) + "x" + to_string((ulong)level.ny) + "x" +
                to_string((ulong)level.allocated_nz) +
                " (physical z=" + to_string((ulong)level.physical_nz) + ")");
        }
    }
    if (domain.top_sponge_grid_extend) {
        println("| Top sponge grid | " + alignr(57u, "core Nz=" + to_string((ulong)domain.Nz_cells_core) +
                ", ext=" + to_string((ulong)domain.sponge_cells_cfg) +
                ", total Nz=" + to_string((ulong)domain.lbm_N.z)) + " |");
    }
    if (domain.gpu_memory_estimate.extra_per_device_mb > 0u) {
        println("| GPU Estimate    | " + alignr(57u, to_string(deck.Dx * deck.Dy * deck.Dz) + "x " + to_string(domain.gpu_memory_estimate.total_per_device_mb) +
                " MB (core " + to_string(domain.gpu_memory_estimate.core_per_device_mb) +
                " + extra " + to_string(domain.gpu_memory_estimate.extra_per_device_mb) + ")") + " |");
    } else {
        println("| GPU Estimate    | " + alignr(57u, to_string(deck.Dx * deck.Dy * deck.Dz) + "x " + to_string(domain.gpu_memory_estimate.total_per_device_mb) + " MB") + " |");
    }
}

void print_reference_report(const CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    const bc::ThermalReference& thermal = ctx.thermal;
    const CasePhysics& physics = ctx.physics;
    const CaseDomain& domain = ctx.domain;
    if (!deck.dataset_generation && !deck.profile_generation) {
        println("| SI Reference U  | " + alignl(7u, to_string(fmtf(physics.si_ref_u))) + alignl(50u, "m/s") + " |");
        println("| LBM Reference U | " + alignl(7u, to_string(fmtf(domain.lbm_ref_u))) + alignl(50u, "(Nondimensionalized)") + " |");
        if (thermal.temperature_ref_adaptive) {
            println("| Temp Reference  | " + alignr(57u, to_string(fmtf(thermal.temperature_ref_kelvin)) + " K (auto center of input Tmin/Tmax)") + " |");
        }
        else if (!deck.boussinesq_reference_auto) {
            println("| Temp Reference  | " + alignr(57u, to_string(fmtf(thermal.temperature_ref_kelvin)) + " K (explicit Boussinesq reference)") + " |");
        }
        else {
            println("| Temp Reference  | " + alignr(57u, to_string(fmtf(thermal.temperature_ref_kelvin)) + " K (auto fallback default)") + " |");
        }
        if (thermal.temperature_scale_adaptive) {
            println("| Temp Scale      | " + alignr(57u, to_string(fmtf(thermal.temperature_scale_kelvin)) + " K per 1.0 T_lbm (auto from input range)") + " |");
        }
        else {
            println("| Temp Scale      | " + alignr(57u, to_string(fmtf(thermal.temperature_scale_kelvin)) + " K per 1.0 T_lbm (default)") + " |");
        }
        print_base_transport(ctx, "initial units", physics.base_transport);
        println("| Thermal lattice | " + alignr(57u, string(get_temperature_lattice_name())) + " |");
        println("| Thermal beta    | " + alignr(57u, deck.buoyancy_enabled ? to_string(physics.lbm_beta, 8u) : string("0 (buoyancy inactive)")) + " |");
        println("| Buoyancy g_z    | " + alignr(57u, deck.buoyancy_enabled ? to_string(physics.lbm_buoyancy_gz, 8u) : string("0 (disabled)")) + " |");
        if (deck.buoyancy_enabled && thermal.boussinesq_temperature_range_checked) {
            println("| Boussinesq dT/T | " + alignr(57u,
                    to_string(100.0f * thermal.boussinesq_max_relative_temperature_difference, 5u) +
                    "% max |T-Tref|/Tref") + " |");
            if (thermal.boussinesq_max_relative_temperature_difference >
                k_boussinesq_relative_temperature_warning) {
                println("| WARNING         | max |T-Tref|/Tref exceeds 5%; Boussinesq small-dT accuracy is not assured. |");
            }
        }
        if (physics.lbm_alpha < 1.0e-6f) {
            println("| WARNING         | thermal alpha is very small in LBM units; temperature diffusion may be stiff. |");
        }
    }
}

} // namespace culbm::cases
