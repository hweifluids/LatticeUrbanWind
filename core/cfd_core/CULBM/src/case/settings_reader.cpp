#include "settings.hpp"
#include "deck.hpp"
#include "features.hpp"
#include "geometry.hpp"
#include "memory_fit.hpp"
#include "../app/launch.hpp"
#include "../solver/field_set.hpp"

#include <cstdlib>
#include <sstream>

namespace culbm::cases {

using namespace culbm::bc;

namespace {

std::string read_line_console() {
    std::string out;
    std::cout.flush();
    std::getline(std::cin, out);
    return out;
}

} // namespace

void read_case_settings(CaseSettings& settings) {
    settings.culbm_sgs_enabled = get_subgrid_configuration().enabled;
    settings.wall_model_settings = get_wall_model_configuration();
    settings.wall_model_settings.enabled = true; // production deck default: culbm_wall_model = true

    // ---------------------- read *.luw / *.luwdg ------------------------------
    {
        std::string luw_path_input;
        auto trim = [](std::string s) {
            const char* ws = " \t\r\n";
            size_t b = s.find_first_not_of(ws);
            size_t e = s.find_last_not_of(ws);
            return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
            };

        // Command-line mode: use the first argument as configure file path.
        const std::vector<std::string>& case_arguments = culbm::app::case_arguments();
        if (!case_arguments.empty()) {
            luw_path_input = trim(case_arguments[0]);
            if (case_arguments.size() > 1u) {
                println("| WARNING: extra CLI args ignored (device ID args are no longer supported).    |");
            }
        }

        // Interactive fallback when no path is provided via CLI.
        if (luw_path_input.empty()) {
            println("| Please input configure file path (*.luw, *.luwdg, *.luwpf):                  |");
            luw_path_input = read_line_console();
            luw_path_input = trim(luw_path_input);
        }
        
        if (!luw_path_input.empty()) {
            char q = luw_path_input.front();
            if ((q == '"' || q == '\'') && luw_path_input.size() >= 2 && luw_path_input.back() == q) {
                luw_path_input = luw_path_input.substr(1, luw_path_input.size() - 2);
                luw_path_input = trim(luw_path_input);
            }
            std::filesystem::path p = std::filesystem::path(luw_path_input).make_preferred();
            luw_path_input = p.string();
        }

        if (!luw_path_input.empty()) {
            std::filesystem::path luw_path_fs(luw_path_input);
            std::string ext = luw_path_fs.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".luwdg") {
                settings.dataset_generation = true;
                println("| Dataset generation mode enabled (*.luwdg).                                  |");
            }
            else if (ext == ".luwpf") {
                settings.profile_generation = true;
                println("| Profile forcing mode enabled (*.luwpf).                                     |");
            }
            else if (!ext.empty() && ext != ".luw") {
                println("| WARNING: config suffix is not *.luw, *.luwdg, or *.luwpf; treating as *.luw.|");
            }
            else if (ext.empty()) {
                println("| WARNING: config has no suffix; treating as *.luw.                           |");
            }
        }

        std::ifstream fin;

        if (!luw_path_input.empty()) {
            std::filesystem::path luw_path_fs(luw_path_input);
            fin.open(luw_path_fs);
            if (fin.is_open()) settings.conf_used_path = luw_path_fs.string();
        }

        // helpers
        auto second_val = [](const std::string& rng) {
            size_t c = rng.find(','); size_t r = rng.find(']', c);
            return (float)atof(rng.substr(c + 1, r - c - 1).c_str());
            };
        auto parse_pair_float = [&](const std::string& rng, float& a, float& b) {
            size_t lb = rng.find('[');
            size_t rb = rng.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) { a = 0.0f; b = 0.0f; return; }
            std::string inside = rng.substr(lb + 1, rb - lb - 1);
            std::stringstream ss(inside);
            std::string token;
            std::vector<float> vals;
            while (std::getline(ss, token, ',')) {
                vals.push_back((float)atof(trim(token).c_str()));
            }
            if (vals.size() >= 1) a = vals[0];
            if (vals.size() >= 2) b = vals[1];
            };
        auto parse_triplet_uint = [&](const std::string& rng, uint& a, uint& b, uint& c) {
            size_t lb = rng.find('[');
            size_t rb = rng.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) return;
            std::string inside = rng.substr(lb + 1, rb - lb - 1);
            std::stringstream ss(inside);
            std::string token;
            uint vals[3] = { a, b, c };
            int i = 0;
            while (std::getline(ss, token, ',') && i < 3) {
                vals[i++] = static_cast<uint>(atoi(trim(token).c_str()));
            }
            if (i == 3) { a = vals[0]; b = vals[1]; c = vals[2]; }
            };
        auto parse_triplet_float = [&](const std::string& rng, float& a, float& b, float& c) -> bool {
            std::string s = trim(rng);
            size_t lb = s.find('[');
            size_t rb = s.find(']', lb);
            std::string inside = (lb != std::string::npos && rb != std::string::npos && rb > lb)
                ? s.substr(lb + 1, rb - lb - 1)
                : s;
            std::stringstream ss(inside);
            std::string token;
            float vals[3] = { a, b, c };
            int i = 0;
            while (std::getline(ss, token, ',') && i < 3) {
                const std::string t = trim(token);
                if (t.empty()) return false;
                char* endptr = nullptr;
                const float parsed = std::strtof(t.c_str(), &endptr);
                if (endptr == t.c_str()) return false;
                vals[i++] = parsed;
            }
            if (i != 3) return false;
            a = vals[0];
            b = vals[1];
            c = vals[2];
            return true;
            };
        auto parse_float_list = [&](const std::string& rng, std::vector<float>& out) {
            out.clear();
            std::string s = trim(rng);
            size_t lb = s.find('[');
            size_t rb = s.find(']', lb);
            std::string inside = (lb != std::string::npos && rb != std::string::npos && rb > lb)
                ? s.substr(lb + 1, rb - lb - 1)
                : s;
            std::stringstream ss(inside);
            std::string token;
            while (std::getline(ss, token, ',')) {
                std::string t = trim(token);
                if (!t.empty()) out.push_back((float)atof(t.c_str()));
            }
            };

        if (!fin.is_open()) {
            println("ERROR: config not found. Please provide a valid *.luw, *.luwdg, or *.luwpf and rerun.");
            wait();
            exit(-1);
        }

        else {
            const ParsedDeck entries = read_deck_entries(fin);
            std::string particle_advection_error;
            if (!culbm::particle_advection::parse_config(
                    entries.values,
                    settings.particle_advection_settings,
                    particle_advection_error)) {
                println("| ERROR: invalid CULBM particle-advection config: " +
                        particle_advection_error + " |");
                wait();
                exit(-1);
            }
            std::string surface_ibm_error;
            if (!culbm::surface_ibm::parse_config(
                    entries.values,
                    settings.surface_ibm_settings,
                    surface_ibm_error)) {
                println("| ERROR: invalid CULBM SurfaceIBM config: " +
                        surface_ibm_error + " |");
                wait();
                exit(-1);
            }
            // Memory fitting below consults the immutable runtime snapshot, so
            // publish SurfaceIBM immediately after strict deck validation and
            // before any grid-size/VRAM estimate or LBM construction.
            culbm::surface_ibm::set_config(settings.surface_ibm_settings);
            std::string mesh_control_val, gpu_memory_val, cell_size_val;
            auto parse_bool_setting = [&](const std::string& txt, bool& out) -> bool {
                return deck_try_parse_bool(txt, out);
            };
            auto parse_boolish_setting = [&](const std::string& txt, bool& out) -> bool {
                return deck_try_parse_bool(txt, out);
            };
            auto fail_culbm_setting = [&](const std::string& message) {
                println("| ERROR: " + message + " |");
                wait();
                exit(-1);
            };
            // Deck entries are already normalized here, so handlers stay insensitive
            // to original line order, section order, key aliases, and quoted bool tokens.
            for (const auto& entry : entries.values) {
                const std::string& key = entry.first;
                const std::string& val = entry.second;
                auto unquote = [&](std::string s) {
                    return deck_unquote(std::move(s));
                    };

                if (key == "casename")        settings.caseName = unquote(val);
                else if (key == "datetime") {
                    settings.datetime = unquote(val);
                    settings.datetime_from_config = true;
                }
                else if (key == "downstream_bc") settings.downstream_bc = unquote(val);
                else if (key == "high_order") {
                    bool parsed = false;
                    std::string v = unquote(val);
                    if (!v.empty() && parse_bool_setting(v, parsed)) settings.use_high_order = parsed;
                }
                else if (key == "flux_correction") {
                    bool parsed = false;
                    std::string v = unquote(val);
                    if (!v.empty() && parse_bool_setting(v, parsed)) settings.flux_correction = parsed;
                }
                else if (key == "buoyancy") {
                    std::string v = unquote(val);
                    if (v.empty()) continue;
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    settings.buoyancy_explicit = true;
                    bool parsed = true;
                    if (parse_bool_setting(v, parsed)) settings.buoyancy_enabled = parsed;
                    else settings.buoyancy_enabled = true; // keep default behavior for any non-false explicit value
                }
                else if (key == "culbm_boussinesq_reference_temperature_k") {
                    std::string v = trim(unquote(val));
                    std::string normalized = v;
                    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
                    if (normalized == "auto") {
                        settings.boussinesq_reference_auto = true;
                        settings.boussinesq_reference_setting_kelvin = k_temperature_ref_kelvin;
                    }
                    else {
                        char* endptr = nullptr;
                        const float parsed = std::strtof(v.c_str(), &endptr);
                        const bool has_trailing_text = endptr != nullptr &&
                            !trim(std::string(endptr)).empty();
                        if (v.empty() || endptr == v.c_str() || has_trailing_text ||
                            !std::isfinite(parsed) || parsed <= 0.0f) {
                            println("| ERROR: culbm_boussinesq_reference_temperature_k must be auto or a positive finite Kelvin value. |");
                            wait();
                            exit(-1);
                        }
                        settings.boussinesq_reference_auto = false;
                        settings.boussinesq_reference_setting_kelvin = parsed;
                    }
                }
                else if (key == "culbm_velocity_set") {
                    std::string v = unquote(val);
                    std::transform(v.begin(), v.end(), v.begin(), ::toupper);
                    if (v == "D3Q19" || v == "19") {
                        settings.momentum_lattice_setting = Momentum_Lattice::Q19;
                    }
                    else if (v == "D3Q27" || v == "27") {
                        settings.momentum_lattice_setting = Momentum_Lattice::Q27;
                    }
                    else {
                        fail_culbm_setting("culbm_velocity_set must be D3Q19 or D3Q27.");
                    }
                }
                else if (key == "culbm_ddf_storage") {
                    std::string v = unquote(val);
                    std::transform(v.begin(), v.end(), v.begin(), ::toupper);
                    if (v == "FP16C") {
                        settings.ddf_storage_setting = DDF_Storage::FP16C;
                    }
                    else if (v == "FP16S") {
                        settings.ddf_storage_setting = DDF_Storage::FP16S;
                    }
                    else if (v == "FP32") {
                        settings.ddf_storage_setting = DDF_Storage::FP32;
                    }
                    else {
                        fail_culbm_setting("culbm_ddf_storage must be FP16C, FP16S or FP32.");
                    }
                }
                else if (key == "culbm_temperature_lattice" ||
                         key == "temperature_lattice") {
                    std::string v = unquote(val);
                    std::transform(v.begin(), v.end(), v.begin(), ::toupper);
                    if (v == "D3Q7" || v == "7") {
                        settings.temperature_lattice_setting = Temperature_Lattice::Q7;
                    }
                    else if (v == "D3Q19" || v == "19") {
                        settings.temperature_lattice_setting = Temperature_Lattice::Q19;
                    }
                    else {
                        println("| ERROR: culbm_temperature_lattice must be D3Q7 or D3Q19.                     |");
                        wait();
                        exit(-1);
                    }
                }
                else if (key == "culbm_sgs_enable") {
                    bool parsed = false;
                    if (!deck_try_parse_bool(val, parsed)) {
                        fail_culbm_setting(
                            "culbm_sgs_enable must be a bool-like value.");
                    }
                    settings.culbm_sgs_enabled = parsed;
                }
                else if (key == "culbm_sgs_cs") {
                    double parsed = 0.0;
                    if (!deck_try_parse_finite_double(val, parsed) ||
                        parsed < 0.0 ||
                        !std::isfinite((float)parsed)) {
                        fail_culbm_setting(
                            "culbm_sgs_cs must be a finite value >= 0.");
                    }
                    settings.culbm_sgs_cs = parsed;
                    settings.culbm_sgs_cs_explicit = true;
                }
                else if (key == "culbm_sgs_cs_squared") {
                    double parsed = 0.0;
                    if (!deck_try_parse_finite_double(val, parsed) ||
                        parsed < 0.0) {
                        fail_culbm_setting(
                            "culbm_sgs_cs_squared must be a finite value >= 0.");
                    }
                    const double cs = std::sqrt(parsed);
                    if (!std::isfinite((float)cs)) {
                        fail_culbm_setting(
                            "sqrt(culbm_sgs_cs_squared) exceeds FP32 range.");
                    }
                    settings.culbm_sgs_cs = cs;
                    settings.culbm_sgs_cs_squared_explicit = true;
                }
                else if (key == "culbm_sgs_scalar_coupling") {
                    bool parsed = false;
                    if (!deck_try_parse_bool(val, parsed)) {
                        fail_culbm_setting(
                            "culbm_sgs_scalar_coupling must be a bool-like value.");
                    }
                    settings.culbm_sgs_scalar_coupling = parsed;
                }
                else if (key == "culbm_wall_model") {
                    bool parsed = false;
                    if (!deck_try_parse_bool(val, parsed)) {
                        fail_culbm_setting(
                            "culbm_wall_model must be a bool-like value.");
                    }
                    settings.wall_model_settings.enabled = parsed;
                }
                else if (key == "culbm_wall_model_ibm") {
                    bool parsed = false;
                    if (!deck_try_parse_bool(val, parsed)) {
                        fail_culbm_setting(
                            "culbm_wall_model_ibm must be a bool-like value.");
                    }
                    settings.wall_model_settings.surface_ibm = parsed;
                }
                else if (key == "culbm_wall_model_z0_ground_m" ||
                         key == "culbm_wall_model_z0_building_m") {
                    double parsed = 0.0;
                    if (!deck_try_parse_finite_double(val, parsed) ||
                        parsed < 0.0 ||
                        !std::isfinite((float)parsed)) {
                        fail_culbm_setting(
                            key + " must be a finite roughness length >= 0 m (0 = smooth wall).");
                    }
                    if (key == "culbm_wall_model_z0_ground_m")
                        settings.wall_model_settings.z0_ground_m = (float)parsed;
                    else
                        settings.wall_model_settings.z0_building_m = (float)parsed;
                }
                else if (key == "culbm_wall_model_sgs_layers") {
                    double parsed = 0.0;
                    if (!deck_try_parse_finite_double(val, parsed) ||
                        parsed < 0.0 || parsed > 64.0 ||
                        std::floor(parsed) != parsed) {
                        fail_culbm_setting(
                            "culbm_wall_model_sgs_layers must be an integer in [0, 64].");
                    }
                    settings.wall_model_settings.sgs_layers = (uint)parsed;
                }
                else if (key == "culbm_wall_model_kappa") {
                    double parsed = 0.0;
                    if (!deck_try_parse_finite_double(val, parsed) ||
                        !(parsed > 0.0) || parsed > 1.0) {
                        fail_culbm_setting(
                            "culbm_wall_model_kappa must lie in (0, 1].");
                    }
                    settings.wall_model_settings.kappa = (float)parsed;
                }
                else if (key == "culbm_sgs_turbulent_prandtl") {
                    double parsed = 0.0;
                    if (!deck_try_parse_finite_double(val, parsed) ||
                        parsed <= 0.0 ||
                        !std::isfinite((float)parsed) ||
                        !((float)parsed > 0.0f)) {
                        fail_culbm_setting(
                            "culbm_sgs_turbulent_prandtl must be finite and > 0.");
                    }
                    settings.culbm_sgs_turbulent_prandtl = parsed;
                }
                else if (key == "culbm_base_kinematic_viscosity_m2_s") {
                    double parsed = 0.0;
                    if (!deck_try_parse_finite_double(val, parsed) ||
                        parsed <= 0.0 ||
                        !std::isfinite((float)parsed) ||
                        !((float)parsed > 0.0f)) {
                        fail_culbm_setting(
                            "culbm_base_kinematic_viscosity_m2_s must be finite and > 0.");
                    }
                    settings.culbm_base_kinematic_viscosity_m2_s = parsed;
                }
                else if (key == "culbm_base_thermal_diffusivity_m2_s") {
                    double parsed = 0.0;
                    if (!deck_try_parse_finite_double(val, parsed) ||
                        parsed <= 0.0 ||
                        !std::isfinite((float)parsed) ||
                        !((float)parsed > 0.0f)) {
                        fail_culbm_setting(
                            "culbm_base_thermal_diffusivity_m2_s must be finite and > 0.");
                    }
                    settings.culbm_base_thermal_diffusivity_m2_s = parsed;
                }
                else if (key == "culbm_sgs_diagnostics_interval") {
                    uint parsed = 0u;
                    if (!deck_try_parse_uint(val, parsed)) {
                        fail_culbm_setting(
                            "culbm_sgs_diagnostics_interval must be an integer >= 0.");
                    }
                    settings.culbm_sgs_diagnostics_interval = parsed;
                }
                else if (key == "downstream_bc_yaw") settings.downstream_bc_yaw = unquote(val);
                else if (key == "downstream_open_face") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (raw.empty()) {
                    }
                    else if (!parse_bool_setting(raw, parsed)) {
                        println("| WARNING: downstream_open_face expects a bool-like value. Keep default false.  |");
                    }
                    else {
                        settings.downstream_open_face = parsed;
                    }
                }
                else if (key == "base_height") {
                    if (!unquote(val).empty()) settings.z_si_offset = (float)atof(val.c_str());
                }
                else if (key == "z_limit") {
                    if (!unquote(val).empty()) settings.z_limit_override = (float)atof(val.c_str());
                }
                else if (key == "memory_lbm") {
                    if (!unquote(val).empty()) settings.memory = (uint)atoi(val.c_str());
                }
                else if (key == "si_x_cfd") {
                    if (!unquote(val).empty()) settings.si_size.x = second_val(val);
                }
                else if (key == "si_y_cfd") {
                    if (!unquote(val).empty()) settings.si_size.y = second_val(val);
                }
                else if (key == "si_z_cfd") {
                    if (!unquote(val).empty()) settings.si_size.z = second_val(val);
                }
                else if (key == "enable_buffer_nudging") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (raw.empty()) {
                    }
                    else if (!parse_boolish_setting(raw, parsed)) {
                        println("| WARNING: enable_buffer_nudging expects a bool-like value. Keep current/default.|");
                    }
                    else {
                        settings.enable_buffer_nudging = parsed;
                    }
                }
                else if (key == "buffer_thickness_m") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.buffer_thickness_m = (float)atof(raw.c_str());
                }
                else if (key == "buffer_tau_s") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.buffer_tau_s = (float)atof(raw.c_str());
                }
                else if (key == "buffer_nudge_vertical") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (raw.empty()) {
                    }
                    else if (!parse_boolish_setting(raw, parsed)) {
                        println("| WARNING: buffer_nudge_vertical expects a bool-like value. Keep current/default.|");
                    }
                    else {
                        settings.buffer_nudge_vertical = parsed ? 1 : 0;
                    }
                }
                else if (key == "enable_top_sponge") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (raw.empty()) {
                    }
                    else if (!parse_boolish_setting(raw, parsed)) {
                        println("| WARNING: enable_top_sponge expects a bool-like value. Keep current/default.   |");
                    }
                    else {
                        settings.enable_top_sponge = parsed;
                    }
                }
                else if (key == "sponge_thickness_m") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.sponge_thickness_m = (float)atof(raw.c_str());
                }
                else if (key == "sponge_tau_s") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.sponge_tau_s = (float)atof(raw.c_str());
                }
                else if (key == "sponge_ref_mode") {
                    std::string raw = unquote(val);
                    std::string v = raw;
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    if (v == "0" || v == "mode0" || v == "mode_0") {
                        settings.sponge_ref_mode = 0;
                    }
                    else if (v == "1" || v == "mode1" || v == "mode_1" || v == "geostrophic") {
                        settings.sponge_ref_mode = 1;
                    }
                    else {
                        char* endptr = nullptr;
                        const long parsed = std::strtol(raw.c_str(), &endptr, 10);
                        if (endptr != raw.c_str() && *endptr == '\0') {
                            settings.sponge_ref_mode = (int)parsed;
                        }
                        else {
                            println("| WARNING: sponge_ref_mode parse failed. Fallback to mode 0.                  |");
                            settings.sponge_ref_mode = 0;
                        }
                    }
                }
                else if (key == "mesh_control") mesh_control_val = unquote(val);
                else if (key == "gpu_memory")   gpu_memory_val = unquote(val);
                else if (key == "cell_size")    cell_size_val = unquote(val);
                else if (key == "cuda_vertical_nesting_agl") {
                    parse_float_list(val, settings.cuda_vertical_nesting_agl_si);
                }
                else if (key == "n_gpu") {
                    if (!unquote(val).empty()) parse_triplet_uint(val, settings.Dx, settings.Dy, settings.Dz);
                }
                else if (key == "research_output") {
                    if (!unquote(val).empty()) settings.research_output_steps = (uint)atoi(val.c_str());
                }
                else if (key == "probes_output") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) {
                        const int v = atoi(raw.c_str());
                        settings.probes_output_defined = true;
                        if (v > 0) settings.probes_output_steps = (uint)v;
                        else {
                            settings.probes_output_steps = 0u;
                            println("| WARNING: probes_output must be > 0 to take effect. Fallback to legacy window. |");
                        }
                    }
                }
                else if (key == "unsteady_output") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) {
                        const int v = atoi(raw.c_str());
                        settings.unsteady_output_interval = v > 0 ? static_cast<uint>(v) : 0u;
                    }
                }
                else if (key == "culbm_unsteady_start_step") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) {
                        const long long v = atoll(raw.c_str());
                        settings.culbm_unsteady_start_step = v > 0ll ? static_cast<ulong>(v) : 0u;
                    }
                }
                else if (key == "run_nstep") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) {
                        const long long v = atoll(raw.c_str());
                        settings.run_nstep_override = v > 0ll ? static_cast<ulong>(v) : 0ull;
                    }
                }
                else if (key == "purge_avg") {
                    if (!unquote(val).empty()) {
                        const int v = atoi(val.c_str());
                        settings.purge_avg_steps = v > 0 ? static_cast<uint>(v) : 0u;
                    }
                }
                else if (key == "purge_avg_stride") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) {
                        const int v = atoi(raw.c_str());
                        if (v > 0) {
                            settings.purge_avg_stride = static_cast<uint>(v);
                        }
                        else {
                            settings.purge_avg_stride = 1u;
                            println("| WARNING: purge_avg_stride must be > 0. Fallback to 1 (every step).            |");
                        }
                    }
                }
                else if (key == "output_tke_ti_tls") {
                    const std::string list_text = trim(unquote(val));
                    if (list_text.empty()) {
                    }
                    else {
                        const size_t lb = list_text.find('[');
                        const size_t rb = list_text.find(']', lb);
                        if (lb == std::string::npos || rb == std::string::npos || rb <= lb) {
                            println("| WARNING: output_tke_ti_tls expects [tke,ti,tls]. Keep default output set.    |");
                        }
                        else {
                            settings.output_tke_in_avg_vtk = false;
                            settings.output_ti_in_avg_vtk = false;
                            settings.output_tls_in_avg_vtk = false;
                            const std::string inside = list_text.substr(lb + 1u, rb - lb - 1u);
                            std::stringstream ss(inside);
                            std::string token;
                            while (std::getline(ss, token, ',')) {
                                std::string item = trim(token);
                                std::transform(item.begin(), item.end(), item.begin(), ::tolower);
                                if (item.empty()) continue;
                                if (item == "tke") settings.output_tke_in_avg_vtk = true;
                                else if (item == "ti") settings.output_ti_in_avg_vtk = true;
                                else if (item == "tls") settings.output_tls_in_avg_vtk = true;
                                else println("| WARNING: output_tke_ti_tls token '" + item + "' is unknown and ignored.       |");
                            }
                        }
                    }
                }
                else if (key == "culbm_output_global_vtk" || key == "physical_output_global_vtk") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (!raw.empty() && parse_bool_setting(raw, parsed)) settings.culbm_output_global_vtk = parsed;
                    else if (!raw.empty()) println("| WARNING: culbm_output_global_vtk expects a bool-like value. Disabled.       |");
                }
                else if (key == "culbm_output_global_netcdf" || key == "physical_output_global_netcdf") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (!raw.empty() && parse_bool_setting(raw, parsed)) settings.culbm_output_global_netcdf = parsed;
                    else if (!raw.empty()) println("| WARNING: culbm_output_global_netcdf expects a bool-like value. Disabled.    |");
                }
                else if (key == "culbm_output_global_vtkhdf" || key == "physical_output_global_vtkhdf") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (!raw.empty() && parse_bool_setting(raw, parsed)) settings.culbm_output_global_vtkhdf = parsed;
                    else if (!raw.empty()) println("| WARNING: culbm_output_global_vtkhdf expects a bool-like value. Disabled.    |");
                }
                else if (key == "culbm_output_agl_vtk" || key == "physical_output_agl_vtk") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (!raw.empty() && parse_bool_setting(raw, parsed)) settings.culbm_output_agl_vtk = parsed;
                    else if (!raw.empty()) println("| WARNING: culbm_output_agl_vtk expects a bool-like value. Disabled.          |");
                }
                else if (key == "culbm_output_agl_netcdf" || key == "physical_output_agl_netcdf") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (!raw.empty() && parse_bool_setting(raw, parsed)) settings.culbm_output_agl_netcdf = parsed;
                    else if (!raw.empty()) println("| WARNING: culbm_output_agl_netcdf expects a bool-like value. Disabled.       |");
                }
                else if (key == "validation") settings.validation_status = unquote(val);
                else if (key == "probes") {
                    settings.probes_raw_text = trim(val);
                    settings.probes_defined = !settings.probes_raw_text.empty();
                }
                else if (key == "coriolis_term") {
                    std::string v = unquote(val);
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    bool parsed = false;
                    if (!v.empty() && parse_bool_setting(v, parsed)) settings.enable_coriolis = parsed;
                }
                else if (key == "turb_inflow_enable" || key == "vk_inlet_enable") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (raw.empty()) {
                    }
                    else if (!parse_bool_setting(raw, parsed)) {
                        println("| WARNING: turb_inflow_enable expects a bool-like value. Keep current/default.  |");
                    }
                    else {
                        settings.vk_inlet_settings.enable = parsed;
                    }
                }
                else if (key == "vk_inlet_ti") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.vk_inlet_settings.ti = (float)atof(raw.c_str());
                }
                else if (key == "vk_inlet_sigma") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.vk_inlet_settings.sigma_si = (float)atof(raw.c_str());
                }
                else if (key == "vk_inlet_l") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.vk_inlet_settings.L_si = (float)atof(raw.c_str());
                }
                else if (key == "vk_inlet_nmodes") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.vk_inlet_settings.nmodes = atoi(raw.c_str());
                }
                else if (key == "vk_inlet_seed") {
                    const std::string raw = unquote(val);
                    if (raw.empty()) continue;
                    char* endptr = nullptr;
                    const unsigned long long parsed = std::strtoull(raw.c_str(), &endptr, 10);
                    if (endptr == raw.c_str()) {
                        println("| WARNING: vk_inlet_seed parse failed. Keep previous/default seed.             |");
                    }
                    else {
                        settings.vk_inlet_settings.seed = (ulong)parsed;
                    }
                }
                else if (key == "vk_inlet_update_stride") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.vk_inlet_settings.update_stride = atoi(raw.c_str());
                }
                else if (key == "vk_inlet_uc_mode") {
                    std::string v = unquote(val);
                    if (v.empty()) continue;
                    std::transform(v.begin(), v.end(), v.begin(), ::toupper);
                    if (v == "NORM_MEAN") settings.vk_inlet_settings.uc_mode = VkInletUcMode::NORM_MEAN;
                    else if (v == "NORMAL_COMPONENT") settings.vk_inlet_settings.uc_mode = VkInletUcMode::NORMAL_COMPONENT;
                    else {
                        println("| WARNING: vk_inlet_uc_mode invalid. Use NORMAL_COMPONENT or NORM_MEAN.        |");
                    }
                }
                else if (key == "vk_inlet_same_realization_all_faces") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (raw.empty()) {
                    }
                    else if (!parse_bool_setting(raw, parsed)) {
                        println("| WARNING: vk_inlet_same_realization_all_faces expects a bool-like value.       |");
                    }
                    else {
                        settings.vk_inlet_settings.same_realization_all_faces = parsed;
                    }
                }
                else if (key == "vk_inlet_stride_interpolation") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (raw.empty()) {
                    }
                    else if (!parse_bool_setting(raw, parsed)) {
                        println("| WARNING: vk_inlet_stride_interpolation expects a bool-like value.             |");
                    }
                    else {
                        settings.vk_inlet_settings.stride_interpolation = parsed;
                    }
                }
                else if (key == "vk_inlet_inflow_only") {
                    bool parsed = false;
                    const std::string raw = unquote(val);
                    if (raw.empty()) {
                    }
                    else if (!parse_bool_setting(raw, parsed)) {
                        println("| WARNING: vk_inlet_inflow_only expects a bool-like value.                      |");
                    }
                    else {
                        settings.vk_inlet_settings.inflow_only = parsed;
                    }
                }
                else if (key == "vk_inlet_face_mode") {
                    std::string v = unquote(val);
                    if (v.empty()) continue;
                    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
                        return c == '-' ? '_' : (char)std::toupper(c);
                    });
                    if (v == "AUTO" || v == "AUTO_SIDES" || v == "BY_INFLOW_ONLY" ||
                        v == "BUSINESS_DEFAULT" || v == "DEFAULT") {
                        settings.vk_inlet_settings.face_mode = VkInletFaceMode::AUTO_SIDES;
                    } else if (v == "TARGET_INFLOW" || v == "INFLOW" || v == "TARGET" || v == "UPSTREAM_ONLY") {
                        settings.vk_inlet_settings.face_mode = VkInletFaceMode::TARGET_INFLOW;
                    } else if (v == "EXCLUDE_DOWNSTREAM" || v == "EXCEPT_DOWNSTREAM" ||
                               v == "ALL_EXCEPT_DOWNSTREAM" || v == "NON_DOWNSTREAM") {
                        settings.vk_inlet_settings.face_mode = VkInletFaceMode::EXCLUDE_DOWNSTREAM;
                    } else if (v == "EXCLUDE_DOWNSTREAM_SIDES" || v == "EXCEPT_DOWNSTREAM_SIDES" ||
                               v == "SIDE_EXCEPT_DOWNSTREAM" || v == "SIDES_EXCEPT_DOWNSTREAM" ||
                               v == "NON_DOWNSTREAM_SIDES" || v == "SIDE_FACES_EXCEPT_DOWNSTREAM") {
                        settings.vk_inlet_settings.face_mode = VkInletFaceMode::EXCLUDE_DOWNSTREAM_SIDES;
                    } else if (v == "ALL_SIDES" || v == "SIDE_FACES" ||
                               v == "ALL_SIDE_FACES" || v == "SIDES_ONLY" ||
                               v == "ALL_SIDES_NO_TOP") {
                        settings.vk_inlet_settings.face_mode = VkInletFaceMode::ALL_SIDES;
                    } else if (v == "ALL" || v == "ALL_SELECTED" || v == "ALL_FACES") {
                        settings.vk_inlet_settings.face_mode = VkInletFaceMode::ALL_SELECTED;
                    } else {
                        println("| WARNING: vk_inlet_face_mode invalid. Use AUTO_SIDES/ALL_SIDES/etc.          |");
                    }
                }
                else if (key == "vk_inlet_anisotropy" ||
                         key == "vk_inlet_anisotropy_scale" ||
                         key == "vk_inlet_aniso_scale") {
                    if (unquote(val).empty()) continue;
                    float ax = settings.vk_inlet_settings.anisotropy_scale.x;
                    float ay = settings.vk_inlet_settings.anisotropy_scale.y;
                    float az = settings.vk_inlet_settings.anisotropy_scale.z;
                    if (!parse_triplet_float(unquote(val), ax, ay, az)) {
                        println("| WARNING: vk_inlet_anisotropy expects [a,b,c]. Keep previous/default.         |");
                    } else {
                        settings.vk_inlet_settings.anisotropy_scale = float3(ax, ay, az);
                    }
                }

                else if (key == "cut_lon_manual") {
                    if (!unquote(val).empty()) {
                        parse_pair_float(val, settings.cut_lon_min_deg, settings.cut_lon_max_deg);
                        settings.has_cut_lon_manual = true;
                    }
                }
                else if (key == "cut_lat_manual") {
                    if (!unquote(val).empty()) {
                        parse_pair_float(val, settings.cut_lat_min_deg, settings.cut_lat_max_deg);
                        settings.has_cut_lat_manual = true;
                    }
                }
                else if (key == "utm_crs") {
                    const std::string raw = unquote(val);
                    if (!raw.empty()) settings.utm_crs_from_config = raw;
                }
                else if (key == "rotate_deg") {
                    const std::string raw = unquote(val);
                    if (raw.empty()) continue;
                    char* endptr = nullptr;
                    const double parsed = std::strtod(raw.c_str(), &endptr);
                    if (endptr != raw.c_str() && std::isfinite(parsed)) {
                        settings.rotate_deg_from_config = parsed;
                        settings.has_rotate_deg = true;
                    }
                }
                else if (key == "inflow") {
                    if (!unquote(val).empty()) parse_float_list(val, settings.inflow_list);
                }
                else if (key == "angle") {
                    if (!unquote(val).empty()) parse_float_list(val, settings.angle_list);
                }

            }
            if (settings.culbm_sgs_cs_explicit &&
                settings.culbm_sgs_cs_squared_explicit) {
                fail_culbm_setting(
                    "culbm_sgs_cs and culbm_sgs_cs_squared are mutually exclusive.");
            }
            // si_size.z += z_si_offset;
            if (!settings.memory) settings.memory = 6000u;
            if (settings.Dx == 0u) settings.Dx = 1u;
            if (settings.Dy == 0u) settings.Dy = 1u;
            if (settings.Dz == 0u) settings.Dz = 1u;

            if (settings.vk_inlet_settings.ti < 0.0f) {
                println("| WARNING: vk_inlet_ti is negative. It is clamped to 0.                       |");
                settings.vk_inlet_settings.ti = 0.0f;
            }
            if (settings.vk_inlet_settings.ti > 1.0f && settings.vk_inlet_settings.ti <= 100.0f) {
                println("| WARNING: vk_inlet_ti > 1 detected. Interpret as percent and divide by 100.  |");
                settings.vk_inlet_settings.ti *= 0.01f;
            }
            if (settings.vk_inlet_settings.sigma_si < 0.0f) {
                println("| WARNING: vk_inlet_sigma is negative. It is clamped to 0.                    |");
                settings.vk_inlet_settings.sigma_si = 0.0f;
            }
            if (settings.vk_inlet_settings.L_si < 0.0f) {
                println("| WARNING: vk_inlet_l is negative. It is clamped to 0.                        |");
                settings.vk_inlet_settings.L_si = 0.0f;
            }
            if (settings.vk_inlet_settings.nmodes <= 0) {
                println("| WARNING: vk_inlet_nmodes <= 0. Fallback to default " + to_string((ulong)k_vk_nmodes_default) + ".                     |");
                settings.vk_inlet_settings.nmodes = k_vk_nmodes_default;
            }
            if (settings.vk_inlet_settings.nmodes > k_vk_nmodes_max) {
                println("| WARNING: vk_inlet_nmodes is too large. Clamped to " + to_string((ulong)k_vk_nmodes_max) + ".          |");
                settings.vk_inlet_settings.nmodes = k_vk_nmodes_max;
            }
            if (settings.vk_inlet_settings.update_stride <= 0) {
                println("| WARNING: vk_inlet_update_stride <= 0. Fallback to 1.                         |");
                settings.vk_inlet_settings.update_stride = 1;
            }
            auto sanitize_vk_aniso_component = [&](float& v, const char* axis_name) {
                if (!std::isfinite(v)) {
                    println("| WARNING: vk_inlet_anisotropy " + string(axis_name) + " is non-finite. Reset to 1.0.           |");
                    v = 1.0f;
                } else if (v < 0.0f) {
                    println("| WARNING: vk_inlet_anisotropy " + string(axis_name) + " is negative. Reset to 1.0.             |");
                    v = 1.0f;
                }
            };
            sanitize_vk_aniso_component(settings.vk_inlet_settings.anisotropy_scale.x, "x");
            sanitize_vk_aniso_component(settings.vk_inlet_settings.anisotropy_scale.y, "y");
            sanitize_vk_aniso_component(settings.vk_inlet_settings.anisotropy_scale.z, "z");
            if (settings.vk_inlet_settings.enable && !(settings.vk_inlet_settings.L_si > 0.0f)) {
                println("| WARNING: turb_inflow_enable=true but L is invalid. VK inlet disabled.         |");
                settings.vk_inlet_settings.enable = false;
            }
            if (settings.vk_inlet_settings.enable && !(settings.vk_inlet_settings.ti > 0.0f || settings.vk_inlet_settings.sigma_si > 0.0f)) {
                println("| WARNING: turb_inflow_enable=true but TI/sigma is invalid. VK inlet disabled.  |");
                settings.vk_inlet_settings.enable = false;
            }

            apply_lattice_selection(settings);
            // mesh_control
            auto is_nonempty = [&](const std::string& s) {
                return !trim(s).empty();
                };

            settings.parent =
                std::filesystem::path(settings.conf_used_path).parent_path().string();
            bool applied = false;
            if (is_nonempty(mesh_control_val) && mesh_control_val == "gpu_memory") {
                if (is_nonempty(gpu_memory_val)) {
                    uint mm = static_cast<uint>(atoi(trim(gpu_memory_val).c_str()));
                    if (mm > 0u) {
                        settings.preloaded_geometry_path =
                            select_culbm_geometry_candidate(
                                std::filesystem::path(settings.parent),
                                settings.caseName,
                                settings.profile_generation);
                        if (settings.preloaded_geometry_path.empty()) {
                            println("| ERROR: gpu_memory fitting requires the STL geometry to exist.               |");
                            wait();
                            exit(-1);
                        }
                        settings.preloaded_geometry_mesh.reset(
                            read_stl(settings.preloaded_geometry_path.string()));
                        if (!settings.preloaded_geometry_mesh) {
                            println("| ERROR: gpu_memory fitting could not read the selected STL geometry.          |");
                            wait();
                            exit(-1);
                        }
                        const std::uint64_t fit_voxelization_mb =
                            voxelization_transient_vram_mb_per_device(
                                settings.preloaded_geometry_mesh->triangle_number);
                        settings.memory = mm;
                        settings.cell_m = fit_cell_size_to_gpu_memory_request(
                            settings, settings.memory, settings.cuda_vertical_nesting_agl_si,
                            &settings.particle_advection_settings,
                            fit_voxelization_mb);
                        settings.gpu_memory_mesh_control_requested = true;
                        applied = true;
                    }
                }
            }

            else if (is_nonempty(mesh_control_val) && mesh_control_val == "cell_size") {
                if (is_nonempty(cell_size_val)) {
                    float cs = static_cast<float>(atof(trim(cell_size_val).c_str()));
                    if (cs > 0.0f && std::isfinite(cs)) { settings.cell_m = cs; applied = true; }
                }
            }

            if (!applied) {
                settings.cell_m = 20.0f;
            }

        }
    }

    if (!settings.temperature_lattice_applied) {
        if (!set_momentum_lattice(settings.momentum_lattice_setting)) {
            println("| ERROR: culbm_velocity_set cannot change after a solver instance exists.      |");
            wait();
            exit(-1);
        }
        if (!set_ddf_storage(settings.ddf_storage_setting)) {
            println("| ERROR: culbm_ddf_storage cannot change after a solver instance exists.       |");
            wait();
            exit(-1);
        }
    }
    if (!settings.temperature_lattice_applied &&
        !set_temperature_lattice(settings.temperature_lattice_setting)) {
        println("| ERROR: requested temperature lattice is unsupported by this momentum build. |");
        wait();
        exit(-1);
    }
}

void finish_profile_settings(CaseSettings& deck) {
    if (deck.profile_generation && !deck.datetime_from_config) {
        println("| Profile mode    | datetime not set in *.luwpf, use fallback " + deck.datetime + " |");
    }
    if (deck.profile_generation && deck.enable_coriolis && !(deck.has_cut_lon_manual && deck.has_cut_lat_manual)) {
        println("| WARNING: coriolis_term=true but cut_lon_manual/cut_lat_manual is missing in *.luwpf. |");
        println("| WARNING: Coriolis is auto-disabled for Profile mode.                         |");
        deck.enable_coriolis = false;
    }
}

void confirm_validation(const CaseSettings& deck) {
    // ---------------------- Validation Check -----------------------------------
    if (!deck.dataset_generation && !deck.profile_generation) {
        std::string v_check = deck.validation_status;
        std::transform(v_check.begin(), v_check.end(), v_check.begin(), ::tolower);

        if (v_check != "pass" && v_check != "true" && v_check != "1") {
            println("|-----------------------------------------------------------------------------|");
            println("| WARNING: Validation status is '" + deck.validation_status + "'. Pre-processing may be incomplete or invalid. |");
            println("| Do you wish to continue with the simulation? (Press ENTER or type 'y'/'yes'):  |");
            std::string user_choice;
            std::getline(std::cin, user_choice);

            // trim (lambda from file-reading scope must be redefined here)
            auto trim = [](std::string s) {
                const char* ws = " \t";
                size_t b = s.find_first_not_of(ws);
                size_t e = s.find_last_not_of(ws);
                return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
                };
            user_choice = trim(user_choice);
            std::transform(user_choice.begin(), user_choice.end(), user_choice.begin(), ::tolower);

            if (user_choice != "" && user_choice != "y" && user_choice != "yes") {
                println("| Aborting simulation at user's request.                                      |");
                println("|-----------------------------------------------------------------------------|");
                wait();
                exit(0); // Graceful exit
            }
            println("| User confirmed to proceed despite validation warning.                       |");
        }
    }
}

} // namespace culbm::cases
