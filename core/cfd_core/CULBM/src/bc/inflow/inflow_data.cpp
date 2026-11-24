#include "inflow_data.hpp"
#include "../../case/context.hpp"
#include "../../case/report.hpp"

namespace culbm::bc {

using namespace culbm::cases;

void load_inflow(CaseContext& ctx) {
    CaseSettings& deck = ctx.deck;
    InflowData& inflow = ctx.inflow;
    ThermalReference& thermal = ctx.thermal;
    CasePhysics& physics = ctx.physics;
    inflow.csv_has_temperature = false;
    inflow.csv_rows_with_temperature = 0ul;
    inflow.csv_temp_min_si = k_temperature_ref_kelvin;
    inflow.csv_temp_max_si = k_temperature_ref_kelvin;
    inflow.csv_has_patch = false;
    inflow.csv_rows_with_patch = 0ul;
    inflow.csv_samples_valid = true;
    thermal.temperature_ref_kelvin = deck.boussinesq_reference_auto
        ? k_temperature_ref_kelvin
        : deck.boussinesq_reference_setting_kelvin;
    thermal.temperature_scale_kelvin = k_temperature_ref_kelvin;
    thermal.temperature_ref_adaptive = false;
    thermal.temperature_scale_adaptive = false;
    thermal.boussinesq_temperature_range_checked = false;
    thermal.boussinesq_max_relative_temperature_difference = 0.0f;
    if (!deck.dataset_generation && !deck.profile_generation) {
        const std::string csv_path_ref = deck.parent + "/proj_temp/SurfData_" + deck.datetime + ".csv";
        inflow.samples_si = read_samples(
            csv_path_ref,
            &inflow.csv_has_temperature,
            &inflow.csv_rows_with_temperature,
            &inflow.csv_temp_min_si,
            &inflow.csv_temp_max_si,
            &inflow.csv_has_patch,
            &inflow.csv_rows_with_patch,
            &inflow.csv_samples_valid
        );
        if (!inflow.csv_samples_valid) {
            println("| ERROR: invalid boundary temperature input. Aborting before GPU allocation.  |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        if (inflow.samples_si.empty()) {
            println("| ERROR: no inlet samples when computing si_ref_u. Aborting...                |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        float max_u = 0.0f;
        for (const auto& s : inflow.samples_si) {
            const float speed = std::sqrt(
                s.u.x * s.u.x +
                s.u.y * s.u.y +
                s.u.z * s.u.z
            );
            if (speed > max_u) {
                max_u = speed;
            }
        }
        if (inflow.csv_has_temperature && inflow.csv_rows_with_temperature > 0ul) {
            float tmin = inflow.csv_temp_min_si;
            float tmax = inflow.csv_temp_max_si;
            if (tmin > tmax) std::swap(tmin, tmax);
            if (std::isfinite(tmin) && std::isfinite(tmax) && tmax > 0.0f) {
                if (deck.boussinesq_reference_auto) {
                    const float tref = 0.5f * (tmin + tmax);
                    if (std::isfinite(tref) && tref > 0.0f) {
                        thermal.temperature_ref_kelvin = tref;
                        thermal.temperature_ref_adaptive = true;
                    }
                }
                const float thalf = 0.5f * (tmax - tmin);
                if (std::isfinite(thalf) && thalf > 1.0e-6f) {
                    thermal.temperature_scale_kelvin = thalf;
                    thermal.temperature_scale_adaptive = true;
                }
                else {
                    thermal.temperature_scale_kelvin = 1.0f; // keep affine map well-conditioned even for nearly uniform input temperature
                    thermal.temperature_scale_adaptive = true;
                }
                if (std::isfinite(thermal.temperature_ref_kelvin) &&
                    thermal.temperature_ref_kelvin > 0.0f) {
                    thermal.boussinesq_max_relative_temperature_difference =
                        std::max(std::fabs(tmin - thermal.temperature_ref_kelvin),
                                 std::fabs(tmax - thermal.temperature_ref_kelvin)) /
                        thermal.temperature_ref_kelvin;
                    thermal.boussinesq_temperature_range_checked =
                        std::isfinite(thermal.boussinesq_max_relative_temperature_difference);
                }
            }
        }
        physics.si_ref_u = max_u;
    }
    else if (deck.dataset_generation) {
        if (deck.inflow_list.empty()) {
            println("| ERROR: dataset generation requires inflow list (inflow=[...]).              |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        physics.si_ref_u = *std::max_element(deck.inflow_list.begin(), deck.inflow_list.end());
    }
    else { // profile_generation
        const float profile_domain_agl_si = deck.si_size.z - deck.z_si_offset;
        if (profile_domain_agl_si <= 0.0f) {
            println("| ERROR: invalid profile domain height. Check si_z_cfd/base_height.           |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        if (deck.z_limit_override > 0.0f) {
            println("| Profile z_limit | terrain-clearance cap only; profile interpolation uses DEM |");
        }

        const std::string profile_path = deck.parent + "/wind_bc/profile.dat";
        auto profile_samples = read_profile_dat(profile_path);
        if (profile_samples.empty()) {
            println("| ERROR: no profile samples found. Aborting...                                |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }

        std::sort(profile_samples.begin(), profile_samples.end(),
                  [](const ProfileSample& a, const ProfileSample& b) { return a.z < b.z; });

        std::vector<float> z_vals;
        std::vector<float> u_vals;
        z_vals.reserve(profile_samples.size());
        u_vals.reserve(profile_samples.size());
        for (const auto& s : profile_samples) {
            if (!std::isfinite(s.z) || !std::isfinite(s.u)) continue;
            if (!z_vals.empty() && std::fabs(s.z - z_vals.back()) < 1e-6f) {
                u_vals.back() = s.u; // keep last value for duplicate z
                continue;
            }
            z_vals.push_back(s.z);
            u_vals.push_back(s.u);
        }
        if (z_vals.size() < 2u) {
            println("| ERROR: profile.dat needs at least two valid samples. Aborting...            |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }

        if (profile_domain_agl_si > 1.0f && z_vals.back() <= 1.5f) {
            for (float& z : z_vals) {
                z *= profile_domain_agl_si;
            }
            println("| Profile z unit  | normalized -> scaled by domain AGL height                 |");
        }

        float max_u = 0.0f;
        for (const float v : u_vals) {
            if (v > max_u) max_u = v;
        }
        if (max_u <= 0.0f) {
            println("| ERROR: profile.dat has non-positive max U. Aborting...                      |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        physics.si_ref_u = max_u;
        inflow.profile_z_samples_si = z_vals;
        inflow.profile_u_samples_si = u_vals;

        println("| Profile samples | " + alignr(57u, to_string(z_vals.size())) + " |");
        println("| Profile z range | " + alignr(24u, to_string(fmtf(z_vals.front()))) +
                " to " + alignl(16u, to_string(fmtf(z_vals.back()))) + " m |");
        println("| Profile domain  | " + alignr(57u, to_string(fmtf(profile_domain_agl_si))) + " m AGL |");
    }
}

void convert_samples_to_lattice(CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    const CaseDomain& domain = ctx.domain;
    InflowData& inflow = ctx.inflow;
    const CasePhysics& physics = ctx.physics;
    inflow.use_temperature_bc = false;
    inflow.T_bc_min_lbm = 1.0f;
    inflow.T_bc_max_lbm = 1.0f;
    if (!deck.dataset_generation && !deck.profile_generation) {
        if (inflow.samples_si.empty()) { println("ERROR: no inlet samples. Aborting."); wait(); exit(-1); }
        inflow.use_temperature_bc = inflow.csv_has_temperature;

        if (inflow.csv_has_temperature) {
            println("| T column        | detected (" + to_string(inflow.csv_rows_with_temperature) + " rows)                               |");
            println("| CSV T range SI  | " + alignr(24u, to_string(fmtf(inflow.csv_temp_min_si))) +
                    " to " + alignl(16u, to_string(fmtf(inflow.csv_temp_max_si))) + " K |");
            if (!deck.buoyancy_enabled) {
                println("| Temperature BC  | passive transport enabled; buoyancy feedback disabled           |");
            }
            else {
                println("| Temperature BC  | enabled from CSV T (Kelvin -> nondimensionalized)               |");
            }
        }
        else {
            println("| T column        | not found, keep legacy velocity-only boundary behavior           |");
        }

        // convert samples to LBM units
        inflow.samples.reserve(inflow.samples_si.size());
        ulong out_of_temp_range = 0ul;
        for (const auto& s : inflow.samples_si) {
            SamplePoint sp;
            sp.p = float3(units.x(s.p.x), units.x(s.p.y), units.x(s.p.z-domain.cuda_ground_trim_si)) + float3(0.5f);
            sp.u = s.u * physics.u_scale;
            sp.patch = s.patch;
            if (inflow.use_temperature_bc) {
                sp.T = units.T(s.T); // Kelvin -> nondimensional temperature using selected T_ref and input-range scale
                if (s.T < k_temperature_min_kelvin || s.T > k_temperature_max_kelvin) out_of_temp_range++;
            }
            else {
                sp.T = 1.0f;
            }
            inflow.samples.push_back(sp);
        }
        if (inflow.use_temperature_bc && out_of_temp_range > 0ul) {
            println("| WARNING: " + to_string(out_of_temp_range) + " temperature samples are outside [-50C, 70C].                |");
        }
        if (inflow.use_temperature_bc) {
            inflow.T_bc_min_lbm = units.T(inflow.csv_temp_min_si);
            inflow.T_bc_max_lbm = units.T(inflow.csv_temp_max_si);
            if (inflow.T_bc_min_lbm > inflow.T_bc_max_lbm) std::swap(inflow.T_bc_min_lbm, inflow.T_bc_max_lbm);
        }
    }
}

float3 wind_velocity_lbmu(const float inflow_si, const float angle_deg, const float u_scale) {
    const float deg2rad = 3.14159265358979323846f / 180.0f;
    const float angle_rad = angle_deg * deg2rad;
    const float speed_lbmu = inflow_si * u_scale;
    return float3(-sinf(angle_rad) * speed_lbmu, -cosf(angle_rad) * speed_lbmu, 0.0f);
}

} // namespace culbm::bc
