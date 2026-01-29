#include "setup.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cfloat> 
#include <chrono>   
#include <ctime>   
#include <iomanip> 
#include <thread>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <cmath>

#include "interpolation.hpp"
#include "interpolation_hd.hpp"
#include "fluxcorrection.hpp"
extern float coriolis_f_lbmu;

// ────────────── Global configuration ───────────────
std::string caseName = "example";
std::string datetime = "20990101120000";
float        z_si_offset = 50.0f;
std::string downstream_bc = "+y";
std::string downstream_bc_yaw = "INTERNAL ERROR";
uint         memory = 20000u;
float cell_m = 20.0f;
float3       si_size = float3(0);
uint Dx = 1u, Dy = 1u, Dz = 1u;
std::string conf_used_path = "Integrated defaults (*.luw not found)";
bool use_high_order = false;
bool flux_correction = false;
uint research_output_steps = 0u; 
std::string validation_status = "fail"; 
bool enable_coriolis = false;       

struct SamplePoint { float3 p; float3 u; };
struct ProfileSample { float z; float u; };

static std::string trim_copy(std::string s) {
    const char* ws = " \t\r\n";
    size_t b = s.find_first_not_of(ws);
    size_t e = s.find_last_not_of(ws);
    return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
}

// helper: read profile.dat (columns z,U in SI units, with header)
static std::vector<ProfileSample> read_profile_dat(const string& path) {
    std::vector<ProfileSample> out;
    std::ifstream fin(path);
    if (!fin.is_open()) {
        println("ERROR: could not open profile file " + path);
        return out;
    }
    std::string line;
    ulong line_no = 0ul;
    while (std::getline(fin, line)) {
        line_no++;
        size_t cmt = line.find("//"); if (cmt != std::string::npos) line.erase(cmt);
        cmt = line.find('#'); if (cmt != std::string::npos) line.erase(cmt);
        line = trim_copy(line);
        if (line.empty()) continue;
        for (char& ch : line) { if (ch == ',' || ch == ';') ch = ' '; }
        std::stringstream ss(line);
        float z = 0.0f, u = 0.0f;
        if (!(ss >> z >> u)) {
            continue; // likely header or malformed line
        }
        if (!std::isfinite(z) || !std::isfinite(u)) {
            println("WARNING: invalid z/U in profile at line " + to_string(line_no));
            continue;
        }
        out.push_back({ z, u });
    }
    return out;
}

static float interpolate_profile_cubic(const std::vector<float>& z,
                                       const std::vector<float>& u,
                                       const float zq) {
    const size_t n = z.size();
    if (n == 0u) return 0.0f;
    if (n == 1u) return u[0];
    if (zq <= z.front()) return u.front();
    if (zq >= z.back()) return u.back();

    auto it = std::upper_bound(z.begin(), z.end(), zq);
    size_t i1 = (it == z.begin()) ? 0u : static_cast<size_t>(it - z.begin() - 1u);
    size_t i2 = std::min(i1 + 1u, n - 1u);
    const float z0 = z[i1];
    const float z1 = z[i2];
    const float denom = z1 - z0;
    if (denom <= 0.0f) return u[i1];
    const float t = (zq - z0) / denom;

    auto slope_at = [&](size_t i) -> float {
        if (n < 2u) return 0.0f;
        if (i == 0u) {
            const float dz = z[1] - z[0];
            return (dz != 0.0f) ? (u[1] - u[0]) / dz : 0.0f;
        }
        if (i + 1u >= n) {
            const float dz = z[n - 1u] - z[n - 2u];
            return (dz != 0.0f) ? (u[n - 1u] - u[n - 2u]) / dz : 0.0f;
        }
        const float dz = z[i + 1u] - z[i - 1u];
        return (dz != 0.0f) ? (u[i + 1u] - u[i - 1u]) / dz : 0.0f;
    };

    const float m0 = slope_at(i1);
    const float m1 = slope_at(i2);
    const float va = m0 * denom;
    const float vb = m1 * denom;
    return hermite_spline(u[i1], u[i2], va, vb, t);
}

// for Coriolis source term
float coriolis_Omegax_lbmu = 0.0f;
float coriolis_Omegay_lbmu = 0.0f;
float coriolis_Omegaz_lbmu = 0.0f;
float cut_lon_min_deg = 0.0f; // longitude/latitude given in *.luw, degrees
float cut_lon_max_deg = 0.0f;
float cut_lat_min_deg = 0.0f;
float cut_lat_max_deg = 0.0f;


// helper: read SurfData.csv (columns X,Y,Z,u,v,w in SI units, with header)
static std::vector<SamplePoint> read_samples(const string& csv_path) {
    std::vector<SamplePoint> out;
    std::ifstream fin(csv_path);
    if (!fin.is_open()) {
        println("ERROR: could not open CSV " + csv_path);
        return out;
    }
    string line; std::getline(fin, line); // skip header
    ulong line_no = 1ul;
    while (std::getline(fin, line)) {
        line_no++;
        std::stringstream ss(line);
        string token; float vals[6]; int i = 0;
        while (std::getline(ss, token, ',')) { vals[i++] = (float)atof(token.c_str()); }
        if (i != 6) {
            println("WARNING: malformed line " + to_string(line_no) + " in CSV");
            continue;
        }
        SamplePoint sp; sp.p = float3(vals[0], vals[1], vals[2]); sp.u = float3(vals[3], vals[4], vals[5]);
        out.push_back(sp);
    }
    return out;
}

// helper: return current local time in "YYYY-MM-DD hh:mm:ss" format
 static string now_str() {
     using namespace std::chrono;
     auto now = system_clock::now();
     std::time_t tt = system_clock::to_time_t(now);
     std::tm tm{};
    #if defined(_WIN32)
         if (localtime_s(&tm, &tt) != 0) {
             throw std::runtime_error("Failed to get local time.");
         }
    #else
         if (localtime_r(&tt, &tm) == nullptr) {
             throw std::runtime_error("Failed to get local time.");
         }
    #endif

     std::stringstream ss;
     ss << std::put_time(&tm, "%F %T"); // %F = YYYY-MM-DD, %T = hh:mm:ss
     return ss.str();
 }

static float3 wind_velocity_lbmu(const float inflow_si, const float angle_deg, const float u_scale) {
    const float deg2rad = 3.14159265358979323846f / 180.0f;
    const float angle_rad = angle_deg * deg2rad;
    const float speed_lbmu = inflow_si * u_scale;
    return float3(-sinf(angle_rad) * speed_lbmu, -cosf(angle_rad) * speed_lbmu, 0.0f);
}

void main_setup() {
  //println("|-----------------------------------------------------------------------------|");
    println("|                                                                             |");
    println("|      LatticeUrbanWind LUW: Towards Micrometeorology Fastest Simulation      |");
    println("|                                                                             |");
    println("|                                        Developed by Huanxia Wei's Team      |");
    println("|                                        Version - v3.5-251119                |");
    println("|                                                                             |");
    println("|-----------------------------------------------------------------------------|");
 
    bool dataset_generation = false;
    bool profile_generation = false;
    std::vector<float> inflow_list;
    std::vector<float> angle_list;
    std::vector<float> profile_u_si;
    float z_limit_override = 0.0f;
    float z_limit_si = 0.0f;
    const float profile_dz_si = 0.1f;

    // ---------------------- read *.luw / *.luwdg ------------------------------
    std::string parent; // project directory that contains the configure file
    {
        // prompt for configure file path when not directly provided
        println("| Please input configure file path (*.luw, *.luwdg, *.luwpf):                  |");
        std::string luw_path_input;
        std::getline(std::cin, luw_path_input);

        // trim
        auto trim = [](std::string s) {
            const char* ws = " \t";
            size_t b = s.find_first_not_of(ws);
            size_t e = s.find_last_not_of(ws);
            return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
            };
        luw_path_input = trim(luw_path_input);
        
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
                dataset_generation = true;
                println("| Dataset generation mode enabled (*.luwdg).                                  |");
            }
            else if (ext == ".luwpf") {
                profile_generation = true;
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
            if (fin.is_open()) conf_used_path = luw_path_fs.string();
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
            std::string line;
            std::string mesh_control_val, gpu_memory_val, cell_size_val;
            while (std::getline(fin, line)) {
                size_t cmt = line.find("//"); if (cmt != std::string::npos) line.erase(cmt);
                size_t eq = line.find('=');  if (eq == std::string::npos) continue;

                std::string key = trim(line.substr(0, eq));
                std::string val = trim(line.substr(eq + 1));
                std::transform(key.begin(), key.end(), key.begin(), ::tolower);

                auto unquote = [&](std::string s) {
                    s = trim(s);
                    if (!s.empty()) {
                        char q = s.front();
                        if ((q == '"' || q == '\'') && s.size() >= 2 && s.back() == q) {
                            s = trim(s.substr(1, s.size() - 2));
                        }
                    }
                    return s;
                    };

                if (key == "casename")        caseName = unquote(val);
                else if (key == "datetime")   datetime = unquote(val);
                else if (key == "downstream_bc") downstream_bc = unquote(val);
                else if (key == "high_order") {
                    std::string v = unquote(val);
                    if (v == "true" || v == "1") use_high_order = true;
                }
                else if (key == "flux_correction") {
                    std::string v = unquote(val);
                    if (v == "true" || v == "1") flux_correction = true;
                }
                else if (key == "downstream_bc_yaw") downstream_bc_yaw = unquote(val);
                else if (key == "base_height") z_si_offset = (float)atof(val.c_str());
                else if (key == "z_limit")     z_limit_override = (float)atof(val.c_str());
                else if (key == "memory_lbm")  memory = (uint)atoi(val.c_str());
                else if (key == "si_x_cfd")    si_size.x = second_val(val);
                else if (key == "si_y_cfd")    si_size.y = second_val(val);
                else if (key == "si_z_cfd")    si_size.z = second_val(val);
                else if (key == "mesh_control") mesh_control_val = unquote(val);
                else if (key == "gpu_memory")   gpu_memory_val = unquote(val);
                else if (key == "cell_size")    cell_size_val = unquote(val);
                else if (key == "n_gpu")        parse_triplet_uint(val, Dx, Dy, Dz);
                else if (key == "research_output") research_output_steps = (uint)atoi(val.c_str());
                else if (key == "validation") validation_status = unquote(val);
                else if (key == "coriolis_term") {
                    std::string v = unquote(val);
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    if (v == "true" || v == "1") enable_coriolis = true;
                }

                else if (key == "cut_lon_manual") parse_pair_float(val, cut_lon_min_deg, cut_lon_max_deg);
                else if (key == "cut_lat_manual") parse_pair_float(val, cut_lat_min_deg, cut_lat_max_deg);
                else if (key == "inflow") parse_float_list(val, inflow_list);
                else if (key == "angle") parse_float_list(val, angle_list);


            }
            // si_size.z += z_si_offset;
            if (!memory) memory = 6000u;
            if (Dx == 0u) Dx = 1u;
            if (Dy == 0u) Dy = 1u;
            if (Dz == 0u) Dz = 1u;

            // mesh_control
            auto is_nonempty = [&](const std::string& s) {
                return !trim(s).empty();
                };

            bool applied = false;
            if (is_nonempty(mesh_control_val) && mesh_control_val == "gpu_memory") {
                if (is_nonempty(gpu_memory_val)) {
                    uint mm = static_cast<uint>(atoi(trim(gpu_memory_val).c_str()));
                    if (mm > 0u) {
                        memory = mm;
                        const uint3 Nfit = resolution(si_size, memory); 
                        cell_m = si_size.x / std::max(1u, Nfit.x);
                        applied = true;
                    }
                }
            }

            else if (is_nonempty(mesh_control_val) && mesh_control_val == "cell_size") {
                if (is_nonempty(cell_size_val)) {
                    float cs = static_cast<float>(atof(trim(cell_size_val).c_str()));
                    if (cs > 0.0f && std::isfinite(cs)) { cell_m = cs; applied = true; }
                }
            }

            if (!applied) {
                cell_m = 20.0f;
            }
            parent = std::filesystem::path(conf_used_path).parent_path().string();

        }
    }

    // ---------------------- Validation Check -----------------------------------
    if (!dataset_generation && !profile_generation) {
        std::string v_check = validation_status;
        std::transform(v_check.begin(), v_check.end(), v_check.begin(), ::tolower);

        if (v_check != "pass" && v_check != "true" && v_check != "1") {
            println("|-----------------------------------------------------------------------------|");
            println("| WARNING: Validation status is '" + validation_status + "'. Pre-processing may be incomplete or invalid. |");
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


    auto fmtf = [](float v, int prec = 4) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(prec) << v;
        return os.str();
    };

    println("|                                                                             |");
    println("|-----------------------------------------------------------------------------|");
    println("|                            PARAMETER INFORMATION                            |");
    println("|-----------------------------------------------------------------------------|");
 // println("| Grid Domains    | " + alignr(57u,to_string()) + " |");
    println("| Configure deck  | " + alignr(57u, to_string(conf_used_path)) + " |");
    println("| Casename / Time | " + alignr(40u, to_string(caseName)) + alignr(17u, to_string(datetime)) + " |");
    println("| Basement Height | " + alignr(55u, to_string(fmtf(z_si_offset))) + " m |");
    println("| SI Size (m)     | " + alignr(12u, " X:") + alignl(11u, to_string(fmtf(si_size.x))) + "   Y: " + alignl(11u, to_string(fmtf(si_size.y))) + "   Z: " + alignl(11u, to_string(fmtf(si_size.z))) + " | ");
    println("| Downstream BC   | " + alignr(57u, to_string(downstream_bc)) + " |");
    println("| Normal Yaw      | " + alignr(53u, to_string(downstream_bc_yaw)) + " deg |");
    println("| GPU Decompose   | " + alignr(49u, to_string(Dx)) + ", " + alignr(2u, to_string(Dy)) + ", " + alignr(2u, to_string(Dz)) + " |");
    println("| VRAM Request    | " + alignr(54u, to_string(memory)) + " MB |");

    const float lbm_ref_u = 0.10f; // 0.1731 is Ma=0.3
    float si_ref_u = 10.0f;
    const float si_nu = 1.48E-5f, si_rho = 1.225f;

    //const uint3 lbm_N = resolution(si_size, memory);
    const uint Nx_cells = std::max(1, (int)(si_size.x / cell_m + 0.5f));
    const uint Ny_cells = std::max(1, (int)(si_size.y / cell_m + 0.5f));
    #ifndef D2Q9
    const uint Nz_cells = std::max(1, (int)(si_size.z / cell_m + 0.5f));
    #else
    const uint Nz_cells = 1u;   
    #endif
    const uint3 lbm_N = uint3(Nx_cells, Ny_cells, Nz_cells);

    println("|-----------------------------------------------------------------------------|");
    println("|                          DOMAIN AND TRANFORMATION                           |");
    println("|-----------------------------------------------------------------------------|");
    println("| Grid Resolution | " + alignr(45u, to_string(lbm_N.x)) + "," + alignr(5u, to_string(lbm_N.y)) + "," + alignr(5u, to_string(lbm_N.z)) + " |");

    const uint mem_per_dev_mb = vram_required_mb_per_device(lbm_N.x, lbm_N.y, lbm_N.z, Dx, Dy, Dz);
    const uint mem_total_mb = vram_required_mb_total(lbm_N.x, lbm_N.y, lbm_N.z, Dx, Dy, Dz);
    if (!dataset_generation && !profile_generation) {
        const std::string csv_path_ref = parent + "/proj_temp/SurfData_" + datetime + ".csv";
        auto samples_ref = read_samples(csv_path_ref);
        if (samples_ref.empty()) {
            println("| ERROR: no inlet samples when computing si_ref_u. Aborting...                |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        float max_u = 0.0f;
        for (const auto& s : samples_ref) {
            const float speed = std::sqrt(
                s.u.x * s.u.x +
                s.u.y * s.u.y +
                s.u.z * s.u.z
            );
            if (speed > max_u) {
                max_u = speed;
            }
        }
        si_ref_u = max_u;
    }
    else if (dataset_generation) {
        if (inflow_list.empty()) {
            println("| ERROR: dataset generation requires inflow list (inflow=[...]).              |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        si_ref_u = *std::max_element(inflow_list.begin(), inflow_list.end());
    }
    else { // profile_generation
        z_limit_si = si_size.z - z_si_offset;
        if (z_limit_override > 0.0f) {
            if (std::fabs(z_limit_override - z_limit_si) > 1e-3f) {
                println("| WARNING: z_limit differs from si_z_cfd - base_height. Using z_limit value. |");
            }
            z_limit_si = z_limit_override;
        }
        if (z_limit_si <= 0.0f) {
            println("| ERROR: invalid z_limit for profile mode. Check si_z_cfd/base_height.        |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }

        const std::string profile_path = parent + "/wind_bc/profile.dat";
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
        si_ref_u = max_u;

        const uint steps = static_cast<uint>(std::ceil(z_limit_si / profile_dz_si));
        profile_u_si.resize(steps + 1u);
        for (uint i = 0u; i <= steps; ++i) {
            const float zq = std::min(z_limit_si, i * profile_dz_si);
            float u_val = interpolate_profile_cubic(z_vals, u_vals, zq);
            if (u_val < 0.0f) u_val = 0.0f;
            profile_u_si[i] = u_val;
        }

        float u_min_prof = profile_u_si.front();
        float u_max_prof = profile_u_si.front();
        for (const float v : profile_u_si) {
            if (v < u_min_prof) u_min_prof = v;
            if (v > u_max_prof) u_max_prof = v;
        }

        println("| Profile samples | " + alignr(57u, to_string(z_vals.size())) + " |");
        println("| Profile z range | " + alignr(24u, to_string(fmtf(z_vals.front()))) +
                " to " + alignl(16u, to_string(fmtf(z_vals.back()))) + " m |");
        println("| Profile z_limit | " + alignr(57u, to_string(fmtf(z_limit_si))) + " m |");
        println("| Profile U range | " + alignr(24u, to_string(fmtf(u_min_prof))) +
                " to " + alignl(16u, to_string(fmtf(u_max_prof))) + " m/s |");
    }

    units.set_m_kg_s((float)lbm_N.y, lbm_ref_u, 1.0f, si_size.y, si_ref_u, si_rho);

    float u_scale = lbm_ref_u / si_ref_u;
    float z_off = units.x(z_si_offset);

    float lbm_nu = units.nu(si_nu);
    auto update_coriolis = [&](const bool show_info) {
        if (enable_coriolis) {
            // 1. center in degrees
            const float center_lon_deg = 0.5f * (cut_lon_min_deg + cut_lon_max_deg);
            const float center_lat_deg = 0.5f * (cut_lat_min_deg + cut_lat_max_deg);

            // 2. Earth rotation in SI
            const float Omega_earth_si = 7.292115e-5f; // [1/s]
            const float deg2rad = 3.14159265358979323846f / 180.0f;
            const float lat_rad = center_lat_deg * deg2rad;

            // local ENU frame: x east, y north, z up
            const float Omegax_si = 0.0f;
            const float Omegay_si = Omega_earth_si * cosf(lat_rad);
            const float Omegaz_si = Omega_earth_si * sinf(lat_rad);

            // 3. convert to lattice units
            // dx_SI is your cell size in meters, dt_SI = dx_SI * (u_lbm / u_SI)
            const float dx_si = cell_m;
            const float dt_si = dx_si * (lbm_ref_u / si_ref_u);

            coriolis_Omegax_lbmu = Omegax_si * dt_si;
            coriolis_Omegay_lbmu = Omegay_si * dt_si;
            coriolis_Omegaz_lbmu = Omegaz_si * dt_si;

            if (show_info) {
                println("| Coriolis enabled. Center lon,lat = " + to_string(center_lon_deg, 6u) + "," + to_string(center_lat_deg, 6u) + "               |");
                println("| Coriolis Omega(lbmu)    = ("
                    + to_string(coriolis_Omegax_lbmu, 8u) + ","
                    + to_string(coriolis_Omegay_lbmu, 8u) + ","
                    + to_string(coriolis_Omegaz_lbmu, 8u) + ") per step                        |");
            }

            const float omega_si = 7.292e-5f;           // Earth rotation [1/s]
            float f_si = 0.0f;
            if (cut_lat_min_deg != 0.0f || cut_lat_max_deg != 0.0f) {
                f_si = 2.0f * omega_si * std::sin(center_lat_deg * deg2rad);
            }
            coriolis_f_lbmu = f_si * dt_si;
            if (show_info) {
                std::ostringstream os;
                os << "| Coriolis (f): center(lon,lat)=(" << center_lon_deg << ", " << center_lat_deg << ") deg.";
                println(os.str());
                std::ostringstream os2;
                os2 << "| Coriolis (f): f_SI=" << f_si << " 1/s, f_LBM=" << coriolis_f_lbmu << ".";
                println(os2.str());
            }
        }
        else if (show_info) {
            println("| Coriolis term disabled by 'coriolis_term' setting in .luw.                  |");
        }
    };

    if (!dataset_generation && !profile_generation) {
        println("| SI Reference U  | " + alignl(7u, to_string(fmtf(si_ref_u))) + alignl(50u, "m/s") + " |");
        println("| LBM Reference U | " + alignl(7u, to_string(fmtf(lbm_ref_u))) + alignl(50u, "(Nondimensionalized)") + " |");
    }

    update_coriolis(!dataset_generation && !profile_generation);
 
    std::vector<SamplePoint> samples_si;
    std::vector<SamplePoint> samples;
    if (!dataset_generation && !profile_generation) {
        // read CSV
        const std::string csv_path = parent + "/proj_temp/SurfData_" + datetime + ".csv";
        samples_si = read_samples(csv_path);
        if (samples_si.empty()) { println("ERROR: no inlet samples. Aborting."); wait(); exit(-1); }

        // convert samples to LBM units
        samples.reserve(samples_si.size());
        for (const auto& s : samples_si) {
            SamplePoint sp;
            sp.p = float3(units.x(s.p.x), units.x(s.p.y), units.x(s.p.z));
            sp.u = s.u * u_scale;
            samples.push_back(sp);
        }
    }


	// ------------------------------- MESH LOADING -------------------------------
    println("|-----------------------------------------------------------------------------|");
    println("|                        LOADING GEOMETRY AND VOXELIZE                        |");
    println("|-----------------------------------------------------------------------------|");
    println("| Loading buildings as geometry, meshing...                                   |");

    std::string stl_path;
    {
        std::filesystem::path dir = std::filesystem::path(parent) / "proj_temp";
        if (!std::filesystem::exists(dir)) {
            println("ERROR: directory not found: " + dir.string());
            wait(); exit(-1);
        }
        const std::string prefix = caseName;

        // First try {parent}/proj_temp/{caseName}_DG.stl
        const std::filesystem::path dg_candidate = dir / (prefix + "_DG.stl");
        if (std::filesystem::exists(dg_candidate) && std::filesystem::is_regular_file(dg_candidate)) {
            stl_path = dg_candidate.string();
        }
        else {
            // Fallback to any *_DG.stl
            const std::string dg_suffix = "_DG.stl";
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                const std::string fname = entry.path().filename().string();
                if (fname.size() >= dg_suffix.size() &&
                    fname.substr(fname.size() - dg_suffix.size()) == dg_suffix) {
                    stl_path = entry.path().string();
                    break; // first match
                }
            }
        }

        if (stl_path.empty()) {
            // Fallback to any *.stl
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                if (entry.path().extension() == ".stl") {
                    stl_path = entry.path().string();
                    break; // first match
                }
            }
        }

        if (stl_path.empty()) {
            println("ERROR: no STL file. Tried " + dg_candidate.string() + ", *_DG.stl, then *.stl under " + dir.string());
            wait(); exit(-1);
        }

    }
    Mesh* mesh = read_stl(stl_path);


    println("| Time code: " + now_str() + "                                              |");

    if (!mesh) { println("ERROR: failed to load STL"); wait(); exit(-1); }
    const float target_lbm_x = units.x(si_size.x); const float scale_geom = target_lbm_x / mesh->get_bounding_box_size().x; mesh->scale(scale_geom);
    mesh->translate(float3(1.0f - mesh->pmin.x, 1.0f - mesh->pmin.y, 1.0f - mesh->pmin.z));

    println("| Geometry scaled by " + to_string(scale_geom, 4u) + ", ready for voxelization.                           |");

    auto run_lbm = [&](LBM& lbm, const std::string& vtk_prefix, const bool extra_spacing) {
        println("|-----------------------------------------------------------------------------|");
        println("|                           LBM SOLVER INFORMATION                            |");

        lbm.graphics.visualization_modes = VIS_FLAG_SURFACE | VIS_Q_CRITERION;

        const ulong lbm_T = 20001ull;
        const uint  vtk_dt = 20000u;        // export VTK

        auto print_task_finished = [&]() {
            println("| Task finished   | " + alignr(57u, now_str()) + " |");
            if (extra_spacing) {
                println("|                                                                             |");
            }
        };

        const std::string vtk_dir = parent + "/proj_temp/vtk/" + vtk_prefix + datetime + "_raw_";
        const std::string snapshots_dir = parent + "/proj_temp/snapshots";
        std::filesystem::create_directories(std::filesystem::path(vtk_dir).parent_path());
        std::filesystem::create_directories(snapshots_dir);

#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
        const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
        // Camera
        lbm.graphics.set_camera_free(
            float3(0.6f * Nx, -0.7f * Ny, 2.2f * Nz), // upstream & elevated
            -45.0f,                               // yaw
            30.0f,                                // pitch
            80.0f);                               // FOV

        lbm.run(0u, lbm_T); // initialize fields and graphics buffers
        while (lbm.get_t() < lbm_T) {
            // ------------------ off-screen PNG rendering (optional video) ------------------
            if (lbm.graphics.next_frame(lbm_T, 1.0f)) {
                {
                    auto __old_cwd = std::filesystem::current_path();
                    std::filesystem::current_path(snapshots_dir);
                    lbm.graphics.write_frame();
                    std::filesystem::current_path(__old_cwd);
                }
            }

            // ------------------ synchronous VTK output ------------------
            if (lbm.get_t() % vtk_dt == 0u) {
                // velocity (vector field)
                lbm.u.write_device_to_vtk(vtk_dir, true);
                // density as proxy for pressure (p = c_s^2 * (rho-1) can be computed in ParaView)
                lbm.rho.write_device_to_vtk(vtk_dir, true);
            }

            lbm.run(1u, lbm_T);
        }
        if (research_output_steps > 0) {
            println("|-----------------------------------------------------------------------------|");
            println("| Starting research output phase: " + to_string(research_output_steps) + " steps...                        |");

            std::vector<std::string> vtk_filenames;
            vtk_filenames.reserve(research_output_steps);

            const ulong lbm_T_research = lbm.get_t() + research_output_steps;

            while (lbm.get_t() < lbm_T_research) {
                lbm.run(1u, lbm_T_research);
                lbm.u.write_device_to_vtk(vtk_dir, true);
                // density as proxy for pressure (p = c_s^2 * (rho-1) can be computed in ParaView)
                lbm.rho.write_device_to_vtk(vtk_dir, true);
            }

            println("| Research output phase complete. Writing transform.info...             |");

            // 4. Write transform.info
            std::string info_path = parent + "/proj_temp/transform.info";
            std::ofstream info_file(info_path);
            if (info_file.is_open()) {
                // dt_si = (cell_m / 1.0) * (lbm_ref_u / si_ref_u) * 1.0
                const float dt_si = cell_m * (lbm_ref_u / si_ref_u);

                info_file << "dt = " << std::fixed << std::setprecision(10) << dt_si << "s\n";

                info_file.close();
                println("| Successfully wrote " + info_path + " |");
            }
            else {
                println("ERROR: Could not open " + info_path + " for writing.");
            }
        }
        print_task_finished();

#else // GRAPHICS + INTERACTIVE or pure CLI
        while (lbm.get_t() < lbm_T) {
            if (lbm.get_t() % vtk_dt == 0u) {
                const string vtk_file = vtk_dir + "step_" + to_string(lbm.get_t(), 5u) + ".vtk";
                lbm.write_vtk(vtk_file, true, true);
            }
            lbm.run(1u, lbm_T);
        }
        if (research_output_steps > 0) {
            println("|-----------------------------------------------------------------------------|");
            println("| Starting research output phase: " + to_string(research_output_steps) + " steps...                        |");

            std::vector<std::string> vtk_filenames;
            vtk_filenames.reserve(research_output_steps);

            const ulong lbm_T_research = lbm.get_t() + research_output_steps;

            while (lbm.get_t() < lbm_T_research) {
                lbm.run(1u, lbm_T_research);
                lbm.u.write_device_to_vtk(vtk_dir, true);
                // density as proxy for pressure (p = c_s^2 * (rho-1) can be computed in ParaView)
                lbm.rho.write_device_to_vtk(vtk_dir, true);
            }

            println("| Research output phase complete. Writing transform.info...             |");

            // 4. write transform.info
            std::string info_path = parent + "/proj_temp/transform.info";
            std::ofstream info_file(info_path);
            if (info_file.is_open()) {
                // dt_si = (cell_m / 1.0) * (lbm_ref_u / si_ref_u) * 1.0
                const float dt_si = cell_m * (lbm_ref_u / si_ref_u);

                info_file << "dt = " << std::fixed << std::setprecision(10) << dt_si << "s\n";

                info_file.close();
                println("| Successfully wrote " + info_path + " |");
            }
            else {
                println("ERROR: Could not open " + info_path + " for writing.");
            }
        }
        print_task_finished();
#endif
    };

    auto format_tag = [](float v) {
        std::string s = to_string(v, 3u);
        size_t dot = s.find('.');
        if (dot != std::string::npos) {
            while (!s.empty() && s.back() == '0') s.pop_back();
            if (!s.empty() && s.back() == '.') s.pop_back();
        }
        return s.empty() ? std::string("0") : s;
    };

    if (!dataset_generation && !profile_generation) {
        // const uint Dx = 2u, Dy = 1u, Dz = 1u;
        LBM lbm(lbm_N, Dx, Dy, Dz, lbm_nu);

        // LBM lbm(lbm_N, lbm_nu);   // single GPU mode

        const float3 origin_lbmu = lbm.position(0u, 0u, 0u);   // (0.5,0.5,0.5)
        for (auto& sp : samples) {
            sp.p.x += origin_lbmu.x;
            sp.p.y += origin_lbmu.y;
            sp.p.z += origin_lbmu.z;
        }
        const float z0_lbmu = origin_lbmu.z;

        lbm.voxelize_mesh_on_device(mesh);
        println("| Voxelization done.                                                          |");
        println("|-----------------------------------------------------------------------------|");
        println("|                         BUILD BOUNDARY CONDITIONS                           |");
        println("|-----------------------------------------------------------------------------|");
        println("| CDF data loaded | " + alignl(57u, to_string(samples_si.size())) + " |");

        std::vector<float3> P; P.reserve(samples.size());
        std::vector<float3> Uv; Uv.reserve(samples.size());
        for (const auto& s : samples) { P.push_back(s.p); Uv.push_back(s.u); }

        if (use_high_order) {
            KNNInterpolatorHD knn_hd(std::move(P), std::move(Uv));
            // Calculate z base threshold in LBM units
            const float z_base_threshold_lbmu = units.x(z_si_offset) + z0_lbmu;
            InletVelocityFieldHD inlet_hd(knn_hd, z_base_threshold_lbmu);
            apply_inlet_outlet_hd(lbm, downstream_bc, inlet_hd);
            println("| [" + now_str() + "] Boundary initialization complete (high-order).        |");

            if (flux_correction) {
                println("| [" + now_str() + "] Starting flux correction...                           |");
                double avg_du = 0.0, net_b = 0.0, net_a = 0.0;
                auto eval = [&](const float3& q)->float3 { return inlet_hd(q); };
                apply_flux_correction(lbm, downstream_bc, eval, /*show_report=*/true,
                    &avg_du, &net_b, &net_a);
                // std::fprintf(stdout, "[flux] check: net_before=%.9e, net_after=%.9e, avg|Ku|=%.9e m/s\n", net_b, net_a, avg_du);
            }
            else {
                println("| Skipping flux correction. Use flux_correction = true to enable.   |");
            }
        }
        else {
            NearestNeighborInterpolator nn(std::move(P), std::move(Uv));
            InletVelocityField inlet(nn, z0_lbmu, z_off);
            apply_inlet_outlet(lbm, downstream_bc, inlet);
            println("| [" + now_str() + "] Boundary initialization complete (low-order).             |");

            if (flux_correction) {
                println("| [" + now_str() + "] Starting flux correction...                           |");
                double avg_du = 0.0, net_b = 0.0, net_a = 0.0;
                auto eval = [&](const float3& q)->float3 { return inlet(q); };
                apply_flux_correction(lbm, downstream_bc, eval, /*show_report=*/true,
                    &avg_du, &net_b, &net_a);
            }
            else {
                println("| Skipping flux correction. Use flux_correction = true to enable.   |");
            }
        }

        run_lbm(lbm, "", false);
    }
    else if (dataset_generation) {
        if (angle_list.empty()) {
            println("| ERROR: dataset generation requires angle list (angle=[...]).                |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }

        const uint total_cases = static_cast<uint>(inflow_list.size() * angle_list.size());
        uint case_index = 0u;

        auto initialize_uniform_velocity = [&](LBM& lbm, const float3& u) {
            const ulong Ntot = lbm.get_N();
            for (ulong n = 0ull; n < Ntot; ++n) {
                lbm.u.x[n] = u.x;
                lbm.u.y[n] = u.y;
                lbm.u.z[n] = u.z;
            }
        };

        auto apply_velocity_boundaries = [&](LBM& lbm, const float3& u) {
            const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
            const bool has_ground = (Nz > 1u);
            const ulong Ntot = lbm.get_N();
            for (ulong n = 0ull; n < Ntot; ++n) {
                uint x = 0u, y = 0u, z = 0u;
                lbm.coordinates(n, x, y, z);
                if (has_ground && z == 0u) {
                    lbm.flags[n] = TYPE_S;
                    continue;
                }
                const bool is_boundary =
                    (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || (has_ground && z == Nz - 1u));
                if (is_boundary) {
                    lbm.flags[n] = TYPE_E;
                    lbm.u.x[n] = u.x;
                    lbm.u.y[n] = u.y;
                    lbm.u.z[n] = u.z;
                }
            }
        };

        for (const float inflow_si : inflow_list) {
            for (const float angle_deg : angle_list) {
                ++case_index;
                const uint remaining = total_cases - case_index;

                si_ref_u = inflow_si;
                u_scale = lbm_ref_u / si_ref_u;
                units.set_m_kg_s((float)lbm_N.y, lbm_ref_u, 1.0f, si_size.y, si_ref_u, si_rho);
                z_off = units.x(z_si_offset);
                lbm_nu = units.nu(si_nu);
                update_coriolis(false);

                const std::string case_label = to_string(case_index) + "/" + to_string(total_cases) +
                    " (remaining " + to_string(remaining) + ")";
                println("|-----------------------------------------------------------------------------|");
                println("| Dataset case    | " + alignr(57u, case_label) + " |");
                println("| Inflow / Angle  | " + alignr(57u, format_tag(inflow_si) + " m/s, " + format_tag(angle_deg) + " deg") + " |");
                println("| SI Reference U  | " + alignr(57u, format_tag(si_ref_u) + " m/s") + " |");

                const float3 inflow_lbmu = wind_velocity_lbmu(inflow_si, angle_deg, u_scale);

                LBM lbm(lbm_N, Dx, Dy, Dz, lbm_nu);

                lbm.voxelize_mesh_on_device(mesh);
                println("| Voxelization done.                                                          |");
                println("|-----------------------------------------------------------------------------|");
                println("|                         BUILD BOUNDARY CONDITIONS                           |");
                println("|-----------------------------------------------------------------------------|");
                initialize_uniform_velocity(lbm, inflow_lbmu);
                apply_velocity_boundaries(lbm, inflow_lbmu);
                println("| " + alignl(75u, "[" + now_str() + "] Boundary initialization complete (dataset).") + " |");

                const std::string vtk_prefix = "DG_" + format_tag(inflow_si) + "_" + format_tag(angle_deg) + "_";
                run_lbm(lbm, vtk_prefix, true);
            }
        }
    }
    else { // profile_generation
        if (angle_list.empty()) {
            println("| ERROR: profile forcing requires angle list (angle=[...]).                   |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        if (profile_u_si.empty()) {
            println("| ERROR: profile forcing requires profile_u_si to be initialized.             |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }

        std::vector<float> profile_u_lbmu(profile_u_si.size(), 0.0f);
        for (size_t i = 0; i < profile_u_si.size(); ++i) {
            profile_u_lbmu[i] = profile_u_si[i] * u_scale;
        }

        const uint total_cases = static_cast<uint>(angle_list.size());
        uint case_index = 0u;

        auto compute_u_by_z = [&](LBM& lbm) {
            const uint Nz = lbm.get_Nz();
            std::vector<float> u_by_z(Nz, 0.0f);
            const float z_base_lbmu = lbm.position(0u, 0u, 0u).z + units.x(z_si_offset);
            const float inv_dz = 1.0f / profile_dz_si;
            const uint last_idx = static_cast<uint>(profile_u_lbmu.size() - 1u);
            for (uint z = 0u; z < Nz; ++z) {
                const float pos_z = lbm.position(0u, 0u, z).z;
                float u_mag = 0.0f;
                if (pos_z >= z_base_lbmu) {
                    float z_above_si = units.si_x(pos_z - z_base_lbmu);
                    if (z_above_si < 0.0f) z_above_si = 0.0f;
                    long idx_l = std::lround(z_above_si * inv_dz);
                    if (idx_l < 0l) idx_l = 0l;
                    uint idx = static_cast<uint>(idx_l);
                    if (idx > last_idx) idx = last_idx;
                    u_mag = profile_u_lbmu[idx];
                }
                u_by_z[z] = u_mag;
            }
            return u_by_z;
        };

        auto initialize_profile_velocity = [&](LBM& lbm, const std::vector<float>& u_by_z,
                                                const float dir_x, const float dir_y) {
            const ulong Ntot = lbm.get_N();
            for (ulong n = 0ull; n < Ntot; ++n) {
                uint x = 0u, y = 0u, z = 0u;
                lbm.coordinates(n, x, y, z);
                const float u_mag = u_by_z[z];
                lbm.u.x[n] = dir_x * u_mag;
                lbm.u.y[n] = dir_y * u_mag;
                lbm.u.z[n] = 0.0f;
            }
        };

        auto apply_profile_boundaries = [&](LBM& lbm, const std::vector<float>& u_by_z,
                                             const float dir_x, const float dir_y) {
            const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
            const bool has_ground = (Nz > 1u);
            const ulong Ntot = lbm.get_N();
            for (ulong n = 0ull; n < Ntot; ++n) {
                uint x = 0u, y = 0u, z = 0u;
                lbm.coordinates(n, x, y, z);
                if (has_ground && z == 0u) {
                    lbm.flags[n] = TYPE_S;
                    continue;
                }
                const bool is_boundary =
                    (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || (has_ground && z == Nz - 1u));
                if (is_boundary) {
                    lbm.flags[n] = TYPE_E;
                    const float u_mag = u_by_z[z];
                    lbm.u.x[n] = dir_x * u_mag;
                    lbm.u.y[n] = dir_y * u_mag;
                    lbm.u.z[n] = 0.0f;
                }
            }
        };

        for (const float angle_deg : angle_list) {
            ++case_index;
            const uint remaining = total_cases - case_index;

            const std::string case_label = to_string(case_index) + "/" + to_string(total_cases) +
                " (remaining " + to_string(remaining) + ")";
            println("|-----------------------------------------------------------------------------|");
            println("| Profile case    | " + alignr(57u, case_label) + " |");
            println("| Angle           | " + alignr(57u, format_tag(angle_deg) + " deg") + " |");
            println("| SI Reference U  | " + alignr(57u, format_tag(si_ref_u) + " m/s") + " |");

            const float deg2rad = 3.14159265358979323846f / 180.0f;
            const float angle_rad = angle_deg * deg2rad;
            const float dir_x = -sinf(angle_rad);
            const float dir_y = -cosf(angle_rad);

            LBM lbm(lbm_N, Dx, Dy, Dz, lbm_nu);

            lbm.voxelize_mesh_on_device(mesh);
            println("| Voxelization done.                                                          |");
            println("|-----------------------------------------------------------------------------|");
            println("|                         BUILD BOUNDARY CONDITIONS                           |");
            println("|-----------------------------------------------------------------------------|");

            const std::vector<float> u_by_z = compute_u_by_z(lbm);
            initialize_profile_velocity(lbm, u_by_z, dir_x, dir_y);
            apply_profile_boundaries(lbm, u_by_z, dir_x, dir_y);
            println("| " + alignl(75u, "[" + now_str() + "] Boundary initialization complete (profile).") + " |");

            const std::string vtk_prefix = "ANG_" + format_tag(angle_deg) + "_";
            run_lbm(lbm, vtk_prefix, true);
        }
    }
}
