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

void main_setup() {
  //println("|-----------------------------------------------------------------------------|");
    println("|                                                                             |");
    println("|      LatticeUrbanWind LUW: Towards Micrometeorology Fastest Simulation      |");
    println("|                                                                             |");
    println("|                                        Developed by Huanxia Wei's Team      |");
    println("|                                        Version - v3.5-251119                |");
    println("|                                                                             |");
    println("|-----------------------------------------------------------------------------|");
 
    // ---------------------- read *.luw -----------------------------------
    std::string parent; // directory that contains the *.luw configure file
    {
        // prompt for configure file path when not directly provided
        println("| Please input configure file path (*.luw):                                   |");
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

        if (!fin.is_open()) {
            println("ERROR: *.luw not found. Please provide a valid *.luw and rerun.");
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
    {
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
    {
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

    units.set_m_kg_s((float)lbm_N.y, lbm_ref_u, 1.0f, si_size.y, si_ref_u, si_rho);

    const float z_off = units.x(z_si_offset); 

    const float lbm_nu = units.nu(si_nu);
    println("| SI Reference U  | " + alignl(7u, to_string(fmtf(si_ref_u))) + alignl(50u, "m/s") + " |");
    println("| LBM Reference U | " + alignl(7u, to_string(fmtf(lbm_ref_u))) + alignl(50u, "(Nondimensionalized)") + " |");
    // Coriolis based on domain center lat/lon from *.luw
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

        // 4. give a short host-side info
        println("| Coriolis enabled. Center lon,lat = " + to_string(center_lon_deg, 6u) + "," + to_string(center_lat_deg, 6u) + "               |");
        println("| Coriolis Omega(lbmu)    = ("
            + to_string(coriolis_Omegax_lbmu, 8u) + ","
            + to_string(coriolis_Omegay_lbmu, 8u) + ","
            + to_string(coriolis_Omegaz_lbmu, 8u) + ") per step                        |");
    }
    else {
        println("| Coriolis term disabled by 'coriolis_term' setting in .luw.                  |");
    }


    // read CSV
    const std::string csv_path = parent + "/proj_temp/SurfData_" + datetime + ".csv";

    auto samples_si = read_samples(csv_path);

    if (samples_si.empty()) { println("ERROR: no inlet samples. Aborting."); wait(); exit(-1); }

    // convert samples to LBM units
    const float u_scale = lbm_ref_u / si_ref_u;
    if (enable_coriolis)
    {
        const float deg2rad = 3.14159265358979323846f / 180.0f;
        const float center_lon_deg = 0.5f * (cut_lon_min_deg + cut_lon_max_deg);
        const float center_lat_deg = 0.5f * (cut_lat_min_deg + cut_lat_max_deg);
        const float omega_si = 7.292e-5f;           // Earth rotation [1/s]
        float f_si = 0.0f;
        if (cut_lat_min_deg != 0.0f || cut_lat_max_deg != 0.0f) {
            f_si = 2.0f * omega_si * std::sin(center_lat_deg * deg2rad);
        }
        const float dt_si = cell_m * (lbm_ref_u / si_ref_u);
        coriolis_f_lbmu = f_si * dt_si;
        {
            std::ostringstream os;
            os << "| Coriolis (f): center(lon,lat)=(" << center_lon_deg << ", " << center_lat_deg << ") deg.";
            println(os.str());
        }
        {
            std::ostringstream os;
            os << "| Coriolis (f): f_SI=" << f_si << " 1/s, f_LBM=" << coriolis_f_lbmu << ".";
            println(os.str());
        }
    }

    std::vector<SamplePoint> samples; samples.reserve(samples_si.size());
    for (const auto& s : samples_si) { SamplePoint sp; sp.p = float3(units.x(s.p.x), units.x(s.p.y), units.x(s.p.z)); sp.u = s.u * u_scale; samples.push_back(sp);}

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

        // First try {parent}/proj_temp/{caseName}_DEM.stl
        const std::filesystem::path dem_candidate = dir / (prefix + "_DEM.stl");
        if (std::filesystem::exists(dem_candidate) && std::filesystem::is_regular_file(dem_candidate)) {
            stl_path = dem_candidate.string();
        }
        else {
            // Fallback to any {caseName}*.stl
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                const std::string fname = entry.path().filename().string();
                if (fname.size() >= prefix.size() + 4 &&
                    fname.substr(0, prefix.size()) == prefix &&
                    entry.path().extension() == ".stl") {
                    stl_path = entry.path().string();
                    break; // first match
                }
            }
        }

        if (stl_path.empty()) {
            println("ERROR: no STL file. Tried " + dem_candidate.string() + " then " + prefix + "*.stl under " + dir.string());
            wait(); exit(-1);
        }

    }
    Mesh* mesh = read_stl(stl_path);


    println("| Time code: " + now_str() + "                                              |");

    if (!mesh) { println("ERROR: failed to load STL"); wait(); exit(-1); }
    const float target_lbm_x = units.x(si_size.x); const float scale_geom = target_lbm_x / mesh->get_bounding_box_size().x; mesh->scale(scale_geom);
    mesh->translate(float3(1.0f - mesh->pmin.x, 1.0f - mesh->pmin.y, 1.0f - mesh->pmin.z));

    println("| Geometry scaled by " + to_string(scale_geom, 4u) + ", voxelizing...                                    |");

    lbm.voxelize_mesh_on_device(mesh);
    println("| Voxelization done.                                                          |");
    println("|-----------------------------------------------------------------------------|");
    println("|                         BUILD BOUNDARY CONDITIONS                           |");
    println("|-----------------------------------------------------------------------------|");
    println("| CDF data loaded | " + alignl(57u, to_string(samples_si.size())) + " |");

    std::vector<float3> P; P.reserve(samples.size());
    std::vector<float3> Uv; Uv.reserve(samples.size());
    for (const auto& s : samples) { P.push_back(s.p); Uv.push_back(s.u); }

    const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();

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
            //std::fprintf(stdout, "[flux] check: net_before=%.9e, 
            // net_after=%.9e, avg|Δu|=%.9e m/s\n", net_b, net_a, avg_du);
        }
        else{
            println("| Skipping flux correction. Use flux_correction = true to enable.   |"); }
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
            println("| Skipping flux correction. Use flux_correction = true to enable.   |"); }
    }

    println("|-----------------------------------------------------------------------------|");
    println("|                           LBM SOLVER INFORMATION                            |");

    // ------------------------------------------------------------------- RUN LBM --------------------------------------------------------------------
    lbm.graphics.visualization_modes = VIS_FLAG_SURFACE | VIS_Q_CRITERION;

    const ulong lbm_T = 40001ull; 
    const uint  vtk_dt = 10000u;        // export VTK

    const std::string vtk_dir = parent + "/proj_temp/vtk/" + datetime + "_raw_";
    const std::string snapshots_dir = parent + "/proj_temp/snapshots";
    std::filesystem::create_directories(std::filesystem::path(vtk_dir).parent_path());
    std::filesystem::create_directories(snapshots_dir);

    //const string vtk_dir = get_exe_path() + "vtk/"; // ensure this directory exists beforehand

#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
    // Camera
    lbm.graphics.set_camera_free(
        float3(0.6f * Nx, -0.7f * Ny, 2.2f * Nz), // upstream & elevated
        -45.0f,                               // yaw
        30.0f,                                // pitch
        80.0f);                               // FOV

    lbm.run(0u, lbm_T); // initialize fields and graphics buffers
    while (lbm.get_t() < lbm_T) {
        // ------------------ off‑screen PNG rendering (optional video) ------------------
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
            const string ts = to_string(lbm.get_t());
            // velocity (vector field)
            lbm.u.write_device_to_vtk(vtk_dir, true);
            // density as proxy for pressure (p = c_s^2 * (rho-1) can be computed in ParaView)
            lbm.rho.write_device_to_vtk(vtk_dir, true);
        }

        lbm.run(1u, lbm_T);
    }
    println("| Task finished: " + now_str() + "                                                |");

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
#endif
}
