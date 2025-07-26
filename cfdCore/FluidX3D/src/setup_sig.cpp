#include "setup.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>

#include <cfloat>  // lwg
#include <chrono>   
#include <ctime>    // std::localtime
#include <iomanip>  // std::put_time

// ────────────── Global configuration ───────────────
std::string caseName = "hebei";
std::string datetime = "20250721015000";
float        z_si_offset = 50.0f;
uint         memory = 16000u;
float3       si_size = float3(0);

struct SamplePoint { float3 p; float3 u; };

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

// // helper: return current local time in "YYYY-MM-DD hh:mm:ss" format
// static string now_str() {
//     using namespace std::chrono;
//     auto now = system_clock::now();
//     std::time_t tt = system_clock::to_time_t(now);
//     std::tm tm;
//     if (localtime_s(&tm, &tt) != 0) {
//         throw std::runtime_error("Failed to get local time.");
//     }
//     std::stringstream ss;
//     ss << std::put_time(&tm, "%F %T"); // %F = YYYY-MM-DD, %T = hh:mm:ss
//     return ss.str();
// }

// helper: return current local time in "YYYY-MM-DD hh:mm:ss" format
static string now_str() {
    // 获取当前时间戳（自1970年1月1日以来的秒数）
    time_t now = time(nullptr);
    struct tm* local_time = localtime(&now);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);
    //std::cout << "格式化后: " << buffer << std::endl;
    return buffer;
}

void main_setup() {
    println("|-----------------------------------------------------------------------------|");
    println("|                                                                             |");
    println("|      Urban Simulation with WRF bridge by Huanxia Wei for NMIC (CMA)         |");
    println("|                                            Version - 250713 - Draft         |");
    println("|                                                                             |");
    println("|-----------------------------------------------------------------------------|");
 
    // ---------------------- read conf.txt -----------------------------------
    {
        const std::string conf_path = get_exe_path() + std::string("../../../conf.txt");
        std::ifstream fin(conf_path);

        // trim
        auto trim = [](std::string s) {
            const char* ws = " \t";
            size_t b = s.find_first_not_of(ws);
            size_t e = s.find_last_not_of(ws);
            return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
        };
        // get values
        auto second_val = [](const std::string& rng) {
            size_t c = rng.find(','); size_t r = rng.find(']', c);
            return (float)atof(rng.substr(c + 1, r - c - 1).c_str());
        };

        if (!fin.is_open()) {
            println("WARNING: conf.txt not found, using integrated defaults...");
            caseName = "shanghai";
            si_size  = float3(3518.36f, 4438.94f, 1000.0f + z_si_offset);
        } else {
            std::string line;
            while (std::getline(fin, line)) {
                size_t cmt = line.find("//"); if (cmt != std::string::npos) line.erase(cmt);
                size_t eq  = line.find('=');  if (eq  == std::string::npos) continue;

                std::string key = trim(line.substr(0, eq));
                std::string val = trim(line.substr(eq + 1));

                if (key == "casename")        caseName     = val;
                else if (key == "datetime")   datetime = val;
                else if (key == "base_height") z_si_offset = (float)atof(val.c_str());
                else if (key == "memory_lbm")  memory      = (uint)atoi(val.c_str());
                else if (key == "si_x_cfd")    si_size.x   = second_val(val);
                else if (key == "si_y_cfd")    si_size.y   = second_val(val);
                else if (key == "si_z_cfd")    si_size.z   = second_val(val);
            }
            si_size.z += z_si_offset;             
            if (!memory) memory = 6000u;          
        }
    }

    // print configurations
    println("| Using conf: caseName=" + caseName +
           ", datetime=" + datetime +
           ", base_height=" + to_string(z_si_offset) +
           ", memory_lbm="  + to_string(memory) +
           ", si_size=["    + to_string(si_size.x) + "," +
                              to_string(si_size.y) + "," +
                              to_string(si_size.z) + "] |");

    const float lbm_ref_u = 0.05f, si_ref_u = 2.0f; 
    const float si_nu = 1.48E-5f, si_rho = 1.225f;

    const uint3 lbm_N = resolution(si_size, memory);
    println("| Grid resolution Nx,Ny,Nz = " + to_string(lbm_N.x) + "," + to_string(lbm_N.y) + "," + to_string(lbm_N.z) + "                                    |");
    units.set_m_kg_s((float)lbm_N.y, lbm_ref_u, 1.0f, si_size.y, si_ref_u, si_rho);

    const float z_off = units.x(z_si_offset); // 对应的 LBM 单位

    const float lbm_nu = units.nu(si_nu);
    println("| LBM viscosity = " + to_string(lbm_nu, 6u) + "                                                    |");

    // read CSV
    
    const std::string csv_path = get_exe_path() + std::string("../../../wrfInput/") + caseName + "/SurfData_"+datetime+".csv";

    auto samples_si = read_samples(csv_path);
    println("| WRF data loaded = " + to_string(samples_si.size()) + "                                                     |");
    if (samples_si.empty()) { println("ERROR: no inlet samples. Aborting."); wait(); exit(-1); }

    // convert samples to LBM units
    const float u_scale = lbm_ref_u / si_ref_u;
    std::vector<SamplePoint> samples; samples.reserve(samples_si.size());
    for (const auto& s : samples_si) { SamplePoint sp; sp.p = float3(units.x(s.p.x),units.x(s.p.y), units.x(s.p.z + z_si_offset));sp.u = s.u * u_scale; samples.push_back(sp); }

    LBM lbm(lbm_N, lbm_nu);
    // ── 把 CSV 样本整体平移到格点中心 (0.5,0.5,0.5) ────────────────
    const float3 origin_lbmu = lbm.position(0u, 0u, 0u);   // (0.5,0.5,0.5)
    for (auto& sp : samples) {
        sp.p.x += origin_lbmu.x;
        sp.p.y += origin_lbmu.y;
        sp.p.z += origin_lbmu.z;        // 与 50 m 提升叠加后 → 完全对齐
    }
    const float z0_lbmu = origin_lbmu.z;                    // 域底中心

    //const float z0_lbmu = lbm.position(0u, 0u, 0u).z;   // 域底中心

	// ------------------------------- MESH LOADING -------------------------------
    println("|-----------------------------------------------------------------------------|");
    println("| Loading buildings as geometry, meshing...                                   |");
    println("|-----------------------------------------------------------------------------|");
    Mesh* mesh = read_stl(get_exe_path() + std::string("../../../geoData/") + caseName + "/" + caseName + "_with_base.stl");

    println("Time code = " + now_str());

    if (!mesh) { println("ERROR: failed to load STL"); wait(); exit(-1); }
    const float target_lbm_x = units.x(si_size.x); const float scale_geom = target_lbm_x / mesh->get_bounding_box_size().x; mesh->scale(scale_geom);
    mesh->translate(float3(1.0f - mesh->pmin.x, 1.0f - mesh->pmin.y, 1.0f - mesh->pmin.z));

    println("| Geometry scaled by " + to_string(scale_geom, 4u) + ", voxelizing...                                    |");

    lbm.voxelize_mesh_on_device(mesh);
    println("|-----------------------------------------------------------------------------|");
    println("Voxelization done.");
    println("Connecting to WRF data (single-threading only, as draft version).");
    println("Time code = " + now_str());

    // nearest neighbour interp
    auto interp_nn = [&](const float3& pos) {
        float best = FLT_MAX;
        float3 u(0);
        for (const auto& sp : samples) {
            float3 d = pos - sp.p;
            float  d2 = dot(d, d);
            if (d2 < best) { best = d2; u = sp.u; }
        }
        return u;
        };

    auto inlet_velocity = [&](const float3& pos) -> float3 {
        const float z_phys = pos.z - z0_lbmu;   // move basement
        if (z_phys < z_off)                     // < 50 m is static
            return float3(0.0f);

        return interp_nn(pos);                  // intepolation
        };


    const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
    parallel_for(lbm.get_N(), [&](ulong n) { uint x = 0, y = 0, z = 0; lbm.coordinates(n, x, y, z); const float3 pos = lbm.position(x, y, z);
    if (z == 0u) { lbm.flags[n] = TYPE_S; return; } 
    //bool inlet = (x == 0u || x == Nx - 1u || y == 0u || z == Nz - 1u); 
    //bool outlet = (y == Ny - 1u);
    bool inlet = (x == 0u || x == Nx - 1u
        || y == 0u || y == Ny - 1u 
        || z == Nz - 1u); 
    bool outlet = false; 


    if (inlet) {
        lbm.flags[n] = TYPE_E;
        float3 u = inlet_velocity(pos);              
        lbm.u.x[n] = u.x;
        lbm.u.y[n] = u.y;
        lbm.u.z[n] = u.z;
    }
    else if (outlet) { lbm.flags[n] = TYPE_E; }
        });
    println("Boundary initialization complete.");
    println("Time code = " + now_str());

    // ------------------------------------------------------------------- graphics & run --------------------------------------------------------------------
    lbm.graphics.visualization_modes = VIS_FLAG_SURFACE | VIS_Q_CRITERION;

    const ulong lbm_T = 40001ull;   // total simulation time steps (~steady‑state)
    const uint  vtk_dt = 20000u;        // export VTK every 20 time steps
    const std::string vtk_dir = get_exe_path() + std::string("../../../caseData/") + caseName + "/"+datetime+"_";
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
            lbm.graphics.write_frame(); // default path: exe_dir/export/frame_#####.png
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
#else // GRAPHICS + INTERACTIVE or pure CLI
    while (lbm.get_t() < lbm_T) {
        if (lbm.get_t() % vtk_dt == 0u) {
            const string vtk_file = vtk_dir + "step_" + to_string(lbm.get_t(), 5u) + ".vtk";
            lbm.write_vtk(vtk_file, true, true);
        }
        lbm.run(1u, lbm_T);
    }
#endif
}
