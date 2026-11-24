#include "terrain.hpp"
#include "context.hpp"
#include "deck.hpp"
#include "report.hpp"

#include <cfloat>
#include <cstdlib>
#include <sstream>

namespace culbm::cases {

using namespace culbm::bc;

std::vector<DemCsvPoint> read_interpolated_dem_csv(
    const string& path,
    float* x_min, float* x_max,
    float* y_min, float* y_max,
    float* elev_min, float* elev_max) {
    if (x_min) *x_min = 0.0f;
    if (x_max) *x_max = 0.0f;
    if (y_min) *y_min = 0.0f;
    if (y_max) *y_max = 0.0f;
    if (elev_min) *elev_min = 0.0f;
    if (elev_max) *elev_max = 0.0f;

    std::vector<DemCsvPoint> out;
    std::ifstream fin(path);
    if (!fin.is_open()) return out;

    auto split_csv = [](const string& s) -> std::vector<string> {
        std::vector<string> out_cols;
        std::stringstream ss(s);
        string tok;
        while (std::getline(ss, tok, ',')) out_cols.push_back(trim_copy(tok));
        return out_cols;
    };
    auto to_lower_copy = [](string s) -> string {
        for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
        return s;
    };
    auto find_col = [&](const std::vector<string>& cols, const string& key) -> int {
        const string key_lc = to_lower_copy(key);
        for (size_t i = 0u; i < cols.size(); ++i) {
            if (to_lower_copy(cols[i]) == key_lc) return (int)i;
        }
        return -1;
    };

    string header_line;
    if (!std::getline(fin, header_line)) return out;
    const std::vector<string> header_cols = split_csv(header_line);
    int idx_x = find_col(header_cols, "x");
    int idx_y = find_col(header_cols, "y");
    int idx_e = find_col(header_cols, "elevation");
    if (idx_e < 0) idx_e = find_col(header_cols, "z");
    const bool use_named = (idx_x >= 0) && (idx_y >= 0) && (idx_e >= 0);

    float xmin = +FLT_MAX, xmax = -FLT_MAX;
    float ymin = +FLT_MAX, ymax = -FLT_MAX;
    float emin = +FLT_MAX, emax = -FLT_MAX;

    string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        for (char& ch : line) {
            if (ch == ';' || ch == '\t') ch = ',';
        }
        const std::vector<string> cols = split_csv(line);
        float x = 0.0f, y = 0.0f, e = 0.0f;

        if (use_named) {
            const int need_max = std::max(idx_x, std::max(idx_y, idx_e));
            if ((int)cols.size() <= need_max) continue;
            x = (float)atof(cols[idx_x].c_str());
            y = (float)atof(cols[idx_y].c_str());
            e = (float)atof(cols[idx_e].c_str());
        }
        else {
            if (cols.size() < 3u) continue;
            x = (float)atof(cols[0].c_str());
            y = (float)atof(cols[1].c_str());
            e = (float)atof(cols[2].c_str());
        }

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(e)) continue;
        out.push_back(DemCsvPoint{ x, y, e });
        xmin = fminf(xmin, x); xmax = fmaxf(xmax, x);
        ymin = fminf(ymin, y); ymax = fmaxf(ymax, y);
        emin = fminf(emin, e); emax = fmaxf(emax, e);
    }

    if (!out.empty()) {
        if (x_min) *x_min = xmin;
        if (x_max) *x_max = xmax;
        if (y_min) *y_min = ymin;
        if (y_max) *y_max = ymax;
        if (elev_min) *elev_min = emin;
        if (elev_max) *elev_max = emax;
    }
    return out;
}

void load_terrain(CaseContext& ctx) {
    const CaseSettings& deck = ctx.deck;
    const CaseGeometry& geometry = ctx.geometry;
    const CaseDomain& domain = ctx.domain;
    CaseTerrain& terrain = ctx.terrain;
    if (deck.profile_generation || deck.culbm_output_agl_requested) {
        const std::string dem_csv_path = deck.parent + "/proj_temp/interpolated_dem.csv";
        terrain.profile_dem_points_raw = read_interpolated_dem_csv(
            dem_csv_path,
            &terrain.profile_dem_xmin, &terrain.profile_dem_xmax,
            &terrain.profile_dem_ymin, &terrain.profile_dem_ymax,
            &terrain.profile_dem_emin, &terrain.profile_dem_emax
        );
        terrain.profile_dem_loaded = !terrain.profile_dem_points_raw.empty();
        if (terrain.profile_dem_loaded) {
            print_kv_row("Terrain DEM", "Loaded " + to_string((ulong)terrain.profile_dem_points_raw.size()) + " points from interpolated_dem.csv");
            print_kv_row("DEM bounds SI", "x=[" + to_string(terrain.profile_dem_xmin, 3u) + ", " + to_string(terrain.profile_dem_xmax, 3u) +
                    "], y=[" + to_string(terrain.profile_dem_ymin, 3u) + ", " + to_string(terrain.profile_dem_ymax, 3u) +
                    "], elev=[" + to_string(terrain.profile_dem_emin, 3u) + ", " + to_string(terrain.profile_dem_emax, 3u) + "]");
        }
        else if (deck.profile_generation) {
            print_kv_row("Terrain DEM", "interpolated_dem.csv not found or empty, fallback to flat ground");
        }
        else {
            print_kv_row("Physical AGL DEM", "interpolated_dem.csv not found or empty");
        }
    }

    // The generated foundation is an implementation aid, not part of the
    // requested physical datum.  Both normal and Profile STL preprocessors
    // place its top at stl_min + base_height, even though their raw z=0
    // differs.  Rebase only z; x/y remain the metric coordinates carried by
    // the STL/solver domain.
    terrain.physical_output_z_zero_si =
        geometry.stl_min_si.z + domain.cuda_ground_base_height_si;
    terrain.physical_output_dem_ready = false;
    if (deck.culbm_output_agl_requested) {
        const float dem_rx = terrain.profile_dem_xmax - terrain.profile_dem_xmin;
        const float dem_ry = terrain.profile_dem_ymax - terrain.profile_dem_ymin;
        if (!terrain.profile_dem_loaded || !(dem_rx > 1.0e-6f && dem_ry > 1.0e-6f &&
                                     geometry.stl_size_si.x > 1.0e-6f && geometry.stl_size_si.y > 1.0e-6f)) {
            println("| ERROR: AGL output requires proj_temp/interpolated_dem.csv with valid x/y.   |");
            println("|        Disable culbm_output_agl_* or regenerate the terrain DEM.             |");
            wait(); exit(-1);
        }
        const float dem_to_stl_sx = geometry.stl_size_si.x / dem_rx;
        const float dem_to_stl_sy = geometry.stl_size_si.y / dem_ry;
        std::vector<SamplePoint> physical_ground_samples;
        physical_ground_samples.reserve(terrain.profile_dem_points_raw.size());
        for (const DemCsvPoint& dp : terrain.profile_dem_points_raw) {
            const float x_stl = geometry.stl_min_si.x + (dp.x - terrain.profile_dem_xmin) * dem_to_stl_sx;
            const float y_stl = geometry.stl_min_si.y + (dp.y - terrain.profile_dem_ymin) * dem_to_stl_sy;
            if (!std::isfinite(x_stl) || !std::isfinite(y_stl) || !std::isfinite(dp.elevation)) continue;
            SamplePoint sample;
            sample.p = float3(x_stl, y_stl, 0.0f);
            // interpolated_dem.csv is stored relative to its lowest terrain
            // point.  That is exactly the requested global z datum after the
            // artificial foundation has been removed.
            sample.T = dp.elevation;
            sample.patch = k_patch_bottom;
            physical_ground_samples.push_back(sample);
        }
        terrain.physical_output_ground_plane.build_from_patch0_samples(physical_ground_samples, 0.0f);
        terrain.physical_output_dem_ready = terrain.physical_output_ground_plane.has_samples();
        if (!terrain.physical_output_dem_ready) {
            println("| ERROR: no usable DEM samples remain for requested AGL output.                |");
            wait(); exit(-1);
        }
        print_kv_row("Physical AGL DEM", to_string((ulong)physical_ground_samples.size()) +
            " mapped samples; terrain-relative points will omit z <= DEM");
    }
    if (deck.culbm_output_physical_requested) {
        print_kv_row("Physical output", "final step only; global VTK=" +
            string(deck.culbm_output_global_vtk ? "on" : "off") + ", global NetCDF=" +
            string(deck.culbm_output_global_netcdf ? "on" : "off") + ", AGL VTK=" +
            string(deck.culbm_output_agl_vtk ? "on" : "off") + ", AGL NetCDF=" +
            string(deck.culbm_output_agl_netcdf ? "on" : "off") + ", global VTKHDF=" +
            string(deck.culbm_output_global_vtkhdf ? "on" : "off"));
    }
}

std::vector<float> make_physical_output_ground_grid(const CaseContext& ctx, const uint nx, const uint ny,
                                                    const float3& global_origin, const float spacing_si) {
    const CaseSettings& deck = ctx.deck;
    const CaseTerrain& terrain = ctx.terrain;
    std::vector<float> ground;
    if (!deck.culbm_output_agl_requested) return ground;
    ground.assign((size_t)nx * (size_t)ny, 0.0f);
    parallel_for((ulong)nx * (ulong)ny, [&](const ulong id) {
        const uint x = (uint)(id % (ulong)nx);
        const uint y = (uint)(id / (ulong)nx);
        const float gx = global_origin.x + (float)x * spacing_si;
        const float gy = global_origin.y + (float)y * spacing_si;
        float gz = terrain.physical_output_ground_plane.eval_xy(gx, gy);
        if (!std::isfinite(gz)) gz = 0.0f;
        ground[(size_t)id] = gz;
    });
    return ground;
}

} // namespace culbm::cases
