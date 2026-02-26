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
#include <mutex>
#include <cctype>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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
uint purge_avg_steps = 0u;
ulong run_nstep_override = 0ull;
std::string validation_status = "fail"; 
bool enable_coriolis = false;       

constexpr float k_temperature_ref_kelvin = 293.15f; // 20 C -> T_lbm = 1.0
constexpr float k_temperature_min_kelvin = 223.15f; // -50 C
constexpr float k_temperature_max_kelvin = 343.15f; // 70 C
constexpr int k_patch_bottom = 0;
constexpr int k_patch_top = 1;
constexpr int k_patch_south = 2;
constexpr int k_patch_north = 3;
constexpr int k_patch_west = 4;
constexpr int k_patch_east = 5;

struct SamplePoint {
    float3 p;
    float3 u;
    float T = k_temperature_ref_kelvin;
    int patch = -1; // -1 means unspecified/legacy CSV
};
struct ProfileSample { float z; float u; };
struct DemCsvPoint { float x; float y; float elevation; };

class GroundTemperaturePlane2D {
public:
    void build_from_patch0_samples(const std::vector<SamplePoint>& samples, const float default_T) {
        x_raw_.clear();
        y_raw_.clear();
        t_raw_.clear();
        x_coords_.clear();
        y_coords_.clear();
        t_grid_.clear();
        structured_ready_ = false;
        default_T_ = default_T;

        for (const auto& s : samples) {
            if (s.patch != 0) continue;
            x_raw_.push_back(s.p.x);
            y_raw_.push_back(s.p.y);
            t_raw_.push_back(s.T);
        }

        if (t_raw_.empty()) return;

        double t_sum = 0.0;
        for (const float t : t_raw_) t_sum += (double)t;
        default_T_ = (float)(t_sum / (double)t_raw_.size());

        float xmin = x_raw_[0], xmax = x_raw_[0];
        float ymin = y_raw_[0], ymax = y_raw_[0];
        for (size_t i = 1u; i < x_raw_.size(); ++i) {
            xmin = fminf(xmin, x_raw_[i]);
            xmax = fmaxf(xmax, x_raw_[i]);
            ymin = fminf(ymin, y_raw_[i]);
            ymax = fmaxf(ymax, y_raw_[i]);
        }

        const float tol_x = std::max(1e-6f, 1e-6f * fmaxf(1.0f, xmax - xmin));
        const float tol_y = std::max(1e-6f, 1e-6f * fmaxf(1.0f, ymax - ymin));
        x_coords_ = unique_sorted_with_tol_(x_raw_, tol_x);
        y_coords_ = unique_sorted_with_tol_(y_raw_, tol_y);

        const size_t nx = x_coords_.size();
        const size_t ny = y_coords_.size();
        if (nx == 0u || ny == 0u) return;

        std::vector<double> t_sum_grid(nx * ny, 0.0);
        std::vector<uint> t_cnt_grid(nx * ny, 0u);

        for (size_t i = 0u; i < t_raw_.size(); ++i) {
            const size_t ix = nearest_index_(x_coords_, x_raw_[i]);
            const size_t iy = nearest_index_(y_coords_, y_raw_[i]);
            const size_t id = iy * nx + ix;
            t_sum_grid[id] += (double)t_raw_[i];
            t_cnt_grid[id] += 1u;
        }

        t_grid_.assign(nx * ny, default_T_);
        for (size_t id = 0u; id < t_grid_.size(); ++id) {
            if (t_cnt_grid[id] > 0u) {
                t_grid_[id] = (float)(t_sum_grid[id] / (double)t_cnt_grid[id]);
            }
        }

        structured_ready_ = (nx >= 2u && ny >= 2u);
    }

    bool has_samples() const { return !t_raw_.empty(); }
    bool structured_ready() const { return structured_ready_; }
    size_t raw_count() const { return t_raw_.size(); }
    size_t nx() const { return x_coords_.size(); }
    size_t ny() const { return y_coords_.size(); }

    float eval_xy(const float xq, const float yq) const {
        if (!has_samples()) return default_T_;
        if (!structured_ready_) return eval_nearest_raw_(xq, yq);

        const size_t nxv = x_coords_.size();
        const size_t nyv = y_coords_.size();
        if (nxv < 2u || nyv < 2u) return eval_nearest_raw_(xq, yq);

        const float x0 = x_coords_.front();
        const float x1 = x_coords_.back();
        const float y0 = y_coords_.front();
        const float y1 = y_coords_.back();
        const float x = fminf(fmaxf(xq, x0), x1);
        const float y = fminf(fmaxf(yq, y0), y1);

        const size_t ix1 = upper_index_(x_coords_, x);
        const size_t iy1 = upper_index_(y_coords_, y);
        const size_t ix0 = (ix1 == 0u) ? 0u : (ix1 - 1u);
        const size_t iy0 = (iy1 == 0u) ? 0u : (iy1 - 1u);
        const size_t ixa = (ix0 >= nxv - 1u) ? (nxv - 2u) : ix0;
        const size_t iya = (iy0 >= nyv - 1u) ? (nyv - 2u) : iy0;
        const size_t ixb = ixa + 1u;
        const size_t iyb = iya + 1u;

        const float xa = x_coords_[ixa];
        const float xb = x_coords_[ixb];
        const float ya = y_coords_[iya];
        const float yb = y_coords_[iyb];
        const float tx = (fabsf(xb - xa) > 1e-12f) ? ((x - xa) / (xb - xa)) : 0.0f;
        const float ty = (fabsf(yb - ya) > 1e-12f) ? ((y - ya) / (yb - ya)) : 0.0f;

        const float t00 = t_grid_[iya * nxv + ixa];
        const float t10 = t_grid_[iya * nxv + ixb];
        const float t01 = t_grid_[iyb * nxv + ixa];
        const float t11 = t_grid_[iyb * nxv + ixb];

        const float t0 = t00 + tx * (t10 - t00);
        const float t1 = t01 + tx * (t11 - t01);
        return t0 + ty * (t1 - t0);
    }

private:
    static std::vector<float> unique_sorted_with_tol_(std::vector<float> vals, const float tol) {
        if (vals.empty()) return vals;
        std::sort(vals.begin(), vals.end());
        std::vector<float> out;
        out.reserve(vals.size());
        float last = vals[0];
        out.push_back(last);
        for (size_t i = 1u; i < vals.size(); ++i) {
            const float v = vals[i];
            if (fabsf(v - last) > tol) {
                out.push_back(v);
                last = v;
            }
            else {
                // Keep representative coordinate centered inside tolerance cluster.
                out.back() = 0.5f * (out.back() + v);
                last = out.back();
            }
        }
        return out;
    }

    static size_t nearest_index_(const std::vector<float>& arr, const float v) {
        if (arr.empty()) return 0u;
        auto it = std::lower_bound(arr.begin(), arr.end(), v);
        if (it == arr.begin()) return 0u;
        if (it == arr.end()) return arr.size() - 1u;
        const size_t i1 = (size_t)(it - arr.begin());
        const size_t i0 = i1 - 1u;
        const float d0 = fabsf(v - arr[i0]);
        const float d1 = fabsf(v - arr[i1]);
        return (d1 < d0) ? i1 : i0;
    }

    static size_t upper_index_(const std::vector<float>& arr, const float v) {
        auto it = std::upper_bound(arr.begin(), arr.end(), v);
        if (it == arr.end()) return arr.size() - 1u;
        return (size_t)(it - arr.begin());
    }

    float eval_nearest_raw_(const float xq, const float yq) const {
        if (t_raw_.empty()) return default_T_;
        float best_d2 = FLT_MAX;
        float best_t = default_T_;
        for (size_t i = 0u; i < t_raw_.size(); ++i) {
            const float dx = xq - x_raw_[i];
            const float dy = yq - y_raw_[i];
            const float d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_t = t_raw_[i];
            }
        }
        return best_t;
    }

private:
    std::vector<float> x_raw_;
    std::vector<float> y_raw_;
    std::vector<float> t_raw_;
    std::vector<float> x_coords_;
    std::vector<float> y_coords_;
    std::vector<float> t_grid_;
    bool structured_ready_ = false;
    float default_T_ = 1.0f;
};

static const char* patch_name(const int patch) {
    switch (patch) {
    case k_patch_bottom: return "bottom";
    case k_patch_top:    return "top";
    case k_patch_south:  return "south";
    case k_patch_north:  return "north";
    case k_patch_west:   return "west";
    case k_patch_east:   return "east";
    default:             return "unknown";
    }
}

static int downstream_to_patch(const std::string& downstream_bc_local) {
    if (downstream_bc_local == "+y") return k_patch_north;
    if (downstream_bc_local == "-y") return k_patch_south;
    if (downstream_bc_local == "+x") return k_patch_east;
    if (downstream_bc_local == "-x") return k_patch_west;
    return -1;
}

static int boundary_cell_to_patch(const uint x, const uint y, const uint z, const uint Nx, const uint Ny, const uint Nz) {
    if (z == Nz - 1u) return k_patch_top;
    if (x == 0u) return k_patch_west;
    if (x == Nx - 1u) return k_patch_east;
    if (y == 0u) return k_patch_south;
    if (y == Ny - 1u) return k_patch_north;
    return -1;
}

static bool patch_surface_coordinates(const int patch, const float3& p, float& a, float& b) {
    switch (patch) {
    case k_patch_bottom:
    case k_patch_top:
        a = p.x;
        b = p.y;
        return true;
    case k_patch_south:
    case k_patch_north:
        a = p.x;
        b = p.z;
        return true;
    case k_patch_west:
    case k_patch_east:
        a = p.y;
        b = p.z;
        return true;
    default:
        break;
    }
    a = 0.0f;
    b = 0.0f;
    return false;
}

class PatchSurfaceField2D {
public:
    struct SurfaceSample {
        float a;
        float b;
        float3 v;
    };

    template <typename ValueFn>
    void build_from_patch(const std::vector<SamplePoint>& samples,
                          const int patch,
                          ValueFn value_fn,
                          const float3& default_value) {
        clear_(default_value);
        std::vector<SurfaceSample> raw;
        raw.reserve(samples.size());
        for (const auto& s : samples) {
            if (s.patch != patch) continue;
            float a = 0.0f;
            float b = 0.0f;
            if (!patch_surface_coordinates(patch, s.p, a, b)) continue;
            raw.push_back(SurfaceSample{ a, b, value_fn(s) });
        }
        if (raw.empty()) return;
        raw_count_ = raw.size();

        double sx = 0.0, sy = 0.0, sz = 0.0;
        float amin = raw[0].a, amax = raw[0].a;
        float bmin = raw[0].b, bmax = raw[0].b;
        for (const auto& rs : raw) {
            sx += (double)rs.v.x;
            sy += (double)rs.v.y;
            sz += (double)rs.v.z;
            amin = fminf(amin, rs.a);
            amax = fmaxf(amax, rs.a);
            bmin = fminf(bmin, rs.b);
            bmax = fmaxf(bmax, rs.b);
        }
        const double inv_n = 1.0 / (double)raw.size();
        default_value_ = float3((float)(sx * inv_n), (float)(sy * inv_n), (float)(sz * inv_n));

        const float tol_a = fmaxf(1e-6f, 1e-6f * fmaxf(1.0f, amax - amin));
        const float tol_b = fmaxf(1e-6f, 1e-6f * fmaxf(1.0f, bmax - bmin));

        std::sort(raw.begin(), raw.end(), [](const SurfaceSample& lhs, const SurfaceSample& rhs) {
            if (lhs.a < rhs.a) return true;
            if (lhs.a > rhs.a) return false;
            return lhs.b < rhs.b;
        });

        std::vector<std::vector<SurfaceSample>> grouped_cols;
        std::vector<double> col_a_sum;
        std::vector<uint> col_a_cnt;
        grouped_cols.reserve(raw.size());
        col_a_sum.reserve(raw.size());
        col_a_cnt.reserve(raw.size());
        for (const auto& rs : raw) {
            if (grouped_cols.empty()) {
                grouped_cols.push_back(std::vector<SurfaceSample>{ rs });
                col_a_sum.push_back((double)rs.a);
                col_a_cnt.push_back(1u);
                continue;
            }
            const size_t cid = grouped_cols.size() - 1u;
            const float a_rep = (float)(col_a_sum[cid] / (double)col_a_cnt[cid]);
            if (fabsf(rs.a - a_rep) <= tol_a) {
                grouped_cols[cid].push_back(rs);
                col_a_sum[cid] += (double)rs.a;
                col_a_cnt[cid] += 1u;
            }
            else {
                grouped_cols.push_back(std::vector<SurfaceSample>{ rs });
                col_a_sum.push_back((double)rs.a);
                col_a_cnt.push_back(1u);
            }
        }

        a_coords_.resize(grouped_cols.size());
        b_cols_.resize(grouped_cols.size());
        v_cols_.resize(grouped_cols.size());

        for (size_t c = 0u; c < grouped_cols.size(); ++c) {
            a_coords_[c] = (float)(col_a_sum[c] / (double)col_a_cnt[c]);
            auto& col = grouped_cols[c];
            std::sort(col.begin(), col.end(), [](const SurfaceSample& lhs, const SurfaceSample& rhs) {
                return lhs.b < rhs.b;
            });

            std::vector<float>& bvec = b_cols_[c];
            std::vector<float3>& vvec = v_cols_[c];
            bvec.reserve(col.size());
            vvec.reserve(col.size());
            std::vector<double> vsx;
            std::vector<double> vsy;
            std::vector<double> vsz;
            std::vector<uint> vcnt;
            vsx.reserve(col.size());
            vsy.reserve(col.size());
            vsz.reserve(col.size());
            vcnt.reserve(col.size());
            for (const auto& rs : col) {
                if (bvec.empty() || fabsf(rs.b - bvec.back()) > tol_b) {
                    bvec.push_back(rs.b);
                    vvec.push_back(rs.v);
                    vsx.push_back((double)rs.v.x);
                    vsy.push_back((double)rs.v.y);
                    vsz.push_back((double)rs.v.z);
                    vcnt.push_back(1u);
                }
                else {
                    const size_t i = bvec.size() - 1u;
                    bvec[i] = 0.5f * (bvec[i] + rs.b);
                    vsx[i] += (double)rs.v.x;
                    vsy[i] += (double)rs.v.y;
                    vsz[i] += (double)rs.v.z;
                    vcnt[i] += 1u;
                    const double inv = 1.0 / (double)vcnt[i];
                    vvec[i] = float3((float)(vsx[i] * inv), (float)(vsy[i] * inv), (float)(vsz[i] * inv));
                }
            }
        }
    }

    bool has_samples() const { return raw_count_ > 0u; }
    size_t raw_count() const { return raw_count_; }
    size_t column_count() const { return a_coords_.size(); }

    float3 eval(const float a, const float b) const {
        if (a_coords_.empty()) return default_value_;
        if (a_coords_.size() == 1u) return eval_column_(0u, b);

        size_t i0 = 0u;
        size_t i1 = 0u;
        if (a <= a_coords_.front()) {
            i0 = i1 = 0u;
        }
        else if (a >= a_coords_.back()) {
            i0 = i1 = a_coords_.size() - 1u;
        }
        else {
            auto it = std::upper_bound(a_coords_.begin(), a_coords_.end(), a);
            i1 = (size_t)(it - a_coords_.begin());
            i0 = i1 - 1u;
        }

        const float3 v0 = eval_column_(i0, b);
        if (i0 == i1) return v0;
        const float3 v1 = eval_column_(i1, b);
        const float a0 = a_coords_[i0];
        const float a1 = a_coords_[i1];
        const float t = (fabsf(a1 - a0) > 1e-12f) ? ((a - a0) / (a1 - a0)) : 0.0f;
        return float3(
            v0.x + t * (v1.x - v0.x),
            v0.y + t * (v1.y - v0.y),
            v0.z + t * (v1.z - v0.z));
    }

    bool below_sample_support(const float a, const float b, const float eps = 1e-4f) const {
        if (a_coords_.empty()) return false;
        if (a_coords_.size() == 1u) {
            if (b_cols_[0].empty()) return false;
            return b < (b_cols_[0].front() - eps);
        }

        size_t i0 = 0u;
        size_t i1 = 0u;
        if (a <= a_coords_.front()) {
            i0 = i1 = 0u;
        }
        else if (a >= a_coords_.back()) {
            i0 = i1 = a_coords_.size() - 1u;
        }
        else {
            auto it = std::upper_bound(a_coords_.begin(), a_coords_.end(), a);
            i1 = (size_t)(it - a_coords_.begin());
            i0 = i1 - 1u;
        }

        if (b_cols_[i0].empty()) return false;
        const float bmin0 = b_cols_[i0].front();
        float bmin = bmin0;
        if (i1 != i0) {
            if (b_cols_[i1].empty()) return false;
            const float bmin1 = b_cols_[i1].front();
            const float a0 = a_coords_[i0];
            const float a1 = a_coords_[i1];
            const float t = (fabsf(a1 - a0) > 1e-12f) ? ((a - a0) / (a1 - a0)) : 0.0f;
            bmin = bmin0 + t * (bmin1 - bmin0);
        }
        return b < (bmin - eps);
    }

private:
    void clear_(const float3& default_value) {
        raw_count_ = 0u;
        default_value_ = default_value;
        a_coords_.clear();
        b_cols_.clear();
        v_cols_.clear();
    }

    float3 eval_column_(const size_t col_id, const float b) const {
        if (col_id >= b_cols_.size() || col_id >= v_cols_.size()) return default_value_;
        const std::vector<float>& bvec = b_cols_[col_id];
        const std::vector<float3>& vvec = v_cols_[col_id];
        if (bvec.empty() || vvec.empty()) return default_value_;
        if (bvec.size() == 1u) return vvec[0];

        if (b <= bvec.front()) return vvec.front();
        if (b >= bvec.back()) return vvec.back();

        auto it = std::upper_bound(bvec.begin(), bvec.end(), b);
        size_t i1 = (size_t)(it - bvec.begin());
        size_t i0 = i1 - 1u;
        if (i1 >= bvec.size()) i1 = bvec.size() - 1u;
        const float b0 = bvec[i0];
        const float b1 = bvec[i1];
        const float t = (fabsf(b1 - b0) > 1e-12f) ? ((b - b0) / (b1 - b0)) : 0.0f;
        const float3& v0 = vvec[i0];
        const float3& v1 = vvec[i1];
        return float3(
            v0.x + t * (v1.x - v0.x),
            v0.y + t * (v1.y - v0.y),
            v0.z + t * (v1.z - v0.z));
    }

private:
    size_t raw_count_ = 0u;
    float3 default_value_ = float3(0.0f);
    std::vector<float> a_coords_;
    std::vector<std::vector<float>> b_cols_;
    std::vector<std::vector<float3>> v_cols_;
};

static unsigned detect_available_worker_threads_setup() {
#if defined(_WIN32)
    using GetActiveProcessorCountFn = DWORD(WINAPI*)(WORD);
    HMODULE kernel32 = GetModuleHandleA("kernel32.dll");
    if (kernel32 != nullptr) {
        auto get_active_processor_count =
            reinterpret_cast<GetActiveProcessorCountFn>(GetProcAddress(kernel32, "GetActiveProcessorCount"));
        if (get_active_processor_count != nullptr) {
            constexpr WORD all_processor_groups = 0xFFFFu;
            const DWORD win_hw = get_active_processor_count(all_processor_groups);
            if (win_hw > 0u) return static_cast<unsigned>(win_hw);
        }
    }
#endif
    const unsigned hw0 = std::thread::hardware_concurrency();
    return hw0 == 0u ? 4u : hw0;
}

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

// helper: read interpolated_dem.csv (columns x,y,elevation in SI units)
static std::vector<DemCsvPoint> read_interpolated_dem_csv(
    const string& path,
    float* x_min = nullptr, float* x_max = nullptr,
    float* y_min = nullptr, float* y_max = nullptr,
    float* elev_min = nullptr, float* elev_max = nullptr
) {
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


// helper: read SurfData.csv (columns X,Y,Z,u,v,w[,T][,patch] in SI units, with header)
static std::vector<SamplePoint> read_samples(const string& csv_path,
                                             bool* has_temperature_column = nullptr,
                                             ulong* rows_with_temperature = nullptr,
                                             float* temperature_min_si = nullptr,
                                             float* temperature_max_si = nullptr,
                                             bool* has_patch_column = nullptr,
                                             ulong* rows_with_patch = nullptr) {
    if (has_temperature_column) *has_temperature_column = false;
    if (rows_with_temperature) *rows_with_temperature = 0ul;
    if (temperature_min_si) *temperature_min_si = k_temperature_ref_kelvin;
    if (temperature_max_si) *temperature_max_si = k_temperature_ref_kelvin;
    if (has_patch_column) *has_patch_column = false;
    if (rows_with_patch) *rows_with_patch = 0ul;

    std::vector<SamplePoint> out;
    std::ifstream fin(csv_path);
    if (!fin.is_open()) {
        println("ERROR: could not open CSV " + csv_path);
        return out;
    }
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
    if (!std::getline(fin, header_line)) {
        println("WARNING: empty CSV " + csv_path);
        return out;
    }

    const std::vector<string> header_cols = split_csv(header_line);
    const int idx_x = find_col(header_cols, "x");
    const int idx_y = find_col(header_cols, "y");
    const int idx_z = find_col(header_cols, "z");
    const int idx_u = find_col(header_cols, "u");
    const int idx_v = find_col(header_cols, "v");
    const int idx_w = find_col(header_cols, "w");
    const int idx_t = find_col(header_cols, "t");
    const int idx_patch = find_col(header_cols, "patch");
    const bool use_named_columns =
        (idx_x >= 0) && (idx_y >= 0) && (idx_z >= 0) &&
        (idx_u >= 0) && (idx_v >= 0) && (idx_w >= 0);
    bool has_patch = (idx_patch >= 0);

    string line;
    ulong line_no = 1ul;
    bool has_temperature = false;
    ulong temp_rows = 0ul;
    ulong patch_rows = 0ul;
    float tmin = +FLT_MAX;
    float tmax = -FLT_MAX;
    while (std::getline(fin, line)) {
        line_no++;
        const std::vector<string> cols = split_csv(line);
        if (cols.empty()) continue;

        SamplePoint sp;

        if (use_named_columns) {
            const int required_max_idx = std::max(
                std::max(std::max(idx_x, idx_y), std::max(idx_z, idx_u)),
                std::max(idx_v, idx_w)
            );
            if ((int)cols.size() <= required_max_idx) {
                println("WARNING: malformed line " + to_string(line_no) + " in CSV (missing required columns)");
                continue;
            }

            sp.p = float3(
                (float)atof(cols[idx_x].c_str()),
                (float)atof(cols[idx_y].c_str()),
                (float)atof(cols[idx_z].c_str())
            );
            sp.u = float3(
                (float)atof(cols[idx_u].c_str()),
                (float)atof(cols[idx_v].c_str()),
                (float)atof(cols[idx_w].c_str())
            );

            if ((idx_t >= 0) && ((int)cols.size() > idx_t)) {
                sp.T = (float)atof(cols[idx_t].c_str());
                has_temperature = true;
                temp_rows++;
                tmin = fmin(tmin, sp.T);
                tmax = fmax(tmax, sp.T);
            }
            if ((idx_patch >= 0) && ((int)cols.size() > idx_patch)) {
                sp.patch = (int)std::lround((double)atof(cols[idx_patch].c_str()));
                has_patch = true;
                patch_rows++;
            }

            out.push_back(sp);
            continue;
        }

        // Fallback for legacy CSV with positional columns.
        std::stringstream ss(line);
        string token;
        float vals[8] = { 0.0f };
        int n_cols = 0;
        while (std::getline(ss, token, ',')) {
            if (n_cols < 8) vals[n_cols] = (float)atof(trim_copy(token).c_str());
            n_cols++;
        }
        if (n_cols < 6 || n_cols > 8) {
            println("WARNING: malformed line " + to_string(line_no) + " in CSV (expect 6~8 columns)");
            continue;
        }

        sp.p = float3(vals[0], vals[1], vals[2]);
        sp.u = float3(vals[3], vals[4], vals[5]);

        bool row_has_temperature = false;
        if (n_cols >= 8) {
            // X,Y,Z,u,v,w,T,patch
            sp.T = vals[6];
            row_has_temperature = true;
            sp.patch = (int)std::lround((double)vals[7]);
            has_patch = true;
            patch_rows++;
        } else if (n_cols == 7) {
            // Ambiguous: X,Y,Z,u,v,w,T or X,Y,Z,u,v,w,patch
            const float c7 = vals[6];
            const bool looks_like_patch = (c7 >= -0.5f) && (c7 <= 5.5f) && (fabsf(c7 - roundf(c7)) <= 1e-4f);
            if (!looks_like_patch) {
                sp.T = c7;
                row_has_temperature = true;
            }
            else {
                sp.patch = (int)std::lround((double)c7);
                has_patch = true;
                patch_rows++;
            }
        }

        if (row_has_temperature) {
            has_temperature = true;
            temp_rows++;
            tmin = fmin(tmin, sp.T);
            tmax = fmax(tmax, sp.T);
        }

        out.push_back(sp);
    }
    if (has_temperature_column) *has_temperature_column = has_temperature;
    if (rows_with_temperature) *rows_with_temperature = temp_rows;
    if (temperature_min_si) *temperature_min_si = has_temperature ? tmin : k_temperature_ref_kelvin;
    if (temperature_max_si) *temperature_max_si = has_temperature ? tmax : k_temperature_ref_kelvin;
    if (has_patch_column) *has_patch_column = has_patch;
    if (rows_with_patch) *rows_with_patch = patch_rows;
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
     ss << std::put_time(&tm, "%Y%m%d %T"); // YYYYMMDD hh:mm:ss
     return ss.str();
 }
static string now_stamp_str() {
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
    ss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return ss.str();
}
static string init_console_log_file(const string& project_dir) {
    if (project_dir.empty()) return "";
    std::filesystem::path log_dir = std::filesystem::path(project_dir) / "proj_temp";
    std::error_code ec;
    std::filesystem::create_directories(log_dir, ec);
    if (ec) return "";
    const std::string log_name = now_stamp_str() + "_lbm.log";
    const std::filesystem::path log_path = log_dir / log_name;
    return set_console_log_file(log_path.string()) ? log_path.string() : "";
}

static void write_avg_vtk(const string& filename, const uint Nx, const uint Ny, const uint Nz,
                          const float* u_avg, const float* rho_avg, const float* T_avg, const uchar* flags,
                          const float* M2_u, const float* M2_v, const float* M2_w,
                          const ulong avg_count, const bool convert_to_si_units,
                          const bool print_saved_message=true) {
    if (!u_avg || !rho_avg) return;
    float spacing = 1.0f;
    float u_factor = 1.0f;
    float rho_factor = 1.0f;
    float T_factor = 1.0f;
    float T_offset = 0.0f;
    if (convert_to_si_units) {
        spacing = units.si_x(1.0f);
        u_factor = units.si_u(1.0f);
        rho_factor = units.si_rho(1.0f);
        T_factor = units.si_dT(1.0f);
        T_offset = units.si_T(0.0f);
    }
    float3 origin = spacing * float3(0.5f - 0.5f * (float)Nx, 0.5f - 0.5f * (float)Ny, 0.5f - 0.5f * (float)Nz);
    if (convert_to_si_units) origin += vtk_origin_shift;
    const ulong points = (ulong)Nx * (ulong)Ny * (ulong)Nz;

    create_folder(filename);
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        println("ERROR: could not open " + filename + " for writing.");
        return;
    }

    const size_t slash = filename.find_last_of("/\\");
    const string base = (slash == string::npos) ? filename : filename.substr(slash + 1);
    const string header =
        "# vtk DataFile Version 3.0\nFluidX3D " + base + "\nBINARY\nDATASET STRUCTURED_POINTS\n"
        "DIMENSIONS " + to_string(Nx) + " " + to_string(Ny) + " " + to_string(Nz) + "\n"
        "ORIGIN " + to_string(origin.x) + " " + to_string(origin.y) + " " + to_string(origin.z) + "\n"
        "SPACING " + to_string(spacing) + " " + to_string(spacing) + " " + to_string(spacing) + "\n"
        "POINT_DATA " + to_string(points) + "\n";
    file.write(header.c_str(), header.length());

    auto write_field = [&](const string& name, const float* data, const uint components, const float factor, const float offset = 0.0f) {
        const string field_header =
            "SCALARS " + name + " float " + to_string(components) + "\nLOOKUP_TABLE default\n";
        file.write(field_header.c_str(), field_header.length());

        const uint threads = std::max(1u, (uint)std::thread::hardware_concurrency());
        const uint chunk_size_MB = 4u * threads;
        const ulong chunk_elements = std::max(1ull, (1048576ull * (ulong)chunk_size_MB) / ((ulong)components * sizeof(float)));
        std::vector<float> buffer(chunk_elements * (ulong)components);

        const ulong chunks = points / chunk_elements;
        const ulong remainder = points % chunk_elements;
        for (ulong c = 0u; c < chunks + 1ull; ++c) {
            const ulong count = (c < chunks) ? chunk_elements : remainder;
            if (count == 0ull) break;
            parallel_for(count, [&](ulong i) {
                const ulong idx = c * chunk_elements + i;
                const ulong base_i = i * (ulong)components;
                const ulong src_i = idx * (ulong)components;
                for (uint d = 0u; d < components; ++d) {
                    buffer[base_i + d] = reverse_bytes(data[src_i + d] * factor + offset);
                }
            });
            file.write(reinterpret_cast<const char*>(buffer.data()), count * (ulong)components * sizeof(float));
        }
    };

    write_field("u_avg", u_avg, 3u, u_factor);
    write_field("rho_avg", rho_avg, 1u, rho_factor);
    if (T_avg) {
        write_field("T_avg", T_avg, 1u, T_factor, T_offset);
    }
    std::vector<float> fluid(points, 1.0f);
    if (flags) {
        parallel_for(points, [&](ulong n) {
            fluid[n] = (flags[n] & TYPE_S) ? 0.0f : 1.0f;
        });
    }
    write_field("fluid", fluid.data(), 1u, 1.0f);

    std::vector<float> tke(points, 0.0f);
    if (avg_count > 1ull && M2_u && M2_v && M2_w) {
        const float inv_n = 1.0f / (float)avg_count;
        parallel_for(points, [&](ulong n) {
            const float var_u = M2_u[n] * inv_n;
            const float var_v = M2_v[n] * inv_n;
            const float var_w = M2_w[n] * inv_n;
            tke[n] = 0.5f * (var_u + var_v + var_w);
        });
    }
    write_field("tke", tke.data(), 1u, u_factor * u_factor);

    file.close();
    if (print_saved_message) {
        info.allow_printing.lock();
        print_info("File \"" + filename + "\" saved.");
        info.allow_printing.unlock();
    }
}

static float3 wind_velocity_lbmu(const float inflow_si, const float angle_deg, const float u_scale) {
    const float deg2rad = 3.14159265358979323846f / 180.0f;
    const float angle_rad = angle_deg * deg2rad;
    const float speed_lbmu = inflow_si * u_scale;
    return float3(-sinf(angle_rad) * speed_lbmu, -cosf(angle_rad) * speed_lbmu, 0.0f);
}

static string hr_plain() {
    return "|"+string((uint)CONSOLE_WIDTH-2u, '-')+"|";
}
static void print_section_title(const string& title) {
    println(hr_plain());
    println("|"+alignc((uint)CONSOLE_WIDTH-2u, title)+"|");
    println(hr_plain());
}
static void print_kv_row(const string& key, const string& value) {
    println("| "+key+" | "+value+" |");
}
static std::string read_line_console() {
    std::string out;
    std::cout.flush();
    std::getline(std::cin, out);
    return out;
}

void main_setup() {
    println(hr_plain());
    println("|"+alignc((uint)CONSOLE_WIDTH-2u, "LatticeUrbanWind LUW: Towards Micrometeorology Fastest Simulation")+"|");
    println("|"+alignc((uint)CONSOLE_WIDTH-2u, "Developed by Huanxia Wei's Team")+"|");
    println("|"+alignc((uint)CONSOLE_WIDTH-2u, "Version - v3.5-251119")+"|");
    println(hr_plain());
    info.print_logo();
 
    bool dataset_generation = false;
    bool profile_generation = false;
    std::vector<float> inflow_list;
    std::vector<float> angle_list;
    std::vector<float> profile_u_si;
    float z_limit_override = 0.0f;
    float z_limit_si = 0.0f;
    const float profile_dz_si = 0.1f;
    bool buoyancy_enabled = true; // default-on unless explicitly false/False/0
    bool buoyancy_explicit = false;
    bool datetime_from_config = false;
    bool has_cut_lon_manual = false;
    bool has_cut_lat_manual = false;

    // ---------------------- read *.luw / *.luwdg ------------------------------
    std::string parent; // project directory that contains the configure file
    {
        std::string luw_path_input;
        auto trim = [](std::string s) {
            const char* ws = " \t\r\n";
            size_t b = s.find_first_not_of(ws);
            size_t e = s.find_last_not_of(ws);
            return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
            };

        // Command-line mode: use the first argument as configure file path.
        if (!main_arguments.empty()) {
            luw_path_input = trim(main_arguments[0]);
            if (main_arguments.size() > 1u) {
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
                else if (key == "datetime") {
                    datetime = unquote(val);
                    datetime_from_config = true;
                }
                else if (key == "downstream_bc") downstream_bc = unquote(val);
                else if (key == "high_order") {
                    std::string v = unquote(val);
                    if (v == "true" || v == "1") use_high_order = true;
                }
                else if (key == "flux_correction") {
                    std::string v = unquote(val);
                    if (v == "true" || v == "1") flux_correction = true;
                }
                else if (key == "buoyancy") {
                    std::string v = unquote(val);
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    buoyancy_explicit = true;
                    if (v == "false" || v == "0") buoyancy_enabled = false;
                    else buoyancy_enabled = true; // keep default behavior for any non-false explicit value
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
                else if (key == "run_nstep") {
                    const long long v = atoll(unquote(val).c_str());
                    run_nstep_override = v > 0ll ? static_cast<ulong>(v) : 0ull;
                }
                else if (key == "purge_avg") {
                    const int v = atoi(val.c_str());
                    purge_avg_steps = v > 0 ? static_cast<uint>(v) : 0u;
                }
                else if (key == "validation") validation_status = unquote(val);
                else if (key == "coriolis_term") {
                    std::string v = unquote(val);
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    if (v == "true" || v == "1") enable_coriolis = true;
                }

                else if (key == "cut_lon_manual") {
                    parse_pair_float(val, cut_lon_min_deg, cut_lon_max_deg);
                    has_cut_lon_manual = true;
                }
                else if (key == "cut_lat_manual") {
                    parse_pair_float(val, cut_lat_min_deg, cut_lat_max_deg);
                    has_cut_lat_manual = true;
                }
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

    if (profile_generation && !datetime_from_config) {
        println("| Profile mode    | datetime not set in *.luwpf, use fallback " + datetime + " |");
    }
    if (profile_generation && enable_coriolis && !(has_cut_lon_manual && has_cut_lat_manual)) {
        println("| WARNING: coriolis_term=true but cut_lon_manual/cut_lat_manual is missing in *.luwpf. |");
        println("| WARNING: Coriolis is auto-disabled for Profile mode.                         |");
        enable_coriolis = false;
    }

    const std::string lbm_log_path = init_console_log_file(parent);
    if (!lbm_log_path.empty()) {
        println("| Console log     | " + lbm_log_path + " |");
    }
    else {
        println("| WARNING: failed to create proj_temp log file.                               |");
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

    println("|"+string((uint)CONSOLE_WIDTH-2u, ' ')+"|");
    print_section_title("PARAMETER INFORMATION");
 // println("| Grid Domains    | " + alignr(57u,to_string()) + " |");
    println("| Configure deck  | " + alignr(57u, to_string(conf_used_path)) + " |");
    println("| Casename / Time | " + alignr(40u, to_string(caseName)) + alignr(17u, to_string(datetime)) + " |");
    println("| Basement Height | " + alignr(55u, to_string(fmtf(z_si_offset))) + " m |");
    println("| SI Size (m)     | " + alignr(12u, " X:") + alignl(11u, to_string(fmtf(si_size.x))) + "   Y: " + alignl(11u, to_string(fmtf(si_size.y))) + "   Z: " + alignl(11u, to_string(fmtf(si_size.z))) + " | ");
    println("| Downstream BC   | " + alignr(57u, to_string(downstream_bc)) + " |");
    println("| Normal Yaw      | " + alignr(53u, to_string(downstream_bc_yaw)) + " deg |");
    println("| GPU Decompose   | " + alignr(49u, to_string(Dx)) + ", " + alignr(2u, to_string(Dy)) + ", " + alignr(2u, to_string(Dz)) + " |");
    println("| VRAM Request    | " + alignr(54u, to_string(memory)) + " MB |");
    println("| Run Steps       | " + alignr(57u, run_nstep_override > 0ull ? to_string(run_nstep_override) + " (run_nstep)" : string("20001 (default)")) + " |");
    println("| Purge Avg Steps | " + alignr(57u, purge_avg_steps > 0u ? to_string(purge_avg_steps) : string("off")) + " |");
    println("| Buoyancy Switch | " + alignr(57u, buoyancy_enabled ? (buoyancy_explicit ? string("true") : string("true (default)")) : string("false")) + " |");

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

    print_section_title("DOMAIN AND TRANSFORMATION");
    const ulong n_cells_total = (ulong)lbm_N.x*(ulong)lbm_N.y*(ulong)lbm_N.z;
    println("| Grid Resolution | " + alignr(45u, to_string(lbm_N.x)) + "," + alignr(5u, to_string(lbm_N.y)) + "," + alignr(5u, to_string(lbm_N.z)) + " (nCell = " + to_string(n_cells_total) + ") |");

    const uint mem_per_dev_mb = vram_required_mb_per_device(lbm_N.x, lbm_N.y, lbm_N.z, Dx, Dy, Dz);
    const uint mem_total_mb = vram_required_mb_total(lbm_N.x, lbm_N.y, lbm_N.z, Dx, Dy, Dz);
    std::vector<SamplePoint> samples_si;
    bool csv_has_temperature = false;
    ulong csv_rows_with_temperature = 0ul;
    float csv_temp_min_si = k_temperature_ref_kelvin;
    float csv_temp_max_si = k_temperature_ref_kelvin;
    bool csv_has_patch = false;
    ulong csv_rows_with_patch = 0ul;
    float temperature_ref_kelvin = k_temperature_ref_kelvin;
    float temperature_scale_kelvin = k_temperature_ref_kelvin;
    bool temperature_ref_adaptive = false;
    bool temperature_scale_adaptive = false;
    if (!dataset_generation && !profile_generation) {
        const std::string csv_path_ref = parent + "/proj_temp/SurfData_" + datetime + ".csv";
        samples_si = read_samples(
            csv_path_ref,
            &csv_has_temperature,
            &csv_rows_with_temperature,
            &csv_temp_min_si,
            &csv_temp_max_si,
            &csv_has_patch,
            &csv_rows_with_patch
        );
        if (samples_si.empty()) {
            println("| ERROR: no inlet samples when computing si_ref_u. Aborting...                |");
            println("|-----------------------------------------------------------------------------|");
            wait();
            exit(-1);
        }
        float max_u = 0.0f;
        for (const auto& s : samples_si) {
            const float speed = std::sqrt(
                s.u.x * s.u.x +
                s.u.y * s.u.y +
                s.u.z * s.u.z
            );
            if (speed > max_u) {
                max_u = speed;
            }
        }
        if (csv_has_temperature && csv_rows_with_temperature > 0ul) {
            float tmin = csv_temp_min_si;
            float tmax = csv_temp_max_si;
            if (tmin > tmax) std::swap(tmin, tmax);
            if (std::isfinite(tmin) && std::isfinite(tmax) && tmax > 0.0f) {
                const float tref = 0.5f * (tmin + tmax);
                if (std::isfinite(tref) && tref > 0.0f) {
                    temperature_ref_kelvin = tref;
                    temperature_ref_adaptive = true;
                }
                const float thalf = 0.5f * (tmax - tmin);
                if (std::isfinite(thalf) && thalf > 1.0e-6f) {
                    temperature_scale_kelvin = thalf;
                    temperature_scale_adaptive = true;
                }
                else {
                    temperature_scale_kelvin = 1.0f; // keep affine map well-conditioned even for nearly uniform input temperature
                    temperature_scale_adaptive = true;
                }
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

        if (z_limit_si > 1.0f && z_vals.back() <= 1.5f) {
            for (float& z : z_vals) {
                z *= z_limit_si;
            }
            println("| Profile z unit  | normalized -> scaled by z_limit                           |");
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

    units.set_m_kg_s_K((float)lbm_N.y, lbm_ref_u, 1.0f, 1.0f, si_size.y, si_ref_u, si_rho, temperature_scale_kelvin);
    units.set_temperature_reference(1.0f, temperature_ref_kelvin); // T_lbm=1.0 always maps to adaptive T_ref

    float u_scale = lbm_ref_u / si_ref_u;
    float z_off = units.x(z_si_offset);

    float lbm_nu = units.nu(si_nu);
    const float si_alpha_air = 2.10E-5f; // thermal diffusivity of air [m^2/s]
    const float si_beta_air = 1.0f / temperature_ref_kelvin; // volumetric thermal expansion [1/K]
    float lbm_alpha = units.alpha(si_alpha_air);
    float lbm_beta = buoyancy_enabled ? units.beta(si_beta_air) : 0.0f;
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
                print_kv_row("Coriolis", "enabled. center(lon,lat)=(" + to_string(center_lon_deg, 6u) + ", " + to_string(center_lat_deg, 6u) + ") deg");
                print_kv_row("", "Omega(lbmu)=(" + to_string(coriolis_Omegax_lbmu, 8u) + ", " + to_string(coriolis_Omegay_lbmu, 8u) + ", " + to_string(coriolis_Omegaz_lbmu, 8u) + ") per step");
            }

            const float omega_si = 7.292e-5f;           // Earth rotation [1/s]
            float f_si = 0.0f;
            if (cut_lat_min_deg != 0.0f || cut_lat_max_deg != 0.0f) {
                f_si = 2.0f * omega_si * std::sin(center_lat_deg * deg2rad);
            }
            coriolis_f_lbmu = f_si * dt_si;
            if (show_info) {
                print_kv_row("", "f_SI=" + to_string(f_si, 8u) + " 1/s, f_LBM=" + to_string(coriolis_f_lbmu, 8u));
            }
        }
        else if (show_info) {
            print_kv_row("Coriolis", "disabled by 'coriolis_term' setting in .luw");
        }
    };

    if (!dataset_generation && !profile_generation) {
        println("| SI Reference U  | " + alignl(7u, to_string(fmtf(si_ref_u))) + alignl(50u, "m/s") + " |");
        println("| LBM Reference U | " + alignl(7u, to_string(fmtf(lbm_ref_u))) + alignl(50u, "(Nondimensionalized)") + " |");
        if (temperature_ref_adaptive) {
            println("| Temp Reference  | " + alignr(57u, to_string(fmtf(temperature_ref_kelvin)) + " K (auto center of input Tmin/Tmax)") + " |");
        }
        else {
            println("| Temp Reference  | " + alignr(57u, to_string(fmtf(temperature_ref_kelvin)) + " K (default)") + " |");
        }
        if (temperature_scale_adaptive) {
            println("| Temp Scale      | " + alignr(57u, to_string(fmtf(temperature_scale_kelvin)) + " K per 1.0 T_lbm (auto from input range)") + " |");
        }
        else {
            println("| Temp Scale      | " + alignr(57u, to_string(fmtf(temperature_scale_kelvin)) + " K per 1.0 T_lbm (default)") + " |");
        }
        println("| Thermal alpha   | " + alignr(57u, to_string(lbm_alpha, 8u)) + " |");
        println("| Thermal tau_T   | " + alignr(57u, to_string(2.0f * lbm_alpha + 0.5f, 8u)) + " |");
        println("| Thermal beta    | " + alignr(57u, buoyancy_enabled ? to_string(lbm_beta, 8u) : string("0 (disabled by buoyancy=false)")) + " |");
        if (lbm_alpha < 1.0e-6f) {
            println("| WARNING         | thermal alpha is very small in LBM units; temperature diffusion may be stiff. |");
        }
    }

    update_coriolis(!dataset_generation && !profile_generation);
 
    std::vector<SamplePoint> samples;
    bool use_temperature_bc = false;
    float T_bc_min_lbm = 1.0f;
    float T_bc_max_lbm = 1.0f;
    if (!dataset_generation && !profile_generation) {
        if (samples_si.empty()) { println("ERROR: no inlet samples. Aborting."); wait(); exit(-1); }
        use_temperature_bc = buoyancy_enabled && csv_has_temperature;

        if (csv_has_temperature) {
            println("| T column        | detected (" + to_string(csv_rows_with_temperature) + " rows)                               |");
            println("| CSV T range SI  | " + alignr(24u, to_string(fmtf(csv_temp_min_si))) +
                    " to " + alignl(16u, to_string(fmtf(csv_temp_max_si))) + " K |");
            if (!buoyancy_enabled) {
                println("| Temperature BC  | buoyancy=false, ignore T column                                 |");
            }
            else {
                println("| Temperature BC  | enabled from CSV T (Kelvin -> nondimensionalized)               |");
            }
        }
        else {
            println("| T column        | not found, keep legacy velocity-only boundary behavior           |");
        }

        // convert samples to LBM units
        samples.reserve(samples_si.size());
        ulong out_of_temp_range = 0ul;
        for (const auto& s : samples_si) {
            SamplePoint sp;
            sp.p = float3(units.x(s.p.x), units.x(s.p.y), units.x(s.p.z));
            sp.u = s.u * u_scale;
            sp.patch = s.patch;
            if (use_temperature_bc) {
                sp.T = units.T(s.T); // Kelvin -> nondimensional temperature (adaptive affine scaling from Tmin/Tmax)
                if (s.T < k_temperature_min_kelvin || s.T > k_temperature_max_kelvin) out_of_temp_range++;
            }
            else {
                sp.T = 1.0f;
            }
            samples.push_back(sp);
        }
        if (use_temperature_bc && out_of_temp_range > 0ul) {
            println("| WARNING: " + to_string(out_of_temp_range) + " temperature samples are outside [-50C, 70C].                |");
        }
        if (use_temperature_bc) {
            T_bc_min_lbm = units.T(csv_temp_min_si);
            T_bc_max_lbm = units.T(csv_temp_max_si);
            if (T_bc_min_lbm > T_bc_max_lbm) std::swap(T_bc_min_lbm, T_bc_max_lbm);
        }
    }


	// ------------------------------- MESH LOADING -------------------------------
    print_section_title("LOADING GEOMETRY AND VOXELIZE");
    print_kv_row("Geometry", "Loading buildings as geometry, meshing...");

    std::vector<DemCsvPoint> profile_dem_points_raw;
    float profile_dem_xmin = 0.0f, profile_dem_xmax = 0.0f;
    float profile_dem_ymin = 0.0f, profile_dem_ymax = 0.0f;
    float profile_dem_emin = 0.0f, profile_dem_emax = 0.0f;
    bool profile_dem_loaded = false;

    std::string stl_path;
    {
        std::filesystem::path dir = std::filesystem::path(parent) / "proj_temp";
        if (!std::filesystem::exists(dir)) {
            println("ERROR: directory not found: " + dir.string());
            wait(); exit(-1);
        }
        const std::string prefix = caseName;

        const std::filesystem::path dem_pf_candidate = dir / (prefix + "_DEM_PF.stl");
        const std::filesystem::path dg_candidate = dir / (prefix + "_DG.stl");
        if (profile_generation &&
            std::filesystem::exists(dem_pf_candidate) &&
            std::filesystem::is_regular_file(dem_pf_candidate)) {
            stl_path = dem_pf_candidate.string();
        }
        else if (std::filesystem::exists(dg_candidate) && std::filesystem::is_regular_file(dg_candidate)) {
            stl_path = dg_candidate.string();
        }
        else if (profile_generation) {
            // Profile mode fallback: any *_DEM_PF.stl
            const std::string dem_pf_suffix = "_DEM_PF.stl";
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                const std::string fname = entry.path().filename().string();
                if (fname.size() >= dem_pf_suffix.size() &&
                    fname.substr(fname.size() - dem_pf_suffix.size()) == dem_pf_suffix) {
                    stl_path = entry.path().string();
                    break;
                }
            }
        }
        if (stl_path.empty()) {
            // Fallback to any *_DG.stl
            const std::string dg_suffix = "_DG.stl";
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                const std::string fname = entry.path().filename().string();
                if (fname.size() >= dg_suffix.size() &&
                    fname.substr(fname.size() - dg_suffix.size()) == dg_suffix) {
                    stl_path = entry.path().string();
                    break;
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
            if (profile_generation) {
                println("ERROR: no STL file. Tried " + dem_pf_candidate.string() + ", *_DEM_PF.stl, " +
                        dg_candidate.string() + ", *_DG.stl, then *.stl under " + dir.string());
            }
            else {
                println("ERROR: no STL file. Tried " + dg_candidate.string() + ", *_DG.stl, then *.stl under " + dir.string());
            }
            wait(); exit(-1);
        }

    }
    Mesh* mesh = read_stl(stl_path);


    print_kv_row("Time code", "[" + now_str() + "]");

    if (!mesh) { println("ERROR: failed to load STL"); wait(); exit(-1); }
    const float3 stl_size_si = mesh->get_bounding_box_size();
    const float3 stl_min_si = mesh->pmin;
    const float3 stl_max_si = mesh->pmax;
    const float3 domain_min_si = float3(
        units.si_x(0.5f - 0.5f * (float)lbm_N.x),
        units.si_x(0.5f - 0.5f * (float)lbm_N.y),
        units.si_x(0.5f - 0.5f * (float)lbm_N.z));
    vtk_origin_shift = stl_min_si - domain_min_si;
    const float target_lbm_x = units.x(si_size.x);
    const float scale_geom = target_lbm_x / stl_size_si.x;
    mesh->scale(scale_geom);
    mesh->translate(float3(1.0f - mesh->pmin.x, 1.0f - mesh->pmin.y, 1.0f - mesh->pmin.z));

    print_kv_row("Geometry STL", stl_path);
    print_kv_row("STL bounds SI", "x=[" + to_string(stl_min_si.x, 3u) + ", " + to_string(stl_max_si.x, 3u) +
            "], y=[" + to_string(stl_min_si.y, 3u) + ", " + to_string(stl_max_si.y, 3u) +
            "], z=[" + to_string(stl_min_si.z, 3u) + ", " + to_string(stl_max_si.z, 3u) + "]");
    print_kv_row("Geometry", "scaled by " + to_string(scale_geom, 4u) + ", ready for voxelization");

    if (profile_generation) {
        const std::string dem_csv_path = parent + "/proj_temp/interpolated_dem.csv";
        profile_dem_points_raw = read_interpolated_dem_csv(
            dem_csv_path,
            &profile_dem_xmin, &profile_dem_xmax,
            &profile_dem_ymin, &profile_dem_ymax,
            &profile_dem_emin, &profile_dem_emax
        );
        profile_dem_loaded = !profile_dem_points_raw.empty();
        if (profile_dem_loaded) {
            print_kv_row("Terrain DEM", "Loaded " + to_string((ulong)profile_dem_points_raw.size()) + " points from interpolated_dem.csv");
            print_kv_row("DEM bounds SI", "x=[" + to_string(profile_dem_xmin, 3u) + ", " + to_string(profile_dem_xmax, 3u) +
                    "], y=[" + to_string(profile_dem_ymin, 3u) + ", " + to_string(profile_dem_ymax, 3u) +
                    "], elev=[" + to_string(profile_dem_emin, 3u) + ", " + to_string(profile_dem_emax, 3u) + "]");
        }
        else {
            print_kv_row("Terrain DEM", "interpolated_dem.csv not found or empty, fallback to flat ground");
        }
    }

    auto run_lbm = [&](LBM& lbm, const std::string& vtk_prefix, const bool extra_spacing) {
        print_section_title("LBM SOLVER INFORMATION");

        lbm.graphics.visualization_modes = VIS_FLAG_SURFACE | VIS_Q_CRITERION;

        const ulong default_run_steps = 20001ull;
        const ulong base_run_steps = run_nstep_override > 0ull ? run_nstep_override : default_run_steps;
        const ulong extra_run_steps = research_output_steps > 0u ? (ulong)research_output_steps : 0ull;
        const ulong total_steps = base_run_steps + extra_run_steps;
        auto print_task_finished = [&]() {
            print_kv_row("Task finished", "[" + now_str() + "]");
            if (extra_spacing) {
                println("|"+string((uint)CONSOLE_WIDTH-2u, ' ')+"|");
            }
        };

        const std::string vtk_dir = parent + "/proj_temp/vtk/" + vtk_prefix + datetime + "_raw_";
        const std::string snapshots_dir = parent + "/proj_temp/snapshots";
        std::filesystem::create_directories(std::filesystem::path(vtk_dir).parent_path());
        std::filesystem::create_directories(snapshots_dir);
        if (use_temperature_bc) {
            print_kv_row("Export mode", "include temperature T field in Kelvin");
        }
        std::vector<string> vtk_saved_files;
        vtk_saved_files.reserve(16u);

        auto push_vtk_saved_file = [&](const string& filename) {
            vtk_saved_files.push_back(filename);
        };

        auto flush_vtk_saved_files = [&]() {
            if (vtk_saved_files.empty()) return;
            std::lock_guard<std::mutex> lock(info.allow_printing);
            print("\r");
            bool first = true;
            for (const string& filename : vtk_saved_files) {
                print_kv_row(first ? "VTK file" : "", filename + " saved");
                first = false;
            }
            vtk_saved_files.clear();
        };

        print_kv_row("Run steps", to_string(total_steps) + (run_nstep_override > 0ull ? " (run_nstep override)" : " (default)"));
        if (extra_run_steps > 0ull) {
            print_kv_row("Extra steps", to_string(extra_run_steps) + " (legacy research_output)");
        }

        const bool do_avg = purge_avg_steps > 0u;
        const ulong avg_window = do_avg ? std::min((ulong)purge_avg_steps, total_steps) : 0ull;
        const ulong avg_start_t = avg_window > 0ull ? (total_steps - avg_window + 1ull) : max_ulong;
        const ulong Ncells = lbm.get_N();
#ifdef TEMPERATURE
        const bool include_temperature_avg = use_temperature_bc;
#else
        const bool include_temperature_avg = false;
#endif

        std::vector<float> avg_u;
        std::vector<float> avg_rho;
        std::vector<float> avg_T;
        std::vector<float> M2_u;
        std::vector<float> M2_v;
        std::vector<float> M2_w;
        ulong avg_count = 0ull;
        if (avg_window > 0ull) {
            avg_u.assign((size_t)(Ncells * 3ull), 0.0f);
            avg_rho.assign((size_t)Ncells, 0.0f);
            if (include_temperature_avg) {
                avg_T.assign((size_t)Ncells, 0.0f);
            }
            M2_u.assign((size_t)Ncells, 0.0f);
            M2_v.assign((size_t)Ncells, 0.0f);
            M2_w.assign((size_t)Ncells, 0.0f);
        }

        auto enqueue_read_u_rho = [&]() {
#ifndef UPDATE_FIELDS
            for (uint d = 0u; d < lbm.get_D(); ++d) lbm.lbm_domain[d]->enqueue_update_fields();
#endif
            for (uint d = 0u; d < lbm.get_D(); ++d) {
                lbm.lbm_domain[d]->u.enqueue_read_from_device();
                lbm.lbm_domain[d]->rho.enqueue_read_from_device();
#ifdef TEMPERATURE
                if (include_temperature_avg) {
                    lbm.lbm_domain[d]->T.enqueue_read_from_device();
                }
#endif
            }
            for (uint d = 0u; d < lbm.get_D(); ++d) lbm.lbm_domain[d]->finish_queue();
        };

        auto accumulate_from_buffers = [&]() {
            ++avg_count;
            const float inv_n = 1.0f / (float)avg_count;
            float* u_avg = avg_u.data();
            float* rho_avg = avg_rho.data();
            float* m2_u = M2_u.data();
            float* m2_v = M2_v.data();
            float* m2_w = M2_w.data();
#ifdef TEMPERATURE
            float* t_avg = include_temperature_avg ? avg_T.data() : nullptr;
#endif
            parallel_for(Ncells, [&](ulong n) {
                const ulong i3 = 3ull * n;
                const float ux = lbm.u.x[n];
                const float uy = lbm.u.y[n];
                const float uz = lbm.u.z[n];
                float mean_u = u_avg[i3];
                float mean_v = u_avg[i3 + 1];
                float mean_w = u_avg[i3 + 2];

                const float delta_u = ux - mean_u;
                mean_u += delta_u * inv_n;
                const float delta2_u = ux - mean_u;
                m2_u[n] += delta_u * delta2_u;
                u_avg[i3] = mean_u;

                const float delta_v = uy - mean_v;
                mean_v += delta_v * inv_n;
                const float delta2_v = uy - mean_v;
                m2_v[n] += delta_v * delta2_v;
                u_avg[i3 + 1] = mean_v;

                const float delta_w = uz - mean_w;
                mean_w += delta_w * inv_n;
                const float delta2_w = uz - mean_w;
                m2_w[n] += delta_w * delta2_w;
                u_avg[i3 + 2] = mean_w;

                const float r = lbm.rho[n];
                rho_avg[n]    += (r - rho_avg[n]) * inv_n;
#ifdef TEMPERATURE
                if (t_avg) {
                    const float T = lbm.T[n];
                    t_avg[n] += (T - t_avg[n]) * inv_n;
                }
#endif
            });
        };

        auto maybe_accumulate_avg = [&]() {
            if (avg_window == 0ull) return;
            if (lbm.get_t() < avg_start_t) return;
            enqueue_read_u_rho();
            accumulate_from_buffers();
        };

        auto finalize_avg = [&]() {
            if (avg_count == 0ull) {
                flush_vtk_saved_files();
                return;
            }
            lbm.flags.read_from_device();
            std::vector<uchar> flags_host((size_t)Ncells, (uchar)0u);
            parallel_for(Ncells, [&](ulong n) {
                flags_host[(size_t)n] = lbm.flags[n];
            });
            const std::string avg_name = vtk_prefix + datetime + "_avg";
            const std::string avg_file = default_filename(parent + "/proj_temp/vtk/", avg_name, ".vtk", lbm.get_t());
#ifdef TEMPERATURE
            const float* T_avg_ptr = include_temperature_avg ? avg_T.data() : nullptr;
#else
            const float* T_avg_ptr = nullptr;
#endif
            write_avg_vtk(avg_file, lbm.get_Nx(), lbm.get_Ny(), lbm.get_Nz(),
                          avg_u.data(), avg_rho.data(), T_avg_ptr, flags_host.data(),
                          M2_u.data(), M2_v.data(), M2_w.data(),
                          avg_count, true, false);
            push_vtk_saved_file(avg_file);
            flush_vtk_saved_files();
            print_kv_row("Avg samples", to_string(avg_count));
        };

        auto write_final_transient = [&]() {
            const ulong vtk_t = lbm.get_t();
            push_vtk_saved_file(default_filename(vtk_dir, "u", ".vtk", vtk_t));
            lbm.u.write_device_to_vtk(vtk_dir, true, false);
            push_vtk_saved_file(default_filename(vtk_dir, "rho", ".vtk", vtk_t));
            lbm.rho.write_device_to_vtk(vtk_dir, true, false);
#ifdef TEMPERATURE
            if (use_temperature_bc) {
                push_vtk_saved_file(default_filename(vtk_dir, "T", ".vtk", vtk_t));
                lbm.T.write_device_to_vtk(vtk_dir, true, false);
            }
#endif
        };

        auto maybe_write_transform_info = [&]() {
            if (extra_run_steps == 0ull) return;
            println("| Writing transform.info...                                                  |");
            std::string info_path = parent + "/proj_temp/transform.info";
            std::ofstream info_file(info_path);
            if (info_file.is_open()) {
                const float dt_si = cell_m * (lbm_ref_u / si_ref_u);
                info_file << "dt = " << std::fixed << std::setprecision(10) << dt_si << "s\n";
                info_file.close();
                println("| Successfully wrote " + info_path + " |");
            }
            else {
                println("ERROR: Could not open " + info_path + " for writing.");
            }
        };

#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
        const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
        // Camera
        lbm.graphics.set_camera_free(
            float3(0.6f * Nx, -0.7f * Ny, 2.2f * Nz), // upstream & elevated
            -45.0f,                               // yaw
            30.0f,                                // pitch
            80.0f);                               // FOV

        lbm.run(0u, total_steps); // initialize fields and graphics buffers
        while (lbm.get_t() < total_steps) {
            // ------------------ off-screen PNG rendering (optional video) ------------------
            if (lbm.graphics.next_frame(total_steps, 1.0f)) {
                {
                    auto __old_cwd = std::filesystem::current_path();
                    std::filesystem::current_path(snapshots_dir);
                    lbm.graphics.write_frame();
                    std::filesystem::current_path(__old_cwd);
                }
            }
            lbm.run(1u, total_steps);
            maybe_accumulate_avg();
        }
        write_final_transient();
        maybe_write_transform_info();
        finalize_avg();
        print_task_finished();

#else // GRAPHICS + INTERACTIVE or pure CLI
        while (lbm.get_t() < total_steps) {
            lbm.run(1u, total_steps);
            maybe_accumulate_avg();
        }
        write_final_transient();
        maybe_write_transform_info();
        finalize_avg();
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
        print_section_title("DEVICE INFORMATION");
        LBM lbm(lbm_N, Dx, Dy, Dz, lbm_nu, 0.0f, 0.0f, 0.0f, 0.0f, lbm_alpha, lbm_beta);
        lbm.set_coriolis(coriolis_Omegax_lbmu, coriolis_Omegay_lbmu, coriolis_Omegaz_lbmu);

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
        print_section_title("BUILD BOUNDARY CONDITIONS");
        println("| CDF data loaded | " + alignl(57u, to_string(samples_si.size())) + " |");

        std::vector<float3> P; P.reserve(samples.size());
        std::vector<float3> Uv; Uv.reserve(samples.size());
        for (const auto& s : samples) { P.push_back(s.p); Uv.push_back(s.u); }
        std::vector<float3> P_temp;
        std::vector<float3> Tv;
        if (use_temperature_bc) {
            P_temp.reserve(samples.size());
            Tv.reserve(samples.size());
            for (const auto& s : samples) {
                P_temp.push_back(s.p);
                Tv.push_back(float3(s.T, 0.0f, 0.0f));
            }
        }

        const bool use_patch_face_bc = csv_has_patch;
        std::vector<PatchSurfaceField2D> patch_velocity_fields(6);
        std::vector<ulong> patch_counts(6, 0ul);
        if (use_patch_face_bc) {
            for (const auto& s : samples) {
                if (s.patch >= 0 && s.patch <= 5) patch_counts[(size_t)s.patch] += 1ul;
            }
            println("| Patch samples   | " + alignr(8u, string(patch_name(k_patch_bottom))) + " = " +
                    alignl(47u, to_string(patch_counts[(size_t)k_patch_bottom])) + " |");
            for (int patch = k_patch_top; patch <= k_patch_east; ++patch) {
                patch_velocity_fields[(size_t)patch].build_from_patch(
                    samples, patch,
                    [](const SamplePoint& s) { return s.u; },
                    float3(0.0f));
                println("|                 | " + alignr(8u, string(patch_name(patch))) + " = " +
                        alignl(47u, to_string(patch_counts[(size_t)patch])) + " |");
            }
        }

#ifdef TEMPERATURE
        std::vector<PatchSurfaceField2D> patch_temperature_fields(6);
        if (use_patch_face_bc && use_temperature_bc) {
            std::vector<ulong> t_patch_counts(6, 0ul);
            std::vector<float> t_patch_min(6, +FLT_MAX);
            std::vector<float> t_patch_max(6, -FLT_MAX);
            for (const auto& s : samples) {
                if (s.patch < 0 || s.patch > 5) continue;
                const size_t pid = (size_t)s.patch;
                t_patch_counts[pid]++;
                t_patch_min[pid] = fminf(t_patch_min[pid], s.T);
                t_patch_max[pid] = fmaxf(t_patch_max[pid], s.T);
            }
            for (int patch = k_patch_top; patch <= k_patch_east; ++patch) {
                patch_temperature_fields[(size_t)patch].build_from_patch(
                    samples, patch,
                    [](const SamplePoint& s) { return float3(s.T, 0.0f, 0.0f); },
                    float3(1.0f, 0.0f, 0.0f));
                if (t_patch_counts[(size_t)patch] > 0ul) {
                    println("| T patch         | " + string(patch_name(patch)) + ": n=" + to_string(t_patch_counts[(size_t)patch]) +
                            ", SI " + to_string(fmtf(units.si_T(t_patch_min[(size_t)patch]))) + " .. " +
                            to_string(fmtf(units.si_T(t_patch_max[(size_t)patch]))) + " K                    |");
                }
                else {
                    println("| T patch         | " + string(patch_name(patch)) + ": n=0                                           |");
                }
            }
        }
#endif

#ifdef TEMPERATURE
        GroundTemperaturePlane2D ground_temp_plane;
        bool use_ground_temperature_bc = false;
        if (use_temperature_bc && csv_has_patch) {
            ground_temp_plane.build_from_patch0_samples(samples, 1.0f);
            use_ground_temperature_bc = ground_temp_plane.has_samples();
            if (use_ground_temperature_bc) {
                println("| Ground T plane  | enabled from patch=0 (" + to_string(ground_temp_plane.raw_count()) +
                        " samples, grid " + to_string(ground_temp_plane.nx()) + "x" +
                        to_string(ground_temp_plane.ny()) + ", mode=" +
                        (ground_temp_plane.structured_ready() ? string("2D bilinear") : string("2D nearest")) + ") |");
            }
            else {
                println("| Ground T plane  | patch column detected, but no patch=0 samples found        |");
            }
        }
        auto apply_ground_temperature_plane_bc = [&](const string& order_tag) {
            if (!use_ground_temperature_bc) return;

            const uint Nx = lbm.get_Nx();
            const uint Ny = lbm.get_Ny();

            std::vector<float> xy_temp_cache((ulong)Nx * (ulong)Ny, std::numeric_limits<float>::quiet_NaN());
            ulong mapped_cells = 0ul;
            ulong unique_xy = 0ul;

            for (ulong n = 0ul; n < lbm.get_N(); ++n) {
                if ((lbm.flags[n] & TYPE_S) == 0u) continue; // solid only

                uint x = 0u, y = 0u, z = 0u;
                lbm.coordinates(n, x, y, z);
                (void)z;

                const ulong xyid = (ulong)y * (ulong)Nx + (ulong)x;
                float Txy = xy_temp_cache[xyid];
                if (!std::isfinite(Txy)) {
                    const float3 pos_xy = lbm.position(x, y, 0u); // ignore z by construction
                    Txy = ground_temp_plane.eval_xy(pos_xy.x, pos_xy.y);
                    if (use_temperature_bc) {
                        Txy = fminf(fmaxf(Txy, T_bc_min_lbm), T_bc_max_lbm);
                    }
                    xy_temp_cache[xyid] = Txy;
                    unique_xy++;
                }

                // Solid temperature uses ground patch 2D interpolation for the whole vertical column at (x,y).
                lbm.T[n] = Txy;
                lbm.flags[n] = (uchar)(lbm.flags[n] | TYPE_S | TYPE_T);
                mapped_cells++;
            }

            println("| Ground T plane  | mapped " + to_string(mapped_cells) +
                    " solid cells, unique (x,y)=" + to_string(unique_xy) +
                    " [" + order_tag + "]                                |");
            if (mapped_cells == 0ul) {
                println("| Ground T plane  | WARNING: no solid cells were found                          |");
            }
        };

        auto report_temperature_bc_summary = [&](const string& tag) {
            if (!use_temperature_bc) return;
            ulong type_t_total = 0ul;
            ulong type_t_solid = 0ul;
            ulong type_t_fluid = 0ul;
            ulong type_t_invalid = 0ul;
            float t_solid_min = +FLT_MAX, t_solid_max = -FLT_MAX;
            float t_fluid_min = +FLT_MAX, t_fluid_max = -FLT_MAX;
            for (ulong n = 0ul; n < lbm.get_N(); ++n) {
                if ((lbm.flags[n] & TYPE_T) == 0u) continue;
                type_t_total++;
                const float Tn = lbm.T[n];
                if (!std::isfinite(Tn)) {
                    type_t_invalid++;
                    continue;
                }
                if ((lbm.flags[n] & TYPE_S) != 0u) {
                    type_t_solid++;
                    t_solid_min = fminf(t_solid_min, Tn);
                    t_solid_max = fmaxf(t_solid_max, Tn);
                }
                else {
                    type_t_fluid++;
                    t_fluid_min = fminf(t_fluid_min, Tn);
                    t_fluid_max = fmaxf(t_fluid_max, Tn);
                }
            }

            println("| Temperature BC  | summary [" + tag + "]: TYPE_T total=" + to_string(type_t_total) +
                    ", solid=" + to_string(type_t_solid) + ", fluid=" + to_string(type_t_fluid) + "            |");
            if (type_t_solid > 0ul) {
                println("| Temperature BC  | solid TYPE_T range SI: " +
                        to_string(fmtf(units.si_T(t_solid_min))) + " .. " +
                        to_string(fmtf(units.si_T(t_solid_max))) + " K                      |");
            }
            if (type_t_fluid > 0ul) {
                println("| Temperature BC  | fluid TYPE_T range SI: " +
                        to_string(fmtf(units.si_T(t_fluid_min))) + " .. " +
                        to_string(fmtf(units.si_T(t_fluid_max))) + " K                      |");
            }
            if (type_t_invalid > 0ul) {
                println("| Temperature BC  | WARNING: non-finite TYPE_T cells = " + to_string(type_t_invalid) + "                         |");
            }
        };
#endif // TEMPERATURE

        if (use_patch_face_bc) {
            const uint Nx = lbm.get_Nx();
            const uint Ny = lbm.get_Ny();
            const uint Nz = lbm.get_Nz();
            const int downstream_patch = downstream_to_patch(downstream_bc);
            std::vector<uchar> voxel_flags((size_t)lbm.get_N());
            for (ulong n = 0ul; n < lbm.get_N(); ++n) {
                voxel_flags[(size_t)n] = lbm.flags[n];
            }

            auto set_zero_velocity = [&](const ulong n) {
                lbm.u.x[n] = 0.0f;
                lbm.u.y[n] = 0.0f;
                lbm.u.z[n] = 0.0f;
            };

            // Ensure cells below terrain are always solid (robust against tiny STL/domain mismatch gaps).
            PatchSurfaceField2D ground_height_field;
            ground_height_field.build_from_patch(
                samples, k_patch_bottom,
                [](const SamplePoint& s) { return float3(s.p.z, 0.0f, 0.0f); },
                float3(z0_lbmu, 0.0f, 0.0f));
            if (ground_height_field.has_samples()) {
                std::atomic<ulong> terrain_clipped_to_solid{ 0ul };
                parallel_for(lbm.get_N(), [&](ulong n) {
                    if ((lbm.flags[n] & TYPE_S) != 0u) return;
                    uint x = 0u, y = 0u, z = 0u;
                    lbm.coordinates(n, x, y, z);
                    const float3 pos = lbm.position(x, y, z);
                    const float zg = ground_height_field.eval(pos.x, pos.y).x;
                    if (pos.z < zg) {
                        lbm.flags[n] = TYPE_S;
                        set_zero_velocity(n);
                        terrain_clipped_to_solid.fetch_add(1ul, std::memory_order_relaxed);
                    }
                });
                if (terrain_clipped_to_solid.load(std::memory_order_relaxed) > 0ul) {
                    println("| Terrain clip    | below-terrain cells forced to solid: " +
                            to_string(terrain_clipped_to_solid.load()) + "                    |");
                }
                for (ulong n = 0ul; n < lbm.get_N(); ++n) {
                    voxel_flags[(size_t)n] = lbm.flags[n];
                }
            }

            auto voxel_is_solid = [&](const uint xi, const uint yi, const uint zi) -> bool {
                return (voxel_flags[(size_t)lbm.index(xi, yi, zi)] & TYPE_S) != 0u;
            };

            std::atomic<ulong> velocity_mapped{ 0ul };
            std::atomic<ulong> velocity_missing_patch{ 0ul };
            std::atomic<ulong> outlet_cells{ 0ul };
            std::atomic<ulong> velocity_grounded{ 0ul };
            std::atomic<ulong> velocity_below_patch_support{ 0ul };
            parallel_for(lbm.get_N(), [&](ulong n) {
                uint x = 0u, y = 0u, z = 0u;
                lbm.coordinates(n, x, y, z);
                if (z == 0u) {
                    lbm.flags[n] = TYPE_S;
                    set_zero_velocity(n);
                    return;
                }

                const int patch = boundary_cell_to_patch(x, y, z, Nx, Ny, Nz);
                if (patch < 0) return;

                const bool voxel_self_solid = (voxel_flags[(size_t)n] & TYPE_S) != 0u;
                bool side_below_ground = false;
                if (patch == k_patch_west && Nx > 1u) {
                    side_below_ground = voxel_is_solid(1u, y, z);
                }
                else if (patch == k_patch_east && Nx > 1u) {
                    side_below_ground = voxel_is_solid(Nx - 2u, y, z);
                }
                else if (patch == k_patch_south && Ny > 1u) {
                    side_below_ground = voxel_is_solid(x, 1u, z);
                }
                else if (patch == k_patch_north && Ny > 1u) {
                    side_below_ground = voxel_is_solid(x, Ny - 2u, z);
                }

                if (voxel_self_solid || side_below_ground) {
                    lbm.flags[n] = TYPE_S;
                    set_zero_velocity(n);
                    velocity_grounded.fetch_add(1ul, std::memory_order_relaxed);
                    return;
                }

                const PatchSurfaceField2D& face_field = patch_velocity_fields[(size_t)patch];
                if (!face_field.has_samples()) {
                    velocity_missing_patch.fetch_add(1ul, std::memory_order_relaxed);
                    return;
                }

                const float3 pos = lbm.position(x, y, z);
                float a = 0.0f;
                float b = 0.0f;
                if (!patch_surface_coordinates(patch, pos, a, b)) {
                    velocity_missing_patch.fetch_add(1ul, std::memory_order_relaxed);
                    return;
                }

                const bool is_side_patch =
                    (patch == k_patch_west || patch == k_patch_east ||
                     patch == k_patch_south || patch == k_patch_north);
                if (is_side_patch && face_field.below_sample_support(a, b)) {
                    lbm.flags[n] = TYPE_S;
                    set_zero_velocity(n);
                    velocity_below_patch_support.fetch_add(1ul, std::memory_order_relaxed);
                    return;
                }

                lbm.flags[n] = TYPE_E;
                if (patch == downstream_patch) {
                    outlet_cells.fetch_add(1ul, std::memory_order_relaxed);
                    return;
                }

                const float3 u = face_field.eval(a, b);
                lbm.u.x[n] = u.x;
                lbm.u.y[n] = u.y;
                lbm.u.z[n] = u.z;
                velocity_mapped.fetch_add(1ul, std::memory_order_relaxed);
            });
            println("| Velocity BC     | patch-driven 2D mapping: " + to_string(velocity_mapped.load()) + " cells                 |");
            if (velocity_grounded.load(std::memory_order_relaxed) > 0ul) {
                println("|                 | underground no-slip cells: " + to_string(velocity_grounded.load()) + "                     |");
            }
            if (velocity_below_patch_support.load(std::memory_order_relaxed) > 0ul) {
                println("|                 | side cells below terrain support -> solid: " + to_string(velocity_below_patch_support.load()) + "     |");
            }
            if (outlet_cells.load(std::memory_order_relaxed) > 0ul) {
                println("|                 | downstream outlet cells: " + to_string(outlet_cells.load()) + " (no fixed velocity)        |");
            }
            if (velocity_missing_patch.load(std::memory_order_relaxed) > 0ul) {
                println("|                 | WARNING: missing patch samples for " + to_string(velocity_missing_patch.load()) + " cells         |");
            }

#ifdef TEMPERATURE
            if (use_temperature_bc) {
                std::atomic<ulong> temperature_mapped{ 0ul };
                std::atomic<ulong> temperature_missing_patch{ 0ul };
                parallel_for(lbm.get_N(), [&](ulong n) {
                    uint x = 0u, y = 0u, z = 0u;
                    lbm.coordinates(n, x, y, z);
                    if (z == 0u) return;

                    const int patch = boundary_cell_to_patch(x, y, z, Nx, Ny, Nz);
                    if (patch < 0) return;
                    if ((lbm.flags[n] & TYPE_S) != 0u) return; // solid temperature is handled by ground patch plane

                    const PatchSurfaceField2D& face_field_t = patch_temperature_fields[(size_t)patch];
                    if (!face_field_t.has_samples()) {
                        temperature_missing_patch.fetch_add(1ul, std::memory_order_relaxed);
                        return;
                    }

                    const float3 pos = lbm.position(x, y, z);
                    float a = 0.0f;
                    float b = 0.0f;
                    if (!patch_surface_coordinates(patch, pos, a, b)) {
                        temperature_missing_patch.fetch_add(1ul, std::memory_order_relaxed);
                        return;
                    }
                    float Tn = face_field_t.eval(a, b).x;
                    if (use_temperature_bc) {
                        Tn = fminf(fmaxf(Tn, T_bc_min_lbm), T_bc_max_lbm);
                    }
                    lbm.T[n] = Tn;
                    lbm.flags[n] = (uchar)(lbm.flags[n] | TYPE_T);
                    temperature_mapped.fetch_add(1ul, std::memory_order_relaxed);
                });
                println("| Temperature BC  | patch-driven 2D mapping: " + to_string(temperature_mapped.load()) + " cells              |");
                if (temperature_missing_patch.load(std::memory_order_relaxed) > 0ul) {
                    println("|                 | WARNING: missing patch samples for " + to_string(temperature_missing_patch.load()) + " cells      |");
                }
            }
            apply_ground_temperature_plane_bc("patch-2d");
            report_temperature_bc_summary("patch-2d");
#endif // TEMPERATURE
            print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

            if (flux_correction) {
                print_kv_row("Flux correction", "starting. Time: [" + now_str() + "]");
                double avg_du = 0.0, net_b = 0.0, net_a = 0.0;
                auto eval = [&](const float3& q)->float3 {
                    if (downstream_patch < k_patch_top || downstream_patch > k_patch_east) return float3(0.0f);
                    const PatchSurfaceField2D& f = patch_velocity_fields[(size_t)downstream_patch];
                    if (!f.has_samples()) return float3(0.0f);
                    float a = 0.0f;
                    float b = 0.0f;
                    if (!patch_surface_coordinates(downstream_patch, q, a, b)) return float3(0.0f);
                    return f.eval(a, b);
                };
                apply_flux_correction(lbm, downstream_bc, eval, /*show_report=*/true,
                    &avg_du, &net_b, &net_a);
#ifdef TEMPERATURE
                if (use_temperature_bc) report_temperature_bc_summary("patch-2d/post-flux");
#endif
            }
            else {
                print_kv_row("Flux correction", "skipped. Set flux_correction=true to enable");
            }
        }
        else if (use_high_order) {
            KNNInterpolatorHD knn_hd(std::move(P), std::move(Uv));
            // Calculate z base threshold in LBM units
            const float z_base_threshold_lbmu = units.x(z_si_offset) + z0_lbmu;
            InletVelocityFieldHD inlet_hd(knn_hd, z_base_threshold_lbmu);
            apply_inlet_outlet_hd(lbm, downstream_bc, inlet_hd);

#ifdef TEMPERATURE
            if (use_temperature_bc) {
                KNNInterpolatorHD knn_hd_T(std::move(P_temp), std::move(Tv));
                const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
                std::vector<ulong> temp_boundary_indices;
                temp_boundary_indices.reserve((ulong)Nx * (ulong)Ny + (ulong)Nx * (ulong)Nz + (ulong)Ny * (ulong)Nz);
                for (ulong n = 0ul; n < lbm.get_N(); ++n) {
                    uint x = 0u, y = 0u, z = 0u;
                    lbm.coordinates(n, x, y, z);
                    if (z == 0u) continue;
                    const bool on_outer = (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u);
                    if (on_outer) temp_boundary_indices.push_back(n);
                }

                const ulong Nbc = (ulong)temp_boundary_indices.size();
                println("| [" + now_str() + "] temperature BC init (HD): " + to_string(Nbc) + " cells.                    |");
                if (Nbc > 0ul) {
                    unsigned num_threads = detect_available_worker_threads_setup();
#ifdef _MSC_VER
                    char* env_dup = nullptr;
                    size_t env_len = 0;
                    if (_dupenv_s(&env_dup, &env_len, "LBM_NUM_THREADS") == 0 && env_dup != nullptr) {
                        const long v = std::strtol(env_dup, nullptr, 10);
                        if (v > 0) num_threads = (unsigned)std::min<long>(v, 512l);
                        std::free(env_dup);
                    }
#else
                    if (const char* env_p = std::getenv("LBM_NUM_THREADS")) {
                        const long v = std::strtol(env_p, nullptr, 10);
                        if (v > 0) num_threads = (unsigned)std::min<long>(v, 512l);
                    }
#endif
                    println("| Temperature BC  | worker threads = " + to_string(num_threads) + "                                   |");
                    // Split temperature boundary cells into 5 faces, same priority/order as HD velocity mapping.
                    std::vector<std::vector<ulong>> face_indices(5);
                    for (int f = 0; f < 5; ++f) {
                        face_indices[f].reserve(Nbc / 5ul + 1ul);
                    }
                    for (ulong i = 0ul; i < Nbc; ++i) {
                        const ulong n = temp_boundary_indices[i];
                        uint x = 0u, y = 0u, z = 0u;
                        lbm.coordinates(n, x, y, z);
                        int face = -1;
                        if (x == 0u) face = 0;
                        else if (x == Nx - 1u) face = 1;
                        else if (y == 0u) face = 2;
                        else if (y == Ny - 1u) face = 3;
                        else if (z == Nz - 1u) face = 4;
                        if (face >= 0) face_indices[face].push_back(n);
                    }

                    ulong mapped_total = 0ul;
                    for (int f = 0; f < 5; ++f) {
                        const std::vector<ulong>& fi = face_indices[f];
                        const ulong Nface = (ulong)fi.size();
                        if (Nface == 0ul) continue;

                        const char* face_name = "unknown";
                        if (f == 0) face_name = "x == 0";
                        else if (f == 1) face_name = "x == Nx - 1";
                        else if (f == 2) face_name = "y == 0";
                        else if (f == 3) face_name = "y == Ny - 1";
                        else if (f == 4) face_name = "z == Nz - 1";

                        println("| [" + now_str() + "] temperature BC init (HD) surface " + to_string(f + 1) +
                                "/5 (" + string(face_name) + "): " + to_string(Nface) + " cells");

                        const ulong chunk = std::max<ulong>(1024ul, Nface / ((ulong)num_threads * 32ul) + 1ul);
                        std::atomic<ulong> next_idx{ 0ul };
                        std::atomic<ulong> processed{ 0ul };
                        std::atomic<bool> done{ false };

                        std::thread progress_thread([&]() {
                            while (!done.load(std::memory_order_relaxed)) {
                                const double pct = 100.0 * (double)processed.load(std::memory_order_relaxed) / (double)Nface;
                                reprint("| [" + now_str() + "] temperature BC init (HD): " + to_string((float)pct, 3u) + "% |");
                                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                            }
                            println("| [" + now_str() + "] temperature BC init (HD): 100.000% |");
                        });

                        std::vector<std::thread> workers;
                        workers.reserve(num_threads);
                        for (unsigned t = 0u; t < num_threads; ++t) {
                            workers.emplace_back([&]() {
                                ulong local_done = 0ul;
                                for (;;) {
                                    const ulong start = next_idx.fetch_add(chunk, std::memory_order_relaxed);
                                    if (start >= Nface) break;
                                    const ulong end = std::min<ulong>(start + chunk, Nface);
                                    for (ulong i = start; i < end; ++i) {
                                        const ulong n = fi[i];
                                        uint x = 0u, y = 0u, z = 0u;
                                        lbm.coordinates(n, x, y, z);
                                        const float3 pos = lbm.position(x, y, z);
                                        float Tn = (pos.z < z_base_threshold_lbmu) ? 1.0f : knn_hd_T.eval(pos).x;
                                        if (use_temperature_bc) {
                                            Tn = fminf(fmaxf(Tn, T_bc_min_lbm), T_bc_max_lbm);
                                        }
                                        lbm.T[n] = Tn;
                                        lbm.flags[n] = (uchar)(lbm.flags[n] | TYPE_T);
                                        if (++local_done == 512ul) {
                                            processed.fetch_add(local_done, std::memory_order_relaxed);
                                            local_done = 0ul;
                                        }
                                    }
                                }
                                if (local_done) processed.fetch_add(local_done, std::memory_order_relaxed);
                            });
                        }
                        for (auto& th : workers) th.join();
                        done.store(true, std::memory_order_relaxed);
                        if (progress_thread.joinable()) progress_thread.join();
                        mapped_total += Nface;
                    }
                    println("| Temperature BC  | per-face interpolation done on 5 boundary surfaces        |");
                    println("| Temperature BC  | mapped " + to_string(mapped_total) + "/" + to_string(Nbc) + " cells (high-order)      |");
                }
            }
            apply_ground_temperature_plane_bc("high-order");
            report_temperature_bc_summary("high-order");
#endif // TEMPERATURE
            print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

            if (flux_correction) {
                print_kv_row("Flux correction", "starting. Time: [" + now_str() + "]");
                double avg_du = 0.0, net_b = 0.0, net_a = 0.0;
                auto eval = [&](const float3& q)->float3 { return inlet_hd(q); };
                apply_flux_correction(lbm, downstream_bc, eval, /*show_report=*/true,
                    &avg_du, &net_b, &net_a);
#ifdef TEMPERATURE
                if (use_temperature_bc) report_temperature_bc_summary("high-order/post-flux");
#endif
                // std::fprintf(stdout, "[flux] check: net_before=%.9e, net_after=%.9e, avg|Ku|=%.9e m/s\n", net_b, net_a, avg_du);
            }
            else {
                print_kv_row("Flux correction", "skipped. Set flux_correction=true to enable");
            }
        }
        else {
            NearestNeighborInterpolator nn(std::move(P), std::move(Uv));
            InletVelocityField inlet(nn, z0_lbmu, z_off);
            apply_inlet_outlet(lbm, downstream_bc, inlet);

#ifdef TEMPERATURE
            if (use_temperature_bc) {
                NearestNeighborInterpolator nn_T(std::move(P_temp), std::move(Tv));
                std::atomic<ulong> temperature_mapped{ 0ul };
                const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
                const float z_temperature_threshold_lbmu = z0_lbmu + z_off;
                parallel_for(lbm.get_N(), [&](ulong n) {
                    uint x = 0u, y = 0u, z = 0u;
                    lbm.coordinates(n, x, y, z);
                    if (z == 0u) return;

                    bool outlet = false;
                    if (downstream_bc == "+y")      outlet = (y == Ny - 1u);
                    else if (downstream_bc == "-y") outlet = (y == 0u);
                    else if (downstream_bc == "+x") outlet = (x == Nx - 1u);
                    else if (downstream_bc == "-x") outlet = (x == 0u);

                    const bool inlet_face =
                        ((x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u) && !outlet);
                    if (!inlet_face) return;

                    const float3 pos = lbm.position(x, y, z);
                    float Tn = (pos.z < z_temperature_threshold_lbmu) ? 1.0f : nn_T.eval(pos).x;
                    if (use_temperature_bc) {
                        Tn = fminf(fmaxf(Tn, T_bc_min_lbm), T_bc_max_lbm);
                    }
                    lbm.T[n] = Tn;
                    lbm.flags[n] = (uchar)(lbm.flags[n] | TYPE_T);
                    temperature_mapped.fetch_add(1ul, std::memory_order_relaxed);
                });
                println("| Temperature BC  | mapped " + to_string(temperature_mapped.load()) + " cells (low-order)                     |");
            }
            apply_ground_temperature_plane_bc("low-order");
            report_temperature_bc_summary("low-order");
#endif // TEMPERATURE
            print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

            if (flux_correction) {
                print_kv_row("Flux correction", "starting. Time: [" + now_str() + "]");
                double avg_du = 0.0, net_b = 0.0, net_a = 0.0;
                auto eval = [&](const float3& q)->float3 { return inlet(q); };
                apply_flux_correction(lbm, downstream_bc, eval, /*show_report=*/true,
                    &avg_du, &net_b, &net_a);
#ifdef TEMPERATURE
                if (use_temperature_bc) report_temperature_bc_summary("low-order/post-flux");
#endif
            }
            else {
                print_kv_row("Flux correction", "skipped. Set flux_correction=true to enable");
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
                units.set_m_kg_s_K((float)lbm_N.y, lbm_ref_u, 1.0f, 1.0f, si_size.y, si_ref_u, si_rho, temperature_scale_kelvin);
                units.set_temperature_reference(1.0f, temperature_ref_kelvin);
                z_off = units.x(z_si_offset);
                lbm_nu = units.nu(si_nu);
                lbm_alpha = units.alpha(si_alpha_air);
                lbm_beta = buoyancy_enabled ? units.beta(si_beta_air) : 0.0f;
                update_coriolis(false);

                const std::string case_label = to_string(case_index) + "/" + to_string(total_cases) +
                    " (remaining " + to_string(remaining) + ")";
                println("|-----------------------------------------------------------------------------|");
                println("| Dataset case    | " + alignr(57u, case_label) + " |");
                println("| Inflow / Angle  | " + alignr(57u, format_tag(inflow_si) + " m/s, " + format_tag(angle_deg) + " deg") + " |");
                println("| SI Reference U  | " + alignr(57u, format_tag(si_ref_u) + " m/s") + " |");

                const float3 inflow_lbmu = wind_velocity_lbmu(inflow_si, angle_deg, u_scale);

                print_section_title("DEVICE INFORMATION");
                LBM lbm(lbm_N, Dx, Dy, Dz, lbm_nu, 0.0f, 0.0f, 0.0f, 0.0f, lbm_alpha, lbm_beta);
                lbm.set_coriolis(coriolis_Omegax_lbmu, coriolis_Omegay_lbmu, coriolis_Omegaz_lbmu);

                lbm.voxelize_mesh_on_device(mesh);
                println("| Voxelization done.                                                          |");
                print_section_title("BUILD BOUNDARY CONDITIONS");
                initialize_uniform_velocity(lbm, inflow_lbmu);
                apply_velocity_boundaries(lbm, inflow_lbmu);
                print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

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

        const float3 origin_lbmu = float3(
            0.5f - 0.5f * (float)lbm_N.x,
            0.5f - 0.5f * (float)lbm_N.y,
            0.5f - 0.5f * (float)lbm_N.z
        );
        const float flat_ground_lbmu = origin_lbmu.z + units.x(z_si_offset);
        GroundTemperaturePlane2D profile_ground_plane;
        bool use_profile_dem_ground = false;
        float ground_z_min_lbm = flat_ground_lbmu;
        float ground_z_max_lbm = flat_ground_lbmu;

        if (profile_dem_loaded) {
            const float dem_rx = profile_dem_xmax - profile_dem_xmin;
            const float dem_ry = profile_dem_ymax - profile_dem_ymin;
            const float stl_rx = stl_size_si.x;
            const float stl_ry = stl_size_si.y;
            if (dem_rx > 1.0e-6f && dem_ry > 1.0e-6f && stl_rx > 1.0e-6f && stl_ry > 1.0e-6f) {
                const float dem_to_stl_sx = stl_rx / dem_rx;
                const float dem_to_stl_sy = stl_ry / dem_ry;
                const float scale_warn = fmaxf(fabsf(dem_to_stl_sx - 1.0f), fabsf(dem_to_stl_sy - 1.0f));
                const float off_warn_x = fabsf(profile_dem_xmin - stl_min_si.x) / stl_rx;
                const float off_warn_y = fabsf(profile_dem_ymin - stl_min_si.y) / stl_ry;
                if (scale_warn > 0.02f || off_warn_x > 0.02f || off_warn_y > 0.02f) {
                    println("| Terrain DEM     | WARNING: DEM/STL XY bounds mismatch. Apply affine bounds alignment. |");
                    println("|                 | DEM->STL scale x=" + to_string(dem_to_stl_sx, 6u) +
                            ", y=" + to_string(dem_to_stl_sy, 6u) + "                             |");
                }

                std::vector<SamplePoint> dem_ground_samples;
                dem_ground_samples.reserve(profile_dem_points_raw.size());
                ground_z_min_lbm = +FLT_MAX;
                ground_z_max_lbm = -FLT_MAX;
                for (const auto& dp : profile_dem_points_raw) {
                    const float x_stl_raw = stl_min_si.x + (dp.x - profile_dem_xmin) * dem_to_stl_sx;
                    const float y_stl_raw = stl_min_si.y + (dp.y - profile_dem_ymin) * dem_to_stl_sy;
                    const float z_stl_raw = z_si_offset + dp.elevation;

                    const float x_lbm = origin_lbmu.x + (x_stl_raw - stl_min_si.x) * scale_geom;
                    const float y_lbm = origin_lbmu.y + (y_stl_raw - stl_min_si.y) * scale_geom;
                    const float z_lbm = origin_lbmu.z + (z_stl_raw - stl_min_si.z) * scale_geom;

                    if (!std::isfinite(x_lbm) || !std::isfinite(y_lbm) || !std::isfinite(z_lbm)) continue;
                    SamplePoint sp;
                    sp.p = float3(x_lbm, y_lbm, 0.0f);
                    sp.T = z_lbm;
                    sp.patch = k_patch_bottom;
                    dem_ground_samples.push_back(sp);
                    ground_z_min_lbm = fminf(ground_z_min_lbm, z_lbm);
                    ground_z_max_lbm = fmaxf(ground_z_max_lbm, z_lbm);
                }

                if (!dem_ground_samples.empty()) {
                    profile_ground_plane.build_from_patch0_samples(dem_ground_samples, flat_ground_lbmu);
                    use_profile_dem_ground = profile_ground_plane.has_samples();
                }
                if (use_profile_dem_ground) {
                    const float gmin_si = units.si_x(ground_z_min_lbm - origin_lbmu.z);
                    const float gmax_si = units.si_x(ground_z_max_lbm - origin_lbmu.z);
                    println("| Terrain DEM     | profile ground enabled. z(SI) range " +
                            to_string(gmin_si, 3u) + " .. " + to_string(gmax_si, 3u) + " m |");
                }
                else {
                    println("| Terrain DEM     | no valid points after mapping, fallback to flat ground     |");
                }
            }
            else {
                println("| Terrain DEM     | invalid DEM or STL XY range, fallback to flat ground       |");
            }
        }

        const uint total_cases = static_cast<uint>(angle_list.size());
        uint case_index = 0u;
        const bool profile_single_case_standard_output = (total_cases == 1u);
        if (profile_single_case_standard_output) {
            println("| Output logic    | standard luw naming: <datetime>_raw_* and <datetime>_avg  |");
        }
        else {
            println("| Output logic    | multi-angle mode keeps ANG_<angle>_ prefix to avoid overwrite |");
            println("|                 | each case still uses same run_nstep/purge_avg workflow      |");
        }

        auto profile_speed_lbmu = [&](const float pos_z, const float ground_z) {
            if (pos_z <= ground_z) return 0.0f;
            const float inv_dz = 1.0f / profile_dz_si;
            const uint last_idx = static_cast<uint>(profile_u_lbmu.size() - 1u);
            float z_agl_si = units.si_x(pos_z - ground_z);
            if (z_agl_si < 0.0f) z_agl_si = 0.0f;
            long idx_l = std::lround(z_agl_si * inv_dz);
            if (idx_l < 0l) idx_l = 0l;
            uint idx = static_cast<uint>(idx_l);
            if (idx > last_idx) idx = last_idx;
            return profile_u_lbmu[idx];
        };

        auto initialize_profile_velocity = [&](LBM& lbm,
                                               const std::vector<float>& ground_xy,
                                               const float dir_x, const float dir_y) {
            const uint Nx = lbm.get_Nx();
            const ulong Ntot = lbm.get_N();
            parallel_for(Ntot, [&](ulong n) {
                uint x = 0u, y = 0u, z = 0u;
                lbm.coordinates(n, x, y, z);
                if ((lbm.flags[n] & TYPE_S) != 0u) {
                    lbm.u.x[n] = 0.0f;
                    lbm.u.y[n] = 0.0f;
                    lbm.u.z[n] = 0.0f;
                    return;
                }
                const float ground_z = ground_xy[(ulong)y * (ulong)Nx + (ulong)x];
                const float u_mag = profile_speed_lbmu(lbm.position(x, y, z).z, ground_z);
                lbm.u.x[n] = dir_x * u_mag;
                lbm.u.y[n] = dir_y * u_mag;
                lbm.u.z[n] = 0.0f;
            });
        };

        auto apply_profile_boundaries = [&](LBM& lbm,
                                            const std::vector<float>& ground_xy,
                                            const float dir_x, const float dir_y) {
            const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
            const ulong Ntot = lbm.get_N();
            std::atomic<ulong> mapped_bc{ 0ul };
            std::atomic<ulong> terrain_solid_bc{ 0ul };
            parallel_for(Ntot, [&](ulong n) {
                uint x = 0u, y = 0u, z = 0u;
                lbm.coordinates(n, x, y, z);
                if (z == 0u) {
                    lbm.flags[n] = TYPE_S;
                    lbm.u.x[n] = 0.0f;
                    lbm.u.y[n] = 0.0f;
                    lbm.u.z[n] = 0.0f;
                    return;
                }
                const bool is_boundary =
                    (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u);
                if (!is_boundary) return;
                if ((lbm.flags[n] & TYPE_S) != 0u) return;

                const float ground_z = ground_xy[(ulong)y * (ulong)Nx + (ulong)x];
                const float pos_z = lbm.position(x, y, z).z;
                if (pos_z <= ground_z) {
                    lbm.flags[n] = TYPE_S;
                    lbm.u.x[n] = 0.0f;
                    lbm.u.y[n] = 0.0f;
                    lbm.u.z[n] = 0.0f;
                    terrain_solid_bc.fetch_add(1ul, std::memory_order_relaxed);
                    return;
                }
                lbm.flags[n] = (uchar)(lbm.flags[n] | TYPE_E);
                const float u_mag = profile_speed_lbmu(pos_z, ground_z);
                lbm.u.x[n] = dir_x * u_mag;
                lbm.u.y[n] = dir_y * u_mag;
                lbm.u.z[n] = 0.0f;
                mapped_bc.fetch_add(1ul, std::memory_order_relaxed);
            });
            println("| Velocity BC     | profile boundaries mapped: " + to_string(mapped_bc.load()) + " cells                |");
            if (terrain_solid_bc.load(std::memory_order_relaxed) > 0ul) {
                println("|                 | boundary cells below local terrain -> solid: " +
                        to_string(terrain_solid_bc.load()) + "                     |");
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

            print_section_title("DEVICE INFORMATION");
            LBM lbm(lbm_N, Dx, Dy, Dz, lbm_nu, 0.0f, 0.0f, 0.0f, 0.0f, lbm_alpha, lbm_beta);
            lbm.set_coriolis(coriolis_Omegax_lbmu, coriolis_Omegay_lbmu, coriolis_Omegaz_lbmu);

            lbm.voxelize_mesh_on_device(mesh);
            println("| Voxelization done.                                                          |");
            print_section_title("BUILD BOUNDARY CONDITIONS");

            const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
            std::vector<float> ground_xy((ulong)Nx * (ulong)Ny, flat_ground_lbmu);
            if (use_profile_dem_ground) {
                const float zmin = lbm.position(0u, 0u, 0u).z;
                const float zmax = lbm.position(0u, 0u, Nz - 1u).z;
                parallel_for((ulong)Nx * (ulong)Ny, [&](ulong id) {
                    const uint x = (uint)(id % (ulong)Nx);
                    const uint y = (uint)(id / (ulong)Nx);
                    const float3 pxy = lbm.position(x, y, 0u);
                    float zg = profile_ground_plane.eval_xy(pxy.x, pxy.y);
                    if (!std::isfinite(zg)) zg = flat_ground_lbmu;
                    zg = fminf(fmaxf(zg, zmin), zmax);
                    ground_xy[id] = zg;
                });
                float gmin = +FLT_MAX, gmax = -FLT_MAX;
                for (const float zg : ground_xy) {
                    gmin = fminf(gmin, zg);
                    gmax = fmaxf(gmax, zg);
                }
                println("| Terrain ground  | mapped z(SI) range " +
                        to_string(units.si_x(gmin - origin_lbmu.z), 3u) + " .. " +
                        to_string(units.si_x(gmax - origin_lbmu.z), 3u) + " m                     |");
            }

            if (use_profile_dem_ground) {
                std::atomic<ulong> terrain_clipped_to_solid{ 0ul };
                parallel_for(lbm.get_N(), [&](ulong n) {
                    if ((lbm.flags[n] & TYPE_S) != 0u) return;
                    uint x = 0u, y = 0u, z = 0u;
                    lbm.coordinates(n, x, y, z);
                    const float pos_z = lbm.position(x, y, z).z;
                    const float ground_z = ground_xy[(ulong)y * (ulong)Nx + (ulong)x];
                    if (pos_z < ground_z) {
                        lbm.flags[n] = TYPE_S;
                        lbm.u.x[n] = 0.0f;
                        lbm.u.y[n] = 0.0f;
                        lbm.u.z[n] = 0.0f;
                        terrain_clipped_to_solid.fetch_add(1ul, std::memory_order_relaxed);
                    }
                });
                if (terrain_clipped_to_solid.load(std::memory_order_relaxed) > 0ul) {
                    println("| Terrain clip    | below-terrain cells forced to solid: " +
                            to_string(terrain_clipped_to_solid.load()) + "                    |");
                }
            }

            initialize_profile_velocity(lbm, ground_xy, dir_x, dir_y);
            apply_profile_boundaries(lbm, ground_xy, dir_x, dir_y);
            print_kv_row("Boundary init", "complete. Time: [" + now_str() + "]");

            if (flux_correction) {
                print_kv_row("Flux correction", "starting. Time: [" + now_str() + "]");
                double avg_du = 0.0, net_b = 0.0, net_a = 0.0;
                auto eval = [&](const float3& q)->float3 {
                    uint x = 0u, y = 0u, z = 0u;
                    lbm.coordinates(q, x, y, z);
                    const float ground_z = ground_xy[(ulong)y * (ulong)Nx + (ulong)x];
                    const float u_mag = profile_speed_lbmu(q.z, ground_z);
                    return float3(dir_x * u_mag, dir_y * u_mag, 0.0f);
                };
                apply_flux_correction(lbm, downstream_bc, eval, /*show_report=*/true,
                    &avg_du, &net_b, &net_a);
            }
            else {
                print_kv_row("Flux correction", "skipped. Set flux_correction=true to enable");
            }

            const std::string vtk_prefix = profile_single_case_standard_output
                ? std::string("")
                : ("ANG_" + format_tag(angle_deg) + "_");
            run_lbm(lbm, vtk_prefix, true);
        }
    }
}
