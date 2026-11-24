#pragma once

// Boundary patches of a LUW case and the 2D sample fields defined on them.
//
// Patch ids follow the SurfData patch column: 0 bottom (ground), 1 top,
// 2 south (y=0), 3 north (y=Ny-1), 4 west (x=0), 5 east (x=Nx-1). A patch
// field interpolates scattered samples over the two in-plane coordinates of
// its face; GroundPlane2D interpolates patch-0 samples over (x, y) and serves
// the ground temperature and terrain height fields.

#include "inflow/samples.hpp"

#include <cfloat>

namespace culbm::bc {

constexpr int k_patch_bottom = 0;
constexpr int k_patch_top = 1;
constexpr int k_patch_south = 2;
constexpr int k_patch_north = 3;
constexpr int k_patch_west = 4;
constexpr int k_patch_east = 5;

class GroundPlane2D {
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

inline const char* patch_name(const int patch) {
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

inline int downstream_to_patch(const std::string& downstream_bc_local) {
    if (downstream_bc_local == "+y") return k_patch_north;
    if (downstream_bc_local == "-y") return k_patch_south;
    if (downstream_bc_local == "+x") return k_patch_east;
    if (downstream_bc_local == "-x") return k_patch_west;
    return -1;
}

inline int boundary_cell_to_patch(const uint x, const uint y, const uint z, const uint Nx, const uint Ny, const uint Nz) {
    if (z == Nz - 1u) return k_patch_top;
    if (x == 0u) return k_patch_west;
    if (x == Nx - 1u) return k_patch_east;
    if (y == 0u) return k_patch_south;
    if (y == Ny - 1u) return k_patch_north;
    return -1;
}

inline bool is_downstream_boundary_cell(const uint x, const uint y, const uint z,
                                        const uint Nx, const uint Ny, const uint Nz,
                                        const std::string& downstream_bc_local) {
    (void)z;
    (void)Nz;
    if (downstream_bc_local == "+y") return y == Ny - 1u;
    if (downstream_bc_local == "-y") return y == 0u;
    if (downstream_bc_local == "+x") return x == Nx - 1u;
    if (downstream_bc_local == "-x") return x == 0u;
    return false;
}

inline bool patch_surface_coordinates(const int patch, const float3& p, float& a, float& b) {
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

// Face ids of the outlet given as "+x", "-x", "+y", "-y".
// VK inlet convention: 0 west, 1 east, 2 south, 3 north; -1 unknown.
inline int downstream_face_id_from_bc(const std::string& bc) {
    if (bc == "-x") return 0;
    if (bc == "+x") return 1;
    if (bc == "-y") return 2;
    if (bc == "+y") return 3;
    return -1;
}
// Buffer nudging and scalar outlet convention: 1 west, 2 east, 3 south,
// 4 north; 0 none.
inline int buffer_face_id_from_bc(const std::string& bc) {
    if (bc == "-x") return 1;      // west (x=0)
    if (bc == "+x") return 2;      // east (x=Nx-1)
    if (bc == "-y") return 3;      // south (y=0)
    if (bc == "+y") return 4;      // north (y=Ny-1)
    return 0;                      // none/unknown
}
// Outlet of a batch case: the dominant axis of the flow direction.
inline std::string profile_downstream_bc_from_dir(const float dir_x, const float dir_y) {
    if (fabsf(dir_x) >= fabsf(dir_y)) {
        return dir_x >= 0.0f ? "+x" : "-x";
    }
    return dir_y >= 0.0f ? "+y" : "-y";
}

} // namespace culbm::bc
