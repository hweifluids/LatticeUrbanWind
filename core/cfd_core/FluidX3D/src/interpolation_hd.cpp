#include "interpolation_hd.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>
#include <cstdio>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static inline unsigned detect_available_worker_threads_hd() {
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

static inline std::string now_str_local_hd() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return os.str();
}

// Local helpers for high order surface interpolation (HD)
namespace {
    // Solve a 6x6 linear system with three right hand sides using Gaussian elimination
    // A * ax = bx, A * ay = by, A * az = bz
    static bool solve_6x6_3rhs(double A[6][6],
        double bx[6],
        double by[6],
        double bz[6],
        double ax[6],
        double ay[6],
        double az[6]) {
        double a[6][6];
        double rx[6];
        double ry[6];
        double rz[6];

        for (int i = 0; i < 6; ++i) {
            rx[i] = bx[i];
            ry[i] = by[i];
            rz[i] = bz[i];
            for (int j = 0; j < 6; ++j) {
                a[i][j] = A[i][j];
            }
        }

        for (int k = 0; k < 6; ++k) {
            int pivot = k;
            double max_abs = std::fabs(a[k][k]);
            for (int i = k + 1; i < 6; ++i) {
                const double v = std::fabs(a[i][k]);
                if (v > max_abs) {
                    max_abs = v;
                    pivot = i;
                }
            }

            if (max_abs < 1e-18) {
                return false;
            }

            if (pivot != k) {
                for (int j = 0; j < 6; ++j) {
                    std::swap(a[k][j], a[pivot][j]);
                }
                std::swap(rx[k], rx[pivot]);
                std::swap(ry[k], ry[pivot]);
                std::swap(rz[k], rz[pivot]);
            }

            const double diag = a[k][k];
            const double inv_diag = 1.0 / diag;

            for (int i = k + 1; i < 6; ++i) {
                const double factor = a[i][k] * inv_diag;
                if (factor == 0.0) continue;
                for (int j = k; j < 6; ++j) {
                    a[i][j] -= factor * a[k][j];
                }
                rx[i] -= factor * rx[k];
                ry[i] -= factor * ry[k];
                rz[i] -= factor * rz[k];
            }
        }

        double ax_tmp[6] = {};
        double ay_tmp[6] = {};
        double az_tmp[6] = {};

        for (int i = 5; i >= 0; --i) {
            double sx = rx[i];
            double sy = ry[i];
            double sz = rz[i];
            for (int j = i + 1; j < 6; ++j) {
                sx -= a[i][j] * ax_tmp[j];
                sy -= a[i][j] * ay_tmp[j];
                sz -= a[i][j] * az_tmp[j];
            }
            const double diag = a[i][i];
            if (std::fabs(diag) < 1e-18) {
                return false;
            }
            const double inv_diag = 1.0 / diag;
            ax_tmp[i] = sx * inv_diag;
            ay_tmp[i] = sy * inv_diag;
            az_tmp[i] = sz * inv_diag;
        }

        for (int i = 0; i < 6; ++i) {
            ax[i] = ax_tmp[i];
            ay[i] = ay_tmp[i];
            az[i] = az_tmp[i];
        }

        return true;
    }

    // Map 3D position to local 2D surface coordinates
    static inline void surface_local_coords(int plane,
        const float3& p,
        const float3& pos,
        float xmin,
        float xmax,
        float ymin,
        float ymax,
        float zmax,
        float& s1,
        float& s2) {
        (void)xmax;
        (void)ymin;
        (void)ymax;
        (void)zmax;
        switch (plane) {
        case 0:
        case 1:
            s1 = p.y - pos.y;
            s2 = p.z - pos.z;
            break;
        case 2:
        case 3:
            s1 = p.x - pos.x;
            s2 = p.z - pos.z;
            break;
        case 4:
        default:
            s1 = p.x - pos.x;
            s2 = p.y - pos.y;
            break;
        }
    }
}

float3 KNNInterpolatorHD::eval(const float3& pos) const {
    constexpr int K = 64;
    constexpr float eps2 = 1e-16f;

    const int Pn = static_cast<int>(P_.size());
    if (Pn == 0) {
        return float3{ 0.0f, 0.0f, 0.0f };
    }

    static std::atomic<bool> banner_printed{ false };
    if (!banner_printed.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stdout,
            "| [%s] using high order surface based inlet interpolator (HD)\n",
            now_str_local_hd().c_str());
        std::fflush(stdout);
    }

    float xmin = P_[0].x;
    float xmax = P_[0].x;
    float ymin = P_[0].y;
    float ymax = P_[0].y;
    float zmin = P_[0].z;
    float zmax = P_[0].z;

    for (int i = 1; i < Pn; ++i) {
        const float x = P_[i].x;
        const float y = P_[i].y;
        const float z = P_[i].z;
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
    }

    const float ex = xmax - xmin;
    const float ey = ymax - ymin;
    const float ez = zmax - zmin;
    float max_extent = ex;
    if (ey > max_extent) max_extent = ey;
    if (ez > max_extent) max_extent = ez;

    const float plane_tol = 1e-5f * max_extent + 1e-6f;

    const float d0 = std::fabs(pos.x - xmin);
    const float d1 = std::fabs(pos.x - xmax);
    const float d2 = std::fabs(pos.y - ymin);
    const float d3 = std::fabs(pos.y - ymax);
    const float d4 = std::fabs(pos.z - zmax);

    int plane = 0;
    float dmin = d0;
    if (d1 < dmin) { dmin = d1; plane = 1; }
    if (d2 < dmin) { dmin = d2; plane = 2; }
    if (d3 < dmin) { dmin = d3; plane = 3; }
    if (d4 < dmin) { dmin = d4; plane = 4; }

    float best_r2[K];
    int   best_i[K];
    int   filled = 0;
    float max_r2_kept = 0.0f;

    for (int i = 0; i < Pn; ++i) {
        const float3& p = P_[i];

        bool on_plane = false;
        switch (plane) {
        case 0:
            on_plane = (std::fabs(p.x - xmin) <= plane_tol);
            break;
        case 1:
            on_plane = (std::fabs(p.x - xmax) <= plane_tol);
            break;
        case 2:
            on_plane = (std::fabs(p.y - ymin) <= plane_tol);
            break;
        case 3:
            on_plane = (std::fabs(p.y - ymax) <= plane_tol);
            break;
        case 4:
        default:
            on_plane = (std::fabs(p.z - zmax) <= plane_tol);
            break;
        }

        if (!on_plane) {
            continue;
        }

        float s1, s2;
        surface_local_coords(plane, p, pos, xmin, xmax, ymin, ymax, zmax, s1, s2);
        const float r2 = s1 * s1 + s2 * s2;

        if (r2 <= eps2) {
            return U_[i];
        }

        if (filled < K) {
            best_r2[filled] = r2;
            best_i[filled] = i;
            if (r2 > max_r2_kept) max_r2_kept = r2;
            ++filled;
        }
        else {
            int worst_k = 0;
            float worst_r2 = best_r2[0];
            for (int k = 1; k < K; ++k) {
                if (best_r2[k] > worst_r2) {
                    worst_r2 = best_r2[k];
                    worst_k = k;
                }
            }
            if (r2 < worst_r2) {
                best_r2[worst_k] = r2;
                best_i[worst_k] = i;

                max_r2_kept = best_r2[0];
                for (int k = 1; k < K; ++k) {
                    if (best_r2[k] > max_r2_kept) {
                        max_r2_kept = best_r2[k];
                    }
                }
            }
        }
    }

    const int used = filled;
    if (used == 0) {
        return float3{ 0.0f, 0.0f, 0.0f };
    }

    const double R2 = static_cast<double>(std::max(max_r2_kept, 1e-12f));
    const double sigma2 = 0.25 * R2;

    if (used >= 6) {
        double A[6][6] = {};
        double bx[6] = {};
        double by[6] = {};
        double bz[6] = {};

        for (int kk = 0; kk < used; ++kk) {
            const int idx = best_i[kk];
            const float3& p = P_[idx];

            float s1f, s2f;
            surface_local_coords(plane, p, pos, xmin, xmax, ymin, ymax, zmax, s1f, s2f);

            const double q1 = static_cast<double>(s1f);
            const double q2 = static_cast<double>(s2f);
            const double r2d = q1 * q1 + q2 * q2;

            const double w = std::exp(-r2d / (2.0 * sigma2));

            const double phi[6] = {
                1.0,
                q1,
                q2,
                q1 * q1,
                q1 * q2,
                q2 * q2
            };

            for (int i = 0; i < 6; ++i) {
                const double wi = w * phi[i];
                for (int j = 0; j < 6; ++j) {
                    A[i][j] += wi * phi[j];
                }
            }

            const float3& u = U_[idx];
            for (int i = 0; i < 6; ++i) {
                const double wphi = w * phi[i];
                bx[i] += wphi * static_cast<double>(u.x);
                by[i] += wphi * static_cast<double>(u.y);
                bz[i] += wphi * static_cast<double>(u.z);
            }
        }

        double ax[6] = {};
        double ay[6] = {};
        double az[6] = {};
        if (solve_6x6_3rhs(A, bx, by, bz, ax, ay, az)) {
            return float3{
                static_cast<float>(ax[0]),
                static_cast<float>(ay[0]),
                static_cast<float>(az[0])
            };
        }
    }

    double wx = 0.0;
    double wy = 0.0;
    double wz = 0.0;
    double wsum = 0.0;

    for (int k = 0; k < used; ++k) {
        const int idx = best_i[k];
        const float3& p = P_[idx];

        float s1f, s2f;
        surface_local_coords(plane, p, pos, xmin, xmax, ymin, ymax, zmax, s1f, s2f);

        const double q1 = static_cast<double>(s1f);
        const double q2 = static_cast<double>(s2f);
        const double r2d = q1 * q1 + q2 * q2;

        const double w = std::exp(-r2d / (2.0 * sigma2));

        const float3& u = U_[idx];
        wx += w * static_cast<double>(u.x);
        wy += w * static_cast<double>(u.y);
        wz += w * static_cast<double>(u.z);
        wsum += w;
    }

    if (wsum <= 0.0) {
        return float3{ 0.0f, 0.0f, 0.0f };
    }

    const double inv = 1.0 / wsum;
    return float3{
        static_cast<float>(wx * inv),
        static_cast<float>(wy * inv),
        static_cast<float>(wz * inv)
    };
}


float3 InletVelocityFieldHD::operator()(const float3& pos) const {
	// Check if below z threshold
    if (pos.z < z_base_lbmu_) {
		// Velocity zero below threshold
        return float3{ 0.0f, 0.0f, 0.0f };
    }
        return interp_.eval(pos);
}

// ======================= Parellel computing =======================

static inline void print_progress_inline_hd(double pct, bool final_flush = false) {
    std::fprintf(stdout, "\r| [%s] inlet/outlet init (HD): %6.3f%%                      |",
                 now_str_local_hd().c_str(), pct);
    if (final_flush) std::fflush(stdout);
}

void apply_inlet_outlet_hd(LBM& lbm,
                           const std::string& downstream_bc,
                           const InletVelocityFieldHD& inlet,
                           unsigned long /*min_work_per_thread*/,
                           bool show_progress) {
    const uint Nx = lbm.get_Nx();
    const uint Ny = lbm.get_Ny();
    const uint Nz = lbm.get_Nz();
    const unsigned long long Ntot = lbm.get_N();

    const unsigned hw  = detect_available_worker_threads_hd();

    unsigned num_threads = hw;
#ifdef _MSC_VER
    char* env_dup = nullptr;
    size_t env_len = 0;
    if (_dupenv_s(&env_dup, &env_len, "LBM_NUM_THREADS") == 0 && env_dup != nullptr) {
        const long v = std::strtol(env_dup, nullptr, 10);
        if (v > 0) num_threads = static_cast<unsigned>(std::min<long>(v, hw));
        std::free(env_dup);
    }
#else
    if (const char* env_p = std::getenv("LBM_NUM_THREADS")) {
        const long v = std::strtol(env_p, nullptr, 10);
        if (v > 0) num_threads = static_cast<unsigned>(std::min<long>(v, hw));
    }
#endif

    if (show_progress) {
        std::fprintf(stdout,
            "| [%s] inlet/outlet init (HD) using %u worker threads (hardware reports %u)\n",
            now_str_local_hd().c_str(),
            num_threads,
            hw);
        std::fflush(stdout);
    }

    std::vector<std::vector<unsigned long long>> heavy_lists(num_threads);


    std::atomic<unsigned long long> processed{0ull};
    const unsigned long long chunk =
        std::max<unsigned long long>(16384ull,
            Ntot / (static_cast<unsigned long long>(num_threads) * 32ull) + 1ull);
    std::atomic<unsigned long long> next{0ull};

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    std::atomic<bool> progress_stop{ false };
    std::thread progress_thread;
    if (show_progress) {
        std::fprintf(stdout, "| [%s] inlet/outlet init (HD) step 1/2: classifying boundary cells...\n",
            now_str_local_hd().c_str());
        std::fflush(stdout);
    }


    for (unsigned t = 0u; t < num_threads; ++t) {
        workers.emplace_back([&, Ntot, Nx, Ny, Nz, chunk, t]() {
            unsigned long long local_incr = 0ull;
            for (;;) {
                const unsigned long long start_idx = next.fetch_add(chunk, std::memory_order_relaxed);
                if (start_idx >= Ntot) break;
                const unsigned long long end_idx = std::min<unsigned long long>(start_idx + chunk, Ntot);

                for (unsigned long long n = start_idx; n < end_idx; ++n) {
                    uint x = 0u, y = 0u, z = 0u;
                    lbm.coordinates(n, x, y, z);
                    const float3 pos = lbm.position(x, y, z);

                    if (z == 0u) {
                        lbm.flags[n] = TYPE_S;
                    } else {
                        bool outlet = false;
                        if (downstream_bc == "+y")      outlet = (y == Ny - 1u);
                        else if (downstream_bc == "-y") outlet = (y == 0u);
                        else if (downstream_bc == "+x") outlet = (x == Nx - 1u);
                        else if (downstream_bc == "-x") outlet = (x == 0u);

                        const bool on_outer =
                            (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u);

                        //if (on_outer && !outlet) {
                        //    lbm.flags[n] = TYPE_E;
                        //    heavy_lists[t].push_back(n);
                        //}
                        //else if (outlet) {
                        //    lbm.flags[n] = TYPE_E;
                        //}
                        if (on_outer) {
                            lbm.flags[n] = TYPE_E;
                            heavy_lists[t].push_back(n);
                        }
                    }

                    if (++local_incr == 2048ull) {
                        processed.fetch_add(local_incr, std::memory_order_relaxed);
                        local_incr = 0ull;
                    }
                }
            }
            if (local_incr) processed.fetch_add(local_incr, std::memory_order_relaxed);
        });
    }

    for (auto& th : workers) th.join();

    std::vector<unsigned long long> heavy_indices;
    {
        size_t total_heavy = 0;
        for (unsigned t = 0u; t < num_threads; ++t) {
            total_heavy += heavy_lists[t].size();
        }
        heavy_indices.reserve(total_heavy);
        for (unsigned t = 0u; t < num_threads; ++t) {
            heavy_indices.insert(heavy_indices.end(),
                heavy_lists[t].begin(), heavy_lists[t].end());
        }
    }

    if (!heavy_indices.empty()) {
        const unsigned long long Nheavy = static_cast<unsigned long long>(heavy_indices.size());

        // Split heavy boundary nodes into five faces:
        // f = 0: x == 0
        // f = 1: x == Nx - 1
        // f = 2: y == 0
        // f = 3: y == Ny - 1
        // f = 4: z == Nz - 1
        std::vector<std::vector<unsigned long long>> face_indices(5);
        const unsigned long long reserve_per_face = Nheavy / 5ull + 1ull;
        for (int f = 0; f < 5; ++f) {
            face_indices[f].reserve(reserve_per_face);
        }

        for (unsigned long long idx = 0ull; idx < Nheavy; ++idx) {
            const unsigned long long n = heavy_indices[idx];
            uint x = 0u, y = 0u, z = 0u;
            lbm.coordinates(n, x, y, z);

            int face = -1;
            if (x == 0u) {
                face = 0;
            }
            else if (x == Nx - 1u) {
                face = 1;
            }
            else if (y == 0u) {
                face = 2;
            }
            else if (y == Ny - 1u) {
                face = 3;
            }
            else if (z == Nz - 1u) {
                face = 4;
            }

            if (face >= 0) {
                face_indices[face].push_back(n);
            }
        }

        for (int f = 0; f < 5; ++f) {
            const std::vector<unsigned long long>& fi = face_indices[f];
            const unsigned long long Nface = static_cast<unsigned long long>(fi.size());
            if (Nface == 0ull) {
                continue;
            }

            const char* face_name = nullptr;
            switch (f) {
            case 0: face_name = "x == 0"; break;
            case 1: face_name = "x == Nx - 1"; break;
            case 2: face_name = "y == 0"; break;
            case 3: face_name = "y == Ny - 1"; break;
            case 4: face_name = "z == Nz - 1"; break;
            default: face_name = "unknown"; break;
            }

            if (show_progress) {
                std::fprintf(stdout,
                    "| [%s] inlet/outlet init (HD) surface %d/5 (%s): %llu cells\n",
                    now_str_local_hd().c_str(),
                    f + 1,
                    face_name,
                    static_cast<unsigned long long>(Nface));
                std::fflush(stdout);

                processed.store(0ull, std::memory_order_relaxed);
                progress_stop.store(false, std::memory_order_relaxed);
                progress_thread = std::thread([&, Nface]() {
                    using namespace std::chrono;
                    while (!progress_stop.load(std::memory_order_relaxed)) {
                        const double pct = 100.0 * double(processed.load(std::memory_order_relaxed))
                            / double(Nface);
                        print_progress_inline_hd(pct);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    print_progress_inline_hd(100.0, true);
                    std::fputc('\n', stdout);
                    });
            }

            std::atomic<unsigned long long> next_heavy{ 0ull };
            const unsigned long long heavy_chunk =
                std::max<unsigned long long>(64ull,
                    Nface / (static_cast<unsigned long long>(num_threads) * 32ull) + 1ull);

            workers.clear();
            workers.reserve(num_threads);
            for (unsigned t = 0u; t < num_threads; ++t) {
                workers.emplace_back([Nface, heavy_chunk, &fi, &next_heavy, &processed, &lbm, &inlet]() {
                    unsigned long long local_incr = 0ull;
                    for (;;) {
                        const unsigned long long start_idx_h =
                            next_heavy.fetch_add(heavy_chunk, std::memory_order_relaxed);
                        if (start_idx_h >= Nface) break;
                        const unsigned long long end_idx_h =
                            std::min<unsigned long long>(start_idx_h + heavy_chunk, Nface);

                        for (unsigned long long idx = start_idx_h; idx < end_idx_h; ++idx) {
                            const unsigned long long n = fi[idx];
                            uint x = 0u, y = 0u, z = 0u;
                            lbm.coordinates(n, x, y, z);
                            const float3 pos = lbm.position(x, y, z);
                            const float3 u = inlet(pos);
                            lbm.u.x[n] = u.x;
                            lbm.u.y[n] = u.y;
                            lbm.u.z[n] = u.z;
                            if (++local_incr == 2048ull) {
                                processed.fetch_add(local_incr, std::memory_order_relaxed);
                                local_incr = 0ull;
                            }
                        }
                    }
                    if (local_incr) {
                        processed.fetch_add(local_incr, std::memory_order_relaxed);
                    }
                    });
            }
            for (auto& th : workers) th.join();

            if (show_progress) {
                progress_stop.store(true, std::memory_order_relaxed);
                if (progress_thread.joinable()) {
                    progress_thread.join();
                }
            }
        }
    }

}
