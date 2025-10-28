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

// ======================= Interpolator =======================

float3 KNNInterpolatorHD::eval(const float3& pos) const {
    constexpr int K = 48;
    constexpr float eps2 = 1e-16f;

    const int Pn = static_cast<int>(P_.size());
    if (Pn == 0) return float3{0.0f, 0.0f, 0.0f};

    float best_r2[K];
    int   best_i[K];
    int   filled = 0;
    float worst_r2 = -1.0f;
    int   worst_k  = -1;
    float max_r2_kept = 0.0f;

    for (int i = 0; i < Pn; ++i) {
        const float dx = pos.x - P_[i].x;
        const float dy = pos.y - P_[i].y;
        const float dz = pos.z - P_[i].z;
        const float r2 = dx*dx + dy*dy + dz*dz;

        if (r2 <= eps2) return U_[i];

        if (filled < K) {
            best_r2[filled] = r2;
            best_i[filled]  = i;
            if (r2 > max_r2_kept) max_r2_kept = r2;
            if (r2 > worst_r2) { worst_r2 = r2; worst_k = filled; }
            ++filled;
        } else if (r2 < worst_r2) {
            best_r2[worst_k] = r2;
            best_i[worst_k]  = i;
            worst_r2 = best_r2[0]; worst_k = 0; max_r2_kept = best_r2[0];
            for (int k = 1; k < K; ++k) {
                if (best_r2[k] > worst_r2) { worst_r2 = best_r2[k]; worst_k = k; }
                if (best_r2[k] > max_r2_kept) max_r2_kept = best_r2[k];
            }
        }
    }

    const int used = filled;
    if (used == 0) return float3{0.0f, 0.0f, 0.0f};

    const float delta2 = 1e-6f * std::max(max_r2_kept, 1e-12f);

    double wx = 0.0, wy = 0.0, wz = 0.0, wsum = 0.0;
    for (int k = 0; k < used; ++k) {
        const float r2 = best_r2[k];
        const int i    = best_i[k];
        const double w = 1.0 / double(r2 + delta2);
        wx += w * double(U_[i].x);
        wy += w * double(U_[i].y);
        wz += w * double(U_[i].z);
        wsum += w;
    }
    if (wsum == 0.0) return float3{0.0f, 0.0f, 0.0f};
    const float inv = static_cast<float>(1.0 / wsum);
    return float3{ static_cast<float>(wx) * inv,
                   static_cast<float>(wy) * inv,
                   static_cast<float>(wz) * inv };
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
    std::fprintf(stdout, "\r| [%s] inlet/outlet init (HD): %6.2f%%                      |",
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

    const unsigned hw0 = std::thread::hardware_concurrency();
    const unsigned hw  = hw0 == 0u ? 4u : hw0;

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

    std::atomic<unsigned long long> processed{0ull};
    const unsigned long long chunk =
        std::max<unsigned long long>(16384ull,
            Ntot / (static_cast<unsigned long long>(num_threads) * 32ull) + 1ull);
    std::atomic<unsigned long long> next{0ull};

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    std::atomic<bool> progress_stop{false};
    std::thread progress_thread;
    if (show_progress) {
        progress_thread = std::thread([&] {
            using namespace std::chrono;
            while (!progress_stop.load(std::memory_order_relaxed)) {
                const double pct = 100.0 * double(processed.load(std::memory_order_relaxed)) / double(Ntot);
                print_progress_inline_hd(pct);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            print_progress_inline_hd(100.0, true);
            std::fputc('\n', stdout);
        });
    }

    for (unsigned t = 0u; t < num_threads; ++t) {
        workers.emplace_back([&, Ntot, Nx, Ny, Nz, chunk]() {
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

                        if (on_outer && !outlet) {
                            lbm.flags[n] = TYPE_E;
                            const float3 u = inlet(pos);
                            lbm.u.x[n] = u.x;
                            lbm.u.y[n] = u.y;
                            lbm.u.z[n] = u.z;
                        } else if (outlet) {
                            lbm.flags[n] = TYPE_E;
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
    if (show_progress) {
        progress_stop.store(true, std::memory_order_relaxed);
        progress_thread.join();
    }
}
