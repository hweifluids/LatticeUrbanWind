#include "interpolation.hpp"
#include <cfloat>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdlib>

static inline std::string now_str_local() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%F %T");
    return ss.str();
}

float3 NearestNeighborInterpolator::eval(const float3& pos) const {
    float best = FLT_MAX;
    float3 u(0);
    for (size_t i = 0; i < P.size(); ++i) {
        const float3 d = pos - P[i];
        const float d2 = dot(d, d);
        if (d2 < best) { best = d2; u = U[i]; }
    }
    return u;
}

float3 InletVelocityField::operator()(const float3& pos) const {
    const float z_phys = pos.z - z0_;
    if (z_phys < zoff_) return float3(0.0f);
    return interp_.eval(pos);
}

void apply_inlet_outlet(LBM& lbm,
    const std::string& downstream_bc,
    const InletVelocityField& inlet,
    unsigned long min_work_per_thread,
    bool show_progress) {
    const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
    const unsigned hw0 = std::thread::hardware_concurrency();
    const unsigned hw = hw0 == 0u ? 4u : hw0;

    unsigned num_threads = hw;
    #ifdef _MSC_VER
        char* env_dup = nullptr;
        size_t env_len = 0;
        if (_dupenv_s(&env_dup, &env_len, "LBM_NUM_THREADS") == 0 && env_dup != nullptr) {
            const long v = std::strtol(env_dup, nullptr, 10);
            if (v > 0) {
                num_threads = static_cast<unsigned>(std::min<long>(v, hw));
            }
            std::free(env_dup);
        }
    #else
        if (const char* env_p = std::getenv("LBM_NUM_THREADS")) {
            const long v = std::strtol(env_p, nullptr, 10);
            if (v > 0) {
                num_threads = static_cast<unsigned>(std::min<long>(v, hw));
            }
        }
    #endif



    println("| Threads used for BC connection: " + to_string(num_threads) + "                                 |");

    std::atomic<unsigned long long> processed{ 0ull };
    std::atomic<bool> done{ false };

    std::thread monitor;
    if (show_progress) {
        monitor = std::thread([&]() {
            const unsigned long long N = lbm.get_N();
            while (!done.load(std::memory_order_relaxed)) {
                const double pct = (N == 0ull)
                    ? 100.0
                    : 100.0 * static_cast<double>(processed.load(std::memory_order_relaxed)) / static_cast<double>(N);
                std::fprintf(stdout, "\r[%s] inlet/outlet init: %6.2f%%",
                    now_str_local().c_str(), pct);
                std::fflush(stdout);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::fprintf(stdout, "\r| [%s] inlet/outlet init: %6.2f%%                         |",
                now_str_local().c_str(), 100.0);
            std::fputc('\n', stdout);
            });
    }


    std::vector<std::thread> workers; workers.reserve(num_threads);
    const unsigned long long Ntot = lbm.get_N();
    const unsigned long long chunk =
        std::max<unsigned long long>(16384ull, Ntot / (static_cast<unsigned long long>(num_threads) * 32ull) + 1ull);
    std::atomic<unsigned long long> next{ 0ull };

    for (unsigned t = 0u; t < num_threads; ++t) {
        workers.emplace_back([&, Ntot, chunk]() {
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
                    }
                    else {
                        bool outlet = false;
                        if (downstream_bc == "+y")      outlet = (y == Ny - 1u);
                        else if (downstream_bc == "-y") outlet = (y == 0u);
                        else if (downstream_bc == "+x") outlet = (x == Nx - 1u);
                        else if (downstream_bc == "-x") outlet = (x == 0u);

                        const bool inlet_face =
                            ((x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u) && !outlet);

                        if (inlet_face) {
                            lbm.flags[n] = TYPE_E;
                            const float3 u = inlet(pos);
                            lbm.u.x[n] = u.x;
                            lbm.u.y[n] = u.y;
                            lbm.u.z[n] = u.z;
                        }
                        else if (outlet) {
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
    done.store(true, std::memory_order_relaxed);
    if (monitor.joinable()) monitor.join();
}
