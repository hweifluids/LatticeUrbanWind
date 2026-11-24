#include "fluxcorrection.hpp"
#include "../../case/report.hpp"
#include "../../run/console_log.hpp"

namespace culbm::bc {

enum class FaceKind { ZTop, XMin, XMax, YMin, YMax, None };

static inline FaceKind pick_face(uint x, uint y, uint z, uint Nx, uint Ny, uint Nz) {
    if (z == Nz - 1u) return FaceKind::ZTop;
    if (x == 0u)      return FaceKind::XMin;
    if (x == Nx - 1u) return FaceKind::XMax;
    if (y == 0u)      return FaceKind::YMin;
    if (y == Ny - 1u) return FaceKind::YMax;
    return FaceKind::None;
}

struct BoundaryCell {
    unsigned long long n = 0ull;
    FaceKind face = FaceKind::None;
};

static inline uint detect_flux_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return (hw == 0u) ? 1u : static_cast<uint>(hw);
}

void apply_flux_correction(culbm::solver::Solver& solver,
                           const std::string& downstream_bc,
                           const std::function<float3(const float3&)>& inlet_eval,
                           bool show_report,
                           double* avg_delta_mps,
                           double* net_before,
                           double* net_after) {
    const uint Nx = solver.grid().size(0u);
    const uint Ny = solver.grid().size(1u);
    const uint Nz = solver.grid().size(2u);
    const unsigned long long Ntot = solver.grid().cells();

    auto is_downstream = [&](uint x, uint y, uint z) -> bool {
        if (downstream_bc == "+y") return y == Ny - 1u;
        if (downstream_bc == "-y") return y == 0u;
        if (downstream_bc == "+x") return x == Nx - 1u;
        if (downstream_bc == "-x") return x == 0u;
        return false;
    };

    const bool fill_downstream = static_cast<bool>(inlet_eval);
    const uint threads = (Ntot < 4096ull) ? 1u : detect_flux_threads();

    std::vector<std::vector<BoundaryCell>> boundary_per_thread(threads);
    for (uint t = 0u; t < threads; ++t) boundary_per_thread[t].reserve(1024u);

    parallel_for((ulong)Ntot, threads, [&](ulong n, uint t) {
        uint x, y, z;
        solver.grid().coordinates(n, x, y, z);
        if (z == 0u) return;

        const FaceKind fk = pick_face(x, y, z, Nx, Ny, Nz);
        if (fk == FaceKind::None) return;

        const uchar flagsn = solver.flags()[n];
        if ((flagsn & CELL_SOLID) != 0u) return; // keep solids untouched (including terrain-clipped side cells)
        solver.flags()[n] = (uchar)(flagsn | CELL_EQUILIBRIUM); // preserve CELL_THERMAL and any existing auxiliary bits
        boundary_per_thread[t].push_back(BoundaryCell{ (unsigned long long)n, fk });

        if (fill_downstream && is_downstream(x, y, z)) {
            const float3 pos = solver.grid().position(x, y, z);
            const float3 u = inlet_eval(pos);
            solver.velocity().x[n] = u.x; solver.velocity().y[n] = u.y; solver.velocity().z[n] = u.z;
        }
    });

    size_t boundary_count = 0u;
    for (uint t = 0u; t < threads; ++t) boundary_count += boundary_per_thread[t].size();
    std::vector<BoundaryCell> boundary_cells;
    boundary_cells.reserve(boundary_count);
    for (uint t = 0u; t < threads; ++t) {
        boundary_cells.insert(boundary_cells.end(), boundary_per_thread[t].begin(), boundary_per_thread[t].end());
    }

    double S_in = 0.0;
    double S_out = 0.0;
    double net = 0.0;
    double netZTop = 0.0;
    double netXMin = 0.0;
    double netXMax = 0.0;
    double netYMin = 0.0;
    double netYMax = 0.0;
    unsigned long long B = 0ull;

    for (const BoundaryCell& bc : boundary_cells) {
        const unsigned long long n = bc.n;
        const FaceKind fk = bc.face;
        ++B;

        float vn = 0.0f;
        switch (fk) {
            case FaceKind::ZTop: vn =  solver.velocity().z[n]; break;
            case FaceKind::XMin: vn = -solver.velocity().x[n]; break;
            case FaceKind::XMax: vn =  solver.velocity().x[n]; break;
            case FaceKind::YMin: vn = -solver.velocity().y[n]; break;
            case FaceKind::YMax: vn =  solver.velocity().y[n]; break;
            default: break;
        }
        net += static_cast<double>(vn);
        switch (fk) {
            case FaceKind::ZTop: netZTop += static_cast<double>(vn); break;
            case FaceKind::XMin: netXMin += static_cast<double>(vn); break;
            case FaceKind::XMax: netXMax += static_cast<double>(vn); break;
            case FaceKind::YMin: netYMin += static_cast<double>(vn); break;
            case FaceKind::YMax: netYMax += static_cast<double>(vn); break;
            default: break;
        }
        if (vn < 0.0f) S_in  += static_cast<double>(-vn);
        else           S_out += static_cast<double>( vn);
    }

    if (net_before) *net_before = net;

    const unsigned long long B_used = B;
    const double delta = (B_used > 0ull) ? (-net / static_cast<double>(B_used)) : 0.0;

    double sum_delta_abs = 0.0;
    unsigned long long M = 0ull;
    double sumZTop = 0.0;  unsigned long long cntZTop = 0ull;
    double sumXMin = 0.0;  unsigned long long cntXMin = 0ull;
    double sumXMax = 0.0;  unsigned long long cntXMax = 0ull;
    double sumYMin = 0.0;  unsigned long long cntYMin = 0ull;
    double sumYMax = 0.0;  unsigned long long cntYMax = 0ull;

    for (const BoundaryCell& bc : boundary_cells) {
        const unsigned long long n = bc.n;
        const FaceKind fk = bc.face;

        const float ox = solver.velocity().x[n], oy = solver.velocity().y[n], oz = solver.velocity().z[n];

        float* comp = nullptr;
        float sgn = 1.0f;

        switch (fk) {
            case FaceKind::ZTop: comp = &solver.velocity().z[n]; sgn = +1.0f; break;
            case FaceKind::XMin: comp = &solver.velocity().x[n]; sgn = -1.0f; break;
            case FaceKind::XMax: comp = &solver.velocity().x[n]; sgn = +1.0f; break;
            case FaceKind::YMin: comp = &solver.velocity().y[n]; sgn = -1.0f; break;
            case FaceKind::YMax: comp = &solver.velocity().y[n]; sgn = +1.0f; break;
            default: break;
        }

        if (!comp) continue;

        const float before = *comp;
        *comp = before + sgn * static_cast<float>(delta);

        const float dx = solver.velocity().x[n] - ox;
        const float dy = solver.velocity().y[n] - oy;
        const float dz = solver.velocity().z[n] - oz;
        const double dmag = std::sqrt(double(dx) * dx + double(dy) * dy + double(dz) * dz);
        sum_delta_abs += dmag;
        switch (fk) {
        case FaceKind::ZTop: sumZTop += dmag; ++cntZTop; break;
        case FaceKind::XMin: sumXMin += dmag; ++cntXMin; break;
        case FaceKind::XMax: sumXMax += dmag; ++cntXMax; break;
        case FaceKind::YMin: sumYMin += dmag; ++cntYMin; break;
        case FaceKind::YMax: sumYMax += dmag; ++cntYMax; break;
        default: break;
        }

        ++M;

    }

    double net_after_val = 0.0;
    for (const BoundaryCell& bc : boundary_cells) {
        const unsigned long long n = bc.n;
        const FaceKind fk = bc.face;
        float vn = 0.0f;
        switch (fk) {
            case FaceKind::ZTop: vn =  solver.velocity().z[n]; break;
            case FaceKind::XMin: vn = -solver.velocity().x[n]; break;
            case FaceKind::XMax: vn =  solver.velocity().x[n]; break;
            case FaceKind::YMin: vn = -solver.velocity().y[n]; break;
            case FaceKind::YMax: vn =  solver.velocity().y[n]; break;
            default: break;
        }
        net_after_val += static_cast<double>(vn);
    }

    if (avg_delta_mps) *avg_delta_mps = (M > 0ull) ? (sum_delta_abs / double(M)) : 0.0;
    if (net_after)     *net_after     = net_after_val;

    if (show_report) {
        const double avgZTop = (cntZTop ? (sumZTop / double(cntZTop)) : 0.0);
        const double avgXMin = (cntXMin ? (sumXMin / double(cntXMin)) : 0.0);
        const double avgXMax = (cntXMax ? (sumXMax / double(cntXMax)) : 0.0);
        const double avgYMin = (cntYMin ? (sumYMin / double(cntYMin)) : 0.0);
        const double avgYMax = (cntYMax ? (sumYMax / double(cntYMax)) : 0.0);

        println("| Flux correction | S_in=" + to_string(S_in, 3u) + ", S_out=" + to_string(S_out, 3u) +
                ", net_before=" + to_string(net, 3u) + " |");
        println("| Flux correction | net by face: Xn=" + to_string(netXMin, 3u) +
                ", Xp=" + to_string(netXMax, 3u) + ", Yn=" + to_string(netYMin, 3u) +
                ", Yp=" + to_string(netYMax, 3u) + ", Zp(w)=" + to_string(netZTop, 3u) + " |");
        println("| Flux correction | avg_dU=" + to_string(delta, 3u) + " m/s, corrected=" +
                to_string((ulong)B_used) + ", net_after=" + to_string(net_after_val, 3u) + " |");
        println("| Flux correction | per-face dU: Xn=" + to_string(avgXMin, 3u) +
                ", Xp=" + to_string(avgXMax, 3u) + ", Yn=" + to_string(avgYMin, 3u) +
                ", Yp=" + to_string(avgYMax, 3u) + ", Zp=" + to_string(avgZTop, 3u) + " m/s |");
    }
}

void run_case_flux_correction(culbm::solver::Solver& solver, const bool enabled, const std::string& downstream_bc,
                              const std::function<float3(const float3&)>& eval,
                              const std::function<void()>& after) {
    if (!enabled) {
        cases::print_kv_row("Flux correction", "skipped. Set flux_correction=true to enable");
        return;
    }
    cases::print_kv_row("Flux correction", "starting. Time: [" + run::now_str() + "]");
    cases::emit_progress_stage("flux_correction",
                               "Flux correction",
                               "Balancing boundary mass flux",
                               0ll,
                               1ll,
                               true);
    double avg_du = 0.0, net_b = 0.0, net_a = 0.0;
    apply_flux_correction(solver, downstream_bc, eval, /*show_report=*/true,
        &avg_du, &net_b, &net_a);
    cases::emit_progress_stage("flux_correction",
                               "Flux correction",
                               "avg dU = " + to_string(avg_du, 3u) + " m/s, net after = " + to_string(net_a, 3u),
                               1ll,
                               1ll,
                               false);
    if (after) after();
}

} // namespace culbm::bc
