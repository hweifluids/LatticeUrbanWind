#include "fluxcorrection.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

enum class FaceKind { ZTop, XMin, XMax, YMin, YMax, None };

static inline FaceKind pick_face(uint x, uint y, uint z, uint Nx, uint Ny, uint Nz) {
    if (z == Nz - 1u) return FaceKind::ZTop;
    if (x == 0u)      return FaceKind::XMin;
    if (x == Nx - 1u) return FaceKind::XMax;
    if (y == 0u)      return FaceKind::YMin;
    if (y == Ny - 1u) return FaceKind::YMax;
    return FaceKind::None;
}

void apply_flux_correction(LBM& lbm,
                           const std::string& downstream_bc,
                           const std::function<float3(const float3&)>& inlet_eval,
                           bool show_report,
                           double* avg_delta_mps,
                           double* net_before,
                           double* net_after) {
    const uint Nx = lbm.get_Nx();
    const uint Ny = lbm.get_Ny();
    const uint Nz = lbm.get_Nz();
    const unsigned long long Ntot = lbm.get_N();

    auto is_downstream = [&](uint x, uint y, uint z) -> bool {
        if (downstream_bc == "+y") return y == Ny - 1u;
        if (downstream_bc == "-y") return y == 0u;
        if (downstream_bc == "+x") return x == Nx - 1u;
        if (downstream_bc == "-x") return x == 0u;
        return false;
    };

    const bool fill_downstream = static_cast<bool>(inlet_eval);

    for (unsigned long long n = 0ull; n < Ntot; ++n) {
        uint x=0u, y=0u, z=0u;
        lbm.coordinates(n, x, y, z);
        if (z == 0u) continue;
        const bool on_outer = (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == Nz - 1u);
        if (!on_outer) continue;

        lbm.flags[n] = TYPE_E;

        if (fill_downstream && is_downstream(x, y, z)) {
            const float3 pos = lbm.position(x, y, z);
            const float3 u   = inlet_eval(pos);
            lbm.u.x[n] = u.x; lbm.u.y[n] = u.y; lbm.u.z[n] = u.z;
        }
    }

    double S_in = 0.0;
    double S_out = 0.0;
    double net = 0.0;
    unsigned long long B = 0ull;

    for (unsigned long long n = 0ull; n < Ntot; ++n) {
        uint x=0u, y=0u, z=0u;
        lbm.coordinates(n, x, y, z);
        if (z == 0u) continue;
        const FaceKind fk = pick_face(x, y, z, Nx, Ny, Nz);
        if (fk == FaceKind::None) continue;
        ++B;

        float vn = 0.0f;
        switch (fk) {
            case FaceKind::ZTop: vn =  lbm.u.z[n]; break;
            case FaceKind::XMin: vn = -lbm.u.x[n]; break;
            case FaceKind::XMax: vn =  lbm.u.x[n]; break;
            case FaceKind::YMin: vn = -lbm.u.y[n]; break;
            case FaceKind::YMax: vn =  lbm.u.y[n]; break;
            default: break;
        }
        net += static_cast<double>(vn);
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

    for (unsigned long long n = 0ull; n < Ntot; ++n) {
        uint x=0u, y=0u, z=0u;
        lbm.coordinates(n, x, y, z);
        if (z == 0u) continue;
        const FaceKind fk = pick_face(x, y, z, Nx, Ny, Nz);
        if (fk == FaceKind::None) continue;

        const float ox = lbm.u.x[n], oy = lbm.u.y[n], oz = lbm.u.z[n];

        float* comp = nullptr;
        float sgn = 1.0f;

        switch (fk) {
            case FaceKind::ZTop: comp = &lbm.u.z[n]; sgn = +1.0f; break;
            case FaceKind::XMin: comp = &lbm.u.x[n]; sgn = -1.0f; break;
            case FaceKind::XMax: comp = &lbm.u.x[n]; sgn = +1.0f; break;
            case FaceKind::YMin: comp = &lbm.u.y[n]; sgn = -1.0f; break;
            case FaceKind::YMax: comp = &lbm.u.y[n]; sgn = +1.0f; break;
            default: break;
        }

        if (!comp) continue;

        const float before = *comp;
        *comp = before + sgn * static_cast<float>(delta);

        const float dx = lbm.u.x[n] - ox;
        const float dy = lbm.u.y[n] - oy;
        const float dz = lbm.u.z[n] - oz;
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
    for (unsigned long long n = 0ull; n < Ntot; ++n) {
        uint x=0u, y=0u, z=0u;
        lbm.coordinates(n, x, y, z);
        if (z == 0u) continue;
        const FaceKind fk = pick_face(x, y, z, Nx, Ny, Nz);
        if (fk == FaceKind::None) continue;
        float vn = 0.0f;
        switch (fk) {
            case FaceKind::ZTop: vn =  lbm.u.z[n]; break;
            case FaceKind::XMin: vn = -lbm.u.x[n]; break;
            case FaceKind::XMax: vn =  lbm.u.x[n]; break;
            case FaceKind::YMin: vn = -lbm.u.y[n]; break;
            case FaceKind::YMax: vn =  lbm.u.y[n]; break;
            default: break;
        }
        net_after_val += static_cast<double>(vn);
    }

    if (avg_delta_mps) *avg_delta_mps = (M > 0ull) ? (sum_delta_abs / double(M)) : 0.0;
    if (net_after)     *net_after     = net_after_val;

    if (show_report) {
        std::fprintf(stdout,
            "| [flux correction] S_in=%.3e, S_out=%.3e, net_before=%.3e    |\n", S_in, S_out, net);
        std::fprintf(stdout,
            "|       avg|du|=%.3e m/s, Corrected cell: %llu, net_after=%.3e   |\n", 
            delta, (unsigned long long)B_used, net_after_val);

        const double avgZTop = (cntZTop ? (sumZTop / double(cntZTop)) : 0.0);
        const double avgXMin = (cntXMin ? (sumXMin / double(cntXMin)) : 0.0);
        const double avgXMax = (cntXMax ? (sumXMax / double(cntXMax)) : 0.0);
        const double avgYMin = (cntYMin ? (sumYMin / double(cntYMin)) : 0.0);
        const double avgYMax = (cntYMax ? (sumYMax / double(cntYMax)) : 0.0);

        std::fprintf(stdout,
            "|             Per-faced avg|du|: XMin=%.3e, XMax=%.3e m/s           |\n",
            avgXMin, avgXMax);
        std::fprintf(stdout,
            "|          (cont.) ZTop=%.3e, YMin=%.3e, YMax=%.3e m/s          |\n",
            avgZTop, avgYMin, avgYMax);
    }
}
