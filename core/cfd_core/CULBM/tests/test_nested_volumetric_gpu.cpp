// GPU regression test for NestedVolumetricBridge (Rohde 2006 volumetric
// population-conserving 2:1 vertical interface):
//   1. Linear Couette shear continuity across the 2:1 interface (zero kink).
//   2. Closed-box mass conservation across a 3-level hierarchy.
//   3. Single-GPU vs Dual-GPU (Dx=2) bit-exact parity (non-thermal & thermal)
//      and SurfaceIBM building-crossing-interface parity.

#include "cuda_nested_solver.hpp"
#include "solver/solver.hpp"
#include "surface_ibm.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

float coriolis_Omegax_lbmu = 0.0f, coriolis_Omegay_lbmu = 0.0f, coriolis_Omegaz_lbmu = 0.0f;
bool buffer_nudging_active = false, top_sponge_active = false;
int buffer_n_cells = 1, buffer_nudge_vertical = 0, buffer_downstream_face_id = 0;
int sponge_n_cells = 1, sponge_ref_mode = 0;
float buffer_inv_tau_lbmu = 0.0f, sponge_inv_tau_lbmu = 0.0f;

namespace {

void configure_plain(const bool thermal, const DDF_Storage storage) {
    set_thermal_lattice_enabled(thermal);
    set_momentum_lattice(Momentum_Lattice::Q19);
    set_ddf_storage(storage);
    Subgrid_Configuration sub;
    sub.enabled = false;
    set_subgrid_configuration(sub);
    culbm::surface_ibm::Config ibm;
    ibm.enabled = false;
    culbm::surface_ibm::set_config(ibm);
    Wall_Model_Configuration wm;
    wm.enabled = false;
    set_wall_model_configuration(wm);
}

void verify_couette_continuity() {
    configure_plain(false, DDF_Storage::FP32);
    constexpr uint nx = 8u, ny = 8u;
    constexpr double height = 20.0; // 0-10 m fine (dx=1.25), 10-20 m coarse (dx=2.5)
    constexpr double u_lid = 0.04;
    auto plan = culbm::make_vertical_nesting_plan({ 10.0 }, 2.5, height, nx, ny, 1u, 1u, 1u);
    culbm::NestedVerticalLbm hierarchy;
    hierarchy.plan = plan;
    hierarchy.coarse_dt_si = 2.5 / 78.0;

    const double z_top = height + 0.5 * 2.5;
    for (auto it = plan.levels.rbegin(); it != plan.levels.rend(); ++it) {
        culbm::NestedOwnedLbmLevel lvl;
        lvl.plan = *it;
        culbm::solver::CaseCoefficients coeff;
        coeff.nu = 0.05f * (float)it->ratio_to_coarse;
        lvl.solver = std::make_unique<culbm::solver::Solver>(it->nx, it->ny, it->allocated_nz, 1u, 1u, 1u, coeff);
        for (uint k = 0u; k < it->allocated_nz; ++k) {
            const double z = it->z_begin_agl_si + ((double)k - 0.5) * it->dx_si;
            const bool ground = (k == 0u && it->z_begin_agl_si == 0.0);
            const bool top = (k == it->physical_end_z() && it->z_end_agl_si >= height - 1e-6);
            for (uint j = 0u; j < it->ny; ++j) {
                for (uint i = 0u; i < it->nx; ++i) {
                    const ulong n = lvl.solver->grid().index(i, j, k);
                    lvl.solver->flags()[n] = ground ? CELL_SOLID : (top ? CELL_EQUILIBRIUM : 0u);
                    lvl.solver->density()[n] = 1.0f;
                    lvl.solver->velocity().x[n] = ground ? 0.0f : (float)(u_lid * std::clamp(z / z_top, 0.0, 1.0));
                    lvl.solver->velocity().y[n] = 0.0f;
                    lvl.solver->velocity().z[n] = 0.0f;
                }
            }
        }
        hierarchy.levels.push_back(std::move(lvl));
    }

    culbm::NestedLbmScheduler scheduler(hierarchy.make_scheduler_levels());
    scheduler.set_interface_hook([](culbm::NestedLbmLevel& u, culbm::NestedLbmLevel& l, culbm::NestedExchangePhase ph) {
        if (!culbm::NestedVolumetricBridge::supported(u, l))
            throw std::runtime_error("Volumetric bridge unexpectedly unsupported");
        culbm::NestedVolumetricBridge::exchange(u, l, ph);
    });
    scheduler.initialize();
    for (int s = 0; s < 1000; ++s) scheduler.advance_one_coarse_step();
    scheduler.finish();

    std::vector<std::pair<double, double>> profile;
    for (auto it = hierarchy.levels.rbegin(); it != hierarchy.levels.rend(); ++it) {
        it->solver->velocity().download();
        const auto& p = it->plan;
        for (uint k = p.physical_first_z(); k < p.physical_end_z(); ++k) {
            const double z = p.z_begin_agl_si + ((double)k - 0.5) * p.dx_si;
            double sum = 0.0;
            for (uint j = 0u; j < p.ny; ++j)
                for (uint i = 0u; i < p.nx; ++i)
                    sum += it->solver->velocity().x[it->solver->grid().index(i, j, k)];
            profile.push_back({ z, sum / (double)(p.nx * p.ny) });
        }
    }
    const double slope_fine = (profile[7].second - profile[0].second) / (profile[7].first - profile[0].first);
    const double slope_iface = (profile[8].second - profile[7].second) / (profile[8].first - profile[7].first);
    const double slope_coarse = (profile[11].second - profile[8].second) / (profile[11].first - profile[8].first);
    const double kink_rel = std::abs(slope_iface - slope_fine) / std::abs(slope_fine);
    const double coarse_rel = std::abs(slope_coarse - slope_fine) / std::abs(slope_fine);
    std::cout << "COUETTE slope_fine=" << slope_fine << " slope_iface=" << slope_iface
              << " slope_coarse=" << slope_coarse << " kink_rel=" << kink_rel << "\n";
    if (!(kink_rel < 1.0e-3 && coarse_rel < 1.0e-3))
        throw std::runtime_error("Couette shear kink exceeds 0.1%");
}

void verify_mass_conservation() {
    configure_plain(false, DDF_Storage::FP32);
    constexpr uint nx = 16u, ny = 16u;
    auto plan = culbm::make_vertical_nesting_plan({ 5.0, 10.0 }, 2.5, 30.0, nx, ny, 1u, 1u, 1u);
    culbm::NestedVerticalLbm hierarchy;
    hierarchy.plan = plan;
    hierarchy.coarse_dt_si = 2.5 / 78.0;

    for (auto it = plan.levels.rbegin(); it != plan.levels.rend(); ++it) {
        culbm::NestedOwnedLbmLevel lvl;
        lvl.plan = *it;
        culbm::solver::CaseCoefficients coeff;
        coeff.nu = 0.02f * (float)it->ratio_to_coarse;
        lvl.solver = std::make_unique<culbm::solver::Solver>(it->nx, it->ny, it->allocated_nz, 1u, 1u, 1u, coeff);
        for (uint k = 0u; k < it->allocated_nz; ++k) {
            const double z = it->z_begin_agl_si + ((double)k - 0.5) * it->dx_si;
            const bool wall = (k == 0u && it->z_begin_agl_si == 0.0) ||
                              (k == it->physical_end_z() && it->z_end_agl_si >= 29.999);
            for (uint j = 0u; j < it->ny; ++j) {
                for (uint i = 0u; i < it->nx; ++i) {
                    const ulong n = lvl.solver->grid().index(i, j, k);
                    lvl.solver->flags()[n] = wall ? CELL_SOLID : 0u;
                    if (wall) {
                        lvl.solver->density()[n] = 1.0f;
                        lvl.solver->velocity().x[n] = lvl.solver->velocity().y[n] = lvl.solver->velocity().z[n] = 0.0f;
                    } else {
                        const double x = (i + 0.5) * it->dx_si;
                        const double y = (j + 0.5) * it->dx_si;
                        const double s1 = std::sin(2.0 * 3.141592653589793 * x / 40.0);
                        const double s2 = std::cos(2.0 * 3.141592653589793 * y / 40.0);
                        const double sz = std::sin(3.141592653589793 * z / 30.0);
                        lvl.solver->density()[n] = (float)(1.0 + 0.02 * s1 * s2 * sz);
                        lvl.solver->velocity().x[n] = (float)(0.04 * sz + 0.015 * s2 * sz);
                        lvl.solver->velocity().y[n] = (float)(0.015 * s1 * sz);
                        lvl.solver->velocity().z[n] = (float)(0.02 * s1 * s2 * sz);
                    }
                }
            }
        }
        hierarchy.levels.push_back(std::move(lvl));
    }

    culbm::NestedLbmScheduler scheduler(hierarchy.make_scheduler_levels());
    scheduler.set_interface_hook([](culbm::NestedLbmLevel& u, culbm::NestedLbmLevel& l, culbm::NestedExchangePhase ph) {
        culbm::NestedVolumetricBridge::exchange(u, l, ph);
    });
    scheduler.initialize();

    auto total_mass = [&]() {
        double m = 0.0;
        for (auto& lvl : hierarchy.levels) {
            lvl.solver->density().download();
            const auto& p = lvl.plan;
            const double dV = p.dx_si * p.dx_si * p.dx_si;
            for (uint k = p.physical_first_z(); k < p.physical_end_z(); ++k)
                for (uint j = 0u; j < p.ny; ++j)
                    for (uint i = 0u; i < p.nx; ++i)
                        m += (double)lvl.solver->density()[lvl.solver->grid().index(i, j, k)] * dV;
        }
        return m;
    };

    const double m0 = total_mass();
    for (int s = 0; s < 100; ++s) scheduler.advance_one_coarse_step();
    scheduler.finish();
    const double m100 = total_mass();
    const double rel = std::abs(m100 - m0) / m0;
    std::cout << "MASS m0=" << std::setprecision(10) << m0 << " m100=" << m100
              << " rel=" << std::scientific << rel << std::defaultfloat << "\n";
    if (!(rel < 2.0e-5))
        throw std::runtime_error("Volumetric nested mass drift exceeds tolerance");
}

std::vector<float> run_building_case(const uint domains, const bool thermal, const bool ibm_enabled) {
    configure_plain(thermal, DDF_Storage::FP16C);
    culbm::surface_ibm::Config ibm;
    ibm.enabled = ibm_enabled;
    culbm::surface_ibm::set_config(ibm);

    constexpr uint nx = 28u, ny = 20u, nz = 15u;
    auto plan = culbm::make_vertical_nesting_plan({ 5.0, 10.0 }, 2.5, 32.5, nx, ny, domains, 1u, 1u);
    culbm::NestedSeedState seed;
    seed.nx = nx; seed.ny = ny; seed.nz = nz;
    const std::size_t n = (std::size_t)seed.cell_count();
    seed.rho.assign(n, 1.0f);
    seed.ux.assign(n, 0.025f);
    seed.uy.assign(n, 0.005f);
    seed.uz.assign(n, 0.0f);
    seed.flags.assign(n, 0u);
    if (thermal) {
        seed.temperature.assign(n, 1.0f);
        for (uint k = 0u; k < nz; ++k)
            for (uint j = 0u; j < ny; ++j)
                for (uint i = 0u; i < nx; ++i)
                    seed.temperature[(std::size_t)i + (std::size_t)nx * ((std::size_t)j + (std::size_t)ny * k)] =
                        1.0f + 0.04f * (float)k / (float)nz;
    }
    for (uint i = 0u; i < nx * ny; ++i) {
        seed.flags[i] = CELL_SOLID;
        seed.ux[i] = seed.uy[i] = 0.0f;
        if (thermal && !ibm_enabled) {
            seed.flags[i] |= CELL_THERMAL;
            seed.temperature[i] = 1.05f;
        }
    }

    const float3 a(9.25f, 5.25f, 0.5f), x(6.0f, 1.5f, 0.0f), y(-0.75f, 3.0f, 0.0f), z(0.0f, 0.0f, 6.0f);
    const std::array<float3, 8> v = { a, a + x, a + x + y, a + y, a + z, a + x + z, a + x + y + z, a + y + z };
    const uint f[12][3] = { {0,2,1},{0,3,2},{4,5,6},{4,6,7},{0,1,5},{0,5,4},
                            {1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7} };
    Mesh mesh(12u, float3(0.0f));
    for (uint t = 0u; t < 12u; ++t) { mesh.p0[t] = v[f[t][0]]; mesh.p1[t] = v[f[t][1]]; mesh.p2[t] = v[f[t][2]]; }
    mesh.find_bounds();

    culbm::NestedLbmPhysics phys;
    phys.coarse_nu = 0.01f;
    phys.coarse_alpha = thermal ? 0.014f : 0.0f;
    phys.coarse_dt_si = 2.5 / 78.0;

    culbm::NestedVerticalLbm hierarchy;
    std::string err;
    if (!culbm::build_nested_vertical_lbm(plan, seed, mesh, domains, 1u, 1u, phys, hierarchy, err))
        throw std::runtime_error(err);

    culbm::NestedLbmScheduler scheduler(hierarchy.make_scheduler_levels());
    scheduler.set_interface_hook([](culbm::NestedLbmLevel& u, culbm::NestedLbmLevel& l, culbm::NestedExchangePhase ph) {
        culbm::NestedVolumetricBridge::exchange(u, l, ph);
    });
    scheduler.initialize();
    for (int s = 0; s < 24; ++s) scheduler.advance_one_coarse_step();
    scheduler.finish();

    std::vector<float> out;
    for (auto& lvl : hierarchy.levels) {
        auto& slv = *lvl.solver;
        slv.density().download();
        slv.velocity().download();
        if (thermal) slv.temperature().download();
        const auto& p = lvl.plan;
        for (uint k = p.physical_first_z(); k < p.physical_end_z(); ++k) {
            for (uint j = 0u; j < p.ny; ++j) {
                for (uint i = 0u; i < p.nx; ++i) {
                    const ulong idx = slv.grid().index(i, j, k);
                    out.push_back(slv.density()[idx]);
                    out.push_back(slv.velocity().x[idx]);
                    out.push_back(slv.velocity().y[idx]);
                    out.push_back(slv.velocity().z[idx]);
                    if (thermal) out.push_back(slv.temperature()[idx]);
                }
            }
        }
    }
    return out;
}

void verify_dual_gpu_parity(const bool thermal, const bool ibm_enabled, const double tol) {
    const auto a = run_building_case(1u, thermal, ibm_enabled);
    const auto b = run_building_case(2u, thermal, ibm_enabled);
    if (a.size() != b.size()) throw std::runtime_error("Single vs Dual output size mismatch");
    double max_diff = 0.0;
    std::size_t diff_cnt = 0u;
    for (std::size_t i = 0u; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            throw std::runtime_error("Non-finite field value");
        const double d = std::abs((double)a[i] - (double)b[i]);
        if (d > 0.0) ++diff_cnt;
        if (d > max_diff) max_diff = d;
    }
    std::cout << "DUAL_PARITY thermal=" << thermal << " ibm=" << ibm_enabled
              << " values=" << a.size() << " diff_cnt=" << diff_cnt
              << " max_abs_diff=" << std::scientific << max_diff << std::defaultfloat << "\n";
    if (max_diff > tol)
        throw std::runtime_error("Dual-GPU volumetric parity exceeded tolerance");
}

} // namespace

int main() {
    try {
        if (cuInit(0u) != CUDA_SUCCESS) {
            std::cout << "SKIP: CUDA driver unavailable.\n";
            return 77;
        }
        int gpu_count = 0;
        if (cuDeviceGetCount(&gpu_count) != CUDA_SUCCESS || gpu_count <= 0) {
            std::cout << "SKIP: no CUDA device available.\n";
            return 77;
        }
        verify_couette_continuity();
        verify_mass_conservation();
        if (gpu_count >= 2) {
            verify_dual_gpu_parity(false, false, 0.0);
            verify_dual_gpu_parity(true, false, 0.0);
            verify_dual_gpu_parity(false, true, 5.0e-5);
            verify_dual_gpu_parity(true, true, 1.0e-4);
        } else {
            std::cout << "Single GPU host: skipping Dx=2 dual-GPU parity checks.\n";
        }
        std::cout << "NESTED_VOLUMETRIC_GPU_PASS\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "NESTED_VOLUMETRIC_GPU_FAIL: " << err.what() << "\n";
        return 1;
    }
}
