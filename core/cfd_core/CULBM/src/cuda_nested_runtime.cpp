#include "cuda_nested_runtime.hpp"

#include "cuda_nested_interface.hpp"
#include "solver/solver.hpp"
#include "cuda_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace culbm {
namespace {

void nested_require(const bool condition, const string& message) {
    if (!condition) fatal_error("Nested CUDA runtime: " + message);
}

std::uint64_t plane_size(const std::uint32_t nx, const std::uint32_t ny) {
    return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny);
}

std::uint64_t plane_index(const std::uint32_t x,
                          const std::uint32_t y,
                          const std::uint32_t nx) {
    return static_cast<std::uint64_t>(x) +
           static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(nx);
}

bool is_solid(const unsigned char flag) {
    return (flag & CELL_SOLID) == CELL_SOLID;
}

struct OwnedFaceDomain {
    culbm::solver::Subdomain* domain = nullptr;
    std::uint32_t x0 = 0u;
    std::uint32_t y0 = 0u;
    std::uint32_t nx = 0u;
    std::uint32_t ny = 0u;
    std::uint32_t local_z = 0u;
    std::uint32_t hx = 0u;
    std::uint32_t hy = 0u;
};

std::uint32_t face_z(const NestedLbmLevel& level,
                     const NestedPlaneSide side,
                     const NestedPlaneLocation location) {
    nested_require(level.solver != nullptr, "nested level has no LBM instance");
    const culbm::solver::Solver& solver = *level.solver;
    const std::uint32_t nz = solver.grid().size(2u);
    if (location == NestedPlaneLocation::Boundary) {
        const std::uint32_t z = side == NestedPlaneSide::Bottom
            ? level.lower_ghost_z
            : level.upper_ghost_z;
        nested_require(z < nz, "interface ghost row is outside allocated LBM rows");
        return z;
    }
    nested_require(level.physical_first_z < level.physical_end_z &&
                   level.physical_end_z <= nz,
                   "physical z range is invalid for the allocated LBM rows");
    return side == NestedPlaneSide::Bottom
        ? level.physical_first_z
        : level.physical_end_z - 1u;
}

std::vector<OwnedFaceDomain> owned_face_domains(culbm::solver::Solver& solver,
                                                const std::uint32_t z_global) {
    const std::uint32_t dx_count = solver.grid().split(0u);
    const std::uint32_t dy_count = solver.grid().split(1u);
    const std::uint32_t dz_count = solver.grid().split(2u);
    nested_require(dx_count > 0u && dy_count > 0u && dz_count > 0u,
                   "invalid Cartesian decomposition");
    nested_require(solver.grid().size(0u) % dx_count == 0u &&
                   solver.grid().size(1u) % dy_count == 0u &&
                   solver.grid().size(2u) % dz_count == 0u,
                   "nested faces require exactly divisible LBM domains");

    const std::uint32_t nx_inner = solver.grid().size(0u) / dx_count;
    const std::uint32_t ny_inner = solver.grid().size(1u) / dy_count;
    const std::uint32_t nz_inner = solver.grid().size(2u) / dz_count;
    std::vector<OwnedFaceDomain> result;
    for (std::uint32_t local = 0u; local < solver.local_count(); ++local) {
        const std::uint32_t global = solver.distributed()
            ? solver.owned_subdomain()
            : local;
        const std::uint32_t domain_x = (global % (dx_count * dy_count)) % dx_count;
        const std::uint32_t domain_y = (global % (dx_count * dy_count)) / dx_count;
        const std::uint32_t domain_z = global / (dx_count * dy_count);
        const std::uint32_t z0 = domain_z * nz_inner;
        if (z_global < z0 || z_global >= z0 + nz_inner) continue;
        OwnedFaceDomain face;
        face.domain = &solver.local(local);
        face.x0 = domain_x * nx_inner;
        face.y0 = domain_y * ny_inner;
        face.nx = nx_inner;
        face.ny = ny_inner;
        face.hx = dx_count > 1u ? 1u : 0u;
        face.hy = dy_count > 1u ? 1u : 0u;
        const std::uint32_t hz = dz_count > 1u ? 1u : 0u;
        face.local_z = (z_global - z0) + hz;
        result.push_back(face);
    }
    return result;
}

void allreduce_plane_component(std::vector<float>& values) {
    if (!mpi_is_distributed() || values.empty()) return;
    constexpr std::size_t chunk = 64u * 1024u * 1024u;
    for (std::size_t offset = 0u; offset < values.size(); offset += chunk) {
        const std::size_t remaining = values.size() - offset;
        const std::size_t count = std::min(chunk, remaining);
        mpi_allreduce_sum_float(values.data() + offset,
                                static_cast<std::uint32_t>(count));
    }
}

void allreduce_plane_fields(NestedMacroPlane& plane) {
    if (!mpi_is_distributed() || plane.rho.empty()) return;
    const std::size_t count = plane.rho.size();
#ifdef TEMPERATURE
    constexpr std::size_t components = 6u;
#else
    constexpr std::size_t components = 5u;
#endif
    nested_require(
        count <= std::numeric_limits<std::size_t>::max()/components,
        "MPI plane reduction buffer size overflows size_t");

    // A plane is sparse before the collective: every rank fills only its
    // owned x/y region. Pack all macro components and the encoded flags into
    // one collective so an interface exchange performs one MPI latency round
    // instead of five/six. Element order within each component is unchanged.
    thread_local std::vector<float> packed;
    packed.resize(count*components);
    std::size_t offset = 0u;
    const auto append = [&](const std::vector<float>& values) {
        std::copy(values.begin(), values.end(), packed.begin()+offset);
        offset += count;
    };
    append(plane.rho);
    append(plane.ux);
    append(plane.uy);
    append(plane.uz);
#ifdef TEMPERATURE
    append(plane.temperature);
#endif
    for (std::size_t i = 0u; i < count; ++i)
        packed[offset+i] = static_cast<float>(plane.flags[i]);

    allreduce_plane_component(packed);

    offset = 0u;
    const auto extract = [&](std::vector<float>& values) {
        std::copy(packed.begin()+offset, packed.begin()+offset+count,
                  values.begin());
        offset += count;
    };
    extract(plane.rho);
    extract(plane.ux);
    extract(plane.uy);
    extract(plane.uz);
#ifdef TEMPERATURE
    extract(plane.temperature);
#endif
    for (std::size_t i = 0u; i < count; ++i)
        plane.flags[i] = static_cast<unsigned char>(std::lround(packed[offset+i]));
}

void enqueue_read_face(const OwnedFaceDomain& face) {
    culbm::solver::Subdomain& domain = *face.domain;
    const ulong x0 = face.hx;
    const ulong x1 = x0 + face.nx;
    const ulong y0 = face.hy;
    const ulong y1 = y0 + face.ny;
    const ulong z0 = face.local_z;
    const ulong local_nx = domain.extent(0u);
    const ulong local_ny = domain.extent(1u);
    domain.fields().density.download_plane_async(x0, x1, y0, y1, z0,
                                           local_nx, local_ny);
    domain.fields().velocity.download_plane_async(x0, x1, y0, y1, z0,
                                         local_nx, local_ny);
#ifdef TEMPERATURE
    if (domain.fields().temperature.size() > 0u)
        domain.fields().temperature.download_plane_async(x0, x1, y0, y1, z0,
                                             local_nx, local_ny);
#endif
    domain.fields().flags.download_plane_async(x0, x1, y0, y1, z0,
                                             local_nx, local_ny);
}

void enqueue_write_face(const OwnedFaceDomain& face,
                        const bool write_macros,
                        const bool write_flags) {
    culbm::solver::Subdomain& domain = *face.domain;
    const ulong x0 = face.hx;
    const ulong x1 = x0 + face.nx;
    const ulong y0 = face.hy;
    const ulong y1 = y0 + face.ny;
    const ulong z0 = face.local_z;
    const ulong local_nx = domain.extent(0u);
    const ulong local_ny = domain.extent(1u);
    if (write_macros) {
        domain.fields().density.upload_plane_async(x0, x1, y0, y1, z0,
                                              local_nx, local_ny);
        domain.fields().velocity.upload_plane_async(x0, x1, y0, y1, z0,
                                            local_nx, local_ny);
#ifdef TEMPERATURE
        if (domain.fields().temperature.size() > 0u)
            domain.fields().temperature.upload_plane_async(x0, x1, y0, y1, z0,
                                                local_nx, local_ny);
#endif
    }
    if (write_flags) {
        domain.fields().flags.upload_plane_async(x0, x1, y0, y1, z0,
                                                local_nx, local_ny);
    }
}

void copy_face_to_plane(const OwnedFaceDomain& face,
                        NestedMacroPlane& plane) {
    const culbm::solver::Subdomain& domain = *face.domain;
    const ulong local_nx = domain.extent(0u);
    const ulong local_ny = domain.extent(1u);
    const bool has_temp = domain.fields().temperature.size() > 0u;
    for (std::uint32_t y = 0u; y < face.ny; ++y) {
        for (std::uint32_t x = 0u; x < face.nx; ++x) {
            const ulong local = static_cast<ulong>(face.hx + x) +
                (static_cast<ulong>(face.hy + y) +
                 static_cast<ulong>(face.local_z) * local_ny) * local_nx;
            const std::uint64_t global = plane_index(face.x0 + x, face.y0 + y, plane.nx);
            plane.rho[global] = domain.fields().density[local];
            plane.ux[global] = domain.fields().velocity.component(0u)[local];
            plane.uy[global] = domain.fields().velocity.component(1u)[local];
            plane.uz[global] = domain.fields().velocity.component(2u)[local];
#ifdef TEMPERATURE
            plane.temperature[global] = has_temp ? domain.fields().temperature[local] : 1.0f;
#endif
            plane.flags[global] = domain.fields().flags[local];
        }
    }
}

void copy_plane_to_face(const NestedMacroPlane& plane,
                        const OwnedFaceDomain& face,
                        const bool write_macros,
                        const bool write_flags) {
    culbm::solver::Subdomain& domain = *face.domain;
    const ulong local_nx = domain.extent(0u);
    const ulong local_ny = domain.extent(1u);
    const bool has_temp = domain.fields().temperature.size() > 0u;
    for (std::uint32_t y = 0u; y < face.ny; ++y) {
        for (std::uint32_t x = 0u; x < face.nx; ++x) {
            const ulong local = static_cast<ulong>(face.hx + x) +
                (static_cast<ulong>(face.hy + y) +
                 static_cast<ulong>(face.local_z) * local_ny) * local_nx;
            const std::uint64_t global = plane_index(face.x0 + x, face.y0 + y, plane.nx);
            if (write_macros) {
                domain.fields().density[local] = plane.rho[global];
                domain.fields().velocity.component(0u)[local] = plane.ux[global];
                domain.fields().velocity.component(1u)[local] = plane.uy[global];
                domain.fields().velocity.component(2u)[local] = plane.uz[global];
#ifdef TEMPERATURE
                if (has_temp) domain.fields().temperature[local] = plane.temperature[global];
#endif
            }
            if (write_flags) domain.fields().flags[local] = plane.flags[global];
        }
    }
}

void write_plane_internal(NestedLbmLevel& level,
                          const NestedPlaneSide side,
                          const NestedMacroPlane& plane,
                          const NestedPlaneLocation location,
                          const bool write_macros,
                          const bool write_flags) {
    nested_require(level.solver != nullptr, "nested level has no LBM instance");
    culbm::solver::Solver& solver = *level.solver;
    nested_require(plane.valid(), "attempted to write an invalid macro plane");
    nested_require(plane.nx == solver.grid().size(0u) && plane.ny == solver.grid().size(1u),
                   "plane dimensions do not match the target LBM face");
    const std::vector<OwnedFaceDomain> faces =
        owned_face_domains(solver, face_z(level, side, location));
    for (const OwnedFaceDomain& face : faces)
        copy_plane_to_face(plane, face, write_macros, write_flags);
    for (const OwnedFaceDomain& face : faces)
        enqueue_write_face(face, write_macros, write_flags);
    for (const OwnedFaceDomain& face : faces)
        face.domain->synchronize();
}

unsigned char with_interface_boundary_type(const unsigned char existing,
                                           const bool open) {
    const unsigned char boundary_mask =
        static_cast<unsigned char>(CELL_SOLID | CELL_EQUILIBRIUM);
    unsigned char result = static_cast<unsigned char>(
        (existing & ~boundary_mask) | (open ? CELL_EQUILIBRIUM : CELL_SOLID));
#ifdef TEMPERATURE
    // The macro bridge writes the donor temperature to T, not to gi.  Mark an
    // open interface as a prescribed-temperature node so the thermal kernel
    // consumes that bridged macro value instead of reconstructing T from the
    // stale ghost-node distributions.  A closed aperture is an adiabatic
    // internal solid; do not inherit an unrelated seed CELL_THERMAL bit there.
    result = open
        ? static_cast<unsigned char>(result | CELL_THERMAL)
        : static_cast<unsigned char>(result & ~CELL_THERMAL);
#endif
    return result;
}

NestedInterfacePlane to_interface_plane(const NestedMacroPlane& plane) {
    nested_require(plane.valid(), "cannot convert an invalid macro plane");
    NestedInterfacePlane result;
    result.nx = plane.nx;
    result.ny = plane.ny;
    const std::size_t count = static_cast<std::size_t>(plane.size());
    result.macros.resize(count);
    result.flags.resize(count);
    for (std::size_t i = 0u; i < count; ++i) {
        result.macros[i].rho = plane.rho[i];
        result.macros[i].u = { plane.ux[i], plane.uy[i], plane.uz[i] };
        result.macros[i].temperature = plane.temperature[i];
        result.flags[i] = plane.flags[i];
    }
    return result;
}

NestedMacroPlane from_interface_plane(const NestedInterfacePlane& plane) {
    nested_require(plane.has_storage(), "cannot convert an invalid interface plane");
    NestedMacroPlane result;
    result.nx = plane.nx;
    result.ny = plane.ny;
    const std::size_t count = plane.cell_count();
    result.rho.resize(count);
    result.ux.resize(count);
    result.uy.resize(count);
    result.uz.resize(count);
    result.temperature.resize(count);
    result.flags.resize(count);
    for (std::size_t i = 0u; i < count; ++i) {
        result.rho[i] = static_cast<float>(plane.macros[i].rho);
        result.ux[i] = static_cast<float>(plane.macros[i].u[0]);
        result.uy[i] = static_cast<float>(plane.macros[i].u[1]);
        result.uz[i] = static_cast<float>(plane.macros[i].u[2]);
        result.temperature[i] = static_cast<float>(plane.macros[i].temperature);
        result.flags[i] = plane.flags[i];
    }
    return result;
}

} // namespace

std::uint64_t NestedMacroPlane::size() const {
    return plane_size(nx, ny);
}

bool NestedMacroPlane::valid() const {
    const std::uint64_t count = size();
    return nx > 0u && ny > 0u && rho.size() == count && ux.size() == count &&
           uy.size() == count && uz.size() == count && temperature.size() == count &&
           flags.size() == count &&
           (aperture_mask.empty() || aperture_mask.size() == count);
}

NestedMacroPlane NestedMacroPlaneBridge::read(
    NestedLbmLevel& level,
    const NestedPlaneSide side,
    const NestedPlaneLocation location) {
    nested_require(level.solver != nullptr, "nested level has no LBM instance");
    culbm::solver::Solver& solver = *level.solver;
    const std::uint32_t nx = solver.grid().size(0u);
    const std::uint32_t ny = solver.grid().size(1u);
    nested_require(nx > 0u && ny > 0u && solver.grid().size(2u) > 0u,
                   "cannot read a plane from an empty LBM grid");

#ifndef UPDATE_FIELDS
    // With UPDATE_FIELDS the stream/collide kernel keeps rho/u current.  The
    // fallback is retained for custom builds without it; it is a computation,
    // not a full host transfer.
    solver.refresh_macros();
#endif

    NestedMacroPlane result;
    result.nx = nx;
    result.ny = ny;
    const std::uint64_t count = result.size();
    result.rho.resize(count);
    result.ux.resize(count);
    result.uy.resize(count);
    result.uz.resize(count);
#ifdef TEMPERATURE
    // In MPI mode each rank contributes only a sparse x/y segment.  The
    // neutral contribution must be zero; otherwise the default temperature
    // would be added once by every non-owning rank during MPI_SUM.
    result.temperature.resize(count, 0.0f);
#else
    result.temperature.resize(count, 1.0f);
#endif
    // Allreduce uses SUM because each rank owns disjoint x/y cells. A fluid
    // flag is not the additive identity, so distributed sparse entries must
    // begin at zero rather than CELL_FLUID.
    result.flags.assign(
        count, mpi_is_distributed()
            ? static_cast<unsigned char>(0u)
            : static_cast<unsigned char>(CELL_FLUID));

    const std::vector<OwnedFaceDomain> faces =
        owned_face_domains(solver, face_z(level, side, location));
    for (const OwnedFaceDomain& face : faces) enqueue_read_face(face);
    for (const OwnedFaceDomain& face : faces) face.domain->synchronize();
    for (const OwnedFaceDomain& face : faces) {
        copy_face_to_plane(face, result);
    }

    // One rank owns each x/y segment in MPI mode.  Sum the sparse global
    // plane contributions so all ranks can derive the same aperture and then
    // write only their owned plane segment back to the GPU.
    allreduce_plane_fields(result);
    return result;
}

void NestedMacroPlaneBridge::write(NestedLbmLevel& level,
                                   const NestedPlaneSide side,
                                   const NestedMacroPlane& plane,
                                   const NestedPlaneWriteOptions options,
                                   const NestedPlaneLocation location) {
    write_plane_internal(level, side, plane, location, true, options.write_flags);
}

NestedMacroPlane NestedMacroPlaneBridge::restrict_4_to_1(
    const NestedMacroPlane& fine) {
    nested_require(fine.valid(), "attempted to restrict an invalid fine plane");
    nested_require((fine.nx % 2u) == 0u && (fine.ny % 2u) == 0u,
                   "a 4:1 plane restriction requires even fine nx and ny");

    NestedMacroPlane coarse;
    coarse.nx = fine.nx / 2u;
    coarse.ny = fine.ny / 2u;
    const std::uint64_t count = coarse.size();
    coarse.rho.resize(count);
    coarse.ux.resize(count);
    coarse.uy.resize(count);
    coarse.uz.resize(count);
    coarse.temperature.resize(count);
    coarse.flags.resize(count);
    coarse.aperture_mask.resize(count, 0u);

    for (std::uint32_t y = 0u; y < coarse.ny; ++y) {
        for (std::uint32_t x = 0u; x < coarse.nx; ++x) {
            const std::uint32_t fx = 2u * x;
            const std::uint32_t fy = 2u * y;
            const std::uint64_t q[4] = {
                plane_index(fx, fy, fine.nx),
                plane_index(fx + 1u, fy, fine.nx),
                plane_index(fx, fy + 1u, fine.nx),
                plane_index(fx + 1u, fy + 1u, fine.nx)
            };
            const std::uint64_t p = plane_index(x, y, coarse.nx);
            coarse.rho[p] = 0.25f *
                (fine.rho[q[0]] + fine.rho[q[1]] + fine.rho[q[2]] + fine.rho[q[3]]);
            coarse.ux[p] = 0.25f *
                (fine.ux[q[0]] + fine.ux[q[1]] + fine.ux[q[2]] + fine.ux[q[3]]);
            coarse.uy[p] = 0.25f *
                (fine.uy[q[0]] + fine.uy[q[1]] + fine.uy[q[2]] + fine.uy[q[3]]);
            coarse.uz[p] = 0.25f *
                (fine.uz[q[0]] + fine.uz[q[1]] + fine.uz[q[2]] + fine.uz[q[3]]);
            coarse.temperature[p] = 0.25f *
                (fine.temperature[q[0]] + fine.temperature[q[1]] +
                 fine.temperature[q[2]] + fine.temperature[q[3]]);

            std::uint8_t open_mask = 0u;
            for (std::uint32_t child = 0u; child < 4u; ++child) {
                if (!is_solid(fine.flags[q[child]]))
                    open_mask |= static_cast<std::uint8_t>(1u << child);
            }
            coarse.aperture_mask[p] = open_mask;
            // A mixed aperture cannot be represented by one ordinary LBM
            // flag.  Preserve the only unambiguous case here and keep the
            // 4-bit mask for the link-wise bridge.
            coarse.flags[p] = open_mask == 0u
                ? static_cast<unsigned char>(CELL_SOLID)
                : static_cast<unsigned char>(CELL_FLUID);
        }
    }
    return coarse;
}

NestedMacroPlane NestedMacroPlaneBridge::prolong_1_to_4(
    const NestedMacroPlane& coarse) {
    nested_require(coarse.valid(), "attempted to prolong an invalid coarse plane");

    NestedMacroPlane fine;
    fine.nx = coarse.nx * 2u;
    fine.ny = coarse.ny * 2u;
    const std::uint64_t count = fine.size();
    fine.rho.resize(count);
    fine.ux.resize(count);
    fine.uy.resize(count);
    fine.uz.resize(count);
    fine.temperature.resize(count);
    fine.flags.resize(count);

    for (std::uint32_t y = 0u; y < coarse.ny; ++y) {
        for (std::uint32_t x = 0u; x < coarse.nx; ++x) {
            const std::uint64_t p = plane_index(x, y, coarse.nx);
            const std::uint8_t mask = coarse.aperture_mask.empty()
                ? static_cast<std::uint8_t>(0x0Fu)
                : coarse.aperture_mask[p];
            for (std::uint32_t child_y = 0u; child_y < 2u; ++child_y) {
                for (std::uint32_t child_x = 0u; child_x < 2u; ++child_x) {
                    const std::uint32_t child = child_x + 2u * child_y;
                    const std::uint64_t q = plane_index(
                        2u * x + child_x, 2u * y + child_y, fine.nx);
                    fine.rho[q] = coarse.rho[p];
                    fine.ux[q] = coarse.ux[p];
                    fine.uy[q] = coarse.uy[p];
                    fine.uz[q] = coarse.uz[p];
                    fine.temperature[q] = coarse.temperature[p];
                    fine.flags[q] = (mask & (1u << child)) != 0u
                        ? coarse.flags[p]
                        : static_cast<unsigned char>(CELL_SOLID);
                }
            }
        }
    }
    return fine;
}

struct NestedDeviceFacePair {
    culbm::solver::Subdomain* upper_domain = nullptr;
    culbm::solver::Subdomain* lower_domain = nullptr;
    gpu::MirroredBuffer<uint> status; // [0]=invalid, [1]=flags_changed
    gpu::Launch exchange;
    bool status_dirty = false;
};

struct NestedDeviceFace {
    std::vector<NestedDeviceFacePair> pairs;
    bool first_exchange = true;

    static bool match_pairs(NestedLbmLevel& upper, NestedLbmLevel& lower,
                            std::vector<OwnedFaceDomain>& upper_donor,
                            std::vector<OwnedFaceDomain>& upper_ghost,
                            std::vector<OwnedFaceDomain>& lower_donor,
                            std::vector<OwnedFaceDomain>& lower_ghost) {
        if (!upper.solver || !lower.solver ||
            upper.solver->distributed() || lower.solver->distributed()) return false;
        if (lower.solver->grid().size(0u) != 2u * upper.solver->grid().size(0u) ||
            lower.solver->grid().size(1u) != 2u * upper.solver->grid().size(1u)) return false;
        upper_donor = owned_face_domains(*upper.solver, upper.physical_first_z);
        upper_ghost = owned_face_domains(*upper.solver, upper.lower_ghost_z);
        lower_donor = owned_face_domains(*lower.solver, lower.physical_end_z - 1u);
        lower_ghost = owned_face_domains(*lower.solver, lower.upper_ghost_z);
        if (upper_donor.empty() || upper_donor.size() != upper_ghost.size() ||
            upper_donor.size() != lower_donor.size() ||
            upper_donor.size() != lower_ghost.size()) return false;
        for (std::size_t i = 0u; i < upper_donor.size(); ++i) {
            if (upper_donor[i].domain != upper_ghost[i].domain ||
                lower_donor[i].domain != lower_ghost[i].domain) return false;
            if (upper_donor[i].domain->device().context() !=
                lower_donor[i].domain->device().context()) return false;
            if (lower_donor[i].x0 != 2u * upper_donor[i].x0 ||
                lower_donor[i].y0 != 2u * upper_donor[i].y0 ||
                lower_donor[i].nx != 2u * upper_donor[i].nx ||
                lower_donor[i].ny != 2u * upper_donor[i].ny) return false;
        }
        return true;
    }

    bool matches(const std::vector<OwnedFaceDomain>& upper_donor,
                 const std::vector<OwnedFaceDomain>& lower_donor) const {
        if (pairs.size() != upper_donor.size()) return false;
        for (std::size_t i = 0u; i < pairs.size(); ++i) {
            if (pairs[i].upper_domain != upper_donor[i].domain ||
                pairs[i].lower_domain != lower_donor[i].domain) return false;
        }
        return true;
    }

    NestedDeviceFace(const std::vector<OwnedFaceDomain>& upper_donor,
                     const std::vector<OwnedFaceDomain>& upper_ghost,
                     const std::vector<OwnedFaceDomain>& lower_donor,
                     const std::vector<OwnedFaceDomain>& lower_ghost) {
        pairs.reserve(upper_donor.size());
        for (std::size_t i = 0u; i < upper_donor.size(); ++i) {
            NestedDeviceFacePair pair;
            pair.upper_domain = upper_donor[i].domain;
            pair.lower_domain = lower_donor[i].domain;
            auto& c = *pair.upper_domain;
            auto& f = *pair.lower_domain;
            pair.status = gpu::MirroredBuffer<uint>(c.device(), 2u, 1u, gpu::HostMemory::pinned, 0u);
            const ulong inner_count = (ulong)upper_donor[i].nx * (ulong)upper_donor[i].ny;
            const ulong cp = (ulong)c.extent(0u) * (ulong)c.extent(1u);
            const ulong fp = (ulong)f.extent(0u) * (ulong)f.extent(1u);
            const bool c_temp = c.fields().temperature.size() > 0u;
            const bool f_temp = f.fields().temperature.size() > 0u;
            pair.exchange = gpu::Launch(c.device(), "culbm_nested_face_exchange", inner_count).push(
                c.fields().density, c.fields().velocity, c.fields().flags,
                c_temp ? c.fields().temperature : c.fields().density,
                f.fields().density, f.fields().velocity, f.fields().flags,
                f_temp ? f.fields().temperature : f.fields().density,
                upper_donor[i].nx, upper_donor[i].ny,
                upper_donor[i].hx, upper_donor[i].hy, c.extent(0u), cp,
                lower_donor[i].hx, lower_donor[i].hy, f.extent(0u), fp,
                c.cells(), f.cells(),
                (ulong)upper_donor[i].local_z, (ulong)upper_ghost[i].local_z,
                (ulong)lower_donor[i].local_z, (ulong)lower_ghost[i].local_z,
                pair.status);
            pairs.push_back(std::move(pair));
        }
    }
};

void NestedMacroPlaneBridge::exchange(NestedLbmLevel& upper,
                                      NestedLbmLevel& lower,
                                      const NestedPlaneWriteOptions options) {
#ifdef UPDATE_FIELDS
    std::vector<OwnedFaceDomain> upper_donor, upper_ghost, lower_donor, lower_ghost;
    if (NestedDeviceFace::match_pairs(upper, lower, upper_donor, upper_ghost, lower_donor, lower_ghost)) {
        auto& face = upper.lower_face_cache;
        if (!face || !face->matches(upper_donor, lower_donor))
            face = std::make_shared<NestedDeviceFace>(upper_donor, upper_ghost, lower_donor, lower_ghost);
        // Each level owns its own stream on the shared GPU context. Finish fine
        // donors first, reset the flags-changed word only if previously set,
        // launch exchanges and pinned 8-byte status downloads across all GPUs
        // in parallel, then synchronize every upper stream before returning.
        for (auto& pair : face->pairs) pair.lower_domain->synchronize();
        for (auto& pair : face->pairs) {
            if (pair.status_dirty) {
                pair.status[1] = 0u;
                pair.status.upload_async();
                pair.status_dirty = false;
            }
            pair.exchange.submit();
            pair.status.download_async();
        }
        bool any_flags_changed = face->first_exchange;
        face->first_exchange = false;
        for (auto& pair : face->pairs) {
            pair.upper_domain->synchronize();
            nested_require(pair.status[0] == 0u, "non-finite macro or non-positive interface density");
            if (pair.status[1] != 0u) {
                any_flags_changed = true;
                pair.status_dirty = true;
            }
        }
        if (any_flags_changed && (upper.solver->grid().any_split() || lower.solver->grid().any_split())) {
            upper.solver->publish_flags();
            lower.solver->publish_flags();
        }
        return;
    }
#endif
    exchange_host(upper, lower, options);
}

void NestedMacroPlaneBridge::exchange_host(NestedLbmLevel& upper,
                                      NestedLbmLevel& lower,
                                      const NestedPlaneWriteOptions options) {
    // Donor/aperture data always comes from adjacent *interior* rows.  The
    // z=0/Nz-1 rows are reserved interface boundary cells and are only
    // written below; treating them as donors would exchange ghost-to-ghost.
    const NestedMacroPlane lower_top = read(
        lower, NestedPlaneSide::Top, NestedPlaneLocation::AdjacentInterior);
    const NestedMacroPlane upper_bottom = read(
        upper, NestedPlaneSide::Bottom, NestedPlaneLocation::AdjacentInterior);
    const NestedInterfacePlane lower_native = to_interface_plane(lower_top);
    const NestedInterfacePlane upper_native = to_interface_plane(upper_bottom);
    std::vector<NestedInterfaceAperture> apertures;
    nested_require(nested_build_plane_apertures(upper_native, lower_native, apertures),
                   "interface planes are not a valid 2:1 coarse/fine pair");

    NestedInterfacePlane lower_to_upper = upper_native;
    NestedInterfacePlane upper_to_lower = lower_native;
    const NestedInterfaceMacro solid = nested_canonical_solid_macro(1.0);
    nested_require(
        nested_restrict_plane_mean4(
            lower_native, apertures, solid, lower_to_upper).valid,
        "4:1 macro restriction failed");
    nested_require(
        nested_prolongate_plane_constant(
            upper_native, apertures, upper_to_lower).valid,
        "constant macro prolongation failed");

    // Explicitly prepare physical interface boundary flags.  A normal fluid
    // flag here would allow the per-slab periodic halo path to leak through
    // the interface.  Open apertures become CELL_EQUILIBRIUM|CELL_THERMAL in thermal builds so
    // the exchanged macro temperature drives the thermal DDFs; closed
    // apertures and all closed fine slots become CELL_SOLID.
    NestedMacroPlane upper_boundary = read(
        upper, NestedPlaneSide::Bottom, NestedPlaneLocation::Boundary);
    NestedMacroPlane lower_boundary = read(
        lower, NestedPlaneSide::Top, NestedPlaneLocation::Boundary);
    for (std::uint32_t y = 0u; y < upper_boundary.ny; ++y) {
        for (std::uint32_t x = 0u; x < upper_boundary.nx; ++x) {
            const std::size_t parent = static_cast<std::size_t>(
                plane_index(x, y, upper_boundary.nx));
            const NestedInterfaceAperture& aperture = apertures[parent];
            upper_boundary.flags[parent] = with_interface_boundary_type(
                upper_boundary.flags[parent], aperture.is_open());
            const std::uint32_t fx = 2u * x;
            const std::uint32_t fy = 2u * y;
            for (std::uint32_t child_y = 0u; child_y < 2u; ++child_y) {
                for (std::uint32_t child_x = 0u; child_x < 2u; ++child_x) {
                    const std::uint32_t slot = child_x + 2u * child_y;
                    const std::size_t child = static_cast<std::size_t>(
                        plane_index(fx + child_x, fy + child_y, lower_boundary.nx));
                    lower_boundary.flags[child] = with_interface_boundary_type(
                        lower_boundary.flags[child], aperture.slot_is_open(slot));
                }
            }
        }
    }
    write_plane_internal(upper, NestedPlaneSide::Bottom, upper_boundary,
                         NestedPlaneLocation::Boundary, false, true);
    write_plane_internal(lower, NestedPlaneSide::Top, lower_boundary,
                         NestedPlaneLocation::Boundary, false, true);

    // Interface flags are owned by the aperture preparation above; do not
    // overwrite them with donor flags while injecting macro values.
    NestedPlaneWriteOptions macro_options = options;
    macro_options.write_flags = false;
    write(upper, NestedPlaneSide::Bottom,
          from_interface_plane(lower_to_upper), macro_options);
    write(lower, NestedPlaneSide::Top,
          from_interface_plane(upper_to_lower), macro_options);

    // D3Q19 scalar links include z-x/z-y diagonals.  At an x/y domain seam,
    // adjacent physical cells therefore inspect interface flags held in a
    // neighboring domain's halo.  Plane writes update only owned cells, so
    // publish the new open/closed aperture classification before either
    // level advances.
    upper.solver->publish_flags();
    lower.solver->publish_flags();
}

struct NestedVolumetricPair {
    culbm::solver::Subdomain* upper_domain = nullptr;
    culbm::solver::Subdomain* lower_domain = nullptr;
    gpu::MirroredBuffer<uint> status; // [0]=invalid, [1]=flags_changed
    gpu::Buffer<float> stage_mom;     // q_mom_face * (c_nx * c_ny)
    gpu::Buffer<uchar> stage_closed;  // c_nx * c_ny
    gpu::Buffer<float> stage_temp;    // q_temp_face * (c_nx * c_ny) when thermal active
    // 1-cell border staging buffers for multi-GPU x/y exchange of stage_s0
    gpu::MirroredBuffer<float> to_upper, to_lower, from_upper, from_lower;
    gpu::Launch prepare;
    gpu::Launch push_down;
    gpu::Launch stage_s0;
    gpu::Launch pack_stage;
    gpu::Launch unpack_stage;
    gpu::Launch apply_s1;
    uint push_timing_slot = 0u;
    uint stage_timing_slot = 0u;
    uint apply_timing_slot = 0u;
    uint c_nx = 0u;
    uint c_ny = 0u;
    uint border_stride_comp = 0u;
    bool status_dirty = false;
};

struct NestedVolumetricFace {
    std::vector<NestedVolumetricPair> pairs;
    std::uint32_t split_x = 1u;
    std::uint32_t split_y = 1u;
    bool first_prepare = true;

    bool matches(const std::vector<OwnedFaceDomain>& upper_donor,
                 const std::vector<OwnedFaceDomain>& lower_donor) const {
        if (pairs.size() != upper_donor.size()) return false;
        for (std::size_t i = 0u; i < pairs.size(); ++i) {
            if (pairs[i].upper_domain != upper_donor[i].domain ||
                pairs[i].lower_domain != lower_donor[i].domain) return false;
        }
        return true;
    }

    NestedVolumetricFace(const std::vector<OwnedFaceDomain>& upper_donor,
                         const std::vector<OwnedFaceDomain>& upper_ghost,
                         const std::vector<OwnedFaceDomain>& lower_donor,
                         const std::vector<OwnedFaceDomain>& lower_ghost,
                         const std::uint32_t dx_split,
                         const std::uint32_t dy_split)
        : split_x(dx_split), split_y(dy_split) {
        pairs.reserve(upper_donor.size());
        const bool any_xy_split = split_x > 1u || split_y > 1u;
        for (std::size_t i = 0u; i < upper_donor.size(); ++i) {
            NestedVolumetricPair pair;
            pair.upper_domain = upper_donor[i].domain;
            pair.lower_domain = lower_donor[i].domain;
            auto& c = *pair.upper_domain;
            auto& f = *pair.lower_domain;
            const ulong c_inner_count = (ulong)upper_donor[i].nx * (ulong)upper_donor[i].ny;
            const ulong f_inner_count = 4ull * c_inner_count;
            const ulong cp = (ulong)c.extent(0u) * (ulong)c.extent(1u);
            const ulong fp = (ulong)f.extent(0u) * (ulong)f.extent(1u);
            const bool has_temp = c.fields().thermal_ddf.size() > 0u;
            const uint q_mom_face = get_momentum_lattice() == Momentum_Lattice::Q27 ? 9u : 5u;
            const uint q_temp_face = has_temp ? get_temperature_transfers() : 0u;

            pair.c_nx = c.extent(0u);
            pair.c_ny = c.extent(1u);
            pair.border_stride_comp = q_mom_face + q_temp_face + 1u;
            pair.status = gpu::MirroredBuffer<uint>(c.device(), 2u, 1u, gpu::HostMemory::pinned, 0u);
            pair.stage_mom = gpu::Buffer<float>(c.device(), cp, q_mom_face, gpu::Init::zero);
            pair.stage_closed = gpu::Buffer<uchar>(c.device(), cp, 1u, gpu::Init::zero);
            if (has_temp) {
                pair.stage_temp = gpu::Buffer<float>(c.device(), cp, q_temp_face, gpu::Init::zero);
            }
            if (any_xy_split) {
                const ulong max_len = (ulong)std::max(pair.c_nx, pair.c_ny);
                const ulong cap = max_len * (ulong)pair.border_stride_comp;
                pair.to_upper = gpu::MirroredBuffer<float>(c.device(), cap, 1u, gpu::HostMemory::pinned, 0.0f);
                pair.to_lower = gpu::MirroredBuffer<float>(c.device(), cap, 1u, gpu::HostMemory::pinned, 0.0f);
                pair.from_upper = gpu::MirroredBuffer<float>(c.device(), cap, 1u, gpu::HostMemory::pinned, 0.0f);
                pair.from_lower = gpu::MirroredBuffer<float>(c.device(), cap, 1u, gpu::HostMemory::pinned, 0.0f);
                const auto& temp_ref = has_temp ? pair.stage_temp : pair.stage_mom;
                pair.pack_stage = gpu::Launch(c.device(), "culbm_nested_vol_pack_stage", max_len).push(
                    (uint)0u, pair.c_nx, pair.c_ny, q_mom_face, q_temp_face,
                    pair.stage_mom, pair.stage_closed, temp_ref,
                    pair.to_upper, pair.to_lower);
                pair.unpack_stage = gpu::Launch(c.device(), "culbm_nested_vol_unpack_stage", max_len).push(
                    (uint)0u, pair.c_nx, pair.c_ny, q_mom_face, q_temp_face,
                    pair.stage_mom, pair.stage_closed, temp_ref,
                    pair.from_upper, pair.from_lower);
            }

            pair.prepare = gpu::Launch(c.device(), "culbm_nested_vol_prepare", c_inner_count).push(
                c.fields().flags, f.fields().flags,
                upper_donor[i].nx, upper_donor[i].ny,
                upper_donor[i].hx, upper_donor[i].hy, c.extent(0u), cp,
                lower_donor[i].hx, lower_donor[i].hy, f.extent(0u), fp,
                (ulong)upper_donor[i].local_z, (ulong)upper_ghost[i].local_z,
                (ulong)lower_donor[i].local_z, (ulong)lower_ghost[i].local_z,
                pair.status);

            pair.push_down = gpu::Launch(c.device(), "culbm_nested_vol_push_down", f_inner_count);
            pair.push_down.push(c.fields().ddf, c.fields().flags, f.fields().ddf, f.fields().flags);
            if (has_temp) {
                pair.push_down.push(c.fields().thermal_ddf, f.fields().thermal_ddf, (uint)1u);
            }
            pair.push_down.push(
                upper_donor[i].nx, upper_donor[i].ny,
                upper_donor[i].hx, upper_donor[i].hy, c.extent(0u), c.extent(1u), c.extent(2u), (ulong)c.cells(),
                lower_donor[i].hx, lower_donor[i].hy, f.extent(0u), f.extent(1u), f.extent(2u), (ulong)f.cells(),
                upper_donor[i].local_z, upper_ghost[i].local_z,
                lower_donor[i].local_z, lower_ghost[i].local_z);
            pair.push_timing_slot = pair.push_down.slot_count();
            pair.push_down.push((uint)0u, (uint)0u, (uint)0u, pair.status);

            pair.stage_s0 = gpu::Launch(c.device(), "culbm_nested_vol_stage_s0", c_inner_count);
            pair.stage_s0.push(f.fields().ddf, f.fields().flags);
            if (has_temp) {
                pair.stage_s0.push(f.fields().thermal_ddf, (uint)1u, pair.stage_temp);
            }
            pair.stage_s0.push(
                pair.stage_mom, pair.stage_closed,
                upper_donor[i].nx, upper_donor[i].ny,
                upper_donor[i].hx, upper_donor[i].hy, c.extent(0u), c.extent(1u),
                lower_donor[i].hx, lower_donor[i].hy, f.extent(0u), f.extent(1u), f.extent(2u), (ulong)f.cells(),
                lower_donor[i].local_z);
            pair.stage_timing_slot = pair.stage_s0.slot_count();
            pair.stage_s0.push((uint)0u, pair.status);

            pair.apply_s1 = gpu::Launch(c.device(), "culbm_nested_vol_apply_s1", c_inner_count);
            pair.apply_s1.push(c.fields().ddf, c.fields().flags, f.fields().ddf, f.fields().flags);
            if (has_temp) {
                pair.apply_s1.push(c.fields().thermal_ddf, f.fields().thermal_ddf, (uint)1u, pair.stage_temp);
            }
            pair.apply_s1.push(
                pair.stage_mom, pair.stage_closed,
                upper_donor[i].nx, upper_donor[i].ny,
                upper_donor[i].hx, upper_donor[i].hy, c.extent(0u), c.extent(1u), c.extent(2u), (ulong)c.cells(),
                lower_donor[i].hx, lower_donor[i].hy, f.extent(0u), f.extent(1u), f.extent(2u), (ulong)f.cells(),
                upper_donor[i].local_z, upper_ghost[i].local_z,
                lower_donor[i].local_z);
            pair.apply_timing_slot = pair.apply_s1.slot_count();
            pair.apply_s1.push((uint)0u, (uint)0u, pair.status);

            pairs.push_back(std::move(pair));
        }
    }

    void exchange_stage_halo() {
        for (uint axis = 0u; axis < 2u; ++axis) {
            const uint split = axis == 0u ? split_x : split_y;
            if (split <= 1u) continue;
            for (auto& pair : pairs) {
                const uint len = axis == 0u ? pair.c_ny : pair.c_nx;
                pair.pack_stage.over(len).bind(0u, axis).submit();
                pair.upper_domain->device().halo_sources_ready();
            }
            for (auto& pair : pairs) pair.upper_domain->device().halo_copies_begin();
            for (std::size_t i = 0u; i < pairs.size(); ++i) {
                auto& self = pairs[i];
                const auto& pos = self.upper_domain->placement().cell;
                const uint nx_pos = axis == 0u ? ((pos[0] + 1u) % split_x) : pos[0];
                const uint ny_pos = axis == 1u ? ((pos[1] + 1u) % split_y) : pos[1];
                const std::size_t upper_idx = (std::size_t)(nx_pos + ny_pos * split_x);
                auto& nb = pairs[upper_idx];
                const uint len = axis == 0u ? self.c_ny : self.c_nx;
                const std::size_t bytes = (std::size_t)len * (std::size_t)self.border_stride_comp * sizeof(float);
                gpu::copy_halo(nb.from_lower.buffer(), self.to_upper, 0u, bytes);
                gpu::copy_halo(self.from_upper.buffer(), nb.to_lower, 0u, bytes);
            }
            for (auto& pair : pairs) pair.upper_domain->device().halo_copies_end();
            for (auto& pair : pairs) {
                const uint len = axis == 0u ? pair.c_ny : pair.c_nx;
                pair.unpack_stage.over(len).bind(0u, axis).submit();
                pair.upper_domain->device().halo_buffers_consumed();
            }
            for (std::size_t i = 0u; i < pairs.size(); ++i) {
                const auto& pos = pairs[i].upper_domain->placement().cell;
                const uint up_x = axis == 0u ? ((pos[0] + 1u) % split_x) : pos[0];
                const uint up_y = axis == 1u ? ((pos[1] + 1u) % split_y) : pos[1];
                const uint lo_x = axis == 0u ? ((pos[0] + split_x - 1u) % split_x) : pos[0];
                const uint lo_y = axis == 1u ? ((pos[1] + split_y - 1u) % split_y) : pos[1];
                auto& dev = pairs[i].upper_domain->device();
                dev.wait_for_peer_copies(pairs[up_x + up_y * split_x].upper_domain->device());
                if ((lo_x != up_x) || (lo_y != up_y))
                    dev.wait_for_peer_copies(pairs[lo_x + lo_y * split_x].upper_domain->device());
            }
        }
    }
};

bool NestedVolumetricBridge::supported(NestedLbmLevel& upper, NestedLbmLevel& lower) {
    std::vector<OwnedFaceDomain> ud, ug, ld, lg;
    return NestedDeviceFace::match_pairs(upper, lower, ud, ug, ld, lg);
}

void NestedVolumetricBridge::exchange(NestedLbmLevel& upper,
                                      NestedLbmLevel& lower,
                                      const NestedExchangePhase phase) {
    std::vector<OwnedFaceDomain> upper_donor, upper_ghost, lower_donor, lower_ghost;
    nested_require(
        NestedDeviceFace::match_pairs(upper, lower, upper_donor, upper_ghost, lower_donor, lower_ghost),
        "volumetric interface currently requires aligned same-device subdomains");
    auto& face = upper.lower_vol_cache;
    if (!face || !face->matches(upper_donor, lower_donor)) {
        face = std::make_shared<NestedVolumetricFace>(
            upper_donor, upper_ghost, lower_donor, lower_ghost,
            upper.solver->grid().split(0u), upper.solver->grid().split(1u));
    }

    const std::uint64_t t_c = upper.solver->time();
    const std::uint64_t t_f = lower.solver->time();
    const uint c_next_odd = (uint)(t_c & 1ull);
    const uint f_next_odd = (uint)(t_f & 1ull);

    if (phase == NestedExchangePhase::Initial) {
        for (auto& pair : face->pairs) pair.lower_domain->synchronize();
        for (auto& pair : face->pairs) {
            pair.prepare.submit();
            pair.status.download_async();
        }
        bool any_changed = face->first_prepare;
        face->first_prepare = false;
        for (auto& pair : face->pairs) {
            pair.upper_domain->synchronize();
            nested_require(pair.status[0] == 0u, "non-finite population at volumetric interface");
            if (pair.status[1] != 0u) {
                any_changed = true;
                pair.status_dirty = true;
            }
        }
        if (any_changed && (upper.solver->grid().any_split() || lower.solver->grid().any_split())) {
            upper.solver->publish_flags();
            lower.solver->publish_flags();
        }
        return;
    }

    if (phase == NestedExchangePhase::BeforeFinerStep) {
        const uint substep = (uint)(t_f - 2ull * t_c);
        nested_require(substep <= 1u, "volumetric BeforeFinerStep subcycling phase mismatch");
        for (auto& pair : face->pairs) pair.lower_domain->synchronize();
        for (auto& pair : face->pairs) {
            pair.push_down.bind(pair.push_timing_slot, c_next_odd, f_next_odd, substep).submit();
            pair.status.download_async();
        }
        for (auto& pair : face->pairs) {
            pair.upper_domain->synchronize();
            nested_require(pair.status[0] == 0u, "non-finite population during volumetric coarse->fine push");
        }
        return;
    }

    if (phase == NestedExchangePhase::AfterFinerStep) {
        if (t_f == 2ull * t_c) return;
        const uint substep = (uint)(t_f - 2ull * t_c - 1ull);
        nested_require(substep <= 1u, "volumetric AfterFinerStep subcycling phase mismatch");
        for (auto& pair : face->pairs) pair.lower_domain->synchronize();
        if (substep == 0u) {
            for (auto& pair : face->pairs) {
                pair.stage_s0.bind(pair.stage_timing_slot, f_next_odd).submit();
                pair.status.download_async();
            }
            face->exchange_stage_halo();
            for (auto& pair : face->pairs) {
                pair.upper_domain->synchronize();
                nested_require(pair.status[0] == 0u, "non-finite population during volumetric fine->coarse stage");
            }
        } else {
            for (auto& pair : face->pairs) {
                pair.apply_s1.bind(pair.apply_timing_slot, c_next_odd, f_next_odd).submit();
                pair.status.download_async();
            }
            for (auto& pair : face->pairs) {
                pair.upper_domain->synchronize();
                nested_require(pair.status[0] == 0u, "non-finite population during volumetric fine->coarse apply");
            }
        }
        return;
    }
}

NestedLbmScheduler::NestedLbmScheduler(std::vector<NestedLbmLevel> levels)
    : levels_(std::move(levels)) {
    validate_or_fail();
    level_time_si_.assign(levels_.size(), 0.0);
}

void NestedLbmScheduler::set_level_before_step_hook(NestedLevelStepHook hook) {
    before_step_hook_ = std::move(hook);
}

void NestedLbmScheduler::set_level_after_step_hook(NestedLevelStepHook hook) {
    after_step_hook_ = std::move(hook);
}

void NestedLbmScheduler::set_interface_hook(NestedInterfaceHook hook) {
    interface_hook_ = std::move(hook);
}

void NestedLbmScheduler::validate_or_fail() const {
    nested_require(!levels_.empty(), "at least one LBM level is required");
    for (std::size_t i = 0u; i < levels_.size(); ++i) {
        const NestedLbmLevel& level = levels_[i];
        nested_require(level.solver != nullptr,
                       "level " + std::to_string(i) + " has no LBM instance");
        nested_require(std::isfinite(level.dt_si) && level.dt_si > 0.0,
                       "level " + std::to_string(i) + " has an invalid SI time step");
        const std::uint32_t expected_ratio = i == 0u ? 1u : 2u;
        nested_require(level.refinement_to_parent == expected_ratio,
                       "only 2:1 adjacent vertical refinement is supported");
        if (i > 0u) {
            const double expected_dt =
                levels_[i - 1u].dt_si / static_cast<double>(level.refinement_to_parent);
            const double tolerance = 1.0e-10 * std::max(1.0, expected_dt);
            nested_require(std::fabs(level.dt_si - expected_dt) <= tolerance,
                           "adjacent levels must use SI time steps in their 2:1 ratio");
        }
    }
}

void NestedLbmScheduler::call_interface(const std::size_t upper_index,
                                        const NestedExchangePhase phase) {
    if (!interface_hook_ || upper_index + 1u >= levels_.size()) return;
    interface_hook_(levels_[upper_index], levels_[upper_index + 1u], phase);
}

void NestedLbmScheduler::initialize() {
    if (initialized_) return;
    validate_or_fail();
    for (NestedLbmLevel& level : levels_) level.solver->initialize_once();
    for (std::size_t upper = 0u; upper + 1u < levels_.size(); ++upper)
        call_interface(upper, NestedExchangePhase::Initial);
    initialized_ = true;
}

void NestedLbmScheduler::advance_recursive(const std::size_t level_index) {
    NestedLbmLevel& level = levels_[level_index];
    if (level_index + 1u < levels_.size()) {
        const std::uint32_t child_steps = levels_[level_index + 1u].refinement_to_parent;
        for (std::uint32_t child_step = 0u; child_step < child_steps; ++child_step) {
            call_interface(level_index, NestedExchangePhase::BeforeFinerStep);
            advance_recursive(level_index + 1u);
            call_interface(level_index, NestedExchangePhase::AfterFinerStep);
        }
    }

    if (before_step_hook_)
        before_step_hook_(level, level_time_si_[level_index], level.solver->time());
    level.solver->advance();
    level_time_si_[level_index] += level.dt_si;
    if (after_step_hook_)
        after_step_hook_(level, level_time_si_[level_index], level.solver->time());
}

void NestedLbmScheduler::advance_one_coarse_step() {
    if (!initialized_) initialize();
    advance_recursive(0u);
    // Preserve the fully prepared common-time interface state after the
    // parent advance.  The bridge itself remains the strict 4:1 arithmetic
    // restriction plus constant fine copy used by the existing solver path.
    for (std::size_t upper = 0u; upper + 1u < levels_.size(); ++upper)
        call_interface(upper, NestedExchangePhase::AfterFinerStep);
    ++coarse_step_count_;
    physical_time_si_ = level_time_si_.front();
    for (std::size_t i = 1u; i < level_time_si_.size(); ++i) {
        const double tolerance = 1.0e-9 * std::max(1.0, physical_time_si_);
        nested_require(std::fabs(level_time_si_[i] - physical_time_si_) <= tolerance,
                       "subcycling left nested levels at different physical times");
    }
}

void NestedLbmScheduler::finish() {
    for (NestedLbmLevel& level : levels_) level.solver->synchronize();
}

void NestedRunDriver::initialize() {
    scheduler_.initialize();
}

void NestedRunDriver::run_one() {
    scheduler_.advance_one_coarse_step();
}

void NestedRunDriver::run(
    const std::uint64_t coarse_steps,
    const std::function<void(NestedLbmScheduler&)>& after_step) {
    scheduler_.initialize();
    for (std::uint64_t step = 0u; step < coarse_steps; ++step) {
        scheduler_.advance_one_coarse_step();
        if (after_step) after_step(scheduler_);
    }
    scheduler_.finish();
}

} // namespace culbm
