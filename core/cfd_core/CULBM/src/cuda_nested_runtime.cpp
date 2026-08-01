#include "cuda_nested_runtime.hpp"

#include "cuda_nested_interface.hpp"
#include "lbm.hpp"
#include "cuda_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace culbm {
namespace {

void nested_require(const bool condition, const string& message) {
    if (!condition) print_error("Nested CUDA runtime: " + message);
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
    return (flag & TYPE_S) == TYPE_S;
}

struct OwnedFaceDomain {
    LBM_Domain* domain = nullptr;
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
    nested_require(level.lbm != nullptr, "nested level has no LBM instance");
    const LBM& lbm = *level.lbm;
    const std::uint32_t nz = lbm.get_Nz();
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

std::vector<OwnedFaceDomain> owned_face_domains(LBM& lbm,
                                                const std::uint32_t z_global) {
    const std::uint32_t dx_count = lbm.get_Dx();
    const std::uint32_t dy_count = lbm.get_Dy();
    const std::uint32_t dz_count = lbm.get_Dz();
    nested_require(dx_count > 0u && dy_count > 0u && dz_count > 0u,
                   "invalid Cartesian decomposition");
    nested_require(lbm.get_Nx() % dx_count == 0u &&
                   lbm.get_Ny() % dy_count == 0u &&
                   lbm.get_Nz() % dz_count == 0u,
                   "nested faces require exactly divisible LBM domains");

    const std::uint32_t nx_inner = lbm.get_Nx() / dx_count;
    const std::uint32_t ny_inner = lbm.get_Ny() / dy_count;
    const std::uint32_t nz_inner = lbm.get_Nz() / dz_count;
    std::vector<OwnedFaceDomain> result;
    for (std::uint32_t local = 0u; local < lbm.get_D(); ++local) {
        const std::uint32_t global = lbm.is_distributed()
            ? lbm.get_global_domain_id()
            : local;
        const std::uint32_t domain_x = (global % (dx_count * dy_count)) % dx_count;
        const std::uint32_t domain_y = (global % (dx_count * dy_count)) / dx_count;
        const std::uint32_t domain_z = global / (dx_count * dy_count);
        const std::uint32_t z0 = domain_z * nz_inner;
        if (z_global < z0 || z_global >= z0 + nz_inner) continue;
        OwnedFaceDomain face;
        face.domain = lbm.lbm_domain[local];
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
    LBM_Domain& domain = *face.domain;
    const ulong x0 = face.hx;
    const ulong x1 = x0 + face.nx;
    const ulong y0 = face.hy;
    const ulong y1 = y0 + face.ny;
    const ulong z0 = face.local_z;
    const ulong local_nx = domain.get_Nx();
    const ulong local_ny = domain.get_Ny();
    domain.rho.read_from_device_2d_pitched(x0, x1, y0, y1, z0,
                                           local_nx, local_ny, -1, false);
    domain.u.read_from_device_2d_pitched(x0, x1, y0, y1, z0,
                                         local_nx, local_ny, -1, false);
#ifdef TEMPERATURE
    domain.T.read_from_device_2d_pitched(x0, x1, y0, y1, z0,
                                         local_nx, local_ny, -1, false);
#endif
    domain.flags.read_from_device_2d_pitched(x0, x1, y0, y1, z0,
                                             local_nx, local_ny, -1, false);
}

void enqueue_write_face(const OwnedFaceDomain& face,
                        const bool write_macros,
                        const bool write_flags) {
    LBM_Domain& domain = *face.domain;
    const ulong x0 = face.hx;
    const ulong x1 = x0 + face.nx;
    const ulong y0 = face.hy;
    const ulong y1 = y0 + face.ny;
    const ulong z0 = face.local_z;
    const ulong local_nx = domain.get_Nx();
    const ulong local_ny = domain.get_Ny();
    if (write_macros) {
        domain.rho.write_to_device_2d_pitched(x0, x1, y0, y1, z0,
                                              local_nx, local_ny, -1, false);
        domain.u.write_to_device_2d_pitched(x0, x1, y0, y1, z0,
                                            local_nx, local_ny, -1, false);
#ifdef TEMPERATURE
        domain.T.write_to_device_2d_pitched(x0, x1, y0, y1, z0,
                                            local_nx, local_ny, -1, false);
#endif
    }
    if (write_flags) {
        domain.flags.write_to_device_2d_pitched(x0, x1, y0, y1, z0,
                                                local_nx, local_ny, -1, false);
    }
}

void copy_face_to_plane(const OwnedFaceDomain& face,
                        NestedMacroPlane& plane) {
    const LBM_Domain& domain = *face.domain;
    const ulong local_nx = domain.get_Nx();
    const ulong local_ny = domain.get_Ny();
    for (std::uint32_t y = 0u; y < face.ny; ++y) {
        for (std::uint32_t x = 0u; x < face.nx; ++x) {
            const ulong local = static_cast<ulong>(face.hx + x) +
                (static_cast<ulong>(face.hy + y) +
                 static_cast<ulong>(face.local_z) * local_ny) * local_nx;
            const std::uint64_t global = plane_index(face.x0 + x, face.y0 + y, plane.nx);
            plane.rho[global] = domain.rho[local];
            plane.ux[global] = domain.u.x[local];
            plane.uy[global] = domain.u.y[local];
            plane.uz[global] = domain.u.z[local];
#ifdef TEMPERATURE
            plane.temperature[global] = domain.T[local];
#endif
            plane.flags[global] = domain.flags[local];
        }
    }
}

void copy_plane_to_face(const NestedMacroPlane& plane,
                        const OwnedFaceDomain& face,
                        const bool write_macros,
                        const bool write_flags) {
    LBM_Domain& domain = *face.domain;
    const ulong local_nx = domain.get_Nx();
    const ulong local_ny = domain.get_Ny();
    for (std::uint32_t y = 0u; y < face.ny; ++y) {
        for (std::uint32_t x = 0u; x < face.nx; ++x) {
            const ulong local = static_cast<ulong>(face.hx + x) +
                (static_cast<ulong>(face.hy + y) +
                 static_cast<ulong>(face.local_z) * local_ny) * local_nx;
            const std::uint64_t global = plane_index(face.x0 + x, face.y0 + y, plane.nx);
            if (write_macros) {
                domain.rho[local] = plane.rho[global];
                domain.u.x[local] = plane.ux[global];
                domain.u.y[local] = plane.uy[global];
                domain.u.z[local] = plane.uz[global];
#ifdef TEMPERATURE
                domain.T[local] = plane.temperature[global];
#endif
            }
            if (write_flags) domain.flags[local] = plane.flags[global];
        }
    }
}

void write_plane_internal(NestedLbmLevel& level,
                          const NestedPlaneSide side,
                          const NestedMacroPlane& plane,
                          const NestedPlaneLocation location,
                          const bool write_macros,
                          const bool write_flags) {
    nested_require(level.lbm != nullptr, "nested level has no LBM instance");
    LBM& lbm = *level.lbm;
    nested_require(plane.valid(), "attempted to write an invalid macro plane");
    nested_require(plane.nx == lbm.get_Nx() && plane.ny == lbm.get_Ny(),
                   "plane dimensions do not match the target LBM face");
    const std::vector<OwnedFaceDomain> faces =
        owned_face_domains(lbm, face_z(level, side, location));
    for (const OwnedFaceDomain& face : faces)
        copy_plane_to_face(plane, face, write_macros, write_flags);
    for (const OwnedFaceDomain& face : faces)
        enqueue_write_face(face, write_macros, write_flags);
    for (const OwnedFaceDomain& face : faces)
        face.domain->finish_queue();
}

unsigned char with_interface_boundary_type(const unsigned char existing,
                                           const bool open) {
    const unsigned char boundary_mask =
        static_cast<unsigned char>(TYPE_S | TYPE_E);
    unsigned char result = static_cast<unsigned char>(
        (existing & ~boundary_mask) | (open ? TYPE_E : TYPE_S));
#ifdef TEMPERATURE
    // The macro bridge writes the donor temperature to T, not to gi.  Mark an
    // open interface as a prescribed-temperature node so the thermal kernel
    // consumes that bridged macro value instead of reconstructing T from the
    // stale ghost-node distributions.  A closed aperture is an adiabatic
    // internal solid; do not inherit an unrelated seed TYPE_T bit there.
    result = open
        ? static_cast<unsigned char>(result | TYPE_T)
        : static_cast<unsigned char>(result & ~TYPE_T);
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
    nested_require(level.lbm != nullptr, "nested level has no LBM instance");
    LBM& lbm = *level.lbm;
    const std::uint32_t nx = lbm.get_Nx();
    const std::uint32_t ny = lbm.get_Ny();
    nested_require(nx > 0u && ny > 0u && lbm.get_Nz() > 0u,
                   "cannot read a plane from an empty LBM grid");

#ifndef UPDATE_FIELDS
    // With UPDATE_FIELDS the stream/collide kernel keeps rho/u current.  The
    // fallback is retained for custom builds without it; it is a computation,
    // not a full host transfer.
    lbm.update_fields();
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
    // begin at zero rather than TYPE_F.
    result.flags.assign(
        count, mpi_is_distributed()
            ? static_cast<unsigned char>(0u)
            : static_cast<unsigned char>(TYPE_F));

    const std::vector<OwnedFaceDomain> faces =
        owned_face_domains(lbm, face_z(level, side, location));
    for (const OwnedFaceDomain& face : faces) enqueue_read_face(face);
    for (const OwnedFaceDomain& face : faces) face.domain->finish_queue();
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
                ? static_cast<unsigned char>(TYPE_S)
                : static_cast<unsigned char>(TYPE_F);
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
                        : static_cast<unsigned char>(TYPE_S);
                }
            }
        }
    }
    return fine;
}

void NestedMacroPlaneBridge::exchange(NestedLbmLevel& upper,
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
    // the interface.  Open apertures become TYPE_E|TYPE_T in thermal builds so
    // the exchanged macro temperature drives the thermal DDFs; closed
    // apertures and all closed fine slots become TYPE_S.
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
    upper.lbm->external_synchronize_flags();
    lower.lbm->external_synchronize_flags();
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
        nested_require(level.lbm != nullptr,
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
    for (NestedLbmLevel& level : levels_) level.lbm->external_initialize();
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
        before_step_hook_(level, level_time_si_[level_index], level.lbm->get_t());
    level.lbm->external_step();
    level_time_si_[level_index] += level.dt_si;
    if (after_step_hook_)
        after_step_hook_(level, level_time_si_[level_index], level.lbm->get_t());
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
    for (NestedLbmLevel& level : levels_) level.lbm->external_finish();
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
