#include "vtk_output.hpp"
#include "../core/build_features.hpp"
#include "../core/lattice_units.hpp"
#include "../solver/field_view.hpp"

namespace culbm::run {

namespace {

string vtk_triple(const uint a, const uint b, const uint c) {
    return to_string(a) + " " + to_string(b) + " " + to_string(c);
}

string vtk_triple(const float a, const float b, const float c) {
    return to_string(a) + " " + to_string(b) + " " + to_string(c);
}

// Header of a legacy binary structured-points file with nx x ny x nz points.
string structured_points_header(const string& title, const uint nx, const uint ny, const uint nz,
                                const float3& origin, const float spacing) {
    const ulong points = (ulong)nx * (ulong)ny * (ulong)nz;
    return "# vtk DataFile Version 3.0\n" + title + "\nBINARY\nDATASET STRUCTURED_POINTS\n"
        "DIMENSIONS " + vtk_triple(nx, ny, nz) + "\n"
        "ORIGIN " + vtk_triple(origin.x, origin.y, origin.z) + "\n"
        "SPACING " + vtk_triple(spacing, spacing, spacing) + "\n"
        "POINT_DATA " + to_string(points) + "\n";
}

// Name of the file without its directory.
string file_base_name(const string& filename) {
    const size_t separator = filename.find_last_of("/\\");
    return separator == string::npos ? filename : filename.substr(separator + 1u);
}

// Creates the folder of `filename` and opens it for binary output; reports
// a failure on the console.
bool open_binary_output(std::ofstream& file, const string& filename) {
    create_folder(filename);
    file.open(filename, std::ios::binary | std::ios::out);
    if (file.is_open()) return true;
    println("ERROR: could not open " + filename + " for writing.");
    return false;
}

} // namespace

void write_avg_vtk(const string& filename, const uint Nx, const uint Ny, const uint Nz,
                   const float* u_avg, const float* rho_avg, const float* T_avg, const uchar* flags,
                   const float* M2_u, const float* M2_v, const float* M2_w,
                   const ulong avg_count, const bool convert_to_si_units, const AvgTurbulenceOutput& turbulence,
                   const bool print_saved_message, const uint Nz_write,
                   const float spacing_override, const float3* origin_override) {
    if (!culbm::mpi_is_root() || !u_avg || !rho_avg) return;
    // Lattice units, or the SI value of one lattice unit of each quantity.
    float spacing = convert_to_si_units ? units.si_x(1.0f) : 1.0f;
    const float u_factor = convert_to_si_units ? units.si_u(1.0f) : 1.0f;
    const float rho_factor = convert_to_si_units ? units.si_rho(1.0f) : 1.0f;
    const float T_factor = convert_to_si_units ? units.si_dT(1.0f) : 1.0f;
    const float T_offset = convert_to_si_units ? units.si_T(0.0f) : 0.0f;
    if (spacing_override > 0.0f) spacing = spacing_override;
    float3 origin = spacing * float3(0.5f - 0.5f * (float)Nx, 0.5f - 0.5f * (float)Ny, 0.5f - 0.5f * (float)Nz);
    if (origin_override) origin = *origin_override;
    else if (convert_to_si_units) origin += vtk_origin_shift;
    const uint Nz_out = (Nz_write > 0u && Nz_write < Nz) ? Nz_write : Nz;
    const ulong points = (ulong)Nx * (ulong)Ny * (ulong)Nz_out;

    {
    std::ofstream file;
    if (!open_binary_output(file, filename)) return;
    const string header = structured_points_header("CULBM " + file_base_name(filename), Nx, Ny, Nz_out, origin, spacing);
    file.write(header.c_str(), header.length());

    auto write_field = [&](const string& name, const float* data, const uint components, const float factor, const float offset = 0.0f) {
        const string field_header =
            "SCALARS " + name + " float " + to_string(components) + "\nLOOKUP_TABLE default\n";
        file.write(field_header.c_str(), field_header.length());

        const uint threads = std::max(1u, (uint)std::thread::hardware_concurrency());
        const uint chunk_size_MB = 4u * threads;
        const ulong chunk_elements = std::max((ulong)1u, ((ulong)1048576u * (ulong)chunk_size_MB) / ((ulong)components * (ulong)sizeof(float)));
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
    const bool write_tke = turbulence.tke;
    const bool write_ti = turbulence.ti;
    const bool write_tls = turbulence.tls;
    std::vector<float> fluid(points, 1.0f);
    std::vector<float> tke;
    std::vector<float> ti;
    std::vector<float> tls;
    if (write_tke) tke.assign((size_t)points, 0.0f);
    if (write_ti) ti.assign((size_t)points, 0.0f);
    if (write_tls) tls.assign((size_t)points, 0.0f);
    const bool has_m2 = (avg_count > 1ull && M2_u && M2_v && M2_w);
    const float inv_n = has_m2 ? (1.0f / (float)avg_count) : 0.0f;
    const float grid_dx = fmaxf(spacing, 1.0e-12f);
    const ulong Nxul = (ulong)Nx;
    const ulong Nyul = (ulong)Ny;
    const ulong Nzul = (ulong)Nz_out;
    const ulong plane = Nxul * Nyul;
    const float tls_cap = (float)std::max(std::max(Nx, Ny), Nz_out) * grid_dx;
    const auto sample_u = [&](const ulong idx, const uint comp) -> float {
        return u_avg[3ull * idx + (ulong)comp] * u_factor;
    };
    parallel_for(points, [&](ulong n) {
        const bool solid = flags && ((flags[n] & CELL_SOLID) != 0u);
        fluid[n] = solid ? 0.0f : 1.0f;
        if (!has_m2 || solid) return;
        if (!(write_tke || write_ti || write_tls)) return;
        const float var_u = fmaxf(M2_u[n] * inv_n, 0.0f);
        const float var_v = fmaxf(M2_v[n] * inv_n, 0.0f);
        const float var_w = fmaxf(M2_w[n] * inv_n, 0.0f);
        const float var_sum = var_u + var_v + var_w;
        if (write_tke) {
            tke[n] = 0.5f * var_sum;
        }
        if (write_ti) {
            const ulong i3 = 3ull * n;
            const float umag = sqrtf(sq(u_avg[i3]) + sq(u_avg[i3 + 1ull]) + sq(u_avg[i3 + 2ull]));
            if (umag > 1.0e-9f && var_sum > 0.0f) {
                const float urms = sqrtf(var_sum * (1.0f / 3.0f));
                ti[n] = urms / umag;
            }
        }
        if (!write_tls) return;

        // Local turbulence length scale estimate from local k and mean strain:
        // TLS = sqrt(k) / |S|, where |S| = sqrt(2*Sij*Sij).
        const ulong z = n / plane;
        const ulong rem = n - z * plane;
        const ulong y = rem / Nxul;
        const ulong x = rem - y * Nxul;
        const ulong xm = (x > 0ull) ? (x - 1ull) : x;
        const ulong xp = (x + 1ull < Nxul) ? (x + 1ull) : x;
        const ulong ym = (y > 0ull) ? (y - 1ull) : y;
        const ulong yp = (y + 1ull < Nyul) ? (y + 1ull) : y;
        const ulong zm = (z > 0ull) ? (z - 1ull) : z;
        const ulong zp = (z + 1ull < Nzul) ? (z + 1ull) : z;
        const ulong idx_xm = xm + (y + z * Nyul) * Nxul;
        const ulong idx_xp = xp + (y + z * Nyul) * Nxul;
        const ulong idx_ym = x + (ym + z * Nyul) * Nxul;
        const ulong idx_yp = x + (yp + z * Nyul) * Nxul;
        const ulong idx_zm = x + (y + zm * Nyul) * Nxul;
        const ulong idx_zp = x + (y + zp * Nyul) * Nxul;
        const float inv_dx = (xp > xm) ? (1.0f / ((float)(xp - xm) * grid_dx)) : 0.0f;
        const float inv_dy = (yp > ym) ? (1.0f / ((float)(yp - ym) * grid_dx)) : 0.0f;
        const float inv_dz = (zp > zm) ? (1.0f / ((float)(zp - zm) * grid_dx)) : 0.0f;

        const float duxdx = (sample_u(idx_xp, 0u) - sample_u(idx_xm, 0u)) * inv_dx;
        const float duydx = (sample_u(idx_xp, 1u) - sample_u(idx_xm, 1u)) * inv_dx;
        const float duzdx = (sample_u(idx_xp, 2u) - sample_u(idx_xm, 2u)) * inv_dx;
        const float duxdy = (sample_u(idx_yp, 0u) - sample_u(idx_ym, 0u)) * inv_dy;
        const float duydy = (sample_u(idx_yp, 1u) - sample_u(idx_ym, 1u)) * inv_dy;
        const float duzdy = (sample_u(idx_yp, 2u) - sample_u(idx_ym, 2u)) * inv_dy;
        const float duxdz = (sample_u(idx_zp, 0u) - sample_u(idx_zm, 0u)) * inv_dz;
        const float duydz = (sample_u(idx_zp, 1u) - sample_u(idx_zm, 1u)) * inv_dz;
        const float duzdz = (sample_u(idx_zp, 2u) - sample_u(idx_zm, 2u)) * inv_dz;

        const float Sxx = duxdx;
        const float Syy = duydy;
        const float Szz = duzdz;
        const float Sxy = 0.5f * (duxdy + duydx);
        const float Sxz = 0.5f * (duxdz + duzdx);
        const float Syz = 0.5f * (duydz + duzdy);
        const float S_mag = sqrtf(fmaxf(0.0f, 2.0f * (sq(Sxx) + sq(Syy) + sq(Szz) + 2.0f * (sq(Sxy) + sq(Sxz) + sq(Syz)))));

        const float k_local = 0.5f * var_sum * (u_factor * u_factor);
        const float tls_local = (S_mag > 1.0e-10f && k_local > 0.0f) ? (sqrtf(k_local) / S_mag) : 0.0f;
        tls[n] = fminf(fmaxf(tls_local, 0.0f), tls_cap);
    });
    write_field("fluid", fluid.data(), 1u, 1.0f);
    if (write_tke) write_field("tke", tke.data(), 1u, u_factor * u_factor);
    if (write_ti) write_field("TI", ti.data(), 1u, 1.0f);
    if (write_tls) write_field("TLS", tls.data(), 1u, 1.0f);
    }
    if (print_saved_message) {
        const StatusPause pause;
        log_info("File \"" + filename + "\" saved.");
    }
}

void write_nested_raw_vtk(const string& filename, const uint Nx, const uint Ny, const uint Nz,
                          const float* data, const uint components,
                          const float spacing_si, const float3& origin_si,
                          const float factor, const float offset) {
    if (!culbm::mpi_is_root() || !data || Nx == 0u || Ny == 0u || Nz == 0u ||
        components == 0u || !(spacing_si > 0.0f)) return;
    const ulong points = (ulong)Nx * (ulong)Ny * (ulong)Nz;
    std::ofstream file;
    if (!open_binary_output(file, filename)) return;
    const string header = structured_points_header("CULBM nested " + file_base_name(filename), Nx, Ny, Nz, origin_si, spacing_si) +
        "SCALARS data float " + to_string(components) + "\nLOOKUP_TABLE default\n";
    file.write(header.c_str(), header.length());

    const uint threads = std::max(1u, (uint)std::thread::hardware_concurrency());
    const ulong chunk_elements = std::max(
        (ulong)1u,
        ((ulong)1048576u * (ulong)(4u * threads)) /
        ((ulong)components * (ulong)sizeof(float)));
    std::vector<float> buffer(chunk_elements * (ulong)components);
    for (ulong offset_cells = 0u; offset_cells < points; offset_cells += chunk_elements) {
        const ulong count = std::min(chunk_elements, points - offset_cells);
        parallel_for(count, [&](ulong i) {
            const ulong source = (offset_cells + i) * (ulong)components;
            const ulong destination = i * (ulong)components;
            for (uint component = 0u; component < components; ++component) {
                buffer[destination + component] = reverse_bytes(
                    data[source + component] * factor + offset);
            }
        });
        file.write(reinterpret_cast<const char*>(buffer.data()),
                   count * (ulong)components * sizeof(float));
    }
}

} // namespace culbm::run
