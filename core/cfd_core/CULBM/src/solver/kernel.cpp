#include "kernel.hpp"
#include "particle_advection_kernels.hpp"

string get_cuda_kernel_source() {
    string source;
    source.reserve(164001u);
    source += '\n';
#include "kernel_cuda_source.inc"
    source += R"CUDA(

#ifdef TEMPERATURE
// Macro-field mirror used for diagnostics on the GEA atmospheric validation
// walls.  Population-level impermeable/free-slip behavior is provided by the
// FREE_SLIP_BOUNDARIES path in stream_collide.  The production case/deck target
// does not define that extension or call this kernel.
extern "C" __global__ void culbm_update_free_slip_boundaries(
    float* rho,
    float* u,
    float* T,
    const uchar* flags,
    const uint face_mask)
{
    const uxx n = cuda_global_thread_id(0);
    if (n >= (uxx)def_N || is_halo(n)) return;
    if ((flags[n] & TYPE_BO) != TYPE_S) return;

    const uint3 xyz = coordinates(n);
    const int x_global = (int)xyz.x + def_Ox;
    const int y_global = (int)xyz.y + def_Oy;
    const int z_global = (int)xyz.z + def_Oz;

    const bool west =
        (face_mask & 1u) != 0u && x_global == 0;
    const bool east =
        (face_mask & 2u) != 0u &&
        x_global == (int)def_Nx_global - 1;
    const bool south =
        (face_mask & 4u) != 0u && y_global == 0;
    const bool north =
        (face_mask & 8u) != 0u &&
        y_global == (int)def_Ny_global - 1;
    const bool bottom =
        (face_mask & 16u) != 0u && z_global == 0;
    const bool top =
        (face_mask & 32u) != 0u &&
        z_global == (int)def_Nz_global - 1;
    if (!(west || east || south || north || bottom || top)) return;

    uint3 interior = xyz;
    if (west) ++interior.x;
    if (east) --interior.x;
    if (south) ++interior.y;
    if (north) --interior.y;
    if (bottom) ++interior.z;
    if (top) --interior.z;
    if (interior.x >= def_Nx ||
        interior.y >= def_Ny ||
        interior.z >= def_Nz) return;

    const uxx source_index = index(interior);
    rho[n] = rho[source_index];
    u[n] = (west || east) ? 0.0f : u[source_index];
    u[(ulong)def_N + (ulong)n] =
        (south || north)
            ? 0.0f
            : u[(ulong)def_N + (ulong)source_index];
    u[2ull * (ulong)def_N + (ulong)n] =
        (bottom || top)
            ? 0.0f
            : u[2ull * (ulong)def_N + (ulong)source_index];
    // TYPE_S without TYPE_T uses the production scalar bounce-back path.
    // Keeping the macro value mirrored makes diagnostics at the ghost wall
    // consistent without imposing a Dirichlet thermal boundary.
    T[n] = T[source_index];
}
#endif

extern "C" __global__ void culbm_accumulate_statistics(
    const float* rho,
    const float* u,
#ifdef TEMPERATURE
    const float* T,
#endif
    float* mean_u,
    float* mean_rho,
    float* M2_u,
#ifdef TEMPERATURE
    float* mean_T,
#endif
    const float inverse_sample_count)
{
    const ulong n = cuda_global_thread_id(0);
    if (n >= (ulong)def_N) return;

    const float ux = u[n];
    const float uy = u[(ulong)def_N + n];
    const float uz = u[2ull * (ulong)def_N + n];

    float mean = mean_u[n];
    float delta = ux - mean;
    mean += delta * inverse_sample_count;
    M2_u[n] += delta * (ux - mean);
    mean_u[n] = mean;

    mean = mean_u[(ulong)def_N + n];
    delta = uy - mean;
    mean += delta * inverse_sample_count;
    M2_u[(ulong)def_N + n] += delta * (uy - mean);
    mean_u[(ulong)def_N + n] = mean;

    mean = mean_u[2ull * (ulong)def_N + n];
    delta = uz - mean;
    mean += delta * inverse_sample_count;
    M2_u[2ull * (ulong)def_N + n] += delta * (uz - mean);
    mean_u[2ull * (ulong)def_N + n] = mean;

    mean = mean_rho[n];
    delta = rho[n] - mean;
    mean_rho[n] = mean + delta * inverse_sample_count;

#ifdef TEMPERATURE
    if (mean_T != nullptr) {
        mean = mean_T[n];
        delta = T[n] - mean;
        mean_T[n] = mean + delta * inverse_sample_count;
    }
#endif
}

extern "C" __global__ void culbm_gather_probe_velocity(
    const float* u,
    const ulong* indices,
    float* values,
    const ulong count)
{
    const ulong i = cuda_global_thread_id(0);
    if (i >= count) return;
    const ulong n = indices[i];
    if (n >= (ulong)def_N) {
        values[i] = 0.0f;
        values[count + i] = 0.0f;
        values[2ull * count + i] = 0.0f;
        return;
    }
    values[i] = u[n];
    values[count + i] = u[(ulong)def_N + n];
    values[2ull * count + i] = u[2ull * (ulong)def_N + n];
}

)CUDA";
    source += culbm::particle_advection_cuda_source();
    return source;
}
