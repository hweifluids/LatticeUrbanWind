#include "particle_advection_kernels.hpp"

namespace culbm {

std::string particle_advection_cuda_source() {
    return R"CUDA(

#define CULBM_ADVECTION_UNUSED  0u
#define CULBM_ADVECTION_PENDING 1u
#define CULBM_ADVECTION_ACTIVE  2u
#define CULBM_ADVECTION_EXPIRED 3u
#define CULBM_ADVECTION_ESCAPED 4u
#define CULBM_ADVECTION_SOLID   5u
#define CULBM_ADVECTION_INVALID 6u

__device__ __forceinline__ bool culbm_advection_finite3(const float3 p) {
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}

__device__ __forceinline__ bool culbm_advection_inside_global(
    const float3 p,
    const uint nx,
    const uint ny,
    const uint nz)
{
    return culbm_advection_finite3(p) &&
        p.x >= -0.5f && p.x < (float)nx - 0.5f &&
        p.y >= -0.5f && p.y < (float)ny - 0.5f &&
        p.z >= -0.5f && p.z < (float)nz - 0.5f;
}

__device__ __forceinline__ uint culbm_advection_owner_axis(
    const float coordinate,
    const uint extent,
    const uint domains)
{
    int cell = (int)floorf(coordinate + 0.5f);
    cell = cell < 0 ? 0 : (cell >= (int)extent ? (int)extent - 1 : cell);
    const uint inner = extent / domains;
    const uint owner = (uint)cell / inner;
    return owner < domains ? owner : domains - 1u;
}

__device__ __forceinline__ uint culbm_advection_owner(
    const float3 p,
    const uint global_nx,
    const uint global_ny,
    const uint global_nz,
    const uint domains_x,
    const uint domains_y,
    const uint domains_z)
{
    if (!culbm_advection_inside_global(p, global_nx, global_ny, global_nz)) {
        return 0xFFFFFFFFu;
    }
    const uint dx = culbm_advection_owner_axis(p.x, global_nx, domains_x);
    const uint dy = culbm_advection_owner_axis(p.y, global_ny, domains_y);
    const uint dz = culbm_advection_owner_axis(p.z, global_nz, domains_z);
    return dx + (dy + dz * domains_y) * domains_x;
}

__device__ __forceinline__ ulong culbm_advection_local_index(
    const uint x,
    const uint y,
    const uint z,
    const uint nx,
    const uint ny)
{
    return (ulong)x + ((ulong)y + (ulong)z * (ulong)ny) * (ulong)nx;
}

__device__ __forceinline__ bool culbm_advection_point_is_fluid(
    const float3 global_position,
    const uchar* flags,
    const uint local_nx,
    const uint local_ny,
    const uint local_nz,
    const int offset_x,
    const int offset_y,
    const int offset_z)
{
    const float lx = global_position.x - (float)offset_x;
    const float ly = global_position.y - (float)offset_y;
    const float lz = global_position.z - (float)offset_z;
    if (lx < -0.5f || lx >= (float)local_nx - 0.5f ||
        ly < -0.5f || ly >= (float)local_ny - 0.5f ||
        lz < -0.5f || lz >= (float)local_nz - 0.5f) {
        return false;
    }
    int x = (int)floorf(lx + 0.5f);
    int y = (int)floorf(ly + 0.5f);
    int z = (int)floorf(lz + 0.5f);
    x = x < 0 ? 0 : (x >= (int)local_nx ? (int)local_nx - 1 : x);
    y = y < 0 ? 0 : (y >= (int)local_ny ? (int)local_ny - 1 : y);
    z = z < 0 ? 0 : (z >= (int)local_nz ? (int)local_nz - 1 : z);
    const ulong index = culbm_advection_local_index(
        (uint)x, (uint)y, (uint)z, local_nx, local_ny);
    return (flags[index] & (uchar)1u) == 0u;
}

__device__ __forceinline__ bool culbm_advection_velocity(
    const float3 global_position,
    const float alpha,
    const float* previous_u,
    const float* current_u,
    const uchar* flags,
    const uint global_nx,
    const uint global_ny,
    const uint global_nz,
    const uint local_nx,
    const uint local_ny,
    const uint local_nz,
    const int offset_x,
    const int offset_y,
    const int offset_z,
    float3* velocity)
{
    // The ownership volume extends half a cell beyond the first/last lattice
    // sample. Clamp only at those physical boundaries so the half-cell shell
    // has a well-defined nearest-boundary velocity. Internal CUDA-domain
    // interfaces remain un-clamped and use the communicated one-cell halo.
    const float gx = fminf(
        fmaxf(global_position.x, 0.0f), (float)global_nx - 1.0f);
    const float gy = fminf(
        fmaxf(global_position.y, 0.0f), (float)global_ny - 1.0f);
    const float gz = fminf(
        fmaxf(global_position.z, 0.0f), (float)global_nz - 1.0f);
    const float lx = gx - (float)offset_x;
    const float ly = gy - (float)offset_y;
    const float lz = gz - (float)offset_z;
    if (lx < 0.0f || lx > (float)local_nx - 1.0f ||
        ly < 0.0f || ly > (float)local_ny - 1.0f ||
        lz < 0.0f || lz > (float)local_nz - 1.0f) {
        *velocity = cuda_float3(0.0f);
        return false;
    }

    int x0 = (int)floorf(lx);
    int y0 = (int)floorf(ly);
    int z0 = (int)floorf(lz);
    float tx = lx - (float)x0;
    float ty = ly - (float)y0;
    float tz = lz - (float)z0;
    if (x0 >= (int)local_nx - 1) { x0 = (int)local_nx - 2; tx = 1.0f; }
    if (y0 >= (int)local_ny - 1) { y0 = (int)local_ny - 2; ty = 1.0f; }
    if (z0 >= (int)local_nz - 1) { z0 = (int)local_nz - 2; tz = 1.0f; }
    if (x0 < 0 || y0 < 0 || z0 < 0 ||
        local_nx < 2u || local_ny < 2u || local_nz < 2u) {
        *velocity = cuda_float3(0.0f);
        return false;
    }

    const ulong local_count =
        (ulong)local_nx * (ulong)local_ny * (ulong)local_nz;
    float3 sum = cuda_float3(0.0f);
    float fluid_weight = 0.0f;
    for (uint corner = 0u; corner < 8u; ++corner) {
        const uint ox = corner & 1u;
        const uint oy = (corner >> 1u) & 1u;
        const uint oz = (corner >> 2u) & 1u;
        const float wx = ox != 0u ? tx : 1.0f - tx;
        const float wy = oy != 0u ? ty : 1.0f - ty;
        const float wz = oz != 0u ? tz : 1.0f - tz;
        const float weight = wx * wy * wz;
        if (weight <= 0.0f) continue;
        const ulong index = culbm_advection_local_index(
            (uint)(x0 + (int)ox),
            (uint)(y0 + (int)oy),
            (uint)(z0 + (int)oz),
            local_nx,
            local_ny);
        if ((flags[index] & (uchar)1u) != 0u) continue;
        const float ux = fmaf(
            alpha,
            current_u[index] - previous_u[index],
            previous_u[index]);
        const float uy = fmaf(
            alpha,
            current_u[local_count + index] -
                previous_u[local_count + index],
            previous_u[local_count + index]);
        const float uz = fmaf(
            alpha,
            current_u[2ull * local_count + index] -
                previous_u[2ull * local_count + index],
            previous_u[2ull * local_count + index]);
        sum += weight * cuda_float3(ux, uy, uz);
        fluid_weight += weight;
    }
    if (!(fluid_weight > 1.0e-7f)) {
        *velocity = cuda_float3(0.0f);
        return false;
    }
    *velocity = sum / fluid_weight;
    return culbm_advection_finite3(*velocity);
}

__device__ __forceinline__ bool culbm_advection_interpolation_covered(
    const float3 global_position,
    const uint global_nx,
    const uint global_ny,
    const uint global_nz,
    const uint local_nx,
    const uint local_ny,
    const uint local_nz,
    const int offset_x,
    const int offset_y,
    const int offset_z)
{
    if (local_nx < 2u || local_ny < 2u || local_nz < 2u) return false;
    const float gx = fminf(
        fmaxf(global_position.x, 0.0f), (float)global_nx - 1.0f);
    const float gy = fminf(
        fmaxf(global_position.y, 0.0f), (float)global_ny - 1.0f);
    const float gz = fminf(
        fmaxf(global_position.z, 0.0f), (float)global_nz - 1.0f);
    const float lx = gx - (float)offset_x;
    const float ly = gy - (float)offset_y;
    const float lz = gz - (float)offset_z;
    return lx >= 0.0f && lx <= (float)local_nx - 1.0f &&
        ly >= 0.0f && ly <= (float)local_ny - 1.0f &&
        lz >= 0.0f && lz <= (float)local_nz - 1.0f;
}

__device__ __forceinline__ bool culbm_advection_stage_velocity(
    const float3 position,
    const float alpha,
    const float* previous_u,
    const float* current_u,
    const uchar* flags,
    const uint global_nx,
    const uint global_ny,
    const uint global_nz,
    const uint local_nx,
    const uint local_ny,
    const uint local_nz,
    const int offset_x,
    const int offset_y,
    const int offset_z,
    float3* velocity,
    uint* failure)
{
    if (!culbm_advection_inside_global(
            position, global_nx, global_ny, global_nz)) {
        *failure = CULBM_ADVECTION_ESCAPED;
        *velocity = cuda_float3(0.0f);
        return false;
    }
    if (!culbm_advection_interpolation_covered(
            position, global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z)) {
        *failure = CULBM_ADVECTION_INVALID;
        *velocity = cuda_float3(0.0f);
        return false;
    }
    if (!culbm_advection_velocity(
            position, alpha, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z, velocity)) {
        *failure = culbm_advection_point_is_fluid(
            position, flags, local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z)
            ? CULBM_ADVECTION_INVALID
            : CULBM_ADVECTION_SOLID;
        return false;
    }
    return true;
}

template<int Method>
__device__ __forceinline__ bool culbm_advection_integrate(
    const float3 p,
    const float* previous_u,
    const float* current_u,
    const uchar* flags,
    const uint global_nx,
    const uint global_ny,
    const uint global_nz,
    const uint local_nx,
    const uint local_ny,
    const uint local_nz,
    const int offset_x,
    const int offset_y,
    const int offset_z,
    float3* result,
    uint* failure)
{
    *result = p;
    float3 k1 = cuda_float3(0.0f);
    if (!culbm_advection_stage_velocity(
            p, 0.0f, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z, &k1, failure)) {
        return false;
    }

    if constexpr (Method == 1) {
        *result = p + k1;
        return true;
    }

    if constexpr (Method == 2) {
        float3 k2 = cuda_float3(0.0f);
        const float3 stage2 = p + k1;
        *result = stage2;
        if (!culbm_advection_stage_velocity(
                stage2, 1.0f, previous_u, current_u, flags,
                global_nx, global_ny, global_nz,
                local_nx, local_ny, local_nz,
                offset_x, offset_y, offset_z, &k2, failure)) {
            return false;
        }
        *result = p + 0.5f * (k1 + k2);
        return true;
    }

    if constexpr (Method == 4) {
        float3 k2 = cuda_float3(0.0f);
        float3 k3 = cuda_float3(0.0f);
        float3 k4 = cuda_float3(0.0f);
        const float3 stage2 = p + 0.5f * k1;
        *result = stage2;
        if (!culbm_advection_stage_velocity(
                stage2, 0.5f, previous_u, current_u, flags,
                global_nx, global_ny, global_nz,
                local_nx, local_ny, local_nz,
                offset_x, offset_y, offset_z, &k2, failure)) {
            return false;
        }
        const float3 stage3 = p + 0.5f * k2;
        *result = stage3;
        if (!culbm_advection_stage_velocity(
                stage3, 0.5f, previous_u, current_u, flags,
                global_nx, global_ny, global_nz,
                local_nx, local_ny, local_nz,
                offset_x, offset_y, offset_z, &k3, failure)) {
            return false;
        }
        const float3 stage4 = p + k3;
        *result = stage4;
        if (!culbm_advection_stage_velocity(
                stage4, 1.0f, previous_u, current_u, flags,
                global_nx, global_ny, global_nz,
                local_nx, local_ny, local_nz,
                offset_x, offset_y, offset_z, &k4, failure)) {
            return false;
        }
        *result = p + (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
        return true;
    }

    // Canonical upper-sign seven-stage, sixth-order explicit RK tableau.
    constexpr float sqrt5 = 2.2360679774997896964f;
    constexpr float c2 = (5.0f - sqrt5) / 10.0f;
    constexpr float c3 = (5.0f + sqrt5) / 10.0f;
    constexpr float a21 = c2;
    constexpr float a31 = -sqrt5 / 10.0f;
    constexpr float a32 = (5.0f + 2.0f * sqrt5) / 10.0f;
    constexpr float a41 = (-15.0f + 7.0f * sqrt5) / 20.0f;
    constexpr float a42 = (-1.0f + sqrt5) / 4.0f;
    constexpr float a43 = (15.0f - 7.0f * sqrt5) / 10.0f;
    constexpr float a51 = (5.0f - sqrt5) / 60.0f;
    constexpr float a53 = 1.0f / 6.0f;
    constexpr float a54 = (15.0f + 7.0f * sqrt5) / 60.0f;
    constexpr float a61 = (5.0f + sqrt5) / 60.0f;
    constexpr float a63 = (9.0f - 5.0f * sqrt5) / 12.0f;
    constexpr float a64 = 1.0f / 6.0f;
    constexpr float a65 = (-5.0f + 3.0f * sqrt5) / 10.0f;
    constexpr float a71 = 1.0f / 6.0f;
    constexpr float a73 = (-55.0f + 25.0f * sqrt5) / 12.0f;
    constexpr float a74 = (-25.0f - 7.0f * sqrt5) / 12.0f;
    constexpr float a75 = 5.0f - 2.0f * sqrt5;
    constexpr float a76 = (5.0f + sqrt5) / 2.0f;

    float3 k2 = cuda_float3(0.0f);
    float3 k3 = cuda_float3(0.0f);
    float3 k4 = cuda_float3(0.0f);
    float3 k5 = cuda_float3(0.0f);
    float3 k6 = cuda_float3(0.0f);
    float3 k7 = cuda_float3(0.0f);
    const float3 stage2 = p + a21 * k1;
    *result = stage2;
    if (!culbm_advection_stage_velocity(
            stage2, c2, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z, &k2, failure)) {
        return false;
    }
    const float3 stage3 = p + a31 * k1 + a32 * k2;
    *result = stage3;
    if (!culbm_advection_stage_velocity(
            stage3, c3, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z, &k3, failure)) {
        return false;
    }
    const float3 stage4 = p + a41 * k1 + a42 * k2 + a43 * k3;
    *result = stage4;
    if (!culbm_advection_stage_velocity(
            stage4, c2, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z, &k4, failure)) {
        return false;
    }
    const float3 stage5 = p + a51 * k1 + a53 * k3 + a54 * k4;
    *result = stage5;
    if (!culbm_advection_stage_velocity(
            stage5, c3, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z, &k5, failure)) {
        return false;
    }
    const float3 stage6 =
        p + a61 * k1 + a63 * k3 + a64 * k4 + a65 * k5;
    *result = stage6;
    if (!culbm_advection_stage_velocity(
            stage6, c2, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z, &k6, failure)) {
        return false;
    }
    const float3 stage7 =
        p + a71 * k1 + a73 * k3 + a74 * k4 +
            a75 * k5 + a76 * k6;
    *result = stage7;
    if (!culbm_advection_stage_velocity(
            stage7, 1.0f, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z, &k7, failure)) {
        return false;
    }
    *result = p +
        (1.0f / 12.0f) * k1 +
        (5.0f / 12.0f) * k5 +
        (5.0f / 12.0f) * k6 +
        (1.0f / 12.0f) * k7;
    return true;
}

extern "C" __global__ void culbm_advection_copy_velocity(
    float* destination,
    const float* source,
    const ulong value_count)
{
    const ulong index = cuda_global_thread_id(0);
    if (index < value_count) destination[index] = source[index];
}

extern "C" __global__ void culbm_advection_initialize_regular(
    float* positions,
    ulong* birth_steps,
    ulong* death_steps,
    ulong* termination_steps,
    uchar* statuses,
    const ulong* release_steps,
    const ulong release_count,
    const uint count_x,
    const uint count_y,
    const uint count_z,
    const float min_x,
    const float min_y,
    const float min_z,
    const float spacing_x,
    const float spacing_y,
    const float spacing_z,
    const ulong lifetime_steps,
    const ulong particle_count,
    const uint global_nx,
    const uint global_ny,
    const uint global_nz,
    const uint domains_x,
    const uint domains_y,
    const uint domains_z,
    const uint domain_id)
{
    const ulong id = cuda_global_thread_id(0);
    if (id >= particle_count) return;
    const ulong points_per_release =
        (ulong)count_x * (ulong)count_y * (ulong)count_z;
    const ulong batch = id / points_per_release;
    if (batch >= release_count) return;
    const ulong point = id - batch * points_per_release;
    const uint ix = (uint)(point % (ulong)count_x);
    const uint iy = (uint)((point / (ulong)count_x) % (ulong)count_y);
    const uint iz = (uint)(point / ((ulong)count_x * (ulong)count_y));
    const float3 p = cuda_float3(
        min_x + (float)ix * spacing_x,
        min_y + (float)iy * spacing_y,
        min_z + (float)iz * spacing_z);
    const ulong birth = release_steps[batch];
    birth_steps[id] = birth;
    death_steps[id] =
        lifetime_steps == 0xFFFFFFFFFFFFFFFFull
        ? 0xFFFFFFFFFFFFFFFFull
        : (birth > 0xFFFFFFFFFFFFFFFFull - lifetime_steps
            ? 0ull
            : birth + lifetime_steps);
    termination_steps[id] = 0xFFFFFFFFFFFFFFFFull;
    if (culbm_advection_owner(
            p, global_nx, global_ny, global_nz,
            domains_x, domains_y, domains_z) != domain_id) {
        statuses[id] = (uchar)CULBM_ADVECTION_UNUSED;
        return;
    }
    positions[id] = p.x;
    positions[particle_count + id] = p.y;
    positions[2ull * particle_count + id] = p.z;
    statuses[id] = (uchar)CULBM_ADVECTION_PENDING;
}

extern "C" __global__ void culbm_advection_activate(
    const float* positions,
    const ulong* birth_steps,
    const ulong* death_steps,
    ulong* termination_steps,
    uchar* statuses,
    const uchar* flags,
    const ulong current_step,
    const ulong particle_count,
    const uint global_nx,
    const uint global_ny,
    const uint global_nz,
    const uint domains_x,
    const uint domains_y,
    const uint domains_z,
    const uint domain_id,
    const uint local_nx,
    const uint local_ny,
    const uint local_nz,
    const int offset_x,
    const int offset_y,
    const int offset_z)
{
    const ulong id = cuda_global_thread_id(0);
    if (id >= particle_count ||
        statuses[id] != (uchar)CULBM_ADVECTION_PENDING ||
        birth_steps[id] > current_step) {
        return;
    }
    const float3 p = cuda_float3(
        positions[id],
        positions[particle_count + id],
        positions[2ull * particle_count + id]);
    const uint owner = culbm_advection_owner(
        p, global_nx, global_ny, global_nz,
        domains_x, domains_y, domains_z);
    if (owner == 0xFFFFFFFFu) {
        statuses[id] = (uchar)CULBM_ADVECTION_ESCAPED;
        termination_steps[id] = current_step;
        return;
    }
    if (owner != domain_id) {
        statuses[id] = (uchar)CULBM_ADVECTION_INVALID;
        termination_steps[id] = current_step;
        return;
    }
    if (!culbm_advection_point_is_fluid(
            p, flags, local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z)) {
        statuses[id] = (uchar)CULBM_ADVECTION_SOLID;
        termination_steps[id] = current_step;
        return;
    }
    if (death_steps[id] < birth_steps[id]) {
        statuses[id] = (uchar)CULBM_ADVECTION_INVALID;
        termination_steps[id] = current_step;
        return;
    }
    statuses[id] = (uchar)CULBM_ADVECTION_ACTIVE;
}

template<int Method>
__device__ __forceinline__ void culbm_advection_step_impl(
    float* positions,
    const ulong* birth_steps,
    const ulong* death_steps,
    ulong* termination_steps,
    uchar* statuses,
    const float* previous_u,
    const float* current_u,
    const uchar* flags,
    const ulong step_begin,
    uint* migration_count,
    ulong* migration_ids,
    uint* migration_destinations,
    float* migration_positions,
    const ulong particle_count,
    const uint global_nx,
    const uint global_ny,
    const uint global_nz,
    const uint domains_x,
    const uint domains_y,
    const uint domains_z,
    const uint domain_id,
    const uint local_nx,
    const uint local_ny,
    const uint local_nz,
    const int offset_x,
    const int offset_y,
    const int offset_z)
{
    const ulong id = cuda_global_thread_id(0);
    if (id >= particle_count ||
        statuses[id] != (uchar)CULBM_ADVECTION_ACTIVE) {
        return;
    }
    const ulong death = death_steps[id];
    if (death != 0xFFFFFFFFFFFFFFFFull && step_begin >= death) {
        statuses[id] = (uchar)CULBM_ADVECTION_EXPIRED;
        termination_steps[id] = step_begin;
        return;
    }

    const float3 p = cuda_float3(
        positions[id],
        positions[particle_count + id],
        positions[2ull * particle_count + id]);
    float3 next = p;
    uint failure = CULBM_ADVECTION_INVALID;
    if (!culbm_advection_integrate<Method>(
            p, previous_u, current_u, flags,
            global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z,
            &next, &failure)) {
        positions[id] = next.x;
        positions[particle_count + id] = next.y;
        positions[2ull * particle_count + id] = next.z;
        statuses[id] = (uchar)failure;
        termination_steps[id] = step_begin + 1ull;
        return;
    }

    positions[id] = next.x;
    positions[particle_count + id] = next.y;
    positions[2ull * particle_count + id] = next.z;
    const ulong step_end = step_begin + 1ull;
    if (!culbm_advection_inside_global(
            next, global_nx, global_ny, global_nz)) {
        statuses[id] = (uchar)CULBM_ADVECTION_ESCAPED;
        termination_steps[id] = step_end;
        return;
    }
    if (!culbm_advection_interpolation_covered(
            next, global_nx, global_ny, global_nz,
            local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z)) {
        statuses[id] = (uchar)CULBM_ADVECTION_INVALID;
        termination_steps[id] = step_end;
        return;
    }
    if (!culbm_advection_point_is_fluid(
            next, flags, local_nx, local_ny, local_nz,
            offset_x, offset_y, offset_z)) {
        statuses[id] = (uchar)CULBM_ADVECTION_SOLID;
        termination_steps[id] = step_end;
        return;
    }
    if (death != 0xFFFFFFFFFFFFFFFFull && step_end >= death) {
        statuses[id] = (uchar)CULBM_ADVECTION_EXPIRED;
        termination_steps[id] = step_end;
        return;
    }

    const uint destination = culbm_advection_owner(
        next, global_nx, global_ny, global_nz,
        domains_x, domains_y, domains_z);
    if (destination != domain_id) {
        const uint queue_index = cuda_atomic_add(migration_count, 1u);
        if ((ulong)queue_index < particle_count) {
            migration_ids[queue_index] = id;
            migration_destinations[queue_index] = destination;
            migration_positions[queue_index] = next.x;
            migration_positions[particle_count + (ulong)queue_index] = next.y;
            migration_positions[2ull * particle_count + (ulong)queue_index] = next.z;
            statuses[id] = (uchar)CULBM_ADVECTION_UNUSED;
        } else {
            statuses[id] = (uchar)CULBM_ADVECTION_INVALID;
            termination_steps[id] = step_end;
        }
    }
}

#define CULBM_ADVECTION_STEP_KERNEL(name, method) \
extern "C" __global__ void name( \
    float* positions, \
    const ulong* birth_steps, \
    const ulong* death_steps, \
    ulong* termination_steps, \
    uchar* statuses, \
    const float* previous_u, \
    const float* current_u, \
    const uchar* flags, \
    const ulong step_begin, \
    uint* migration_count, \
    ulong* migration_ids, \
    uint* migration_destinations, \
    float* migration_positions, \
    const ulong particle_count, \
    const uint global_nx, \
    const uint global_ny, \
    const uint global_nz, \
    const uint domains_x, \
    const uint domains_y, \
    const uint domains_z, \
    const uint domain_id, \
    const uint local_nx, \
    const uint local_ny, \
    const uint local_nz, \
    const int offset_x, \
    const int offset_y, \
    const int offset_z) \
{ \
    culbm_advection_step_impl<method>( \
        positions, birth_steps, death_steps, termination_steps, statuses, \
        previous_u, current_u, flags, step_begin, migration_count, \
        migration_ids, migration_destinations, migration_positions, \
        particle_count, global_nx, global_ny, global_nz, \
        domains_x, domains_y, domains_z, domain_id, \
        local_nx, local_ny, local_nz, offset_x, offset_y, offset_z); \
}

CULBM_ADVECTION_STEP_KERNEL(culbm_advection_step_euler, 1)
CULBM_ADVECTION_STEP_KERNEL(culbm_advection_step_rk2, 2)
CULBM_ADVECTION_STEP_KERNEL(culbm_advection_step_rk4, 4)
CULBM_ADVECTION_STEP_KERNEL(culbm_advection_step_rk6, 6)

#undef CULBM_ADVECTION_STEP_KERNEL

extern "C" __global__ void culbm_advection_insert_migrants(
    const ulong* incoming_ids,
    const float* incoming_positions,
    const ulong incoming_count,
    const ulong particle_count,
    float* positions,
    ulong* termination_steps,
    uchar* statuses)
{
    const ulong index = cuda_global_thread_id(0);
    if (index >= incoming_count) return;
    const ulong id = incoming_ids[index];
    if (id >= particle_count) return;
    positions[id] = incoming_positions[index];
    positions[particle_count + id] =
        incoming_positions[particle_count + index];
    positions[2ull * particle_count + id] =
        incoming_positions[2ull * particle_count + index];
    termination_steps[id] = 0xFFFFFFFFFFFFFFFFull;
    statuses[id] = (uchar)CULBM_ADVECTION_ACTIVE;
}

)CUDA";
}

} // namespace culbm
