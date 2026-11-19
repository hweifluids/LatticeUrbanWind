

#ifdef SURFACE_IBM

__device__ __forceinline__ float surface_ibm_roma3_phi(const float r) {
  const float a = fabsf(r);
  if (a >= 1.5f)
    return 0.0f;
  if (a <= 0.5f) {
    return (1.0f + sqrtf(fmaxf(0.0f, 1.0f - 3.0f * a * a))) / 3.0f;
  }
  const float q = 1.0f - a;
  return (5.0f - 3.0f * a - sqrtf(fmaxf(0.0f, 1.0f - 3.0f * q * q))) / 6.0f;
}

__device__ __forceinline__ bool surface_ibm_owned_local(const int x, const int y, const int z) {
  if (x < 0 || y < 0 || z < 0 || x >= (int)def_Nx || y >= (int)def_Ny || z >= (int)def_Nz)
    return false;
  if (def_Dx > 1u && (x == 0 || x == (int)def_Nx - 1))
    return false;
  if (def_Dy > 1u && (y == 0 || y == (int)def_Ny - 1))
    return false;
  if (def_Dz > 1u && (z == 0 || z == (int)def_Nz - 1))
    return false;
  return true;
}

__device__ __forceinline__ void surface_ibm_axis_weights(const float position, const int center, float weights[3]) {
  // nearest-grid-point construction guarantees r in [-0.5, 0.5].  All
  // three Roma weights share the same radical, so evaluate one sqrt rather
  // than three mathematically redundant calls to roma3_phi.
  const float r = position - (float)center;
  const float radical = sqrtf(fmaxf(0.0f, 1.0f - 3.0f * r * r));
  weights[0] = (2.0f - 3.0f * r - radical) / 6.0f;
  weights[1] = (1.0f + radical) / 3.0f;
  weights[2] = (2.0f + 3.0f * r - radical) / 6.0f;
}

// The bounded specialization uses the same normalized weights for J and S.
// It acts only on the owning slab's physical rows; interface TYPE_E rows
// remain exclusively owned by the macro bridge. The uniform specialization
// compiles to the original Roma weights without a branch or extra storage.
__device__ __forceinline__ void surface_ibm_z_weights(
    const float position, const int center, float weights[3]) {
    surface_ibm_axis_weights(position, center, weights);
#ifdef SURFACE_IBM_BOUNDED_Z
    float sum = 0.0f;
    for(int i=0; i<3; ++i) {
        if(center+i-1 < def_ibm_z_min || center+i-1 > def_ibm_z_max) weights[i]=0.0f;
        sum += weights[i];
    }
    const float inverse = sum > 0.0f ? 1.0f/sum : 0.0f;
    for(int i=0; i<3; ++i) weights[i] *= inverse;
#endif
}

extern "C" __global__ void surface_ibm_reset_active_force(const ulong *active_cells, float *force, float *applied_force,
                                                          const ulong active_count) {
  const ulong i = (ulong)cuda_global_thread_id(0);
  if (i >= active_count)
    return;
  const ulong n = active_cells[i];
  force[n] = 0.0f;
  force[(ulong)def_N + n] = 0.0f;
  force[2ull * (ulong)def_N + n] = 0.0f;
  applied_force[i] = 0.0f;
  applied_force[active_count + i] = 0.0f;
  applied_force[2ull * active_count + i] = 0.0f;
}

extern "C" __global__ void surface_ibm_mark_active_cells(const ulong *active_cells, uchar *active_mask,
                                                         const ulong active_count) {
  const ulong i = (ulong)cuda_global_thread_id(0);
  if (i >= active_count)
    return;
  active_mask[active_cells[i]] = (uchar)1u;
}

extern "C" __global__ void surface_ibm_begin_device_step(float *metrics, float *reaction, uint *control,
                                                         float *marker_force, const ulong marker_count) {
  const ulong m = (ulong)cuda_global_thread_id(0);
  if (m < marker_count) {
    marker_force[m] = 0.0f;
    marker_force[marker_count + m] = 0.0f;
    marker_force[2ull * marker_count + m] = 0.0f;
  }
  if (m != 0ull)
    return;
  for (uint i = 0u; i < 4u; ++i)
    metrics[i] = 0.0f;
  for (uint i = 0u; i < 6u; ++i)
    reaction[i] = 0.0f;
  control[0] = 1u; // forcing loop is active
  control[1] = 0u; // number of force updates applied this step
  control[2] = 0u; // persistent non-finite force-update error
}

extern "C" __global__ void surface_ibm_reset_device_measure(float *metrics, uint *control, const uint unconditional) {
  if (cuda_global_thread_id(0) != 0u)
    return;
  if (unconditional == 0u && control[0] == 0u)
    return;
  for (uint i = 0u; i < 4u; ++i)
    metrics[i] = 0.0f;
}

extern "C" __global__ void surface_ibm_finish_device_iteration(const float *metrics, uint *control,
                                                               const float tolerance) {
  if (cuda_global_thread_id(0) != 0u || control[0] == 0u)
    return;
  ++control[1];
  if (metrics[3] == 0.0f && metrics[0] <= tolerance) {
    control[0] = 0u;
  }
}

// Each thread handles one marker/domain fragment.  Only cells owned by this
// domain participate; the host sums fragments across local GPUs and then MPI
// ranks.  Output is SoA: ux, uy, uz, rho, delta-weight, invalid-sample count.
extern "C" __global__ void surface_ibm_interpolate_partial(const float *marker_position, const float *u,
                                                           const float *rho, float *partial,
                                                           const ulong fragment_count) {
  const ulong f = (ulong)cuda_global_thread_id(0);
  if (f >= fragment_count)
    return;

  const float px = marker_position[f];
  const float py = marker_position[fragment_count + f];
  const float pz = marker_position[2ull * fragment_count + f];
  const int cx = (int)floorf(px + 0.5f);
  const int cy = (int)floorf(py + 0.5f);
  const int cz = (int)floorf(pz + 0.5f);
  float weights_x[3], weights_y[3], weights_z[3];
  surface_ibm_axis_weights(px, cx, weights_x);
  surface_ibm_axis_weights(py, cy, weights_y);
  surface_ibm_z_weights(pz, cz, weights_z);

  float ux = 0.0f, uy = 0.0f, uz = 0.0f;
  float rh = 0.0f, sum_w = 0.0f, invalid = 0.0f;
  for (int dz = -1; dz <= 1; ++dz) {
    const int gz = cz + dz;
    const float wz = weights_z[dz + 1];
    for (int dy = -1; dy <= 1; ++dy) {
      const int gy = cy + dy;
      const float wy = weights_y[dy + 1];
      for (int dx = -1; dx <= 1; ++dx) {
        const int gx = cx + dx;
        const float w = weights_x[dx + 1] * wy * wz;
        if (w == 0.0f)
          continue;
        const int lx = gx - def_Ox;
        const int ly = gy - def_Oy;
        const int lz = gz - def_Oz;
        if (!surface_ibm_owned_local(lx, ly, lz))
          continue;
        const uxx n = index(cuda_uint3((uint)lx, (uint)ly, (uint)lz));
        const float rhoi = rho[n];
        const float uxi = u[n];
        const float uyi = u[(ulong)def_N + (ulong)n];
        const float uzi = u[2ull * (ulong)def_N + (ulong)n];
        if (!(rhoi > 1.0e-12f) || !isfinite(rhoi) || !isfinite(uxi) || !isfinite(uyi) || !isfinite(uzi)) {
          invalid += 1.0f;
          continue;
        }
        ux += w * uxi;
        uy += w * uyi;
        uz += w * uzi;
        rh += w * rhoi;
        sum_w += w;
      }
    }
  }
  partial[f] = ux;
  partial[fragment_count + f] = uy;
  partial[2ull * fragment_count + f] = uz;
  partial[3ull * fragment_count + f] = rh;
  partial[4ull * fragment_count + f] = sum_w;
  partial[5ull * fragment_count + f] = invalid;
}

// Single-device fast path.  Every marker has exactly one local fragment, so
// interpolation, residual evaluation and marker-force update can remain on
// the GPU.  Only the fixed-size metrics/reaction buffers are read by the host.
// metrics = {max_slip, weighted_slip_squared, volume, invalid_count}.
extern "C" __global__ void surface_ibm_direct_forcing_step(const float *marker_position, const float *wall_velocity,
                                                           const float *marker_volume, const float *marker_coupling,
                                                           const float *u, const float *rho, float *force,
                                                           float *marker_metrics, float *marker_force, uint *control,
                                                           const ulong marker_count, const float relaxation,
                                                           const uint mode) {
  const ulong m = (ulong)cuda_global_thread_id(0);
  if (m >= marker_count)
    return;
  // mode: 0 = unconditional final diagnostic, 1 = controlled update,
  // 2 = controlled post-application diagnostic.
  if (mode != 0u && control[0] == 0u)
    return;
  if (mode != 1u) {
    marker_metrics[m] = 0.0f;
    marker_metrics[marker_count + m] = 0.0f;
    marker_metrics[2ull * marker_count + m] = 0.0f;
    marker_metrics[3ull * marker_count + m] = 0.0f;
  }
  const float px = marker_position[m];
  const float py = marker_position[marker_count + m];
  const float pz = marker_position[2ull * marker_count + m];
  const int cx = (int)floorf(px + 0.5f);
  const int cy = (int)floorf(py + 0.5f);
  const int cz = (int)floorf(pz + 0.5f);
  float weights_x[3], weights_y[3], weights_z[3];
  surface_ibm_axis_weights(px, cx, weights_x);
  surface_ibm_axis_weights(py, cy, weights_y);
  surface_ibm_z_weights(pz, cz, weights_z);

  float ux = 0.0f, uy = 0.0f, uz = 0.0f;
  float rh = 0.0f, sum_w = 0.0f;
  uint invalid = 0u;
  for (int dz = -1; dz <= 1; ++dz) {
    const int gz = cz + dz;
    const float wz = weights_z[dz + 1];
    for (int dy = -1; dy <= 1; ++dy) {
      const int gy = cy + dy;
      const float wy = weights_y[dy + 1];
      for (int dx = -1; dx <= 1; ++dx) {
        const int gx = cx + dx;
        const float w = weights_x[dx + 1] * wy * wz;
        if (w == 0.0f)
          continue;
        const int lx = gx - def_Ox;
        const int ly = gy - def_Oy;
        const int lz = gz - def_Oz;
        if (!surface_ibm_owned_local(lx, ly, lz))
          continue;
        const uxx n = index(cuda_uint3((uint)lx, (uint)ly, (uint)lz));
        const float rhoi = rho[n];
        const float uxi = u[n];
        const float uyi = u[(ulong)def_N + (ulong)n];
        const float uzi = u[2ull * (ulong)def_N + (ulong)n];
        if (!(rhoi > 1.0e-12f) || !isfinite(rhoi) || !isfinite(uxi) || !isfinite(uyi) || !isfinite(uzi)) {
          invalid = 1u;
          continue;
        }
        ux += w * uxi;
        uy += w * uyi;
        uz += w * uzi;
        rh += w * rhoi;
        sum_w += w;
      }
    }
  }

  if (invalid != 0u || !isfinite(sum_w) || fabsf(sum_w - 1.0f) > 2.0e-4f || !(rh > 1.0e-12f) || !isfinite(rh)) {
    if (mode == 1u)
      atomicExch(&control[2], 1u);
    else
      marker_metrics[3ull * marker_count + m] = 1.0f;
    return;
  }
  const float inv_weight = 1.0f / sum_w;
  ux *= inv_weight;
  uy *= inv_weight;
  uz *= inv_weight;
  rh *= inv_weight;
  const float rx = wall_velocity[m] - ux;
  const float ry = wall_velocity[marker_count + m] - uy;
  const float rz = wall_velocity[2ull * marker_count + m] - uz;
  const float slip = sqrtf(rx * rx + ry * ry + rz * rz);
  const float volume = marker_volume[m];
  const float coupling = marker_coupling[m];
  if (!isfinite(slip) || !isfinite(volume) || !(volume > 0.0f) || !isfinite(coupling) || !(coupling > 1.0e-8f)) {
    if (mode == 1u)
      atomicExch(&control[2], 1u);
    else
      marker_metrics[3ull * marker_count + m] = 1.0f;
    return;
  }
  if (mode != 1u) {
    marker_metrics[m] = slip;
    marker_metrics[marker_count + m] = volume * slip * slip;
    marker_metrics[2ull * marker_count + m] = volume;
  }

  // The explicit IBM force is rebuilt from zero every time step.  At least
  // one update is therefore required even when the incoming slip is small;
  // otherwise the wall force alternates between zero and nonzero.  A
  // separate post-application interpolation controls later iterations.
  if (mode != 1u)
    return;
  const float factor = 2.0f * rh * relaxation / coupling;
  const float dfx = volume * factor * rx;
  const float dfy = volume * factor * ry;
  const float dfz = volume * factor * rz;
  if (!isfinite(dfx) || !isfinite(dfy) || !isfinite(dfz)) {
    atomicExch(&control[2], 1u);
    return;
  }
  marker_force[m] += dfx;
  marker_force[marker_count + m] += dfy;
  marker_force[2ull * marker_count + m] += dfz;

  for (int dz = -1; dz <= 1; ++dz) {
    const int gz = cz + dz;
    const float wz = weights_z[dz + 1];
    for (int dy = -1; dy <= 1; ++dy) {
      const int gy = cy + dy;
      const float wy = weights_y[dy + 1];
      for (int dx = -1; dx <= 1; ++dx) {
        const int gx = cx + dx;
        const float w = weights_x[dx + 1] * wy * wz;
        if (w == 0.0f)
          continue;
        const int lx = gx - def_Ox;
        const int ly = gy - def_Oy;
        const int lz = gz - def_Oz;
        if (!surface_ibm_owned_local(lx, ly, lz))
          continue;
        const uxx n = index(cuda_uint3((uint)lx, (uint)ly, (uint)lz));
        atomicAdd(&force[n], w * dfx);
        atomicAdd(&force[(ulong)def_N + (ulong)n], w * dfy);
        atomicAdd(&force[2ull * (ulong)def_N + (ulong)n], w * dfz);
      }
    }
  }
}

// Each block reduces 64 marker diagnostics before one scalar atomic update.
// This remains parallel for million-marker surfaces while reducing output
// contention by cuda_threads_per_block.  The final measurement also reduces
// cumulative marker reaction.
extern "C" __global__ void surface_ibm_reduce_device_measure(const float *marker_position, const float *marker_metrics,
                                                             const float *marker_force, float *metrics, float *reaction,
                                                             const ulong marker_count, const uint *control,
                                                             const uint unconditional) {
  if (unconditional == 0u && control[0] == 0u)
    return;
  const uint tid = (uint)cuda_local_thread_id(0);
  float local[10] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const ulong m = (ulong)cuda_global_thread_id(0);
  if (m < marker_count) {
    local[0] = fmaxf(local[0], marker_metrics[m]);
    local[1] += marker_metrics[marker_count + m];
    local[2] += marker_metrics[2ull * marker_count + m];
    local[3] += marker_metrics[3ull * marker_count + m];
    if (unconditional != 0u) {
      const float rx = -marker_force[m];
      const float ry = -marker_force[marker_count + m];
      const float rz = -marker_force[2ull * marker_count + m];
      const float px = marker_position[m];
      const float py = marker_position[marker_count + m];
      const float pz = marker_position[2ull * marker_count + m];
      local[4] += rx;
      local[5] += ry;
      local[6] += rz;
      local[7] += py * rz - pz * ry;
      local[8] += pz * rx - px * rz;
      local[9] += px * ry - py * rx;
    }
  }
  __shared__ float cache[10][cuda_threads_per_block];
#pragma unroll
  for (uint component = 0u; component < 10u; ++component)
    cache[component][tid] = local[component];
  __syncthreads();
  for (uint stride = cuda_threads_per_block / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      cache[0][tid] = fmaxf(cache[0][tid], cache[0][tid + stride]);
#pragma unroll
      for (uint component = 1u; component < 4u; ++component)
        cache[component][tid] += cache[component][tid + stride];
      if (unconditional != 0u) {
#pragma unroll
        for (uint component = 4u; component < 10u; ++component)
          cache[component][tid] += cache[component][tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid != 0u)
    return;
  atomicMax((uint *)&metrics[0], __float_as_uint(cache[0][0]));
  for (uint component = 1u; component < 4u; ++component)
    atomicAdd(&metrics[component], cache[component][0]);
  if (unconditional != 0u) {
    for (uint component = 0u; component < 6u; ++component)
      atomicAdd(&reaction[component], cache[component + 4u][0]);
  }
}

// Interpolate the x component of a field that was produced by spreading one
// unit force-density value at every marker. The resulting J*S row sum is a
// marker-local coupling normalization for multi-direct forcing. Output uses
// the first two lanes of the ordinary partial buffer: value and weight.
extern "C" __global__ void surface_ibm_interpolate_coupling(const float *marker_position, const float *field,
                                                            float *partial, const ulong fragment_count) {
  const ulong f = (ulong)cuda_global_thread_id(0);
  if (f >= fragment_count)
    return;

  const float px = marker_position[f];
  const float py = marker_position[fragment_count + f];
  const float pz = marker_position[2ull * fragment_count + f];
  const int cx = (int)floorf(px + 0.5f);
  const int cy = (int)floorf(py + 0.5f);
  const int cz = (int)floorf(pz + 0.5f);
  float weights_x[3], weights_y[3], weights_z[3];
  surface_ibm_axis_weights(px, cx, weights_x);
  surface_ibm_axis_weights(py, cy, weights_y);
  surface_ibm_z_weights(pz, cz, weights_z);
  float value = 0.0f, sum_w = 0.0f;
  for (int dz = -1; dz <= 1; ++dz) {
    const int gz = cz + dz;
    const float wz = weights_z[dz + 1];
    for (int dy = -1; dy <= 1; ++dy) {
      const int gy = cy + dy;
      const float wy = weights_y[dy + 1];
      for (int dx = -1; dx <= 1; ++dx) {
        const int gx = cx + dx;
        const float w = weights_x[dx + 1] * wy * wz;
        if (w == 0.0f)
          continue;
        const int lx = gx - def_Ox;
        const int ly = gy - def_Oy;
        const int lz = gz - def_Oz;
        if (!surface_ibm_owned_local(lx, ly, lz))
          continue;
        const uxx n = index(cuda_uint3((uint)lx, (uint)ly, (uint)lz));
        const float sample = field[n];
        if (isfinite(sample))
          value += w * sample;
        sum_w += w;
      }
    }
  }
  partial[f] = value;
  partial[fragment_count + f] = sum_w;
}

// fragment_force already contains the marker volume weight
// (area_lu2*support_thickness_lu).  A fragment writes owned cells only, so no
// Eulerian force halo reduction is needed; overlapping markers still require
// atomics within a domain.
extern "C" __global__ void surface_ibm_spread(const float *marker_position, const float *fragment_force, float *force,
                                              const ulong fragment_count) {
  const ulong f = (ulong)cuda_global_thread_id(0);
  if (f >= fragment_count)
    return;

  const float px = marker_position[f];
  const float py = marker_position[fragment_count + f];
  const float pz = marker_position[2ull * fragment_count + f];
  const float fx = fragment_force[f];
  const float fy = fragment_force[fragment_count + f];
  const float fz = fragment_force[2ull * fragment_count + f];
  if (!isfinite(fx) || !isfinite(fy) || !isfinite(fz))
    return;
  if (fx == 0.0f && fy == 0.0f && fz == 0.0f)
    return;

  const int cx = (int)floorf(px + 0.5f);
  const int cy = (int)floorf(py + 0.5f);
  const int cz = (int)floorf(pz + 0.5f);
  float weights_x[3], weights_y[3], weights_z[3];
  surface_ibm_axis_weights(px, cx, weights_x);
  surface_ibm_axis_weights(py, cy, weights_y);
  surface_ibm_z_weights(pz, cz, weights_z);
  for (int dz = -1; dz <= 1; ++dz) {
    const int gz = cz + dz;
    const float wz = weights_z[dz + 1];
    for (int dy = -1; dy <= 1; ++dy) {
      const int gy = cy + dy;
      const float wy = weights_y[dy + 1];
      for (int dx = -1; dx <= 1; ++dx) {
        const int gx = cx + dx;
        const float w = weights_x[dx + 1] * wy * wz;
        if (w == 0.0f)
          continue;
        const int lx = gx - def_Ox;
        const int ly = gy - def_Oy;
        const int lz = gz - def_Oz;
        if (!surface_ibm_owned_local(lx, ly, lz))
          continue;
        const uxx n = index(cuda_uint3((uint)lx, (uint)ly, (uint)lz));
        atomicAdd(&force[n], w * fx);
        atomicAdd(&force[(ulong)def_N + (ulong)n], w * fy);
        atomicAdd(&force[2ull * (ulong)def_N + (ulong)n], w * fz);
      }
    }
  }
}

// Apply only the difference between the newly accumulated Eulerian force and
// the force already represented in velocity.  The compact snapshot is indexed
// by the unique active-cell list, so direct-forcing iterations never scan the
// bulk domain and velocity observes the fully reduced atomic spread.
extern "C" __global__ void surface_ibm_apply_active_force_delta(const ulong *active_cells, const float *force,
                                                                float *applied_force, const float *rho, float *u,
                                                                const ulong active_count, const uint *control,
                                                                const uint controlled) {
  const ulong i = (ulong)cuda_global_thread_id(0);
  if (i >= active_count)
    return;
  if (controlled != 0u && control[0] == 0u)
    return;
  const ulong n = active_cells[i];
  const float rhoi = rho[n];
  if (!(rhoi > 1.0e-12f) || !isfinite(rhoi))
    return;
  const float fx = force[n];
  const float fy = force[(ulong)def_N + n];
  const float fz = force[2ull * (ulong)def_N + n];
  const float old_x = applied_force[i];
  const float old_y = applied_force[active_count + i];
  const float old_z = applied_force[2ull * active_count + i];
  if (!isfinite(fx) || !isfinite(fy) || !isfinite(fz) || !isfinite(old_x) || !isfinite(old_y) || !isfinite(old_z))
    return;
  applied_force[i] = fx;
  applied_force[active_count + i] = fy;
  applied_force[2ull * active_count + i] = fz;
  const float factor = 0.5f / rhoi;
  u[n] = fmaf(factor, fx - old_x, u[n]);
  u[(ulong)def_N + n] = fmaf(factor, fy - old_y, u[(ulong)def_N + n]);
  u[2ull * (ulong)def_N + n] = fmaf(factor, fz - old_z, u[2ull * (ulong)def_N + n]);
}

#endif // SURFACE_IBM
