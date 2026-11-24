// Native CUDA device kernels for CULBM (D3Q19/D3Q27 momentum, D3Q7/D3Q19
// thermal, Esoteric-Pull in-place streaming, FP16C/FP16S/FP32 storage,
// Smagorinsky SGS, equilibrium wall-stress models, multi-GPU halo packing,
// mesh voxelization and synthetic von Karman inflow).

__device__ __forceinline__ float square_f(const float v) { return v * v; }

__device__ __forceinline__ uint3 cell_coord(const cell_idx_t idx) {
  const uint plane_xy = (uint)kcfg::plane_xy;
  const uint xy = (uint)(idx % (cell_idx_t)plane_xy);
  return make_uint3(xy % kcfg::nx, xy / kcfg::nx, (uint)(idx / (cell_idx_t)plane_xy));
}

__device__ __forceinline__ cell_idx_t cell_linear(const uint3 c) {
  return (cell_idx_t)c.x + ((cell_idx_t)c.y + (cell_idx_t)c.z * (cell_idx_t)kcfg::ny) * (cell_idx_t)kcfg::nx;
}

__device__ __forceinline__ float3 cell_center_pos(const uint3 c) {
  return make_float3((float)c.x + 0.5f - 0.5f * (float)kcfg::nx, (float)c.y + 0.5f - 0.5f * (float)kcfg::ny,
                     (float)c.z + 0.5f - 0.5f * (float)kcfg::nz);
}

__device__ __forceinline__ bool is_halo_coord(const uint3 c) {
  return ((kcfg::split_x > 1u) && (c.x == 0u || c.x + 1u >= kcfg::nx)) ||
         ((kcfg::split_y > 1u) && (c.y == 0u || c.y + 1u >= kcfg::ny)) ||
         ((kcfg::split_z > 1u) && (c.z == 0u || c.z + 1u >= kcfg::nz));
}

__device__ __forceinline__ bool is_halo_index(const cell_idx_t idx) { return is_halo_coord(cell_coord(idx)); }

// ---------------------------------------------------------------------------
// DDF storage codecs (FP16C bit-exact custom 16-bit, FP16S scaled IEEE half, FP32)
// ---------------------------------------------------------------------------

__device__ __forceinline__ ulong ddf_slot(const cell_idx_t cell, const uint dir) {
  return (ulong)dir * kcfg::cells + (ulong)cell;
}

// Custom 16-bit floating-point format: 1 sign bit, 4-bit exponent (bias 15),
// 11-bit mantissa. With --ftz=false, exponent-bias scaling by 2^112 / 2^-112
// handles normals, subnormals and signed zeros in a few instructions while
// matching the reference bit layout across all 65,536 codes and all finite
// IEEE-754 float32 inputs.
__device__ __forceinline__ float decode_fp16c(const ushort code) {
  const uint bits = ((uint)(code & 0x8000u) << 16) | ((uint)(code & 0x7FFFu) << 12);
  return __uint_as_float(bits) * 0x1p112f;
}

__device__ __forceinline__ ushort encode_fp16c(const float value) {
  const uint bits = __float_as_uint(__fmul_rz(value, 0x1p-112f)) + 0x00000800u;
  return (ushort)(((bits & 0x80000000u) >> 16) | ((bits >> 12) & 0x7FFFu));
}

#if defined(DDF_FP16C)
__device__ __forceinline__ float ddf_load(const ddf_elem_t *ptr, const ulong offset) {
  return decode_fp16c(ptr[offset]);
}
__device__ __forceinline__ void ddf_store(ddf_elem_t *ptr, const ulong offset, const float val) {
  ptr[offset] = encode_fp16c(val);
}
#elif defined(DDF_FP16S)
__device__ __forceinline__ float ddf_load(const ddf_elem_t *ptr, const ulong offset) {
  return __half2float(ptr[offset]) * 3.0517578E-5f;
}
__device__ __forceinline__ void ddf_store(ddf_elem_t *ptr, const ulong offset, const float val) {
  ptr[offset] = __float2half_rn(val * 32768.0f);
}
#else
__device__ __forceinline__ float ddf_load(const ddf_elem_t *ptr, const ulong offset) { return ptr[offset]; }
__device__ __forceinline__ void ddf_store(ddf_elem_t *ptr, const ulong offset, const float val) { ptr[offset] = val; }
#endif

// ---------------------------------------------------------------------------
// Discrete velocity descriptors and branchless periodic neighbor offsets
// ---------------------------------------------------------------------------

__device__ __forceinline__ constexpr int vel_cx(const uint q) {
  constexpr int tbl[27] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
  return tbl[q];
}

__device__ __forceinline__ constexpr int vel_cy(const uint q) {
  constexpr int tbl[27] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
  return tbl[q];
}

__device__ __forceinline__ constexpr int vel_cz(const uint q) {
  constexpr int tbl[27] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};
  return tbl[q];
}

__device__ __forceinline__ constexpr float mom_weight(const uint q) {
#if defined(D3Q27)
  return q == 0u ? kcfg::w0 : (q < 7u ? kcfg::ws : (q < 19u ? kcfg::we : kcfg::wc));
#else
  return q == 0u ? kcfg::w0 : (q < 7u ? kcfg::ws : kcfg::we);
#endif
}

struct AxisOffsets {
  cell_idx_t x0, xp, xm;
  cell_idx_t y0, yp, ym;
  cell_idx_t z0, zp, zm;
};

__device__ __forceinline__ AxisOffsets make_axis_offsets(const uint3 c) {
  const cell_idx_t stride_y = (cell_idx_t)kcfg::nx;
  const cell_idx_t stride_z = (cell_idx_t)kcfg::plane_xy;
  AxisOffsets o;
  o.x0 = (cell_idx_t)c.x;
  o.xp = (c.x + 1u == kcfg::nx) ? (cell_idx_t)0u : (cell_idx_t)(c.x + 1u);
  o.xm = (c.x == 0u) ? (cell_idx_t)(kcfg::nx - 1u) : (cell_idx_t)(c.x - 1u);
  o.y0 = (cell_idx_t)c.y * stride_y;
  o.yp = ((c.y + 1u == kcfg::ny) ? (cell_idx_t)0u : (cell_idx_t)(c.y + 1u)) * stride_y;
  o.ym = ((c.y == 0u) ? (cell_idx_t)(kcfg::ny - 1u) : (cell_idx_t)(c.y - 1u)) * stride_y;
  o.z0 = (cell_idx_t)c.z * stride_z;
  o.zp = ((c.z + 1u == kcfg::nz) ? (cell_idx_t)0u : (cell_idx_t)(c.z + 1u)) * stride_z;
  o.zm = ((c.z == 0u) ? (cell_idx_t)(kcfg::nz - 1u) : (cell_idx_t)(c.z - 1u)) * stride_z;
  return o;
}

__device__ __forceinline__ cell_idx_t neighbor_cell(const AxisOffsets &o, const uint q) {
  const cell_idx_t cx = vel_cx(q) > 0 ? o.xp : (vel_cx(q) < 0 ? o.xm : o.x0);
  const cell_idx_t cy = vel_cy(q) > 0 ? o.yp : (vel_cy(q) < 0 ? o.ym : o.y0);
  const cell_idx_t cz = vel_cz(q) > 0 ? o.zp : (vel_cz(q) < 0 ? o.zm : o.z0);
  return cx + cy + cz;
}

// ---------------------------------------------------------------------------
// Precomputed damping-layer rate tables (exact device sinf evaluation once)
// ---------------------------------------------------------------------------

#ifdef BUFFER_NUDGING
__device__ float g_buffer_rate[kcfg::buffer_cells + 1u];
#endif
#ifdef TOP_SPONGE
__device__ float g_sponge_rate[kcfg::sponge_cells > 0u ? kcfg::sponge_cells : 1u];
#endif

#if defined(BUFFER_NUDGING) || defined(TOP_SPONGE)
extern "C" __global__ void culbm_init_damping_tables() {
  const uint d = (uint)cuda_global_thread_id(0);
#ifdef BUFFER_NUDGING
  if (d <= kcfg::buffer_cells) {
    const float xi = kcfg::buffer_cells > 0u ? (1.0f - (float)d / (float)kcfg::buffer_cells) : 1.0f;
    float w = sinf(1.5707963267948966f * xi);
    w *= w;
    g_buffer_rate[d] = w * kcfg::buffer_inv_tau;
  }
#endif
#ifdef TOP_SPONGE
  if (d < kcfg::sponge_cells) {
    const float xi =
        (kcfg::sponge_cells > 1u) ? (1.0f - (float)d / (float)(kcfg::sponge_cells - 1u)) : 1.0f;
    float s = sinf(1.5707963267948966f * xi);
    g_sponge_rate[d] = kcfg::sponge_inv_tau * s * s;
  }
#endif
}
#endif

__device__ __forceinline__ bool is_damping_ref_plane(const uint3 c) {
  bool on_plane = false;
#ifdef BUFFER_NUDGING
  if constexpr (kcfg::has_west && kcfg::downstream_face != 1)
    on_plane = on_plane || ((int)c.x == kcfg::west_x);
  if constexpr (kcfg::has_east && kcfg::downstream_face != 2)
    on_plane = on_plane || ((int)c.x == kcfg::east_x);
  if constexpr (kcfg::has_south && kcfg::downstream_face != 3)
    on_plane = on_plane || ((int)c.y == kcfg::south_y);
  if constexpr (kcfg::has_north && kcfg::downstream_face != 4)
    on_plane = on_plane || ((int)c.y == kcfg::north_y);
  if constexpr (kcfg::has_top)
    on_plane = on_plane || ((int)c.z == kcfg::top_z);
#endif
#ifdef TOP_SPONGE
  if constexpr (kcfg::has_top)
    on_plane = on_plane || ((int)c.z == kcfg::top_z);
#endif
  return on_plane;
}

__device__ __forceinline__ bool is_top_sponge_ref_plane(const uint3 c) {
#ifdef TOP_SPONGE
  if constexpr (kcfg::has_top)
    return (int)c.z == kcfg::top_z;
#endif
  (void)c;
  return false;
}

// ---------------------------------------------------------------------------
// Momentum moments, equilibrium and Guo forcing (bit-exact evaluation order)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float eq_component(const float w_rho, const float w_rhom1, const float neg3_u2,
                                              const float three_eu) {
  return fmaf(w_rho, fmaf(0.5f, fmaf(three_eu, three_eu, neg3_u2), three_eu), w_rhom1);
}

__device__ __forceinline__ float dir_dot(const uint q, const float ax, const float ay, const float az) {
  return (float)vel_cx(q) * ax + (float)vel_cy(q) * ay + (float)vel_cz(q) * az;
}

__device__ __forceinline__ float guo_source_component(const uint q, const float ux, const float uy, const float uz,
                                                      const float fx, const float fy, const float fz,
                                                      const float u_dot_f_scaled) {
  const float wi = mom_weight(q);
  if (q == 0u)
    return 9.0f * wi * u_dot_f_scaled;
  const float ef = dir_dot(q, fx, fy, fz);
  const float eu = dir_dot(q, ux, uy, uz);
  return 9.0f * wi * fmaf(ef, eu + 0.33333334f, u_dot_f_scaled);
}

// Exact left-to-right summation order so floating-point rounding is unchanged.
__device__ __forceinline__ void reconstruct_moments(const float *pop, float *rho_out, float *ux_out, float *uy_out,
                                                    float *uz_out) {
  float r = pop[0];
#pragma unroll
  for (uint q = 1u; q < kcfg::q_mom; ++q)
    r += pop[q];
  r += 1.0f;

  float mx = pop[1] - pop[2];
  mx = mx + pop[7] - pop[8];
  mx = mx + pop[9] - pop[10];
  mx = mx + pop[13] - pop[14];
  mx = mx + pop[15] - pop[16];

  float my = pop[3] - pop[4];
  my = my + pop[7] - pop[8];
  my = my + pop[11] - pop[12];
  my = my + pop[14] - pop[13];
  my = my + pop[17] - pop[18];

  float mz = pop[5] - pop[6];
  mz = mz + pop[9] - pop[10];
  mz = mz + pop[11] - pop[12];
  mz = mz + pop[16] - pop[15];
  mz = mz + pop[18] - pop[17];

#if defined(D3Q27)
  mx = mx + pop[19] - pop[20];
  mx = mx + pop[21] - pop[22];
  mx = mx + pop[23] - pop[24];
  mx = mx + pop[26] - pop[25];

  my = my + pop[19] - pop[20];
  my = my + pop[21] - pop[22];
  my = my + pop[24] - pop[23];
  my = my + pop[25] - pop[26];

  mz = mz + pop[19] - pop[20];
  mz = mz + pop[22] - pop[21];
  mz = mz + pop[23] - pop[24];
  mz = mz + pop[25] - pop[26];
#endif

  *rho_out = r;
  *ux_out = mx / r;
  *uy_out = my / r;
  *uz_out = mz / r;
}

__device__ __forceinline__ void eval_equilibrium_full(const float rho, const float ux, const float uy, const float uz,
                                                      float *feq) {
  const float rhom1 = rho - 1.0f;
  const float neg3_u2 = -3.0f * (square_f(ux) + square_f(uy) + square_f(uz));
  const float u3x = 3.0f * ux;
  const float u3y = 3.0f * uy;
  const float u3z = 3.0f * uz;
  feq[0] = kcfg::w0 * fmaf(rho, 0.5f * neg3_u2, rhom1);
#pragma unroll
  for (uint q = 1u; q < kcfg::q_mom; q += 2u) {
    const float wi = mom_weight(q);
    const float w_rho = wi * rho;
    const float w_rhom1 = wi * rhom1;
    const float eu3 = dir_dot(q, u3x, u3y, u3z);
    feq[q] = eq_component(w_rho, w_rhom1, neg3_u2, eu3);
    feq[q + 1u] = eq_component(w_rho, w_rhom1, neg3_u2, -eu3);
  }
}

// Esoteric-Pull load / store: odd/even step parity swaps the storage slot of
// opposite pairs in-place so each streaming step touches half local and half
// neighbor slots.
__device__ __forceinline__ void pull_populations(const cell_idx_t cell, const AxisOffsets &o, const ddf_elem_t *buf,
                                                 const uint q_count, const bool odd_step, float *pop) {
  pop[0] = ddf_load(buf, ddf_slot(cell, 0u));
#pragma unroll
  for (uint q = 1u; q < q_count; q += 2u) {
    const uint slot_self = odd_step ? q : (q + 1u);
    const uint slot_nb = odd_step ? (q + 1u) : q;
    pop[q] = ddf_load(buf, ddf_slot(cell, slot_self));
    pop[q + 1u] = ddf_load(buf, ddf_slot(neighbor_cell(o, q), slot_nb));
  }
}

__device__ __forceinline__ void push_populations(const cell_idx_t cell, const AxisOffsets &o, const float *pop,
                                                 const uint q_count, const bool odd_step, ddf_elem_t *buf) {
  ddf_store(buf, ddf_slot(cell, 0u), pop[0]);
#pragma unroll
  for (uint q = 1u; q < q_count; q += 2u) {
    const uint slot_self = odd_step ? q : (q + 1u);
    const uint slot_nb = odd_step ? (q + 1u) : q;
    ddf_store(buf, ddf_slot(neighbor_cell(o, q), slot_nb), pop[q]);
    ddf_store(buf, ddf_slot(cell, slot_self), pop[q + 1u]);
  }
}

#ifdef FREE_SLIP_BOUNDARIES
__device__ __forceinline__ void pull_outgoing_populations(const cell_idx_t cell, const AxisOffsets &o,
                                                          const ddf_elem_t *buf, const uint q_count,
                                                          const bool odd_step, float *outgoing) {
#pragma unroll
  for (uint q = 1u; q < q_count; q += 2u) {
    const uint slot_out_p = odd_step ? q : (q + 1u);
    const uint slot_out_m = odd_step ? (q + 1u) : q;
    outgoing[q] = ddf_load(buf, ddf_slot(neighbor_cell(o, q), slot_out_p));
    outgoing[q + 1u] = ddf_load(buf, ddf_slot(cell, slot_out_m));
  }
}

// Replace halfway bounce-back at axis-aligned outer ghost walls by specular
// reflection. For every population travelling into a wall, bounce-back has
// already returned it in the opposite direction; remap to the specular link.
__device__ __forceinline__ void apply_outer_free_slip(const uint3 xyz, const AxisOffsets &o, float *populations,
                                                      const float *outgoing, const uchar *flags,
                                                      const uint direction_count, const bool adiabatic_scalar) {
  const int xg = (int)xyz.x + kcfg::origin_x;
  const int yg = (int)xyz.y + kcfg::origin_y;
  const int zg = (int)xyz.z + kcfg::origin_z;
  const uchar wall_tag = (uchar)(CELL_SOLID | CELL_AUX_X);
  const uchar wall_mask = (uchar)(CELL_BOUNDARY_MASK | CELL_AUX_X);

  for (uint q = 1u; q < direction_count; ++q) {
    const uchar wall_flag = flags[neighbor_cell(o, q)];
    if ((wall_flag & wall_mask) != wall_tag)
      continue;
    if (adiabatic_scalar && (wall_flag & CELL_THERMAL))
      continue;

    const int cx = vel_cx(q);
    const int cy = vel_cy(q);
    const int cz = vel_cz(q);
    const bool flip_x = (cx < 0 && xg == 1 && (flags[neighbor_cell(o, 2u)] & wall_mask) == wall_tag) ||
                        (cx > 0 && xg == (int)kcfg::global_nx - 2 &&
                         (flags[neighbor_cell(o, 1u)] & wall_mask) == wall_tag);
    const bool flip_y = (cy < 0 && yg == 1 && (flags[neighbor_cell(o, 4u)] & wall_mask) == wall_tag) ||
                        (cy > 0 && yg == (int)kcfg::global_ny - 2 &&
                         (flags[neighbor_cell(o, 3u)] & wall_mask) == wall_tag);
    const bool flip_z = (cz < 0 && zg == 1 && (flags[neighbor_cell(o, 6u)] & wall_mask) == wall_tag) ||
                        (cz > 0 && zg == (int)kcfg::global_nz - 2 &&
                         (flags[neighbor_cell(o, 5u)] & wall_mask) == wall_tag);
    if (!(flip_x || flip_y || flip_z))
      continue;

    const int rx = flip_x ? -cx : cx;
    const int ry = flip_y ? -cy : cy;
    const int rz = flip_z ? -cz : cz;
    uint reflected = 0u;
    for (uint k = 1u; k < direction_count; ++k) {
      if (vel_cx(k) == rx && vel_cy(k) == ry && vel_cz(k) == rz) {
        reflected = k;
        break;
      }
    }
    if (reflected != 0u)
      populations[reflected] = outgoing[q];
  }
}
#endif

// ---------------------------------------------------------------------------
// Thermal lattice helpers (D3Q7 / D3Q19)
// ---------------------------------------------------------------------------

#ifdef TEMPERATURE
__device__ __forceinline__ constexpr float thermal_weight(const uint q) {
#ifdef TEMPERATURE_D3Q19
  return q == 0u ? (1.0f / 3.0f) : (q < 7u ? (1.0f / 18.0f) : (1.0f / 36.0f));
#else
  return q == 0u ? 0.25f : 0.125f;
#endif
}

__device__ __forceinline__ float thermal_eq_component(const uint q, const float T, const float Tm1, const float ux,
                                                      const float uy, const float uz) {
#ifdef TEMPERATURE_D3Q19
  const float wi = thermal_weight(q);
  const float eu = dir_dot(q, ux, uy, uz);
  return fmaf(3.0f * wi * T, eu, wi * Tm1);
#else
  if (q == 0u)
    return fmaf(0.25f, T, -0.25f);
  const float wsT4 = 0.5f * T;
  const float wsTm1 = 0.125f * Tm1;
  const float u_proj = (q == 1u) ? ux : (q == 2u) ? -ux : (q == 3u) ? uy : (q == 4u) ? -uy : (q == 5u) ? uz : -uz;
  return fmaf(wsT4, u_proj, wsTm1);
#endif
}

__device__ __forceinline__ void apply_thermal_solid_bc(float *gh, const cell_idx_t cell, const AxisOffsets &o,
                                                       const uchar *flags, const float *T, const ddf_elem_t *gi,
                                                       const bool odd_step) {
#pragma unroll
  for (uint q = 1u; q < kcfg::q_temp; q += 2u) {
    const cell_idx_t jp = neighbor_cell(o, q);
    const cell_idx_t jm = neighbor_cell(o, q + 1u);
    const float wi2 = 2.0f * thermal_weight(q);
    const uchar fp = flags[jp];
    const uchar fm = flags[jm];
    const uint slot_p = odd_step ? q : (q + 1u);
    const uint slot_m = odd_step ? (q + 1u) : q;
    if (fp & CELL_SOLID) {
      const float outgoing = ddf_load(gi, ddf_slot(jp, slot_p));
      gh[q + 1u] = (fp & CELL_THERMAL) ? fmaf(wi2, T[jp] - 1.0f, -outgoing) : outgoing;
    }
    if (fm & CELL_SOLID) {
      const float outgoing = ddf_load(gi, ddf_slot(cell, slot_m));
      gh[q] = (fm & CELL_THERMAL) ? fmaf(wi2, T[jm] - 1.0f, -outgoing) : outgoing;
    }
  }
}

// First-order scalar outlet: copy adjacent-interior macro temperature to the
// open boundary, then let the CELL_THERMAL path rebuild equilibrium populations
// in stream_collide.
extern "C" __global__ void update_temperature_outlet(const ddf_elem_t *gi, float *T, uchar *flags, const ulong t,
                                                     const int face) {
  const cell_idx_t a = (cell_idx_t)cuda_global_thread_id(0);
  uint3 xyz;
  if (face == 1 || face == 2) {
    if (a >= (cell_idx_t)kcfg::area_x)
      return;
    if ((face == 1 && !kcfg::has_west) || (face == 2 && !kcfg::has_east))
      return;
    xyz = make_uint3((uint)(face == 1 ? kcfg::west_x : kcfg::east_x), (uint)(a % (cell_idx_t)kcfg::ny),
                     (uint)(a / (cell_idx_t)kcfg::ny));
  } else if (face == 3 || face == 4) {
    if (a >= (cell_idx_t)kcfg::area_y)
      return;
    if ((face == 3 && !kcfg::has_south) || (face == 4 && !kcfg::has_north))
      return;
    xyz = make_uint3((uint)(a % (cell_idx_t)kcfg::nx), (uint)(face == 3 ? kcfg::south_y : kcfg::north_y),
                     (uint)(a / (cell_idx_t)kcfg::nx));
  } else {
    return;
  }
  if (is_halo_coord(xyz))
    return;
  const cell_idx_t n = cell_linear(xyz);
  const uchar flagsn = flags[n];
  if ((flagsn & CELL_BOUNDARY_MASK) != CELL_EQUILIBRIUM)
    return;

  const int xg = (int)xyz.x + kcfg::origin_x;
  const int zg = (int)xyz.z + kcfg::origin_z;
  if (zg <= 0 || zg >= (int)kcfg::global_nz - 1)
    return;
  if ((face == 3 || face == 4) && (xg <= 0 || xg >= (int)kcfg::global_nx - 1))
    return;

  uint3 inner_xyz = xyz;
  if (face == 1)
    ++inner_xyz.x;
  else if (face == 2)
    --inner_xyz.x;
  else if (face == 3)
    ++inner_xyz.y;
  else
    --inner_xyz.y;
  if (inner_xyz.x >= kcfg::nx || inner_xyz.y >= kcfg::ny)
    return;
  const cell_idx_t inner = cell_linear(inner_xyz);
  const uchar flags_inner = flags[inner];
  if ((flags_inner & CELL_BOUNDARY_MASK) == CELL_SOLID || (flags_inner & CELL_PHASE_MASK) == CELL_GAS)
    return;

  float Tin;
  if (flags_inner & CELL_THERMAL) {
    Tin = T[inner];
  } else {
    const AxisOffsets inner_o = make_axis_offsets(inner_xyz);
    const bool odd_step = (t & 1ull) != 0ull;
    float gh[kcfg::q_temp];
    pull_populations(inner, inner_o, gi, kcfg::q_temp, odd_step, gh);
    apply_thermal_solid_bc(gh, inner, inner_o, flags, T, gi, odd_step);
    Tin = 1.0f;
#pragma unroll
    for (uint i = 0u; i < kcfg::q_temp; ++i)
      Tin += gh[i];
  }
  T[n] = Tin;
  flags[n] = (uchar)(flagsn | CELL_THERMAL);
}
#endif

// ---------------------------------------------------------------------------
// Equilibrium wall-stress models (link-wise bounce-back and SurfaceIBM core)
// ---------------------------------------------------------------------------

#ifdef WALL_MODEL
__device__ __forceinline__ bool wall_model_solid(const uchar flag) {
  return (flag & (CELL_BOUNDARY_MASK | CELL_AUX_X)) == CELL_SOLID;
}

__device__ __forceinline__ float wall_model_stress_factor(const float ut, const bool smooth, const float cd) {
  if (!smooth)
    return cd * ut;
  const float nu = kcfg::wm_nu;
  if (0.5f * ut <= kcfg::wm_re_laminar * nu)
    return 2.0f * nu;
  float utau = sqrtf(2.0f * nu * ut);
  for (uint k = 0u; k < 6u; ++k) {
    const float L = logf(0.5f * utau / nu) + kcfg::wm_log_E;
    utau = fmaxf(utau - (utau * L - kcfg::wm_kappa * ut) / (L + 1.0f), 0.5f * utau);
  }
  return utau * utau / ut;
}

__device__ __forceinline__ constexpr uint wall_model_reverse(const uint q) { return q + 1u - 2u * (1u - (q & 1u)); }

#if defined(D3Q27)
__device__ __forceinline__ void wall_model_corner_swap(const uint faces, float *fhn, const AxisOffsets &o,
                                                       const uchar *flags, float *dp) {
  const uint corner_face[12] = {0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u};
  const uint corner_P[12] = {25u, 24u, 19u, 21u, 23u, 26u, 19u, 21u, 21u, 26u, 19u, 23u};
  const uint corner_M[12] = {20u, 22u, 26u, 23u, 20u, 22u, 24u, 25u, 20u, 24u, 22u, 25u};
  uint candidate = 0u, seen = 0u, shared = 0u;
#pragma unroll
  for (uint q = 0u; q < 12u; ++q) {
    if (((faces >> corner_face[q]) & 1u) == 0u)
      continue;
    if (!wall_model_solid(flags[neighbor_cell(o, wall_model_reverse(corner_P[q]))]) ||
        !wall_model_solid(flags[neighbor_cell(o, wall_model_reverse(corner_M[q]))]))
      continue;
    candidate |= 1u << q;
    const uint links = (1u << (corner_P[q] - 19u)) | (1u << (corner_M[q] - 19u));
    shared |= seen & links;
    seen |= links;
  }
#pragma unroll
  for (uint q = 0u; q < 12u; ++q) {
    const uint P = corner_P[q], M = corner_M[q];
    if (((candidate >> q) & 1u) == 0u || (shared & ((1u << (P - 19u)) | (1u << (M - 19u)))) != 0u)
      continue;
    const float a = fhn[M], b = fhn[P];
    fhn[P] = a;
    fhn[M] = b;
    dp[0] += (a - b) * (float)(vel_cx(P) - vel_cx(M));
    dp[1] += (a - b) * (float)(vel_cy(P) - vel_cy(M));
    dp[2] += (a - b) * (float)(vel_cz(P) - vel_cz(M));
  }
}
#endif

struct WallModelState {
  float f[kcfg::q_mom];
  float ux, uy, uz;
};

__device__ __noinline__ WallModelState apply_wall_model_slow(const uint faces, WallModelState state,
                                                             const AxisOffsets o, const uchar *flags,
                                                             const uint ibm_face_mask, const float rhon) {
  float *fhn = state.f;
  const uint ibm_faces = faces & ibm_face_mask;
  const uint pair_face[12] = {0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u};
  const uint pair_P[12] = {14u, 16u, 7u, 9u, 13u, 18u, 7u, 11u, 15u, 17u, 9u, 11u};
  const uint pair_M[12] = {8u, 10u, 13u, 15u, 8u, 12u, 14u, 17u, 10u, 12u, 16u, 18u};
  const uint pair_b[12] = {1u, 2u, 1u, 2u, 0u, 2u, 0u, 2u, 0u, 1u, 0u, 1u};
  const uint pair_conflict[12] = {0x050u, 0x500u, 0x050u, 0x500u, 0x005u, 0xA00u,
                                  0x005u, 0xA00u, 0x00Au, 0x0A0u, 0x00Au, 0x0A0u};
  uint candidate = 0u;
#pragma unroll
  for (uint q = 0u; q < 12u; ++q) {
    if (((faces >> pair_face[q]) & 1u) == 0u)
      continue;
    if (wall_model_solid(flags[neighbor_cell(o, wall_model_reverse(pair_P[q]))]) &&
        wall_model_solid(flags[neighbor_cell(o, wall_model_reverse(pair_M[q]))]))
      candidate |= 1u << q;
  }
  uint active = 0u;
#pragma unroll
  for (uint q = 0u; q < 12u; ++q) {
    if (((candidate >> q) & 1u) != 0u && (candidate & pair_conflict[q]) == 0u)
      active |= 1u << q;
  }
  if (active == 0u)
    return state;

  float dp[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
  for (uint q = 0u; q < 12u; ++q) {
    if (((active >> q) & 1u) == 0u)
      continue;
    const uint P = pair_P[q], M = pair_M[q];
    const float a = fhn[M], b = fhn[P];
    fhn[P] = a;
    fhn[M] = b;
    dp[pair_b[q]] += 2.0f * (a - b);
  }
#if defined(D3Q27)
  {
    uint edge_faces = 0u;
#pragma unroll
    for (uint f = 0u; f < 6u; ++f)
      edge_faces |= (uint)((active & (3u << (2u * f))) != 0u) << f;
    wall_model_corner_swap(edge_faces, fhn, o, flags, dp);
  }
#endif
  const float rho_inv = 1.0f / rhon;
  const float usx = fmaf(dp[0], rho_inv, state.ux);
  const float usy = fmaf(dp[1], rho_inv, state.uy);
  const float usz = fmaf(dp[2], rho_inv, state.uz);
#pragma unroll
  for (uint f = 0u; f < 6u; ++f) {
    if ((active & (3u << (2u * f))) == 0u || ((ibm_faces >> f) & 1u) != 0u)
      continue;
    const uint a = f >> 1;
    const float ut = a == 0u ? sqrtf(usy * usy + usz * usz)
                   : a == 1u ? sqrtf(usx * usx + usz * usz)
                             : sqrtf(usx * usx + usy * usy);
    const float k = rhon * (f == 5u ? wall_model_stress_factor(ut, kcfg::wm_smooth_h, kcfg::wm_cd_h)
                                    : wall_model_stress_factor(ut, kcfg::wm_smooth_v, kcfg::wm_cd_v));
#pragma unroll
    for (uint q = 2u * f; q < 2u * f + 2u; ++q) {
      if (((active >> q) & 1u) == 0u)
        continue;
      const uint b = pair_b[q];
      const float T = k * (b == 0u ? usx : b == 1u ? usy : usz);
      fhn[pair_P[q]] -= 0.5f * T;
      fhn[pair_M[q]] += 0.5f * T;
      dp[b] -= T;
    }
  }
  state.ux = fmaf(dp[0], rho_inv, state.ux);
  state.uy = fmaf(dp[1], rho_inv, state.uy);
  state.uz = fmaf(dp[2], rho_inv, state.uz);
  return state;
}

__device__ __forceinline__ void apply_wall_model_core(float *fhn, const AxisOffsets &o, const uchar *flags,
                                                      const uint ibm_face_mask, const float rhon, float *uxn,
                                                      float *uyn, float *uzn) {
  const uint faces = (uint)wall_model_solid(flags[o.xp + o.y0 + o.z0]) |
                     ((uint)wall_model_solid(flags[o.xm + o.y0 + o.z0]) << 1) |
                     ((uint)wall_model_solid(flags[o.x0 + o.yp + o.z0]) << 2) |
                     ((uint)wall_model_solid(flags[o.x0 + o.ym + o.z0]) << 3) |
                     ((uint)wall_model_solid(flags[o.x0 + o.y0 + o.zp]) << 4) |
                     ((uint)wall_model_solid(flags[o.x0 + o.y0 + o.zm]) << 5);
  if (faces == 0u)
    return;
  WallModelState state;
#pragma unroll
  for (uint q = 0u; q < kcfg::q_mom; ++q)
    state.f[q] = fhn[q];
  state.ux = *uxn;
  state.uy = *uyn;
  state.uz = *uzn;
  state = apply_wall_model_slow(faces, state, o, flags, ibm_face_mask, rhon);
#pragma unroll
  for (uint q = 0u; q < kcfg::q_mom; ++q)
    fhn[q] = state.f[q];
  *uxn = state.ux;
  *uyn = state.uy;
  *uzn = state.uz;
}
#endif

// ---------------------------------------------------------------------------
// Smagorinsky-Lilly subgrid model (with optional near-wall mixing length)
// ---------------------------------------------------------------------------

#ifdef SUBGRID
#ifdef WALL_MODEL_SGS
__device__ __forceinline__ float subgrid_prefactor_at(const uint3 xyz, const ushort *wall_model_surface) {
  const uint zs = (uint)wall_model_surface[xyz.x + xyz.y * kcfg::nx];
  const float zw = (float)((int)xyz.z + kcfg::origin_z - (int)zs) + 0.5f;
  if (zs == 0xFFFFu || zw <= 0.0f || zw >= kcfg::wm_sgs_layers)
    return kcfg::sgs_prefactor;
  return fmaxf(kcfg::sgs_prefactor, 25.4558441f * square_f(kcfg::wm_kappa * zw));
}
#endif

__device__ __forceinline__ float compute_subgrid_tau_eff(const float *f, const float rho, const float ux,
                                                         const float uy, const float uz, const float fx,
                                                         const float fy, const float fz, const float prefactor) {
#ifdef SUBGRID_FIXED_EDDY_VISCOSITY
  (void)f;
  (void)rho;
  (void)ux;
  (void)uy;
  (void)uz;
  (void)fx;
  (void)fy;
  (void)fz;
  (void)prefactor;
  return kcfg::tau + 3.0f * kcfg::sgs_fixed_nu_t;
#else
  const float shifted_p = (rho - 1.0f) * 0.3333333333333333f;
  float Hxx = f[1] + f[2] + f[7] + f[8] + f[9] + f[10] + f[13] + f[14] + f[15] + f[16] -
              fmaf(rho, ux * ux, shifted_p);
  float Hyy = f[3] + f[4] + f[7] + f[8] + f[11] + f[12] + f[13] + f[14] + f[17] + f[18] -
              fmaf(rho, uy * uy, shifted_p);
  float Hzz = f[5] + f[6] + f[9] + f[10] + f[11] + f[12] + f[15] + f[16] + f[17] + f[18] -
              fmaf(rho, uz * uz, shifted_p);
  float Hxy = f[7] + f[8] - f[13] - f[14] - rho * ux * uy;
  float Hxz = f[9] + f[10] - f[15] - f[16] - rho * ux * uz;
  float Hyz = f[11] + f[12] - f[17] - f[18] - rho * uy * uz;
#if defined(D3Q27)
  const float k_ppp = f[19] + f[20], k_ppm = f[21] + f[22], k_pmp = f[23] + f[24], k_mpp = f[25] + f[26];
  const float k_all = (k_ppp + k_ppm) + (k_pmp + k_mpp);
  Hxx += k_all;
  Hyy += k_all;
  Hzz += k_all;
  Hxy += (k_ppp + k_ppm) - (k_pmp + k_mpp);
  Hxz += (k_ppp + k_pmp) - (k_ppm + k_mpp);
  Hyz += (k_ppp + k_mpp) - (k_ppm + k_pmp);
#endif
#ifdef VOLUME_FORCE
  Hxx = fmaf(ux, fx, Hxx);
  Hyy = fmaf(uy, fy, Hyy);
  Hzz = fmaf(uz, fz, Hzz);
  Hxy = fmaf(0.5f, fmaf(ux, fy, uy * fx), Hxy);
  Hxz = fmaf(0.5f, fmaf(ux, fz, uz * fx), Hxz);
  Hyz = fmaf(0.5f, fmaf(uy, fz, uz * fy), Hyz);
#endif
  const float Q = square_f(Hxx) + square_f(Hyy) + square_f(Hzz) +
                  2.0f * (square_f(Hxy) + square_f(Hxz) + square_f(Hyz));
  return 0.5f * (kcfg::tau + sqrtf(square_f(kcfg::tau) + prefactor * sqrtf(Q) / rho));
#endif
}
#endif

// ---------------------------------------------------------------------------
// Shared damping force assembly (buffer nudging and top sponge)
// ---------------------------------------------------------------------------

#ifdef BUFFER_NUDGING
__device__ __forceinline__ void apply_buffer_nudging(const cell_idx_t cell, const uint3 xyz, const AxisOffsets &o,
                                                     const float *u, const float rhon, const float uxn,
                                                     const float uyn, const float uzn, float &fxn, float &fyn,
                                                     float &fzn) {
  const int xg = (int)xyz.x + kcfg::origin_x;
  const int yg = (int)xyz.y + kcfg::origin_y;
  const int zg = (int)xyz.z + kcfg::origin_z;
  const int nbuf = (int)kcfg::buffer_cells;
  const int dw = xg;
  const int de = (int)(kcfg::global_nx - 1u) - xg;
  const int ds = yg;
  const int dn = (int)(kcfg::global_ny - 1u) - yg;
  const int dt = (int)(kcfg::global_nz - 1u) - zg;
  const bool in_w = (kcfg::downstream_face != 1) && kcfg::has_west && dw >= 0 && dw <= nbuf;
  const bool in_e = (kcfg::downstream_face != 2) && kcfg::has_east && de >= 0 && de <= nbuf;
  const bool in_s = (kcfg::downstream_face != 3) && kcfg::has_south && ds >= 0 && ds <= nbuf;
  const bool in_n = (kcfg::downstream_face != 4) && kcfg::has_north && dn >= 0 && dn <= nbuf;
  const bool in_t = kcfg::has_top && dt >= 0 && dt <= nbuf;
  if (!(in_w || in_e || in_s || in_n || in_t))
    return;

  uint d_min = kcfg::buffer_cells + 1u;
  cell_idx_t n_ref = cell;
  if (in_w && (uint)dw < d_min) {
    d_min = (uint)dw;
    n_ref = (cell_idx_t)kcfg::west_x + o.y0 + o.z0;
  }
  if (in_e && (uint)de < d_min) {
    d_min = (uint)de;
    n_ref = (cell_idx_t)kcfg::east_x + o.y0 + o.z0;
  }
  if (in_s && (uint)ds < d_min) {
    d_min = (uint)ds;
    n_ref = o.x0 + (cell_idx_t)kcfg::south_y * (cell_idx_t)kcfg::nx + o.z0;
  }
  if (in_n && (uint)dn < d_min) {
    d_min = (uint)dn;
    n_ref = o.x0 + (cell_idx_t)kcfg::north_y * (cell_idx_t)kcfg::nx + o.z0;
  }
  if (in_t && (uint)dt < d_min) {
    d_min = (uint)dt;
    n_ref = o.x0 + o.y0 + (cell_idx_t)kcfg::top_z * (cell_idx_t)kcfg::plane_xy;
  }
  const float rate = __ldg(&g_buffer_rate[d_min]);
  const float ax = rate * (u[n_ref] - uxn);
  const float ay = rate * (u[kcfg::cells + (ulong)n_ref] - uyn);
  const float az = kcfg::buffer_nudge_z ? rate * (u[2ull * kcfg::cells + (ulong)n_ref] - uzn) : 0.0f;
  fxn += rhon * ax;
  fyn += rhon * ay;
  fzn += rhon * az;
}
#endif

// ---------------------------------------------------------------------------
// Initialization kernel
// ---------------------------------------------------------------------------

extern "C" __global__ void initialize(ddf_elem_t *fi, const float *rho, float *u, uchar *flags
#ifdef VOLUME_FORCE
                                      ,
                                      const float fx, const float fy, const float fz, const float omega_x,
                                      const float omega_y, const float omega_z
#ifdef TEMPERATURE
                                      ,
                                      const float buoyancy_gx, const float buoyancy_gy, const float buoyancy_gz
#endif
#endif
#ifdef FORCE_FIELD
                                      ,
                                      const float *F, const uint read_force_field
#endif
#ifdef TEMPERATURE
                                      ,
                                      ddf_elem_t *gi, const float *T
#endif
) {
  const cell_idx_t n = (cell_idx_t)cuda_global_thread_id(0);
  if (n >= (cell_idx_t)kcfg::cells)
    return;
  const uint3 xyz = cell_coord(n);
  if (is_halo_coord(xyz))
    return;
  const AxisOffsets o = make_axis_offsets(xyz);
  const uchar flagsn = flags[n];
  const uchar flagsn_bo = flagsn & CELL_BOUNDARY_MASK;
  if (flagsn_bo == CELL_SOLID) {
    u[n] = 0.0f;
    u[kcfg::cells + (ulong)n] = 0.0f;
    u[2ull * kcfg::cells + (ulong)n] = 0.0f;
  }

  const float rhon = rho[n];
  const float uxn = u[n];
  const float uyn = u[kcfg::cells + (ulong)n];
  const float uzn = u[2ull * kcfg::cells + (ulong)n];
  float feq[kcfg::q_mom];
  eval_equilibrium_full(rhon, uxn, uyn, uzn, feq);

#if defined(EQUILIBRIUM_BOUNDARIES) && defined(VOLUME_FORCE)
  if (flagsn_bo == CELL_EQUILIBRIUM) {
    float fxn = fx - 2.0f * rhon * (omega_y * uzn - omega_z * uyn);
    float fyn = fy - 2.0f * rhon * (omega_z * uxn - omega_x * uzn);
    float fzn = fz - 2.0f * rhon * (omega_x * uyn - omega_y * uxn);
#ifdef FORCE_FIELD
    if (read_force_field != 0u) {
      fxn += F[n];
      fyn += F[kcfg::cells + (ulong)n];
      fzn += F[2ull * kcfg::cells + (ulong)n];
    }
#endif
#ifdef TEMPERATURE
    const float dT = T[n] - kcfg::T_ref;
    fxn -= buoyancy_gx * kcfg::beta * dT;
    fyn -= buoyancy_gy * kcfg::beta * dT;
    fzn -= buoyancy_gz * kcfg::beta * dT;
#endif
    const float uF = -0.33333334f * fmaf(uxn, fxn, fmaf(uyn, fyn, uzn * fzn));
#pragma unroll
    for (uint q = 0u; q < kcfg::q_mom; ++q) {
      const float Si = guo_source_component(q, uxn, uyn, uzn, fxn, fyn, fzn, uF);
      feq[q] = fmaf(-0.5f, Si, feq[q]);
    }
  }
#endif

#ifdef TEMPERATURE
  {
    const float Tn = T[n];
    const float Tm1 = Tn - 1.0f;
    ddf_store(gi, ddf_slot(n, 0u), thermal_eq_component(0u, Tn, Tm1, uxn, uyn, uzn));
#pragma unroll
    for (uint q = 1u; q < kcfg::q_temp; q += 2u) {
      const float gp = thermal_eq_component(q, Tn, Tm1, uxn, uyn, uzn);
      const float gm = thermal_eq_component(q + 1u, Tn, Tm1, uxn, uyn, uzn);
      ddf_store(gi, ddf_slot(neighbor_cell(o, q), q + 1u), gp);
      ddf_store(gi, ddf_slot(n, q), gm);
    }
  }
#endif
  push_populations(n, o, feq, kcfg::q_mom, true, fi);
}

// ---------------------------------------------------------------------------
// Main LBM time-step kernel: stream_collide (in-place Esoteric-Pull)
// ---------------------------------------------------------------------------

#if defined(D3Q19)
extern "C" __global__ void __launch_bounds__(cuda_threads_per_block, 10) stream_collide(
#else
extern "C" __global__ void stream_collide(
#endif
    ddf_elem_t *fi, float *rho, float *u, uchar *flags, const ulong t,
                                          const float fx, const float fy, const float fz, const float omega_x,
                                          const float omega_y, const float omega_z
#ifdef TEMPERATURE
                                          ,
                                          const float buoyancy_gx, const float buoyancy_gy, const float buoyancy_gz
#endif
#ifdef FORCE_FIELD
                                          ,
                                          const float *F, const uint read_force_field
#endif
                                          ,
                                          const uint write_macros
#ifdef TEMPERATURE
                                          ,
                                          ddf_elem_t *gi, float *T
#endif
#ifdef SURFACE_IBM
                                          ,
                                          const float *F_ibm, const uchar *surface_ibm_active
#endif
#ifdef WALL_MODEL_SGS
                                          ,
                                          const ushort *wall_model_surface
#endif
) {
  const cell_idx_t n = (cell_idx_t)cuda_global_thread_id(0);
  if (n >= (cell_idx_t)kcfg::cells)
    return;
  const uint3 xyz = cell_coord(n);
  if (is_halo_coord(xyz))
    return;
  const uchar flagsn = flags[n];
  const uchar flagsn_bo = flagsn & CELL_BOUNDARY_MASK;
  const uchar flagsn_phase = flagsn & CELL_PHASE_MASK;
  if (flagsn_bo == CELL_SOLID || flagsn_phase == CELL_GAS)
    return;

  const AxisOffsets o = make_axis_offsets(xyz);
  const bool odd_step = (t & 1ull) != 0ull;
  float fhn[kcfg::q_mom];
  pull_populations(n, o, fi, kcfg::q_mom, odd_step, fhn);

#ifdef FREE_SLIP_BOUNDARIES
  {
    float outgoing[kcfg::q_mom];
    outgoing[0] = fhn[0];
    pull_outgoing_populations(n, o, fi, kcfg::q_mom, odd_step, outgoing);
    apply_outer_free_slip(xyz, o, fhn, outgoing, flags, kcfg::q_mom, false);
  }
#endif

  float rhon, uxn, uyn, uzn;
#ifdef EQUILIBRIUM_BOUNDARIES
  const bool is_eq = (flagsn_bo == CELL_EQUILIBRIUM);
  if (is_eq) {
    rhon = rho[n];
    uxn = u[n];
    uyn = u[kcfg::cells + (ulong)n];
    uzn = u[2ull * kcfg::cells + (ulong)n];
  } else
#else
  constexpr bool is_eq = false;
#endif
  {
    reconstruct_moments(fhn, &rhon, &uxn, &uyn, &uzn);
#ifdef WALL_MODEL
#ifdef WALL_MODEL_IBM
    const uint ibm_face_mask = (uint)surface_ibm_active[n] >> 1u;
#else
    const uint ibm_face_mask = 0u;
#endif
    apply_wall_model_core(fhn, o, flags, ibm_face_mask, rhon, &uxn, &uyn, &uzn);
#endif
  }

  float fxn = fx - 2.0f * rhon * (omega_y * uzn - omega_z * uyn);
  float fyn = fy - 2.0f * rhon * (omega_z * uxn - omega_x * uzn);
  float fzn = fz - 2.0f * rhon * (omega_x * uyn - omega_y * uxn);

#ifdef BUFFER_NUDGING
  if (!is_eq)
    apply_buffer_nudging(n, xyz, o, u, rhon, uxn, uyn, uzn, fxn, fyn, fzn);
#endif

#ifdef TOP_SPONGE
  bool in_top_sponge = false;
  float sponge_sigma = 0.0f;
  cell_idx_t sponge_ref = n;
  if constexpr (kcfg::has_top) {
    if (!is_eq) {
      const int d_t = (int)(kcfg::global_nz - 2u) - ((int)xyz.z + kcfg::origin_z);
      if (d_t >= 0 && d_t < (int)kcfg::sponge_cells) {
        in_top_sponge = true;
        sponge_sigma = __ldg(&g_sponge_rate[d_t]);
        sponge_ref = o.x0 + o.y0 + (cell_idx_t)kcfg::top_z * (cell_idx_t)kcfg::plane_xy;
        fxn += rhon * sponge_sigma * (u[sponge_ref] - uxn);
        fyn += rhon * sponge_sigma * (u[kcfg::cells + (ulong)sponge_ref] - uyn);
        fzn += rhon * sponge_sigma * (u[2ull * kcfg::cells + (ulong)sponge_ref] - uzn);
      }
    }
  }
#endif

#ifdef FORCE_FIELD
  if (read_force_field != 0u) {
    fxn += F[n];
    fyn += F[kcfg::cells + (ulong)n];
    fzn += F[2ull * kcfg::cells + (ulong)n];
  }
#endif

#ifdef SURFACE_IBM
  if (surface_ibm_active[n] != (uchar)0u) {
    fxn += F_ibm[n];
    fyn += F_ibm[kcfg::cells + (ulong)n];
    fzn += F_ibm[2ull * kcfg::cells + (ulong)n];
  }
#endif

#ifdef TEMPERATURE
  float ghn[kcfg::q_temp];
  float Tn;
  {
    pull_populations(n, o, gi, kcfg::q_temp, odd_step, ghn);
    apply_thermal_solid_bc(ghn, n, o, flags, T, gi, odd_step);
#ifdef FREE_SLIP_BOUNDARIES
    {
      float thermal_outgoing[kcfg::q_temp];
      thermal_outgoing[0] = ghn[0];
      pull_outgoing_populations(n, o, gi, kcfg::q_temp, odd_step, thermal_outgoing);
      apply_outer_free_slip(xyz, o, ghn, thermal_outgoing, flags, kcfg::q_temp, true);
    }
#endif
    if (flagsn & CELL_THERMAL) {
      Tn = T[n];
    } else {
      Tn = 0.0f;
#pragma unroll
      for (uint q = 0u; q < kcfg::q_temp; ++q)
        Tn += ghn[q];
      Tn += 1.0f;
    }
#ifdef TOP_SPONGE
    if (in_top_sponge && (flagsn & CELL_THERMAL) == 0u)
      Tn = fmaf(sponge_sigma, T[sponge_ref] - Tn, Tn);
#endif
    const float dT = Tn - kcfg::T_ref;
    fxn -= buoyancy_gx * kcfg::beta * dT;
    fyn -= buoyancy_gy * kcfg::beta * dT;
    fzn -= buoyancy_gz * kcfg::beta * dT;
  }
#endif

#ifdef VOLUME_FORCE
  if (is_eq) {
    uxn = cuda_clamp(uxn, -kcfg::cs, kcfg::cs);
    uyn = cuda_clamp(uyn, -kcfg::cs, kcfg::cs);
    uzn = cuda_clamp(uzn, -kcfg::cs, kcfg::cs);
  } else {
    const float inv_2rho = 0.5f / rhon;
    uxn = cuda_clamp(fmaf(fxn, inv_2rho, uxn), -kcfg::cs, kcfg::cs);
    uyn = cuda_clamp(fmaf(fyn, inv_2rho, uyn), -kcfg::cs, kcfg::cs);
    uzn = cuda_clamp(fmaf(fzn, inv_2rho, uzn), -kcfg::cs, kcfg::cs);
  }
#else
  uxn = cuda_clamp(uxn, -kcfg::cs, kcfg::cs);
  uyn = cuda_clamp(uyn, -kcfg::cs, kcfg::cs);
  uzn = cuda_clamp(uzn, -kcfg::cs, kcfg::cs);
#endif

#ifdef SUBGRID
#ifdef WALL_MODEL_SGS
  const float sgs_pref = subgrid_prefactor_at(xyz, wall_model_surface);
#else
  const float sgs_pref = kcfg::sgs_prefactor;
#endif
  const float tau_eff = compute_subgrid_tau_eff(fhn, rhon, uxn, uyn, uzn, fxn, fyn, fzn, sgs_pref);
#endif

  // Thermal collision and in-place Esoteric-Pull store (streamed on the fly so
  // no geq[] array stays live in registers and ghn[] dies before momentum).
#ifdef TEMPERATURE
  {
    const float Tm1 = Tn - 1.0f;
    const bool fixed_T = (flagsn & CELL_THERMAL) != 0u;
#ifdef SUBGRID_SCALAR
    const float w_T = 1.0f / fmaf(tau_eff - kcfg::tau, kcfg::sgs_scalar_tau_factor, 1.0f / kcfg::omega_T);
#else
    const float w_T = kcfg::omega_T;
#endif
    const float om_wT = 1.0f - w_T;
#ifdef UPDATE_FIELDS
    if (!fixed_T && (write_macros != 0u || is_top_sponge_ref_plane(xyz)))
      T[n] = Tn;
#endif
    const float geq0 = thermal_eq_component(0u, Tn, Tm1, uxn, uyn, uzn);
    ddf_store(gi, ddf_slot(n, 0u), fixed_T ? geq0 : fmaf(om_wT, ghn[0], w_T * geq0));
#pragma unroll
    for (uint q = 1u; q < kcfg::q_temp; q += 2u) {
      const float geq_p = thermal_eq_component(q, Tn, Tm1, uxn, uyn, uzn);
      const float geq_m = thermal_eq_component(q + 1u, Tn, Tm1, uxn, uyn, uzn);
      const float gp = fixed_T ? geq_p : fmaf(om_wT, ghn[q], w_T * geq_p);
      const float gm = fixed_T ? geq_m : fmaf(om_wT, ghn[q + 1u], w_T * geq_m);
      const uint slot_self = odd_step ? q : (q + 1u);
      const uint slot_nb = odd_step ? (q + 1u) : q;
      ddf_store(gi, ddf_slot(neighbor_cell(o, q), slot_nb), gp);
      ddf_store(gi, ddf_slot(n, slot_self), gm);
    }
  }
#endif

#ifdef UPDATE_FIELDS
  if (!is_eq && (write_macros != 0u || is_damping_ref_plane(xyz))) {
    rho[n] = rhon;
    u[n] = uxn;
    u[kcfg::cells + (ulong)n] = uyn;
    u[2ull * kcfg::cells + (ulong)n] = uzn;
  }
#endif

  // Momentum collision and in-place Esoteric-Pull store, streamed per opposite
  // pair so neither feq[] nor Fin[] needs a register array.
  const float rhom1 = rhon - 1.0f;
  const float neg3_u2 = -3.0f * (square_f(uxn) + square_f(uyn) + square_f(uzn));
  const float u3x = 3.0f * uxn;
  const float u3y = 3.0f * uyn;
  const float u3z = 3.0f * uzn;
#ifdef VOLUME_FORCE
  const float uF = -0.33333334f * fmaf(uxn, fxn, fmaf(uyn, fyn, uzn * fzn));
#else
  const float uF = 0.0f;
#endif
#ifdef SUBGRID
  const float w = 1.0f / tau_eff;
#else
  const float w = kcfg::omega;
#endif

#if defined(SRT)
  const float om_w = 1.0f - w;
  const float c_tau = fmaf(w, -0.5f, 1.0f);
  {
    const float feq0 = kcfg::w0 * fmaf(rhon, 0.5f * neg3_u2, rhom1);
#ifdef VOLUME_FORCE
    const float F0 = 9.0f * kcfg::w0 * uF;
    const float out0 = is_eq ? fmaf(-0.5f, F0, feq0) : fmaf(om_w, fhn[0], fmaf(w, feq0, F0 * c_tau));
#else
    const float out0 = is_eq ? feq0 : fmaf(om_w, fhn[0], w * feq0);
#endif
    ddf_store(fi, ddf_slot(n, 0u), out0);
  }
#pragma unroll
  for (uint q = 1u; q < kcfg::q_mom; q += 2u) {
    const float wi = mom_weight(q);
    const float w_rho = wi * rhon;
    const float w_rhom1 = wi * rhom1;
    const float eu3 = dir_dot(q, u3x, u3y, u3z);
    const float feq_p = eq_component(w_rho, w_rhom1, neg3_u2, eu3);
    const float feq_m = eq_component(w_rho, w_rhom1, neg3_u2, -eu3);
#ifdef VOLUME_FORCE
    const float F_p = guo_source_component(q, uxn, uyn, uzn, fxn, fyn, fzn, uF);
    const float F_m = guo_source_component(q + 1u, uxn, uyn, uzn, fxn, fyn, fzn, uF);
    const float out_p = is_eq ? fmaf(-0.5f, F_p, feq_p) : fmaf(om_w, fhn[q], fmaf(w, feq_p, F_p * c_tau));
    const float out_m = is_eq ? fmaf(-0.5f, F_m, feq_m) : fmaf(om_w, fhn[q + 1u], fmaf(w, feq_m, F_m * c_tau));
#else
    const float out_p = is_eq ? feq_p : fmaf(om_w, fhn[q], w * feq_p);
    const float out_m = is_eq ? feq_m : fmaf(om_w, fhn[q + 1u], w * feq_m);
#endif
    const uint slot_self = odd_step ? q : (q + 1u);
    const uint slot_nb = odd_step ? (q + 1u) : q;
    ddf_store(fi, ddf_slot(neighbor_cell(o, q), slot_nb), out_p);
    ddf_store(fi, ddf_slot(n, slot_self), out_m);
  }
#elif defined(TRT)
  const float wp = w;
  const float wm = 1.0f / (kcfg::trt_magic / (1.0f / w - 0.5f) + 0.5f);
  const float half_wp = 0.5f * wp;
  const float half_wm = 0.5f * wm;
  const float c_taup = fmaf(wp, -0.25f, 0.5f);
  const float c_taum = fmaf(wm, -0.25f, 0.5f);
  {
    const float feq0 = kcfg::w0 * fmaf(rhon, 0.5f * neg3_u2, rhom1);
#ifdef VOLUME_FORCE
    const float F0 = 9.0f * kcfg::w0 * uF;
    const float F0_trt = fmaf(c_taup, F0 + F0, c_taum * (F0 - F0));
    const float out0 = is_eq ? fmaf(-0.5f, F0, feq0)
                             : fmaf(half_wp, feq0 - fhn[0] + feq0 - fhn[0],
                                    fmaf(half_wm, feq0 - feq0 - fhn[0] + fhn[0], fhn[0] + F0_trt));
#else
    const float out0 = is_eq ? feq0
                             : fmaf(half_wp, feq0 - fhn[0] + feq0 - fhn[0],
                                    fmaf(half_wm, feq0 - feq0 - fhn[0] + fhn[0], fhn[0]));
#endif
    ddf_store(fi, ddf_slot(n, 0u), out0);
  }
#pragma unroll
  for (uint q = 1u; q < kcfg::q_mom; q += 2u) {
    const float wi = mom_weight(q);
    const float w_rho = wi * rhon;
    const float w_rhom1 = wi * rhom1;
    const float eu3 = dir_dot(q, u3x, u3y, u3z);
    const float feq_p = eq_component(w_rho, w_rhom1, neg3_u2, eu3);
    const float feq_m = eq_component(w_rho, w_rhom1, neg3_u2, -eu3);
    const float fp = fhn[q];
    const float fm = fhn[q + 1u];
#ifdef VOLUME_FORCE
    const float F_p = guo_source_component(q, uxn, uyn, uzn, fxn, fyn, fzn, uF);
    const float F_m = guo_source_component(q + 1u, uxn, uyn, uzn, fxn, fyn, fzn, uF);
    const float src_p = fmaf(c_taup, F_p + F_m, c_taum * (F_p - F_m));
    const float src_m = fmaf(c_taup, F_m + F_p, c_taum * (F_m - F_p));
    const float out_p = is_eq ? fmaf(-0.5f, F_p, feq_p)
                              : fmaf(half_wp, feq_p - fp + feq_m - fm,
                                     fmaf(half_wm, feq_p - feq_m - fp + fm, fp + src_p));
    const float out_m = is_eq ? fmaf(-0.5f, F_m, feq_m)
                              : fmaf(half_wp, feq_m - fm + feq_p - fp,
                                     fmaf(half_wm, feq_m - feq_p - fm + fp, fm + src_m));
#else
    const float out_p = is_eq ? feq_p
                              : fmaf(half_wp, feq_p - fp + feq_m - fm, fmaf(half_wm, feq_p - feq_m - fp + fm, fp));
    const float out_m = is_eq ? feq_m
                              : fmaf(half_wp, feq_m - fm + feq_p - fp, fmaf(half_wm, feq_m - feq_p - fm + fp, fm));
#endif
    const uint slot_self = odd_step ? q : (q + 1u);
    const uint slot_nb = odd_step ? (q + 1u) : q;
    ddf_store(fi, ddf_slot(neighbor_cell(o, q), slot_nb), out_p);
    ddf_store(fi, ddf_slot(n, slot_self), out_m);
  }
#endif
}

// ---------------------------------------------------------------------------
// Macroscopic field reconstruction & SGS diagnostics (update_fields)
// ---------------------------------------------------------------------------

extern "C" __global__ void update_fields(const ddf_elem_t *fi, float *rho, float *u, const uchar *flags,
                                         const ulong t, const float fx, const float fy, const float fz,
                                         const float omega_x, const float omega_y, const float omega_z
#ifdef TEMPERATURE
                                         ,
                                         const float buoyancy_gx, const float buoyancy_gy, const float buoyancy_gz
#endif
#ifdef FORCE_FIELD
                                         ,
                                         const float *F, const uint read_force_field
#endif
#ifdef TEMPERATURE
                                         ,
                                         const ddf_elem_t *gi, float *T
#endif
#ifdef SUBGRID
                                         ,
                                         const uint write_fields, const uint collect_subgrid_statistics,
                                         double *subgrid_sum, float *subgrid_max, ulong *subgrid_count
#endif
#ifdef SURFACE_IBM
                                         ,
                                         const float *F_ibm, const ulong *surface_ibm_active_cells,
                                         const ulong surface_ibm_active_count, const uint surface_ibm_indexed_mode
#endif
#ifdef WALL_MODEL_SGS
                                         ,
                                         const ushort *wall_model_surface
#endif
#ifdef WALL_MODEL_IBM
                                         ,
                                         const uchar *wall_model_ibm_mask
#endif
) {
  const cell_idx_t work_item = (cell_idx_t)cuda_global_thread_id(0);
#ifdef SURFACE_IBM
  const cell_idx_t n = surface_ibm_indexed_mode != 0u
                           ? (work_item < (cell_idx_t)surface_ibm_active_count
                                  ? (cell_idx_t)surface_ibm_active_cells[work_item]
                                  : (cell_idx_t)kcfg::cells)
                           : work_item;
#else
  const cell_idx_t n = work_item;
#endif

#ifdef SUBGRID
  float sampled_nu_t = 0.0f, sampled_rho_nu_t = 0.0f;
  ulong sampled_count = 0ull;
  if (n < (cell_idx_t)kcfg::cells) {
    const uint3 xyz = cell_coord(n);
    if (!is_halo_coord(xyz)) {
#else
  if (n >= (cell_idx_t)kcfg::cells)
    return;
  const uint3 xyz = cell_coord(n);
  if (is_halo_coord(xyz))
    return;
#endif
      const uchar flagsn = flags[n];
      const uchar flagsn_bo = flagsn & CELL_BOUNDARY_MASK;
      const uchar flagsn_phase = flagsn & CELL_PHASE_MASK;
#ifdef SUBGRID
      if (flagsn_bo != CELL_SOLID && flagsn_phase != CELL_GAS) {
#else
  if (flagsn_bo == CELL_SOLID || flagsn_phase == CELL_GAS)
    return;
#endif
        const AxisOffsets o = make_axis_offsets(xyz);
        const bool odd_step = (t & 1ull) != 0ull;
        float fhn[kcfg::q_mom];
        pull_populations(n, o, fi, kcfg::q_mom, odd_step, fhn);
#ifdef FREE_SLIP_BOUNDARIES
        {
          float outgoing[kcfg::q_mom];
          outgoing[0] = fhn[0];
          pull_outgoing_populations(n, o, fi, kcfg::q_mom, odd_step, outgoing);
          apply_outer_free_slip(xyz, o, fhn, outgoing, flags, kcfg::q_mom, false);
        }
#endif
        float rhon, uxn, uyn, uzn;
#ifdef EQUILIBRIUM_BOUNDARIES
        const bool is_eq = (flagsn_bo == CELL_EQUILIBRIUM);
        if (is_eq) {
          rhon = rho[n];
          uxn = u[n];
          uyn = u[kcfg::cells + (ulong)n];
          uzn = u[2ull * kcfg::cells + (ulong)n];
        } else
#else
  constexpr bool is_eq = false;
#endif
        {
          reconstruct_moments(fhn, &rhon, &uxn, &uyn, &uzn);
#ifdef WALL_MODEL
#ifdef WALL_MODEL_IBM
          const uint ibm_face_mask = (uint)wall_model_ibm_mask[n] >> 1u;
#else
          const uint ibm_face_mask = 0u;
#endif
          apply_wall_model_core(fhn, o, flags, ibm_face_mask, rhon, &uxn, &uyn, &uzn);
#endif
        }

        float fxn = fx - 2.0f * rhon * (omega_y * uzn - omega_z * uyn);
        float fyn = fy - 2.0f * rhon * (omega_z * uxn - omega_x * uzn);
        float fzn = fz - 2.0f * rhon * (omega_x * uyn - omega_y * uxn);

#ifdef BUFFER_NUDGING
        if (!is_eq)
          apply_buffer_nudging(n, xyz, o, u, rhon, uxn, uyn, uzn, fxn, fyn, fzn);
#endif

#ifdef TOP_SPONGE
        bool in_top_sponge = false;
        float sponge_sigma = 0.0f;
        cell_idx_t sponge_ref = n;
        if constexpr (kcfg::has_top) {
          if (!is_eq) {
            const int d_t = (int)(kcfg::global_nz - 2u) - ((int)xyz.z + kcfg::origin_z);
            if (d_t >= 0 && d_t < (int)kcfg::sponge_cells) {
              in_top_sponge = true;
              sponge_sigma = __ldg(&g_sponge_rate[d_t]);
              sponge_ref = o.x0 + o.y0 + (cell_idx_t)kcfg::top_z * (cell_idx_t)kcfg::plane_xy;
              fxn += rhon * sponge_sigma * (u[sponge_ref] - uxn);
              fyn += rhon * sponge_sigma * (u[kcfg::cells + (ulong)sponge_ref] - uyn);
              fzn += rhon * sponge_sigma * (u[2ull * kcfg::cells + (ulong)sponge_ref] - uzn);
            }
          }
        }
#endif

#ifdef FORCE_FIELD
        if (read_force_field != 0u) {
          fxn += F[n];
          fyn += F[kcfg::cells + (ulong)n];
          fzn += F[2ull * kcfg::cells + (ulong)n];
        }
#endif

#ifdef SURFACE_IBM
        fxn += F_ibm[n];
        fyn += F_ibm[kcfg::cells + (ulong)n];
        fzn += F_ibm[2ull * kcfg::cells + (ulong)n];
#endif

#ifdef TEMPERATURE
        {
          float ghn[kcfg::q_temp];
          pull_populations(n, o, gi, kcfg::q_temp, odd_step, ghn);
          apply_thermal_solid_bc(ghn, n, o, flags, T, gi, odd_step);
#ifdef FREE_SLIP_BOUNDARIES
          {
            float thermal_outgoing[kcfg::q_temp];
            thermal_outgoing[0] = ghn[0];
            pull_outgoing_populations(n, o, gi, kcfg::q_temp, odd_step, thermal_outgoing);
            apply_outer_free_slip(xyz, o, ghn, thermal_outgoing, flags, kcfg::q_temp, true);
          }
#endif
          float Tn;
          if (flagsn & CELL_THERMAL) {
            Tn = T[n];
          } else {
            Tn = 0.0f;
#pragma unroll
            for (uint q = 0u; q < kcfg::q_temp; ++q)
              Tn += ghn[q];
            Tn += 1.0f;
#ifdef SUBGRID
            if (write_fields != 0u)
#endif
              T[n] = Tn;
          }
#ifdef TOP_SPONGE
          if (in_top_sponge && (flagsn & CELL_THERMAL) == 0u) {
            Tn = fmaf(sponge_sigma, T[sponge_ref] - Tn, Tn);
#ifdef SUBGRID
            if (write_fields != 0u)
#endif
              T[n] = Tn;
          }
#endif
          const float dT = Tn - kcfg::T_ref;
          fxn -= buoyancy_gx * kcfg::beta * dT;
          fyn -= buoyancy_gy * kcfg::beta * dT;
          fzn -= buoyancy_gz * kcfg::beta * dT;
        }
#endif

#ifdef VOLUME_FORCE
        if (is_eq) {
          uxn = cuda_clamp(uxn, -kcfg::cs, kcfg::cs);
          uyn = cuda_clamp(uyn, -kcfg::cs, kcfg::cs);
          uzn = cuda_clamp(uzn, -kcfg::cs, kcfg::cs);
        } else {
          const float inv_2rho = 0.5f / rhon;
          uxn = cuda_clamp(fmaf(fxn, inv_2rho, uxn), -kcfg::cs, kcfg::cs);
          uyn = cuda_clamp(fmaf(fyn, inv_2rho, uyn), -kcfg::cs, kcfg::cs);
          uzn = cuda_clamp(fmaf(fzn, inv_2rho, uzn), -kcfg::cs, kcfg::cs);
        }
#else
    uxn = cuda_clamp(uxn, -kcfg::cs, kcfg::cs);
    uyn = cuda_clamp(uyn, -kcfg::cs, kcfg::cs);
    uzn = cuda_clamp(uzn, -kcfg::cs, kcfg::cs);
#endif

#ifdef SUBGRID
        if (collect_subgrid_statistics != 0u && !is_eq) {
#ifdef WALL_MODEL_SGS
          const float sgs_pref = subgrid_prefactor_at(xyz, wall_model_surface);
#else
          const float sgs_pref = kcfg::sgs_prefactor;
#endif
          const float tau_eff = compute_subgrid_tau_eff(fhn, rhon, uxn, uyn, uzn, fxn, fyn, fzn, sgs_pref);
          sampled_nu_t = (tau_eff - kcfg::tau) * 0.3333333333333333f;
          sampled_rho_nu_t = rhon * sampled_nu_t;
          sampled_count = 1ull;
        }
        if (write_fields != 0u) {
#endif
          if (!is_eq) {
            rho[n] = rhon;
            u[n] = uxn;
            u[kcfg::cells + (ulong)n] = uyn;
            u[2ull * kcfg::cells + (ulong)n] = uzn;
          }
#ifdef SUBGRID
        }
      }
    }
  }
  if (collect_subgrid_statistics != 0u) {
    double sum_nu_t = (double)sampled_nu_t;
    double sum_rho_nu_t = (double)sampled_rho_nu_t;
    float max_nu_t = sampled_nu_t;
    float max_rho_nu_t = sampled_rho_nu_t;
    ulong count = sampled_count;
#pragma unroll
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
      sum_nu_t += __shfl_down_sync(0xFFFFFFFFu, sum_nu_t, offset);
      sum_rho_nu_t += __shfl_down_sync(0xFFFFFFFFu, sum_rho_nu_t, offset);
      max_nu_t = fmaxf(max_nu_t, __shfl_down_sync(0xFFFFFFFFu, max_nu_t, offset));
      max_rho_nu_t = fmaxf(max_rho_nu_t, __shfl_down_sync(0xFFFFFFFFu, max_rho_nu_t, offset));
      count += __shfl_down_sync(0xFFFFFFFFu, count, offset);
    }
    if ((threadIdx.x & 31u) == 0u) {
      atomicAdd(&subgrid_sum[0], sum_nu_t);
      atomicAdd(&subgrid_sum[1], sum_rho_nu_t);
      atomicMax((uint *)&subgrid_max[0], __float_as_uint(max_nu_t));
      atomicMax((uint *)&subgrid_max[1], __float_as_uint(max_rho_nu_t));
      atomicAdd((unsigned long long *)subgrid_count, (unsigned long long)count);
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// Halo packing and unpacking (compile-time unrolled per axis, zero stack frame)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint face_area_of(const uint axis) {
  return axis == 0u ? kcfg::area_x : (axis == 1u ? kcfg::area_y : kcfg::area_z);
}

__device__ __forceinline__ uint3 face_coord(const uint flat, const uint axis, const uint plane_coord) {
  if (axis == 0u)
    return make_uint3(plane_coord, flat % kcfg::ny, flat / kcfg::ny);
  if (axis == 1u)
    return make_uint3(flat % kcfg::nx, plane_coord, flat / kcfg::nx);
  return make_uint3(flat % kcfg::nx, flat / kcfg::nx, plane_coord);
}

__device__ __forceinline__ constexpr uint mom_face_dir(const uint slot) {
  constexpr uchar tbl[2u * kcfg::dims * kcfg::q_mom_face] = {
#if defined(D3Q19)
      1, 7, 13, 9, 15, 2, 8, 14, 10, 16, 3, 7, 14, 11, 17, 4, 8, 13, 12, 18, 5, 9, 16, 11, 18, 6, 10, 15, 12, 17
#elif defined(D3Q27)
      1, 7,  13, 9,  15, 19, 26, 21, 23, 2, 8,  14, 10, 16, 20, 25, 22, 24, 3,  7,  14, 11, 17, 19, 24, 21, 25,
      4, 8,  13, 12, 18, 20, 23, 22, 26, 5, 9,  16, 11, 18, 19, 22, 23, 25, 6,  10, 15, 12, 17, 20, 21, 24, 26
#endif
  };
  return (uint)tbl[slot];
}

template <uint Side>
__device__ __forceinline__ void pack_mom_side(const uint flat, const uint area, const uint3 xyz, const bool odd_step,
                                              ddf_raw_t *dst, const ddf_raw_t *fi) {
  const cell_idx_t cell = cell_linear(xyz);
  const AxisOffsets o = make_axis_offsets(xyz);
#pragma unroll
  for (uint b = 0u; b < kcfg::q_mom_face; ++b) {
    const uint dir = mom_face_dir(Side * kcfg::q_mom_face + b);
    const cell_idx_t src_cell = (dir & 1u) ? neighbor_cell(o, dir) : cell;
    const uint src_slot = odd_step ? ((dir & 1u) ? (dir + 1u) : (dir - 1u)) : dir;
    dst[(ulong)b * (ulong)area + (ulong)flat] = __ldg(&fi[ddf_slot(src_cell, src_slot)]);
  }
}

template <uint Side>
__device__ __forceinline__ void unpack_mom_side(const uint flat, const uint area, const uint3 xyz, const bool odd_step,
                                                const ddf_raw_t *src, ddf_raw_t *fi) {
  const cell_idx_t cell = cell_linear(xyz);
  const AxisOffsets o = make_axis_offsets(xyz);
#pragma unroll
  for (uint b = 0u; b < kcfg::q_mom_face; ++b) {
    const uint dir = mom_face_dir(Side * kcfg::q_mom_face + b);
    const cell_idx_t dst_cell = (dir & 1u) ? cell : neighbor_cell(o, dir - 1u);
    const uint dst_slot = odd_step ? dir : ((dir & 1u) ? (dir + 1u) : (dir - 1u));
    fi[ddf_slot(dst_cell, dst_slot)] = __ldg(&src[(ulong)b * (ulong)area + (ulong)flat]);
  }
}

extern "C" __global__ void culbm_halo_pack_fi(const uint axis, const ulong t, ddf_raw_t *buf_p, ddf_raw_t *buf_m,
                                              const ddf_raw_t *fi) {
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const bool odd_step = (t & 1ull) != 0ull;
  if (axis == 0u) {
    const uint y = flat % kcfg::ny, z = flat / kcfg::ny;
    pack_mom_side<0u>(flat, area, make_uint3(kcfg::nx - 2u, y, z), odd_step, buf_p, fi);
    pack_mom_side<1u>(flat, area, make_uint3(1u, y, z), odd_step, buf_m, fi);
  } else if (axis == 1u) {
    const uint x = flat % kcfg::nx, z = flat / kcfg::nx;
    pack_mom_side<2u>(flat, area, make_uint3(x, kcfg::ny - 2u, z), odd_step, buf_p, fi);
    pack_mom_side<3u>(flat, area, make_uint3(x, 1u, z), odd_step, buf_m, fi);
  } else {
    const uint x = flat % kcfg::nx, y = flat / kcfg::nx;
    pack_mom_side<4u>(flat, area, make_uint3(x, y, kcfg::nz - 2u), odd_step, buf_p, fi);
    pack_mom_side<5u>(flat, area, make_uint3(x, y, 1u), odd_step, buf_m, fi);
  }
}

extern "C" __global__ void culbm_halo_unpack_fi(const uint axis, const ulong t, const ddf_raw_t *buf_p,
                                                const ddf_raw_t *buf_m, ddf_raw_t *fi) {
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const bool odd_step = (t & 1ull) != 0ull;
  if (axis == 0u) {
    const uint y = flat % kcfg::ny, z = flat / kcfg::ny;
    unpack_mom_side<0u>(flat, area, make_uint3(kcfg::nx - 1u, y, z), odd_step, buf_p, fi);
    unpack_mom_side<1u>(flat, area, make_uint3(0u, y, z), odd_step, buf_m, fi);
  } else if (axis == 1u) {
    const uint x = flat % kcfg::nx, z = flat / kcfg::nx;
    unpack_mom_side<2u>(flat, area, make_uint3(x, kcfg::ny - 1u, z), odd_step, buf_p, fi);
    unpack_mom_side<3u>(flat, area, make_uint3(x, 0u, z), odd_step, buf_m, fi);
  } else {
    const uint x = flat % kcfg::nx, y = flat / kcfg::nx;
    unpack_mom_side<4u>(flat, area, make_uint3(x, y, kcfg::nz - 1u), odd_step, buf_p, fi);
    unpack_mom_side<5u>(flat, area, make_uint3(x, y, 0u), odd_step, buf_m, fi);
  }
}

__device__ __forceinline__ void pack_macro_cell(const uint flat, const uint area, const cell_idx_t cell, char *dst,
                                                const float *rho, const float *u, const uchar *flags) {
  float *fbuf = (float *)dst;
  fbuf[flat] = rho[cell];
  fbuf[area + flat] = u[cell];
  fbuf[2u * area + flat] = u[kcfg::cells + (ulong)cell];
  fbuf[3u * area + flat] = u[2ull * kcfg::cells + (ulong)cell];
  ((uchar *)dst)[16u * area + flat] = flags[cell];
}

__device__ __forceinline__ void unpack_macro_cell(const uint flat, const uint area, const cell_idx_t cell,
                                                  const char *src, float *rho, float *u, uchar *flags) {
  const float *fbuf = (const float *)src;
  rho[cell] = fbuf[flat];
  u[cell] = fbuf[area + flat];
  u[kcfg::cells + (ulong)cell] = fbuf[2u * area + flat];
  u[2ull * kcfg::cells + (ulong)cell] = fbuf[3u * area + flat];
  flags[cell] = ((const uchar *)src)[16u * area + flat];
}

extern "C" __global__ void culbm_halo_pack_rho_u_flags(const uint axis, const ulong t, char *buf_p, char *buf_m,
                                                       const float *rho, const float *u, const uchar *flags) {
  (void)t;
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const uint hi = (axis == 0u ? kcfg::nx : (axis == 1u ? kcfg::ny : kcfg::nz)) - 2u;
  pack_macro_cell(flat, area, cell_linear(face_coord(flat, axis, hi)), buf_p, rho, u, flags);
  pack_macro_cell(flat, area, cell_linear(face_coord(flat, axis, 1u)), buf_m, rho, u, flags);
}

extern "C" __global__ void culbm_halo_unpack_rho_u_flags(const uint axis, const ulong t, const char *buf_p,
                                                         const char *buf_m, float *rho, float *u, uchar *flags) {
  (void)t;
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const uint hi = (axis == 0u ? kcfg::nx : (axis == 1u ? kcfg::ny : kcfg::nz)) - 1u;
  unpack_macro_cell(flat, area, cell_linear(face_coord(flat, axis, hi)), buf_p, rho, u, flags);
  unpack_macro_cell(flat, area, cell_linear(face_coord(flat, axis, 0u)), buf_m, rho, u, flags);
}

extern "C" __global__ void culbm_halo_pack_flags(const uint axis, const ulong t, uchar *buf_p, uchar *buf_m,
                                                 const uchar *flags) {
  (void)t;
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const uint hi = (axis == 0u ? kcfg::nx : (axis == 1u ? kcfg::ny : kcfg::nz)) - 2u;
  buf_p[flat] = flags[cell_linear(face_coord(flat, axis, hi))];
  buf_m[flat] = flags[cell_linear(face_coord(flat, axis, 1u))];
}

extern "C" __global__ void culbm_halo_unpack_flags(const uint axis, const ulong t, const uchar *buf_p,
                                                   const uchar *buf_m, uchar *flags) {
  (void)t;
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const uint hi = (axis == 0u ? kcfg::nx : (axis == 1u ? kcfg::ny : kcfg::nz)) - 1u;
  flags[cell_linear(face_coord(flat, axis, hi))] = buf_p[flat];
  flags[cell_linear(face_coord(flat, axis, 0u))] = buf_m[flat];
}

#ifdef TEMPERATURE
__device__ __forceinline__ constexpr uint temp_face_dir(const uint slot) {
#ifdef TEMPERATURE_D3Q19
  constexpr uchar tbl[30] = {1, 7,  13, 9,  15, 2, 8,  14, 10, 16, 3,  7,  14, 11, 17,
                             4, 8,  13, 12, 18, 5, 9,  16, 11, 18, 6,  10, 15, 12, 17};
  return (uint)tbl[slot];
#else
  return slot + 1u;
#endif
}

template <uint Side>
__device__ __forceinline__ void pack_temp_side(const uint flat, const uint area, const uint3 xyz, const bool odd_step,
                                               ddf_raw_t *dst, const ddf_raw_t *gi) {
  const cell_idx_t cell = cell_linear(xyz);
  const AxisOffsets o = make_axis_offsets(xyz);
#pragma unroll
  for (uint b = 0u; b < kcfg::q_temp_face; ++b) {
    const uint dir = temp_face_dir(Side * kcfg::q_temp_face + b);
    const cell_idx_t src_cell = (dir & 1u) ? neighbor_cell(o, dir) : cell;
    const uint src_slot = odd_step ? ((dir & 1u) ? (dir + 1u) : (dir - 1u)) : dir;
    dst[(ulong)b * (ulong)area + (ulong)flat] = __ldg(&gi[ddf_slot(src_cell, src_slot)]);
  }
}

template <uint Side>
__device__ __forceinline__ void unpack_temp_side(const uint flat, const uint area, const uint3 xyz, const bool odd_step,
                                                 const ddf_raw_t *src, ddf_raw_t *gi) {
  const cell_idx_t cell = cell_linear(xyz);
  const AxisOffsets o = make_axis_offsets(xyz);
#pragma unroll
  for (uint b = 0u; b < kcfg::q_temp_face; ++b) {
    const uint dir = temp_face_dir(Side * kcfg::q_temp_face + b);
    const cell_idx_t dst_cell = (dir & 1u) ? cell : neighbor_cell(o, dir - 1u);
    const uint dst_slot = odd_step ? dir : ((dir & 1u) ? (dir + 1u) : (dir - 1u));
    gi[ddf_slot(dst_cell, dst_slot)] = __ldg(&src[(ulong)b * (ulong)area + (ulong)flat]);
  }
}

extern "C" __global__ void culbm_halo_pack_gi(const uint axis, const ulong t, ddf_raw_t *buf_p, ddf_raw_t *buf_m,
                                              const ddf_raw_t *gi) {
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const bool odd_step = (t & 1ull) != 0ull;
  if (axis == 0u) {
    const uint y = flat % kcfg::ny, z = flat / kcfg::ny;
    pack_temp_side<0u>(flat, area, make_uint3(kcfg::nx - 2u, y, z), odd_step, buf_p, gi);
    pack_temp_side<1u>(flat, area, make_uint3(1u, y, z), odd_step, buf_m, gi);
  } else if (axis == 1u) {
    const uint x = flat % kcfg::nx, z = flat / kcfg::nx;
    pack_temp_side<2u>(flat, area, make_uint3(x, kcfg::ny - 2u, z), odd_step, buf_p, gi);
    pack_temp_side<3u>(flat, area, make_uint3(x, 1u, z), odd_step, buf_m, gi);
  } else {
    const uint x = flat % kcfg::nx, y = flat / kcfg::nx;
    pack_temp_side<4u>(flat, area, make_uint3(x, y, kcfg::nz - 2u), odd_step, buf_p, gi);
    pack_temp_side<5u>(flat, area, make_uint3(x, y, 1u), odd_step, buf_m, gi);
  }
}

extern "C" __global__ void culbm_halo_unpack_gi(const uint axis, const ulong t, const ddf_raw_t *buf_p,
                                                const ddf_raw_t *buf_m, ddf_raw_t *gi) {
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const bool odd_step = (t & 1ull) != 0ull;
  if (axis == 0u) {
    const uint y = flat % kcfg::ny, z = flat / kcfg::ny;
    unpack_temp_side<0u>(flat, area, make_uint3(kcfg::nx - 1u, y, z), odd_step, buf_p, gi);
    unpack_temp_side<1u>(flat, area, make_uint3(0u, y, z), odd_step, buf_m, gi);
  } else if (axis == 1u) {
    const uint x = flat % kcfg::nx, z = flat / kcfg::nx;
    unpack_temp_side<2u>(flat, area, make_uint3(x, kcfg::ny - 1u, z), odd_step, buf_p, gi);
    unpack_temp_side<3u>(flat, area, make_uint3(x, 0u, z), odd_step, buf_m, gi);
  } else {
    const uint x = flat % kcfg::nx, y = flat / kcfg::nx;
    unpack_temp_side<4u>(flat, area, make_uint3(x, y, kcfg::nz - 1u), odd_step, buf_p, gi);
    unpack_temp_side<5u>(flat, area, make_uint3(x, y, 0u), odd_step, buf_m, gi);
  }
}

extern "C" __global__ void culbm_halo_pack_fi_gi(const uint axis, const ulong t, ddf_raw_t *buf_p, ddf_raw_t *buf_m,
                                                 const ddf_raw_t *fi, const ddf_raw_t *gi) {
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const bool odd_step = (t & 1ull) != 0ull;
  const ulong temp_off = (ulong)kcfg::q_mom_face * (ulong)area;
  if (axis == 0u) {
    const uint y = flat % kcfg::ny, z = flat / kcfg::ny;
    const uint3 cp = make_uint3(kcfg::nx - 2u, y, z);
    const uint3 cm = make_uint3(1u, y, z);
    pack_mom_side<0u>(flat, area, cp, odd_step, buf_p, fi);
    pack_temp_side<0u>(flat, area, cp, odd_step, buf_p + temp_off, gi);
    pack_mom_side<1u>(flat, area, cm, odd_step, buf_m, fi);
    pack_temp_side<1u>(flat, area, cm, odd_step, buf_m + temp_off, gi);
  } else if (axis == 1u) {
    const uint x = flat % kcfg::nx, z = flat / kcfg::nx;
    const uint3 cp = make_uint3(x, kcfg::ny - 2u, z);
    const uint3 cm = make_uint3(x, 1u, z);
    pack_mom_side<2u>(flat, area, cp, odd_step, buf_p, fi);
    pack_temp_side<2u>(flat, area, cp, odd_step, buf_p + temp_off, gi);
    pack_mom_side<3u>(flat, area, cm, odd_step, buf_m, fi);
    pack_temp_side<3u>(flat, area, cm, odd_step, buf_m + temp_off, gi);
  } else {
    const uint x = flat % kcfg::nx, y = flat / kcfg::nx;
    const uint3 cp = make_uint3(x, y, kcfg::nz - 2u);
    const uint3 cm = make_uint3(x, y, 1u);
    pack_mom_side<4u>(flat, area, cp, odd_step, buf_p, fi);
    pack_temp_side<4u>(flat, area, cp, odd_step, buf_p + temp_off, gi);
    pack_mom_side<5u>(flat, area, cm, odd_step, buf_m, fi);
    pack_temp_side<5u>(flat, area, cm, odd_step, buf_m + temp_off, gi);
  }
}

extern "C" __global__ void culbm_halo_unpack_fi_gi(const uint axis, const ulong t, const ddf_raw_t *buf_p,
                                                   const ddf_raw_t *buf_m, ddf_raw_t *fi, ddf_raw_t *gi) {
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const bool odd_step = (t & 1ull) != 0ull;
  const ulong temp_off = (ulong)kcfg::q_mom_face * (ulong)area;
  if (axis == 0u) {
    const uint y = flat % kcfg::ny, z = flat / kcfg::ny;
    const uint3 cp = make_uint3(kcfg::nx - 1u, y, z);
    const uint3 cm = make_uint3(0u, y, z);
    unpack_mom_side<0u>(flat, area, cp, odd_step, buf_p, fi);
    unpack_temp_side<0u>(flat, area, cp, odd_step, buf_p + temp_off, gi);
    unpack_mom_side<1u>(flat, area, cm, odd_step, buf_m, fi);
    unpack_temp_side<1u>(flat, area, cm, odd_step, buf_m + temp_off, gi);
  } else if (axis == 1u) {
    const uint x = flat % kcfg::nx, z = flat / kcfg::nx;
    const uint3 cp = make_uint3(x, kcfg::ny - 1u, z);
    const uint3 cm = make_uint3(x, 0u, z);
    unpack_mom_side<2u>(flat, area, cp, odd_step, buf_p, fi);
    unpack_temp_side<2u>(flat, area, cp, odd_step, buf_p + temp_off, gi);
    unpack_mom_side<3u>(flat, area, cm, odd_step, buf_m, fi);
    unpack_temp_side<3u>(flat, area, cm, odd_step, buf_m + temp_off, gi);
  } else {
    const uint x = flat % kcfg::nx, y = flat / kcfg::nx;
    const uint3 cp = make_uint3(x, y, kcfg::nz - 1u);
    const uint3 cm = make_uint3(x, y, 0u);
    unpack_mom_side<4u>(flat, area, cp, odd_step, buf_p, fi);
    unpack_temp_side<4u>(flat, area, cp, odd_step, buf_p + temp_off, gi);
    unpack_mom_side<5u>(flat, area, cm, odd_step, buf_m, fi);
    unpack_temp_side<5u>(flat, area, cm, odd_step, buf_m + temp_off, gi);
  }
}

extern "C" __global__ void culbm_halo_pack_T(const uint axis, const ulong t, float *buf_p, float *buf_m,
                                             const float *T) {
  (void)t;
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const uint hi = (axis == 0u ? kcfg::nx : (axis == 1u ? kcfg::ny : kcfg::nz)) - 2u;
  buf_p[flat] = T[cell_linear(face_coord(flat, axis, hi))];
  buf_m[flat] = T[cell_linear(face_coord(flat, axis, 1u))];
}

extern "C" __global__ void culbm_halo_unpack_T(const uint axis, const ulong t, const float *buf_p, const float *buf_m,
                                               float *T) {
  (void)t;
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(axis);
  if (flat >= area)
    return;
  const uint hi = (axis == 0u ? kcfg::nx : (axis == 1u ? kcfg::ny : kcfg::nz)) - 1u;
  T[cell_linear(face_coord(flat, axis, hi))] = buf_p[flat];
  T[cell_linear(face_coord(flat, axis, 0u))] = buf_m[flat];
}
#endif

// ---------------------------------------------------------------------------
// Mesh voxelization kernels (unbinned fallback + 2D tile-binned fast path)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void voxelize_test_triangle(const uint i, const float *p0, const float *p1, const float *p2,
                                                       const float3 r_origin, const float3 r_dir, uint &hit_pos,
                                                       uint &hit_neg, float *distances) {
  const uint tx = 3u * i;
  const float3 a = make_float3(p0[tx], p0[tx + 1u], p0[tx + 2u]);
  const float3 b = make_float3(p1[tx], p1[tx + 1u], p1[tx + 2u]);
  const float3 c = make_float3(p2[tx], p2[tx + 1u], p2[tx + 2u]);
  const float3 e1 = b - a, e2 = c - a, rv = r_origin - a;
  const float3 hv = cross(r_dir, e2);
  const float3 qv = cross(rv, e1);
  const float det = dot(e1, hv);
  const float inv_det = 1.0f / det;
  const float bary_u = inv_det * dot(rv, hv);
  const float bary_v = inv_det * dot(r_dir, qv);
  const float dist = inv_det * dot(e2, qv);
  if (det != 0.0f && bary_u >= 0.0f && bary_u < 1.0f && bary_v >= 0.0f && bary_u + bary_v < 1.0f) {
    if (dist > 0.0f) {
      if (hit_pos < 64u && dist < 65536.0f)
        distances[hit_pos] = dist;
      ++hit_pos;
    } else {
      ++hit_neg;
    }
  }
}

__device__ __forceinline__ void voxelize_apply_ray(const uint direction, const uint3 xyz, const float3 domain_offset,
                                                   const float *bbu, const uint hit_pos, const uint hit_neg,
                                                   float *distances, ddf_elem_t *fi, float *u, uchar *flags,
                                                   const ulong t, const uchar flag) {
  const float x1 = bbu[4], y1 = bbu[5], z1 = bbu[6];
  const float cx = bbu[7], cy = bbu[8], cz = bbu[9];
  const float ux = bbu[10], uy = bbu[11], uz = bbu[12];
  const float rx = bbu[13], ry = bbu[14], rz = bbu[15];

  const uint sort_count = cuda_min(hit_pos, 64u);
  for (uint i = 1u; i < sort_count; ++i) {
    const float key = distances[i];
    int k = (int)i - 1;
    while (k >= 0 && distances[k] > key) {
      distances[k + 1] = distances[k];
      --k;
    }
    distances[k + 1] = key;
  }

  bool inside = ((hit_pos & 1u) != 0u) && ((hit_neg & 1u) != 0u);
  const bool set_u = (square_f(ux) + square_f(uy) + square_f(uz) + square_f(rx) + square_f(ry) + square_f(rz)) > 0.0f;
  uint cursor = (uint)((hit_pos & 1u) != (hit_neg & 1u));
  const uint h0 = direction == 0u ? xyz.x : (direction == 1u ? xyz.y : xyz.z);
  const uint hmax = direction == 0u   ? (uint)cuda_clamp((int)x1 - kcfg::origin_x, 0, (int)kcfg::nx)
                    : direction == 1u ? (uint)cuda_clamp((int)y1 - kcfg::origin_y, 0, (int)kcfg::ny)
                                      : (uint)cuda_clamp((int)z1 - kcfg::origin_z, 0, (int)kcfg::nz);
  const float hmesh = hit_pos > 0u ? ((float)h0 + distances[cuda_min(hit_pos - 1u, 63u)]) : (float)h0;
  const bool odd_step = (t & 1ull) != 0ull;

  for (uint h = h0; h < hmax; ++h) {
    const float ray_tol = 8.0f * 1.1920928955078125e-7f * fmaxf(1.0f, fmaxf(fabsf((float)h), fabsf(hmesh)));
    while (cursor < hit_pos && (float)h > (float)h0 + distances[cuda_min(cursor, 63u)] + ray_tol) {
      inside = !inside;
      ++cursor;
    }
    inside = inside && (cursor < hit_pos && (float)h < hmesh - ray_tol);
    const uint3 c_cur =
        make_uint3(direction == 0u ? h : xyz.x, direction == 1u ? h : xyz.y, direction == 2u ? h : xyz.z);
    const cell_idx_t cell = cell_linear(c_cur);
    uchar fl = flags[cell];
    const float3 p = cell_center_pos(c_cur) + domain_offset;
    const float3 u_set = make_float3(ux, uy, uz) + cross(make_float3(cx, cy, cz) - p, make_float3(rx, ry, rz));
    if (inside) {
      fl = (uchar)((fl & ~CELL_BOUNDARY_MASK) | flag);
      if (set_u && flag == CELL_SOLID) {
        u[cell] = u_set.x;
        u[kcfg::cells + (ulong)cell] = u_set.y;
        u[2ull * kcfg::cells + (ulong)cell] = u_set.z;
      }
    } else if ((fl & CELL_BOUNDARY_MASK) == CELL_SOLID) {
      const float3 un = make_float3(u[cell], u[kcfg::cells + (ulong)cell], u[2ull * kcfg::cells + (ulong)cell]);
      if (un.x == u_set.x && un.y == u_set.y && un.z == u_set.z) {
        if (set_u) {
          const AxisOffsets cell_o = make_axis_offsets(c_cur);
          float feq[kcfg::q_mom];
          eval_equilibrium_full(1.0f, un.x, un.y, un.z, feq);
          push_populations(cell, cell_o, feq, kcfg::q_mom, odd_step, fi);
        }
        fl = ((fl & CELL_BOUNDARY_MASK) == CELL_BOUNDARY_MASK) ? (uchar)(fl & ~CELL_BOUNDARY_MASK)
                                                               : (uchar)(fl & ~flag);
      }
    }
    flags[cell] = fl;
  }
}

extern "C" __global__ void voxelize_mesh(const uint direction, ddf_elem_t *fi, float *u, uchar *flags, const ulong t,
                                         const uchar flag, const float *p0, const float *p1, const float *p2,
                                         const float *bbu) {
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(direction);
  if (flat >= area)
    return;
  const uint tri_count = cuda_as_uint(bbu[0]);
  const float x0 = bbu[1], y0 = bbu[2], z0 = bbu[3];
  const float x1 = bbu[4], y1 = bbu[5], z1 = bbu[6];
  const uint3 xyz =
      direction == 0u
          ? make_uint3((uint)cuda_clamp((int)x0 - kcfg::origin_x, 0, (int)kcfg::nx - 1), flat % kcfg::ny,
                       flat / kcfg::ny)
      : direction == 1u
          ? make_uint3(flat / kcfg::nz, (uint)cuda_clamp((int)y0 - kcfg::origin_y, 0, (int)kcfg::ny - 1),
                       flat % kcfg::nz)
          : make_uint3(flat % kcfg::nx, flat / kcfg::nx,
                       (uint)cuda_clamp((int)z0 - kcfg::origin_z, 0, (int)kcfg::nz - 1));
  const float3 domain_offset =
      make_float3(0.5f * (float)((int)kcfg::nx + 2 * kcfg::origin_x) - 0.5f,
                  0.5f * (float)((int)kcfg::ny + 2 * kcfg::origin_y) - 0.5f,
                  0.5f * (float)((int)kcfg::nz + 2 * kcfg::origin_z) - 0.5f);
  const float3 r_origin = cell_center_pos(xyz) + domain_offset;
  const float3 r_dir = make_float3((float)(direction == 0u), (float)(direction == 1u), (float)(direction == 2u));

  const bool outside = direction == 0u   ? (r_origin.y < y0 || r_origin.z < z0 || r_origin.y >= y1 || r_origin.z >= z1)
                       : direction == 1u ? (r_origin.x < x0 || r_origin.z < z0 || r_origin.x >= x1 || r_origin.z >= z1)
                                         : (r_origin.x < x0 || r_origin.y < y0 || r_origin.x >= x1 || r_origin.y >= y1);
  if (outside)
    return;

  uint hit_pos = 0u, hit_neg = 0u;
  float distances[64];
  for (uint i = 0u; i < tri_count; ++i)
    voxelize_test_triangle(i, p0, p1, p2, r_origin, r_dir, hit_pos, hit_neg, distances);
  voxelize_apply_ray(direction, xyz, domain_offset, bbu, hit_pos, hit_neg, distances, fi, u, flags, t, flag);
}

extern "C" __global__ void voxelize_mesh_binned(const uint direction, ddf_elem_t *fi, float *u, uchar *flags,
                                                const ulong t, const uchar flag, const float *p0, const float *p1,
                                                const float *p2, const float *bbu, const uint bin_size,
                                                const uint nbins_u, const uint *bin_offsets, const uint *bin_tris) {
  const uint flat = (uint)cuda_global_thread_id(0);
  const uint area = face_area_of(direction);
  if (flat >= area)
    return;
  const float x0 = bbu[1], y0 = bbu[2], z0 = bbu[3];
  const float x1 = bbu[4], y1 = bbu[5], z1 = bbu[6];
  const uint3 xyz =
      direction == 0u
          ? make_uint3((uint)cuda_clamp((int)x0 - kcfg::origin_x, 0, (int)kcfg::nx - 1), flat % kcfg::ny,
                       flat / kcfg::ny)
      : direction == 1u
          ? make_uint3(flat / kcfg::nz, (uint)cuda_clamp((int)y0 - kcfg::origin_y, 0, (int)kcfg::ny - 1),
                       flat % kcfg::nz)
          : make_uint3(flat % kcfg::nx, flat / kcfg::nx,
                       (uint)cuda_clamp((int)z0 - kcfg::origin_z, 0, (int)kcfg::nz - 1));
  const float3 domain_offset =
      make_float3(0.5f * (float)((int)kcfg::nx + 2 * kcfg::origin_x) - 0.5f,
                  0.5f * (float)((int)kcfg::ny + 2 * kcfg::origin_y) - 0.5f,
                  0.5f * (float)((int)kcfg::nz + 2 * kcfg::origin_z) - 0.5f);
  const float3 r_origin = cell_center_pos(xyz) + domain_offset;
  const float3 r_dir = make_float3((float)(direction == 0u), (float)(direction == 1u), (float)(direction == 2u));

  const bool outside = direction == 0u   ? (r_origin.y < y0 || r_origin.z < z0 || r_origin.y >= y1 || r_origin.z >= z1)
                       : direction == 1u ? (r_origin.x < x0 || r_origin.z < z0 || r_origin.x >= x1 || r_origin.z >= z1)
                                         : (r_origin.x < x0 || r_origin.y < y0 || r_origin.x >= x1 || r_origin.y >= y1);
  if (outside)
    return;

  const uint u_local = direction == 0u ? xyz.y : xyz.x;
  const uint v_local = direction == 2u ? xyz.y : xyz.z;
  const uint bin_id = (u_local / bin_size) + (v_local / bin_size) * nbins_u;
  const uint k_begin = bin_offsets[bin_id];
  const uint k_end = bin_offsets[bin_id + 1u];

  uint hit_pos = 0u, hit_neg = 0u;
  float distances[64];
  for (uint k = k_begin; k < k_end; ++k)
    voxelize_test_triangle(bin_tris[k], p0, p1, p2, r_origin, r_dir, hit_pos, hit_neg, distances);
  voxelize_apply_ray(direction, xyz, domain_offset, bbu, hit_pos, hit_neg, distances, fi, u, flags, t, flag);
}

extern "C" __global__ void unvoxelize_mesh(uchar *flags, const uchar flag, float x0, float y0, float z0, float x1,
                                           float y1, float z1) {
  const cell_idx_t n = (cell_idx_t)cuda_global_thread_id(0);
  if (n >= (cell_idx_t)kcfg::cells)
    return;
  const float3 domain_offset =
      make_float3(0.5f * (float)((int)kcfg::nx + 2 * kcfg::origin_x) - 0.5f,
                  0.5f * (float)((int)kcfg::ny + 2 * kcfg::origin_y) - 0.5f,
                  0.5f * (float)((int)kcfg::nz + 2 * kcfg::origin_z) - 0.5f);
  const float3 p = cell_center_pos(cell_coord(n)) + domain_offset;
  if (p.x >= x0 - 1.0f && p.y >= y0 - 1.0f && p.z >= z0 - 1.0f && p.x <= x1 + 1.0f && p.y <= y1 + 1.0f &&
      p.z <= z1 + 1.0f)
    flags[n] &= (uchar)~flag;
}

extern "C" __global__ void finalize_voxelize_and(uchar *flags, float x0, float y0, float z0, float x1, float y1,
                                                 float z1) {
  const cell_idx_t n = (cell_idx_t)cuda_global_thread_id(0);
  if (n >= (cell_idx_t)kcfg::cells)
    return;
  const float3 domain_offset =
      make_float3(0.5f * (float)((int)kcfg::nx + 2 * kcfg::origin_x) - 0.5f,
                  0.5f * (float)((int)kcfg::ny + 2 * kcfg::origin_y) - 0.5f,
                  0.5f * (float)((int)kcfg::nz + 2 * kcfg::origin_z) - 0.5f);
  const float3 p = cell_center_pos(cell_coord(n)) + domain_offset;
  if (p.x >= x0 - 1.0f && p.y >= y0 - 1.0f && p.z >= z0 - 1.0f && p.x <= x1 + 1.0f && p.y <= y1 + 1.0f &&
      p.z <= z1 + 1.0f) {
    uchar f = flags[n];
    const bool both = (f & CELL_AUX_X) != 0u && (f & CELL_AUX_Y) != 0u;
    f = both ? (uchar)((f & ~CELL_BOUNDARY_MASK) | CELL_SOLID) : (uchar)(f & ~CELL_SOLID);
    f = (uchar)(f & ~(CELL_AUX_X | CELL_AUX_Y));
    flags[n] = f;
  }
}

// ---------------------------------------------------------------------------
// Synthetic von Karman turbulence inlet
// ---------------------------------------------------------------------------

extern "C" __global__ void vk_inlet_apply(const uint use_interp, const float t0, const float t1, const float alpha,
                                          const ulong point_count, const ulong mode_count, const ulong mode_stride,
                                          const ulong *point_cell, const uchar *point_face, const float *point_data,
                                          const float *mode_data, float *u) {
  const ulong i = cuda_global_thread_id(0);
  if (i >= point_count)
    return;
  const ulong n = point_cell[i];
  const ulong N = point_count;
  const ulong M = mode_count;
  const ulong V = mode_stride;
  const ulong fid = (ulong)(point_face[i] & (uchar)0x07u);
  const float px = point_data[i];
  const float py = point_data[N + i];
  const float pz = point_data[2ull * N + i];
  const float ubx = point_data[3ull * N + i];
  const float uby = point_data[4ull * N + i];
  const float ubz = point_data[5ull * N + i];
  const float sigma = point_data[6ull * N + i];
  if (fid >= 5ull || !(sigma > 0.0f)) {
    u[n] = ubx;
    u[kcfg::cells + n] = uby;
    u[2ull * kcfg::cells + n] = ubz;
    return;
  }
  const ulong fbase = fid * M;
  float qx = 0.0f, qy = 0.0f, qz = 0.0f;
  for (ulong m = 0ull; m < M; ++m) {
    const ulong idx = fbase + m;
    const float kx = __ldg(&mode_data[idx]);
    const float ky = __ldg(&mode_data[V + idx]);
    const float kz = __ldg(&mode_data[2ull * V + idx]);
    const float omega = __ldg(&mode_data[3ull * V + idx]);
    const float cx = __ldg(&mode_data[4ull * V + idx]);
    const float sx = __ldg(&mode_data[5ull * V + idx]);
    const float cy = __ldg(&mode_data[6ull * V + idx]);
    const float sy = __ldg(&mode_data[7ull * V + idx]);
    const float cz = __ldg(&mode_data[8ull * V + idx]);
    const float sz = __ldg(&mode_data[9ull * V + idx]);
    const float spatial = fmaf(kx, px, fmaf(ky, py, kz * pz));
    const float phase0 = fmaf(omega, t0, spatial);
    float s0, c0;
    __sincosf(phase0, &s0, &c0);
    float vx = fmaf(cx, c0, sx * s0);
    float vy = fmaf(cy, c0, sy * s0);
    float vz = fmaf(cz, c0, sz * s0);
    if (use_interp != 0u) {
      const float phase1 = fmaf(omega, t1, spatial);
      float s1, c1;
      __sincosf(phase1, &s1, &c1);
      const float vx1 = fmaf(cx, c1, sx * s1);
      const float vy1 = fmaf(cy, c1, sy * s1);
      const float vz1 = fmaf(cz, c1, sz * s1);
      vx = fmaf(alpha, vx1 - vx, vx);
      vy = fmaf(alpha, vy1 - vy, vy);
      vz = fmaf(alpha, vz1 - vz, vz);
    }
    qx += vx;
    qy += vy;
    qz += vz;
  }
  u[n] = fmaf(sigma, qx, ubx);
  u[kcfg::cells + n] = fmaf(sigma, qy, uby);
  u[2ull * kcfg::cells + n] = fmaf(sigma, qz, ubz);
}
