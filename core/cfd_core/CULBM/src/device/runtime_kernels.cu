

#ifdef TEMPERATURE
// Macro-field mirror used for diagnostics on the GEA atmospheric validation
// walls.  Population-level impermeable/free-slip behavior is provided by the
// FREE_SLIP_BOUNDARIES path in stream_collide.  The production case/deck target
// does not define that extension or call this kernel.
extern "C" __global__ void culbm_update_free_slip_boundaries(float *rho, float *u, float *T, const uchar *flags,
                                                             const uint face_mask) {
  const cell_idx_t n = cuda_global_thread_id(0);
  if (n >= (cell_idx_t)kcfg::cells || is_halo_index(n))
    return;
  if ((flags[n] & CELL_BOUNDARY_MASK) != CELL_SOLID)
    return;

  const uint3 xyz = cell_coord(n);
  const int x_global = (int)xyz.x + kcfg::origin_x;
  const int y_global = (int)xyz.y + kcfg::origin_y;
  const int z_global = (int)xyz.z + kcfg::origin_z;

  const bool west = (face_mask & 1u) != 0u && x_global == 0;
  const bool east = (face_mask & 2u) != 0u && x_global == (int)kcfg::global_nx - 1;
  const bool south = (face_mask & 4u) != 0u && y_global == 0;
  const bool north = (face_mask & 8u) != 0u && y_global == (int)kcfg::global_ny - 1;
  const bool bottom = (face_mask & 16u) != 0u && z_global == 0;
  const bool top = (face_mask & 32u) != 0u && z_global == (int)kcfg::global_nz - 1;
  if (!(west || east || south || north || bottom || top))
    return;

  uint3 interior = xyz;
  if (west)
    ++interior.x;
  if (east)
    --interior.x;
  if (south)
    ++interior.y;
  if (north)
    --interior.y;
  if (bottom)
    ++interior.z;
  if (top)
    --interior.z;
  if (interior.x >= kcfg::nx || interior.y >= kcfg::ny || interior.z >= kcfg::nz)
    return;

  const cell_idx_t source_index = cell_linear(interior);
  rho[n] = rho[source_index];
  u[n] = (west || east) ? 0.0f : u[source_index];
  u[(ulong)kcfg::cells + (ulong)n] = (south || north) ? 0.0f : u[(ulong)kcfg::cells + (ulong)source_index];
  u[2ull * (ulong)kcfg::cells + (ulong)n] = (bottom || top) ? 0.0f : u[2ull * (ulong)kcfg::cells + (ulong)source_index];
  // CELL_SOLID without CELL_THERMAL uses the production scalar bounce-back path.
  // Keeping the macro value mirrored makes diagnostics at the ghost wall
  // consistent without imposing a Dirichlet thermal boundary.
  T[n] = T[source_index];
}
#endif

extern "C" __global__ void culbm_accumulate_statistics(const float *rho, const float *u,
#ifdef TEMPERATURE
                                                       const float *T,
#endif
                                                       float *mean_u, float *mean_rho, float *M2_u,
#ifdef TEMPERATURE
                                                       float *mean_T,
#endif
                                                       const float inverse_sample_count) {
  const ulong n = cuda_global_thread_id(0);
  if (n >= (ulong)kcfg::cells || is_halo_index((cell_idx_t)n))
    return;

  const float ux = u[n];
  const float uy = u[(ulong)kcfg::cells + n];
  const float uz = u[2ull * (ulong)kcfg::cells + n];

  float mean = mean_u[n];
  float delta = ux - mean;
  mean += delta * inverse_sample_count;
  M2_u[n] += delta * (ux - mean);
  mean_u[n] = mean;

  mean = mean_u[(ulong)kcfg::cells + n];
  delta = uy - mean;
  mean += delta * inverse_sample_count;
  M2_u[(ulong)kcfg::cells + n] += delta * (uy - mean);
  mean_u[(ulong)kcfg::cells + n] = mean;

  mean = mean_u[2ull * (ulong)kcfg::cells + n];
  delta = uz - mean;
  mean += delta * inverse_sample_count;
  M2_u[2ull * (ulong)kcfg::cells + n] += delta * (uz - mean);
  mean_u[2ull * (ulong)kcfg::cells + n] = mean;

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

extern "C" __global__ void culbm_gather_probe_velocity(const float *u, const ulong *indices, float *values,
                                                       const ulong count) {
  const ulong i = cuda_global_thread_id(0);
  if (i >= count)
    return;
  const ulong n = indices[i];
  if (n >= (ulong)kcfg::cells) {
    values[i] = 0.0f;
    values[count + i] = 0.0f;
    values[2ull * count + i] = 0.0f;
    return;
  }
  values[i] = u[n];
  values[count + i] = u[(ulong)kcfg::cells + n];
  values[2ull * count + i] = u[2ull * (ulong)kcfg::cells + n];
}

// Both slab allocations live in the same CUDA primary context (one launch per
// aligned subdomain pair on multi-GPU). Each thread owns one interior coarse
// face cell and its four fine children. Donors and targets are disjoint so both
// directions see the same state as the host bridge. status[0] flags non-finite
// or non-positive values; status[1] flags whether any interface flag changed.
extern "C" __global__ void culbm_nested_face_exchange(
    float* cr, float* cu, uchar* cf, float* ct,
    float* fr, float* fu, uchar* ff, float* ft,
    const uint c_inner_nx, const uint c_inner_ny,
    const uint c_hx, const uint c_hy, const uint c_nx, const ulong cp,
    const uint f_hx, const uint f_hy, const uint f_nx, const ulong fp,
    const ulong cn, const ulong fn,
    const ulong cd, const ulong cg, const ulong fd, const ulong fg,
    uint* status) {
    const ulong p = cuda_global_thread_id(0);
    const ulong total = (ulong)c_inner_nx * (ulong)c_inner_ny;
    if (p >= total) return;
    const uint cx = (uint)(p % (ulong)c_inner_nx);
    const uint cy = (uint)(p / (ulong)c_inner_nx);
    const ulong c_xy = (ulong)(cx + c_hx) + (ulong)(cy + c_hy) * (ulong)c_nx;
    const ulong coarse = cd * cp + c_xy;
    const ulong ghost = cg * cp + c_xy;
    const ulong f_base = (ulong)(2u * cx + f_hx) + (ulong)(2u * cy + f_hy) * (ulong)f_nx;
    const ulong child[4] = {f_base, f_base + 1ull, f_base + (ulong)f_nx, f_base + (ulong)f_nx + 1ull};
    unsigned mask = 0u;
    for (unsigned j = 0u; j < 4u; ++j)
        if ((ff[fd * fp + child[j]] & CELL_SOLID) == 0u) mask |= 1u << j;
    if ((cf[coarse] & CELL_SOLID) != 0u || __popc(mask) < 3) mask = 0u;
    const uchar flags_mask = CELL_SOLID | CELL_EQUILIBRIUM
#ifdef TEMPERATURE
        | CELL_THERMAL
#endif
        ;
    const uchar open = CELL_EQUILIBRIUM
#ifdef TEMPERATURE
        | CELL_THERMAL
#endif
        ;
    const uchar new_cf = (uchar)((cf[ghost] & ~flags_mask) | (mask ? open : CELL_SOLID));
    bool flag_changed = (new_cf != cf[ghost]);
    cf[ghost] = new_cf;
    for (unsigned j = 0u; j < 4u; ++j) {
        const ulong dest = fg * fp + child[j];
        const uchar new_ff = (uchar)((ff[dest] & ~flags_mask) | (((mask >> j) & 1u) ? open : CELL_SOLID));
        if (new_ff != ff[dest]) flag_changed = true;
        ff[dest] = new_ff;
    }
    if (flag_changed) atomicOr(&status[1], 1u);
    // For a closed slot the reference writes that level's adjacent donor.
    // An open parent averages four slots, using the canonical solid for a
    // blocked child. Double accumulation reproduces the host macro arithmetic.
    for (int field = 0; field <
#ifdef TEMPERATURE
        5
#else
        4
#endif
        ; ++field) {
        float* cv = field == 0 ? cr : field == 4 ? ct : cu + (ulong)(field - 1) * cn;
        float* fv = field == 0 ? fr : field == 4 ? ft : fu + (ulong)(field - 1) * fn;
        const float parent = cv[coarse];
        if (mask && (!isfinite(parent) || (field == 0 && parent <= 0.0f)))
            atomicOr(&status[0], 1u);
        double sum = 0.0;
        for (unsigned j = 0u; j < 4u; ++j) {
            const bool flows = ((mask >> j) & 1u) != 0u;
            const float donor = fv[fd * fp + child[j]];
            if (flows && (!isfinite(donor) || (field == 0 && donor <= 0.0f)))
                atomicOr(&status[0], 1u);
            sum += flows ? (double)donor : (field == 0 || field == 4 ? 1.0 : 0.0);
            fv[fg * fp + child[j]] = flows ? parent : donor;
        }
        cv[ghost] = mask ? (float)(0.25 * sum) : parent;
    }
}

// ---------------------------------------------------------------------------
// Volumetric population-conserving 2:1 horizontal interface (Rohde et al. 2006)
// ---------------------------------------------------------------------------
// Open interface ghost cells are tagged CELL_GAS (non-solid, non-equilibrium) so
// stream_collide and update_fields skip them without triggering solid bounce-back
// or wall-model stress, while closed slots remain CELL_SOLID for building walls.

__device__ __forceinline__ ulong nested_ddf_offset(
    const uint x, const uint y, const uint z, const uint slot,
    const uint nx, const uint ny, const ulong cells) {
    return (ulong)slot * cells + (ulong)x + ((ulong)y + (ulong)z * (ulong)ny) * (ulong)nx;
}

__device__ __forceinline__ uint nested_wrap_axis(const int c, const uint n) {
    return c < 0 ? (uint)(c + (int)n) : (c >= (int)n ? (uint)(c - (int)n) : (uint)c);
}

__device__ __forceinline__ float nested_load_outgoing_mom(
    const ddf_elem_t *fi, const uint x, const uint y, const uint z,
    const uint dir, const bool next_odd,
    const uint nx, const uint ny, const uint nz, const ulong cells) {
    const uint odd = dir & 1u;
    const uint sx = odd ? nested_wrap_axis((int)x + vel_cx(dir), nx) : x;
    const uint sy = odd ? nested_wrap_axis((int)y + vel_cy(dir), ny) : y;
    const uint sz = odd ? nested_wrap_axis((int)z + vel_cz(dir), nz) : z;
    const uint slot = next_odd ? dir : (odd ? (dir + 1u) : (dir - 1u));
    return ddf_load(fi, nested_ddf_offset(sx, sy, sz, slot, nx, ny, cells));
}

__device__ __forceinline__ void nested_store_incoming_mom(
    ddf_elem_t *fi, const uint x, const uint y, const uint z,
    const uint dir, const bool next_odd, const float val,
    const uint nx, const uint ny, const uint nz, const ulong cells) {
    const uint odd = dir & 1u;
    const uint dx = odd ? x : nested_wrap_axis((int)x - vel_cx(dir), nx);
    const uint dy = odd ? y : nested_wrap_axis((int)y - vel_cy(dir), ny);
    const uint dz = odd ? z : nested_wrap_axis((int)z - vel_cz(dir), nz);
    const uint slot = next_odd ? dir : (odd ? (dir + 1u) : (dir - 1u));
    ddf_store(fi, nested_ddf_offset(dx, dy, dz, slot, nx, ny, cells), val);
}

#ifdef TEMPERATURE
__device__ __forceinline__ float nested_load_outgoing_temp(
    const ddf_elem_t *gi, const uint x, const uint y, const uint z,
    const uint dir, const bool next_odd,
    const uint nx, const uint ny, const uint nz, const ulong cells) {
    const uint odd = dir & 1u;
    const uint sx = odd ? nested_wrap_axis((int)x + vel_cx(dir), nx) : x;
    const uint sy = odd ? nested_wrap_axis((int)y + vel_cy(dir), ny) : y;
    const uint sz = odd ? nested_wrap_axis((int)z + vel_cz(dir), nz) : z;
    const uint slot = next_odd ? dir : (odd ? (dir + 1u) : (dir - 1u));
    return ddf_load(gi, nested_ddf_offset(sx, sy, sz, slot, nx, ny, cells));
}

__device__ __forceinline__ void nested_store_incoming_temp(
    ddf_elem_t *gi, const uint x, const uint y, const uint z,
    const uint dir, const bool next_odd, const float val,
    const uint nx, const uint ny, const uint nz, const ulong cells) {
    const uint odd = dir & 1u;
    const uint dx = odd ? x : nested_wrap_axis((int)x - vel_cx(dir), nx);
    const uint dy = odd ? y : nested_wrap_axis((int)y - vel_cy(dir), ny);
    const uint dz = odd ? z : nested_wrap_axis((int)z - vel_cz(dir), nz);
    const uint slot = next_odd ? dir : (odd ? (dir + 1u) : (dir - 1u));
    ddf_store(gi, nested_ddf_offset(dx, dy, dz, slot, nx, ny, cells), val);
}
#endif

extern "C" __global__ void culbm_nested_vol_prepare(
    uchar *cf, uchar *ff,
    const uint c_inner_nx, const uint c_inner_ny,
    const uint c_hx, const uint c_hy, const uint c_nx, const ulong cp,
    const uint f_hx, const uint f_hy, const uint f_nx, const ulong fp,
    const ulong cd, const ulong cg, const ulong fd, const ulong fg,
    uint *status) {
    const ulong p = cuda_global_thread_id(0);
    const ulong total = (ulong)c_inner_nx * (ulong)c_inner_ny;
    if (p >= total) return;
    const uint cx = (uint)(p % (ulong)c_inner_nx);
    const uint cy = (uint)(p / (ulong)c_inner_nx);
    const ulong c_xy = (ulong)(cx + c_hx) + (ulong)(cy + c_hy) * (ulong)c_nx;
    const ulong coarse = cd * cp + c_xy;
    const ulong ghost = cg * cp + c_xy;
    const ulong f_base = (ulong)(2u * cx + f_hx) + (ulong)(2u * cy + f_hy) * (ulong)f_nx;
    const ulong child[4] = {f_base, f_base + 1ull, f_base + (ulong)f_nx, f_base + (ulong)f_nx + 1ull};

    const uchar c_flag = cf[coarse];
    const bool c_solid = (c_flag & CELL_SOLID) != 0u;
    const bool c_eq = (c_flag & CELL_BOUNDARY_MASK) == CELL_EQUILIBRIUM;
    unsigned raw_mask = 0u;
    for (unsigned j = 0u; j < 4u; ++j) {
        if ((ff[fd * fp + child[j]] & CELL_SOLID) == 0u) raw_mask |= 1u << j;
    }
    unsigned mask = (!c_solid && (__popc(raw_mask) >= 3 || (c_eq && raw_mask != 0u))) ? raw_mask : 0u;

    const uchar clear_mask = (uchar)(CELL_BOUNDARY_MASK | CELL_PHASE_MASK | CELL_THERMAL);
    const uchar open_tag = (uchar)CELL_GAS;
    const uchar closed_tag = (uchar)CELL_SOLID;
    const uchar new_cf = (uchar)((cf[ghost] & ~clear_mask) | (mask ? open_tag : closed_tag));
    bool changed = (new_cf != cf[ghost]);
    cf[ghost] = new_cf;
    for (unsigned j = 0u; j < 4u; ++j) {
        const ulong dest = fg * fp + child[j];
        const bool child_open = ((mask >> j) & 1u) != 0u;
        const uchar new_ff = (uchar)((ff[dest] & ~clear_mask) | (child_open ? open_tag : closed_tag));
        if (new_ff != ff[dest]) changed = true;
        ff[dest] = new_ff;
    }
    if (changed) atomicOr(&status[1], 1u);
}

// Coarse -> Fine downward population explosion (BeforeFinerStep, substep 0 or 1).
extern "C" __global__ void culbm_nested_vol_push_down(
    const ddf_elem_t *c_fi, const uchar *cf,
    ddf_elem_t *f_fi, const uchar *ff,
#ifdef TEMPERATURE
    const ddf_elem_t *c_gi, ddf_elem_t *f_gi, const uint has_temp,
#endif
    const uint c_inner_nx, const uint c_inner_ny,
    const uint c_hx, const uint c_hy, const uint c_nx, const uint c_ny, const uint c_nz, const ulong c_cells,
    const uint f_hx, const uint f_hy, const uint f_nx, const uint f_ny, const uint f_nz, const ulong f_cells,
    const uint cd, const uint cg, const uint fd, const uint fg,
    const uint c_next_odd, const uint f_next_odd, const uint substep,
    uint *status) {
    const uint f_inner_nx = 2u * c_inner_nx;
    const uint f_inner_ny = 2u * c_inner_ny;
    const ulong p = cuda_global_thread_id(0);
    const ulong total = (ulong)f_inner_nx * (ulong)f_inner_ny;
    if (p >= total) return;

    const uint fx_in = (uint)(p % (ulong)f_inner_nx);
    const uint fy_in = (uint)(p / (ulong)f_inner_nx);
    const uint fx = fx_in + f_hx;
    const uint fy = fy_in + f_hy;
    const ulong fp = (ulong)f_nx * (ulong)f_ny;
    const ulong cp = (ulong)c_nx * (ulong)c_ny;
    const uchar f_flag = ff[(ulong)fd * fp + (ulong)fx + (ulong)fy * (ulong)f_nx];
    if ((f_flag & CELL_BOUNDARY_MASK) == CELL_SOLID || (f_flag & CELL_BOUNDARY_MASK) == CELL_EQUILIBRIUM) return;

    const bool c_odd = c_next_odd != 0u;
    const bool f_odd = f_next_odd != 0u;

#pragma unroll
    for (uint b = 0u; b < kcfg::q_mom_face; ++b) {
        const uint dir = mom_face_dir(5u * kcfg::q_mom_face + b); // downward (cz = -1)
        const int vx = vel_cx(dir);
        const int vy = vel_cy(dir);

        // Check if fine ghost cell in direction -c is open (CELL_GAS).
        const uint gx = nested_wrap_axis((int)fx - vx, f_nx);
        const uint gy = nested_wrap_axis((int)fy - vy, f_ny);
        const uchar g_flag = ff[(ulong)fg * fp + (ulong)gx + (ulong)gy * (ulong)f_nx];
        if ((g_flag & CELL_PHASE_MASK) != CELL_GAS) continue;

        // Rohde 2-step donor coarse cell in inner coordinates:
        //   substep 0 (1 fine step from fg):   floor((fx_in - vx) / 2)
        //   substep 1 (2 fine steps from fg+1): (fx_in / 2) - vx
        int px_in = substep == 0u ? (((int)fx_in - vx) >> 1) : ((int)(fx_in >> 1) - vx);
        int py_in = substep == 0u ? (((int)fy_in - vy) >> 1) : ((int)(fy_in >> 1) - vy);
        uint px, py;
        if (c_hx > 0u) {
            px = (uint)(px_in + (int)c_hx);
        } else {
            px = nested_wrap_axis(px_in, c_nx);
        }
        if (c_hy > 0u) {
            py = (uint)(py_in + (int)c_hy);
        } else {
            py = nested_wrap_axis(py_in, c_ny);
        }

        // If the shifted donor coarse cell is solid (e.g. substep 1 beside a building corner),
        // fall back to the direct substep-0 coarse parent covering (gx, gy).
        if ((cf[(ulong)cd * cp + (ulong)px + (ulong)py * (ulong)c_nx] & CELL_SOLID) != 0u) {
            const int fpx = ((int)fx_in - vx) >> 1;
            const int fpy = ((int)fy_in - vy) >> 1;
            px = c_hx > 0u ? (uint)(fpx + (int)c_hx) : nested_wrap_axis(fpx, c_nx);
            py = c_hy > 0u ? (uint)(fpy + (int)c_hy) : nested_wrap_axis(fpy, c_ny);
            if ((cf[(ulong)cd * cp + (ulong)px + (ulong)py * (ulong)c_nx] & CELL_SOLID) != 0u) continue;
        }

        const float val = nested_load_outgoing_mom(c_fi, px, py, cd, dir, c_odd, c_nx, c_ny, c_nz, c_cells);
        if (!isfinite(val)) atomicOr(&status[0], 1u);
        nested_store_incoming_mom(f_fi, fx, fy, fd, dir, f_odd, val, f_nx, f_ny, f_nz, f_cells);
    }

#ifdef TEMPERATURE
    if (has_temp != 0u) {
#pragma unroll
        for (uint b = 0u; b < kcfg::q_temp_face; ++b) {
            const uint dir = temp_face_dir(5u * kcfg::q_temp_face + b);
            const int vx = vel_cx(dir);
            const int vy = vel_cy(dir);
            const uint gx = nested_wrap_axis((int)fx - vx, f_nx);
            const uint gy = nested_wrap_axis((int)fy - vy, f_ny);
            if ((ff[(ulong)fg * fp + (ulong)gx + (ulong)gy * (ulong)f_nx] & CELL_PHASE_MASK) != CELL_GAS) continue;
            const int px_in = substep == 0u ? (((int)fx_in - vx) >> 1) : ((int)(fx_in >> 1) - vx);
            const int py_in = substep == 0u ? (((int)fy_in - vy) >> 1) : ((int)(fy_in >> 1) - vy);
            uint px = c_hx > 0u ? (uint)(px_in + (int)c_hx) : nested_wrap_axis(px_in, c_nx);
            uint py = c_hy > 0u ? (uint)(py_in + (int)c_hy) : nested_wrap_axis(py_in, c_ny);
            if ((cf[(ulong)cd * cp + (ulong)px + (ulong)py * (ulong)c_nx] & CELL_SOLID) != 0u) {
                const int fpx = ((int)fx_in - vx) >> 1;
                const int fpy = ((int)fy_in - vy) >> 1;
                px = c_hx > 0u ? (uint)(fpx + (int)c_hx) : nested_wrap_axis(fpx, c_nx);
                py = c_hy > 0u ? (uint)(fpy + (int)c_hy) : nested_wrap_axis(fpy, c_ny);
                if ((cf[(ulong)cd * cp + (ulong)px + (ulong)py * (ulong)c_nx] & CELL_SOLID) != 0u) continue;
            }
            const float val = nested_load_outgoing_temp(c_gi, px, py, cd, dir, c_odd, c_nx, c_ny, c_nz, c_cells);
            if (!isfinite(val)) atomicOr(&status[0], 1u);
            nested_store_incoming_temp(f_gi, fx, fy, fd, dir, f_odd, val, f_nx, f_ny, f_nz, f_cells);
        }
    }
#endif
}

// Fine -> Coarse upward population coalescence (AfterFinerStep, substep 0 and 1).
// Substep 0 reduces ONLY the 4 local fine children of each coarse cell into a
// 2D coarse staging plane (with 1-cell halo on multi-GPU). A tiny 1-cell border
// exchange syncs that coarse plane across GPUs before substep 1 gathers the
// 1-cell coarse neighbor from substep 0 plus the 1-cell fine neighbor from
// substep 1, achieving exact single-GPU / multi-GPU parity with zero 2-cell halos.

extern "C" __global__ void culbm_nested_vol_stage_s0(
    const ddf_elem_t *f_fi, const uchar *ff,
#ifdef TEMPERATURE
    const ddf_elem_t *f_gi, const uint has_temp, float *stage_temp,
#endif
    float *stage_mom, uchar *stage_closed,
    const uint c_inner_nx, const uint c_inner_ny,
    const uint c_hx, const uint c_hy, const uint c_nx, const uint c_ny,
    const uint f_hx, const uint f_hy, const uint f_nx, const uint f_ny, const uint f_nz, const ulong f_cells,
    const uint fd, const uint f_next_odd, uint *status) {
    const ulong p = cuda_global_thread_id(0);
    const ulong total = (ulong)c_inner_nx * (ulong)c_inner_ny;
    if (p >= total) return;

    const uint cx_in = (uint)(p % (ulong)c_inner_nx);
    const uint cy_in = (uint)(p / (ulong)c_inner_nx);
    const uint cx = cx_in + c_hx;
    const uint cy = cy_in + c_hy;
    const ulong cp = (ulong)c_nx * (ulong)c_ny;
    const ulong c_flat = (ulong)cx + (ulong)cy * (ulong)c_nx;
    const ulong fp = (ulong)f_nx * (ulong)f_ny;
    const uint base_fx = 2u * cx_in + f_hx;
    const uint base_fy = 2u * cy_in + f_hy;
    const bool f_odd = f_next_odd != 0u;

    uint open_mask = 0u;
    uint closed_cnt = 0u;
    for (uint ay = 0u; ay < 2u; ++ay) {
        for (uint ax = 0u; ax < 2u; ++ax) {
            const uint sfx = base_fx + ax;
            const uint sfy = base_fy + ay;
            const uchar fl = ff[(ulong)fd * fp + (ulong)sfx + (ulong)sfy * (ulong)f_nx];
            if ((fl & CELL_SOLID) == 0u) open_mask |= 1u << (ax + 2u * ay);
            else ++closed_cnt;
        }
    }
    stage_closed[c_flat] = (uchar)closed_cnt;

#pragma unroll
    for (uint b = 0u; b < kcfg::q_mom_face; ++b) {
        const uint dir = mom_face_dir(4u * kcfg::q_mom_face + b); // upward (cz = +1)
        float sum_open = 0.0f;
        for (uint ay = 0u; ay < 2u; ++ay) {
            for (uint ax = 0u; ax < 2u; ++ax) {
                if (((open_mask >> (ax + 2u * ay)) & 1u) == 0u) continue;
                const float fv = nested_load_outgoing_mom(
                    f_fi, base_fx + ax, base_fy + ay, fd, dir, f_odd, f_nx, f_ny, f_nz, f_cells);
                if (!isfinite(fv)) atomicOr(&status[0], 1u);
                sum_open += fv;
            }
        }
        stage_mom[(ulong)b * cp + c_flat] = sum_open;
    }

#ifdef TEMPERATURE
    if (has_temp != 0u) {
#pragma unroll
        for (uint b = 0u; b < kcfg::q_temp_face; ++b) {
            const uint dir = temp_face_dir(4u * kcfg::q_temp_face + b);
            float sum_open = 0.0f;
            for (uint ay = 0u; ay < 2u; ++ay) {
                for (uint ax = 0u; ax < 2u; ++ax) {
                    if (((open_mask >> (ax + 2u * ay)) & 1u) == 0u) continue;
                    const float fv = nested_load_outgoing_temp(
                        f_gi, base_fx + ax, base_fy + ay, fd, dir, f_odd, f_nx, f_ny, f_nz, f_cells);
                    if (!isfinite(fv)) atomicOr(&status[0], 1u);
                    sum_open += fv;
                }
            }
            stage_temp[(ulong)b * cp + c_flat] = sum_open;
        }
    }
#endif
}

// Pack/unpack 1-cell border of the 2D coarse stage_s0 plane along x (axis=0) or y (axis=1).
extern "C" __global__ void culbm_nested_vol_pack_stage(
    const uint axis, const uint c_nx, const uint c_ny, const uint mom_comp, const uint temp_comp,
    const float *stage_mom, const uchar *stage_closed, const float *stage_temp,
    float *buf_hi, float *buf_lo) {
    const uint idx = (uint)cuda_global_thread_id(0);
    const uint len = axis == 0u ? c_ny : c_nx;
    if (idx >= len) return;
    const ulong cp = (ulong)c_nx * (ulong)c_ny;
    const uint stride_comp = mom_comp + temp_comp + 1u;
    const ulong hi_flat = axis == 0u
        ? ((ulong)(c_nx - 2u) + (ulong)idx * (ulong)c_nx)
        : ((ulong)idx + (ulong)(c_ny - 2u) * (ulong)c_nx);
    const ulong lo_flat = axis == 0u
        ? (1ull + (ulong)idx * (ulong)c_nx)
        : ((ulong)idx + (ulong)c_nx);
    for (uint k = 0u; k < mom_comp; ++k) {
        buf_hi[(ulong)k * len + idx] = stage_mom[(ulong)k * cp + hi_flat];
        buf_lo[(ulong)k * len + idx] = stage_mom[(ulong)k * cp + lo_flat];
    }
    for (uint k = 0u; k < temp_comp; ++k) {
        buf_hi[(ulong)(mom_comp + k) * len + idx] = stage_temp[(ulong)k * cp + hi_flat];
        buf_lo[(ulong)(mom_comp + k) * len + idx] = stage_temp[(ulong)k * cp + lo_flat];
    }
    buf_hi[(ulong)(stride_comp - 1u) * len + idx] = (float)stage_closed[hi_flat];
    buf_lo[(ulong)(stride_comp - 1u) * len + idx] = (float)stage_closed[lo_flat];
}

extern "C" __global__ void culbm_nested_vol_unpack_stage(
    const uint axis, const uint c_nx, const uint c_ny, const uint mom_comp, const uint temp_comp,
    float *stage_mom, uchar *stage_closed, float *stage_temp,
    const float *buf_hi, const float *buf_lo) {
    const uint idx = (uint)cuda_global_thread_id(0);
    const uint len = axis == 0u ? c_ny : c_nx;
    if (idx >= len) return;
    const ulong cp = (ulong)c_nx * (ulong)c_ny;
    const uint stride_comp = mom_comp + temp_comp + 1u;
    const ulong hi_flat = axis == 0u
        ? ((ulong)(c_nx - 1u) + (ulong)idx * (ulong)c_nx)
        : ((ulong)idx + (ulong)(c_ny - 1u) * (ulong)c_nx);
    const ulong lo_flat = axis == 0u
        ? ((ulong)idx * (ulong)c_nx)
        : (ulong)idx;
    for (uint k = 0u; k < mom_comp; ++k) {
        stage_mom[(ulong)k * cp + hi_flat] = buf_hi[(ulong)k * len + idx];
        stage_mom[(ulong)k * cp + lo_flat] = buf_lo[(ulong)k * len + idx];
    }
    for (uint k = 0u; k < temp_comp; ++k) {
        stage_temp[(ulong)k * cp + hi_flat] = buf_hi[(ulong)(mom_comp + k) * len + idx];
        stage_temp[(ulong)k * cp + lo_flat] = buf_lo[(ulong)(mom_comp + k) * len + idx];
    }
    stage_closed[hi_flat] = (uchar)(buf_hi[(ulong)(stride_comp - 1u) * len + idx] + 0.5f);
    stage_closed[lo_flat] = (uchar)(buf_lo[(ulong)(stride_comp - 1u) * len + idx] + 0.5f);
}

extern "C" __global__ void culbm_nested_vol_apply_s1(
    ddf_elem_t *c_fi, const uchar *cf,
    const ddf_elem_t *f_fi, const uchar *ff,
#ifdef TEMPERATURE
    ddf_elem_t *c_gi, const ddf_elem_t *f_gi, const uint has_temp,
    const float *stage_temp,
#endif
    const float *stage_mom, const uchar *stage_closed,
    const uint c_inner_nx, const uint c_inner_ny,
    const uint c_hx, const uint c_hy, const uint c_nx, const uint c_ny, const uint c_nz, const ulong c_cells,
    const uint f_hx, const uint f_hy, const uint f_nx, const uint f_ny, const uint f_nz, const ulong f_cells,
    const uint cd, const uint cg, const uint fd,
    const uint c_next_odd, const uint f_next_odd, uint *status) {
    const ulong p = cuda_global_thread_id(0);
    const ulong total = (ulong)c_inner_nx * (ulong)c_inner_ny;
    if (p >= total) return;

    const uint cx_in = (uint)(p % (ulong)c_inner_nx);
    const uint cy_in = (uint)(p / (ulong)c_inner_nx);
    const uint cx = cx_in + c_hx;
    const uint cy = cy_in + c_hy;
    const ulong cp = (ulong)c_nx * (ulong)c_ny;
    const ulong fp = (ulong)f_nx * (ulong)f_ny;
    const uchar c_flag = cf[(ulong)cd * cp + (ulong)cx + (ulong)cy * (ulong)c_nx];
    if ((c_flag & CELL_BOUNDARY_MASK) == CELL_SOLID || (c_flag & CELL_BOUNDARY_MASK) == CELL_EQUILIBRIUM) return;

    const bool c_odd = c_next_odd != 0u;
    const bool f_odd = f_next_odd != 0u;
    const uint f_inner_nx = 2u * c_inner_nx;
    const uint f_inner_ny = 2u * c_inner_ny;

#pragma unroll
    for (uint b = 0u; b < kcfg::q_mom_face; ++b) {
        const uint dir = mom_face_dir(4u * kcfg::q_mom_face + b);      // upward (cz = +1)
        const uint opp = mom_face_dir(5u * kcfg::q_mom_face + b);      // opposite downward (cz = -1)
        const int vx = vel_cx(dir);
        const int vy = vel_cy(dir);

        // Coarse ghost cell (cx - vx, cy - vy, cg) must be open (CELL_GAS).
        const uint gx = c_hx > 0u ? (uint)((int)cx - vx) : nested_wrap_axis((int)cx - vx, c_nx);
        const uint gy = c_hy > 0u ? (uint)((int)cy - vy) : nested_wrap_axis((int)cy - vy, c_ny);
        if ((cf[(ulong)cg * cp + (ulong)gx + (ulong)gy * (ulong)c_nx] & CELL_PHASE_MASK) != CELL_GAS) continue;

        const ulong s0_flat = (ulong)gx + (ulong)gy * (ulong)c_nx;
        const uint s0_closed = (uint)stage_closed[s0_flat];
        float sum8 = stage_mom[(ulong)b * cp + s0_flat];

        float c_bounce = 0.0f;
        bool have_bounce = false;
        if (s0_closed > 0u) {
            c_bounce = nested_load_outgoing_mom(c_fi, cx, cy, cd, opp, c_odd, c_nx, c_ny, c_nz, c_cells);
            have_bounce = true;
            sum8 += (float)s0_closed * c_bounce;
        }

        // Substep 1 fine cells: 2*cx_in - vx + {0, 1}, 2*cy_in - vy + {0, 1}
        const int base_fx = 2 * (int)cx_in - vx;
        const int base_fy = 2 * (int)cy_in - vy;
        for (int ay = 0; ay < 2; ++ay) {
            for (int ax = 0; ax < 2; ++ax) {
                const uint sfx = f_hx > 0u
                    ? (uint)(base_fx + ax + (int)f_hx)
                    : nested_wrap_axis(base_fx + ax, f_inner_nx);
                const uint sfy = f_hy > 0u
                    ? (uint)(base_fy + ay + (int)f_hy)
                    : nested_wrap_axis(base_fy + ay, f_inner_ny);
                const uchar ff_cell = ff[(ulong)fd * fp + (ulong)sfx + (ulong)sfy * (ulong)f_nx];
                if ((ff_cell & CELL_SOLID) == 0u) {
                    const float fv = nested_load_outgoing_mom(f_fi, sfx, sfy, fd, dir, f_odd, f_nx, f_ny, f_nz, f_cells);
                    if (!isfinite(fv)) atomicOr(&status[0], 1u);
                    sum8 += fv;
                } else {
                    if (!have_bounce) {
                        c_bounce = nested_load_outgoing_mom(c_fi, cx, cy, cd, opp, c_odd, c_nx, c_ny, c_nz, c_cells);
                        have_bounce = true;
                    }
                    sum8 += c_bounce;
                }
            }
        }

        const float avg8 = 0.125f * sum8;
        if (!isfinite(avg8)) atomicOr(&status[0], 1u);
        nested_store_incoming_mom(c_fi, cx, cy, cd, dir, c_odd, avg8, c_nx, c_ny, c_nz, c_cells);
    }

#ifdef TEMPERATURE
    if (has_temp != 0u) {
#pragma unroll
        for (uint b = 0u; b < kcfg::q_temp_face; ++b) {
            const uint dir = temp_face_dir(4u * kcfg::q_temp_face + b);
            const uint opp = temp_face_dir(5u * kcfg::q_temp_face + b);
            const int vx = vel_cx(dir);
            const int vy = vel_cy(dir);
            const uint gx = c_hx > 0u ? (uint)((int)cx - vx) : nested_wrap_axis((int)cx - vx, c_nx);
            const uint gy = c_hy > 0u ? (uint)((int)cy - vy) : nested_wrap_axis((int)cy - vy, c_ny);
            if ((cf[(ulong)cg * cp + (ulong)gx + (ulong)gy * (ulong)c_nx] & CELL_PHASE_MASK) != CELL_GAS) continue;

            const ulong s0_flat = (ulong)gx + (ulong)gy * (ulong)c_nx;
            const uint s0_closed = (uint)stage_closed[s0_flat];
            float sum8 = stage_temp[(ulong)b * cp + s0_flat];
            float c_bounce = 0.0f;
            bool have_bounce = false;
            if (s0_closed > 0u) {
                c_bounce = nested_load_outgoing_temp(c_gi, cx, cy, cd, opp, c_odd, c_nx, c_ny, c_nz, c_cells);
                have_bounce = true;
                sum8 += (float)s0_closed * c_bounce;
            }
            const int base_fx = 2 * (int)cx_in - vx;
            const int base_fy = 2 * (int)cy_in - vy;
            for (int ay = 0; ay < 2; ++ay) {
                for (int ax = 0; ax < 2; ++ax) {
                    const uint sfx = f_hx > 0u
                        ? (uint)(base_fx + ax + (int)f_hx)
                        : nested_wrap_axis(base_fx + ax, f_inner_nx);
                    const uint sfy = f_hy > 0u
                        ? (uint)(base_fy + ay + (int)f_hy)
                        : nested_wrap_axis(base_fy + ay, f_inner_ny);
                    const uchar ff_cell = ff[(ulong)fd * fp + (ulong)sfx + (ulong)sfy * (ulong)f_nx];
                    if ((ff_cell & CELL_SOLID) == 0u) {
                        const float fv = nested_load_outgoing_temp(f_gi, sfx, sfy, fd, dir, f_odd, f_nx, f_ny, f_nz, f_cells);
                        if (!isfinite(fv)) atomicOr(&status[0], 1u);
                        sum8 += fv;
                    } else {
                        if (!have_bounce) {
                            c_bounce = nested_load_outgoing_temp(c_gi, cx, cy, cd, opp, c_odd, c_nx, c_ny, c_nz, c_cells);
                            have_bounce = true;
                        }
                        sum8 += c_bounce;
                    }
                }
            }
            const float avg8 = 0.125f * sum8;
            if (!isfinite(avg8)) atomicOr(&status[0], 1u);
            nested_store_incoming_temp(c_gi, cx, cy, cd, dir, c_odd, avg8, c_nx, c_ny, c_nz, c_cells);
        }
    }
#endif
}


