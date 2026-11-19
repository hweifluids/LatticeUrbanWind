

#ifdef TEMPERATURE
// Macro-field mirror used for diagnostics on the GEA atmospheric validation
// walls.  Population-level impermeable/free-slip behavior is provided by the
// FREE_SLIP_BOUNDARIES path in stream_collide.  The production case/deck target
// does not define that extension or call this kernel.
extern "C" __global__ void culbm_update_free_slip_boundaries(float *rho, float *u, float *T, const uchar *flags,
                                                             const uint face_mask) {
  const uxx n = cuda_global_thread_id(0);
  if (n >= (uxx)def_N || is_halo(n))
    return;
  if ((flags[n] & TYPE_BO) != TYPE_S)
    return;

  const uint3 xyz = coordinates(n);
  const int x_global = (int)xyz.x + def_Ox;
  const int y_global = (int)xyz.y + def_Oy;
  const int z_global = (int)xyz.z + def_Oz;

  const bool west = (face_mask & 1u) != 0u && x_global == 0;
  const bool east = (face_mask & 2u) != 0u && x_global == (int)def_Nx_global - 1;
  const bool south = (face_mask & 4u) != 0u && y_global == 0;
  const bool north = (face_mask & 8u) != 0u && y_global == (int)def_Ny_global - 1;
  const bool bottom = (face_mask & 16u) != 0u && z_global == 0;
  const bool top = (face_mask & 32u) != 0u && z_global == (int)def_Nz_global - 1;
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
  if (interior.x >= def_Nx || interior.y >= def_Ny || interior.z >= def_Nz)
    return;

  const uxx source_index = index(interior);
  rho[n] = rho[source_index];
  u[n] = (west || east) ? 0.0f : u[source_index];
  u[(ulong)def_N + (ulong)n] = (south || north) ? 0.0f : u[(ulong)def_N + (ulong)source_index];
  u[2ull * (ulong)def_N + (ulong)n] = (bottom || top) ? 0.0f : u[2ull * (ulong)def_N + (ulong)source_index];
  // TYPE_S without TYPE_T uses the production scalar bounce-back path.
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
  if (n >= (ulong)def_N)
    return;

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

extern "C" __global__ void culbm_gather_probe_velocity(const float *u, const ulong *indices, float *values,
                                                       const ulong count) {
  const ulong i = cuda_global_thread_id(0);
  if (i >= count)
    return;
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

// Both slab allocations live in the same CUDA primary context. Each thread
// owns one coarse face and its four children. Donors and targets are disjoint
// so both directions see exactly the same state as the host bridge.
extern "C" __global__ void culbm_nested_face_exchange(
    float* cr, float* cu, uchar* cf, float* ct,
    float* fr, float* fu, uchar* ff, float* ft,
    const uint nx, const ulong cp, const ulong fp, const ulong cn, const ulong fn,
    const ulong cd, const ulong cg, const ulong fd, const ulong fg, uint* invalid) {
    const ulong p = cuda_global_thread_id(0);
    if (p >= cp) return;
    const ulong coarse = cd * cp + p, ghost = cg * cp + p;
    const ulong first = (p / nx) * (4ull * nx) + 2ull * (p % nx);
    const ulong child[4] = {first, first+1ull, first+2ull*nx, first+2ull*nx+1ull};
    unsigned mask = 0;
    for (unsigned j=0; j<4; ++j)
        if ((ff[fd*fp+child[j]] & TYPE_S) == 0) mask |= 1u << j;
    if ((cf[coarse] & TYPE_S) || __popc(mask) < 3) mask = 0;
    const uchar flags_mask = TYPE_S | TYPE_E
#ifdef TEMPERATURE
        | TYPE_T
#endif
        ;
    const uchar open = TYPE_E
#ifdef TEMPERATURE
        | TYPE_T
#endif
        ;
    cf[ghost] = (cf[ghost] & ~flags_mask) | (mask ? open : TYPE_S);
    for (unsigned j=0; j<4; ++j) {
        const ulong dest = fg*fp+child[j];
        ff[dest] = (ff[dest] & ~flags_mask) | ((mask >> j) & 1u ? open : TYPE_S);
    }
    // For a closed slot the reference writes that level's adjacent donor.
    // An open parent averages four slots, using the canonical solid for a
    // blocked child. Double accumulation reproduces the host macro arithmetic.
    for (int field=0; field<
#ifdef TEMPERATURE
        5
#else
        4
#endif
        ; ++field) {
        float* cv = field==0 ? cr : field==4 ? ct : cu+(ulong)(field-1)*cn;
        float* fv = field==0 ? fr : field==4 ? ft : fu+(ulong)(field-1)*fn;
        const float parent = cv[coarse];
        if (mask && (!isfinite(parent) || (field==0 && parent<=0.0f)))
            atomicOr(invalid,1u);
        double sum = 0.0;
        for (unsigned j=0; j<4; ++j) {
            const bool flows = ((mask >> j) & 1u)!=0u;
            const float donor = fv[fd*fp+child[j]];
            if (flows && (!isfinite(donor) || (field==0 && donor<=0.0f)))
                atomicOr(invalid,1u);
            sum += flows ? (double)donor : (field==0 || field==4 ? 1.0 : 0.0);
            fv[fg*fp+child[j]] = flows ? parent : donor;
        }
        cv[ghost] = mask ? (float)(0.25 * sum) : parent;
    }
}

