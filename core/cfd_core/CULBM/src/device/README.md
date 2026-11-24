# CULBM native CUDA sources

This directory is the source of truth for device code compiled by NVRTC.
CMake embeds these files mechanically into generated fragments under the build
directory; the generated `.inc` files are not source files and must not be
edited or committed.

The runtime assembles the fragments in this order:

1. `cuda_prelude.cuh` supplies CUDA headers and common device helpers.
2. Per-subdomain definitions from `culbm::solver::kernel_definitions()`
   (`src/solver/kernel_set.cpp`) specialize grid
   dimensions, collision models, precision and enabled features.
3. `lbm_kernels.cu` supplies the core lattice and utility kernels.
4. `runtime_kernels.cu` supplies CULBM diagnostics and boundary helpers.
5. `particle_advection_kernels.cu` and `surface_ibm_kernels.cu` supply their
   optional subsystems.

These files are CUDA C++ even though CMake does not compile them with `nvcc`.
At runtime, CULBM concatenates them and invokes NVRTC for the selected GPU
architecture. Run `CULBM.CudaSourceEmbedding` after editing them to verify that
the embedded bytes still match the canonical files.
