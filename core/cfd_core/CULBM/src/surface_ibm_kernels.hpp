#pragma once

#include <string>

namespace culbm::surface_ibm {

// Native CUDA kernels appended to the common NVRTC translation unit.  The
// body is guarded by SURFACE_IBM, which is injected per LBM domain only when
// the optional runtime is enabled before solver construction.
std::string surface_ibm_cuda_source();

} // namespace culbm::surface_ibm
