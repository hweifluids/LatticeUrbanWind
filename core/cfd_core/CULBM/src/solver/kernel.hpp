#pragma once

#include "../core/core.hpp"

// Common native CUDA body. Per-domain preprocessor definitions are prepended
// by culbm::solver::kernel_source() before NVRTC compiles it.
string get_cuda_kernel_source();
