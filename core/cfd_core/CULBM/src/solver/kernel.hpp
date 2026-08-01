#pragma once

#include "utilities.hpp"

// Common native CUDA body. Per-domain preprocessor definitions are prepended
// by LBM_Domain before this source is compiled with NVRTC.
string get_cuda_kernel_source();
