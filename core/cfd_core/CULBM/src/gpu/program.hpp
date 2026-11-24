#pragma once

// Runtime compilation of the CULBM kernels with NVRTC.

#include "device.hpp"

namespace culbm::gpu {

// Compile the device prelude plus `native_source` for `properties` and load the
// result into the current context. The PTX is cached next to the executable,
// keyed by a hash of the source and the compiler options (CULBM_PTX_CACHE=0
// disables the cache). CULBM_DUMP_SOURCE / CULBM_DUMP_PTX write the compiled
// source / PTX next to the executable.
CUmodule load_kernel_module(const DeviceProperties& properties, const string& native_source);

} // namespace culbm::gpu
