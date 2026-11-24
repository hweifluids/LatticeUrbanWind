#pragma once

// CUDA driver and NVRTC entry points of the CULBM GPU layer.

#include <cuda.h>
#include <nvrtc.h>

#include <cstdint>
#include <string>

namespace culbm::gpu {

// Threads per block every CULBM kernel is written for (cuda_threads_per_block in
// the device prelude). Individual launches may override it.
constexpr unsigned default_block_size = 64u;

// Readable name and description of a driver status code.
std::string describe(CUresult result);

// Abort the run with a message naming the failed call.
void check(CUresult result, const char* what);
void check(nvrtcResult result, const char* what);

// Bind `context` to the calling thread.
void make_current(CUcontext context);

// Initialise the driver API once per process.
void ensure_driver();

} // namespace culbm::gpu
