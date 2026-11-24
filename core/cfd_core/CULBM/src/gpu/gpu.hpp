#pragma once

// CULBM GPU layer (umbrella header), namespace culbm::gpu.
//
//   driver.hpp     status checks, context binding, driver initialisation
//   device.hpp     DeviceProperties, Device (context, streams, module), halo protocol
//   program.hpp    NVRTC compilation and the PTX cache
//   buffer.hpp     Buffer<T>, HostMirror<T>, MirroredBuffer<T>, copy_halo
//   launch.hpp     Launch: kernel, extent, block size and argument slots
//   selection.hpp  GPU choice for a run and peer-aware domain placement
//   vram.hpp       allocation bookkeeping (MiB rounding, planned tallies)

#include "driver.hpp"
#include "device.hpp"
#include "program.hpp"
#include "buffer.hpp"
#include "launch.hpp"
#include "selection.hpp"
#include "vram.hpp"

// Short name for solver code outside namespace culbm.
namespace gpu = culbm::gpu;
