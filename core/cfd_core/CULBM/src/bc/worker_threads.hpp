#pragma once

// Worker threads of the host-side boundary construction.

namespace culbm::bc {

// Active processors of all processor groups (Windows), else the hardware
// concurrency; 4 when unknown.
unsigned available_worker_threads();

} // namespace culbm::bc
