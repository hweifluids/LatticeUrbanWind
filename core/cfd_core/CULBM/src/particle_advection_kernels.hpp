#pragma once

#include <string>

namespace culbm {

enum class ParticleStatus : unsigned char {
    unused = 0u,
    pending = 1u,
    active = 2u,
    expired = 3u,
    escaped = 4u,
    solid = 5u,
    invalid = 6u
};

// Native CUDA source shared by the production NVRTC module and the focused
// GPU regression test.  Keeping it separate from the generated solver source
// makes the advection kernels independently testable.
std::string particle_advection_cuda_source();

} // namespace culbm
