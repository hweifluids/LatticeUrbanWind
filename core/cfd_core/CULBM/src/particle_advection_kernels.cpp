#include "particle_advection_kernels.hpp"

namespace culbm {

std::string particle_advection_cuda_source() {
    std::string source;
    source.reserve(25929u);
#include "culbm_particle_advection_kernels.inc"
    return source;
}

} // namespace culbm
