#include "surface_ibm_kernels.hpp"

namespace culbm::surface_ibm {

std::string surface_ibm_cuda_source() {
    std::string source;
    source.reserve(20657u);
#include "culbm_surface_ibm_kernels.inc"
    return source;
}

} // namespace culbm::surface_ibm
