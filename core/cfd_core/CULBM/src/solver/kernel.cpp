#include "kernel.hpp"
#include "particle_advection_kernels.hpp"
#include "../surface_ibm_kernels.hpp"

string get_cuda_kernel_source() {
    string source;
    source.reserve(262000u);
    source += '\n';
#include "culbm_lbm_kernels.inc"
#include "culbm_runtime_kernels.inc"
    source += culbm::particle_advection_cuda_source();
    source += culbm::surface_ibm::surface_ibm_cuda_source();
    return source;
}
