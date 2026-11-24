#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

std::string read_normalized(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if(!input) throw std::runtime_error(std::string("cannot open ")+path);
    const std::string bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    std::string normalized;
    normalized.reserve(bytes.size());
    for(size_t i=0u; i<bytes.size(); ++i) {
        if(bytes[i]=='\r') {
            if(i+1u<bytes.size()&&bytes[i+1u]=='\n') ++i;
            normalized.push_back('\n');
        } else {
            normalized.push_back(bytes[i]);
        }
    }
    return normalized;
}

std::string embedded_cuda_prelude() {
    std::string source;
#include "culbm_cuda_prelude.inc"
    return source;
}

std::string embedded_lbm_kernels() {
    std::string source;
#include "culbm_lbm_kernels.inc"
    return source;
}

std::string embedded_runtime_kernels() {
    std::string source;
#include "culbm_runtime_kernels.inc"
    return source;
}

std::string embedded_particle_advection_kernels() {
    std::string source;
#include "culbm_particle_advection_kernels.inc"
    return source;
}

std::string embedded_surface_ibm_kernels() {
    std::string source;
#include "culbm_surface_ibm_kernels.inc"
    return source;
}

bool compare_source(
    const char* label,
    const char* path,
    const std::string& embedded) {
    const std::string canonical = read_normalized(path);
    if(canonical==embedded) return true;
    size_t mismatch = 0u;
    const size_t shared = std::min(canonical.size(), embedded.size());
    while(mismatch<shared&&canonical[mismatch]==embedded[mismatch]) ++mismatch;
    std::cerr << label << " differs at byte " << mismatch
              << " (canonical=" << canonical.size()
              << ", embedded=" << embedded.size() << ")\n";
    return false;
}

} // namespace

int main(int argc, char** argv) {
    if(argc!=6) {
        std::cerr << "expected five canonical CUDA source paths\n";
        return 2;
    }
    try {
        bool ok = true;
        ok &= compare_source("CUDA prelude", argv[1], embedded_cuda_prelude());
        ok &= compare_source("LBM kernels", argv[2], embedded_lbm_kernels());
        ok &= compare_source(
            "runtime kernels", argv[3], embedded_runtime_kernels());
        ok &= compare_source(
            "particle-advection kernels",
            argv[4],
            embedded_particle_advection_kernels());
        ok &= compare_source(
            "SurfaceIBM kernels", argv[5], embedded_surface_ibm_kernels());
        if(!ok) return 1;
        std::cout << "All embedded CUDA sources match their canonical files.\n";
        return 0;
    } catch(const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 2;
    }
}
