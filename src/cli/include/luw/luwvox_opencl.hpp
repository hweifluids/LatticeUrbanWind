#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace luw::cli {

struct OpenClKrigingRequest {
    // Zero-based index in the GPU-only list obtained by flattening each
    // OpenCL platform's devices in platform/device enumeration order.
    int deviceIndex = 0;
    std::size_t queryCount = 0;
    std::size_t neighborCount = 0;
    // Row-major [query][neighbor][xy], with a per-row local origin already
    // removed to retain float32 precision for large projected coordinates.
    std::vector<float> localPointsXY;
    std::vector<float> localElevations;
    std::vector<float> localDistances;
    float range = 1.0F;
    float partialSill = 1.0F;
    float nugget = 0.02F;
    float fallbackPower = 2.0F;
};

struct OpenClKrigingResult {
    std::vector<float> predictions;
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
};

struct OpenClKrigingBackendInfo {
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
};

using OpenClVoxelBackendInfo = OpenClKrigingBackendInfo;

struct OpenClIdwRequest {
    int deviceIndex = 0;
    std::size_t queryCount = 0;
    std::size_t neighborCount = 0;
    // Row-major [query][neighbor]. Neighbor order is preserved by the kernel.
    std::vector<float> elevations;
    std::vector<float> distances;
    float power = 2.0F;
};

struct OpenClIdwResult {
    std::vector<float> predictions;
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
};

struct OpenClGaussianRequest {
    int deviceIndex = 0;
    std::size_t width = 0;
    std::size_t height = 0;
    // Row-major [height][width].
    std::vector<float> values;
    // Keep the host-side value in double so the kernel radius uses the same
    // boundary rule as the CPU implementation.
    double sigma = 0.0;
};

struct OpenClGaussianResult {
    std::vector<float> values;
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
};

enum class OpenClTerrainApproach {
    Idw,
    Kriging,
};

struct OpenClTerrainInterpolationRequest {
    int deviceIndex = 0;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t neighborCount = 0;
    double xMin = 0.0;
    double xMax = 0.0;
    double yMin = 0.0;
    double yMax = 0.0;
    // Row-major [sample][xy]. Double precision is intentional: neighbor
    // ordering follows the CPU (distance squared, original sample index)
    // contract even for large projected coordinates.
    std::vector<double> pointsXY;
    std::vector<double> elevations;
    OpenClTerrainApproach approach = OpenClTerrainApproach::Idw;
    double idwPower = 2.0;
    double smoothSigma = 0.0;
    // Diagnostic/test-only. Production leaves this false so no complete host
    // neighbor table is retained. Kriging still streams distance chunks to
    // compute its exact CPU-compatible median/range model.
    bool includeNeighborData = false;
};

struct OpenClTerrainInterpolationResult {
    // The unsmoothed interpolation and the final (possibly smoothed) surface.
    std::vector<float> rawValues;
    std::vector<float> values;
    std::vector<std::uint64_t> neighborIndices;
    std::vector<double> neighborSquaredDistances;
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
    bool usedRegularGridFastPath = false;
};

struct OpenClTerrainMeshRequest {
    int deviceIndex = 0;
    std::size_t width = 0;
    std::size_t height = 0;
    double xMin = 0.0;
    double xMax = 0.0;
    double yMin = 0.0;
    double yMax = 0.0;
    double baseHeight = 0.0;
    // Row-major [height][width].
    std::vector<double> elevations;
};

struct OpenClTerrainMeshResult {
    // Row-major triangle order matching terrainTriangles(); nine doubles per
    // triangle (a.xyz, b.xyz, c.xyz).
    std::vector<double> triangleVertices;
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
};

class OpenClKrigingError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using OpenClVoxelError = OpenClKrigingError;

// Compiles and caches the OpenCL program on first use, then executes in
// allocation-bounded chunks. The unified voxel program requires FP64 so DEM
// distance ordering matches the CPU (distance squared, source index) contract.
// Throws OpenClKrigingError when no GPU backend is available or when the kernel
// cannot run; it never selects a CPU backend.
OpenClVoxelBackendInfo requireNativeOpenClVoxelBackend(int deviceIndex = 0);
bool nativeOpenClVoxelRequiresWarmup(int deviceIndex = 0) noexcept;
OpenClKrigingBackendInfo requireNativeOpenClKrigingBackend(int deviceIndex = 0);
OpenClKrigingResult runNativeOpenClKriging(const OpenClKrigingRequest& request);
OpenClIdwResult runNativeOpenClIdw(const OpenClIdwRequest& request);
OpenClGaussianResult runNativeOpenClGaussian(
    const OpenClGaussianRequest& request);
OpenClTerrainInterpolationResult runNativeOpenClTerrainInterpolation(
    const OpenClTerrainInterpolationRequest& request);
OpenClTerrainMeshResult runNativeOpenClTerrainMesh(
    const OpenClTerrainMeshRequest& request);
bool nativeOpenClVoxelFp64CapabilityIsSufficient(
    std::uint64_t capabilityBits) noexcept;
bool nativeOpenClKrigingRequiresWarmup(int deviceIndex = 0) noexcept;

} // namespace luw::cli
