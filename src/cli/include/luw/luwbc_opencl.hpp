#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace luw::cli {

// Fused U/V/W[/T] horizontal interpolation.  Values are field-major:
//   input [(component * layerCount + layer) * sourceLayerSize + sourcePoint]
//   output[(component * layerCount + layer) * targetPointCount + targetPoint]
// Each target uses either one nearest-neighbor entry or three barycentric
// entries.  Only entries below counts[target] are consumed.
struct OpenClLuwbcHorizontalRequest {
    // Zero-based index in the GPU-only list obtained by flattening each
    // OpenCL platform's devices in platform/device enumeration order.
    int deviceIndex = 0;
    std::size_t sourceLayerSize = 0;
    std::size_t layerCount = 0;
    std::size_t targetPointCount = 0;
    std::size_t componentCount = 3;
    std::vector<float> values;
    // Target-major [target][3].
    std::vector<std::uint32_t> indices;
    // Retained as double because the established CPU path computes
    // barycentric weights and accumulation in double before the final float
    // store.
    std::vector<double> weights;
    // Exactly 1 (nearest) or 3 (barycentric) for every target.
    std::vector<std::uint32_t> counts;
};

// Fused U/V/W[/T] vertical linear interpolation.  Values are field-major:
//   input [(component * sourceLevels.size() + level) * pointCount + point]
//   output[(component * targetLevels.size() + level) * pointCount + point]
// sourceLevels must be finite and strictly increasing.  Targets outside the
// source interval are clamped to the first/last source plane.
struct OpenClLuwbcVerticalRequest {
    int deviceIndex = 0;
    std::size_t pointCount = 0;
    std::size_t componentCount = 3;
    std::vector<double> sourceLevels;
    std::vector<double> targetLevels;
    std::vector<float> values;
};

// Batched DEM inverse-distance interpolation.  The host retains ownership of
// spatial-neighbor discovery so that VTK's distance-tie ordering remains the
// single source of truth.  neighborIndices is target-major with exactly
// neighborCount entries per target, in the order returned by the CPU locator.
// The OpenCL kernel preserves the established luwbc rule
// max(euclideanDistance, 1e-10), including coincident/exact-hit points.
struct OpenClLuwbcDemIdwRequest {
    int deviceIndex = 0;
    std::size_t neighborCount = 0;
    std::vector<double> sourceX;
    std::vector<double> sourceY;
    std::vector<double> sourceValues;
    // Paired target coordinates, not Cartesian coordinate axes.
    std::vector<double> targetX;
    std::vector<double> targetY;
    std::vector<std::uint32_t> neighborIndices;
};

// Batched terrain-first moving-least-squares plan construction.  Neighbor
// search remains on the host for deterministic VTK tie ordering; the GPU
// computes the Wendland-C2 kernel and the fixed 1x1, 3x3, or 6x6 regularized
// system for every target.  Returned weights are target-major and float, as in
// TerrainFirstPlan's established CPU storage contract.  Production terrain-
// first plans use min(24, validSourcePointCount) neighbors.  A degenerate
// system uses normalized Wendland weights on-device, matching the CPU numeric
// rule; it never switches execution to the CPU backend.
struct OpenClLuwbcMlsRequest {
    int deviceIndex = 0;
    std::size_t neighborCount = 0;
    std::vector<double> sourceX;
    std::vector<double> sourceY;
    // Paired target coordinates, not Cartesian coordinate axes.
    std::vector<double> targetX;
    std::vector<double> targetY;
    std::vector<std::uint32_t> neighborIndices;
};

struct OpenClLuwbcResult {
    std::vector<float> values;
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
};

struct OpenClLuwbcDoubleResult {
    std::vector<double> values;
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
};

struct OpenClLuwbcBackendInfo {
    std::string device;
    int deviceIndex = 0;
    bool compiledThisCall = false;
};

class OpenClLuwbcError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Creates the context, queue, and all kernels on first use of each device.
// All failures are explicit; these functions never execute a CPU fallback.
OpenClLuwbcBackendInfo requireNativeOpenClLuwbcBackend(int deviceIndex = 0);
OpenClLuwbcResult runNativeOpenClLuwbcHorizontal(
    const OpenClLuwbcHorizontalRequest& request);
OpenClLuwbcResult runNativeOpenClLuwbcVertical(
    const OpenClLuwbcVerticalRequest& request);
OpenClLuwbcDoubleResult runNativeOpenClLuwbcDemIdw(
    const OpenClLuwbcDemIdwRequest& request);
OpenClLuwbcResult runNativeOpenClLuwbcMlsWeights(
    const OpenClLuwbcMlsRequest& request);
bool nativeOpenClLuwbcRequiresWarmup(int deviceIndex = 0) noexcept;

} // namespace luw::cli
