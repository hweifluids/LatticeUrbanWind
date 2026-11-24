#include "luw/luwvox_opencl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <utility>

#ifndef LUW_HAS_OPENCL
#define LUW_HAS_OPENCL 0
#endif

#if LUW_HAS_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#endif

namespace luw::cli {

#if LUW_HAS_OPENCL
namespace {

std::string infoString(cl_device_id device, cl_device_info property) {
    std::size_t length = 0U;
    if (clGetDeviceInfo(device, property, 0, nullptr, &length) != CL_SUCCESS ||
        length == 0U) {
        return "unknown";
    }
    std::string value(length, '\0');
    if (clGetDeviceInfo(device, property, length, value.data(), nullptr) !=
        CL_SUCCESS) {
        return "unknown";
    }
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

void requireSuccess(cl_int status, std::string_view operation) {
    if (status != CL_SUCCESS) {
        throw OpenClKrigingError(std::string(operation) +
                                 " failed (OpenCL error " +
                                 std::to_string(status) + ")");
    }
}

struct EnumeratedGpuDevice {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
};

std::vector<EnumeratedGpuDevice> enumerateGpuDevices() {
    cl_uint platformCount = 0U;
    cl_int status = clGetPlatformIDs(0, nullptr, &platformCount);
    if (status != CL_SUCCESS || platformCount == 0U) return {};

    std::vector<cl_platform_id> platforms(platformCount);
    requireSuccess(clGetPlatformIDs(platformCount, platforms.data(), nullptr),
                   "enumerating OpenCL platforms");

    std::vector<EnumeratedGpuDevice> result;
    for (const cl_platform_id platform : platforms) {
        cl_uint deviceCount = 0U;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
                                &deviceCount);
        if (status == CL_DEVICE_NOT_FOUND ||
            (status == CL_SUCCESS && deviceCount == 0U)) {
            continue;
        }
        requireSuccess(status, "counting OpenCL GPU devices");

        std::vector<cl_device_id> devices(deviceCount);
        requireSuccess(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                                      deviceCount, devices.data(), nullptr),
                       "enumerating OpenCL GPU devices");
        for (const cl_device_id device : devices) {
            result.push_back({platform, device});
        }
    }
    return result;
}

EnumeratedGpuDevice selectGpuDevice(int requestedIndex) {
    const std::vector<EnumeratedGpuDevice> devices = enumerateGpuDevices();
    if (requestedIndex < 0 ||
        static_cast<std::size_t>(requestedIndex) >= devices.size()) {
        throw OpenClKrigingError(
            "requested OpenCL GPU device index " +
            std::to_string(requestedIndex) + ", but only " +
            std::to_string(devices.size()) +
            " OpenCL GPU device(s) are available");
    }
    return devices[static_cast<std::size_t>(requestedIndex)];
}

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           std::string_view description) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw OpenClKrigingError(std::string(description) + " size overflow");
    }
    return left * right;
}

std::size_t checkedSum(std::size_t left, std::size_t right,
                       std::string_view description) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw OpenClKrigingError(std::string(description) + " size overflow");
    }
    return left + right;
}

struct RegularGridLayout {
    bool available = false;
    std::size_t width = 0U;
    std::size_t height = 0U;
    double x0 = 0.0;
    double y0 = 0.0;
    double dx = 0.0;
    double dy = 0.0;
};

struct SpatialBins {
    std::size_t width = 1U;
    std::size_t height = 1U;
    double minX = 0.0;
    double maxX = 0.0;
    double minY = 0.0;
    double maxY = 0.0;
    std::vector<cl_ulong> offsets;
    std::vector<cl_ulong> indices;
};

RegularGridLayout detectRegularGridLayout(
    const std::vector<double>& pointsXY) {
    const std::size_t pointCount = pointsXY.size() / 2U;
    if (pointCount == 0U || pointsXY.size() != pointCount * 2U) return {};
    const double x0 = pointsXY[0];
    const double y0 = pointsXY[1];
    std::size_t width = 1U;
    while (width < pointCount && pointsXY[width * 2U + 1U] == y0) {
        ++width;
    }
    if (width == 0U || pointCount % width != 0U) return {};
    const std::size_t height = pointCount / width;
    const double dx = width > 1U ? pointsXY[2] - x0 : 0.0;
    const double dy = height > 1U ? pointsXY[width * 2U + 1U] - y0 : 0.0;
    if ((width > 1U && (!std::isfinite(dx) || dx == 0.0)) ||
        (height > 1U && (!std::isfinite(dy) || dy == 0.0))) {
        return {};
    }
    for (std::size_t row = 0U; row < height; ++row) {
        const double expectedY = y0 + static_cast<double>(row) * dy;
        for (std::size_t column = 0U; column < width; ++column) {
            const std::size_t index = row * width + column;
            const double expectedX = x0 + static_cast<double>(column) * dx;
            const double x = pointsXY[index * 2U];
            const double y = pointsXY[index * 2U + 1U];
            if (!std::isfinite(x) || !std::isfinite(y) ||
                x != expectedX || y != expectedY) {
                return {};
            }
        }
    }
    return {true, width, height, x0, y0, dx, dy};
}

SpatialBins buildSpatialBins(const std::vector<double>& pointsXY) {
    const std::size_t pointCount = pointsXY.size() / 2U;
    if (pointCount == 0U || pointsXY.size() != pointCount * 2U) {
        throw OpenClKrigingError(
            "cannot build OpenCL spatial bins for an empty DEM");
    }
    SpatialBins bins;
    bins.minX = bins.maxX = pointsXY[0];
    bins.minY = bins.maxY = pointsXY[1];
    for (std::size_t index = 0U; index < pointCount; ++index) {
        bins.minX = std::min(bins.minX, pointsXY[index * 2U]);
        bins.maxX = std::max(bins.maxX, pointsXY[index * 2U]);
        bins.minY = std::min(bins.minY, pointsXY[index * 2U + 1U]);
        bins.maxY = std::max(bins.maxY, pointsXY[index * 2U + 1U]);
    }
    const double spanX = bins.maxX - bins.minX;
    const double spanY = bins.maxY - bins.minY;
    const std::size_t targetBins = std::max<std::size_t>(
        1U, checkedSum(pointCount, 7U, "OpenCL spatial-bin target") / 8U);
    constexpr std::size_t maximumDimension = 4096U;
    if (spanX > 0.0 && spanY > 0.0) {
        const double aspect = spanX / spanY;
        const double estimatedWidth =
            std::sqrt(static_cast<double>(targetBins) * aspect);
        bins.width = !std::isfinite(estimatedWidth) ||
                             estimatedWidth >=
                                 static_cast<double>(maximumDimension)
            ? maximumDimension
            : std::max<std::size_t>(
                  1U, static_cast<std::size_t>(std::ceil(estimatedWidth)));
        bins.height = std::clamp<std::size_t>(
            checkedSum(targetBins, bins.width - 1U,
                       "OpenCL spatial-bin height") /
                bins.width,
            1U, maximumDimension);
    } else if (spanX > 0.0) {
        bins.width = std::min(targetBins, maximumDimension);
    } else if (spanY > 0.0) {
        bins.height = std::min(targetBins, maximumDimension);
    }
    const std::size_t binCount = checkedProduct(
        bins.width, bins.height, "OpenCL spatial bins");
    bins.offsets.assign(checkedSum(binCount, 1U, "OpenCL spatial-bin offsets"),
                        0U);
    const auto binForPoint = [&](std::size_t pointIndex) {
        std::size_t column = 0U;
        std::size_t row = 0U;
        if (bins.width > 1U) {
            const double normalized =
                (pointsXY[pointIndex * 2U] - bins.minX) / spanX;
            column = std::min(
                bins.width - 1U,
                static_cast<std::size_t>(normalized *
                                         static_cast<double>(bins.width)));
        }
        if (bins.height > 1U) {
            const double normalized =
                (pointsXY[pointIndex * 2U + 1U] - bins.minY) / spanY;
            row = std::min(
                bins.height - 1U,
                static_cast<std::size_t>(normalized *
                                         static_cast<double>(bins.height)));
        }
        return row * bins.width + column;
    };
    for (std::size_t point = 0U; point < pointCount; ++point) {
        ++bins.offsets[binForPoint(point) + 1U];
    }
    std::partial_sum(bins.offsets.begin(), bins.offsets.end(),
                     bins.offsets.begin());
    std::vector<cl_ulong> cursors = bins.offsets;
    bins.indices.resize(pointCount);
    for (std::size_t point = 0U; point < pointCount; ++point) {
        const std::size_t bin = binForPoint(point);
        bins.indices[static_cast<std::size_t>(cursors[bin]++)] =
            static_cast<cl_ulong>(point);
    }
    return bins;
}

struct KrigingParameters {
    double range = 1.0;
    double partialSill = 1.0;
    double nugget = 0.02;
};

KrigingParameters estimateKrigingParameters(
    const std::vector<double>& pointsXY,
    const std::vector<double>& elevations,
    std::vector<double> positiveDistances) {
    const double mean =
        std::accumulate(elevations.begin(), elevations.end(), 0.0) /
        static_cast<double>(std::max<std::size_t>(elevations.size(), 1U));
    double variance = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const double elevation : elevations) {
        const double delta = elevation - mean;
        variance += delta * delta;
        minimum = std::min(minimum, elevation);
        maximum = std::max(maximum, elevation);
    }
    variance /=
        static_cast<double>(std::max<std::size_t>(elevations.size(), 1U));
    if (!std::isfinite(variance) || variance <= 1.0e-12) {
        variance = std::max(std::pow((maximum - minimum) * 0.25, 2.0), 1.0);
    }

    double range = 1.0;
    if (!positiveDistances.empty()) {
        const std::size_t middle = positiveDistances.size() / 2U;
        std::nth_element(positiveDistances.begin(),
                         positiveDistances.begin() +
                             static_cast<std::ptrdiff_t>(middle),
                         positiveDistances.end());
        double median = positiveDistances[middle];
        if (positiveDistances.size() % 2U == 0U) {
            const auto lower = std::max_element(
                positiveDistances.begin(),
                positiveDistances.begin() +
                    static_cast<std::ptrdiff_t>(middle));
            median = (*lower + median) * 0.5;
        }
        range = std::max(median * 3.0, 1.0);
    } else if (!pointsXY.empty()) {
        double minX = pointsXY[0], maxX = pointsXY[0];
        double minY = pointsXY[1], maxY = pointsXY[1];
        for (std::size_t index = 0U; index < elevations.size(); ++index) {
            minX = std::min(minX, pointsXY[index * 2U]);
            maxX = std::max(maxX, pointsXY[index * 2U]);
            minY = std::min(minY, pointsXY[index * 2U + 1U]);
            maxY = std::max(maxY, pointsXY[index * 2U + 1U]);
        }
        range = std::max(std::hypot(maxX - minX, maxY - minY) / 6.0, 1.0);
    }
    const double nugget = std::max(variance * 0.02, 1.0e-9);
    return {range, std::max(variance - nugget, 1.0e-9), nugget};
}

class Backend final {
public:
    explicit Backend(int deviceIndex) : deviceIndex_(deviceIndex) {
        const EnumeratedGpuDevice selected = selectGpuDevice(deviceIndex);
        platform_ = selected.platform;
        device_ = selected.device;
        try {
            const cl_context_properties contextProperties[] = {
                CL_CONTEXT_PLATFORM,
                reinterpret_cast<cl_context_properties>(platform_), 0};
            cl_int status = CL_SUCCESS;
            context_ = clCreateContext(contextProperties, 1, &device_, nullptr,
                                       nullptr, &status);
            requireSuccess(status, "creating OpenCL context");
            queue_ = clCreateCommandQueue(context_, device_, 0, &status);
            requireSuccess(status, "creating OpenCL command queue");
            cl_device_fp_config doubleConfig = 0;
            requireSuccess(clGetDeviceInfo(device_, CL_DEVICE_DOUBLE_FP_CONFIG,
                                           sizeof(doubleConfig), &doubleConfig,
                                           nullptr),
                           "querying OpenCL double-precision capability");
            if (!nativeOpenClVoxelFp64CapabilityIsSufficient(
                    static_cast<std::uint64_t>(doubleConfig))) {
                throw OpenClKrigingError(
                    "selected OpenCL GPU does not support FP64, which is "
                    "required for CPU-compatible DEM neighbor ordering");
            }
            std::array<const char*, 4U> sources = kernelSources();
            std::array<std::size_t, 4U> sourceLengths{};
            for (std::size_t index = 0U; index < sources.size(); ++index) {
                sourceLengths[index] = std::strlen(sources[index]);
            }
            program_ = clCreateProgramWithSource(
                context_, static_cast<cl_uint>(sources.size()), sources.data(),
                sourceLengths.data(), &status);
            requireSuccess(status, "creating OpenCL preprocessing program");
            status = clBuildProgram(program_, 1, &device_, "-cl-std=CL1.2",
                                    nullptr, nullptr);
            if (status != CL_SUCCESS) {
                std::size_t logLength = 0U;
                (void)clGetProgramBuildInfo(program_, device_,
                                            CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                            &logLength);
                std::string log(logLength, '\0');
                if (logLength != 0U) {
                    (void)clGetProgramBuildInfo(program_, device_,
                                                CL_PROGRAM_BUILD_LOG, logLength,
                                                log.data(), nullptr);
                }
                throw OpenClKrigingError(
                    "OpenCL preprocessing kernel compilation failed: " + log);
            }
            krigingKernel_ =
                clCreateKernel(program_, "ordinary_kriging", &status);
            requireSuccess(status, "creating OpenCL kriging kernel");
            idwKernel_ = clCreateKernel(program_, "idw_interpolate", &status);
            requireSuccess(status, "creating OpenCL IDW kernel");
            gaussianHorizontalKernel_ =
                clCreateKernel(program_, "gaussian_horizontal", &status);
            requireSuccess(status, "creating OpenCL horizontal Gaussian kernel");
            gaussianVerticalKernel_ =
                clCreateKernel(program_, "gaussian_vertical", &status);
            requireSuccess(status, "creating OpenCL vertical Gaussian kernel");
            neighborKernel_ =
                clCreateKernel(program_, "select_dem_neighbors", &status);
            requireSuccess(status, "creating OpenCL DEM neighbor kernel");
            residentIdwKernel_ =
                clCreateKernel(program_, "idw_from_neighbors", &status);
            requireSuccess(status, "creating resident OpenCL IDW kernel");
            residentKrigingKernel_ =
                clCreateKernel(program_, "kriging_from_neighbors", &status);
            requireSuccess(status, "creating resident OpenCL kriging kernel");
            terrainMeshKernel_ =
                clCreateKernel(program_, "terrain_mesh_triangles", &status);
            requireSuccess(status, "creating OpenCL terrain mesh kernel");
            (void)clGetDeviceInfo(device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                  sizeof(maximumAllocation_),
                                  &maximumAllocation_, nullptr);
            if (maximumAllocation_ == 0U)
                maximumAllocation_ = 128U * 1024U * 1024U;
            (void)clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,
                                  sizeof(globalMemory_), &globalMemory_,
                                  nullptr);
        } catch (...) {
            releaseResources();
            throw;
        }
    }

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    ~Backend() { releaseResources(); }

    std::string deviceName() const {
        return infoString(device_, CL_DEVICE_NAME);
    }

    int deviceIndex() const noexcept { return deviceIndex_; }

    std::vector<float> run(const OpenClKrigingRequest& request) {
        std::scoped_lock lock(runMutex_);
        if (request.queryCount == 0U) return {};
        if (request.neighborCount == 0U || request.neighborCount > 16U) {
            throw OpenClKrigingError(
                "native OpenCL kriging supports 1 to 16 neighboring points; "
                "got " +
                std::to_string(request.neighborCount));
        }
        const std::size_t samples = checkedProduct(
            request.queryCount, request.neighborCount,
            "OpenCL kriging sample");
        const std::size_t pointValues = checkedProduct(
            samples, 2U, "OpenCL kriging point");
        if (request.localPointsXY.size() != pointValues ||
            request.localElevations.size() != samples ||
            request.localDistances.size() != samples) {
            throw OpenClKrigingError(
                "inconsistent native OpenCL kriging buffer dimensions");
        }
        const std::size_t bytesPerQuery =
            request.neighborCount * 4U * sizeof(float) + sizeof(float);
        const std::size_t allocationLimited =
            static_cast<std::size_t>(maximumAllocation_) /
            std::max<std::size_t>(request.neighborCount * 2U * sizeof(float),
                                  1U);
        const std::size_t chunkSize = std::max<std::size_t>(
            1U, std::min<std::size_t>(
                    {request.queryCount, 65536U, allocationLimited,
                     (256U * 1024U * 1024U) /
                         std::max<std::size_t>(bytesPerQuery, 1U)}));
        std::vector<float> predictions(request.queryCount);
        for (std::size_t start = 0; start < request.queryCount;
             start += chunkSize) {
            const std::size_t count =
                std::min(chunkSize, request.queryCount - start);
            executeChunk(request, start, count, predictions.data() + start);
        }
        return predictions;
    }

    std::vector<float> run(const OpenClIdwRequest& request) {
        std::scoped_lock lock(runMutex_);
        if (!std::isfinite(request.power) || request.power <= 0.0F) {
            throw OpenClKrigingError(
                "native OpenCL IDW power must be finite and greater than zero");
        }
        if (request.neighborCount == 0U ||
            request.neighborCount >
                static_cast<std::size_t>(std::numeric_limits<cl_int>::max())) {
            throw OpenClKrigingError(
                "native OpenCL IDW requires a neighbor count in [1, " +
                std::to_string(std::numeric_limits<cl_int>::max()) + "]");
        }
        const std::size_t samples = checkedProduct(
            request.queryCount, request.neighborCount, "OpenCL IDW sample");
        if (request.elevations.size() != samples ||
            request.distances.size() != samples) {
            throw OpenClKrigingError(
                "inconsistent native OpenCL IDW buffer dimensions");
        }
        if (request.queryCount == 0U) return {};

        const std::size_t bytesPerInputQuery = checkedProduct(
            request.neighborCount, sizeof(float), "OpenCL IDW query");
        if (static_cast<cl_ulong>(bytesPerInputQuery) > maximumAllocation_) {
            throw OpenClKrigingError(
                "one OpenCL IDW query exceeds the device maximum allocation");
        }
        const std::size_t bytesPerQuery = checkedSum(
            checkedProduct(bytesPerInputQuery, 2U, "OpenCL IDW query"),
            sizeof(float), "OpenCL IDW query");
        const std::size_t allocationLimited =
            static_cast<std::size_t>(maximumAllocation_) / bytesPerInputQuery;
        const std::size_t chunkSize = std::max<std::size_t>(
            1U, std::min<std::size_t>(
                    {request.queryCount, 65536U, allocationLimited,
                     (256U * 1024U * 1024U) / bytesPerQuery}));
        std::vector<float> predictions(request.queryCount);
        for (std::size_t start = 0; start < request.queryCount;
             start += chunkSize) {
            const std::size_t count =
                std::min(chunkSize, request.queryCount - start);
            executeIdwChunk(request, start, count,
                            predictions.data() + start);
        }
        return predictions;
    }

    std::vector<float> run(const OpenClGaussianRequest& request) {
        std::scoped_lock lock(runMutex_);
        const std::size_t count = checkedProduct(
            request.width, request.height, "OpenCL Gaussian grid");
        if (request.values.size() != count) {
            throw OpenClKrigingError(
                "inconsistent native OpenCL Gaussian buffer dimensions");
        }
        if (!std::isfinite(request.sigma)) {
            throw OpenClKrigingError(
                "native OpenCL Gaussian sigma must be finite");
        }
        if (count == 0U || request.sigma <= 0.0) return request.values;
        if (request.width >
                static_cast<std::size_t>(std::numeric_limits<cl_long>::max()) ||
            request.height >
                static_cast<std::size_t>(std::numeric_limits<cl_long>::max())) {
            throw OpenClKrigingError(
                "native OpenCL Gaussian dimensions exceed the kernel index range");
        }

        const double radiusValue = 4.0 * request.sigma + 0.5;
        if (!std::isfinite(radiusValue) ||
            radiusValue >=
                static_cast<double>((std::numeric_limits<cl_int>::max() - 1) /
                                    2)) {
            throw OpenClKrigingError(
                "native OpenCL Gaussian radius exceeds the kernel index range");
        }
        const int radius =
            std::max(1, static_cast<int>(radiusValue));
        const std::size_t weightCount = checkedSum(
            checkedProduct(static_cast<std::size_t>(radius), 2U,
                           "OpenCL Gaussian weights"),
            1U, "OpenCL Gaussian weights");
        const std::size_t weightBytes = checkedProduct(
            weightCount, sizeof(float), "OpenCL Gaussian weights");
        const std::size_t valueBytes = checkedProduct(
            count, sizeof(float), "OpenCL Gaussian values");
        if (static_cast<cl_ulong>(weightBytes) > maximumAllocation_ ||
            static_cast<cl_ulong>(valueBytes) > maximumAllocation_) {
            throw OpenClKrigingError(
                "native OpenCL Gaussian buffers exceed the device maximum allocation");
        }
        const std::size_t residentBytes = checkedSum(
            checkedProduct(valueBytes, 3U, "OpenCL Gaussian resident buffers"),
            weightBytes, "OpenCL Gaussian resident buffers");
        if (globalMemory_ != 0U &&
            static_cast<cl_ulong>(residentBytes) > globalMemory_) {
            throw OpenClKrigingError(
                "native OpenCL Gaussian buffers exceed device global memory");
        }

        std::vector<float> weights(weightCount);
        double weightSum = 0.0;
        for (int offset = -radius; offset <= radius; ++offset) {
            const double scaled =
                static_cast<double>(offset) / request.sigma;
            weightSum += std::exp(-0.5 * scaled * scaled);
        }
        if (!std::isfinite(weightSum) || weightSum <= 0.0) {
            throw OpenClKrigingError(
                "native OpenCL Gaussian weights could not be normalized");
        }
        for (int offset = -radius; offset <= radius; ++offset) {
            const double scaled =
                static_cast<double>(offset) / request.sigma;
            weights[static_cast<std::size_t>(offset + radius)] =
                static_cast<float>(std::exp(-0.5 * scaled * scaled) /
                                   weightSum);
        }
        return executeGaussian(request, weights, radius, valueBytes,
                               weightBytes);
    }

    OpenClTerrainInterpolationResult run(
        const OpenClTerrainInterpolationRequest& request) {
        std::scoped_lock lock(runMutex_);
        const std::size_t queryCount = checkedProduct(
            request.width, request.height, "OpenCL terrain query grid");
        if (queryCount == 0U) {
            throw OpenClKrigingError(
                "native OpenCL terrain interpolation requires a non-empty grid");
        }
        if (request.pointsXY.size() % 2U != 0U ||
            request.pointsXY.size() / 2U != request.elevations.size() ||
            request.elevations.empty()) {
            throw OpenClKrigingError(
                "inconsistent native OpenCL DEM point/elevation dimensions");
        }
        const std::size_t pointCount = request.elevations.size();
        if (request.neighborCount == 0U ||
            request.neighborCount > pointCount) {
            throw OpenClKrigingError(
                "native OpenCL terrain interpolation neighbor count must be "
                "in [1, DEM point count]");
        }
        if (request.approach == OpenClTerrainApproach::Kriging &&
            request.neighborCount > 16U) {
            throw OpenClKrigingError(
                "native OpenCL kriging supports 1 to 16 neighboring points; got " +
                std::to_string(request.neighborCount));
        }
        if (!std::isfinite(request.idwPower) || request.idwPower <= 0.0) {
            throw OpenClKrigingError(
                "native OpenCL IDW power must be finite and greater than zero");
        }
        if (!std::isfinite(request.smoothSigma)) {
            throw OpenClKrigingError(
                "native OpenCL Gaussian sigma must be finite");
        }
        for (const double coordinate : request.pointsXY) {
            if (!std::isfinite(coordinate)) {
                throw OpenClKrigingError(
                    "native OpenCL DEM coordinates must be finite");
            }
        }
        for (const double elevation : request.elevations) {
            if (!std::isfinite(elevation)) {
                throw OpenClKrigingError(
                    "native OpenCL DEM elevations must be finite");
            }
        }
        for (const double bound : {request.xMin, request.xMax, request.yMin,
                                   request.yMax}) {
            if (!std::isfinite(bound)) {
                throw OpenClKrigingError(
                    "native OpenCL terrain grid bounds must be finite");
            }
        }
        if (request.width >
                static_cast<std::size_t>(std::numeric_limits<cl_ulong>::max()) ||
            request.height >
                static_cast<std::size_t>(std::numeric_limits<cl_ulong>::max()) ||
            pointCount >
                static_cast<std::size_t>(std::numeric_limits<cl_ulong>::max()) ||
            request.neighborCount >
                static_cast<std::size_t>(
                    std::numeric_limits<cl_long>::max() - 2)) {
            throw OpenClKrigingError(
                "native OpenCL terrain dimensions exceed the kernel index range");
        }

        const RegularGridLayout layout =
            detectRegularGridLayout(request.pointsXY);
        SpatialBins bins;
        if (layout.available) {
            bins.offsets = {0U, 0U};
            bins.indices = {0U};
        } else {
            bins = buildSpatialBins(request.pointsXY);
        }
        const std::size_t neighborSamples = checkedProduct(
            queryCount, request.neighborCount, "OpenCL DEM neighbor");
        const std::size_t pointBytes = checkedProduct(
            request.pointsXY.size(), sizeof(double), "OpenCL DEM coordinates");
        const std::size_t elevationBytes = checkedProduct(
            pointCount, sizeof(double), "OpenCL DEM elevations");
        const std::size_t valueBytes = checkedProduct(
            queryCount, sizeof(float), "OpenCL terrain values");
        const std::size_t binOffsetBytes = checkedProduct(
            bins.offsets.size(), sizeof(cl_ulong),
            "OpenCL spatial-bin offsets");
        const std::size_t binIndexBytes = checkedProduct(
            bins.indices.size(), sizeof(cl_ulong),
            "OpenCL spatial-bin indices");
        for (const auto [bytes, description] :
             {std::pair{pointBytes, "DEM coordinate"},
              std::pair{elevationBytes, "DEM elevation"},
              std::pair{binOffsetBytes, "spatial-bin offset"},
              std::pair{binIndexBytes, "spatial-bin index"},
              std::pair{valueBytes, "terrain value"}}) {
            if (static_cast<cl_ulong>(bytes) > maximumAllocation_) {
                throw OpenClKrigingError(
                    std::string("native OpenCL ") + description +
                    " buffer exceeds the device maximum allocation");
            }
        }
        std::size_t residentBytes = checkedSum(
            checkedSum(pointBytes, elevationBytes,
                       "OpenCL terrain resident buffers"),
            valueBytes, "OpenCL terrain resident buffers");
        residentBytes = checkedSum(
            residentBytes,
            checkedSum(binOffsetBytes, binIndexBytes,
                       "OpenCL spatial-bin resident buffers"),
            "OpenCL terrain resident buffers");
        if (request.smoothSigma > 0.0) {
            residentBytes = checkedSum(
                residentBytes,
                checkedProduct(valueBytes, 2U,
                               "OpenCL terrain Gaussian buffers"),
                "OpenCL terrain resident buffers");
        }
        if (globalMemory_ != 0U &&
            static_cast<cl_ulong>(residentBytes) >= globalMemory_) {
            throw OpenClKrigingError(
                "native OpenCL terrain pipeline buffers exceed device global memory");
        }
        const std::size_t bytesPerNeighborBufferQuery = checkedProduct(
            request.neighborCount, sizeof(cl_ulong),
            "OpenCL DEM neighbor chunk");
        const std::size_t bytesPerNeighborQuery = checkedProduct(
            bytesPerNeighborBufferQuery, 2U,
            "OpenCL DEM neighbor chunk");
        const std::size_t allocationLimited =
            static_cast<std::size_t>(maximumAllocation_) /
            bytesPerNeighborBufferQuery;
        std::size_t memoryLimited = queryCount;
        if (globalMemory_ != 0U) {
            const cl_ulong available =
                globalMemory_ - static_cast<cl_ulong>(residentBytes);
            memoryLimited = static_cast<std::size_t>(available) /
                            bytesPerNeighborQuery;
        }
        const std::size_t chunkQueryCount = std::min<std::size_t>(
            {queryCount, 65536U, allocationLimited, memoryLimited});
        if (chunkQueryCount == 0U) {
            throw OpenClKrigingError(
                "selected OpenCL GPU has insufficient memory for one DEM neighbor query");
        }
        const std::size_t chunkNeighborSamples = checkedProduct(
            chunkQueryCount, request.neighborCount,
            "OpenCL DEM neighbor chunk");
        const std::size_t chunkIndexBytes = checkedProduct(
            chunkNeighborSamples, sizeof(cl_ulong),
            "OpenCL DEM neighbor-index chunk");
        const std::size_t chunkDistanceBytes = checkedProduct(
            chunkNeighborSamples, sizeof(double),
            "OpenCL DEM neighbor-distance chunk");

        cl_int status = CL_SUCCESS;
        Buffer points{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pointBytes,
            const_cast<double*>(request.pointsXY.data()), &status)};
        requireSuccess(status, "allocating resident GPU DEM coordinate buffer");
        Buffer elevations{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, elevationBytes,
            const_cast<double*>(request.elevations.data()), &status)};
        requireSuccess(status, "allocating resident GPU DEM elevation buffer");
        Buffer binOffsets{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, binOffsetBytes,
            bins.offsets.data(), &status)};
        requireSuccess(status, "allocating resident GPU spatial-bin offsets");
        Buffer binIndices{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, binIndexBytes,
            bins.indices.data(), &status)};
        requireSuccess(status, "allocating resident GPU spatial-bin indices");
        Buffer neighborIndices{clCreateBuffer(
            context_, CL_MEM_READ_WRITE, chunkIndexBytes, nullptr, &status)};
        requireSuccess(status, "allocating resident GPU neighbor-index buffer");
        Buffer neighborDistances{clCreateBuffer(
            context_, CL_MEM_READ_WRITE, chunkDistanceBytes, nullptr, &status)};
        requireSuccess(status, "allocating resident GPU neighbor-distance buffer");
        Buffer rawValues{clCreateBuffer(context_, CL_MEM_READ_WRITE, valueBytes,
                                       nullptr, &status)};
        requireSuccess(status, "allocating resident GPU terrain-value buffer");

        const cl_ulong clPointCount = static_cast<cl_ulong>(pointCount);
        const cl_ulong clWidth = static_cast<cl_ulong>(request.width);
        const cl_ulong clHeight = static_cast<cl_ulong>(request.height);
        const cl_ulong clNeighborCount =
            static_cast<cl_ulong>(request.neighborCount);
        const cl_int regular = layout.available ? 1 : 0;
        const cl_ulong layoutWidth = static_cast<cl_ulong>(layout.width);
        const cl_ulong layoutHeight = static_cast<cl_ulong>(layout.height);
        const cl_ulong binWidth = static_cast<cl_ulong>(bins.width);
        const cl_ulong binHeight = static_cast<cl_ulong>(bins.height);
        cl_uint argument = 0U;
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_mem),
                                      &points.value),
                       "setting GPU DEM coordinates");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_ulong),
                                      &clPointCount),
                       "setting GPU DEM point count");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_ulong),
                                      &clWidth),
                       "setting GPU terrain width");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_ulong),
                                      &clHeight),
                       "setting GPU terrain height");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &request.xMin),
                       "setting GPU terrain x minimum");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &request.xMax),
                       "setting GPU terrain x maximum");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &request.yMin),
                       "setting GPU terrain y minimum");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &request.yMax),
                       "setting GPU terrain y maximum");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_ulong),
                                      &clNeighborCount),
                       "setting GPU terrain neighbor count");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_int),
                                      &regular),
                       "setting GPU regular-grid mode");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_ulong),
                                      &layoutWidth),
                       "setting GPU DEM layout width");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_ulong),
                                      &layoutHeight),
                       "setting GPU DEM layout height");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &layout.x0),
                       "setting GPU DEM layout x origin");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &layout.y0),
                       "setting GPU DEM layout y origin");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &layout.dx),
                       "setting GPU DEM layout x spacing");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &layout.dy),
                       "setting GPU DEM layout y spacing");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_ulong),
                                      &binWidth),
                       "setting GPU spatial-bin width");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_ulong),
                                      &binHeight),
                       "setting GPU spatial-bin height");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &bins.minX),
                       "setting GPU spatial-bin x minimum");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &bins.maxX),
                       "setting GPU spatial-bin x maximum");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &bins.minY),
                       "setting GPU spatial-bin y minimum");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(double),
                                      &bins.maxY),
                       "setting GPU spatial-bin y maximum");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_mem),
                                      &binOffsets.value),
                       "setting GPU spatial-bin offsets");
        requireSuccess(clSetKernelArg(neighborKernel_, argument++, sizeof(cl_mem),
                                      &binIndices.value),
                       "setting GPU spatial-bin indices");
        OpenClTerrainInterpolationResult result;
        if (request.includeNeighborData) {
            if (neighborSamples > result.neighborIndices.max_size() ||
                neighborSamples >
                    result.neighborSquaredDistances.max_size()) {
                throw OpenClKrigingError(
                    "OpenCL DEM neighbor diagnostics exceed host container limits");
            }
            try {
                result.neighborIndices.resize(neighborSamples);
                result.neighborSquaredDistances.resize(neighborSamples);
            } catch (const std::bad_alloc&) {
                throw OpenClKrigingError(
                    "insufficient host memory for requested OpenCL DEM neighbor diagnostics");
            }
        }
        const auto enqueueNeighborChunk =
            [&](std::size_t start, std::size_t count) {
                const cl_ulong queryStart = static_cast<cl_ulong>(start);
                cl_uint changingArgument = 24U;
                requireSuccess(clSetKernelArg(
                                   neighborKernel_, changingArgument++,
                                   sizeof(cl_ulong), &queryStart),
                               "setting GPU DEM query start");
                requireSuccess(clSetKernelArg(
                                   neighborKernel_, changingArgument++,
                                   sizeof(cl_mem), &neighborIndices.value),
                               "setting GPU neighbor-index output");
                requireSuccess(clSetKernelArg(
                                   neighborKernel_, changingArgument++,
                                   sizeof(cl_mem), &neighborDistances.value),
                               "setting GPU neighbor-distance output");
                requireSuccess(clEnqueueNDRangeKernel(
                                   queue_, neighborKernel_, 1, nullptr, &count,
                                   nullptr, 0, nullptr, nullptr),
                               "running GPU DEM neighbor selection");
            };
        const auto readDiagnosticChunk =
            [&](std::size_t start, std::size_t count,
                bool readIndices, bool readDistances,
                std::vector<double>* distanceScratch) {
                const std::size_t samples = checkedProduct(
                    count, request.neighborCount,
                    "OpenCL DEM diagnostic chunk");
                if (readIndices) {
                    requireSuccess(clEnqueueReadBuffer(
                                       queue_, neighborIndices.value, CL_TRUE,
                                       0, samples * sizeof(cl_ulong),
                                       result.neighborIndices.data() +
                                           start * request.neighborCount,
                                       0, nullptr, nullptr),
                                   "reading GPU DEM neighbor indices");
                }
                if (readDistances) {
                    if (distanceScratch != nullptr) {
                        distanceScratch->resize(samples);
                        requireSuccess(clEnqueueReadBuffer(
                                           queue_, neighborDistances.value,
                                           CL_TRUE, 0,
                                           samples * sizeof(double),
                                           distanceScratch->data(), 0, nullptr,
                                           nullptr),
                                       "reading GPU DEM neighbor distances");
                        if (request.includeNeighborData) {
                            std::copy(
                                distanceScratch->begin(),
                                distanceScratch->end(),
                                result.neighborSquaredDistances.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        start * request.neighborCount));
                        }
                    } else {
                        requireSuccess(clEnqueueReadBuffer(
                                           queue_, neighborDistances.value,
                                           CL_TRUE, 0,
                                           samples * sizeof(double),
                                           result.neighborSquaredDistances.data() +
                                               start * request.neighborCount,
                                           0, nullptr, nullptr),
                                       "reading GPU DEM neighbor distances");
                    }
                }
            };

        KrigingParameters model;
        if (request.approach == OpenClTerrainApproach::Kriging) {
            std::vector<double> positiveDistances;
            if (neighborSamples > positiveDistances.max_size()) {
                throw OpenClKrigingError(
                    "exact OpenCL kriging median exceeds host container limits");
            }
            try {
                positiveDistances.reserve(neighborSamples);
            } catch (const std::bad_alloc&) {
                throw OpenClKrigingError(
                    "insufficient host memory for exact OpenCL kriging distance median");
            }
            std::vector<double> chunkDistances;
            for (std::size_t start = 0U; start < queryCount;
                 start += chunkQueryCount) {
                const std::size_t count =
                    std::min(chunkQueryCount, queryCount - start);
                enqueueNeighborChunk(start, count);
                readDiagnosticChunk(start, count,
                                    request.includeNeighborData, true,
                                    &chunkDistances);
                for (const double squared : chunkDistances) {
                    if (std::isfinite(squared) && squared > 0.0) {
                        positiveDistances.push_back(std::sqrt(squared));
                    }
                }
            }
            model = estimateKrigingParameters(
                request.pointsXY, request.elevations,
                std::move(positiveDistances));
        }

        cl_kernel interpolationKernel =
            request.approach == OpenClTerrainApproach::Kriging
                ? residentKrigingKernel_
                : residentIdwKernel_;
        for (std::size_t start = 0U; start < queryCount;
             start += chunkQueryCount) {
            const std::size_t count =
                std::min(chunkQueryCount, queryCount - start);
            enqueueNeighborChunk(start, count);
            if (request.approach == OpenClTerrainApproach::Idw &&
                request.includeNeighborData) {
                readDiagnosticChunk(start, count, true, true, nullptr);
            }
            argument = 0U;
            requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                          sizeof(cl_mem), &points.value),
                           "setting resident GPU interpolation coordinates");
            requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                          sizeof(cl_mem), &elevations.value),
                           "setting resident GPU interpolation elevations");
            requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                          sizeof(cl_mem), &neighborIndices.value),
                           "setting resident GPU interpolation indices");
            requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                          sizeof(cl_mem), &neighborDistances.value),
                           "setting resident GPU interpolation distances");
            requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                          sizeof(cl_ulong), &clNeighborCount),
                           "setting resident GPU interpolation neighbor count");
            if (request.approach == OpenClTerrainApproach::Kriging) {
                requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                              sizeof(double), &model.range),
                               "setting resident GPU kriging range");
                requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                              sizeof(double), &model.partialSill),
                               "setting resident GPU kriging partial sill");
                requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                              sizeof(double), &model.nugget),
                               "setting resident GPU kriging nugget");
            }
            requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                          sizeof(double), &request.idwPower),
                           "setting resident GPU interpolation power");
            const cl_ulong outputStart = static_cast<cl_ulong>(start);
            requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                          sizeof(cl_ulong), &outputStart),
                           "setting resident GPU interpolation output start");
            requireSuccess(clSetKernelArg(interpolationKernel, argument++,
                                          sizeof(cl_mem), &rawValues.value),
                           "setting resident GPU interpolation output");
            requireSuccess(clEnqueueNDRangeKernel(
                               queue_, interpolationKernel, 1, nullptr, &count,
                               nullptr, 0, nullptr, nullptr),
                           request.approach == OpenClTerrainApproach::Kriging
                               ? "running resident GPU kriging"
                               : "running resident GPU IDW");
        }

        Buffer gaussianTemporary;
        Buffer gaussianOutput;
        Buffer gaussianWeights;
        cl_mem finalValues = rawValues.value;
        if (request.smoothSigma > 0.0) {
            const double radiusValue = 4.0 * request.smoothSigma + 0.5;
            if (!std::isfinite(radiusValue) ||
                radiusValue >= static_cast<double>(
                                   (std::numeric_limits<cl_int>::max() - 1) /
                                   2)) {
                throw OpenClKrigingError(
                    "native OpenCL Gaussian radius exceeds the kernel index range");
            }
            const int radius =
                std::max(1, static_cast<int>(radiusValue));
            const std::size_t weightCount = checkedSum(
                checkedProduct(static_cast<std::size_t>(radius), 2U,
                               "OpenCL Gaussian weights"),
                1U, "OpenCL Gaussian weights");
            const std::size_t weightBytes = checkedProduct(
                weightCount, sizeof(float), "OpenCL Gaussian weights");
            if (static_cast<cl_ulong>(weightBytes) > maximumAllocation_) {
                throw OpenClKrigingError(
                    "native OpenCL Gaussian weights exceed the device maximum allocation");
            }
            std::vector<float> weights(weightCount);
            double weightSum = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                const double scaled =
                    static_cast<double>(offset) / request.smoothSigma;
                weightSum += std::exp(-0.5 * scaled * scaled);
            }
            if (!std::isfinite(weightSum) || weightSum <= 0.0) {
                throw OpenClKrigingError(
                    "native OpenCL Gaussian weights could not be normalized");
            }
            for (int offset = -radius; offset <= radius; ++offset) {
                const double scaled =
                    static_cast<double>(offset) / request.smoothSigma;
                weights[static_cast<std::size_t>(offset + radius)] =
                    static_cast<float>(std::exp(-0.5 * scaled * scaled) /
                                       weightSum);
            }
            gaussianTemporary.value = clCreateBuffer(
                context_, CL_MEM_READ_WRITE, valueBytes, nullptr, &status);
            requireSuccess(status,
                           "allocating resident GPU Gaussian temporary buffer");
            gaussianOutput.value = clCreateBuffer(
                context_, CL_MEM_READ_WRITE, valueBytes, nullptr, &status);
            requireSuccess(status,
                           "allocating resident GPU Gaussian output buffer");
            gaussianWeights.value = clCreateBuffer(
                context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightBytes,
                weights.data(), &status);
            requireSuccess(status,
                           "allocating resident GPU Gaussian weight buffer");
            enqueueGaussian(rawValues.value, gaussianTemporary.value,
                            gaussianOutput.value, gaussianWeights.value,
                            radius, clWidth, clHeight, queryCount);
            finalValues = gaussianOutput.value;
        }

        result.rawValues.resize(queryCount);
        requireSuccess(clEnqueueReadBuffer(queue_, rawValues.value, CL_TRUE, 0,
                                           valueBytes,
                                           result.rawValues.data(), 0, nullptr,
                                           nullptr),
                       "reading resident GPU terrain values");
        if (finalValues == rawValues.value) {
            result.values = result.rawValues;
        } else {
            result.values.resize(queryCount);
            requireSuccess(clEnqueueReadBuffer(queue_, finalValues, CL_TRUE, 0,
                                               valueBytes,
                                               result.values.data(), 0,
                                               nullptr, nullptr),
                           "reading resident GPU smoothed terrain values");
        }
        result.usedRegularGridFastPath = layout.available;
        return result;
    }

    std::vector<double> run(const OpenClTerrainMeshRequest& request) {
        std::scoped_lock lock(runMutex_);
        const std::size_t gridPoints = checkedProduct(
            request.width, request.height, "OpenCL terrain mesh grid");
        if (request.elevations.size() != gridPoints) {
            throw OpenClKrigingError(
                "inconsistent native OpenCL terrain mesh dimensions");
        }
        if (request.width < 2U || request.height < 2U) return {};
        for (const double value : request.elevations) {
            if (!std::isfinite(value)) {
                throw OpenClKrigingError(
                    "native OpenCL terrain mesh elevations must be finite");
            }
        }
        for (const double value : {request.xMin, request.xMax, request.yMin,
                                   request.yMax, request.baseHeight}) {
            if (!std::isfinite(value)) {
                throw OpenClKrigingError(
                    "native OpenCL terrain mesh bounds must be finite");
            }
        }
        const std::size_t cells = checkedProduct(
            request.width - 1U, request.height - 1U,
            "OpenCL terrain mesh cells");
        const std::size_t cellTriangles = checkedProduct(
            cells, 4U, "OpenCL terrain mesh triangles");
        const std::size_t rowTriangles = checkedProduct(
            request.height - 1U, 4U,
            "OpenCL terrain mesh row boundary triangles");
        const std::size_t columnTriangles = checkedProduct(
            request.width - 1U, 4U,
            "OpenCL terrain mesh column boundary triangles");
        const std::size_t triangleCount = checkedSum(
            checkedSum(cellTriangles, rowTriangles,
                       "OpenCL terrain mesh triangles"),
            columnTriangles, "OpenCL terrain mesh triangles");
        const std::size_t valueCount = checkedProduct(
            triangleCount, 9U, "OpenCL terrain mesh vertices");
        const std::size_t inputBytes = checkedProduct(
            gridPoints, sizeof(double), "OpenCL terrain mesh elevations");
        if (request.width >
                static_cast<std::size_t>(std::numeric_limits<cl_ulong>::max()) ||
            request.height >
                static_cast<std::size_t>(std::numeric_limits<cl_ulong>::max()) ||
            triangleCount >
                static_cast<std::size_t>(std::numeric_limits<cl_ulong>::max())) {
            throw OpenClKrigingError(
                "native OpenCL terrain mesh dimensions exceed the kernel index range");
        }
        if (static_cast<cl_ulong>(inputBytes) > maximumAllocation_) {
            throw OpenClKrigingError(
                "native OpenCL terrain mesh elevation buffer exceeds the device maximum allocation");
        }
        constexpr std::size_t bytesPerTriangle = 9U * sizeof(double);
        const std::size_t allocationLimited =
            static_cast<std::size_t>(maximumAllocation_) / bytesPerTriangle;
        std::size_t memoryLimited = triangleCount;
        if (globalMemory_ != 0U) {
            if (static_cast<cl_ulong>(inputBytes) >= globalMemory_) {
                throw OpenClKrigingError(
                    "native OpenCL terrain mesh elevation buffer exhausts device global memory");
            }
            memoryLimited = static_cast<std::size_t>(
                globalMemory_ - static_cast<cl_ulong>(inputBytes)) /
                bytesPerTriangle;
        }
        const std::size_t chunkTriangleCount = std::min<std::size_t>(
            {triangleCount, 65536U, allocationLimited, memoryLimited});
        if (chunkTriangleCount == 0U) {
            throw OpenClKrigingError(
                "selected OpenCL GPU has insufficient memory for one terrain triangle");
        }
        const std::size_t chunkOutputBytes = checkedProduct(
            chunkTriangleCount, bytesPerTriangle,
            "OpenCL terrain mesh output chunk");
        cl_int status = CL_SUCCESS;
        Buffer elevations{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputBytes,
            const_cast<double*>(request.elevations.data()), &status)};
        requireSuccess(status, "allocating GPU terrain mesh elevation buffer");
        Buffer output{clCreateBuffer(context_, CL_MEM_WRITE_ONLY, chunkOutputBytes,
                                    nullptr, &status)};
        requireSuccess(status, "allocating GPU terrain mesh output buffer");
        const cl_ulong width = static_cast<cl_ulong>(request.width);
        const cl_ulong height = static_cast<cl_ulong>(request.height);
        cl_uint argument = 0U;
        requireSuccess(clSetKernelArg(terrainMeshKernel_, argument++,
                                      sizeof(cl_mem), &elevations.value),
                       "setting GPU terrain mesh elevations");
        requireSuccess(clSetKernelArg(terrainMeshKernel_, argument++,
                                      sizeof(cl_ulong), &width),
                       "setting GPU terrain mesh width");
        requireSuccess(clSetKernelArg(terrainMeshKernel_, argument++,
                                      sizeof(cl_ulong), &height),
                       "setting GPU terrain mesh height");
        requireSuccess(clSetKernelArg(terrainMeshKernel_, argument++,
                                      sizeof(double), &request.xMin),
                       "setting GPU terrain mesh x minimum");
        requireSuccess(clSetKernelArg(terrainMeshKernel_, argument++,
                                      sizeof(double), &request.xMax),
                       "setting GPU terrain mesh x maximum");
        requireSuccess(clSetKernelArg(terrainMeshKernel_, argument++,
                                      sizeof(double), &request.yMin),
                       "setting GPU terrain mesh y minimum");
        requireSuccess(clSetKernelArg(terrainMeshKernel_, argument++,
                                      sizeof(double), &request.yMax),
                       "setting GPU terrain mesh y maximum");
        requireSuccess(clSetKernelArg(terrainMeshKernel_, argument++,
                                      sizeof(double), &request.baseHeight),
                       "setting GPU terrain mesh base height");
        std::vector<double> result(valueCount);
        for (std::size_t start = 0U; start < triangleCount;
             start += chunkTriangleCount) {
            const std::size_t count =
                std::min(chunkTriangleCount, triangleCount - start);
            const cl_ulong triangleStart = static_cast<cl_ulong>(start);
            cl_uint changingArgument = 8U;
            requireSuccess(clSetKernelArg(terrainMeshKernel_, changingArgument++,
                                          sizeof(cl_ulong), &triangleStart),
                           "setting GPU terrain mesh triangle start");
            requireSuccess(clSetKernelArg(terrainMeshKernel_, changingArgument++,
                                          sizeof(cl_mem), &output.value),
                           "setting GPU terrain mesh output");
            requireSuccess(clEnqueueNDRangeKernel(queue_, terrainMeshKernel_, 1,
                                                  nullptr, &count, nullptr, 0,
                                                  nullptr, nullptr),
                           "running GPU terrain mesh generation");
            requireSuccess(clEnqueueReadBuffer(
                               queue_, output.value, CL_TRUE, 0,
                               count * bytesPerTriangle,
                               result.data() + start * 9U, 0, nullptr, nullptr),
                           "reading GPU terrain mesh vertices");
        }
        return result;
    }

private:
    struct Buffer {
        cl_mem value = nullptr;
        ~Buffer() {
            if (value != nullptr) clReleaseMemObject(value);
        }
    };

    void executeChunk(const OpenClKrigingRequest& request, std::size_t start,
                      std::size_t count, float* predictions) {
        const std::size_t firstSample = start * request.neighborCount;
        const std::size_t sampleCount = count * request.neighborCount;
        cl_int status = CL_SUCCESS;
        Buffer points{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sampleCount * 2U * sizeof(float),
            const_cast<float*>(request.localPointsXY.data() +
                               firstSample * 2U),
            &status)};
        requireSuccess(status, "allocating GPU local-point buffer");
        Buffer elevations{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sampleCount * sizeof(float),
            const_cast<float*>(request.localElevations.data() + firstSample),
            &status)};
        requireSuccess(status, "allocating GPU elevation buffer");
        Buffer distances{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sampleCount * sizeof(float),
            const_cast<float*>(request.localDistances.data() + firstSample),
            &status)};
        requireSuccess(status, "allocating GPU distance buffer");
        Buffer output{clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                     count * sizeof(float), nullptr, &status)};
        requireSuccess(status, "allocating GPU output buffer");
        const cl_int neighbors =
            static_cast<cl_int>(request.neighborCount);
        cl_uint argument = 0U;
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(cl_mem),
                                      &points.value),
                       "setting GPU points");
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(cl_mem),
                                      &elevations.value),
                       "setting GPU elevations");
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(cl_mem),
                                      &distances.value),
                       "setting GPU distances");
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(cl_int),
                                      &neighbors),
                       "setting GPU neighbor count");
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(float),
                                      &request.range),
                       "setting GPU variogram range");
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(float),
                                      &request.partialSill),
                       "setting GPU partial sill");
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(float),
                                      &request.nugget),
                       "setting GPU nugget");
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(float),
                                      &request.fallbackPower),
                       "setting GPU fallback power");
        requireSuccess(clSetKernelArg(krigingKernel_, argument++, sizeof(cl_mem),
                                      &output.value),
                       "setting GPU output");
        const std::size_t global = count;
        requireSuccess(clEnqueueNDRangeKernel(queue_, krigingKernel_, 1, nullptr,
                                              &global, nullptr, 0, nullptr,
                                              nullptr),
                       "running GPU kriging kernel");
        requireSuccess(clEnqueueReadBuffer(queue_, output.value, CL_TRUE, 0,
                                           count * sizeof(float), predictions,
                                            0, nullptr, nullptr),
                       "reading GPU kriging predictions");
    }

    void executeIdwChunk(const OpenClIdwRequest& request, std::size_t start,
                         std::size_t count, float* predictions) {
        const std::size_t firstSample = start * request.neighborCount;
        const std::size_t sampleCount = count * request.neighborCount;
        const std::size_t sampleBytes = sampleCount * sizeof(float);
        cl_int status = CL_SUCCESS;
        Buffer elevations{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sampleBytes,
            const_cast<float*>(request.elevations.data() + firstSample),
            &status)};
        requireSuccess(status, "allocating GPU IDW elevation buffer");
        Buffer distances{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sampleBytes,
            const_cast<float*>(request.distances.data() + firstSample),
            &status)};
        requireSuccess(status, "allocating GPU IDW distance buffer");
        Buffer output{clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                     count * sizeof(float), nullptr, &status)};
        requireSuccess(status, "allocating GPU IDW output buffer");
        const cl_int neighbors = static_cast<cl_int>(request.neighborCount);
        cl_uint argument = 0U;
        requireSuccess(clSetKernelArg(idwKernel_, argument++, sizeof(cl_mem),
                                      &elevations.value),
                       "setting GPU IDW elevations");
        requireSuccess(clSetKernelArg(idwKernel_, argument++, sizeof(cl_mem),
                                      &distances.value),
                       "setting GPU IDW distances");
        requireSuccess(clSetKernelArg(idwKernel_, argument++, sizeof(cl_int),
                                      &neighbors),
                       "setting GPU IDW neighbor count");
        requireSuccess(clSetKernelArg(idwKernel_, argument++, sizeof(float),
                                      &request.power),
                       "setting GPU IDW power");
        requireSuccess(clSetKernelArg(idwKernel_, argument++, sizeof(cl_mem),
                                      &output.value),
                       "setting GPU IDW output");
        const std::size_t global = count;
        requireSuccess(clEnqueueNDRangeKernel(queue_, idwKernel_, 1, nullptr,
                                              &global, nullptr, 0, nullptr,
                                              nullptr),
                       "running GPU IDW kernel");
        requireSuccess(clEnqueueReadBuffer(queue_, output.value, CL_TRUE, 0,
                                           count * sizeof(float), predictions,
                                           0, nullptr, nullptr),
                       "reading GPU IDW predictions");
    }

    void enqueueGaussian(cl_mem input, cl_mem temporary, cl_mem output,
                         cl_mem weightBuffer, int radius, cl_ulong width,
                         cl_ulong height, std::size_t global) {
        cl_uint argument = 0U;
        requireSuccess(clSetKernelArg(gaussianHorizontalKernel_, argument++,
                                      sizeof(cl_mem), &input),
                       "setting horizontal Gaussian input");
        requireSuccess(clSetKernelArg(gaussianHorizontalKernel_, argument++,
                                      sizeof(cl_mem), &weightBuffer),
                       "setting horizontal Gaussian weights");
        requireSuccess(clSetKernelArg(gaussianHorizontalKernel_, argument++,
                                      sizeof(cl_int), &radius),
                       "setting horizontal Gaussian radius");
        requireSuccess(clSetKernelArg(gaussianHorizontalKernel_, argument++,
                                      sizeof(cl_ulong), &width),
                       "setting horizontal Gaussian width");
        requireSuccess(clSetKernelArg(gaussianHorizontalKernel_, argument++,
                                      sizeof(cl_ulong), &height),
                       "setting horizontal Gaussian height");
        requireSuccess(clSetKernelArg(gaussianHorizontalKernel_, argument++,
                                      sizeof(cl_mem), &temporary),
                       "setting horizontal Gaussian output");

        argument = 0U;
        requireSuccess(clSetKernelArg(gaussianVerticalKernel_, argument++,
                                      sizeof(cl_mem), &temporary),
                       "setting vertical Gaussian input");
        requireSuccess(clSetKernelArg(gaussianVerticalKernel_, argument++,
                                      sizeof(cl_mem), &weightBuffer),
                       "setting vertical Gaussian weights");
        requireSuccess(clSetKernelArg(gaussianVerticalKernel_, argument++,
                                      sizeof(cl_int), &radius),
                       "setting vertical Gaussian radius");
        requireSuccess(clSetKernelArg(gaussianVerticalKernel_, argument++,
                                      sizeof(cl_ulong), &width),
                       "setting vertical Gaussian width");
        requireSuccess(clSetKernelArg(gaussianVerticalKernel_, argument++,
                                      sizeof(cl_ulong), &height),
                       "setting vertical Gaussian height");
        requireSuccess(clSetKernelArg(gaussianVerticalKernel_, argument++,
                                      sizeof(cl_mem), &output),
                       "setting vertical Gaussian output");
        requireSuccess(clEnqueueNDRangeKernel(
                           queue_, gaussianHorizontalKernel_, 1, nullptr,
                           &global, nullptr, 0, nullptr, nullptr),
                       "running horizontal Gaussian kernel");
        requireSuccess(clEnqueueNDRangeKernel(
                           queue_, gaussianVerticalKernel_, 1, nullptr,
                           &global, nullptr, 0, nullptr, nullptr),
                       "running vertical Gaussian kernel");
    }

    std::vector<float> executeGaussian(
        const OpenClGaussianRequest& request,
        const std::vector<float>& weights, int radius,
        std::size_t valueBytes, std::size_t weightBytes) {
        cl_int status = CL_SUCCESS;
        Buffer input{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, valueBytes,
            const_cast<float*>(request.values.data()), &status)};
        requireSuccess(status, "allocating GPU Gaussian input buffer");
        Buffer temporary{clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                        valueBytes, nullptr, &status)};
        requireSuccess(status, "allocating GPU Gaussian temporary buffer");
        Buffer output{clCreateBuffer(context_, CL_MEM_WRITE_ONLY, valueBytes,
                                     nullptr, &status)};
        requireSuccess(status, "allocating GPU Gaussian output buffer");
        Buffer weightBuffer{clCreateBuffer(
            context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightBytes,
            const_cast<float*>(weights.data()), &status)};
        requireSuccess(status, "allocating GPU Gaussian weight buffer");

        const std::size_t global = request.values.size();
        enqueueGaussian(input.value, temporary.value, output.value,
                        weightBuffer.value, radius,
                        static_cast<cl_ulong>(request.width),
                        static_cast<cl_ulong>(request.height), global);
        std::vector<float> result(request.values.size());
        requireSuccess(clEnqueueReadBuffer(queue_, output.value, CL_TRUE, 0,
                                           valueBytes, result.data(), 0,
                                           nullptr, nullptr),
                       "reading GPU Gaussian values");
        return result;
    }

    void releaseResources() noexcept {
        if (terrainMeshKernel_ != nullptr) clReleaseKernel(terrainMeshKernel_);
        if (residentKrigingKernel_ != nullptr)
            clReleaseKernel(residentKrigingKernel_);
        if (residentIdwKernel_ != nullptr) clReleaseKernel(residentIdwKernel_);
        if (neighborKernel_ != nullptr) clReleaseKernel(neighborKernel_);
        if (gaussianVerticalKernel_ != nullptr)
            clReleaseKernel(gaussianVerticalKernel_);
        if (gaussianHorizontalKernel_ != nullptr)
            clReleaseKernel(gaussianHorizontalKernel_);
        if (idwKernel_ != nullptr) clReleaseKernel(idwKernel_);
        if (krigingKernel_ != nullptr) clReleaseKernel(krigingKernel_);
        if (program_ != nullptr) clReleaseProgram(program_);
        if (queue_ != nullptr) clReleaseCommandQueue(queue_);
        if (context_ != nullptr) clReleaseContext(context_);
        gaussianVerticalKernel_ = nullptr;
        gaussianHorizontalKernel_ = nullptr;
        terrainMeshKernel_ = nullptr;
        residentKrigingKernel_ = nullptr;
        residentIdwKernel_ = nullptr;
        neighborKernel_ = nullptr;
        idwKernel_ = nullptr;
        krigingKernel_ = nullptr;
        program_ = nullptr;
        queue_ = nullptr;
        context_ = nullptr;
    }

    static std::array<const char*, 4U> kernelSources() {
        // MSVC limits one string literal to 16,380 bytes. OpenCL explicitly
        // accepts an ordered source-string array and concatenates it into one
        // translation unit, so split only at complete kernel boundaries.
        return {R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

inline int dem_candidate_before(const double distance_squared,
                                const ulong index,
                                const double other_distance_squared,
                                const ulong other_index) {
    return distance_squared < other_distance_squared ||
           (distance_squared == other_distance_squared && index < other_index);
}

inline void consider_dem_candidate(
    __global ulong* neighbor_indices,
    __global double* neighbor_distances_squared,
    const ulong output_offset,
    const ulong neighbor_count,
    const ulong point_index,
    const double distance_squared) {
    ulong position = neighbor_count - 1UL;
    if (!dem_candidate_before(
            distance_squared, point_index,
            neighbor_distances_squared[output_offset + position],
            neighbor_indices[output_offset + position])) {
        return;
    }
    while (position > 0UL &&
           dem_candidate_before(
               distance_squared, point_index,
               neighbor_distances_squared[output_offset + position - 1UL],
               neighbor_indices[output_offset + position - 1UL])) {
        neighbor_distances_squared[output_offset + position] =
            neighbor_distances_squared[output_offset + position - 1UL];
        neighbor_indices[output_offset + position] =
            neighbor_indices[output_offset + position - 1UL];
        --position;
    }
    neighbor_distances_squared[output_offset + position] = distance_squared;
    neighbor_indices[output_offset + position] = point_index;
}

__kernel void select_dem_neighbors(
    __global const double2* points,
    const ulong point_count,
    const ulong query_width,
    const ulong query_height,
    const double query_x_min,
    const double query_x_max,
    const double query_y_min,
    const double query_y_max,
    const ulong neighbor_count,
    const int regular_layout,
    const ulong layout_width,
    const ulong layout_height,
    const double layout_x0,
    const double layout_y0,
    const double layout_dx,
    const double layout_dy,
    const ulong bin_width,
    const ulong bin_height,
    const double bin_min_x,
    const double bin_max_x,
    const double bin_min_y,
    const double bin_max_y,
    __global const ulong* bin_offsets,
    __global const ulong* bin_indices,
    const ulong query_start,
    __global ulong* neighbor_indices,
    __global double* neighbor_distances_squared) {
    const ulong local_query = (ulong)get_global_id(0);
    const ulong query = query_start + local_query;
    const ulong row = query / query_width;
    const ulong column = query % query_width;
    if (row >= query_height) return;
    const double query_x = query_width <= 1UL
        ? query_x_min
        : query_x_min + (query_x_max - query_x_min) * (double)column /
              (double)(query_width - 1UL);
    const double query_y = query_height <= 1UL
        ? query_y_min
        : query_y_min + (query_y_max - query_y_min) * (double)row /
              (double)(query_height - 1UL);
    const ulong output_offset = local_query * neighbor_count;
    for (ulong neighbor = 0UL; neighbor < neighbor_count; ++neighbor) {
        neighbor_indices[output_offset + neighbor] = (ulong)(-1L);
        neighbor_distances_squared[output_offset + neighbor] = INFINITY;
    }

    if (regular_layout != 0) {
        long center_column = 0L;
        long center_row = 0L;
        if (layout_width > 1UL) {
            const double fractional = (query_x - layout_x0) / layout_dx;
            if (fractional <= 0.0) {
                center_column = 0L;
            } else if (fractional >= (double)(layout_width - 1UL)) {
                center_column = (long)(layout_width - 1UL);
            } else {
                center_column = (long)floor(fractional + 0.5);
            }
        }
        if (layout_height > 1UL) {
            const double fractional = (query_y - layout_y0) / layout_dy;
            if (fractional <= 0.0) {
                center_row = 0L;
            } else if (fractional >= (double)(layout_height - 1UL)) {
                center_row = (long)(layout_height - 1UL);
            } else {
                center_row = (long)floor(fractional + 0.5);
            }
        }
        const long radius = (long)neighbor_count + 2L;
        const long first_column = max(0L, center_column - radius);
        const long last_column = min((long)layout_width - 1L,
                                     center_column + radius);
        const long first_row = max(0L, center_row - radius);
        const long last_row = min((long)layout_height - 1L,
                                  center_row + radius);
        for (long candidate_row = first_row; candidate_row <= last_row;
             ++candidate_row) {
            for (long candidate_column = first_column;
                 candidate_column <= last_column; ++candidate_column) {
                const ulong point_index =
                    (ulong)candidate_row * layout_width +
                    (ulong)candidate_column;
                const double2 point = points[point_index];
                const double dx = query_x - point.x;
                const double dy = query_y - point.y;
                const double distance_squared = dx * dx + dy * dy;
                consider_dem_candidate(
                    neighbor_indices, neighbor_distances_squared,
                    output_offset, neighbor_count, point_index,
                    distance_squared);
            }
        }
    } else {
        const double span_x = bin_max_x - bin_min_x;
        const double span_y = bin_max_y - bin_min_y;
        long center_column = 0L;
        long center_row = 0L;
        if (bin_width > 1UL && span_x > 0.0) {
            const double normalized = (query_x - bin_min_x) / span_x;
            if (normalized <= 0.0) center_column = 0L;
            else if (normalized >= 1.0)
                center_column = (long)bin_width - 1L;
            else
                center_column = min((long)bin_width - 1L,
                    (long)floor(normalized * (double)bin_width));
        }
        if (bin_height > 1UL && span_y > 0.0) {
            const double normalized = (query_y - bin_min_y) / span_y;
            if (normalized <= 0.0) center_row = 0L;
            else if (normalized >= 1.0)
                center_row = (long)bin_height - 1L;
            else
                center_row = min((long)bin_height - 1L,
                    (long)floor(normalized * (double)bin_height));
        }
        const long maximum_radius =
            max((long)bin_width, (long)bin_height);
        for (long radius = 0L; radius < maximum_radius; ++radius) {
            const long first_column = max(0L, center_column - radius);
            const long last_column = min((long)bin_width - 1L,
                                         center_column + radius);
            const long first_row = max(0L, center_row - radius);
            const long last_row = min((long)bin_height - 1L,
                                      center_row + radius);
            for (long candidate_row = first_row;
                 candidate_row <= last_row; ++candidate_row) {
                for (long candidate_column = first_column;
                     candidate_column <= last_column; ++candidate_column) {
                    const long row_delta = candidate_row >= center_row
                        ? candidate_row - center_row
                        : center_row - candidate_row;
                    const long column_delta = candidate_column >= center_column
                        ? candidate_column - center_column
                        : center_column - candidate_column;
                    if (max(row_delta, column_delta) != radius) {
                        continue;
                    }
                    const ulong bin = (ulong)candidate_row * bin_width +
                                      (ulong)candidate_column;
                    for (ulong position = bin_offsets[bin];
                         position < bin_offsets[bin + 1UL]; ++position) {
                        const ulong point_index = bin_indices[position];
                        const double2 point = points[point_index];
                        const double dx = query_x - point.x;
                        const double dy = query_y - point.y;
                        consider_dem_candidate(
                            neighbor_indices,
                            neighbor_distances_squared, output_offset,
                            neighbor_count, point_index,
                            dx * dx + dy * dy);
                    }
                }
            }
            if (first_column == 0L &&
                last_column == (long)bin_width - 1L &&
                first_row == 0L && last_row == (long)bin_height - 1L) {
                break;
            }
            const double worst = neighbor_distances_squared[
                output_offset + neighbor_count - 1UL];
            if (!isfinite(worst)) continue;
            double unseen_lower_bound = INFINITY;
            if (first_column > 0L) {
                const double boundary = bin_min_x + span_x *
                    (double)first_column / (double)bin_width;
                const double delta = query_x - boundary;
                unseen_lower_bound = fmin(unseen_lower_bound, delta * delta);
            }
            if (last_column < (long)bin_width - 1L) {
                const double boundary = bin_min_x + span_x *
                    (double)(last_column + 1L) / (double)bin_width;
                const double delta = boundary - query_x;
                unseen_lower_bound = fmin(unseen_lower_bound, delta * delta);
            }
            if (first_row > 0L) {
                const double boundary = bin_min_y + span_y *
                    (double)first_row / (double)bin_height;
                const double delta = query_y - boundary;
                unseen_lower_bound = fmin(unseen_lower_bound, delta * delta);
            }
            if (last_row < (long)bin_height - 1L) {
                const double boundary = bin_min_y + span_y *
                    (double)(last_row + 1L) / (double)bin_height;
                const double delta = boundary - query_y;
                unseen_lower_bound = fmin(unseen_lower_bound, delta * delta);
            }
            if (worst < unseen_lower_bound) break;
        }
    }
}

__kernel void idw_from_neighbors(
    __global const double2* points,
    __global const double* elevations,
    __global const ulong* neighbor_indices,
    __global const double* neighbor_distances_squared,
    const ulong neighbor_count,
    const double power_value,
    const ulong output_start,
    __global float* predictions) {
    (void)points;
    const ulong local_query = (ulong)get_global_id(0);
    const ulong offset = local_query * neighbor_count;
    double minimum_distance = INFINITY;
    for (ulong neighbor = 0UL; neighbor < neighbor_count; ++neighbor) {
        minimum_distance = fmin(
            minimum_distance,
            fmax(sqrt(neighbor_distances_squared[offset + neighbor]),
                 1.0e-10));
    }
    double weighted = 0.0;
    double weight_sum = 0.0;
    for (ulong neighbor = 0UL; neighbor < neighbor_count; ++neighbor) {
        const double distance = fmax(
            sqrt(neighbor_distances_squared[offset + neighbor]), 1.0e-10);
        const double weight = pow(minimum_distance / distance, power_value);
        weighted += weight * elevations[neighbor_indices[offset + neighbor]];
        weight_sum += weight;
    }
    predictions[output_start + local_query] =
        (float)(weight_sum > 0.0 ? weighted / weight_sum : 0.0);
}

inline double variogram_double(const double distance,
                               const double range_value,
                               const double partial_sill,
                               const double nugget) {
    const double gamma = partial_sill *
        (1.0 - exp(-fmax(distance, 0.0) / range_value));
    return gamma + (distance > 0.0 ? nugget : 0.0);
}

)CLC",
R"CLC(
__kernel void kriging_from_neighbors(
    __global const double2* points,
    __global const double* elevations,
    __global const ulong* neighbor_indices,
    __global const double* neighbor_distances_squared,
    const ulong neighbor_count_value,
    const double range_value,
    const double partial_sill,
    const double nugget,
    const double fallback_power,
    const ulong output_start,
    __global float* predictions) {
    const ulong local_query = (ulong)get_global_id(0);
    const ulong offset = local_query * neighbor_count_value;
    const int neighbor_count = (int)neighbor_count_value;
    double matrix[17 * 17];
    double rhs[17];
    double weights[17];
    const int system_size = neighbor_count + 1;
    double local_min = INFINITY;
    double local_max = -INFINITY;
    int exact_index = -1;
    const double first_elevation =
        elevations[neighbor_indices[offset]];
    for (int i = 0; i < neighbor_count; ++i) {
        const double elevation =
            elevations[neighbor_indices[offset + (ulong)i]];
        local_min = fmin(local_min, elevation);
        local_max = fmax(local_max, elevation);
        if (neighbor_distances_squared[offset + (ulong)i] <= 1.0e-20 &&
            exact_index < 0) {
            exact_index = i;
        }
    }
    if (exact_index >= 0) {
        predictions[output_start + local_query] = (float)elevations[
            neighbor_indices[offset + (ulong)exact_index]];
        return;
    }
    if (neighbor_count == 1 || local_max - local_min <= 1.0e-12) {
        predictions[output_start + local_query] = (float)first_elevation;
        return;
    }
    for (int i = 0; i < system_size; ++i) {
        rhs[i] = 0.0;
        weights[i] = 0.0;
        for (int j = 0; j < system_size; ++j)
            matrix[i * system_size + j] = 0.0;
    }
    const double regularization = fmax(nugget * 1.0e-6, 1.0e-10);
    for (int row_index = 0; row_index < neighbor_count; ++row_index) {
        const double2 a = points[
            neighbor_indices[offset + (ulong)row_index]];
        for (int column_index = 0; column_index < neighbor_count;
             ++column_index) {
            if (row_index == column_index) continue;
            const double2 b = points[
                neighbor_indices[offset + (ulong)column_index]];
            const double2 delta = a - b;
            matrix[row_index * system_size + column_index] =
                variogram_double(sqrt(dot(delta, delta)), range_value,
                                 partial_sill, nugget);
        }
        matrix[row_index * system_size + row_index] = regularization;
        matrix[row_index * system_size + neighbor_count] = 1.0;
        matrix[neighbor_count * system_size + row_index] = 1.0;
        rhs[row_index] = variogram_double(
            sqrt(neighbor_distances_squared[offset + (ulong)row_index]),
            range_value, partial_sill, nugget);
    }
    rhs[neighbor_count] = 1.0;
    int singular = 0;
    for (int pivot = 0; pivot < system_size; ++pivot) {
        int pivot_row = pivot;
        double pivot_value = fabs(matrix[pivot * system_size + pivot]);
        for (int candidate = pivot + 1; candidate < system_size;
             ++candidate) {
            const double candidate_value =
                fabs(matrix[candidate * system_size + pivot]);
            if (candidate_value > pivot_value) {
                pivot_value = candidate_value;
                pivot_row = candidate;
            }
        }
        if (!isfinite(pivot_value) || pivot_value <= 1.0e-14) {
            singular = 1;
            break;
        }
        if (pivot_row != pivot) {
            for (int column_index = pivot; column_index < system_size;
                 ++column_index) {
                const double temporary =
                    matrix[pivot * system_size + column_index];
                matrix[pivot * system_size + column_index] =
                    matrix[pivot_row * system_size + column_index];
                matrix[pivot_row * system_size + column_index] = temporary;
            }
            const double temporary = rhs[pivot];
            rhs[pivot] = rhs[pivot_row];
            rhs[pivot_row] = temporary;
        }
        const double diagonal = matrix[pivot * system_size + pivot];
        for (int target_row = pivot + 1; target_row < system_size;
             ++target_row) {
            const double factor =
                matrix[target_row * system_size + pivot] / diagonal;
            matrix[target_row * system_size + pivot] = 0.0;
            for (int column_index = pivot + 1;
                 column_index < system_size; ++column_index) {
                matrix[target_row * system_size + column_index] -=
                    factor * matrix[pivot * system_size + column_index];
            }
            rhs[target_row] -= factor * rhs[pivot];
        }
    }
    if (!singular) {
        for (int row_index = system_size - 1; row_index >= 0;
             --row_index) {
            double value = rhs[row_index];
            for (int column_index = row_index + 1;
                 column_index < system_size; ++column_index) {
                value -= matrix[row_index * system_size + column_index] *
                         weights[column_index];
            }
            const double diagonal =
                matrix[row_index * system_size + row_index];
            if (!isfinite(diagonal) || fabs(diagonal) <= 1.0e-14) {
                singular = 1;
                break;
            }
            weights[row_index] = value / diagonal;
        }
    }
    double prediction = 0.0;
    if (!singular) {
        for (int i = 0; i < neighbor_count; ++i) {
            prediction += weights[i] *
                elevations[neighbor_indices[offset + (ulong)i]];
        }
        if (!isfinite(prediction)) singular = 1;
    }
    if (singular) {
        int rowwise_all_close = 1;
        for (int i = 0; i < neighbor_count; ++i) {
            const double elevation =
                elevations[neighbor_indices[offset + (ulong)i]];
            if (fabs(elevation - first_elevation) >
                1.0e-8 + 1.0e-5 * fabs(first_elevation)) {
                rowwise_all_close = 0;
            }
        }
        if (rowwise_all_close) {
            prediction = first_elevation;
        } else {
            double minimum_distance = INFINITY;
            for (int i = 0; i < neighbor_count; ++i) {
                minimum_distance = fmin(
                    minimum_distance,
                    fmax(sqrt(neighbor_distances_squared[
                             offset + (ulong)i]), 1.0e-10));
            }
            double weighted = 0.0;
            double weight_sum = 0.0;
            for (int i = 0; i < neighbor_count; ++i) {
                const double distance = fmax(
                    sqrt(neighbor_distances_squared[offset + (ulong)i]),
                    1.0e-10);
                const double weight =
                    pow(minimum_distance / distance, fallback_power);
                weighted += weight * elevations[
                    neighbor_indices[offset + (ulong)i]];
                weight_sum += weight;
            }
            prediction = weight_sum > 0.0
                ? weighted / weight_sum
                : first_elevation;
        }
    }
    predictions[output_start + local_query] =
        (float)clamp(prediction, local_min, local_max);
}

)CLC",
R"CLC(
inline double terrain_grid_x(const ulong column, const ulong width,
                             const double x_min, const double x_max) {
    return width <= 1UL
        ? x_min
        : x_min + (x_max - x_min) * (double)column /
              (double)(width - 1UL);
}

inline double terrain_grid_y(const ulong row, const ulong height,
                             const double y_min, const double y_max) {
    return height <= 1UL
        ? y_min
        : y_min + (y_max - y_min) * (double)row /
              (double)(height - 1UL);
}

inline void store_terrain_triangle(__global double* output,
                                   const ulong local_triangle,
                                   const double3 a, const double3 b,
                                   const double3 c) {
    const ulong offset = local_triangle * 9UL;
    output[offset] = a.x;
    output[offset + 1UL] = a.y;
    output[offset + 2UL] = a.z;
    output[offset + 3UL] = b.x;
    output[offset + 4UL] = b.y;
    output[offset + 5UL] = b.z;
    output[offset + 6UL] = c.x;
    output[offset + 7UL] = c.y;
    output[offset + 8UL] = c.z;
}

__kernel void terrain_mesh_triangles(
    __global const double* elevations,
    const ulong width,
    const ulong height,
    const double x_min,
    const double x_max,
    const double y_min,
    const double y_max,
    const double base_height,
    const ulong triangle_start,
    __global double* output) {
    const ulong local_triangle = (ulong)get_global_id(0);
    const ulong triangle = triangle_start + local_triangle;
    const ulong cells_x = width - 1UL;
    const ulong cells_y = height - 1UL;
    const ulong cell_triangles = cells_x * cells_y * 4UL;
    const ulong row_boundary_triangles = cells_y * 4UL;
    double3 a;
    double3 b;
    double3 c;
    double3 d;
    ulong triangle_offset = triangle;
    if (triangle_offset < cell_triangles) {
        const ulong cell = triangle_offset / 4UL;
        const ulong which = triangle_offset % 4UL;
        const ulong row = cell / cells_x;
        const ulong column = cell % cells_x;
        const double x0 = terrain_grid_x(column, width, x_min, x_max);
        const double x1 = terrain_grid_x(column + 1UL, width, x_min, x_max);
        const double y0 = terrain_grid_y(row, height, y_min, y_max);
        const double y1 = terrain_grid_y(row + 1UL, height, y_min, y_max);
        const double3 v0 = (double3)(x0, y0,
            base_height + elevations[row * width + column]);
        const double3 v1 = (double3)(x1, y0,
            base_height + elevations[row * width + column + 1UL]);
        const double3 v2 = (double3)(x0, y1,
            base_height + elevations[(row + 1UL) * width + column]);
        const double3 v3 = (double3)(x1, y1,
            base_height + elevations[(row + 1UL) * width + column + 1UL]);
        const double3 b0 = (double3)(x0, y0, base_height);
        const double3 b1 = (double3)(x1, y0, base_height);
        const double3 b2 = (double3)(x0, y1, base_height);
        const double3 b3 = (double3)(x1, y1, base_height);
        if (which == 0UL) { a = v0; b = v1; c = v2; }
        else if (which == 1UL) { a = v1; b = v3; c = v2; }
        else if (which == 2UL) { a = b0; b = b2; c = b1; }
        else { a = b1; b = b2; c = b3; }
    } else if ((triangle_offset -= cell_triangles) < row_boundary_triangles) {
        const ulong row = triangle_offset / 4UL;
        const ulong which = triangle_offset % 4UL;
        const int right = which >= 2UL;
        const int second = (which & 1UL) != 0UL;
        const ulong column = right ? width - 1UL : 0UL;
        const double x = terrain_grid_x(column, width, x_min, x_max);
        const double y0 = terrain_grid_y(row, height, y_min, y_max);
        const double y1 = terrain_grid_y(row + 1UL, height, y_min, y_max);
        const double3 top0 = (double3)(x, y0,
            base_height + elevations[row * width + column]);
        const double3 top1 = (double3)(x, y1,
            base_height + elevations[(row + 1UL) * width + column]);
        const double3 bottom0 = (double3)(x, y0, base_height);
        const double3 bottom1 = (double3)(x, y1, base_height);
        if (!right) { a = top0; b = top1; c = bottom1; d = bottom0; }
        else { a = top0; b = bottom0; c = bottom1; d = top1; }
        if (second) { b = c; c = d; }
    } else {
        triangle_offset -= row_boundary_triangles;
        const ulong column = triangle_offset / 4UL;
        const ulong which = triangle_offset % 4UL;
        const int back = which >= 2UL;
        const int second = (which & 1UL) != 0UL;
        const ulong row = back ? height - 1UL : 0UL;
        const double x0 = terrain_grid_x(column, width, x_min, x_max);
        const double x1 = terrain_grid_x(column + 1UL, width, x_min, x_max);
        const double y = terrain_grid_y(row, height, y_min, y_max);
        const double3 top0 = (double3)(x0, y,
            base_height + elevations[row * width + column]);
        const double3 top1 = (double3)(x1, y,
            base_height + elevations[row * width + column + 1UL]);
        const double3 bottom0 = (double3)(x0, y, base_height);
        const double3 bottom1 = (double3)(x1, y, base_height);
        if (!back) { a = top0; b = bottom0; c = bottom1; d = top1; }
        else { a = top0; b = top1; c = bottom1; d = bottom0; }
        if (second) { b = c; c = d; }
    }
    store_terrain_triangle(output, local_triangle, a, b, c);
}

)CLC",
R"CLC(
inline float variogram(float distance, float range_value, float partial_sill, float nugget) {
    float gamma = partial_sill * (1.0f - exp(-fmax(distance, 0.0f) / range_value));
    return gamma + (distance > 0.0f ? nugget : 0.0f);
}

__kernel void idw_interpolate(
    __global const float* elevations,
    __global const float* distances,
    const int neighbor_count,
    const float power_value,
    __global float* predictions) {
    const size_t query = get_global_id(0);
    const size_t offset = query * (size_t)neighbor_count;
    float weighted = 0.0f;
    float weight_sum = 0.0f;
    float minimum_distance = INFINITY;
    for (int neighbor = 0; neighbor < neighbor_count; ++neighbor) {
        minimum_distance = fmin(
            minimum_distance,
            fmax(distances[offset + neighbor], 1.0e-10f));
    }
    for (int neighbor = 0; neighbor < neighbor_count; ++neighbor) {
        const float distance = fmax(distances[offset + neighbor], 1.0e-10f);
        const float weight = pow(minimum_distance / distance, power_value);
        weighted += weight * elevations[offset + neighbor];
        weight_sum += weight;
    }
    predictions[query] = weight_sum > 0.0f ? weighted / weight_sum : 0.0f;
}

inline long reflected_index(long index, const long extent) {
    if (extent <= 1) return 0;
    while (index < 0 || index >= extent) {
        if (index < 0) index = -index - 1;
        if (index >= extent) index = 2 * extent - index - 1;
    }
    return index;
}

__kernel void gaussian_horizontal(
    __global const float* values,
    __global const float* weights,
    const int radius,
    const ulong width,
    const ulong height,
    __global float* result) {
    const ulong linear = (ulong)get_global_id(0);
    if (linear >= width * height) return;
    const long row = (long)(linear / width);
    const long column = (long)(linear % width);
    float value = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
        const long reflected = reflected_index(column + (long)offset,
                                               (long)width);
        value += weights[offset + radius] *
                 values[(ulong)row * width + (ulong)reflected];
    }
    result[linear] = value;
}

__kernel void gaussian_vertical(
    __global const float* values,
    __global const float* weights,
    const int radius,
    const ulong width,
    const ulong height,
    __global float* result) {
    const ulong linear = (ulong)get_global_id(0);
    if (linear >= width * height) return;
    const long row = (long)(linear / width);
    const ulong column = linear % width;
    float value = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
        const long reflected = reflected_index(row + (long)offset,
                                               (long)height);
        value += weights[offset + radius] *
                 values[(ulong)reflected * width + column];
    }
    result[linear] = value;
}

__kernel void ordinary_kriging(
    __global const float2* local_points,
    __global const float* local_elevations,
    __global const float* local_distances,
    const int neighbor_count,
    const float range_value,
    const float partial_sill,
    const float nugget,
    const float fallback_power,
    __global float* predictions) {
    const int query = (int)get_global_id(0);
    const int offset = query * neighbor_count;
    float matrix[17 * 17];
    float rhs[17];
    float weights[17];
    const int system_size = neighbor_count + 1;
    float local_min = INFINITY;
    float local_max = -INFINITY;
    int exact_index = -1;
    float first_elevation = local_elevations[offset];
    for (int i = 0; i < neighbor_count; ++i) {
        float elevation = local_elevations[offset + i];
        local_min = fmin(local_min, elevation);
        local_max = fmax(local_max, elevation);
        if (local_distances[offset + i] <= 1.0e-10f && exact_index < 0)
            exact_index = i;
    }
    if (exact_index >= 0) {
        predictions[query] = local_elevations[offset + exact_index];
        return;
    }
    if (neighbor_count == 1 || local_max - local_min <= 1.0e-12f) {
        predictions[query] = first_elevation;
        return;
    }
    for (int i = 0; i < system_size; ++i) {
        rhs[i] = 0.0f;
        weights[i] = 0.0f;
        for (int j = 0; j < system_size; ++j)
            matrix[i * system_size + j] = 0.0f;
    }
    const float regularization = fmax(nugget * 1.0e-6f, 1.0e-10f);
    for (int row = 0; row < neighbor_count; ++row) {
        float2 a = local_points[offset + row];
        for (int column = 0; column < neighbor_count; ++column) {
            if (row == column) continue;
            float2 delta = a - local_points[offset + column];
            matrix[row * system_size + column] = variogram(
                sqrt(dot(delta, delta)), range_value, partial_sill, nugget);
        }
        matrix[row * system_size + row] = regularization;
        matrix[row * system_size + neighbor_count] = 1.0f;
        matrix[neighbor_count * system_size + row] = 1.0f;
        rhs[row] = variogram(local_distances[offset + row], range_value,
                             partial_sill, nugget);
    }
    rhs[neighbor_count] = 1.0f;
    int singular = 0;
    for (int pivot = 0; pivot < system_size; ++pivot) {
        int pivot_row = pivot;
        float pivot_value = fabs(matrix[pivot * system_size + pivot]);
        for (int candidate = pivot + 1; candidate < system_size; ++candidate) {
            float candidate_value =
                fabs(matrix[candidate * system_size + pivot]);
            if (candidate_value > pivot_value) {
                pivot_value = candidate_value;
                pivot_row = candidate;
            }
        }
        if (pivot_value < 1.0e-8f) {
            singular = 1;
            break;
        }
        if (pivot_row != pivot) {
            for (int column = 0; column < system_size; ++column) {
                float temporary = matrix[pivot * system_size + column];
                matrix[pivot * system_size + column] =
                    matrix[pivot_row * system_size + column];
                matrix[pivot_row * system_size + column] = temporary;
            }
            float temporary = rhs[pivot];
            rhs[pivot] = rhs[pivot_row];
            rhs[pivot_row] = temporary;
        }
        float diagonal = matrix[pivot * system_size + pivot];
        for (int column = pivot + 1; column < system_size; ++column)
            matrix[pivot * system_size + column] /= diagonal;
        rhs[pivot] /= diagonal;
        matrix[pivot * system_size + pivot] = 1.0f;

        for (int target_row = pivot + 1; target_row < system_size;
             ++target_row) {
            float factor = matrix[target_row * system_size + pivot];
            if (fabs(factor) <= 1.0e-12f) {
                matrix[target_row * system_size + pivot] = 0.0f;
                continue;
            }
            for (int column = pivot + 1; column < system_size; ++column) {
                matrix[target_row * system_size + column] -=
                    factor * matrix[pivot * system_size + column];
            }
            rhs[target_row] -= factor * rhs[pivot];
            matrix[target_row * system_size + pivot] = 0.0f;
        }
    }
    if (!singular) {
        for (int target_row = system_size - 1; target_row >= 0;
             --target_row) {
            float value = rhs[target_row];
            for (int column = target_row + 1; column < system_size; ++column)
                value -= matrix[target_row * system_size + column] *
                         weights[column];
            weights[target_row] = value;
        }
    }
    float prediction = 0.0f;
    if (!singular) {
        for (int i = 0; i < neighbor_count; ++i)
            prediction += weights[i] * local_elevations[offset + i];
        if (!isfinite(prediction)) singular = 1;
    }
    if (singular) {
        int rowwise_all_close = 1;
        for (int i = 0; i < neighbor_count; ++i) {
            if (fabs(local_elevations[offset + i] - first_elevation) >
                1.0e-8f + 1.0e-5f * fabs(first_elevation)) {
                rowwise_all_close = 0;
            }
        }
        if (rowwise_all_close) {
            prediction = first_elevation;
        } else {
            float minimum_distance = INFINITY;
            for (int i = 0; i < neighbor_count; ++i) {
                minimum_distance = fmin(
                    minimum_distance,
                    fmax(local_distances[offset + i], 1.0e-10f));
            }
            float weighted = 0.0f;
            float weight_sum = 0.0f;
            for (int i = 0; i < neighbor_count; ++i) {
                float distance =
                    fmax(local_distances[offset + i], 1.0e-10f);
                float weight =
                    pow(minimum_distance / distance, fallback_power);
                weighted += weight * local_elevations[offset + i];
                weight_sum += weight;
            }
            prediction = weight_sum > 0.0f
                             ? weighted / weight_sum
                             : first_elevation;
        }
    }
    predictions[query] = clamp(prediction, local_min, local_max);
}
)CLC"};
    }

    int deviceIndex_ = 0;
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel krigingKernel_ = nullptr;
    cl_kernel idwKernel_ = nullptr;
    cl_kernel gaussianHorizontalKernel_ = nullptr;
    cl_kernel gaussianVerticalKernel_ = nullptr;
    cl_kernel neighborKernel_ = nullptr;
    cl_kernel residentIdwKernel_ = nullptr;
    cl_kernel residentKrigingKernel_ = nullptr;
    cl_kernel terrainMeshKernel_ = nullptr;
    cl_ulong maximumAllocation_ = 0U;
    cl_ulong globalMemory_ = 0U;
    std::mutex runMutex_;
};

std::unordered_map<int, std::unique_ptr<Backend>>& backendInstances() {
    static std::unordered_map<int, std::unique_ptr<Backend>> backends;
    return backends;
}

std::mutex& backendInstancesMutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace
#endif

bool nativeOpenClVoxelRequiresWarmup(int deviceIndex) noexcept {
#if LUW_HAS_OPENCL
    try {
        std::scoped_lock lock(backendInstancesMutex());
        return !backendInstances().contains(deviceIndex);
    } catch (...) {
        return true;
    }
#else
    (void)deviceIndex;
    return true;
#endif
}

OpenClVoxelBackendInfo requireNativeOpenClVoxelBackend(int deviceIndex) {
#if LUW_HAS_OPENCL
    std::scoped_lock lock(backendInstancesMutex());
    auto& backends = backendInstances();
    auto match = backends.find(deviceIndex);
    const bool compileNow = match == backends.end();
    if (compileNow) {
        match = backends.emplace(deviceIndex,
                                 std::make_unique<Backend>(deviceIndex)).first;
    }
    return {match->second->deviceName(), match->second->deviceIndex(),
            compileNow};
#else
    (void)deviceIndex;
    throw OpenClKrigingError("OpenCL support is not enabled in this build");
#endif
}

bool nativeOpenClKrigingRequiresWarmup(int deviceIndex) noexcept {
    return nativeOpenClVoxelRequiresWarmup(deviceIndex);
}

OpenClKrigingBackendInfo requireNativeOpenClKrigingBackend(int deviceIndex) {
    return requireNativeOpenClVoxelBackend(deviceIndex);
}

OpenClKrigingResult runNativeOpenClKriging(
    const OpenClKrigingRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClKrigingBackendInfo backendInfo =
        requireNativeOpenClVoxelBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    OpenClKrigingResult result{backend->run(request), backendInfo.device,
                               backendInfo.deviceIndex,
                               backendInfo.compiledThisCall};
    return result;
#else
    (void)request;
    throw OpenClKrigingError("OpenCL support is not enabled in this build");
#endif
}

OpenClIdwResult runNativeOpenClIdw(const OpenClIdwRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClVoxelBackendInfo backendInfo =
        requireNativeOpenClVoxelBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    return {backend->run(request), backendInfo.device,
            backendInfo.deviceIndex, backendInfo.compiledThisCall};
#else
    (void)request;
    throw OpenClKrigingError("OpenCL support is not enabled in this build");
#endif
}

OpenClGaussianResult runNativeOpenClGaussian(
    const OpenClGaussianRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClVoxelBackendInfo backendInfo =
        requireNativeOpenClVoxelBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    return {backend->run(request), backendInfo.device,
            backendInfo.deviceIndex, backendInfo.compiledThisCall};
#else
    (void)request;
    throw OpenClKrigingError("OpenCL support is not enabled in this build");
#endif
}

OpenClTerrainInterpolationResult runNativeOpenClTerrainInterpolation(
    const OpenClTerrainInterpolationRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClVoxelBackendInfo backendInfo =
        requireNativeOpenClVoxelBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    OpenClTerrainInterpolationResult result = backend->run(request);
    result.device = backendInfo.device;
    result.deviceIndex = backendInfo.deviceIndex;
    result.compiledThisCall = backendInfo.compiledThisCall;
    return result;
#else
    (void)request;
    throw OpenClKrigingError("OpenCL support is not enabled in this build");
#endif
}

OpenClTerrainMeshResult runNativeOpenClTerrainMesh(
    const OpenClTerrainMeshRequest& request) {
#if LUW_HAS_OPENCL
    const OpenClVoxelBackendInfo backendInfo =
        requireNativeOpenClVoxelBackend(request.deviceIndex);
    Backend* backend = nullptr;
    {
        std::scoped_lock lock(backendInstancesMutex());
        backend = backendInstances().at(request.deviceIndex).get();
    }
    return {backend->run(request), backendInfo.device,
            backendInfo.deviceIndex, backendInfo.compiledThisCall};
#else
    (void)request;
    throw OpenClKrigingError("OpenCL support is not enabled in this build");
#endif
}

bool nativeOpenClVoxelFp64CapabilityIsSufficient(
    std::uint64_t capabilityBits) noexcept {
    return capabilityBits != 0U;
}

} // namespace luw::cli
