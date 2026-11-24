#include "luw/luwvox_opencl.hpp"
#include "luw/luwvox_kriging_compat.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

float cpuIdw(const float* elevations, const float* distances,
             std::size_t count, float power) {
    double minimumDistance = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < count; ++index) {
        minimumDistance = std::min(
            minimumDistance,
            std::max(static_cast<double>(distances[index]), 1.0e-10));
    }
    double weighted = 0.0;
    double weightSum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double distance =
            std::max(static_cast<double>(distances[index]), 1.0e-10);
        const double weight =
            std::pow(minimumDistance / distance, static_cast<double>(power));
        weighted += weight * static_cast<double>(elevations[index]);
        weightSum += weight;
    }
    return weightSum > 0.0 ? static_cast<float>(weighted / weightSum) : 0.0F;
}

std::size_t cpuReflectedIndex(std::ptrdiff_t index, std::size_t size) {
    if (size <= 1U) return 0U;
    const std::ptrdiff_t extent = static_cast<std::ptrdiff_t>(size);
    while (index < 0 || index >= extent) {
        if (index < 0) index = -index - 1;
        if (index >= extent) index = 2 * extent - index - 1;
    }
    return static_cast<std::size_t>(index);
}

std::vector<float> cpuGaussian(const std::vector<float>& values,
                               std::size_t width, std::size_t height,
                               double sigma) {
    if (sigma <= 0.0 || values.empty()) return values;
    const int radius =
        std::max(1, static_cast<int>(4.0 * sigma + 0.5));
    std::vector<double> weights(static_cast<std::size_t>(radius * 2 + 1));
    double sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const double scaled =
            static_cast<double>(offset) / sigma;
        const double weight = std::exp(-0.5 * scaled * scaled);
        weights[static_cast<std::size_t>(offset + radius)] = weight;
        sum += weight;
    }
    for (double& weight : weights) weight /= sum;

    std::vector<double> temporary(values.size(), 0.0);
    std::vector<float> result(values.size(), 0.0F);
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            double value = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                value += weights[static_cast<std::size_t>(offset + radius)] *
                         static_cast<double>(values[row * width +
                             cpuReflectedIndex(
                                 static_cast<std::ptrdiff_t>(column) + offset,
                                 width)]);
            }
            temporary[row * width + column] = value;
        }
    }
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            double value = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                value += weights[static_cast<std::size_t>(offset + radius)] *
                         temporary[cpuReflectedIndex(
                                       static_cast<std::ptrdiff_t>(row) + offset,
                                       height) *
                                       width +
                                   column];
            }
            result[row * width + column] = static_cast<float>(value);
        }
    }
    return result;
}

struct CpuNeighbor {
    double squaredDistance = 0.0;
    std::uint64_t index = 0U;
};

std::vector<CpuNeighbor> cpuNeighbors(const std::vector<double>& pointsXY,
                                      double queryX, double queryY,
                                      std::size_t count) {
    std::vector<CpuNeighbor> candidates(pointsXY.size() / 2U);
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        const double dx = queryX - pointsXY[index * 2U];
        const double dy = queryY - pointsXY[index * 2U + 1U];
        candidates[index] = {dx * dx + dy * dy,
                             static_cast<std::uint64_t>(index)};
    }
    count = std::min(count, candidates.size());
    const auto compare = [](const CpuNeighbor& left, const CpuNeighbor& right) {
        return left.squaredDistance < right.squaredDistance ||
               (left.squaredDistance == right.squaredDistance &&
                left.index < right.index);
    };
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<std::ptrdiff_t>(count),
                      candidates.end(), compare);
    candidates.resize(count);
    return candidates;
}

float cpuIdwFromPoints(const std::vector<double>& elevations,
                       const std::vector<CpuNeighbor>& neighbors,
                       double power) {
    double minimumDistance = std::numeric_limits<double>::infinity();
    for (const CpuNeighbor& neighbor : neighbors) {
        minimumDistance = std::min(
            minimumDistance,
            std::max(std::sqrt(neighbor.squaredDistance), 1.0e-10));
    }
    double weighted = 0.0;
    double weightSum = 0.0;
    for (const CpuNeighbor& neighbor : neighbors) {
        const double distance =
            std::max(std::sqrt(neighbor.squaredDistance), 1.0e-10);
        const double weight = std::pow(minimumDistance / distance, power);
        weighted += weight * elevations[static_cast<std::size_t>(neighbor.index)];
        weightSum += weight;
    }
    return static_cast<float>(weighted / weightSum);
}

std::vector<double> regularPoints(std::size_t width, std::size_t height,
                                  double x0, double y0, double dx,
                                  double dy) {
    std::vector<double> points(width * height * 2U);
    for (std::size_t row = 0U; row < height; ++row) {
        for (std::size_t column = 0U; column < width; ++column) {
            const std::size_t index = row * width + column;
            points[index * 2U] = x0 + static_cast<double>(column) * dx;
            points[index * 2U + 1U] = y0 + static_cast<double>(row) * dy;
        }
    }
    return points;
}

struct TestVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

std::vector<double> cpuTerrainMesh(
    const luw::cli::OpenClTerrainMeshRequest& request) {
    std::vector<double> result;
    const auto x = [&](std::size_t column) {
        return request.xMin + (request.xMax - request.xMin) *
            static_cast<double>(column) /
            static_cast<double>(request.width - 1U);
    };
    const auto y = [&](std::size_t row) {
        return request.yMin + (request.yMax - request.yMin) *
            static_cast<double>(row) /
            static_cast<double>(request.height - 1U);
    };
    const auto append = [&](const TestVec3& a, const TestVec3& b,
                            const TestVec3& c) {
        result.insert(result.end(),
                      {a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z});
    };
    const auto quad = [&](const TestVec3& a, const TestVec3& b,
                          const TestVec3& c, const TestVec3& d) {
        append(a, b, c);
        append(a, c, d);
    };
    const auto top = [&](std::size_t row, std::size_t column) {
        return TestVec3{x(column), y(row), request.baseHeight +
            request.elevations[row * request.width + column]};
    };
    const auto bottom = [&](const TestVec3& value) {
        return TestVec3{value.x, value.y, request.baseHeight};
    };
    for (std::size_t row = 0U; row + 1U < request.height; ++row) {
        for (std::size_t column = 0U; column + 1U < request.width;
             ++column) {
            const TestVec3 v0 = top(row, column);
            const TestVec3 v1 = top(row, column + 1U);
            const TestVec3 v2 = top(row + 1U, column);
            const TestVec3 v3 = top(row + 1U, column + 1U);
            append(v0, v1, v2);
            append(v1, v3, v2);
            append(bottom(v0), bottom(v2), bottom(v1));
            append(bottom(v1), bottom(v2), bottom(v3));
        }
    }
    for (std::size_t row = 0U; row + 1U < request.height; ++row) {
        const TestVec3 left0 = top(row, 0U);
        const TestVec3 left1 = top(row + 1U, 0U);
        quad(left0, left1, bottom(left1), bottom(left0));
        const TestVec3 right0 = top(row, request.width - 1U);
        const TestVec3 right1 = top(row + 1U, request.width - 1U);
        quad(right0, bottom(right0), bottom(right1), right1);
    }
    for (std::size_t column = 0U; column + 1U < request.width; ++column) {
        const TestVec3 front0 = top(0U, column);
        const TestVec3 front1 = top(0U, column + 1U);
        quad(front0, bottom(front0), bottom(front1), front1);
        const TestVec3 back0 = top(request.height - 1U, column);
        const TestVec3 back1 = top(request.height - 1U, column + 1U);
        quad(back0, back1, bottom(back1), bottom(back0));
    }
    return result;
}

} // namespace

int main() {
    if (luw::cli::nativeOpenClVoxelFp64CapabilityIsSufficient(0U) ||
        !luw::cli::nativeOpenClVoxelFp64CapabilityIsSufficient(1U)) {
        return fail("OpenCL voxel FP64 capability gate changed");
    }
    const std::array<double, 3> batchConstant{5.0, 5.0 + 4.0e-13,
                                              5.0 - 4.0e-13};
    if (!luw::cli::detail::krigingBatchConstant(batchConstant) ||
        !luw::cli::detail::krigingRowwiseAllClose(batchConstant)) {
        return fail("strict near-constant samples must bypass the primary solve");
    }

    const std::array<double, 3> fallbackOnly{5.0, 5.0 + 1.0e-6,
                                             5.0 - 1.0e-6};
    if (luw::cli::detail::krigingBatchConstant(fallbackOnly) ||
        !luw::cli::detail::krigingRowwiseAllClose(fallbackOnly)) {
        return fail("fallback-only near-constant samples must reach the primary solve");
    }

    for (const int invalidIndex : {-1, std::numeric_limits<int>::max()}) {
        try {
            (void)luw::cli::requireNativeOpenClVoxelBackend(invalidIndex);
            return fail("invalid OpenCL GPU device index was accepted");
        } catch (const luw::cli::OpenClKrigingError& exception) {
            const std::string message = exception.what();
            if (message.find(std::to_string(invalidIndex)) == std::string::npos ||
                message.find("available") == std::string::npos) {
                std::cerr << "device-index error omitted requested/available details: "
                          << message << '\n';
                return 1;
            }
        }
    }

    try {
        luw::cli::OpenClKrigingRequest invalidRequest;
        invalidRequest.deviceIndex = std::numeric_limits<int>::max();
        (void)luw::cli::runNativeOpenClKriging(invalidRequest);
        return fail("OpenCL kriging request ignored its device index");
    } catch (const luw::cli::OpenClKrigingError& exception) {
        const std::string message = exception.what();
        if (message.find(std::to_string(std::numeric_limits<int>::max())) ==
                std::string::npos ||
            message.find("available") == std::string::npos) {
            std::cerr << "request device-index error omitted requested/available details: "
                      << message << '\n';
            return 1;
        }
    }

    try {
        constexpr std::size_t queries = 3U;
        constexpr std::size_t neighbors = 4U;
        luw::cli::OpenClKrigingRequest request;
        request.deviceIndex = 0;
        request.queryCount = queries;
        request.neighborCount = neighbors;
        request.range = 3.0F;
        request.partialSill = 1.0F;
        request.nugget = 0.02F;
        request.fallbackPower = 2.0F;
        const float points[] = {
            0, 0, 1, 0, 0, 1, 1, 1,
            0, 0, 1, 0, 0, 1, 1, 1,
            0, 0, 1, 0, 0, 1, 1, 1,
        };
        request.localPointsXY.assign(std::begin(points), std::end(points));
        request.localElevations = {
            1.0F, 2.0F, 3.0F, 4.0F,
            5.0F, 5.000040F, 4.999960F, 5.000049F,
            0.0F, 2.0F, 4.0F, 6.0F,
        };
        constexpr float diagonal = 0.7071067811865476F;
        request.localDistances = {
            0.0F, 1.0F, 1.0F, 1.41421356237F,
            diagonal, diagonal, diagonal, diagonal,
            diagonal, diagonal, diagonal, diagonal,
        };
        const bool expectedWarmup =
            luw::cli::nativeOpenClVoxelRequiresWarmup(request.deviceIndex);
        const luw::cli::OpenClKrigingResult result =
            luw::cli::runNativeOpenClKriging(request);
        if (result.predictions.size() != queries || result.device.empty() ||
            result.deviceIndex != request.deviceIndex) {
            std::cerr << "invalid OpenCL kriging result metadata\n";
            return 1;
        }
        if (expectedWarmup && !result.compiledThisCall) {
            std::cerr << "first OpenCL call did not report kernel compilation\n";
            return 1;
        }
        if (std::abs(result.predictions[0] - 1.0F) > 1.0e-6F) {
            std::cerr << "exact-hit GPU kriging behavior changed\n";
            return 1;
        }
        constexpr float expectedPrimarySolve =
            (5.0F + 5.000040F + 4.999960F + 5.000049F) / 4.0F;
        if (std::abs(result.predictions[1] - expectedPrimarySolve) > 2.0e-5F ||
            std::abs(result.predictions[1] - 5.0F) <= 2.0e-6F) {
            std::cerr << "near-constant GPU kriging bypassed the primary solve: "
                      << result.predictions[1] << '\n';
            return 1;
        }
        if (std::abs(result.predictions[2] - 3.0F) > 1.0e-3F) {
            std::cerr << "symmetric CPU/GPU kriging parity failed: "
                      << result.predictions[2] << '\n';
            return 1;
        }

        luw::cli::OpenClKrigingRequest singular;
        singular.deviceIndex = request.deviceIndex;
        singular.queryCount = 1U;
        singular.neighborCount = 2U;
        singular.localPointsXY = {0.0F, 0.0F, 0.0F, 0.0F};
        singular.localElevations = {0.0F, 10.0F};
        singular.localDistances = {1.0F, 2.0F};
        singular.range = 3.0F;
        singular.partialSill = 1.0F;
        singular.nugget = 0.0F;
        singular.fallbackPower = 2.0F;
        const luw::cli::OpenClKrigingResult singularResult =
            luw::cli::runNativeOpenClKriging(singular);
        if (singularResult.predictions.size() != 1U ||
            std::abs(singularResult.predictions.front() - 2.0F) > 1.0e-5F) {
            std::cerr << "singular GPU kriging did not use the on-device IDW fallback\n";
            return 1;
        }
        if (singularResult.compiledThisCall ||
            luw::cli::nativeOpenClVoxelRequiresWarmup(request.deviceIndex)) {
            std::cerr << "per-device OpenCL backend cache was not reused\n";
            return 1;
        }

        luw::cli::OpenClIdwRequest idw;
        idw.deviceIndex = request.deviceIndex;
        idw.queryCount = 3U;
        idw.neighborCount = 4U;
        idw.power = 2.0F;
        idw.elevations = {
            7.0F, 9.0F, 100.0F, 200.0F,
            1.0F, 3.0F, 6.0F, 10.0F,
            2.0F, -1.0F, 8.0F, 4.0F,
        };
        idw.distances = {
            0.0F, 0.0F, 1.0F, 2.0F,
            1.0F, 2.0F, 3.0F, 4.0F,
            0.5F, 1.5F, 2.5F, 3.5F,
        };
        const luw::cli::OpenClIdwResult idwResult =
            luw::cli::runNativeOpenClIdw(idw);
        if (idwResult.predictions.size() != idw.queryCount ||
            idwResult.deviceIndex != idw.deviceIndex ||
            idwResult.device != result.device || idwResult.compiledThisCall) {
            std::cerr << "invalid OpenCL IDW result/cache metadata\n";
            return 1;
        }
        for (std::size_t query = 0; query < idw.queryCount; ++query) {
            const std::size_t offset = query * idw.neighborCount;
            const float expected = cpuIdw(idw.elevations.data() + offset,
                                          idw.distances.data() + offset,
                                          idw.neighborCount, idw.power);
            if (std::abs(idwResult.predictions[query] - expected) > 2.0e-5F) {
                std::cerr << "OpenCL IDW CPU parity failed at query " << query
                          << ": " << idwResult.predictions[query]
                          << " vs " << expected << '\n';
                return 1;
            }
        }
        bool malformedIdwRejected = false;
        try {
            luw::cli::OpenClIdwRequest malformed = idw;
            malformed.distances.pop_back();
            (void)luw::cli::runNativeOpenClIdw(malformed);
        } catch (const luw::cli::OpenClKrigingError& exception) {
            malformedIdwRejected =
                std::string(exception.what()).find("buffer dimensions") !=
                std::string::npos;
        }
        if (!malformedIdwRejected) {
            std::cerr << "malformed OpenCL IDW buffers were accepted\n";
            return 1;
        }
        bool invalidIdwPowerRejected = false;
        try {
            luw::cli::OpenClIdwRequest invalidPower = idw;
            invalidPower.power = 0.0F;
            (void)luw::cli::runNativeOpenClIdw(invalidPower);
        } catch (const luw::cli::OpenClKrigingError& exception) {
            invalidIdwPowerRejected =
                std::string(exception.what()).find("power") !=
                std::string::npos;
        }
        if (!invalidIdwPowerRejected) {
            std::cerr << "invalid OpenCL IDW power was accepted\n";
            return 1;
        }

        luw::cli::OpenClGaussianRequest gaussian;
        gaussian.deviceIndex = request.deviceIndex;
        gaussian.width = 4U;
        gaussian.height = 3U;
        gaussian.sigma = 0.8F;
        gaussian.values = {
            0.0F, 1.0F, 4.0F, 9.0F,
            2.0F, 3.0F, 8.0F, 7.0F,
            -1.0F, 5.0F, 6.0F, 10.0F,
        };
        const std::vector<float> gaussianExpected = cpuGaussian(
            gaussian.values, gaussian.width, gaussian.height, gaussian.sigma);
        const luw::cli::OpenClGaussianResult gaussianResult =
            luw::cli::runNativeOpenClGaussian(gaussian);
        if (gaussianResult.values.size() != gaussian.values.size() ||
            gaussianResult.deviceIndex != gaussian.deviceIndex ||
            gaussianResult.device != result.device ||
            gaussianResult.compiledThisCall) {
            std::cerr << "invalid OpenCL Gaussian result/cache metadata\n";
            return 1;
        }
        for (std::size_t index = 0; index < gaussianExpected.size(); ++index) {
            if (std::abs(gaussianResult.values[index] -
                         gaussianExpected[index]) > 2.0e-4F) {
                std::cerr << "OpenCL reflected Gaussian CPU parity failed at "
                          << index << ": " << gaussianResult.values[index]
                          << " vs " << gaussianExpected[index] << '\n';
                return 1;
            }
        }

        luw::cli::OpenClGaussianRequest boundarySigma = gaussian;
        boundarySigma.sigma = 0.624999999;
        for (float& value : boundarySigma.values) value *= 10000.0F;
        const std::vector<float> boundaryExpected = cpuGaussian(
            boundarySigma.values, boundarySigma.width, boundarySigma.height,
            boundarySigma.sigma);
        const luw::cli::OpenClGaussianResult boundaryResult =
            luw::cli::runNativeOpenClGaussian(boundarySigma);
        for (std::size_t index = 0; index < boundaryExpected.size(); ++index) {
            if (std::abs(boundaryResult.values[index] -
                         boundaryExpected[index]) > 2.0e-2F) {
                std::cerr << "OpenCL Gaussian double-sigma radius boundary "
                             "changed at "
                          << index << '\n';
                return 1;
            }
        }

        luw::cli::OpenClGaussianRequest noSmoothing = gaussian;
        noSmoothing.sigma = 0.0F;
        const luw::cli::OpenClGaussianResult unchanged =
            luw::cli::runNativeOpenClGaussian(noSmoothing);
        if (unchanged.values != gaussian.values || unchanged.compiledThisCall) {
            std::cerr << "sigma<=0 OpenCL Gaussian request changed its values\n";
            return 1;
        }

        bool malformedGaussianRejected = false;
        try {
            luw::cli::OpenClGaussianRequest malformed = gaussian;
            malformed.values.pop_back();
            (void)luw::cli::runNativeOpenClGaussian(malformed);
        } catch (const luw::cli::OpenClKrigingError& exception) {
            malformedGaussianRejected =
                std::string(exception.what()).find("buffer dimensions") !=
                std::string::npos;
        }
        if (!malformedGaussianRejected) {
            std::cerr << "malformed OpenCL Gaussian buffers were accepted\n";
            return 1;
        }
        bool nonfiniteGaussianRejected = false;
        try {
            luw::cli::OpenClGaussianRequest nonfinite = gaussian;
            nonfinite.sigma = std::numeric_limits<double>::infinity();
            (void)luw::cli::runNativeOpenClGaussian(nonfinite);
        } catch (const luw::cli::OpenClKrigingError& exception) {
            nonfiniteGaussianRejected =
                std::string(exception.what()).find("sigma") !=
                std::string::npos;
        }
        if (!nonfiniteGaussianRejected) {
            std::cerr << "non-finite OpenCL Gaussian sigma was accepted\n";
            return 1;
        }

        bool gaussianOverflowRejected = false;
        try {
            luw::cli::OpenClGaussianRequest oversized;
            oversized.deviceIndex = request.deviceIndex;
            oversized.width = std::numeric_limits<std::size_t>::max();
            oversized.height = 2U;
            oversized.sigma = 1.0F;
            (void)luw::cli::runNativeOpenClGaussian(oversized);
        } catch (const luw::cli::OpenClKrigingError& exception) {
            gaussianOverflowRejected =
                std::string(exception.what()).find("overflow") !=
                std::string::npos;
        }
        if (!gaussianOverflowRejected) {
            std::cerr << "overflowing OpenCL Gaussian dimensions were accepted\n";
            return 1;
        }

        luw::cli::OpenClTerrainInterpolationRequest terrain;
        terrain.deviceIndex = request.deviceIndex;
        terrain.pointsXY = regularPoints(5U, 4U, 1000000.0, 2000000.0,
                                         2.0, 3.0);
        terrain.elevations.resize(20U);
        for (std::size_t index = 0U; index < terrain.elevations.size(); ++index) {
            terrain.elevations[index] =
                static_cast<double>((index * 7U) % 13U) +
                static_cast<double>(index) * 0.125;
        }
        terrain.width = 3U;
        terrain.height = 2U;
        terrain.neighborCount = 6U;
        terrain.xMin = 1000001.0;
        terrain.xMax = 1000005.0;
        terrain.yMin = 2000001.5;
        terrain.yMax = 2000004.5;
        terrain.idwPower = 2.0;
        terrain.smoothSigma = 0.8;
        terrain.includeNeighborData = true;
        const luw::cli::OpenClTerrainInterpolationResult terrainResult =
            luw::cli::runNativeOpenClTerrainInterpolation(terrain);
        if (!terrainResult.usedRegularGridFastPath ||
            terrainResult.rawValues.size() != terrain.width * terrain.height ||
            terrainResult.values.size() != terrainResult.rawValues.size() ||
            terrainResult.neighborIndices.size() !=
                terrainResult.rawValues.size() * terrain.neighborCount ||
            terrainResult.neighborSquaredDistances.size() !=
                terrainResult.neighborIndices.size()) {
            return fail("resident regular-grid OpenCL terrain metadata is invalid");
        }
        std::vector<float> expectedRaw(terrainResult.rawValues.size());
        for (std::size_t query = 0U; query < expectedRaw.size(); ++query) {
            const std::size_t row = query / terrain.width;
            const std::size_t column = query % terrain.width;
            const double queryX = terrain.xMin +
                (terrain.xMax - terrain.xMin) * static_cast<double>(column) /
                static_cast<double>(terrain.width - 1U);
            const double queryY = terrain.yMin +
                (terrain.yMax - terrain.yMin) * static_cast<double>(row) /
                static_cast<double>(terrain.height - 1U);
            const std::vector<CpuNeighbor> expected = cpuNeighbors(
                terrain.pointsXY, queryX, queryY, terrain.neighborCount);
            expectedRaw[query] = cpuIdwFromPoints(
                terrain.elevations, expected, terrain.idwPower);
            for (std::size_t neighbor = 0U;
                 neighbor < terrain.neighborCount; ++neighbor) {
                const std::size_t offset =
                    query * terrain.neighborCount + neighbor;
                if (terrainResult.neighborIndices[offset] !=
                        expected[neighbor].index ||
                    terrainResult.neighborSquaredDistances[offset] !=
                        expected[neighbor].squaredDistance) {
                    std::cerr << "regular-grid GPU (distance^2,index) order "
                                 "changed at query " << query << " neighbor "
                              << neighbor << '\n';
                    return 1;
                }
            }
            if (std::abs(terrainResult.rawValues[query] -
                         expectedRaw[query]) > 2.0e-5F) {
                return fail("resident regular-grid GPU IDW parity failed");
            }
        }
        const std::vector<float> expectedSmoothed = cpuGaussian(
            expectedRaw, terrain.width, terrain.height, terrain.smoothSigma);
        for (std::size_t index = 0U; index < expectedSmoothed.size(); ++index) {
            if (std::abs(terrainResult.values[index] -
                         expectedSmoothed[index]) > 2.0e-4F) {
                return fail("resident GPU interpolation/Gaussian fusion parity failed");
            }
        }

        luw::cli::OpenClTerrainInterpolationRequest irregular = terrain;
        irregular.width = 4U;
        irregular.height = 3U;
        irregular.xMin = 999999.25;
        irregular.xMax = 1000009.75;
        irregular.yMin = 1999999.25;
        irregular.yMax = 2000010.25;
        irregular.neighborCount = 7U;
        irregular.smoothSigma = 0.0;
        irregular.pointsXY.erase(irregular.pointsXY.begin() + 14,
                                 irregular.pointsXY.begin() + 16);
        irregular.elevations.erase(irregular.elevations.begin() + 7);
        irregular.pointsXY[6] += 0.03125;
        const luw::cli::OpenClTerrainInterpolationResult irregularResult =
            luw::cli::runNativeOpenClTerrainInterpolation(irregular);
        if (irregularResult.usedRegularGridFastPath) {
            return fail("gapped/near-regular DEM bypassed uniform-bin CSR search");
        }
        for (std::size_t query = 0U; query < irregularResult.rawValues.size();
             ++query) {
            const std::size_t row = query / irregular.width;
            const std::size_t column = query % irregular.width;
            const double queryX = irregular.xMin +
                (irregular.xMax - irregular.xMin) *
                    static_cast<double>(column) /
                    static_cast<double>(irregular.width - 1U);
            const double queryY = irregular.yMin +
                (irregular.yMax - irregular.yMin) *
                    static_cast<double>(row) /
                    static_cast<double>(irregular.height - 1U);
            const std::vector<CpuNeighbor> expected = cpuNeighbors(
                irregular.pointsXY, queryX, queryY,
                irregular.neighborCount);
            for (std::size_t neighbor = 0U;
                 neighbor < irregular.neighborCount; ++neighbor) {
                const std::size_t offset =
                    query * irregular.neighborCount + neighbor;
                if (irregularResult.neighborIndices[offset] !=
                        expected[neighbor].index ||
                    irregularResult.neighborSquaredDistances[offset] !=
                        expected[neighbor].squaredDistance) {
                    return fail("uniform-bin CSR GPU kNN order/parity failed");
                }
            }
        }

        luw::cli::OpenClTerrainInterpolationRequest exactKriging = terrain;
        exactKriging.width = 1U;
        exactKriging.height = 1U;
        exactKriging.xMin = exactKriging.xMax = 1000004.0;
        exactKriging.yMin = exactKriging.yMax = 2000003.0;
        exactKriging.neighborCount = 4U;
        exactKriging.approach = luw::cli::OpenClTerrainApproach::Kriging;
        exactKriging.smoothSigma = 0.0;
        exactKriging.includeNeighborData = true;
        const auto exactKrigingResult =
            luw::cli::runNativeOpenClTerrainInterpolation(exactKriging);
        if (exactKrigingResult.neighborIndices.front() != 7U ||
            exactKrigingResult.neighborSquaredDistances.front() != 0.0 ||
            std::abs(exactKrigingResult.rawValues.front() -
                     static_cast<float>(exactKriging.elevations[7])) > 1.0e-6F) {
            return fail("resident GPU exact-hit kriging behavior changed");
        }
        bool excessiveKrigingNeighborsRejected = false;
        try {
            luw::cli::OpenClTerrainInterpolationRequest excessive = terrain;
            excessive.approach = luw::cli::OpenClTerrainApproach::Kriging;
            excessive.neighborCount = 17U;
            (void)luw::cli::runNativeOpenClTerrainInterpolation(excessive);
        } catch (const luw::cli::OpenClKrigingError& exception) {
            excessiveKrigingNeighborsRejected =
                std::string(exception.what()).find("1 to 16") !=
                std::string::npos;
        }
        if (!excessiveKrigingNeighborsRejected) {
            return fail("resident GPU kriging accepted more than 16 neighbors");
        }
        bool zeroNeighborsRejected = false;
        try {
            luw::cli::OpenClTerrainInterpolationRequest invalid = terrain;
            invalid.neighborCount = 0U;
            (void)luw::cli::runNativeOpenClTerrainInterpolation(invalid);
        } catch (const luw::cli::OpenClKrigingError& exception) {
            zeroNeighborsRejected =
                std::string(exception.what()).find("neighbor count") !=
                std::string::npos;
        }
        if (!zeroNeighborsRejected) {
            return fail("resident GPU terrain accepted zero neighbors");
        }
        bool nonfiniteCoordinatesRejected = false;
        try {
            luw::cli::OpenClTerrainInterpolationRequest invalid = terrain;
            invalid.pointsXY.front() =
                std::numeric_limits<double>::infinity();
            (void)luw::cli::runNativeOpenClTerrainInterpolation(invalid);
        } catch (const luw::cli::OpenClKrigingError& exception) {
            nonfiniteCoordinatesRejected =
                std::string(exception.what()).find("coordinates") !=
                std::string::npos;
        }
        if (!nonfiniteCoordinatesRejected) {
            return fail("resident GPU terrain accepted non-finite coordinates");
        }

        luw::cli::OpenClTerrainMeshRequest mesh;
        mesh.deviceIndex = request.deviceIndex;
        mesh.width = 3U;
        mesh.height = 2U;
        mesh.xMin = 10.0;
        mesh.xMax = 16.0;
        mesh.yMin = -2.0;
        mesh.yMax = 4.0;
        mesh.baseHeight = 7.0;
        mesh.elevations = {0.0, 1.0, 4.0, 2.0, 5.0, 9.0};
        const auto meshResult = luw::cli::runNativeOpenClTerrainMesh(mesh);
        const std::vector<double> expectedMesh = cpuTerrainMesh(mesh);
        if (meshResult.triangleVertices != expectedMesh ||
            meshResult.triangleVertices.size() != 20U * 9U) {
            return fail("GPU terrain mesh triangle values/order changed");
        }

        luw::cli::OpenClTerrainInterpolationRequest benchmark;
        benchmark.deviceIndex = request.deviceIndex;
        benchmark.pointsXY = regularPoints(64U, 64U, 0.0, 0.0, 2.0, 2.0);
        benchmark.elevations.resize(64U * 64U);
        for (std::size_t index = 0U; index < benchmark.elevations.size(); ++index) {
            benchmark.elevations[index] =
                std::sin(static_cast<double>(index) * 0.017);
        }
        benchmark.width = 128U;
        benchmark.height = 128U;
        benchmark.neighborCount = 8U;
        benchmark.xMin = 0.5;
        benchmark.xMax = 125.5;
        benchmark.yMin = 0.5;
        benchmark.yMax = 125.5;
        benchmark.idwPower = 2.0;
        const auto gpuStarted = std::chrono::steady_clock::now();
        const auto benchmarkResult =
            luw::cli::runNativeOpenClTerrainInterpolation(benchmark);
        const double gpuMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gpuStarted).count();
        const auto cpuStarted = std::chrono::steady_clock::now();
        double cpuChecksum = 0.0;
        for (std::size_t row = 0U; row < benchmark.height; ++row) {
            for (std::size_t column = 0U; column < benchmark.width; ++column) {
                const double queryX = 0.5 + 125.0 *
                    static_cast<double>(column) /
                    static_cast<double>(benchmark.width - 1U);
                const double queryY = 0.5 + 125.0 *
                    static_cast<double>(row) /
                    static_cast<double>(benchmark.height - 1U);
                cpuChecksum += cpuIdwFromPoints(
                    benchmark.elevations,
                    cpuNeighbors(benchmark.pointsXY, queryX, queryY,
                                 benchmark.neighborCount),
                    benchmark.idwPower);
            }
        }
        const double cpuMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpuStarted).count();
        if (benchmarkResult.values.size() !=
                benchmark.width * benchmark.height ||
            !std::isfinite(cpuChecksum)) {
            return fail("medium OpenCL terrain benchmark produced invalid data");
        }
        std::cout << "OpenCL resident terrain benchmark (64x64 DEM -> 128x128, k=8): "
                  << gpuMilliseconds << " ms; CPU brute-force reference (same 128x128 queries): "
                  << cpuMilliseconds << " ms\n";

        luw::cli::OpenClTerrainInterpolationRequest chunked = terrain;
        chunked.width = 257U;
        chunked.height = 257U;
        chunked.xMin = 999999.0;
        chunked.xMax = 1000009.0;
        chunked.yMin = 1999999.0;
        chunked.yMax = 2000010.0;
        chunked.neighborCount = 4U;
        chunked.smoothSigma = 0.6;
        chunked.includeNeighborData = false;
        const auto chunkedResult =
            luw::cli::runNativeOpenClTerrainInterpolation(chunked);
        if (chunkedResult.values.size() != 257U * 257U ||
            chunkedResult.rawValues.size() != chunkedResult.values.size()) {
            return fail("allocation-bounded multi-chunk terrain output is incomplete");
        }

        luw::cli::OpenClTerrainMeshRequest chunkedMesh;
        chunkedMesh.deviceIndex = request.deviceIndex;
        chunkedMesh.width = 130U;
        chunkedMesh.height = 130U;
        chunkedMesh.xMin = -3.0;
        chunkedMesh.xMax = 126.0;
        chunkedMesh.yMin = 4.0;
        chunkedMesh.yMax = 262.0;
        chunkedMesh.baseHeight = 9.0;
        chunkedMesh.elevations.resize(
            chunkedMesh.width * chunkedMesh.height);
        for (std::size_t index = 0U;
             index < chunkedMesh.elevations.size(); ++index) {
            chunkedMesh.elevations[index] =
                static_cast<double>(index % 97U) * 0.25;
        }
        const auto chunkedMeshResult =
            luw::cli::runNativeOpenClTerrainMesh(chunkedMesh);
        const std::vector<double> expectedChunkedMesh =
            cpuTerrainMesh(chunkedMesh);
        if (chunkedMeshResult.triangleVertices != expectedChunkedMesh) {
            return fail("allocation-bounded multi-chunk terrain mesh order changed");
        }

        try {
            const luw::cli::OpenClVoxelBackendInfo second =
                luw::cli::requireNativeOpenClVoxelBackend(1);
            if (second.device.empty() || second.deviceIndex != 1) {
                std::cerr << "nonzero OpenCL voxel device selection returned "
                             "invalid metadata\n";
                return 1;
            }
            luw::cli::OpenClTerrainInterpolationRequest secondTerrain = terrain;
            secondTerrain.deviceIndex = 1;
            const auto secondTerrainResult =
                luw::cli::runNativeOpenClTerrainInterpolation(secondTerrain);
            if (secondTerrainResult.neighborIndices !=
                    terrainResult.neighborIndices ||
                secondTerrainResult.neighborSquaredDistances !=
                    terrainResult.neighborSquaredDistances ||
                secondTerrainResult.values.size() != terrainResult.values.size()) {
                return fail("OpenCL device 1 terrain kNN/order parity failed");
            }
            luw::cli::OpenClTerrainMeshRequest secondMesh = mesh;
            secondMesh.deviceIndex = 1;
            if (luw::cli::runNativeOpenClTerrainMesh(secondMesh)
                    .triangleVertices != expectedMesh) {
                return fail("OpenCL device 1 terrain mesh order parity failed");
            }
            std::cout << "OpenCL voxel device 1: " << second.device << '\n';
        } catch (const luw::cli::OpenClKrigingError& exception) {
            const std::string message = exception.what();
            if (message.find("but only 1 OpenCL GPU device(s) are available") ==
                std::string::npos) {
                throw;
            }
        }

        std::cout << "OpenCL voxel device: " << result.device << '\n';
        return 0;
    } catch (const luw::cli::OpenClKrigingError& exception) {
        const std::string message = exception.what();
        if (message.find("but only 0 OpenCL GPU device(s) are available") !=
            std::string::npos) {
            std::cout << "OpenCL voxel test skipped: " << message << '\n';
            return 77;
        }
        if (message.find("does not support FP64") != std::string::npos) {
            std::cout << "OpenCL voxel strict FP64 rejection verified: "
                      << message << '\n';
            return 0;
        }
        std::cerr << "OpenCL voxel backend test failed: " << message << '\n';
        return 1;
    }
}
