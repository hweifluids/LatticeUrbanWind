#include "luw/luwbc_opencl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#ifndef LUW_HAS_OPENCL
#define LUW_HAS_OPENCL 0
#endif

#ifndef LUW_LUWBC_MAX_WORK_ITEMS
#define LUW_LUWBC_MAX_WORK_ITEMS (1U << 22U)
#endif

namespace {

constexpr std::size_t kConfiguredMaximumWorkItems =
    static_cast<std::size_t>(LUW_LUWBC_MAX_WORK_ITEMS);
static_assert(
    LUW_HAS_OPENCL == 0 ||
        (kConfiguredMaximumWorkItems >= 4U &&
         kConfiguredMaximumWorkItems <= 1024U),
    "luw_luwbc_opencl_tests must set LUW_LUWBC_MAX_WORK_ITEMS to a small "
    "positive value so the deterministic multi-chunk test stays lightweight");

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

bool close(float actual, float expected, float tolerance = 2.0e-5F) {
    return std::abs(actual - expected) <= tolerance;
}

bool closeDouble(double actual, double expected,
                 double tolerance = 2.0e-11) {
    return std::abs(actual - expected) <=
           tolerance * std::max(1.0, std::abs(expected));
}

bool backendAbsent(const std::string& message) {
    return message.find("only 0 OpenCL GPU device(s) are available") !=
               std::string::npos ||
           message.find("OpenCL support is not enabled") != std::string::npos;
}

std::vector<float> horizontalOracle(
    const luw::cli::OpenClLuwbcHorizontalRequest& request) {
    const std::size_t planes = request.componentCount * request.layerCount;
    std::vector<float> output(planes * request.targetPointCount);
    for (std::size_t plane = 0U; plane < planes; ++plane) {
        for (std::size_t target = 0U; target < request.targetPointCount;
             ++target) {
            double value = 0.0;
            for (std::uint32_t corner = 0U;
                 corner < request.counts[target]; ++corner) {
                value += request.weights[target * 3U + corner] *
                         request.values[
                             plane * request.sourceLayerSize +
                             request.indices[target * 3U + corner]];
            }
            output[plane * request.targetPointCount + target] =
                static_cast<float>(value);
        }
    }
    return output;
}

std::vector<double> demIdwOracle(
    const luw::cli::OpenClLuwbcDemIdwRequest& request) {
    std::vector<double> output(request.targetX.size());
    for (std::size_t target = 0U; target < request.targetX.size(); ++target) {
        long double weighted = 0.0L;
        long double weights = 0.0L;
        for (std::size_t entry = 0U; entry < request.neighborCount; ++entry) {
            const std::size_t source = request.neighborIndices[
                target * request.neighborCount + entry];
            const double dx = request.sourceX[source] - request.targetX[target];
            const double dy = request.sourceY[source] - request.targetY[target];
            const double distance =
                std::max(std::sqrt(dx * dx + dy * dy), 1.0e-10);
            const double weight = 1.0 / (distance * distance);
            weighted += weight * request.sourceValues[source];
            weights += weight;
        }
        output[target] = static_cast<double>(weighted / weights);
    }
    return output;
}

bool solveSmallSystemOracle(std::vector<double> matrix,
                            std::vector<double> right,
                            std::size_t dimension,
                            std::vector<double>& solution) {
    solution.assign(dimension, 0.0);
    for (std::size_t pivot = 0U; pivot < dimension; ++pivot) {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1U; row < dimension; ++row) {
            if (std::abs(matrix[row * dimension + pivot]) >
                std::abs(matrix[best * dimension + pivot])) {
                best = row;
            }
        }
        if (std::abs(matrix[best * dimension + pivot]) <= 1.0e-18) {
            return false;
        }
        if (best != pivot) {
            for (std::size_t column = pivot; column < dimension; ++column) {
                std::swap(matrix[pivot * dimension + column],
                          matrix[best * dimension + column]);
            }
            std::swap(right[pivot], right[best]);
        }
        const double diagonal = matrix[pivot * dimension + pivot];
        for (std::size_t column = pivot; column < dimension; ++column) {
            matrix[pivot * dimension + column] /= diagonal;
        }
        right[pivot] /= diagonal;
        for (std::size_t row = 0U; row < dimension; ++row) {
            if (row == pivot) continue;
            const double factor = matrix[row * dimension + pivot];
            if (factor == 0.0) continue;
            for (std::size_t column = pivot; column < dimension; ++column) {
                matrix[row * dimension + column] -=
                    factor * matrix[pivot * dimension + column];
            }
            right[row] -= factor * right[pivot];
        }
    }
    solution = std::move(right);
    return true;
}

std::vector<float> mlsOracle(
    const luw::cli::OpenClLuwbcMlsRequest& request) {
    const std::size_t neighbors = request.neighborCount;
    const std::size_t basisDimension =
        neighbors >= 6U ? 6U : (neighbors >= 3U ? 3U : 1U);
    std::vector<float> output(request.targetX.size() * neighbors);
    for (std::size_t target = 0U; target < request.targetX.size(); ++target) {
        std::vector<double> distance(neighbors);
        double h = 0.0;
        for (std::size_t entry = 0U; entry < neighbors; ++entry) {
            const std::size_t source = request.neighborIndices[
                target * neighbors + entry];
            const double dx = request.sourceX[source] - request.targetX[target];
            const double dy = request.sourceY[source] - request.targetY[target];
            distance[entry] = std::sqrt(dx * dx + dy * dy);
            h = std::max(h, distance[entry]);
        }
        if (!std::isfinite(h) || h <= 1.0e-12) h = 1.0;
        h *= 1.000001;
        std::vector<double> kernel(neighbors);
        std::vector<std::vector<double>> basis(
            neighbors, std::vector<double>(basisDimension, 0.0));
        double kernelSum = 0.0;
        for (std::size_t entry = 0U; entry < neighbors; ++entry) {
            const std::size_t source = request.neighborIndices[
                target * neighbors + entry];
            const double xn =
                (request.sourceX[source] - request.targetX[target]) / h;
            const double yn =
                (request.sourceY[source] - request.targetY[target]) / h;
            const double radius = std::sqrt(xn * xn + yn * yn);
            const double compact = std::max(0.0, 1.0 - radius);
            kernel[entry] =
                std::pow(compact, 4.0) * (4.0 * radius + 1.0);
            kernelSum += kernel[entry];
            basis[entry][0] = 1.0;
            if (basisDimension >= 3U) {
                basis[entry][1] = xn;
                basis[entry][2] = yn;
            }
            if (basisDimension == 6U) {
                basis[entry][3] = xn * xn;
                basis[entry][4] = xn * yn;
                basis[entry][5] = yn * yn;
            }
        }
        if (!std::isfinite(kernelSum) || kernelSum <= 1.0e-14) {
            std::fill(kernel.begin(), kernel.end(), 1.0);
            kernelSum = static_cast<double>(neighbors);
        }
        std::vector<double> coefficients(neighbors);
        if (basisDimension == 1U) {
            for (std::size_t entry = 0U; entry < neighbors; ++entry) {
                coefficients[entry] = kernel[entry] / kernelSum;
            }
        } else {
            std::vector<double> matrix(
                basisDimension * basisDimension, 0.0);
            for (std::size_t entry = 0U; entry < neighbors; ++entry) {
                for (std::size_t row = 0U; row < basisDimension; ++row) {
                    for (std::size_t column = 0U;
                         column < basisDimension; ++column) {
                        matrix[row * basisDimension + column] +=
                            kernel[entry] * basis[entry][row] *
                            basis[entry][column];
                    }
                }
            }
            double trace = 0.0;
            for (std::size_t diagonal = 0U; diagonal < basisDimension;
                 ++diagonal) {
                trace += matrix[diagonal * basisDimension + diagonal];
            }
            const double regularization =
                1.0e-10 * trace / static_cast<double>(basisDimension) +
                1.0e-12;
            for (std::size_t diagonal = 0U; diagonal < basisDimension;
                 ++diagonal) {
                matrix[diagonal * basisDimension + diagonal] +=
                    regularization;
            }
            std::vector<double> right(basisDimension, 0.0);
            right[0] = 1.0;
            std::vector<double> solved;
            if (solveSmallSystemOracle(matrix, right, basisDimension, solved)) {
                double sum = 0.0;
                for (std::size_t entry = 0U; entry < neighbors; ++entry) {
                    coefficients[entry] = kernel[entry] * std::inner_product(
                        basis[entry].begin(), basis[entry].end(),
                        solved.begin(), 0.0);
                    sum += coefficients[entry];
                }
                if (!std::isfinite(sum) || std::abs(sum) <= 1.0e-14) {
                    for (std::size_t entry = 0U; entry < neighbors; ++entry) {
                        coefficients[entry] = kernel[entry] / kernelSum;
                    }
                } else {
                    for (double& value : coefficients) value /= sum;
                    double negativeMass = 0.0;
                    for (const double value : coefficients) {
                        if (value < 0.0) negativeMass += std::abs(value);
                    }
                    if (negativeMass > 0.08) {
                        const double alpha = std::min(
                            1.0, (negativeMass - 0.08) / 0.30);
                        double limitedSum = 0.0;
                        for (std::size_t entry = 0U; entry < neighbors;
                             ++entry) {
                            coefficients[entry] =
                                (1.0 - alpha) * coefficients[entry] +
                                alpha * kernel[entry] / kernelSum;
                            limitedSum += coefficients[entry];
                        }
                        for (double& value : coefficients) {
                            value /= limitedSum;
                        }
                    }
                }
            } else {
                for (std::size_t entry = 0U; entry < neighbors; ++entry) {
                    coefficients[entry] = kernel[entry] / kernelSum;
                }
            }
        }
        for (std::size_t entry = 0U; entry < neighbors; ++entry) {
            output[target * neighbors + entry] =
                static_cast<float>(coefficients[entry]);
        }
    }
    return output;
}

int verifyHorizontal(std::size_t componentCount) {
    luw::cli::OpenClLuwbcHorizontalRequest request;
    request.deviceIndex = 0;
    request.sourceLayerSize = 4U;
    request.layerCount = 2U;
    request.targetPointCount = 3U;
    request.componentCount = componentCount;
    const std::size_t planes = componentCount * request.layerCount;
    request.values.resize(planes * request.sourceLayerSize);
    for (std::size_t component = 0U; component < componentCount; ++component) {
        for (std::size_t layer = 0U; layer < request.layerCount; ++layer) {
            for (std::size_t point = 0U; point < request.sourceLayerSize;
                 ++point) {
                request.values[(component * request.layerCount + layer) *
                                   request.sourceLayerSize +
                               point] =
                    static_cast<float>(component * 100U + layer * 10U + point);
            }
        }
    }
    request.indices = {
        2U, 0U, 0U,
        0U, 1U, 2U,
        1U, 2U, 3U,
    };
    request.weights = {
        1.0, 0.0, 0.0,
        0.20000000000000004, 0.29999999999999993, 0.5,
        0.25, 0.25, 0.5,
    };
    request.counts = {1U, 3U, 3U};
    const std::vector<float> expected = horizontalOracle(request);
    const luw::cli::OpenClLuwbcResult result =
        luw::cli::runNativeOpenClLuwbcHorizontal(request);
    if (result.values.size() != expected.size() || result.device.empty() ||
        result.deviceIndex != request.deviceIndex) {
        return fail("invalid fused horizontal result metadata/dimensions");
    }
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        if (!close(result.values[index], expected[index])) {
            return fail("fused horizontal CPU/GPU oracle mismatch at " +
                        std::to_string(index) + ": actual=" +
                        std::to_string(result.values[index]) +
                        ", expected=" + std::to_string(expected[index]));
        }
    }
    return 0;
}

int verifyVertical() {
    luw::cli::OpenClLuwbcVerticalRequest request;
    request.deviceIndex = 0;
    request.pointCount = 2U;
    request.componentCount = 4U;
    request.sourceLevels = {0.0, 7.0, 19.0};
    request.targetLevels = {
        -5.0, 0.0, 7.0 - 5.0e-7, 7.0 + 5.0e-7, 13.0, 19.0, 30.0};
    request.values.resize(request.componentCount * request.sourceLevels.size() *
                          request.pointCount);
    for (std::size_t component = 0U; component < request.componentCount;
         ++component) {
        for (std::size_t level = 0U; level < request.sourceLevels.size();
             ++level) {
            for (std::size_t point = 0U; point < request.pointCount; ++point) {
                request.values[(component * request.sourceLevels.size() + level) *
                                   request.pointCount +
                               point] =
                    static_cast<float>(component * 100U + level * 10U + point);
            }
        }
    }
    const luw::cli::OpenClLuwbcResult result =
        luw::cli::runNativeOpenClLuwbcVertical(request);
    const std::vector<float> levelValues{
        0.0F, 0.0F, 10.0F, 10.0F, 15.0F, 20.0F, 20.0F};
    const std::size_t expectedCount = request.componentCount *
                                      request.targetLevels.size() *
                                      request.pointCount;
    if (result.values.size() != expectedCount || result.deviceIndex != 0) {
        return fail("invalid fused vertical result metadata/dimensions");
    }
    for (std::size_t component = 0U; component < request.componentCount;
         ++component) {
        for (std::size_t level = 0U; level < request.targetLevels.size(); ++level) {
            for (std::size_t point = 0U; point < request.pointCount; ++point) {
                const float expected =
                    static_cast<float>(component * 100U + point) +
                    levelValues[level];
                const std::size_t index =
                    (component * request.targetLevels.size() + level) *
                        request.pointCount +
                    point;
                if (!close(result.values[index], expected)) {
                    return fail("fused vertical clamp/lerp mismatch at " +
                                std::to_string(index));
                }
            }
        }
    }
    return 0;
}

int verifyResamplePipelineChunks() {
    // Four output planes and (2 * ceiling/4 + 1) targets force at least three
    // submissions without timing assertions.  The test target supplies a
    // small ceiling while production retains the 2^22 default.
    const std::size_t pointsPerSubmission =
        kConfiguredMaximumWorkItems / 4U;
    const std::size_t pointCount = pointsPerSubmission * 2U + 1U;

    luw::cli::OpenClLuwbcHorizontalRequest horizontal;
    horizontal.deviceIndex = 0;
    horizontal.componentCount = 4U;
    horizontal.layerCount = 1U;
    horizontal.sourceLayerSize = 3U;
    horizontal.targetPointCount = pointCount;
    horizontal.values.resize(4U * horizontal.sourceLayerSize);
    for (std::size_t plane = 0U; plane < 4U; ++plane) {
        horizontal.values[plane * 3U] = static_cast<float>(plane * 20U + 1U);
        horizontal.values[plane * 3U + 1U] =
            static_cast<float>(plane * 20U + 3U);
        horizontal.values[plane * 3U + 2U] =
            static_cast<float>(plane * 20U + 8U);
    }
    horizontal.indices.resize(pointCount * 3U);
    horizontal.weights.resize(pointCount * 3U);
    horizontal.counts.assign(pointCount, 3U);
    for (std::size_t point = 0U; point < pointCount; ++point) {
        horizontal.indices[point * 3U] = 0U;
        horizontal.indices[point * 3U + 1U] = 1U;
        horizontal.indices[point * 3U + 2U] = 2U;
        const double firstWeight =
            0.1 + 0.01 * static_cast<double>(point % 7U);
        horizontal.weights[point * 3U] = firstWeight;
        horizontal.weights[point * 3U + 1U] = 0.25;
        horizontal.weights[point * 3U + 2U] = 0.75 - firstWeight;
    }
    const luw::cli::OpenClLuwbcResult horizontalResult =
        luw::cli::runNativeOpenClLuwbcHorizontal(horizontal);
    if (horizontalResult.values.size() != 4U * pointCount) {
        return fail("horizontal ping-pong multi-chunk dimensions changed");
    }
    for (std::size_t plane = 0U; plane < 4U; ++plane) {
        for (std::size_t point = 0U; point < pointCount; ++point) {
            double expected = 0.0;
            for (std::size_t corner = 0U; corner < 3U; ++corner) {
                expected += horizontal.weights[point * 3U + corner] *
                            horizontal.values[plane * 3U + corner];
            }
            const std::size_t offset = plane * pointCount + point;
            if (!close(horizontalResult.values[offset],
                       static_cast<float>(expected))) {
                return fail(
                    "horizontal ping-pong multi-chunk ordering mismatch at " +
                    std::to_string(offset));
            }
        }
    }

    luw::cli::OpenClLuwbcVerticalRequest vertical;
    vertical.deviceIndex = 0;
    vertical.componentCount = 4U;
    vertical.pointCount = pointCount;
    vertical.sourceLevels = {0.0, 10.0};
    vertical.targetLevels = {std::nextafter(1.0e-6, 0.0)};
    vertical.values.resize(8U * pointCount);
    for (std::size_t component = 0U; component < 4U; ++component) {
        for (std::size_t point = 0U; point < pointCount; ++point) {
            const float first =
                static_cast<float>(component * 1000U + point);
            vertical.values[(component * 2U) * pointCount + point] = first;
            vertical.values[(component * 2U + 1U) * pointCount + point] =
                first + 1.0e8F;
        }
    }
    const auto verifyVerticalValues =
        [&](const luw::cli::OpenClLuwbcResult& actual,
            bool snappedToLower) -> int {
            if (actual.values.size() != 4U * pointCount) {
                return fail("vertical ping-pong multi-chunk dimensions changed");
            }
            for (std::size_t component = 0U; component < 4U; ++component) {
                for (std::size_t point = 0U; point < pointCount; ++point) {
                    const float first = vertical.values[
                        (component * 2U) * pointCount + point];
                    const float second = vertical.values[
                        (component * 2U + 1U) * pointCount + point];
                    const float expected = snappedToLower
                        ? first
                        : static_cast<float>(
                              static_cast<double>(first) +
                              (vertical.targetLevels.front() / 10.0) *
                                  static_cast<double>(second - first));
                    const std::size_t offset = component * pointCount + point;
                    if (!close(actual.values[offset], expected, 1.0e-4F)) {
                        return fail(
                            "vertical ping-pong/cache ordering mismatch at " +
                            std::to_string(offset));
                    }
                }
            }
            return 0;
        };

    const luw::cli::OpenClLuwbcResult firstVertical =
        luw::cli::runNativeOpenClLuwbcVertical(vertical);
    if (const int status = verifyVerticalValues(firstVertical, true);
        status != 0) {
        return status;
    }
    // Exact same axes exercise the per-device LRU hit.
    const luw::cli::OpenClLuwbcResult cachedVertical =
        luw::cli::runNativeOpenClLuwbcVertical(vertical);
    if (const int status = verifyVerticalValues(cachedVertical, true);
        status != 0) {
        return status;
    }
    // One representable step crosses the strict 1e-6 snap threshold.  A cache
    // keyed less precisely than the complete axis bits would reuse stale plan
    // entries and return the lower source value here.
    vertical.targetLevels.front() = 1.0e-6;
    const luw::cli::OpenClLuwbcResult invalidatedVertical =
        luw::cli::runNativeOpenClLuwbcVertical(vertical);
    if (const int status = verifyVerticalValues(invalidatedVertical, false);
        status != 0) {
        return status;
    }

    std::cout << "OpenCL luwbc ping-pong pipeline verified with at least 3 "
                 "horizontal and vertical chunks\n";
    return 0;
}

int verifyDemIdw(int deviceIndex) {
    luw::cli::OpenClLuwbcDemIdwRequest request;
    request.deviceIndex = deviceIndex;
    request.neighborCount = 4U;
    request.sourceX = {0.0, 1.0, 0.0, 1.0, 0.0, 2.0};
    request.sourceY = {0.0, 0.0, 1.0, 1.0, 0.0, 0.5};
    request.sourceValues = {10.0, 20.0, 30.0, 45.0, 50.0, -5.0};
    request.targetX = {0.0, 0.4, 1.75};
    request.targetY = {0.0, 0.6, 0.4};
    // This is intentionally not re-sorted by the GPU.  The first target also
    // has two exact/coincident hits, locking the CPU distance-floor behavior.
    request.neighborIndices = {
        0U, 4U, 1U, 2U,
        2U, 0U, 3U, 1U,
        5U, 3U, 1U, 2U,
    };
    const std::vector<double> expected = demIdwOracle(request);
    const luw::cli::OpenClLuwbcDoubleResult result =
        luw::cli::runNativeOpenClLuwbcDemIdw(request);
    if (result.values.size() != expected.size() || result.device.empty() ||
        result.deviceIndex != deviceIndex) {
        return fail("invalid DEM IDW result metadata/dimensions");
    }
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        if (!closeDouble(result.values[index], expected[index])) {
            return fail("DEM IDW CPU/GPU oracle mismatch at " +
                        std::to_string(index) + ": actual=" +
                        std::to_string(result.values[index]) +
                        ", expected=" + std::to_string(expected[index]));
        }
    }
    if (!closeDouble(result.values.front(), 30.0)) {
        return fail("DEM IDW coincident-point distance-floor behavior changed");
    }
    return 0;
}

luw::cli::OpenClLuwbcMlsRequest makeMlsRequest(
    int deviceIndex, std::size_t neighborCount, bool degenerate) {
    luw::cli::OpenClLuwbcMlsRequest request;
    request.deviceIndex = deviceIndex;
    request.neighborCount = neighborCount;
    request.sourceX.resize(neighborCount);
    request.sourceY.resize(neighborCount);
    for (std::size_t index = 0U; index < neighborCount; ++index) {
        if (degenerate) {
            request.sourceX[index] = 2.0;
            request.sourceY[index] = -3.0;
        } else {
            request.sourceX[index] =
                static_cast<double>(index % 6U) * 0.73 +
                static_cast<double>((index / 6U) % 2U) * 0.11;
            request.sourceY[index] =
                static_cast<double>(index / 6U) * 0.81 +
                static_cast<double>((index * index) % 5U) * 0.037;
        }
    }
    request.targetX = degenerate ? std::vector<double>{2.0}
                                 : std::vector<double>{0.37, 2.12, -1.4};
    request.targetY = degenerate ? std::vector<double>{-3.0}
                                 : std::vector<double>{0.44, 1.37, 4.2};
    request.neighborIndices.resize(
        request.targetX.size() * neighborCount);
    for (std::size_t target = 0U; target < request.targetX.size(); ++target) {
        for (std::size_t entry = 0U; entry < neighborCount; ++entry) {
            std::size_t source = entry;
            if (target == 1U) {
                source = (entry * 5U + 1U) % neighborCount;
            } else if (target == 2U) {
                source = neighborCount - 1U - entry;
            }
            request.neighborIndices[target * neighborCount + entry] =
                static_cast<std::uint32_t>(source);
        }
    }
    return request;
}

int verifyMls(int deviceIndex) {
    for (const std::size_t neighborCount : {2U, 3U, 6U, 24U}) {
        const luw::cli::OpenClLuwbcMlsRequest request =
            makeMlsRequest(deviceIndex, neighborCount, false);
        const std::vector<float> expected = mlsOracle(request);
        const luw::cli::OpenClLuwbcResult result =
            luw::cli::runNativeOpenClLuwbcMlsWeights(request);
        if (result.values.size() != expected.size() || result.device.empty() ||
            result.deviceIndex != deviceIndex) {
            return fail("invalid MLS result metadata/dimensions for k=" +
                        std::to_string(neighborCount));
        }
        for (std::size_t index = 0U; index < expected.size(); ++index) {
            if (!close(result.values[index], expected[index], 8.0e-5F)) {
                return fail("MLS CPU/GPU oracle mismatch for k=" +
                            std::to_string(neighborCount) + " at " +
                            std::to_string(index) + ": actual=" +
                            std::to_string(result.values[index]) +
                            ", expected=" + std::to_string(expected[index]));
            }
        }
        for (std::size_t target = 0U; target < request.targetX.size();
             ++target) {
            double sum = 0.0;
            for (std::size_t entry = 0U; entry < neighborCount; ++entry) {
                sum += result.values[target * neighborCount + entry];
            }
            if (std::abs(sum - 1.0) > 2.0e-5) {
                return fail("MLS partition of unity changed for k=" +
                            std::to_string(neighborCount));
            }
        }
    }

    const luw::cli::OpenClLuwbcMlsRequest degenerate =
        makeMlsRequest(deviceIndex, 6U, true);
    const std::vector<float> expected = mlsOracle(degenerate);
    const luw::cli::OpenClLuwbcResult result =
        luw::cli::runNativeOpenClLuwbcMlsWeights(degenerate);
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        if (!close(result.values[index], expected[index], 8.0e-5F)) {
            return fail("degenerate/exact-hit MLS CPU/GPU oracle mismatch");
        }
    }
    return 0;
}

int verifyMediumSpatialBatch(int deviceIndex) {
    constexpr std::size_t targetCount = (1U << 15U) + 17U;
    luw::cli::OpenClLuwbcDemIdwRequest idw;
    idw.deviceIndex = deviceIndex;
    idw.neighborCount = 12U;
    for (std::size_t source = 0U; source < 24U; ++source) {
        idw.sourceX.push_back(static_cast<double>(source % 6U) * 0.9);
        idw.sourceY.push_back(static_cast<double>(source / 6U) * 1.1);
        idw.sourceValues.push_back(
            std::sin(static_cast<double>(source) * 0.37) * 80.0 +
            static_cast<double>(source));
    }
    idw.targetX.resize(targetCount);
    idw.targetY.resize(targetCount);
    idw.neighborIndices.resize(targetCount * idw.neighborCount);
    for (std::size_t target = 0U; target < targetCount; ++target) {
        idw.targetX[target] =
            static_cast<double>(target % 257U) * 0.019 - 0.7;
        idw.targetY[target] =
            static_cast<double>((target / 257U) % 131U) * 0.023 - 0.4;
        const std::size_t first = target % 13U;
        for (std::size_t entry = 0U; entry < idw.neighborCount; ++entry) {
            idw.neighborIndices[target * idw.neighborCount + entry] =
                static_cast<std::uint32_t>((first + entry) % 24U);
        }
    }
    const auto idwCpuStart = std::chrono::steady_clock::now();
    const std::vector<double> expectedIdw = demIdwOracle(idw);
    const auto idwCpuEnd = std::chrono::steady_clock::now();
    const auto idwStart = std::chrono::steady_clock::now();
    const luw::cli::OpenClLuwbcDoubleResult idwResult =
        luw::cli::runNativeOpenClLuwbcDemIdw(idw);
    const auto idwEnd = std::chrono::steady_clock::now();
    if (idwResult.values.size() != expectedIdw.size()) {
        return fail("medium DEM IDW batch returned the wrong size");
    }
    for (std::size_t index = 0U; index < expectedIdw.size(); ++index) {
        if (!closeDouble(idwResult.values[index], expectedIdw[index])) {
            return fail("medium DEM IDW chunk/oracle mismatch at " +
                        std::to_string(index));
        }
    }

    luw::cli::OpenClLuwbcMlsRequest mls;
    mls.deviceIndex = deviceIndex;
    mls.neighborCount = 24U;
    mls.sourceX = idw.sourceX;
    mls.sourceY = idw.sourceY;
    mls.targetX = idw.targetX;
    mls.targetY = idw.targetY;
    mls.neighborIndices.resize(targetCount * mls.neighborCount);
    for (std::size_t target = 0U; target < targetCount; ++target) {
        const std::size_t rotation = target % mls.neighborCount;
        for (std::size_t entry = 0U; entry < mls.neighborCount; ++entry) {
            mls.neighborIndices[target * mls.neighborCount + entry] =
                static_cast<std::uint32_t>(
                    (rotation + entry) % mls.neighborCount);
        }
    }
    const auto mlsCpuStart = std::chrono::steady_clock::now();
    const std::vector<float> expectedMls = mlsOracle(mls);
    const auto mlsCpuEnd = std::chrono::steady_clock::now();
    const auto mlsStart = std::chrono::steady_clock::now();
    const luw::cli::OpenClLuwbcResult mlsResult =
        luw::cli::runNativeOpenClLuwbcMlsWeights(mls);
    const auto mlsEnd = std::chrono::steady_clock::now();
    if (mlsResult.values.size() != expectedMls.size()) {
        return fail("medium MLS batch returned the wrong size");
    }
    for (std::size_t index = 0U; index < expectedMls.size(); ++index) {
        if (!close(mlsResult.values[index], expectedMls[index], 8.0e-5F)) {
            return fail("medium MLS chunk/oracle mismatch at " +
                        std::to_string(index));
        }
    }
    const auto idwMilliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(idwEnd - idwStart)
            .count();
    const auto idwCpuMilliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(idwCpuEnd -
                                                               idwCpuStart)
            .count();
    const auto mlsMilliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(mlsEnd - mlsStart)
            .count();
    const auto mlsCpuMilliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(mlsCpuEnd -
                                                               mlsCpuStart)
            .count();
    std::cout << "OpenCL luwbc medium spatial batch (" << targetCount
              << " targets, device " << deviceIndex
              << "): IDW CPU/GPU " << idwCpuMilliseconds << "/"
              << idwMilliseconds << " ms, MLS CPU/GPU "
              << mlsCpuMilliseconds << "/" << mlsMilliseconds << " ms\n";
    return 0;
}

int verifyValidation() {
    try {
        luw::cli::OpenClLuwbcHorizontalRequest request;
        request.deviceIndex = 0;
        request.sourceLayerSize = 1U;
        request.layerCount = std::numeric_limits<std::size_t>::max();
        request.targetPointCount = 0U;
        request.componentCount = 4U;
        (void)luw::cli::runNativeOpenClLuwbcHorizontal(request);
        return fail("horizontal dimension overflow was accepted");
    } catch (const luw::cli::OpenClLuwbcError& exception) {
        if (std::string(exception.what()).find("overflow") == std::string::npos) {
            return fail("horizontal overflow error was not explicit");
        }
    }

    try {
        luw::cli::OpenClLuwbcHorizontalRequest request;
        request.deviceIndex = 0;
        request.sourceLayerSize = 2U;
        request.layerCount = 1U;
        request.targetPointCount = 1U;
        request.componentCount = 3U;
        request.values.resize(6U);
        request.indices = {2U, 0U, 0U};
        request.weights = {1.0, 0.0, 0.0};
        request.counts = {1U};
        (void)luw::cli::runNativeOpenClLuwbcHorizontal(request);
        return fail("out-of-range horizontal source index was accepted");
    } catch (const luw::cli::OpenClLuwbcError& exception) {
        if (std::string(exception.what()).find("source index") ==
            std::string::npos) {
            return fail("horizontal index error was not explicit");
        }
    }

    try {
        luw::cli::OpenClLuwbcVerticalRequest request;
        request.deviceIndex = 0;
        request.pointCount = 1U;
        request.componentCount = 3U;
        request.sourceLevels = {0.0F, 0.0F};
        request.targetLevels = {0.0F};
        request.values.resize(6U);
        (void)luw::cli::runNativeOpenClLuwbcVertical(request);
        return fail("non-increasing vertical source levels were accepted");
    } catch (const luw::cli::OpenClLuwbcError& exception) {
        if (std::string(exception.what()).find("strictly increasing") ==
            std::string::npos) {
            return fail("vertical level-order error was not explicit");
        }
    }

    try {
        luw::cli::OpenClLuwbcDemIdwRequest request;
        request.deviceIndex = 0;
        request.neighborCount = 25U;
        request.sourceX.resize(25U, 0.0);
        request.sourceY.resize(25U, 0.0);
        request.sourceValues.resize(25U, 1.0);
        request.targetX = {0.0};
        request.targetY = {0.0};
        request.neighborIndices.resize(25U, 0U);
        (void)luw::cli::runNativeOpenClLuwbcDemIdw(request);
        return fail("DEM IDW neighborCount above 24 was accepted");
    } catch (const luw::cli::OpenClLuwbcError& exception) {
        if (std::string(exception.what()).find("neighborCount") ==
            std::string::npos) {
            return fail("DEM IDW neighbor-count error was not explicit");
        }
    }

    try {
        luw::cli::OpenClLuwbcMlsRequest request;
        request.deviceIndex = 0;
        request.neighborCount = 1U;
        request.sourceX = {0.0};
        request.sourceY = {0.0};
        request.targetX = {0.0};
        request.targetY = {0.0};
        request.neighborIndices = {1U};
        (void)luw::cli::runNativeOpenClLuwbcMlsWeights(request);
        return fail("out-of-range MLS source index was accepted");
    } catch (const luw::cli::OpenClLuwbcError& exception) {
        if (std::string(exception.what()).find("source index") ==
            std::string::npos) {
            return fail("MLS source-index error was not explicit");
        }
    }

    try {
        luw::cli::OpenClLuwbcDemIdwRequest request;
        request.deviceIndex = 0;
        request.neighborCount = 1U;
        request.sourceX = {0.0};
        request.sourceY = {0.0};
        request.sourceValues = {
            std::numeric_limits<double>::quiet_NaN()};
        request.targetX = {0.0};
        request.targetY = {0.0};
        request.neighborIndices = {0U};
        (void)luw::cli::runNativeOpenClLuwbcDemIdw(request);
        return fail("non-finite DEM IDW source value was accepted");
    } catch (const luw::cli::OpenClLuwbcError& exception) {
        if (std::string(exception.what()).find("non-finite") ==
            std::string::npos) {
            return fail("DEM IDW non-finite error was not explicit");
        }
    }
    return 0;
}

} // namespace

int main() {
#if !LUW_HAS_OPENCL
    std::cout << "OpenCL luwbc test skipped: OpenCL support is not enabled in this build\n";
    return 77;
#else
    const bool expectedWarmup =
        luw::cli::nativeOpenClLuwbcRequiresWarmup(0);
    luw::cli::OpenClLuwbcBackendInfo first;
    try {
        first = luw::cli::requireNativeOpenClLuwbcBackend(0);
    } catch (const luw::cli::OpenClLuwbcError& exception) {
        const std::string message = exception.what();
        if (backendAbsent(message)) {
            std::cout << "OpenCL luwbc test skipped: " << message << '\n';
            return 77;
        }
        return fail("OpenCL luwbc preflight failed on an enumerated GPU: " +
                    message);
    }

    try {
        if (first.device.empty() || first.deviceIndex != 0 ||
            (expectedWarmup && !first.compiledThisCall)) {
            return fail("invalid first luwbc OpenCL backend metadata");
        }
        if (luw::cli::nativeOpenClLuwbcRequiresWarmup(0)) {
            return fail("luwbc OpenCL backend was not cached after preflight");
        }

        for (const int invalidIndex :
             {-1, std::numeric_limits<int>::max()}) {
            try {
                (void)luw::cli::requireNativeOpenClLuwbcBackend(invalidIndex);
                return fail("invalid luwbc OpenCL GPU device index was accepted");
            } catch (const luw::cli::OpenClLuwbcError& exception) {
                const std::string message = exception.what();
                if (message.find(std::to_string(invalidIndex)) ==
                        std::string::npos ||
                    message.find("available") == std::string::npos) {
                    return fail(
                        "device-index error omitted requested/available details: " +
                        message);
                }
            }
        }

        if (const int status = verifyHorizontal(3U); status != 0) return status;
        if (const int status = verifyHorizontal(4U); status != 0) return status;
        if (const int status = verifyVertical(); status != 0) return status;
        if (const int status = verifyResamplePipelineChunks(); status != 0) {
            return status;
        }
        if (const int status = verifyDemIdw(0); status != 0) return status;
        if (const int status = verifyMls(0); status != 0) return status;
        if (const int status = verifyMediumSpatialBatch(0); status != 0) {
            return status;
        }
        if (const int status = verifyValidation(); status != 0) return status;

        const luw::cli::OpenClLuwbcBackendInfo cached =
            luw::cli::requireNativeOpenClLuwbcBackend(0);
        if (cached.compiledThisCall || cached.device != first.device ||
            cached.deviceIndex != first.deviceIndex) {
            return fail("per-device luwbc OpenCL backend cache was not reused");
        }
        try {
            const luw::cli::OpenClLuwbcBackendInfo second =
                luw::cli::requireNativeOpenClLuwbcBackend(1);
            if (second.device.empty() || second.deviceIndex != 1) {
                return fail("nonzero luwbc OpenCL device selection returned "
                            "invalid metadata");
            }
            if (const int status = verifyDemIdw(1); status != 0) return status;
            if (const int status = verifyMls(1); status != 0) return status;
            std::cout << "OpenCL luwbc device 1: " << second.device << '\n';
        } catch (const luw::cli::OpenClLuwbcError& exception) {
            const std::string message = exception.what();
            if (message.find("but only 1 OpenCL GPU device(s) are available") ==
                std::string::npos) {
                throw;
            }
        }
        std::cout << "OpenCL luwbc device: " << first.device << '\n';
        return 0;
    } catch (const luw::cli::OpenClLuwbcError& exception) {
        return fail("unexpected luwbc OpenCL execution failure after successful "
                    "preflight: " +
                    std::string(exception.what()));
    }
#endif
}
