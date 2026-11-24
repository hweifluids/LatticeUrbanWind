#include "tke_viz_internal.hpp"

#include "post_internal.hpp"

// Keep declarations aligned with the VTK::libproj binary selected by CMake.
#include <vtk_libproj.h>
#include <vtk_zlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace luw::post::tke {

namespace {

namespace fs = std::filesystem;
using detail::Projection;
using detail::pathUtf8;

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr std::size_t maskChunkRows = 256;
constexpr std::size_t boundarySamples = 800;

[[nodiscard]] std::string lower(std::string_view text) {
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return result;
}

[[nodiscard]] std::string normalizedField(std::string_view text) {
    std::string result;
    for (const unsigned char character : text) {
        if (std::isalnum(character) != 0) result.push_back(static_cast<char>(std::tolower(character)));
    }
    return result;
}

[[nodiscard]] std::pair<double, double> sortedPair(double first, double second) {
    return first <= second ? std::pair{first, second} : std::pair{second, first};
}

[[nodiscard]] int roundedGridCount(double span, double step) {
    // Python's round() is ties-to-even. nearbyint follows the default IEEE
    // rounding mode and therefore preserves that edge-case behaviour.
    const double rounded = std::nearbyint(span / step);
    if (!std::isfinite(rounded) || rounded > static_cast<double>(std::numeric_limits<int>::max() - 1)) {
        throw std::runtime_error("Target visualization grid is too large");
    }
    return std::max(2, static_cast<int>(rounded) + 1);
}

void projectionApply(const Projection& projection,
                     std::vector<double>& x,
                     std::vector<double>& y) {
    if (x.size() != y.size()) throw std::logic_error("Projection coordinate arrays differ in size");
    std::vector<PJ_COORD> coordinates(x.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
        coordinates[index] = proj_coord(x[index], y[index], 0.0, 0.0);
    }
    projection.apply(coordinates);
    for (std::size_t index = 0; index < x.size(); ++index) {
        x[index] = coordinates[index].xy.x;
        y[index] = coordinates[index].xy.y;
    }
}

[[nodiscard]] std::vector<double> axis(double origin, double spacing, int count) {
    std::vector<double> values(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        values[static_cast<std::size_t>(index)] = origin + static_cast<double>(index) * spacing;
    }
    return values;
}

[[nodiscard]] std::vector<int> sampledIndices(int count, std::size_t maximum) {
    const int samples = std::min(count, static_cast<int>(maximum));
    std::vector<int> result(static_cast<std::size_t>(samples));
    if (samples <= 1) {
        if (samples == 1) result[0] = 0;
        return result;
    }
    for (int index = 0; index < samples; ++index) {
        result[static_cast<std::size_t>(index)] = static_cast<int>(std::nearbyint(
            static_cast<double>(index) * static_cast<double>(count - 1)
            / static_cast<double>(samples - 1)));
    }
    return result;
}

[[nodiscard]] const LegacyVtkField* findInsensitive(const LegacyStructuredPoints& vtk,
                                                    std::string_view candidate) {
    const std::string wantedLower = lower(candidate);
    const std::string wantedNormalized = normalizedField(candidate);
    for (const LegacyVtkField& field : vtk.fields) {
        if (lower(field.name) == wantedLower || normalizedField(field.name) == wantedNormalized) return &field;
    }
    return nullptr;
}

[[nodiscard]] bool pointField(const LegacyStructuredPoints& vtk,
                              const LegacyVtkField& field,
                              std::size_t components) {
    return field.association == LegacyVtkAssociation::PointData
        && field.tuples == vtk.pointCount
        && field.components == components;
}

[[nodiscard]] bool vectorPointField(const LegacyStructuredPoints& vtk,
                                    const LegacyVtkField& field) {
    return field.association == LegacyVtkAssociation::PointData
        && field.tuples == vtk.pointCount
        && field.components >= 3;
}

} // namespace

class GeoModel final {
public:
    explicit GeoModel(const RuntimeConfig& config)
        : utmCrs_(config.utmCrs.empty() ? config.targetCrs : config.utmCrs),
          llToUtm_("EPSG:4326", utmCrs_),
          utmToLl_(utmCrs_, "EPSG:4326") {
        const auto [lonLow, lonHigh] = sortedPair(config.domainLon.first, config.domainLon.second);
        const auto [latLow, latHigh] = sortedPair(config.domainLat.first, config.domainLat.second);
        domainLon_ = {lonLow, lonHigh};
        domainLat_ = {latLow, latHigh};

        std::array<std::pair<double, double>, 4> corners{
            llToUtm_.apply(lonLow, latLow), llToUtm_.apply(lonHigh, latLow),
            llToUtm_.apply(lonHigh, latHigh), llToUtm_.apply(lonLow, latHigh)};
        if (config.rotateDegrees) {
            rotateDegrees_ = *config.rotateDegrees;
        } else {
            rotateDegrees_ = -std::atan2(corners[1].second - corners[0].second,
                                         corners[1].first - corners[0].first) * 180.0 / pi;
        }
        for (const auto& [x, y] : corners) {
            pivotX_ += x;
            pivotY_ += y;
        }
        pivotX_ /= 4.0;
        pivotY_ /= 4.0;
        xOriginRotated_ = std::numeric_limits<double>::infinity();
        yOriginRotated_ = std::numeric_limits<double>::infinity();
        for (const auto& [x, y] : corners) {
            const auto [rotatedX, rotatedY] = rotate(x, y, rotateDegrees_);
            xOriginRotated_ = std::min(xOriginRotated_, rotatedX);
            yOriginRotated_ = std::min(yOriginRotated_, rotatedY);
        }
    }

    void localToLonLat(std::vector<double>& x, std::vector<double>& y) const {
        for (std::size_t index = 0; index < x.size(); ++index) {
            const auto [utmX, utmY] = rotate(x[index] + xOriginRotated_,
                                             y[index] + yOriginRotated_,
                                             -rotateDegrees_);
            x[index] = utmX;
            y[index] = utmY;
        }
        projectionApply(utmToLl_, x, y);
    }

    void lonLatToLocal(std::vector<double>& x, std::vector<double>& y) const {
        projectionApply(llToUtm_, x, y);
        for (std::size_t index = 0; index < x.size(); ++index) {
            const auto [rotatedX, rotatedY] = rotate(x[index], y[index], rotateDegrees_);
            x[index] = rotatedX - xOriginRotated_;
            y[index] = rotatedY - yOriginRotated_;
        }
    }

    [[nodiscard]] const std::string& utmCrs() const noexcept { return utmCrs_; }
    [[nodiscard]] double rotateDegrees() const noexcept { return rotateDegrees_; }
    [[nodiscard]] std::pair<double, double> domainLon() const noexcept { return domainLon_; }
    [[nodiscard]] std::pair<double, double> domainLat() const noexcept { return domainLat_; }

private:
    [[nodiscard]] std::pair<double, double> rotate(double x, double y, double degrees) const {
        const double radians = degrees * pi / 180.0;
        const double cosine = std::cos(radians);
        const double sine = std::sin(radians);
        return {
            cosine * (x - pivotX_) - sine * (y - pivotY_) + pivotX_,
            sine * (x - pivotX_) + cosine * (y - pivotY_) + pivotY_};
    }

    std::string utmCrs_;
    double rotateDegrees_ = 0.0;
    double pivotX_ = 0.0;
    double pivotY_ = 0.0;
    double xOriginRotated_ = 0.0;
    double yOriginRotated_ = 0.0;
    std::pair<double, double> domainLon_{};
    std::pair<double, double> domainLat_{};
    Projection llToUtm_;
    Projection utmToLl_;
};

std::string WindFields::label() const {
    if (vector != nullptr) return vector->name;
    if (u == nullptr || v == nullptr || w == nullptr) return {};
    return u->name + "," + v->name + "," + w->name;
}

Bounds localBounds(const LegacyStructuredPoints& vtk,
                   std::optional<double> minX,
                   std::optional<double> maxX,
                   std::optional<double> minY,
                   std::optional<double> maxY) {
    const auto defaultX = sortedPair(vtk.origin[0],
        vtk.origin[0] + static_cast<double>(vtk.dimensions[0] - 1) * vtk.spacing[0]);
    const auto defaultY = sortedPair(vtk.origin[1],
        vtk.origin[1] + static_cast<double>(vtk.dimensions[1] - 1) * vtk.spacing[1]);
    Bounds result;
    result.defaultMinX = defaultX.first;
    result.defaultMaxX = defaultX.second;
    result.defaultMinY = defaultY.first;
    result.defaultMaxY = defaultY.second;
    const auto actualX = sortedPair(minX.value_or(result.defaultMinX), maxX.value_or(result.defaultMaxX));
    const auto actualY = sortedPair(minY.value_or(result.defaultMinY), maxY.value_or(result.defaultMaxY));
    result.minX = actualX.first;
    result.maxX = actualX.second;
    result.minY = actualY.first;
    result.maxY = actualY.second;
    return result;
}

Bounds geographicBounds(const RuntimeConfig& config,
                        std::optional<double> minX,
                        std::optional<double> maxX,
                        std::optional<double> minY,
                        std::optional<double> maxY) {
    Projection projection("EPSG:4326", config.targetCrs);
    const auto minimum = projection.apply(config.minLon, config.minLat);
    const auto maximum = projection.apply(config.maxLon, config.maxLat);
    Bounds result;
    result.defaultMinX = minimum.first;
    result.defaultMaxX = maximum.first;
    result.defaultMinY = minimum.second;
    result.defaultMaxY = maximum.second;
    const auto actualX = sortedPair(minX.value_or(result.defaultMinX), maxX.value_or(result.defaultMaxX));
    const auto actualY = sortedPair(minY.value_or(result.defaultMinY), maxY.value_or(result.defaultMaxY));
    result.minX = actualX.first;
    result.maxX = actualX.second;
    result.minY = actualY.first;
    result.maxY = actualY.second;
    return result;
}

Crop localCrop(const LegacyStructuredPoints& vtk, const Bounds& bounds) {
    const std::vector<double> xValues = axis(vtk.origin[0], vtk.spacing[0], vtk.dimensions[0]);
    const std::vector<double> yValues = axis(vtk.origin[1], vtk.spacing[1], vtk.dimensions[1]);
    std::vector<int> xIndices;
    std::vector<int> yIndices;
    for (int index = 0; index < vtk.dimensions[0]; ++index) {
        const double value = xValues[static_cast<std::size_t>(index)];
        if (value >= bounds.minX && value <= bounds.maxX) xIndices.push_back(index);
    }
    for (int index = 0; index < vtk.dimensions[1]; ++index) {
        const double value = yValues[static_cast<std::size_t>(index)];
        if (value >= bounds.minY && value <= bounds.maxY) yIndices.push_back(index);
    }
    if (xIndices.empty() || yIndices.empty()) {
        throw std::runtime_error("Visualization range does not overlap VTK local XY domain.");
    }
    Crop result;
    result.xStart = xIndices.front();
    result.xEnd = xIndices.back() + 1;
    result.yStart = yIndices.front();
    result.yEnd = yIndices.back() + 1;
    result.insideCount = static_cast<std::size_t>(result.nx()) * static_cast<std::size_t>(result.ny());
    result.insideRatio = result.insideCount == 0 ? 0.0 : 1.0;
    const auto domainX = sortedPair(xValues.front(), xValues.back());
    const auto domainY = sortedPair(yValues.front(), yValues.back());
    result.domainPolygonX = {domainX.first, domainX.second, domainX.second, domainX.first, domainX.first};
    result.domainPolygonY = {domainY.first, domainY.first, domainY.second, domainY.second, domainY.first};
    result.selectedRectangleX = {bounds.minX, bounds.maxX, bounds.maxX, bounds.minX, bounds.minX};
    result.selectedRectangleY = {bounds.minY, bounds.minY, bounds.maxY, bounds.maxY, bounds.minY};
    return result;
}

Crop geographicCrop(const LegacyStructuredPoints& vtk,
                    const RuntimeConfig& config,
                    const Bounds& bounds) {
    const int nx = vtk.dimensions[0];
    const int ny = vtk.dimensions[1];
    const std::vector<double> xValues = axis(vtk.origin[0], vtk.spacing[0], nx);
    const std::vector<double> yValues = axis(vtk.origin[1], vtk.spacing[1], ny);
    auto model = std::make_shared<GeoModel>(config);
    Projection llToTarget("EPSG:4326", config.targetCrs);

    const std::vector<int> sampleX = sampledIndices(nx, boundarySamples);
    const std::vector<int> sampleY = sampledIndices(ny, boundarySamples);
    std::vector<double> boundaryX;
    std::vector<double> boundaryY;
    boundaryX.reserve(sampleX.size() * 2 + sampleY.size() * 2 + 1);
    boundaryY.reserve(boundaryX.capacity());
    for (const int index : sampleX) { boundaryX.push_back(xValues[static_cast<std::size_t>(index)]); boundaryY.push_back(yValues.front()); }
    for (const int index : sampleY) { boundaryX.push_back(xValues.back()); boundaryY.push_back(yValues[static_cast<std::size_t>(index)]); }
    for (auto iterator = sampleX.rbegin(); iterator != sampleX.rend(); ++iterator) { boundaryX.push_back(xValues[static_cast<std::size_t>(*iterator)]); boundaryY.push_back(yValues.back()); }
    for (auto iterator = sampleY.rbegin(); iterator != sampleY.rend(); ++iterator) { boundaryX.push_back(xValues.front()); boundaryY.push_back(yValues[static_cast<std::size_t>(*iterator)]); }
    if (!boundaryX.empty()) { boundaryX.push_back(boundaryX.front()); boundaryY.push_back(boundaryY.front()); }
    model->localToLonLat(boundaryX, boundaryY);
    projectionApply(llToTarget, boundaryX, boundaryY);

    int xStart = nx;
    int xEnd = 0;
    int yStart = ny;
    int yEnd = 0;
    std::size_t inside = 0;
    for (int blockStart = 0; blockStart < ny; blockStart += static_cast<int>(maskChunkRows)) {
        const int blockEnd = std::min(ny, blockStart + static_cast<int>(maskChunkRows));
        const std::size_t count = static_cast<std::size_t>(blockEnd - blockStart) * static_cast<std::size_t>(nx);
        std::vector<double> projectedX(count);
        std::vector<double> projectedY(count);
        for (int y = blockStart; y < blockEnd; ++y) {
            for (int x = 0; x < nx; ++x) {
                const std::size_t local = static_cast<std::size_t>(y - blockStart) * static_cast<std::size_t>(nx)
                    + static_cast<std::size_t>(x);
                projectedX[local] = xValues[static_cast<std::size_t>(x)];
                projectedY[local] = yValues[static_cast<std::size_t>(y)];
            }
        }
        model->localToLonLat(projectedX, projectedY);
        projectionApply(llToTarget, projectedX, projectedY);
        for (int y = blockStart; y < blockEnd; ++y) {
            for (int x = 0; x < nx; ++x) {
                const std::size_t local = static_cast<std::size_t>(y - blockStart) * static_cast<std::size_t>(nx)
                    + static_cast<std::size_t>(x);
                if (projectedX[local] < bounds.minX || projectedX[local] > bounds.maxX
                    || projectedY[local] < bounds.minY || projectedY[local] > bounds.maxY) continue;
                ++inside;
                xStart = std::min(xStart, x);
                xEnd = std::max(xEnd, x + 1);
                yStart = std::min(yStart, y);
                yEnd = std::max(yEnd, y + 1);
            }
        }
    }
    if (inside == 0) {
        throw std::runtime_error(
            "Visualization range does not overlap transformed VTK domain after local->lon/lat->TARGET_CRS mapping.");
    }
    Crop result;
    result.xStart = xStart;
    result.xEnd = xEnd;
    result.yStart = yStart;
    result.yEnd = yEnd;
    result.insideCount = inside;
    const std::size_t rectangleCount = static_cast<std::size_t>(result.nx()) * static_cast<std::size_t>(result.ny());
    result.insideRatio = rectangleCount == 0 ? 0.0 : static_cast<double>(inside) / static_cast<double>(rectangleCount);
    result.utmCrs = model->utmCrs();
    result.rotateDegrees = model->rotateDegrees();
    result.domainLon = model->domainLon();
    result.domainLat = model->domainLat();
    result.domainPolygonX = std::move(boundaryX);
    result.domainPolygonY = std::move(boundaryY);
    result.selectedRectangleX = {bounds.minX, bounds.maxX, bounds.maxX, bounds.minX, bounds.minX};
    result.selectedRectangleY = {bounds.minY, bounds.minY, bounds.maxY, bounds.maxY, bounds.minY};
    result.model = std::move(model);
    return result;
}

TargetGrid localTargetGrid(const LegacyStructuredPoints& vtk,
                           const RuntimeConfig& config,
                           const Bounds& bounds) {
    double step = config.cellSize;
    if (!std::isfinite(step) || step <= 0.0) step = std::min(std::abs(vtk.spacing[0]), std::abs(vtk.spacing[1]));
    if (!(step > 0.0)) step = 10.0;
    TargetGrid result;
    result.nx = roundedGridCount(bounds.maxX - bounds.minX, step);
    result.ny = roundedGridCount(bounds.maxY - bounds.minY, step);
    result.minX = bounds.minX;
    result.maxX = bounds.maxX;
    result.minY = bounds.minY;
    result.maxY = bounds.maxY;
    result.step = step;
    const std::size_t count = static_cast<std::size_t>(result.nx) * static_cast<std::size_t>(result.ny);
    result.xIndex.resize(count);
    result.yIndex.resize(count);
    result.valid.resize(count);
    for (int y = 0; y < result.ny; ++y) {
        const double targetY = result.ny == 1 ? bounds.minY
            : bounds.minY + static_cast<double>(y) * (bounds.maxY - bounds.minY) / static_cast<double>(result.ny - 1);
        for (int x = 0; x < result.nx; ++x) {
            const double targetX = result.nx == 1 ? bounds.minX
                : bounds.minX + static_cast<double>(x) * (bounds.maxX - bounds.minX) / static_cast<double>(result.nx - 1);
            const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(result.nx)
                + static_cast<std::size_t>(x);
            const double sourceX = (targetX - vtk.origin[0]) / vtk.spacing[0];
            const double sourceY = (targetY - vtk.origin[1]) / vtk.spacing[1];
            result.valid[index] = sourceX >= 0.0 && sourceX <= static_cast<double>(vtk.dimensions[0] - 1)
                && sourceY >= 0.0 && sourceY <= static_cast<double>(vtk.dimensions[1] - 1);
            result.xIndex[index] = std::clamp(sourceX, 0.0, static_cast<double>(vtk.dimensions[0] - 1));
            result.yIndex[index] = std::clamp(sourceY, 0.0, static_cast<double>(vtk.dimensions[1] - 1));
        }
    }
    return result;
}

TargetGrid geographicTargetGrid(const LegacyStructuredPoints& vtk,
                                const RuntimeConfig& config,
                                const Bounds& bounds,
                                const Crop& crop) {
    if (!crop.model) throw std::logic_error("Geographic crop is missing its transform model");
    double step = config.cellSize;
    if (!std::isfinite(step) || step <= 0.0) step = std::min(std::abs(vtk.spacing[0]), std::abs(vtk.spacing[1]));
    if (!(step > 0.0)) step = 10.0;
    TargetGrid result;
    result.nx = roundedGridCount(bounds.maxX - bounds.minX, step);
    result.ny = roundedGridCount(bounds.maxY - bounds.minY, step);
    result.minX = bounds.minX;
    result.maxX = bounds.maxX;
    result.minY = bounds.minY;
    result.maxY = bounds.maxY;
    result.step = step;
    const double radians = -crop.rotateDegrees * pi / 180.0;
    result.vectorCos = std::cos(radians);
    result.vectorSin = std::sin(radians);
    const std::size_t count = static_cast<std::size_t>(result.nx) * static_cast<std::size_t>(result.ny);
    std::vector<double> localX(count);
    std::vector<double> localY(count);
    for (int y = 0; y < result.ny; ++y) {
        const double targetY = bounds.minY + static_cast<double>(y) * (bounds.maxY - bounds.minY)
            / static_cast<double>(result.ny - 1);
        for (int x = 0; x < result.nx; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(result.nx)
                + static_cast<std::size_t>(x);
            localX[index] = bounds.minX + static_cast<double>(x) * (bounds.maxX - bounds.minX)
                / static_cast<double>(result.nx - 1);
            localY[index] = targetY;
        }
    }
    Projection targetToLl(config.targetCrs, "EPSG:4326");
    projectionApply(targetToLl, localX, localY);
    crop.model->lonLatToLocal(localX, localY);
    result.xIndex.resize(count);
    result.yIndex.resize(count);
    result.valid.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        const double sourceX = (localX[index] - vtk.origin[0]) / vtk.spacing[0];
        const double sourceY = (localY[index] - vtk.origin[1]) / vtk.spacing[1];
        result.valid[index] = sourceX >= 0.0 && sourceX <= static_cast<double>(vtk.dimensions[0] - 1)
            && sourceY >= 0.0 && sourceY <= static_cast<double>(vtk.dimensions[1] - 1);
        result.xIndex[index] = std::clamp(sourceX, 0.0, static_cast<double>(vtk.dimensions[0] - 1));
        result.yIndex[index] = std::clamp(sourceY, 0.0, static_cast<double>(vtk.dimensions[1] - 1));
    }
    return result;
}

std::array<HeightItem, 9> heightPlan(const LegacyStructuredPoints& vtk) {
    std::array<HeightItem, 9> result{};
    for (std::size_t index = 0; index < targetHeights.size(); ++index) {
        const int z = static_cast<int>(std::nearbyint(
            (static_cast<double>(targetHeights[index]) - baseHeight) / layerStep));
        result[index] = {targetHeights[index], z, z >= 0 && z < vtk.dimensions[2], baseHeight + layerStep * z};
    }
    return result;
}

ZCrop computeZCrop(const LegacyStructuredPoints& vtk, std::optional<double> maximumZ) {
    const int nz = vtk.dimensions[2];
    if (!maximumZ) {
        return {0, nz, nz, vtk.origin[2],
                vtk.origin[2] + static_cast<double>(nz - 1) * vtk.spacing[2], false};
    }
    if (vtk.spacing[2] <= 0.0) {
        throw std::runtime_error("Z crop requires positive dz, got dz=" + std::to_string(vtk.spacing[2]));
    }
    const double relative = (*maximumZ - vtk.origin[2]) / vtk.spacing[2];
    const int end = std::min(nz, std::max(0, static_cast<int>(std::floor(relative + 1.0e-9)) + 1));
    if (end <= 0) {
        std::ostringstream message;
        message << "crop_z_max=" << *maximumZ << " is below the first VTK layer z=" << vtk.origin[2];
        throw std::runtime_error(message.str());
    }
    return {0, end, end, vtk.origin[2],
            vtk.origin[2] + static_cast<double>(end - 1) * vtk.spacing[2], end < nz};
}

WindFields selectWindFields(const LegacyStructuredPoints& vtk, bool exactLegacyNames) {
    if (exactLegacyNames) {
        const LegacyVtkField* field = vtk.field("u_avg");
        if (field == nullptr) throw std::runtime_error("VTK missing 'u_avg' field");
        if (!vectorPointField(vtk, *field)) throw std::runtime_error("Field 'u_avg' must have at least 3 components");
        return {.vector = field};
    }
    for (const std::string_view candidate : {"u_avg", "velocity", "wind", "uvw"}) {
        const LegacyVtkField* field = findInsensitive(vtk, candidate);
        if (field != nullptr && vectorPointField(vtk, *field)) return {.vector = field};
    }
    constexpr std::array<std::array<std::string_view, 3>, 4> triplets{{
        {"u_filtered", "v_filtered", "w_filtered"}, {"u", "v", "w"},
        {"u_avg_x", "u_avg_y", "u_avg_z"}, {"ux", "uy", "uz"}}};
    for (const auto& triplet : triplets) {
        const LegacyVtkField* u = findInsensitive(vtk, triplet[0]);
        const LegacyVtkField* v = findInsensitive(vtk, triplet[1]);
        const LegacyVtkField* w = findInsensitive(vtk, triplet[2]);
        if (u != nullptr && v != nullptr && w != nullptr
            && pointField(vtk, *u, 1) && pointField(vtk, *v, 1) && pointField(vtk, *w, 1)) {
            return {.u = u, .v = v, .w = w};
        }
    }
    for (const LegacyVtkField& field : vtk.fields) {
        if (vectorPointField(vtk, field)) return {.vector = &field};
    }
    throw std::runtime_error("VTK missing a wind vector field or scalar u/v/w component fields");
}

const LegacyVtkField& selectTkeField(const LegacyStructuredPoints& vtk, bool exactLegacyName) {
    if (exactLegacyName) {
        const LegacyVtkField* field = vtk.field("tke");
        if (field == nullptr) throw std::runtime_error("VTK missing 'tke' field");
        if (!pointField(vtk, *field, 1)) throw std::runtime_error("Field 'tke' must be scalar");
        return *field;
    }
    for (const std::string_view candidate : {
        "tke", "tke_filtered", "k", "tke_avg", "k_avg", "sgs_tke",
        "turbulence_k", "tkenergy"}) {
        const LegacyVtkField* field = findInsensitive(vtk, candidate);
        if (field != nullptr && pointField(vtk, *field, 1)) return *field;
    }
    std::ostringstream available;
    for (std::size_t index = 0; index < vtk.fields.size(); ++index) {
        if (index != 0) available << ", ";
        available << vtk.fields[index].name;
    }
    throw std::runtime_error("VTK missing TKE scalar field. Available fields=[" + available.str() + "]");
}

namespace {

template <typename Value>
[[nodiscard]] Value byteSwapped(Value value) {
    static_assert(std::is_trivially_copyable_v<Value>);
    std::array<unsigned char, sizeof(Value)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(Value));
    std::reverse(bytes.begin(), bytes.end());
    std::memcpy(&value, bytes.data(), sizeof(Value));
    return value;
}

template <typename Value>
void writeBigEndian(std::ostream& stream, Value value) {
    if constexpr (sizeof(Value) > 1) {
        if constexpr (std::endian::native == std::endian::little) value = byteSwapped(value);
    }
    stream.write(reinterpret_cast<const char*>(&value), sizeof(Value));
}

void writeConvertedValue(std::ostream& stream, LegacyVtkScalarType type, float value) {
    switch (type) {
    case LegacyVtkScalarType::Int8: writeBigEndian(stream, static_cast<std::int8_t>(value)); break;
    case LegacyVtkScalarType::UInt8: writeBigEndian(stream, static_cast<std::uint8_t>(value)); break;
    case LegacyVtkScalarType::Int16: writeBigEndian(stream, static_cast<std::int16_t>(value)); break;
    case LegacyVtkScalarType::UInt16: writeBigEndian(stream, static_cast<std::uint16_t>(value)); break;
    case LegacyVtkScalarType::Int32: writeBigEndian(stream, static_cast<std::int32_t>(value)); break;
    case LegacyVtkScalarType::UInt32: writeBigEndian(stream, static_cast<std::uint32_t>(value)); break;
    case LegacyVtkScalarType::Float32: writeBigEndian(stream, value); break;
    case LegacyVtkScalarType::Float64: writeBigEndian(stream, static_cast<double>(value)); break;
    }
}

[[nodiscard]] std::string fixed(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

[[nodiscard]] std::vector<float> resample(const std::vector<float>& source,
                                          int sourceNx,
                                          int sourceNy,
                                          const TargetGrid& grid) {
    const std::size_t count = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny);
    if (source.size() != static_cast<std::size_t>(sourceNx) * static_cast<std::size_t>(sourceNy)) {
        throw std::logic_error("Scalar plane size does not match VTK dimensions");
    }
    std::vector<float> result(count, std::numeric_limits<float>::quiet_NaN());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t signedIndex = 0; signedIndex < static_cast<std::int64_t>(count); ++signedIndex) {
        const std::size_t index = static_cast<std::size_t>(signedIndex);
        if (grid.valid[index] == 0) continue;
        const double x = grid.xIndex[index];
        const double y = grid.yIndex[index];
        const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, sourceNx - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, sourceNy - 1);
        const int x1 = std::min(x0 + 1, sourceNx - 1);
        const int y1 = std::min(y0 + 1, sourceNy - 1);
        const double wx = x - static_cast<double>(x0);
        const double wy = y - static_cast<double>(y0);
        const auto at = [&](int sx, int sy) -> double {
            return source[static_cast<std::size_t>(sy) * static_cast<std::size_t>(sourceNx)
                          + static_cast<std::size_t>(sx)];
        };
        const double top = at(x0, y0) * (1.0 - wx) + at(x1, y0) * wx;
        const double bottom = at(x0, y1) * (1.0 - wx) + at(x1, y1) * wx;
        result[index] = static_cast<float>(top * (1.0 - wy) + bottom * wy);
    }
    return result;
}

[[nodiscard]] double percentile(std::vector<float> values, double percent) {
    values.erase(std::remove_if(values.begin(), values.end(), [](float value) {
        return !std::isfinite(value);
    }), values.end());
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = (static_cast<double>(values.size()) - 1.0) * percent / 100.0;
    const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(position));
    const std::size_t upperIndex = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lowerIndex);
    return static_cast<double>(values[lowerIndex]) * (1.0 - fraction)
        + static_cast<double>(values[upperIndex]) * fraction;
}

struct Color final {
    unsigned char red = 255;
    unsigned char green = 255;
    unsigned char blue = 255;
};

class Canvas final {
public:
    Canvas(int width, int height)
        : width_(width), height_(height), pixels_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {}

    void pixel(int x, int y, Color color) {
        if (x < 0 || y < 0 || x >= width_ || y >= height_) return;
        pixels_[static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x)] = color;
    }

    [[nodiscard]] Color pixel(int x, int y) const {
        return pixels_[static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x)];
    }

    void fill(int left, int top, int right, int bottom, Color color) {
        left = std::max(left, 0);
        top = std::max(top, 0);
        right = std::min(right, width_);
        bottom = std::min(bottom, height_);
        for (int y = top; y < bottom; ++y) {
            auto begin = pixels_.begin() + static_cast<std::ptrdiff_t>(y) * width_ + left;
            std::fill(begin, begin + (right - left), color);
        }
    }

    void blend(int left, int top, int right, int bottom, Color color, double alpha) {
        left = std::max(left, 0);
        top = std::max(top, 0);
        right = std::min(right, width_);
        bottom = std::min(bottom, height_);
        for (int y = top; y < bottom; ++y) for (int x = left; x < right; ++x) {
            const Color old = pixel(x, y);
            pixel(x, y, {
                static_cast<unsigned char>(std::lround(old.red * (1.0 - alpha) + color.red * alpha)),
                static_cast<unsigned char>(std::lround(old.green * (1.0 - alpha) + color.green * alpha)),
                static_cast<unsigned char>(std::lround(old.blue * (1.0 - alpha) + color.blue * alpha))});
        }
    }

    void line(int x0, int y0, int x1, int y1, Color color, int thickness = 1) {
        const int dx = std::abs(x1 - x0);
        const int sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0);
        const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        while (true) {
            fill(x0 - thickness / 2, y0 - thickness / 2,
                 x0 + (thickness + 1) / 2, y0 + (thickness + 1) / 2, color);
            if (x0 == x1 && y0 == y1) break;
            const int twice = 2 * error;
            if (twice >= dy) { error += dy; x0 += sx; }
            if (twice <= dx) { error += dx; y0 += sy; }
        }
    }

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Color> pixels_;
};

[[nodiscard]] std::array<unsigned char, 7> glyph(char character) {
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    switch (c) {
    case 'A': return {14,17,17,31,17,17,17}; case 'B': return {30,17,17,30,17,17,30};
    case 'C': return {14,17,16,16,16,17,14}; case 'D': return {30,17,17,17,17,17,30};
    case 'E': return {31,16,16,30,16,16,31}; case 'F': return {31,16,16,30,16,16,16};
    case 'G': return {14,17,16,23,17,17,15}; case 'H': return {17,17,17,31,17,17,17};
    case 'I': return {31,4,4,4,4,4,31}; case 'J': return {7,2,2,2,18,18,12};
    case 'K': return {17,18,20,24,20,18,17}; case 'L': return {16,16,16,16,16,16,31};
    case 'M': return {17,27,21,21,17,17,17}; case 'N': return {17,25,21,19,17,17,17};
    case 'O': return {14,17,17,17,17,17,14}; case 'P': return {30,17,17,30,16,16,16};
    case 'Q': return {14,17,17,17,21,18,13}; case 'R': return {30,17,17,30,20,18,17};
    case 'S': return {15,16,16,14,1,1,30}; case 'T': return {31,4,4,4,4,4,4};
    case 'U': return {17,17,17,17,17,17,14}; case 'V': return {17,17,17,17,17,10,4};
    case 'W': return {17,17,17,21,21,21,10}; case 'X': return {17,17,10,4,10,17,17};
    case 'Y': return {17,17,10,4,4,4,4}; case 'Z': return {31,1,2,4,8,16,31};
    case '0': return {14,17,19,21,25,17,14}; case '1': return {4,12,4,4,4,4,14};
    case '2': return {14,17,1,2,4,8,31}; case '3': return {30,1,1,14,1,1,30};
    case '4': return {2,6,10,18,31,2,2}; case '5': return {31,16,16,30,1,1,30};
    case '6': return {14,16,16,30,17,17,14}; case '7': return {31,1,2,4,8,8,8};
    case '8': return {14,17,17,14,17,17,14}; case '9': return {14,17,17,15,1,1,14};
    case '.': return {0,0,0,0,0,6,6}; case ',': return {0,0,0,0,6,6,4};
    case ':': return {0,6,6,0,6,6,0}; case '-': return {0,0,0,31,0,0,0};
    case '+': return {0,4,4,31,4,4,0}; case '/': return {1,2,2,4,8,8,16};
    case '(': return {2,4,8,8,8,4,2}; case ')': return {8,4,2,2,2,4,8};
    case '[': return {14,8,8,8,8,8,14}; case ']': return {14,2,2,2,2,2,14};
    case '_': return {0,0,0,0,0,0,31}; case '=': return {0,31,0,31,0,0,0};
    case '^': return {4,10,17,0,0,0,0}; case '|': return {4,4,4,4,4,4,4};
    default: return {};
    }
}

[[nodiscard]] int textWidth(std::string_view text, int scale) {
    return static_cast<int>(text.size()) * 6 * scale;
}

void drawText(Canvas& canvas, int x, int y, std::string_view text, int scale, Color color) {
    for (const char character : text) {
        const auto rows = glyph(character);
        for (int row = 0; row < 7; ++row) for (int column = 0; column < 5; ++column) {
            if ((rows[static_cast<std::size_t>(row)] & (1U << (4 - column))) == 0) continue;
            canvas.fill(x + column * scale, y + row * scale,
                        x + (column + 1) * scale, y + (row + 1) * scale, color);
        }
        x += 6 * scale;
    }
}

[[nodiscard]] Color turbo(double normalized) {
    const double x = std::clamp(normalized, 0.0, 1.0);
    const auto channel = [](double value) -> unsigned char {
        return static_cast<unsigned char>(std::lround(std::clamp(value, 0.0, 255.0)));
    };
    const double r = 34.61 + x * (1172.33 + x * (-10793.56 + x * (33300.12 + x * (-38394.49 + x * 14825.05))));
    const double g = 23.31 + x * (557.33 + x * (1225.33 + x * (-3574.96 + x * (1073.77 + x * 707.56))));
    const double b = 27.20 + x * (3211.10 + x * (-15327.97 + x * (27814.00 + x * (-22569.18 + x * 6838.66))));
    return {channel(r), channel(g), channel(b)};
}

[[nodiscard]] Color magma(double normalized) {
    struct Stop { double x; Color color; };
    static constexpr std::array<Stop, 9> stops{{
        {0.000, {0,0,4}}, {0.125, {27,12,65}}, {0.250, {79,12,107}},
        {0.375, {120,28,109}}, {0.500, {165,44,96}}, {0.625, {207,68,70}},
        {0.750, {237,105,37}}, {0.875, {251,155,6}}, {1.000, {252,253,191}}}};
    const double value = std::clamp(normalized, 0.0, 1.0);
    std::size_t upper = 1;
    while (upper + 1 < stops.size() && value > stops[upper].x) ++upper;
    const Stop& a = stops[upper - 1];
    const Stop& b = stops[upper];
    const double f = (value - a.x) / (b.x - a.x);
    const auto blend = [f](unsigned char left, unsigned char right) {
        return static_cast<unsigned char>(std::lround(left * (1.0 - f) + right * f));
    };
    return {blend(a.color.red, b.color.red), blend(a.color.green, b.color.green), blend(a.color.blue, b.color.blue)};
}

void writeU32(std::ostream& stream, std::uint32_t value) {
    const std::array<unsigned char, 4> bytes{
        static_cast<unsigned char>((value >> 24U) & 0xffU),
        static_cast<unsigned char>((value >> 16U) & 0xffU),
        static_cast<unsigned char>((value >> 8U) & 0xffU),
        static_cast<unsigned char>(value & 0xffU)};
    stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void writeChunk(std::ostream& stream, std::string_view type, std::span<const unsigned char> data) {
    writeU32(stream, static_cast<std::uint32_t>(data.size()));
    stream.write(type.data(), static_cast<std::streamsize>(type.size()));
    if (!data.empty()) stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    uLong checksum = crc32(0L, Z_NULL, 0);
    checksum = crc32(checksum, reinterpret_cast<const Bytef*>(type.data()), static_cast<uInt>(type.size()));
    if (!data.empty()) checksum = crc32(checksum, data.data(), static_cast<uInt>(data.size()));
    writeU32(stream, static_cast<std::uint32_t>(checksum));
}

void writeScaledPng(const fs::path& path, const Canvas& canvas, int outputWidth, int outputHeight) {
    if (outputWidth <= 0 || outputHeight <= 0) throw std::runtime_error("PNG dimensions must be positive");
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot create PNG: " + pathUtf8(path));
    static constexpr std::array<unsigned char, 8> signature{137,80,78,71,13,10,26,10};
    stream.write(reinterpret_cast<const char*>(signature.data()), signature.size());
    std::array<unsigned char, 13> ihdr{};
    for (int shift = 0; shift < 4; ++shift) {
        ihdr[static_cast<std::size_t>(shift)] = static_cast<unsigned char>(
            (static_cast<std::uint32_t>(outputWidth) >> (24 - shift * 8)) & 0xffU);
        ihdr[static_cast<std::size_t>(4 + shift)] = static_cast<unsigned char>(
            (static_cast<std::uint32_t>(outputHeight) >> (24 - shift * 8)) & 0xffU);
    }
    ihdr[8] = 8;
    ihdr[9] = 2;
    writeChunk(stream, "IHDR", ihdr);

    z_stream compressor{};
    if (deflateInit(&compressor, Z_DEFAULT_COMPRESSION) != Z_OK) {
        throw std::runtime_error("Cannot initialize PNG compressor");
    }
    try {
        std::vector<unsigned char> row(static_cast<std::size_t>(outputWidth) * 3U + 1U);
        std::vector<unsigned char> compressed(256U * 1024U);
        int previousSourceY = -1;
        auto drain = [&](int flush) {
            int status = Z_OK;
            do {
                compressor.next_out = compressed.data();
                compressor.avail_out = static_cast<uInt>(compressed.size());
                status = deflate(&compressor, flush);
                if (status != Z_OK && status != Z_STREAM_END) {
                    throw std::runtime_error("PNG compression failed");
                }
                const std::size_t produced = compressed.size() - compressor.avail_out;
                if (produced != 0) writeChunk(stream, "IDAT", std::span(compressed.data(), produced));
            } while (compressor.avail_out == 0 || (flush == Z_FINISH && status != Z_STREAM_END));
        };
        for (int y = 0; y < outputHeight; ++y) {
            const int sourceY = std::min(canvas.height() - 1,
                static_cast<int>(static_cast<std::int64_t>(y) * canvas.height() / outputHeight));
            if (sourceY != previousSourceY) {
                row[0] = 0;
                for (int x = 0; x < outputWidth; ++x) {
                    const int sourceX = std::min(canvas.width() - 1,
                        static_cast<int>(static_cast<std::int64_t>(x) * canvas.width() / outputWidth));
                    const Color color = canvas.pixel(sourceX, sourceY);
                    row[1U + static_cast<std::size_t>(x) * 3U] = color.red;
                    row[2U + static_cast<std::size_t>(x) * 3U] = color.green;
                    row[3U + static_cast<std::size_t>(x) * 3U] = color.blue;
                }
                previousSourceY = sourceY;
            }
            compressor.next_in = row.data();
            compressor.avail_in = static_cast<uInt>(row.size());
            while (compressor.avail_in != 0) drain(Z_NO_FLUSH);
        }
        drain(Z_FINISH);
        (void)deflateEnd(&compressor);
    } catch (...) {
        (void)deflateEnd(&compressor);
        throw;
    }
    writeChunk(stream, "IEND", {});
    if (!stream) throw std::runtime_error("Failed while writing PNG: " + pathUtf8(path));
}

struct PlotRect final { int left; int top; int width; int height; };

[[nodiscard]] PlotRect aspectContained(PlotRect cell, const TargetGrid& grid) {
    const double xSpan = std::max(grid.maxX - grid.minX, std::numeric_limits<double>::epsilon());
    const double ySpan = std::max(grid.maxY - grid.minY, std::numeric_limits<double>::epsilon());
    const double aspect = xSpan / ySpan;
    int width = cell.width;
    int height = static_cast<int>(std::nearbyint(static_cast<double>(width) / aspect));
    if (height > cell.height) {
        height = cell.height;
        width = static_cast<int>(std::nearbyint(static_cast<double>(height) * aspect));
    }
    width = std::max(width, 1);
    height = std::max(height, 1);
    return {cell.left + (cell.width - width) / 2, cell.top + (cell.height - height) / 2, width, height};
}

void renderScalarPanel(Canvas& canvas,
                       const PlotRect& rectangle,
                       const std::vector<float>& values,
                       const TargetGrid& grid,
                       double minimum,
                       double maximum,
                       const std::function<Color(double)>& colorMap) {
    const double range = maximum > minimum ? maximum - minimum : 1.0;
    for (int py = 0; py < rectangle.height; ++py) {
        const int gridY = std::clamp(
            static_cast<int>(static_cast<std::int64_t>(rectangle.height - 1 - py) * grid.ny / rectangle.height),
            0, grid.ny - 1);
        for (int px = 0; px < rectangle.width; ++px) {
            const int gridX = std::clamp(
                static_cast<int>(static_cast<std::int64_t>(px) * grid.nx / rectangle.width),
                0, grid.nx - 1);
            const float value = values[static_cast<std::size_t>(gridY) * static_cast<std::size_t>(grid.nx)
                                       + static_cast<std::size_t>(gridX)];
            if (!std::isfinite(value)) {
                canvas.pixel(rectangle.left + px, rectangle.top + py, {245,245,245});
            } else {
                canvas.pixel(rectangle.left + px, rectangle.top + py,
                             colorMap((static_cast<double>(value) - minimum) / range));
            }
        }
    }
}

struct Arrow final { int x = 0; int y = 0; float u = 0.0F; float v = 0.0F; };

struct WindLayer final {
    std::vector<float> speed;
    std::vector<Arrow> arrows;
};

void drawArrows(Canvas& canvas,
                const PlotRect& rectangle,
                const std::vector<Arrow>& arrows,
                const TargetGrid& grid) {
    const Color white{255,255,255};
    const double length = std::max(5.0, std::min(rectangle.width, rectangle.height) * 0.025);
    for (const Arrow& arrow : arrows) {
        if (!std::isfinite(arrow.u) || !std::isfinite(arrow.v)) continue;
        const double magnitude = std::hypot(static_cast<double>(arrow.u), static_cast<double>(arrow.v));
        if (!(magnitude > 0.0)) continue;
        const double dx = static_cast<double>(arrow.u) / magnitude * length;
        const double dy = -static_cast<double>(arrow.v) / magnitude * length;
        const int centerX = rectangle.left + static_cast<int>(std::nearbyint(
            static_cast<double>(arrow.x) / std::max(grid.nx - 1, 1) * (rectangle.width - 1)));
        const int centerY = rectangle.top + rectangle.height - 1 - static_cast<int>(std::nearbyint(
            static_cast<double>(arrow.y) / std::max(grid.ny - 1, 1) * (rectangle.height - 1)));
        const int x0 = static_cast<int>(std::nearbyint(centerX - dx * 0.5));
        const int y0 = static_cast<int>(std::nearbyint(centerY - dy * 0.5));
        const int x1 = static_cast<int>(std::nearbyint(centerX + dx * 0.5));
        const int y1 = static_cast<int>(std::nearbyint(centerY + dy * 0.5));
        canvas.line(x0, y0, x1, y1, white, 2);
        const double angle = std::atan2(dy, dx);
        constexpr double headAngle = 0.65;
        const double headLength = length * 0.35;
        canvas.line(x1, y1,
                    static_cast<int>(std::nearbyint(x1 - headLength * std::cos(angle - headAngle))),
                    static_cast<int>(std::nearbyint(y1 - headLength * std::sin(angle - headAngle))), white, 2);
        canvas.line(x1, y1,
                    static_cast<int>(std::nearbyint(x1 - headLength * std::cos(angle + headAngle))),
                    static_cast<int>(std::nearbyint(y1 - headLength * std::sin(angle + headAngle))), white, 2);
    }
}

void drawColorbar(Canvas& canvas,
                  int left,
                  int top,
                  int width,
                  int height,
                  const std::function<Color(double)>& colorMap,
                  double minimum,
                  double maximum,
                  std::string_view label) {
    const Color black{0,0,0};
    for (int y = 0; y < height; ++y) {
        const double normalized = 1.0 - static_cast<double>(y) / std::max(height - 1, 1);
        canvas.fill(left, top + y, left + width, top + y + 1, colorMap(normalized));
    }
    canvas.line(left, top, left + width, top, black);
    canvas.line(left, top + height, left + width, top + height, black);
    canvas.line(left, top, left, top + height, black);
    canvas.line(left + width, top, left + width, top + height, black);
    for (int tick = 0; tick <= 4; ++tick) {
        const int y = top + height - height * tick / 4;
        canvas.line(left + width, y, left + width + 7, y, black);
        const std::string text = fixed(minimum + (maximum - minimum) * tick / 4.0, 2);
        drawText(canvas, left + width + 12, y - 3, text, 1, black);
    }
    drawText(canvas, left - 12, top + height + 26, label, 1, black);
}

[[nodiscard]] std::pair<int, int> checkedFigureDimensions(int widthInches,
                                                          int heightInches,
                                                          int dpi) {
    if (dpi <= 0) throw std::runtime_error("Figure DPI must be positive");
    const std::int64_t width = static_cast<std::int64_t>(widthInches) * dpi;
    const std::int64_t height = static_cast<std::int64_t>(heightInches) * dpi;
    if (width > std::numeric_limits<int>::max() || height > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Requested figure dimensions exceed PNG limits");
    }
    return {static_cast<int>(width), static_cast<int>(height)};
}

[[nodiscard]] std::vector<float> readComponent(LegacyVtkPlaneReader& reader,
                                               const LegacyVtkField& field,
                                               std::size_t z,
                                               std::size_t component) {
    return reader.readPointComponentPlane(field, z, component);
}

} // namespace

void exportCroppedVtk(const LegacyStructuredPoints& vtk,
                      const Crop& crop,
                      const ZCrop& zCrop,
                      const fs::path& outputPath,
                      std::ostream& output) {
    const int nx = vtk.dimensions[0];
    const int ny = vtk.dimensions[1];
    const int nz = vtk.dimensions[2];
    if (!(0 <= crop.xStart && crop.xStart < crop.xEnd && crop.xEnd <= nx
          && 0 <= crop.yStart && crop.yStart < crop.yEnd && crop.yEnd <= ny)) {
        throw std::runtime_error("Invalid crop window for VTK export");
    }
    if (!(0 <= zCrop.zStart && zCrop.zStart < zCrop.zEnd && zCrop.zEnd <= nz)) {
        throw std::runtime_error("Invalid z crop window for VTK export");
    }
    if (!outputPath.parent_path().empty()) fs::create_directories(outputPath.parent_path());
    std::ofstream stream(outputPath, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot create cropped VTK: " + pathUtf8(outputPath));
    const int nxOut = crop.nx();
    const int nyOut = crop.ny();
    const int nzOut = zCrop.zEnd - zCrop.zStart;
    const std::size_t outputPoints = static_cast<std::size_t>(nxOut) * static_cast<std::size_t>(nyOut)
        * static_cast<std::size_t>(nzOut);
    stream << "# vtk DataFile Version 3.0\n"
           << "Cropped from " << pathUtf8(vtk.path.filename()) << "\n"
           << "BINARY\n"
           << "DATASET STRUCTURED_POINTS\n"
           << "DIMENSIONS " << nxOut << ' ' << nyOut << ' ' << nzOut << "\n"
           << std::fixed << std::setprecision(10)
           << "ORIGIN " << vtk.origin[0] + crop.xStart * vtk.spacing[0] << ' '
           << vtk.origin[1] + crop.yStart * vtk.spacing[1] << ' '
           << vtk.origin[2] + zCrop.zStart * vtk.spacing[2] << "\n"
           << "SPACING " << vtk.spacing[0] << ' ' << vtk.spacing[1] << ' ' << vtk.spacing[2] << "\n"
           << "POINT_DATA " << outputPoints << "\n";

    LegacyVtkPlaneReader reader(vtk);
    std::ifstream rawInput;
    if (vtk.encoding == LegacyVtkEncoding::Binary) {
        rawInput.open(vtk.path, std::ios::binary);
        if (!rawInput) throw std::runtime_error("Cannot reopen source VTK for cropped export");
    }
    for (const LegacyVtkField& field : vtk.fields) {
        if (field.association != LegacyVtkAssociation::PointData || field.tuples != vtk.pointCount) {
            output << "Skip field '" << field.name << "' during export: tuples=" << field.tuples
                   << " != n_points=" << vtk.pointCount << '\n';
            continue;
        }
        if (field.kind == LegacyVtkFieldKind::Vectors && field.components == 3) {
            stream << "VECTORS " << field.name << ' ' << field.vtkTypeToken << "\n";
        } else {
            stream << "SCALARS " << field.name << ' ' << field.vtkTypeToken;
            if (field.components != 1) stream << ' ' << field.components;
            stream << "\nLOOKUP_TABLE default\n";
        }
        if (vtk.encoding == LegacyVtkEncoding::Binary) {
            const std::size_t tupleBytes = field.components * field.scalarSize();
            const std::size_t sourceRowBytes = static_cast<std::size_t>(nx) * tupleBytes;
            const std::size_t sourcePlaneBytes = static_cast<std::size_t>(ny) * sourceRowBytes;
            const std::size_t outputRowBytes = static_cast<std::size_t>(nxOut) * tupleBytes;
            std::vector<char> row(outputRowBytes);
            for (int z = zCrop.zStart; z < zCrop.zEnd; ++z) for (int y = crop.yStart; y < crop.yEnd; ++y) {
                const std::uint64_t offset = field.byteOffset
                    + static_cast<std::uint64_t>(z) * sourcePlaneBytes
                    + static_cast<std::uint64_t>(y) * sourceRowBytes
                    + static_cast<std::uint64_t>(crop.xStart) * tupleBytes;
                rawInput.clear();
                rawInput.seekg(static_cast<std::streamoff>(offset));
                rawInput.read(row.data(), static_cast<std::streamsize>(row.size()));
                if (rawInput.gcount() != static_cast<std::streamsize>(row.size())) {
                    throw std::runtime_error("Read failure while exporting cropped VTK field " + field.name);
                }
                stream.write(row.data(), static_cast<std::streamsize>(row.size()));
            }
        } else {
            for (int z = zCrop.zStart; z < zCrop.zEnd; ++z) {
                const std::vector<float> plane = reader.readPointPlane(field, static_cast<std::size_t>(z));
                for (int y = crop.yStart; y < crop.yEnd; ++y) for (int x = crop.xStart; x < crop.xEnd; ++x) {
                    const std::size_t tuple = static_cast<std::size_t>(y) * static_cast<std::size_t>(nx)
                        + static_cast<std::size_t>(x);
                    for (std::size_t component = 0; component < field.components; ++component) {
                        writeConvertedValue(stream, field.scalarType,
                                            plane[tuple * field.components + component]);
                    }
                }
            }
        }
    }
    if (!stream) throw std::runtime_error("Failed while writing cropped VTK: " + pathUtf8(outputPath));
    output << "Saved cropped VTK: " << pathUtf8(outputPath) << '\n';
}

void renderRangeDiagnostic(const Crop& crop,
                           const fs::path& outputPath,
                           std::string_view targetCrs,
                           int dpi) {
    if (crop.domainPolygonX.empty() || crop.domainPolygonY.empty()) return;
    Canvas canvas(900, 800);
    const Color black{0,0,0};
    const Color blue{31,119,180};
    const Color red{214,39,40};
    const Color gridColor{220,220,220};
    constexpr PlotRect plot{100, 90, 690, 600};
    double minX = *std::min_element(crop.domainPolygonX.begin(), crop.domainPolygonX.end());
    double maxX = *std::max_element(crop.domainPolygonX.begin(), crop.domainPolygonX.end());
    double minY = *std::min_element(crop.domainPolygonY.begin(), crop.domainPolygonY.end());
    double maxY = *std::max_element(crop.domainPolygonY.begin(), crop.domainPolygonY.end());
    minX = std::min(minX, *std::min_element(crop.selectedRectangleX.begin(), crop.selectedRectangleX.end()));
    maxX = std::max(maxX, *std::max_element(crop.selectedRectangleX.begin(), crop.selectedRectangleX.end()));
    minY = std::min(minY, *std::min_element(crop.selectedRectangleY.begin(), crop.selectedRectangleY.end()));
    maxY = std::max(maxY, *std::max_element(crop.selectedRectangleY.begin(), crop.selectedRectangleY.end()));
    double xSpan = std::max(maxX - minX, 1.0);
    double ySpan = std::max(maxY - minY, 1.0);
    minX -= xSpan * 0.05; maxX += xSpan * 0.05;
    minY -= ySpan * 0.05; maxY += ySpan * 0.05;
    xSpan = maxX - minX; ySpan = maxY - minY;
    const auto point = [&](double x, double y) {
        const int px = plot.left + static_cast<int>(std::nearbyint((x - minX) / xSpan * plot.width));
        const int py = plot.top + plot.height - static_cast<int>(std::nearbyint((y - minY) / ySpan * plot.height));
        return std::pair{px, py};
    };
    for (int tick = 0; tick <= 5; ++tick) {
        const int x = plot.left + plot.width * tick / 5;
        const int y = plot.top + plot.height * tick / 5;
        canvas.line(x, plot.top, x, plot.top + plot.height, gridColor);
        canvas.line(plot.left, y, plot.left + plot.width, y, gridColor);
        const std::string xText = fixed(minX + xSpan * tick / 5.0, 0);
        const std::string yText = fixed(maxY - ySpan * tick / 5.0, 0);
        drawText(canvas, x - textWidth(xText, 1) / 2, plot.top + plot.height + 12, xText, 1, black);
        drawText(canvas, plot.left - textWidth(yText, 1) - 12, y - 3, yText, 1, black);
    }
    const auto selectedMinimum = point(crop.selectedRectangleX[0], crop.selectedRectangleY[0]);
    const auto selectedMaximum = point(crop.selectedRectangleX[2], crop.selectedRectangleY[2]);
    canvas.blend(std::min(selectedMinimum.first, selectedMaximum.first),
                 std::min(selectedMinimum.second, selectedMaximum.second),
                 std::max(selectedMinimum.first, selectedMaximum.first),
                 std::max(selectedMinimum.second, selectedMaximum.second), red, 0.10);
    for (std::size_t index = 1; index < crop.domainPolygonX.size(); ++index) {
        const auto previous = point(crop.domainPolygonX[index - 1], crop.domainPolygonY[index - 1]);
        const auto current = point(crop.domainPolygonX[index], crop.domainPolygonY[index]);
        canvas.line(previous.first, previous.second, current.first, current.second, blue, 3);
    }
    for (std::size_t index = 1; index < crop.selectedRectangleX.size(); ++index) {
        const auto previous = point(crop.selectedRectangleX[index - 1], crop.selectedRectangleY[index - 1]);
        const auto current = point(crop.selectedRectangleX[index], crop.selectedRectangleY[index]);
        canvas.line(previous.first, previous.second, current.first, current.second, red, 3);
    }
    canvas.line(plot.left, plot.top, plot.left, plot.top + plot.height, black);
    canvas.line(plot.left, plot.top + plot.height, plot.left + plot.width, plot.top + plot.height, black);
    const std::string title = "RANGE DIAGNOSTIC: VTK TOTAL DOMAIN VS SELECTED RANGE";
    drawText(canvas, 450 - textWidth(title, 2) / 2, 30, title, 2, black);
    drawText(canvas, 450 - textWidth("VTK TOTAL DOMAIN", 1) / 2, 715, "VTK TOTAL DOMAIN", 1, blue);
    drawText(canvas, 450 - textWidth("SELECTED RANGE", 1) / 2, 735, "SELECTED RANGE", 1, red);
    const std::string xLabel = "X (" + std::string(targetCrs) + ")";
    drawText(canvas, 450 - textWidth(xLabel, 1) / 2, 765, xLabel, 1, black);
    const auto [width, height] = checkedFigureDimensions(9, 8, dpi);
    writeScaledPng(outputPath, canvas, width, height);
}

void renderWindFigure(const LegacyStructuredPoints& vtk,
                      const WindFields& fields,
                      const std::array<HeightItem, 9>& plan,
                      const TargetGrid& grid,
                      int quiverStep,
                      const fs::path& outputPath,
                      int dpi) {
    const int sourceNx = vtk.dimensions[0];
    const int sourceNy = vtk.dimensions[1];
    const int step = std::max(1, quiverStep);
    LegacyVtkPlaneReader reader(vtk);
    std::array<WindLayer, 9> layers;
    std::vector<float> normalizationValues;
    for (std::size_t layerIndex = 0; layerIndex < plan.size(); ++layerIndex) {
        if (!plan[layerIndex].valid) continue;
        const std::size_t z = static_cast<std::size_t>(plan[layerIndex].zIndex);
        std::vector<float> sourceU;
        std::vector<float> sourceV;
        std::vector<float> sourceW;
        if (fields.vector != nullptr) {
            sourceU = readComponent(reader, *fields.vector, z, 0);
            sourceV = readComponent(reader, *fields.vector, z, 1);
            sourceW = readComponent(reader, *fields.vector, z, 2);
        } else {
            sourceU = readComponent(reader, *fields.u, z, 0);
            sourceV = readComponent(reader, *fields.v, z, 0);
            sourceW = readComponent(reader, *fields.w, z, 0);
        }
        std::vector<float> u = resample(sourceU, sourceNx, sourceNy, grid);
        std::vector<float> v = resample(sourceV, sourceNx, sourceNy, grid);
        const std::vector<float> w = resample(sourceW, sourceNx, sourceNy, grid);
        WindLayer& layer = layers[layerIndex];
        layer.speed.resize(u.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t signedIndex = 0; signedIndex < static_cast<std::int64_t>(u.size()); ++signedIndex) {
            const std::size_t index = static_cast<std::size_t>(signedIndex);
            const float localU = u[index];
            const float localV = v[index];
            u[index] = static_cast<float>(grid.vectorCos * localU - grid.vectorSin * localV);
            v[index] = static_cast<float>(grid.vectorSin * localU + grid.vectorCos * localV);
            layer.speed[index] = std::sqrt(u[index] * u[index] + v[index] * v[index] + w[index] * w[index]);
        }
        for (int y = 0; y < grid.ny; y += step) for (int x = 0; x < grid.nx; x += step) {
            const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(grid.nx)
                + static_cast<std::size_t>(x);
            layer.arrows.push_back({x, y, u[index], v[index]});
        }
        for (int y = 0; y < grid.ny; y += 4) for (int x = 0; x < grid.nx; x += 4) {
            const float value = layer.speed[static_cast<std::size_t>(y) * static_cast<std::size_t>(grid.nx)
                                            + static_cast<std::size_t>(x)];
            if (std::isfinite(value)) normalizationValues.push_back(value);
        }
    }
    double minimum = percentile(normalizationValues, 2.0);
    double maximum = percentile(normalizationValues, 98.0);
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || maximum <= minimum) {
        if (normalizationValues.empty()) {
            minimum = 0.0; maximum = 1.0;
        } else {
            const auto [minIterator, maxIterator] = std::minmax_element(normalizationValues.begin(), normalizationValues.end());
            minimum = *minIterator;
            maximum = *maxIterator;
            if (maximum <= minimum) maximum = minimum + 1.0;
        }
    }

    Canvas canvas(1800, 1600);
    const Color black{0,0,0};
    constexpr int left = 45;
    constexpr int top = 115;
    constexpr int right = 1650;
    constexpr int bottom = 1510;
    constexpr int gapX = 24;
    constexpr int gapY = 28;
    const int cellWidth = (right - left - gapX * 2) / 3;
    const int cellHeight = (bottom - top - gapY * 2) / 3;
    for (std::size_t index = 0; index < plan.size(); ++index) {
        const int column = static_cast<int>(index % 3);
        const int row = static_cast<int>(index / 3);
        const PlotRect cell{left + column * (cellWidth + gapX), top + row * (cellHeight + gapY),
                            cellWidth, cellHeight};
        const PlotRect plot = aspectContained({cell.left, cell.top + 25, cell.width, cell.height - 25}, grid);
        const std::string panelTitle = std::to_string(plan[index].targetHeight) + "M (Z="
            + std::to_string(plan[index].zIndex) + ")";
        drawText(canvas, cell.left + cell.width / 2 - textWidth(panelTitle, 2) / 2,
                 cell.top, panelTitle, 2, black);
        if (!plan[index].valid) {
            const std::string out = "OUT_OF_RANGE";
            drawText(canvas, cell.left + cell.width / 2 - textWidth(out, 2) / 2,
                     cell.top + cell.height / 2, out, 2, black);
            continue;
        }
        renderScalarPanel(canvas, plot, layers[index].speed, grid, minimum, maximum, turbo);
        drawArrows(canvas, plot, layers[index].arrows, grid);
    }
    const std::string title = "3D WIND ARROWS (9 LAYERS) | " + pathUtf8(vtk.path.filename());
    drawText(canvas, 900 - textWidth(title, 3) / 2, 28, title, 3, black);
    const std::string subtitle = "HEIGHT=-50+10*Z, QUIVER_STEP=" + std::to_string(step)
        + ", TARGET_GRID=" + std::to_string(grid.nx) + "X" + std::to_string(grid.ny);
    drawText(canvas, 900 - textWidth(subtitle, 2) / 2, 65, subtitle, 2, black);
    drawColorbar(canvas, 1680, 260, 28, 1020, turbo, minimum, maximum, "3D WIND SPEED (M/S)");
    const auto [width, height] = checkedFigureDimensions(18, 16, dpi);
    writeScaledPng(outputPath, canvas, width, height);
}

void renderTkeFigure(const LegacyStructuredPoints& vtk,
                     const LegacyVtkField& field,
                     const std::array<HeightItem, 9>& plan,
                     const TargetGrid& grid,
                     const fs::path& outputPath,
                     int dpi) {
    LegacyVtkPlaneReader reader(vtk);
    std::array<std::vector<float>, 9> layers;
    std::vector<double> lowValues;
    std::vector<double> highValues;
    for (std::size_t index = 0; index < plan.size(); ++index) {
        if (!plan[index].valid) continue;
        const std::vector<float> source = readComponent(reader, field,
            static_cast<std::size_t>(plan[index].zIndex), 0);
        layers[index] = resample(source, vtk.dimensions[0], vtk.dimensions[1], grid);
        const double low = percentile(layers[index], 2.0);
        const double high = percentile(layers[index], 98.0);
        if (std::isfinite(low)) lowValues.push_back(low);
        if (std::isfinite(high)) highValues.push_back(high);
    }
    double minimum = lowValues.empty() ? 0.0 : *std::min_element(lowValues.begin(), lowValues.end());
    double maximum = highValues.empty() ? 1.0 : *std::max_element(highValues.begin(), highValues.end());
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || maximum <= minimum) {
        std::vector<float> all;
        for (const auto& values : layers) for (float value : values) if (std::isfinite(value)) all.push_back(value);
        if (all.empty()) {
            minimum = 0.0; maximum = 1.0;
        } else {
            const auto [minIterator, maxIterator] = std::minmax_element(all.begin(), all.end());
            minimum = *minIterator;
            maximum = *maxIterator;
            if (maximum <= minimum) maximum = minimum + 1.0;
        }
    }

    Canvas canvas(1800, 1600);
    const Color black{0,0,0};
    constexpr int left = 45;
    constexpr int top = 115;
    constexpr int right = 1650;
    constexpr int bottom = 1510;
    constexpr int gapX = 24;
    constexpr int gapY = 28;
    const int cellWidth = (right - left - gapX * 2) / 3;
    const int cellHeight = (bottom - top - gapY * 2) / 3;
    for (std::size_t index = 0; index < plan.size(); ++index) {
        const int column = static_cast<int>(index % 3);
        const int row = static_cast<int>(index / 3);
        const PlotRect cell{left + column * (cellWidth + gapX), top + row * (cellHeight + gapY),
                            cellWidth, cellHeight};
        const PlotRect plot = aspectContained({cell.left, cell.top + 25, cell.width, cell.height - 25}, grid);
        const std::string panelTitle = std::to_string(plan[index].targetHeight) + "M (Z="
            + std::to_string(plan[index].zIndex) + ")";
        drawText(canvas, cell.left + cell.width / 2 - textWidth(panelTitle, 2) / 2,
                 cell.top, panelTitle, 2, black);
        if (!plan[index].valid) {
            const std::string out = "OUT_OF_RANGE";
            drawText(canvas, cell.left + cell.width / 2 - textWidth(out, 2) / 2,
                     cell.top + cell.height / 2, out, 2, black);
            continue;
        }
        renderScalarPanel(canvas, plot, layers[index], grid, minimum, maximum, magma);
    }
    const std::string title = "TKE (9 LAYERS) | " + pathUtf8(vtk.path.filename());
    drawText(canvas, 900 - textWidth(title, 3) / 2, 28, title, 3, black);
    const std::string subtitle = "HEIGHT=-50+10*Z, TARGET_GRID="
        + std::to_string(grid.nx) + "X" + std::to_string(grid.ny);
    drawText(canvas, 900 - textWidth(subtitle, 2) / 2, 65, subtitle, 2, black);
    drawColorbar(canvas, 1680, 260, 28, 1020, magma, minimum, maximum, "TKE (M^2/S^2)");
    const auto [width, height] = checkedFigureDimensions(18, 16, dpi);
    writeScaledPng(outputPath, canvas, width, height);
}

std::vector<fs::path> discoverVtkFiles(const fs::path& directory,
                                       std::string_view patternList) {
    std::vector<std::string> patterns;
    std::size_t start = 0;
    while (start <= patternList.size()) {
        const std::size_t comma = patternList.find(',', start);
        std::string pattern(patternList.substr(start, comma == std::string_view::npos
            ? patternList.size() - start : comma - start));
        const std::size_t first = pattern.find_first_not_of(" \t\r\n");
        const std::size_t last = pattern.find_last_not_of(" \t\r\n");
        if (first != std::string::npos) patterns.push_back(pattern.substr(first, last - first + 1));
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    if (patterns.empty()) patterns = {"*_avg-*.vtk", "*_avg_*.vtk"};
    auto wildcardRegex = [](std::string_view pattern) {
        std::string expression = "^";
        for (const char character : pattern) {
            if (character == '*') expression += ".*";
            else if (character == '?') expression += '.';
            else {
                if (std::string_view(R"(\.^$|()[]{}+)").find(character) != std::string_view::npos) expression += '\\';
                expression += character;
            }
        }
        expression += '$';
#ifdef _WIN32
        return std::regex(expression, std::regex::ECMAScript | std::regex::icase);
#else
        return std::regex(expression, std::regex::ECMAScript);
#endif
    };
    std::vector<std::regex> matchers;
    for (const std::string& pattern : patterns) matchers.push_back(wildcardRegex(pattern));
    std::set<fs::path> unique;
    std::error_code iteratorError;
    for (fs::directory_iterator iterator(directory, iteratorError), end;
         !iteratorError && iterator != end; iterator.increment(iteratorError)) {
        std::error_code typeError;
        if (!iterator->is_regular_file(typeError) || typeError) continue;
        const std::string name = pathUtf8(iterator->path().filename());
        if (std::any_of(matchers.begin(), matchers.end(), [&](const std::regex& matcher) {
            return std::regex_match(name, matcher);
        })) unique.insert(fs::absolute(iterator->path()).lexically_normal());
    }
    if (iteratorError) throw std::runtime_error("Cannot enumerate VTK input directory: " + iteratorError.message());
    std::vector<fs::path> result(unique.begin(), unique.end());
    const std::regex rawAverage(R"(^.+_avg[-_](\d+)\.vtk$)");
    auto step = [&](const fs::path& path) -> std::optional<long long> {
        std::smatch match;
        const std::string name = pathUtf8(path.filename());
        if (!std::regex_match(name, match, rawAverage)) return std::nullopt;
        try { return std::stoll(match[1].str()); } catch (const std::exception&) { return std::nullopt; }
    };
    std::sort(result.begin(), result.end(), [&](const fs::path& left, const fs::path& right) {
        const auto leftStep = step(left);
        const auto rightStep = step(right);
        if (leftStep && rightStep && *leftStep != *rightStep) return *leftStep < *rightStep;
        if (leftStep.has_value() != rightStep.has_value()) return leftStep.has_value();
        if (!leftStep) {
            std::error_code leftError, rightError;
            const auto leftTime = fs::last_write_time(left, leftError);
            const auto rightTime = fs::last_write_time(right, rightError);
            if (!leftError && !rightError && leftTime != rightTime) return leftTime < rightTime;
        }
        return pathUtf8(left.filename()) < pathUtf8(right.filename());
    });
    if (result.empty()) {
        std::ostringstream message;
        message << "No VTK files found in " << pathUtf8(directory) << " with patterns=[";
        for (std::size_t index = 0; index < patterns.size(); ++index) {
            if (index != 0) message << ", ";
            message << patterns[index];
        }
        message << ']';
        throw std::runtime_error(message.str());
    }
    return result;
}

} // namespace luw::post::tke
