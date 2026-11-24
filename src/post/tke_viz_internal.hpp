#pragma once

#include "luw/legacy_vtk.hpp"

#include <array>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace luw::post::tke {

inline constexpr std::array<int, 9> targetHeights{50, 100, 150, 200, 300, 400, 500, 600, 800};
inline constexpr double baseHeight = -50.0;
inline constexpr double layerStep = 10.0;

struct RuntimeConfig final {
    std::filesystem::path configPath;
    std::filesystem::path dataDirectory = ".";
    std::string dataDirectorySource;
    std::filesystem::path outputDirectory = "./tke_viz_output";
    std::string targetCrs = "LOCAL_XY";
    double minLon = 0.0;
    double maxLon = 0.0;
    double minLat = 0.0;
    double maxLat = 0.0;
    double cellSize = 10.0;
    int visDpi = 1200;
    std::optional<double> cropZMax;
    std::string fileGlob = "*_avg-*.vtk,*_avg_*.vtk";
    std::pair<double, double> domainLon{};
    std::pair<double, double> domainLat{};
    std::string utmCrs;
    std::optional<double> rotateDegrees;
};

struct Bounds final {
    double minX = 0.0;
    double maxX = 0.0;
    double minY = 0.0;
    double maxY = 0.0;
    double defaultMinX = 0.0;
    double defaultMaxX = 0.0;
    double defaultMinY = 0.0;
    double defaultMaxY = 0.0;
};

class GeoModel;

struct Crop final {
    int xStart = 0;
    int xEnd = 0;
    int yStart = 0;
    int yEnd = 0;
    std::size_t insideCount = 0;
    double insideRatio = 0.0;
    std::string utmCrs = "LOCAL_XY";
    double rotateDegrees = 0.0;
    std::pair<double, double> domainLon{};
    std::pair<double, double> domainLat{};
    std::vector<double> domainPolygonX;
    std::vector<double> domainPolygonY;
    std::array<double, 5> selectedRectangleX{};
    std::array<double, 5> selectedRectangleY{};
    std::shared_ptr<GeoModel> model;

    [[nodiscard]] int nx() const noexcept { return xEnd - xStart; }
    [[nodiscard]] int ny() const noexcept { return yEnd - yStart; }
};

struct TargetGrid final {
    int nx = 0;
    int ny = 0;
    double minX = 0.0;
    double maxX = 0.0;
    double minY = 0.0;
    double maxY = 0.0;
    double step = 10.0;
    double vectorCos = 1.0;
    double vectorSin = 0.0;
    std::vector<double> xIndex;
    std::vector<double> yIndex;
    std::vector<unsigned char> valid;
};

struct HeightItem final {
    int targetHeight = 0;
    int zIndex = 0;
    bool valid = false;
    double mappedHeight = 0.0;
};

struct ZCrop final {
    int zStart = 0;
    int zEnd = 0;
    int nzOut = 0;
    double zMin = 0.0;
    double zMaxKept = 0.0;
    bool enabled = false;
};

struct WindFields final {
    const LegacyVtkField* vector = nullptr;
    const LegacyVtkField* u = nullptr;
    const LegacyVtkField* v = nullptr;
    const LegacyVtkField* w = nullptr;

    [[nodiscard]] std::string label() const;
};

[[nodiscard]] Bounds localBounds(const LegacyStructuredPoints& vtk,
                                 std::optional<double> minX,
                                 std::optional<double> maxX,
                                 std::optional<double> minY,
                                 std::optional<double> maxY);
[[nodiscard]] Bounds geographicBounds(const RuntimeConfig& config,
                                      std::optional<double> minX,
                                      std::optional<double> maxX,
                                      std::optional<double> minY,
                                      std::optional<double> maxY);
[[nodiscard]] Crop localCrop(const LegacyStructuredPoints& vtk, const Bounds& bounds);
[[nodiscard]] Crop geographicCrop(const LegacyStructuredPoints& vtk,
                                  const RuntimeConfig& config,
                                  const Bounds& bounds);
[[nodiscard]] TargetGrid localTargetGrid(const LegacyStructuredPoints& vtk,
                                         const RuntimeConfig& config,
                                         const Bounds& bounds);
[[nodiscard]] TargetGrid geographicTargetGrid(const LegacyStructuredPoints& vtk,
                                              const RuntimeConfig& config,
                                              const Bounds& bounds,
                                              const Crop& crop);
[[nodiscard]] std::array<HeightItem, 9> heightPlan(const LegacyStructuredPoints& vtk);
[[nodiscard]] ZCrop computeZCrop(const LegacyStructuredPoints& vtk,
                                 std::optional<double> maximumZ);
[[nodiscard]] WindFields selectWindFields(const LegacyStructuredPoints& vtk,
                                          bool exactLegacyNames);
[[nodiscard]] const LegacyVtkField& selectTkeField(const LegacyStructuredPoints& vtk,
                                                   bool exactLegacyName);

void exportCroppedVtk(const LegacyStructuredPoints& vtk,
                      const Crop& crop,
                      const ZCrop& zCrop,
                      const std::filesystem::path& outputPath,
                      std::ostream& output);

void renderRangeDiagnostic(const Crop& crop,
                           const std::filesystem::path& outputPath,
                           std::string_view targetCrs,
                           int dpi);
void renderWindFigure(const LegacyStructuredPoints& vtk,
                      const WindFields& fields,
                      const std::array<HeightItem, 9>& plan,
                      const TargetGrid& grid,
                      int quiverStep,
                      const std::filesystem::path& outputPath,
                      int dpi);
void renderTkeFigure(const LegacyStructuredPoints& vtk,
                     const LegacyVtkField& field,
                     const std::array<HeightItem, 9>& plan,
                     const TargetGrid& grid,
                     const std::filesystem::path& outputPath,
                     int dpi);

[[nodiscard]] std::vector<std::filesystem::path> discoverVtkFiles(
    const std::filesystem::path& directory,
    std::string_view patternList);

} // namespace luw::post::tke
