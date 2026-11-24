#pragma once

#include "luw/command_registry.hpp"

#include <iosfwd>
#include <vector>

namespace luw::cli {

struct BuildingScaleDistribution final {
    std::vector<double> valuesDescending;
    std::vector<double> cumulativeWeights;
    std::vector<double> binEdges;
    std::vector<double> binCenters;
    std::vector<double> probabilityDensity;
};

// NumPy-compatible unweighted Freedman-Diaconis bins, area*height weighted
// histogram density, and descending weighted cumulative distribution.
[[nodiscard]] BuildingScaleDistribution calculateBuildingScaleDistribution(
    const std::vector<double>& values,
    const std::vector<double>& weights);

int run_buildingscale(const CommandArguments& arguments,
                      std::ostream& output,
                      std::ostream& error);

} // namespace luw::cli
