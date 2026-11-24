#pragma once

#include <algorithm>
#include <cmath>
#include <span>

namespace luw::cli::detail {

// Matches np.ptp(local_elevations, axis=1) <= 1e-12 in the primary batched
// kriging path.
inline bool krigingBatchConstant(std::span<const double> elevations) {
    if (elevations.empty()) return true;
    const auto [minimum, maximum] =
        std::minmax_element(elevations.begin(), elevations.end());
    return *maximum - *minimum <= 1.0e-12;
}

// Matches np.allclose(local_elevations, local_elevations[0]) in the Python
// row-wise fallback (NumPy defaults: rtol=1e-5, atol=1e-8).
inline bool krigingRowwiseAllClose(std::span<const double> elevations) {
    if (elevations.empty()) return true;
    const double reference = elevations.front();
    return std::all_of(elevations.begin(), elevations.end(),
                       [reference](double elevation) {
                           return std::abs(elevation - reference) <=
                                  1.0e-8 + 1.0e-5 * std::abs(reference);
                       });
}

} // namespace luw::cli::detail
