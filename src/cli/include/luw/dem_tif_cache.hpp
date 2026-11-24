#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

namespace luw::cli {

struct DemLonLatBounds {
    double lonMin = 0.0;
    double lonMax = 0.0;
    double latMin = 0.0;
    double latMax = 0.0;
};

struct DemTifCacheRequest {
    std::filesystem::path projectHome;
    std::filesystem::path installDatabaseDem;
    std::string caseName;
    DemLonLatBounds targetBounds;

    // Production callers leave this null so the legacy five-second console
    // prompt is used. Tests may provide a stream and/or a shorter timeout to
    // exercise the same answer handling without touching global stdin.
    std::istream* coverageInput = nullptr;
    double coverageTimeoutSeconds = 5.0;
};

class DemTifCacheCancelled final : public std::runtime_error {
  public:
    DemTifCacheCancelled()
        : std::runtime_error("DEM GeoTIFF conversion canceled by user") {}
};

struct DemTifCacheResult {
    std::filesystem::path shapefile;
    std::filesystem::path sourceRaster;
    bool created = false;
    std::size_t featureCount = 0U;
};

// Reproduces dem_tif_to_shp.ensure_dem_shp_from_tif without a Python runtime.
// An existing <casename>.shp is never modified. The caller supplies the
// resolved installation database/dem.tif path so this helper remains usable by
// both luwbc and luwvox without owning runtime-path policy.
std::optional<DemTifCacheResult> ensureDemShapefileFromTif(
    const DemTifCacheRequest& request,
    std::ostream& output);

} // namespace luw::cli
