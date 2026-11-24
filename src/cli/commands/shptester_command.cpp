#include "luw/shptester_command.hpp"

#ifndef LUW_HAS_SHPTESTER_GIS
#define LUW_HAS_SHPTESTER_GIS 0
#endif

#if LUW_HAS_SHPTESTER_GIS
#include "shptester_diagnostics.hpp"

#define GEOS_USE_ONLY_R_API
#include <geos_c.h>

#include <cpl_error.h>
#include <gdal_priv.h>
#include <ogr_geometry.h>
#include <ogrsf_frmts.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace luw::cli {

namespace {

constexpr const char* legacy_usage =
    "Usage: python check_degenerate_shp.py <path_to_shp>";

#if LUW_HAS_SHPTESTER_GIS

struct DatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) {
            GDALClose(dataset);
        }
    }
};

struct FeatureCloser {
    void operator()(OGRFeature* feature) const noexcept {
        if (feature != nullptr) {
            OGRFeature::DestroyFeature(feature);
        }
    }
};

class QuietGdalErrors final {
public:
    QuietGdalErrors() {
        CPLErrorReset();
        CPLPushErrorHandler(CPLQuietErrorHandler);
    }

    ~QuietGdalErrors() {
        CPLPopErrorHandler();
    }

    QuietGdalErrors(const QuietGdalErrors&) = delete;
    QuietGdalErrors& operator=(const QuietGdalErrors&) = delete;
};

std::string last_gdal_error(const std::string& fallback) {
    const char* message = CPLGetLastErrorMsg();
    if (message != nullptr && *message != '\0') {
        return message;
    }
    return fallback;
}

struct ReasonCount {
    std::string reason;
    std::uint64_t count = 0;
};

void increment_reason(std::vector<ReasonCount>& reasons, const std::string& reason) {
    const auto existing = std::find_if(
        reasons.begin(), reasons.end(), [&](const ReasonCount& item) {
            return item.reason == reason;
        });
    if (existing != reasons.end()) {
        ++existing->count;
        return;
    }
    reasons.push_back({reason, 1});
}

#endif

}  // namespace

#if LUW_HAS_SHPTESTER_GIS

namespace detail {

namespace {

bool same_ring_coordinate(const OGRLinearRing& ring, int left, int right) {
    if (ring.getX(left) != ring.getX(right)
        || ring.getY(left) != ring.getY(right)) {
        return false;
    }
    if (ring.Is3D() && ring.getZ(left) != ring.getZ(right)) {
        return false;
    }
    if (ring.IsMeasured() && ring.getM(left) != ring.getM(right)) {
        return false;
    }
    return true;
}

std::string shapely_geometry_type_name(OGRwkbGeometryType type) {
    switch (wkbFlatten(type)) {
    case wkbPoint:
        return "Point";
    case wkbLineString:
        return "LineString";
    case wkbPolygon:
        return "Polygon";
    case wkbMultiPoint:
        return "MultiPoint";
    case wkbMultiLineString:
        return "MultiLineString";
    case wkbMultiPolygon:
        return "MultiPolygon";
    case wkbGeometryCollection:
        return "GeometryCollection";
    case wkbCircularString:
        return "CircularString";
    case wkbCompoundCurve:
        return "CompoundCurve";
    case wkbCurvePolygon:
        return "CurvePolygon";
    case wkbMultiCurve:
        return "MultiCurve";
    case wkbMultiSurface:
        return "MultiSurface";
    case wkbPolyhedralSurface:
        return "PolyhedralSurface";
    case wkbTIN:
        return "TIN";
    case wkbTriangle:
        return "Triangle";
    case wkbUnknown:
        return "Unknown";
    case wkbNone:
        return "None";
    default:
        return OGRGeometryTypeToName(type);
    }
}

}  // namespace

struct ShpTesterGeometryDiagnoser::Impl {
    GEOSContextHandle_t context = nullptr;
    GEOSWKBReader* reader = nullptr;

    Impl() {
        context = GEOS_init_r();
        if (context == nullptr) {
            throw std::runtime_error("Could not initialize GEOS.");
        }
        reader = GEOSWKBReader_create_r(context);
        if (reader == nullptr) {
            GEOS_finish_r(context);
            context = nullptr;
            throw std::runtime_error("Could not initialize the GEOS WKB reader.");
        }
    }

    ~Impl() {
        if (reader != nullptr) {
            GEOSWKBReader_destroy_r(context, reader);
        }
        if (context != nullptr) {
            GEOS_finish_r(context);
        }
    }

    [[nodiscard]] std::pair<bool, std::string>
    validity(const OGRGeometry& geometry) const {
        std::vector<unsigned char> wkb(geometry.WkbSize());
        if (geometry.exportToWkb(wkbNDR, wkb.data(), wkbVariantIso)
            != OGRERR_NONE) {
            throw std::runtime_error("Could not export an OGR geometry for GEOS validity checking.");
        }

        GEOSGeometry* raw_geometry =
            GEOSWKBReader_read_r(context, reader, wkb.data(), wkb.size());
        if (raw_geometry == nullptr) {
            throw std::runtime_error("GEOS could not read an OGR geometry for validity checking.");
        }
        const auto geometry_deleter = [this](GEOSGeometry* value) {
            GEOSGeom_destroy_r(context, value);
        };
        const std::unique_ptr<GEOSGeometry, decltype(geometry_deleter)> geos_geometry(
            raw_geometry, geometry_deleter);

        const char valid = GEOSisValid_r(context, geos_geometry.get());
        if (valid == 1) {
            return {true, {}};
        }

        char* raw_reason = GEOSisValidReason_r(context, geos_geometry.get());
        std::string reason = raw_reason != nullptr ? raw_reason : "Unknown";
        if (raw_reason != nullptr) {
            GEOSFree_r(context, raw_reason);
        }
        return {false, std::move(reason)};
    }
};

ShpTesterGeometryDiagnoser::ShpTesterGeometryDiagnoser()
    : impl_(std::make_unique<Impl>()) {}

ShpTesterGeometryDiagnoser::~ShpTesterGeometryDiagnoser() = default;
ShpTesterGeometryDiagnoser::ShpTesterGeometryDiagnoser(
    ShpTesterGeometryDiagnoser&&) noexcept = default;
ShpTesterGeometryDiagnoser& ShpTesterGeometryDiagnoser::operator=(
    ShpTesterGeometryDiagnoser&&) noexcept = default;

ShpTesterDiagnosis ShpTesterGeometryDiagnoser::diagnose(
    const OGRGeometry* geometry) const {
    if (geometry == nullptr) {
        return {true, "null"};
    }
    if (geometry->IsEmpty()) {
        return {true, "empty"};
    }

    const auto diagnose_polygon_part = [&](const OGRPolygon& polygon) {
        if (polygon.IsEmpty()) {
            return ShpTesterDiagnosis{true, "empty"};
        }
        const OGRLinearRing* exterior = polygon.getExteriorRing();
        const int point_count = exterior != nullptr ? exterior->getNumPoints() : 0;
        if (point_count < 4) {
            return ShpTesterDiagnosis{true, "too_few_points"};
        }
        if (!same_ring_coordinate(*exterior, 0, point_count - 1)) {
            return ShpTesterDiagnosis{true, "ring_not_closed"};
        }
        if (polygon.get_Area() <= 0.0) {
            return ShpTesterDiagnosis{true, "zero_area"};
        }
        const auto [valid, reason] = impl_->validity(polygon);
        if (!valid) {
            return ShpTesterDiagnosis{true, "invalid: " + reason};
        }
        return ShpTesterDiagnosis{false, "ok"};
    };

    const OGRwkbGeometryType type = wkbFlatten(geometry->getGeometryType());
    if (type == wkbPolygon) {
        return diagnose_polygon_part(*static_cast<const OGRPolygon*>(geometry));
    }
    if (type == wkbMultiPolygon) {
        const auto& multi = *static_cast<const OGRMultiPolygon*>(geometry);
        if (multi.getNumGeometries() == 0) {
            // Kept for source parity.  OGR, like Shapely, normally reports an
            // empty collection above before this branch can be reached.
            return {true, "empty_multipolygon"};
        }
        for (int index = 0; index < multi.getNumGeometries(); ++index) {
            const ShpTesterDiagnosis part =
                diagnose_polygon_part(*multi.getGeometryRef(index));
            if (part.degenerate) {
                return {true, "multipart: " + part.reason};
            }
        }
        if (multi.get_Area() <= 0.0) {
            return {true, "zero_area"};
        }
        const auto [valid, reason] = impl_->validity(multi);
        if (!valid) {
            return {true, "invalid: " + reason};
        }
        return {false, "ok"};
    }

    return {true, "not_polygon_type: "
                      + shapely_geometry_type_name(geometry->getGeometryType())};
}

}  // namespace detail

#endif

int run_shptester(const std::vector<std::string>& args,
                  std::ostream& out,
                  std::ostream& err) {
    if (args.size() != 1) {
        err << legacy_usage << '\n';
        return 2;
    }

#if !LUW_HAS_SHPTESTER_GIS
    (void)out;
    err << "Failed to read shapefile: GDAL/OGR with GEOS support is unavailable "
           "in this build.\n";
    return 1;
#else
    GDALAllRegister();
    QuietGdalErrors quiet_errors;
    const unsigned int open_flags =
        GDAL_OF_VECTOR | GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR;
    std::unique_ptr<GDALDataset, DatasetCloser> dataset(
        static_cast<GDALDataset*>(
            GDALOpenEx(args.front().c_str(), open_flags, nullptr, nullptr, nullptr)));
    if (!dataset) {
        err << "Failed to read shapefile: "
            << last_gdal_error(args.front() + ": No such file or directory") << '\n';
        return 1;
    }

    OGRLayer* layer = dataset->GetLayer(0);
    if (layer == nullptr) {
        err << "Failed to read shapefile: "
            << last_gdal_error("No layers found in " + args.front()) << '\n';
        return 1;
    }
    OGRFeatureDefn* definition = layer->GetLayerDefn();
    if (definition == nullptr || definition->GetGeomFieldCount() == 0) {
        err << "No geometry column found.\n";
        return 1;
    }

    detail::ShpTesterGeometryDiagnoser diagnoser;
    std::uint64_t total = 0;
    std::uint64_t degenerate = 0;
    std::vector<ReasonCount> reasons;
    layer->ResetReading();
    while (true) {
        CPLErrorReset();
        std::unique_ptr<OGRFeature, FeatureCloser> feature(layer->GetNextFeature());
        if (!feature) {
            if (CPLGetLastErrorType() >= CE_Failure) {
                err << "Failed to read shapefile: "
                    << last_gdal_error("Could not read a feature from " + args.front())
                    << '\n';
                return 1;
            }
            break;
        }
        ++total;
        const detail::ShpTesterDiagnosis diagnosis =
            diagnoser.diagnose(feature->GetGeometryRef());
        if (diagnosis.degenerate) {
            ++degenerate;
            increment_reason(reasons, diagnosis.reason);
        }
    }

    out << "Total features: " << total << '\n';
    out << "Degenerate features: " << degenerate << '\n';
    std::stable_sort(reasons.begin(), reasons.end(),
                     [](const ReasonCount& left, const ReasonCount& right) {
                         return left.count > right.count;
                     });
    for (const ReasonCount& reason : reasons) {
        out << reason.count << '\t' << reason.reason << '\n';
    }
    return 0;
#endif
}

}  // namespace luw::cli
