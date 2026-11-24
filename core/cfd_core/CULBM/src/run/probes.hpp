#pragma once

// Velocity probes of a LUW case: deck tokens (lon:lat, center/centre with an
// optional grid or metre offset), the geographic mapping from lon/lat to the
// local CFD frame (UTM WGS84, rotated like the preprocessor), and the
// resolved lattice columns with their sampled time series.

#include "../core/core.hpp"

namespace culbm::cases {
struct CaseContext;
}

namespace culbm::run {

enum class ProbeOffsetMode {
    NONE = 0,
    GRID_CELLS = 1,
    METERS = 2
};

struct ProbeOffset {
    ProbeOffsetMode mode = ProbeOffsetMode::NONE;
    int north_cells = 0;
    int east_cells = 0;
    double north_m = 0.0;
    double east_m = 0.0;
    std::string label;
};

struct ProbeRequest {
    std::string raw_token;
    double lon_deg = 0.0;
    double lat_deg = 0.0;
    bool uses_center = false;
    ProbeOffset offset;
};

struct ProbeGeoMapping {
    bool valid = false;
    int utm_zone = 0;
    bool utm_north = true;
    double rotate_deg = 0.0;
    double pivot_x = 0.0;
    double pivot_y = 0.0;
    double x_min_rot = 0.0;
    double y_min_rot = 0.0;
    double center_lon_deg = 0.0;
    double center_lat_deg = 0.0;
    double east_dx = 1.0;
    double east_dy = 0.0;
    double north_dx = 0.0;
    double north_dy = 1.0;
};

struct ResolvedProbe {
    ProbeRequest request;
    std::string file_stem;
    std::string label;
    uint x = 0u;
    uint y = 0u;
    double snapped_x_si = 0.0;
    double snapped_y_si = 0.0;
    std::vector<uint> z_indices;
    std::vector<float> heights_si;
    std::vector<double> times_si;
    std::vector<float> velocity_series_si; // [time][level][uvw]
};

struct ProbePlan {
    std::vector<ProbeRequest> probe_requests;
    ProbeGeoMapping probe_geo_mapping;
};

int auto_utm_zone_from_lon(double lon_deg);
bool parse_utm_zone_from_crs(const std::string& utm_crs, int& zone, bool& north);
bool lonlat_to_utm_wgs84(double lon_deg, double lat_deg, int zone, bool north_hemisphere,
                         double& x_utm, double& y_utm);
void rotate_xy(double x, double y, double deg, double cx, double cy, double& xr, double& yr);
ProbeGeoMapping build_probe_geo_mapping(float cut_lon_min_deg, float cut_lon_max_deg,
                                        float cut_lat_min_deg, float cut_lat_max_deg,
                                        const std::string& utm_crs, bool has_rotate_deg,
                                        double rotate_deg_override);
bool project_probe_local_xy(double lon_deg, double lat_deg, const ProbeGeoMapping& mapping,
                            double& x_local_si, double& y_local_si);
std::vector<std::string> split_probe_tokens(const std::string& raw);
bool parse_probe_offset(const std::string& raw_offset, ProbeOffset& offset, std::string& error);
bool parse_probe_request(const std::string& raw_token, ProbeRequest& request, std::string& error);
uint snap_probe_index(double coord_si, uint n_cells, float cell_m);
std::string make_probe_file_stem(const ProbeRequest& request, const ProbeGeoMapping& mapping,
                                 const std::string& prefix);

// Parses the deck probes and builds the geographic mapping (ctx.probes).
void plan_probes(cases::CaseContext& ctx);

} // namespace culbm::run
