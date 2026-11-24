#include "probes.hpp"
#include "../case/context.hpp"
#include "../case/deck.hpp"

#include <cstdlib>

namespace culbm::run {

using namespace culbm::cases;

int auto_utm_zone_from_lon(const double lon_deg) {
    int zone = (int)std::floor((lon_deg + 180.0) / 6.0) + 1;
    if (zone < 1) zone = 1;
    if (zone > 60) zone = 60;
    return zone;
}

bool parse_utm_zone_from_crs(const std::string& utm_crs, int& zone, bool& north) {
    std::string s = trim(utm_crs);
    if (s.empty()) return false;
    std::string digits;
    for (char ch : s) {
        if (ch >= '0' && ch <= '9') digits.push_back(ch);
    }
    if (digits.empty()) return false;
    const int code = atoi(digits.c_str());
    if (code >= 32601 && code <= 32660) {
        zone = code - 32600;
        north = true;
        return true;
    }
    if (code >= 32701 && code <= 32760) {
        zone = code - 32700;
        north = false;
        return true;
    }
    return false;
}

bool lonlat_to_utm_wgs84(const double lon_deg, const double lat_deg,
                                const int zone, const bool north_hemisphere,
                                double& x_utm, double& y_utm) {
    if (!(zone >= 1 && zone <= 60)) return false;
    if (!std::isfinite(lon_deg) || !std::isfinite(lat_deg)) return false;
    if (lat_deg <= -90.0 || lat_deg >= 90.0) return false;

    constexpr double kPi = 3.1415926535897932384626433832795;
    constexpr double a = 6378137.0;
    constexpr double f = 1.0 / 298.257223563;
    constexpr double k0 = 0.9996;
    constexpr double false_easting = 500000.0;
    constexpr double false_northing = 10000000.0;
    const double e2 = f * (2.0 - f);
    const double ep2 = e2 / (1.0 - e2);

    const double phi = lat_deg * (kPi / 180.0);
    const double lambda = lon_deg * (kPi / 180.0);
    const double lambda0 = ((double)zone * 6.0 - 183.0) * (kPi / 180.0);

    const double sin_phi = std::sin(phi);
    const double cos_phi = std::cos(phi);
    const double tan_phi = std::tan(phi);

    const double N = a / std::sqrt(1.0 - e2 * sin_phi * sin_phi);
    const double T = tan_phi * tan_phi;
    const double C = ep2 * cos_phi * cos_phi;
    const double A = cos_phi * (lambda - lambda0);

    const double M =
        a * ((1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2 * e2 * e2 / 256.0) * phi
           - (3.0 * e2 / 8.0 + 3.0 * e2 * e2 / 32.0 + 45.0 * e2 * e2 * e2 / 1024.0) * std::sin(2.0 * phi)
           + (15.0 * e2 * e2 / 256.0 + 45.0 * e2 * e2 * e2 / 1024.0) * std::sin(4.0 * phi)
           - (35.0 * e2 * e2 * e2 / 3072.0) * std::sin(6.0 * phi));

    x_utm = false_easting + k0 * N * (
        A
        + (1.0 - T + C) * A * A * A / 6.0
        + (5.0 - 18.0 * T + T * T + 72.0 * C - 58.0 * ep2) * A * A * A * A * A / 120.0);

    y_utm = k0 * (
        M
        + N * tan_phi * (
            A * A / 2.0
            + (5.0 - T + 9.0 * C + 4.0 * C * C) * A * A * A * A / 24.0
            + (61.0 - 58.0 * T + T * T + 600.0 * C - 330.0 * ep2) * A * A * A * A * A * A / 720.0));

    if (!north_hemisphere) y_utm += false_northing;
    return std::isfinite(x_utm) && std::isfinite(y_utm);
}

void rotate_xy(const double x, const double y, const double deg,
                      const double cx, const double cy,
                      double& xr, double& yr) {
    const double th = deg * (3.1415926535897932384626433832795 / 180.0);
    const double c = std::cos(th);
    const double s = std::sin(th);
    const double dx = x - cx;
    const double dy = y - cy;
    xr = c * dx - s * dy + cx;
    yr = s * dx + c * dy + cy;
}

ProbeGeoMapping build_probe_geo_mapping(const float cut_lon_min_deg,
                                               const float cut_lon_max_deg,
                                               const float cut_lat_min_deg,
                                               const float cut_lat_max_deg,
                                               const std::string& utm_crs,
                                               const bool has_rotate_deg,
                                               const double rotate_deg_override) {
    ProbeGeoMapping mapping;
    if (!std::isfinite(cut_lon_min_deg) || !std::isfinite(cut_lon_max_deg) ||
        !std::isfinite(cut_lat_min_deg) || !std::isfinite(cut_lat_max_deg)) {
        return mapping;
    }
    const double lon_lo = std::min((double)cut_lon_min_deg, (double)cut_lon_max_deg);
    const double lon_hi = std::max((double)cut_lon_min_deg, (double)cut_lon_max_deg);
    const double lat_lo = std::min((double)cut_lat_min_deg, (double)cut_lat_max_deg);
    const double lat_hi = std::max((double)cut_lat_min_deg, (double)cut_lat_max_deg);
    if (!(lon_hi > lon_lo) || !(lat_hi > lat_lo)) return mapping;

    int utm_zone = 0;
    bool utm_north = true;
    if (!parse_utm_zone_from_crs(utm_crs, utm_zone, utm_north)) {
        utm_zone = auto_utm_zone_from_lon(0.5 * (lon_lo + lon_hi));
        utm_north = (0.5 * (lat_lo + lat_hi)) >= 0.0;
    }

    double x00 = 0.0, y00 = 0.0;
    double x10 = 0.0, y10 = 0.0;
    double x11 = 0.0, y11 = 0.0;
    double x01 = 0.0, y01 = 0.0;
    if (!lonlat_to_utm_wgs84(lon_lo, lat_lo, utm_zone, utm_north, x00, y00) ||
        !lonlat_to_utm_wgs84(lon_hi, lat_lo, utm_zone, utm_north, x10, y10) ||
        !lonlat_to_utm_wgs84(lon_hi, lat_hi, utm_zone, utm_north, x11, y11) ||
        !lonlat_to_utm_wgs84(lon_lo, lat_hi, utm_zone, utm_north, x01, y01)) {
        return mapping;
    }

    const double cx = 0.25 * (x00 + x10 + x11 + x01);
    const double cy = 0.25 * (y00 + y10 + y11 + y01);
    const double rotate_deg = has_rotate_deg
        ? rotate_deg_override
        : (-std::atan2(y10 - y00, x10 - x00) * 180.0 / 3.1415926535897932384626433832795);

    double xr00 = 0.0, yr00 = 0.0;
    double xr10 = 0.0, yr10 = 0.0;
    double xr11 = 0.0, yr11 = 0.0;
    double xr01 = 0.0, yr01 = 0.0;
    rotate_xy(x00, y00, rotate_deg, cx, cy, xr00, yr00);
    rotate_xy(x10, y10, rotate_deg, cx, cy, xr10, yr10);
    rotate_xy(x11, y11, rotate_deg, cx, cy, xr11, yr11);
    rotate_xy(x01, y01, rotate_deg, cx, cy, xr01, yr01);

    const double th = rotate_deg * (3.1415926535897932384626433832795 / 180.0);
    mapping.valid = true;
    mapping.utm_zone = utm_zone;
    mapping.utm_north = utm_north;
    mapping.rotate_deg = rotate_deg;
    mapping.pivot_x = cx;
    mapping.pivot_y = cy;
    mapping.x_min_rot = std::min(std::min(xr00, xr10), std::min(xr11, xr01));
    mapping.y_min_rot = std::min(std::min(yr00, yr10), std::min(yr11, yr01));
    mapping.center_lon_deg = 0.5 * (lon_lo + lon_hi);
    mapping.center_lat_deg = 0.5 * (lat_lo + lat_hi);
    mapping.east_dx = std::cos(th);
    mapping.east_dy = std::sin(th);
    mapping.north_dx = -std::sin(th);
    mapping.north_dy = std::cos(th);
    return mapping;
}

bool project_probe_local_xy(const double lon_deg, const double lat_deg,
                                   const ProbeGeoMapping& mapping,
                                   double& x_local_si, double& y_local_si) {
    if (!mapping.valid) return false;
    double x_utm = 0.0, y_utm = 0.0;
    if (!lonlat_to_utm_wgs84(lon_deg, lat_deg, mapping.utm_zone, mapping.utm_north, x_utm, y_utm)) {
        return false;
    }
    double x_rot = 0.0, y_rot = 0.0;
    rotate_xy(x_utm, y_utm, mapping.rotate_deg, mapping.pivot_x, mapping.pivot_y, x_rot, y_rot);
    x_local_si = x_rot - mapping.x_min_rot;
    y_local_si = y_rot - mapping.y_min_rot;
    return std::isfinite(x_local_si) && std::isfinite(y_local_si);
}

std::vector<std::string> split_probe_tokens(const std::string& raw) {
    std::string s = trim(raw);
    const size_t lb = s.find('[');
    const size_t rb = s.rfind(']');
    if (lb != std::string::npos && rb != std::string::npos && rb > lb) {
        s = s.substr(lb + 1u, rb - lb - 1u);
    }

    std::vector<std::string> out;
    std::string token;
    char quote = '\0';
    for (char ch : s) {
        if (quote != '\0') {
            token.push_back(ch);
            if (ch == quote) quote = '\0';
            continue;
        }
        if (ch == '"' || ch == '\'') {
            quote = ch;
            token.push_back(ch);
            continue;
        }
        if (ch == ',') {
            const std::string t = trim(token);
            if (!t.empty()) out.push_back(t);
            token.clear();
            continue;
        }
        token.push_back(ch);
    }
    const std::string tail = trim(token);
    if (!tail.empty()) out.push_back(tail);
    return out;
}

bool parse_probe_offset(const std::string& raw_offset, ProbeOffset& offset, std::string& error) {
    offset = ProbeOffset{};
    std::string s = strip_all_whitespace(raw_offset);
    if (s.empty()) return true;
    s = to_upper_ascii(s);
    const bool has_digit = std::any_of(s.begin(), s.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
    offset.label = s;
    if (!has_digit) {
        offset.mode = ProbeOffsetMode::GRID_CELLS;
        for (char ch : s) {
            if (ch == 'N') offset.north_cells += 1;
            else if (ch == 'S') offset.north_cells -= 1;
            else if (ch == 'E') offset.east_cells += 1;
            else if (ch == 'W') offset.east_cells -= 1;
            else {
                error = "grid offset can only contain N/S/E/W";
                return false;
            }
        }
        return true;
    }

    offset.mode = ProbeOffsetMode::METERS;
    size_t i = 0u;
    while (i < s.size()) {
        const char dir = s[i];
        if (!(dir == 'N' || dir == 'S' || dir == 'E' || dir == 'W')) {
            error = "meter offset must use N/S/E/W followed by a number";
            return false;
        }
        i++;
        char* endptr = nullptr;
        const double value = std::strtod(s.c_str() + i, &endptr);
        if (endptr == s.c_str() + i || !std::isfinite(value)) {
            error = "meter offset is missing a numeric value after direction";
            return false;
        }
        if (value < 0.0) {
            error = "meter offset cannot be negative";
            return false;
        }
        if (dir == 'N') offset.north_m += value;
        else if (dir == 'S') offset.north_m -= value;
        else if (dir == 'E') offset.east_m += value;
        else if (dir == 'W') offset.east_m -= value;
        i = (size_t)(endptr - s.c_str());
    }
    return true;
}

bool parse_probe_request(const std::string& raw_token, ProbeRequest& request, std::string& error) {
    request = ProbeRequest{};
    request.raw_token = trim(raw_token);
    std::string token = request.raw_token;
    if (token.empty()) {
        error = "empty probe token";
        return false;
    }

    auto parse_center_token = [&](const std::string& word, const std::string& raw_offset) -> bool {
        const std::string key = to_lower_ascii(trim(word));
        if (key != "center" && key != "centre") return false;
        request.uses_center = true;
        return parse_probe_offset(raw_offset, request.offset, error);
    };

    if (token.front() == '"' || token.front() == '\'') {
        const char quote = token.front();
        const size_t close = token.find(quote, 1u);
        if (close == std::string::npos) {
            error = "quoted probe token is missing the closing quote";
            return false;
        }
        const std::string inner = token.substr(1u, close - 1u);
        const std::string rest = trim(token.substr(close + 1u));
        if (!parse_center_token(inner, rest)) {
            if (error.empty()) error = "quoted probe token only supports center/centre";
            return false;
        }
        return true;
    }

    {
        const std::string token_lc = to_lower_ascii(token);
        const std::string k_center = "center";
        const std::string k_centre = "centre";
        if (begins_with(token_lc, k_center)) {
            const std::string rest = trim(token.substr(k_center.size()));
            return parse_center_token(k_center, rest);
        }
        if (begins_with(token_lc, k_centre)) {
            const std::string rest = trim(token.substr(k_centre.size()));
            return parse_center_token(k_centre, rest);
        }
    }

    const size_t colon = token.find(':');
    if (colon == std::string::npos) {
        error = "probe must be lon:lat, center, or centre";
        return false;
    }

    const std::string lon_text = trim(token.substr(0u, colon));
    const std::string lat_and_offset = trim(token.substr(colon + 1u));
    if (lon_text.empty() || lat_and_offset.empty()) {
        error = "probe lon:lat is incomplete";
        return false;
    }

    char* lon_end = nullptr;
    const double lon_deg = std::strtod(lon_text.c_str(), &lon_end);
    if (lon_end == lon_text.c_str() || *lon_end != '\0' || !std::isfinite(lon_deg)) {
        error = "invalid probe longitude";
        return false;
    }

    char* lat_end = nullptr;
    const double lat_deg = std::strtod(lat_and_offset.c_str(), &lat_end);
    if (lat_end == lat_and_offset.c_str() || !std::isfinite(lat_deg)) {
        error = "invalid probe latitude";
        return false;
    }

    request.lon_deg = lon_deg;
    request.lat_deg = lat_deg;
    return parse_probe_offset(trim(lat_and_offset.substr((size_t)(lat_end - lat_and_offset.c_str()))), request.offset, error);
}

uint snap_probe_index(const double coord_si, const uint n_cells, const float cell_m) {
    if (n_cells == 0u || !(cell_m > 0.0f)) return 0u;
    const long idx = (long)std::llround(coord_si / (double)cell_m);
    if (idx <= 0l) return 0u;
    if ((ulong)idx >= (ulong)n_cells) return n_cells - 1u;
    return (uint)idx;
}

std::string make_probe_file_stem(const ProbeRequest& request,
                                        const ProbeGeoMapping& mapping,
                                        const std::string& prefix) {
    const double lon_deg = request.uses_center ? mapping.center_lon_deg : request.lon_deg;
    const double lat_deg = request.uses_center ? mapping.center_lat_deg : request.lat_deg;
    std::string stem = format_decimal_trimmed(lon_deg) + "_" + format_decimal_trimmed(lat_deg);
    if (!request.offset.label.empty()) stem += "_" + sanitize_filename_component(request.offset.label);
    if (!prefix.empty()) stem = sanitize_filename_component(prefix) + stem;
    return sanitize_filename_component(stem);
}

void plan_probes(CaseContext& ctx) {
    CaseSettings& deck = ctx.deck;
    ProbePlan& probes = ctx.probes;
    if (deck.probes_defined) {
        const std::vector<std::string> probe_tokens = split_probe_tokens(deck.probes_raw_text);
        for (const std::string& token : probe_tokens) {
            ProbeRequest request;
            std::string error;
            if (!parse_probe_request(token, request, error)) {
                println("| WARNING: ignore probe token '" + token + "': " + error + "                        |");
                continue;
            }
            probes.probe_requests.push_back(request);
        }
        if (probes.probe_requests.empty()) {
            println("| WARNING: probes is defined but no valid probe token was parsed.                |");
        }
        else if (!(deck.has_cut_lon_manual && deck.has_cut_lat_manual)) {
            println("| WARNING: probes requires cut_lon_manual/cut_lat_manual for lon-lat mapping.    |");
        }
        else {
            probes.probe_geo_mapping = build_probe_geo_mapping(
                deck.cut_lon_min_deg, deck.cut_lon_max_deg,
                deck.cut_lat_min_deg, deck.cut_lat_max_deg,
                deck.utm_crs_from_config,
                deck.has_rotate_deg,
                deck.rotate_deg_from_config);
            if (!probes.probe_geo_mapping.valid) {
                println("| WARNING: failed to build probes geographic mapping. Probes are disabled.       |");
            }
        }
    }
}

} // namespace culbm::run
