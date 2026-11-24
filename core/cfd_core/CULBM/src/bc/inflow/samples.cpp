#include "samples.hpp"
#include "../../case/deck.hpp"

#include <cfloat>
#include <cstdlib>
#include <sstream>

namespace culbm::bc {

using cases::trim_copy;

std::vector<ProfileSample> read_profile_dat(const string& path) {
    std::vector<ProfileSample> out;
    std::ifstream fin(path);
    if (!fin.is_open()) {
        println("ERROR: could not open profile file " + path);
        return out;
    }
    std::string line;
    ulong line_no = 0ul;
    while (std::getline(fin, line)) {
        line_no++;
        size_t cmt = line.find("//"); if (cmt != std::string::npos) line.erase(cmt);
        cmt = line.find('#'); if (cmt != std::string::npos) line.erase(cmt);
        line = trim_copy(line);
        if (line.empty()) continue;
        for (char& ch : line) { if (ch == ',' || ch == ';') ch = ' '; }
        std::stringstream ss(line);
        float z = 0.0f, u = 0.0f;
        if (!(ss >> z >> u)) {
            continue; // likely header or malformed line
        }
        if (!std::isfinite(z) || !std::isfinite(u)) {
            println("WARNING: invalid z/U in profile at line " + to_string(line_no));
            continue;
        }
        out.push_back({ z, u });
    }
    return out;
}

float interpolate_profile_cubic(const std::vector<float>& z,
                                const std::vector<float>& u,
                                const float zq) {
    const size_t n = z.size();
    if (n == 0u) return 0.0f;
    if (n == 1u) return u[0];
    if (zq <= z.front()) return u.front();
    if (zq >= z.back()) return u.back();

    auto it = std::upper_bound(z.begin(), z.end(), zq);
    size_t i1 = (it == z.begin()) ? 0u : static_cast<size_t>(it - z.begin() - 1u);
    size_t i2 = std::min(i1 + 1u, n - 1u);
    const float z0 = z[i1];
    const float z1 = z[i2];
    const float denom = z1 - z0;
    if (denom <= 0.0f) return u[i1];
    const float t = (zq - z0) / denom;

    auto slope_at = [&](size_t i) -> float {
        if (n < 2u) return 0.0f;
        if (i == 0u) {
            const float dz = z[1] - z[0];
            return (dz != 0.0f) ? (u[1] - u[0]) / dz : 0.0f;
        }
        if (i + 1u >= n) {
            const float dz = z[n - 1u] - z[n - 2u];
            return (dz != 0.0f) ? (u[n - 1u] - u[n - 2u]) / dz : 0.0f;
        }
        const float dz = z[i + 1u] - z[i - 1u];
        return (dz != 0.0f) ? (u[i + 1u] - u[i - 1u]) / dz : 0.0f;
    };

    const float m0 = slope_at(i1);
    const float m1 = slope_at(i2);
    const float va = m0 * denom;
    const float vb = m1 * denom;
    return hermite_spline(u[i1], u[i2], va, vb, t);
}

std::vector<SamplePoint> read_samples(const string& csv_path,
                                      bool* has_temperature_column,
                                      ulong* rows_with_temperature,
                                      float* temperature_min_si,
                                      float* temperature_max_si,
                                      bool* has_patch_column,
                                      ulong* rows_with_patch,
                                      bool* samples_valid) {
    if (has_temperature_column) *has_temperature_column = false;
    if (rows_with_temperature) *rows_with_temperature = 0ul;
    if (temperature_min_si) *temperature_min_si = k_temperature_ref_kelvin;
    if (temperature_max_si) *temperature_max_si = k_temperature_ref_kelvin;
    if (has_patch_column) *has_patch_column = false;
    if (rows_with_patch) *rows_with_patch = 0ul;
    if (samples_valid) *samples_valid = true;

    std::vector<SamplePoint> out;
    std::ifstream fin(csv_path);
    if (!fin.is_open()) {
        println("ERROR: could not open CSV " + csv_path);
        return out;
    }
    auto split_csv = [](const string& s) -> std::vector<string> {
        std::vector<string> out_cols;
        std::stringstream ss(s);
        string tok;
        while (std::getline(ss, tok, ',')) out_cols.push_back(trim_copy(tok));
        return out_cols;
    };

    auto to_lower_copy = [](string s) -> string {
        for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
        return s;
    };

    auto find_col = [&](const std::vector<string>& cols, const string& key) -> int {
        const string key_lc = to_lower_copy(key);
        for (size_t i = 0u; i < cols.size(); ++i) {
            if (to_lower_copy(cols[i]) == key_lc) return (int)i;
        }
        return -1;
    };

    auto parse_positive_finite_kelvin = [](const string& token, float& value) -> bool {
        char* endptr = nullptr;
        const char* const begin = token.c_str();
        const float parsed = std::strtof(begin, &endptr);
        if (endptr == begin || *endptr != '\0' || !std::isfinite(parsed) || parsed <= 0.0f) {
            return false;
        }
        value = parsed;
        return true;
    };

    auto report_invalid_temperature = [&](const ulong line_no,
                                          const string& line,
                                          const string& token) {
        println("ERROR: invalid temperature T in " + csv_path + " at line " +
                to_string(line_no) + ": expected a complete, finite Kelvin value > 0; got '" +
                (token.empty() ? string("<missing>") : token) + "'.");
        println("ERROR: CSV line " + to_string(line_no) + " content: " + line);
        if (samples_valid) *samples_valid = false;
    };

    string header_line;
    if (!std::getline(fin, header_line)) {
        println("WARNING: empty CSV " + csv_path);
        return out;
    }

    const std::vector<string> header_cols = split_csv(header_line);
    const int idx_x = find_col(header_cols, "x");
    const int idx_y = find_col(header_cols, "y");
    const int idx_z = find_col(header_cols, "z");
    const int idx_u = find_col(header_cols, "u");
    const int idx_v = find_col(header_cols, "v");
    const int idx_w = find_col(header_cols, "w");
    const int idx_t = find_col(header_cols, "t");
    const int idx_patch = find_col(header_cols, "patch");
    const bool use_named_columns =
        (idx_x >= 0) && (idx_y >= 0) && (idx_z >= 0) &&
        (idx_u >= 0) && (idx_v >= 0) && (idx_w >= 0);
    const bool named_temperature_column = (idx_t >= 0);
    bool has_patch = (idx_patch >= 0);

    string line;
    ulong line_no = 1ul;
    bool has_temperature = named_temperature_column;
    ulong temp_rows = 0ul;
    ulong patch_rows = 0ul;
    float tmin = +FLT_MAX;
    float tmax = -FLT_MAX;
    ulong first_positional_row_without_temperature = 0ul;
    string first_positional_line_without_temperature;
    while (std::getline(fin, line)) {
        line_no++;
        const std::vector<string> cols = split_csv(line);
        if (cols.empty()) continue;

        SamplePoint sp;

        if (use_named_columns) {
            const int required_max_idx = std::max(
                std::max(std::max(idx_x, idx_y), std::max(idx_z, idx_u)),
                std::max(idx_v, idx_w)
            );
            if ((int)cols.size() <= required_max_idx) {
                println("WARNING: malformed line " + to_string(line_no) + " in CSV (missing required columns)");
                continue;
            }

            sp.p = float3(
                (float)atof(cols[idx_x].c_str()),
                (float)atof(cols[idx_y].c_str()),
                (float)atof(cols[idx_z].c_str())
            );
            sp.u = float3(
                (float)atof(cols[idx_u].c_str()),
                (float)atof(cols[idx_v].c_str()),
                (float)atof(cols[idx_w].c_str())
            );

            if (named_temperature_column) {
                const string temperature_token = ((int)cols.size() > idx_t)
                    ? cols[(size_t)idx_t]
                    : string();
                if (!parse_positive_finite_kelvin(temperature_token, sp.T)) {
                    report_invalid_temperature(line_no, line, temperature_token);
                    return {};
                }
                temp_rows++;
                tmin = fmin(tmin, sp.T);
                tmax = fmax(tmax, sp.T);
            }
            if ((idx_patch >= 0) && ((int)cols.size() > idx_patch)) {
                sp.patch = (int)std::lround((double)atof(cols[idx_patch].c_str()));
                has_patch = true;
                patch_rows++;
            }

            out.push_back(sp);
            continue;
        }

        // Fallback for legacy CSV with positional columns.
        std::stringstream ss(line);
        string token;
        float vals[8] = { 0.0f };
        int n_cols = 0;
        while (std::getline(ss, token, ',')) {
            if (n_cols < 8) vals[n_cols] = (float)atof(trim_copy(token).c_str());
            n_cols++;
        }
        if (n_cols < 6 || n_cols > 8) {
            println("WARNING: malformed line " + to_string(line_no) + " in CSV (expect 6~8 columns)");
            continue;
        }

        sp.p = float3(vals[0], vals[1], vals[2]);
        sp.u = float3(vals[3], vals[4], vals[5]);

        bool row_has_temperature = false;
        if (named_temperature_column) {
            const string temperature_token = ((int)cols.size() > idx_t)
                ? cols[(size_t)idx_t]
                : string();
            if (!parse_positive_finite_kelvin(temperature_token, sp.T)) {
                report_invalid_temperature(line_no, line, temperature_token);
                return {};
            }
            row_has_temperature = true;
            if ((idx_patch >= 0) && ((int)cols.size() > idx_patch)) {
                sp.patch = (int)std::lround((double)atof(cols[(size_t)idx_patch].c_str()));
                has_patch = true;
                patch_rows++;
            }
        }
        else if (n_cols >= 8) {
            // X,Y,Z,u,v,w,T,patch
            has_temperature = true;
            const string temperature_token = cols.size() > 6u ? cols[6u] : string();
            if (!parse_positive_finite_kelvin(temperature_token, sp.T)) {
                report_invalid_temperature(line_no, line, temperature_token);
                return {};
            }
            row_has_temperature = true;
            sp.patch = (int)std::lround((double)vals[7]);
            has_patch = true;
            patch_rows++;
        } else if (n_cols == 7) {
            // Ambiguous: X,Y,Z,u,v,w,T or X,Y,Z,u,v,w,patch
            const float c7 = vals[6];
            const bool looks_like_patch = (c7 >= -0.5f) && (c7 <= 5.5f) && (fabsf(c7 - roundf(c7)) <= 1e-4f);
            if (!looks_like_patch) {
                const string temperature_token = cols.size() > 6u ? cols[6u] : string();
                if (!parse_positive_finite_kelvin(temperature_token, sp.T)) {
                    report_invalid_temperature(line_no, line, temperature_token);
                    return {};
                }
                row_has_temperature = true;
            }
            else {
                sp.patch = (int)std::lround((double)c7);
                has_patch = true;
                patch_rows++;
            }
        }

        if (row_has_temperature) {
            has_temperature = true;
            temp_rows++;
            tmin = fmin(tmin, sp.T);
            tmax = fmax(tmax, sp.T);
        }
        else if (first_positional_row_without_temperature == 0ul) {
            first_positional_row_without_temperature = line_no;
            first_positional_line_without_temperature = line;
        }

        out.push_back(sp);
    }
    if (has_temperature && first_positional_row_without_temperature != 0ul) {
        report_invalid_temperature(first_positional_row_without_temperature,
                                   first_positional_line_without_temperature,
                                   string());
        println("ERROR: positional CSV rows cannot mix temperature-bearing samples with samples that omit T.");
        return {};
    }
    if (has_temperature_column) *has_temperature_column = has_temperature;
    if (rows_with_temperature) *rows_with_temperature = temp_rows;
    if (temperature_min_si) *temperature_min_si = has_temperature ? tmin : k_temperature_ref_kelvin;
    if (temperature_max_si) *temperature_max_si = has_temperature ? tmax : k_temperature_ref_kelvin;
    if (has_patch_column) *has_patch_column = has_patch;
    if (rows_with_patch) *rows_with_patch = patch_rows;
    return out;
}

} // namespace culbm::bc
