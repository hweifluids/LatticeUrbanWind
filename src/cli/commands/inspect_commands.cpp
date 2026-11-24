#include "luw/inspect_commands.hpp"

#include <vtk_netcdf.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef LUW_HAS_GDAL
#define LUW_HAS_GDAL 0
#endif

#if LUW_HAS_GDAL
#include <cpl_conv.h>
#include <cpl_error.h>
#include <gdal_priv.h>
#include <gdal_version.h>
#include <ogrsf_frmts.h>
#endif

namespace luw::cli {
namespace {

namespace fs = std::filesystem;

std::string utf8_text(const fs::path& path) {
#if defined(__cpp_lib_char8_t)
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

std::string lower_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

fs::path absolute_normalized(const std::string& input) {
    std::error_code ec;
#if defined(__cpp_lib_char8_t)
    const auto* first = reinterpret_cast<const char8_t*>(input.data());
    const fs::path decoded(std::u8string(first, first + input.size()));
#else
    const fs::path decoded = fs::u8path(input);
#endif
    fs::path result = fs::absolute(decoded, ec);
    if (ec) {
        throw std::runtime_error("Could not resolve input path: " + input +
                                 " (" + ec.message() + ")");
    }
    return result.lexically_normal();
}

struct ResolveSpec {
    std::string extension;
    std::string derived_folder;
    std::string direct_missing_label;
    std::string no_candidate_label;
    std::string multiple_warning;
    std::string list_heading;
    std::string using_label;
};

fs::path resolve_input_path(const std::string& input,
                            const ResolveSpec& spec,
                            std::ostream& out,
                            std::ostream& err) {
    const fs::path input_path = absolute_normalized(input);
    const bool has_target_extension =
        lower_ascii(utf8_text(input_path.extension())) == spec.extension;

    std::error_code ec;
    if (has_target_extension) {
        if (fs::is_regular_file(input_path, ec) && !ec) {
            return input_path;
        }
        throw std::runtime_error(spec.direct_missing_label + utf8_text(input_path));
    }

    ec.clear();
    if (!fs::is_regular_file(input_path, ec) || ec) {
        throw std::runtime_error("Input file does not exist: " + utf8_text(input_path));
    }

    const fs::path derived = input_path.parent_path() / fs::path(spec.derived_folder);
    ec.clear();
    if (!fs::is_directory(derived, ec) || ec) {
        throw std::runtime_error("Derived folder '" + spec.derived_folder +
                                 "' does not exist: " + utf8_text(derived));
    }

    std::vector<fs::path> candidates;
    fs::directory_iterator iterator(derived, ec);
    const fs::directory_iterator end;
    while (!ec && iterator != end) {
        const auto& entry = *iterator;
        if (lower_ascii(utf8_text(entry.path().extension())) == spec.extension) {
            candidates.push_back(entry.path());
        }
        iterator.increment(ec);
    }
    if (ec) {
        throw std::runtime_error("Could not inspect folder: " + utf8_text(derived) +
                                 " (" + ec.message() + ")");
    }

    std::sort(candidates.begin(), candidates.end(), [](const fs::path& lhs,
                                                       const fs::path& rhs) {
        return utf8_text(lhs.filename()) < utf8_text(rhs.filename());
    });

    if (candidates.empty()) {
        throw std::runtime_error(spec.no_candidate_label + utf8_text(derived));
    }
    if (candidates.size() > 1U) {
        err << "Warning: " << spec.multiple_warning << '\n';
        out << spec.list_heading << '\n';
        for (const auto& candidate : candidates) {
            out << "  " << utf8_text(candidate.filename()) << '\n';
        }
    }

    out << spec.using_label << utf8_text(candidates.front()) << '\n';
    return candidates.front();
}

const ResolveSpec kNetCdfResolve{
    ".nc",
    "wind_bc",
    "Input .nc does not exist: ",
    "No .nc files found in folder: ",
    "Multiple .nc files found under 'wind_bc'. The first NetCDF in alphabetical "
    "order will be used.",
    "All detected NetCDF files:",
    "Using NetCDF: ",
};

const ResolveSpec kShapefileResolve{
    ".shp",
    "building_db",
    "Input .shp does not exist: ",
    "No .shp files found in folder: ",
    "Multiple .shp files found under 'building_db'. The first shapefile in "
    "alphabetical order will be used.",
    "All detected shapefiles:",
    "Using shapefile: ",
};

void check_nc(int status, std::string_view operation) {
    if (status != NC_NOERR) {
        throw std::runtime_error(std::string(operation) + ": " + nc_strerror(status));
    }
}

class NcFile {
public:
    explicit NcFile(const fs::path& path) {
        const int status = nc_open(utf8_text(path).c_str(), NC_NOWRITE, &id_);
        if (status != NC_NOERR) {
            throw std::runtime_error("Failed to open NetCDF: " +
                                     std::string(nc_strerror(status)));
        }
    }

    NcFile(const NcFile&) = delete;
    NcFile& operator=(const NcFile&) = delete;

    ~NcFile() {
        if (id_ >= 0) {
            (void)nc_close(id_);
        }
    }

    [[nodiscard]] int id() const noexcept { return id_; }

private:
    int id_ = -1;
};

std::optional<std::string> nc_string_attribute(int ncid,
                                                int varid,
                                                const std::string& name) {
    nc_type type = NC_NAT;
    std::size_t length = 0;
    if (nc_inq_att(ncid, varid, name.c_str(), &type, &length) != NC_NOERR) {
        return std::nullopt;
    }
    if (type == NC_CHAR) {
        std::string result(length, '\0');
        if (length > 0U &&
            nc_get_att_text(ncid, varid, name.c_str(), result.data()) != NC_NOERR) {
            return std::nullopt;
        }
        return result;
    }
    if (type == NC_STRING && length == 1U) {
        char* value = nullptr;
        if (nc_get_att_string(ncid, varid, name.c_str(), &value) != NC_NOERR) {
            return std::nullopt;
        }
        std::string result = value != nullptr ? value : "";
        (void)nc_free_string(1U, &value);
        return result;
    }
    return std::nullopt;
}

std::vector<long double> nc_numeric_attribute(int ncid,
                                               int varid,
                                               const std::string& name) {
    nc_type type = NC_NAT;
    std::size_t length = 0;
    if (nc_inq_att(ncid, varid, name.c_str(), &type, &length) != NC_NOERR ||
        type == NC_CHAR || type == NC_STRING || length == 0U) {
        return {};
    }
    std::vector<double> values(length);
    if (nc_get_att_double(ncid, varid, name.c_str(), values.data()) != NC_NOERR) {
        return {};
    }
    std::vector<long double> result;
    result.reserve(values.size());
    for (const double value : values) {
        result.push_back(static_cast<long double>(value));
    }
    return result;
}

bool is_integer_nc_type(nc_type type) {
    switch (type) {
    case NC_BYTE:
    case NC_UBYTE:
    case NC_SHORT:
    case NC_USHORT:
    case NC_INT:
    case NC_UINT:
    case NC_INT64:
    case NC_UINT64:
        return true;
    default:
        return false;
    }
}

bool is_numeric_nc_type(nc_type type) {
    return is_integer_nc_type(type) || type == NC_FLOAT || type == NC_DOUBLE;
}

std::string nc_dtype_name(nc_type type) {
    switch (type) {
    case NC_BYTE:
        return "int8";
    case NC_UBYTE:
        return "uint8";
    case NC_CHAR:
        return "|S1";
    case NC_SHORT:
        return "int16";
    case NC_USHORT:
        return "uint16";
    case NC_INT:
        return "int32";
    case NC_UINT:
        return "uint32";
    case NC_INT64:
        return "int64";
    case NC_UINT64:
        return "uint64";
    case NC_FLOAT:
        return "float32";
    case NC_DOUBLE:
        return "float64";
    case NC_STRING:
        return "<U";
    default:
        return "object";
    }
}

std::int64_t days_from_civil(int year, unsigned month, unsigned day) {
    year -= month <= 2U ? 1 : 0;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned year_of_era = static_cast<unsigned>(year - era * 400);
    const unsigned day_of_year =
        (153U * (month > 2U ? month - 3U : month + 9U) + 2U) / 5U + day - 1U;
    const unsigned day_of_era =
        year_of_era * 365U + year_of_era / 4U - year_of_era / 100U + day_of_year;
    return static_cast<std::int64_t>(era) * 146097 +
           static_cast<std::int64_t>(day_of_era) - 719468;
}

bool valid_civil_date(int year, unsigned month, unsigned day) {
    if (month < 1U || month > 12U || day < 1U) {
        return false;
    }
    constexpr std::array<unsigned, 12> days_per_month{
        31U, 28U, 31U, 30U, 31U, 30U, 31U, 31U, 30U, 31U, 30U, 31U};
    unsigned maximum = days_per_month[month - 1U];
    const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    if (month == 2U && leap) {
        maximum = 29U;
    }
    return day <= maximum;
}

struct CfTimeConversion {
    long double origin_nanoseconds = 0.0L;
    long double unit_nanoseconds = 0.0L;
};

std::optional<CfTimeConversion> parse_cf_time_units(const std::string& units,
                                                    const std::string& calendar) {
    const std::string normalized_calendar = lower_ascii(calendar);
    if (!normalized_calendar.empty() && normalized_calendar != "standard" &&
        normalized_calendar != "gregorian" &&
        normalized_calendar != "proleptic_gregorian") {
        return std::nullopt;
    }

    static const std::regex pattern(
        R"(^\s*(nanoseconds?|microseconds?|milliseconds?|seconds?|minutes?|hours?|days?)\s+since\s+([+-]?\d{1,6})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2})(?::(\d{1,2})(?::(\d{1,2}(?:\.\d*)?))?)?)?(?:\s*(Z|[+-]\d{1,2}:?\d{2}))?\s*$)",
        std::regex::icase);
    std::smatch match;
    if (!std::regex_match(units, match, pattern)) {
        return std::nullopt;
    }

    try {
        const std::string unit = lower_ascii(match[1].str());
        const int year = std::stoi(match[2].str());
        const unsigned month = static_cast<unsigned>(std::stoul(match[3].str()));
        const unsigned day = static_cast<unsigned>(std::stoul(match[4].str()));
        const unsigned hour = match[5].matched
                                  ? static_cast<unsigned>(std::stoul(match[5].str()))
                                  : 0U;
        const unsigned minute = match[6].matched
                                    ? static_cast<unsigned>(std::stoul(match[6].str()))
                                    : 0U;
        const long double second =
            match[7].matched ? std::stold(match[7].str()) : 0.0L;
        if (!valid_civil_date(year, month, day) || hour > 23U || minute > 59U ||
            second < 0.0L || second >= 61.0L) {
            return std::nullopt;
        }

        long double multiplier = 0.0L;
        if (unit.rfind("nanosecond", 0U) == 0U) {
            multiplier = 1.0L;
        } else if (unit.rfind("microsecond", 0U) == 0U) {
            multiplier = 1.0e3L;
        } else if (unit.rfind("millisecond", 0U) == 0U) {
            multiplier = 1.0e6L;
        } else if (unit.rfind("second", 0U) == 0U) {
            multiplier = 1.0e9L;
        } else if (unit.rfind("minute", 0U) == 0U) {
            multiplier = 60.0e9L;
        } else if (unit.rfind("hour", 0U) == 0U) {
            multiplier = 3600.0e9L;
        } else if (unit.rfind("day", 0U) == 0U) {
            multiplier = 86400.0e9L;
        }

        int timezone_seconds = 0;
        if (match[8].matched && match[8].str() != "Z" && match[8].str() != "z") {
            const std::string zone = match[8].str();
            const int sign = zone.front() == '-' ? -1 : 1;
            std::string digits = zone.substr(1U);
            digits.erase(std::remove(digits.begin(), digits.end(), ':'), digits.end());
            if (digits.size() != 4U) {
                return std::nullopt;
            }
            timezone_seconds =
                sign * (std::stoi(digits.substr(0U, 2U)) * 3600 +
                        std::stoi(digits.substr(2U, 2U)) * 60);
        }

        const long double epoch_seconds =
            static_cast<long double>(days_from_civil(year, month, day)) * 86400.0L +
            static_cast<long double>(hour * 3600U + minute * 60U) + second -
            static_cast<long double>(timezone_seconds);
        return CfTimeConversion{epoch_seconds * 1.0e9L, multiplier};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

struct NcDimensionInfo {
    int id = -1;
    std::string name;
    std::size_t size = 0;
};

struct NcVariableInfo {
    int id = -1;
    std::string name;
    nc_type type = NC_NAT;
    std::vector<int> dimension_ids;
    std::vector<std::string> dimensions;
    std::vector<std::size_t> shape;
    std::vector<std::string> attributes;
    bool coordinate = false;
    bool datetime = false;
    bool integer_values = false;
};

struct NcDatasetInfo {
    std::vector<NcDimensionInfo> dimensions;
    std::vector<NcVariableInfo> variables;
    std::vector<std::string> global_attributes;
};

std::vector<std::string> nc_attribute_names(int ncid, int varid, int count) {
    std::vector<std::string> result;
    result.reserve(static_cast<std::size_t>(std::max(count, 0)));
    for (int index = 0; index < count; ++index) {
        std::array<char, NC_MAX_NAME + 1> name{};
        check_nc(nc_inq_attname(ncid, varid, index, name.data()),
                 "Failed to read NetCDF attribute name");
        result.emplace_back(name.data());
    }
    return result;
}

std::vector<std::string> split_words(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

NcDatasetInfo read_nc_structure(int ncid) {
    int dimension_count = 0;
    int variable_count = 0;
    int global_attribute_count = 0;
    int unlimited_dimension = -1;
    check_nc(nc_inq(ncid, &dimension_count, &variable_count, &global_attribute_count,
                    &unlimited_dimension),
             "Failed to inspect NetCDF dataset");

    NcDatasetInfo dataset;
    dataset.dimensions.reserve(static_cast<std::size_t>(dimension_count));
    for (int id = 0; id < dimension_count; ++id) {
        std::array<char, NC_MAX_NAME + 1> name{};
        std::size_t size = 0;
        check_nc(nc_inq_dim(ncid, id, name.data(), &size),
                 "Failed to inspect NetCDF dimension");
        dataset.dimensions.push_back({id, name.data(), size});
    }

    dataset.variables.reserve(static_cast<std::size_t>(variable_count));
    for (int id = 0; id < variable_count; ++id) {
        std::array<char, NC_MAX_NAME + 1> name{};
        nc_type type = NC_NAT;
        int dimension_total = 0;
        int attribute_total = 0;
        check_nc(nc_inq_var(ncid, id, name.data(), &type, &dimension_total, nullptr,
                            &attribute_total),
                 "Failed to inspect NetCDF variable");

        std::vector<int> dimension_ids(static_cast<std::size_t>(dimension_total));
        if (dimension_total > 0) {
            check_nc(nc_inq_vardimid(ncid, id, dimension_ids.data()),
                     "Failed to inspect NetCDF variable dimensions");
        }

        NcVariableInfo variable;
        variable.id = id;
        variable.name = name.data();
        variable.type = type;
        variable.dimension_ids = dimension_ids;
        variable.integer_values = is_integer_nc_type(type);
        for (const int dimension_id : dimension_ids) {
            const auto found = std::find_if(
                dataset.dimensions.begin(), dataset.dimensions.end(),
                [dimension_id](const NcDimensionInfo& dimension) {
                    return dimension.id == dimension_id;
                });
            if (found == dataset.dimensions.end()) {
                throw std::runtime_error("NetCDF variable references an unknown dimension");
            }
            variable.dimensions.push_back(found->name);
            variable.shape.push_back(found->size);
        }
        variable.attributes = nc_attribute_names(ncid, id, attribute_total);

        const auto scale = nc_numeric_attribute(ncid, id, "scale_factor");
        const auto offset = nc_numeric_attribute(ncid, id, "add_offset");
        const bool has_mask =
            std::find(variable.attributes.begin(), variable.attributes.end(),
                      "_FillValue") != variable.attributes.end() ||
            std::find(variable.attributes.begin(), variable.attributes.end(),
                      "missing_value") != variable.attributes.end();
        if (!scale.empty() || !offset.empty() ||
            (has_mask && is_integer_nc_type(type))) {
            variable.integer_values = false;
        }

        const auto units = nc_string_attribute(ncid, id, "units");
        const auto calendar = nc_string_attribute(ncid, id, "calendar");
        if (units && parse_cf_time_units(*units, calendar.value_or("standard"))) {
            variable.datetime = true;
            variable.integer_values = true;
        }
        dataset.variables.push_back(std::move(variable));
    }

    // xarray exposes dimensions in first-use order (normally the dimensions of
    // the first data variable), rather than the order of the raw dimension
    // declarations in the NetCDF header.
    std::vector<NcDimensionInfo> ordered_dimensions;
    ordered_dimensions.reserve(dataset.dimensions.size());
    std::set<int> ordered_dimension_ids;
    for (const auto& variable : dataset.variables) {
        for (const int dimension_id : variable.dimension_ids) {
            if (!ordered_dimension_ids.insert(dimension_id).second) {
                continue;
            }
            const auto found = std::find_if(
                dataset.dimensions.begin(), dataset.dimensions.end(),
                [dimension_id](const NcDimensionInfo& dimension) {
                    return dimension.id == dimension_id;
                });
            if (found != dataset.dimensions.end()) {
                ordered_dimensions.push_back(*found);
            }
        }
    }
    for (const auto& dimension : dataset.dimensions) {
        if (ordered_dimension_ids.insert(dimension.id).second) {
            ordered_dimensions.push_back(dimension);
        }
    }
    dataset.dimensions = std::move(ordered_dimensions);

    std::set<std::string> coordinate_names;
    for (const auto& dimension : dataset.dimensions) {
        const auto variable = std::find_if(
            dataset.variables.begin(), dataset.variables.end(),
            [&dimension](const NcVariableInfo& candidate) {
                return candidate.name == dimension.name && candidate.dimensions.size() == 1U &&
                       candidate.dimensions.front() == dimension.name;
            });
        if (variable != dataset.variables.end()) {
            coordinate_names.insert(variable->name);
        }
    }
    for (const auto& variable : dataset.variables) {
        if (const auto coordinates =
                nc_string_attribute(ncid, variable.id, "coordinates")) {
            for (const auto& name : split_words(*coordinates)) {
                coordinate_names.insert(name);
            }
        }
    }
    for (auto& variable : dataset.variables) {
        variable.coordinate = coordinate_names.contains(variable.name);
    }

    dataset.global_attributes =
        nc_attribute_names(ncid, NC_GLOBAL, global_attribute_count);
    return dataset;
}

std::size_t element_count(const NcVariableInfo& variable) {
    std::size_t count = 1U;
    for (const std::size_t dimension : variable.shape) {
        if (dimension == 0U) {
            return 0U;
        }
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error("Variable is too large to inspect safely");
        }
        count *= dimension;
    }
    return count;
}

struct NumericValues {
    std::vector<long double> values;
    bool integer_values = false;
    bool datetime = false;
};

bool equals_fill_value(long double value, const std::vector<long double>& fills) {
    return std::any_of(fills.begin(), fills.end(), [value](long double fill) {
        return (std::isnan(value) && std::isnan(fill)) || value == fill;
    });
}

NumericValues read_numeric_values(int ncid, const NcVariableInfo& variable) {
    if (!is_numeric_nc_type(variable.type)) {
        throw std::runtime_error("unsupported non-numeric variable type " +
                                 nc_dtype_name(variable.type));
    }

    const std::size_t count = element_count(variable);
    std::vector<double> raw(count);
    if (count > 0U) {
        check_nc(nc_get_var_double(ncid, variable.id, raw.data()),
                 "failed to read values");
    }

    std::vector<long double> fills =
        nc_numeric_attribute(ncid, variable.id, "_FillValue");
    const auto missing = nc_numeric_attribute(ncid, variable.id, "missing_value");
    fills.insert(fills.end(), missing.begin(), missing.end());

    const auto scales = nc_numeric_attribute(ncid, variable.id, "scale_factor");
    const auto offsets = nc_numeric_attribute(ncid, variable.id, "add_offset");
    const long double scale = scales.empty() ? 1.0L : scales.front();
    const long double offset = offsets.empty() ? 0.0L : offsets.front();

    std::optional<CfTimeConversion> time_conversion;
    if (const auto units = nc_string_attribute(ncid, variable.id, "units")) {
        const auto calendar = nc_string_attribute(ncid, variable.id, "calendar");
        time_conversion = parse_cf_time_units(*units, calendar.value_or("standard"));
    }

    NumericValues result;
    result.integer_values = variable.integer_values;
    result.datetime = time_conversion.has_value();
    result.values.reserve(raw.size());
    for (const double raw_value : raw) {
        long double value = static_cast<long double>(raw_value);
        if (std::isnan(raw_value) || equals_fill_value(value, fills)) {
            result.values.push_back(std::numeric_limits<long double>::quiet_NaN());
            continue;
        }
        value = value * scale + offset;
        if (time_conversion) {
            value = time_conversion->origin_nanoseconds +
                    value * time_conversion->unit_nanoseconds;
            value = std::round(value);
        }
        result.values.push_back(value);
    }
    return result;
}

std::string format_number(long double value,
                          bool integer_value,
                          bool force_decimal = false) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value < 0.0L ? "-inf" : "inf";
    }
    if (integer_value &&
        value >= static_cast<long double>(std::numeric_limits<std::int64_t>::min()) &&
        value <= static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
        std::ostringstream stream;
        stream << static_cast<std::int64_t>(std::llround(value));
        return stream.str();
    }

    const double narrowed = static_cast<double>(value);
    std::array<char, 128> buffer{};
    const auto conversion = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                          narrowed, std::chars_format::general);
    std::string result;
    if (conversion.ec == std::errc{}) {
        result.assign(buffer.data(), conversion.ptr);
    } else {
        std::ostringstream stream;
        stream << std::setprecision(std::numeric_limits<double>::max_digits10)
               << narrowed;
        result = stream.str();
    }
    if (force_decimal && result.find_first_of(".eE") == std::string::npos) {
        result += ".0";
    }
    return result;
}

std::string format_numpy_preview_number(long double value, nc_type type) {
    if (is_integer_nc_type(type)) {
        return format_number(value, true);
    }
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value < 0.0L ? "-inf" : "inf";
    }
    std::ostringstream stream;
    stream << std::setprecision(8) << static_cast<double>(value);
    std::string result = stream.str();
    if (result.find_first_of(".eE") == std::string::npos) {
        result += '.';
    }
    return result;
}

std::string format_name_tuple(const std::vector<std::string>& values) {
    if (values.empty()) {
        return "()";
    }
    std::ostringstream stream;
    stream << '(';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0U) {
            stream << ", ";
        }
        stream << '\'' << values[index] << '\'';
    }
    if (values.size() == 1U) {
        stream << ',';
    }
    stream << ')';
    return stream.str();
}

std::string format_shape_tuple(const std::vector<std::size_t>& values) {
    if (values.empty()) {
        return "()";
    }
    std::ostringstream stream;
    stream << '(';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0U) {
            stream << ", ";
        }
        stream << values[index];
    }
    if (values.size() == 1U) {
        stream << ',';
    }
    stream << ')';
    return stream.str();
}

std::string format_dimension_signature(const NcVariableInfo& variable) {
    if (variable.dimensions.empty()) {
        return "";
    }
    std::ostringstream stream;
    stream << '(';
    for (std::size_t index = 0; index < variable.dimensions.size(); ++index) {
        if (index > 0U) {
            stream << ", ";
        }
        stream << variable.dimensions[index];
    }
    stream << ')';
    return stream.str();
}

std::string human_bytes(std::size_t bytes) {
    if (bytes < 1000U) {
        return std::to_string(bytes) + "B";
    }
    if (bytes < 1000U * 1000U) {
        return std::to_string(static_cast<std::size_t>(
                                  std::llround(static_cast<double>(bytes) / 1000.0))) +
               "kB";
    }
    return std::to_string(static_cast<std::size_t>(
                              std::llround(static_cast<double>(bytes) / 1000000.0))) +
           "MB";
}

std::size_t variable_bytes(int ncid, const NcVariableInfo& variable) {
    std::size_t type_size = 0U;
    if (nc_inq_type(ncid, variable.type, nullptr, &type_size) != NC_NOERR) {
        return 0U;
    }
    if (variable.datetime) {
        type_size = sizeof(std::int64_t);
    }
    const std::size_t count = element_count(variable);
    if (type_size != 0U && count > std::numeric_limits<std::size_t>::max() / type_size) {
        return std::numeric_limits<std::size_t>::max();
    }
    return count * type_size;
}

std::string attribute_text(int ncid, int varid, const std::string& name) {
    nc_type type = NC_NAT;
    std::size_t length = 0;
    check_nc(nc_inq_att(ncid, varid, name.c_str(), &type, &length),
             "Failed to inspect attribute " + name);
    if (type == NC_CHAR) {
        std::string value(length, '\0');
        if (length > 0U) {
            check_nc(nc_get_att_text(ncid, varid, name.c_str(), value.data()),
                     "Failed to read attribute " + name);
        }
        return value;
    }
    if (type == NC_STRING) {
        std::vector<char*> values(length, nullptr);
        if (length > 0U) {
            check_nc(nc_get_att_string(ncid, varid, name.c_str(), values.data()),
                     "Failed to read attribute " + name);
        }
        std::ostringstream stream;
        if (length > 1U) {
            stream << '[';
        }
        for (std::size_t index = 0; index < length; ++index) {
            if (index > 0U) {
                stream << ", ";
            }
            if (length > 1U) {
                stream << '\'';
            }
            stream << (values[index] != nullptr ? values[index] : "");
            if (length > 1U) {
                stream << '\'';
            }
        }
        if (length > 1U) {
            stream << ']';
        }
        if (length > 0U) {
            (void)nc_free_string(length, values.data());
        }
        return stream.str();
    }

    std::vector<double> values(length);
    if (length > 0U) {
        check_nc(nc_get_att_double(ncid, varid, name.c_str(), values.data()),
                 "Failed to read attribute " + name);
    }
    std::ostringstream stream;
    if (length > 1U) {
        stream << '[';
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0U) {
            stream << ' ';
        }
        if (length > 1U) {
            stream << format_numpy_preview_number(values[index], type);
        } else {
            stream << format_number(values[index], is_integer_nc_type(type),
                                    !is_integer_nc_type(type));
        }
    }
    if (length > 1U) {
        stream << ']';
    }
    return stream.str();
}

bool hidden_decoding_attribute(const std::string& name) {
    return name == "_FillValue" || name == "missing_value" ||
           name == "scale_factor" || name == "add_offset" ||
           name == "coordinates";
}

void print_nc_overview(int ncid,
                       const NcDatasetInfo& dataset,
                       std::ostream& out) {
    std::size_t total_bytes = 0U;
    for (const auto& variable : dataset.variables) {
        const std::size_t bytes = variable_bytes(ncid, variable);
        if (bytes > std::numeric_limits<std::size_t>::max() - total_bytes) {
            total_bytes = std::numeric_limits<std::size_t>::max();
            break;
        }
        total_bytes += bytes;
    }

    out << "<xarray.Dataset> Size: " << human_bytes(total_bytes) << '\n';
    out << "Dimensions:  (";
    for (std::size_t index = 0; index < dataset.dimensions.size(); ++index) {
        if (index > 0U) {
            out << ", ";
        }
        out << dataset.dimensions[index].name << ": " << dataset.dimensions[index].size;
    }
    out << ")\n";

    std::set<std::string> printed_coordinates;
    bool has_coordinates = false;
    for (const auto& dimension : dataset.dimensions) {
        const auto variable = std::find_if(
            dataset.variables.begin(), dataset.variables.end(),
            [&dimension](const NcVariableInfo& candidate) {
                return candidate.coordinate && candidate.name == dimension.name;
            });
        if (variable == dataset.variables.end()) {
            continue;
        }
        if (!has_coordinates) {
            out << "Coordinates:\n";
            has_coordinates = true;
        }
        const std::string dtype = variable->datetime ? "datetime64[ns]"
                                                     : nc_dtype_name(variable->type);
        out << "  * " << variable->name << " " << format_dimension_signature(*variable)
            << " " << dtype << " " << human_bytes(variable_bytes(ncid, *variable))
            << " ...\n";
        printed_coordinates.insert(variable->name);
    }
    for (const auto& variable : dataset.variables) {
        if (!variable.coordinate || printed_coordinates.contains(variable.name)) {
            continue;
        }
        if (!has_coordinates) {
            out << "Coordinates:\n";
            has_coordinates = true;
        }
        const std::string dtype = variable.datetime ? "datetime64[ns]"
                                                    : nc_dtype_name(variable.type);
        out << "    " << variable.name << " " << format_dimension_signature(variable)
            << " " << dtype << " " << human_bytes(variable_bytes(ncid, variable))
            << " ...\n";
    }

    std::vector<std::string> dimensions_without_coordinates;
    for (const auto& dimension : dataset.dimensions) {
        if (!printed_coordinates.contains(dimension.name)) {
            dimensions_without_coordinates.push_back(dimension.name);
        }
    }
    if (!dimensions_without_coordinates.empty()) {
        out << "Dimensions without coordinates: ";
        for (std::size_t index = 0; index < dimensions_without_coordinates.size();
             ++index) {
            if (index > 0U) {
                out << ", ";
            }
            out << dimensions_without_coordinates[index];
        }
        out << '\n';
    }

    bool has_data_variables = false;
    for (const auto& variable : dataset.variables) {
        if (variable.coordinate) {
            continue;
        }
        if (!has_data_variables) {
            out << "Data variables:\n";
            has_data_variables = true;
        }
        out << "    " << variable.name << " " << format_dimension_signature(variable)
            << " " << nc_dtype_name(variable.type) << " "
            << human_bytes(variable_bytes(ncid, variable)) << " ...\n";
    }
    if (!has_data_variables) {
        out << "Data variables:\n"
               "    *empty*\n";
    }

    if (!dataset.global_attributes.empty()) {
        out << "Attributes:\n";
        for (const auto& name : dataset.global_attributes) {
            out << "    " << name << ": " << attribute_text(ncid, NC_GLOBAL, name)
                << '\n';
        }
    }
}

long double quantile_sorted(const std::vector<long double>& sorted, long double q) {
    if (sorted.empty()) {
        return std::numeric_limits<long double>::quiet_NaN();
    }
    if (sorted.size() == 1U) {
        return sorted.front();
    }
    const long double position =
        q * static_cast<long double>(sorted.size() - 1U);
    const auto lower_index = static_cast<std::size_t>(std::floor(position));
    const auto upper_index = static_cast<std::size_t>(std::ceil(position));
    if (lower_index == upper_index) {
        return sorted[lower_index];
    }
    const long double fraction = position - static_cast<long double>(lower_index);
    return sorted[lower_index] +
           (sorted[upper_index] - sorted[lower_index]) * fraction;
}

struct PreviewData {
    std::vector<long double> values;
    nc_type type = NC_NAT;
};

void print_nc_dataset_info(int ncid,
                           const NcDatasetInfo& dataset,
                           std::ostream& out) {
    out << "\n===== Dataset Overview =====\n";
    print_nc_overview(ncid, dataset, out);

    out << "\n===== Variable Ranges (All Variables) =====\n";
    std::unordered_map<int, PreviewData> previews;
    for (const auto& variable : dataset.variables) {
        try {
            const NumericValues data = read_numeric_values(ncid, variable);
            if (data.values.empty()) {
                out << variable.name << ": empty variable\n";
                continue;
            }

            PreviewData preview;
            preview.type = data.integer_values ? variable.type : NC_DOUBLE;
            preview.values.assign(data.values.begin(),
                                  data.values.begin() +
                                      static_cast<std::ptrdiff_t>(
                                          std::min<std::size_t>(2U, data.values.size())));
            previews.emplace(variable.id, std::move(preview));

            std::vector<long double> finite_or_infinite;
            finite_or_infinite.reserve(data.values.size());
            std::copy_if(data.values.begin(), data.values.end(),
                         std::back_inserter(finite_or_infinite),
                         [](long double value) { return !std::isnan(value); });
            if (finite_or_infinite.empty()) {
                out << variable.name
                    << ": min = nan, q05 = nan, q95 = nan, max = nan\n";
                continue;
            }
            std::sort(finite_or_infinite.begin(), finite_or_infinite.end());
            const long double minimum = finite_or_infinite.front();
            const long double maximum = finite_or_infinite.back();
            const long double q05 = quantile_sorted(finite_or_infinite, 0.05L);
            const long double q95 = quantile_sorted(finite_or_infinite, 0.95L);
            out << variable.name << ": min = "
                << format_number(minimum, data.integer_values,
                                 !data.integer_values && !data.datetime)
                << ", q05 = "
                << format_number(q05, data.datetime, !data.datetime)
                << ", q95 = "
                << format_number(q95, data.datetime, !data.datetime)
                << ", max = "
                << format_number(maximum, data.integer_values,
                                 !data.integer_values && !data.datetime)
                << '\n';
        } catch (const std::exception& exception) {
            out << variable.name << ": failed to compute min/max/quantiles ("
                << exception.what() << ")\n";
        }
    }

    out << "\n===== Variable Details =====\n";
    for (const auto& variable : dataset.variables) {
        if (variable.coordinate) {
            continue;
        }
        out << "\nVariable: " << variable.name << '\n';
        out << "  Dimensions: " << format_name_tuple(variable.dimensions) << '\n';
        out << "  Shape: " << format_shape_tuple(variable.shape) << '\n';
        out << "  Data type: " << nc_dtype_name(variable.type) << '\n';
        const auto preview = previews.find(variable.id);
        if (preview == previews.end()) {
            out << "  Unable to preview data values.\n";
        } else {
            out << "  Preview first two values: [";
            for (std::size_t index = 0; index < preview->second.values.size(); ++index) {
                if (index > 0U) {
                    out << ' ';
                }
                out << format_numpy_preview_number(preview->second.values[index],
                                                   preview->second.type);
            }
            out << "]\n";
        }
        for (const auto& attribute : variable.attributes) {
            if (!hidden_decoding_attribute(attribute)) {
                out << "  Attribute " << attribute << ": "
                    << attribute_text(ncid, variable.id, attribute) << '\n';
            }
        }
    }

    if (!dataset.global_attributes.empty()) {
        out << "\n===== Global Attributes =====\n";
        for (const auto& attribute : dataset.global_attributes) {
            out << attribute << ": " << attribute_text(ncid, NC_GLOBAL, attribute)
                << '\n';
        }
    }
}

void inspect_netcdf(const fs::path& path, std::ostream& out) {
    out << "========================================\n";
    out << "Loading NetCDF\n";
    out << "Path: " << utf8_text(path) << '\n';
    out << "========================================\n";

    NcFile file(path);
    const NcDatasetInfo dataset = read_nc_structure(file.id());
    print_nc_dataset_info(file.id(), dataset, out);
}

#if LUW_HAS_GDAL

struct GdalDatasetCloser {
    void operator()(GDALDataset* dataset) const noexcept {
        if (dataset != nullptr) {
            GDALClose(dataset);
        }
    }
};

struct OgrFeatureCloser {
    void operator()(OGRFeature* feature) const noexcept {
        if (feature != nullptr) {
            OGRFeature::DestroyFeature(feature);
        }
    }
};

struct OgrTransformCloser {
    void operator()(OGRCoordinateTransformation* transform) const noexcept {
        if (transform != nullptr) {
            OCTDestroyCoordinateTransformation(transform);
        }
    }
};

using GdalDatasetPtr = std::unique_ptr<GDALDataset, GdalDatasetCloser>;
using OgrFeaturePtr = std::unique_ptr<OGRFeature, OgrFeatureCloser>;
using OgrTransformPtr =
    std::unique_ptr<OGRCoordinateTransformation, OgrTransformCloser>;

class QuietGdalErrors {
public:
    QuietGdalErrors() {
        CPLErrorReset();
        CPLPushErrorHandler(CPLQuietErrorHandler);
    }
    QuietGdalErrors(const QuietGdalErrors&) = delete;
    QuietGdalErrors& operator=(const QuietGdalErrors&) = delete;
    ~QuietGdalErrors() { CPLPopErrorHandler(); }
};

std::string last_gdal_error(std::string fallback) {
    const char* message = CPLGetLastErrorMsg();
    if (message != nullptr && *message != '\0') {
        return message;
    }
    return fallback;
}

std::string spatial_reference_display(const OGRSpatialReference& reference) {
    const char* authority = reference.GetAuthorityName(nullptr);
    const char* code = reference.GetAuthorityCode(nullptr);
    if (authority != nullptr && code != nullptr) {
        return std::string(authority) + ":" + code;
    }
    const char* name = reference.GetName();
    return name != nullptr ? name : "unknown CRS";
}

bool is_epsg_4326(const OGRSpatialReference& source) {
    OGRSpatialReference target;
    if (target.importFromEPSG(4326) != OGRERR_NONE) {
        return false;
    }
    OGRSpatialReference copy(source);
    copy.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    target.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return copy.IsSame(&target) != 0;
}

std::string ogr_field_dtype(const OGRFieldDefn& field) {
    switch (field.GetType()) {
    case OFTInteger:
    case OFTInteger64:
        return "int64";
    case OFTReal:
        return "float64";
    case OFTDate:
    case OFTTime:
    case OFTDateTime:
        return "datetime64[ms]";
    case OFTString:
    case OFTWideString:
        return "str";
    default:
        return "object";
    }
}

std::string quote_dataframe_value(const std::string& value) {
    if (value.find_first_of(" \t\r\n") == std::string::npos) {
        return value;
    }
    return '"' + value + '"';
}

std::string feature_field_text(const OGRFeature& feature, int index) {
    if (!feature.IsFieldSetAndNotNull(index)) {
        return "None";
    }
    const OGRFieldDefn* field = feature.GetFieldDefnRef(index);
    if (field == nullptr) {
        return "None";
    }
    switch (field->GetType()) {
    case OFTInteger:
        return std::to_string(feature.GetFieldAsInteger(index));
    case OFTInteger64:
        return std::to_string(feature.GetFieldAsInteger64(index));
    case OFTReal:
        return format_number(feature.GetFieldAsDouble(index), false);
    default:
        return quote_dataframe_value(feature.GetFieldAsString(index));
    }
}

std::string geometry_preview(const OGRGeometry* geometry) {
    if (geometry == nullptr || geometry->IsEmpty()) {
        return "None";
    }
    char* wkt = nullptr;
    if (geometry->exportToWkt(&wkt) != OGRERR_NONE || wkt == nullptr) {
        return "<geometry>";
    }
    std::string result(wkt);
    CPLFree(wkt);
    constexpr std::size_t kPreviewLimit = 72U;
    if (result.size() > kPreviewLimit) {
        result.resize(kPreviewLimit - 3U);
        result += "...";
    }
    return result;
}

struct ShapefileScan {
    bool has_bounds = false;
    OGREnvelope bounds{};
    std::vector<std::vector<std::string>> rows;
    bool transformation_failed = false;
    std::string transformation_error;
};

ShapefileScan scan_shapefile(OGRLayer& layer,
                             OGRCoordinateTransformation* transformation,
                             std::size_t preview_limit) {
    ShapefileScan result;
    layer.ResetReading();
    while (true) {
        OgrFeaturePtr feature(layer.GetNextFeature());
        if (!feature) {
            break;
        }

        OGRGeometry* geometry = feature->GetGeometryRef();
        if (geometry != nullptr && transformation != nullptr &&
            geometry->transform(transformation) != OGRERR_NONE) {
            result.transformation_failed = true;
            result.transformation_error =
                last_gdal_error("coordinate transformation failed");
            return result;
        }

        if (geometry != nullptr && !geometry->IsEmpty()) {
            OGREnvelope envelope;
            geometry->getEnvelope(&envelope);
            if (!result.has_bounds) {
                result.bounds = envelope;
                result.has_bounds = true;
            } else {
                result.bounds.MinX = std::min(result.bounds.MinX, envelope.MinX);
                result.bounds.MinY = std::min(result.bounds.MinY, envelope.MinY);
                result.bounds.MaxX = std::max(result.bounds.MaxX, envelope.MaxX);
                result.bounds.MaxY = std::max(result.bounds.MaxY, envelope.MaxY);
            }
        }

        if (result.rows.size() < preview_limit) {
            std::vector<std::string> row;
            const int field_count = feature->GetFieldCount();
            row.reserve(static_cast<std::size_t>(field_count + 1));
            for (int index = 0; index < field_count; ++index) {
                row.push_back(feature_field_text(*feature, index));
            }
            row.push_back(geometry_preview(geometry));
            result.rows.push_back(std::move(row));
        }
    }
    return result;
}

void print_crs_json(const OGRSpatialReference& reference, std::ostream& out) {
#if GDAL_VERSION_NUM >= GDAL_COMPUTE_VERSION(3, 1, 0)
    char* json = nullptr;
    const char* options[] = {"MULTILINE=YES", "INDENTATION_WIDTH=2", nullptr};
    if (reference.exportToPROJJSON(&json, options) == OGRERR_NONE && json != nullptr) {
        out << json << '\n';
        CPLFree(json);
        return;
    }
    if (json != nullptr) {
        CPLFree(json);
    }
#endif
    char* wkt = nullptr;
    if (reference.exportToPrettyWkt(&wkt, FALSE) == OGRERR_NONE && wkt != nullptr) {
        out << wkt << '\n';
        CPLFree(wkt);
        return;
    }
    if (wkt != nullptr) {
        CPLFree(wkt);
    }
    throw std::runtime_error(last_gdal_error("Failed to parse CRS"));
}

void print_shapefile_metadata(OGRLayer& layer,
                              const ShapefileScan& scan,
                              std::ostream& out) {
    const OGRFeatureDefn* definition = layer.GetLayerDefn();
    if (definition == nullptr) {
        throw std::runtime_error("Failed to read shapefile schema");
    }

    std::vector<std::string> column_names;
    std::vector<std::string> dtypes;
    const int field_count = definition->GetFieldCount();
    column_names.reserve(static_cast<std::size_t>(field_count + 1));
    dtypes.reserve(static_cast<std::size_t>(field_count + 1));
    for (int index = 0; index < field_count; ++index) {
        const OGRFieldDefn* field = definition->GetFieldDefn(index);
        column_names.emplace_back(field != nullptr ? field->GetNameRef() : "<unknown>");
        dtypes.push_back(field != nullptr ? ogr_field_dtype(*field) : "object");
    }
    column_names.emplace_back("geometry");
    dtypes.emplace_back("geometry");

    out << "Columns:\n[";
    for (std::size_t index = 0; index < column_names.size(); ++index) {
        if (index > 0U) {
            out << ", ";
        }
        out << '\'' << column_names[index] << '\'';
    }
    out << "]\n\nDtypes:\n";
    std::size_t width = 0U;
    for (const auto& name : column_names) {
        width = std::max(width, name.size());
    }
    for (std::size_t index = 0; index < column_names.size(); ++index) {
        out << std::left << std::setw(static_cast<int>(width + 2U)) << column_names[index]
            << dtypes[index] << '\n';
    }
    out << std::right << "dtype: object\n";

    out << "\nFirst 5 rows:\n";
    if (scan.rows.empty()) {
        out << "Empty GeoDataFrame\nColumns: [";
        for (std::size_t index = 0; index < column_names.size(); ++index) {
            if (index > 0U) {
                out << ", ";
            }
            out << column_names[index];
        }
        out << "]\nIndex: []\n";
        return;
    }
    out << "  ";
    for (const auto& name : column_names) {
        out << name << '\t';
    }
    out << '\n';
    for (std::size_t row_index = 0; row_index < scan.rows.size(); ++row_index) {
        out << row_index << ' ';
        for (const auto& value : scan.rows[row_index]) {
            out << value << '\t';
        }
        out << '\n';
    }
    out << "\n[" << scan.rows.size() << " rows x " << column_names.size()
        << " columns]\n";
}

void inspect_shapefile_native(const fs::path& path, std::ostream& out) {
    out << "========================================\n";
    out << "Loading shapefile\n";
    out << "Path: " << utf8_text(path) << '\n';
    out << "========================================\n";

    GDALAllRegister();
    QuietGdalErrors quiet_errors;
    GdalDatasetPtr dataset(static_cast<GDALDataset*>(
        GDALOpenEx(utf8_text(path).c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
                   nullptr, nullptr, nullptr)));
    if (!dataset) {
        throw std::runtime_error("Failed to read shapefile: " +
                                 last_gdal_error("unknown GDAL/OGR error"));
    }
    OGRLayer* layer = dataset->GetLayer(0);
    if (layer == nullptr) {
        throw std::runtime_error("Failed to read shapefile: no vector layer found");
    }

    out << "\nCRS full details\n";
    out << "========================================\n";
    const OGRSpatialReference* layer_reference = layer->GetSpatialRef();
    std::optional<OGRSpatialReference> source;
    if (layer_reference == nullptr) {
        out << "CRS: None\n";
    } else {
        source.emplace(*layer_reference);
        try {
            print_crs_json(*source, out);
        } catch (const std::exception& exception) {
            out << "Failed to parse CRS: " << exception.what() << '\n';
        }
    }

    std::optional<OGRSpatialReference> transformation_source;
    std::optional<OGRSpatialReference> transformation_target;
    OgrTransformPtr transformation;
    if (!source) {
        out << "\nShapefile has no CRS information. Assuming EPSG:4326.\n";
    } else if (!is_epsg_4326(*source)) {
        out << "\nReprojecting to EPSG:4326 from "
            << spatial_reference_display(*source) << ".\n";
        transformation_source.emplace(*source);
        transformation_target.emplace();
        transformation_source->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        CPLErrorReset();
        if (transformation_target->importFromEPSG(4326) == OGRERR_NONE) {
            transformation_target->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            transformation.reset(OGRCreateCoordinateTransformation(
                &*transformation_source, &*transformation_target));
        }
        if (!transformation) {
            out << "\nCould not evaluate CRS EPSG code. Proceeding without "
                   "reprojection. Details: "
                << last_gdal_error("could not create coordinate transformation")
                << '\n';
        }
    }

    ShapefileScan scan = scan_shapefile(*layer, transformation.get(), 5U);
    if (scan.transformation_failed) {
        out << "\nCould not evaluate CRS EPSG code. Proceeding without "
               "reprojection. Details: "
            << scan.transformation_error << '\n';
        scan = scan_shapefile(*layer, nullptr, 5U);
    }

    out << "\nRange of longitude and latitude\n";
    out << "========================================\n";
    out << std::fixed << std::setprecision(6);
    if (scan.has_bounds) {
        out << "Longitude (X): " << scan.bounds.MinX << " to " << scan.bounds.MaxX
            << '\n';
        out << "Latitude  (Y): " << scan.bounds.MinY << " to " << scan.bounds.MaxY
            << '\n';
    } else {
        out << "Longitude (X): nan to nan\n";
        out << "Latitude  (Y): nan to nan\n";
    }
    out << std::defaultfloat;

    out << "\nMeta data\n";
    out << "========================================\n";
    print_shapefile_metadata(*layer, scan, out);
}

#endif  // LUW_HAS_GDAL

}  // namespace

int run_cdfinspect(const std::vector<std::string>& args,
                   std::ostream& out,
                   std::ostream& err) {
    if (args.size() != 1U) {
        out << "Usage: python cdfInspect.py <input_file_path>\n";
        return 1;
    }

    fs::path path;
    try {
        path = resolve_input_path(args.front(), kNetCdfResolve, out, err);
    } catch (const std::runtime_error& exception) {
        out << "Error: " << exception.what() << '\n';
        return 1;
    }

    try {
        inspect_netcdf(path, out);
        return 0;
    } catch (const std::exception& exception) {
        out << "Unexpected failure: " << exception.what() << '\n';
        return 1;
    }
}

int run_shpinspect(const std::vector<std::string>& args,
                   std::ostream& out,
                   std::ostream& err) {
    if (args.size() != 1U) {
        out << "Usage: python shpInspect.py <input_file_path>\n";
        return 1;
    }

    fs::path path;
    try {
        path = resolve_input_path(args.front(), kShapefileResolve, out, err);
#if LUW_HAS_GDAL
        inspect_shapefile_native(path, out);
#else
        (void)path;
        throw std::runtime_error(
            "GDAL/OGR support is unavailable in this build; rebuild with "
            "LUW_ENABLE_GEOSPATIAL=ON and a discoverable GDAL installation.");
#endif
        return 0;
    } catch (const std::runtime_error& exception) {
        out << "Error: " << exception.what() << '\n';
        return 1;
    }
}

}  // namespace luw::cli
