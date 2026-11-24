#pragma once

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Use VTK's configured PROJ wrapper so its symbol mangling always matches
// VTK::libproj, even when another PROJ SDK appears earlier on the include path.
#include <vtk_libproj.h>
#include <vtk_netcdf.h>

namespace luw::post::detail {

inline std::string pathUtf8(const std::filesystem::path& path) {
#if defined(__cpp_lib_char8_t)
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.u8string();
#endif
}

inline void ncCheck(int status, const std::string& context) {
    if (status != NC_NOERR) {
        throw std::runtime_error(context + ": " + nc_strerror(status));
    }
}

class NetcdfFile final {
public:
    explicit NetcdfFile(const std::filesystem::path& path) {
        const std::string utf8 = pathUtf8(path);
        ncCheck(nc_create(utf8.c_str(), NC_NETCDF4 | NC_CLOBBER, &id_),
                "Cannot create NetCDF file " + utf8);
    }

    ~NetcdfFile() {
        if (id_ >= 0) nc_close(id_);
    }

    NetcdfFile(const NetcdfFile&) = delete;
    NetcdfFile& operator=(const NetcdfFile&) = delete;

    [[nodiscard]] int id() const noexcept { return id_; }

    void close() {
        if (id_ < 0) return;
        const int id = std::exchange(id_, -1);
        ncCheck(nc_close(id), "Failed to close NetCDF file");
    }

private:
    int id_ = -1;
};

class Projection final {
public:
    Projection(std::string sourceCrs, std::string targetCrs)
        : context_(proj_context_create()) {
        if (!context_) throw std::runtime_error("Failed to create PROJ context");
        PJ* raw = proj_create_crs_to_crs(context_.get(), sourceCrs.c_str(), targetCrs.c_str(), nullptr);
        if (!raw) {
            throw std::runtime_error("Failed to create geodetic transform from "
                                     + sourceCrs + " to " + targetCrs);
        }
        PJ* normalized = proj_normalize_for_visualization(context_.get(), raw);
        proj_destroy(raw);
        transform_.reset(normalized);
        if (!transform_) {
            throw std::runtime_error("Failed to normalize geodetic transform from "
                                     + sourceCrs + " to " + targetCrs);
        }
    }

    [[nodiscard]] std::pair<double, double> apply(double x, double y) const {
        proj_errno_reset(transform_.get());
        const PJ_COORD result = proj_trans(transform_.get(), PJ_FWD, proj_coord(x, y, 0.0, 0.0));
        const int error = proj_errno(transform_.get());
        if (error != 0 || !std::isfinite(result.xy.x) || !std::isfinite(result.xy.y)) {
            const char* description = error == 0 ? nullptr : proj_errno_string(error);
            throw std::runtime_error("PROJ transformation failed"
                                     + std::string(description ? ": " : "")
                                     + std::string(description ? description : ""));
        }
        return {result.xy.x, result.xy.y};
    }

    void apply(std::vector<PJ_COORD>& coordinates) const {
        if (coordinates.empty()) return;
        proj_errno_reset(transform_.get());
        const int status = proj_trans_array(transform_.get(), PJ_FWD,
                                            coordinates.size(), coordinates.data());
        const int error = status != 0 ? status : proj_errno(transform_.get());
        if (error != 0) {
            const char* description = proj_errno_string(error);
            throw std::runtime_error("PROJ transformation failed: "
                                     + std::string(description ? description : "unknown error"));
        }
        for (const PJ_COORD& coordinate : coordinates) {
            if (!std::isfinite(coordinate.xy.x) || !std::isfinite(coordinate.xy.y)) {
                throw std::runtime_error("PROJ transformation produced a non-finite coordinate");
            }
        }
    }

private:
    struct ContextDeleter final {
        void operator()(PJ_CONTEXT* context) const noexcept {
            (void)proj_context_destroy(context);
        }
    };
    struct TransformDeleter final {
        void operator()(PJ* transform) const noexcept { proj_destroy(transform); }
    };

    std::unique_ptr<PJ_CONTEXT, ContextDeleter> context_;
    std::unique_ptr<PJ, TransformDeleter> transform_;
};

inline void setTextAttribute(int fileId, int variableId,
                             const char* name, const std::string& value) {
    ncCheck(nc_put_att_text(fileId, variableId, name, value.size(), value.c_str()),
            "Failed to set NetCDF attribute '" + std::string(name) + "'");
}

inline int defineCompressedFloatVariable(int fileId,
                                         const char* name,
                                         const std::vector<int>& dimensions,
                                         int compressionLevel = 3,
                                         bool shuffle = true,
                                         bool nanFill = true) {
    int variable = -1;
    ncCheck(nc_def_var(fileId, name, NC_FLOAT,
                       static_cast<int>(dimensions.size()), dimensions.data(), &variable),
            "Failed to define NetCDF variable '" + std::string(name) + "'");
    ncCheck(nc_def_var_deflate(fileId, variable, shuffle ? 1 : 0,
                               compressionLevel > 0 ? 1 : 0, compressionLevel),
            "Failed to configure NetCDF compression for '" + std::string(name) + "'");
    if (nanFill) {
        const float fill = std::numeric_limits<float>::quiet_NaN();
        ncCheck(nc_def_var_fill(fileId, variable, 0, &fill),
                "Failed to configure NetCDF fill value for '" + std::string(name) + "'");
    }
    return variable;
}

inline int defineCompressedDoubleVariable(int fileId,
                                          const char* name,
                                          const std::vector<int>& dimensions,
                                          int compressionLevel = 3,
                                          bool shuffle = true) {
    int variable = -1;
    ncCheck(nc_def_var(fileId, name, NC_DOUBLE,
                       static_cast<int>(dimensions.size()), dimensions.data(), &variable),
            "Failed to define NetCDF variable '" + std::string(name) + "'");
    ncCheck(nc_def_var_deflate(fileId, variable, shuffle ? 1 : 0,
                               compressionLevel > 0 ? 1 : 0, compressionLevel),
            "Failed to configure NetCDF compression for '" + std::string(name) + "'");
    return variable;
}

} // namespace luw::post::detail
