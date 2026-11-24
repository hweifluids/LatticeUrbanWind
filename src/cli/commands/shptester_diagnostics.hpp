#pragma once

#include <memory>
#include <string>

class OGRGeometry;

namespace luw::cli::detail {

struct ShpTesterDiagnosis {
    bool degenerate = false;
    std::string reason;
};

// A single instance is intentionally reused for a complete layer scan.  GEOS
// contexts and WKB readers are not free to construct for every feature.
class ShpTesterGeometryDiagnoser final {
public:
    ShpTesterGeometryDiagnoser();
    ~ShpTesterGeometryDiagnoser();

    ShpTesterGeometryDiagnoser(const ShpTesterGeometryDiagnoser&) = delete;
    ShpTesterGeometryDiagnoser& operator=(const ShpTesterGeometryDiagnoser&) = delete;

    ShpTesterGeometryDiagnoser(ShpTesterGeometryDiagnoser&&) noexcept;
    ShpTesterGeometryDiagnoser& operator=(ShpTesterGeometryDiagnoser&&) noexcept;

    [[nodiscard]] ShpTesterDiagnosis diagnose(const OGRGeometry* geometry) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace luw::cli::detail
