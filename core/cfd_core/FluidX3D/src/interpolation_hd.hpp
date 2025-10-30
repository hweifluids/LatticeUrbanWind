#pragma once
#include "setup.hpp"
#include <vector>
#include <string>

struct InletInterpolatorHD {
    virtual ~InletInterpolatorHD() = default;
    virtual float3 eval(const float3& pos) const = 0;
};

class KNNInterpolatorHD final : public InletInterpolatorHD {
public:
    KNNInterpolatorHD(std::vector<float3> points, std::vector<float3> velocities)
        : P_(std::move(points)), U_(std::move(velocities)) {}

    float3 eval(const float3& pos) const override;

private:
    std::vector<float3> P_;
    std::vector<float3> U_;
};

class InletVelocityFieldHD {
public:
    InletVelocityFieldHD(const InletInterpolatorHD& interp, float z_base_lbmu)
        : interp_(interp), z_base_lbmu_(z_base_lbmu) {
    }

    float3 operator()(const float3& pos) const;

private:
    const InletInterpolatorHD& interp_;
    float z_base_lbmu_;
};

void apply_inlet_outlet_hd(LBM& lbm,
                           const std::string& downstream_bc,
                           const InletVelocityFieldHD& inlet,
                           unsigned long min_work_per_thread = 500000ull,
                           bool show_progress = true);
