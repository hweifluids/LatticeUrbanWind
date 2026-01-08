#pragma once
#include "setup.hpp"
#include <vector>
#include <string>

struct InletInterpolator {
    virtual ~InletInterpolator() = default;
    virtual float3 eval(const float3& pos) const = 0;
};

class NearestNeighborInterpolator final : public InletInterpolator {
public:
    NearestNeighborInterpolator(std::vector<float3> points,
        std::vector<float3> velocities)
        : P(std::move(points)), U(std::move(velocities)) {
    }

    float3 eval(const float3& pos) const override;

private:
    const std::vector<float3> P;
    const std::vector<float3> U;
};

class ConstantInletInterpolator final : public InletInterpolator {
public:
    explicit ConstantInletInterpolator(const float3& velocity) : velocity_(velocity) {}
    float3 eval(const float3& pos) const override { (void)pos; return velocity_; }

private:
    float3 velocity_;
};

class InletVelocityField {
public:
    InletVelocityField(const InletInterpolator& interp, float z0_lbmu, float z_off)
        : interp_(interp), z0_(z0_lbmu), zoff_(z_off) {
    }

    float3 operator()(const float3& pos) const;

private:
    const InletInterpolator& interp_;
    float z0_;
    float zoff_;
};

void apply_inlet_outlet(LBM& lbm,
    const std::string& downstream_bc,
    const InletVelocityField& inlet,
    unsigned long min_work_per_thread = 500000ull,
    bool show_progress = true);
