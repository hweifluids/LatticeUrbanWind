#pragma once
#include "setup.hpp"
#include <vector>
#include <string>

// 本头文件提供“高阶插值”版本的独立接口，符号名与基础版不同，可同时编译链接。

struct InletInterpolatorHD {
    virtual ~InletInterpolatorHD() = default;
    virtual float3 eval(const float3& pos) const = 0;
};

// K 近邻反距离加权插值器（固定 K，线性扫描样本集）
class KNNInterpolatorHD final : public InletInterpolatorHD {
public:
    KNNInterpolatorHD(std::vector<float3> points, std::vector<float3> velocities)
        : P_(std::move(points)), U_(std::move(velocities)) {}

    float3 eval(const float3& pos) const override;

private:
    std::vector<float3> P_;
    std::vector<float3> U_;
};

// 入口速度场包装器：在 z0 平面之上施加 zoff 的采样偏移
class InletVelocityFieldHD {
public:
    InletVelocityFieldHD(const InletInterpolatorHD& interp, float z0, float zoff)
        : interp_(interp), z0_(z0), zoff_(zoff) {}

    float3 operator()(const float3& pos) const;

private:
    const InletInterpolatorHD& interp_;
    float z0_;
    float zoff_;
};

// 高阶版本的入口与出口赋值
void apply_inlet_outlet_hd(LBM& lbm,
                           const std::string& downstream_bc,
                           const InletVelocityFieldHD& inlet,
                           unsigned long min_work_per_thread = 500000ull,
                           bool show_progress = true);
