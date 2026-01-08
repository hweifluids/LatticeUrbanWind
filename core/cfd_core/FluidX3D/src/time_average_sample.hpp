// time_average_sample.hpp
#pragma once

#include "lbm.hpp"  // 包含 LBM 类的定义
#include <vector>
#include <string>

// 时均场采样函数声明（无默认参数）
void output_mean_field_sampling(const LBM& lbm,
    const std::vector<float>& mean_u_x,
    const std::vector<float>& mean_u_y,
    const std::vector<float>& mean_u_z,
    const std::vector<float>& mean_rho,
    ulong time_average_count,
    float u_scale,
    const std::string& output_path,
    float x_min, float x_max,
    float y_min, float y_max,
    float z_min, float z_max,
    float sample_interval_xy_si,  // 改为 xy 方向采样间隔
    float sample_interval_z_si,   // 添加 z 方向采样间隔参数
    float sample_xkm
    );