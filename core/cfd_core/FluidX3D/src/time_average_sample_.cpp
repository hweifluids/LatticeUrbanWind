#include "time_average_sample.hpp"
#include <fstream>      // 用于文件输出 std::ofstream
#include <vector>       // 用于 std::vector
#include <string>       // 用于 std::string
#include <sstream>      // 用于 std::ostringstream
#include <iomanip>      // 用于 std::setprecision, std::fixed
#include <cmath>        // 用于 sqrt
#include <algorithm>    // 用于 std::min

// 添加必要的函数声明
extern void println(const std::string& message);
extern Units units;  // 假设 Units 类已定义
//
//// 时均场采样函数
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
    float sample_interval_xy_si,  
    float sample_interval_z_si,  
    float sample_xkm
    ) {

    if (time_average_count == 0) {
        println("| 警告: 没有时均场数据可输出 |");
        return;
    }

    std::ofstream fout(output_path);
    if (!fout.is_open()) {
        println("| 错误: 无法创建输出文件 " + output_path + " |");
        return;
    }

    fout << "x,y,z,mean_velocity_x,mean_velocity_y,mean_velocity_z,mean_velocity_magnitude,mean_density,mean_pressure" << std::endl;

    const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
    const float c_s2 = 1.0f / 3.0f;

    uint samples_written = 0u;
    auto fmtf = [](float v, int prec = 2) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(prec) << v;
        return os.str();
        };

  

    for (float z_si = z_min; z_si <= z_max; z_si += sample_interval_z_si) {  // z 用 sample_interval_z_si
        for (float y_si = y_min; y_si <= y_max; y_si += sample_interval_xy_si) {  // y 用 sample_interval_xy_si
            for (float x_si = x_min; x_si <= x_max; x_si += sample_interval_xy_si) {  // x 用 sample_interval_xy_si
                // 坐标偏移和转换
                float x_lbm = units.x(x_si);
                float y_lbm = units.x(y_si);
                float z_lbm = units.x(z_si);

                // 找到最近的网格点
                uint x_idx = std::min(Nx - 1u, (uint)(x_lbm + 0.5f));
                uint y_idx = std::min(Ny - 1u, (uint)(y_lbm + 0.5f));
                uint z_idx = std::min(Nz - 1u, (uint)(z_lbm + 0.5f));

                ulong n = lbm.index(x_idx, y_idx, z_idx);

                // 获取时均场数据并转换单位
                float mean_vel_x = mean_u_x[n] / u_scale;
                float mean_vel_y = mean_u_y[n] / u_scale;
                float mean_vel_z = mean_u_z[n] / u_scale;
                float mean_vel_mag = sqrt(mean_vel_x * mean_vel_x + mean_vel_y * mean_vel_y + mean_vel_z * mean_vel_z);
                float mean_density = mean_rho[n];
                float mean_pressure = c_s2 * (mean_density - 1.0f);

                // 写入CSV行
                fout << fmtf(x_si, 1) << "," //将坐标转换为[-1000,1000][-800,800]写入csv表格中
                    << fmtf(y_si, 1) << ","
                    << fmtf(z_si, 1) << ","
                    << fmtf(mean_vel_x, 6) << ","
                    << fmtf(mean_vel_y, 6) << ","
                    << fmtf(mean_vel_z, 6) << ","
                    << fmtf(mean_vel_mag, 6) << ","
                    << fmtf(mean_density, 6) << ","
                    << fmtf(mean_pressure, 6) << std::endl;

                samples_written++;

            }
        }
    }

    fout.close();

    // 计算预期总点数
    uint total_x_points = (uint)((x_max - x_min) / sample_interval_xy_si) + 1;
    uint total_y_points = (uint)((y_max - y_min) / sample_interval_xy_si) + 1;
    uint total_z_points = (uint)((z_max - z_min) / sample_interval_z_si) + 1;
    uint expected_samples = total_x_points * total_y_points * total_z_points;

    println("| 时均场采样完成: " + to_string(samples_written) + "/" + to_string(expected_samples) + " 个点 |");
    println("| 文件位置: " + output_path + " |");
}









