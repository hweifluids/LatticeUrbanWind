# 分析产物说明

本目录由 `analyze_results.py` 生成。最终物理量只来自严格正常完成的 run；
失败或未完成 run 的末帧不会进入比较表。

## 本次扫描状态

- 完成：31
- 失败：5
- 未完成：0

## 最终关键量

- 热泡最细主线 h=7.8125 m：w_max=17.565 m/s（无同网格 GEA 表值），theta'_max=1.564 K（无同网格 GEA 表值）。
- 密度流前缘相对 GEA（h=100/50/25 m）：-7.67% / -2.59% / +3.80%。
- 减小时间步（相对同 h、dt=0.1 主线）：热泡 w_max h=7.8125, dt=0.025: -8.15%, Ma=0.089 / h=7.8125, dt=0.05: -6.72%, Ma=0.182 / h=15.625, dt=0.05: -1.69%, Ma=0.089；密度流前缘 h=12.5, dt=0.025: -0.32%, Ma=0.107 / h=12.5, dt=0.05: -0.24%, Ma=0.213 / h=25, dt=0.05: -0.13%, Ma=0.102。
- D3Q7：热泡 w_max -0.41%，密度流前缘 -0.10%。
- numeric viscosity：热泡 w_max -0.27%，密度流前缘 0.36%。
- 背景密度映射对热泡 theta'_max：rho_ref=0.90 / 0.55 kg/m3 相对地面映射 -2.88% / -10.69%。
- 粗网格：原 SRT dt=0.1 s 与 TRT dt=0.1 s 均失败；SRT dt=1 s 在两个算例均走到终止时刻。后者是稳定性替代设置，不是原 dt 的严格复现。

## 图表

- `bubble_grid_convergence.*`：热泡终态极值与 GEA AV15 表 1。
- `density_front_comparison.*`：密度流 -1 K 前缘与 GEA AV75 区间。
- `density_profile_z1200.*`：所有 h<=100 m 已完成主线的 t=900 s 水平剖面；只作网格间定性审计。
- `transient_diagnostics.*`：主要瞬态量；失败 run 只显示有效前缀。
- `rising_thermal_snapshots.*`、`density_current_snapshots.*`：最细已完成主网格场。
- `sensitivity_comparison.*`：相对同网格主计算的变化。
- `fine_grid_time_step_convergence.*`：最细 D3Q19 网格主指标与最大 Mach 的 dt 收敛。
- `fine_grid_scaling_comparison.*`：h/2 时固定 dt、dt∝h 与 dt∝h² 三条缩放路径。
- `stability_matrix.*`：粗网格 SRT/TRT 与 dt 的完成时长及成败。
- `bubble_viscosity_profile.*`：rho_ref=1.161/0.90/0.55 黏度映射敏感性。

PNG 为 200 dpi；SVG 中场数据采用栅格层，文字与坐标保持矢量，避免文件过大。

注意：未显式记录 `grid_registration` 的旧热泡 run 按当前 runner 审计结果标记为
`legacy_half_cell_shift`，不视为 exact GEA setup。存在修正版本时，比较表按
官方注册优先、完成状态次优先的规则选择；密度流旧结果与官方边界注册一致。
