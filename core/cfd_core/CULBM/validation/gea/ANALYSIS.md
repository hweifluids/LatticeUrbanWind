# GEA 验证结果分析

`analyze_results.py` 扫描 `results/**/run_config.json`，结合终态、时序、场快照和
论文参考值，生成可审计的比较表与出版质量图表。

```powershell
python .\analyze_results.py
```

也可指定非默认目录：

```powershell
python .\analyze_results.py `
  --results D:\gea-results `
  --reference .\reference_metrics.csv `
  --output .\analysis
```

## 完成状态

脚本仅在以下条件同时成立时把 run 标为 `complete`：

1. 存在可解析的 `final_summary.json`；
2. 终态时间及 `metrics.csv` 末时刻均在浮点容差内等于
   `run_config.json` 的 `end_time_s`；
3. `finite=true`；
4. `positive_density=true`；
5. 必需终态物理量均为有限数；
6. 错误日志中没有 `GEA_ABORT`；
7. `final_summary.json` 不早于本次 `run_config.json`；
8. 未显式标记 `completed=false`；
9. 若存在 `suite_manifest*.csv`，对应最新清单记录的 `exit_code=0`。

`suite_manifest*.csv` 是可选输入：套件仍在运行、尚未生成清单时，脚本仍可依据
run 自身文件分类。脚本合并基础清单和算例专用清单；同一 run 采用修改时间最新
的记录，非零退出码会把对应 run 强制标为 `failed`。若某个 `run_config.json`
比匹配清单更新，说明已开始新一轮计算，该 run 会忽略旧记录，直到新清单生成。

数值失稳为 `failed`，仍在运行或缺少正常终态为 `incomplete`。后二者在
`summary.csv` 中保留进度和最后有效诊断值，但所有“最终物理量”列为空，防止把
异常末帧误用于验证结论。

`grid_registration` 是单独的审计字段。未显式记录该字段的旧热泡结果按已知
runner 行为标为 `legacy_half_cell_shift`：其 cell centers 是
`h/2,...,L-h/2`，而官方 GEA 热泡网格是 `0,h,...,L-h`，因此旧结果不是 exact
setup。密度流旧结果按 `official_boundary_aligned` 标记。若结果树中同时存在
后续修正版本，主比较按“官方注册优先、完成状态次优先”选择，旧结果仍保留在
`summary.csv` 中。路径位于 `registration_sensitivity/` 下的旧注册结果始终按
敏感性数据处理，即使其子目录名为 `main`，也不会进入主结果。

## 输出

- `analysis/summary.csv`：全部 run 的状态、参数、进度和严格终态结果。
- `analysis/main_grid_comparison.csv`：主网格逐指标对 GEA 参考值的绝对/相对偏差。
- `analysis/sensitivity_comparison.csv`：敏感性 run 与同算例同网格主 run 的比较。
- `analysis/time_step_convergence.csv`：同一 D3Q19 动态黏度网格上
  `dt=0.1/0.05/0.025 s` 的主指标、最大 Mach、相对主线及相邻时间步变化。
- `analysis/fine_grid_scaling_comparison.csv`：新增 h/2 网格在固定 `dt`、
  acoustic scaling (`dt∝h`) 与 diffusive scaling (`dt∝h²`) 下相对上一层最细
  主网格的主指标、`tau` 与最大 Mach。
- `analysis/stability_comparison.csv`：两个失败粗网格的主 SRT、TRT 和放大时间步
  SRT 稳定性矩阵。
- `analysis/viscosity_profile_comparison.csv`：热泡 h=31.25 m 使用地面、
  约 3 km 和约 8 km 背景密度进行 `mu/rho_ref` 映射时的逐指标偏差。
- `analysis/bubble_grid_convergence.{png,svg}`：热泡终态极值网格收敛。
- `analysis/density_front_comparison.{png,svg}`：密度流前缘及原文区间。
- `analysis/density_profile_z1200.{png,svg}`：密度流所有已完成且 `h<=100 m` 主线在
  `z=1200 m, t=900 s` 的水平温位扰动剖面。论文 Fig. 17 未提供 raw reference，
  该图只能审计 CULBM 网格间的相位/幅值及定性形态，不能计算对文献的
  \(L_p\) 误差。
- `analysis/transient_diagnostics.{png,svg}`：瞬态演化；失败 run 仅画有效前缀。
- `analysis/*_snapshots.{png,svg}`：最细已完成主网格的温位扰动场和速度矢量。
- `analysis/sensitivity_comparison.{png,svg}`：时间步、热格子和黏度映射敏感性。
- `analysis/fine_grid_time_step_convergence.{png,svg}`：新增最细 D3Q19 网格的
  主指标与最大 Mach 随时间步缩小的收敛诊断。
- `analysis/fine_grid_scaling_comparison.{png,svg}`：三条网格/时间步缩放路径的
  主指标变化及对应最大 Mach。
- `analysis/stability_matrix.{png,svg}`：粗网格碰撞模型/时间步的完成率和状态。
- `analysis/bubble_viscosity_profile.{png,svg}`：`rho_ref=1.161/0.90/0.55 kg/m3`
  （地面/约 3 km/约 8 km）黏度映射敏感性。

`reference_metrics.csv` 中热泡值来自 arXiv:2302.04836 的 Table 1；密度流值来自
Section 4.3 / Table 3 的 AV75 前缘区间。冷池前缘按物理定义采用
`theta_prime=-1 K` 交点；论文相邻文字中的 `+1 K` 与负温位扰动冷池不相容。

官方 GEA 设置审计基于本地 shallow checkout 的实际 HEAD
`bdf0b929fc92163cb946520e5f8bd3d1fd6ca45f`。`analysis/README.md` 每次运行时
自动写入本批数据的关键主线偏差、dt/2、D3Q7、numeric viscosity、背景密度映射
和粗网格稳定性摘要。
