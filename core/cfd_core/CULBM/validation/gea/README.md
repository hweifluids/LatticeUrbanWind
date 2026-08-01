# GEA 大气基准的 CULBM 数值验证

本目录复现 arXiv:2302.04836 中的两个二维大气基准，并提供冻结的人工黏度（AV）
基线与独立的 Smagorinsky-Lilly SGS/LES 敏感性目标：

- rising thermal bubble（AV15）；
- density current（AV75）。

这里验证的是 **CULBM 的弱可压 Boussinesq analogue**，不是对 GEA/OpenFOAM
可压缩干空气方程的逐方程复现。初值、计算时长、网格族、时间步、自由滑移壁面、
`Pr=1` 和论文中的人工动力黏度数值被尽可能对齐；状态方程、静力背景、黏度的空间
依赖和离散方法仍然不同。因此结果适合回答“CULBM 对这两个经典浮力流动的网格收敛
和定量偏差如何”，不能单独用来证明两个程序求解了完全相同的数学问题。

基准来源：

- 论文：<https://arxiv.org/abs/2302.04836>；
- 官方 GEA 仓库：<https://github.com/GEA-Geophysical-and-Environmental-Apps/GEA>；
- 本次参数审计使用的本地官方仓库提交：
  `bdf0b929fc92163cb946520e5f8bd3d1fd6ca45f`。

论文中的 AV15 热泡终态参考值位于 Table 1，AV75 密度流前缘参考区间位于
Table 3。结构化参考值保存在 `reference_metrics.csv`。

## 基准定义

### Rising thermal bubble

| 项目 | 设置 |
|---|---|
| 连续域（论文表述） | `[0,5000] × [0,10000] m²` |
| 终止时间 / 时间步 | `1020 s` / `0.1 s` |
| 背景位温 | `300 K` |
| 扰动 | `θ′=2(1-r/2000)` K，`r≤2000 m` |
| 论文中心 | `(xc,zc)=(5000,2000) m` |
| 官方仓库离散中心 | `(xc,zc)=(0,2000) m` |
| 人工动力黏度 / Pr | `μa=15 Pa s` / `1` |
| 网格间距 | `250, 125, 62.5, 31.25, 15.625 m` |

论文把半圆热泡放在右壁，官方 GEA case 把它放在左壁；二者通过水平镜像对应，
所以全域极值可直接比较，场图比较时必须先统一朝向。主 runner 默认遵循官方仓库，
即 `--bubble-center-x 0`。该选项只移动初值，不改变下述网格注册；直接设为
`5000` 并不等价于重新生成论文表述的常规 `[0,L]` 网格，不能作为主对比设置。

官方热泡网格还有一个容易遗漏的离散注册细节：其 block bounds 是
`[-h/2,L-h/2]`，所以物理 cell centers 是 `0,h,...,L-h`，而不是
`h/2,3h/2,...,L-h/2`。runner 对热泡采用相同注册；两侧 ghost wall 分别位于
`-h/2` 和 `L-h/2`。这忠实匹配 checked-in GEA 离散 case，但与论文写出的连续域
边界仍有半格的表述差异。`--write-vtk` 和 canonical snapshot CSV 使用相同注册。

### Density current

| 项目 | 设置 |
|---|---|
| 域 | `[0,25600] × [0,6400] m²` |
| 终止时间 / 时间步 | `900 s` / `0.1 s` |
| 背景位温 | `300 K` |
| 扰动 | `θ′=-7.5[1+cos(πr)]` K，`r≤1` |
| 椭圆半径 / 中心 | `(xr,zr)=(4000,2000) m`，`(xc,zc)=(0,3000) m` |
| 人工动力黏度 / Pr | `μa=75 Pa s` / `1` |
| 网格间距 | `400, 200, 100, 50, 25 m` |

密度流使用常规 bounds `[0,L]`，物理 cell centers 为
`h/2,3h/2,...,L-h/2`。论文正文把冷池前缘写成地面 `θ′=+1 K` 的位置；这与整个
冷异常的符号不相容，是明显的符号笔误。runner 使用物理一致的 `θ′=-1 K`：
在底层 cell centers（`z=h/2`）的 y 平均剖面上找最右侧向上的阈值穿越，并在相邻
x cell centers 之间线性插值。`front_bracket_left_m/right_m` 是 CULBM 相邻网格
中心构成的宽 `h` 区间；论文 Table 3 的区间是独立参考区间，不能把两者的区间宽度
当成同一离散量。

## CULBM 映射

### 冻结的 AV 基线

主验证二进制 `CULBM-gea-validation` 默认使用：

- 动量 D3Q19；
- 热标量 D3Q19；
- SRT/BGK 碰撞；
- FP32 DDF；
- 关闭生产 `SUBGRID` 和 `FP16C`；
- x/z 外壁为显式 link-wise specular free-slip，标量绝热；
- y 方向 4 个周期格，模拟 OpenFOAM `empty` 的二维方向。

若初值和边界保持 y 均匀，解应保持二维；`max_y_theta_spread_K` 用于检查是否出现
非预期的 y 向分叉。

物理到格子单位的主映射为

```text
rho_ref = 100000 / (287 × 300) = 1.16144018583 kg/m³
nu_SI   = mu_a / rho_ref
nu_LBM  = nu_SI × dt / h²
tau     = 0.5 + 3 nu_LBM
alpha_LBM = nu_LBM                  (Pr = 1)
g_z,LBM = -9.81 × dt² / h
```

因此热泡 `ν=12.915 m²/s`，密度流 `ν=64.575 m²/s`。这是把 GEA 的常量动力黏度
用地面背景密度换算成 **全域常量运动黏度**；GEA 中实际的 `μa/ρ(z)` 随静力密度
变化，二者只在选定参考密度处对齐。

标量使用仿射表示

```text
T_LBM = 1 + theta_prime / theta_scale
beta_LBM = theta_scale / 300
```

热泡默认 `theta_scale=2 K`，密度流默认 `15 K`，故浮力乘积始终为
`theta_prime/300`。CULBM 的 `T` 在本验证中代表位温扰动的归一化标量，不是 GEA
总能量方程中的热力学温度。

### SGS/LES 目标与系数映射

`CULBM-gea-validation-sgs` 是单独的 SGS 敏感性目标。其编译默认值为：

| 设置 | 默认值 |
|---|---:|
| 动量 SGS | Smagorinsky-Lilly，开启 |
| 热标量 SGS | 开启，`alpha_t=nu_t/Pr_t` |
| `Pr_t` | `1` |
| `Cs²` | `0.453989945` |
| `Cs` | `sqrt(Cs²)=0.673787760` |
| 请求的基础 `nu` / `alpha` | 均为 `1.48e-5 m²/s` |
| 最小动量 tau | `0.5001`（进入 FP32 后按可表示值向上取整） |

`--sgs-cs` 接受传统 Smagorinsky 常数 `Cs`，runner 内部将其平方；
`--sgs-cs-squared` 直接接受 `Cs²`，二者互斥。默认 GEA 映射对应 kernel prefactor
`18 sqrt(2) Cs² = 11.5566973`。`--sgs-scalar off` 只关闭热标量的湍流扩散耦合，
不会关闭动量 SGS；`--sgs-pr-t` 控制开启耦合时的湍流 Prandtl 数。

两个可执行文件的默认配置是整体配置，不能只靠切换 `--sgs` 得到彼此等价的算例：

- AV 目标加 `--sgs on` 仍保留 AV 基础输运和 AV 目标的默认 `Cs²`；
- SGS 目标加 `--sgs off` 仅关闭涡黏度，仍保留 SGS 目标的基础输运请求和 tau floor。

因此正式 A/B 对照应记录完整命令，并以 `run_config.json` 中的请求值、有效值和
SGS 参数为准。

### tau floor、Pr 和有效基础输运

SGS 目标不再把 D3Q7 与 D3Q19 的最小 thermal tau 都硬设为 `0.5001`。未显式传入
`--min-thermal-tau` 时，runner 先由 FP32 可表示的动量下限求基础运动黏度下限，
再保持请求的基础输运比：

```text
r = alpha_requested / nu_requested
nu_floor,LBM    = (tau_floor,momentum - 0.5) / 3
alpha_floor,LBM = r * nu_floor,LBM
tau_floor,T     = 0.5 + alpha_floor,LBM / cs_T²

cs_T² = 1/3  (D3Q19)
cs_T² = 1/4  (D3Q7)
```

默认 `r=1`，所以两种热格子都保持 `alpha_floor=nu_floor`：D3Q19 的 thermal tau
约为 `0.5001`，D3Q7 约为 `0.5001333`。这避免了“两个 tau 数值相同”在 D3Q7 下
产生 `alpha/nu=0.75`。显式 `--min-thermal-tau VALUE` 始终优先，并将 derived 标志
置为 false；显式传入 `0.5` 会关闭 thermal floor。其他显式值应由用户自行确认是否
符合所需的基础 Prandtl 数。

tau floor 是 FP32 稳定性/可表示性措施，不是分子物性。floor 触发后，实际基础输运为

```text
nu_effective,SI = nu_floor,LBM * h² / dt
alpha_effective,SI = alpha_floor,LBM * h² / dt
```

所以它随网格间距 `h` 的平方增大、随 `dt` 的倒数增大。按名义
`tau_floor=0.5001` 估算，`nu_floor,LBM≈3.333e-5`；例如 `dt=0.1 s` 时：

| h | floor 后约等效基础 `nu` |
|---:|---:|
| `400 m` | `53.3 m²/s` |
| `12.5 m` | `0.0521 m²/s` |
| `5 m` | `0.00833 m²/s` |

这些值均不能称为分子输运。`1.48e-5 m²/s` 只是 requested base transport；一旦
floor 生效，报告必须使用 `effective_nu_m2_s` 和 `effective_alpha_m2_s`。由于 FP32
向上取整，精确值以输出为准：`run_config.json` 的 `minimum_thermal_tau` 是实际解析后
使用的 thermal floor，`minimum_thermal_tau_derived` 说明其是否由动量下限推导；
console 的 `GEA_CONFIG` 始终输出同样信息，floor 实际触发时还会输出
`GEA_TAU_FLOOR`。

## 完整网格矩阵

`run_suite.ps1` 的主矩阵不是 smoke test。每个 case 都运行论文给出的全部五级网格
至正式终止时间：

| Case | h (m) | 物理 x×z 格数 | 含 ghost 的 CULBM 网格 | dt (s) | 步数 |
|---|---:|---:|---:|---:|---:|
| Rising thermal | 250 | 20×40 | 22×4×42 | 0.1 | 10200 |
| Rising thermal | 125 | 40×80 | 42×4×82 | 0.1 | 10200 |
| Rising thermal | 62.5 | 80×160 | 82×4×162 | 0.1 | 10200 |
| Rising thermal | 31.25 | 160×320 | 162×4×322 | 0.1 | 10200 |
| Rising thermal | 15.625 | 320×640 | 322×4×642 | 0.1 | 10200 |
| Density current | 400 | 64×16 | 66×4×18 | 0.1 | 9000 |
| Density current | 200 | 128×32 | 130×4×34 | 0.1 | 9000 |
| Density current | 100 | 256×64 | 258×4×66 | 0.1 | 9000 |
| Density current | 50 | 512×128 | 514×4×130 | 0.1 | 9000 |
| Density current | 25 | 1024×256 | 1026×4×258 | 0.1 | 9000 |

默认还执行六个敏感性 run：

- 两个最细网格的 `dt=0.05 s`；
- 两个最细网格的热标量 D3Q7；
- 热泡 `h=31.25 m`、密度流 `h=50 m` 的 numeric-viscosity 解释，即把
  `15/75` 直接视为 `m²/s`。

numeric-viscosity 只是歧义/模型敏感性，不属于主 GEA 对齐结果。

### D3Q19 加密一级延拓

`run_fine_grid_d3q19.ps1` 在论文表格最细网格之外各加密一级：

| Case | h (m) | 物理 x×z 格数 | 含 ghost 的 CULBM 网格 | dt (s) | 步数 |
|---|---:|---:|---:|---:|---:|
| Rising thermal | 7.8125 | 640×1280 | 642×4×1282 | 0.1 | 10200 |
| Density current | 12.5 | 2048×512 | 2050×4×514 | 0.1 | 9000 |

脚本同时执行两个 `dt=0.05 s` 的等马赫数控制组。网格减半而保持 `dt=0.1 s`
会使格子速度和 Mach 数约增一倍；`dt` 同时减半则保持与上一层最细主网格近似相同
的格子 Mach 数。因此 `dt=0.1 s` 用于直接延拓原主矩阵，`dt=0.05 s` 用于区分
空间加密效应和弱可压时间步效应。论文 Table 1/3 没有这两个新增 h 的 GEA 表值，
新增点只能用于 CULBM 自身的网格/时间步趋势审计，不能伪称同网格文献误差。

## 构建和运行

从仓库根目录执行：

```powershell
cmake -S core/cfd_core/CULBM `
  -B core/cfd_core/CULBM/build `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCULBM_ENABLE_MPI=OFF `
  -DBUILD_TESTING=ON

cmake --build core/cfd_core/CULBM/build `
  --target CULBM-gea-validation CULBM-gea-validation-sgs `
           CULBM-gea-validation-trt `
  --parallel
```

直接运行一个 SGS 算例：

```powershell
$sgsExe = 'core/cfd_core/CULBM/build/CULBM-gea-validation-sgs.exe'
& $sgsExe `
  --case density-current `
  --h 50 `
  --dt 0.1 `
  --thermal D3Q19 `
  --sample-seconds 5 `
  --output D:\gea-validation\sgs-density-h50-new
```

也可让主矩阵脚本使用 SGS 可执行文件；建议先用 `-MainOnly`，避免把 AV 专用的
numeric-viscosity 敏感性项混入 SGS 矩阵：

```powershell
core/cfd_core/CULBM/validation/gea/run_suite.ps1 `
  -Executable $sgsExe `
  -MainOnly `
  -ResultsRoot D:\gea-validation\sgs-main-20260731
```

现有套件脚本不传 `-Executable` 时仍运行冻结的 `CULBM-gea-validation` AV 基线。

在两张 GPU 上成对并行执行完整主矩阵和敏感性矩阵：

```powershell
core/cfd_core/CULBM/validation/gea/run_suite.ps1 `
  -ResultsRoot D:\gea-validation\run-20260731
```

只运行正式主矩阵：

```powershell
core/cfd_core/CULBM/validation/gea/run_suite.ps1 `
  -MainOnly `
  -ResultsRoot D:\gea-validation\main-20260731
```

也可用 `-CaseFilter rising-thermal` 或 `-CaseFilter density-current` 单独选择一个
case。脚本把每个进程限制到一张 GPU，并在两个 GPU 上同时跑一对独立 case；这不是
一个 case 的域分解。

在两张 GPU 上执行 D3Q19 加密一级主计算和等马赫数控制：

```powershell
core/cfd_core/CULBM/validation/gea/run_fine_grid_d3q19.ps1 `
  -ResultsRoot D:\gea-validation\fine-d3q19-20260731
```

该脚本固定并校验 runner SHA-256，`dt=0.1 s` 主计算保留完整标准时刻切片，
`dt=0.05 s` 控制组保留全时序指标及密度流终态剖面，但不重复写大体积二维切片。
若这一级在 `dt=0.05 s` 下仍未进入目标低 Mach 区间，可继续执行不写重复切片的
`dt=0.025 s` 控制：

```powershell
core/cfd_core/CULBM/validation/gea/run_fine_grid_d3q19.ps1 `
  -RunSet low-mach `
  -ResultsRoot D:\gea-validation\fine-d3q19-20260731
```

相对上一层最细主网格，`h` 减半且 `dt` 减为四分之一是 LBM 的 diffusive
scaling：`nu_LBM=nu_SI dt/h²` 与 `tau` 保持不变，同时格子速度/Mach 约减半。
因此该组既是低 Mach 控制，也是在相同松弛参数下审计细网格趋势的关键配对。

脚本在启动整个矩阵前检查所有将使用的 run 目录，只要任一目录非空就拒绝运行，
防止失败复跑混入旧 snapshot/profile。直接调用 runner 时也应始终使用新的空输出目录：

```powershell
$exe = 'core/cfd_core/CULBM/build/CULBM-gea-validation.exe'
& $exe `
  --case rising-thermal `
  --h 125 `
  --dt 0.1 `
  --thermal D3Q19 `
  --viscosity dynamic `
  --sample-seconds 5 `
  --output D:\gea-validation\bubble-h125-new
```

`run_stability_matrix.ps1` 是粗网格失稳诊断，不是主结果替代品。它对两个最粗网格
分别测试 SRT `dt=1 s` 和仅动量改为 TRT、`dt=0.1 s` 的控制组。改变时间步或碰撞
算子会改变误差来源，结果必须单独标为 stability sensitivity。

```powershell
core/cfd_core/CULBM/validation/gea/run_stability_matrix.ps1 `
  -ResultsRoot D:\gea-validation\stability-20260731
```

`run_viscosity_profile_diagnostic.ps1` 在热泡 `h=31.25 m` 上把 GEA 的
`mu_a=15 Pa s` 分别用约 3 km 和 8 km 的静力背景密度
`rho_ref=0.90/0.55 kg/m³` 映射为 `nu=16.67/27.27 m²/s`，并各自完整运行到
`t=1020 s`。它用于量化“GEA 常动力黏度、CULBM 常运动黏度”这一模型差异，
不替代以地面密度映射的主结果。

```powershell
core/cfd_core/CULBM/validation/gea/run_viscosity_profile_diagnostic.ps1 `
  -ResultsRoot D:\gea-validation\viscosity-profile-20260731
```

## 输出与指标

每个 run 目录包含：

- `run_config.json`：模型标签、格子/碰撞、注册、单位映射、网格和松弛参数；
- `metrics.csv`：默认每 5 s 一行的时序诊断；
- `final_summary.json`：完成或检测到数值失稳时的末次摘要；
- `snapshots/slice_t*.csv`：指定时刻的 y 平均二维场；
- `profile_z1200_tfinal.csv`：仅密度流完成后写出的 `z=1200 m` 线性插值剖面；
- `vtk/t*/`：仅传入 `--write-vtk` 时生成；
- `console.stdout.log` / `console.stderr.log`：由套件脚本保存；
- suite 根目录的 `suite_manifest*.csv`：退出码、状态、GPU 和耗时。

热泡 snapshot 时刻为 `0,300,600,900,1020 s`；密度流为
`0,300,600,750,900 s`。

全域指标在所有物理 x/z cells 和全部 y cells 上计算，ghost 不参与：

- `u_min/max_m_s`、`w_min/max_m_s`：水平和竖直速度极值；
- `theta_prime_min/max/mean/rms_K`：位温扰动统计；
- `rho_min/max/mean_lbm`：弱可压格子密度统计；
- `max_speed_m_s`、`max_mach`：速度和格子声速定义的最大 Mach 数；
- `max_y_theta_spread_K`：同一 x/z 位置跨 y 的最大位温差；
- `front_x_m` 和 bracket：仅密度流，定义见上文。

密度流的 `z=1200 m` profile 对夹住目标高度的两层 cell centers 做线性插值，并对
y 平均。二维 snapshot 也对 y 平均；这些 CSV 是场图和剖面对比的 canonical 数据。

运行后执行：

```powershell
python core/cfd_core/CULBM/validation/gea/analyze_results.py `
  --results D:\gea-validation\run-20260731 `
  --reference core/cfd_core/CULBM/validation/gea/reference_metrics.csv `
  --output D:\gea-validation\run-20260731-analysis
```

分析器的完成判据和图表清单见 `ANALYSIS.md`。

## 失败判据与解释约束

runner 退出码：

- `0`：达到终止时间；
- `2`：参数、I/O、初始化或运行时异常；
- `3`：采样时检测到非有限字段或非正密度；
- `77`：无可用 CUDA device。

非有限/非正密度在采样或 snapshot 时检查，默认间隔不超过 5 s。`max_mach>0.1`
只产生弱可压误差警告，`max_mach>0.3` 产生高误差警告，但两者不会自动判失败。
套件把任意非零退出码记为 failed；分析器还要求终态时间、有限量和 manifest 状态
同时通过，避免把 abort 摘要当成有效终态。

SRT 的低黏度稳定性由 `tau-0.5` 控制。主设置中两个最粗网格尤其接近下限：

```text
热泡 h=250 m:       tau ≈ 0.500061992
密度流 h=400 m:    tau ≈ 0.500121078
```

若它们失稳，应保留为主矩阵的数值验证结论，再用时间步/TRT 控制组定位原因；不能
静默用控制组替换论文 `dt=0.1 s` 的主结果。

## 已知模型差异

偏差分析至少要区分以下来源：

1. **控制方程**：GEA 求解带总能量和理想气体状态方程的可压缩干空气 Euler/LES；
   CULBM 求解弱可压动量方程和被动扩散标量，并用 Boussinesq 浮力反馈。
2. **静力背景**：GEA 初始化随高度变化的 `p(z),ρ(z),T(z)`；CULBM 以
   `ρ_LBM=1` 和均匀归一化背景开始，没有真实大气密度下降。
3. **黏度**：GEA 使用常量动力黏度 `μa`，其等效运动黏度随 `1/ρ(z)` 变化；
   主 CULBM 映射使用地面背景密度得到全域常量 `ν`。
4. **热力学变量**：GEA 的位温由压力和热力学温度定义并进入焓/总能量耦合；
   CULBM 直接输运归一化 `θ′`。
5. **离散方法**：GEA 是有限体积、BDF1 和压力/速度分裂；CULBM 主结果是
   D3Q19 SRT 格子 Boltzmann。耗散、色散和边界截断误差不会相同。
6. **二维实现**：GEA 使用 `empty` patch；CULBM 使用 4-cell 周期薄层。正常时
   `max_y_theta_spread_K` 应接近舍入误差，但它仍是数值实现差异。
7. **弱可压误差**：格子 Mach 数随 `dt/h` 缩放。网格与时间步敏感性必须和物理
   模型误差一起报告。

热泡初始最大扰动仅为背景的约 `0.67%`，Boussinesq 近似较温和；密度流达到
`-15 K`（约 `5%`），预期模型形式误差更明显。因而即使网格趋于稳定，CULBM 与
GEA 的终态极值或前缘仍可能存在系统偏差。只有在网格、`dt`、热格子和黏度解释
敏感性都量化后，才应把剩余差异主要归因于方程/状态模型。
