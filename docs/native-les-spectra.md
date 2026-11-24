# 原生 LES 频谱分析

`les_spectra` 是旧版 `core/tools_core/les_spectra.py` 的 C++20 原生替代，运行时不会启动、嵌入或查找 Python。统一入口和兼容命令名等价：

```text
luw les_spectra wind.vtk --output-dir spectra
les_spectra wind.vtk --quick-test --test-height-count 3 --test-3d-layers 24
```

未提供输入文件时，仍使用 `uvwrho-guangzhou4kmC3_20250910091000.vtk`。高度起点和间隔默认均为 50 m，`--workers -1` 使用全部可用工作线程；其余参数名称、默认值、quick-test 截取规则和 argparse 风格错误状态保持兼容。

## 输入契约

- 输入必须是 BINARY legacy VTK `STRUCTURED_POINTS`。
- 必须提供 `DIMENSIONS`、`SPACING`、`ORIGIN`、`POINT_DATA`、`SCALARS` 和 `LOOKUP_TABLE default`。
- 首个标量块必须声明为 `float`；数据按 legacy VTK 的大端 float32 顺序读取，可包含一个或多个分量。
- `POINT_DATA` 必须等于三个维度的乘积，且文件必须包含声明的完整标量 payload。

程序只读映射 VTK 文件，不复制完整输入场。点是否有效仍按“任意分量不等于零”判断；NaN 与旧版 NumPy 表达式一样会被视作非零值。

## 计算兼容

- 目标高度使用与 `numpy.arange` 一致的起点/终点语义，并用 ties-to-even 规则映射到最近 z 层；映射到同一层的目标会保留第一个并去重。
- 二维谱仅使用前三个分量。每个分量在有效点上减去 float64 均值，无效点置零，再执行 float32 二维 FFT；归一化、三分量累加、float32 落盘和 `fftshift` 顺序与旧版一致。
- 对数色阶仍取所有正谱值 `log10` 后的 5% 与 99.5% 分位数。
- 三维谱从第一个覆盖率与 1.0 `isclose` 的层开始。x 轴采用实数 FFT 的半谱，内部模式按偶/奇数长度赋予 Hermitian 权重，再按 `dk=max(2π/(nΔ))` 汇入球壳；`E(k)`、补偿谱和 Nyquist 定义保持不变。
- 三维常驻工作集约为一个 float32 子体、一个 complex64 半谱和一个 int32 壳层索引，不需要把输入 VTK 完整加载到内存。

生产依赖中存在 `pocketfft_hdronly.h` 时，CMake 自动启用整轴批量 pocketfft：二维频谱一次处理两个轴，三维频谱一次执行多轴 `r2c`，并直接使用 `--workers`。普通 no-vcpkg 检查构建会使用 VTK 内置 kissfft fallback；该路径保持文件和计算契约，但 VTK 若以 double kissfft 构建，会在每一维后显式量化回 complex64。生产性能和最接近 SciPy 的数值应以 pocketfft 路径为准。

## 输出

输出目录和文件名保持为：

- `layers_kxky/kxky_zNNNNm.png`，1364×1188；
- `kxky_overview.png`，4136×`880×行数`；
- 存在 100/200/300/400/500/600 m 目标时，`kxky_overview_100_200_300_400_500_600_largefont.png`，4290×2640；
- `isotropic_spectrum.png`，1892×1276；
- `isotropic_spectrum_plain.png`，1760×1188；
- `kxky_layer_metadata.csv`、`isotropic_spectrum.csv` 和 `run_summary.json`。

CSV 列名、CRLF 行尾、JSON 字段名称/层次、绝对输出路径和控制台进度顺序均保持兼容。原生 PNG 保留 Magma 色阶、面板组织、坐标/标题语义、Nyquist/半 Nyquist 标记、补偿谱和 -5/3 参考线；字体栅格、抗锯齿和 Matplotlib 的约束布局不会逐像素相同。

## 验证结果

生成式 8×6×13、四分量 BINARY VTK 同时覆盖了无效点、ties-to-even 高度、目标去重、quick-test 和完整三分量 3D FFT。与旧脚本通过 `PYTHONDONTWRITEBYTECODE=1 python -B` 得到的 oracle 比较：

- 层记录、覆盖率、有效点数、壳层中心和壳层权重逐值相同；
- pocketfft 路径的非近零 `E(k)` 最大相对差约 `9.8e-8`；严格为零的理论模态会因 FFT 实现/编译器舍入产生约 `1e-15` 的噪声差，因此依赖这些模态的 5% 色阶下限在该合成场上相差约 `0.079` 个 log10 单位；
- JSON 三组字段顺序和全部 PNG 文件名/像素尺寸一致，PNG 均可完整写出；
- 同一 quick-test fixture、单 worker 下，原生程序报告约 1.0 s，旧 Python 脚本报告约 3.9 s；含进程启动的墙钟约为 1.13 s 对 6.04 s。该小型基准用于确认没有性能回退，不代表真实 LES 数据的固定加速比。

VTK fallback 在有效能量上的误差也处于约 `1e-7` 相对量级，但不声明与 SciPy 逐末位一致；它的职责是保证不含 vcpkg 依赖的检查构建仍可运行。
