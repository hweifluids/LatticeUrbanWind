# 原生 GPU 加速路线图

## 目标与边界

本计划评估原生前后处理模块增加 OpenCL GPU 后端的实际价值。目标是缩短真实案例的
端到端墙钟时间，同时保留 CPU 权威实现、输入输出格式、计算步骤、提示与错误边界。

CFD 求解器独立于本计划：CULBM 是活动的默认 CUDA 模块，FluidX3D 是默认不安装、仅以
`LUW_ENABLE_FLUIDX3D=ON` 显式启用的独立冻结 OpenCL 模块。本计划只讨论原生前后处理
加速，不修改、编译或运行任一 CFD 求解器。
“可以写成 kernel”不等于“值得上 GPU”：文件解析、压缩、PROJ/GDAL/GEOS 拓扑、变长
网格输出等任务通常受 I/O、分支发散或兼容语义限制，必须先做真实 profile。

## 统一后端契约

1. CPU 继续作为默认和数值 oracle。所有前处理计算核心统一读取
   `preprocess_use_gpu` 与 `preprocess_gpu_device`；CLI 使用
   `--preprocess-use-gpu BOOL` 与 `--preprocess-gpu-device N`，不再提出或实现第二套
   设备选择参数。
2. `preprocess_gpu_device` 是 GPU-only、全局从 `0` 开始的设备号：按 platform 顺序，
   再按各 platform 的 `CL_DEVICE_TYPE_GPU` 顺序扁平化。CPU OpenCL device 不计入。
3. 用户显式选择 GPU 时，命令必须在创建相关输出前预检 GPU type、设备号、必要扩展、
   全局显存、单次最大分配、context/queue 和全部 kernel。任一步失败均写 stderr、
   返回非零，禁止静默 CPU fallback。
4. 本计划不引入隐式 `auto`。未来即使另行批准 `auto`，也不能改变显式 GPU 的
   fail-fast 契约。
5. 公共 GPU CLI 参数只能由统一 dispatcher 解析，命令 handler 不得增加局部别名、
   重复解析或设备选择旁路。各模块可保留适合自身 kernel 的 program/buffer cache，
   但必须遵守同一 GPU-only 编号、能力检查和 fail-fast 语义。
6. CPU 与 GPU 的算法选择必须正交。地形算法只允许 `idw` 与 `kriging`；
   `kriging_gpu` 不保留为配置值或兼容别名。GPU 内部为奇异方程执行 IDW 等数值稳定分支可以保留，
   但必须仍在设备上完成，并与“后端降级”明确区分。
7. GPU 失败前不得留下部分 CSV/STL/VTK/NetCDF/PNG；需要长时间写出的命令继续采用
   staging + 排他提交。

本轮 full-dependency Release 主程序与 GUI 已构建成功，串行 CTest `30/30` 通过；
OpenCL GPU-only 设备 0 和 1 均已通过 `luwvox` FP64 k-NN、Kriging/IDW/Gaussian、
规则地形三角网格，以及 `luwbc` FP64 融合重采样、DEM IDW 和 terrain-first MLS
定向验证。fresh-prefix `build-production-install-20260720_071053` 也已通过三个 CLI help、
`detect_opencl` 和 GUI offscreen self-check。真实大型数据端到端性能、峰值显存和 NVIDIA
以外厂商仍待验证。详细调用契约见
[`native-preprocess-compute.md`](native-preprocess-compute.md)。

## 候选总览

优先级含义：P1 为高潜力、应优先原型；P2 需真实 profile 后实施；P3 仅在共享 kernel
已经存在且端到端占比足够时接入；“不建议”表示先做 CPU/I/O 优化。

| 模块/阶段 | 当前主要计算 | GPU 潜力 | 结论 |
|---|---|---:|---|
| `luwvox` kriging / IDW 邻点搜索 | CPU KD-tree k-NN 曾是主要串行/主机瓶颈 | 已实现，P1 | 完整规则 DEM 直接索引；带缺口/扰动及不规则 DEM 使用稳定 uniform-bin CSR + GPU ring 扩展，按 `(distance²,index)` 严格停止；设备 0/1 和跨 65536 查询分块通过 |
| `luwvox` Kriging/IDW + Gaussian smoothing | 每网格点局部预测；两次可分离卷积 | 已实现，P1 | DEM/elevation/raw surface 驻留设备，预测直接连接 Gaussian；第二张 lookup surface 复用 host raw grid；真实大场性能与显存仍待测 |
| `luwvox` 地形规则网格三角面生成 | 每 cell 生成固定三角面 | 已实现，P2 | 按 CPU 原顺序以最多 65536 triangles 分块生成/回读并接入生产 STL；最终主机 STL 写出仍不可省略 |
| `luwvox` 建筑拉伸 | Earcut + 变长侧面/顶底三角面 | 低 | 已逐建筑 OpenMP；分支和变长输出不适合首批 GPU 化 |
| `luwbc` WRF/CF U/V/W/T 重采样 | 水平重心权重、垂向插值，多字段多面 | 已实现，P1 | CPU 规划权重并精确缓存；CPU 融合字段，GPU 使用双 queue/slot 的 FP64 水平/竖直批处理；设备 0/1 与至少三分块测试通过 |
| `luwbc` DEM IDW / terrain-first MLS | k=12/24 邻域、最多 6×6 小系统 | 已实现，P2 | VTK host 邻点顺序保持权威，GPU FP64 执行 IDW/Wendland-C2 MLS；CPU/GPU terrain CSV 字节一致，32,785 目标合成 MLS 为 188/12 ms |
| `transluw --woDEM` | 每 VTK 点 CPU KD-tree k=12 IDW | 中高，P2 | 先做规则 DEM 快路；要求 FP64、固定距离/index tie 顺序 |
| `les_spectra` | 2D/3D FFT、能量与球壳归并 | 很高，P1 | 首批 pilot；先 GPU FFT、CPU 归并，再评估融合 reduction |
| `vtk2nc_new` | float64 spline prefilter + 每点 16-tap cubic | 高，P1 | 行/列 IIR 并行和采样 kernel；设备必须支持可靠 FP64 |
| `season_average` | 多方向五场加权累加、三线性重采样 | 高，P2 | 先重采样，再做 VRAM-resident/分块累加；固定方向累加顺序 |
| `tke_viz` / `cut_vis` / batch | 九层 bilinear、旋转、模长、颜色映射 | 中低，P3 | 可复用规则网格 kernel，但 PNG/VTK 写出常是端到端瓶颈 |
| `visluw` | 裁剪/转置、模长、最大值、直方图 | 中低，P3 | 只在大 VTK profile 证明计算占主导后接入共享 kernel |
| `visdem` / preview raster | 重采样、颜色映射、像素生成 | 中低，P3 | GDAL warp、JPEG/PNG 压缩和读回可能吃掉收益 |
| `luwcut` | GEOS MakeValid/Intersection/UnaryUnion | 很低/高风险 | 不建议 GPU；拓扑、要素顺序和变长几何难保持兼容 |
| `buildingscale` | GEOS union/最小旋转矩形、PDF/CDF | 低 | 先改进 CPU union/逐建筑并行；绘图 GPU 化收益有限 |
| `dgprepare` | Manifold boolean、STL I/O | 很低/高风险 | 不建议 OpenCL 重写；先评估启用 Manifold/TBB 并行 |
| `dem_tif_cache` | GDAL RasterIO/rasterize/polygonize | 很低 | 已改为源块对齐分条和双缓冲 RasterIO，显著减少裁切区完整数组；I/O/GDAL 主导，不建立形式化 GPU 开关 |
| `vtk_avg_to_utm_asl_nc` | 少量逐点旋转后 NetCDF 压缩 | 很低 | 保留 CPU；传输成本高于每点计算量 |
| 旧 `vtk2nc` | AoS→SoA 拆分、NetCDF 压缩 | 很低 | 保留 CPU/I/O 流水 |
| inspect、validate、deck、progress、wrapper | 解析、诊断、编排或文件扫描 | 无明显价值 | 不增加 GPU 后端 |

## 先于 GPU 的直接优化

- `luwvox` 已复用一次原始插值并分别执行两种平滑，避免重复邻域查询和预测；真实数据
  上的端到端提速仍需基准确认。
- `luwbc` 已用 exact-bit、deck-scope、设备隔离的有界 LRU 复用投影、水平、竖直、
  DEM 邻域和 terrain-first MLS 计划；现有输入契约每次只读取 NetCDF `Time=0`，因此
  不擅自引入没有文件命名契约的单文件多时次输出，而是优化连续时间戳文件调用。
- `dem_tif_cache` 和 `luwvox` 直接 GeoTIFF 读取已按约 8 MiB/源块分条；前者在读取与
  mask/value 写入间双缓冲，后者使用轴对齐矩形快筛或 prepared geometry。
- `luwcut`/`buildingscale` 先优化 GEOS union 策略和要素级 CPU 并行。
- `dgprepare` 先验证 Manifold 的 TBB 构建；当前依赖配置未启用并行 boolean。
- NetCDF/PNG/JPEG/NPZ 密集模块先做分块、双缓冲、压缩线程和 I/O/计算流水 profile。

## 分阶段实施

### 阶段 0：基线与共享运行时

- 为候选命令记录 read、host preprocess、H2D、kernel、D2H、compress/write、峰值 RAM/VRAM。
- 以统一两键建立共享 GPU-only preflight、全局 0-based GPU selector、program cache、
  buffer planner 和错误测试。
- 固化三档真实 fixture：能完全驻留的小型、接近 break-even 的中型、需要分块的大型。
- 同时记录冷启动与同进程 warm run；首次 kernel 编译单列，不从结果中隐藏。

### 阶段 1：`luwvox` 完整规则网格流水

1. 已消除两次原始插值重复，复用一次原始网格后分别平滑。
2. DEM 点/elevation、邻点分块、预测与 Gaussian 缓冲已接成设备驻留流水；完整规则
   DEM 使用直接索引，不规则 DEM 使用 uniform-bin CSR GPU k-NN，不存在 O(Q×N) CPU
   fallback。
3. OpenCL Kriging、IDW、separable Gaussian 和稳定顺序地形网格已通过 GPU 设备 0/1
   定向数值与跨分块测试；规则 TIF/PKL DEM 的真实大场端到端收益仍待验证。
4. 显式 GPU 遇到不支持的能力、布局或显存不足会报错，不转 CPU。为保留 Kriging
   精确 range 中位数，设备缓冲分块有界但主机 positive-distance 临时数据仍为 O(Q×k)。

### 阶段 2：FFT pilot

- 为 `les_spectra` 选用可部署、跨厂商的 OpenCL FFT 实现；先替换 2D/3D transform，
  保留 CPU 均值、mask 和球壳归并以缩小数值风险。
- 锁定 Parseval、频率轴、能量谱、CSV/JSON/PNG schema 与现有容差，然后再判断是否把
  能量和 reduction 融合进设备端。

### 阶段 3：高价值重采样

- `luwbc`：CPU 生成并跨连续时间戳精确缓存几何/权重计划；GPU 以 FP64 执行融合
  U/V/W/T 水平/垂向重采样、DEM IDW 和 terrain-first MLS。双 queue/slot 让 H2D、
  kernel、D2H 分块重叠；设备 0/1、CPU/GPU CSV、缓存失效和错误路径已通过，真实大场
  端到端收益仍待验证。
- `vtk2nc_new`：要求 FP64，按行/列并行 spline prefilter，并缓存目标映射做 cubic sampling。
- `season_average`：先接共享 trilinear kernel，再做保持方向顺序的设备常驻加权累加。

### 阶段 4：条件候选

- `transluw --woDEM` 仅为规则 DEM 做 FP64 k-NN/IDW 原型；不规则输入在显式 GPU 下报
  unsupported。必须逐项复刻 `(distance²,index)` tie 顺序。
- TKE/visluw/visdem 只复用已经成熟的 bilinear、magnitude、histogram、colormap kernel；
  若 PNG/NetCDF/NPZ 写出仍占主导，则不发布 GPU 开关。

## 数值与性能验收门槛

- CPU 现有测试和 byte-exact 输出不得放宽；GPU 使用独立 oracle/tolerance，不反向降低 CPU 基线。
- `luwbc` 为保持 CPU 的 double 权重/分数语义必须要求 GPU FP64，并固定运算次序、禁用
  fast-math；不满足能力时显式 GPU 直接失败。DEM IDW 与 MLS 也遵循该要求。
- `luwvox` 的预测/Gaussian 最终值仍为 float，但 k-NN 使用 FP64
  `(distance²,source index)` 稳定排序、地形网格使用 double 坐标，所以整个 GPU program
  也明确要求 FP64；Kriging 的 CPU/GPU 邻点上限统一为 16。跨厂商 `exp/pow` 不宣称逐位一致。
- 每个 GPU 后端至少覆盖：无 OpenCL 构建、无 platform、只有 CPU device、kernel 编译失败、
  FP64 缺失、显存/最大分配不足、运行失败、用户取消和成功路径。
- 在可获得设备上分别记录 NVIDIA、AMD、Intel；不能获得的厂商明确标为未验证。
- 发布门槛建议为真实大型 fixture 端到端至少 1.5×，必须包含 H2D/D2H 和输出写入；
  仅 kernel 快而端到端未达门槛的实现不进入默认交付。
- 默认 CPU 路径不得因新增 GPU 抽象产生可测回退；小输入的 GPU break-even 尺寸必须记录。
- 当前 Release 构建、串行 `30/30` CTest、GPU 设备 0/1 定向测试和 fresh-prefix 隔离
  自检已通过。合成核心负载中 `luwvox` k-NN/IDW 为 5.89 ms 对 CPU 暴力参考
  437.83 ms，`luwbc` MLS 为 12 ms 对 188 ms；小 DEM IDW 则为 3 ms 对 2 ms，说明
  break-even 不能忽略。AMD/Intel、真实大型数据端到端吞吐与峰值显存仍是证据缺口。

## 明确不做

- 不修改任何 CFD 求解器核心。
- 不为追求“有 GPU 选项”重写 GEOS、GDAL polygonize、Manifold boolean、PROJ、NetCDF/zlib。
- 不静默选择不同算法、不在显式 GPU 失败时返回 CPU 结果、不以 kernel-only 时间宣传加速。
