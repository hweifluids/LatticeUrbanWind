# Python 到原生实现兼容性矩阵

## 审计口径

本表沿用原生重写工作以提交 `8e4cad8ffbb98efb743c183aa363a6c191270373`
建立的迁移基线。该基线通过 `git ls-files '*.py'` 得到 33 个生产、安装辅助和旧测试
Python 文件；下表逐一列出，未把缓存、虚拟环境或依赖包计入。FluidX3D 不在这 33 个
文件的迁移范围内。合并后的仓库将 CULBM 作为活动默认求解器，将 FluidX3D 作为默认
不安装、仅通过 `LUW_ENABLE_FLUIDX3D=ON` 启用的独立冻结求解器；两个求解器均独立于
本表的 Python 到 C++20 迁移清单。

CULBM 另保留四个离线验证/报告 Python 脚本：`tests/compare_backends.py`、
`tests/compare_physical_vtk.py`、`validation/gea/analyze_results.py` 和
`validation/gea/report/build_report.py`。它们由 Python-free 门禁精确 allowlist，
不属于安装器、CLI、GUI、前后处理或求解器启动链，也不计入上述历史 33 文件统计。

状态含义：

- **原生等价**：主要公开契约已有 C++ 实现，并在确定性 fixture 或旧实现 oracle
  上通过了对应验证。它不自动代表所有平台、真实大数据或未列出的边界均已验证。
- **被原生模块吸收**：旧文件是内部模块、安装辅助或测试，其能力已并入新的公共
  原生层，或因不再存在 Python 环境而有意取消独立入口。
- **原生替代（行为调整）**：原生能力保留，但当前产品契约明确改变旧工程行为；
  该差异不是需要恢复的兼容缺口。
- **仍待验证/已知差异**：即使已有历史生产依赖门禁证据，仍存在明确的安全、后端或
  绘图差异，或真实生产规模、跨平台和人工交互路径尚未完整验收。不得据此声称严格
  全量等价。

本文表格中的构建、CTest、GPU 和 oracle 结果是原生重写工作树截至 2026-07-20 保存的
历史证据；本次文档整合没有重新执行这些项目。表中额外限制仍以“尚未验证或差异”列
为准，历史证据不能自动外推为当前合并树在所有平台和真实数据上的完整验收。

## 当前合并后的生产边界

1. CLI、GUI、前处理、编排和后处理均由 C++20 原生实现，生产路径不启动或嵌入 Python。
2. `cfd_solver` 缺失或为 `culbm` 时由 `runluw` 启动 CULBM；只有显式 `fluidx3d`
   且该后端已通过 opt-in 安装时才启动 FluidX3D。CULBM 是活动模块，FluidX3D 是
   默认关闭的独立冻结模块；所选可执行文件缺失时明确失败，不进行求解器回退。
3. SurfaceIBM 只由 CULBM 支持，`ibm_enabler` 默认 `false`；与 FluidX3D 同时启用会在
   求解器启动前被拒绝。
4. 默认完整安装依次运行 `installer/2_build_native.*` 和 `installer/3_compile_cfdcore.*`；
   前者构建原生应用层，后者只构建自包含 CULBM。仅当两个阶段均显式设置
   `LUW_ENABLE_FLUIDX3D=ON` 时才额外安装和构建独立 FluidX3D。

## 原生重写工作树的历史门禁与剩余边界

1. **历史 full-dependency Release 门禁已完成。** 主程序/GUI 构建和 30/30 CTest 当时已通过，
   包括 `luwbc`、`luwcut`、`luwvox`、DEM GeoTIFF 缓存、`buildingscale`、
   `shpinspect`、`shptester` 和 `visdem` 的 full-GDAL/GEOS 路径，以及 Manifold
   3.4.1 boolean union 分支。`native.luwcut` 已实际写出并验证
   38400×38400、JFIF 4800 DPI 的流式 JPEG。
2. **历史安装运行时闭包曾通过全新前缀隔离验证。** 当时通过了 `luw --help`、
   `luw luwbc --help`、`luw luwvox --help`、`luw detect_opencl` 和 GUI offscreen
   runtime self-check；安装版 `shpinspect`/`vtk2nc_new` 的生产依赖验证也属于该历史覆盖。
   当时没有运行现已改造为默认构建 CULBM、可显式附加 FluidX3D 的完整多阶段安装器。
3. **历史门禁未覆盖 CFD。** 上述门禁只覆盖原生 GUI、编排和前后处理层，不能用来
   声称当前 CULBM 或 FluidX3D 已由同一轮测试验证。
4. **GUI 人工交互验收尚不完整。** GUI 当时已进入统一 Release 构建并在 fresh-prefix
   通过 offscreen runtime self-check，但终端输入、错误重输和完整工作流中的全部人工
   交互尚未逐项验收。
5. **图形产物不逐像素相同。** `visluw`、`tke_viz`、`cut_vis`、
   `batch_tke_geo_viz`、`season_average`、`les_spectra`、`buildingscale`、
   `luwcut` 和 `visdem` 使用原生 rasterizer。已验证的文件名、像素尺寸、数据、
   面板和色阶语义与 Matplotlib 的字体、抗锯齿、箭头自动缩放、布局及压缩字节
   是两种不同契约。
6. **真实生产规模的性能证据仍不完整。** 小 fixture 上 `tke_viz` 约快 4.1 倍，
   `les_spectra` 墙钟约为 1.13 s 对 6.04 s；当时的合成核心负载中 `luwvox` k-NN/IDW
   为 GPU/CPU 5.89/437.83 ms、`luwbc` MLS 为 12/188 ms，但小 DEM IDW 为 3/2 ms。
   这些不是完整项目墙钟；其余大 NetCDF、GIS、体素化和多 GiB
   VTK 路径采用流式 I/O、OpenMP/OpenCL 和有界内存设计，但尚未在同一真实生产案例
   上全面证明“不弱于旧 Python”。
7. **保留有意安全差异。** 旧 GIS/绘图脚本会删除 Shapefile sidecar 或覆盖已有输出；
   原生实现按重写约束拒绝覆盖、拒绝不安全 `casename`，不会先删除旧文件。
8. **Python/Numba runtime 准备是明确的移除边界。** `luw accelerator_report`
   已补齐只读 CUDA/OpenCL JSON，`detect_opencl` 补齐严格文本报告；旧 requirements
   import、site-packages wheel 复制/删除、环境变量修改和 Numba cache 只服务已移除的
   Python runtime，原生兼容模式明确返回 `not applicable`，不会伪造或执行这些副作用。
9. **统一前处理设备契约曾通过历史门禁。** deck 使用 `preprocess_use_gpu` 与
   `preprocess_gpu_device`，CLI 使用同名 kebab-case 选项；设备号是 GPU-only 全局
   0-based，公共参数只由 dispatcher 解析，无命令局部旁路。算法只保留
   `idw/kriging`，没有 `kriging_gpu` 兼容值；Kriging CPU/GPU 均最多 16 邻点；
   `luwbc` 和 `luwvox` 的稳定邻域/地形 GPU program 均严格要求 FP64。设备 0/1
   实测均通过，显式 GPU 失败不降级 CPU。详见
   [`native-preprocess-compute.md`](native-preprocess-compute.md)。

## 基线外恢复的历史公开入口

原 README 宣称但审计基线已删除的 `transluw` 不在上述 33 个文件内。原生重写工作从
Git 历史删除提交 `3a57c6b` 的父提交恢复行为基线，并以
[`transluw_command.cpp`](../src/post/transluw_command.cpp) 注册纯 C++20 命令及两端
wrapper。`native.transluw` 已通过；标准 U/RHO+DEM 和 16 源点 k=12 tie fixture
与历史 PyVista/SciPy oracle 的 dataset type、拓扑、array 集合、dtype 和全值一致，
后者 `dem_height` 最大绝对差为 `1.7763568394002505e-15`。旧版会执行任意 pickle
GLOBAL 且覆盖输出；原生 restricted reader 与拒绝覆盖属于明确安全收紧。详细证据见
[`native-transluw.md`](native-transluw.md)。

## 逐文件矩阵

### 核心前处理与公共模块（12）

| # | 原 Python 文件 | 旧入口与主要 I/O | 当前原生映射与验证证据 | 状态 | 尚未验证或差异 |
|---:|---|---|---|---|---|
| 1 | `core/accelerator_runtime.py` | 旧 GUI 内部模块兼 CLI；检查 requirements/import，准备 Numba CUDA wheel 布局，探测 CUDA/OpenCL，输出 JSON | `luw accelerator_report` / wrapper 由 [`accelerator_report_command.cpp`](../src/cli/commands/accelerator_report_command.cpp) 输出兼容顶层 JSON，并保留无模式显示帮助、实际模式仅在 `--json` 下输出的 CLI 分支；[`gui/src/StartupDiagnostics.cpp`](../gui/src/StartupDiagnostics.cpp) 直接探测；`native.accelerator_report` 与 no-SDK 变体通过，本机只读报告含两 CUDA device、两 OpenCL platform/三 device | 被原生模块吸收 | Python identity/checks 固定为空并标记不适用；wheel 发现/复制删除、环境修改、Numba cache/runtime version 有意不复刻；host 字符串不逐字相同，Linux/macOS 未实机验证 |
| 2 | `core/bridge_core/1_buildBC.py` | `luwbc`/GUI；读 deck、CF/WRF NetCDF、DEM SHP/TIF，写六面 CSV、可选 VTI、`dem_grid.pkl`、deck 与进度 | [`src/cli/commands/luwbc_command.cpp`](../src/cli/commands/luwbc_command.cpp)；通用平地 CSV 与旧 oracle 逐字节一致，WRF 与 production GeoTIFF→SHP→DEM 通过；投影/水平/竖直/DEM/MLS exact-bit 有界缓存、CPU 多字段批处理、GPU 双槽融合重采样、DEM IDW/MLS 均已接入，设备 0/1 与 CPU/GPU terrain CSV 字节一致测试通过 | 仍待验证/已知差异 | GPU 必须支持 FP64，否则状态 1 且不降级 CPU；合成 MLS 已验证潜力但真实生产规模端到端未跑；旧 WRF 合法交错网格会因 NaN 失败而原生成功，是明确修复；单文件仍按旧契约取 `Time=0` |
| 3 | `core/bridge_core/2_shpCutter.py` | `luwcut`/GUI；读 deck 与唯一 building SHP，清理、裁剪、union、过滤，写 cut SHP 与可选 JPEG preview | [`src/cli/commands/luwcut_command.cpp`](../src/cli/commands/luwcut_command.cpp)，production full-GDAL `native.luwcut` 已通过，并实际验证 38400×38400、JFIF 4800 DPI 流式 JPEG 及已有预览保留路径 | 仍待验证/已知差异 | 与旧 Python/Shapely 的更广几何差分集仍不完整；旧版删除/覆盖 sidecar，原生拒绝覆盖；绘图不逐像素相同；真实大 Shapefile 性能未全面基准 |
| 4 | `core/bridge_core/3_voxelization.py` | `luwvox`/GUI；读 cut SHP、SHP/TIF/PKL DEM 与 deck/CLI，以 `idw/kriging` 选择算法、以统一两键选择 CPU/GPU，写 STL、投影元数据和进度 | [`src/cli/commands/luwvox_command.cpp`](../src/cli/commands/luwvox_command.cpp) 与 [`luwvox_opencl.cpp`](../src/cli/commands/luwvox_opencl.cpp)；规则 DEM 直接索引、不规则 uniform-bin CSR GPU k-NN、驻留 Kriging/IDW/Gaussian、稳定顺序地形三角网格已在 GPU-only 设备 0/1 和跨分块 fixture 通过；Kriging CPU/GPU 均最多 16 邻点 | 仍待验证/已知差异 | `kriging_gpu` 不接受；GPU program 要求 FP64 且失败不降级；Kriging 为精确中位数仍有 host O(Q×k) 临时内存；建筑变长 extrusion 保留 CPU；真实场吞吐与峰值显存尚未全面对比 |
| 5 | `core/bridge_core/auto_UTM.py` | 内部 UTM zone/EPSG 推断，读取 deck 经纬度范围 | 公式被 `luwbc`、`luwcut`、`luwvox`、`vtk2nc_new`、`buildingscale` 等投影路径吸收 | 被原生模块吸收 | 无同名独立入口；尚无覆盖 dateline、UTM zone 边界和极区的专门旧实现差分集 |
| 6 | `core/bridge_core/dem_tif_to_shp.py` | 内部 helper；按 deck bbox 扩 20%，TIF/TIFF 裁切 polygonize 为 `terrain_db/<casename>.shp`，含覆盖提示 | [`src/cli/commands/dem_tif_cache.cpp`](../src/cli/commands/dem_tif_cache.cpp)，供 `luwbc`/`luwvox` 共用；旧 WGS84、projected/no-nodata、NaN/±Inf oracle 已固化；源块对齐分条、双缓冲 RasterIO 及跨多条带 fixture 随 full-GDAL `native.dem_tif_cache` 通过 | 仍待验证/已知差异 | 原生 staging/no-overwrite、sidecar 保护和路径逃逸拒绝有意不同于旧删除/覆盖行为；完整值/mask 栅格仍由 GDAL MEM 保存以供 polygonize，真实超大 DEM 吞吐和峰值内存尚未基准 |
| 7 | `core/datagen_core/dgPrepare_stlinput.py` | 旧 GUI 直接入口；读 `.luwdg/.luwpf` 与 building STL，建 base、尝试 boolean，写 `<casename>_DG.stl`、`.bak` 和 deck 范围 | `luw dgprepare` / wrapper，由 [`dgprepare_command.cpp`](../src/cli/commands/dgprepare_command.cpp) 实现；production build 已链接 Manifold 3.4.1，`native.validate_and_dgprepare` 中封闭 cube 的真实 boolean union 和非法四面体的 concatenation fallback 均已通过 | 仍待验证/已知差异 | 真实大型 STL 性能未测；等价 boolean 后端可能产生不同三角剖分，因此面数和 STL 字节排列不承诺相同 |
| 8 | `core/deck_io.py` | 内部 deck 解析/修改/渲染；别名、注释、重复键、未知键、UTF-8 容错 | [`src/common/deck_document.cpp`](../src/common/deck_document.cpp)；`native.deck_and_terrain_config` 含 Python golden byte-exact render | 原生等价 | 等价结论限于已覆盖 token、编码和渲染 fixture；所有未知现实 deck 仍需回归样本 |
| 9 | `core/deck_schema.py` | 内部 schema/section/field/alias/bool token | [`src/common/deck_schema.cpp`](../src/common/deck_schema.cpp) 读取同一 [`core/deck_schema.json`](../core/deck_schema.json)，由 deck/config CTest 与 GUI schema 共用 | 原生等价 | schema JSON 今后变更时仍需同步 C++ kind 解析测试 |
| 10 | `core/luw_progress.py` | 内部 `[[LUW_PROGRESS]]` JSON 行协议，0.5% bucket 与 150 ms 节流 | [`src/common/progress.cpp`](../src/common/progress.cpp) 与 GUI [`CommandRunner.cpp`](../gui/src/CommandRunner.cpp) 吸收；工作流和各阶段测试消费同一字段 | 被原生模块吸收 | JSON 语义字段已保留，但原始 key 顺序、单调时钟边界和每一帧字节序列没有独立 golden 比较 |
| 11 | `core/terr_voxel_config.py` | 内部 CLI > deck > default 合并、校验、来源和 warning | 算法配置位于 [`src/common/terrain_voxel_config.cpp`](../src/common/terrain_voxel_config.cpp)，设备配置由 `preprocess_compute_config` 公共层独立管理；算法仅 `idw/kriging`，设备优先级为 CLI > deck > default；公共 CLI 仅由 dispatcher 消费；Kriging CPU/GPU 均为 1..16 邻点 | 原生替代（行为调整） | 统一两键、CLI scope、无局部旁路和算法/设备正交已纳入 30/30 门禁；尚未对大量真实 deck 建立完整回归集 |
| 12 | `core/terr_voxel_gpu.py` | 内部 Numba CUDA ordinary-kriging kernel、warmup、16 邻点限制与 fallback | 原生 OpenCL kernels 位于 [`luwvox_opencl.cpp`](../src/cli/commands/luwvox_opencl.cpp)；FP64 稳定 k-NN、Kriging exact-hit/常值/on-device 奇异系统 IDW、IDW/Gaussian、地形网格、设备 0/1 与跨分块均通过实机测试 | 被原生模块吸收 | OpenCL 后端失败不降级 CPU；CUDA 被 OpenCL 替代，编译提示和末位结果不承诺相同；真实场吞吐/峰值显存尚未与 Numba 比较，AMD/Intel 尚未实测 |

### 独立分析、检查与后处理工具（17）

| # | 原 Python 文件 | 旧入口与主要 I/O | 当前原生映射与验证证据 | 状态 | 尚未验证或差异 |
|---:|---|---|---|---|---|
| 13 | `core/tools_core/batch_tke_geo_viz.py` | 直接脚本；读 LUW-family 配置和批量 legacy VTK，写 geo crop VTK、风/TKE PNG | `luw batch_tke_geo_viz` / wrapper，共用 [`tke_viz_core.cpp`](../src/post/tke_viz_core.cpp)；`native.tke_viz` 中 batch BINARY crop 与旧 oracle 逐字节一致 | 仍待验证/已知差异 | PNG 不逐像素；大型城市批处理未测；原生对 ASCII 的支持属于扩展，ASCII 高精度裁剪不承诺旧脚本理想化结果 |
| 14 | `core/tools_core/buildingscale.py` | 直接脚本；读建筑 SHP，计算短边和 area×height 加权 PDF/CDF，写默认/`--out` 图像 | `luw buildingscale` / wrapper，GDAL/OGR 数值核心与 3000×900 PNG/JPEG writer；Python 数值/CLI edge oracle 和 production full-GDAL Shapefile/PNG/JPEG 测试均已通过 | 仍待验证/已知差异 | PDF/SVG/TIFF/WebP 等旧 Matplotlib 后缀不支持；拒绝覆盖；字体和像素不同；真实超大建筑数据集性能未全面基准 |
| 15 | `core/tools_core/cdfInspect.py` | `cdfinspect`/GUI；直接 NC 或 deck→`wind_bc` 首文件，向控制台输出 dataset、范围、分位数、变量/全局属性 | [`inspect_commands.cpp`](../src/cli/commands/inspect_commands.cpp)，`native.inspect_commands`；真实示例的维度与 min/q05/q95/max 精确对齐 | 仍待验证/已知差异 | xarray `Dataset.__repr__` 的空格、折行和值折叠没有逐字符复刻；大型/特殊 NetCDF dtype 与 CF 编码仍需更多样本 |
| 16 | `core/tools_core/cleanluw.py` | `cleanluw`；递归删除 case `proj_temp` 内文件但保留目录，不改 deck | [`cleanluw_command.cpp`](../src/cli/commands/cleanluw_command.cpp)，`native.cleanluw` 已验证边界、输出与退出码 | 原生等价 | 本审计没有调用清理命令；它仍是必须由用户显式请求的破坏性入口 |
| 17 | `core/tools_core/cut_vis.py` | 直接脚本；单 VTK local/geo 裁剪、九层风/TKE 图与可选 crop VTK | `luw cut_vis` / wrapper，共用 `tke_viz_core`；`native.tke_viz` 已通过 | 仍待验证/已知差异 | 与 `tke_viz` 相同的 Matplotlib/config.py/ASCII 精度差异；真实大型 VTK 未人工视觉验收 |
| 18 | `core/tools_core/les_spectra.py` | 直接脚本；内存映射 BINARY VTK，2D/3D FFT，写 CSV/JSON 与多张频谱 PNG | `luw les_spectra` / wrapper，pocketfft 生产后端及 kissfft fallback；`native.les_spectra` 和旧 oracle 已运行 | 仍待验证/已知差异 | 非零 `E(k)` 最大相对差约 `9.8e-8`，理论零模态噪声约 `1e-15` 可使测试色限差 `0.079 log10`；PNG 不逐像素；大型 3D 实场未跑 |
| 19 | `core/tools_core/makeluw.py` | `makeluw`/GUI；固定六阶段子进程、实时转发、日志；旧版即使子步骤失败也最终成功 | [`makeluw.cpp`](../src/cli/commands/makeluw.cpp) 保留阶段与日志，并向计算子步骤传统一 CLI override；继续收集后续诊断但任一子步骤失败最终返回 1，退出码与 override 透传已纳入 30/30 门禁 | 原生替代（行为调整） | 真实案例六阶段 end-to-end 及安装后终端/GUI 人工全流程尚未执行 |
| 20 | `core/tools_core/prerunValidate.py` | `luwval`/GUI；读 STL/CSV 范围，补 deck 字段，写 validation/high-order/flux 修正 | [`validate_command.cpp`](../src/cli/commands/validate_command.cpp)，`native.validate_and_dgprepare` 及实际兼容命令已通过 | 原生等价 | 只验证确定性小 STL/CSV；尚无真实超大 STL/CSV 吞吐基准 |
| 21 | `core/tools_core/season_average.py` | 直接脚本；读 profile/windrose/多方向 VTK，权重合成与可选重采样，写 VTK、summary、五张 PNG | `luw season_average` / wrapper；`native.season_average` 的非重采样和 spacing=2 VTK/summary 与旧 oracle 均逐字节一致 | 仍待验证/已知差异 | PNG 字体/抗锯齿/压缩不同；>512 MiB mmap 分支和几十 GiB 真实案例未跑 |
| 22 | `core/tools_core/shpInspect.py` | `shpinspect`/GUI；直接 SHP 或 deck→`building_db` 首文件，控制台输出 CRS JSON、WGS84 bounds、列/dtype/前五行 | [`inspect_commands.cpp`](../src/cli/commands/inspect_commands.cpp)，production full-GDAL `native.inspect_commands` 已通过；隔离安装版 `shpinspect` 已读取 EPSG:3857 输入、输出 EPSG:4326 范围，并确认输入 hash 不变 | 仍待验证/已知差异 | GeoPandas dataframe/dtype/repr 不逐字符复刻；更多复杂字段类型和真实超大 Shapefile 尚需样本 |
| 23 | `core/tools_core/shpTester.py` | 直接脚本；只读逐 feature 判退化原因并按 `Counter.most_common()` 输出 | `luw shptester` / wrapper，GDAL/OGR + GEOS C；旧 Polygon/Point/mixed GeoJSON console oracle 和 production full-GDAL `native.shptester` 均已通过，并验证只读输入契约 | 仍待验证/已知差异 | GEOS 版本升级可能改变 invalid reason 文案，虽当前抽查 reason 与旧实现一致；真实超大图层性能未全面基准 |
| 24 | `core/tools_core/tke_viz.py` | 直接脚本；功能与 `cut_vis.py` 同源，写九层风/TKE、诊断 PNG 与可选 crop VTK | `luw tke_viz` / wrapper；`native.tke_viz`、Python oracle、PNG 解码和 BINARY crop SHA 已通过，小 fixture 原生约快 4.1 倍 | 仍待验证/已知差异 | 不执行任意 `config.py`，只解析普通字面量；PNG 字体/quiver/layout 不逐像素；大型实场未测 |
| 25 | `core/tools_core/visdem.py` | 直接脚本；读首 band GeoTIFF，bilinear 重投影 EPSG:4326，统计，写 2D 四面板和 3D PNG，并阻塞 `plt.show()` | `luw visdem` / wrapper，GDAL 原生重投影与静态 rasterizer；production Release `native.visdem` 已通过 5×6→4×7、4770×3570/2955×2498 RGBA、控制台、只读与拒绝覆盖验证 | 仍待验证/已知差异 | 不复刻两个交互 Matplotlib 窗口；字体、抗锯齿和压缩字节不同；旧 `savefig` 覆盖而原生拒绝；真实大 DEM 尚未人工视觉验收和全面基准 |
| 26 | `core/tools_core/visluw.py` | `visluw`/GUI；读 deck/VTK，交互裁剪，写 NPZ、逐层/分布 PNG、可选 lon/lat NetCDF | `luw visluw` / wrapper；`native.visluw` 对 8 dtype×ASCII/BINARY×SCALARS/VECTORS/FIELD 的 NPZ/NetCDF/稳定 INFO 全值 oracle 16/16 一致 | 仍待验证/已知差异 | PNG 不逐像素；旧 NumPy 整数溢出 RuntimeWarning 未伪造；交互错误重输未全覆盖，真实大体积与 ZIP64 >4 GiB 未测 |
| 27 | `core/tools_core/vtk2nc.py` | `vtk2nc`/GUI；发现结果 VTK，选风字段，按旧固定坐标/50 m pedestal 写压缩 NetCDF | `luw vtk2nc` / wrapper，shared legacy VTK reader；`native.vtk_post` 的 ASCII/BINARY 输出与旧 oracle 核心 8 变量 exact equal | 原生等价 | reader 范围仍是 `STRUCTURED_POINTS` 和已列字段类型；超大真实 ASCII/BINARY 性能未测 |
| 28 | `core/tools_core/vtk2nc_new.py` | 直接脚本；批量 VTK，旋转 UTM→完整规则 lon/lat 网格，SciPy cubic，写多变量 NetCDF | `luw vtk2nc_new` / wrapper；`native.vtk2nc_new` 对 19 变量 schema、dtype、filters、chunks、attrs 和全部数组 bit-exact，控制台 23/23 行一致；隔离安装版也已成功生成 NetCDF | 原生等价 | oracle 和安装自检均为合成小网格；真实生产大网格性能与不同 PROJ 版本仍需验收 |
| 29 | `core/tools_core/vtk_avg_to_utm_asl_nc.py` | 直接脚本；单 case/多 case、Range/summary 地形 ASL，avg VTK 风旋转，写逐层压缩 UTM/ASL NetCDF | `luw vtk_avg_to_utm_asl_nc` / wrapper；`native.vtk_post` 锁定四城市发现、Range/summary/显式值优先级、limit/skip/overwrite、命名和 schema；5 组旧/新 NetCDF 的全部 47 数值 exact equal，详见 [`native-vtk-avg-to-utm-asl-nc.md`](native-vtk-avg-to-utm-asl-nc.md) | 原生等价（有版本化元数据差异） | pyproj PROJ 9.5.1 与 VTK bundled PROJ 8.1 的 WKT scope 文本不同但 CRS 语义相等；原生额外接受 ASCII VTK；真实大文件性能尚未验收 |

### 安装辅助（2）

| # | 原 Python 文件 | 旧入口与主要 I/O | 当前原生映射与验证证据 | 状态 | 尚未验证或差异 |
|---:|---|---|---|---|---|
| 30 | `installer/2_get-pip.py` | vendored `get-pip` bootstrap；仅为旧 venv 安装 pip/setuptools/wheel | 原生安装链使用 CMake、系统 C++ toolchain 和 vcpkg；[`installer/2_build_native.cmd`](../installer/2_build_native.cmd) / [`.sh`](../installer/2_build_native.sh) 不创建 Python 环境；历史 fresh-prefix 曾通过三个 CLI help、`detect_opencl` 与 GUI offscreen self-check | 被原生模块吸收 | 不保留 `get-pip` 的独立联网 CLI，这是去 Python 后有意移除；完整安装还必须继续运行 `installer/3_compile_cfdcore.*` 构建默认 CULBM；FluidX3D 仅在 `LUW_ENABLE_FLUIDX3D=ON` 时附加构建 |
| 31 | `installer/3_detect_opencl.py` | 旧安装诊断；pyopencl 枚举 platform/device/type/driver/CU/clock/global memory，特定退出码 | `luw detect_opencl` / POSIX、PowerShell wrapper 与纯原生 `installer/3_detect_opencl.cmd/.sh`，由 [`detect_opencl_command.cpp`](../src/cli/commands/detect_opencl_command.cpp) 调用 OpenCL C API；两项诊断测试及 installer 退出码透传测试通过，本机两 platform/三 device 成功报告与旧 oracle 的 27 行、1020 字符一致 | 原生等价 | 无可用 platform 的实际主机与 Linux 尚未实机运行；状态 1/2/3、固定提示及格式分支由原生单测锁定 |

### 旧 Python 测试（2）

| # | 原 Python 文件 | 旧入口与主要 I/O | 当前原生映射与验证证据 | 状态 | 尚未验证或差异 |
|---:|---|---|---|---|---|
| 32 | `tests/test_deck_io.py` | unittest：bool、alias、section 顺序、未知键和 terrain 字段 round-trip | 测试用例被 [`tests/native/test_deck.cpp`](../tests/native/test_deck.cpp) 吸收，并增加 golden bytes、UTF-8 和数字 token 边界 | 被原生模块吸收 | 后续 schema 扩展仍需同步新增 fixture；不再保留 Python unittest 入口 |
| 33 | `tests/test_terr_voxel_config.py` | unittest：CLI/deck/default 优先级、旧 GPU choice、Inferred alias、非法值 warning | 算法测试由 [`tests/native/test_terrain_voxel_config.cpp`](../tests/native/test_terrain_voxel_config.cpp) 吸收；统一 CPU/GPU 两键、dispatcher 唯一解析入口、Kriging 1..16 和 OpenCL 测试已纳入 30/30 门禁 | 被原生模块吸收 | 不保留旧 `kriging_gpu` choice；更广真实 deck 和参数组合仍需持续回归 |

## 公共边界与验证结论

- 新仓库的生产入口统一为 `luw <command>`，并为已注册命令安装 POSIX 与
  PowerShell 兼容 wrapper；旧内部 Python import 关系不再构成运行时依赖。
- 审计基线之外的历史公开 `transluw` 已恢复为原生命令；它不改变 33 文件的状态
  统计，验证与安全差异单列于上文及 `docs/native-transluw.md`。
- `native.python_free_tree` 检查仓库自有 `bin/cmake/core/gui/installer/src/tests`
  中的 `.py/.pyc/.pyo`，精确排除上文四个 CULBM 离线脚本，并检查高可信 Python
  embedding/launcher API。它是“没有 Python 生产运行时路径”的门禁，不是上述数值、
  控制台或图形等价性的替代证据。
- CFD 由 `runluw` 根据 deck 选择并转发参数：缺失/`culbm` 选择活动默认 CULBM，
  显式 `fluidx3d` 只选择已经通过 `LUW_ENABLE_FLUIDX3D=ON` 安装的独立冻结后端；
  SurfaceIBM 仅可与 CULBM 配合且默认关闭。
  本矩阵没有把两个 CFD 源码树算入历史 33 个 Python 文件。
- 2026-07-20 的 full-dependency Release、30/30 CTest、Python-free gate、GPU 设备 0/1
  定向运行及 fresh-prefix 隔离自检均是重写工作树的历史证据，不是本次合并重新执行的结果。
- 历史结果表明原生重写层曾通过所列本机门禁，但不等同于当前合并树在所有平台、全部真实
  大数据、GUI 人工交互、图形像素和性能上均已严格等价；仍应保留已记录的兼容差异和
  规模化验收边界。
