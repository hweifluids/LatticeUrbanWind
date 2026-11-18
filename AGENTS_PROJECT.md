# AGENTS_PROJECT.md

## 项目级持续说明

本文件记录对后续 agents 有持续价值的项目知识。若后续任务发现新的、可复用的构建方式、运行方式、排查结论、约定或长期注意事项，应更新本文件；不要只把这类信息写进一次性的 `codexlog/`。

## 原生产品架构与求解器选择

生产 CLI、前处理、编排、后处理和 GUI 均为 C++20 原生实现，入口统一为
`luw <command>`；不得恢复 Python 解释器、虚拟环境、Python 子进程或生产 `.py`
入口。历史 Python 行为与原生映射保存在 `docs/native-compatibility-matrix.md`。
CULBM 的四个 `tests/` / `validation/` 离线分析脚本是精确许可的科研验证工具，
不属于生产运行链，也不得据此让安装器或 GUI 重新依赖 Python。

规范 deck 键为 `cfd_solver = "culbm"|"fluidx3d"`。缺失该键时也必须选择 CULBM；
FluidX3D 只允许显式 opt-in。`ibm_enabler` 缺失时为 `false`，SurfaceIBM 只受 CULBM
支持；`cfd_solver=fluidx3d` 与 `ibm_enabler=true` 是配置错误，不得静默忽略 IBM、
自动关闭 IBM 或回退到另一个求解器。GUI Solve 必须通过 `luw runluw <deck>`，不能
自行直启某个 solver，以免 CLI/GUI 选择策略分叉。

原生根 CMake 工程不无条件 `add_subdirectory(CULBM)`，因为 CULBM 配置要求 CUDA
Toolkit。原生前后处理可独立构建；安装阶段再由 `installer/3_compile_cfdcore.*`
构建默认 CULBM。只有调用方显式设置 `LUW_ENABLE_FLUIDX3D=ON` 时，根安装规则、CFD
编译脚本和运行自检才暴露、构建或要求 FluidX3D。FluidX3D 是冻结兼容模块，CULBM
是活动模块；二者是彼此独立的构建与运行包。

## 代码变更版本日期

每次修改会影响运行行为的代码后，应同步更新对应求解器启动横幅中的版本日期。位置：

```text
core/cfd_core/FluidX3D/src/setup.cpp
core/cfd_core/CULBM/src/solver/setup.cpp
```

当前格式分别为：

```text
Version - v3.5-YYMMDD
Version - CULBM-v3.5-YYMMDD-<variant>
```

例如 2026-04-24 对应 `Version - v3.5-260424`。更新后需要重新编译对应平台的 CFD core；本地 Windows 与 HPC/Linux 是不同 binary，必须分别编译。

## C++ 跨平台类型注意

`core/cfd_core/CULBM/src/solver/utilities.hpp` 中 `ulong` 定义为 `uint64_t`。在 Windows/MinGW 上它可能等价于 `unsigned long long`，而在 Linux GCC 上常等价于 `unsigned long`。因此修改 `ulong` 相关计算时，尤其是传给 `std::min` / `std::max` 的参数，不要混用 `1ull`、`0ull` 等 `unsigned long long` 字面量；应显式转换为 `ulong`，例如：

```cpp
std::max(current_t + (ulong)1u, avg_start_t);
std::min(avg_start_t - (ulong)1u, total_steps);
```

否则 Linux GCC 可能报 `deduced conflicting types`，而本机 Windows 语法检查不一定能暴露该问题。

## CFD Core 编译

Windows 下优先使用仓库脚本编译默认 CULBM：

```powershell
cmd /c "installer\3_compile_cfdcore.cmd < NUL"
```

说明：

- 命令应从仓库根目录执行。
- `< NUL` 用于避免脚本末尾 `pause` 在非交互环境中挂起。
- CULBM 由自身 CMake 工程构建，要求 CUDA Toolkit 12+；输出为
  `core\cfd_core\CULBM\bin\CULBM.exe`。
- 默认流程不会读取或构建 FluidX3D。需要同时构建该独立可选后端时，先执行
  `set LUW_ENABLE_FLUIDX3D=ON`；其阶段会通过 `vswhere.exe` 或常见安装路径定位 MSBuild。
- 当前本机可用路径示例：`C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe`。
- 项目文件请求 `PlatformToolset=v143`；脚本会在必要时切换到已安装的新工具集，例如 `v145`。
- 仅当显式启用时，FluidX3D 输出为
  `core\cfd_core\FluidX3D\bin\FluidX3D.exe`。其构建可能出现 `lodepng.cpp` 的
  `C4267`、`C4334` 等既有 warning；只要最终为 `0 Error(s)` 且
  `Build succeeded.`，该可选后端编译视为通过。

以下 MinGW `g++` 快速语法检查仅适用于显式启用的 FluidX3D 后端，不能替代其 MSVC
Release 链接构建，也不是 CULBM 验证步骤：

```powershell
g++ -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -fsyntax-only src/setup.cpp
g++ -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -fsyntax-only src/info.cpp
```

以上命令需在 `core\cfd_core\FluidX3D` 目录下执行。

## 长时间发展后的瞬态窗口

CULBM deck 的 `culbm_unsteady_start_step`（缺省 `0`）只控制周期 raw-u 写盘的
开始步数，输出仍对齐 `unsteady_output` 的整数倍；最终常规结果不受影响。
大型 FTLE 工作应先估算窗口文件总量，再将此门槛设到所需窗口起点，避免把
不需要的发展阶段全部输出。该选项同时覆盖均匀与垂直嵌套主循环，嵌套时单位是粗层步数。

SurfaceIBM 会清除支持带内部分固体标记以执行强制耦合，故启用 IBM 后最终 TYPE_S
不能独立代表连续 STL 的物理内外。轨迹和 FTLE 的物理流体掩码应按真实几何另行建立；
不要将 IBM 内侧支持节点当成建筑外部流体，也不要用膨胀固体的方法删除整个近壁首层。
当前 `src/device/particle_advection_kernels.cu` 的粒子位置与插值判定仍直接读取
LBM `flags` 的 `TYPE_S` 位，因此启用 SurfaceIBM 时必须另外核查原生粒子相对真实
STL 的位置，不能把 `status=active` 当作真实几何外部的证明。NUS 2 m case 已发现
距实体表面约一个网格以内的穿透。若以保存的瞬态速度另行推进修正粒子，应明确实际
覆盖的时间窗，不能用冻结场补齐未保存的早期历史。

FTLE 在出域或碰壁后保留最后共同有效积分时长时，各种子使用的时间窗并不相同。
全域高百分位可能主要由短驻留种子占据：应同时报告完整时间窗比例、有效时长分布，
并区分原始标量与显示平滑后的超水平集统计。不能仅凭全域 q95 等值面、外边界封口
或近壁有限值覆盖率，宣称已经识别出长时 material barriers。60/120 s 对照只衡量
积分时长敏感性；快照加密和积分步长减半是另两类独立检查。

关闭 IBM 后，轨迹与 FTLE 应重新使用最终 TYPE_S 体素固体作为 CFD 的实际边界；
连续 STL 相对体素的差异另作几何离散审计。长粒子历史可用原生在线推进保存，
不必为全部粒龄归档三维速度。扩展速度支持域时，在线粒子还要分配全网格的
previous_velocity（额外 12 Bytes/cell），不得仅用 LBM 核心显存估算能否装下。
需要固定时长主 FTLE 时，应只从完整共同积分时长的种子计算主图百分位，并把
短驻留标量单独保留；扩域不能代替实际完整时长覆盖率检查。

profile 模式在 `setup.cpp::interpolate_profile_cubic` 使用非单调限制的 Hermite 插值，
随后采样成 0.1 m 的当地 DEM AGL 查表。对在 z=0 补零、z=10 m 开始变化很小的稀疏
年均剖面，不能默认 native 插值仍单调：NUS 案例实测会在约 13.3 m 超调 0.289 m/s。
科研 case 可在写入 profile.dat 前，按原有统计拟合方法加密到 0.1 m，并验证实际
float32 查表值；这不改变统计剖面，能避免第二次稀疏三次插值制造人工低空急流。

## season_average 后处理

原生命令 `luw season_average` 可直接指定平均 VTK 输入目录并按目标 spacing 重采样输出：

```powershell
luw season_average <case>\conf.luwpf --vtk-dir <case>\RESULTS\crop\cropped_vtk_stage2 --output-spacing 10 --skip-figures
```

说明：

- `--vtk-dir` 可指向绝对路径或相对 case 目录的路径，用于选择修正后的 VTK 源目录。
- `--output-spacing 10` 会在 x/y/z 三个方向重采样；为保持输入物理范围，实际写出的 `SPACING` 可能是接近 10 m 的值，例如由 19.9991264 m 输入得到 9.9995632 m。
- 修正后 VTK 若使用独立标量 `u/v/w/tke`，脚本会合成为 `u_avg` vector，并同时输出 `u/v/w/vm/tke` 标量。
- `--skip-figures` 只生成 VTK 和 summary，适合几十 GiB 级别的大体积输出。

历史四城含 sigma 的结果来自仓库外
`D:/averagedField/New0518_average/synthesize_windrose_season_average.py`。
2026-09-12 核查时，原生 `src/post/season_average_command.cpp` 尚未写出
`sigma_u/sigma_v/sigma_w/sigma_speed`，不能把同名命令当作该外部版本的等价替代。
该历史链先在完整水平层执行 FFT/去趋势/行条纹修正，再地理裁剪、裁剪后修正，
最后对各方向重采样并累计一阶/二阶矩；先在粗网格计算 sigma 再插值不等价。
历史 `batch_tke_geo_viz.py` 的 `crop_cell_size` 用于制图，VTK 导出保持源间距。
外部经验修正不是生产求解器的一部分，也不能无条件推广为新城市的无损清理。

## HPC/Linux CFD Core 编译

本地默认 Windows 编译只会更新本地 CULBM；显式 opt-in 时才另行更新本地 FluidX3D。
两者都不会更新 HPC 上的 Linux 可执行文件。若作业脚本运行的是 HPC 路径，例如：

```text
/work/home/.../LatticeUrbanWind-HPC0424/core/cfd_core/CULBM/bin/CULBM
```

则必须在该 HPC 仓库副本中同步源码后重新编译 Linux 版 CULBM。生产多节点构建必须
开启强制 MPI 检查，避免 MPI 开发环境未加载时静默生成单进程二进制：

```sh
cd /work/home/.../LatticeUrbanWind-HPC0424
cmake -S core/cfd_core/CULBM -B core/cfd_core/CULBM/build-hpc \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCULBM_ENABLE_MPI=ON -DCULBM_REQUIRE_MPI=ON
cmake --build core/cfd_core/CULBM/build-hpc --target CULBM --parallel
```

如需同时编译默认 CULBM 与显式可选的 FluidX3D，也可使用仓库安装脚本并要求 CULBM
必须链接 MPI：

```sh
cd /work/home/.../LatticeUrbanWind-HPC0424
LUW_CULBM_REQUIRE_MPI=ON LUW_ENABLE_FLUIDX3D=ON sh installer/3_compile_cfdcore.sh
```

诊断提示：如果运行日志仍出现旧字符串 `Avg benchmark | mean-field Steps/s = ...` 或 `ETA model | normal Steps/s = dynamic, mean-field Steps/s = ...`，说明正在使用旧版求解器。包含 timing-plan 改动的新版本应在 `SOLVER START` 前打印 `LBM TIMING PLAN`，并出现 `sample extra = ... s`、`Preprocess time`、`Estimated total` 等输出。

Timing plan 注意事项：

- `Info::print_initialize()` 会把 `runtime_lbm_timestep_smooth` 初始化为 `1.0` 秒/步。第一轮 LBM case 如果没有历史 normal Steps/s，不能直接用这个值估算 warm-up，否则会出现严重偏长的 `1s/step` 初始 ETA。
- 当前实现会在长循环前先运行少量 normal warm-up 步作为 `Normal benchmark`，再打印 `LBM TIMING PLAN`。这些 probe 步是实际计算步，会计入总步数，不会额外增加步数。
- 如果确实没有 normal 样本，timing plan 应显示 `pending normal-speed sample`，不要打印看似确定的 fallback ETA。

## 山地 LUW 风边界顶高

原生 `src/cli/commands/luwbc_command.cpp` 中风场垂直层按 AGL 解释。对地形起伏较大的 LUW 标准模式案例，固定 CFD 顶高不能只用 `base_height + wind_top_agl`，否则最高地形可能高于顶边界。应使用全域最高地形作为基准：

```text
z_top = ground_z_max + min(wind_top_agl, z_limit_agl)
```

其中 `ground_z_max = base_height + max(relative_dem)`。林芝 20 m 案例中 DEM 相对起伏约 1.39 km，原始风场最高层为 995 m AGL；若不按最高地形抬升顶面，边界 CSV 顶面会低于高山区域。

## W 分量边界与通量修正

原生 `src/cli/commands/luwbc_command.cpp` 会读取输入 NetCDF 中的 `w`/`W` 变量；若不存在则回退为 0。当前边界 CSV 的 `top/south/north/west/east` patch 会保留采样得到的 `w` 分量，不再强制写成 0；`bottom` patch 仍用于地形/地面信息，不作为常规速度入口面。

`core/cfd_core/CULBM/src/solver/fluxcorrection.cpp` 的通量修正按外边界法向速度计算净通量：`x/y` 侧面使用对应水平法向分量，顶面 `Zp` 使用 `u.z`，因此带 W 的顶边界会进入通量平衡；侧边界上的 W 是切向分量，不应计入侧面法向通量。日志中 `Flux correction | net by face ... Zp(w)=...` 可用于确认 W 是否实际参与顶面通量修正。

## VK 入口面选择与 x 向波纹排查

剖面工况中 VK turbulent inlet 默认业务语义由 `vk_inlet_inflow_only` 控制，且 top 面不作为默认 VK inlet：

- `vk_inlet_inflow_only=true`：选择除 downstream/outlet 以外的三个侧面。
- `vk_inlet_inflow_only=false`：选择 west/east/south/north 四个侧面。
- `vk_inlet_face_mode` 保留为诊断覆盖项；正常业务配置不需要写该字段，默认 `AUTO_SIDES` 会按上面规则解析。

北京 debug case 的 0 度测试显示：

- `TARGET_INFLOW`：只对 downstream 对侧的真实入口面施加 VK，x 向 k=4 人工波纹显著下降。
- `EXCLUDE_DOWNSTREAM`：排除 downstream 但保留其余四面（0 度时 west/east/north/top）会复现原始 x 向 k=4 大尺度波纹；top 面点数约 190 万，远大于侧面点数，是高风险强相干扰动来源。
- `EXCLUDE_DOWNSTREAM_SIDES`：排除 downstream 且排除 top，仅保留非 downstream 的三个侧面（0 度时 west/east/north）。北京 debug case 中该模式未复现 k=4 大波纹，k=4 与 `TARGET_INFLOW` 同量级，说明 top 面是决定性触发面，侧面多入口本身不是主因。
- `ALL_SIDES`：四个侧面都施加 VK，但排除 top。北京 debug case 中该模式仍未复现原始 x 向 k=4 大波纹，说明 downstream 侧面加入 VK 也不是主因；top 面参与 VK 才是当前波纹的决定性触发条件。
- `ALL_SELECTED` 且 `enable_top_sponge=false`：禁用 top sponge 后，west/east/south/north/top 五面全部施加 VK，仍会复现 x 向 k=4 大波纹，且 k=4 接近原始全边界量级。因此波纹不是 top sponge/grid extension 本身造成的，而是 top 面 VK 扰动与域内流场耦合造成的。
- 默认业务语义测试：`vk_inlet_inflow_only=true` 的三侧面和 `false` 的四侧面都未复现原始 x 向 k=4 大波纹；四侧面的 k=4 比三侧面通常略高，但仍远低于 top VK 参与时的量级。

相关源码位置为 `core/cfd_core/CULBM/src/solver/setup.cpp` 的 `VkInletFaceMode` 和 `VonKarmanInletUpdater`。新增或调试 VK 入口策略时，应优先确认日志中的 `VK inlet target`、`VK inlet face` 和频谱诊断，不要只看切片图。

## Coriolis 与高空出口 u 分量

北京 debug case 中 `angle=0` 表示来流从 `+y/north` 向 `-y/south`，主流速度主要体现在 `v` 分量。若 `coriolis_term=true`，北半球向南气流会向右偏转，在当前坐标中表现为负 `u` 分量；该偏转在高空、出口附近沿程累积后最明显。

四侧面默认 VK case 的 0 度对照显示：

- `coriolis_term=true`：1200 m AGL、south/outlet 0-5 km 内 `u_mean≈-1.13 m/s`，但 `uvw_mean≈7.49 m/s`，因此图上 `u` 的低值区不是总风速低速区。
- `coriolis_term=false`：同一区域 `u_mean≈-0.02 m/s`，`uvw_mean≈7.51 m/s`；高空出口的 `u` 负值区基本消失。

因此遇到出口高空 `u` 图上的明显低值区时，应优先检查 `v/uvw` 和 `coriolis_term`，不要直接解释为出口低速或 VK 波纹。

以下排查只适用于显式设置 `LUW_ENABLE_FLUIDX3D=ON` 的独立 FluidX3D 后端。其
HPC/Linux 链接阶段如果出现类似：

```text
temp/fluxcorrection.o: file not recognized: 不可识别的文件格式
collect2: 错误：ld 返回 1
```

通常表示 `core/cfd_core/FluidX3D/temp/` 中残留了其他平台或旧编译器生成的 `.o` 文件。`make.sh` 的多核 `make` 分支应先执行 `make clean`，再执行目标平台编译；如果在旧脚本或手工编译环境中遇到该问题，可先运行：

```sh
cd /work/home/.../LatticeUrbanWind-HPC0424/core/cfd_core/FluidX3D
rm -rf temp bin/FluidX3D
./make.sh
```

## CULBM CUDA 包

CUDA 求解器包名、CMake 工程名和主可执行文件名统一为 `CULBM`：

```text
core/cfd_core/CULBM/
core/cfd_core/CULBM/bin/CULBM.exe
```

内部公开编译宏与环境变量使用 `CULBM_*`，C++ 命名空间使用 `culbm`。不要再新增
旧包名、旧宏前缀或旧命名空间标识。

CULBM 的当前工程边界是独立、自包含的 CUDA 求解器包，不得在架构、构建或运行说明中
把它表述为继承、托管于或依赖 FluidX3D。主机端源码固定在
`core/cfd_core/CULBM/src/solver/` 与 `core/cfd_core/CULBM/src/`；可直接维护的原生 CUDA
唯一真源位于 `core/cfd_core/CULBM/src/device/`。CMake 只在 build 目录机械生成嵌入
片段，运行时由 NVRTC 直接编译 CUDA C++，不得恢复 OpenCL 转译或从
`core/cfd_core/FluidX3D` 等相邻源码树复制、staging、补丁生成的流程。

CULBM 由 CULBM/LUW 团队独立开发和维护，并受其包内独立
`core/cfd_core/CULBM/LICENSE.md` 约束，许可人统一写作 `CULBM developers`，不得列写
个人姓名。仓库根 `LICENSE.md` 是 LUW 许可；包含、使用或分发 CULBM 的 LUW 同时受
CULBM 许可约束，涉及 CULBM 的权利或义务冲突时以 CULBM 许可为准。不得在架构、
构建、运行、来源或许可说明中把 CULBM 表述为源于、继承、改编自或托管于 FluidX3D，
也不得在 CULBM 包内恢复 FluidX3D 来源声明或许可
副本。FluidX3D 只作为独立、冻结、显式启用的可选后端存在；启用、使用或分发该后端
时，其包内许可再作为额外约束适用。

默认 MPI-enabled 构建在普通直接启动时不会初始化 MPI，仍走单进程本地任意多卡
路径；检测到 `PMI_RANK`、`PMIX_RANK`、`OMPI_COMM_WORLD_RANK` 等 launcher rank
标记时可自动初始化。MPI 标准不保证这些环境标记，因此生产 MPI 命令应显式设置
`CULBM_MPI=1`。LBM 对象在构造时固定并缓存 distributed 状态，不要在构造后才
初始化 MPI。MPI 运行时要求 `MPI_THREAD_MULTIPLE`；求解通信使用独立
communicator 和进程内串行锁。

生产多节点构建应传 `-DCULBM_REQUIRE_MPI=ON`。非 MPI CULBM 二进制检测到
`CULBM_MPI=1` 或 Slurm/OpenMPI/PMI 等 launcher rank 标记时必须立即失败，不能让多个
rank 各自退化为独立的本地多卡整场计算。

生产 CULBM 不定义 `GRAPHICS`，不创建图形快照/halo，也不写出离线图像帧；LodePNG
源码、PNG 纹理解码和帧文件导出均不应恢复到默认数值路径。保留的交互显示源码不是
受支持的 headless 生产功能，若未来显式启用，仍须保证其通信与活动物理场隔离。

历史 staging 补丁仅作为本地回归材料归档在
`core/cfd_core/CULBM/test-output/legacy-staging-pipeline/`，不属于活动源码或发布
内容。后续 CULBM 主机端功能修改应直接落在 `src/solver/`、`src/` 或 `CMakeLists.txt`；
设备代码应直接修改 `src/device/*.cu` / `*.cuh`，不得手工编辑 build 目录生成的 `.inc`。

### CULBM halo 优化与性能回归约束

默认启用 `TEMPERATURE` 时，CULBM 把相同方向、相同时间奇偶相位的 `fi` 和
`gi` halo 合并成一次 raw-bit 批量通信。联合 buffer 每边界格至少按
`(transfers + temperature_transfers) * sizeof(fpxx)` 分配；温度 D3Q7 每面传
1 个 DDF，温度 D3Q19 每面传 5 个 DDF。修改速度集或精度时必须同步检查实际分配、
VRAM 估算和 host/MPI 传输长度。

halo 优化必须保持 `fpxx` 原始位模式、Esoteric-Pull 奇偶相位、p/m 方向映射以及
x/y/z 阶段顺序。不得用浮点重算、插值或并行方向交换替代原始复制；当前方向顺序
还承担边/角数据传播。主机中转日志必须区分锁页缓冲实际成功的异步路径和 pageable
同步保底，不能仅根据预选路由声称使用 pinned memory。

性能结论应固定二进制哈希、算例、GPU/驱动、卡数、分解和实际传输路径。稳定 A/B
需严格串行并交错运行，排除预热区间，报告运行级样本分布，同时逐位比较最终字段；
不要只用 16 步内置 `normal Steps/s` 宣称优化收益。2026-07-29 的本机基准为
`202×200×25`、10,001 步、2 × RTX 2080 Ti、无 P2P 的 pinned-host 回退：
冻结版 `E0B330DE6803...` 为 1622.0 Steps/s，批处理版 `CB4BEB3B02C9...` 为
2991.0 Steps/s，匹配中位数提升 84.40%，4/4 `u/rho` payload 逐位一致。该数字
不得外推到 P2P/NVLink、多节点 MPI、其他 GPU 或不同网格规模。

Windows 可用以下方式做独立配置与验证：

```powershell
cd core/cfd_core/CULBM
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCULBM_ENABLE_MPI=OFF
cmake --build build --target CULBM --parallel
ctest --test-dir build --output-on-failure
```

### CULBM 温度速度集与边界

CULBM 的动量核心仍由 `defines.hpp` 编译为 D3Q19；温度是独立的标量分布，deck
可用 `culbm_temperature_lattice = D3Q7|D3Q19` 在同一可执行文件中运行时选择，
默认 D3Q7。进程级选择必须发生在构造任何 LBM 对象之前。D3Q7 的
`cs_T^2=1/4`、`tau_T=0.5+4*alpha_lbm`；D3Q19 的
`cs_T^2=1/3`、`tau_T=0.5+3*alpha_lbm`。不得恢复旧的
`tau_T=0.5+2*alpha_lbm`。

温度 DDF 数、显存/带宽估算、NVRTC define 与 halo 传输数都必须从同一运行时选择
取得，不能再硬编码 7 或单个标量 halo。FP16C 下 `gi+T` 分别为
18 Byte/cell（D3Q7）和 42 Byte/cell（D3Q19）。

标量固壁使用显式 link-wise 边界：裸 `TYPE_S` 是绝热反弹，
`TYPE_S|TYPE_T` 是定温 anti-bounce-back。由于 Esoteric-Pull 在奇偶步把出射
DDF 存在不同槽位，边界必须按当前 `t` 从正确槽位取流体侧出射值；直接使用固体
初始化 DDF 会在首轮污染流体均值。非固体 `TYPE_T` 仍为 on-node 定温。

Guo forcing 下的宏观物理速度为 `sum(c_i f_i)/rho + F/(2 rho)`。温度平衡态必须在
总力组装和半力速度修正之后生成。`TYPE_E` 中保存的速度是规定的物理速度，因此其
初始化及主碰撞输出都必须使用未经 SRT/TRT 缩放的原始 Guo 源项，写成
`f_eq(rho,u_bc)-S(u_bc,F_total)/2`；不能直接写 `f_eq`，也不能使用已乘松弛修正的
源项。当前 `TYPE_E` 仍是 full-equilibrium 边界，不应描述为 Zou–He 或二阶非平衡
外推边界。

温度输运与浮力反馈相互独立：CSV 有温度列时，即使 `buoyancy=false` 也应继续被动
输运。Boussinesq 重力使用独立于一般体积力的向量；生产 setup 设标准向下重力，
垂直嵌套 ratio 为 `r` 时格子重力缩放为粗层的 `1/r`。相关生产 GPU 回归为
`CULBM.TemperatureAnalyticGPU` 和 `CULBM.TemperatureHaloMultiGPU`。

生产 setup 将 `buoyancy` 解释为请求而非无条件激活：只有常规 `.luw` 工况实际采用
有效的 `SurfData_*.csv` 温度列时才能打开 Boussinesq 反馈。请求为 true 但没有温度
边界时（包括当前 `.luwdg` / `.luwpf` 模式），CULBM 与 FluidX3D 都必须自动回退，
把有效 `beta` 置零；CULBM 还须把独立浮力重力向量置零，并输出 request、active 与
warning。下游零梯度温度出口及 293.15 K 内部参考值都不是温度输入边界。显式
`buoyancy=false` 不得阻止已有 CSV 温度作为被动标量输运。

2026-09-01 的绵阳涪城 5 m、均匀 6 m/s、22.5°、南侧开口单因素复算确认：保持
可执行文件、几何、SGS、入口/出口、50000 步和 200 样本统计窗不变，仅把默认
`buoyancy=true` 改为 `false` 后，全域 400 m AGL 平均 TKE 从 113.775 降至
0.038 m²/s²（-99.97%），平均垂直速度从 -5.305 变为 +0.012 m/s；400 m 的
99.3% 配对有效点 TKE 降低。关闭后的存储 TKE 与独立末时刻偏离能量比值仍约 0.988，
排除统计器被关闭。该结果说明默认热浮力反馈是此代表工况高 TKE/强下沉的主导来源。
缺少经验证的温度边界、参考温度和目标大气稳定度时，生产 setup 现在会自动关闭
浮力；中性城市风 deck 仍建议显式使用 `buoyancy=false` 以记录建模意图。恢复热浮力
前应输出并闭合验证温度场与浮力功。该结论尚未在年均剖面驱动和其他风向重复，不能
直接外推到所有 case 或真实大气。

默认生产定义 `UPDATE_FIELDS`；显式用 `CULBM_DISABLE_UPDATE_FIELDS` 构建的手动宏观场
重建仍必须与 `stream_collide` 使用同一总力和顺序，包括 BUFFER_NUDGING、TOP_SPONGE、
顶部温度松弛后的浮力以及 Guo 半力速度。`TYPE_E` 保存的是规定物理速度，在该路径也
不得施加半力偏移。专用回归为 `CULBM.ManualFieldsForcingGPU`。

`SurfData_*.csv` 使用命名 `T` 列时，每个实际采用的样本都必须包含可完整解析、有限且
严格大于 0 K 的温度；不得把空值或非法文本经 `atof` 静默变成 0 K，也不得回落到
`SamplePoint` 的默认温度。错误必须在构造 LBM/GPU 分配前带文件和行号终止。完全没有
`T` 列的旧速度边界仍须兼容；新文件应使用命名表头，避免旧式 7 列格式中 `T/patch`
启发式判断的歧义。

即使自定义构建关闭 `UPDATE_FIELDS`，手动 `update_fields` kernel 也必须与主
`stream_collide` 使用同一套宏观物理速度定义：重构全部 Coriolis、buffer nudging、
top sponge、逐点外力与温度浮力后，仅对普通流体加 `F/(2 rho)`；`TYPE_E` 始终保留
规定的物理 Dirichlet 速度。顶部标量 sponge 必须先更新 `T`，再用该温度计算浮力。
对应非默认路径回归为 `CULBM.ManualFieldsForcingGPU`。

### CULBM SGS LES

生产 CULBM 的 Smagorinsky-Lilly 配置必须在构造任何 `LBM` 实例前通过
`set_subgrid_configuration()` 固定，并由 `device_defines()` 作为 NVRTC 常量注入；
不得把逐格 SGS 计算或温度涡扩散率移到 CPU。规范 deck 键为
`culbm_sgs_enable`、`culbm_sgs_cs`（或互斥的 `culbm_sgs_cs_squared`）、
`culbm_sgs_scalar_coupling`、`culbm_sgs_turbulent_prandtl`、
`culbm_base_kinematic_viscosity_m2_s`、`culbm_base_thermal_diffusivity_m2_s` 和
`culbm_sgs_diagnostics_interval`。传统 `C_s` 与 `C_s^2` 不得混用；GEA 式 (13)
的 `0.453989945` 是 `C_s^2`，对应传统 `C_s=0.673787760`。该算例优先写
`culbm_sgs_cs_squared=0.453989945`，不得误写成 `culbm_sgs_cs=0.454`。

温度 SGS 使用与动量碰撞同一点的涡黏度，闭合为
`alpha_t=nu_t/Pr_t`；D3Q7/D3Q19 必须分别使用其 `cs_T^2=1/4`、`1/3` 计算局部
温度松弛率。基础 `nu` 与基础 `alpha` 是两个独立 SI 参数，不得为了复现 AV75 而在
SGS 模式中把二者重新绑成同一个硬编码值。垂直嵌套各层保持相同无量纲 `C_s/Pr_t`，
基础格子输运系数仍按各层 `dx/dt` 规则缩放。

由非平衡二阶矩计算 SGS 应变时必须包含 Guo 力矩修正
`Pi_neq,corr=Pi_neq+(uF+Fu)/2`，其中 `u` 是加过 `F/(2rho)` 的物理速度，`F` 是浮力、
Coriolis、nudging、sponge、逐点力与全局力汇总后的力密度。主碰撞和诊断复算必须使用
同一修正；不得把均匀体力产生的领先阶动量源误判为剪切应变。

生产 FP32 碰撞若把请求的基础输运换算成 `tau<=0.5`，不得静默以零耗散继续运行。
当前 setup 对 `nu/alpha` 使用同一个最小放大系数，直到动量和所选温度格子的实际 FP32
`tau` 都严格大于 `0.5`；这样保持请求的基础 Prandtl 比，并在启动日志中同时报告请求值、
有效 SI/LBM 值、共同倍数和告警。数据集模式每次改变参考速度/时间尺度后都必须重新求解
该倍数。此修正只是 FP32 可表示性下限，不应在报告中仍称为原始分子输运系数。

`nu_t` 平均/峰值诊断必须来自 GPU 所用的同一局部 SGS 公式；输出应明确是瞬时空间
统计而非时间平均。诊断关闭或非采样步不能增加逐格 CPU 工作。性能验收必须用相同
D3Q19/D3Q19 网格、步数、卡数与输出设置做 SGS off/on 稳态 A/B，并报告
`steps/s_on / steps/s_off`；不得只比较包含 NVRTC 编译或文件写出的总墙钟时间。

TRT 动量碰撞保持标准 magic parameter `Lambda=3/16`，kernel 与主机诊断应共享同一
常量，不得为掩盖不稳定而静默 cap `omega-`。启动信息打印 `tau+/-`、`omega+/-` 和
碰撞本征幅值；`|1-omega|>0.98` 只表示每次碰撞不足 2% 的幅值衰减并触发
case-dependent 告警，不是稳定性证明。`CULBM.TemperatureAnalyticGPU.TRT` 只锁定短时
Guo/温度/边界实现；在 `tau+`、`tau_T` 都逼近 0.5 的 GEA 长时算例中 TRT 仍可能失稳，
不能以短时回归宣称长期可靠，也不能用 TRT 控制组替换默认 SRT 的物理验证。

### CULBM CUDA 高度嵌套网格

嵌套 SurfaceIBM 对每层裁剪 STL 表面积分片，使用本层格距、时间步和归一化物理行支撑。
单卡与同一 CUDA primary context 内的相邻层可在设备上完成双向面交换。
`CULBM.NestedSurfaceIBMGPU` 检查跨两个接口的斜向建筑面积、虚拟行零力、
欧拉力与壁面反力。双卡可运行 `CULBM-nested-surface-ibm-test --dual`。
平地使用压缩固体行的半格反弹，固体定温壁继续沿用热边界处理。

`cuda_vertical_nesting_agl` 只由 CULBM 解析，值是压缩单层地基顶面的固定水平 SI
高度（m），不是随 DEM 变化的 local AGL：`[h]` 表示地面至 `h` 使用粗尺度的 1/2；
`[h0,h1]` 表示依次使用 1/4、1/2、1 倍粗尺度。计划必须分别满足每层 SI 厚度整格、
至少两个物理格，以及每层 `Nx/Ny/Nz(含 ghost)` 可被 `Dx/Dy/Dz` 整除；这不是 GPU
数量上限。

嵌套层的顺序在 plan 中为地面到天空、在 scheduler 中为天空到地面。每个内部接口
只能做严格四槽 4→1 算术均值与粗到细常量复制；不要改成高阶/时间插值，固体槽不应
被 open-slot 重新归一化。当前生产 hook 保留完整双向 `NestedMacroPlaneBridge::exchange`
和原有调用顺序。曾验证过“缓存 aperture、只做方向性传输”的缩减调度在 1 步一致但
32 步不再 payload 字节一致，因此不得将其替换进生产路径，除非完成长期等价验证。

几何必须先在公共最细 CUDA raster 上直接体素化，再对每一层从该最细逻辑体素直接
多数表决；半数与半数以上固体均为固体。不要递归地从粗层再粗化。公共最细 z=0 只有
一个压缩地基层；STL 基础板下方的生成实体需移除，profile DEM 也应在公共最细层补入。
顶层 sponge 仅属于最上层，写出/平均/probe 必须裁掉 sponge 延伸层。

接口面传输的安全优化位于 `src/cuda.hpp` 的
`Memory::read_from_device_2d_pitched` / `write_to_device_2d_pitched`：它以 CUDA
Driver `CUDA_MEMCPY2D` 复制一个局部 z 面，替代按 y 行循环的调用。2026-07-29 在
200×200 两层、32 步、单 RTX 2080 Ti 上，与逐行传输的 `u/rho` VTK payload 字节一致，
日志求解时间 17 s→1 s；三层 2 GPU + VK 和 top-sponge smoke 也成功。该性能数字不可
外推到其他硬件、网格、P2P 或 MPI。修改该 DMA 路径后，至少重新执行上述 byte-level
单卡对照及多卡功能 smoke。

## STL 垂直零点约定

不要把不同前处理模式的 STL z=0 语义混用：

- 普通 `.luw` 全流程走原生 `luw makeluw -> luwvox`（实现位于
  `src/cli/commands/luwvox_command.cpp`）。该路径构造的
  基础板为 `z=0 .. base_height`；无 DEM 时地面是 `z=base_height`，有 DEM 时本地地面是
  `base_height + relative_dem(x,y)`。DEM 的绝对海拔会先减去最低值，插值网格也会再次
  归一到本域最低值为 0。因此普通模式 STL 的 `z=0` 是基础板底面，不是地面。
- `.luwdg` 与 `.luwpf` 的 GUI 全流程走原生 `luw dgprepare`（实现位于
  `src/cli/commands/dgprepare_command.cpp`）。它在原始
  网格最低面下方增加厚度为 `base_height` 的基础板，再整体平移到 STL 最低 z 为
  `-base_height`；原始网格最低面因而落在 `z=0`。这一路径中，`z<0` 才是人工基础板。
- 排查 CULBM 的 VTK 高程时，必须先根据实际生成器确认上述模式。普通 raw/avg VTK 还会
  保留压缩后的固体地基层；嵌套输出则会裁掉该层，不能把数组第 0 层直接解释成地面面。

## DEM 最低海拔与输入范围

`src/cli/commands/dem_tif_cache.cpp::expandedBounds` 在读取 GeoTIFF 时，将
`cut_lon_manual`、`cut_lat_manual` 向每侧各扩展原跨度的 20%；
`luwvox_command.cpp::loadDemRaster` 的直接栅格路径也使用这一范围。因此，
输入 DEM 的 `Datum offset applied` 可能由原始 cut 范围外的最低点决定；查询
指定区域最低海拔时，应分别报告原始范围和扩展范围，不能混用。

2026-11-04 检查本机 `database/dem.tif`：EPSG:4326、1 角秒、Int16，声明
`NoData=0`。LUW 排除数据集 NoData 和 `-9999`，保留有效负高程；不能将该文件的
0 值当成有效海面最低点。高程按 LUW 约定以米解释，文件本身没有 band unit 或垂直
基准标签。最低点经纬度应标明是像元中心；并列最低像元应全部保留。此文件的静态
高程不受 deck 中 `datetime` 影响；后续插值、局部归零和人工基础板高度另行处理。

## CULBM 物理坐标导出

CULBM 的 `culbm_output_global_vtk` / `culbm_output_global_netcdf` /
`culbm_output_global_vtkhdf` 统一使用 STL 的米制 x/y，并用
`stl_min_z + base_height` 重基准 z；这保证普通 `.luw` 与
`.luwdg`/`.luwpf` 均以“去除人工基础板后的最低地面 datum”为 `z=0`。结构化结果裁掉
z<0 的地基层，但不应根据固体 flag 再删除建筑或地形体素。

`culbm_output_agl_vtk` / `culbm_output_agl_netcdf` 必须读取
`proj_temp/interpolated_dem.csv`，按 DEM 与 STL 的 x/y 边界做仿射对齐、连续双线性采样，
并仅导出 `z > terrain(x,y)` 的点；导出坐标是 `z_agl=z_global-terrain(x,y)`。缺失 DEM
时必须报错，不得回退为平地。五种物理产品均只在最终时步写出，以免无意把非稳态输出
的存储量成倍放大。内置 NetCDF 写出器是 CDF-2，不依赖外部 NetCDF 库，单变量约受
4 GiB 上限约束。

`culbm_output_global_vtkhdf` 是第 5 个、仅全局坐标的结构化产品，写入
`RESULTS/vtk/*.vtkhdf` 的标准 `VTKHDF/ImageData` 布局。它是可选的 HDF5 CMake 依赖：
`CULBM_ENABLE_VTKHDF=ON` 时自动查找 HDF5；无 HDF5 的构建仍支持前四种产品，但若 deck
请求 VTKHDF 必须在求解前报出明确的重建提示。当前 writer 使用 128×128×1 HDF5 chunks，
并在可用时使用无损 shuffle+deflate level 4；不要把它改写为显式 `StructuredGrid`
点坐标，因为那会失去全局均匀网格的主要空间优势。Windows 动态链接 HDF5 时，运行环境
还必须让 `hdf5.dll` 及其依赖位于 `PATH`；vcpkg 的 `installed/x64-windows/bin` 是常见来源。

GB 级格式基准可通过仅在显式请求时构建的
`CULBM_BUILD_PHYSICAL_OUTPUT_BENCHMARKS=ON` 目标
`CULBM-physical-output-gib-benchmark` 复现。它按索引即时生成确定性的多尺度伪湍流字段，
不需要在 RAM 中复制完整场；默认 `512×512×384`、无温度的原始 point payload 为
1,711,276,032 B（1.59 GiB）。2026-07-30 本机 HDF5/Deflate 实测：legacy VTK
1,711,276,347 B / 6.17 s，CDF-2 NetCDF 2,013,278,476 B / 8.44 s，VTKHDF
1,253,377,829 B / 42.33 s；VTKHDF 比 legacy 小 26.76%、比 NetCDF 小 37.74%，但写入
耗时约为 legacy 的 6.86 倍。此字段刻意保留高频宽带噪声，因此该压缩率可作为高熵场的
保守参考，不能替代真实算例的存储或性能结论。基准会生成数 GiB 临时文件，勿纳入版本控制。

## CULBM GEA 大气基准验证

`core/cfd_core/CULBM/validation/gea/` 保存 arXiv:2302.04836 中 Rising thermal
bubble 与 Density current 的可复现验证流程。D3Q19 加密一级可通过
`run_fine_grid_d3q19.ps1` 运行；脚本会锁定 `build/CULBM-gea-validation.exe`
的 SHA-256，并拒绝复用非空输出目录。

加密网格时固定物理 `dt` 会随 `h` 减半而约翻倍格子 Mach 数，不能把该结果直接解释为
纯空间收敛。至少应同时检查声学尺度 `dt ∝ h` 和扩散尺度 `dt ∝ h²`；分析脚本分别输出
`fine_grid_scaling_comparison.csv` 与 `time_step_convergence.csv`。论文没有
`h=7.8125 m` 热泡或 `h=12.5 m` 密度流的同网格表值，因此这些点只用于 CULBM 内部
加密趋势，正式跨代码相对误差仍只能在论文提供的共同网格上计算，不得以最近论文网格
伪装成同网格参考。

### GEA 密度流的闭合模型对照

密度流云图必须按相同稳定化/闭合模型比较。CULBM 主验证使用全域常数
`mu_a=75 Pa s` 映射的 AV75-like 模型，并关闭 SGS；它应首先与论文 Figure 7/8
的 GEA AV75 结果做结构对照。论文 Figure 9/13 的 `h=12.5 m` 云图分别属于
Smagorinsky 与 kEqn LES，不能把它们当作 CULBM AV75 的同模型小尺度验收基准。
论文 Figure 12 中 Smagorinsky `h=12.5 m` 在 900 s 的空间平均涡黏度仅约
`1.4-1.5 Pa s`，远低于 AV75 的 `75 Pa s`，所以 LES 图出现更多小尺度是预期的。

2026-08-01 的 h=5 只读诊断还确认：报告使用的 5 m 到 10 m 严格 2x2 显示平均
在 900 s 保留约 99.96% 的 100 m 高通 RMS 和 99.88% 的梯度 RMS，不是云图平滑的
主因。详细可复现材料位于
`core/cfd_core/CULBM/test-output/density-current-dualgpu-h5-validation-20260801_031341/diagnostics/turbulence-structure-20260801_163826/`。

2026-08-01 的最终 force-aware D3Q19/D3Q19 常系数 Smagorinsky SGS 诊断运行
（h=12.5 m、dt=0.05 s、900 s）得到近地前缘 14982.571 m，相对论文同闭合
GEA Smagorinsky 中点低 6.435%，且全程峰值 Ma=0.280843；因此它确认 GPU SGS、
标量耦合、force-aware 应力和性能约束均工作，但不能标为最终弱可压物理校验通过。
统一 25 m 网格的梯度、二维涡量、短波能量与冷池周长代理都显示 SGS 解析出更多
短尺度变化，但 AV75/SGS 对照同时改变基础输运、dt 和闭合，不能作单变量因果结论。
正式报告保存在
`output/reports/CULBM_density_current_sgs_comparison_report_20260801_211208.docx` 与
`output/pdf/CULBM_density_current_sgs_comparison_report_20260801_211208.pdf`；下一项首要
控制是同网格 dt=0.025 s 的时间步敏感性运行。

融合对照中必须进一步区分黏性定义：GEA AV75 是常动力黏度 `75 Pa s`，CULBM
AV75-like deck 使用 `viscosity_mode=mu/rho_ref`，即固定常运动黏度
`nu=64.575 m2/s`；两者是名义映射而非相同本构。论文 Figure 7 的 AV75 h=25 m
四时刻原图是
`density-current-dualgpu-h5-validation-20260801_031341/diagnostics/turbulence-structure-20260801_163826/av75-page13-000..003.png`；
`reference_rasters/gea-004..007.png` 实际属于 Figure 10 的 Smagorinsky h=25 m，
不得标为 AV75。CULBM AV75-like h=5/h=12.5 在共同 25 m 网格的 900 s 指标为
`r=0.975214`、冷池 IoU `0.902600`、L1 `0.042708 K`，但 Linf 达 `8.789068 K`，
所以只能说明整体形态相近，不能声称逐点一致。对应融合报告保存在
`output/reports/CULBM_density_current_fused_closure_report_20260801.docx` 与
`output/pdf/CULBM_density_current_fused_closure_report_20260801.pdf`。

Ref.[29]（Marras、Nazarov、Giraldo，JCP 301 (2015) 77–101）Figure 5 的正式
Elsevier 300 dpi 图源为
`https://ars.els-cdn.com/content/image/1-s2.0-S0021999115004799-gr005_lrg.jpg`
（2310×623）。云图物理窗口是 `x=0–18 km、z=0–4 km`，不是 0–16 km；右列为
`h=12.5 m`，三行依次为 600、750、900 s。900 s 前缘的文献定义是地面
`theta'=-1 K`，表值 15056 m。2026-08-02 的坐标审计用该大图色条反演得到约
15044.5 m，与表值相差 11.5 m（约 0.39 个出版源像素），可用于检查方向和坐标映射，
但出版栅格不能用于逐点误差、相关系数、IoU 或能谱。错误地把整幅 0–18 km 栅格重标
为 0–16 km 会把前缘压缩约 11.1%，左右翻转无法修复。

2026-08-03 的原三列 AV75 四时刻图仍保留在
`output/reports/CULBM_GEA_AV75_density_current_comparison_20260803.{png,svg}`；加入当前求解器
h=25 m 新跑后的四列图位于
`output/reports/CULBM_GEA_AV75_density_current_h25_h12p5_h5_comparison_20260803.{png,svg}`，
可复现脚本与来源审计位于
`core/cfd_core/CULBM/test-output/density-current-av75-comparison-20260803/`。文献列必须使用
Girfoglio、Quaini、Rozza（2023）Figure 7 的 `av75-page13-000..003.png`；
`gea-004..007.png` 是 Smagorinsky h=25 m，仍不得误标为 AV75。

新的平台 h=25 m 正式运行是
`density-current-av75-h25-current-20260803_222850/run`：h=25 m、dt=0.05 s、
D3Q19/D3Q19、SRT、FP32、SGS off、18000 步，当前 executable SHA-256 为
`1EEC8EC6B5D078782988BC9E382BA9D2C47C6A8C05C7717E1FAF760C3629A9D7`；单进程
259.815 s 完成，31 个 metrics 与 5 个 1024×256 快照均 finite/positive，密度全程
[0.992088,1.008492]，但全程峰值 Mach=0.131352，解释时必须保留弱可压误差提示。
早先 `...20260803_222356/run` 被两个进程并发写入，已明确标记 contaminated，禁止使用。

四列形态图的可视锋位统一定义为正在显示的二维场中、跨至少五个连续行持久的最右
`theta'=-1 K` 冷到暖等值线鼻端；地面交点另存为 audit metric，不能混称。该修正把
300 s 的 GEA/h25/h12.5/h5 标记分别改为 3978.944/4229.277/4230.789/4232.336 m；
旧 GEA 3767.295 m 是数值正确但不适合标视觉鼻端的地面交点。GEA 900 s 地面交点
14804.538 m 仍落在 Table 3 的 [14783,14808] m 格点区间内。出版栅格只作坐标与形态
证据；图中按用户要求显示标签 `AV75`，本构差异仍以本节前述来源说明为准。

### `.luwdg` dataset 速度开放出口实验

2026-09-04 的绵阳 22.5°、均匀 6 m/s、10000 步受控实验确认：实验版在
`downstream_open_face=false` 时，其 `avg/raw_rho/raw_u` 三个 VTK 与生产版逐文件 MD5
一致；新代码路径在开关关闭时是 no-op。开启逐步一阶零梯度速度出口后，目标子域
（i=190..381、j=221..444，距 south/-y 出口约 1.105–2.220 km）平均风速增加
15.05%，TKE 增加 23.69%，平均密度下降 1.37%；末态净体积通量为
`+5.343e6 m3/s`，占总出流 17.68%。因此该实现不是可接受的生产开放边界，不能把
广域加速解释为可信的局部出口改善。

实现审计还发现：`.luwdg` 均匀 dataset 分支虽然解析 `flux_correction=true`，但在
`setup.cpp` 的该路由中没有调用 `apply_flux_correction`；而出口核每步重写出口
`rho/u`，所以只补初始化通量修正也未必充分。后续应优先评估压力/密度出口或对流
出口配合动态质量约束，并增加逐步质量与各面通量门禁。可复现脚本与 MCP 报告载荷在
`output/reports/CULBM_open_downstream_comparison_20260904/`；数值结果和独立验证回执在
`D:/averagedField/New0613_EUHPC/cmp_open_new/comparison_open_vs_closed_20260904/`。

## 本地粒子展示与离线渲染

2026-09-14 的重庆渝中/新加坡展示材料位于
`output/pollution_showcase_20260914/`。两套均匀 6 m 剖面 case 各用一张本机
GPU，由 `CUDA_VISIBLE_DEVICES` 隔离、原生 `luw runluw` 分派至 CULBM；每套
26,000 步、317,376 个 prescribed RK4 粒子、47 份 CSV，实际均退出 0，所有
快照 `invalid=0`。这是展示工况与输出完整性检查，不是浓度、沉降或网格收敛验证。
`solid` 是粒子碰撞终止状态，不应解释为经校准的沉积量。

渲染粒子必须与 STL 使用同一米制坐标，普通 `luwvox` STL 应扣除人工地基层
`base_height`。PyVista 0.47.1 的 `points_gaussian` 默认把显示半径按粒子点集
包围盒长度缩放；制作不同时刻对比时应显式固定 `actor.mapper.scale_factor`
的米制值，否则会把点集变长造成的光斑变大误画成更多扩散。该显示尺度不是物理粒径。

本机保留的 `build-runluw-agent` CLI 没有完整 GDAL 支持。完整依赖的旧
`LatticeUrbanWind_rebuild/build-production` 可完成裁剪，但其 `luwvox` 在独立
进程中缺少 `GDALAllRegister()` 初始化，无法打开有效 Shapefile；对应原生测试
先调用该函数，因而会遮蔽独立 CLI 问题。本次通过工件目录中的小型 C++ 启动器
初始化驱动后，直接链接并调用原有体素实现完成前处理，没有修改生产算法。
若后续修复该问题，应增加真正新进程打开 Shapefile 的 CLI 回归。PROJ 数据库
须与该构建配套；旧 pyproj 的 layout minor 4 数据库不能供要求 minor >=6
的原生 PROJ 使用。配套路径与本次运行证据保存在工件说明及 `native_env.ps1`。

展示图和动画应使用完整输出尺寸的渲染视口。把较矮视口的截图贴入标题或页脚
留白，会让地形与建筑在画面内部出现硬截断；隐藏文字时也必须移除该布局。
平行投影若需保持横向显示尺度并扩展上下视野，应将 `parallel_scale` 按新旧
视口高度之比缩放。2026-09-14 修订采用白色背景、浅灰地面，仅保留排放源
颜色图例；入口为 `output/pollution_showcase_20260914/render_white_showcase.py`，
当前成品为同目录的 `white_showcase/`，此前版本另存归档。地面材质只改变
显示颜色，不得修改 DEM、STL 或粒子坐标；相同帧应固定粒子显示半径与颜色。

粒子 case 的显存估算不能只引用启动时的 `Required VRAM`：它不包含后续可选
统计缓存及独立粒子模块。2026-09-14 新加坡展示日志记录 464×424×91、约 6 m
网格，GPU 常驻 Welford 统计启用后的 `Memory Usage` 为 1943 MiB，随后粒子
模块预检额外 236 MiB，合计约 2.13 GiB；这不是整卡峰值实测。维持同一物理域、
存储精度、统计与粒子设置时，三维均匀加密可用显存约随间距的负三次方增长作
初估，另留驱动、域间 halo/通信和分配余量。双卡总容量只有在同一个 case 做
域分解时才可一起使用；当前粒子模块支持同进程本地多域，但显式拒绝 MPI
分布式运行。不得将容量估算当作已验证的加密运行结果。

2026-09-19 的 NUS UTown 5 m 单卡案例保存在
`E:/#_PAPERS_#/IP_J40_SCP-FTLE/Overleaf/LUW_NUS/`。以 NUS 官方校园地图
University Town 点为中心，沿用 2782.879×2543.967 m 水平域与 225° 来流。
`scripts/place_sources.py` 用邻域建筑覆盖率、开放区净空、横风投影间距和顺风错开
共同筛选三个源；本例相邻横风间距 226/198 m，在下游保持部分空间交叠。
这是该展示工况的选址实例，不能把其统计交叠值当作分子混合效率或浓度验证。

容量与结果报告须分别保存实际求解网格和导出场尺寸：上述 case 启动日志为
557×509×110（含海绵），分析 VTK 为 557×509×91（裁剪后的核心场），不能
通过后者反推显存规模。改变 cell_size 后，渲染时刻应从实际 SGS step/time
或等效原始时间记录读取，不能沿用旧 6 m case 的硬编码时间尺度。

按截图圈选重布源时，应先把截图配准到原图，再使用已保存的相机对原生几何
射线求交；屏幕距离不能直接当作米制距离或横风距离。2026-09-19 的 NUS 修订
位于上述目录的 `source_revision_20260919/`，保留原图、配准参数与实际源位移。
STL 表面净空不能替代求解器体素检查：细窄楼间空地上方的种子仍可能落入
5 m 建筑边缘固体格点。应按原生粒子核的 float32 格点坐标及
`floor(x+0.5)` 最近格点规则检查 flags，并在源斑内重新采样冲突位置。
相同几何、网格及坐标基准的既有 flags 可用于预检，但须核对输入一致性，并
用本次正式输出再次确认；不得仅凭显示网格或视觉判断宣布种子位于流体内。

CULBM 的 `unsteady_output` 是 LBM 步数间隔，不是秒。非零时按该间隔导出
`raw_u-<step>.vtk`，末步还可能多一份非等间隔输出；向 FTLE 输入序列时须按
步号筛选严格等间隔窗口，并以原生 float32 时间换算值计算 `physical_dt`。
`raw_u` 的三个 SI 速度分量为大端 float32、VTK x-fast 顺序，原始数组名为
`data`。其 z 原点仍含基础板；与粒子和物理地形统一时应减去 `base_height`，
不能照搬已去基础板的 global VTK 原点再重复平移。按字节保留速度负载、只改
数组名和坐标头即可接入 Streamcenter+；在同一 NTFS 卷内可填充 VTK 标题以
保持头长不变，再原子移动完整文件，避免为 TB 级序列重复写入整个负载。

原生 SGS CSV 的 `time_s` 使用 `units.si_t(step)`：先计算单精度
`float(step) * unit_s`，再按 12 位有效数字写入。长时间序列不能以固定
1e-5 s 残差对这些值作双精度直线拟合来判断变步长；约 3000 s 时正常舍入
即可达到 1e-4 s。应以原生 `units.si_t(1)` 的 float32 值建立时间轴，
逐条重现上述乘法及打印精度，并拒绝不符的时间记录。单条 `time_s/step`
只能给近似步长。NUS 无 IBM 长算的正式 187 条记录及相邻 float32 步长
拒绝检查保存在 case 的 `qa/solver_time_validation.json`。

LUW 原生粒子插值会排除固体角点并重归一化权重；一般 FTLE 三线性插值可能
包含固体内的零速度，两者近壁轨迹不能无条件视为同一数值问题。有限开放域
的长时 FTLE 必须检查离域和固体端点、屏蔽无效梯度邻域；高 FTLE 等值面只能
作为屏障候选区展示，不能单凭该图宣称已验证严格的 material barrier。

FTLE 性能试算图必须同时注明原始风场起始时刻 t0、积分长度 T、种子间距与
显示采样间距，不能只靠文件名中的分辨率判断其科学含义。NUS 的 2 m benchmark
实际使用启动阶段约 1–6 s 的速度，T 仅约 5 s；其全域外观不能与 t0≈400 s、
T=300/600 s 的图直接比较。开放域应先扫描积分窗及驻留/有效覆盖，再选择长时
窗口；分别记录初末点合格数与梯度邻域合格数，不能把后者的损失都解释为离域。
等值面图与全体采样散点叠加会改变结构可读性，应在相同数据与显示方式下做比较。

2026-09-20 的 NUS 连续面修订在同一 t0≈400 s 风场上扫描 10/30/60/120 s，
4 m 种子有效梯度覆盖分别约 88.9/74.6/56.0/27.8%；60 s 精化为 2 m 种子后为
57.4%。这些值只适用于该有限域与筛选规则，不能固定成其他案例的推荐积分窗。
用于展示的 FTLE 空间平滑必须单独注明尺度并保留原标量；本例采用 σ=8 m 的
有效权重归一化 Gaussian，仅保留原有效区域。等值面顶点平滑后仍须重新检查
实际顶点和三角形中心的有效支持，并再次筛选新分裂的小连通片；不得补齐无效
空洞来伪造连续屏障。2 m 种子不等于经 8 m 显示平滑的图仍能解析 2 m 结构。
PyVista 的 `clean()` 可能把退化三角形转换成线或点单元；连续面交付应在最终
清理后显式移除这些单元，并检查保存后 VTP 的 `n_verts/n_lines` 均为零。
按 `faces` 数组筛选前也应先去除点/线单元：PolyData 的完整单元编号包含排在
多边形前面的点/线，直接把三角形布尔掩码当完整单元掩码会错筛边界面。

NUS 同一 60.2 s 窗口、4 m 诊断种子，将锚点从约 400 s 推到 900 s 后，
有效梯度覆盖仍为约 55.99%/55.90%；z=200 m 掩码交并比约 96.3%。矩形有效区
主要可由前向轨迹的有限域驻留范围解释，不能据此判断启动扰动传播或统计未收敛。
锚点统计稳定性须独立验证：全域 SGS/平均速度接近平台不足以证明局部均值与
二阶统计量收敛，也不能把稀疏短时间块间的差异直接认定为启动漂移。

有限域提前离域的种子可以保留有定义的驻留时间受限 FTLE，但必须同时输出实际
受支持的积分时长。不能对不同退出时刻的终点直接求空间梯度，也不能把停在边界
后的原生 FTLE 直接解除掩码。2026-09-20 的 Streamcenter+ `FlowMapProbe` 增加
`residence_ftle` 研究模式：中心和六个求导邻点在同一个时间检查点计算梯度，失去
支持后保存上一个共同时间的 FTLE 与时长；速度分量的全域上界保守约束 RK4 阶段
不越过输入 AABB。这个时长可能短于真实退出时刻，需做检查步长敏感性测试；
可变时长结果不能标成统一 T 的长时屏障。解析平移、旋转、定常/非定常应变及
反向积分测试位于 NUS case 的 `scripts/test_residence_ftle.py`，结果记录在
Streamcenter+ 根目录 `.tests/nus_utown_2m_ftle_20260919/residence_validation_20260920/`。

NUS 该方法在约 900 s 锚点、60.201 s 上限、2 m 种子上保留了全部
91,169,482 个初始可求导流体种子的值，其中 51,629,045 个有完整窗口，其余
39,540,437 个保留较短历史。配套成果在案例 `residence_revision_20260920/`。
4 m 检查步长减半时 95% 的 FTLE 绝对差异不超过 1.702e-4 s^-1，但约
0.0605% 点超过 0.01 s^-1，最大的样本涉及近壁掩码触发时刻变化；不得略去这个
尾部或宣称全域已收敛。原生支持原因位 2 来自流体掩码，而该掩码也包含标为
固体的外边界行，所以不能单凭这个原因位把所有截短都归为建筑壁面。

NUS 竖直截面对照位于案例 `building_sections_20260920/`，出图入口是
`scripts/render_building_sections.py`。速度须匹配 FTLE 锚点时刻；截面内 `(u,w)`
或 `(v,w)` 流线应明确标为瞬时投影速度场流线，不能称为三维粒子轨迹。
原生 residence FTLE 二进制是 C-order `(x,y,z)`、z 最快，打包 VTKHDF 是
`(z,y,x)`，截取同一平面时须独立比对以防转置错误。Matplotlib 默认流线绘图
积分可能跳过薄墙；本例一格近壁保护加可见线段每 0.5 m 支持检查消除了该问题，
但不应把该显示检查当作三维轨迹或积分收敛验证。

2026-09-20 截面修订进一步确认：速度热图和固体填色必须采用同一最终求解体素
分类。不得用“插值邻点全部流体”的支持掩码隐藏速度、同时用单个最近体素给
建筑填色；这会把有速度的真实流体画成白洞。当前图直接使用最近原生网格列的
速度与正式末步 flags，以 `TYPE_S=1` 判断固体；`TYPE_E=2` 是平衡边界，不能
用 `flags & 15` 一概当成实体建筑。FTLE 的梯度/轨迹有效掩码另有含义，缺值
流体不得填成固体或任意补值。当前成品位于 `building_sections_20260920/voxel_revision_20260920/`。
用户对本组截面要求只保留图、坐标轴和色标，去除建筑突出标记、标题、说明和
位置图；所有文字统一22 pt Times New Roman，PDF已验证仅嵌入该字体。

同日逐点核查 `near_wall_ftle_exclusion_audit.json` 确认：当前可视 yz 截面内
2,427个最终 flags 判为流体的 FTLE 种子，全在积分开始前即被筛掉；其中1,426个
来自 `safe = ~binary_dilation(solid, iterations=1)` 的一层近壁保护，另1,001个
是中心安全但六个中心差分邻点中有不安全点。保存的 `initial_stencil.u8` 与该
独立重建、以及切片 FTLE 的有限值掩码完全一致，没有“初始有效而后来无值”的点。
因此这批近壁白带不是 CFD 速度缺失，也不是提前离域造成，不能只在渲染层填补。
其数值修复需改变近壁种子支持/流体侧求导方案；不能把这些缺值解释为物理上
没有 FTLE，或把缺值流体改标成固体。

用户随后暂缓近壁FTLE数值修复，并要求截面改用平滑填色。当前显示版在
`building_sections_20260920/smooth_revision_20260920/`：对全部角点有效的单元作
双线性显示插值，近固体/缺值处回退到原单元值；原始单元有效掩码与最终体素
几何不变。显示细分不能当作新的物理分辨率，也不能藉此填补FTLE的NaN。

同日 q95 边界封口版位于 NUS 案例
`residence_revision_20260920/boundary_capped_q95_20260920/`。封口表示超阈值区域
与有限数据域交集的边界，封口内部不必等于等值阈值；应与域内等值面区分，
不能称为额外发现的物理屏障。只在最外有效网格的六个平面构造封口，不能用
通用 fill-holes 填满近壁无效孔洞，或把保护层延拓成有值数据。本例平面为
x=6/1384、y=6/1264、z=6/446 m，顶面在 q95 下没有高值区域。外部构造层的
marching contour 顶点投影到准确平面后，需要保留接缝、固定边界平滑、检查
顶点与面心支持、朝外法向和三角面单元；VTP 用 BoundaryFaceId 区分封口与
域内面。解析盒体/半空间、缺值孔洞及正式网格检查见该案例的
`scripts/test_ftle_domain_caps.py` 与 `scripts/validate_ftle_domain_caps.py`。

NUS 紧凑四联图入口为案例 `scripts/render_compact_composite.py`，成品位于
`compact_composite_20260920/`。排列为速度剖面、FTLE 剖面、底行粒子/三维 q95；
所有最终文字以同一 22 pt Times New Roman 直接排版，PDF 仅嵌入该字体。
比较两张三维图时，除相机参数相同，还必须共用裁切框、最终宽高和缩放比例；
分别按内容裁切会破坏几何对齐。按用户最新要求，FTLE 三维面板除编号 (d) 外，
左下用同色小圆点标注 FTLE⁺ = 0.07 s⁻¹（显示取两位小数，实际阈值不变）；上标须使用 Times New Roman
原生字形，“+”根据实际时间窗确认是正向积分。粒子 Source 图例也放左下。
该拼图保留粒子约1000 s、FTLE锚点约900 s的原时刻，
不因视角一致就声称是同时刻的数据对比。

紧凑拼图还须检查文字与后绘制面板画布的重叠，而不只是文字是否位于整张画布内。
NUS 四联图曾因下排三维 axes 的白色背景覆盖上排 `y / m` 下缘约 8 px；现增加
0.1 英寸行间距，并检查标签下缘与下排 axes 顶部至少有 4 pt 净空。PNG 原像素
局部与 PDF 文本边界须分别核对；当前净空约 21.67 px / 5.00 pt。图 (b) 色标
与图 (d) 使用同一实际积分方向上标。拼图专用流线宽度与箭头尺寸通过参数传入，
保留独立截面渲染的默认样式。

## ERA5 最终版与点时间序列取样

需要排除 ERA5T 时，不能把 CDS 的 `product_type=reanalysis` 当作最终版证明。
应保存版本核查证据；ECMWF GRIB 的 `expver=0001` 为最终 ERA5，`0005` 为 ERA5T。
若按最近完整一年制备剖面，应先确定最终版截止月，再取连续十二个完整月，避免
把近期 ERA5T 静默混入年平均。不同镜像的更新截止日期可能晚于或早于 CDS，
不能把一个镜像的截止日期直接当成所有数据源的统一截止日期。

官方 `reanalysis-era5-single-levels-timeseries` 可高效提取单点逐小时数据；
`reanalysis-era5-pressure-levels-timeseries` 虽标题含 hourly，实际时间分辨率为
00/06/12/18 UTC、13 个压力层。必须核对返回时间坐标，并记录高空样本数；
不能把每六小时样本或其时间插值称为原始的 8760 个逐小时高空场。
年均标量速度剖面应明确先求逐时/逐样本水平风速模长再平均，还是先平均 U/V。

## 大网格 MPI 输出的主机内存与磁盘预算

当前 `src/solver/lbm.hpp::Memory_Container` 在 distributed 模式下为每个 rank
分配全域 `distributed_data`；`read_from_device()` 使用 allgather，
`lbm.cpp::read_device_statistics()` 还构造全域 mean/M2 view 与输出数组。
因此每 rank 只使用一张 GPU，不代表主机内存也只保存局部网格。多节点预算必须
同时计入各 rank 的全域主机副本、统计回读及最终写出峰值，不能只除总显存。

均匀网格的周期 raw-u VTK 为每点 3 个 float32，即 12 B/点，保留固体格且输出
范围由核心网格决定；当前没有周期 raw-u 的中心 ROI 写盘选项。后处理裁剪可以
减少归档量，但不能直接减少求解器已经写出的全域原始数据量。`unsteady_output=1`
与起写步 `Nconv+1`、终止步 `Nconv+500` 配合可产生 500 个连续样本；起写门槛
本身不是收敛检测器。LES 中必须区分统计平稳验收与末端有限时间窗的平均/TKE，
尤其要报告 500 个相邻 dt 样本仅跨越 `499*dt` 的实际时长。

跨格距预算应先用各自 `dt` 匹配发展物理时长，再比较步数和耗时，不能简单保持
`run_nstep` 相同。相同剖面下 CULBM 的参考格子速度固定，故 `dt` 随实际格距变化；
格点数约按格距的三次方变化，同物理时间的推进步数还约按格距反比变化。
同为 500 个相邻快照时，两套网格也不覆盖相同物理时长。已有同几何/同网格的
短测可校准容量和步速，但更换剖面、风向和长时发展后仍需另验稳定性与统计收敛。

Arrhenius 的 `GPU/buildenv-gcccuda/2026.03-cu13.0` 使用 MPICH，现场模块设置
`SLURM_MPI_TYPE=pmi2`；显式 `srun --mpi=pmix` 会在初始化时报
`launcher not compatible with PMI1 client`。多节点作业应先用同一 binary 的小型
MPI 检查确认 launcher，不能把单卡编译/短测通过当成 MPI 启动已验证。

2026-10-30 现场调度器要求 GPU 作业申请量至少为 `(nodes-1)*4+1`；
八节点各申请一张 GPU 会被拒绝。GPU 节点 `RealMemory=485854 MiB` 另有
`MemSpecLimit=65536 MiB`，故 `--mem=450G` 不可申请，`400G` 已通过调度检查。
实际规则、保留内存、`PriorityFlags=MAX_TRES` 和 Req/AllocTRES 必须实时复核。

`unsteady_output=0` 只关闭周期 raw-u；普通流程仍写最终瞬时场，不能据此承诺
“仅平均场”。本次绵阳案例在 `output/mianyang_20261029_preflight/production_2m_stats/`
保存独立原生 C++ 控制层，沿用已归档 CULBM 动力学内核，并在 GPU 上累积统计、
按子域直接写中心 ROI 的均值/TKE/掩码。该案例控制层不是公共 deck 功能；不要
将其环境开关解释为普通已安装 binary 的能力。块间监测验收、技术检查的绕过开关
及科研结果状态分开记录，技术检查输出不能作为物理收敛结果。

2026-10-30 的 `vram8_bill8_20261030` 原生容量检查使用两节点、每节点四卡，
ReqTRES、AllocTRES 和最终 sacct 均为 `billing=8,gres/gpu=8`。它直接构造八个
与 2 m 案例相同的 `377×2999×532` 本地子域，跳过全域主机副本和 MPI 编排；
核心/halo/Welford 及原动力学核函数不变。实测每卡核心+统计约 63.97 GiB，
另加保守入口容量预留后约 64.50 GiB；八卡同时驻留约 30 s、推进 32 步通过。
这是显存容量检查，不代表含真实几何、全域主机副本和跨节点交换的正式流程已在
billing=8 下跑通。后续引用必须保留这个适用范围；不能用容量测试的主机 RSS
代替原完整 MPI 流程的主机内存预算。

5 m、仅平均/TKE 的最新配置保存在
`output/mianyang_20261029_preflight/production_5m_stats_20261031/`，不要与
`variants/5m/` 中历史的500帧瞬态方案混用。按相同物理时长换算后，最低发展
72025步、监测块5064步/每8步采样、最少统计20256步，dt约0.118709883 s。
单卡/双卡模板都采用单进程本地GPU（`CULBM_MPI=0`，清除launcher rank环境），
不走跨节点MPI主机副本路径。5 m同网格已有单卡短测：正常33.938步/s、日志GPU
33077 MiB、主机峰值约30.24 GiB；双卡约16.30 GiB/卡是静态模型，尚未实测。
新控制层的步数已经随格距调整，不能直接用旧2 m专用binary只改cell_size来运行。

16个独立风向可在Arrhenius四个整节点上各用一张GPU并行；本例实际
`--nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=72`
配合每节点4GPU、`--mem=256G --exclusive`，ReqTRES与AllocTRES均为
`billing=16,gres/gpu=16,node=4`。任务步骤使用`--mpi=none --gpus-per-task=1
--gpu-bind=single:1`，求解器子进程清除全部已支持的launcher rank标记并设
`CULBM_MPI=0`；应以16个互异GPU UUID验证绑定，不能只比较每个任务都重映射为0的
`CUDA_VISIBLE_DEVICES`。整批allocation在最慢风向结束后释放，先结束风向的尾段
仍计入整批billing。当前最终ERA5剖面的5 m正式首10,000步实测33.147–34.852步/s，
`setup.cpp`的`dt_si`按名义`cell_m=5`打印0.11875645305626485 s，但
`Units::set_m_kg_s_K`使用实际`si_size.y/Ny`，该工况真实float32求解步长为
0.11870988458395004 s，两者相差约0.0392%。本例最终审计器v2在全场审计后校正
JSON/监测CSV的物理时间，并以`.native_controller`保留原始元数据；不可把控制层
名义步长直接当作`Units::si_t(1)`。首段速度和短技术检查均不能替代长时统计平稳性验收。
## CULBM 几何与节点坐标（261116）

STL 最低坐标映射到格点坐标 `+0.5`，整数索引表示节点中心。均匀网格
raw SI 坐标因此为 `stl_min + (index - 0.5) * dx`；z 再加上实际移除的
地基厚度 `cuda_ground_trim_si`。全局物理输出仍按
`stl_min_z + base_height` 重设地面 datum。整格基础板压缩后，保留的固体
地面节点为 z=-dx/2，首个空气节点为 z=+dx/2，半格反弹壁面位于 z=0。

几何平移、边界样本、平地/DEM 的 profile AGL 参考和输出原点必须使用同一
变换。DEM 的原始高程先以未压缩的基础板顶面为基准，再减去实际移除的
格层数；不能在已减小的 base_height 上再次减去 STL 最低高程。

嵌套网格沿用既有的地面面映射：粗、细格点坐标中的地面均为 +0.5。
去掉最细固体地面行后，首个导出节点的 raw z 为
`stl_min_z + cuda_ground_trim_si + finest_dx/2`；不能从粗网格平移量
与细网格总层数重新推导这个 z 原点。

体素化射线交点保留浮点距离，并在同一个节点使用共同的舍入容差判断交点。
不能把交点截断到整数格层，否则共面基础板/建筑底面会被分到不同层，
非整格屋顶也会丢失有效固体节点。零交点射线不得读取未初始化的距离。

此修复会改变旧版本的固体占据和坐标标签。已有流场检查点及按旧原点处理的
结果应保留版本标识；更新几何后从初始条件重新计算，不能只平移旧结果
代替重算。

## AIJ Case E 近地面低估排查（261117）

2.5 m 几何修复算例的 profile 分支仍存在出口语义问题：
`apply_profile_boundaries` 先设 `TYPE_E`，随后因 `downstream_open_face`
而跳过赋值；内部初始化的速度却仍保留。`TYPE_E` 动量核持续使用保存的
rho/u，纯风构建中没有随后更新出口速度的代码。因此这条路径实际是固定
初始剖面的平衡边界，不能仅凭 deck 的 `downstream_open_face=true` 或
日志中的 `no fixed velocity` 将它认作速度零梯度出口。AIJ 三个 200 s
统计块在出口的速度完全一致，TKE 仅有 float32 舍入量。正式修复需要检查
动量边界的分布函数和质量通量，不能只修改标签或温度出口。

AIJ 的 1 m canonical 入流只是中间采样网格；将含 z=0 补零和 z=1.25 m
首个实测值的 AF 均值剖面先采到整数米，再采到 2.5 m 的中心节点，会丢失
首层折点：U(1.25) 从 2.847 降到 2.4492 m/s，随后在 2 m 的离散插值为
2.6598 m/s，原 AF 线性插值为 2.964 m/s。均值应在求解器实际节点直接按
原 AF 求值，脉动另行插值；不要将这个确定性损失归为随机入流采样误差。

当前该算例采用 SRT、常系数 Smagorinsky Cs=0.17、普通无滑移反弹，
没有近壁阻尼或壁面应力模型。2.5 m 网格的首个空气节点为 1.25 m。
原生平均场在无建筑上游已出现明显近地面沿程衰减；这定位了近壁计算路径
的问题，但不能由单张误差图分摊壁面模型、碰撞格式和网格各自的贡献。
小型恒定入流对照的无滑移分支出现强振荡或回流；应保存并报告其物理异常，
不能以进程退出码 0 当作验证通过，也不能据其数值拟合正式 AIJ 参数。

风洞记录为平均标量速率，时均速度矢量的模不是同一统计量。已有原生
一、二阶矩可给出解析场平均速率的上界，不能重建精确的平均速率；空间
插值后的节点二阶矩还包含空间差异的上界。SGS 未解析运动也不在该上界内。
以后在线采样应同时累计测点的 |u(t)|，并保留时均矢量作方向性比较。

## CULBM 反弹壁面模型（261118）

`culbm_wall_model`（deck 缺省 true）在 `stream_collide` 与 `update_fields` 中对
`TYPE_S` 轴向面做"切向镜面交换 + 壁面律应力注入"。库级 `Wall_Model_Configuration{}`
缺省 false：直接构造 `LBM` 的 GPU 回归、GEA 验证等驱动保持原无滑移；只有 setup 的
deck 解析默认打开。配置与 SGS 一样必须在构造第一个 `LBM` 前确定，常数按每个域的
`dx/dt` 注入 NVRTC；嵌套层由 `Wall_Model_Level_Scope(ratio)` 在构造前提供层尺度，
每个域启动时打印 `Wall model: dx=..., Cd ground=...` 供核对。

实现要点（修改前先读）：
- Esoteric-Pull 的隐式反弹是两步延迟：格点第 t 步从固体槽位读回的是自己 t−2 步射向
  固体的分布（奇偶步各一套槽位），不是 1 步。温度 BB 与 FREE_SLIP 显式读 1 步前出射值；
  动量壁面模型沿用隐式两步时序。
- 每面每个切向方向取一对离壁斜向链 (P,M)：先交换（自由滑移），用交换后的首层速度求
  `T=ρCd|u_t|u_b`，再 P−T/2、M+T/2。壁面每步恰好取走 T，质量和法向动量严格不变。
  用无滑移反射值计算壁面函数输入会系统偏低约 1/3，不得改回。只有一条链命中固体的台阶
  边缘保留无滑移。只读本格点已载入分布与静态 flags，单卡与 `Dx=2` 双卡逐位一致。
- 近壁 SGS 层 `culbm_wall_model_sgs_layers`（缺省 3）：`LBM::initialize()` 用全局 host
  flags 为每列求"底部实体之上首个流体行"（ushort/列，含 halo），前 L 行 Smagorinsky
  长度取 `max(Cs·dx, κ z_w)`；只覆盖朝上表面。SurfaceIBM 工况编译
  `WALL_MODEL_GROUND_ONLY`，壁面模型与 SGS 层都只作用于压实地基之上。
- 凹角（两固壁面相交）处两面的切向对共享同一条斜向链（如 16 号链同属"地面 x 对"与
  "+x 墙 z 对"）；两次交换线性叠加得到 `f16←f9+f10−f16` 这类非反射组合，曾在 AIJ
  5 m 体素建筑的单格凹坑/窄缝首层泵出 12–29 m/s 虚假射流，CTest 凹井用例中直接发散。
  现规则：冲突的两对都保持无滑移（90° 凹角两次镜面反射等价于反弹），每条链至多参与
  一次交换；回归用例 `test_concave_pocket`（凹井 |u| 须 <0.3 倍来流，修复后约 2e-3 倍）。
- 贴壁格点正则化（只保留 ρ、ρu、Π）已试过：不改善稳定性，还使首点偏低、外区偏快，
  不要再加回。

验证（`tests/test_wall_model_gpu.cpp`；CTest `CULBM.WallModelGPU` 用 FP32 构建，
`CULBM.WallModelMultiGPU` 用 FP16C 比较 1/2 卡逐位；`--channel/--benchmark` 研究模式，
开放槽道 `--open 1` 需 `CULBM-wall-model-gpu-fp32-freeslip-test`；材料与脚本在
`core/cfd_core/CULBM/test-output/wall-model-20261118/`）：
- 层流体积力槽道：首层自由滑移速度与 `sqrt(gh/Cd)` 差 0.1%，与同算例中心抛物线 −0.2%
  的 SRT 离散误差同级。FP16C 存储吞掉 1e-6 量级体积力增量（无壁面模型的泊肃叶也差
  2.3%），体积力解析测试必须用 FP32 构建。
- 开放槽道 h=16Δ、z0/h=1.25e-3、Cs=0.173：首点 U+ 偏差 无模型 −5.3 / 仅壁面律 −0.3 /
  壁面律+L3 −0.1；外区 +0.2 / +5.7 / +2.5；体积流速 −3.8% / +34% / +15%。外区约
  +2.5u* 的截距偏移在 h=8/16/32 基本不变，是近壁 RANS 化层与上方 LES 的界面失配；
  平台或指数延伸只把它推高（+1.9~2.1）并加厚 RANS 区，故保留 3 行阶跃。全局加大 Cs
  也能减小失配（Cs=0.35 时 +1.3），但会过度耗散自由剪切层。
- 体积力周期槽道在 max|u|≈0.10（Ma≈0.18）、FP32、τ0≈0.5 下，无论有无壁面模型都会在
  1e5 步量级突发 NaN（追踪起点在槽道中部而非贴壁层）；壁面模型只因降低壁阻提高了
  定力槽道流速。研究槽道应取 u*≤0.003（max|u|≲0.075）。生产 `lbm_ref_u=0.10` 与该
  极限同量级，长时周期算例需留意。
- AIJ Case E 示例（0°，300 s 平均，上游无建筑区，来流 profile.dat）：5 m 网格首层
  2.5 m，距入口 400/785 m 处无模型 0.72/0.63 m/s（来流 3.04），z0=0.03 为 2.66/2.30，
  z0=0.005 为 2.99/2.70；2.5 m 网格首层 1.25 m（来流 2.85），无模型 0.52/0.59，
  z0=0.03 为 1.88/1.58，z0=0.003 为 2.43/2.16。建筑区首层平均风速（凹角修复后）
  5 m：0.62→2.03（z0=0.03）/2.39（z0=0.005）m/s；2.5 m：0.60→1.54/2.02（z0=0.003）m/s。
  该来流近地段远比 z0=0.03 光滑（1.25–12.5 m 拟合 z0≈1e-4–3e-3 m），z0 应与来流剖面/
  上风地表一致，否则上游出现内边界层调整；10–20 m 高度开/关两算例同样低于来流约
  10–15%，不能归因于壁面应力。
- 示例 `CaseE(Niigata).xls` 的 80 个测点（实尺度、原点为再现区中心）加
  (1010.5, 1000.6) m 即与示例 `CaseE_PF.stl` 对齐（恒等变换只有 4% 测点落入建筑体素）。
  N 风向 2 m 风速比（以来流 15.9 m 的 3.93 m/s 归一）：实验均值 0.345/0.422（建设前/后），
  2.5 m 无模型 0.124、壁面模型 0.147（z0=0.03）/0.156（z0=0.003），5 m 为 0.075/0.085。
  壁面模型消除了上游近地衰减，但建筑区测点低估主要随街谷分辨率变化，不是地面应力问题；
  后续应检验垂直嵌套近地加密。Excel 二进制表可用本机 Excel COM 另存 CSV 读取。
- `culbm_wall_model=false` 与开工前二进制（261117）在 AIJ 5 m 400 步上 raw_u/raw_rho/avg
  逐字节一致；旧二进制副本在 `test-output/wall-model-20261118/baseline_bin/`。
- 性能：壁面模型几乎全部开销来自寄存器压力（stream_collide 80→101 寄存器时
  occupancy 约 75%→56%，密集街区慢 6.9%）。冲突规则保证每条链至多一次交换后，改为
  原地交换、按面即时求应力系数，寄存器回到 84。FP16C 单卡 256×256×128、交错 7 次
  中位：密集街区（近壁格点 4.8%）仅壁面律 +0.8%、加 SGS 层 −1.1%，稀疏立方体（1.2%）
  −1.5%。修改设备代码后用 `CULBM_DUMP_PTX=1 CULBM_PTX_CACHE=0` 导出 PTX，再以 CUDA
  `ptxas -arch=sm_75 -v` 核对寄存器数，不要只看 MLUPS。

嵌套网格平均阶段每个样本在 CPU 上执行 `stitch_nested_physical_fields`，大 `purge_avg`
会让嵌套算例在平均段极慢（与壁面模型无关）；冒烟测试应把 `purge_avg` 设小。
