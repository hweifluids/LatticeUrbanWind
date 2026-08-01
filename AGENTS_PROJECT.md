# AGENTS_PROJECT.md

## 项目级持续说明

本文件记录对后续 agents 有持续价值的项目知识。若后续任务发现新的、可复用的构建方式、运行方式、排查结论、约定或长期注意事项，应更新本文件；不要只把这类信息写进一次性的 `codexlog/`。

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

`core/cfd_core/FluidX3D/src/utilities.hpp` 中 `ulong` 定义为 `uint64_t`。在 Windows/MinGW 上它可能等价于 `unsigned long long`，而在 Linux GCC 上常等价于 `unsigned long`。因此修改 `ulong` 相关计算时，尤其是传给 `std::min` / `std::max` 的参数，不要混用 `1ull`、`0ull` 等 `unsigned long long` 字面量；应显式转换为 `ulong`，例如：

```cpp
std::max(current_t + (ulong)1u, avg_start_t);
std::min(avg_start_t - (ulong)1u, total_steps);
```

否则 Linux GCC 可能报 `deduced conflicting types`，而本机 Windows 语法检查不一定能暴露该问题。

## CFD Core 编译

Windows 下优先使用仓库脚本编译 FluidX3D CFD core：

```powershell
cmd /c "installer\3_compile_cfdcore.cmd < NUL"
```

说明：

- 命令应从仓库根目录执行。
- `< NUL` 用于避免脚本末尾 `pause` 在非交互环境中挂起。
- 脚本会通过 `vswhere.exe` 或常见安装路径定位 MSBuild。
- 当前本机可用路径示例：`C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe`。
- 项目文件请求 `PlatformToolset=v143`；脚本会在必要时切换到已安装的新工具集，例如 `v145`。
- 输出可执行文件为 `core\cfd_core\FluidX3D\bin\FluidX3D.exe`。
- 构建过程中可能出现 `lodepng.cpp` 的 `C4267`、`C4334` 等既有 warning；只要最终为 `0 Error(s)` 且 `Build succeeded.`，本次 CFD core 编译视为通过。

快速语法检查可用 MinGW `g++`，但它不能替代 MSVC Release 链接构建：

```powershell
g++ -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -fsyntax-only src/setup.cpp
g++ -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -fsyntax-only src/info.cpp
```

以上命令需在 `core\cfd_core\FluidX3D` 目录下执行。

## season_average 后处理

`core/tools_core/season_average.py` 可直接指定平均 VTK 输入目录并按目标 spacing 重采样输出：

```powershell
python core\tools_core\season_average.py <case>\conf.luwpf --vtk-dir <case>\RESULTS\crop\cropped_vtk_stage2 --output-spacing 10 --skip-figures
```

说明：

- `--vtk-dir` 可指向绝对路径或相对 case 目录的路径，用于选择修正后的 VTK 源目录。
- `--output-spacing 10` 会在 x/y/z 三个方向重采样；为保持输入物理范围，实际写出的 `SPACING` 可能是接近 10 m 的值，例如由 19.9991264 m 输入得到 9.9995632 m。
- 修正后 VTK 若使用独立标量 `u/v/w/tke`，脚本会合成为 `u_avg` vector，并同时输出 `u/v/w/vm/tke` 标量。
- `--skip-figures` 只生成 VTK 和 summary，适合几十 GiB 级别的大体积输出。

## HPC/Linux CFD Core 编译

本地 Windows 编译只会更新 `core\cfd_core\FluidX3D\bin\FluidX3D.exe`，不会更新 HPC 上的 Linux 可执行文件。若作业脚本运行的是 HPC 路径，例如：

```text
/work/home/.../LatticeUrbanWind-HPC0424/core/cfd_core/FluidX3D/bin/FluidX3D
```

则必须在该 HPC 仓库副本中同步源码后重新编译 Linux 版 FluidX3D。优先使用仓库脚本：

```sh
cd /work/home/.../LatticeUrbanWind-HPC0424
sh installer/3_compile_cfdcore.sh
```

或直接：

```sh
cd /work/home/.../LatticeUrbanWind-HPC0424/core/cfd_core/FluidX3D
chmod +x make.sh
./make.sh
```

诊断提示：如果运行日志仍出现旧字符串 `Avg benchmark | mean-field Steps/s = ...` 或 `ETA model | normal Steps/s = dynamic, mean-field Steps/s = ...`，说明正在使用旧版求解器。包含 timing-plan 改动的新版本应在 `SOLVER START` 前打印 `LBM TIMING PLAN`，并出现 `sample extra = ... s`、`Preprocess time`、`Estimated total` 等输出。

Timing plan 注意事项：

- `Info::print_initialize()` 会把 `runtime_lbm_timestep_smooth` 初始化为 `1.0` 秒/步。第一轮 LBM case 如果没有历史 normal Steps/s，不能直接用这个值估算 warm-up，否则会出现严重偏长的 `1s/step` 初始 ETA。
- 当前实现会在长循环前先运行少量 normal warm-up 步作为 `Normal benchmark`，再打印 `LBM TIMING PLAN`。这些 probe 步是实际计算步，会计入总步数，不会额外增加步数。
- 如果确实没有 normal 样本，timing plan 应显示 `pending normal-speed sample`，不要打印看似确定的 fallback ETA。

## 山地 LUW 风边界顶高

`core/bridge_core/1_buildBC.py` 中风场垂直层按 AGL 解释。对地形起伏较大的 LUW 标准模式案例，固定 CFD 顶高不能只用 `base_height + wind_top_agl`，否则最高地形可能高于顶边界。应使用全域最高地形作为基准：

```text
z_top = ground_z_max + min(wind_top_agl, z_limit_agl)
```

其中 `ground_z_max = base_height + max(relative_dem)`。林芝 20 m 案例中 DEM 相对起伏约 1.39 km，原始风场最高层为 995 m AGL；若不按最高地形抬升顶面，边界 CSV 顶面会低于高山区域。

## W 分量边界与通量修正

`core/bridge_core/1_buildBC.py` 会读取输入 NetCDF 中的 `w`/`W` 变量；若不存在则回退为 0。当前边界 CSV 的 `top/south/north/west/east` patch 会保留采样得到的 `w` 分量，不再强制写成 0；`bottom` patch 仍用于地形/地面信息，不作为常规速度入口面。

`core/cfd_core/FluidX3D/src/fluxcorrection.cpp` 的通量修正按外边界法向速度计算净通量：`x/y` 侧面使用对应水平法向分量，顶面 `Zp` 使用 `u.z`，因此带 W 的顶边界会进入通量平衡；侧边界上的 W 是切向分量，不应计入侧面法向通量。日志中 `Flux correction | net by face ... Zp(w)=...` 可用于确认 W 是否实际参与顶面通量修正。

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

相关源码位置为 `core/cfd_core/FluidX3D/src/setup.cpp` 的 `VkInletFaceMode` 和 `VonKarmanInletUpdater`。新增或调试 VK 入口策略时，应优先确认日志中的 `VK inlet target`、`VK inlet face` 和频谱诊断，不要只看切片图。

## Coriolis 与高空出口 u 分量

北京 debug case 中 `angle=0` 表示来流从 `+y/north` 向 `-y/south`，主流速度主要体现在 `v` 分量。若 `coriolis_term=true`，北半球向南气流会向右偏转，在当前坐标中表现为负 `u` 分量；该偏转在高空、出口附近沿程累积后最明显。

四侧面默认 VK case 的 0 度对照显示：

- `coriolis_term=true`：1200 m AGL、south/outlet 0-5 km 内 `u_mean≈-1.13 m/s`，但 `uvw_mean≈7.49 m/s`，因此图上 `u` 的低值区不是总风速低速区。
- `coriolis_term=false`：同一区域 `u_mean≈-0.02 m/s`，`uvw_mean≈7.51 m/s`；高空出口的 `u` 负值区基本消失。

因此遇到出口高空 `u` 图上的明显低值区时，应优先检查 `v/uvw` 和 `coriolis_term`，不要直接解释为出口低速或 VK 波纹。

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

CULBM 是保留 FluidX3D 来源与许可证的派生实现，不是 clean-room 实现。其已适配
宿主求解器源码固定在 `core/cfd_core/CULBM/src/solver/`，CUDA 与 MPI 后端代码位于
`core/cfd_core/CULBM/src/`。CMake 直接编译这些包内源码，不得恢复从
`core/cfd_core/FluidX3D` 或其他相邻源码树复制、staging、补丁生成的构建流程。
许可证全文位于 `core/cfd_core/CULBM/THIRD_PARTY_LICENSES/FluidX3D-LICENSE.md`，
LUW 许可证也保存在同一目录，派生关系和第三方说明位于
`core/cfd_core/CULBM/NOTICE.md`。

默认 MPI-enabled 构建在普通直接启动时不会初始化 MPI，仍走单进程本地任意多卡
路径；检测到 `PMI_RANK`、`PMIX_RANK`、`OMPI_COMM_WORLD_RANK` 等 launcher rank
标记时可自动初始化。MPI 标准不保证这些环境标记，因此生产 MPI 命令应显式设置
`CULBM_MPI=1`。LBM 对象在构造时固定并缓存 distributed 状态，不要在构造后才
初始化 MPI。MPI 运行时要求 `MPI_THREAD_MULTIPLE`；求解通信使用独立
communicator 和进程内串行锁。

生产 CULBM 不定义 `GRAPHICS`，不创建图形快照/halo，也不写出离线图像帧；LodePNG
源码、PNG 纹理解码和帧文件导出均不应恢复到默认数值路径。保留的交互显示源码不是
受支持的 headless 生产功能，若未来显式启用，仍须保证其通信与活动物理场隔离。

历史 staging 补丁仅作为本地回归材料归档在
`core/cfd_core/CULBM/test-output/legacy-staging-pipeline/`，不属于活动源码或发布
内容。后续 CULBM 功能修改应直接落在 `src/solver/`、`src/` 或 `CMakeLists.txt`。

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

TRT 动量碰撞保持标准 magic parameter `Lambda=3/16`，kernel 与主机诊断应共享同一
常量，不得为掩盖不稳定而静默 cap `omega-`。启动信息打印 `tau+/-`、`omega+/-` 和
碰撞本征幅值；`|1-omega|>0.98` 只表示每次碰撞不足 2% 的幅值衰减并触发
case-dependent 告警，不是稳定性证明。`CULBM.TemperatureAnalyticGPU.TRT` 只锁定短时
Guo/温度/边界实现；在 `tau+`、`tau_T` 都逼近 0.5 的 GEA 长时算例中 TRT 仍可能失稳，
不能以短时回归宣称长期可靠，也不能用 TRT 控制组替换默认 SRT 的物理验证。

### CULBM CUDA 高度嵌套网格

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

- 普通 `.luw` 全流程走 `makeluw.py -> bridge_core/3_voxelization.py`。该脚本构造的
  基础板为 `z=0 .. base_height`；无 DEM 时地面是 `z=base_height`，有 DEM 时本地地面是
  `base_height + relative_dem(x,y)`。DEM 的绝对海拔会先减去最低值，插值网格也会再次
  归一到本域最低值为 0。因此普通模式 STL 的 `z=0` 是基础板底面，不是地面。
- `.luwdg` 与 `.luwpf` 的 GUI 全流程走 `datagen_core/dgPrepare_stlinput.py`。它在原始
  网格最低面下方增加厚度为 `base_height` 的基础板，再整体平移到 STL 最低 z 为
  `-base_height`；原始网格最低面因而落在 `z=0`。这一路径中，`z<0` 才是人工基础板。
- 排查 CULBM 的 VTK 高程时，必须先根据实际生成器确认上述模式。普通 raw/avg VTK 还会
  保留压缩后的固体地基层；嵌套输出则会裁掉该层，不能把数组第 0 层直接解释成地面面。

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
