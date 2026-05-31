# AGENTS_PROJECT.md

## 项目级持续说明

本文件记录对后续 agents 有持续价值的项目知识。若后续任务发现新的、可复用的构建方式、运行方式、排查结论、约定或长期注意事项，应更新本文件；不要只把这类信息写进一次性的 `codexlog/`。

## 代码变更版本日期

每次修改会影响运行行为的代码后，应同步更新求解器启动横幅中的版本日期。位置：

```text
core/cfd_core/FluidX3D/src/setup.cpp
```

当前格式为：

```text
Version - v3.5-YYMMDD
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
