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
