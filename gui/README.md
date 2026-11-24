# LUW Studio GUI

`./gui` 提供一个独立的 C++20 GUI 工程，面向 `LatticeUrbanWind` 的三种运行模式：

- `*.luw`
- `*.luwdg`
- `*.luwpf`

当前工程基于 `Qt6 Widgets + VTK`，目标是提供：

- 工程树导航和参数面板
- deck 文件结构化编辑与原始文本同步
- 前处理/求解/后处理命令编排与 console 转发
- 内嵌 VTK 可视化与基础后处理
- DG / PF 批量边界条件专用展示区域
- `cut_vis` / `visluw` / `vtk2nc` 原生命令集成入口

## 依赖

- Qt 6.8+
- VTK 9.x，需包含 `GUISupportQt`
- CMake 3.24+
- 可用的 C++20 编译器

## 构建

```powershell
cmake -S gui -B gui/build
cmake --build gui/build --config Release
```

## 运行

构建后的可执行文件需要从仓库环境中运行，以便正确定位：

- `core/`
- `bin/`
- `examples/`
- `core/deck_schema.json`

GUI 通过原生 `luw` 调度器运行前处理、求解和后处理，不启动或嵌入 Python。求解器按 deck 的 `cfd_solver` 选择：字段缺失或值为 `culbm` 时使用活动的默认 CULBM；只有显式写为 `fluidx3d` 且已通过 `LUW_ENABLE_FLUIDX3D=ON` 单独安装该后端时，才使用冻结的可选 FluidX3D。所选可执行文件缺失时明确失败，不回退到另一求解器。

SurfaceIBM 仅由 CULBM 支持，`ibm_enabler` 默认为 `false`。GUI 会阻止在 FluidX3D 被选中时启用 IBM，也会阻止 IBM 已启用的 deck 切换到 FluidX3D。仓库保留的四个 CULBM Python 分析脚本只用于离线验证，不属于 GUI 或生产运行链。
