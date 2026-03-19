# LUW Studio GUI

`./gui` 提供一个独立的 C++ GUI 工程，面向 `LatticeUrbanWind` 的三种运行模式：

- `*.luw`
- `*.luwdg`
- `*.luwpf`

当前工程基于 `Qt6 Widgets + VTK`，目标是提供：

- 工程树导航和参数面板
- deck 文件结构化编辑与原始文本同步
- 前处理/求解/后处理命令编排与 console 转发
- 内嵌 VTK 可视化与基础后处理
- DG / PF 批量边界条件专用展示区域
- `cut_vis.py` / `visluw.py` / `vtk2nc.py` 集成入口

## 依赖

- Qt 6.5+
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
- `.venv/`

GUI 会优先在仓库内寻找 Python、FluidX3D 可执行文件和后处理脚本。
