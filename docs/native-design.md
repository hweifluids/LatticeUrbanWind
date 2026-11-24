# 原生重写设计约束

## 兼容边界

1. deck 读取保留 UTF-8 容错、行尾归一化、别名、引号内 `//`、重复键末项覆盖、未知键保留及 schema 顺序渲染。
2. 原命令名称、参数顺序、工作目录、阶段顺序、进度帧格式和关键控制台文本保持兼容。
3. NetCDF、Shapefile、CSV、STL、legacy VTK 与 deck 文件继续使用原文件名、坐标约定、维度顺序和数值单位。
4. CPU 密集计算使用预计算插值结构、分块处理和受控并行；不通过启动解释器或创建脚本子进程执行数值计算。
5. CULBM 是活动开发且默认选择的 CUDA 求解器；FluidX3D 是默认不暴露、仅以 `LUW_ENABLE_FLUIDX3D=ON` 安装并显式选择的独立冻结 OpenCL 求解器。两个 CFD 模块均独立于 C++20 前后处理层构建，原生调度层负责选择、定位、参数转发和输出转发。
6. `cfd_solver` 缺失或规范化值为 `culbm` 时选择 CULBM；只有 `fluidx3d` 显式值选择 FluidX3D。非法值在启动求解器前失败。
7. SurfaceIBM 只属于 CULBM，且 `ibm_enabler` 缺失时按 `false` 处理；`ibm_enabler=true` 与 FluidX3D 的组合必须在启动前拒绝。

## 依赖方向

```text
GUI -> native CLI -> common configuration/runtime
                  -> GIS pipeline (GDAL/OGR/GEOS)
                  -> VTK/NetCDF post-processing
                  -> CULBM executable (active default)
                  -> FluidX3D executable (frozen, separately installed opt-in)
```

GUI 与 CLI 共享 `[[LUW_PROGRESS]]` JSON 行协议。普通终端模式继续输出旧式文本进度；GUI 模式解析协议行而不吞掉普通 stdout/stderr。

## Python 边界

生产 CLI、GUI、前处理、编排、后处理和求解器启动链均为原生代码，不查找、启动或嵌入 Python。仓库仅保留以下四个 CULBM 离线验证/报告脚本，并由 Python-free 门禁精确 allowlist；它们不安装为生产命令，也不会被生产入口调用：

- `core/cfd_core/CULBM/tests/compare_backends.py`
- `core/cfd_core/CULBM/tests/compare_physical_vtk.py`
- `core/cfd_core/CULBM/validation/gea/analyze_results.py`
- `core/cfd_core/CULBM/validation/gea/report/build_report.py`

默认安装顺序是先运行 `installer/2_build_native.*`，再运行 `installer/3_compile_cfdcore.*`；后者只构建自包含的 CULBM。只有两个阶段都显式设置 `LUW_ENABLE_FLUIDX3D=ON` 时，才额外暴露、安装和构建独立 FluidX3D 后端。

## 验证策略

- 对 deck 使用逐字节 golden oracle。
- 对命令调度验证退出码、继续执行语义、工作目录和参数转发。
- 对 GIS/NetCDF/VTK 使用程序生成的小型确定性数据集，检查 schema、CRS、维度、字段、边界和数值容差。
- 对 STL 验证几何边界、面/体语义与下游可读性；不同库产生的合法 STL 字节排列不作为相等条件。
- 先运行受影响的最小测试，再执行完整原生 CTest；GUI 另做完整链接构建。
