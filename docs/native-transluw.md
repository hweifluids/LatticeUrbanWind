# `transluw` 原生兼容说明

## 范围与历史基线

审计基线提交已经删除 `transluw`，但其 README 仍把它列为公开命令。本次实现以
原仓库删除提交 `3a57c6b` 的父提交中的
`core/tools_core/transluw.py`、`bin/transluw` 和 `bin/transluw.ps1` 为行为基线，
恢复 `luw transluw` 以及 POSIX/PowerShell 同名 wrapper。生产路径是 C++20 + VTK，
不会启动、嵌入或查找 Python。

## 输入发现与输出

命令接受一个 legacy VTK 或文本 deck：

```text
luw transluw C:\case\conf.luw
luw transluw C:\case\proj_temp\vtk\20260719_raw_u-000000123.vtk --tailname review
```

- 单 VTK 模式从输入父目录取得 case 名；文件名匹配
  `<datetime>_raw_u-#########`、`<datetime>_raw_rho-#########` 时分别使用 `uvw`、
  `rho` 前缀，否则使用 stem 与 `mesh`。
- deck 模式继续使用旧脚本的大小写不敏感文本正则读取 `casename`、`datetime`，
  并在 `proj_temp/vtk` 中分别选择数值最大的精确 9 位 U/RHO step。
- U 存在时始终作为合并基底；另一数据集的全部 point/cell arrays 被复制，重名时
  分别追加 `_u` 或 `_rho`。只按点数与单元数执行旧拓扑兼容检查。
- 共用 offset 固定为 `(-xmin, -ymin, -zmin-50)`。结果写入 `RESULTS`，命名为
  `{uvw|rho|uvwrho}-<case>_<datetime>[_tail].vtk`；单文件 fallback 使用 `mesh`。
- `--absonly` 只在 deck 模式跳过 woDEM；`--tailname` 与旧脚本一样先去除首尾空白。

读取与写出由 VTK legacy I/O 完成。`STRUCTURED_POINTS`/`vtkImageData`、
`RECTILINEAR_GRID` 和所有 point-set legacy 数据集分别通过 origin、coordinate
arrays 或显式 points 平移；原 scalar dtype、components、point/cell association 和
基底拓扑均被保留。输出仍是 binary legacy VTK，但 header/array 排列或压缩外的字节
布局不承诺与 PyVista writer 逐字节相同。

## DEM 与安全边界

存在 `proj_temp/dem_grid.pkl` 且未指定 `--absonly` 时，原生路径复现旧步骤：

1. 读取 `dem_grid`、`x_grid`、`y_grid`、`x_min`、`y_min`；
2. 抵消 `x_min/y_min`，在相对 XY 范围内使用 12 邻点、power=2 的 IDW；
3. 先把 cell arrays 转为 point arrays，再计算 `new_z=z-dem_height`；
4. 丢弃 `new_z<0` 的点，并写
   `{prefix}-<case>_woDEM_<datetime>[_tail].vtk`；
5. 输出与 PyVista 最终 `extract_points(..., include_cells=False)` 相同的
   `UnstructuredGrid`：每点一个 `VTK_VERTEX`，包含 `vtkOriginalPointIds`、全部可对应
   point arrays 和 float64 `dem_height`。

pickle reader 只实现旧 `luwbc` 所写 protocol 3 与近期 protocol 4 的 NumPy
float32/float64 C-order ndarray/dict 子集。`GLOBAL` 从不执行，`REDUCE` 只允许 NumPy
`_reconstruct` 和 `dtype`；恶意 `os.system` fixture 会以 warning 被拒绝，主 VTK
仍按旧 DEM 异常隔离行为成功保留。这是相对 Python `pickle.load` 的有意安全收紧。

所有输出先通过排他创建保留目标名。已有目标会明确失败且原字节保持不变；旧
PyVista `.save()` 会覆盖同名文件，这是为满足本次重写的数据保护约束而保留的有意
差异。写出中途失败时，新建的部分文件会留下供诊断，不会通过删除进行“清理”。

## 验证证据

`native.transluw` 在独立 `build-transluw-smoke` Release 构建中通过，覆盖：

- U/RHO 最新 step、point/cell array 全量合并及重名 suffix；
- 单 U、单 RHO、fallback 单文件命名，`--absonly`、CLI 错误和拒绝覆盖；
- protocol-3 DEM、cell-to-point、顶点拓扑、`dem_height` 与恶意 GLOBAL/REDUCE；
- 16 个对称 DEM 源点、半格查询和第 12 邻居距离并列边界。

历史脚本通过原仓库 Git object 在内存中 `compile/exec`，使用原 `.venv` 的
`python -B` 作为只读 oracle；新仓库没有生成 `.py/.pyc/.pyo`。标准与 16 点
fixture 的 4 对 native/Python 输出均具有相同 dataset type、点/单元拓扑、array
集合、dtype 和数值。标准 fixture 坐标及 `dem_height` 最大绝对差为 0；对称 k=12
fixture 的 `dem_height` 最大绝对差为 `1.7763568394002505e-15`。fixture 保留在
`build-transluw-smoke/tests/native/transluw-retained-fixtures`。另一次同 fixture 运行在把
native/oracle case 根路径归一为 `<CASE>` 后，6 行 stdout 逐字符一致。

上述证据是小型确定性 fixture；尚未用真实多 GiB solver 输出做生产规模吞吐基准。
实现使用 VTK 原生内存数据、`O(log N)` KD-tree 邻点查询和 OpenMP 点级并行，不引入
Python/PyVista/SciPy 的对象与导入开销，但生产规模“不弱于旧 Python”仍应由真实案例
基准最终确认。
