# Native `luwbc`

`luwbc` is the C++ replacement for `core/bridge_core/1_buildBC.py`. It reads the
existing deck and NetCDF formats directly; it does not start Python or import a
Python runtime.

## Terrain compatibility layers

1. Flat terrain is always available and has no GDAL dependency.
2. A build with GDAL/OGR reads the first `terrain_db/*.shp` and the legacy
   elevation-field aliases (`elevation`, `Elevation`, `ELEVATION`, `height`,
   `Height`, `z`, `Z`). If no shapefile is present, the first GeoTIFF is read
   directly. Both inputs are transformed to the selected WGS84 UTM CRS.
3. Terrain is interpolated with the legacy 12-neighbour inverse-distance rule.
   Boundary wind is sampled terrain-first on native NetCDF columns, followed by
   quadratic Wendland-C2 moving-least-squares interpolation to the output grid.
4. `proj_temp/dem_grid.pkl` remains readable by the legacy voxelizer. It is a
   protocol-3 pickle dictionary containing C-order float64 `dem_grid[ny,nx]`,
   float64 `x_grid[nx]` and `y_grid[ny]`, scalar `dx`, `dy`, `base_height`,
   `rotate_deg`, `x_min`, `y_min`, `x_max`, `y_max`, and the two-float tuple
   `pivot_xy`.

When GDAL is not compiled in, an existing DEM is reported and the command uses
the flat-terrain path. This keeps inspection-only builds usable without silently
claiming that terrain was loaded.

## CPU/GPU execution and reusable plans

The backend is selected only by the shared `preprocess_use_gpu` and
`preprocess_gpu_device` policy (or its dispatcher-owned CLI overrides). CPU is
the default. An explicit GPU request selects a zero-based GPU-only OpenCL
device and fails if that device, FP64, memory, program build, or execution is
unavailable; it never substitutes a CPU result.

The projected wind grid, horizontal barycentric map, vertical axis map, DEM
neighbour map, and terrain-first MLS map use exact, bounded LRU caches. Keys
include the normalized deck scope and exact double bit patterns; terrain GPU
plans also include the selected device. A coordinate/grid/axis/device change
therefore invalidates the relevant entry, and CPU plans cannot satisfy a GPU
request. Current NetCDF compatibility intentionally consumes `Time=0`; plan
reuse accelerates sequential timestamp files/calls without inventing a new
multi-time output naming contract.

CPU horizontal and vertical interpolation process U/V/W[/T] as fused batches.
The GPU path additionally uses two independent in-order queues and two bounded
slots so uploads, kernels, and downloads of adjacent chunks can overlap. DEM
IDW and Wendland-C2 MLS execute in FP64 on the selected device while VTK host
neighbour discovery remains the deterministic source of index/tie order.

## Compatibility regression

`native.luwbc` generates and retains both a generic tiny NetCDF case and a WRF
staggered-grid case under the build tree. It checks TC-to-K conversion, U/V/W
destaggering, W export, VTI opt-in output, six-patch ordering, timestamped CSV
copying, top-height calculation, deck rewrite, and legacy completion output.

The generic flat fixture is also compared byte-for-byte with the Python
reference CSV during migration validation. The WRF fixture intentionally covers
a legacy coordinate-construction defect. For a two-point mass dimension and a
three-point staggered dimension, Python creates index coordinates `0..1` and
`0..2`; after dense interpolation, the staggered adjacent-pair midpoints span
`0.005..1.995`. Reinterpolating mass-grid `XLONG/XLAT`, which only span `0..1`,
onto those midpoints makes the right 20000 of 40000 points NaN and ultimately
raises `Not enough valid points for triangulation after dropping NaNs`.

Native `luwbc` averages adjacent U/V/W samples by array position and retains the
`XLONG/XLAT` mass grid. The non-constant regression locks U first = `(2+4)/2 =
3`, V first = `(-6-2)/2 = -4`, and top W = `(3+5)/2 = 4`, while requiring the
valid WRF fixture to complete. Reproducing the legacy crash is deliberately not
offered as a compatibility mode.

2026-07-20 的 full-GDAL Release `native.luwbc` 还覆盖了 GeoTIFF 发现、20% 扩展
裁切、`terrain_db/<casename>.shp` 缓存写出、4 个 DEM polygon 只读回载及后续边界
条件生成；该项已随 30/30 串行生产 CTest 通过。NaN elevation 在 Shapefile 写出
边界被转换为与 Fiona 相同的 DBF NULL，正负无穷仍保留。

本轮回归还强制水平和竖直 OpenCL 流水各跨越至少三个 chunk，验证 GPU 0/1、同 deck
缓存命中、网格/竖直轴/跨 deck 失效，以及 CPU/GPU 平地和 terrain 最终 CSV
逐字节一致。32785 个目标点的 Release 合成核心基准为：DEM IDW CPU/GPU 2/3 ms，
MLS CPU/GPU 188/12 ms。该数字包含当前测试调用开销，但不是 NetCDF/GDAL/CSV 完整
项目墙钟；真实城市级 break-even、峰值 RAM/VRAM 和跨厂商 OpenCL 仍需单独验收。
