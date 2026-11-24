# `vtk_avg_to_utm_asl_nc` 原生兼容说明

## 入口与模式

原生入口为：

```text
luw vtk_avg_to_utm_asl_nc [options]
```

`bin/vtk_avg_to_utm_asl_nc` 与 `bin/vtk_avg_to_utm_asl_nc.ps1` 只转发到该原生
命令，不启动或嵌入 Python。

- `--case-dir` 处理单个 case；配置查找顺序为显式 `--config`、`conf.luwpf`、
  `conf.luwdg`、`conf.luw`。
- 不带 `--case-dir` 时按 `beijing`、`chongqing`、`shanghai`、`singapore` 的固定
  顺序发现 case，并要求各目录存在 `conf.luwpf`。
- 裸 `--cases` 产生的空列表与不传该选项相同，回退上述四城市；非空列表才筛选。
- deck 的 `cut_lon_manual`/`cut_lat_manual` 由共享 parser 先按 min/max 排序，倒序
  边界与旧 `parse_luw_config` 生成相同的旋转原点和 UTM 网格。
- 输入默认位于 `RESULTS/crop/cropped_vtk_raw_assembled`，匹配
  `ANG_*_avg-*_cropped.vtk`；排序后再应用 `--limit`。
- 输出名固定为 `<VTK stem>_utm_asl.nc`，默认写入 `RESULTS/nc_utm_asl`。

## 地形 ASL 优先级

这里保留旧脚本两个模式并不相同的优先级：

1. 单 case：有限的 `--terrain-min-asl` > `prep_summary.json` 的
   `dem_absolute_min_m` > `--range-file`/根目录 `Range.txt` 中与 deck casename
   对应的值。
2. 多 case：始终使用 Range 文件；即使传入 `--terrain-min-asl` 或各 case 存在
   summary，也不改变批处理地形值。

summary 的值继续使用旧 `float(value)` 的常见 JSON coercion：number、bool 和有限
数字字符串均可读取；NaN/Inf、null、数组、对象或不可解析字符串不作为有效地形值。
Range 解析保留同一行城市名加多个 `m` 数值时取最后一个值、城市块中的
`terrain_min_asl_m = ...` 和单独 `<number> m` 三种形式。多 case 模式仍要求 Range
中四个标准城市全部存在，即使 `--cases` 只选其中一部分。

## `--limit` 与覆盖

- 正整数 `--limit N` 对每个 case 独立取排序后的前 N 个文件。
- `--limit 0`、负数或不传该选项均处理全部文件。
- 默认遇到已有 NetCDF 时输出 `Skip existing: ...`，其字节和修改时间保持不变，
  该文件计入成功数。
- `--overwrite` 保留旧 `Dataset(..., "w")` 行为，以 NetCDF clobber 模式重建目标。

## NetCDF 契约

输出维度顺序为 `z,y,x`。变量顺序、dtype 与维度为：

| 变量 | dtype | 维度 |
|---|---|---|
| `x` | float64 | `x` |
| `y` | float64 | `y` |
| `z` | float32 | `z` |
| `easting` | float64 | `y,x` |
| `northing` | float64 | `y,x` |
| `crs` | int32 | scalar |
| `u`,`v`,`w` 与附加 scalar | float32 | `z,y,x` |

坐标与数据变量使用 shuffle + zlib 指定 level；三维数据 chunk 为
`(1,min(ny,256),min(nx,256))`，数据 `_FillValue` 为 float32 NaN。保留
`source_field`、CF wind/altitude、grid mapping、处理说明、VTK origin/spacing、
case、angle、terrain/pedestal 等旧属性。

逐层读取只保留一个源平面及输出平面；水平风按旧公式旋转：

```text
u_utm = cos(theta) * u_model + sin(theta) * v_model
v_utm = -sin(theta) * u_model + cos(theta) * v_model
```

保留 `z_vtk >= pedestal_height - 1e-6` 的层，并写为
`z_vtk - pedestal_height + terrain_min_asl`。

## 差分验证

本机旧 oracle 使用原仓库 `.venv`、`PYTHONDONTWRITEBYTECODE=1` 与 `python -B`，
fixture 保留于忽略目录
`build-oracle-retained/vtk_avg_multicase_binary_20260719_232922`。验证包括：

- 四城市、每城两个 BINARY `STRUCTURED_POINTS` avg VTK；四种 Range 文本形式、
  冲突 summary、`--limit 1` 与排序输出名；
- 单 case 数字字符串 summary；
- 5 组旧/新 NetCDF 的维度、变量顺序、dtype、chunks、filters、属性名和全部数组。

每组 47 个 NetCDF 数值（含坐标、风场和 scalar）逐值 exact equal；非动态属性除
CRS WKT 文本外一致。`source_vtk` 路径和 `history` 时间按各自运行目录/时刻不同。

旧 pyproj 链接 PROJ 9.5.1，而当前依赖精简构建链接 VTK bundled PROJ 8.1.0；两者
为 EPSG:32650 生成的 WKT scope 文案不同。原生输出已改为与 pyproj 一样的单行
WKT2 格式，且 `pyproj.CRS.equals()` 验证两个 WKT 表示同一 CRS，但不能声称跨
PROJ/EPSG 数据库版本逐字一致。生产依赖版本变化时仍应按语义而非 WKT 字节验收。

生成式 C++ 测试 `native.vtk_post` 还锁定显式 Range 文件、多 case 忽略 summary/
显式地形、单 case 优先级、`--limit 1/0`、默认 skip、受控 overwrite、schema、
压缩、chunk、NaN fill 和全部合成风场数值。

## 尚未覆盖

- 真实生产尺寸文件的吞吐和峰值内存尚未与旧脚本做同机基准；实现保持逐 z 平面
  流式读写，不随 `nz` 保存完整三维场。
- oracle 覆盖标准四城市、小网格、float32 scalar 风场；未穷举所有合法 CRS、
  VTK field 命名冲突和超大 NetCDF/HDF5 文件。
- 原生 shared legacy VTK reader 还能读取 ASCII；旧脚本会拒绝 ASCII。这是输入
  能力超集，但若调用方依赖旧版对 ASCII 立即退出的失败行为，结果不完全相同。
