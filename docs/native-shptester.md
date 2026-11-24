# 原生 `shptester`

`shptester` 是旧 `core/tools_core/shpTester.py` 的 C++20 只读替代。生产路径只使用
GDAL/OGR 与 GEOS C API，不启动或嵌入 Python：

```text
luw shptester <path_to_shp>
shptester <path_to_shp>
```

虽然参数名沿用 `path_to_shp`，读取能力与旧 `geopandas.read_file` 一样由 GDAL
驱动决定；测试也用 GeoJSON 覆盖混合 geometry type 的排序行为。

## 控制台与退出契约

- 参数个数不是 1 时向 stderr 原样打印
  `Usage: python check_degenerate_shp.py <path_to_shp>`，退出 2。这里的 Python
  字样只是兼容文本，不是运行时依赖。
- 打开或迭代失败时向 stderr 打印
  `Failed to read shapefile: <GDAL reason>`，退出 1。
- 数据集没有 geometry field 时打印 `No geometry column found.`，退出 1。
- 成功时只向 stdout 打印 feature 总数、退化总数和 reason 计数，退出 0。
- reason 按计数降序；计数相同的项保持该 reason 第一次出现的顺序，等价于
  Python `Counter.most_common()`。

## 判定顺序

每个 feature 只归入第一个命中的原因，顺序严格保留旧脚本：

1. null geometry：`null`；任意 empty geometry：`empty`。
2. `Polygon` 外环少于四点：`too_few_points`；首尾坐标不等：
   `ring_not_closed`；面积不大于零：`zero_area`。
3. 仍无问题但 GEOS 判为无效：`invalid: <GEOS reason>`。
4. `MultiPolygon` 逐 part 使用相同顺序；第一个坏 part 加 `multipart: ` 前缀。
   所有 part 通过后才检查总体面积和总体 validity。
5. 其他 geometry type：`not_polygon_type: <Shapely-style type>`，例如
   `Point`、`LineString` 或 `MultiLineString`。

GEOS reason 通过 OGR ISO WKB 直接交给 `GEOSisValidReason_r`，没有自行翻译；
例如 hole 在 shell 外时为 `Hole lies outside shell[5 5]`。这与 Shapely 的
`explain_validity` 使用同一 GEOS 诊断来源。GEOS 大版本可能调整 reason 文案，
因此原生测试固定检查典型 Polygon/MultiPolygon 文本，差分时也记录双方版本。

## 只读与计算方式

输入以 `GDAL_OF_READONLY | GDAL_OF_VECTOR` 打开。命令不创建、不修复、不覆盖
任何 Shapefile sidecar；测试在运行前后逐字节比较 `.shp/.shx/.dbf` 并比较修改
时间。feature 逐个流式读取，仅累计少量 reason 计数；一个 GEOS context 和一个
WKB reader 在整层扫描中复用，内存不会随 feature 数量或 geometry 总体积累积。

## 验证 fixture

`native.shptester` 在忽略的构建目录中生成并保留：

- Polygon Shapefile：有效 Polygon、null、零面积 Polygon、有效 MultiPolygon；
- Point Shapefile：非 polygon 类型；
- mixed GeoJSON：Point、null、LineString、null，用于验证计数相同项的首次出现
  顺序；
- 内存 OGR geometry：empty、少点、未闭合、零面积、hole 越界、坏 part 和重叠
  MultiPolygon。少点和未闭合环使用内存 fixture，避免 Shapefile 驱动读取时主动
  闭环而掩盖旧脚本分支。

没有 GDAL/GEOS 的 inspection build 仍会编译入口并保留 Usage/退出码；实际读取
会以 `Failed to read shapefile: ... unavailable in this build.` 明确失败。
