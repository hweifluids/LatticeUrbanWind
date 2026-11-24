# 原生 `buildingscale`

`buildingscale` 已由 C++20、GDAL/OGR/PROJ 实现，生产运行不启动或嵌入 Python。

## 用法

```powershell
luw buildingscale C:\case\buildings.shp
luw buildingscale C:\case\buildings.shp --out C:\case\scale.png
```

默认输出为输入目录中的 `<shp_stem>_short_side_cdf.png`。画布保持旧脚本的
10×3 英寸、300 dpi 语义，即 3000×900 像素。显式 `.png` 使用 PNG，`.jpg` 和
`.jpeg` 使用 JPEG；显式路径没有后缀时，和 Matplotlib 一样把实际文件写为
`<path>.png`，完成提示仍显示用户给出的原路径。其他 Matplotlib 格式目前会明确
报“不支持”，不会以错误后缀静默写入 PNG 内容。

为避免破坏用户结果，原生命令发现实际输出目标已存在时会拒绝覆盖。这一点有意
不同于旧 Python 脚本的 `savefig` 覆盖行为；需要重新生成时应选择一个新的
`--out` 路径或由用户自行处理旧文件。

## 计算契约

- 缺失 CRS 会以旧脚本文字
  `Input data has no CRS. Please set a CRS before measuring lengths.` 失败。
- 地理 CRS 先对修复后的全部面做 union centroid，再以同一公式选择南/北半球
  UTM EPSG；投影 CRS保持原坐标。
- 只接受 Polygon/MultiPolygon；MultiPolygon 的面积和最小旋转矩形短边都只取
  最大面积子面。
- 高度字段先按 Shapefile 列顺序匹配旧脚本的精确候选名，再按列顺序匹配包含
  `height` 或 `hgt` 的字段；不存在时使用 1.0。
- 权重为最大子面 footprint area × height。非有限、非正短边或非正权重会被过滤。
- PDF 使用 NumPy 的无权重 Freedman–Diaconis 分箱，再对 area×height 权重做
  histogram density；CDF 按短边降序累计相同权重。横轴保持从最大值反向到 0，
  并保留 2、5、10、20、50 m 参考线。

## 依赖门禁

关闭 `LUW_ENABLE_GEOSPATIAL` 或找不到 GDAL 时，帮助和纯 C++ 数值测试仍可用；
实际 Shapefile 分析会明确提示必须以 GDAL/OGR（含 GEOS）重新构建。生成式测试
在 full-GDAL 构建中创建独立 Shapefile，验证投影/地理 CRS、字段优先级、旋转
矩形、MultiPolygon、输出防覆盖和 PNG/JPEG 尺寸。

该 full-GDAL 分支已在 2026-07-20 的生产 Release 总构建中随 30/30 CTest 通过；
这不扩展上文明确列出的输出格式，也不表示原生字体/像素与 Matplotlib 相同。
