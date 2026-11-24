# 原生 `visdem`

`visdem` 是旧 `core/tools_core/visdem.py` 的 C++20 替代。生产路径使用 GDAL
完成 GeoTIFF 读取、重投影和 PNG 编码，并使用原生位图渲染器生成图表；它不启动、
嵌入或依赖 Python、Rasterio、NumPy、Matplotlib：

```text
luw visdem [path_to_tif]
visdem [path_to_tif]
```

未提供参数时仍使用旧脚本的固定文件名
`AST14DEM_00408232025023230_20251011022044.tif`。提供多个参数时只读取第一个，
额外参数与旧脚本一样被忽略。命令在尝试打开文件前先打印
`Loading DEM file: <原始参数文本>`。

## 计算契约

1. 以 read-only 模式打开 raster band 1，并按旧顺序打印 shape、CRS、bounds、
   resolution 和 nodata metadata。
2. 数值等于 `-9999` 或等于数据集声明 nodata 的像元转为 NaN；其余像元转为
   float32。
3. 使用 GDAL 的建议输出网格重投影到 EPSG:4326，重采样为 bilinear，源与目标
   nodata 均为 NaN。这与旧 Rasterio `calculate_default_transform` 加
   `reproject(..., Resampling.bilinear)` 使用同一 GDAL 网格/warp 语义。
4. 对有效像元打印最小值、最大值、均值、总体标准差、有效数量和百分比。
5. 总览图保留 terrain elevation、方位角 315°/高度角 45°/垂直放大 2 的
   hillshade、两者 overlay，以及 50-bin elevation histogram 和统计框。
6. 3D 图的抽样步长仍为 `max(1, min(height, width) / 200)`（整数除法），并保留
   terrain 色阶和经纬度/高程轴语义。

## 输出文件与安全边界

输出名严格沿用旧脚本对**完整输入字符串**执行大小写敏感的全局替换：

```text
<input>.replace(".tif", "_visualization.png")
<input>.replace(".tif", "_3d_visualization.png")
```

因此常规 `dem.tif` 生成 `dem_visualization.png` 和
`dem_3d_visualization.png`；路径中每一处小写 `.tif` 都会被替换。原生实现不会
覆盖任何已有输出。若旧命名规则让两个输出互相冲突，或像 `DEM.TIF`、无 `.tif`
后缀那样使输出可能等于输入，命令会报错并退出 1。这里有意比旧脚本安全：旧
`savefig` 路径可能覆盖输入或已有图片。`.tiff` 会沿用旧字符串替换得到 `.pngf`
类路径，原生 PNG writer 会明确拒绝该不支持的扩展名。

输入始终用 `GDAL_OF_READONLY | GDAL_OF_RASTER` 打开。专项测试比较运行前后的
GeoTIFF 字节和修改时间；再次运行也会验证两个已有 PNG 的字节保持不变。当前
检查与写出之间仍是普通文件系统的先检查后创建，多个进程同时争用同一路径时不
提供跨进程事务保证。

## 图像兼容范围

- 总览固定为 4770×3570 RGBA PNG，保持旧 `figsize=(16, 12)`、300 DPI 与
  `bbox_inches="tight"` 在测试 fixture 上的像素尺寸和 2×2 面板语义。
- 3D 图固定为 2955×2498 RGBA PNG，保持同一 fixture 的旧输出尺寸、surface、
  colorbar 和三个坐标轴语义。
- 原生位图字体、抗锯齿、色彩插值和 PNG 压缩字节不宣称与 Matplotlib 逐像素
  相同；这部分差异不改变重投影数组、统计量、面板内容、色阶端点或输出尺寸。
- 旧脚本的两次 `plt.show()` 依赖图形后端，可能打开并阻塞交互窗口。原生命令只
  生成确定性的静态 PNG，不打开窗口；这是为无 Python/无显示器批处理保留的明确
  行为差异。

渲染循环可由 OpenMP 并行执行。总览 PNG 写出后会先释放约 65 MiB 的 RGBA 画布，
再分配约 28 MiB 的 3D 画布，不会同时保留两张高分辨率图；DEM 数据与 hillshade
使用连续原生数组，不经过 Python object 或解释器调度。

在同一 retained 6×5 fixture 上各冷启动一次、分别写入全新目录的保守 wall-clock
检查中，Release 优化的原生入口为 6.167 s，旧 `.venv` 的 `python -B` 加 Agg
后端为 7.025 s，原生耗时低 12.2%。该小 fixture 主要衡量固定高分辨率 PNG
渲染和进程启动，不是大 DEM 的统计基准；专项原生入口还使用 Debug GDAL DLL 且
未链接 OpenMP，因此这个结果只证明当前受测路径没有观察到性能回退。

## 验证

`native.visdem` 在忽略的构建目录生成 6×5、EPSG:32630、100 m 分辨率、含一个
`-9999` nodata 的 GeoTIFF。旧 oracle 使用 Rasterio 1.5.0 所带 GDAL 3.12.1，
原生专项 Debug 测试使用 GDAL 3.12.4；两端重投影均为 4×7，控制台 metadata、
经纬度范围、27/28 有效像元及全部统计提示逐字一致。测试还完整解码两张 PNG，
检查尺寸、四通道、只读保证、已有输出拒绝和大写 `.TIF` 输入保护。无 GDAL 的
inspection build 仍会编译入口并明确提示需启用 geospatial 支持。

同一 `native.visdem` 已在 2026-07-20 的 full-GDAL Release 生产构建中再次通过，
并纳入 30/30 串行总门禁；上述小 fixture 性能范围和图像差异说明仍然适用。
