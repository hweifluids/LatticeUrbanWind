# 原生 TKE / 裁剪可视化

`tke_viz` 与 `cut_vis` 对应旧仓库中内容完全相同的两个 Python 脚本，原生版共用同一套计算与渲染核心。`batch_tke_geo_viz` 对应旧版批量地理裁剪流程。

## 单文件模式

本地坐标模式不需要任何地理配置：

```text
luw tke_viz result_avg-100.vtk --skip-config --local-xy --output-dir figures
luw cut_vis result_avg-100.vtk --skip-config --local-xy --output-dir figures
```

地理模式保留 `--target-crs`、经纬度范围、完整 VTK 域范围、UTM CRS、旋转角、目标网格尺寸及 XY 覆盖参数。为兼容旧工作目录，未指定 `--skip-config` 时会读取当前目录中由普通字面量组成的 `config.py`；原生程序不会执行其中的 Python 代码。

输出文件名保持为：

- `<stem>_range_diagnostic.png`
- `<stem>_wind_arrows_9layers.png`
- `<stem>_tke_9layers.png`
- 可选的 `<stem>_cropped.vtk`

九层高度仍使用固定关系 `height=-50+10*z`，目标高度为 50、100、150、200、300、400、500、600、800 m。

## 批处理模式

```text
luw batch_tke_geo_viz C:\case\conf.luwpf
luw batch_tke_geo_viz C:\case\conf.luwpf --crop-z-max 800
```

配置可为 `.luw`、`.luwdg` 或 `.luwpf`。输入目录发现顺序、`crop_debug_input_dir` 覆盖规则、文件 glob、地理裁剪键、Z 上限和 `RESULTS/crop` 输出目录均与旧脚本一致。

## 计算与兼容说明

- 使用共享 legacy VTK reader，支持 ASCII 与 BINARY `STRUCTURED_POINTS`；绘图值与旧脚本一样转换为 float32。
- BINARY 裁剪输出逐行复制原始字段字节，因此 float64、整数、多分量标量和向量的类型及数值不发生隐式截断。
- 地理链保持 `local XY -> UTM 逆旋转 -> lon/lat -> TARGET_CRS`，风矢量同步旋转到目标地图轴；标量使用双线性插值。
- 风速与 TKE 色阶继续使用跨九层的 2%/98% 分位范围，分别使用 Turbo 与 Magma 语义；PNG 的 9×8 或 18×16 英寸乘 DPI 后的像素尺寸保持一致。
- PNG 由原生无窗口渲染器生成。字体栅格、箭头自动缩放、色图采样和约束布局与 Matplotlib 的实现不同，因此不承诺逐像素哈希一致；面板层次、字段、范围、色阶、箭头方向、色标含义和文件尺寸保持一致。
- PNG 按行压缩和写出，高 DPI 批处理不需要同时保留整张最终分辨率 RGB 图；各 VTK 层按需读取，避免加载完整三维场。

Retained 测试使用生成式 7×6×91 VTK 同时验证本地与地理模式。测试夹具上，原生与旧 Python 批处理裁剪 VTK 字节完全一致；80 DPI 单文件流程的实测约为 0.81 s（原生）对 3.34 s（Python），该数字仅用于确认迁移未造成性能回退，并非通用基准。
