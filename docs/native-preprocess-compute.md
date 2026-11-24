# 前处理 CPU/GPU 统一计算策略

## 公共配置

所有原生前处理计算核心共用两个 deck 字段：

```text
preprocess_use_gpu = false
preprocess_gpu_device = 0
```

- `preprocess_use_gpu=false`：使用各算法的 CPU 实现，也是默认值。
- `preprocess_use_gpu=true`：所有已经实现 GPU 后端的前处理核心必须使用 OpenCL GPU。
- `preprocess_gpu_device`：非负整数，表示 OpenCL 运行时中全局、从 `0` 开始的
  **GPU-only** 设备号。CPU 类型的 OpenCL device 不参与编号。

设备编号先按 `clGetPlatformIDs` 返回的 platform 顺序，再按各 platform 的
`clGetDeviceIDs(..., CL_DEVICE_TYPE_GPU, ...)` 顺序扁平化。例如，第一个 platform
有 GPU 0、GPU 1，第二个 platform 有一个 GPU，则其全局编号依次为 0、1、2。
驱动或设备安装发生变化后，OpenCL 枚举顺序也可能变化，因此生产配置应在环境变化后
重新核对 `luw detect_opencl` 报告及计算命令打印的已选设备名称。

`preprocess_gpu_device` 即使在 CPU 模式也必须是合法的非负 32 位整数，但只有
`preprocess_use_gpu=true` 时才会创建对应设备的 context、queue 和 kernel。

## CLI 与优先级

支持公共前处理选项的命令为 `makeluw`、`luwbc`、`luwcut`、`luwvox`、`luwval`
和 `dgprepare`。两种等价拼写如下：

```text
luw makeluw C:\case\conf.luw --preprocess-use-gpu true --preprocess-gpu-device 1
luw luwvox --preprocess-use-gpu=true C:\case\conf.luw --preprocess-gpu-device=1
```

公共选项可位于 `--` 分隔符之前的任意位置；`--` 之后的参数不再由统一 dispatcher
消费。布尔 CLI 值接受 `true/false`、`1/0`、`yes/no` 和 `on/off`。缺值、非法布尔值、
负设备号或整数溢出属于 CLI 错误，返回状态 `2`，计算核心不会启动。

统一 dispatcher 是这两个公共 CLI 参数的唯一解析入口。各命令 handler 只接收解析后的
override，不得再次解析同名参数、增加命令局部别名，或建立另一条设备选择旁路。

取值优先级固定为：

1. 命令内部显式 override（仅供原生嵌入调用和测试）；
2. 本次调用的公共 CLI 选项；
3. deck 中的 `preprocess_use_gpu` / `preprocess_gpu_device`；
4. 默认 `false` / `0`。

`makeluw` 将本次调用的显式 CLI override 传给 `luwbc`、`luwcut`、`luwvox` 和
`luwval`；`cdfinspect`、`shpinspect` 是只读检查阶段，不接收计算设备选项。deck
仍是所有子进程可共同读取的持久配置。工作流继续记录各阶段结果；任一子步骤失败时，
工作流最终返回 `1`，不得再显示为整体成功。

## 算法与设备正交

地形插值算法只允许：

```text
terr_voxel_approach = idw
terr_voxel_approach = kriging
```

算法不再通过名字携带 CPU/GPU 信息。`kriging_gpu` 不是兼容别名，也不是有效配置；
需要 GPU kriging 时应使用：

```text
terr_voxel_approach = kriging
preprocess_use_gpu = true
preprocess_gpu_device = 0
```

同理，IDW 是否使用 GPU 只由统一计算策略决定。命令行 `--terrain-approach` 只选择
`idw` 或 `kriging`，不能覆盖或暗示计算设备。

Kriging 在 CPU 与 GPU 后端使用相同的邻点数契约：`terr_voxel_idw_neighbors` 必须为
`1..16`。超过 16 是配置错误，不能在某个后端截断或使用另一套上限。IDW 仍接受正整数，
其可执行规模由输入和可用内存约束。

## 严格 GPU 失败语义

显式 GPU 模式没有 CPU fallback。以下任一情况都必须在相应 GPU 阶段写入 stderr、
返回状态 `1`，并且不得用 CPU 结果冒充成功：

- 构建时没有 OpenCL headers/loader；
- 没有 `CL_DEVICE_TYPE_GPU` 设备；
- 全局 GPU 设备号越界；
- context、queue 或 program/kernel 创建、编译失败；
- 所选设备缺少该 kernel 要求的能力或显存；
- buffer 分配、kernel 执行或结果读回失败。

GPU kernel 内部为数值稳定性执行的分支仍属于 GPU 计算。例如克里金线性系统奇异时，
在同一 OpenCL kernel 内执行 IDW 数值分支不构成 CPU fallback。

`luwbc` 的融合水平/竖直重采样、DEM IDW 与 terrain-first MLS 使用 `double` 权重和
分数，以保持 CPU 路径的计算语义。`luwvox` 的 DEM k-NN 也使用 FP64
`(distance², source index)` 稳定排序，地形三角网格使用 double 坐标。因此这两个命令
的 GPU 后端都要求所选设备提供 FP64。缺少 FP64 是设备能力错误，返回 `1`，不得改用
单精度邻域排序或 CPU 结果。

部分阶段因计算性质明确保留 CPU。它们仍解析并打印统一 policy，同时明确提示本阶段
没有 GPU kernel、实际在 CPU 上运行；这属于该阶段没有 GPU 实现，而不是 GPU 后端
失败后的静默降级。`detect_opencl` 和 `accelerator_report` 只是诊断，不能替代每个
GPU 计算核心自己的设备与 kernel preflight。

## 本轮实现与验证状态

本轮已完成以下门禁：

- full-dependency Release 主程序与 GUI 构建成功，串行 CTest `30/30` 通过；
- 公共 deck/CLI 解析、优先级、作用域恢复、dispatcher 唯一解析入口、无命令局部旁路，
  以及 `makeluw` override 传递和失败退出状态已纳入门禁；
- 全局 GPU-only 设备选择、显式 GPU 失败不降级 CPU，以及相关错误路径已通过测试；
- `luwvox` 已把规则 DEM 直接索引、不规则 DEM uniform-bin CSR + GPU ring k-NN、
  Kriging/IDW、可分离 Gaussian 与固定顺序地形三角网格接成设备驻留、分块流水；
  GPU-only 设备 0 和 1 均通过稳定 `(distance²,index)`、数值、跨 65536 分块和网格顺序
  验证；
- `luwbc` 已加入投影/水平/竖直/DEM/MLS 精确有界映射缓存、CPU 多字段批处理、GPU
  DEM IDW、terrain-first MLS，以及水平/竖直双 queue 双 slot 流水；GPU-only 设备 0
  和 1 均通过数值和错误路径测试，测试还强制覆盖至少三个流水分块；
- 同 deck 连续时间戳运行会复用映射计划；输入坐标、网格、竖直轴、deck scope 或 GPU
  设备变化会精确失效，CPU/GPU terrain plan 不交叉复用；
- `dem_tif_cache` 按源块对齐分条、双缓冲读取，不再为裁切区同时保留多份完整数组；
  `luwvox` 的直接 GeoTIFF 路径也改为约 8 MiB 多行块、矩形快筛/prepared geometry
  和分块投影；
- fresh-prefix `build-production-install-20260720_071053` 已通过 `luw --help`、
  `luw luwbc --help`、`luw luwvox --help`、`luw detect_opencl` 和 GUI offscreen
  runtime self-check。

本机 Release 合成基准（不是完整案例）给出的墙钟为：`luwvox` 64×64 DEM → 128×128
查询、k=8 时 GPU 5.89 ms、同工作量 CPU 暴力参考 437.83 ms；`luwbc` 32785 目标点时
DEM IDW CPU/GPU 2/3 ms、MLS CPU/GPU 188/12 ms。它说明 k-NN/MLS 有明显加速潜力，
也说明很小的 IDW 会被传输/调度开销抵消；不能把这些 kernel/core fixture 数字当成真实
城市项目端到端加速比。

真实生产规模 TIF/PKL、WRF/CF、峰值 RAM/VRAM、冷/热启动、端到端“不弱于旧
Python”的性能，以及 AMD/Intel OpenCL 设备仍未形成完整证据。`luwvox` Kriging 的
设备邻点缓冲按查询分块有界，但为完全保留旧 range 的精确中位数，主机仍需要
`O(query_count × neighbor_count)` positive-distance 临时数组；超出容器/主存能力会
明确失败，不使用近似中位数。这些限制不能由 `30/30` 小型/定向测试或安装自检替代。

## 保留 CPU 的低收益阶段

- **建筑拉伸**：Earcut 三角化、洞、多边形修复及每栋建筑不同长度的顶/底/侧面输出
  分支密集，最终还要回主机顺序写 STL；现有逐建筑 OpenMP 更适合该结构。
- **`luwcut` / `buildingscale`**：主要成本在 GEOS MakeValid、intersection、union
  和最小旋转矩形。拓扑健壮性、要素顺序与变长 geometry 很难由通用 OpenCL kernel
  保持，GPU 化也无法避免 GEOS 和 Shapefile I/O。
- **`dgprepare`**：核心是 Manifold boolean 和 STL/VTK 转换；应优先使用 Manifold/TBB
  的 CPU 并行，而不是另写一套高风险布尔几何 kernel。
- **`dem_tif_cache`**：受 GDAL RasterIO、裁切、polygonize 和磁盘 I/O 主导；已采用
  源块对齐分条与双缓冲，而不是为它建立收益很低的 OpenCL polygonize 后端。
- **inspect / validate / deck / wrapper**：属于解析、诊断和编排，计算密度不足以抵消
  OpenCL 初始化及数据传输。

这些阶段不会为了形式上的“GPU 覆盖率”重写 GEOS、GDAL、Manifold、PROJ 或文件
编码库。只有真实端到端 profile 证明收益显著且数值/行为契约可锁定时，才重新评估。
