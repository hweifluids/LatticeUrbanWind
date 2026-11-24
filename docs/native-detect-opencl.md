# OpenCL 原生诊断

旧 `installer/3_detect_opencl.py` 已由统一的 C++20 命令替代：

```text
luw detect_opencl
detect_opencl
```

实现直接调用 OpenCL C API，不导入 `pyopencl`，也不启动 Python。报告保留旧版
标题及逐 platform/device 的 `Type`、`Vendor`、`OpenCL`、`Driver`、
`Compute Units`、`Max Clock` 和 `Global Memory` 字段；内存仍按 1024 进位并保留
两位小数。`Type` 特意保留 `pyopencl.device_type.to_string()` 的 `ALL | GPU` 等
历史显示形式。

退出码契约：

- `0`：至少发现一个 platform，并完成报告；单个平台的 device 枚举失败仍按旧版
  打印该平台错误后继续。
- `1`：platform 或 device 属性查询发生意外错误。
- `2`：当前原生构建没有 OpenCL headers/loader 依赖，命令会明确提示重新配置。
- `3`：OpenCL loader 没有返回任何 platform。

安装启动检查会在 `luw --help` 后运行该命令。状态 `3` 只产生 warning，因为没有
可用设备的主机仍可安装并在 `preprocess_use_gpu=false` 时使用 CPU 路径；状态 `1`
或 `2` 保持诊断失败。若用户显式设置 `preprocess_use_gpu=true`，安装诊断的 warning
不构成计算许可，具体计算命令仍必须硬失败。
旧同名 `installer/3_detect_opencl.cmd` 也已恢复为纯原生 launcher wrapper，并新增
POSIX `.sh`；两者保留 probe 提示、安装器 banner 和子命令退出码，不再依赖
`LUW_PYTHON`。`native.installer_detect_opencl_wrapper` 用返回 3 的假 launcher
验证参数转发和状态透传。

诊断成功不等于前处理 GPU 后端可用：`detect_opencl` 会枚举 platform 下的所有
device type，而统一计算策略只接受 `CL_DEVICE_TYPE_GPU`。计算设备号是 GPU-only、
全局从 `0` 开始的编号：按报告中的 platform 顺序，再按每个 platform 内出现的 GPU
顺序扁平化；CPU 类型 OpenCL device 不参与计数。例如在 GPU、CPU、GPU 的报告顺序中，
两个 GPU 的计算编号是 0 和 1，而不是 0 和 2。

前处理统一通过 deck 的 `preprocess_use_gpu` / `preprocess_gpu_device` 或 CLI 的
`--preprocess-use-gpu BOOL` / `--preprocess-gpu-device N` 选择设备。算法只允许
`idw` 与 `kriging`，不提供 `kriging_gpu` 配置或兼容别名。显式 GPU 的核心会在
创建相关输出前独立完成所选设备、context、queue 和 kernel preflight；构建无
OpenCL、无 GPU、设备号越界或任一 OpenCL 阶段失败均返回 `1`，绝不转用 CPU。
两个公共 CLI 参数只由统一 dispatcher 解析并传入命令，不存在命令局部别名或设备
选择旁路。`luwbc` 为保持 CPU 的 double 权重/分数语义要求 FP64；`luwvox` 的稳定
`(distance²,index)` 邻域排序和 double 地形坐标也要求 FP64。缺少 FP64 同样返回
`1`，不改用 CPU。
完整契约见 [`native-preprocess-compute.md`](native-preprocess-compute.md)。

本轮 full-dependency Release 主程序与 GUI 已构建成功，串行 CTest `30/30` 通过；
OpenCL GPU-only 设备 0 和 1 均已通过 `luwvox` k-NN/Kriging/IDW/Gaussian/地形网格
和 `luwbc` 融合重采样/DEM IDW/MLS 定向验证。诊断命令列出设备仍不能替代具体
kernel 的 preflight。fresh-prefix `build-production-install-20260720_071053` 已实际
通过 `luw --help`、`luw luwbc --help`、`luw luwvox --help`、`luw detect_opencl` 和
GUI offscreen runtime self-check；AMD/Intel 设备和真实大型数据性能仍待验证。

本机使用旧仓库虚拟环境以只读方式运行旧脚本，并与 Release 原生命令对比。两
platform、三 device 的成功报告均为 27 行、1020 字符，逐行与逐字符一致；
`native.detect_opencl` 和 `native.detect_opencl_no_sdk` 另行锁定单位、device type、
字段缩进、退出码 1/2/3、缺依赖提示和无平台分支。无平台时，原生实现保留旧
`LogicError('clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR')` 显示文本，但它只是
兼容提示，不表示存在 Python runtime。
