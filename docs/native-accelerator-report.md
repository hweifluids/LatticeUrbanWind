# 原生加速器 JSON 诊断与旧运行时边界

## 公共入口

```text
luw accelerator_report --startup-report installer/requirements.txt --json
accelerator_report --startup-report installer/requirements.txt --json
luw accelerator_report --prepare-cuda-runtime --json
```

`accelerator_report` 只调用 CUDA Driver API 与 OpenCL C API；在旧 CLI 规定的 JSON
模式中输出单行 UTF-8 JSON。
它不启动 Python，不创建 CUDA context，不复制或删除 wheel 文件，不写 manifest，
也不修改 `CUDA_HOME`、`CUDA_PATH`、`PATH`、`LD_LIBRARY_PATH` 或
`DYLD_LIBRARY_PATH`。

## 对 `core/accelerator_runtime.py` 的逐功能审计

| 旧功能 | 旧行为与副作用 | 原生映射与决策 |
|---|---|---|
| requirements import check | 解析 `requirements.txt`，逐个导入 Python module 或检查 distribution；import 本身可执行包初始化代码 | CMake、vcpkg、链接器和 CTest 已取代 Python 包环境。旧顶层 `checks` 与 `requirements_summary` 键保留，但 checks 为空并明确写明 `not applicable` |
| CUDA wheel discovery | 扫描当前 Python 的 site-packages 下 `nvidia/cu13`、`cu12`、split wheels、NVVM 与 libdevice | 原生程序不依赖 Python wheel 布局；不扫描 `.venv` 或用户 site-packages |
| `--prepare-cuda-runtime` | 创建 `sys.prefix/luw_cuda_runtime`，硬链或复制 DLL/so/libdevice，必要时先删除目标文件，并写 `runtime_manifest.json` | 这些动作只服务 Numba CUDA。兼容 flag 仍可读取，但只返回 `applicable:false`、`prepared:false`、`files_written:false` 和当前 CUDA driver 报告 |
| runtime environment | 设置当前 Python 进程的 `CUDA_HOME`/`CUDA_PATH`/`PATH`，Unix 还修改动态库路径，并清除 Numba CUDA path cache | 对原生动态链接/driver API 无意义且会造成隐式环境副作用，因此不复刻；测试逐字节比较相关环境变量前后值 |
| Numba CUDA probe | 导入 Numba，创建 current context，报告 Numba、CUDA runtime、driver 和 compute capability | `numba_version` 与 wheel runtime version 不适用并保留为空；通用的 driver version、设备名和 compute capability 由只读 CUDA Driver API 枚举 |
| PyOpenCL probe | 报告 platform name/vendor/version/device_count，以及 device name/vendor/version/driver/CU | OpenCL C API 复刻同一 JSON 字段，并扩展 type、max clock 与 global-memory bytes；文本报告另由 `detect_opencl` 严格兼容 |
| startup host/Python identity | `platform.platform()`、Python executable/version、requirements absolute path | host 由 Qt 原生系统信息报告；旧 Python identity keys 保留为空。传入 requirements path 只做绝对路径回显，不读取文件 |
| GUI startup consumer | GUI 启动 Python，把 JSON 解析到 accelerator structs | 当前 GUI 的 `StartupDiagnostics` 已直接使用 CUDA Driver/OpenCL C API；CLI 补齐 GUI 内部对象之外的公开、机器可读入口 |

## JSON 契约

普通报告保留旧消费者依赖的顶层键：

- `host_platform`
- `python_executable`、`python_version`（固定为空）
- `requirements_path`、`checks`、`requirements_summary`
- `cuda`、`opencl`

并增加：

- `schema_version: 1`
- `native_executable`、`native_version`
- `native_runtime`：明确 `python_free`，并标记 requirements/Numba/wheel 操作为
  不适用
- `side_effects`：固定声明 files/environment 未修改、仅做 driver queries

CUDA 对象继续提供旧 `configured/prepared/available/warning/summary/error/provider`、
版本和 devices 字段。这里 `configured` 表示 CUDA driver library 可加载；
`prepared` 始终为 false。OpenCL 对象保留旧字段，并在每个 device 中增加
`type`、`max_clock_mhz` 与 `global_memory_bytes`。

严格保留旧 CLI 的输出分支：无参数或只有 `--json` 时打印帮助；
`--startup-report PATH` 接受并回显旧 requirements 路径，但不读取或导入该文件；
`--prepare-cuda-runtime` 优先级仍高于 startup report。后两种实际模式只有同时提供
`--json` 才输出 JSON，不带 `--json` 时完成同一只读探测后静默返回 `0`。
prepare 模式只产生上述无副作用说明。
帮助文本保留旧显示名 `accelerator_runtime.py` 及 argparse 的完整逐行排版；这只是
公共输出兼容字符串，不会启动或依赖 Python。未知参数和缺少 startup-report 值时，
同样按旧 argparse 输出两行 usage、错误行并返回 `2`。help action 与其他参数混用
时保留 argparse 的立即退出优先级；但位于 help 前且缺值的 startup-report 仍返回
参数错误。
参数解析成功时，即使主机没有 CUDA/OpenCL 设备也返回 `0`，availability/warning
写在 JSON 中；未知参数或缺失 path 返回 `2`。

## Installer 兼容入口

旧 `installer/3_detect_opencl.cmd` 已恢复为纯原生 wrapper；它调用
`luw detect_opencl`，保留 `Probing OpenCL...`、安装器 banner 和子命令退出码。
同时提供 `installer/3_detect_opencl.sh`。两者均不查找 `LUW_PYTHON` 或旧 `.py`。

## 验证边界

本机 Release 报告只读枚举到两个 CUDA 设备、两个 OpenCL platform/三个 OpenCL
device；`CUDA_HOME`、`CUDA_PATH` 和 `PATH` 调用前后完全一致。测试同时编译
OpenCL-enabled 与 no-SDK 变体，并验证不存在的 requirements fixture 不会被创建。
installer wrapper 测试用返回 `3` 的假原生 launcher 锁定参数转发、prompt、banner
及退出码透传。

旧 startup report 会在构建 JSON 时隐式执行 CUDA wheel 准备，因此没有直接运行它
作为 oracle，以免写入或替换原虚拟环境文件；本审计依据完整 638 行源码逐字段锁定
schema。Linux/macOS 的动态 driver 实机路径尚未运行。
