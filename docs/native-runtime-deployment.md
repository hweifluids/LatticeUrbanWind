# 原生安装运行时部署

## 目标与布局

`cmake --install` 生成可脱离开发机 Qt、VTK 与 vcpkg `PATH` 的安装树：

- `bin/`：`luw`、`luw_studio`、Windows 非系统 DLL 与兼容 wrapper；
- `lib/`：Linux 非系统共享库；
- `plugins/` 与 `bin/qt.conf`：Qt platform/imageformat 插件及相对插件路径；
- `share/gdal`、`share/proj`：GDAL 数据文件与其 PROJ `proj.db`；
- `share/vtk-<version>/proj`：VTK bundled-libproj 自带的版本匹配数据库；
- `core/`、`database/`、`project_template/`、GUI 图片：运行所需只读资产与默认 CULBM 源码树；显式设置 `LUW_ENABLE_FLUIDX3D=ON` 时才附带独立 FluidX3D 源码树；
- `core/cfd_core/CULBM/bin/`：由安装阶段 3 生成的活动默认 CULBM；
- `core/cfd_core/FluidX3D/bin/`：仅在 `LUW_ENABLE_FLUIDX3D=ON` 时由安装阶段 3 生成的冻结可选 FluidX3D。

Windows 使用 CMake `RUNTIME_DEPENDENCY_SET` 从已链接 target 的目录递归收集
非系统 DLL，排除 Windows `System32/SysWOW64` 与 API-set DLL；Qt 官方部署 API
只部署插件，避免再次复制 Qt DLL；未被程序链接的 generic/networkinformation/tls
插件不会进入安装树，否则它们会额外引入 Qt Network 闭包。Linux 同样收集
非系统共享库，目标的安装
RPATH 固定为 `$ORIGIN/../lib`；GNU/Clang 链接器使用可传递的 `DT_RPATH`，使
VTK/Qt 间接依赖也从安装 `lib` 解析。`/lib` 与 `/usr/lib` 下的系统库保持由
发行版管理。

Windows 安装树不复制 MSVC 编译器运行库；目标机需预装与构建工具链兼容的
x64 Microsoft Visual C++ Redistributable。

CLI 启动时仅在用户没有显式设置对应变量时，把 `GDAL_DATA`、`PROJ_DATA` 与旧版
兼容的 `PROJ_LIB` 指向安装根目录下的数据。用户的显式运行环境始终优先。
PROJ 主数据优先从实际 GDAL target/vcpkg prefix 选择，不能因为
`CMAKE_PREFIX_PATH` 中 VTK 排在前面而误用 VTK 的旧数据库；VTK 自带数据库另按
其原始相对目录部署。原生重写工作树 2026-07-20 的历史 Windows full-GIS 安装曾在
清空 PROJ 环境后同时通过 GDAL `shpinspect` 的 EPSG:3857→4326 转换，以及
`vtk2nc_new` 使用 VTK bundled-libproj 的 EPSG:32651 自动推断、坐标转换和 NetCDF
写出；本次文档整合没有重跑该验收。

## 完整构建顺序

默认完整安装必须依次构建原生应用层和独立 CULBM 模块。Windows 使用：

```powershell
installer\2_build_native.cmd
installer\3_compile_cfdcore.cmd
```

Linux 使用：

```bash
bash installer/2_build_native.sh
bash installer/3_compile_cfdcore.sh
```

`2_build_native` 构建并安装 C++20 CLI、前后处理和 Qt GUI，不创建 Python 环境。
`3_compile_cfdcore` 默认只构建活动 CULBM；两个阶段都不读取或构建 FluidX3D。
只执行其中一个阶段不能得到默认完整运行树。

需要同时部署冻结的独立 FluidX3D 后端时，必须在两个阶段都显式设置
`LUW_ENABLE_FLUIDX3D=ON`；同一变量也控制 `4_testrun` 是否要求其可执行文件：

```powershell
$env:LUW_ENABLE_FLUIDX3D = "ON"
installer\2_build_native.cmd
installer\3_compile_cfdcore.cmd
```

```bash
LUW_ENABLE_FLUIDX3D=ON bash installer/2_build_native.sh
LUW_ENABLE_FLUIDX3D=ON bash installer/3_compile_cfdcore.sh
```

## 原生应用层的隔离验证

只验证 CLI/GUI 运行时闭包时必须使用一个新的前缀。以下命令不会构建或运行 CULBM、
也不会构建或运行 FluidX3D，因此不能替代上面的默认完整两阶段安装：

```powershell
cmake --build build --config Release --parallel 1 --target luw luw_studio
$prefix = Join-Path $PWD ("build-install-" + (Get-Date -Format yyyyMMdd_HHmmss))
cmake --install build --config Release --prefix $prefix
cmake -DLUW_INSTALL_PREFIX=$prefix -P cmake/VerifyInstalledRuntime.cmake
```

验证脚本清除 Qt/GDAL/PROJ 以及动态库搜索环境，只保留安装 `bin` 与系统目录，
检查 `luw --version`、`luw detect_opencl`，并以
`luw_studio -platform offscreen --runtime-self-check` 验证 GUI 加载器、VTK/Qt DLL
和 Qt platform plugin。OpenCL 返回 `3`（无 platform）按既有安装契约视为警告；
其他非零状态失败。验证还会对每个已安装 Qt plugin 执行静态运行库扫描，任一
非系统 Qt DLL 未解析都会失败，避免“offscreen 能启动但闲置 plugin 已损坏”的
假阳性。

可用 `LUW_RUNTIME_SEARCH_DIRECTORIES` 补充非标准依赖目录；非 vcpkg 布局可通过
`LUW_GDAL_DATA_DIRECTORY` 与 `LUW_PROJ_DATA_DIRECTORY` 明确指定数据源目录。
VTK 自带数据库可用 `LUW_VTK_PROJ_DATA_DIRECTORY` 单独指定。

## 求解器选择与 Python 边界

`runluw` 必须接收 deck 路径并从 deck 选择求解器：`cfd_solver` 缺失或为 `culbm`
时运行 CULBM；只有显式 `fluidx3d` 才运行 FluidX3D。SurfaceIBM 仅由 CULBM 支持，
`ibm_enabler` 默认 `false`，与显式 FluidX3D 同时启用时应在进程启动前失败。选择
FluidX3D 前还必须通过 `LUW_ENABLE_FLUIDX3D=ON` 单独安装其可执行文件；缺失时明确失败，
不得回退到 CULBM。

生产安装不部署 Python runtime。仓库中的四个 CULBM `.py` 文件是精确 allowlist 的
离线验证/分析程序，不由安装器、CLI、GUI 或 `runluw` 调用。

## 原生重写工作树 2026-07-20 历史验收

以下是重写工作树合并前保存的历史证据，本次文档整合没有重新执行这些项目：

- 当时的 full-dependency Release 构建和 29 项 CTest 以单节点/串行方式通过；
  当时的 CFD target 不是默认构建目标，该次验收没有编译或运行 CFD。
- 新前缀 `build-production-install-20260720_031419` 的安装与
  `VerifyInstalledRuntime.cmake` 均返回 0；安装树包含 98 个运行 DLL、6 个 Qt
  plugin DLL、`share/gdal/gdalvrt.xsd`、两套版本匹配的 `proj.db`。
- 验证脚本在隔离环境中通过 `luw --version`、`detect_opencl`、Qt plugin 依赖闭包
  和 `luw_studio -platform offscreen --runtime-self-check`。
- 安装版 `shpinspect` 读取投影 Shapefile 后，5 个输入 sidecar 的 SHA-256 前后
  一致；安装版 `vtk2nc_new` 生成 78,145 字节、维度 2×4×5、含 16 个数据变量的
  NetCDF，随后由安装版 `cdfinspect` 成功读取，源 VTK 哈希不变。

这些是确定性小 fixture 和运行时闭包证据；并不替代真实多 GiB VTK、城市级 GIS
数据与长时间 GUI 交互的生产规模性能/人工验收。
