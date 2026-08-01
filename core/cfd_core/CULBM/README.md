# CULBM

这是 LatticeUrbanWind 中的 CUDA 求解器实验模块。CULBM 随包保存并改编
宿主求解器代码，保持现有 `.luw` / `.luwpf` 配置、边界处理、修正、
湍流模型、湍流入口和 I/O 路径，并将设备执行后端改为 CUDA。现有生产参考求解器及
前端、前处理和耦合代码不会被此模块修改。

CULBM 是自包含的源码包，但仍是派生/改编实现；它不是独立实现，也不应表述为
clean-room 重写。归属、来源和适用限制随包保留在 `NOTICE.md` 与
`THIRD_PARTY_LICENSES/` 中。

当前标识为 `CULBM-v3.5-260731-thermal-lattices`；其中 `v3.5` 表示随包保存的
宿主求解器源码代际，不表示 CULBM 拥有原始宿主代码的作者身份。

这里的“脱离原代码”严格指工程边界：CULBM 的配置、编译、测试和运行不读取相邻
宿主源码树，也不在构建目录临时复制或打补丁；`src/solver/` 是由 CULBM 自己
持有、可直接维护和发布的固定派生快照。它不表示这些宿主文件已被 clean-room
重写，也不表示代码谱系或许可证随目录依赖消失。

## 实现方式

CULBM 将改编后的宿主求解器快照随源码保存在 `src/solver/`，并直接编译；构建时
不再从相邻目录 staging 或复制宿主源码。因此配置解析、网格生成、边界设置、
时间推进调度、多域分解、修正和输出仍来自同一套派生宿主逻辑。
在该快照之上，CULBM 注入 CUDA 专用的多卡 halo 调度和地基层压缩。自包含仅减少
构建期外部依赖，不改变其派生来源、原始权利人或适用限制。

当输入 STL 符合 LUW 预处理器生成的基础板特征（配置的 `base_height` 对应 STL
底部与其上方的一对大面积水平面）时，CUDA 版会自动识别该基础板。它会裁掉完整
的基础格层、同步平移几何和边界坐标，并强制 `z=0` 为连续固体层；因此地面不漏
风，同时不再为完整基础厚度分配稠密 LBM 网格。无法确认 STL 特征、或裁剪后不能
满足 z 向多卡分解时，求解器会保留原有网格并打印跳过原因。

运行时，随包保存的原生 CUDA C++ 设备源码（`src/solver/kernel_cuda_source.inc`）
直接与 CUDA 预置代码组合，由 NVRTC 按每张卡的 compute capability 编译为 PTX；
不执行运行时源代码翻译。CUDA Driver API 负责：

- 每卡 primary context、计算/入站/出站非阻塞 stream 和 kernel launch；
- device buffer 及 pinned host transfer buffer；
- 任意 `Dx × Dy × Dz` 周期笛卡尔域的 halo 传输和事件同步；
- 原生 CUDA 的线程索引、向量构造、半精度、原子操作和同步辅助函数。

多卡通信不锁定为两张卡。`D = Dx × Dy × Dz` 个域继续由原
`smart_device_selection` 选择 `D` 张不同的可见 NVIDIA GPU。每条相邻有向边会
独立探测 CUDA P2P：可用时执行 `cuMemcpyPeerAsync`；不可用时执行异步
device→pinned-host→device 回退。发送和接收使用独立缓冲，计算、入站和出站
stream 之间用 ready/complete/consumed 事件封闭缓冲生命周期，因此不存在双向
原地覆盖，也不需要逐轮全卡主机同步。锁页内存分配失败时会退化到普通主机内存
的同步保底路径。

本地多卡启动时，CULBM 还会按每个笛卡尔域面的面积和双向 P2P 可达性评估已选
GPU 的映射；仅当映射得分严格提高时才重排，得分相同或没有 P2P 时保留原选择顺序。
`CULBM_P2P_TOPOLOGY=0` 可关闭这一映射优化。

启用 `TEMPERATURE` 时，同一方向和同一时间奇偶相位的 `fi`、`gi` halo 会合并
为一次批量通信。打包和解包只复制 `fpxx` 原始位模式，不解压、不执行浮点运算，
也不改变 x/y/z 阶段顺序；传输总字节数不变，但 extract/insert kernel、DMA 或
MPI 消息轮次减半。联合 transfer buffer 按
`(transfers + temperature_transfers) * sizeof(fpxx)` 每边界格分配；
温度 D3Q7 每面传 1 个 DDF，温度 D3Q19 每面传 5 个 DDF。运行日志会另行确认
锁页主机缓冲是否实际分配成功，避免把
pageable 同步保底误报成 pinned 异步路径。

同一可执行文件也支持 MPI 多节点模式。普通直接启动时不初始化 MPI，仍走上述
单进程、本地任意多卡路径；由 `mpiexec` / `mpirun` / `srun` 启动时会根据常见
rank 环境标记自动初始化 MPI，rank 数大于 1 时切换为一 rank、一全局笛卡尔
子域、一 GPU。MPI 标准不保证 launcher 必须提供这些环境标记，因此生产 MPI
命令应显式设置 `CULBM_MPI=1`：

- 强制 `MPI ranks = Dx × Dy × Dz`，rank 号采用 x-fast 子域编号；
- 用 `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` 得到节点内 local rank，并绑定
  对应可见 CUDA GPU；调度器若已给每 rank 仅暴露一张卡则自动使用 ordinal 0；
- 每轮 halo 先由 CUDA 写入锁页主机缓冲，再成对提交分块 `MPI_Irecv`/`MPI_Isend`
  并等待同一对正负方向完成，最后异步写回 CUDA 接收缓冲；这不要求 CUDA-aware
  MPI、NCCL 或 GPUDirect；
- 各 rank 只上传和推进自己拥有的子域；设备场读取通过 MPI 重组，只有 rank 0
  写 VTK、均值场和 probe；
- 物体质心、力和力矩使用 MPI 全局归约。

MPI 初始化要求实现提供 `MPI_THREAD_MULTIPLE`。求解通信使用独立 communicator
和进程内串行锁，以避免不同 rank 的调用顺序形成环。

MPI 模式当前以可移植、可在本机验证的锁页主机 staging 为基线。前处理为了保持
前端零改动，会在各 rank 确定性执行并保留一份全局 host 视图；GPU 场和时间推进
是真正按 rank 分布的，但超大规模作业的主机内存仍有进一步分布化空间。

生产构建不定义 `GRAPHICS`，不渲染或写出离线图像帧，也不创建图形快照、图形
halo 或 PNG 解码器。数值时间推进只使用 `fi` 和 `gi` halo；标准输出仅包括 VTK、
均值场和 probe。保留的交互显示源码不属于默认生产路径，也不提供帧文件导出。

设备编译使用 FMA、快速除法/平方根及不启用 FTZ，以贴合生产参考路径。宿主
仍使用生产构建的快速浮点设置；MSVC 下仅对 profile 三次插值函数启用局部
precise 语义，以消除初始化表的编译器差异。

## 支持范围

本实验实现并验证了当前 LUW 时间推进实际启用的功能：

- 动量 D3Q19、SRT、FP16C 和宏观量更新；
- 体积力、逐单元力场和平衡边界；
- 运行时可选 D3Q7/D3Q19 温度输运、温度边界、浮力和非均匀温度输入；
- Smagorinsky-Lilly LES；
- 科氏力、下游缓冲 nudging、顶层 sponge；
- 边界质量通量修正；
- Von Karman 合成湍流入口；
- 任意正整数 `Dx × Dy × Dz` 多卡域分解、halo 交换和网格体素化；
- 任意正整数 `Dx × Dy × Dz` 的 MPI 一域/rank 多节点时间推进；
- 单级均匀 CUDA 网格上的被动粒子平流，包括给定粒子、规则自动散布、
  Euler/Heun-RK2/RK4/RK6，以及单进程本地多 GPU 域间迁移；
- 最终/非稳态 VTK、均值场和 probe CSV 输出。

LUW 的 `defines.hpp` 中未启用、当前流程也不调用的宿主扩展没有纳入本次
实现，包括自由液面、移动边界和宿主原有的颗粒-流体耦合 `PARTICLES` 扩展。
本节所列的被动粒子平流是 CULBM 独立的单向示踪功能，不向流场施加反作用。
CUDA 后端只支持 NVIDIA GPU；其他厂商设备应使用其各自已部署且受支持的求解器。

## 构建

依赖：

- CMake 3.24 或更高版本；
- 支持 C++17 的编译器；
- CUDA Toolkit 12.0 或更高版本（包含 Driver API 和 NVRTC）；
- 可见的 NVIDIA GPU 及相容驱动。
- 提供 `MPI_THREAD_MULTIPLE` 的 MPI 3 实现（多节点可选；Windows 已用
  Microsoft MPI 10，Linux 可用 OpenMPI/MPICH）。找不到 MPI 时仍会构建纯
  单进程 CUDA 模式。

Windows 已使用 Visual Studio 2022 x64 和 CUDA 13.2 验证：

```powershell
cd core\cfd_core\CULBM
.\build.ps1
```

也可显式指定配置和生成器：

```powershell
.\build.ps1 -Configuration RelWithDebInfo -Generator "Visual Studio 17 2022"
```

Linux 构建入口为：

```sh
cd core/cfd_core/CULBM
sh ./build.sh Release
```

Linux 入口已提供，但本次没有 Linux/CUDA 机器验证。构建产物位于
`bin/CULBM.exe`（Windows）或 `bin/CULBM`（Linux）。

## 运行与替换

CUDA 可执行文件接受与现有求解器相同的单个配置文件参数：

```powershell
.\bin\CULBM.exe C:\path\to\case.luw
.\bin\CULBM.exe C:\path\to\profile-case.luwpf
```

卡数和分解方向仍完全由现有 deck 参数控制，例如：

```text
n_gpu = [2, 1, 1]
```

也可使用 `[1, 2, 1]`、`[1, 1, 2]`、`[2, 2, 1]` 等分解；只要
`Dx × Dy × Dz` 不超过不同可见 CUDA 设备数，CUDA 后端没有固定卡数上限。
配置键、默认值、输入目录和结果目录均沿用原求解器。替换时只需由调用方选择
CUDA 可执行文件；本实验没有修改任何调用方代码。首次遇到某个内核源码/编译选项
组合时，NVRTC 编译属于进程启动开销；生成的 PTX 默认缓存到可执行文件目录的
`culbm-ptx-cache/`，后续相同组合会直接加载。求解器打印的 `normal Steps/s` 只
衡量时间推进。

本机或单节点 MPI 验证（deck 中需为 `n_gpu = [2, 1, 1]`）：

```powershell
$env:CULBM_MPI = "1"
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 2 `
  .\bin\CULBM.exe C:\path\to\case.luwpf
```

典型多节点作业同样使用一个 rank 对应一张 GPU。例如 OpenMPI：

```sh
mpirun -x CULBM_MPI=1 -np 8 --map-by ppr:4:node \
  ./bin/CULBM /shared/case.luwpf
```

或由 Slurm 分配 GPU：

```sh
srun --export=ALL,CULBM_MPI=1 \
  --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 \
  ./bin/CULBM /shared/case.luwpf
```

所有 rank 必须能读取同一 deck 和输入文件，并使用一致的 CUDA/NVRTC 版本。
结果目录应位于 rank 0 可写的位置；通常使用共享文件系统。若调度器不负责
`CUDA_VISIBLE_DEVICES`，运行时按节点 local rank 自动选择 GPU。也可为每个
进程显式设置 `CULBM_DEVICE`，值是该进程可见列表中的 CUDA ordinal。

故障诊断时可临时设置以下环境变量；生成文件写到可执行文件目录：

```powershell
$env:CULBM_DUMP_SOURCE = "1" # 保存运行时 CUDA 源
$env:CULBM_DUMP_PTX = "1"    # 保存每卡 PTX
$env:CULBM_PTX_CACHE = "0"   # 禁用 PTX 磁盘缓存
$env:CULBM_DISABLE_P2P = "1" # 强制测试 pinned-host 回退
$env:CULBM_P2P_TOPOLOGY = "0" # 禁用按 P2P 拓扑重排本地多卡映射
$env:CULBM_GROUND_COMPACTION = "0" # 禁用自动基础层压缩，供 A/B 回归使用
$env:CULBM_DEVICE_STATISTICS = "0" # 使用主机均值统计回退，供 A/B 回归使用
$env:CULBM_WRITE_FLAGS = "1" # 额外写出 raw_flags VTK，供几何掩码回归使用
$env:CULBM_STREAM_BLOCK_SIZE = "64" # 诊断用，32 的倍数
$env:CULBM_DEVICE = "0" # 可选：显式绑定当前 rank 的可见 GPU ordinal
$env:CULBM_MPI = "1" # 强制初始化 MPI；MPI 作业推荐显式设置
```

NVRTC 编译失败时，无需设置环境变量也会自动保存完整编译日志和运行时 CUDA 源。
默认 `stream_collide` block size 保持实测更快的 64；最后一个环境变量仅用于
设备专项调优，不建议写入通用 deck。

## GPU 温度输运

动量核心固定使用编译时选择的 D3Q19；温度是独立的标量分布，可在同一
CULBM 可执行文件中按 deck 运行时选择：

```text
culbm_temperature_lattice = D3Q7
culbm_boussinesq_reference_temperature_k = auto
```

也可设为 `D3Q19`。默认值是 D3Q7，以保持已有 deck 的存储规模和行为兼容。
该选择是进程级求解配置，setup 会在构造任何 LBM 对象之前设置本次运行的值。
为防止已分配的 `gi` 尺寸与后续 kernel/halo 解释不一致，存在活动 LBM 实例时
运行库会拒绝切换到另一种温度速度集。
D3Q7 的标量声速平方为 `1/4`，因此
`tau_T = 0.5 + 4 * alpha_lbm`；D3Q19 为 `1/3`，因此
`tau_T = 0.5 + 3 * alpha_lbm`。求解器从同一个 helper 生成 kernel 松弛率和
运行日志，避免两处公式不一致。

两种温度核心的 collision、streaming、固壁边界和多 GPU halo 均在 CUDA kernel
中执行。`TYPE_S` 是显式绝热反弹；`TYPE_S | TYPE_T` 使用 link-wise
anti-bounce-back 施加定温壁；非固体 `TYPE_T` 仍是 on-node 定温边界。
关闭 `buoyancy` 只会禁止温度对动量的反馈，不会再忽略输入 CSV 的温度列，
因此温度仍可作为被动标量输运。

温度平衡态使用 Guo 体积力修正后的物理速度
`u = sum(c_i f_i) / rho + F / (2 rho)`，而不是施力前的原始矩速度。对
`TYPE_E` 定速边界，输入的 `u` 同样定义为物理 Dirichlet 速度；初始化和每个碰撞步
都会写入 `f_eq(rho,u) - S(u,F_total)/2`，使下一个流体节点按同一 Guo 约定重构后
严格得到规定速度，不会额外叠加半个力步。这里仍是现有的 full-equilibrium 边界，
并不等同于二阶非平衡外推或 Zou–He 边界；规定速度仍应满足低马赫 LBM 的适用条件。

`culbm_boussinesq_reference_temperature_k` 定义 Boussinesq 浮力的零浮力参考
温度，单位为 Kelvin。缺省值和显式写入 `auto` 都保持已有行为：使用输入 CSV
温度列的 `(Tmin + Tmax) / 2`；若 CSV 没有有效温度范围，则回退到 293.15 K。
也可显式写入有限正数，例如：

```text
culbm_boussinesq_reference_temperature_k = 298.15
```

兼容别名为 `culbm_temperature_reference_k`。显式参考温度同时决定零浮力温度和
干空气近似 `beta = 1 / Tref`，不会被 CSV 极值重新覆盖。启用浮力且 CSV 含温度时，
启动日志会输出实际 `max |T-Tref| / Tref`；超过 5% 会告警，提示该算例已超出本实现
采用的保守小温差判据。这个 5% 是适用性警戒线，不是数学稳定性边界，因此求解器
不会仅据此中止；若温差不可视为小量，应改用可压缩热流模型，而不是把警告当作可忽略
的数值参数。非有限值、零或负 Kelvin 的显式参考温度会在分配 GPU 资源前直接拒绝。

若 `SurfData_*.csv` 使用命名 `T` 列，则每一条实际采用的边界样本都必须给出完整、
有限且严格大于 0 K 的温度；空单元格、带尾随字符的数值、`nan/inf` 以及温度行与
无温度行混用都会在 GPU 分配前按文件和行号报错。完全没有 `T` 列的旧速度边界仍保持
兼容。旧式无命名 7 列格式需要靠数值范围区分 `T` 与 `patch`，因此新输入应始终使用
完整命名表头，避免该历史歧义。

当前热力模型的可信范围是干空气、常参考温度、小密度差的 Boussinesq 近似。它不包含
大气位温或虚位温、湿度对密度的贡献、凝结/蒸发潜热、静力可压缩分层，也没有独立的
标量 SGS / 湍流 Prandtl 闭合。因此它适合近地层小温差热浮力与被动温度输运；水汽输送
和相变污染物模型应在后续作为新的守恒标量与热力耦合扩展，不能把当前 `T` 字段直接
解释为完整湿空气热力学。

当 `downstream_open_face = true` 时，温度在所选下游侧面自动使用一阶零法向梯度
出口：每个时间步先由相邻内侧格点的当前 DDF 重建温度，再在出口格点建立平衡态。
这一过程同样由 GPU kernel 完成，并按 Esoteric-Pull 当前奇偶相位读取 DDF；不会把
下游面错误地当成周期温度入口。该条件是稳健的平衡外推，不是 Orlanski 对流出口。
多 GPU 时，出口法向上每个分区至少需要两个物理格点；不满足时 setup 会在求解前
明确报错。出口方向必须在 LBM 首次初始化前设定；`reset()` 只重新初始化场，不允许
把旧出口改到另一侧，以免旧 `TYPE_T` 标记残留。若将该标量核心用于污染物浓度，还应
在上游显式给定背景浓度，以定义可能从入口进入的标量，而不是依赖底层周期存储拓扑。

浮力使用独立于一般体积力的重力向量。生产 setup 将标准重力转换成 LBM 单位，
按 Boussinesq 形式耦合；垂直嵌套子层在
`dx, dt -> dx/r, dt/r` 时将格子重力缩放为粗层的 `1/r`。这避免了把重力误当成
全域恒定压力梯度，也修复了此前体积力为零时浮力始终为零的问题。

FP16C 下，温度 D3Q7 的 `gi + T` 占 18 Byte/cell，D3Q19 占
42 Byte/cell；D3Q19 的方向各向同性更好，但显存、显存带宽和 halo payload
相应更大。运行时的显存预算会按所选温度速度集计算。

## GPU 被动粒子平流

被动粒子平流默认关闭。启用后，粒子读取或规则生成、释放、速度插值、生命周期与
状态判断以及 Euler/Runge-Kutta 数值推进会随 LBM 时间推进执行。粒子只读取流场，
不会修改 `rho`、`u`、边界 flag 或其他 LBM 状态。

当前功能只支持单级均匀 CUDA 网格。单进程的一张或多张本地 NVIDIA GPU 均可使用；
多卡粒子跨域后会迁移到目标 CUDA 域。启用平流时，MPI 分布式运行和非空的
`cuda_vertical_nesting_agl` 均会明确报错，当前不会退化为 CPU 平流。未启用平流时，
MPI 和垂直嵌套功能保持原有支持范围。

本地多 GPU 的积分、目标域判定和迁移记录生成都在 kernel 内完成；CPU 只编排紧凑迁移
记录的设备间传输。正常 LBM 稳定性范围下，一个 stage 的位移由相邻域的一层 halo
覆盖。若异常速度使某个 RK stage 一步跳出源域 halo，粒子会明确标为 `invalid`，不会
用 CPU 积分或把缺失速度误当成固体。

平流运行时还会在每张卡保存一份本域的上一帧速度，并创建粒子状态、迁移和接收队列。
这些缓冲不是 LBM 核心分布的一部分。`gpu_memory` 网格拟合会按
`culbm_advection_max_particles` 的保守上界，把全部独立平流分配逐项向上取整后计入
每卡预算。按整卡总容量拟合时预留 512 MiB，其中一半用于 runtime/module、另一半是
分配后的安全保护；实际粒子计划生成后，运行时还会在分配任何粒子缓冲之前查询当前
空闲显存。因桌面显示或其他进程占用导致预算失效时，求解器
会给出“平流所需/当前空闲/保留量”，不会等到半数队列已分配后才由驱动 OOM。
平均场 Welford 缓冲和稀疏 probe 在粒子缓冲之前创建，但属于可选加速；它们会按
当前驱动空闲量预留完整的后续粒子上界和 256 MiB 分配保护。不足时分别退回已有的
host 统计/全场 probe 路径，不会占走已由 `gpu_memory` 承诺给平流的显存。

### 配置

下面列出应在新 deck 中使用的规范键名。所有带 `_s` 的值以秒为单位，带 `_m`
的值以米为单位；未知的 `culbm_advection_*` 键会报错，以避免拼写错误被静默忽略。

| 键 | 默认值 | 说明 |
|---|---|---|
| `culbm_advection_enabled` | `false` | 是否启用被动粒子平流。 |
| `culbm_advection_mode` | `prescribed` | `prescribed` 为文件给定粒子；`automatic` 为定期规则散布。 |
| `culbm_advection_integrator` | `rk4` | `euler`、`heun-rk2`、`rk4` 或 `rk6`。 |
| `culbm_advection_particle_file` | `particle.txt` | `prescribed` 模式输入；相对路径按 case 所在目录解析。 |
| `culbm_advection_release_time_s` | 未设置 | 全局释放起点。`prescribed` 模式下覆盖文件内所有 T；`automatic` 模式下缺省为 0 s。 |
| `culbm_advection_lifetime_s` | 未设置 | 单个粒子的最大存活时间；未设置表示不因寿命而消散，仍可能因出界、进入固体或无效状态而终止。 |
| `culbm_advection_output_interval_s` | 未设置 | 中间 CSV 快照间隔；未设置时只写初始和最终快照。 |
| `culbm_advection_max_particles` | `1000000` | 整个运行计划允许的粒子槽位上限，必须为正整数且不超过 CUDA `uint` 范围。 |
| `culbm_advection_release_interval_s` | 无 | `automatic` 模式必填且必须大于 0；相邻两次名义释放的秒数。 |
| `culbm_advection_auto_spacing_m` | 无 | `automatic` 模式必填的 `[dx, dy, dz]` 规则点阵间距，各分量必须大于 0。 |
| `culbm_advection_auto_min_m` | 自动内域下界 | 可选的 `[xmin, ymin, zmin]`；必须与 `auto_max_m` 同时设置。 |
| `culbm_advection_auto_max_m` | 自动内域上界 | 可选的 `[xmax, ymax, zmax]`；每轴必须严格大于对应下界。 |

解析器为已有 deck 保留了少量别名，但文档和新配置应使用上表的规范键；同一个
设置同时通过规范键和别名给出会作为重复配置报错。

给定粒子的最小配置如下：

```text
culbm_advection_enabled       = true
culbm_advection_mode          = prescribed
culbm_advection_integrator    = rk4
culbm_advection_particle_file = particle.txt

// 可选；省略时各行使用自己的 T，没有 T 的行在 0 s 释放
culbm_advection_release_time_s = 12.5

// 可选；省略即无限寿命
culbm_advection_lifetime_s        = 60
culbm_advection_output_interval_s = 5
culbm_advection_max_particles     = 1000000
```

自动模式在每个释放时刻生成相同的规则笛卡尔点阵：

```text
culbm_advection_enabled            = true
culbm_advection_mode               = automatic
culbm_advection_integrator         = heun-rk2
culbm_advection_release_time_s     = 2
culbm_advection_release_interval_s = 0.5
culbm_advection_auto_spacing_m     = [5, 5, 2]
culbm_advection_auto_min_m         = [100, 200, 1]
culbm_advection_auto_max_m         = [200, 300, 41]
culbm_advection_lifetime_s         = 30
```

`auto_min_m` 和 `auto_max_m` 都省略时，点阵使用每个全局轴向内缩一个格点后的均匀
网格内域。每轴从下界开始按配置间距生成，不越过上界；上界不一定恰好落在点阵上。
当前自动模式仅提供规则点阵，不提供随机散布。每次释放都会占用一整批新的粒子槽位，
因此 `culbm_advection_max_particles` 限制的是“每批点数 × 运行期释放批数”，而不只是
某一时刻的活跃粒子数。落在固体中的规则点会保留稳定 ID，并在释放时标记为 `solid`。

### `particle.txt`

`prescribed` 模式逐行读取 `X Y Z [T]`。X/Y/Z 是米制物理坐标，T 是可选的非负释放
秒数；没有 T 的行等价于 `T=0`。逗号、分号或空白均可作为分隔符，空行以及 `#`、
`//` 后的注释会忽略；第一条有效内容可以是 `x,y,z` 或
`x_m,y_m,z_m,release_time_s` 一类表头。每条数据行必须恰有三列或四列，且所有数值
必须有限。

例如：

```text
x_m,y_m,z_m,release_time_s
105.0, 210.0, 2.0, 0.0
110.0, 210.0, 2.0, 3.5
115.0, 210.0, 2.0, 8.0
```

文件路径为绝对路径时直接使用；相对路径相对于当前 case 目录，默认文件名为
`particle.txt`。设置 `culbm_advection_release_time_s` 后，所有行（包括带 T 的行）
都在同一个全局时刻释放，文件中的 T 会被明确忽略并打印提示。量化后晚于最终
CULBM 帧的给定粒子会在启动时明确报错；恰好落在最终帧的粒子允许释放并出现在最终
快照中。

### 时间量化与数值方法

设一个 LBM 步对应 `dt_s = units.si_t(1)` 秒。释放时刻不是用浮点秒在每一步累加，
而是按 `floor(t / dt_s + 0.5)` 映射到最近的 LBM 步；恰在半步处向后一个步号取整。
这同时适用于文件 T、全局释放时刻、自动释放和中间输出时刻。

自动模式按原始秒数直接构造
`release_time_s + k * release_interval_s`，然后分别量化每次事件，避免反复累加一个
已取整间隔所产生的漂移。若多个名义释放映射到同一个 LBM 步，它们不会丢失，而会在
该步生成多批相同点阵；中间输出的重复步号则合并为一个 CSV 快照。

有限寿命使用 `floor(lifetime_s / dt_s)`，保证离散表示的存活时长不超过配置上限；
粒子在死亡步推进前变为 `expired`。因此，正数但短于一个 LBM 步的寿命会量化为
0 步，粒子可以在释放帧出现，但不会完成一次平流步。未设置
`culbm_advection_lifetime_s` 时，寿命步号为无限。

可选积分器为显式 Euler、Heun 二阶 RK、经典四阶 RK 和七阶段六阶 RK。CUDA kernel
在每个 LBM 步保存前一帧与当前帧速度，并按各 RK stage 的时间位置线性插值速度场；
空间速度采用流体格点的三线性插值。积分 stage、固体/出界检测和生命周期判断均在
GPU 上执行。

### 坐标、状态与 CSV

输入、自动散布边界和 CSV 都使用与 `culbm_output_global_*` 相同的全局米制坐标：
x/y 保留 case STL 的米制坐标，z 以人工地基层顶部为零。这里的 z 不是 CUDA 格点
索引，也不是随 DEM 起伏变化的 local AGL。`particle.txt` 中落在全局计算格之外的
坐标会在启动时直接报错；显式 automatic min/max 超出同一范围时也会直接报错。

粒子快照写到 `<case>/RESULTS/particles/`，文件名为：

```text
<vtk_prefix><datetime>_particles_<LBM-step>.csv
```

CSV 列固定为：

```text
id,x_m,y_m,z_m,release_time_s,age_s,status,termination_time_s
```

`id` 在整个运行中稳定；`release_time_s` 是量化后的实际释放帧时间，`age_s` 在粒子
终止后不再增长。`status` 可为 `pending`、`active`、`expired`、`escaped`、
`solid` 或 `invalid`；尚未终止时 `termination_time_s` 留空。设置
`culbm_advection_output_interval_s` 后，除初始和最终状态外还会写量化后的定期快照。

数值平流、速度场时间/空间插值、自动规则点生成以及粒子状态更新由 CUDA kernel
完成，CPU 不执行轨迹积分。CPU 负责 deck 与 `particle.txt` 解析、秒到步的调度量化、
CSV 序列化，以及当前本地多 GPU 实现中的跨域迁移队列 staging 和传输编排。

## 物理坐标与 AGL 输出

以下五个可选 deck 布尔键只在最终时步写一次；它们不会改变既有 raw/avg VTK、probe
或非稳态输出。文件名保留 case 前缀与步号：VTK 写入 `RESULTS/vtk/`，NetCDF 写入
`RESULTS/netcdf/`。

```text
culbm_output_global_vtk    = true   // 1. 全局结构化 VTK
culbm_output_global_netcdf = true   // 2. 全局结构化 NetCDF-3 (CDF-2)
culbm_output_agl_vtk       = true   // 3. DEM AGL VTK POLYDATA 点集
culbm_output_agl_netcdf    = true   // 4. DEM AGL NetCDF-3 点集
culbm_output_global_vtkhdf = true   // 5. 全局结构化 VTKHDF ImageData
```

1、2、5 保留 STL 的 x/y 米制坐标，并把 z 重基准到“人工地基层顶部”：
`z=0 = stl_min_z + base_height`。这使普通 `.luw`（基础板位于 `0..base_height`）和
`.luwdg`/`.luwpf`（基础板位于 `-base_height..0`）得到同一物理地面 datum。结构化输出
会裁掉 z<0 的基础板行，但仍保留其余固体单元并通过 `flags` 标记；它不是 UTM/经纬度
重投影，若输入 STL 本身采用本地坐标，输出也保持本地米制坐标。

3、4 必须读取 `proj_temp/interpolated_dem.csv`。CULBM 按该 CSV 的 x/y 边界对齐 STL
边界并双线性采样连续 DEM；任何 `z <= terrain(x,y)` 的格点都会被排除，保留下来的点以
`z_agl = z_global - terrain(x,y)` 写出。AGL VTK 是 `POLYDATA`（每个样本一个 vertex），
AGL NetCDF 使用一维 `point` 维度。两者都包含 `x`、`y`、`z_agl`/`ground_z`、`u/v/w`、
`rho`、`flags` 和可用时的 `temperature`。缺少或无效 DEM 时，启用任一 AGL 开关会明确
报错而不会悄悄回退到平地。

NetCDF 写出器为内置、无外部运行时依赖的标准 NetCDF-3 64-bit-offset (CDF-2) 实现。
单个变量受该格式的约 4 GiB 上限约束；超限时导出会报错，避免生成不完整文件。

5 使用标准 `.vtkhdf` 的 HDF5 `VTKHDF/ImageData` 布局，而不是为每一点重复写入坐标的
`StructuredGrid`；它与 1 使用相同的均匀网格 `Origin`、`Spacing` 和非地基 datum。点场以
128×128×1 分块写入；若链接的 HDF5 库有 deflate 编码器，则启用无损 shuffle+deflate
level 4 压缩。该特性由 CMake 的 `CULBM_ENABLE_VTKHDF=ON` 控制，并自动查找 HDF5 C 库；
若构建时找不到 HDF5，其他四种输出仍可用，而启用第 5 个 deck 键会在求解前明确报错。
Windows 若链接动态 HDF5，还需让 `hdf5.dll` 及其依赖位于 `PATH`（例如 vcpkg 的
`installed/x64-windows/bin`）；Linux/HPC 则按集群的 HDF5 module 或 `HDF5_DIR` 配置即可。

### GB 级伪湍流输出基准

可选的 `CULBM_BUILD_PHYSICAL_OUTPUT_BENCHMARKS=ON` 会构建
`CULBM-physical-output-gib-benchmark`。它不运行 LBM，而是通过同一套物理输出写入器按
索引即时合成确定性的多尺度伪湍流速度、密度和 flag 字段；默认网格为
`512 x 512 x 384`，未压缩 point payload 为 1.59 GiB。即时生成避免为基准额外占用一份
GB 级内存副本。

```powershell
cmake -S . -B build-vtkhdf-benchmark -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCULBM_ENABLE_MPI=OFF -DCULBM_ENABLE_VTKHDF=ON `
  -DCULBM_BUILD_PHYSICAL_OUTPUT_BENCHMARKS=ON `
  -DCMAKE_PREFIX_PATH=C:/path/to/vcpkg_installed/x64-windows
cmake --build build-vtkhdf-benchmark --target CULBM-physical-output-gib-benchmark --parallel
$env:PATH = 'C:/path/to/vcpkg_installed/x64-windows/bin;' + $env:PATH
.\build-vtkhdf-benchmark\CULBM-physical-output-gib-benchmark.exe `
  --output .\test-output\physical-output-gib-benchmark
```

它会并列写出全局 legacy VTK、CDF-2 NetCDF 和 VTKHDF，并报告字节数、每点开销、写入时间
以及相对 legacy VTK 的比例。输出目录中的文件均为 GB 级临时基准数据，应在需要时自行
移除；AGL 格式的点集不同，不在此直接比较范围内。

## CUDA 高度嵌套网格

`cuda_vertical_nesting_agl` 是 CULBM 专用的可选 deck 键。它的单位始终是米，
以压缩后单层地基的顶面为固定水平零面；名称沿用旧配置习惯，但它不是随地形变化的
local AGL。空列表或未写该键时仍运行原有单级网格。

```text
// 设定的 cell_size 是最高层的粗网格尺度 H
cuda_vertical_nesting_agl = [50]       // 0–50 m: H/2；其上: H
cuda_vertical_nesting_agl = [50, 100]  // 0–50 m: H/4；50–100 m: H/2；其上: H
```

例如 `cell_size=20 m`、`[50,100]` 产生 5 m、10 m、20 m 三个物理尺度；这与
CUDA/ LBM 内部坐标无关。构建时会把 SI 高度转换为各层的原生格数，并拒绝不能整除
各自尺度的 slab，而不会把 50 m 当作 50 个 LBM 格点。

每个相邻层的水平比例固定为 2:1，接口只做下面两种操作：

- 细到粗：四个固定水平子格的严格算术平均（固体槽位采用固定墙状态参与四项平均，
  不会对剩余流体槽位重新归一化）；
- 粗到细：把粗格宏观量原样常量复制到开放子格。

没有高阶空间插值、斜率限制器或时间插值。为避免接口与常规 halo 混用，接口面由
`TYPE_E`/`TYPE_S` aperture 明确封闭。输出则统一写成公共最细物理尺度：粗层值只在
其逻辑子格上重复，绝不额外插值。

网格计划的限制是明确且与求解器分解相容的：每个物理 slab 至少有两个本层格；其
厚度必须是本层 SI 格长的整数倍；每层 `Nx/Ny` 与含两个接口 ghost 行的 `Nz` 必须
分别能被 deck 的 `Dx/Dy/Dz` 整除，而且每个 z 分区至少拥有两个 allocated rows，
避免把接口 ghost 与 donor 拆到无法同步的单行分区。这不是卡数限制：任意正整数
`Dx×Dy×Dz` 仍然受支持，只要机器实际有相应数量的可见 CUDA 设备。

接口每次改写开放/封闭 aperture 后都会同步 flags halo；这对温度 D3Q19 的
`z±x`/`z±y` 对角 link 是必需的，否则 Dx/Dy 分区缝附近会读到旧接口类型。全局
最顶层保留真实 sky 边界，只有面向相邻层的 ghost 才预封为内部接口；最底
compact-ground 行仍由独立规则封闭。

垂直嵌套显存按三个不同时序的峰值取最大：普通种子格及其 STL 临时缓冲、公共最细
几何栅格及其 STL 临时缓冲、同时驻留的全部 2/3 个最终 LBM level 加各层 VK
updater。STL 的 `p0/p1/p2` 和参数缓冲按四次独立 CUDA 分配分别向上取整；
`gpu_memory` 拟合前会按最终几何选择规则预读三角形数量，公共最细栅格则以完整活动
高度作安全上界。运行时再用实际 STL/地形顶高缩短该临时栅格，并基于
`cuMemGetInfo` 做一次精确峰值预检。容量拟合与构建均同时保留 256 MiB
runtime/module 余量和 256 MiB 分配后保护，避免 D3Q19 额外 DDF 或临时体素化缓冲
在 `cuMemAlloc` 中途才失败。

嵌套模式要求已成功启用 CUDA 压缩地基，以便保留唯一、连续的 `z=0` 固体封闭层。
STL 和 profile DEM 都先落在同一个公共最细 CUDA 光线体素栅格上；光线追踪只覆盖
最高相关 STL/地形高度，随后每一层直接从该最细逻辑体素做多数表决。一个目标格中
半数或更多最细体素为固体时即为固体（恰好一半也为固体）。因此建筑、地形和界面
附近的几何不会因为先粗化某一层而丢失；基础板下方被识别出的生成固体会先移除，
再叠加最细 DEM 地形补集。

顶层 sponge 只安装到最上方粗层，内部嵌套 ghost 行不会误用 sponge；写 VTK、均值
场和 probe 时会裁掉 sponge 扩展层。profile、dataset 和普通 case 都走同一嵌套
调度；VK 入口会在每层使用相同的粗尺度时间实现，但按各层原生坐标与时间步更新。

接口面是嵌套求解的热路径。CULBM 用 CUDA Driver 的二维 DMA 一次复制一个完整的
局部面/字段分量，替代原先逐行的数百次 Driver 调用；保持原有完整双向接口桥和
调用顺序。Windows RTX 2080 Ti 的 200×200 两层 32 步回归中，逐行版本与二维 DMA
版本的 `u`、`rho` VTK binary payload 字节一致；该小算例日志中的求解时间由 17 s
降到 1 s。该数字仅说明该机器、该网格和该运行路径，不能外推到其他 GPU、网格或
MPI 拓扑。

## 物理等价回归

基础层压缩会改变稠密格的 `z` 原点和高度，因而不应用 VTK payload 偏移或逐字节
一致性判断它是否保持物理解。`tests/compare_physical_vtk.py` 按 SI 坐标匹配样本，
可同时读取 `CULBM_WRITE_FLAGS=1` 写出的 `raw_flags`，排除固体单元并验证两侧几何
掩码一致。它检查有限值、相对 L2、非均匀场相关性，以及全域和逐高度平面的 QoI；
逐点误差仅在显式传入 `--require-pointwise` 时才作为硬门槛。

压缩/未压缩 A/B 的典型调用如下。保留的连续地面封闭行不属于比较流场时，可跳过
最低一个匹配平面：

```powershell
python tests\compare_physical_vtk.py `
  --reference <uncompressed-raw_u.vtk> --candidate <compacted-raw_u.vtk> `
  --reference-mask <uncompressed-raw_flags.vtk> `
  --candidate-mask <compacted-raw_flags.vtk> `
  --skip-bottom-planes 1 --report physical-report.json
```

## 参考后端/CUDA 对比测试

`tests/compare_backends.py` 会为外部参考求解器和 CULBM 生成相同且互相隔离的
case，运行多卡求解，读取 VTK 的原始 float payload，并检查：

- 默认逐点绝对误差、相对误差和最多 4 ULP；
- CUDA/参考后端 `normal Steps/s` 比值不低于 `0.5`；
- 使用 `--exercise-output` 时，全部 VTK/CSV 文件集合及 SHA-256 完全一致。

基础性能与数值回归：

```powershell
$reference = "C:\path\to\reference-solver.exe"
python tests\compare_backends.py --reference $reference --cell-size 5 --steps 1001
```

比较器始终要求显式传入 `--reference <基准程序>`，不内置任何机器相关路径。
`--source-case <测试案例目录>` 可选，默认使用仓库示例 case；这两个输入均不是
CULBM 构建或运行依赖。

卡数和拓扑可直接指定，不再限制为 1/2 张卡：

```powershell
python tests\compare_backends.py --reference $reference --domains-x 1 --domains-y 2 --steps 1001
python tests\compare_backends.py --reference $reference --domains-x 1 --domains-z 2 --steps 1001
python tests\compare_backends.py --reference $reference --domains-x 2 --steps 1001 --cuda-mpi-ranks 2
ctest --test-dir build -C Release --output-on-failure
```

CTest 的 `CULBM.DomainTopology` 会穷举 `Dx/Dy/Dz = 1..6`，检查最多
216 个域时正/负邻居环闭合且映射双射；实际运行仍要求有对应数量的物理 GPU。
发现 MPI 时还会运行 `CULBM.MPIRuntime`，用两个真实 MPI rank 检查
节点内 rank、分块 gather、双向 send/recv、全局归约、simulation/graphics
同 tag 不同长度的并发传输，以及运行/绘图/session 控制 collective。

`CULBM.TemperatureAnalyticGPU` 直接调用生产 LBM 和生产 NVRTC kernel，对
D3Q7/D3Q19 分别检查周期 Fourier 扩散与平流、定温/绝热固壁、独立重力浮力、
嵌套层温度桥以及四个水平侧面的零梯度出口；同时锁定 STL 四次分配的 MiB 取整、
公共最细栅格 `Dz` padding 和“临时栅格大于最终层级”的峰值模型。
`CULBM.TemperatureAnalyticGPU.TRT` 将同一组温度、Guo 外力和 `TYPE_E` 解析回归
独立编译为 TRT 动量碰撞版本；它只验证短时解析路径和实现一致性，不代表 GEA 等
非线性大气算例的长期稳定性。TRT 初始化会打印标准 `Lambda=3/16` 对应的
`tau+/-`、`omega+/-` 和 `|1-omega+/-|`；若任一本征支或温度 SRT 支超过 0.98，
只发出 case-dependent 预警，不自动修改松弛系数。
`CULBM.TemperatureHaloMultiGPU` 在至少两张 CUDA 卡时对两种温度速度集比较
单卡与 `Dx=2`、`Dy=2`、`Dz=2` 的全场及分解接口，并覆盖出口分别位于分区法向
和切向的情况。它还用 D3Q19 在 `Dx=2`、`Dy=2` 下检查跨分区缝的嵌套上下层、
开放/封闭 aperture 和连续两种 pull parity；设备不足时按 CTest 约定返回
skip 码 77。

Von Karman、全部外力/修正、标准非均匀温度输入和扩展 I/O 回归：

```powershell
python tests\compare_backends.py --reference $reference --cell-size 20 --steps 1001 --enable-vk
python tests\compare_backends.py --reference $reference --cell-size 20 --steps 64 `
  --enable-forcing-models --enable-flux-correction
python tests\compare_backends.py --reference $reference --case-kind standard --cell-size 20 --steps 64 `
  --enable-forcing-models --enable-flux-correction
python tests\compare_backends.py --reference $reference --cell-size 20 --steps 64 `
  --enable-forcing-models --enable-flux-correction --exercise-output
```

每次测试的控制台日志、deck 和 `compare-report.json` 写入忽略版本控制的
`test-output/<timestamp>.../`。

## 本次验收实测

测试环境为 Windows、2 × NVIDIA GeForce RTX 2080 Ti、CUDA 13.2、驱动
595.79；参考后端和 CUDA 都使用 D3Q19/SRT/FP16C。该驱动在本机
`cuDeviceCanAccessPeer` 返回不可用，所以下列多卡数字全部来自异步
pinned-host 回退，而不是无法在本机实测的 P2P 快路径。

下表是温度速度集仍固定为 D3Q7 时的历史后端对比与性能基线。
`260731-thermal-lattices` 已修正旧 D3Q7 的半扩散率、固壁温度和未生效浮力，
因此温度/浮力 case 不应再要求与保留这些缺陷的旧参考二进制逐字节一致；
当前温度正确性以本节上方的解析 GPU 回归为准。D3Q19 温度核心的显存和带宽
不同，也不能直接套用下表性能数字。

本轮另以 D3Q19 温度核心和自动 RK6 平流完成双卡显存压力回归：
`1376 × 1358 × 171 = 319,531,968` 个格点、`4,271,421` 个粒子，
每卡预算为 `18,885 MB = 16,680 MB LBM + 2,205 MB 平流上界`；运行期按实际
粒子计划、逐分配向上取整后的预检为 `2,170 MB/卡`。2 个 LBM 步、初末粒子快照和最终
`u/rho` 场均完整写出，未出现 CUDA 分配错误。该规模是当时桌面负载下的安全上限
回归，不表示所有运行时背景显存占用下都能固定使用同一网格。
在 force-aware Dirichlet、温度—浮力顺序和输入校验修正完成后，又以
`buoyancy=true`、显式 `Tref=293.15 K` 原规模复跑；GPU 0 分配后仅余约 391 MiB，
2 步正常结束。流式验收确认全部 `4,271,421` 个粒子均为有限 active 状态且发生位移，
最终 `u/rho` 的 `319,531,968` 个点全部有限，SI 密度范围为
`1.187821 .. 1.230341 kg/m^3`。

| Case | 步数 | 参考后端 Steps/s | CUDA Steps/s | CUDA/参考后端 | 数值结果 |
|---|---:|---:|---:|---:|---|
| 8,704,584 cells，单卡基础 profile | 1001 | 369.131 | 385.956 | 1.046 | `u`、`rho` 逐字节一致 |
| 8,704,584 cells，`[2,1,1]` 基础 profile | 1001 | 210.695 | 518.361 | 2.460 | `u`、`rho` 逐字节一致 |
| 8,704,584 cells，2 MPI ranks / 2 GPUs | 1001 | 204.167 | 337.106 | 1.651 | `u`、`rho` 逐字节一致 |
| 1,090,800 cells，2 MPI ranks / 2 GPUs | 1001 | 369.245 | 654.357 | 1.772 | `u`、`rho` 逐字节一致 |
| 1,090,800 cells，`[1,2,1]` | 1001 | 375.804 | 916.144 | 2.438 | `u`、`rho` 逐字节一致 |
| 1,090,800 cells，`[1,1,2]` | 1001 | 247.189 | 838.513 | 3.392 | `u`、`rho` 逐字节一致 |
| Von Karman profile，`[2,1,1]` | 1001 | 375.238 | 955.498 | 2.546 | `u`、`rho` 逐字节一致 |
| 标准 SurfData + 非均匀 `T` + 全部外力 | 1001 | 336.936 | 1104.591 | 3.278 | `u`、`rho`、`T` 逐字节一致 |
| flux correction + 扩展 I/O | 1001 | 439.792 | 1250.625 | 2.844 | `u`、`rho` 及 8 个 I/O 文件逐字节一致 |

大网格 CUDA 双卡/单卡为 `1.343×`（并行效率约 `67.2%`）；优化前同一大网格
双卡为 303.969 Steps/s，本版全新构建验收为 518.361 Steps/s。受 16 步内置
benchmark、GPU 动态频率和系统负载影响，这些是指定机器的单次实测，不应视为
其他 GPU 或驱动组合的固定性能保证。物理机只有两张卡，因此三卡以上只完成了
共享拓扑逻辑的穷举测试，没有声称做过三卡以上硬件实测。

扩展 I/O case 比较了最终场、非稳态场、均值场和 probe 文件，共 8 个文件全部
逐字节一致；测试脚本中的硬门槛仍是用户要求的 CUDA 不低于参考后端的一半。

MPI 路径在同一台双卡机器上用 Microsoft MPI 10 做了“逻辑跨节点”验证：
基础、Von Karman、温度/全部外力/flux correction、均值/probe I/O 均通过，
其中扩展 I/O 的 8 个文件与参考后端逐字节一致。尚未在两台物理主机、Linux
MPI、InfiniBand/UCX 或 CUDA-aware MPI 上实测，因此这里只确认代码和通信协议
具备多节点运行条件，不把本机 MPI 性能外推为真实网络性能。

### 稳定 halo 优化 A/B

上表中的 `normal Steps/s` 来自较短的内置探测，适合功能验收，不作为本轮 halo
优化的稳定性能结论。本轮另用固定 `202 × 200 × 25 = 1,010,000` cells 的
profile 算例运行 10,001 步，并只统计 `SOLVER START` 后第 2001–9000 步的
进度样本。四组进程严格串行，按 R→C、C→R、R→C、C→R 交错执行；每组都比较
最终 VTK 原始 payload。

| 二进制 | SHA-256 前缀 | 双卡稳态中位数 | 四次运行范围 | 结果 |
|---|---|---:|---:|---|
| 优化前冻结版 | `E0B330DE6803` | 1622.0 Steps/s | 1596.0–1901.0 | `u/rho` byte-exact |
| halo 批处理版 | `CB4BEB3B02C9` | 2991.0 Steps/s | 2945.0–3051.0 | `u/rho` byte-exact |

匹配中位数提升为 `1.844×`，即 `+84.40%`；优化版四次运行的变异系数为
`1.47%`。本机 `cuDeviceCanAccessPeer` 返回不可用，日志确认实际走
pinned-host 异步回退，因此该数字只适用于上述网格、2 × RTX 2080 Ti、驱动
595.79 和本地单进程 `[2,1,1]` 分解。P2P/NVLink、多节点网络、其他 GPU 或明显
不同的网格规模需要单独测量。完整命令、每组样本和逐位检查结果保存在
`test-output/optimization-final-cb4beb3b/ab/BENCHMARK_REPORT.md`。

## 来源与许可

CULBM 的来源、版权、第三方组件与分发要求见 `NOTICE.md` 和
`THIRD_PARTY_LICENSES/`。若分发 CULBM 源码、二进制或其生成物，应一并保留这些
文件及各源文件中已有的版权和许可声明。
