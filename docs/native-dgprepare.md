# `dgprepare` 原生几何合并

## 兼容目标

`luw dgprepare <deck>` 取代旧 `core/datagen_core/dgPrepare_stlinput.py`，且生产
路径不启动或嵌入 Python。命令继续读取 `.luwdg`/`.luwpf` 与
`building_db` 中唯一 STL，写出 `proj_temp/<casename>_DG.stl`、原 deck 的
`.bak`，并更新 `si_x_cfd`、`si_y_cfd`、`si_z_cfd`。

原生命令保留旧实现的两条几何路径：

- 封闭且拓扑合法的 building STL 与 base block 执行真实 boolean union；成功时
  输出 `Boolean union succeeded.`、`Union method: boolean` 和
  `Union message: Boolean union succeeded.`。
- 输入不是体、Manifold 状态错误、结果为空、转换失败或 boolean 抛出异常时，
  输出 `Boolean union failed. Falling back to mesh concatenation. Error: ...`，
  再按旧顺序输出 concatenation 点/面计数、cleanup 提示、cleanup 后点/面计数，
  最后输出 `Union method: concatenation` 与同一失败消息。

## 原生计算步骤

1. VTK 以合并同坐标顶点的方式读取 STL，并将位置提升到 double。
2. 按 building bounds、`base_height`、`x_exp_rat`、`y_exp_rat` 建立封闭 base
   block；其顶面与 building 最低面相接。
3. `vtkTriangleFilter` 将两份 `vtkPolyData` 三角化，再转换成 double/64-bit index
   的 `manifold::MeshGL64`。三角顶点顺序保持不变。
4. 分别构造并检查 building、base 的 `manifold::Manifold` 状态和非空性，再用
   `manifold::OpType::Add` 求并集，并再次检查结果。
5. `GetMeshGL64()` 的 position、triangle 与 merge vectors 转回 VTK；merge vectors
   对应的 property vertices 会恢复成同一拓扑顶点。
6. boolean 成功结果或失败后的 concatenation 结果都执行相同平移，使最小点成为
   `[0, 0, -base_height]`，随后写 binary STL 和 deck 范围。

生产依赖由 manifest 中的 Manifold 3.4.1 提供；
`LUW_REQUIRE_FULL_DEPENDENCIES=ON` 时缺少 `manifold::manifold` 会在配置阶段失败。
仅用于依赖检查的 `LUW_REQUIRE_FULL_DEPENDENCIES=OFF` 构建允许缺少 Manifold，
但会明确走上述 concatenation 回退，不会伪报 boolean 成功。

## 已锁定的 oracle

`native.validate_and_dgprepare` 每次生成两类输入：

- 四面体底面朝向错误的 4-face STL：旧实现报“不是 volume”并回退；原生也必须
  报 boolean 失败、使用 concatenation，且输出 bounds 为
  `[0, 0, -2]..[15, 40, 10]`。
- 封闭 12-face cube：带 Manifold 的生产构建必须报告 boolean 成功；输出 bounds
  同上、watertight，体积为 `3200`。旧输出为 16 vertices/28 faces，但三角剖分
  数量不是兼容契约，原生测试不锁定该实现细节。

旧 oracle 与生成物保留在忽略目录
`build-oracle-retained/dgprepare_boolean_oracle_20260719_224300` 和
`build-oracle-retained/dgprepare_boolean_success_oracle_20260719_224315`。
Manifold 3.4.1 的真实 Release 链接分支及上述成功/回退 fixture 已在 2026-07-20
生产总构建与 30/30 CTest 中通过。
尚未完成真实城市级 STL 的 Release 吞吐与峰值内存对比。
