# CULBM / GEA 可移植技术报告

`build_report.py` 从 `../analysis/` 的最新审计 CSV 和
`../reference_metrics.csv` 构建 canonical `artifact.json`。构建器会复制输入快照，
用内存 SQLite 实际执行 `sources/*.sql`，并生成供 source affordance 审计的查询、
派生数据与 `CHART_MAP.md`。

在此目录运行：

```powershell
python .\build_report.py
```

随后从 Data Analytics 插件根目录执行唯一的 HTML 交付路径：

```powershell
npm run report:deliver -- `
  --input C:\1_Development\LatticeUrbanWind\core\cfd_core\CULBM\validation\gea\report\artifact.json `
  --output C:\1_Development\LatticeUrbanWind\core\cfd_core\CULBM\validation\gea\report\report.html
```

当上游计算或 `analyze_results.py` 重跑后，重复以上两条命令即可刷新报告。不要手工
编辑 `artifact.json`、`data/`、`sources/` 或 `report.html`；它们都是可再生输出。
