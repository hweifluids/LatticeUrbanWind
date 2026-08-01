# 报告图表映射

| 报告段落 | 分析问题 | 图形 | 字段 | 结论 | 来源 |
|---|---|---|---|---|---|
| 热泡速度 | w_max 是否趋近 AV15 | 分组柱形 | grid, culbm_w_max, gea_av15_w_max | 完成网格维持较高速度平台 | sources/bubble_wmax.sql |
| 密度流前缘 | 前缘是否随加密趋近 AV75 | 分组柱形 | grid, culbm_front_km, gea_av75_midpoint_km | 偏差变号、非单调 | sources/density_front.sql |

两图重复使用分组柱形是有意的：两段都比较少量离散网格上的同单位实际值与文献基线，不是时间趋势。使用蓝/中性基线的两根语义由共享 reader 处理，并由表格/tooltip 保留精确值。
