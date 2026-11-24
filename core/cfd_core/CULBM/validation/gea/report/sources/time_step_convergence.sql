SELECT
    ROW_NUMBER() OVER (
        ORDER BY "case" ASC, CAST(h_m AS REAL) ASC, CAST(dt_s AS REAL) DESC
    ) AS sequence,
    CASE "case"
        WHEN 'rising_thermal' THEN 'Rising thermal bubble'
        WHEN 'density_current' THEN 'Density current'
        ELSE "case"
    END AS case_label,
    CAST(h_m AS REAL) AS h_m,
    CAST(dt_s AS REAL) AS dt_s,
    CASE status
        WHEN 'complete' THEN '完成'
        WHEN 'failed' THEN '失败'
        ELSE '未完成'
    END AS status_label,
    CASE metric
        WHEN 'w_max_m_s' THEN 'w_max (m/s)'
        WHEN 'front_x_m' THEN 'front_x (m)'
        ELSE metric
    END AS metric_label,
    CAST(NULLIF(value, '') AS REAL) AS metric_value,
    CAST(NULLIF(max_mach, '') AS REAL) AS max_mach,
    CAST(NULLIF(relative_to_dt0p1_percent, '') AS REAL)
        AS relative_to_dt0p1_percent,
    CAST(NULLIF(next_coarser_dt_s, '') AS REAL) AS next_coarser_dt_s,
    CAST(NULLIF(relative_to_next_coarser_dt_percent, '') AS REAL)
        AS relative_to_next_coarser_dt_percent
FROM time_step_convergence
ORDER BY "case" ASC, CAST(h_m AS REAL) ASC, CAST(dt_s AS REAL) DESC
