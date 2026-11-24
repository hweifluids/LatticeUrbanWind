SELECT
    ROW_NUMBER() OVER (
        ORDER BY "case" ASC,
                 CASE scaling
                     WHEN 'fixed_dt' THEN 1
                     WHEN 'acoustic_dt_proportional_h' THEN 2
                     WHEN 'diffusive_dt_proportional_h2' THEN 3
                     ELSE 4
                 END
    ) AS sequence,
    CASE "case"
        WHEN 'rising_thermal' THEN 'Rising thermal bubble'
        WHEN 'density_current' THEN 'Density current'
        ELSE "case"
    END AS case_label,
    CASE scaling
        WHEN 'fixed_dt' THEN 'fixed dt'
        WHEN 'acoustic_dt_proportional_h' THEN 'dt proportional to h'
        WHEN 'diffusive_dt_proportional_h2' THEN 'dt proportional to h^2'
        ELSE scaling
    END AS scaling_label,
    CAST(baseline_h_m AS REAL) AS baseline_h_m,
    CAST(baseline_dt_s AS REAL) AS baseline_dt_s,
    CAST(refined_h_m AS REAL) AS refined_h_m,
    CAST(refined_dt_s AS REAL) AS refined_dt_s,
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
    CAST(NULLIF(baseline_value, '') AS REAL) AS baseline_value,
    CAST(NULLIF(refined_value, '') AS REAL) AS refined_value,
    CAST(NULLIF(relative_change_percent, '') AS REAL) AS relative_change_percent,
    CAST(NULLIF(baseline_tau, '') AS REAL) AS baseline_tau,
    CAST(NULLIF(refined_tau, '') AS REAL) AS refined_tau,
    CAST(NULLIF(baseline_max_mach, '') AS REAL) AS baseline_max_mach,
    CAST(NULLIF(refined_max_mach, '') AS REAL) AS refined_max_mach
FROM fine_grid_scaling_comparison
ORDER BY "case" ASC, sequence ASC
