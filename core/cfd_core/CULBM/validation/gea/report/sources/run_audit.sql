WITH preferred AS (
    SELECT DISTINCT
        run_name,
        "case",
        CAST(h_m AS REAL) AS h_m,
        grid_registration
    FROM main_grid_comparison
    WHERE NULLIF(run_name, '') IS NOT NULL
),
selected AS (
    SELECT s.*
    FROM run_summary AS s
    INNER JOIN preferred AS p
        ON p.run_name = s.run_name
       AND p."case" = s."case"
       AND p.h_m = CAST(s.h_m AS REAL)
       AND p.grid_registration = s.grid_registration
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY selected."case" ASC, CAST(selected.h_m AS REAL) DESC
    ) AS run_order,
    CASE selected."case"
        WHEN 'rising_thermal' THEN 'Rising thermal bubble'
        WHEN 'density_current' THEN 'Density current'
        ELSE selected."case"
    END AS case_label,
    CAST(selected.h_m AS REAL) AS h_m,
    CASE selected.status
        WHEN 'complete' THEN '完成'
        WHEN 'failed' THEN '失败'
        ELSE '未完成'
    END AS status_label,
    selected.status_detail AS status_detail,
    CAST(selected.observed_time_s AS REAL) AS observed_time_s,
    CAST(selected.end_time_s AS REAL) AS target_time_s,
    100.0 * CAST(selected.completion_fraction AS REAL) AS completion_percent,
    CAST(selected.tau AS REAL) AS tau,
    CASE
        WHEN NULLIF(selected.max_mach, '') IS NOT NULL
            THEN CAST(selected.max_mach AS REAL)
        WHEN NULLIF(selected.last_valid_max_mach, '') IS NOT NULL
            THEN CAST(selected.last_valid_max_mach AS REAL)
        ELSE NULL
    END AS diagnostic_max_mach,
    CASE selected."case"
        WHEN 'rising_thermal' THEN 'w_max (m/s)'
        WHEN 'density_current' THEN 'front_x (m)'
        ELSE '关键指标'
    END AS key_metric,
    CASE selected."case"
        WHEN 'rising_thermal' THEN CAST(NULLIF(selected.w_max_m_s, '') AS REAL)
        WHEN 'density_current' THEN CAST(NULLIF(selected.front_x_m, '') AS REAL)
        ELSE NULL
    END AS key_value
FROM selected
ORDER BY selected."case" ASC, CAST(selected.h_m AS REAL) DESC
