SELECT
    ROW_NUMBER() OVER (
        ORDER BY sensitivity_comparison."case" ASC,
                 CAST(sensitivity_comparison.h_m AS REAL) DESC,
                 sensitivity_comparison.run_group ASC,
                 sensitivity_comparison.run_name ASC
    ) AS experiment_order,
    CASE sensitivity_comparison."case"
        WHEN 'rising_thermal' THEN 'Rising thermal bubble'
        WHEN 'density_current' THEN 'Density current'
        ELSE sensitivity_comparison."case"
    END AS case_label,
    sensitivity_comparison.run_group AS run_group,
    sensitivity_comparison.run_name AS run_name,
    CAST(sensitivity_comparison.h_m AS REAL) AS h_m,
    CAST(sensitivity_comparison.dt_s AS REAL) AS dt_s,
    sensitivity_comparison.thermal_lattice AS thermal_lattice,
    sensitivity_comparison.viscosity_mode AS viscosity_mode,
    CAST(NULLIF(sensitivity_comparison.nu_m2_s, '') AS REAL) AS nu_m2_s,
    CASE
        WHEN lower(sensitivity_comparison.run_name) LIKE '%_trt' THEN 'TRT 碰撞'
        WHEN lower(sensitivity_comparison.run_name) LIKE '%_srt' THEN 'SRT 大时间步'
        WHEN sensitivity_comparison.thermal_lattice = 'D3Q7' THEN '热标量 D3Q7'
        WHEN sensitivity_comparison.viscosity_mode LIKE 'numeric%' THEN '数值黏度解释'
        WHEN lower(sensitivity_comparison.run_group) = 'viscosity_profile'
            THEN '等效高度黏度'
        WHEN ABS(CAST(sensitivity_comparison.dt_s AS REAL) - 0.1) > 1.0e-12
            THEN '时间步敏感性'
        ELSE sensitivity_comparison.run_group
    END AS experiment,
    CASE sensitivity_comparison.status
        WHEN 'complete' THEN '完成'
        WHEN 'failed' THEN '失败'
        ELSE '未完成'
    END AS status_label,
    sensitivity_comparison.comparison_metric AS comparison_metric,
    CAST(NULLIF(sensitivity_comparison.matching_main_value, '') AS REAL)
        AS matching_main_value,
    CAST(NULLIF(sensitivity_comparison.relative_to_matching_main_percent, '') AS REAL)
        AS relative_to_main_percent,
    CAST(NULLIF(sensitivity_comparison.max_mach, '') AS REAL) AS max_mach,
    CAST(NULLIF(sensitivity_comparison.theta_prime_min_K, '') AS REAL)
        AS theta_prime_min_K,
    CAST(NULLIF(sensitivity_comparison.theta_prime_max_K, '') AS REAL)
        AS theta_prime_max_K,
    CAST(NULLIF(sensitivity_comparison.front_x_m, '') AS REAL) AS front_x_m,
    CAST(NULLIF(sensitivity_comparison.w_max_m_s, '') AS REAL) AS w_max_m_s
FROM sensitivity_comparison
WHERE lower(sensitivity_comparison.run_group) <> 'registration_sensitivity'
ORDER BY sensitivity_comparison."case" ASC,
         CAST(sensitivity_comparison.h_m AS REAL) DESC,
         sensitivity_comparison.run_group ASC,
         sensitivity_comparison.run_name ASC
