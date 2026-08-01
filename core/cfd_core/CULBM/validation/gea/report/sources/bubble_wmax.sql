SELECT
    'h=' ||
        rtrim(rtrim(printf('%.4f', CAST(w.h_m AS REAL)), '0'), '.') ||
        ' m' AS grid,
    CAST(w.h_m AS REAL) AS h_m,
    CAST(w.culbm_value AS REAL) AS culbm_w_max,
    CASE
        WHEN NULLIF(w.reference_value, '') IS NULL THEN NULL
        ELSE CAST(w.reference_value AS REAL)
    END AS gea_av15_w_max,
    CASE
        WHEN NULLIF(w.difference, '') IS NULL THEN NULL
        ELSE CAST(w.difference AS REAL)
    END AS difference_m_s,
    CASE
        WHEN NULLIF(w.relative_difference_percent, '') IS NULL THEN NULL
        ELSE CAST(w.relative_difference_percent AS REAL)
    END AS relative_difference_percent,
    CAST(t.culbm_value AS REAL) AS culbm_theta_max,
    CASE
        WHEN NULLIF(t.reference_value, '') IS NULL THEN NULL
        ELSE CAST(t.reference_value AS REAL)
    END AS gea_av15_theta_max,
    CASE
        WHEN NULLIF(t.relative_difference_percent, '') IS NULL THEN NULL
        ELSE CAST(t.relative_difference_percent AS REAL)
    END AS theta_relative_difference_percent,
    w.status AS status,
    w.grid_registration AS grid_registration
FROM main_grid_comparison AS w
LEFT JOIN main_grid_comparison AS t
    ON t."case" = w."case"
   AND CAST(t.h_m AS REAL) = CAST(w.h_m AS REAL)
   AND t.metric = 'theta_prime_max_K'
WHERE w."case" = 'rising_thermal'
  AND w.metric = 'w_max_m_s'
  AND w.status = 'complete'
  AND NULLIF(w.culbm_value, '') IS NOT NULL
ORDER BY CAST(w.h_m AS REAL) DESC
