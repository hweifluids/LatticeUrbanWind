SELECT
    'h=' ||
        rtrim(rtrim(printf('%.4f', CAST(h_m AS REAL)), '0'), '.') ||
        ' m' AS grid,
    CAST(h_m AS REAL) AS h_m,
    CAST(culbm_value AS REAL) / 1000.0 AS culbm_front_km,
    CASE
        WHEN NULLIF(reference_value, '') IS NULL THEN NULL
        ELSE CAST(reference_value AS REAL) / 1000.0
    END AS gea_av75_midpoint_km,
    CASE
        WHEN NULLIF(reference_lower, '') IS NULL THEN NULL
        ELSE CAST(reference_lower AS REAL) / 1000.0
    END AS gea_av75_lower_km,
    CASE
        WHEN NULLIF(reference_upper, '') IS NULL THEN NULL
        ELSE CAST(reference_upper AS REAL) / 1000.0
    END AS gea_av75_upper_km,
    CASE
        WHEN NULLIF(difference, '') IS NULL THEN NULL
        ELSE CAST(difference AS REAL)
    END AS difference_m,
    CASE
        WHEN NULLIF(relative_difference_percent, '') IS NULL THEN NULL
        ELSE CAST(relative_difference_percent AS REAL)
    END AS relative_difference_percent,
    status,
    grid_registration
FROM main_grid_comparison
WHERE "case" = 'density_current'
  AND metric = 'front_x_m'
  AND status = 'complete'
  AND NULLIF(culbm_value, '') IS NOT NULL
ORDER BY CAST(h_m AS REAL) DESC
