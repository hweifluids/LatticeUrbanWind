WITH main_unique AS (
    SELECT DISTINCT "case", CAST(h_m AS REAL) AS h_m, status, run_name
    FROM main_grid_comparison
),
bubble_fine AS (
    SELECT
        CAST(w.h_m AS REAL) AS h_m,
        CAST(w.culbm_value AS REAL) AS culbm_w_max,
        CAST(w.reference_value AS REAL) AS gea_w_max,
        CAST(w.relative_difference_percent AS REAL) AS w_relative_percent,
        CAST(t.culbm_value AS REAL) AS culbm_theta_max,
        CAST(t.reference_value AS REAL) AS gea_theta_max,
        CAST(t.relative_difference_percent AS REAL) AS theta_relative_percent
    FROM main_grid_comparison AS w
    LEFT JOIN main_grid_comparison AS t
      ON t."case" = w."case"
     AND CAST(t.h_m AS REAL) = CAST(w.h_m AS REAL)
     AND t.metric = 'theta_prime_max_K'
    WHERE w."case" = 'rising_thermal'
      AND w.metric = 'w_max_m_s'
      AND w.status = 'complete'
      AND NULLIF(w.reference_value, '') IS NOT NULL
    ORDER BY CAST(w.h_m AS REAL) ASC
    LIMIT 1
),
bubble_finest_all AS (
    SELECT
        CAST(w.h_m AS REAL) AS h_m,
        CAST(w.culbm_value AS REAL) AS culbm_w_max,
        CASE
            WHEN NULLIF(w.reference_value, '') IS NULL THEN NULL
            ELSE CAST(w.reference_value AS REAL)
        END AS gea_w_max,
        CAST(t.culbm_value AS REAL) AS culbm_theta_max
    FROM main_grid_comparison AS w
    LEFT JOIN main_grid_comparison AS t
      ON t."case" = w."case"
     AND CAST(t.h_m AS REAL) = CAST(w.h_m AS REAL)
     AND t.metric = 'theta_prime_max_K'
    WHERE w."case" = 'rising_thermal'
      AND w.metric = 'w_max_m_s'
      AND w.status = 'complete'
      AND NULLIF(w.culbm_value, '') IS NOT NULL
    ORDER BY CAST(w.h_m AS REAL) ASC
    LIMIT 1
),
density_ref AS (
    SELECT
        CAST(h_m AS REAL) AS h_m,
        CAST(culbm_value AS REAL) AS culbm_front_m,
        CAST(reference_value AS REAL) AS gea_front_m,
        CAST(difference AS REAL) AS difference_m,
        CAST(relative_difference_percent AS REAL) AS relative_percent
    FROM main_grid_comparison
    WHERE "case" = 'density_current'
      AND metric = 'front_x_m'
      AND status = 'complete'
      AND NULLIF(reference_value, '') IS NOT NULL
),
density_finest_all AS (
    SELECT
        CAST(h_m AS REAL) AS h_m,
        CAST(culbm_value AS REAL) AS culbm_front_m,
        CASE
            WHEN NULLIF(reference_value, '') IS NULL THEN NULL
            ELSE CAST(reference_value AS REAL)
        END AS gea_front_m
    FROM main_grid_comparison
    WHERE "case" = 'density_current'
      AND metric = 'front_x_m'
      AND status = 'complete'
      AND NULLIF(culbm_value, '') IS NOT NULL
    ORDER BY CAST(h_m AS REAL) ASC
    LIMIT 1
)
SELECT
    (SELECT COUNT(*) FROM main_unique) AS main_total,
    (SELECT COUNT(*) FROM main_unique WHERE status = 'complete') AS main_complete,
    (SELECT COUNT(*) FROM main_unique WHERE status = 'failed') AS main_failed,
    (SELECT COUNT(*) FROM main_unique WHERE status = 'incomplete') AS main_incomplete,
    (SELECT h_m FROM bubble_fine) AS bubble_fine_h_m,
    (SELECT culbm_w_max FROM bubble_fine) AS bubble_culbm_w_max,
    (SELECT gea_w_max FROM bubble_fine) AS bubble_gea_w_max,
    (SELECT w_relative_percent FROM bubble_fine) AS bubble_w_relative_percent,
    (SELECT culbm_theta_max FROM bubble_fine) AS bubble_culbm_theta_max,
    (SELECT gea_theta_max FROM bubble_fine) AS bubble_gea_theta_max,
    (SELECT theta_relative_percent FROM bubble_fine) AS bubble_theta_relative_percent,
    (SELECT h_m FROM bubble_finest_all) AS bubble_finest_all_h_m,
    (SELECT culbm_w_max FROM bubble_finest_all) AS bubble_finest_all_w_max,
    (SELECT gea_w_max FROM bubble_finest_all) AS bubble_finest_all_gea_w_max,
    (SELECT culbm_theta_max FROM bubble_finest_all) AS bubble_finest_all_theta_max,
    (SELECT h_m FROM density_finest_all) AS density_finest_all_h_m,
    (SELECT culbm_front_m FROM density_finest_all) AS density_finest_all_front_m,
    (SELECT gea_front_m FROM density_finest_all) AS density_finest_all_gea_front_m,
    (SELECT relative_percent FROM density_ref WHERE h_m = 100.0) AS density_h100_relative_percent,
    (SELECT difference_m FROM density_ref WHERE h_m = 100.0) AS density_h100_difference_m,
    (SELECT relative_percent FROM density_ref WHERE h_m = 50.0) AS density_h50_relative_percent,
    (SELECT difference_m FROM density_ref WHERE h_m = 50.0) AS density_h50_difference_m,
    (SELECT relative_percent FROM density_ref WHERE h_m = 25.0) AS density_h25_relative_percent,
    (SELECT difference_m FROM density_ref WHERE h_m = 25.0) AS density_h25_difference_m,
    (
        SELECT COUNT(*)
        FROM sensitivity_comparison
        WHERE lower(run_group) <> 'registration_sensitivity'
          AND status = 'complete'
    ) AS robustness_complete,
    (
        SELECT COUNT(*)
        FROM sensitivity_comparison
        WHERE lower(run_group) <> 'registration_sensitivity'
          AND status = 'failed'
    ) AS robustness_failed,
    (
        SELECT COUNT(*)
        FROM sensitivity_comparison
        WHERE lower(run_group) <> 'registration_sensitivity'
          AND status = 'incomplete'
    ) AS robustness_incomplete
