#!/usr/bin/env python3
"""Build the canonical portable report artifact for the CULBM/GEA validation.

The analysis CSV files remain the canonical numerical inputs.  This builder copies
the reviewed inputs into the report package, loads them into an in-memory SQLite
database, executes the SQL saved under ``sources/``, and emits ``artifact.json``.
Run the Data Analytics portable-report packager afterwards to produce
``report.html``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TITLE = "CULBM 对 GEA 两个大气基准的数值验证"
PAPER_URL = "https://arxiv.org/abs/2302.04836"
GEA_COMMIT_URL = (
    "https://github.com/GEA-Geophysical-and-Environmental-Apps/GEA/"
    "tree/bdf0b929fc92163cb946520e5f8bd3d1fd6ca45f"
)


BUBBLE_SQL = r"""
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
""".strip()


DENSITY_SQL = r"""
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
""".strip()


RUN_AUDIT_SQL = r"""
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
""".strip()


ROBUSTNESS_SQL = r"""
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
""".strip()


TIME_STEP_SQL = r"""
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
""".strip()


SCALING_SQL = r"""
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
""".strip()


DERIVED_SQL = r"""
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
""".strip()


SQL_QUERIES = {
    "bubble_wmax.sql": BUBBLE_SQL,
    "density_front.sql": DENSITY_SQL,
    "run_audit.sql": RUN_AUDIT_SQL,
    "robustness.sql": ROBUSTNESS_SQL,
    "time_step_convergence.sql": TIME_STEP_SQL,
    "fine_grid_scaling.sql": SCALING_SQL,
    "derived_metrics.sql": DERIVED_SQL,
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def load_csv_table(connection: sqlite3.Connection, table: str, path: Path) -> None:
    columns, rows = read_csv(path)
    column_sql = ", ".join(f"{sql_identifier(column)} TEXT" for column in columns)
    connection.execute(f"CREATE TABLE {sql_identifier(table)} ({column_sql})")
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = (
        f"INSERT INTO {sql_identifier(table)} "
        f"({', '.join(sql_identifier(column) for column in columns)}) "
        f"VALUES ({placeholders})"
    )
    connection.executemany(
        insert_sql,
        [[row.get(column, "") for column in columns] for row in rows],
    )


def query_rows(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description or []]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0]))
        writer.writeheader()
        writer.writerows(row_list)


def as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int:
    numeric = as_number(value)
    return int(numeric) if numeric is not None else 0


def fmt(value: Any, digits: int = 2) -> str:
    numeric = as_number(value)
    return "无可用值" if numeric is None else f"{numeric:.{digits}f}"


def fmt_signed(value: Any, digits: int = 2, suffix: str = "") -> str:
    numeric = as_number(value)
    return "无可用值" if numeric is None else f"{numeric:+.{digits}f}{suffix}"


def fmt_grid(value: Any) -> str:
    numeric = as_number(value)
    return "无可用值" if numeric is None else f"{numeric:.4f}".rstrip("0").rstrip(".")


def find_row(
    rows: list[dict[str, Any]], **criteria: Any
) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return None


def source(
    source_id: str,
    label: str,
    *,
    path: str | None = None,
    href: str | None = None,
    sql: str | None = None,
    description: str | None = None,
    tables: list[str] | None = None,
    filters: list[str] | None = None,
    definitions: list[str] | None = None,
    executed_at: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"id": source_id, "label": label}
    if path:
        item["path"] = path
    if href:
        item["href"] = href
    if sql:
        item["query"] = {
            "engine": "SQLite (Python sqlite3)",
            "sql": sql,
            "description": description,
            "language": "sql",
            "executed_at": executed_at,
            "tables_used": tables or [],
            "filters": filters or [],
            "metric_definitions": definitions or [],
        }
    return item


def markdown_block(
    block_id: str, body: str, source_id: str | None = None
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "id": block_id,
        "type": "markdown",
        "body": body,
        "layout": "full",
    }
    if source_id:
        block["sourceId"] = source_id
    return block


def build_narrative(
    derived: dict[str, Any],
    bubble_rows: list[dict[str, Any]],
    density_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    time_step_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
) -> dict[str, str]:
    total = as_int(derived.get("main_total"))
    complete = as_int(derived.get("main_complete"))
    failed = as_int(derived.get("main_failed"))
    incomplete = as_int(derived.get("main_incomplete"))

    bubble_comparable = [
        row
        for row in bubble_rows
        if as_number(row.get("gea_av15_w_max")) is not None
    ]
    bubble_fine_comparable = min(
        bubble_comparable,
        key=lambda row: as_number(row.get("h_m")) or float("inf"),
        default=None,
    )
    bubble_finest_all = min(
        bubble_rows,
        key=lambda row: as_number(row.get("h_m")) or float("inf"),
        default=None,
    )
    if bubble_fine_comparable:
        bubble_h = fmt_grid(bubble_fine_comparable.get("h_m"))
        bubble_comparison_text = (
            f"在最细的文献同网格比较点 **h={bubble_h} m**，CULBM 的 "
            f"`w_max` 为 **{fmt(bubble_fine_comparable.get('culbm_w_max'), 2)} m/s**，"
            f"GEA AV15 为 {fmt(bubble_fine_comparable.get('gea_av15_w_max'), 2)} m/s，"
            f"相对偏差 **{fmt_signed(bubble_fine_comparable.get('relative_difference_percent'), 2, '%')}**。"
            f"同一网格的 `theta'_max` 为 "
            f"{fmt(bubble_fine_comparable.get('culbm_theta_max'), 3)} K，文献值为 "
            f"{fmt(bubble_fine_comparable.get('gea_av15_theta_max'), 3)} K，偏差 "
            f"{fmt_signed(bubble_fine_comparable.get('theta_relative_difference_percent'), 2, '%')}。"
        )
    else:
        bubble_comparison_text = "当前快照没有可与 GEA 表 1 对齐的已完成热泡主网格。"

    bubble_refinement_text = ""
    if (
        bubble_finest_all
        and as_number(bubble_finest_all.get("gea_av15_w_max")) is None
    ):
        bubble_refinement_text = (
            f" 更细的 CULBM 主网格 **h={fmt_grid(bubble_finest_all.get('h_m'))} m** "
            f"给出 `w_max={fmt(bubble_finest_all.get('culbm_w_max'), 2)} m/s`、"
            f"`theta'_max={fmt(bubble_finest_all.get('culbm_theta_max'), 3)} K`；"
            "GEA 表 1 没有该网格值，因此该点只用于观察 CULBM 自身的加密趋势，"
            "不计算或外推跨代码百分比偏差。"
        )
    bubble_text = bubble_comparison_text + bubble_refinement_text

    bubble_ordered = sorted(
        [
            row
            for row in bubble_rows
            if as_number(row.get("culbm_w_max")) is not None
        ],
        key=lambda row: as_number(row.get("h_m")) or float("inf"),
    )
    bubble_spatial_detail = ""
    if len(bubble_ordered) >= 2:
        finest, previous = bubble_ordered[0], bubble_ordered[1]
        finest_w = as_number(finest.get("culbm_w_max"))
        previous_w = as_number(previous.get("culbm_w_max"))
        finest_theta = as_number(finest.get("culbm_theta_max"))
        previous_theta = as_number(previous.get("culbm_theta_max"))
        w_change = (
            100.0 * (finest_w - previous_w) / abs(previous_w)
            if finest_w is not None and previous_w not in (None, 0)
            else None
        )
        theta_change = (
            100.0 * (finest_theta - previous_theta) / abs(previous_theta)
            if finest_theta is not None and previous_theta not in (None, 0)
            else None
        )
        finest_audit = next(
            (
                row
                for row in audit_rows
                if row.get("case_label") == "Rising thermal bubble"
                and as_number(row.get("h_m")) is not None
                and abs(
                    (as_number(row.get("h_m")) or 0)
                    - (as_number(finest.get("h_m")) or 0)
                )
                < 1.0e-9
            ),
            None,
        )
        bubble_spatial_detail = (
            f"从 h={fmt_grid(previous.get('h_m'))} m 加密到 "
            f"h={fmt_grid(finest.get('h_m'))} m 后，`w_max` 变化 "
            f"**{fmt_signed(w_change, 2, '%')}**、`theta'_max` 变化 "
            f"**{fmt_signed(theta_change, 2, '%')}**；"
            f"新主计算 `max Mach={fmt((finest_audit or {}).get('diagnostic_max_mach'), 3)}`。"
        )

    density_comparable = [
        row
        for row in density_rows
        if as_number(row.get("gea_av75_midpoint_km")) is not None
    ]
    density_parts = [
        (
            f"h={fmt(row.get('h_m'), 3).rstrip('0').rstrip('.')} m："
            f"{fmt_signed(row.get('relative_difference_percent'), 2, '%')} "
            f"({fmt_signed(row.get('difference_m'), 1, ' m')})"
        )
        for row in density_comparable
    ]
    density_text = (
        "；".join(density_parts)
        if density_parts
        else "当前快照没有同时具备 CULBM 与 GEA 前缘参考值的网格。"
    )
    density_signs = {
        1 if (as_number(row.get("relative_difference_percent")) or 0) > 0 else -1
        for row in density_comparable
        if as_number(row.get("relative_difference_percent")) not in (None, 0)
    }
    density_interpretation = (
        "偏差随加密发生变号，不能据此宣称向 GEA AV75 单调收敛。"
        if len(density_signs) > 1
        else "当前网格序列尚不足以证明对 GEA AV75 的单调收敛。"
    )
    density_unmatched = [
        row
        for row in density_rows
        if as_number(row.get("gea_av75_midpoint_km")) is None
    ]
    density_unmatched_text = ""
    if density_unmatched:
        unmatched_parts = [
            (
                f"h={fmt_grid(row.get('h_m'))} m "
                f"(CULBM {fmt(row.get('culbm_front_km'), 3)} km)"
            )
            for row in density_unmatched
        ]
        density_unmatched_text = (
            " GEA 表 3 没有以下同网格 AV75 值："
            + "、".join(unmatched_parts)
            + "；这些点只展示 CULBM 加密轨迹，不计入跨代码百分比误差。"
        )

    density_ordered = sorted(
        [
            row
            for row in density_rows
            if as_number(row.get("culbm_front_km")) is not None
        ],
        key=lambda row: as_number(row.get("h_m")) or float("inf"),
    )
    density_spatial_detail = ""
    if len(density_ordered) >= 2:
        finest, previous = density_ordered[0], density_ordered[1]
        finest_front = as_number(finest.get("culbm_front_km"))
        previous_front = as_number(previous.get("culbm_front_km"))
        front_change = (
            100.0 * (finest_front - previous_front) / abs(previous_front)
            if finest_front is not None and previous_front not in (None, 0)
            else None
        )
        finest_audit = next(
            (
                row
                for row in audit_rows
                if row.get("case_label") == "Density current"
                and as_number(row.get("h_m")) is not None
                and abs(
                    (as_number(row.get("h_m")) or 0)
                    - (as_number(finest.get("h_m")) or 0)
                )
                < 1.0e-9
            ),
            None,
        )
        density_spatial_detail = (
            f"从 h={fmt_grid(previous.get('h_m'))} m 加密到 "
            f"h={fmt_grid(finest.get('h_m'))} m 后，前缘变化 "
            f"**{fmt_signed(front_change, 2, '%')}**；"
            f"新主计算 `max Mach={fmt((finest_audit or {}).get('diagnostic_max_mach'), 3)}`。"
        )

    failed_rows = [row for row in audit_rows if row.get("status_label") == "失败"]
    failed_text = "；".join(
        f"{row.get('case_label')} h={fmt_grid(row.get('h_m'))} m"
        for row in failed_rows
    )
    if not failed_text:
        failed_text = "无"

    completed_robustness = [
        row
        for row in robustness_rows
        if row.get("status_label") == "完成"
        and as_number(row.get("relative_to_main_percent")) is not None
    ]
    max_robustness = max(
        completed_robustness,
        key=lambda row: abs(as_number(row.get("relative_to_main_percent")) or 0),
        default=None,
    )
    robustness_summary = (
        f"共有 **{len(completed_robustness)}** 个已完成控制组可直接和同网格主结果比较；"
        f"其主比较量最大相对变化为 "
        f"**{fmt_signed(max_robustness.get('relative_to_main_percent'), 2, '%')}**"
        f"（{max_robustness.get('case_label')}，{max_robustness.get('experiment')}）"
        if max_robustness
        else "当前快照没有已完成且可与同网格主结果直接比较的控制组。"
    )

    scaling_names = {
        "fixed dt": "固定 dt",
        "dt proportional to h": "dt∝h",
        "dt proportional to h^2": "dt∝h²",
    }
    scaling_case_parts: list[str] = []
    for case_label in ["Rising thermal bubble", "Density current"]:
        case_rows = [
            row for row in scaling_rows if row.get("case_label") == case_label
        ]
        if not case_rows:
            continue
        path_parts = []
        for row in case_rows:
            path_name = scaling_names.get(
                str(row.get("scaling_label")), str(row.get("scaling_label"))
            )
            if (
                row.get("status_label") == "完成"
                and as_number(row.get("relative_change_percent")) is not None
            ):
                result = (
                    f"{fmt_signed(row.get('relative_change_percent'), 2, '%')}"
                    f" (Ma={fmt(row.get('refined_max_mach'), 3)})"
                )
            else:
                result = str(row.get("status_label") or "状态未知")
            path_parts.append(f"{path_name} {result}")
        case_cn = "热泡" if case_label.startswith("Rising") else "密度流"
        scaling_case_parts.append(f"{case_cn}：" + "、".join(path_parts))
    scaling_detail = (
        "h/2 加密相对上一层最细主网格的三路径结果为："
        + "；".join(scaling_case_parts)
        + "。"
        if scaling_case_parts
        else "h/2 加密三路径比较尚无可用结果。"
    )
    scaling_section = (
        "## h/2 三条缩放路径区分空间与时间离散效应\n\n"
        f"{scaling_detail} 固定 `dt` 延续原矩阵但同时提高格子 Mach；"
        "`dt∝h` 近似保持格子速度/Mach；`dt∝h²` 近似保持 `nu_LBM` 与 `tau`。"
        "三条路径必须并列解释，不能把任一路径单独当作纯空间收敛。"
    )

    time_step_case_parts: list[str] = []
    for case_label in ["Rising thermal bubble", "Density current"]:
        case_complete = [
            row
            for row in time_step_rows
            if row.get("case_label") == case_label
            and row.get("status_label") == "完成"
            and as_number(row.get("metric_value")) is not None
        ]
        if not case_complete:
            continue
        finest_h = min(
            as_number(row.get("h_m")) or float("inf") for row in case_complete
        )
        finest_rows = sorted(
            [
                row
                for row in case_complete
                if as_number(row.get("h_m")) is not None
                and abs((as_number(row.get("h_m")) or 0) - finest_h) < 1.0e-9
            ],
            key=lambda row: as_number(row.get("dt_s")) or float("inf"),
            reverse=True,
        )
        path_parts = [
            (
                f"dt={fmt(row.get('dt_s'), 3)} s："
                f"{fmt(row.get('metric_value'), 4)}，"
                f"Ma={fmt(row.get('max_mach'), 3)}，"
                f"相对 dt=0.1 为 "
                f"{fmt_signed(row.get('relative_to_dt0p1_percent'), 2, '%')}"
            )
            for row in finest_rows
        ]
        case_cn = "热泡" if case_label.startswith("Rising") else "密度流"
        time_step_case_parts.append(
            f"{case_cn} h={fmt_grid(finest_h)} m：" + "；".join(path_parts)
        )
    time_step_detail = (
        "最细可用同网格时间步序列为：" + "；".join(time_step_case_parts) + "。"
        if time_step_case_parts
        else "同网格时间步收敛表尚无严格完成的可比较序列。"
    )
    time_step_section = (
        "## 同网格时间步扫描量化 Mach 与终态指标敏感性\n\n"
        f"{time_step_detail} 表中的“相对 dt=0.1”使用同一 case/h 的主时间步作基线，"
        "“逐级变化”只与相邻更粗时间步比较；两列分母不同，不能互换。"
    )

    dt_density_candidates = [
        row
        for row in completed_robustness
        if row.get("case_label") == "Density current"
        and row.get("experiment") == "时间步敏感性"
    ]
    dt_density = min(
        dt_density_candidates,
        key=lambda row: (
            as_number(row.get("h_m")) or float("inf"),
            as_number(row.get("dt_s")) or float("inf"),
        ),
        default=None,
    )
    dt_detail = (
        f"密度流最细已完成的最小时间步控制"
        f"（h={fmt_grid(dt_density.get('h_m'))} m，"
        f"dt={fmt(dt_density.get('dt_s'), 3)} s）"
        f"前缘变化为 "
        f"{fmt_signed(dt_density.get('relative_to_main_percent'), 2, '%')}，"
        f"`max Mach` 降至 {fmt(dt_density.get('max_mach'), 3)}。"
        if dt_density
        else "密度流减半时间步控制组尚无可用完整结果。"
    )

    refined_targets = {
        "Rising thermal bubble": 7.8125,
        "Density current": 12.5,
    }
    refined_control_rows = [
        row
        for row in robustness_rows
        if row.get("experiment") == "时间步敏感性"
        and row.get("case_label") in refined_targets
        and as_number(row.get("h_m")) is not None
        and abs(
            (as_number(row.get("h_m")) or 0)
            - refined_targets[str(row.get("case_label"))]
        )
        < 1.0e-9
    ]
    if refined_control_rows:
        refined_control_parts = []
        for row in refined_control_rows:
            if (
                row.get("status_label") == "完成"
                and as_number(row.get("relative_to_main_percent")) is not None
            ):
                outcome = (
                    f"完成，主比较量变化 "
                    f"{fmt_signed(row.get('relative_to_main_percent'), 2, '%')}，"
                    f"`max Mach={fmt(row.get('max_mach'), 3)}`"
                )
            else:
                outcome = str(row.get("status_label") or "状态未知")
            refined_control_parts.append(
                f"{row.get('case_label')} h={fmt_grid(row.get('h_m'))} m、"
                f"dt={fmt(row.get('dt_s'), 3)} s：{outcome}"
            )
        refined_control_detail = (
            "新增最细 D3Q19 主网格的减小时间步控制为："
            + "；".join(refined_control_parts)
            + "。这些控制只检验时间步/Mach 敏感性，不能补造缺失的 GEA 同网格基线。"
        )
    else:
        refined_control_detail = ""

    q7_density = next(
        (
            row
            for row in completed_robustness
            if row.get("case_label") == "Density current"
            and row.get("experiment") == "热标量 D3Q7"
        ),
        None,
    )
    q7_detail = (
        f"D3Q7 的前缘变化仅 "
        f"{fmt_signed(q7_density.get('relative_to_main_percent'), 2, '%')}，"
        f"但 `theta'_max` 达 {fmt(q7_density.get('theta_prime_max_K'), 3)} K，"
        "说明前缘单一指标会掩盖标量场色散/过冲。"
        if q7_density
        else "D3Q7 控制组尚无可用完整结果。"
    )

    stability_rows = [
        row
        for row in robustness_rows
        if row.get("experiment") in {"SRT 大时间步", "TRT 碰撞"}
    ]
    srt_stability = [
        row
        for row in stability_rows
        if row.get("experiment") == "SRT 大时间步"
    ]
    trt_stability = [
        row for row in stability_rows if row.get("experiment") == "TRT 碰撞"
    ]
    if (
        srt_stability
        and all(row.get("status_label") == "完成" for row in srt_stability)
        and trt_stability
        and all(row.get("status_label") == "失败" for row in trt_stability)
    ):
        stability_detail = (
            f"粗网格稳定性矩阵中，增大到 `dt=1.0 s` 的 SRT "
            f"**{len(srt_stability)}/{len(srt_stability)} 完成**，而保持 "
            f"`dt=0.1 s` 的 TRT **{len(trt_stability)}/{len(trt_stability)} 失败**；"
            "这说明把 `tau-0.5` 拉离零比只改变奇对称动量模态更关键，"
            "且标量/偶对称模态或界面欠解析仍可能触发失稳。"
        )
    elif stability_rows:
        stability_detail = (
            "粗网格稳定性控制组结果混合，必须结合表中的碰撞模型、时间步与状态逐项解释。"
        )
    else:
        stability_detail = "粗网格 SRT/TRT 稳定性矩阵尚无可用结果。"

    viscosity_profile_rows = [
        row
        for row in completed_robustness
        if row.get("experiment") == "等效高度黏度"
    ]
    if viscosity_profile_rows:
        viscosity_changes = [
            abs(as_number(row.get("relative_to_main_percent")) or 0)
            for row in viscosity_profile_rows
        ]
        viscosity_detail = (
            f"两档等效高度黏度在 h=31.25 m 热泡上的 `w_max` 变化绝对值为 "
            f"**{min(viscosity_changes):.2f}%–{max(viscosity_changes):.2f}%**，"
            "明显小于对 GEA 的系统偏差，因此地面常量运动黏度映射不是唯一主因。"
        )
    else:
        viscosity_detail = "等效高度黏度控制组尚无可用完整结果。"

    technical_summary = (
        "## 技术结论：可稳定复现主要流动，但未建立与 GEA 的严格定量等价\n\n"
        "**本工作应解释为 CULBM 的 Boussinesq 类比验证，而不是相同控制方程的代码复现。** "
        f"主矩阵共有 **{total}** 个完整终止时刻算例，其中 **{complete}** 个完成、"
        f"**{failed}** 个失败、**{incomplete}** 个未完成。"
        f"{bubble_text} 密度流前缘相对 GEA AV75 的偏差依次为：{density_text}。"
        f"{density_interpretation}{density_unmatched_text}\n\n"
        f"新增细网格的空间延拓为：{bubble_spatial_detail} "
        f"{density_spatial_detail}\n\n{scaling_detail}\n\n"
        "因此，当前证据支持“边界条件、初值和两个基准的主要动力学已被实算并可审计”，"
        "但不支持把 CULBM 与 GEA 的终态极值/前缘视为数值等价。"
    )

    bubble_section = (
        "## 热泡上升速度系统性偏高\n\n"
        f"{bubble_text} 完成网格上的 `w_max` 没有随加密向 AV15 表值靠拢，"
        "而是维持在更高的平台；这更像有效耗散与方程形式差异造成的系统偏差，"
        "而不是单一粗网格截断误差。下图纳入全部严格完成的 CULBM 主网格；"
        "没有同网格 GEA 表值的新加密点只显示 CULBM 柱，文献柱保持为空。"
        f" {bubble_spatial_detail}"
    )

    density_section = (
        "## 密度流前缘在加密过程中穿过文献基线\n\n"
        f"相对 GEA AV75 区间中点：{density_text}。{density_interpretation} "
        f"{density_unmatched_text} {density_spatial_detail}"
    )

    audit_section = (
        "## 两个最粗主网格失稳，属于验证结果而非缺失数据\n\n"
        f"严格完成判据下，主矩阵失败项为：**{failed_text}**。"
        "失败算例只保留到最后有效采样的诊断量，异常终态不会进入物理比较。"
        "两项粗网格的 SRT 松弛时间非常接近 `tau=0.5` 稳定性边界；"
        "这使“低黏度映射 + 粗空间步长”的碰撞稳定性成为首要数值原因，"
        "同时粗网格对初始扰动界面的解析不足会放大风险。"
    )

    robustness_section = (
        "## 控制组削弱了“时间步/Mach 数是主偏差来源”的解释\n\n"
        f"{robustness_summary} {dt_detail} {refined_control_detail} "
        f"{q7_detail} {stability_detail} "
        f"{viscosity_detail} "
        "这些结果表明，降低弱可压误差或改变标量格子会改变局部过冲与稳定裕度，"
        "但目前并未消除热泡速度平台或密度流前缘的非单调偏差；"
        "剩余差异仍需优先按控制方程、静力背景和耗散模型不一致解释。"
    )

    return {
        "technical_summary": technical_summary,
        "bubble_section": bubble_section,
        "density_section": density_section,
        "audit_section": audit_section,
        "scaling_section": scaling_section,
        "time_step_section": time_step_section,
        "robustness_section": robustness_section,
    }


def build_artifact(
    generated_at: str,
    datasets: dict[str, list[dict[str, Any]]],
    sources: list[dict[str, Any]],
    narrative: dict[str, str],
) -> dict[str, Any]:
    charts = [
        {
            "id": "bubble_wmax_chart",
            "title": "热泡终态最大上升速度：CULBM 与 GEA AV15",
            "subtitle": (
                "t=1020 s；全部已完成主网格；无同网格 GEA 表值时文献柱留空；单位 m/s"
            ),
            "showDescription": True,
            "intent": "comparison",
            "question": "CULBM 的热泡 w_max 是否随网格加密趋近 GEA AV15？",
            "rationale": (
                "完成网格是离散、同单位的分辨率比较；分组柱形能显示具有文献基线"
                "网格上的绝对差距，同时以缺失文献柱诚实保留更细 CULBM 点。"
            ),
            "comparisonContext": {
                "baseline": "GEA AV15, arXiv:2302.04836 Table 1",
                "grain": "每个已完成 CULBM 主网格；GEA 仅覆盖论文表 1 网格",
                "normalization": "t=1020 s 全域竖直速度最大值",
                "semanticFamily": "CULBM 对 GEA",
                "unit": "m/s",
            },
            "type": "bar",
            "dataset": "bubble_wmax",
            "sourceId": "src_bubble_wmax",
            "encodings": {
                "x": {"field": "grid", "type": "ordinal", "label": "网格步长 h"},
                "y": {
                    "fields": ["culbm_w_max", "gea_av15_w_max"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "w_max",
                    "unit": "m/s",
                },
                "tooltip": [
                    {"field": "h_m", "type": "quantitative", "label": "h", "unit": "m"},
                    {
                        "field": "relative_difference_percent",
                        "type": "quantitative",
                        "label": "相对偏差",
                        "unit": "%",
                    },
                    {
                        "field": "culbm_theta_max",
                        "type": "quantitative",
                        "label": "CULBM theta'_max",
                        "unit": "K",
                    },
                ],
            },
            "xAxisTitle": "网格步长 h",
            "yAxisTitle": "w_max (m/s)",
            "valueFormat": "number",
            "unit": "m/s",
            "layout": "full",
            "combinationRationale": (
                "两列均为同一物理量和单位，分别表示实际计算与文献基线。"
            ),
            "maxRows": 10,
        },
        {
            "id": "density_front_chart",
            "title": "密度流终态前缘：CULBM 与 GEA AV75",
            "subtitle": (
                "t=900 s，z=1200 m，theta'=-1 K；无同网格 GEA 表值时文献柱留空；单位 km"
            ),
            "showDescription": True,
            "intent": "comparison",
            "question": "CULBM 密度流前缘如何随网格加密接近或越过 GEA AV75？",
            "rationale": (
                "网格点是离散分辨率比较，分组柱形同时展示绝对前缘与可用文献中点；"
                "超出论文表 3 网格范围的文献值明确保留为空。"
            ),
            "comparisonContext": {
                "baseline": "GEA AV75, arXiv:2302.04836 Table 3",
                "grain": "每个已完成 CULBM 主网格；GEA 仅覆盖论文表 3 网格",
                "normalization": "theta'=-1 K 前缘位置",
                "semanticFamily": "CULBM 对 GEA",
                "unit": "km",
            },
            "type": "bar",
            "dataset": "density_front",
            "sourceId": "src_density_front",
            "encodings": {
                "x": {"field": "grid", "type": "ordinal", "label": "网格步长 h"},
                "y": {
                    "fields": ["culbm_front_km", "gea_av75_midpoint_km"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "前缘位置",
                    "unit": "km",
                },
                "tooltip": [
                    {
                        "field": "difference_m",
                        "type": "quantitative",
                        "label": "CULBM-文献中点",
                        "unit": "m",
                    },
                    {
                        "field": "relative_difference_percent",
                        "type": "quantitative",
                        "label": "相对偏差",
                        "unit": "%",
                    },
                    {
                        "field": "gea_av75_lower_km",
                        "type": "quantitative",
                        "label": "GEA 区间下界",
                        "unit": "km",
                    },
                    {
                        "field": "gea_av75_upper_km",
                        "type": "quantitative",
                        "label": "GEA 区间上界",
                        "unit": "km",
                    },
                ],
            },
            "xAxisTitle": "网格步长 h",
            "yAxisTitle": "前缘位置 (km)",
            "valueFormat": "number",
            "unit": "km",
            "layout": "full",
            "combinationRationale": (
                "两列均为相同阈值和时刻下的前缘位置，文献列采用区间中点。"
            ),
            "maxRows": 10,
        },
    ]

    tables = [
        {
            "id": "run_audit_table",
            "title": "主网格运行审计",
            "subtitle": "严格完成状态、目标时刻、松弛时间与最后有效/终态 Mach 数",
            "showDescription": True,
            "dataset": "run_audit",
            "defaultSort": {"field": "run_order", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_run_audit",
            "layout": "full",
            "columns": [
                {"field": "run_order", "label": "#", "format": "number"},
                {"field": "case_label", "label": "算例", "type": "text"},
                {"field": "h_m", "label": "h (m)", "format": "number"},
                {"field": "status_label", "label": "状态", "type": "text"},
                {
                    "field": "completion_percent",
                    "label": "完成度 (%)",
                    "format": "number",
                },
                {"field": "tau", "label": "tau", "format": "number"},
                {
                    "field": "diagnostic_max_mach",
                    "label": "诊断 max Mach",
                    "format": "number",
                },
                {"field": "key_metric", "label": "关键指标", "type": "text"},
                {"field": "key_value", "label": "终态值", "format": "number"},
            ],
        },
        {
            "id": "robustness_table",
            "title": "敏感性与稳定性控制组",
            "subtitle": "相对同算例、同网格主结果；旧半网格平移注册归档不在此表",
            "showDescription": True,
            "dataset": "robustness",
            "defaultSort": {"field": "experiment_order", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_robustness",
            "layout": "full",
            "columns": [
                {"field": "experiment_order", "label": "#", "format": "number"},
                {"field": "case_label", "label": "算例", "type": "text"},
                {"field": "h_m", "label": "h (m)", "format": "number"},
                {"field": "experiment", "label": "控制变量", "type": "text"},
                {"field": "status_label", "label": "状态", "type": "text"},
                {
                    "field": "relative_to_main_percent",
                    "label": "主比较量变化 (%)",
                    "format": "number",
                    "movement": True,
                    "role": "movement",
                    "semantic": "movement",
                },
                {"field": "max_mach", "label": "max Mach", "format": "number"},
                {
                    "field": "theta_prime_max_K",
                    "label": "theta'_max (K)",
                    "format": "number",
                },
            ],
        },
        {
            "id": "time_step_table",
            "title": "细网格时间步与 Mach 收敛",
            "subtitle": "逐级比较同一 D3Q19 网格；密度流关键值单位为 m",
            "showDescription": True,
            "dataset": "time_step_convergence",
            "defaultSort": {"field": "sequence", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_time_step_convergence",
            "layout": "full",
            "columns": [
                {"field": "sequence", "label": "#", "format": "number"},
                {"field": "case_label", "label": "算例", "type": "text"},
                {"field": "h_m", "label": "h (m)", "format": "number"},
                {"field": "dt_s", "label": "dt (s)", "format": "number"},
                {"field": "status_label", "label": "状态", "type": "text"},
                {"field": "metric_label", "label": "主指标", "type": "text"},
                {"field": "metric_value", "label": "终态值", "format": "number"},
                {"field": "max_mach", "label": "max Mach", "format": "number"},
                {
                    "field": "relative_to_dt0p1_percent",
                    "label": "相对 dt=0.1 (%)",
                    "format": "number",
                    "movement": True,
                    "role": "movement",
                    "semantic": "movement",
                },
                {
                    "field": "relative_to_next_coarser_dt_percent",
                    "label": "逐级变化 (%)",
                    "format": "number",
                    "movement": True,
                    "role": "movement",
                    "semantic": "movement",
                },
            ],
        },
        {
            "id": "fine_grid_scaling_table",
            "title": "h/2 加密的三条缩放路径",
            "subtitle": "均相对上一层最细主网格；固定 dt、等 Mach 与同 tau 分开解释",
            "showDescription": True,
            "dataset": "fine_grid_scaling",
            "defaultSort": {"field": "sequence", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_fine_grid_scaling",
            "layout": "full",
            "columns": [
                {"field": "sequence", "label": "#", "format": "number"},
                {"field": "case_label", "label": "算例", "type": "text"},
                {"field": "scaling_label", "label": "缩放", "type": "text"},
                {"field": "refined_dt_s", "label": "新 dt (s)", "format": "number"},
                {"field": "status_label", "label": "状态", "type": "text"},
                {"field": "metric_label", "label": "主指标", "type": "text"},
                {"field": "refined_value", "label": "新网格值", "format": "number"},
                {
                    "field": "relative_change_percent",
                    "label": "相对上一层 (%)",
                    "format": "number",
                    "movement": True,
                    "role": "movement",
                    "semantic": "movement",
                },
                {"field": "refined_tau", "label": "新 tau", "format": "number"},
                {
                    "field": "refined_max_mach",
                    "label": "新 max Mach",
                    "format": "number",
                },
            ],
        },
    ]

    blocks = [
        markdown_block("title", f"# {TITLE}"),
        # The summary intentionally combines the main-grid, scaling, and sensitivity
        # evidence paths; a single block-wide sourceId would misstate provenance.
        markdown_block("technical_summary", narrative["technical_summary"]),
        markdown_block(
            "bubble_finding", narrative["bubble_section"], "src_bubble_wmax"
        ),
        {
            "id": "bubble_wmax_visual",
            "type": "chart",
            "chartId": "bubble_wmax_chart",
            "layout": "full",
        },
        markdown_block(
            "density_finding", narrative["density_section"], "src_density_front"
        ),
        {
            "id": "density_front_visual",
            "type": "chart",
            "chartId": "density_front_chart",
            "layout": "full",
        },
        markdown_block("run_reliability", narrative["audit_section"], "src_run_audit"),
        {
            "id": "run_audit_visual",
            "type": "table",
            "tableId": "run_audit_table",
            "layout": "full",
        },
        markdown_block(
            "scope_and_definitions",
            "## 比较范围与指标定义\n\n"
            "验证对象为 GEA 论文中的 **Rising thermal bubble** 与 "
            "**Density current**。热泡比较 `t=1020 s` 的全域 "
            "`w_min/w_max` 和 `theta'_min/theta'_max`；密度流比较 "
            "`t=900 s, z=1200 m` 上 `theta'=-1 K` 的最右前缘，并保留阈值"
            "所在相邻网格中心作为 bracket。论文正文相邻位置写作 `+1 K`，"
            "但冷池扰动为负且其表 3 前缘定义与 `-1 K` 一致，因此本验证采用负阈值。\n\n"
            "主比较固定为 D3Q19 动量/热标量、SRT、`dt=0.1 s`、无 SGS。"
            "热泡网格中心按官方 GEA 的 `0,h,...,L-h` 注册；密度流按边界对齐注册。"
            "完成状态要求达到目标时刻、全场有限、格子密度为正，且最新 suite "
            "manifest 没有非零退出码。",
        ),
        markdown_block(
            "method_and_model",
            "## 方法与模型规格：这是受控的 Boussinesq 类比\n\n"
            "CULBM 将论文常量动力黏度按地面参考密度映射为全域常量运动黏度，"
            "并直接输运归一化位温扰动，通过 Boussinesq 项反馈浮力。二维问题在 "
            "CULBM 中以 4-cell 周期薄层表示，外边界使用本次验证增加并单测的精确"
            "镜面 free-slip 反射。\n\n"
            "GEA 基准则求解带总能量、理想气体状态方程与高度变化静力背景的可压缩"
            "干空气方程；常量动力黏度对应的运动黏度会随高度密度下降而增加。"
            "两者还分别采用有限体积/BDF1/压力速度分裂和格子 Boltzmann SRT。"
            "因此，本报告检验的是初值、边界、主要动力学、网格行为与文献指标的"
            "一致程度，而不是逐项相同 PDE 离散的回归测试。\n\n"
            "新增 h/2 网格同时采用三档时间步：`dt=0.1 s` 直接延拓原矩阵，"
            "`dt=0.05 s` 维持与上一层主网格近似相同的格子 Mach，"
            "`dt=0.025 s` 按 diffusive scaling 保持 `nu_LBM` 与 `tau`，并把 "
            "Mach 约减半。三者用于分离固定时间步趋势、等 Mach 趋势和低 Mach/"
            "同松弛参数趋势。",
        ),
        markdown_block(
            "solver_regression",
            "## 求解器边界实现与回归测试已通过\n\n"
            "最终构建的 CTest 为 **8/8 通过**。240 步四墙 free-slip GPU 回归同时覆盖 "
            "D3Q7 与 D3Q19：最大法向速度为 **5.96e-8**，切向速度保留约 "
            "**0.0199985**，温度误差不超过 **1.76e-5**。这把“边界实现本身错误”"
            "从两个基准主要偏差来源中排除；它不证明跨方程模型的物理等价。",
            "src_solver_verification",
        ),
        markdown_block(
            "fine_grid_scaling",
            narrative["scaling_section"],
            "src_fine_grid_scaling",
        ),
        {
            "id": "fine_grid_scaling_visual",
            "type": "table",
            "tableId": "fine_grid_scaling_table",
            "layout": "full",
        },
        markdown_block(
            "time_step_convergence",
            narrative["time_step_section"],
            "src_time_step_convergence",
        ),
        {
            "id": "time_step_visual",
            "type": "table",
            "tableId": "time_step_table",
            "layout": "full",
        },
        markdown_block(
            "robustness", narrative["robustness_section"], "src_robustness"
        ),
        {
            "id": "robustness_visual",
            "type": "table",
            "tableId": "robustness_table",
            "layout": "full",
        },
        markdown_block(
            "limitations",
            "## 不确定性与限制：文献表值可量化，场误差目前不可量化\n\n"
            "- GEA 论文没有随文提供这两个算例的原始场或 `z=1200 m` 数字剖面，"
            "所以只能对表中极值/前缘计算精确偏差；等值线和剖面只能作定性、"
            "网格间审计，不能声称文献场的 `L1/L2/Linf` 误差。\n"
            "- 当 CULBM 加密到论文表 1/表 3 未列出的更细网格时，报告保留真实 "
            "CULBM 点并将 GEA 值记为空；这些点只支持 CULBM 自身的网格/时间步"
            "敏感性判断，不能通过插值或外推生成同网格文献误差。\n"
            "- 两个基准没有解析解，GEA 结果本身是特定人工黏度与有限体积离散下的"
            "文献参考，而非真值。\n"
            "- 细网格主计算的格子 Mach 数偏高，会增加弱可压误差；`dt=0.05/0.025 s`"
            "控制组用于隔离这一项并检查时间步收敛，但不能校正控制方程差异。\n"
            "- 常量地面运动黏度不能复现 GEA 随高度变化的等效运动黏度；热力学、"
            "静力背景、离散耗散和二维实现差异彼此耦合，现有控制组不能唯一分解"
            "每一项的贡献。\n"
            "- 失败粗网格的异常终态被排除；它们证明稳定性边界，而不提供可比较"
            "的终态物理量。",
        ),
        markdown_block(
            "next_steps",
            "## 建议的下一步\n\n"
            "1. 对后续更细网格按 `dt/h` 固定或下降设计时间步，避免把增长的格子 "
            "Mach 数误判为空间离散趋势；严格完成并通过终态审计的 "
            "`dt=0.025 s` 结果才可作为低 Mach 基线。\n"
            "2. 保留本次 TRT 粗网格两案均失败的事实，进一步研究 MRT、cumulant、"
            "regularized 或 entropic 碰撞及标量稳定化；不要用新控制组替换本次 "
            "SRT 主矩阵。\n"
            "3. 增加随高度变化的黏性/扩散映射或真实静力背景能力，单独量化"
            "“常量动力黏度”与“常量运动黏度”的差别。\n"
            "4. 对热泡 `w_max` 平台做耗散预算：比较 SRT/TRT、显式滤波、SGS 与"
            "不同热格子，避免只看终态极值。\n"
            "5. 从 GEA 固定提交导出同网格原始场和 `z=1200 m` 剖面后，再计算"
            "相位对齐与未对齐的场范数误差。\n"
            "6. 将本次 runner、free-slip GPU 回归与关键表值阈值接入持续验证；"
            "失败粗网格应作为已知稳定性测试单列，不与物理一致性门槛混合。",
        ),
        markdown_block(
            "further_questions",
            "## 仍需回答的问题\n\n"
            "- 若在 CULBM 中复现 GEA 的高度变黏性，热泡 `w_max` 的系统高估能下降多少？\n"
            "- TRT 只改善奇对称动量模态时，两个粗网格失败中有多少来自动量碰撞，"
            "多少来自标量/界面欠解析？\n"
            "- 获取 GEA 原始场后，密度流 h=25 m 的前缘超越是整体相位误差，"
            "还是阈值附近局部色散造成？\n"
            "- 对相同 Boussinesq 方程与相同黏性模型做一个中间基准，能否把"
            "“方程形式误差”和“离散误差”拆开？",
        ),
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "CULBM 对 GEA Rising thermal bubble 与 Density current 的"
                "完整网格、稳定性和敏感性技术验证。"
            ),
            "generatedAt": generated_at,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    report_dir = Path(__file__).resolve().parent
    gea_dir = report_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=gea_dir / "analysis",
        help="Directory containing summary/main-grid/sensitivity CSV files.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=gea_dir / "reference_metrics.csv",
        help="Reviewed paper reference metrics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=report_dir / "artifact.json",
        help="Canonical artifact output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.parent != report_dir:
        raise ValueError("Report output must stay inside validation/gea/report/")

    analysis_dir = args.analysis.resolve()
    reference_path = args.reference.resolve()
    inputs = {
        "main_grid_comparison": analysis_dir / "main_grid_comparison.csv",
        "run_summary": analysis_dir / "summary.csv",
        "sensitivity_comparison": analysis_dir / "sensitivity_comparison.csv",
        "time_step_convergence": analysis_dir / "time_step_convergence.csv",
        "fine_grid_scaling_comparison": (
            analysis_dir / "fine_grid_scaling_comparison.csv"
        ),
        "reference_metrics": reference_path,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report input(s): " + ", ".join(missing))

    data_dir = report_dir / "data"
    source_dir = report_dir / "sources"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    copied_inputs: dict[str, Path] = {}
    for table, input_path in inputs.items():
        destination = data_dir / input_path.name
        shutil.copy2(input_path, destination)
        copied_inputs[table] = destination

    for filename, sql in SQL_QUERIES.items():
        (source_dir / filename).write_text(sql + "\n", encoding="utf-8")
    (source_dir / "solver_verification.md").write_text(
        "# CULBM 求解器最终回归证据\n\n"
        "- CTest：8/8 通过。\n"
        "- GPU free-slip 回归：240 步、四墙，D3Q7 与 D3Q19 均覆盖。\n"
        "- 最大法向速度：5.96e-8。\n"
        "- 切向速度保留：约 0.0199985。\n"
        "- 最大温度误差：1.76e-5。\n\n"
        "该证据验证本次精确镜面 free-slip 边界及相关求解器回归；"
        "它不等价于 GEA 与 CULBM 控制方程相同。\n",
        encoding="utf-8",
    )

    connection = sqlite3.connect(":memory:")
    try:
        for table, path in copied_inputs.items():
            load_csv_table(connection, table, path)
        connection.commit()
        bubble_rows = query_rows(connection, BUBBLE_SQL)
        density_rows = query_rows(connection, DENSITY_SQL)
        audit_rows = query_rows(connection, RUN_AUDIT_SQL)
        robustness_rows = query_rows(connection, ROBUSTNESS_SQL)
        time_step_rows = query_rows(connection, TIME_STEP_SQL)
        scaling_rows = query_rows(connection, SCALING_SQL)
        derived_rows = query_rows(connection, DERIVED_SQL)
    finally:
        connection.close()

    if not derived_rows:
        raise RuntimeError("Derived headline query returned no rows")
    if not bubble_rows:
        raise RuntimeError("No completed rising-thermal rows")
    if not density_rows:
        raise RuntimeError("No completed density-current rows")
    if not audit_rows:
        raise RuntimeError("No preferred main-run audit rows")

    datasets = {
        "bubble_wmax": bubble_rows,
        "density_front": density_rows,
        "run_audit": audit_rows,
        "robustness": robustness_rows,
        "time_step_convergence": time_step_rows,
        "fine_grid_scaling": scaling_rows,
        "derived_metrics": derived_rows,
    }
    for dataset_id, rows in datasets.items():
        write_csv(data_dir / f"{dataset_id}.csv", rows)

    generated_at = utc_now()
    source_specs = [
        source(
            "src_bubble_wmax",
            "热泡主网格与 GEA AV15 对比查询",
            path="sources/bubble_wmax.sql",
            sql=BUBBLE_SQL,
            description=(
                "从分析器的逐网格比较表连接 w_max 与 theta'_max，"
                "保留全部严格完成的热泡主计算；论文未列出的更细网格参考值为 NULL。"
            ),
            tables=["data/main_grid_comparison.csv"],
            filters=[
                "case = rising_thermal",
                "metric = w_max_m_s（并连接 theta_prime_max_K）",
                "status = complete",
                "必须存在 CULBM 值；GEA 参考值允许为空",
            ],
            definitions=[
                "w_max：t=1020 s 全域竖直速度最大值，单位 m/s。",
                "相对偏差：100*(CULBM-GEA)/abs(GEA)。",
                "theta'_max：t=1020 s 全域位温扰动最大值，单位 K。",
            ],
            executed_at=generated_at,
        ),
        source(
            "src_density_front",
            "密度流主网格与 GEA AV75 对比查询",
            path="sources/density_front.sql",
            sql=DENSITY_SQL,
            description=(
                "读取完成的密度流主计算前缘；文献区间存在时同时给出中点和上下界。"
            ),
            tables=["data/main_grid_comparison.csv"],
            filters=[
                "case = density_current",
                "metric = front_x_m",
                "status = complete",
            ],
            definitions=[
                "前缘：t=900 s、z=1200 m 上 theta'=-1 K 的最右交点。",
                "GEA 对比值：Table 3 AV75 区间中点；上下界保留在 tooltip。",
                "相对偏差：100*(CULBM-GEA中点)/abs(GEA中点)。",
            ],
            executed_at=generated_at,
        ),
        source(
            "src_run_audit",
            "严格主网格运行审计查询",
            path="sources/run_audit.sql",
            sql=RUN_AUDIT_SQL,
            description=(
                "用主比较表选定的首选 run 连接严格状态汇总，"
                "失败 run 只显示最后有效诊断而不暴露异常终态。"
            ),
            tables=[
                "data/main_grid_comparison.csv",
                "data/summary.csv",
            ],
            filters=["仅保留 main_grid_comparison 选定的首选 run_name"],
            definitions=[
                "complete：达到目标时刻、终态有限、密度为正且最新 manifest 退出码为 0。",
                "diagnostic max Mach：完成 run 用终态值，失败 run 用最后有效采样值。",
            ],
            executed_at=generated_at,
        ),
        source(
            "src_robustness",
            "敏感性与稳定性控制组查询",
            path="sources/robustness.sql",
            sql=ROBUSTNESS_SQL,
            description=(
                "汇总时间步、热格子、黏度解释与 SRT/TRT 控制组，"
                "并保留相对同网格主结果的变化。"
            ),
            tables=["data/sensitivity_comparison.csv"],
            filters=["排除 registration_sensitivity 旧半网格平移归档"],
            definitions=[
                "主比较量：热泡为 w_max，密度流为 front_x。",
                "相对主结果变化：100*(控制组-同网格主结果)/abs(同网格主结果)。",
            ],
            executed_at=generated_at,
        ),
        source(
            "src_time_step_convergence",
            "细网格时间步与 Mach 收敛查询",
            path="sources/time_step_convergence.sql",
            sql=TIME_STEP_SQL,
            description=(
                "对同一 D3Q19 动态黏度网格按 dt 逐级比较主指标和最大格子 Mach，"
                "同时保留相对 dt=0.1 及相邻更粗时间步的变化。"
            ),
            tables=["data/time_step_convergence.csv"],
            filters=[
                "仅含同 case/h 至少两个时间步的 D3Q19 动态黏度结果",
                "严格完成 run 才提供终态主指标和 max Mach",
            ],
            definitions=[
                "热泡主指标为 w_max，单位 m/s；密度流主指标为 front_x，单位 m。",
                "逐级变化：100*(当前 dt 值-相邻更粗 dt 值)/abs(相邻更粗 dt 值)。",
            ],
            executed_at=generated_at,
        ),
        source(
            "src_fine_grid_scaling",
            "h/2 加密三路径比较查询",
            path="sources/fine_grid_scaling.sql",
            sql=SCALING_SQL,
            description=(
                "把新增 h/2 网格的固定 dt、dt∝h 和 dt∝h² 三条路径"
                "分别与上一层最细 dt=0.1 主网格比较。"
            ),
            tables=["data/fine_grid_scaling_comparison.csv"],
            filters=[
                "热泡 15.625→7.8125 m；密度流 25→12.5 m",
                "D3Q19、动态黏度映射；严格完成 run 才给出终态值",
            ],
            definitions=[
                "dt∝h：近似保持格子速度/Mach；dt∝h²：保持 nu_LBM 与 tau。",
                "相对变化：100*(新网格-上一层最细主网格)/abs(上一层值)。",
            ],
            executed_at=generated_at,
        ),
        source(
            "src_derived_metrics",
            "报告头部派生指标查询",
            path="sources/derived_metrics.sql",
            sql=DERIVED_SQL,
            description=(
                "计算主矩阵状态、最细文献可比点、CULBM 全部最细点及密度流可比偏差。"
            ),
            tables=[
                "data/main_grid_comparison.csv",
                "data/sensitivity_comparison.csv",
            ],
            filters=[
                "主矩阵按 case/h/run_name 去重",
                "最细热泡仅用严格完成且有文献值的网格",
                "另保留没有同网格文献值的最细 CULBM 热泡/密度流点",
                "密度流定量偏差只用 h=100/50/25 m 的文献对齐网格",
            ],
            definitions=[
                "主矩阵状态计数以分析器选定的首选 run 为单位。",
                "所有百分比偏差使用文献值或同网格主结果的绝对值作分母。",
            ],
            executed_at=generated_at,
        ),
        source(
            "src_reference_metrics",
            "GEA 论文参考值审计表",
            path="data/reference_metrics.csv",
        ),
        source(
            "src_solver_verification",
            "CULBM 最终 CTest 与 GPU free-slip 回归",
            path="sources/solver_verification.md",
        ),
        source(
            "src_paper",
            "GEA 论文：arXiv:2302.04836",
            href=PAPER_URL,
        ),
        source(
            "src_gea_commit",
            "GEA 官方实现固定提交 bdf0b929",
            href=GEA_COMMIT_URL,
        ),
    ]

    narrative = build_narrative(
        derived_rows[0],
        bubble_rows,
        density_rows,
        audit_rows,
        robustness_rows,
        time_step_rows,
        scaling_rows,
    )
    artifact = build_artifact(generated_at, datasets, source_specs, narrative)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    chart_map = (
        "# 报告图表映射\n\n"
        "| 报告段落 | 分析问题 | 图形 | 字段 | 结论 | 来源 |\n"
        "|---|---|---|---|---|---|\n"
        "| 热泡速度 | w_max 是否趋近 AV15 | 分组柱形 | grid, culbm_w_max, "
        "gea_av15_w_max | 完成网格维持较高速度平台 | "
        "sources/bubble_wmax.sql |\n"
        "| 密度流前缘 | 前缘是否随加密趋近 AV75 | 分组柱形 | grid, "
        "culbm_front_km, gea_av75_midpoint_km | 偏差变号、非单调 | "
        "sources/density_front.sql |\n\n"
        "两图重复使用分组柱形是有意的：两段都比较少量离散网格上的同单位实际值与"
        "文献基线，不是时间趋势。使用蓝/中性基线的两根语义由共享 reader 处理，"
        "并由表格/tooltip 保留精确值。\n"
    )
    (report_dir / "CHART_MAP.md").write_text(chart_map, encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact": str(output),
                "generated_at": generated_at,
                "datasets": {key: len(value) for key, value in datasets.items()},
                "main_total": as_int(derived_rows[0].get("main_total")),
                "main_complete": as_int(derived_rows[0].get("main_complete")),
                "main_failed": as_int(derived_rows[0].get("main_failed")),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
