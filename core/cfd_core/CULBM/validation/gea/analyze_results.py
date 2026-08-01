#!/usr/bin/env python3
"""汇总 CULBM GEA 验证结果并生成可复现图表。

完成状态是严格定义的：必须存在 final_summary.json，且达到配置终止时刻、
所有场有限、密度为正。失败或尚未完成的 run 不会把末帧数值写入最终比较列。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
GREEN = "#009E73"
GREY = "#6B7280"
LIGHT_GREY = "#D1D5DB"

FINAL_METRICS = [
    "u_min_m_s",
    "u_max_m_s",
    "w_min_m_s",
    "w_max_m_s",
    "theta_prime_min_K",
    "theta_prime_max_K",
    "rho_min_lbm",
    "rho_max_lbm",
    "rho_mean_lbm",
    "theta_prime_mean_K",
    "max_speed_m_s",
    "max_mach",
    "max_y_theta_spread_K",
    "front_x_m",
    "front_bracket_left_m",
    "front_bracket_right_m",
]

METRIC_LABELS = {
    "w_min_m_s": r"$w_{\min}$ (m s$^{-1}$)",
    "w_max_m_s": r"$w_{\max}$ (m s$^{-1}$)",
    "theta_prime_min_K": r"$\theta'_{\min}$ (K)",
    "theta_prime_max_K": r"$\theta'_{\max}$ (K)",
    "front_x_m": "Front position (km)",
}


def infer_grid_registration(config: dict[str, Any]) -> str:
    """读取显式注册信息；旧结果按已审计的 runner 行为保守标注。"""
    explicit = config.get("grid_registration")
    if explicit:
        return str(explicit)
    if config.get("case") == "rising_thermal":
        return "legacy_half_cell_shift (centres h/2..L-h/2; not exact GEA setup)"
    if config.get("case") == "density_current":
        return "official_boundary_aligned"
    return "unknown"


def registration_priority(value: Any) -> int:
    token = str(value).lower()
    if any(word in token for word in ("legacy", "shift", "half_cell", "half-cell")):
        return 2
    if any(
        word in token
        for word in ("official", "corrected", "gea_aligned", "gea aligned", "gea ")
    ):
        return 0
    return 1


def primary_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    """返回主验证候选，并允许后续 corrected 分组覆盖旧 main 结果。"""
    if summary.empty:
        return summary
    groups = summary["run_group"].astype(str).str.lower()
    return summary[
        (groups == "main")
        | groups.str.contains("corrected", regex=False)
        | groups.str.contains("primary", regex=False)
    ].copy()


def choose_preferred_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """每个 case/h 选择注册最接近官方、且正常完成的候选。"""
    if frame.empty:
        return frame
    selected = frame.copy()
    selected["_status_rank"] = selected["status"].map(
        {"complete": 0, "failed": 1, "incomplete": 2}
    ).fillna(3)
    selected["_registration_rank"] = selected["grid_registration"].map(
        registration_priority
    )
    selected = selected.sort_values(
        ["case", "h_m", "_registration_rank", "_status_rank", "run_name"]
    )
    return selected.drop_duplicates(["case", "h_m"], keep="first").drop(
        columns=["_status_rank", "_registration_rank"]
    )


def sensitivity_kind(run_group: str, config: dict[str, Any], run_name: str) -> str:
    if run_group == "main":
        return "primary"
    if run_group == "registration_sensitivity":
        return "legacy_grid_registration"
    if run_group == "stability":
        return (
            f"{config.get('momentum_collision', 'unknown')}"
            f"_dt{float(config.get('dt_s', math.nan)):g}"
        )
    if run_group == "viscosity_profile":
        if "z3km" in run_name:
            return "rho_ref_z3km"
        if "z8km" in run_name:
            return "rho_ref_z8km"
        return f"rho_ref_{config.get('rho_ref_kg_m3', 'unknown')}"
    if str(config.get("thermal_lattice", "")).upper() == "D3Q7":
        return "thermal_D3Q7"
    if "numeric" in str(config.get("viscosity_mode", "")).lower():
        return "numeric_viscosity"
    dt_s = float(config.get("dt_s", math.nan))
    if dt_s < 0.1:
        return "dt_half" if math.isclose(dt_s, 0.05) else f"dt_{dt_s:g}"
    return run_group


@dataclass
class RunRecord:
    path: Path
    group: str
    name: str
    config: dict[str, Any]
    summary: dict[str, Any] | None
    metrics: pd.DataFrame
    status: str
    detail: str
    observed_time: float
    last_valid_time: float
    last_valid_max_mach: float
    manifest_exit_code: float


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        # 允许在求解套件仍写入最后一行时执行分析。
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def discover_runs(results_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    manifest_by_path: dict[str, tuple[pd.Series, float]] = {}
    manifest_by_name: dict[tuple[str, str], tuple[pd.Series, float]] = {}
    for manifest_path in sorted(results_root.glob("suite_manifest*.csv")):
        manifest = load_csv(manifest_path)
        try:
            manifest_mtime = manifest_path.stat().st_mtime
        except OSError:
            manifest_mtime = math.nan
        if manifest.empty:
            continue
        for _, item in manifest.iterrows():
            directory = item.get("output_directory")
            if isinstance(directory, str) and directory:
                try:
                    key = str(Path(directory).resolve()).lower()
                    entry = (item, manifest_mtime)
                    old = manifest_by_path.get(key)
                    if old is None or manifest_mtime >= old[1]:
                        manifest_by_path[key] = entry
                    name_key = (
                        str(item.get("group", "")).lower(),
                        Path(directory).name,
                    )
                    old = manifest_by_name.get(name_key)
                    if old is None or manifest_mtime >= old[1]:
                        manifest_by_name[name_key] = entry
                except OSError:
                    pass
    candidates = sorted({path.parent for path in results_root.rglob("run_config.json")})
    for run_path in candidates:
        config = load_json(run_path / "run_config.json")
        if not config:
            continue
        config = dict(config)
        try:
            relative_parts = run_path.relative_to(results_root).parts
        except ValueError:
            relative_parts = run_path.parts
        lowered_parts = [part.lower() for part in relative_parts]
        if not config.get("grid_registration"):
            if "wall_centered" in lowered_parts or "wall-centred" in lowered_parts:
                config["grid_registration"] = (
                    "legacy_wall_centered / half-cell-shifted centres "
                    "(not exact GEA setup)"
                )
            elif any(
                token in lowered_parts
                for token in ("cell_centered", "cell_centred", "official_registration")
            ):
                config["grid_registration"] = "official_cell_centered"
        summary = load_json(run_path / "final_summary.json")
        metrics = load_csv(run_path / "metrics.csv")
        summary_fresh = False
        if summary:
            try:
                summary_fresh = (
                    (run_path / "final_summary.json").stat().st_mtime
                    + 1.0e-2
                    >= (run_path / "run_config.json").stat().st_mtime
                )
            except OSError:
                summary_fresh = False
        group = (
            "registration_sensitivity"
            if "registration_sensitivity" in lowered_parts
            else "unknown"
        )
        if group == "unknown":
            for part in reversed(relative_parts[:-1]):
                if part.lower() in {"main", "sensitivity", "corrected", "primary"}:
                    group = part.lower()
                    break
        if group == "unknown" and len(relative_parts) > 1:
            group = relative_parts[0]
        manifest_entry = manifest_by_path.get(str(run_path.resolve()).lower())
        if manifest_entry is None:
            manifest_entry = manifest_by_name.get((group.lower(), run_path.name))
        # 新一轮 run 可能在旧 suite_manifest 仍存在时启动；旧清单不得覆盖新配置。
        if manifest_entry is not None and finite_number(manifest_entry[1]):
            try:
                if (
                    manifest_entry[1] + 1.0e-2
                    < (run_path / "run_config.json").stat().st_mtime
                ):
                    manifest_entry = None
            except OSError:
                manifest_entry = None
        manifest_row = manifest_entry[0] if manifest_entry is not None else None
        manifest_exit_code = math.nan
        if manifest_row is not None and finite_number(manifest_row.get("exit_code")):
            manifest_exit_code = float(manifest_row["exit_code"])

        observed_time = 0.0
        if not metrics.empty and "time_s" in metrics:
            observed_time = float(pd.to_numeric(metrics["time_s"], errors="coerce").max())
        if summary and finite_number(summary.get("time_s")):
            observed_time = max(observed_time, float(summary["time_s"]))

        valid = metrics
        if not metrics.empty:
            valid = metrics.copy()
            if "finite" in valid:
                valid = valid[pd.to_numeric(valid["finite"], errors="coerce") == 1]
            if "positive_density" in valid:
                valid = valid[
                    pd.to_numeric(valid["positive_density"], errors="coerce") == 1
                ]
        last_valid_time = (
            float(pd.to_numeric(valid["time_s"], errors="coerce").max())
            if not valid.empty and "time_s" in valid
            else math.nan
        )
        last_valid_max_mach = math.nan
        if not valid.empty and "max_mach" in valid:
            last_valid_max_mach = float(
                pd.to_numeric(valid["max_mach"], errors="coerce").iloc[-1]
            )

        end_time = float(config.get("end_time_s", math.nan))
        dt = float(config.get("dt_s", 0.0))
        stderr = ""
        try:
            stderr = (run_path / "console.stderr.log").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            pass
        aborted = "GEA_ABORT" in stderr or "non-finite" in stderr.lower()
        summary_good = bool(
            summary
            and summary.get("finite") is True
            and summary.get("positive_density") is True
        )
        reached_end = bool(
            summary
            and finite_number(summary.get("time_s"))
            and finite_number(end_time)
            and math.isclose(
                float(summary["time_s"]),
                end_time,
                rel_tol=1.0e-10,
                abs_tol=max(abs(dt) * 1.0e-6, 1.0e-8),
            )
        )
        metrics_reached_end = False
        metrics_terminal_valid = False
        if not metrics.empty and "time_s" in metrics:
            metric_times = pd.to_numeric(metrics["time_s"], errors="coerce")
            if metric_times.notna().any():
                terminal_index = metric_times.idxmax()
                terminal_time = float(metric_times.loc[terminal_index])
                metrics_reached_end = math.isclose(
                    terminal_time,
                    end_time,
                    rel_tol=1.0e-10,
                    abs_tol=max(abs(dt) * 1.0e-6, 1.0e-8),
                )
                terminal = metrics.loc[terminal_index]
                metrics_terminal_valid = bool(
                    ("finite" not in terminal or float(terminal["finite"]) == 1.0)
                    and (
                        "positive_density" not in terminal
                        or float(terminal["positive_density"]) == 1.0
                    )
                )
        explicitly_incomplete = bool(
            (summary and summary.get("completed") is False)
            or config.get("completed") is False
        )
        manifest_failed = finite_number(manifest_exit_code) and manifest_exit_code != 0
        required_values_good = bool(
            summary_good
            and all(
                finite_number(summary.get(key))
                for key in FINAL_METRICS
                if key not in {
                    "front_x_m",
                    "front_bracket_left_m",
                    "front_bracket_right_m",
                }
            )
        )

        if (
            summary_good
            and reached_end
            and metrics_reached_end
            and metrics_terminal_valid
            and summary_fresh
            and required_values_good
            and not aborted
            and not manifest_failed
            and not explicitly_incomplete
        ):
            status = "complete"
            detail = "达到目标终止时刻，场有限且密度为正"
        elif manifest_failed or aborted or (summary and not summary_good):
            status = "failed"
            if manifest_failed:
                detail = f"suite_manifest 记录非零退出码 {int(manifest_exit_code)}"
            elif aborted:
                detail = "数值失稳/非正密度终止"
            else:
                detail = "终态有效性检查失败"
        else:
            status = "incomplete"
            if explicitly_incomplete:
                detail = "输出显式标记 completed=false"
            elif summary and not summary_fresh:
                detail = "final_summary 早于 run_config，拒绝陈旧终态"
            elif summary and (not reached_end or not metrics_reached_end):
                detail = "终态/时序时间与配置终止时刻不符"
            else:
                detail = "仍在计算或缺少正常终态文件"

        records.append(
            RunRecord(
                path=run_path,
                group=group,
                name=run_path.name,
                config=config,
                summary=summary,
                metrics=metrics,
                status=status,
                detail=detail,
                observed_time=observed_time,
                last_valid_time=last_valid_time,
                last_valid_max_mach=last_valid_max_mach,
                manifest_exit_code=manifest_exit_code,
            )
        )
    return records


def build_summary(records: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in records:
        cfg = run.config
        end_time = float(cfg.get("end_time_s", math.nan))
        row: dict[str, Any] = {
            "run_group": run.group,
            "run_name": run.name,
            "case": cfg.get("case"),
            "status": run.status,
            "status_detail": run.detail,
            "h_m": cfg.get("h_m"),
            "dt_s": cfg.get("dt_s"),
            "end_time_s": cfg.get("end_time_s"),
            "observed_time_s": run.observed_time,
            "completion_fraction": (
                min(run.observed_time / end_time, 1.0)
                if finite_number(end_time) and end_time > 0
                else math.nan
            ),
            "thermal_lattice": cfg.get("thermal_lattice"),
            "momentum_lattice": cfg.get("momentum_lattice"),
            "momentum_collision": cfg.get("momentum_collision", "SRT"),
            "viscosity_mode": cfg.get("viscosity_mode"),
            "grid_registration": infer_grid_registration(cfg),
            "sensitivity_kind": sensitivity_kind(
                run.group, cfg, run.name
            ),
            "rho_ref_kg_m3": cfg.get("rho_ref_kg_m3"),
            "mu_artificial_Pa_s": cfg.get("mu_artificial_Pa_s"),
            "nu_m2_s": cfg.get("nu_m2_s"),
            "nu_lbm": cfg.get("nu_lbm"),
            "tau": cfg.get("tau"),
            "alpha_lbm": cfg.get("alpha_lbm"),
            "gravity_z_lbm": cfg.get("gravity_z_lbm"),
            "last_valid_time_s": run.last_valid_time,
            "last_valid_max_mach": run.last_valid_max_mach,
            "manifest_exit_code": run.manifest_exit_code,
            "output_directory": run.path.as_posix(),
        }
        # 只有严格完成的结果才进入可比较的终态列。
        for metric in FINAL_METRICS:
            row[metric] = (
                run.summary.get(metric, math.nan)
                if run.status == "complete" and run.summary
                else math.nan
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["run_group", "case", "h_m", "dt_s", "thermal_lattice"],
            ascending=[True, True, False, True, True],
            na_position="last",
        )
    return frame


def build_main_comparison(
    summary: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "case",
        "h_m",
        "metric",
        "units",
        "status",
        "culbm_value",
        "reference_value",
        "reference_lower",
        "reference_upper",
        "difference",
        "relative_difference_percent",
        "within_reference_bracket",
        "run_name",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    main = choose_preferred_runs(primary_candidates(summary))
    rows: list[dict[str, Any]] = []
    units_by_metric = {
        "w_min_m_s": "m/s",
        "w_max_m_s": "m/s",
        "theta_prime_min_K": "K",
        "theta_prime_max_K": "K",
        "front_x_m": "m",
    }
    keys: set[tuple[str, float, str]] = set()
    for _, run in main.iterrows():
        metrics = (
            [
                "w_min_m_s",
                "w_max_m_s",
                "theta_prime_min_K",
                "theta_prime_max_K",
            ]
            if run["case"] == "rising_thermal"
            else ["front_x_m"]
        )
        keys.update((str(run["case"]), float(run["h_m"]), metric) for metric in metrics)
    keys.update(
        (str(ref["case"]), float(ref["h_m"]), str(ref["metric"]))
        for _, ref in reference.iterrows()
    )
    for case, h_m, metric in sorted(keys, key=lambda item: (item[0], -item[1], item[2])):
        match = main[
            (main["case"] == case)
            & np.isclose(
                pd.to_numeric(main["h_m"], errors="coerce"),
                h_m,
                equal_nan=False,
            )
        ]
        run = match.iloc[0] if not match.empty else None
        reference_match = reference[
            (reference["case"] == case)
            & np.isclose(
                pd.to_numeric(reference["h_m"], errors="coerce"),
                h_m,
                equal_nan=False,
            )
            & (reference["metric"] == metric)
        ]
        ref = reference_match.iloc[0] if not reference_match.empty else None
        value = (
            run.get(metric, math.nan)
            if run is not None and run["status"] == "complete"
            else math.nan
        )
        difference = (
            float(value) - float(ref["reference_value"])
            if finite_number(value) and ref is not None
            else math.nan
        )
        ref_value = float(ref["reference_value"]) if ref is not None else math.nan
        relative = (
            100.0 * difference / abs(ref_value)
            if finite_number(difference) and ref_value != 0
            else math.nan
        )
        within = (
            bool(
                float(ref["reference_lower"])
                <= float(value)
                <= float(ref["reference_upper"])
            )
            if finite_number(value) and ref is not None
            else pd.NA
        )
        rows.append(
            {
                "case": case,
                "h_m": h_m,
                "metric": metric,
                "units": ref["units"] if ref is not None else units_by_metric[metric],
                "status": run["status"] if run is not None else "missing",
                "culbm_value": value,
                "reference_value": ref_value,
                "reference_lower": (
                    ref["reference_lower"] if ref is not None else math.nan
                ),
                "reference_upper": (
                    ref["reference_upper"] if ref is not None else math.nan
                ),
                "difference": difference,
                "relative_difference_percent": relative,
                "within_reference_bracket": within,
                "run_name": run["run_name"] if run is not None else "",
                "grid_registration": (
                    run["grid_registration"] if run is not None else ""
                ),
            }
        )
    columns.append("grid_registration")
    return pd.DataFrame(rows, columns=columns)


def build_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    groups = summary["run_group"].astype(str).str.lower()
    sensitivity = summary[
        ~(
            (groups == "main")
            | groups.str.contains("corrected", regex=False)
            | groups.str.contains("primary", regex=False)
        )
    ].copy()
    if sensitivity.empty:
        return sensitivity
    main = choose_preferred_runs(primary_candidates(summary))
    value_metric: list[str] = []
    baseline_value: list[float] = []
    relative: list[float] = []
    for _, row in sensitivity.iterrows():
        metric = "w_max_m_s" if row["case"] == "rising_thermal" else "front_x_m"
        baseline = main[
            (main["case"] == row["case"])
            & np.isclose(
                pd.to_numeric(main["h_m"], errors="coerce"),
                float(row["h_m"]),
                equal_nan=False,
            )
        ]
        base_value = (
            float(baseline.iloc[0][metric])
            if not baseline.empty
            and baseline.iloc[0]["status"] == "complete"
            and finite_number(baseline.iloc[0][metric])
            else math.nan
        )
        actual = float(row[metric]) if finite_number(row[metric]) else math.nan
        value_metric.append(metric)
        baseline_value.append(base_value)
        relative.append(
            100.0 * (actual - base_value) / abs(base_value)
            if finite_number(actual) and finite_number(base_value) and base_value != 0
            else math.nan
        )
    sensitivity["comparison_metric"] = value_metric
    sensitivity["matching_main_value"] = baseline_value
    sensitivity["relative_to_matching_main_percent"] = relative
    return sensitivity


def build_time_step_convergence(summary: pd.DataFrame) -> pd.DataFrame:
    """对同一细网格的 dt 序列做逐级比较，不把 dt 误差混入 h 误差。"""
    columns = [
        "case",
        "h_m",
        "dt_s",
        "status",
        "metric",
        "value",
        "max_mach",
        "baseline_dt_s",
        "baseline_value",
        "relative_to_dt0p1_percent",
        "next_coarser_dt_s",
        "next_coarser_dt_value",
        "relative_to_next_coarser_dt_percent",
        "run_name",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    groups = summary["run_group"].astype(str).str.lower()
    candidates = summary[
        groups.isin(["main", "sensitivity"])
        & (summary["thermal_lattice"].astype(str).str.upper() == "D3Q19")
        & summary["viscosity_mode"].astype(str).str.contains(
            "rho_ref", case=False, regex=False
        )
        & (pd.to_numeric(summary["dt_s"], errors="coerce") <= 0.1)
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    group_sizes = candidates.groupby(["case", "h_m"])["dt_s"].nunique()
    eligible = {
        (str(case), float(h_m))
        for (case, h_m), count in group_sizes.items()
        if int(count) >= 2
    }
    rows: list[dict[str, Any]] = []
    for case, h_m in sorted(eligible):
        case_rows = candidates[
            (candidates["case"] == case)
            & np.isclose(
                pd.to_numeric(candidates["h_m"], errors="coerce"),
                h_m,
                equal_nan=False,
            )
        ].sort_values("dt_s", ascending=False)
        metric = "w_max_m_s" if case == "rising_thermal" else "front_x_m"
        baseline_match = case_rows[
            np.isclose(
                pd.to_numeric(case_rows["dt_s"], errors="coerce"),
                0.1,
                equal_nan=False,
            )
        ]
        baseline_value = (
            float(baseline_match.iloc[0][metric])
            if not baseline_match.empty
            and baseline_match.iloc[0]["status"] == "complete"
            and finite_number(baseline_match.iloc[0][metric])
            else math.nan
        )
        previous_dt = math.nan
        previous_value = math.nan
        for _, run in case_rows.iterrows():
            value = (
                float(run[metric])
                if run["status"] == "complete" and finite_number(run[metric])
                else math.nan
            )
            rows.append(
                {
                    "case": case,
                    "h_m": h_m,
                    "dt_s": float(run["dt_s"]),
                    "status": run["status"],
                    "metric": metric,
                    "value": value,
                    "max_mach": (
                        float(run["max_mach"])
                        if run["status"] == "complete"
                        and finite_number(run["max_mach"])
                        else math.nan
                    ),
                    "baseline_dt_s": 0.1,
                    "baseline_value": baseline_value,
                    "relative_to_dt0p1_percent": (
                        100.0 * (value - baseline_value) / abs(baseline_value)
                        if finite_number(value)
                        and finite_number(baseline_value)
                        and baseline_value != 0
                        else math.nan
                    ),
                    "next_coarser_dt_s": previous_dt,
                    "next_coarser_dt_value": previous_value,
                    "relative_to_next_coarser_dt_percent": (
                        100.0 * (value - previous_value) / abs(previous_value)
                        if finite_number(value)
                        and finite_number(previous_value)
                        and previous_value != 0
                        else math.nan
                    ),
                    "run_name": run["run_name"],
                }
            )
            previous_dt = float(run["dt_s"])
            previous_value = value
    return pd.DataFrame(rows, columns=columns)


def build_fine_grid_scaling_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """比较 h/2 时固定 dt、acoustic scaling 与 diffusive scaling 三条路径。"""
    columns = [
        "case",
        "scaling",
        "baseline_h_m",
        "baseline_dt_s",
        "refined_h_m",
        "refined_dt_s",
        "metric",
        "status",
        "baseline_value",
        "refined_value",
        "relative_change_percent",
        "baseline_tau",
        "refined_tau",
        "baseline_max_mach",
        "refined_max_mach",
        "run_name",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    targets = {
        "rising_thermal": (15.625, 7.8125, "w_max_m_s"),
        "density_current": (25.0, 12.5, "front_x_m"),
    }
    primary = choose_preferred_runs(primary_candidates(summary))
    sensitivity = summary[
        (summary["run_group"].astype(str).str.lower() == "sensitivity")
        & (summary["thermal_lattice"].astype(str).str.upper() == "D3Q19")
        & summary["viscosity_mode"].astype(str).str.contains(
            "rho_ref", case=False, regex=False
        )
    ]
    paths = [
        ("fixed_dt", 0.1),
        ("acoustic_dt_proportional_h", 0.05),
        ("diffusive_dt_proportional_h2", 0.025),
    ]
    rows: list[dict[str, Any]] = []
    for case, (baseline_h, refined_h, metric) in targets.items():
        baseline_match = primary[
            (primary["case"] == case)
            & np.isclose(
                pd.to_numeric(primary["h_m"], errors="coerce"),
                baseline_h,
                equal_nan=False,
            )
            & np.isclose(
                pd.to_numeric(primary["dt_s"], errors="coerce"),
                0.1,
                equal_nan=False,
            )
        ]
        baseline = baseline_match.iloc[0] if not baseline_match.empty else None
        baseline_value = (
            float(baseline[metric])
            if baseline is not None
            and baseline["status"] == "complete"
            and finite_number(baseline[metric])
            else math.nan
        )
        for scaling, refined_dt in paths:
            if math.isclose(refined_dt, 0.1):
                match = primary[
                    (primary["case"] == case)
                    & np.isclose(
                        pd.to_numeric(primary["h_m"], errors="coerce"),
                        refined_h,
                        equal_nan=False,
                    )
                    & np.isclose(
                        pd.to_numeric(primary["dt_s"], errors="coerce"),
                        refined_dt,
                        equal_nan=False,
                    )
                ]
            else:
                match = sensitivity[
                    (sensitivity["case"] == case)
                    & np.isclose(
                        pd.to_numeric(sensitivity["h_m"], errors="coerce"),
                        refined_h,
                        equal_nan=False,
                    )
                    & np.isclose(
                        pd.to_numeric(sensitivity["dt_s"], errors="coerce"),
                        refined_dt,
                        equal_nan=False,
                    )
                ]
            refined = match.iloc[0] if not match.empty else None
            refined_value = (
                float(refined[metric])
                if refined is not None
                and refined["status"] == "complete"
                and finite_number(refined[metric])
                else math.nan
            )
            rows.append(
                {
                    "case": case,
                    "scaling": scaling,
                    "baseline_h_m": baseline_h,
                    "baseline_dt_s": 0.1,
                    "refined_h_m": refined_h,
                    "refined_dt_s": refined_dt,
                    "metric": metric,
                    "status": refined["status"] if refined is not None else "missing",
                    "baseline_value": baseline_value,
                    "refined_value": refined_value,
                    "relative_change_percent": (
                        100.0
                        * (refined_value - baseline_value)
                        / abs(baseline_value)
                        if finite_number(refined_value)
                        and finite_number(baseline_value)
                        and baseline_value != 0
                        else math.nan
                    ),
                    "baseline_tau": (
                        float(baseline["tau"])
                        if baseline is not None and finite_number(baseline["tau"])
                        else math.nan
                    ),
                    "refined_tau": (
                        float(refined["tau"])
                        if refined is not None and finite_number(refined["tau"])
                        else math.nan
                    ),
                    "baseline_max_mach": (
                        float(baseline["max_mach"])
                        if baseline is not None
                        and baseline["status"] == "complete"
                        and finite_number(baseline["max_mach"])
                        else math.nan
                    ),
                    "refined_max_mach": (
                        float(refined["max_mach"])
                        if refined is not None
                        and refined["status"] == "complete"
                        and finite_number(refined["max_mach"])
                        else math.nan
                    ),
                    "run_name": refined["run_name"] if refined is not None else "",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_stability_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    coarse = summary[
        (
            (summary["run_group"] == "main")
            & (
                (
                    (summary["case"] == "rising_thermal")
                    & np.isclose(summary["h_m"].astype(float), 250.0)
                )
                | (
                    (summary["case"] == "density_current")
                    & np.isclose(summary["h_m"].astype(float), 400.0)
                )
            )
        )
        | (summary["run_group"] == "stability")
    ].copy()
    return coarse.sort_values(["case", "dt_s", "momentum_collision"])


def build_viscosity_profile_comparison(
    summary: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[
        (summary["run_group"] == "main")
        & (summary["case"] == "rising_thermal")
        & np.isclose(summary["h_m"].astype(float), 31.25)
    ]
    alternatives = summary[
        (summary["run_group"] == "viscosity_profile")
        & (summary["case"] == "rising_thermal")
        & np.isclose(summary["h_m"].astype(float), 31.25)
    ]
    candidates = pd.concat([baseline, alternatives], ignore_index=True)
    metrics = [
        "w_min_m_s",
        "w_max_m_s",
        "theta_prime_min_K",
        "theta_prime_max_K",
    ]
    rows: list[dict[str, Any]] = []
    base = baseline.iloc[0] if not baseline.empty else None
    for _, run in candidates.iterrows():
        for metric in metrics:
            value = run[metric] if run["status"] == "complete" else math.nan
            base_value = (
                base[metric]
                if base is not None
                and base["status"] == "complete"
                and finite_number(base[metric])
                else math.nan
            )
            ref_match = reference[
                (reference["case"] == "rising_thermal")
                & np.isclose(reference["h_m"].astype(float), 31.25)
                & (reference["metric"] == metric)
            ]
            ref_value = (
                float(ref_match.iloc[0]["reference_value"])
                if not ref_match.empty
                else math.nan
            )
            rows.append(
                {
                    "run_group": run["run_group"],
                    "sensitivity_kind": run["sensitivity_kind"],
                    "rho_ref_kg_m3": run["rho_ref_kg_m3"],
                    "nu_m2_s": run["nu_m2_s"],
                    "metric": metric,
                    "status": run["status"],
                    "culbm_value": value,
                    "primary_value": base_value,
                    "relative_to_primary_percent": (
                        100.0 * (float(value) - float(base_value)) / abs(float(base_value))
                        if finite_number(value)
                        and finite_number(base_value)
                        and float(base_value) != 0
                        else math.nan
                    ),
                    "gea_reference_value": ref_value,
                    "relative_to_gea_percent": (
                        100.0 * (float(value) - ref_value) / abs(ref_value)
                        if finite_number(value) and finite_number(ref_value) and ref_value != 0
                        else math.nan
                    ),
                    "run_name": run["run_name"],
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=LIGHT_GREY, linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot_bubble_convergence(
    summary: pd.DataFrame, reference: pd.DataFrame, output: Path
) -> None:
    metrics = [
        "w_min_m_s",
        "w_max_m_s",
        "theta_prime_min_K",
        "theta_prime_max_K",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    all_main = primary_candidates(summary)
    main = choose_preferred_runs(all_main)
    main = main[main["case"] == "rising_thermal"]
    for ax, metric in zip(axes.flat, metrics):
        ref = reference[
            (reference["case"] == "rising_thermal")
            & (reference["metric"] == metric)
        ].sort_values("h_m", ascending=False)
        completed = main[main["status"] == "complete"].sort_values(
            "h_m", ascending=False
        )
        failed = main[main["status"] == "failed"]
        incomplete = main[main["status"] == "incomplete"]
        if not ref.empty:
            ax.plot(
                ref["h_m"],
                ref["reference_value"],
                color=GREY,
                marker="s",
                linewidth=1.8,
                label="GEA AV15",
            )
        if not completed.empty:
            ax.plot(
                completed["h_m"],
                completed[metric],
                color=BLUE,
                marker="o",
                linewidth=2.0,
                label="CULBM complete*",
            )
        for data, color, marker, label in [
            (failed, RED, "x", "failed"),
            (incomplete, ORANGE, "^", "incomplete"),
        ]:
            if not data.empty:
                ax.scatter(
                    data["h_m"],
                    np.full(len(data), 0.04),
                    transform=ax.get_xaxis_transform(),
                    color=color,
                    marker=marker,
                    s=55,
                    clip_on=False,
                    label=label,
                )
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()
        bubble_ticks = sorted(
            {
                float(value)
                for value in pd.concat([main["h_m"], ref["h_m"]])
                if finite_number(value) and float(value) > 0
            },
            reverse=True,
        )
        ax.set_xticks(bubble_ticks)
        ax.set_xticklabels([f"{value:g}" for value in bubble_ticks])
        ax.set_xlabel("Grid spacing h (m) — finer →")
        ax.set_ylabel(METRIC_LABELS[metric])
        style_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        "Rising thermal: terminal extrema and grid refinement "
        "(main sequence, fixed dt = 0.1 s)",
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "* Official grid registration is preferred. Legacy bubble runs are "
        "half-cell shifted and are not an exact GEA setup.",
        ha="center",
        color=GREY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.89))
    save_figure(fig, output / "bubble_grid_convergence")


def plot_density_front(
    summary: pd.DataFrame, reference: pd.DataFrame, output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    main = choose_preferred_runs(primary_candidates(summary))
    main = main[main["case"] == "density_current"]
    completed = main[main["status"] == "complete"].sort_values(
        "h_m", ascending=False
    )
    if not completed.empty:
        x = completed["h_m"].to_numpy(float)
        y = completed["front_x_m"].to_numpy(float)
        lo = completed["front_bracket_left_m"].to_numpy(float)
        hi = completed["front_bracket_right_m"].to_numpy(float)
        yerr = np.vstack([np.maximum(y - lo, 0), np.maximum(hi - y, 0)])
        ax.errorbar(
            x,
            y / 1000.0,
            yerr=yerr / 1000.0,
            color=BLUE,
            marker="o",
            linewidth=2.0,
            capsize=3,
            label="CULBM (-1 K crossing)",
        )
    ref = reference[
        (reference["case"] == "density_current")
        & (reference["metric"] == "front_x_m")
    ].sort_values("h_m", ascending=False)
    if not ref.empty:
        y = ref["reference_value"].to_numpy(float)
        lo = ref["reference_lower"].to_numpy(float)
        hi = ref["reference_upper"].to_numpy(float)
        ax.errorbar(
            ref["h_m"],
            y / 1000.0,
            yerr=np.vstack([y - lo, hi - y]) / 1000.0,
            color=GREY,
            marker="s",
            linewidth=1.8,
            capsize=3,
            label="GEA AV75 bracket",
        )
    for status, color, marker in [
        ("failed", RED, "x"),
        ("incomplete", ORANGE, "^"),
    ]:
        data = main[main["status"] == status]
        if not data.empty:
            ax.scatter(
                data["h_m"],
                np.full(len(data), 0.04),
                transform=ax.get_xaxis_transform(),
                color=color,
                marker=marker,
                s=60,
                clip_on=False,
                label=status,
            )
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    density_ticks = sorted(
        {
            float(value)
            for value in pd.concat([main["h_m"], ref["h_m"]])
            if finite_number(value) and float(value) > 0
        },
        reverse=True,
    )
    ax.set_xticks(density_ticks)
    ax.set_xticklabels([f"{value:g}" for value in density_ticks])
    ax.set_xlabel("Grid spacing h (m) — finer →")
    ax.set_ylabel("Front position at t = 900 s (km)")
    ax.set_title(
        "Density current front: CULBM versus GEA AV75 "
        "(main sequence, fixed dt = 0.1 s)"
    )
    style_axis(ax)
    ax.legend(frameon=False)
    save_figure(fig, output / "density_front_comparison")


def plot_density_profile(records: list[RunRecord], output: Path) -> None:
    """绘制论文 Fig. 17 对应高度；无论文 raw data，故不计算 Lp 误差。"""
    candidates = [
        run
        for run in records
        if run.group == "main"
        and run.config.get("case") == "density_current"
        and run.status == "complete"
        and float(run.config.get("h_m", math.inf)) <= 100.0
    ]
    preferred: dict[float, RunRecord] = {}
    for run in candidates:
        h = float(run.config["h_m"])
        incumbent = preferred.get(h)
        if incumbent is None or registration_priority(
            infer_grid_registration(run.config)
        ) < registration_priority(infer_grid_registration(incumbent.config)):
            preferred[h] = run

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    profile_runs = [preferred[h] for h in sorted(preferred, reverse=True)]
    colors = plt.get_cmap("viridis")(
        np.linspace(0.18, 0.82, max(len(profile_runs), 1))
    )
    plotted = 0
    for color, run in zip(colors, profile_runs):
        h = float(run.config["h_m"])
        profile_path = run.path / "profile_z1200_tfinal.csv"
        try:
            if (
                profile_path.stat().st_mtime + 1.0e-2
                < (run.path / "run_config.json").stat().st_mtime
            ):
                continue
        except OSError:
            continue
        profile = load_csv(profile_path)
        required = {"x_m", "theta_prime_K"}
        if profile.empty or not required.issubset(profile.columns):
            continue
        x = pd.to_numeric(profile["x_m"], errors="coerce").to_numpy(float)
        theta = pd.to_numeric(
            profile["theta_prime_K"], errors="coerce"
        ).to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(theta)
        if not valid.any():
            continue
        ax.plot(
            x[valid] / 1000.0,
            theta[valid],
            color=color,
            linewidth=1.8,
            label=f"CULBM h={h:g} m",
        )
        plotted += 1
    ax.axhline(-1.0, color=RED, linewidth=1.0, linestyle="--", label="-1 K front threshold")
    ax.set_xlabel("x (km)")
    ax.set_ylabel(r"$\theta'(x,z=1200\,\mathrm{m},t=900\,\mathrm{s})$ (K)")
    ax.set_title(
        "Density current horizontal profile at z = 1200 m "
        "(main sequence, fixed dt = 0.1 s)"
    )
    style_axis(ax)
    if plotted:
        ax.legend(frameon=False, ncol=2)
    else:
        ax.text(
            0.5,
            0.5,
            "No completed primary profiles at h ≤ 100 m",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=GREY,
        )
    fig.text(
        0.5,
        0.01,
        "Paper Fig. 17 has no raw reference values: this plot supports only "
        "qualitative phase/amplitude and inter-grid audit; no literature Lp error.",
        ha="center",
        color=GREY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, output / "density_profile_z1200")


def clean_timeseries(run: RunRecord) -> pd.DataFrame:
    data = run.metrics.copy()
    if data.empty:
        return data
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "finite" in data:
        data = data[data["finite"] == 1]
    if "positive_density" in data:
        data = data[data["positive_density"] == 1]
    # 异常终止前最后若干步可能已严重污染；Mach 0.5 是诊断显示上限，不是完成判据。
    if "max_mach" in data:
        data = data[data["max_mach"] <= 0.5]
    return data


def select_timeseries_runs(
    records: list[RunRecord], case: str, limit: int = 5
) -> list[RunRecord]:
    candidates = [
        run
        for run in records
        if (
            run.group.lower() == "main"
            or "corrected" in run.group.lower()
            or "primary" in run.group.lower()
        )
        and run.config.get("case") == case
    ]
    preferred_by_h: dict[float, RunRecord] = {}
    status_rank = {"complete": 0, "failed": 1, "incomplete": 2}
    for run in candidates:
        h = float(run.config["h_m"])
        incumbent = preferred_by_h.get(h)
        key = (
            registration_priority(infer_grid_registration(run.config)),
            status_rank.get(run.status, 3),
        )
        if incumbent is None:
            preferred_by_h[h] = run
        else:
            incumbent_key = (
                registration_priority(infer_grid_registration(incumbent.config)),
                status_rank.get(incumbent.status, 3),
            )
            if key < incumbent_key:
                preferred_by_h[h] = run
    candidates = list(preferred_by_h.values())
    complete = sorted(
        [r for r in candidates if r.status == "complete"],
        key=lambda r: float(r.config["h_m"]),
    )
    failed = sorted(
        [r for r in candidates if r.status == "failed"],
        key=lambda r: float(r.config["h_m"]),
        reverse=True,
    )
    if len(complete) > 3:
        complete = [complete[0], complete[len(complete) // 2], complete[-1]]
    return (complete + failed[:1])[:limit]


def plot_timeseries(records: list[RunRecord], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), constrained_layout=True)
    specifications = [
        ("rising_thermal", "w_max_m_s", axes[0, 0], r"$w_{\max}$ (m s$^{-1}$)"),
        (
            "rising_thermal",
            "theta_prime_max_K",
            axes[1, 0],
            r"$\theta'_{\max}$ (K)",
        ),
        ("density_current", "front_x_m", axes[0, 1], "Front position (km)"),
        (
            "density_current",
            "theta_prime_min_K",
            axes[1, 1],
            r"$\theta'_{\min}$ (K)",
        ),
    ]
    colors = [BLUE, ORANGE, GREEN, GREY, RED]
    for case, metric, ax, ylabel in specifications:
        complete_values: list[np.ndarray] = []
        for color, run in zip(colors, select_timeseries_runs(records, case)):
            data = clean_timeseries(run)
            if data.empty or metric not in data:
                continue
            y = data[metric].to_numpy(float)
            if metric == "front_x_m":
                y = y / 1000.0
            if run.status == "complete":
                complete_values.append(y[np.isfinite(y)])
            ax.plot(
                data["time_s"],
                y,
                color=RED if run.status == "failed" else color,
                linestyle="--" if run.status != "complete" else "-",
                linewidth=1.5,
                label=(
                    f"h={run.config['h_m']:g} m ({run.status}, "
                    f"{'legacy reg.' if registration_priority(infer_grid_registration(run.config)) == 2 else 'GEA reg.'})"
                ),
            )
            if run.status == "failed":
                ax.axvline(run.observed_time, color=RED, alpha=0.3, linewidth=1)
        if complete_values:
            scale_values = np.concatenate(
                [values for values in complete_values if values.size]
            )
            if scale_values.size:
                lower = float(np.nanmin(scale_values))
                upper = float(np.nanmax(scale_values))
                span = max(upper - lower, 0.05 * max(abs(lower), abs(upper), 1.0))
                ax.set_ylim(lower - 0.08 * span, upper + 0.10 * span)
        ax.set_xlabel("Physical time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            ("Rising thermal" if case == "rising_thermal" else "Density current")
            + f": {METRIC_LABELS.get(metric, metric)}"
        )
        style_axis(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Transient diagnostics (failed prefixes clipped to completed-run scale)"
    )
    save_figure(fig, output / "transient_diagnostics")


def snapshot_time(path: Path) -> float:
    match = re.search(r"slice_t([0-9p]+)\.csv$", path.name)
    if not match:
        return math.nan
    return float(match.group(1).replace("p", "."))


def load_snapshot(path: Path) -> tuple[np.ndarray, ...]:
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError(f"快照列数不符合预期: {path}")
    order = np.lexsort((raw[:, 0], raw[:, 1]))
    raw = raw[order]
    x = np.unique(raw[:, 0])
    z = np.unique(raw[:, 1])
    expected = len(x) * len(z)
    if raw.shape[0] != expected:
        raise ValueError(f"快照不是规则 x-z 网格: {path}")
    shape = (len(z), len(x))
    return (
        x,
        z,
        raw[:, 2].reshape(shape),
        raw[:, 3].reshape(shape),
        raw[:, 4].reshape(shape),
    )


def select_snapshot_run(records: list[RunRecord], case: str) -> RunRecord | None:
    candidates = [
        run
        for run in records
        if (
            run.group.lower() == "main"
            or "corrected" in run.group.lower()
            or "primary" in run.group.lower()
        )
        and run.config.get("case") == case
        and run.status == "complete"
        and list((run.path / "snapshots").glob("slice_t*.csv"))
    ]
    return (
        min(
            candidates,
            key=lambda r: (
                registration_priority(infer_grid_registration(r.config)),
                float(r.config["h_m"]),
            ),
        )
        if candidates
        else None
    )


def plot_snapshots(records: list[RunRecord], case: str, output: Path) -> None:
    run = select_snapshot_run(records, case)
    if run is None:
        return
    available = sorted(
        (run.path / "snapshots").glob("slice_t*.csv"), key=snapshot_time
    )
    end_time = float(run.config["end_time_s"])
    targets = (
        [0.0, 300.0, 600.0, end_time]
        if case == "rising_thermal"
        else [0.0, 300.0, 600.0, 900.0]
    )
    selected: list[Path] = []
    for target in targets:
        closest = min(available, key=lambda p: abs(snapshot_time(p) - target))
        if closest not in selected:
            selected.append(closest)
    fields = [load_snapshot(path) for path in selected]
    vmin = min(float(np.nanmin(field[4])) for field in fields)
    vmax = max(float(np.nanmax(field[4])) for field in fields)
    vmin = min(vmin, -1.0e-9)
    vmax = max(vmax, 1.0e-9)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    if case == "density_current":
        fig, axes = plt.subplots(
            2, 2, figsize=(11.0, 5.6), constrained_layout=True
        )
    else:
        fig, axes = plt.subplots(
            1, len(fields), figsize=(4.0 * len(fields), 4.4), constrained_layout=True
        )
    axes = np.asarray(axes).ravel()
    image = None
    for ax, path, field in zip(axes, selected, fields):
        x, z, u, w, theta = field
        image = ax.pcolormesh(
            x / 1000.0,
            z / 1000.0,
            theta,
            shading="auto",
            cmap="coolwarm",
            norm=norm,
            rasterized=True,
        )
        sx = max(1, len(x) // 24)
        sz = max(1, len(z) // 18)
        xx, zz = np.meshgrid(x / 1000.0, z / 1000.0)
        speed_max = float(np.nanmax(np.hypot(u, w)))
        if speed_max > 1.0e-12:
            arrow_length_km = min(float(np.ptp(x)), float(np.ptp(z))) / 1000.0 / 18.0
            ax.quiver(
                xx[::sz, ::sx],
                zz[::sz, ::sx],
                u[::sz, ::sx],
                w[::sz, ::sx],
                color="#1F2937",
                alpha=0.55,
                pivot="middle",
                angles="xy",
                scale_units="xy",
                scale=speed_max / max(arrow_length_km, 1.0e-9),
                width=0.0022,
            )
        ax.set_title(f"t = {snapshot_time(path):g} s")
        ax.set_xlabel("x (km)")
        ax.set_aspect("equal", adjustable="box")
        style_axis(ax)
    axes[0].set_ylabel("z (km)")
    if case == "density_current" and len(axes) > 2:
        axes[2].set_ylabel("z (km)")
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), label=r"$\theta'$ (K)", shrink=0.82)
    title = "Rising thermal" if case == "rising_thermal" else "Density current"
    registration_label = (
        "legacy half-cell shift"
        if registration_priority(infer_grid_registration(run.config)) == 2
        else "GEA-aligned registration"
    )
    snapshot_mach = float(
        (run.summary or {}).get("max_mach", run.last_valid_max_mach)
    )
    fig.suptitle(
        f"{title}: scalar field + normalized velocity arrows\n"
        f"h={float(run.config['h_m']):g} m, "
        f"dt={float(run.config['dt_s']):g} s, max Ma={snapshot_mach:.3f}, "
        f"{registration_label}"
    )
    save_figure(
        fig,
        output
        / (
            "rising_thermal_snapshots"
            if case == "rising_thermal"
            else "density_current_snapshots"
        ),
    )


def plot_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    sensitivity = sensitivity[sensitivity["run_group"] == "sensitivity"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, case in zip(axes, ["rising_thermal", "density_current"]):
        data = sensitivity[
            (sensitivity["case"] == case) & (sensitivity["status"] == "complete")
        ].copy()
        if data.empty:
            ax.text(
                0.5,
                0.5,
                "No completed sensitivity runs",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=GREY,
            )
            ax.set_axis_off()
            continue
        label_names = {
            "dt_half": "dt/2",
            "thermal_D3Q7": "thermal D3Q7",
            "numeric_viscosity": "numeric viscosity",
        }
        labels = [
            f"{label_names.get(row.sensitivity_kind, row.sensitivity_kind)}\n"
            f"h={row.h_m:g} m"
            for row in data.itertuples()
        ]
        values = data["relative_to_matching_main_percent"].to_numpy(float)
        colors = [BLUE if finite_number(value) else GREY for value in values]
        positions = np.arange(len(data))
        bars = ax.bar(
            positions, np.nan_to_num(values, nan=0.0), color=colors, width=0.65
        )
        for x, value in zip(positions, values):
            if not finite_number(value):
                ax.text(x, 0, "no matching\nmain result", ha="center", va="bottom")
        for bar, value in zip(bars, values):
            if finite_number(value):
                ax.annotate(
                    f"{value:+.2f}%",
                    (
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                    ),
                    xytext=(0, 4 if value >= 0 else -4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8.5,
                )
        ax.axhline(0, color="#111827", linewidth=0.9)
        ax.set_xticks(positions, labels)
        ax.set_ylabel("Change from matching primary run (%)")
        ax.set_title("Rising thermal: max w" if case == "rising_thermal" else "Density current: front")
        style_axis(ax)
    fig.suptitle("Sensitivity to time step, thermal lattice and viscosity mapping")
    save_figure(fig, output / "sensitivity_comparison")


def plot_time_step_convergence(
    convergence: pd.DataFrame, output: Path
) -> None:
    if convergence.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    for column, case in enumerate(["rising_thermal", "density_current"]):
        case_rows = convergence[
            (convergence["case"] == case)
            & (convergence["status"] == "complete")
        ].copy()
        if case_rows.empty:
            for row in range(2):
                axes[row, column].set_axis_off()
            continue
        counts = case_rows.groupby("h_m")["dt_s"].nunique()
        selected_h = min(
            counts[counts == counts.max()].index.astype(float)
        )
        data = case_rows[
            np.isclose(
                pd.to_numeric(case_rows["h_m"], errors="coerce"),
                selected_h,
                equal_nan=False,
            )
        ].sort_values("dt_s", ascending=False)
        dt = data["dt_s"].to_numpy(float)
        value = data["value"].to_numpy(float)
        if case == "density_current":
            value = value / 1000.0
            value_label = "Front position (km)"
            title = "Density current"
        else:
            value_label = r"$w_{\max}$ (m s$^{-1}$)"
            title = "Rising thermal"
        axes[0, column].plot(dt, value, color=BLUE, marker="o", linewidth=2)
        axes[0, column].set_ylabel(value_label)
        axes[0, column].set_title(f"{title}, h={selected_h:g} m")
        axes[1, column].plot(
            dt,
            data["max_mach"].to_numpy(float),
            color=ORANGE,
            marker="o",
            linewidth=2,
        )
        axes[1, column].axhline(
            0.1, color=GREEN, linestyle="--", linewidth=1.3, label="Ma = 0.1"
        )
        axes[1, column].axhline(
            0.3, color=RED, linestyle=":", linewidth=1.3, label="Ma = 0.3"
        )
        axes[1, column].set_ylabel("Maximum lattice Mach")
        axes[1, column].set_xlabel("Physical time step dt (s) — smaller →")
        axes[1, column].legend(frameon=False)
        for row in range(2):
            ax = axes[row, column]
            ax.set_xscale("log", base=2)
            ax.invert_xaxis()
            ax.set_xticks(dt)
            ax.set_xticklabels([f"{item:g}" for item in dt])
            style_axis(ax)
    fig.suptitle(
        "Fine-grid time-step convergence and weak-compressibility diagnostic"
    )
    save_figure(fig, output / "fine_grid_time_step_convergence")


def plot_fine_grid_scaling(
    comparison: pd.DataFrame, output: Path
) -> None:
    if comparison.empty:
        return
    labels = {
        "fixed_dt": "fixed dt",
        "acoustic_dt_proportional_h": "dt ∝ h",
        "diffusive_dt_proportional_h2": "dt ∝ h²",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for ax, case in zip(axes, ["rising_thermal", "density_current"]):
        data = comparison[comparison["case"] == case].copy()
        positions = np.arange(len(data))
        values = data["relative_change_percent"].to_numpy(float)
        bars = ax.bar(
            positions,
            np.nan_to_num(values, nan=0.0),
            color=[BLUE, ORANGE, GREEN],
            width=0.68,
        )
        for bar, value, mach in zip(
            bars, values, data["refined_max_mach"].to_numpy(float)
        ):
            if finite_number(value):
                ax.annotate(
                    f"{value:+.2f}%\nMa={mach:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4 if value >= 0 else -4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8.5,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0,
                    "not complete",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=GREY,
                )
        ax.axhline(0, color="#111827", linewidth=0.9)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            lower = min(0.0, float(np.min(finite_values)))
            upper = max(0.0, float(np.max(finite_values)))
            span = max(upper - lower, 1.0)
            ax.set_ylim(lower - 0.25 * span, upper + 0.18 * span)
        ax.set_xticks(
            positions,
            [labels.get(item, item) for item in data["scaling"]],
        )
        ax.set_ylabel("Change from previous finest main grid (%)")
        ax.set_title(
            "Rising thermal: w_max"
            if case == "rising_thermal"
            else "Density current: front position"
        )
        style_axis(ax)
    fig.suptitle(
        "One-level D3Q19 refinement under fixed, acoustic and diffusive scaling"
    )
    save_figure(fig, output / "fine_grid_scaling_comparison")


def plot_stability_matrix(stability: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, case in zip(axes, ["rising_thermal", "density_current"]):
        data = stability[stability["case"] == case].copy()
        data = data.sort_values(["dt_s", "momentum_collision"])
        labels = [
            f"{row.momentum_collision}, dt={row.dt_s:g} s"
            + (" (primary)" if row.run_group == "main" else "")
            for row in data.itertuples()
        ]
        fractions = 100.0 * data["completion_fraction"].to_numpy(float)
        colors = [GREEN if status == "complete" else RED for status in data["status"]]
        y = np.arange(len(data))
        ax.barh(y, fractions, color=colors, height=0.58)
        for position, (_, row) in zip(y, data.iterrows()):
            ax.text(
                min(float(row["completion_fraction"]) * 100.0 + 2.0, 98.0),
                position,
                f"{row['observed_time_s']:g}/{row['end_time_s']:g} s, {row['status']}",
                va="center",
                ha="right" if float(row["completion_fraction"]) > 0.82 else "left",
                color="#111827",
                fontsize=8.5,
            )
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Completed physical duration (%)")
        ax.set_title(
            "Rising thermal, h=250 m"
            if case == "rising_thermal"
            else "Density current, h=400 m"
        )
        style_axis(ax)
    fig.suptitle(
        "Coarse-grid stability: collision model and time-step matrix", y=0.985
    )
    legend = [
        Line2D([0], [0], color=GREEN, linewidth=8, label="complete"),
        Line2D([0], [0], color=RED, linewidth=8, label="failed"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, output / "stability_matrix")


def plot_viscosity_profile(
    comparison: pd.DataFrame, output: Path
) -> None:
    metrics = [
        "w_min_m_s",
        "w_max_m_s",
        "theta_prime_min_K",
        "theta_prime_max_K",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 7.2))
    for ax, metric in zip(axes.flat, metrics):
        data = comparison[
            (comparison["metric"] == metric)
            & (comparison["status"] == "complete")
        ].sort_values("rho_ref_kg_m3")
        ax.plot(
            data["rho_ref_kg_m3"],
            data["culbm_value"],
            color=BLUE,
            marker="o",
            linewidth=1.8,
            label="CULBM",
        )
        if not data.empty and finite_number(data.iloc[0]["gea_reference_value"]):
            ax.axhline(
                float(data.iloc[0]["gea_reference_value"]),
                color=GREY,
                linestyle="--",
                linewidth=1.5,
                label="GEA AV15",
            )
        for _, row in data.iterrows():
            ax.annotate(
                f"{float(row['culbm_value']):.3g}",
                (float(row["rho_ref_kg_m3"]), float(row["culbm_value"])),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel(r"Viscosity mapping density $\rho_\mathrm{ref}$ (kg m$^{-3}$)")
        ax.set_ylabel(METRIC_LABELS[metric])
        style_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Rising thermal h=31.25 m: sensitivity to density used in μ/ρ mapping",
        y=0.985,
    )
    fig.text(
        0.5,
        0.01,
        "ρref=1.161 kg m⁻³: surface reference; 0.90: ~3 km; 0.55: ~8 km.",
        ha="center",
        color=GREY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.89))
    save_figure(fig, output / "bubble_viscosity_profile")


def write_status_report(
    summary: pd.DataFrame,
    main_comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    stability: pd.DataFrame,
    viscosity_profile: pd.DataFrame,
    output: Path,
) -> None:
    counts = summary["status"].value_counts().to_dict() if not summary.empty else {}

    def one_value(
        frame: pd.DataFrame, mask: pd.Series, column: str, digits: int = 3
    ) -> str:
        selected = frame.loc[mask, column]
        if selected.empty or not finite_number(selected.iloc[0]):
            return "n/a"
        return f"{float(selected.iloc[0]):.{digits}f}"

    def gea_relative_text(frame: pd.DataFrame, metric: str) -> str:
        selected = frame.loc[
            frame["metric"] == metric, "relative_difference_percent"
        ]
        if selected.empty or not finite_number(selected.iloc[0]):
            return "无同网格 GEA 表值"
        return f"相对 GEA {float(selected.iloc[0]):+.2f}%"

    def time_step_details(case: str) -> str:
        controls = standard[
            (standard["case"] == case)
            & (pd.to_numeric(standard["dt_s"], errors="coerce") < 0.1)
        ].sort_values(["h_m", "dt_s"], ascending=[True, True])
        if controls.empty:
            return "n/a"
        return " / ".join(
            (
                f"h={float(row.h_m):g}, dt={float(row.dt_s):g}: "
                f"{float(row.relative_to_matching_main_percent):+.2f}%, "
                f"Ma={float(row.max_mach):.3f}"
            )
            if finite_number(row.relative_to_matching_main_percent)
            and finite_number(row.max_mach)
            else f"h={float(row.h_m):g}, dt={float(row.dt_s):g}: {row.status}"
            for row in controls.itertuples()
        )

    standard = sensitivity[sensitivity["run_group"] == "sensitivity"]
    bubble_complete = main_comparison[
        (main_comparison["case"] == "rising_thermal")
        & (main_comparison["status"] == "complete")
        & (main_comparison["metric"] == "w_max_m_s")
        & main_comparison["culbm_value"].notna()
    ]
    if bubble_complete.empty:
        bubble_finest_h = math.nan
        bubble_finest_label = "n/a"
        bubble_fine = bubble_complete
    else:
        bubble_finest_h = float(bubble_complete["h_m"].min())
        bubble_finest_label = f"{bubble_finest_h:g}"
        bubble_fine = main_comparison[
            (main_comparison["case"] == "rising_thermal")
            & np.isclose(main_comparison["h_m"].astype(float), bubble_finest_h)
        ]
    density_refs = main_comparison[
        (main_comparison["case"] == "density_current")
        & main_comparison["reference_value"].notna()
    ].sort_values("h_m", ascending=False)
    lines = [
        "# 分析产物说明",
        "",
        "本目录由 `analyze_results.py` 生成。最终物理量只来自严格正常完成的 run；",
        "失败或未完成 run 的末帧不会进入比较表。",
        "",
        "## 本次扫描状态",
        "",
        f"- 完成：{counts.get('complete', 0)}",
        f"- 失败：{counts.get('failed', 0)}",
        f"- 未完成：{counts.get('incomplete', 0)}",
        "",
        "## 最终关键量",
        "",
        (
            f"- 热泡最细主线 h={bubble_finest_label} m："
            f"w_max={one_value(bubble_fine, bubble_fine.metric == 'w_max_m_s', 'culbm_value')} m/s"
            f"（{gea_relative_text(bubble_fine, 'w_max_m_s')}），"
            f"theta'_max={one_value(bubble_fine, bubble_fine.metric == 'theta_prime_max_K', 'culbm_value')} K"
            f"（{gea_relative_text(bubble_fine, 'theta_prime_max_K')}）。"
        ),
        (
            "- 密度流前缘相对 GEA（h=100/50/25 m）："
            + " / ".join(
                f"{float(row.relative_difference_percent):+.2f}%"
                for row in density_refs.itertuples()
            )
            + "。"
        ),
        (
            "- 减小时间步（相对同 h、dt=0.1 主线）：热泡 w_max "
            f"{time_step_details('rising_thermal')}；密度流前缘 "
            f"{time_step_details('density_current')}。"
        ),
        (
            "- D3Q7：热泡 w_max "
            f"{one_value(standard, (standard.case == 'rising_thermal') & (standard.sensitivity_kind == 'thermal_D3Q7'), 'relative_to_matching_main_percent', 2)}%，"
            "密度流前缘 "
            f"{one_value(standard, (standard.case == 'density_current') & (standard.sensitivity_kind == 'thermal_D3Q7'), 'relative_to_matching_main_percent', 2)}%。"
        ),
        (
            "- numeric viscosity：热泡 w_max "
            f"{one_value(standard, (standard.case == 'rising_thermal') & (standard.sensitivity_kind == 'numeric_viscosity'), 'relative_to_matching_main_percent', 2)}%，"
            "密度流前缘 "
            f"{one_value(standard, (standard.case == 'density_current') & (standard.sensitivity_kind == 'numeric_viscosity'), 'relative_to_matching_main_percent', 2)}%。"
        ),
        (
            "- 背景密度映射对热泡 theta'_max：rho_ref=0.90 / 0.55 kg/m3 相对地面映射 "
            f"{one_value(viscosity_profile, (viscosity_profile.sensitivity_kind == 'rho_ref_z3km') & (viscosity_profile.metric == 'theta_prime_max_K'), 'relative_to_primary_percent', 2)}% / "
            f"{one_value(viscosity_profile, (viscosity_profile.sensitivity_kind == 'rho_ref_z8km') & (viscosity_profile.metric == 'theta_prime_max_K'), 'relative_to_primary_percent', 2)}%。"
        ),
        (
            "- 粗网格：原 SRT dt=0.1 s 与 TRT dt=0.1 s 均失败；"
            "SRT dt=1 s 在两个算例均走到终止时刻。后者是稳定性替代设置，"
            "不是原 dt 的严格复现。"
        ),
        "",
        "## 图表",
        "",
        "- `bubble_grid_convergence.*`：热泡终态极值与 GEA AV15 表 1。",
        "- `density_front_comparison.*`：密度流 -1 K 前缘与 GEA AV75 区间。",
        "- `density_profile_z1200.*`：所有 h<=100 m 已完成主线的 t=900 s 水平剖面；只作网格间定性审计。",
        "- `transient_diagnostics.*`：主要瞬态量；失败 run 只显示有效前缀。",
        "- `rising_thermal_snapshots.*`、`density_current_snapshots.*`：最细已完成主网格场。",
        "- `sensitivity_comparison.*`：相对同网格主计算的变化。",
        "- `fine_grid_time_step_convergence.*`：最细 D3Q19 网格主指标与最大 Mach 的 dt 收敛。",
        "- `fine_grid_scaling_comparison.*`：h/2 时固定 dt、dt∝h 与 dt∝h² 三条缩放路径。",
        "- `stability_matrix.*`：粗网格 SRT/TRT 与 dt 的完成时长及成败。",
        "- `bubble_viscosity_profile.*`：rho_ref=1.161/0.90/0.55 黏度映射敏感性。",
        "",
        "PNG 为 200 dpi；SVG 中场数据采用栅格层，文字与坐标保持矢量，避免文件过大。",
        "",
        "注意：未显式记录 `grid_registration` 的旧热泡 run 按当前 runner 审计结果标记为",
        "`legacy_half_cell_shift`，不视为 exact GEA setup。存在修正版本时，比较表按",
        "官方注册优先、完成状态次优先的规则选择；密度流旧结果与官方边界注册一致。",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    # Windows 非交互 PowerShell 可能回退到 cp1252，显式使用 UTF-8 输出中文诊断。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--results", type=Path, default=here / "results")
    parser.add_argument("--reference", type=Path, default=here / "reference_metrics.csv")
    parser.add_argument("--output", type=Path, default=here / "analysis")
    args = parser.parse_args()

    results_root = args.results.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference)
    records = discover_runs(results_root)
    if not records:
        raise SystemExit(f"没有发现 run_config.json: {results_root}")

    summary = build_summary(records)
    main_comparison = build_main_comparison(summary, reference)
    sensitivity = build_sensitivity(summary)
    time_step_convergence = build_time_step_convergence(summary)
    fine_grid_scaling = build_fine_grid_scaling_comparison(summary)
    stability = build_stability_comparison(summary)
    viscosity_profile = build_viscosity_profile_comparison(summary, reference)
    summary.to_csv(output / "summary.csv", index=False, float_format="%.10g")
    main_comparison.to_csv(
        output / "main_grid_comparison.csv", index=False, float_format="%.10g"
    )
    sensitivity.to_csv(
        output / "sensitivity_comparison.csv", index=False, float_format="%.10g"
    )
    time_step_convergence.to_csv(
        output / "time_step_convergence.csv",
        index=False,
        float_format="%.10g",
    )
    fine_grid_scaling.to_csv(
        output / "fine_grid_scaling_comparison.csv",
        index=False,
        float_format="%.10g",
    )
    stability.to_csv(
        output / "stability_comparison.csv", index=False, float_format="%.10g"
    )
    viscosity_profile.to_csv(
        output / "viscosity_profile_comparison.csv",
        index=False,
        float_format="%.10g",
    )

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 13,
            "legend.fontsize": 8.5,
            "savefig.facecolor": "white",
        }
    )
    plot_bubble_convergence(summary, reference, output)
    plot_density_front(summary, reference, output)
    plot_density_profile(records, output)
    plot_timeseries(records, output)
    plot_snapshots(records, "rising_thermal", output)
    plot_snapshots(records, "density_current", output)
    plot_sensitivity(sensitivity, output)
    plot_time_step_convergence(time_step_convergence, output)
    plot_fine_grid_scaling(fine_grid_scaling, output)
    plot_stability_matrix(stability, output)
    plot_viscosity_profile(viscosity_profile, output)
    write_status_report(
        summary,
        main_comparison,
        sensitivity,
        stability,
        viscosity_profile,
        output,
    )

    print(
        "ANALYSIS_COMPLETE "
        f"runs={len(records)} complete={(summary.status == 'complete').sum()} "
        f"failed={(summary.status == 'failed').sum()} "
        f"incomplete={(summary.status == 'incomplete').sum()} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
