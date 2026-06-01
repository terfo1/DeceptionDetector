"""Experiment run manifest creation and metric history tracking."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .file_hashing import collect_file_metadata
from .tracking_config import (
    RUN_VERSION_PREFIX,
    TRACKED_REPORT_FILES,
    TRACKING_OUTPUT_DIR,
)


METRIC_FILES = [
    "reports/baselines/baseline_metrics.csv",
    "reports/sequences/sequence_model_metrics.csv",
    "reports/sequences/tcn_metrics.csv",
]

METRIC_HISTORY_COLUMNS = [
    "run_version",
    "dataset_version",
    "split",
    "model_name",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "false_positive_rate",
    "false_negative_rate",
    "number_of_samples",
]


def get_next_run_version(output_dir: str) -> str:
    """Return the next run_vNNN identifier."""
    path = Path(output_dir) / "experiment_runs.csv"
    if not path.exists():
        return f"{RUN_VERSION_PREFIX}001"
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return f"{RUN_VERSION_PREFIX}001"
    if df.empty or "run_version" not in df.columns:
        return f"{RUN_VERSION_PREFIX}001"

    numbers = []
    for value in df["run_version"].dropna().astype(str):
        if value.startswith(RUN_VERSION_PREFIX):
            try:
                numbers.append(int(value.replace(RUN_VERSION_PREFIX, "")))
            except ValueError:
                continue
    return f"{RUN_VERSION_PREFIX}{(max(numbers) + 1 if numbers else 1):03d}"


def collect_model_metrics() -> pd.DataFrame:
    """Load all available model metric CSV files."""
    frames = []
    for path in METRIC_FILES:
        file_path = Path(path)
        if not file_path.exists() or file_path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = path
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    metrics = pd.concat(frames, ignore_index=True, sort=False)
    if {"split", "model_name"}.issubset(metrics.columns):
        metrics = metrics.drop_duplicates(subset=["split", "model_name"], keep="last")
    return metrics


def collect_model_selection() -> dict[str, Any]:
    """Read the selected live model configuration when available."""
    path = Path("reports/model_selection/selected_model_config.json")
    if not path.exists():
        return {
            "selected_model": "",
            "fallback_model": "",
            "disabled_models": [],
            "experimental_models": [],
            "warning": "Missing selected model config.",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "selected_model": "",
            "fallback_model": "",
            "disabled_models": [],
            "experimental_models": [],
            "warning": f"Invalid selected model config JSON: {exc}",
        }
    return {
        "selected_model": data.get("selected_model", ""),
        "fallback_model": data.get("fallback_model", ""),
        "disabled_models": data.get("disabled_models", []),
        "experimental_models": data.get("experimental_models", []),
    }


def collect_best_model_summary(metrics_df: pd.DataFrame) -> dict[str, Any]:
    """Find the best model by F1, preferring test metrics over validation."""
    empty_summary = {
        "best_model_name": "",
        "best_split": "",
        "best_f1": None,
        "best_accuracy": None,
        "best_roc_auc": None,
        "false_positive_rate": None,
        "false_negative_rate": None,
        "number_of_samples": None,
    }
    if metrics_df.empty or "split" not in metrics_df.columns or "model_name" not in metrics_df.columns:
        return empty_summary

    split_used = "test" if (metrics_df["split"].astype(str) == "test").any() else "validation"
    candidates = metrics_df[metrics_df["split"].astype(str) == split_used].copy()
    if candidates.empty or "f1" not in candidates.columns:
        return empty_summary

    candidates["f1_numeric"] = pd.to_numeric(candidates["f1"], errors="coerce")
    candidates = candidates.dropna(subset=["f1_numeric"])
    if candidates.empty:
        return empty_summary

    best = candidates.sort_values("f1_numeric", ascending=False).iloc[0]
    return {
        "best_model_name": _value(best, "model_name"),
        "best_split": split_used,
        "best_f1": _number(best, "f1"),
        "best_accuracy": _number(best, "accuracy"),
        "best_roc_auc": _number(best, "roc_auc"),
        "false_positive_rate": _number(best, "false_positive_rate"),
        "false_negative_rate": _number(best, "false_negative_rate"),
        "number_of_samples": _number(best, "number_of_samples"),
    }


def create_run_manifest(dataset_version: str | None = None) -> dict[str, Any]:
    """Create the latest experiment run manifest from existing reports."""
    output_dir = Path(TRACKING_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_version = get_next_run_version(TRACKING_OUTPUT_DIR)
    metrics_df = collect_model_metrics()
    selection = collect_model_selection()
    best_summary = collect_best_model_summary(metrics_df)
    report_hashes = collect_file_metadata(TRACKED_REPORT_FILES)

    warnings: list[str] = []
    if metrics_df.empty:
        warnings.append("No model metrics were found.")
    if selection.get("warning"):
        warnings.append(str(selection["warning"]))
    missing_reports = report_hashes[report_hashes["exists"] == False]
    for _, row in missing_reports.iterrows():
        warnings.append(f"Missing tracked report file: {row['path']}")
    if not best_summary.get("best_model_name"):
        warnings.append("No best model could be selected from available metrics.")

    manifest = {
        "run_version": run_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_version": dataset_version,
        "git_commit_optional": _git_commit(),
        "model_metrics_summary": _metrics_summary(metrics_df),
        "best_model_summary": best_summary,
        "model_selection": selection,
        "tracked_report_hashes": report_hashes.to_dict(orient="records"),
        "warnings": warnings,
        "note": "This run tracks prototype-level model experiments. Metrics are only scientifically meaningful with sufficient participant count and subject-independent evaluation.",
    }
    return manifest


def save_run_manifest(manifest: dict[str, Any], output_path: str) -> None:
    """Save run manifest JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def append_experiment_run(manifest: dict[str, Any]) -> None:
    """Append a row to experiment_runs.csv."""
    path = Path(TRACKING_OUTPUT_DIR) / "experiment_runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "run_version",
        "created_at",
        "dataset_version",
        "selected_model",
        "fallback_model",
        "best_model_name",
        "best_split",
        "best_f1",
        "best_accuracy",
        "best_roc_auc",
        "false_positive_rate",
        "false_negative_rate",
        "number_of_samples",
        "manifest_path",
        "warning_count",
    ]
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(header)

    selection = manifest.get("model_selection", {})
    best = manifest.get("best_model_summary", {})
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writerow(
            {
                "run_version": manifest.get("run_version", ""),
                "created_at": manifest.get("created_at", ""),
                "dataset_version": manifest.get("dataset_version", ""),
                "selected_model": selection.get("selected_model", ""),
                "fallback_model": selection.get("fallback_model", ""),
                "best_model_name": best.get("best_model_name", ""),
                "best_split": best.get("best_split", ""),
                "best_f1": best.get("best_f1", ""),
                "best_accuracy": best.get("best_accuracy", ""),
                "best_roc_auc": best.get("best_roc_auc", ""),
                "false_positive_rate": best.get("false_positive_rate", ""),
                "false_negative_rate": best.get("false_negative_rate", ""),
                "number_of_samples": best.get("number_of_samples", ""),
                "manifest_path": str(Path(TRACKING_OUTPUT_DIR) / "latest_run_manifest.json"),
                "warning_count": len(manifest.get("warnings", [])),
            }
        )


def save_metric_history(
    metrics_df: pd.DataFrame,
    run_version: str | None = None,
    dataset_version: str | None = None,
) -> None:
    """Append available metrics to metric_history.csv for this run."""
    if metrics_df.empty:
        return
    path = Path(TRACKING_OUTPUT_DIR) / "metric_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = metrics_df.copy()
    rows["run_version"] = run_version or ""
    rows["dataset_version"] = dataset_version or ""
    for column in METRIC_HISTORY_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    rows = rows[METRIC_HISTORY_COLUMNS]

    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path)
            rows = pd.concat([existing, rows], ignore_index=True, sort=False)
        except pd.errors.EmptyDataError:
            pass
    rows.to_csv(path, index=False)


def _metrics_summary(metrics_df: pd.DataFrame) -> list[dict[str, Any]]:
    if metrics_df.empty:
        return []
    columns = [
        "split",
        "model_name",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
        "number_of_samples",
    ]
    available = [column for column in columns if column in metrics_df.columns]
    return metrics_df[available].to_dict(orient="records")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _value(row: pd.Series, column: str) -> str:
    return str(row[column]) if column in row and pd.notna(row[column]) else ""


def _number(row: pd.Series, column: str) -> float | int | None:
    if column not in row or pd.isna(row[column]):
        return None
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        return None
    if column == "number_of_samples":
        return int(value)
    return float(value)
