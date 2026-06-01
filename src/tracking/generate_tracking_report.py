"""Generate dataset versioning and experiment run tracking artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .dataset_versioning import (
    append_dataset_version,
    create_dataset_manifest,
    save_dataset_manifest,
)
from .file_hashing import collect_file_metadata
from .run_tracking import (
    append_experiment_run,
    collect_model_metrics,
    create_run_manifest,
    save_metric_history,
    save_run_manifest,
)
from .tracking_config import (
    TRACKED_PROCESSED_FILES,
    TRACKED_RAW_FILES,
    TRACKED_REPORT_FILES,
    TRACKING_OUTPUT_DIR,
)


REQUIRED_CORE_FILES = [
    "data/raw/trials.csv",
    "data/raw/gaze_samples.csv",
    "data/processed/windows.csv",
    "data/processed/window_features.csv",
    "data/processed/split_report.txt",
    "reports/model_comparison/model_comparison_report.txt",
    "reports/model_selection/model_selection_report.txt",
]


def main() -> None:
    """Create tracking manifests and reports."""
    print("Generating dataset versioning and experiment run tracking report...")
    output_dir = Path(TRACKING_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = create_dataset_manifest()
    dataset_manifest_path = output_dir / "latest_dataset_manifest.json"
    save_dataset_manifest(dataset_manifest, str(dataset_manifest_path))
    append_dataset_version(dataset_manifest)

    run_manifest = create_run_manifest(dataset_manifest.get("dataset_version"))
    run_manifest_path = output_dir / "latest_run_manifest.json"
    save_run_manifest(run_manifest, str(run_manifest_path))
    append_experiment_run(run_manifest)

    file_hashes = collect_file_metadata(
        TRACKED_RAW_FILES + TRACKED_PROCESSED_FILES + TRACKED_REPORT_FILES
    )
    file_hashes.to_csv(output_dir / "file_hashes_latest.csv", index=False)

    metrics_df = collect_model_metrics()
    save_metric_history(
        metrics_df,
        run_version=run_manifest.get("run_version"),
        dataset_version=dataset_manifest.get("dataset_version"),
    )

    _write_tracking_report(dataset_manifest, run_manifest, file_hashes, output_dir)
    _write_tracking_summary(dataset_manifest, run_manifest, output_dir)
    _write_reproducibility_checklist(file_hashes, output_dir)

    stats = dataset_manifest["dataset_statistics"]
    best = run_manifest["best_model_summary"]
    print(f"Dataset version: {dataset_manifest['dataset_version']}")
    print(f"Run version: {run_manifest['run_version']}")
    print(
        "Dataset counts: "
        f"{stats.get('participant_count', 0)} participants, "
        f"{stats.get('trial_count', 0)} trials, "
        f"{stats.get('total_window_count', 0)} windows"
    )
    if best.get("best_model_name"):
        print(
            "Best model by available F1: "
            f"{best['best_model_name']} ({best['best_split']} F1={best['best_f1']})"
        )
    print(f"Files saved to {TRACKING_OUTPUT_DIR}")


def _write_tracking_report(
    dataset_manifest: dict,
    run_manifest: dict,
    file_hashes: pd.DataFrame,
    output_dir: Path,
) -> None:
    stats = dataset_manifest["dataset_statistics"]
    selection = run_manifest.get("model_selection", {})
    best = run_manifest.get("best_model_summary", {})
    warnings = dataset_manifest.get("warnings", []) + run_manifest.get("warnings", [])
    core_status = _core_file_status()
    metrics_table = _metrics_text(run_manifest.get("model_metrics_summary", []))

    lines = [
        "Dataset Versioning and Experiment Run Tracking Report",
        "",
        "1. Overview",
        "This report records dataset version and experiment run metadata for reproducibility. It uses local JSON, CSV, and Markdown files only.",
        "",
        "2. Dataset Version",
        f"Dataset version: {dataset_manifest.get('dataset_version', '')}",
        f"Participant count: {stats.get('participant_count', 0)}",
        f"Session count: {stats.get('session_count', 0)}",
        f"Trial count: {stats.get('trial_count', 0)}",
        f"Gaze sample count: {stats.get('gaze_sample_count', 0)}",
        f"Train windows: {stats.get('train_window_count', 0)}",
        f"Validation windows: {stats.get('validation_window_count', 0)}",
        f"Test windows: {stats.get('test_window_count', 0)}",
        f"Total windows: {stats.get('total_window_count', 0)}",
        f"Train participants: {stats.get('train_participant_count', 0)}",
        f"Validation participants: {stats.get('validation_participant_count', 0)}",
        f"Test participants: {stats.get('test_participant_count', 0)}",
        f"Total split participants: {stats.get('total_split_participants', 0)}",
        "",
        "3. File Hashes",
        "SHA256 hashes are saved in file_hashes_latest.csv and in the JSON manifests to detect whether tracked files changed between runs.",
        f"Tracked files found: {int(file_hashes['exists'].sum()) if not file_hashes.empty else 0}",
        f"Tracked files missing: {int((file_hashes['exists'] == False).sum()) if not file_hashes.empty else 0}",
        "",
        "4. Experiment Run",
        f"Run version: {run_manifest.get('run_version', '')}",
        f"Linked dataset version: {run_manifest.get('dataset_version', '')}",
        f"Selected model: {selection.get('selected_model', '')}",
        f"Fallback model: {selection.get('fallback_model', '')}",
        f"Best model by available F1: {best.get('best_model_name', '')}",
        f"Best split: {best.get('best_split', '')}",
        f"Best F1: {best.get('best_f1', '')}",
        f"Best accuracy: {best.get('best_accuracy', '')}",
        f"Best ROC-AUC: {best.get('best_roc_auc', '')}",
        "",
        "Metrics summary:",
        metrics_table,
        "",
        "5. Reproducibility Status",
        core_status,
        "",
        "6. Warnings",
        _warnings_text(warnings),
        "",
        "7. Scientific Note",
        "Current results are prototype-level unless enough participants are collected and the participant-independent split is preserved. The system estimates deception risk under a controlled experimental protocol; it does not determine definitive lying.",
        "",
        "8. Recommended Next Action",
        "Collect more participants, rerun the full pipeline, regenerate this tracking report, and compare dataset_versions.csv, experiment_runs.csv, and metric_history.csv across runs.",
        "",
    ]
    (output_dir / "tracking_report.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_tracking_summary(dataset_manifest: dict, run_manifest: dict, output_dir: Path) -> None:
    stats = dataset_manifest["dataset_statistics"]
    selection = run_manifest.get("model_selection", {})
    best = run_manifest.get("best_model_summary", {})
    warnings = dataset_manifest.get("warnings", []) + run_manifest.get("warnings", [])
    lines = [
        "# Tracking Summary",
        "",
        f"- Latest dataset version: {dataset_manifest.get('dataset_version', '')}",
        f"- Latest run version: {run_manifest.get('run_version', '')}",
        f"- Participant count: {stats.get('participant_count', 0)}",
        f"- Trial count: {stats.get('trial_count', 0)}",
        f"- Total windows: {stats.get('total_window_count', 0)}",
        f"- Best current model: {best.get('best_model_name', '') or 'not available'}",
        f"- Selected live model: {selection.get('selected_model', '') or 'not available'}",
        "",
        "## Main Warnings",
        _markdown_warning_list(warnings),
        "",
    ]
    (output_dir / "tracking_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_reproducibility_checklist(file_hashes: pd.DataFrame, output_dir: Path) -> None:
    status_by_path = {
        str(row["path"]): bool(row["exists"]) for _, row in file_hashes.iterrows()
    } if not file_hashes.empty else {}
    commands = [
        "python -m src.preprocessing.validate_raw_data",
        "python -m src.preprocessing.build_windows",
        "python -m src.training.create_splits",
        "python -m src.models.train_baselines",
        "python -m src.models.evaluate_baselines",
        "python -m src.training.build_sequence_dataset",
        "python -m src.models.train_sequence_models",
        "python -m src.models.evaluate_sequence_models",
        "python -m src.models.train_tcn_model",
        "python -m src.models.evaluate_tcn_model",
        "python -m src.analysis.generate_model_comparison",
        "python -m src.analysis.generate_threshold_report",
        "python -m src.realtime.validate_selected_model",
        "python -m src.tracking.generate_tracking_report",
    ]
    lines = [
        "# Reproducibility Checklist",
        "",
        "## Raw Data",
        _checklist_items(TRACKED_RAW_FILES, status_by_path),
        "",
        "## Processed Data",
        _checklist_items(TRACKED_PROCESSED_FILES, status_by_path),
        "",
        "## Splits",
        _checklist_items(
            [
                "data/processed/train_windows.csv",
                "data/processed/validation_windows.csv",
                "data/processed/test_windows.csv",
                "data/processed/split_report.txt",
            ],
            status_by_path,
        ),
        "",
        "## Model Artifacts",
        "- [ ] Confirm model files under models/baselines and models/sequences match the tracked reports.",
        "- [ ] Confirm selected live model files are validated by Step 14.",
        "",
        "## Reports",
        _checklist_items(TRACKED_REPORT_FILES, status_by_path),
        "",
        "## Commands to Reproduce",
        "```bash",
        *commands,
        "```",
        "",
    ]
    (output_dir / "reproducibility_checklist.md").write_text("\n".join(lines), encoding="utf-8")


def _core_file_status() -> str:
    lines = []
    for path in REQUIRED_CORE_FILES:
        exists = Path(path).exists()
        status = "OK" if exists else "MISSING"
        lines.append(f"- {path}: {status}")
    return "\n".join(lines)


def _metrics_text(rows: list[dict]) -> str:
    if not rows:
        return "No metrics available."
    lines = []
    for row in rows:
        lines.append(
            f"- {row.get('split', '')} / {row.get('model_name', '')}: "
            f"F1={row.get('f1', '')}, accuracy={row.get('accuracy', '')}, "
            f"ROC-AUC={row.get('roc_auc', '')}, samples={row.get('number_of_samples', '')}"
        )
    return "\n".join(lines)


def _warnings_text(warnings: list[str]) -> str:
    if not warnings:
        return "None."
    return "\n".join(f"- {warning}" for warning in warnings)


def _markdown_warning_list(warnings: list[str]) -> str:
    if not warnings:
        return "- None."
    return "\n".join(f"- {warning}" for warning in warnings[:12])


def _checklist_items(paths: list[str], status_by_path: dict[str, bool]) -> str:
    lines = []
    for path in paths:
        marker = "x" if status_by_path.get(path, Path(path).exists()) else " "
        lines.append(f"- [{marker}] {path}")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Tracking report generation failed: {exc}")
