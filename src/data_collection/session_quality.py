"""Session quality calculation and append-only quality logging."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


SESSION_QUALITY_HEADER = [
    "session_id",
    "participant_id",
    "calibration_status",
    "calibration_quality_score",
    "baseline_completed",
    "baseline_duration_seconds",
    "total_trials",
    "completed_trials",
    "total_gaze_samples",
    "valid_sample_count",
    "invalid_sample_count",
    "valid_ratio",
    "average_samples_per_trial",
    "quality_flag",
    "notes",
    "created_at",
]


def calculate_session_quality(session_id: str, raw_dir: str = "data/raw") -> dict:
    """Calculate session quality from raw sessions, trials, and gaze samples."""
    raw_path = Path(raw_dir)
    sessions = _read_csv(raw_path / "sessions.csv")
    trials = _read_csv(raw_path / "trials.csv")
    gaze = _read_csv(raw_path / "gaze_samples.csv")
    session_metadata = _read_optional_csv(raw_path / "session_metadata.csv")

    if sessions.empty or "session_id" not in sessions.columns:
        raise ValueError("sessions.csv is missing or has no session_id column.")
    session_rows = sessions[sessions["session_id"].astype(str) == str(session_id)]
    if session_rows.empty:
        raise ValueError(f"session_id not found in sessions.csv: {session_id}")
    session_row = session_rows.iloc[-1]
    participant_id = session_row.get("participant_id", "")
    calibration_status = _safe_value(session_row.get("calibration_quality", ""), default="not_started")
    baseline_duration = 0.0
    notes = ""
    if not session_metadata.empty and "session_id" in session_metadata.columns:
        metadata_rows = session_metadata[session_metadata["session_id"].astype(str) == str(session_id)]
        if not metadata_rows.empty:
            metadata = metadata_rows.iloc[-1]
            calibration_status = metadata.get("calibration_status", calibration_status)
            baseline_value = pd.to_numeric(metadata.get("baseline_duration_seconds", 0), errors="coerce")
            baseline_duration = 0.0 if pd.isna(baseline_value) else float(baseline_value)
            notes = str(metadata.get("operator_notes", ""))

    session_trials = trials[trials["session_id"].astype(str) == str(session_id)].copy() if not trials.empty else pd.DataFrame()
    trial_ids = set(session_trials["trial_id"].astype(str)) if "trial_id" in session_trials.columns else set()
    session_gaze = gaze[gaze["trial_id"].astype(str).isin(trial_ids)].copy() if trial_ids and not gaze.empty else pd.DataFrame()

    total_trials = int(len(session_trials))
    completed_trials = int((session_trials["answer"].astype(str) != "timeout").sum()) if "answer" in session_trials.columns else total_trials
    total_gaze_samples = int(len(session_gaze))
    valid_sample_count = int((pd.to_numeric(session_gaze.get("validity", pd.Series(dtype=float)), errors="coerce") == 1).sum()) if total_gaze_samples else 0
    invalid_sample_count = total_gaze_samples - valid_sample_count
    valid_ratio = valid_sample_count / total_gaze_samples if total_gaze_samples else 0.0
    average_samples_per_trial = total_gaze_samples / total_trials if total_trials else 0.0
    quality_flag = classify_quality(valid_ratio, completed_trials, total_trials)

    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "calibration_status": calibration_status,
        "calibration_quality_score": "",
        "baseline_completed": baseline_duration > 0,
        "baseline_duration_seconds": baseline_duration,
        "total_trials": total_trials,
        "completed_trials": completed_trials,
        "total_gaze_samples": total_gaze_samples,
        "valid_sample_count": valid_sample_count,
        "invalid_sample_count": invalid_sample_count,
        "valid_ratio": round(valid_ratio, 4),
        "average_samples_per_trial": round(average_samples_per_trial, 2),
        "quality_flag": quality_flag,
        "notes": notes,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_session_quality(
    row: dict,
    path: str = "data/raw/session_quality.csv",
) -> None:
    """Append one session quality row."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_file(output_path, SESSION_QUALITY_HEADER)
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SESSION_QUALITY_HEADER)
        writer.writerow({column: row.get(column, "") for column in SESSION_QUALITY_HEADER})


def classify_quality(valid_ratio: float, completed_trials: int, expected_trials: int) -> str:
    """Classify session quality from validity and trial completion."""
    if valid_ratio >= 0.85 and completed_trials >= expected_trials and completed_trials > 0:
        return "good"
    if valid_ratio >= 0.70 and completed_trials > 0:
        return "warning"
    return "poor"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required raw file is missing: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _ensure_file(path: Path, header: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(header)


def _safe_value(value, default: str) -> str:
    value = str(value).strip()
    return value if value else default
