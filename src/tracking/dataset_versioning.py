"""Dataset manifest creation and version history tracking."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .file_hashing import collect_file_metadata
from .tracking_config import (
    DATASET_VERSION_PREFIX,
    TRACKED_PROCESSED_FILES,
    TRACKED_RAW_FILES,
    TRACKING_OUTPUT_DIR,
)


def get_next_dataset_version(output_dir: str) -> str:
    """Return the next dataset_vNNN identifier."""
    path = Path(output_dir) / "dataset_versions.csv"
    if not path.exists():
        return f"{DATASET_VERSION_PREFIX}001"
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return f"{DATASET_VERSION_PREFIX}001"
    if df.empty or "dataset_version" not in df.columns:
        return f"{DATASET_VERSION_PREFIX}001"
    numbers = []
    for value in df["dataset_version"].dropna().astype(str):
        if value.startswith(DATASET_VERSION_PREFIX):
            try:
                numbers.append(int(value.replace(DATASET_VERSION_PREFIX, "")))
            except ValueError:
                continue
    return f"{DATASET_VERSION_PREFIX}{(max(numbers) + 1 if numbers else 1):03d}"


def collect_dataset_statistics() -> dict:
    """Collect raw, split, and sequence dataset counts."""
    warnings: list[str] = []
    participants = _read_csv("data/raw/participants.csv", warnings)
    sessions = _read_csv("data/raw/sessions.csv", warnings)
    trials = _read_csv("data/raw/trials.csv", warnings)
    gaze = _read_csv("data/raw/gaze_samples.csv", warnings)
    train = _read_csv("data/processed/train_windows.csv", warnings)
    validation = _read_csv("data/processed/validation_windows.csv", warnings)
    test = _read_csv("data/processed/test_windows.csv", warnings)

    stats = {
        "participant_count": len(participants),
        "session_count": len(sessions),
        "trial_count": len(trials),
        "gaze_sample_count": len(gaze),
        "truth_trial_count": int((pd.to_numeric(trials.get("label", pd.Series(dtype=float)), errors="coerce") == 0).sum()) if not trials.empty else 0,
        "lie_trial_count": int((pd.to_numeric(trials.get("label", pd.Series(dtype=float)), errors="coerce") == 1).sum()) if not trials.empty else 0,
        "train_window_count": len(train),
        "validation_window_count": len(validation),
        "test_window_count": len(test),
        "total_window_count": len(train) + len(validation) + len(test),
        "train_participant_count": _participant_count(train),
        "validation_participant_count": _participant_count(validation),
        "test_participant_count": _participant_count(test),
        "total_split_participants": len(
            _participants(train) | _participants(validation) | _participants(test)
        ),
        "sequence_train_count": _npz_count("data/processed/sequences/train_sequences.npz", warnings),
        "sequence_validation_count": _npz_count("data/processed/sequences/validation_sequences.npz", warnings),
        "sequence_test_count": _npz_count("data/processed/sequences/test_sequences.npz", warnings),
        "warnings": warnings,
    }
    return stats


def create_dataset_manifest() -> dict:
    """Create the latest dataset manifest."""
    output_dir = Path(TRACKING_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_version = get_next_dataset_version(TRACKING_OUTPUT_DIR)
    stats = collect_dataset_statistics()
    raw_hashes = collect_file_metadata(TRACKED_RAW_FILES)
    processed_hashes = collect_file_metadata(TRACKED_PROCESSED_FILES)
    warnings = list(stats.pop("warnings", []))
    warnings.extend(_missing_warnings(raw_hashes, "raw"))
    warnings.extend(_missing_warnings(processed_hashes, "processed"))
    if stats["participant_count"] < 10:
        warnings.append("Participant count is below the preferred reliability threshold of 10.")
    return {
        "dataset_version": dataset_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_statistics": stats,
        "raw_file_hashes": raw_hashes.to_dict(orient="records"),
        "processed_file_hashes": processed_hashes.to_dict(orient="records"),
        "warnings": warnings,
        "note": "This dataset is used for controlled deception-risk estimation research and is not evidence of universal lie detection.",
    }


def save_dataset_manifest(manifest: dict, output_path: str) -> None:
    """Save dataset manifest JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def append_dataset_version(manifest: dict) -> None:
    """Append dataset version row."""
    path = Path(TRACKING_OUTPUT_DIR) / "dataset_versions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "dataset_version",
        "created_at",
        "participant_count",
        "session_count",
        "trial_count",
        "gaze_sample_count",
        "total_window_count",
        "train_window_count",
        "validation_window_count",
        "test_window_count",
        "total_split_participants",
        "manifest_path",
        "warning_count",
    ]
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(header)
    stats = manifest["dataset_statistics"]
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writerow(
            {
                "dataset_version": manifest["dataset_version"],
                "created_at": manifest["created_at"],
                "participant_count": stats.get("participant_count", 0),
                "session_count": stats.get("session_count", 0),
                "trial_count": stats.get("trial_count", 0),
                "gaze_sample_count": stats.get("gaze_sample_count", 0),
                "total_window_count": stats.get("total_window_count", 0),
                "train_window_count": stats.get("train_window_count", 0),
                "validation_window_count": stats.get("validation_window_count", 0),
                "test_window_count": stats.get("test_window_count", 0),
                "total_split_participants": stats.get("total_split_participants", 0),
                "manifest_path": str(Path(TRACKING_OUTPUT_DIR) / "latest_dataset_manifest.json"),
                "warning_count": len(manifest.get("warnings", [])),
            }
        )


def _read_csv(path: str, warnings: list[str]) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        warnings.append(f"Missing file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        warnings.append(f"Empty file: {path}")
        return pd.DataFrame()


def _npz_count(path: str, warnings: list[str]) -> int:
    file_path = Path(path)
    if not file_path.exists():
        warnings.append(f"Missing sequence file: {path}")
        return 0
    try:
        with np.load(file_path, allow_pickle=True) as data:
            return int(data["X"].shape[0])
    except Exception as exc:
        warnings.append(f"Could not read sequence file {path}: {exc}")
        return 0


def _participant_count(df: pd.DataFrame) -> int:
    return len(_participants(df))


def _participants(df: pd.DataFrame) -> set[str]:
    if df.empty or "participant_id" not in df.columns:
        return set()
    return set(df["participant_id"].dropna().astype(str))


def _missing_warnings(df: pd.DataFrame, label: str) -> list[str]:
    return [f"Missing tracked {label} file: {row['path']}" for _, row in df[df["exists"] == False].iterrows()]
