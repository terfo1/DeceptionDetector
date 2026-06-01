"""Utilities for the WebSocket live test client."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GAZE_REQUIRED_COLUMNS = [
    "trial_id",
    "timestamp",
    "gaze_x",
    "gaze_y",
    "pupil_left",
    "pupil_right",
    "blink",
    "fixation",
    "saccade",
    "validity",
]

TRIAL_REQUIRED_COLUMNS = [
    "trial_id",
    "session_id",
    "question_text",
    "instruction",
    "label",
    "answer",
]


def load_raw_stream_data(raw_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate raw gaze samples and trials for replay."""
    raw_path = Path(raw_dir)
    gaze_path = raw_path / "gaze_samples.csv"
    trials_path = raw_path / "trials.csv"
    if not gaze_path.exists():
        raise FileNotFoundError(f"Missing raw gaze file: {gaze_path}")
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing trials file: {trials_path}")

    gaze_df = pd.read_csv(gaze_path)
    trials_df = pd.read_csv(trials_path)
    _require_columns(gaze_df, GAZE_REQUIRED_COLUMNS, "gaze_samples.csv")
    _require_columns(trials_df, TRIAL_REQUIRED_COLUMNS, "trials.csv")

    gaze_df["timestamp"] = pd.to_numeric(gaze_df["timestamp"], errors="coerce")
    if gaze_df["timestamp"].isna().any():
        raise ValueError("gaze_samples.csv contains invalid timestamp values.")
    return gaze_df, trials_df


def get_trial_ids(
    trials_df: pd.DataFrame,
    max_trials: int | None = None,
    trial_id: str | None = None,
) -> list[str]:
    """Return requested trial IDs in trial file order."""
    trial_ids = trials_df["trial_id"].dropna().astype(str).tolist()
    if trial_id is not None:
        if str(trial_id) not in trial_ids:
            raise ValueError(f"trial_id not found in trials.csv: {trial_id}")
        return [str(trial_id)]
    if max_trials is not None:
        return trial_ids[: max(0, int(max_trials))]
    return trial_ids


def build_sample_message(row: pd.Series) -> dict:
    """Build one WebSocket sample message from a raw gaze row."""
    return {
        "type": "sample",
        "data": {
            "timestamp": float(row["timestamp"]),
            "gaze_x": float(row["gaze_x"]),
            "gaze_y": float(row["gaze_y"]),
            "pupil_left": float(row["pupil_left"]),
            "pupil_right": float(row["pupil_right"]),
            "blink": int(row["blink"]),
            "fixation": int(row["fixation"]),
            "saccade": int(row["saccade"]),
            "validity": int(row["validity"]),
        },
    }


def summarize_predictions(predictions_df: pd.DataFrame) -> dict:
    """Summarize received prediction rows."""
    if predictions_df.empty:
        return {
            "total_predictions": 0,
            "mean_probability": np.nan,
            "max_probability": np.nan,
            "min_probability": np.nan,
            "mean_latency_ms": np.nan,
            "max_latency_ms": np.nan,
            "low_count": 0,
            "medium_count": 0,
            "high_count": 0,
            "insufficient_data_count": 0,
        }
    categories = predictions_df["risk_category"].value_counts().to_dict()
    return {
        "total_predictions": int(len(predictions_df)),
        "mean_probability": float(predictions_df["probability"].mean()),
        "max_probability": float(predictions_df["probability"].max()),
        "min_probability": float(predictions_df["probability"].min()),
        "mean_latency_ms": float(predictions_df["latency_ms"].mean()),
        "max_latency_ms": float(predictions_df["latency_ms"].max()),
        "low_count": int(categories.get("low", 0)),
        "medium_count": int(categories.get("medium", 0)),
        "high_count": int(categories.get("high", 0)),
        "insufficient_data_count": int(categories.get("insufficient_data", 0)),
    }


def write_csv_with_header(path, rows, columns) -> None:
    """Write rows to CSV with stable headers."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def _require_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
