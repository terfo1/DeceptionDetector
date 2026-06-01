"""Causal preprocessing for replayed real-time gaze windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .realtime_config import MISSING_VALUE_FILL


BINARY_COLUMNS = ["blink", "fixation", "saccade", "validity"]
NUMERIC_COLUMNS = [
    "sample_id",
    "timestamp",
    "gaze_x",
    "gaze_y",
    "pupil_left",
    "pupil_right",
    *BINARY_COLUMNS,
]


def clean_realtime_sample(sample: dict) -> dict:
    """Clean one sample using only its current values."""
    cleaned = dict(sample)
    for column in NUMERIC_COLUMNS:
        cleaned[column] = _to_float(cleaned.get(column, np.nan))

    validity = cleaned.get("validity", 1.0)
    gaze_x = cleaned.get("gaze_x", np.nan)
    gaze_y = cleaned.get("gaze_y", np.nan)
    pupil_left = cleaned.get("pupil_left", np.nan)
    pupil_right = cleaned.get("pupil_right", np.nan)

    if pd.notna(gaze_x) and (gaze_x < 0 or gaze_x > 1):
        validity = 0
    if pd.notna(gaze_y) and (gaze_y < 0 or gaze_y > 1):
        validity = 0
    if pd.notna(pupil_left) and pupil_left <= 0:
        validity = 0
    if pd.notna(pupil_right) and pupil_right <= 0:
        validity = 0

    for column in BINARY_COLUMNS:
        value = validity if column == "validity" else cleaned.get(column, 0)
        cleaned[column] = int(np.clip(round(_to_float(value, default=0.0)), 0, 1))

    cleaned["pupil_mean"] = np.nanmean([pupil_left, pupil_right])
    if pd.isna(cleaned["pupil_mean"]):
        cleaned["pupil_mean"] = np.nan
    return cleaned


def compute_realtime_gaze_velocity(window_df: pd.DataFrame) -> pd.DataFrame:
    """Compute gaze velocity from current-window samples only."""
    if window_df.empty:
        result = window_df.copy()
        result["gaze_velocity"] = []
        return result

    result = window_df.sort_values("timestamp").reset_index(drop=True).copy()
    previous = result[["timestamp", "gaze_x", "gaze_y"]].shift(1)
    dt = result["timestamp"] - previous["timestamp"]
    dx = result["gaze_x"] - previous["gaze_x"]
    dy = result["gaze_y"] - previous["gaze_y"]
    velocity = np.sqrt((dx ** 2) + (dy ** 2)) / dt
    velocity = velocity.replace([np.inf, -np.inf], 0).fillna(0)
    result["gaze_velocity"] = velocity.where(dt > 0, 0)
    return result


def prepare_window_dataframe(window_samples: pd.DataFrame) -> pd.DataFrame:
    """Clean a rolling window and add current-window derived features."""
    if window_samples.empty:
        return pd.DataFrame()
    rows = [clean_realtime_sample(row.to_dict()) for _, row in window_samples.iterrows()]
    cleaned = pd.DataFrame(rows)
    for column in NUMERIC_COLUMNS + ["pupil_mean"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return compute_realtime_gaze_velocity(cleaned)


def calculate_valid_ratio(window_df: pd.DataFrame) -> float:
    """Calculate valid sample ratio for a window."""
    if window_df.empty or "validity" not in window_df.columns:
        return 0.0
    validity = pd.to_numeric(window_df["validity"], errors="coerce").fillna(0)
    return float((validity == 1).sum() / len(window_df)) if len(window_df) else 0.0


def prepare_baseline_features(window_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Create one aggregated feature row for baseline models."""
    valid_ratio = calculate_valid_ratio(window_df)
    features = {
        "gaze_x_mean": _mean(window_df.get("gaze_x", pd.Series(dtype=float))),
        "gaze_x_std": _std(window_df.get("gaze_x", pd.Series(dtype=float))),
        "gaze_y_mean": _mean(window_df.get("gaze_y", pd.Series(dtype=float))),
        "gaze_y_std": _std(window_df.get("gaze_y", pd.Series(dtype=float))),
        "pupil_left_mean": _mean(window_df.get("pupil_left", pd.Series(dtype=float))),
        "pupil_left_std": _std(window_df.get("pupil_left", pd.Series(dtype=float))),
        "pupil_right_mean": _mean(window_df.get("pupil_right", pd.Series(dtype=float))),
        "pupil_right_std": _std(window_df.get("pupil_right", pd.Series(dtype=float))),
        "pupil_mean_mean": _mean(window_df.get("pupil_mean", pd.Series(dtype=float))),
        "pupil_mean_std": _std(window_df.get("pupil_mean", pd.Series(dtype=float))),
        "blink_rate": _mean(window_df.get("blink", pd.Series(dtype=float))),
        "fixation_rate": _mean(window_df.get("fixation", pd.Series(dtype=float))),
        "saccade_rate": _mean(window_df.get("saccade", pd.Series(dtype=float))),
        "gaze_velocity_mean": _mean(window_df.get("gaze_velocity", pd.Series(dtype=float))),
        "gaze_velocity_std": _std(window_df.get("gaze_velocity", pd.Series(dtype=float))),
        "missing_ratio": round(1.0 - valid_ratio, 4),
    }
    row = {column: features.get(column, MISSING_VALUE_FILL) for column in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    return df.replace([np.inf, -np.inf], np.nan).fillna(MISSING_VALUE_FILL)


def prepare_sequence_tensor(
    window_df: pd.DataFrame,
    feature_columns: list[str],
    target_time_steps: int,
    window_start: float,
    window_end: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create a padded sequence tensor from one causal rolling window."""
    feature_count = len(feature_columns)
    X = np.full((1, target_time_steps, feature_count), MISSING_VALUE_FILL, dtype=np.float32)
    mask = np.zeros((1, target_time_steps), dtype=np.float32)
    if window_df.empty:
        return X, mask, 0

    samples = window_df.sort_values("timestamp").copy()
    denominator = float(window_end) - float(window_start)
    samples["timestamp_norm"] = (
        (samples["timestamp"] - float(window_start)) / denominator
        if denominator > 0
        else MISSING_VALUE_FILL
    )
    for column in feature_columns:
        if column not in samples.columns:
            samples[column] = MISSING_VALUE_FILL
    values = samples[feature_columns].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).fillna(MISSING_VALUE_FILL)
    original_length = int(len(values))
    usable_length = min(original_length, target_time_steps)
    if usable_length > 0:
        X[0, :usable_length, :] = values.iloc[:usable_length].to_numpy(dtype=np.float32)
        mask[0, :usable_length] = 1.0
    return X, mask, original_length


def _to_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").mean(skipna=True)
    return 0.0 if pd.isna(value) else round(float(value), 6)


def _std(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").std(skipna=True, ddof=0)
    return 0.0 if pd.isna(value) else round(float(value), 6)
