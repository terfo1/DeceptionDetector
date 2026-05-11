"""Create sliding windows and aggregate window-level features."""

import numpy as np
import pandas as pd


def create_windows(
    trials_df: pd.DataFrame,
    gaze_df: pd.DataFrame,
    window_size_seconds: float,
    stride_seconds: float,
    min_valid_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create trial-level sliding windows and aggregated feature rows."""
    windows = []
    feature_rows = []
    window_number = 1

    for _, trial in trials_df.iterrows():
        trial_id = trial["trial_id"]
        trial_samples = (
            gaze_df[gaze_df["trial_id"] == trial_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if trial_samples.empty:
            continue

        max_timestamp = float(trial_samples["timestamp"].max())
        starts = _window_starts(max_timestamp, window_size_seconds, stride_seconds)

        for window_start in starts:
            if max_timestamp <= window_size_seconds:
                window_end = max_timestamp
            else:
                window_end = window_start + window_size_seconds

            mask = (
                (trial_samples["timestamp"] >= window_start)
                & (trial_samples["timestamp"] < window_end)
            )
            if window_start == starts[-1]:
                mask = (
                    (trial_samples["timestamp"] >= window_start)
                    & (trial_samples["timestamp"] <= window_end)
                )
            window_samples = trial_samples[mask]
            if window_samples.empty:
                continue

            window_id = f"W{window_number:06d}"
            window_number += 1

            sample_count = int(len(window_samples))
            valid_ratio = float((window_samples["validity"] == 1).sum() / sample_count)
            is_usable = bool(valid_ratio >= min_valid_ratio and sample_count > 0)

            window_row = {
                "window_id": window_id,
                "trial_id": trial_id,
                "session_id": trial["session_id"],
                "question_text": trial["question_text"],
                "instruction": trial["instruction"],
                "label": int(trial["label"]),
                "answer": trial["answer"],
                "response_time": trial["response_time"],
                "window_start": round(float(window_start), 3),
                "window_end": round(float(window_end), 3),
                "valid_ratio": round(valid_ratio, 4),
                "sample_count": sample_count,
                "is_usable": is_usable,
            }
            windows.append(window_row)

            feature_rows.append(
                {
                    "window_id": window_id,
                    "trial_id": trial_id,
                    "session_id": trial["session_id"],
                    "label": int(trial["label"]),
                    "instruction": trial["instruction"],
                    "window_start": round(float(window_start), 3),
                    "window_end": round(float(window_end), 3),
                    "sample_count": sample_count,
                    "valid_ratio": round(valid_ratio, 4),
                    "is_usable": is_usable,
                    **_aggregate_features(window_samples, valid_ratio),
                }
            )

    return pd.DataFrame(windows), pd.DataFrame(feature_rows)


def _window_starts(max_timestamp, window_size_seconds, stride_seconds):
    if max_timestamp <= window_size_seconds:
        return [0.0]

    starts = []
    current = 0.0
    while current + window_size_seconds <= max_timestamp + 1e-9:
        starts.append(round(current, 6))
        current += stride_seconds

    return starts


def _aggregate_features(window_samples, valid_ratio):
    return {
        "gaze_x_mean": _mean(window_samples["gaze_x"]),
        "gaze_x_std": _std(window_samples["gaze_x"]),
        "gaze_y_mean": _mean(window_samples["gaze_y"]),
        "gaze_y_std": _std(window_samples["gaze_y"]),
        "pupil_left_mean": _mean(window_samples["pupil_left"]),
        "pupil_left_std": _std(window_samples["pupil_left"]),
        "pupil_right_mean": _mean(window_samples["pupil_right"]),
        "pupil_right_std": _std(window_samples["pupil_right"]),
        "pupil_mean_mean": _mean(window_samples["pupil_mean"]),
        "pupil_mean_std": _std(window_samples["pupil_mean"]),
        "blink_rate": _mean(window_samples["blink"]),
        "fixation_rate": _mean(window_samples["fixation"]),
        "saccade_rate": _mean(window_samples["saccade"]),
        "gaze_velocity_mean": _mean(window_samples["gaze_velocity"]),
        "gaze_velocity_std": _std(window_samples["gaze_velocity"]),
        "missing_ratio": round(1.0 - valid_ratio, 4),
    }


def _mean(series):
    value = series.mean(skipna=True)
    return 0.0 if pd.isna(value) else round(float(value), 6)


def _std(series):
    value = series.std(skipna=True, ddof=0)
    return 0.0 if pd.isna(value) else round(float(value), 6)
