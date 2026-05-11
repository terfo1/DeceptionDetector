"""Load and clean raw gaze samples for preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_data(raw_dir: str):
    """Load participants, sessions, trials, and gaze samples from raw CSV files."""
    raw_path = Path(raw_dir)
    participants_df = pd.read_csv(raw_path / "participants.csv")
    sessions_df = pd.read_csv(raw_path / "sessions.csv")
    trials_df = pd.read_csv(raw_path / "trials.csv")
    gaze_df = pd.read_csv(raw_path / "gaze_samples.csv")
    return participants_df, sessions_df, trials_df, gaze_df


def clean_gaze_samples(gaze_df: pd.DataFrame) -> pd.DataFrame:
    """Clean gaze samples using only current and past samples within each trial."""
    cleaned = gaze_df.copy()

    numeric_columns = [
        "sample_id",
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
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.sort_values(["trial_id", "timestamp"]).reset_index(drop=True)

    invalid_gaze = (
        cleaned["gaze_x"].notna()
        & ((cleaned["gaze_x"] < 0) | (cleaned["gaze_x"] > 1))
    ) | (
        cleaned["gaze_y"].notna()
        & ((cleaned["gaze_y"] < 0) | (cleaned["gaze_y"] > 1))
    )
    invalid_pupil = (
        cleaned["pupil_left"].notna()
        & (cleaned["pupil_left"] <= 0)
    ) | (
        cleaned["pupil_right"].notna()
        & (cleaned["pupil_right"] <= 0)
    )

    cleaned.loc[invalid_gaze | invalid_pupil, "validity"] = 0

    for column in ["blink", "fixation", "saccade", "validity"]:
        cleaned[column] = (
            cleaned[column]
            .fillna(0)
            .round()
            .clip(lower=0, upper=1)
            .astype(int)
        )

    cleaned["pupil_mean"] = cleaned[["pupil_left", "pupil_right"]].mean(axis=1)
    cleaned["gaze_velocity"] = _calculate_gaze_velocity(cleaned)

    return cleaned


def _calculate_gaze_velocity(gaze_df: pd.DataFrame) -> pd.Series:
    previous = gaze_df.groupby("trial_id")[["timestamp", "gaze_x", "gaze_y"]].shift(1)
    dt = gaze_df["timestamp"] - previous["timestamp"]
    dx = gaze_df["gaze_x"] - previous["gaze_x"]
    dy = gaze_df["gaze_y"] - previous["gaze_y"]

    distance = np.sqrt((dx ** 2) + (dy ** 2))
    velocity = distance / dt
    velocity = velocity.replace([np.inf, -np.inf], 0).fillna(0)
    velocity = velocity.where(dt > 0, 0)
    return velocity
