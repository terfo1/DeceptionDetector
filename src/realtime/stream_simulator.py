"""Replay recorded gaze samples trial-by-trial."""

from __future__ import annotations

import time

import pandas as pd


class RecordedGazeStreamSimulator:
    """Simulate online gaze sample arrival from recorded raw data."""

    def __init__(self, gaze_df: pd.DataFrame, trials_df: pd.DataFrame):
        self.gaze_df = gaze_df.copy()
        self.trials_df = trials_df.copy()

    def get_available_trials(self) -> list[str]:
        """Return trial IDs that have recorded gaze samples."""
        if "trial_id" not in self.gaze_df.columns:
            return []
        return sorted(self.gaze_df["trial_id"].dropna().astype(str).unique())

    def iter_trial_samples(self, trial_id: str, real_time_sleep: bool = False):
        """Yield samples for one trial ordered by timestamp."""
        trial_samples = (
            self.gaze_df[self.gaze_df["trial_id"].astype(str) == str(trial_id)]
            .copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        previous_timestamp = None
        for _, row in trial_samples.iterrows():
            timestamp = float(pd.to_numeric(row["timestamp"], errors="coerce"))
            if real_time_sleep and previous_timestamp is not None:
                delay = max(0.0, timestamp - previous_timestamp)
                time.sleep(delay)
            previous_timestamp = timestamp
            yield {
                "trial_id": str(trial_id),
                "timestamp": timestamp,
                "sample": row.to_dict(),
            }
