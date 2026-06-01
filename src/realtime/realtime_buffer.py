"""Rolling gaze sample buffer for replayed online inference."""

from __future__ import annotations

import pandas as pd


class RealtimeGazeBuffer:
    """Keep only current-trial samples needed for a rolling time window."""

    def __init__(self, window_size_seconds: float):
        self.window_size_seconds = float(window_size_seconds)
        self._samples: list[dict] = []
        self._trial_id: str | None = None

    def add_sample(self, sample: dict) -> None:
        """Add one current sample and reset if the trial changes."""
        trial_id = str(sample.get("trial_id", ""))
        if self._trial_id is None:
            self._trial_id = trial_id
        elif trial_id != self._trial_id:
            self.clear()
            self._trial_id = trial_id
        self._samples.append(dict(sample))

    def get_window(self, current_timestamp: float) -> pd.DataFrame:
        """Return samples from [current - window_size, current]."""
        if not self._samples:
            return pd.DataFrame()
        window_start = float(current_timestamp) - self.window_size_seconds
        df = pd.DataFrame(self._samples)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        window_df = df[
            (df["timestamp"] >= window_start)
            & (df["timestamp"] <= float(current_timestamp))
        ].copy()
        return window_df.sort_values("timestamp").reset_index(drop=True)

    def clear(self) -> None:
        """Remove all buffered samples."""
        self._samples = []
        self._trial_id = None

    def __len__(self) -> int:
        return len(self._samples)
