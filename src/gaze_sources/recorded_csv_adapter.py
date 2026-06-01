"""Recorded CSV gaze source adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import GazeSourceAdapter


REQUIRED_COLUMNS = [
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


class RecordedCsvGazeAdapter(GazeSourceAdapter):
    """Stream gaze samples from recorded raw CSV files."""

    def __init__(
        self,
        gaze_csv_path: str = "data/raw/gaze_samples.csv",
        trials_csv_path: str = "data/raw/trials.csv",
        trial_id: str | None = None,
        max_trials: int | None = None,
        loop: bool = False,
    ):
        self.gaze_csv_path = gaze_csv_path
        self.trials_csv_path = trials_csv_path
        self.trial_id = trial_id
        self.max_trials = max_trials
        self.loop = loop
        self.running = False
        self.samples_df = pd.DataFrame()
        self.index = 0
        self.trial_count = 0

    def start(self) -> None:
        self.samples_df = self._load_samples()
        self.index = 0
        self.running = True

    def stop(self) -> None:
        self.running = False

    def read_sample(self) -> dict | None:
        if not self.running or self.samples_df.empty:
            return None
        if self.index >= len(self.samples_df):
            if self.loop:
                self.index = 0
            else:
                self.stop()
                return None
        row = self.samples_df.iloc[self.index]
        self.index += 1
        return {
            "trial_id": str(row["trial_id"]),
            "timestamp": float(row["timestamp"]),
            "gaze_x": float(row["gaze_x"]),
            "gaze_y": float(row["gaze_y"]),
            "pupil_left": float(row["pupil_left"]),
            "pupil_right": float(row["pupil_right"]),
            "blink": int(row["blink"]),
            "fixation": int(row["fixation"]),
            "saccade": int(row["saccade"]),
            "validity": int(row["validity"]),
        }

    def is_running(self) -> bool:
        return self.running

    def get_source_name(self) -> str:
        return "recorded_csv"

    def get_metadata(self) -> dict:
        return {
            "source": "recorded_csv",
            "gaze_csv_path": self.gaze_csv_path,
            "trials_csv_path": self.trials_csv_path,
            "trial_count": self.trial_count,
            "sample_count": int(len(self.samples_df)),
        }

    def _load_samples(self) -> pd.DataFrame:
        gaze_path = Path(self.gaze_csv_path)
        trials_path = Path(self.trials_csv_path)
        if not gaze_path.exists():
            raise FileNotFoundError(f"Missing gaze CSV file: {gaze_path}")
        if not trials_path.exists():
            raise FileNotFoundError(f"Missing trials CSV file: {trials_path}")
        gaze_df = pd.read_csv(gaze_path)
        trials_df = pd.read_csv(trials_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in gaze_df.columns]
        if missing:
            raise ValueError("gaze_samples.csv is missing columns: " + ", ".join(missing))

        trial_ids = trials_df["trial_id"].dropna().astype(str).tolist()
        if self.trial_id is not None:
            if str(self.trial_id) not in trial_ids:
                raise ValueError(f"trial_id not found in trials.csv: {self.trial_id}")
            trial_ids = [str(self.trial_id)]
        elif self.max_trials is not None:
            trial_ids = trial_ids[: max(0, int(self.max_trials))]
        self.trial_count = len(trial_ids)

        filtered = gaze_df[gaze_df["trial_id"].astype(str).isin(trial_ids)].copy()
        for column in REQUIRED_COLUMNS:
            if column != "trial_id":
                filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered.dropna(subset=["timestamp"])
        return filtered.sort_values(["trial_id", "timestamp"]).reset_index(drop=True)
