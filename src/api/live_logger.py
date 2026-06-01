"""CSV logging for live inference API predictions and sessions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("reports/live_inference")

PREDICTION_COLUMNS = [
    "created_at",
    "session_id",
    "trial_id",
    "model_type",
    "timestamp",
    "probability",
    "smoothed_probability",
    "risk_category",
    "valid_ratio",
    "sample_count",
    "latency_ms",
    "status",
]

SESSION_COLUMNS = [
    "created_at",
    "session_id",
    "model_type",
    "started_at",
    "ended_at",
    "total_predictions",
    "mean_probability",
    "max_probability",
    "mean_latency_ms",
    "max_latency_ms",
    "low_count",
    "medium_count",
    "high_count",
    "insufficient_data_count",
]


class LiveInferenceLogger:
    """Append-only CSV logger for API inference activity."""

    def __init__(self, output_dir: str | Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.output_dir / "live_predictions.csv"
        self.sessions_path = self.output_dir / "live_sessions.csv"
        self.report_path = self.output_dir / "api_run_report.txt"
        self._ensure_csv(self.predictions_path, PREDICTION_COLUMNS)
        self._ensure_csv(self.sessions_path, SESSION_COLUMNS)

    def log_prediction(self, prediction: dict) -> None:
        """Append one prediction row."""
        row = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": prediction.get("session_id", ""),
            "trial_id": prediction.get("trial_id", ""),
            "model_type": prediction.get("model_type", ""),
            "timestamp": prediction.get("timestamp", ""),
            "probability": prediction.get("probability", ""),
            "smoothed_probability": prediction.get("smoothed_probability", ""),
            "risk_category": prediction.get("risk_category", ""),
            "valid_ratio": prediction.get("valid_ratio", ""),
            "sample_count": prediction.get("sample_count", ""),
            "latency_ms": prediction.get("latency_ms", ""),
            "status": prediction.get("status", ""),
        }
        self._append_row(self.predictions_path, row, PREDICTION_COLUMNS)

    def log_session_summary(self, summary: dict) -> None:
        """Append one session summary row."""
        row = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": summary.get("session_id", ""),
            "model_type": summary.get("model_type", ""),
            "started_at": summary.get("started_at", ""),
            "ended_at": summary.get("ended_at", ""),
            "total_predictions": summary.get("total_predictions", 0),
            "mean_probability": summary.get("mean_probability", ""),
            "max_probability": summary.get("max_probability", ""),
            "mean_latency_ms": summary.get("mean_latency_ms", ""),
            "max_latency_ms": summary.get("max_latency_ms", ""),
            "low_count": summary.get("low_count", 0),
            "medium_count": summary.get("medium_count", 0),
            "high_count": summary.get("high_count", 0),
            "insufficient_data_count": summary.get("insufficient_data_count", 0),
        }
        self._append_row(self.sessions_path, row, SESSION_COLUMNS)

    def write_api_run_report(self, info: dict) -> None:
        """Append an API run note."""
        lines = [
            "Live Inference API Run",
            f"created_at: {datetime.now().isoformat(timespec='seconds')}",
            *[f"{key}: {value}" for key, value in info.items()],
            "",
        ]
        with self.report_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))

    def _ensure_csv(self, path: Path, columns: list[str]) -> None:
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)

    def _append_row(self, path: Path, row: dict, columns: list[str]) -> None:
        pd.DataFrame([row], columns=columns).to_csv(
            path,
            mode="a",
            header=False,
            index=False,
        )
