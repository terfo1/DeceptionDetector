"""Stateful live inference session for WebSocket streaming."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.realtime.decision_policy import risk_category, smooth_probabilities
from src.realtime.realtime_buffer import RealtimeGazeBuffer
from src.realtime.realtime_config import MIN_VALID_RATIO
from src.realtime.realtime_predictor import RealtimePredictor


class LiveInferenceSession:
    """Manage one streaming inference session."""

    def __init__(
        self,
        session_id: str,
        model_type: str,
        update_interval_seconds: float,
        window_size_seconds: float,
    ):
        self.session_id = session_id
        self.model_type = model_type
        self.update_interval_seconds = float(update_interval_seconds)
        self.window_size_seconds = float(window_size_seconds)
        self.predictor = RealtimePredictor(model_type)
        self.predictor.load_model()
        self.buffer = RealtimeGazeBuffer(window_size_seconds)
        self.current_trial_id: str | None = None
        self.next_prediction_timestamp = 0.0
        self.probabilities: list[float] = []
        self.latencies: list[float] = []
        self.predictions: list[dict] = []
        self.started_at = datetime.now().isoformat(timespec="seconds")

    def start_trial(self, trial_id: str) -> None:
        """Start a new trial and clear rolling state."""
        self.current_trial_id = trial_id
        self.buffer.clear()
        self.next_prediction_timestamp = 0.0
        self.probabilities = []

    def add_sample(self, sample: dict) -> dict | None:
        """Add a sample and return a prediction if update timing is reached."""
        if self.current_trial_id is None:
            self.start_trial(str(sample.get("trial_id") or "live_trial"))
        sample = dict(sample)
        sample["trial_id"] = self.current_trial_id
        self.buffer.add_sample(sample)
        timestamp = float(sample["timestamp"])
        if not self.should_predict(timestamp):
            return None
        return self.predict(timestamp)

    def should_predict(self, timestamp: float) -> bool:
        """Return whether a prediction should be generated at this timestamp."""
        return float(timestamp) + 1e-9 >= self.next_prediction_timestamp

    def predict(self, current_timestamp: float) -> dict:
        """Predict from the current rolling buffer."""
        window_df = self.buffer.get_window(current_timestamp)
        prediction = self.predictor.predict(window_df, current_timestamp)
        self.probabilities.append(prediction["probability"])
        smoothed = smooth_probabilities(self.probabilities, window_size=3)
        category = risk_category(smoothed, prediction["valid_ratio"], MIN_VALID_RATIO)
        self.next_prediction_timestamp += self.update_interval_seconds
        response = {
            "type": "prediction",
            "session_id": self.session_id,
            "trial_id": self.current_trial_id,
            "model_type": self.model_type,
            "timestamp": float(current_timestamp),
            "probability": prediction["probability"],
            "smoothed_probability": smoothed,
            "risk_category": category,
            "valid_ratio": prediction["valid_ratio"],
            "sample_count": prediction["sample_count"],
            "latency_ms": prediction["latency_ms"],
            "status": "ok",
        }
        self.latencies.append(prediction["latency_ms"])
        self.predictions.append(response)
        return response

    def end_trial(self) -> dict:
        """Return summary for the active trial."""
        trial_predictions = [
            prediction
            for prediction in self.predictions
            if prediction.get("trial_id") == self.current_trial_id
        ]
        probabilities = [prediction["probability"] for prediction in trial_predictions]
        return {
            "type": "trial_summary",
            "session_id": self.session_id,
            "trial_id": self.current_trial_id,
            "prediction_count": len(trial_predictions),
            "mean_probability": float(np.mean(probabilities)) if probabilities else None,
            "max_probability": float(np.max(probabilities)) if probabilities else None,
            "status": "ok",
        }

    def end_session(self) -> dict:
        """Return full session summary."""
        probabilities = [prediction["probability"] for prediction in self.predictions]
        latencies = [prediction["latency_ms"] for prediction in self.predictions]
        categories = [prediction["risk_category"] for prediction in self.predictions]
        return {
            "type": "session_summary",
            "session_id": self.session_id,
            "model_type": self.model_type,
            "started_at": self.started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "total_predictions": len(self.predictions),
            "mean_probability": float(np.mean(probabilities)) if probabilities else None,
            "max_probability": float(np.max(probabilities)) if probabilities else None,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
            "max_latency_ms": float(np.max(latencies)) if latencies else None,
            "low_count": categories.count("low"),
            "medium_count": categories.count("medium"),
            "high_count": categories.count("high"),
            "insufficient_data_count": categories.count("insufficient_data"),
            "status": "ok",
        }

    def reset(self) -> None:
        """Reset buffer and in-memory prediction state."""
        self.buffer.clear()
        self.current_trial_id = None
        self.next_prediction_timestamp = 0.0
        self.probabilities = []
        self.latencies = []
        self.predictions = []
