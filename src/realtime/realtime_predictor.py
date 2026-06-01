"""Load trained models and score causal rolling gaze windows."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.sequence_models import RecurrentSequenceClassifier
from src.models.tcn_model import CausalTCNClassifier

from .decision_policy import risk_category
from .realtime_config import (
    CONTINUOUS_SEQUENCE_FEATURES,
    MIN_VALID_RATIO,
    MODEL_BASELINE_DIR,
    MODEL_SEQUENCE_DIR,
    SEQUENCE_DATA_DIR,
    SUPPORTED_MODEL_TYPES,
    TARGET_TIME_STEPS,
    WINDOW_SIZE_SECONDS,
)
from .realtime_preprocessor import (
    calculate_valid_ratio,
    prepare_baseline_features,
    prepare_sequence_tensor,
    prepare_window_dataframe,
)


class RealtimePredictor:
    """Prediction wrapper for baseline and neural sequence models."""

    def __init__(self, model_type: str):
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns: list[str] = []
        self.sequence_feature_columns: list[str] = []
        self.continuous_indices: list[int] = []
        self.device = None
        self.loaded = False

    def load_model(self) -> None:
        """Load the selected trained model and required preprocessing artifacts."""
        if self.model_type in {"random_forest", "logistic_regression"}:
            self._load_baseline_model()
        else:
            self._load_sequence_model()
        self.loaded = True

    def predict(self, window_df: pd.DataFrame, current_timestamp: float) -> dict:
        """Predict deception-risk probability for the current causal window."""
        if not self.loaded:
            self.load_model()
        start = time.perf_counter()
        prepared = prepare_window_dataframe(window_df)
        valid_ratio = calculate_valid_ratio(prepared)
        original_length = int(len(prepared))

        if self.model_type in {"random_forest", "logistic_regression"}:
            probability = self._predict_baseline(prepared)
        else:
            probability = self._predict_sequence(
                prepared,
                current_timestamp=current_timestamp,
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "probability": float(probability),
            "valid_ratio": float(valid_ratio),
            "risk_category": risk_category(float(probability), valid_ratio, MIN_VALID_RATIO),
            "latency_ms": float(latency_ms),
            "model_type": self.model_type,
            "sample_count": int(len(prepared)),
            "original_length": original_length,
        }

    def _load_baseline_model(self) -> None:
        model_dir = Path(MODEL_BASELINE_DIR)
        model_file = model_dir / f"{self.model_type}.joblib"
        feature_file = model_dir / "feature_columns.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Required baseline model file is missing: {model_file}")
        if not feature_file.exists():
            raise FileNotFoundError(f"Required baseline feature file is missing: {feature_file}")
        self.model = joblib.load(model_file)
        self.feature_columns = json.loads(feature_file.read_text(encoding="utf-8"))
        if self.model_type == "logistic_regression":
            scaler_path = model_dir / "scaler.joblib"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Required logistic regression scaler is missing: {scaler_path}")
            self.scaler = joblib.load(scaler_path)

    def _load_sequence_model(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch is required for neural realtime simulation models. Install torch and rerun."
            ) from exc

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sequence_features_path = Path(SEQUENCE_DATA_DIR) / "sequence_feature_columns.json"
        sequence_scaler_path = Path(SEQUENCE_DATA_DIR) / "sequence_scaler.joblib"
        if not sequence_features_path.exists():
            raise FileNotFoundError(f"Missing sequence feature config: {sequence_features_path}")
        if not sequence_scaler_path.exists():
            raise FileNotFoundError(f"Missing sequence scaler: {sequence_scaler_path}")

        feature_payload = json.loads(sequence_features_path.read_text(encoding="utf-8"))
        self.sequence_feature_columns = list(feature_payload["all_features"])
        continuous = list(feature_payload.get("continuous_features", CONTINUOUS_SEQUENCE_FEATURES))
        self.continuous_indices = [
            self.sequence_feature_columns.index(feature)
            for feature in continuous
            if feature in self.sequence_feature_columns
        ]
        self.scaler = joblib.load(sequence_scaler_path)

        model_dir = Path(MODEL_SEQUENCE_DIR)
        if self.model_type in {"gru", "lstm"}:
            config_path = model_dir / "sequence_model_config.json"
            model_path = model_dir / f"{self.model_type}_model.pt"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing sequence model config: {config_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.model = RecurrentSequenceClassifier(
                input_size=int(config["input_size"]),
                hidden_size=int(config["hidden_size"]),
                num_layers=int(config["num_layers"]),
                dropout=float(config["dropout"]),
                model_type=self.model_type,
            )
        else:
            config_path = model_dir / "tcn_model_config.json"
            model_path = model_dir / "tcn_model.pt"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing TCN model config: {config_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.model = CausalTCNClassifier(
                input_size=int(config["input_size"]),
                num_channels=list(config["tcn_channels"]),
                kernel_size=int(config["kernel_size"]),
                dropout=float(config["dropout"]),
            )

        if not model_path.exists():
            raise FileNotFoundError(f"Required sequence model file is missing: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _predict_baseline(self, window_df: pd.DataFrame) -> float:
        features = prepare_baseline_features(window_df, self.feature_columns)
        if self.model_type == "logistic_regression":
            features = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features)
        return float(probabilities[0, 1])

    def _predict_sequence(self, window_df: pd.DataFrame, current_timestamp: float) -> float:
        import torch

        window_end = float(current_timestamp)
        window_start = max(0.0, window_end - WINDOW_SIZE_SECONDS)
        X, mask, _ = prepare_sequence_tensor(
            window_df=window_df,
            feature_columns=self.sequence_feature_columns,
            target_time_steps=TARGET_TIME_STEPS,
            window_start=window_start,
            window_end=window_end,
        )
        X = self._apply_sequence_scaler(X, mask)
        with torch.no_grad():
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            logit = self.model(X_tensor, mask_tensor)
            probability = torch.sigmoid(logit).detach().cpu().numpy()[0]
        return float(probability)

    def _apply_sequence_scaler(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        scaled = X.copy()
        real_rows = mask.astype(bool)
        if not real_rows.any():
            return scaled
        continuous_values = scaled[:, :, self.continuous_indices]
        continuous_values[real_rows] = self.scaler.transform(continuous_values[real_rows])
        scaled[:, :, self.continuous_indices] = continuous_values
        scaled[~real_rows] = 0.0
        return scaled
