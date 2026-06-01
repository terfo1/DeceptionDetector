"""Registry of trained models available to the realtime prototype."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_model_registry() -> dict:
    """Return model registry metadata."""
    return {
        "logistic_regression": {
            "model_type": "baseline",
            "status": "available",
            "role": "candidate",
            "model_path": "models/baselines/logistic_regression.joblib",
            "required_files": [
                "models/baselines/logistic_regression.joblib",
                "models/baselines/scaler.joblib",
                "models/baselines/feature_columns.json",
            ],
            "reason": "Baseline model with wide probability distribution but not selected.",
        },
        "random_forest": {
            "model_type": "baseline",
            "status": "selected",
            "role": "primary",
            "model_path": "models/baselines/random_forest.joblib",
            "required_files": [
                "models/baselines/random_forest.joblib",
                "models/baselines/feature_columns.json",
            ],
            "reason": "Current best prototype model by available model comparison results.",
        },
        "gru": {
            "model_type": "sequence",
            "status": "fallback",
            "role": "secondary",
            "model_path": "models/sequences/gru_model.pt",
            "required_files": [
                "models/sequences/gru_model.pt",
                "models/sequences/sequence_model_config.json",
                "data/processed/sequences/sequence_feature_columns.json",
                "data/processed/sequences/sequence_scaler.joblib",
            ],
            "reason": "Best current neural sequence model but probability range is narrow.",
        },
        "lstm": {
            "model_type": "sequence",
            "status": "disabled",
            "role": "disabled",
            "model_path": "models/sequences/lstm_model.pt",
            "required_files": [
                "models/sequences/lstm_model.pt",
            ],
            "reason": "Disabled because prediction collapse was detected.",
        },
        "causal_tcn": {
            "model_type": "sequence",
            "status": "experimental",
            "role": "experimental",
            "model_path": "models/sequences/tcn_model.pt",
            "required_files": [
                "models/sequences/tcn_model.pt",
                "models/sequences/tcn_model_config.json",
                "data/processed/sequences/sequence_feature_columns.json",
                "data/processed/sequences/sequence_scaler.joblib",
            ],
            "reason": "Architecturally suitable for future real-time inference, but current probability range is narrow.",
        },
    }


def check_required_files(model_name: str) -> dict:
    """Check required files for one registry model."""
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unsupported registry model: {model_name}")
    required_files = registry[model_name]["required_files"]
    existing_files = [path for path in required_files if Path(path).exists()]
    missing_files = [path for path in required_files if not Path(path).exists()]
    return {
        "model_name": model_name,
        "all_files_exist": len(missing_files) == 0,
        "missing_files": missing_files,
        "existing_files": existing_files,
    }


def validate_model_registry() -> pd.DataFrame:
    """Validate all registered model artifacts."""
    rows = []
    for model_name, entry in get_model_registry().items():
        file_status = check_required_files(model_name)
        rows.append(
            {
                "model_name": model_name,
                "model_type": entry["model_type"],
                "status": entry["status"],
                "role": entry["role"],
                "model_path": entry["model_path"],
                "all_files_exist": file_status["all_files_exist"],
                "missing_files": "; ".join(file_status["missing_files"]),
                "reason": entry["reason"],
            }
        )
    return pd.DataFrame(rows)
