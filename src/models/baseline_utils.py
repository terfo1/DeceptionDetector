"""Shared utilities for baseline model training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .baseline_config import (
    METADATA_COLUMNS,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    TEST_FEATURES_FILE,
    TRAIN_FEATURES_FILE,
    VALIDATION_FEATURES_FILE,
)


def load_split_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test aggregated feature split files."""
    processed_dir = Path(PROCESSED_DATA_DIR)
    return (
        _read_required_csv(processed_dir / TRAIN_FEATURES_FILE),
        _read_required_csv(processed_dir / VALIDATION_FEATURES_FILE),
        _read_required_csv(processed_dir / TEST_FEATURES_FILE),
    )


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[str]:
    """Validate split files and return non-fatal warnings."""
    warnings: list[str] = []
    split_frames = {"train": train_df, "validation": val_df, "test": test_df}

    if train_df.empty:
        raise ValueError("Train split is empty. Baseline models cannot be trained.")
    if val_df.empty:
        warnings.append("Validation split is empty. Validation evaluation will be skipped.")
    if test_df.empty:
        warnings.append("Test split is empty. Test evaluation can be skipped later.")

    for split_name, df in split_frames.items():
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"{split_name} split is missing required label column.")
        if "participant_id" not in df.columns:
            raise ValueError(f"{split_name} split is missing required participant_id column.")

        if not df.empty:
            labels = set(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().astype(int))
            invalid_labels = sorted(labels - {0, 1})
            if invalid_labels:
                raise ValueError(
                    f"{split_name} split contains labels outside 0/1: {invalid_labels}"
                )
            if df[TARGET_COLUMN].isna().any():
                raise ValueError(f"{split_name} split contains missing label values.")

    overlaps = [
        ("train", "validation", _participant_ids(train_df) & _participant_ids(val_df)),
        ("train", "test", _participant_ids(train_df) & _participant_ids(test_df)),
        ("validation", "test", _participant_ids(val_df) & _participant_ids(test_df)),
    ]
    for left, right, overlap in overlaps:
        if overlap:
            raise ValueError(
                f"Participant leakage between {left} and {right}: "
                + ", ".join(sorted(overlap))
            )

    if not get_feature_columns(train_df):
        raise ValueError("No numeric feature columns are available for training.")

    if train_df[TARGET_COLUMN].nunique(dropna=True) < 2:
        warnings.append("Training labels contain only one class; model training may fail.")

    return warnings


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns after excluding metadata and invalid columns."""
    feature_columns: list[str] = []
    metadata = set(METADATA_COLUMNS)

    for column in df.select_dtypes(include=[np.number]).columns:
        if column in metadata or column == TARGET_COLUMN:
            continue

        values = df[column]
        if values.isna().all():
            continue
        if np.isinf(values.to_numpy(dtype=float, copy=True)).any():
            continue

        feature_columns.append(column)

    return feature_columns


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Select model features and target labels without scaling."""
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Feature columns are missing from split data: " + ", ".join(missing_columns)
        )
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    X = df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int)
    return X, y


def calculate_metrics(y_true: Any, y_pred: Any, y_proba: Any) -> dict[str, float]:
    """Calculate binary classification metrics for deceptive-response risk."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_proba_array = np.asarray(y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()
    has_two_classes = len(np.unique(y_true_array)) == 2

    return {
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "balanced_accuracy": balanced_accuracy_score(y_true_array, y_pred_array),
        "precision": precision_score(y_true_array, y_pred_array, zero_division=0),
        "recall": recall_score(y_true_array, y_pred_array, zero_division=0),
        "f1": f1_score(y_true_array, y_pred_array, zero_division=0),
        "roc_auc": roc_auc_score(y_true_array, y_proba_array) if has_two_classes else np.nan,
        "average_precision": (
            average_precision_score(y_true_array, y_proba_array) if has_two_classes else np.nan
        ),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON data with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: str | Path) -> Any:
    """Load JSON data."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required split file is missing: {path}")
    return pd.read_csv(path)


def _participant_ids(df: pd.DataFrame) -> set[str]:
    if df.empty or "participant_id" not in df.columns:
        return set()
    return set(df["participant_id"].dropna().astype(str))
