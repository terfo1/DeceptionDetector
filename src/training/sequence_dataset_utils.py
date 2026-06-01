"""Utilities for building fixed-length neural sequence datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.cleaning import clean_gaze_samples

from .sequence_config import (
    MISSING_VALUE_FILL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_TIME_STEPS,
)


GAZE_FILE = "gaze_samples.csv"
TRAIN_WINDOWS_FILE = "train_windows.csv"
VALIDATION_WINDOWS_FILE = "validation_windows.csv"
TEST_WINDOWS_FILE = "test_windows.csv"

WINDOW_REQUIRED_COLUMNS = [
    "window_id",
    "trial_id",
    "session_id",
    "participant_id",
    "window_start",
    "window_end",
    "label",
    "instruction",
    "valid_ratio",
    "sample_count",
    "is_usable",
]

GAZE_REQUIRED_COLUMNS = [
    "sample_id",
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

METADATA_COLUMNS = [
    "window_id",
    "participant_id",
    "trial_id",
    "session_id",
    "label",
    "instruction",
    "window_start",
    "window_end",
    "valid_ratio",
    "sample_count",
    "is_usable",
    "original_sequence_length",
    "padded_length",
    "sequence_file",
]


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and return a dataframe."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty or unreadable: {path}") from exc


def validate_input_files() -> list[str]:
    """Return validation errors for required raw and split CSV files."""
    required_paths = [
        Path(RAW_DATA_DIR) / GAZE_FILE,
        Path(PROCESSED_DATA_DIR) / TRAIN_WINDOWS_FILE,
        Path(PROCESSED_DATA_DIR) / VALIDATION_WINDOWS_FILE,
        Path(PROCESSED_DATA_DIR) / TEST_WINDOWS_FILE,
    ]
    errors: list[str] = []
    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing required file: {path}")
    return errors


def validate_window_columns(windows_df: pd.DataFrame, split_name: str) -> list[str]:
    """Return validation errors for split window metadata."""
    errors = _missing_column_errors(windows_df, WINDOW_REQUIRED_COLUMNS, split_name)
    if errors or windows_df.empty:
        return errors

    labels = pd.to_numeric(windows_df["label"], errors="coerce")
    if labels.isna().any():
        errors.append(f"{split_name} contains missing or non-numeric label values.")
    else:
        invalid_labels = sorted(set(labels.astype(int)) - {0, 1})
        if invalid_labels:
            errors.append(f"{split_name} labels must be 0 or 1. Found: {invalid_labels}")

    starts = pd.to_numeric(windows_df["window_start"], errors="coerce")
    ends = pd.to_numeric(windows_df["window_end"], errors="coerce")
    if starts.isna().any() or ends.isna().any():
        errors.append(f"{split_name} contains invalid window_start/window_end values.")
    elif (ends <= starts).any():
        errors.append(f"{split_name} contains windows where window_end <= window_start.")

    return errors


def validate_gaze_columns(gaze_df: pd.DataFrame) -> list[str]:
    """Return validation errors for raw gaze sample columns."""
    return _missing_column_errors(gaze_df, GAZE_REQUIRED_COLUMNS, "gaze_samples.csv")


def prepare_clean_gaze_samples(gaze_df: pd.DataFrame) -> pd.DataFrame:
    """Clean gaze samples and ensure sequence-derived feature columns exist."""
    cleaned = clean_gaze_samples(gaze_df)

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
        "pupil_mean",
        "gaze_velocity",
    ]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "pupil_mean" not in cleaned.columns:
        cleaned["pupil_mean"] = cleaned[["pupil_left", "pupil_right"]].mean(axis=1)
    if "gaze_velocity" not in cleaned.columns:
        cleaned["gaze_velocity"] = 0.0

    return cleaned.sort_values(["trial_id", "timestamp"]).reset_index(drop=True)


def extract_window_samples(gaze_df: pd.DataFrame, window_row: pd.Series) -> pd.DataFrame:
    """Extract raw gaze samples that fall inside one processed window."""
    window_start = float(window_row["window_start"])
    window_end = float(window_row["window_end"])
    trial_id = window_row["trial_id"]

    mask = (
        (gaze_df["trial_id"].astype(str) == str(trial_id))
        & (gaze_df["timestamp"] >= window_start)
        & (gaze_df["timestamp"] < window_end)
    )
    samples = gaze_df.loc[mask].copy()
    denominator = window_end - window_start
    samples["timestamp_norm"] = (
        (samples["timestamp"] - window_start) / denominator
        if denominator > 0
        else MISSING_VALUE_FILL
    )
    return samples


def build_fixed_length_sequence(
    samples_df: pd.DataFrame,
    feature_columns: list[str],
    target_time_steps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build one padded or truncated sequence tensor from a window's samples."""
    ordered = samples_df.sort_values("timestamp").copy()
    original_length = len(ordered)

    missing_features = [column for column in feature_columns if column not in ordered.columns]
    if missing_features:
        raise ValueError("Missing sequence feature columns: " + ", ".join(missing_features))

    values = ordered[feature_columns].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).fillna(MISSING_VALUE_FILL)
    feature_count = len(feature_columns)

    sequence_array = np.full(
        (target_time_steps, feature_count),
        MISSING_VALUE_FILL,
        dtype=np.float32,
    )
    sample_mask = np.zeros(target_time_steps, dtype=np.float32)

    usable_length = min(original_length, target_time_steps)
    if usable_length > 0:
        sequence_array[:usable_length] = values.iloc[:usable_length].to_numpy(dtype=np.float32)
        sample_mask[:usable_length] = 1.0

    return sequence_array, sample_mask, original_length


def fit_sequence_scaler(
    X_train: np.ndarray,
    mask_train: np.ndarray,
    continuous_feature_indices: list[int],
):
    """Fit a StandardScaler on real train samples from continuous features only."""
    if X_train.shape[0] == 0:
        raise ValueError("Train split is empty. Cannot fit sequence scaler.")

    real_rows = mask_train.astype(bool)
    if not real_rows.any():
        raise ValueError("Train split has no real gaze samples. Cannot fit sequence scaler.")

    train_continuous = X_train[:, :, continuous_feature_indices][real_rows]
    if train_continuous.size == 0:
        raise ValueError("No continuous train samples are available for scaler fitting.")

    scaler = StandardScaler()
    scaler.fit(train_continuous)
    return scaler


def apply_sequence_scaler(
    X: np.ndarray,
    mask: np.ndarray,
    scaler,
    continuous_feature_indices: list[int],
) -> np.ndarray:
    """Scale continuous features at real sample positions and keep padding unchanged."""
    scaled = X.copy()
    if X.shape[0] == 0:
        return scaled

    real_rows = mask.astype(bool)
    if not real_rows.any():
        return scaled

    continuous_values = scaled[:, :, continuous_feature_indices]
    continuous_values[real_rows] = scaler.transform(continuous_values[real_rows])
    scaled[:, :, continuous_feature_indices] = continuous_values
    scaled[~real_rows] = MISSING_VALUE_FILL
    return scaled


def save_npz(
    output_path: str,
    X: np.ndarray,
    y: np.ndarray,
    masks: np.ndarray,
    window_ids: np.ndarray,
    participant_ids: np.ndarray,
    trial_ids: np.ndarray,
    valid_ratios: np.ndarray,
    original_lengths: np.ndarray,
) -> None:
    """Save one split's sequence arrays."""
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        masks=masks,
        window_ids=window_ids,
        participant_ids=participant_ids,
        trial_ids=trial_ids,
        valid_ratios=valid_ratios,
        original_lengths=original_lengths,
    )


def write_metadata_csv(metadata_rows: list[dict], output_path: str) -> None:
    """Write sequence metadata with stable headers, including for empty splits."""
    metadata_df = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    metadata_df.to_csv(output_path, index=False)


def _missing_column_errors(
    df: pd.DataFrame,
    required_columns: list[str],
    name: str,
) -> list[str]:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        return [f"{name} is missing required columns: {', '.join(missing)}"]
    return []
