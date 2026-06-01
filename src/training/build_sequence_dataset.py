"""Build fixed-length neural sequence datasets from split window files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .sequence_config import (
    BINARY_SEQUENCE_FEATURES,
    CONTINUOUS_SEQUENCE_FEATURES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLING_RATE,
    SEQUENCE_FEATURE_COLUMNS,
    SEQUENCE_OUTPUT_DIR,
    TARGET_COLUMN,
    TARGET_TIME_STEPS,
    WINDOW_SIZE_SECONDS,
)
from .sequence_dataset_utils import (
    apply_sequence_scaler,
    build_fixed_length_sequence,
    extract_window_samples,
    fit_sequence_scaler,
    load_csv,
    prepare_clean_gaze_samples,
    save_npz,
    validate_gaze_columns,
    validate_input_files,
    validate_window_columns,
    write_metadata_csv,
)


SPLIT_FILES = {
    "train": "train_windows.csv",
    "validation": "validation_windows.csv",
    "test": "test_windows.csv",
}


def main() -> None:
    """Create sequence tensors for future neural sequence model training."""
    print("Building neural sequence datasets...")
    try:
        output_dir = Path(SEQUENCE_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_errors = validate_input_files()
        if input_errors:
            raise ValueError("\n".join(input_errors))

        raw_gaze_path = Path(RAW_DATA_DIR) / "gaze_samples.csv"
        split_paths = {
            split_name: Path(PROCESSED_DATA_DIR) / file_name
            for split_name, file_name in SPLIT_FILES.items()
        }

        gaze_df = load_csv(str(raw_gaze_path))
        gaze_errors = validate_gaze_columns(gaze_df)
        if gaze_errors:
            raise ValueError("\n".join(gaze_errors))

        split_windows = {
            split_name: load_csv(str(path))
            for split_name, path in split_paths.items()
        }

        validation_errors: list[str] = []
        for split_name, windows_df in split_windows.items():
            validation_errors.extend(validate_window_columns(windows_df, split_name))
        if validation_errors:
            raise ValueError("\n".join(validation_errors))

        _check_subject_independent_split(split_windows)
        gaze_df = prepare_clean_gaze_samples(gaze_df)

        split_data = {
            split_name: _build_split_sequences(split_name, windows_df, gaze_df)
            for split_name, windows_df in split_windows.items()
        }

        if split_data["train"]["X"].shape[0] == 0:
            raise ValueError("Train split is empty. Sequence scaler cannot be fitted.")

        continuous_indices = [
            SEQUENCE_FEATURE_COLUMNS.index(feature)
            for feature in CONTINUOUS_SEQUENCE_FEATURES
        ]
        scaler = fit_sequence_scaler(
            split_data["train"]["X"],
            split_data["train"]["masks"],
            continuous_indices,
        )
        joblib.dump(scaler, output_dir / "sequence_scaler.joblib")

        for split_name, data in split_data.items():
            data["X"] = apply_sequence_scaler(
                data["X"],
                data["masks"],
                scaler,
                continuous_indices,
            )
            sequence_path = output_dir / f"{split_name}_sequences.npz"
            save_npz(
                output_path=str(sequence_path),
                X=data["X"],
                y=data["y"],
                masks=data["masks"],
                window_ids=data["window_ids"],
                participant_ids=data["participant_ids"],
                trial_ids=data["trial_ids"],
                valid_ratios=data["valid_ratios"],
                original_lengths=data["original_lengths"],
            )
            write_metadata_csv(
                data["metadata_rows"],
                str(output_dir / f"{split_name}_sequence_metadata.csv"),
            )

        _write_feature_columns_json(output_dir / "sequence_feature_columns.json")
        _write_report(
            output_path=output_dir / "sequence_dataset_report.txt",
            input_files=[raw_gaze_path, *split_paths.values()],
            split_data=split_data,
            scaler=scaler,
        )

        print(f"Feature columns: {len(SEQUENCE_FEATURE_COLUMNS)}")
        print(f"Target time steps: {TARGET_TIME_STEPS}")
        print(f"Train sequences: {split_data['train']['X'].shape}")
        print(f"Validation sequences: {split_data['validation']['X'].shape}")
        print(f"Test sequences: {split_data['test']['X'].shape}")
        print("Scaler fitted on train split only.")
        print(f"Files saved to {SEQUENCE_OUTPUT_DIR}")
    except ValueError as exc:
        print(f"Sequence dataset preparation failed:\n{exc}")
    except OSError as exc:
        print(f"File error while preparing sequence datasets:\n{exc}")


def _build_split_sequences(
    split_name: str,
    windows_df: pd.DataFrame,
    gaze_df: pd.DataFrame,
) -> dict:
    feature_count = len(SEQUENCE_FEATURE_COLUMNS)
    sequence_file = f"{split_name}_sequences.npz"

    if windows_df.empty:
        return _empty_split_data(feature_count)

    sequences = []
    masks = []
    y_values = []
    window_ids = []
    participant_ids = []
    trial_ids = []
    valid_ratios = []
    original_lengths = []
    metadata_rows = []

    for _, window_row in windows_df.iterrows():
        samples = extract_window_samples(gaze_df, window_row)
        sequence, sample_mask, original_length = build_fixed_length_sequence(
            samples,
            SEQUENCE_FEATURE_COLUMNS,
            TARGET_TIME_STEPS,
        )

        label = int(pd.to_numeric(window_row[TARGET_COLUMN], errors="raise"))
        valid_ratio = float(pd.to_numeric(window_row["valid_ratio"], errors="coerce"))

        sequences.append(sequence)
        masks.append(sample_mask)
        y_values.append(label)
        window_ids.append(str(window_row["window_id"]))
        participant_ids.append(str(window_row["participant_id"]))
        trial_ids.append(str(window_row["trial_id"]))
        valid_ratios.append(valid_ratio)
        original_lengths.append(original_length)
        metadata_rows.append(
            {
                "window_id": window_row["window_id"],
                "participant_id": window_row["participant_id"],
                "trial_id": window_row["trial_id"],
                "session_id": window_row["session_id"],
                "label": label,
                "instruction": window_row["instruction"],
                "window_start": window_row["window_start"],
                "window_end": window_row["window_end"],
                "valid_ratio": window_row["valid_ratio"],
                "sample_count": window_row["sample_count"],
                "is_usable": window_row["is_usable"],
                "original_sequence_length": original_length,
                "padded_length": TARGET_TIME_STEPS,
                "sequence_file": sequence_file,
            }
        )

    return {
        "X": np.stack(sequences).astype(np.float32),
        "y": np.asarray(y_values, dtype=np.int64),
        "masks": np.stack(masks).astype(np.float32),
        "window_ids": np.asarray(window_ids, dtype=object),
        "participant_ids": np.asarray(participant_ids, dtype=object),
        "trial_ids": np.asarray(trial_ids, dtype=object),
        "valid_ratios": np.asarray(valid_ratios, dtype=np.float32),
        "original_lengths": np.asarray(original_lengths, dtype=np.int64),
        "metadata_rows": metadata_rows,
    }


def _empty_split_data(feature_count: int) -> dict:
    return {
        "X": np.empty((0, TARGET_TIME_STEPS, feature_count), dtype=np.float32),
        "y": np.empty((0,), dtype=np.int64),
        "masks": np.empty((0, TARGET_TIME_STEPS), dtype=np.float32),
        "window_ids": np.empty((0,), dtype=object),
        "participant_ids": np.empty((0,), dtype=object),
        "trial_ids": np.empty((0,), dtype=object),
        "valid_ratios": np.empty((0,), dtype=np.float32),
        "original_lengths": np.empty((0,), dtype=np.int64),
        "metadata_rows": [],
    }


def _check_subject_independent_split(split_windows: dict[str, pd.DataFrame]) -> None:
    participants = {
        split_name: set(windows_df["participant_id"].dropna().astype(str))
        for split_name, windows_df in split_windows.items()
    }
    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]
    errors = []
    for left, right in pairs:
        overlap = participants[left] & participants[right]
        if overlap:
            errors.append(
                f"Participant leakage between {left} and {right}: "
                + ", ".join(sorted(overlap))
            )
    if errors:
        raise ValueError("\n".join(errors))


def _write_feature_columns_json(output_path: Path) -> None:
    payload = {
        "continuous_features": CONTINUOUS_SEQUENCE_FEATURES,
        "binary_features": BINARY_SEQUENCE_FEATURES,
        "all_features": SEQUENCE_FEATURE_COLUMNS,
        "target_time_steps": TARGET_TIME_STEPS,
        "sampling_rate": SAMPLING_RATE,
        "window_size_seconds": WINDOW_SIZE_SECONDS,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_report(
    output_path: Path,
    input_files: list[Path],
    split_data: dict[str, dict],
    scaler,
) -> None:
    lines = [
        "Neural Sequence Dataset Report",
        "==============================",
        f"date_time: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Input files:",
        *[f"- {path}" for path in input_files],
        f"output_directory: {SEQUENCE_OUTPUT_DIR}",
        f"target_time_steps: {TARGET_TIME_STEPS}",
        f"sampling_rate: {SAMPLING_RATE}",
        f"window_size_seconds: {WINDOW_SIZE_SECONDS}",
        "",
        "Sequence feature columns:",
        *[f"- {column}" for column in SEQUENCE_FEATURE_COLUMNS],
        "",
        f"train_sequence_shape: {split_data['train']['X'].shape}",
        f"validation_sequence_shape: {split_data['validation']['X'].shape}",
        f"test_sequence_shape: {split_data['test']['X'].shape}",
        "",
        "Label distribution:",
        _label_distribution_line("train", split_data["train"]["y"]),
        _label_distribution_line("validation", split_data["validation"]["y"]),
        _label_distribution_line("test", split_data["test"]["y"]),
        "",
        "Original sequence lengths:",
        _length_stats_line("train", split_data["train"]["original_lengths"]),
        _length_stats_line("validation", split_data["validation"]["original_lengths"]),
        _length_stats_line("test", split_data["test"]["original_lengths"]),
        "",
        "Empty windows with no samples:",
        f"- train: {_empty_windows(split_data['train']['original_lengths'])}",
        f"- validation: {_empty_windows(split_data['validation']['original_lengths'])}",
        f"- test: {_empty_windows(split_data['test']['original_lengths'])}",
        "",
        "Scaler information:",
        "- type: StandardScaler",
        "- fitted_on: train split only",
        f"- scaled_features: {', '.join(CONTINUOUS_SEQUENCE_FEATURES)}",
        f"- n_features_in: {getattr(scaler, 'n_features_in_', 'unknown')}",
        "",
        "Sequence datasets are prepared for future LSTM/GRU/TCN training. "
        "No neural model is trained in this step.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_distribution_line(split_name: str, y: np.ndarray) -> str:
    if y.size == 0:
        return f"- {split_name}: label 0 = 0, label 1 = 0"
    return (
        f"- {split_name}: label 0 = {int((y == 0).sum())}, "
        f"label 1 = {int((y == 1).sum())}"
    )


def _length_stats_line(split_name: str, lengths: np.ndarray) -> str:
    if lengths.size == 0:
        return f"- {split_name}: average = 0.00, min = 0, max = 0"
    return (
        f"- {split_name}: average = {float(np.mean(lengths)):.2f}, "
        f"min = {int(np.min(lengths))}, max = {int(np.max(lengths))}"
    )


def _empty_windows(lengths: np.ndarray) -> int:
    if lengths.size == 0:
        return 0
    return int((lengths == 0).sum())


if __name__ == "__main__":
    main()
