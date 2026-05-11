"""Create subject-independent train/validation/test split CSV files."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .split_config import (
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    USE_ONLY_USABLE_WINDOWS,
    VALIDATION_RATIO,
)


WINDOWS_REQUIRED_COLUMNS = [
    "window_id",
    "trial_id",
    "session_id",
    "window_start",
    "window_end",
    "label",
    "instruction",
    "valid_ratio",
    "sample_count",
    "is_usable",
]
FEATURES_REQUIRED_COLUMNS = [
    "window_id",
    "trial_id",
    "session_id",
    "label",
    "instruction",
    "is_usable",
]
SESSIONS_REQUIRED_COLUMNS = ["session_id", "participant_id"]


def load_required_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load windows, window features, and sessions after validating file presence."""
    processed_dir = Path(PROCESSED_DATA_DIR)
    raw_dir = Path(RAW_DATA_DIR)
    windows_path = processed_dir / "windows.csv"
    features_path = processed_dir / "window_features.csv"
    sessions_path = raw_dir / "sessions.csv"

    _require_file(windows_path)
    _require_file(features_path)
    _require_file(sessions_path)

    windows_df = _read_csv(windows_path)
    features_df = _read_csv(features_path)
    sessions_df = _read_csv(sessions_path)

    _require_columns(windows_df, WINDOWS_REQUIRED_COLUMNS, "windows.csv")
    _require_columns(features_df, FEATURES_REQUIRED_COLUMNS, "window_features.csv")
    _require_columns(sessions_df, SESSIONS_REQUIRED_COLUMNS, "sessions.csv")

    if windows_df.empty:
        raise ValueError("windows.csv has no rows.")
    if features_df.empty:
        raise ValueError("window_features.csv has no rows.")
    if sessions_df.empty:
        raise ValueError("sessions.csv has no rows.")
    if sessions_df["session_id"].duplicated().any():
        raise ValueError("sessions.csv contains duplicate session_id values.")

    return windows_df, features_df, sessions_df


def attach_participant_ids(df: pd.DataFrame, sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Attach participant_id to rows through session_id."""
    if "participant_id" in df.columns:
        df = df.drop(columns=["participant_id"])

    session_lookup = sessions_df[["session_id", "participant_id"]].copy()
    merged = df.merge(session_lookup, on="session_id", how="left", validate="many_to_one")

    if merged["participant_id"].isna().any():
        missing_sessions = sorted(
            merged.loc[merged["participant_id"].isna(), "session_id"].dropna().astype(str).unique()
        )
        raise ValueError(
            "Some rows could not be assigned participant_id. Unknown session_id values: "
            + ", ".join(missing_sessions[:10])
        )

    return merged


def filter_usable_windows(
    windows_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only windows marked usable in both metadata and feature files."""
    usable_window_ids = set(
        windows_df.loc[_as_bool(windows_df["is_usable"]), "window_id"].astype(str)
    )
    usable_feature_ids = set(
        features_df.loc[_as_bool(features_df["is_usable"]), "window_id"].astype(str)
    )
    keep_ids = usable_window_ids & usable_feature_ids

    filtered_windows = windows_df[windows_df["window_id"].astype(str).isin(keep_ids)].copy()
    filtered_features = features_df[features_df["window_id"].astype(str).isin(keep_ids)].copy()

    if filtered_windows.empty or filtered_features.empty:
        raise ValueError("No usable windows remain after filtering.")

    return filtered_windows, filtered_features


def split_participants(
    participant_ids: list[str],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> dict:
    """Split unique participant IDs into train, validation, and test groups."""
    warnings = []
    unique_participants = sorted({str(participant_id) for participant_id in participant_ids})
    if not unique_participants:
        raise ValueError("No participant_id values found.")

    rng = np.random.default_rng(random_seed)
    shuffled = list(rng.permutation(unique_participants))
    participant_count = len(shuffled)

    if participant_count < 3:
        warnings.append(
            "Subject-independent train/validation/test split requires at least 3 participants."
        )

    if participant_count == 1:
        train = shuffled
        validation = []
        test = []
    elif participant_count == 2:
        train = shuffled[:1]
        validation = []
        test = shuffled[1:]
    else:
        test_count = max(1, int(round(participant_count * test_ratio)))
        validation_count = max(1, int(round(participant_count * validation_ratio)))

        while test_count + validation_count > participant_count - 1:
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                break

        train_count = participant_count - validation_count - test_count
        train = shuffled[:train_count]
        validation = shuffled[train_count : train_count + validation_count]
        test = shuffled[train_count + validation_count :]

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "warnings": warnings,
    }


def create_split_dataframes(
    windows_df: pd.DataFrame,
    features_df: pd.DataFrame,
    split_participants: dict,
) -> dict:
    """Create split-specific dataframes using participant_id membership."""
    train_participants = set(split_participants["train"])
    validation_participants = set(split_participants["validation"])
    test_participants = set(split_participants["test"])

    return {
        "train_windows": windows_df[windows_df["participant_id"].isin(train_participants)].copy(),
        "validation_windows": windows_df[
            windows_df["participant_id"].isin(validation_participants)
        ].copy(),
        "test_windows": windows_df[windows_df["participant_id"].isin(test_participants)].copy(),
        "train_features": features_df[features_df["participant_id"].isin(train_participants)].copy(),
        "validation_features": features_df[
            features_df["participant_id"].isin(validation_participants)
        ].copy(),
        "test_features": features_df[features_df["participant_id"].isin(test_participants)].copy(),
    }


def check_no_participant_leakage(split_dataframes: dict) -> tuple[bool, list[str]]:
    """Check that no participant_id appears in more than one split."""
    participants_by_split = {
        "train": _participants(split_dataframes["train_windows"]),
        "validation": _participants(split_dataframes["validation_windows"]),
        "test": _participants(split_dataframes["test_windows"]),
    }

    errors = []
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        overlap = participants_by_split[left] & participants_by_split[right]
        if overlap:
            errors.append(
                f"Participant leakage between {left} and {right}: "
                + ", ".join(sorted(overlap))
            )

    return len(errors) == 0, errors


def check_window_id_consistency(split_dataframes: dict) -> tuple[bool, list[str]]:
    """Check matching window_id sets between each split's metadata and feature files."""
    errors = []
    split_pairs = [
        ("train", "train_windows", "train_features"),
        ("validation", "validation_windows", "validation_features"),
        ("test", "test_windows", "test_features"),
    ]
    for split_name, windows_key, features_key in split_pairs:
        window_ids = set(split_dataframes[windows_key]["window_id"].astype(str))
        feature_ids = set(split_dataframes[features_key]["window_id"].astype(str))
        if window_ids != feature_ids:
            missing_features = sorted(window_ids - feature_ids)
            extra_features = sorted(feature_ids - window_ids)
            errors.append(
                f"{split_name} window_id mismatch. "
                f"Missing in features: {missing_features[:10]}; "
                f"extra in features: {extra_features[:10]}"
            )

    return len(errors) == 0, errors


def save_split_files(split_dataframes: dict, processed_dir: str) -> None:
    """Save split CSV files to data/processed."""
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dataframes["train_windows"].to_csv(output_dir / "train_windows.csv", index=False)
    split_dataframes["validation_windows"].to_csv(
        output_dir / "validation_windows.csv", index=False
    )
    split_dataframes["test_windows"].to_csv(output_dir / "test_windows.csv", index=False)
    split_dataframes["train_features"].to_csv(
        output_dir / "train_window_features.csv", index=False
    )
    split_dataframes["validation_features"].to_csv(
        output_dir / "validation_window_features.csv", index=False
    )
    split_dataframes["test_features"].to_csv(
        output_dir / "test_window_features.csv", index=False
    )


def write_split_report(
    split_dataframes: dict,
    split_participants: dict,
    warnings: list[str],
    output_path: str,
) -> None:
    """Write a readable split report."""
    leakage_ok, leakage_errors = check_no_participant_leakage(split_dataframes)
    consistency_ok, consistency_errors = check_window_id_consistency(split_dataframes)
    all_participants = (
        set(split_participants["train"])
        | set(split_participants["validation"])
        | set(split_participants["test"])
    )

    lines = [
        "Subject-Independent Split Report",
        "================================",
        f"date_time: {datetime.now().isoformat(timespec='seconds')}",
        f"random_seed: {RANDOM_SEED}",
        f"train_ratio: {TRAIN_RATIO}",
        f"validation_ratio: {VALIDATION_RATIO}",
        f"test_ratio: {TEST_RATIO}",
        f"use_only_usable_windows: {USE_ONLY_USABLE_WINDOWS}",
        f"total_participants: {len(all_participants)}",
        f"train_participants: {', '.join(split_participants['train']) or 'none'}",
        f"validation_participants: {', '.join(split_participants['validation']) or 'none'}",
        f"test_participants: {', '.join(split_participants['test']) or 'none'}",
        "",
        "Windows by split:",
        f"- train: {len(split_dataframes['train_windows'])}",
        f"- validation: {len(split_dataframes['validation_windows'])}",
        f"- test: {len(split_dataframes['test_windows'])}",
        "",
        "Label distribution by split:",
        _label_distribution_line("train", split_dataframes["train_windows"]),
        _label_distribution_line("validation", split_dataframes["validation_windows"]),
        _label_distribution_line("test", split_dataframes["test_windows"]),
        "",
        "Warnings:",
    ]

    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("- This split is not scientifically strong with the current participant count.")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Subject-independent split was used. No participant_id appears in more than one split.",
            f"Leakage check: {'OK' if leakage_ok else 'FAILED'}",
            f"Window ID consistency check: {'OK' if consistency_ok else 'FAILED'}",
        ]
    )

    if leakage_errors:
        lines.append("")
        lines.append("Leakage errors:")
        lines.extend(f"- {error}" for error in leakage_errors)

    if consistency_errors:
        lines.append("")
        lines.append("Window ID consistency errors:")
        lines.extend(f"- {error}" for error in consistency_errors)

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Creating subject-independent dataset splits...")
    try:
        windows_df, features_df, sessions_df = load_required_data()
        windows_df = attach_participant_ids(windows_df, sessions_df)
        features_df = attach_participant_ids(features_df, sessions_df)

        _validate_labels(windows_df, "windows.csv")
        _validate_labels(features_df, "window_features.csv")
        _validate_window_alignment(windows_df, features_df)

        if USE_ONLY_USABLE_WINDOWS:
            windows_df, features_df = filter_usable_windows(windows_df, features_df)
            _validate_window_alignment(windows_df, features_df)

        participant_ids = sorted(windows_df["participant_id"].astype(str).unique())
        participant_split = split_participants(
            participant_ids=participant_ids,
            train_ratio=TRAIN_RATIO,
            validation_ratio=VALIDATION_RATIO,
            test_ratio=TEST_RATIO,
            random_seed=RANDOM_SEED,
        )
        split_dataframes = create_split_dataframes(windows_df, features_df, participant_split)

        leakage_ok, leakage_errors = check_no_participant_leakage(split_dataframes)
        consistency_ok, consistency_errors = check_window_id_consistency(split_dataframes)
        if not leakage_ok:
            raise ValueError("Participant leakage detected: " + "; ".join(leakage_errors))
        if not consistency_ok:
            raise ValueError("Window ID mismatch detected: " + "; ".join(consistency_errors))

        save_split_files(split_dataframes, PROCESSED_DATA_DIR)
        warnings = list(participant_split.get("warnings", []))
        write_split_report(
            split_dataframes=split_dataframes,
            split_participants=participant_split,
            warnings=warnings,
            output_path=str(Path(PROCESSED_DATA_DIR) / "split_report.txt"),
        )

        print(f"Participants found: {len(participant_ids)}")
        print(f"Train participants: {len(participant_split['train'])}")
        print(f"Validation participants: {len(participant_split['validation'])}")
        print(f"Test participants: {len(participant_split['test'])}")
        print(f"Train windows: {len(split_dataframes['train_windows'])}")
        print(f"Validation windows: {len(split_dataframes['validation_windows'])}")
        print(f"Test windows: {len(split_dataframes['test_windows'])}")
        for warning in warnings:
            print(f"Warning: {warning}")
        print("Leakage check: OK")
        print(f"Files saved to {PROCESSED_DATA_DIR}")
    except ValueError as exc:
        print(f"Split creation failed: {exc}")
    except OSError as exc:
        print(f"File error while creating splits: {exc}")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Required file is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required file is empty: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Required file is empty or unreadable: {path}") from exc


def _require_columns(df: pd.DataFrame, required_columns: list[str], file_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{file_name} is missing columns: {', '.join(missing_columns)}")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _validate_labels(df: pd.DataFrame, name: str) -> None:
    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{name} contains non-numeric label values.")
    invalid_labels = sorted(set(labels.astype(int)) - {0, 1})
    if invalid_labels:
        raise ValueError(f"{name} labels must only be 0 or 1. Found: {invalid_labels}")


def _validate_window_alignment(windows_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
    window_ids = set(windows_df["window_id"].astype(str))
    feature_ids = set(features_df["window_id"].astype(str))
    if window_ids != feature_ids:
        raise ValueError("windows.csv and window_features.csv have different window_id sets.")


def _participants(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    return set(df["participant_id"].dropna().astype(str))


def _label_distribution_line(split_name: str, df: pd.DataFrame) -> str:
    if df.empty:
        return f"- {split_name}: label 0 = 0, label 1 = 0"
    labels = pd.to_numeric(df["label"], errors="coerce")
    return (
        f"- {split_name}: label 0 = {int((labels == 0).sum())}, "
        f"label 1 = {int((labels == 1).sum())}"
    )


if __name__ == "__main__":
    main()
