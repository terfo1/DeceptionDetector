"""Validate raw eye-tracking CSV files before preprocessing."""

from pathlib import Path

import pandas as pd

from .config import (
    GAZE_SAMPLES_COLUMNS,
    PARTICIPANTS_COLUMNS,
    RAW_DATA_DIR,
    SESSIONS_COLUMNS,
    TRIALS_COLUMNS,
)


REQUIRED_FILES = {
    "participants": ("participants.csv", PARTICIPANTS_COLUMNS),
    "sessions": ("sessions.csv", SESSIONS_COLUMNS),
    "trials": ("trials.csv", TRIALS_COLUMNS),
    "gaze_samples": ("gaze_samples.csv", GAZE_SAMPLES_COLUMNS),
}


def validate_raw_dataset(raw_dir: str = RAW_DATA_DIR) -> dict:
    """Return validation status, errors, warnings, and row counts."""
    raw_path = Path(raw_dir)
    errors = []
    warnings = []
    counts = {
        "participants": 0,
        "sessions": 0,
        "trials": 0,
        "gaze_samples": 0,
    }
    dataframes = {}

    if not raw_path.exists():
        errors.append(f"Raw data directory does not exist: {raw_path}")
        return _result(errors, warnings, counts)

    for name, (file_name, required_columns) in REQUIRED_FILES.items():
        path = raw_path / file_name
        if not path.exists():
            errors.append(f"Missing required file: {path}")
            continue
        if path.stat().st_size == 0:
            errors.append(f"Required file is empty and has no header: {path}")
            continue

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            errors.append(f"Required file is empty or unreadable: {path}")
            continue
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"Could not read {path}: {exc}")
            continue

        dataframes[name] = df
        counts[name] = len(df)
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            errors.append(f"{file_name} is missing columns: {', '.join(missing_columns)}")

    if errors:
        return _result(errors, warnings, counts)

    participants_df = dataframes["participants"]
    sessions_df = dataframes["sessions"]
    trials_df = dataframes["trials"]
    gaze_df = dataframes["gaze_samples"]

    if participants_df.empty:
        errors.append("participants.csv has no participant rows.")
    if sessions_df.empty:
        errors.append("sessions.csv has no session rows.")
    if trials_df.empty:
        errors.append("trials.csv has no trial rows.")
    if gaze_df.empty:
        errors.append("gaze_samples.csv has no gaze sample rows.")

    _check_relational_integrity(
        participants_df,
        sessions_df,
        trials_df,
        gaze_df,
        errors,
    )
    _check_labels(trials_df, errors)
    _check_gaze_values(gaze_df, errors)

    duplicate_participants = participants_df["participant_id"].duplicated().sum()
    if duplicate_participants:
        warnings.append(f"participants.csv contains {duplicate_participants} duplicate participant_id rows.")

    duplicate_sessions = sessions_df["session_id"].duplicated().sum()
    if duplicate_sessions:
        warnings.append(f"sessions.csv contains {duplicate_sessions} duplicate session_id rows.")

    duplicate_trials = trials_df["trial_id"].duplicated().sum()
    if duplicate_trials:
        warnings.append(f"trials.csv contains {duplicate_trials} duplicate trial_id rows.")

    return _result(errors, warnings, counts)


def _check_relational_integrity(participants_df, sessions_df, trials_df, gaze_df, errors):
    participant_ids = set(participants_df["participant_id"].dropna().astype(str))
    session_participant_ids = set(sessions_df["participant_id"].dropna().astype(str))
    missing_participants = sorted(session_participant_ids - participant_ids)
    if missing_participants:
        errors.append(
            "sessions.csv references unknown participant_id values: "
            + ", ".join(missing_participants[:10])
        )

    session_ids = set(sessions_df["session_id"].dropna().astype(str))
    trial_session_ids = set(trials_df["session_id"].dropna().astype(str))
    missing_sessions = sorted(trial_session_ids - session_ids)
    if missing_sessions:
        errors.append(
            "trials.csv references unknown session_id values: "
            + ", ".join(missing_sessions[:10])
        )

    trial_ids = set(trials_df["trial_id"].dropna().astype(str))
    gaze_trial_ids = set(gaze_df["trial_id"].dropna().astype(str))
    missing_trials = sorted(gaze_trial_ids - trial_ids)
    if missing_trials:
        errors.append(
            "gaze_samples.csv references unknown trial_id values: "
            + ", ".join(missing_trials[:10])
        )


def _check_labels(trials_df, errors):
    labels = pd.to_numeric(trials_df["label"], errors="coerce")
    if labels.isna().any():
        errors.append("trials.csv contains non-numeric label values.")

    invalid_labels = sorted(set(labels.dropna().astype(int)) - {0, 1})
    if invalid_labels:
        errors.append(f"trials.csv labels must only be 0 or 1. Found: {invalid_labels}")

    instructions = trials_df["instruction"].astype(str).str.lower()
    invalid_instructions = sorted(set(instructions) - {"truth", "lie"})
    if invalid_instructions:
        errors.append(
            "trials.csv instruction must only be 'truth' or 'lie'. Found: "
            + ", ".join(invalid_instructions[:10])
        )

    truth_mismatch = ((instructions == "truth") & (labels != 0)).sum()
    lie_mismatch = ((instructions == "lie") & (labels != 1)).sum()
    if truth_mismatch:
        errors.append(f"Found {truth_mismatch} truth trial(s) where label is not 0.")
    if lie_mismatch:
        errors.append(f"Found {lie_mismatch} lie trial(s) where label is not 1.")


def _check_gaze_values(gaze_df, errors):
    numeric_checks = {
        "gaze_x": "gaze_x must be numeric when present.",
        "gaze_y": "gaze_y must be numeric when present.",
        "pupil_left": "pupil_left must be numeric when present.",
        "pupil_right": "pupil_right must be numeric when present.",
        "blink": "blink must be numeric.",
        "fixation": "fixation must be numeric.",
        "saccade": "saccade must be numeric.",
        "validity": "validity must be numeric.",
    }
    converted = {}
    for column, message in numeric_checks.items():
        converted[column] = pd.to_numeric(gaze_df[column], errors="coerce")
        invalid_numeric = converted[column].isna() & gaze_df[column].notna()
        if invalid_numeric.any():
            errors.append(message)

    for column in ["gaze_x", "gaze_y"]:
        values = converted[column].dropna()
        invalid_count = ((values < 0) | (values > 1)).sum()
        if invalid_count:
            errors.append(f"{column} has {invalid_count} value(s) outside [0, 1].")

    for column in ["pupil_left", "pupil_right"]:
        values = converted[column].dropna()
        invalid_count = (values <= 0).sum()
        if invalid_count:
            errors.append(f"{column} has {invalid_count} non-positive value(s).")

    for column in ["blink", "fixation", "saccade", "validity"]:
        values = converted[column].dropna()
        invalid_count = (~values.isin([0, 1])).sum()
        if invalid_count:
            errors.append(f"{column} has {invalid_count} value(s) outside {{0, 1}}.")


def _result(errors, warnings, counts):
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
    }


def _print_report(result):
    print("Raw Dataset Validation Report")
    print("=============================")
    print(f"Status: {'OK' if result['is_valid'] else 'FAILED'}")
    print("")
    print("Counts:")
    for name, count in result["counts"].items():
        print(f"- {name}: {count}")

    if result["warnings"]:
        print("")
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")

    if result["errors"]:
        print("")
        print("Errors:")
        for error in result["errors"]:
            print(f"- {error}")


def main():
    try:
        result = validate_raw_dataset()
    except Exception as exc:  # Defensive fallback for unexpected input problems.
        print("Raw Dataset Validation Report")
        print("=============================")
        print("Status: FAILED")
        print(f"Unexpected validation error: {exc}")
        return

    _print_report(result)


if __name__ == "__main__":
    main()
