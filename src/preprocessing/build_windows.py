"""Build processed sliding-window datasets from raw eye-tracking CSV files."""

from datetime import datetime
from pathlib import Path

from .cleaning import clean_gaze_samples, load_raw_data
from .config import (
    MIN_VALID_RATIO,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    WINDOW_SIZE_SECONDS,
    WINDOW_STRIDE_SECONDS,
)
from .validate_raw_data import validate_raw_dataset
from .windowing import create_windows


def build_windows():
    """Validate raw data, clean samples, create windows, and save processed CSVs."""
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_raw_dataset(RAW_DATA_DIR)
    if not validation["is_valid"]:
        print("Raw dataset validation: FAILED")
        for error in validation["errors"]:
            print(f"- {error}")
        print("Preprocessing stopped. Fix raw data errors and run again.")
        return False

    print("Raw dataset validation: OK")

    try:
        participants_df, sessions_df, trials_df, gaze_df = load_raw_data(RAW_DATA_DIR)
    except Exception as exc:
        print(f"Could not load raw data: {exc}")
        return False

    if trials_df.empty:
        print("No trials found. Preprocessing stopped.")
        return False
    if gaze_df.empty:
        print("No gaze samples found. Preprocessing stopped.")
        return False

    cleaned_gaze_df = clean_gaze_samples(gaze_df)
    windows_df, window_features_df = create_windows(
        trials_df=trials_df,
        gaze_df=cleaned_gaze_df,
        window_size_seconds=WINDOW_SIZE_SECONDS,
        stride_seconds=WINDOW_STRIDE_SECONDS,
        min_valid_ratio=MIN_VALID_RATIO,
    )

    windows_path = processed_dir / "windows.csv"
    features_path = processed_dir / "window_features.csv"
    report_path = processed_dir / "preprocessing_report.txt"

    windows_df.to_csv(windows_path, index=False)
    window_features_df.to_csv(features_path, index=False)

    report_text = _build_report(
        participants_count=len(participants_df),
        sessions_count=len(sessions_df),
        trials_count=len(trials_df),
        gaze_count=len(gaze_df),
        windows_df=windows_df,
        validation=validation,
    )
    report_path.write_text(report_text, encoding="utf-8")

    usable_count = int(windows_df["is_usable"].sum()) if not windows_df.empty else 0
    print(f"Trials loaded: {len(trials_df)}")
    print(f"Raw gaze samples loaded: {len(gaze_df)}")
    print(f"Windows created: {len(windows_df)}")
    print(f"Usable windows: {usable_count}")
    print(f"Processed files saved to {processed_dir}")
    return True


def _build_report(
    participants_count,
    sessions_count,
    trials_count,
    gaze_count,
    windows_df,
    validation,
):
    windows_count = len(windows_df)
    usable_count = int(windows_df["is_usable"].sum()) if not windows_df.empty else 0
    unusable_count = windows_count - usable_count

    lines = [
        "Preprocessing Report",
        "====================",
        f"date_time: {datetime.now().isoformat(timespec='seconds')}",
        f"participants: {participants_count}",
        f"sessions: {sessions_count}",
        f"trials: {trials_count}",
        f"raw_gaze_samples: {gaze_count}",
        f"windows_created: {windows_count}",
        f"usable_windows: {usable_count}",
        f"unusable_windows: {unusable_count}",
        f"window_size_seconds: {WINDOW_SIZE_SECONDS}",
        f"window_stride_seconds: {WINDOW_STRIDE_SECONDS}",
        f"min_valid_ratio: {MIN_VALID_RATIO}",
        "",
        "Label distribution by windows:",
    ]

    if windows_df.empty:
        lines.append("- no windows created")
    else:
        for label, count in windows_df["label"].value_counts().sort_index().items():
            lines.append(f"- label {label}: {count}")

    lines.append("")
    lines.append("Validation warnings:")
    if validation["warnings"]:
        for warning in validation["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    lines.append("")
    return "\n".join(lines)


def main():
    try:
        build_windows()
    except Exception as exc:  # Defensive fallback for unexpected preprocessing problems.
        print(f"Preprocessing failed unexpectedly: {exc}")


if __name__ == "__main__":
    main()
