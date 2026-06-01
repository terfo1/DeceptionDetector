"""Configuration for local tracking artifacts."""

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

REPORTS_DIR = "reports"
TRACKING_OUTPUT_DIR = "reports/tracking"

DATASET_VERSION_PREFIX = "dataset_v"
RUN_VERSION_PREFIX = "run_v"

TRACKED_RAW_FILES = [
    "data/raw/participants.csv",
    "data/raw/participant_metadata.csv",
    "data/raw/sessions.csv",
    "data/raw/session_metadata.csv",
    "data/raw/trials.csv",
    "data/raw/gaze_samples.csv",
    "data/raw/session_quality.csv",
]

TRACKED_PROCESSED_FILES = [
    "data/processed/windows.csv",
    "data/processed/window_features.csv",
    "data/processed/train_windows.csv",
    "data/processed/validation_windows.csv",
    "data/processed/test_windows.csv",
    "data/processed/train_window_features.csv",
    "data/processed/validation_window_features.csv",
    "data/processed/test_window_features.csv",
    "data/processed/sequences/train_sequences.npz",
    "data/processed/sequences/validation_sequences.npz",
    "data/processed/sequences/test_sequences.npz",
]

TRACKED_REPORT_FILES = [
    "data/processed/preprocessing_report.txt",
    "data/processed/split_report.txt",
    "data/processed/sequences/sequence_dataset_report.txt",
    "reports/baselines/baseline_metrics.csv",
    "reports/sequences/sequence_model_metrics.csv",
    "reports/sequences/tcn_metrics.csv",
    "reports/model_comparison/model_ranking.csv",
    "reports/model_comparison/model_comparison_report.txt",
    "reports/threshold_calibration/selected_thresholds.json",
    "reports/model_selection/selected_model_config.json",
    "reports/model_selection/model_selection_report.txt",
]
