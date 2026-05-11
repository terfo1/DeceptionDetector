"""Configuration constants for baseline window-feature models."""

PROCESSED_DATA_DIR = "data/processed"
MODEL_OUTPUT_DIR = "models/baselines"
REPORT_OUTPUT_DIR = "reports/baselines"

TRAIN_FEATURES_FILE = "train_window_features.csv"
VALIDATION_FEATURES_FILE = "validation_window_features.csv"
TEST_FEATURES_FILE = "test_window_features.csv"

RANDOM_SEED = 42

TARGET_COLUMN = "label"

METADATA_COLUMNS = [
    "window_id",
    "trial_id",
    "session_id",
    "participant_id",
    "instruction",
    "answer",
    "window_start",
    "window_end",
    "sample_count",
    "valid_ratio",
    "is_usable",
    "label",
]
