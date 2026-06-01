"""Configuration for model comparison and diagnostics."""

BASELINE_REPORT_DIR = "reports/baselines"
SEQUENCE_REPORT_DIR = "reports/sequences"
PROCESSED_DATA_DIR = "data/processed"
SEQUENCE_DATA_DIR = "data/processed/sequences"
OUTPUT_DIR = "reports/model_comparison"

METRIC_FILES = [
    "reports/baselines/baseline_metrics.csv",
    "reports/sequences/sequence_model_metrics.csv",
    "reports/sequences/tcn_metrics.csv",
]

PREDICTION_FILES = [
    "reports/baselines/validation_predictions.csv",
    "reports/baselines/test_predictions.csv",
    "reports/sequences/validation_predictions.csv",
    "reports/sequences/test_predictions.csv",
    "reports/sequences/tcn_validation_predictions.csv",
    "reports/sequences/tcn_test_predictions.csv",
]

PRIMARY_SPLIT = "test"
PRIMARY_METRIC = "f1"

MIN_RELIABLE_TRAIN_SAMPLES = 100
MIN_RELIABLE_TEST_SAMPLES = 50
MIN_RELIABLE_PARTICIPANTS = 10
