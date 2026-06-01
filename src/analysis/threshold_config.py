"""Configuration for threshold calibration and probability diagnostics."""

OUTPUT_DIR = "reports/threshold_calibration"

PREDICTION_FILES = [
    "reports/baselines/validation_predictions.csv",
    "reports/baselines/test_predictions.csv",
    "reports/sequences/validation_predictions.csv",
    "reports/sequences/test_predictions.csv",
    "reports/sequences/tcn_validation_predictions.csv",
    "reports/sequences/tcn_test_predictions.csv",
]

REALTIME_PREDICTIONS_FILE = "reports/realtime_simulation/realtime_predictions.csv"
REALTIME_TRIAL_SUMMARY_FILE = "reports/realtime_simulation/realtime_trial_summary.csv"

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

DEFAULT_LOW_THRESHOLD = 0.40
DEFAULT_HIGH_THRESHOLD = 0.70

MIN_RELIABLE_SAMPLES = 100
MIN_RELIABLE_PARTICIPANTS = 10

PRIMARY_METRIC = "f1"

SUPPORTED_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "lstm",
    "gru",
    "causal_tcn",
]
