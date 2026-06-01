"""Configuration for offline real-time replay simulation."""

RAW_DATA_DIR = "data/raw"
MODEL_BASELINE_DIR = "models/baselines"
MODEL_SEQUENCE_DIR = "models/sequences"
SEQUENCE_DATA_DIR = "data/processed/sequences"
OUTPUT_DIR = "reports/realtime_simulation"

DEFAULT_MODEL_TYPE = "random_forest"
SUPPORTED_MODEL_TYPES = [
    "random_forest",
    "logistic_regression",
    "gru",
    "lstm",
    "causal_tcn",
]

WINDOW_SIZE_SECONDS = 3.0
UPDATE_INTERVAL_SECONDS = 0.5
SAMPLING_RATE = 60
TARGET_TIME_STEPS = 180
MIN_VALID_RATIO = 0.70

RISK_LOW_THRESHOLD = 0.40
RISK_HIGH_THRESHOLD = 0.70

MISSING_VALUE_FILL = 0.0

SEQUENCE_FEATURE_COLUMNS = [
    "gaze_x",
    "gaze_y",
    "pupil_left",
    "pupil_right",
    "pupil_mean",
    "gaze_velocity",
    "timestamp_norm",
    "blink",
    "fixation",
    "saccade",
    "validity",
]

CONTINUOUS_SEQUENCE_FEATURES = [
    "gaze_x",
    "gaze_y",
    "pupil_left",
    "pupil_right",
    "pupil_mean",
    "gaze_velocity",
    "timestamp_norm",
]
