"""Configuration for neural sequence dataset preparation."""

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
SEQUENCE_OUTPUT_DIR = "data/processed/sequences"

WINDOW_SIZE_SECONDS = 3.0
SAMPLING_RATE = 60
TARGET_TIME_STEPS = 180

TARGET_COLUMN = "label"

CONTINUOUS_SEQUENCE_FEATURES = [
    "gaze_x",
    "gaze_y",
    "pupil_left",
    "pupil_right",
    "pupil_mean",
    "gaze_velocity",
    "timestamp_norm",
]

BINARY_SEQUENCE_FEATURES = [
    "blink",
    "fixation",
    "saccade",
    "validity",
]

SEQUENCE_FEATURE_COLUMNS = CONTINUOUS_SEQUENCE_FEATURES + BINARY_SEQUENCE_FEATURES

MISSING_VALUE_FILL = 0.0
