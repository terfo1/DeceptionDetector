"""Configuration constants for preprocessing raw eye-tracking data."""

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

WINDOW_SIZE_SECONDS = 3.0
WINDOW_STRIDE_SECONDS = 1.0
MIN_VALID_RATIO = 0.7

SIGNAL_COLUMNS = [
    "gaze_x",
    "gaze_y",
    "pupil_left",
    "pupil_right",
    "blink",
    "fixation",
    "saccade",
    "validity",
]

PARTICIPANTS_COLUMNS = ["participant_id", "notes"]
SESSIONS_COLUMNS = [
    "session_id",
    "participant_id",
    "date",
    "device",
    "screen_width",
    "screen_height",
    "sampling_rate",
    "calibration_quality",
]
TRIALS_COLUMNS = [
    "trial_id",
    "session_id",
    "question_text",
    "instruction",
    "label",
    "answer",
    "response_time",
    "start_time",
    "end_time",
]
GAZE_SAMPLES_COLUMNS = [
    "sample_id",
    "trial_id",
    "timestamp",
    *SIGNAL_COLUMNS,
]
