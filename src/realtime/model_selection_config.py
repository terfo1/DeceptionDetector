"""Model selection settings for the live prototype."""

SELECTED_MODEL = "random_forest"
FALLBACK_MODEL = "gru"

DISABLED_MODELS = ["lstm"]
EXPERIMENTAL_MODELS = ["causal_tcn"]

SUPPORTED_MODELS = [
    "logistic_regression",
    "random_forest",
    "gru",
    "lstm",
    "causal_tcn",
]

MODEL_SELECTION_OUTPUT_DIR = "reports/model_selection"

DEFAULT_RISK_THRESHOLDS = {
    "low": 0.40,
    "high": 0.70,
}

PRELIMINARY_MODEL_THRESHOLDS = {
    "random_forest": {
        "binary_threshold": 0.50,
        "low": 0.40,
        "high": 0.70,
        "note": "Use default thresholds for runtime prototype. Calibration thresholds are preliminary.",
    },
    "gru": {
        "binary_threshold": 0.50,
        "low": 0.40,
        "high": 0.70,
        "note": "Fallback neural model. Probability range is narrow on current data.",
    },
    "causal_tcn": {
        "binary_threshold": 0.50,
        "low": 0.40,
        "high": 0.70,
        "note": "Experimental only. Narrow probability range observed.",
    },
}
