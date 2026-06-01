"""Configuration for the local pipeline runner."""

PIPELINE_OUTPUT_DIR = "reports/pipeline"

DEFAULT_MODE = "full"

PIPELINE_STEPS = [
    {
        "name": "validate_raw_data",
        "command": ["python", "-m", "src.preprocessing.validate_raw_data"],
        "required_for_modes": ["validate", "preprocess", "full"],
        "critical": True,
        "description": "Validate raw CSV files and dataset integrity.",
    },
    {
        "name": "build_windows",
        "command": ["python", "-m", "src.preprocessing.build_windows"],
        "required_for_modes": ["preprocess", "full"],
        "critical": True,
        "description": "Clean gaze samples and build sliding windows.",
    },
    {
        "name": "create_splits",
        "command": ["python", "-m", "src.training.create_splits"],
        "required_for_modes": ["split", "full"],
        "critical": True,
        "description": "Create subject-independent train/validation/test splits.",
    },
    {
        "name": "train_baselines",
        "command": ["python", "-m", "src.models.train_baselines"],
        "required_for_modes": ["baselines", "training", "full"],
        "critical": False,
        "description": "Train Logistic Regression and Random Forest baseline models.",
    },
    {
        "name": "evaluate_baselines",
        "command": ["python", "-m", "src.models.evaluate_baselines"],
        "required_for_modes": ["baselines", "training", "full"],
        "critical": False,
        "description": "Evaluate baseline models on test split.",
    },
    {
        "name": "build_sequence_dataset",
        "command": ["python", "-m", "src.training.build_sequence_dataset"],
        "required_for_modes": ["sequences", "neural", "training", "full"],
        "critical": True,
        "description": "Build fixed-length sequence datasets for neural models.",
    },
    {
        "name": "train_sequence_models",
        "command": ["python", "-m", "src.models.train_sequence_models"],
        "required_for_modes": ["neural", "training", "full"],
        "critical": False,
        "description": "Train LSTM and GRU sequence models.",
    },
    {
        "name": "evaluate_sequence_models",
        "command": ["python", "-m", "src.models.evaluate_sequence_models"],
        "required_for_modes": ["neural", "training", "full"],
        "critical": False,
        "description": "Evaluate LSTM and GRU sequence models.",
    },
    {
        "name": "train_tcn_model",
        "command": ["python", "-m", "src.models.train_tcn_model"],
        "required_for_modes": ["neural", "training", "full"],
        "critical": False,
        "description": "Train Causal TCN model.",
    },
    {
        "name": "evaluate_tcn_model",
        "command": ["python", "-m", "src.models.evaluate_tcn_model"],
        "required_for_modes": ["neural", "training", "full"],
        "critical": False,
        "description": "Evaluate Causal TCN model.",
    },
    {
        "name": "generate_model_comparison",
        "command": ["python", "-m", "src.analysis.generate_model_comparison"],
        "required_for_modes": ["reports", "selection", "full"],
        "critical": False,
        "description": "Generate model comparison and diagnostics report.",
    },
    {
        "name": "generate_threshold_report",
        "command": ["python", "-m", "src.analysis.generate_threshold_report"],
        "required_for_modes": ["reports", "selection", "full"],
        "critical": False,
        "description": "Generate threshold calibration and probability diagnostics report.",
    },
    {
        "name": "validate_selected_model",
        "command": ["python", "-m", "src.realtime.validate_selected_model"],
        "required_for_modes": ["selection", "full"],
        "critical": False,
        "description": "Validate selected live prototype model.",
    },
    {
        "name": "generate_tracking_report",
        "command": ["python", "-m", "src.tracking.generate_tracking_report"],
        "required_for_modes": ["tracking", "full"],
        "critical": False,
        "description": "Create dataset version and experiment run tracking reports.",
    },
]

SUPPORTED_MODES = [
    "validate",
    "preprocess",
    "split",
    "baselines",
    "sequences",
    "neural",
    "training",
    "reports",
    "selection",
    "tracking",
    "full",
]
