# Architecture

## Offline ML Pipeline

```text
raw data
-> preprocessing
-> windows
-> subject-independent split
-> baseline models
-> sequence dataset
-> LSTM/GRU/TCN
-> comparison
-> model selection
-> tracking
```

## Live Inference Pipeline

```text
gaze source adapter
-> WebSocket API
-> live buffer
-> selected model
-> risk score
-> dashboard/logs
```

## Module Structure

### `src/data_collection`

Collects controlled experiment data and anonymous participant/session metadata.

Key files: `experiment_app.py`, `data_writer.py`, `participant_metadata.py`, `session_quality.py`, `session_report.py`.

Outputs: `data/raw/*.csv`, `reports/data_collection/*`.

### `src/preprocessing`

Validates raw CSV files, cleans gaze samples, and builds sliding windows.

Outputs: `data/processed/windows.csv`, `data/processed/window_features.csv`.

### `src/training`

Creates subject-independent splits and fixed-length sequence tensors.

Outputs: split CSV files and `data/processed/sequences/*.npz`.

### `src/models`

Trains and evaluates baseline, recurrent, and Causal TCN models.

Outputs: model artifacts under `models/` and metrics under `reports/`.

### `src/analysis`

Compares models, diagnoses prediction collapse, and analyzes thresholds.

Outputs: `reports/model_comparison/` and `reports/threshold_calibration/`.

### `src/realtime`

Runs replay simulation, model selection, and live prediction helpers.

Outputs: `reports/realtime_simulation/` and `reports/model_selection/`.

### `src/api`

Provides the FastAPI/WebSocket live inference service.

Outputs: `reports/live_inference/`.

### `src/gaze_sources`

Defines gaze source adapters for mock, recorded CSV, webcam placeholder, and real eye-tracker placeholder sources.

Outputs: `reports/gaze_sources/`.

### `src/tracking`

Creates dataset version and experiment run manifests.

Outputs: `reports/tracking/`.

### `src/pipeline`

Runs existing pipeline stages in one command and records pass/fail/skipped status.

Outputs: `reports/pipeline/`.
