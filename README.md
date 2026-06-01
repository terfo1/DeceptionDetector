# Real-Time Eye-Tracking-Based Deception Risk Detection

This project estimates deception-risk patterns from eye-tracking signals under a controlled experimental protocol. It includes data collection, preprocessing, model training, model diagnostics, live inference, a minimal dashboard, gaze source adapters, run tracking, and a one-command pipeline runner.

Important: this system is not a universal lie detector. It outputs risk scores for research/prototype use, not final truth/lie judgments.

## Current Status

Implemented through Step 22:

- Tkinter experiment/data collection workflow
- Raw data validation and preprocessing
- Subject-independent train/validation/test split
- Baseline Logistic Regression and Random Forest models
- LSTM, GRU, and Causal TCN sequence models
- Model comparison and threshold diagnostics
- Real-time replay simulation
- Model selection for live prototype
- FastAPI/WebSocket live inference service
- WebSocket test client
- Minimal HTML/JavaScript live monitor
- Gaze source adapter interface
- Data collection quality reports
- Dataset versioning and run tracking
- One-command pipeline runner
- Final documentation and handoff package

## Architecture Overview

Offline pipeline:

```text
raw data -> preprocessing -> windows -> subject-independent split
-> baseline models -> sequence datasets -> LSTM/GRU/TCN
-> comparison -> model selection -> tracking
```

Live prototype:

```text
gaze source adapter -> WebSocket API -> live buffer
-> selected model -> risk score -> dashboard/logs
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Selected Model

- Primary model: `random_forest`
- Fallback model: `gru`
- Disabled model: `lstm`
- Experimental model: `causal_tcn`

Random Forest is selected for the current prototype because it has the strongest available prototype-level test F1 and a wider probability distribution. GRU remains the fallback neural sequence model.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a pipeline dry run:

```bash
python -m src.pipeline.run_pipeline --mode full --dry-run
```

Start the API:

```bash
uvicorn src.api.app:app --reload
```

Check API status:

```text
http://127.0.0.1:8000/status
```

Open the dashboard:

```text
web/live_monitor.html
```

Or:

```text
http://127.0.0.1:8000/static/live_monitor.html
```

## Main Commands

Collect data:

```bash
python -m src.data_collection.collection_checklist
python -m src.data_collection.experiment_app
```

Run the full pipeline:

```bash
python -m src.pipeline.run_pipeline --mode full
```

Run reports only:

```bash
python -m src.pipeline.run_pipeline --mode reports
```

Generate tracking report:

```bash
python -m src.tracking.generate_tracking_report
```

Generate handoff package:

```bash
python scripts/create_handoff_package.py
```

## Project Structure

```text
data/                 Raw and processed datasets
docs/                 Final project guides
models/               Trained model artifacts
reports/              Metrics, diagnostics, logs, and handoff package
scripts/              Utility scripts
src/analysis/         Model comparison and threshold diagnostics
src/api/              FastAPI/WebSocket live inference service
src/data_collection/  Experiment app and session quality tools
src/gaze_sources/     Mock/recorded gaze source adapters
src/models/           Baseline and neural model training/evaluation
src/pipeline/         One-command pipeline runner
src/preprocessing/    Raw data validation and windowing
src/realtime/         Realtime predictor, simulation, and model selection
src/tracking/         Dataset versioning and run tracking
src/training/         Splits and sequence dataset preparation
web/                  Minimal live monitor dashboard
```

## Important Reports

- `reports/model_comparison/model_comparison_report.txt`
- `reports/threshold_calibration/threshold_calibration_report.txt`
- `reports/model_selection/model_selection_report.txt`
- `reports/tracking/tracking_report.txt`
- `reports/pipeline/pipeline_run_report.txt`
- `reports/handoff_package/HANDOFF_README.md`

See [docs/REPORTS_INDEX.md](docs/REPORTS_INDEX.md).

## Detailed Documentation

- [Final Project Overview](docs/FINAL_PROJECT_OVERVIEW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data Collection Guide](docs/DATA_COLLECTION_GUIDE.md)
- [Dataset Guide](docs/DATASET_GUIDE.md)
- [ML Pipeline Guide](docs/ML_PIPELINE_GUIDE.md)
- [Model Evaluation Guide](docs/MODEL_EVALUATION_GUIDE.md)
- [Live Inference Guide](docs/LIVE_INFERENCE_GUIDE.md)
- [Dashboard Guide](docs/DASHBOARD_GUIDE.md)
- [Reproducibility Guide](docs/REPRODUCIBILITY_GUIDE.md)
- [Ethics and Limitations](docs/ETHICS_AND_LIMITATIONS.md)
- [Future Work](docs/FUTURE_WORK.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Reports Index](docs/REPORTS_INDEX.md)

## Limitations

Current results are prototype-level because the dataset is small. Thresholds are preliminary. LSTM collapsed to single-class prediction, and Causal TCN is still experimental. Real deployment requires more participants, real hardware testing, calibration validation, ethics approval, and human oversight.

## Future Work

Next priorities are collecting more participants, rerunning the full pipeline, recalibrating thresholds, integrating a real eye tracker SDK, improving neural sequence models with more data, and adding production engineering only after the research pipeline is stronger.
