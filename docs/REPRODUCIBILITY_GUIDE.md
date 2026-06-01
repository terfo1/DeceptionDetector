# Reproducibility Guide

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Use existing sample data or collect new anonymous participant data with the experiment app.

## Run Full Pipeline

```bash
python -m src.pipeline.run_pipeline --mode full
```

## Generate Tracking Report

```bash
python -m src.tracking.generate_tracking_report
```

## Inspect Reports

Review:

- `reports/pipeline/`
- `reports/model_comparison/`
- `reports/threshold_calibration/`
- `reports/model_selection/`
- `reports/tracking/`

## Dataset and Run Versions

Tracking artifacts include dataset version, run version, file hashes, model metrics, and selected model information. These files help compare future runs after adding participants.
