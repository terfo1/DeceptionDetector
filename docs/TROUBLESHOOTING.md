# Troubleshooting

## API Does Not Start

- Check that `uvicorn` is installed.
- Validate selected model files:

```bash
python -m src.realtime.validate_selected_model
```

## WebSocket Client Cannot Connect

- Start the API first:

```bash
uvicorn src.api.app:app --reload
```

- Check that the URL is `ws://127.0.0.1:8000/ws/live`.

## Validation/Test Split Is Empty

Collect at least 3 participants. Subject-independent splitting needs enough participants to populate train, validation, and test splits.

## LSTM Predicts Only Zeros

This is single-class prediction behavior. The current dataset is too small; collect more data and retrain from scratch.

## All Predictions Are Medium

The model probability range may be narrow. Run:

```bash
python -m src.analysis.generate_threshold_report
```

Then collect more data before treating thresholds as reliable.

## FastAPI Says Model Unavailable

Check:

```text
reports/model_selection/model_registry_status.csv
```

Then rerun:

```bash
python -m src.realtime.validate_selected_model
```

## Pipeline Fails

Open:

```text
reports/pipeline/pipeline_run_report.txt
```

Fix the first failed critical step and rerun the pipeline.
