# Reports Index

## `reports/baselines`

Purpose: baseline model metrics and predictions.

Key files: `baseline_metrics.csv`, `baseline_report.txt`.

Regenerate after training/evaluating baseline models.

## `reports/sequences`

Purpose: LSTM, GRU, and Causal TCN metrics, predictions, and reports.

Regenerate after neural model evaluation.

## `reports/model_comparison`

Purpose: compare Logistic Regression, Random Forest, LSTM, GRU, and Causal TCN.

Regenerate with:

```bash
python -m src.analysis.generate_model_comparison
```

## `reports/threshold_calibration`

Purpose: probability diagnostics and threshold sweeps.

Regenerate with:

```bash
python -m src.analysis.generate_threshold_report
```

## `reports/realtime_simulation`

Purpose: offline replay simulation results.

## `reports/model_selection`

Purpose: selected model, fallback model, disabled models, and registry status.

## `reports/live_inference`

Purpose: FastAPI live inference logs.

## `reports/live_ws_client`

Purpose: WebSocket client end-to-end test output.

## `reports/gaze_sources`

Purpose: gaze source adapter streaming test output.

## `reports/data_collection`

Purpose: collection checklist, session quality summaries, and recommendations.

## `reports/tracking`

Purpose: dataset versions, run manifests, file hashes, and reproducibility checklists.

Regenerate with:

```bash
python -m src.tracking.generate_tracking_report
```

## `reports/pipeline`

Purpose: one-command pipeline execution logs.

Regenerate with:

```bash
python -m src.pipeline.run_pipeline --mode full --dry-run
```
