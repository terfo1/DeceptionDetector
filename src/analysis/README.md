# Model Comparison and Diagnostics

This module compares baseline models, recurrent sequence models, and the Causal TCN model.

Run from the project root:

```bash
python -m src.analysis.generate_model_comparison
```

Outputs:

- `reports/model_comparison/model_comparison_metrics.csv`
- `reports/model_comparison/prediction_diagnostics.csv`
- `reports/model_comparison/model_ranking.csv`
- `reports/model_comparison/model_comparison_report.txt`
- `reports/model_comparison/model_comparison_summary.md`
- `reports/model_comparison/recommendations.md`

The report detects single-class prediction behavior such as the LSTM predicting only class 0. This step reads existing reports and prediction files only; it does not train models.

## Step 13: Threshold Calibration

This step analyzes probability outputs and tests decision thresholds. It does not train models and does not update production thresholds automatically.

Run from the project root:

```bash
python -m src.analysis.generate_threshold_report
```

Outputs:

- `reports/threshold_calibration/threshold_sweep.csv`
- `reports/threshold_calibration/model_probability_diagnostics.csv`
- `reports/threshold_calibration/selected_thresholds.json`
- `reports/threshold_calibration/threshold_calibration_report.txt`
- `reports/threshold_calibration/threshold_calibration_summary.md`
- `reports/threshold_calibration/recommendations.md`
