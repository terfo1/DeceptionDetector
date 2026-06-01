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
