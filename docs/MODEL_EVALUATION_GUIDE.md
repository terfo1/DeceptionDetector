# Model Evaluation Guide

## Metrics

- Accuracy: fraction of correct predictions.
- Balanced accuracy: average recall across classes.
- Precision: fraction of predicted deceptive windows that are truly deceptive.
- Recall: fraction of deceptive windows detected.
- F1: harmonic mean of precision and recall.
- ROC-AUC: ranking quality across thresholds.
- Average precision: precision-recall summary.
- False positive rate: truthful response predicted as deceptive.
- False negative rate: deceptive response predicted as truthful.
- Latency: inference time for live or simulated predictions.

## Current Diagnostics

- `random_forest` is selected for the current live prototype.
- `gru` is the fallback neural sequence model.
- `lstm` collapsed to single-class prediction and is disabled.
- `causal_tcn` remains experimental because its probability range is currently narrow.
- Thresholds are preliminary and should be recalibrated after collecting more participants.

## Commands

```bash
python -m src.analysis.generate_model_comparison
python -m src.analysis.generate_threshold_report
```
