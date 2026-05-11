# Baseline Models

This step trains simple baseline classifiers on aggregated eye-tracking window features.

Models:

1. Logistic Regression
2. Random Forest

Purpose:

Baseline models are used to check whether engineered window-level features contain useful signal before implementing neural sequence models.

Important:

These models do not process raw gaze sequences directly. They use aggregated features from `window_features.csv`.

Train baselines:

```bash
python -m src.models.train_baselines
```

Evaluate baselines on test split:

```bash
python -m src.models.evaluate_baselines
```

Outputs:

- `models/baselines/logistic_regression.joblib`
- `models/baselines/random_forest.joblib`
- `models/baselines/scaler.joblib`
- `models/baselines/feature_columns.json`
- `reports/baselines/baseline_metrics.csv`
- `reports/baselines/baseline_report.txt`
- `reports/baselines/validation_predictions.csv`
- `reports/baselines/test_predictions.csv`

Metrics:

- accuracy
- balanced accuracy
- precision
- recall
- F1
- ROC-AUC
- average precision
- false positive rate
- false negative rate

False positive rate is important because falsely marking a truthful response as deceptive is harmful.
