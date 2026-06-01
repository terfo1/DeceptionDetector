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

## Step 9: Neural Sequence Models

This step trains LSTM and GRU classifiers on fixed-length eye-tracking sequences generated in Step 8.

These models use temporal gaze patterns directly, unlike baseline models that use aggregated window-level features.

Train sequence models:

```bash
python -m src.models.train_sequence_models
```

Evaluate sequence models:

```bash
python -m src.models.evaluate_sequence_models
```

Outputs:

- `models/sequences/lstm_model.pt`
- `models/sequences/gru_model.pt`
- `models/sequences/sequence_model_config.json`
- `models/sequences/lstm_training_history.json`
- `models/sequences/gru_training_history.json`
- `reports/sequences/sequence_model_metrics.csv`
- `reports/sequences/sequence_model_report.txt`
- `reports/sequences/validation_predictions.csv`
- `reports/sequences/test_predictions.csv`

If validation or test splits are empty because there are not enough participants, the code still trains on the train split and skips unavailable validation or test metrics.

These models estimate deception risk only within the controlled experimental protocol. They are not universal lie detectors.

## Step 10: Causal TCN Model

This step trains a Causal Temporal Convolutional Network on fixed-length eye-tracking sequences.

LSTM and GRU models learn temporal patterns recurrently. A Causal TCN uses temporal convolutions with causal padding, which is useful for future real-time inference because predictions avoid future-sample leakage. This is still an offline training step, not real-time deployment.

Train Causal TCN:

```bash
python -m src.models.train_tcn_model
```

Evaluate Causal TCN:

```bash
python -m src.models.evaluate_tcn_model
```

Outputs:

- `models/sequences/tcn_model.pt`
- `models/sequences/tcn_training_history.json`
- `models/sequences/tcn_model_config.json`
- `reports/sequences/tcn_metrics.csv`
- `reports/sequences/tcn_report.txt`
- `reports/sequences/tcn_validation_predictions.csv`
- `reports/sequences/tcn_test_predictions.csv`

If validation or test splits are empty because there are not enough participants, the code still trains on the train split and skips unavailable validation or test metrics.
