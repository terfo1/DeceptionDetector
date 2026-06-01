# Threshold Calibration Recommendations

## Why thresholds are needed
Model probabilities need decision thresholds before they can be mapped into risk categories. The same probability range may behave differently across model families.

## Why current 0.40/0.70 bands failed for Causal TCN
The Causal TCN currently produces probabilities in a narrow band near 0.5. Under 0.40/0.70 thresholds, nearly all replay predictions become medium, so the bands are not informative for that model.

## How to recalibrate after more data
Use validation data for threshold selection, preserve test data for final evaluation, and recalibrate after retraining all models from scratch.

## Which model to use for next real-time simulation
Use random_forest as the default because it currently has the strongest test F1. Also compare GRU because it is the strongest current neural sequence model.

## Commands to rerun after collecting more data

```bash
python -m src.preprocessing.validate_raw_data
python -m src.preprocessing.build_windows
python -m src.training.create_splits
python -m src.models.train_baselines
python -m src.models.evaluate_baselines
python -m src.training.build_sequence_dataset
python -m src.models.train_sequence_models
python -m src.models.evaluate_sequence_models
python -m src.models.train_tcn_model
python -m src.models.evaluate_tcn_model
python -m src.analysis.generate_model_comparison
python -m src.realtime.run_realtime_simulation --model-type random_forest
python -m src.realtime.run_realtime_simulation --model-type gru
python -m src.analysis.generate_threshold_report
```

Do not automatically deploy the preliminary thresholds. They are calibration artifacts only.
