# Reproducibility Checklist

## Raw Data
- [x] data/raw/participants.csv
- [x] data/raw/participant_metadata.csv
- [x] data/raw/sessions.csv
- [x] data/raw/session_metadata.csv
- [x] data/raw/trials.csv
- [x] data/raw/gaze_samples.csv
- [x] data/raw/session_quality.csv

## Processed Data
- [x] data/processed/windows.csv
- [x] data/processed/window_features.csv
- [x] data/processed/train_windows.csv
- [x] data/processed/validation_windows.csv
- [x] data/processed/test_windows.csv
- [x] data/processed/train_window_features.csv
- [x] data/processed/validation_window_features.csv
- [x] data/processed/test_window_features.csv
- [x] data/processed/sequences/train_sequences.npz
- [x] data/processed/sequences/validation_sequences.npz
- [x] data/processed/sequences/test_sequences.npz

## Splits
- [x] data/processed/train_windows.csv
- [x] data/processed/validation_windows.csv
- [x] data/processed/test_windows.csv
- [x] data/processed/split_report.txt

## Model Artifacts
- [ ] Confirm model files under models/baselines and models/sequences match the tracked reports.
- [ ] Confirm selected live model files are validated by Step 14.

## Reports
- [x] data/processed/preprocessing_report.txt
- [x] data/processed/split_report.txt
- [x] data/processed/sequences/sequence_dataset_report.txt
- [x] reports/baselines/baseline_metrics.csv
- [x] reports/sequences/sequence_model_metrics.csv
- [x] reports/sequences/tcn_metrics.csv
- [x] reports/model_comparison/model_ranking.csv
- [x] reports/model_comparison/model_comparison_report.txt
- [x] reports/threshold_calibration/selected_thresholds.json
- [x] reports/model_selection/selected_model_config.json
- [x] reports/model_selection/model_selection_report.txt

## Commands to Reproduce
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
python -m src.analysis.generate_threshold_report
python -m src.realtime.validate_selected_model
python -m src.tracking.generate_tracking_report
```
