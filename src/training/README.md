# Subject-Independent Dataset Splitting

This step creates train/validation/test splits for future model training.

The split is performed by `participant_id`, not by individual windows or trials. This prevents data leakage where windows from the same participant appear in both training and testing data.

Run from the project root:

```bash
python -m src.training.create_splits
```

## Output files

- `data/processed/train_windows.csv`
- `data/processed/validation_windows.csv`
- `data/processed/test_windows.csv`
- `data/processed/train_window_features.csv`
- `data/processed/validation_window_features.csv`
- `data/processed/test_window_features.csv`
- `data/processed/split_report.txt`

For a scientifically meaningful train/validation/test split, at least 3 participants are required. For stronger evaluation, 30+ participants are preferred.

This step only prepares split datasets. It does not train any model.

## Step 8: Neural Sequence Dataset Preparation

This step converts subject-independent split windows into fixed-length time-series tensors for future LSTM, GRU, and Causal TCN models.

Run from the project root:

```bash
python -m src.training.build_sequence_dataset
```

## Sequence output files

- `data/processed/sequences/train_sequences.npz`
- `data/processed/sequences/validation_sequences.npz`
- `data/processed/sequences/test_sequences.npz`
- `data/processed/sequences/train_sequence_metadata.csv`
- `data/processed/sequences/validation_sequence_metadata.csv`
- `data/processed/sequences/test_sequence_metadata.csv`
- `data/processed/sequences/sequence_feature_columns.json`
- `data/processed/sequences/sequence_scaler.joblib`
- `data/processed/sequences/sequence_dataset_report.txt`

The `.npz` files contain:

- `X`: sequence tensor with shape `[windows, time_steps, features]`
- `y`: window labels
- `masks`: real sample mask where padded rows are 0
- `window_ids`
- `participant_ids`
- `trial_ids`
- `valid_ratios`
- `original_lengths`

This step prepares data only. It does not train LSTM, GRU, TCN, Transformer, or real-time inference models.
