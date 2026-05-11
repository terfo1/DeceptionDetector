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
