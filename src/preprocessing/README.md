# Preprocessing Pipeline

Raw gaze samples are converted into sliding windows for future ML models. This step prepares structured processed data, but it does not train a model.

## Pipeline

1. Validate raw CSV files.
2. Clean gaze samples.
3. Calculate `pupil_mean` and `gaze_velocity`.
4. Create sliding windows.
5. Calculate aggregated window features.
6. Save processed CSV files.

## Commands

Validate raw data:

```bash
python -m src.preprocessing.validate_raw_data
```

Build windows:

```bash
python -m src.preprocessing.build_windows
```

## Output files

- `data/processed/windows.csv`
- `data/processed/window_features.csv`
- `data/processed/preprocessing_report.txt`

The processed files are generated from `data/raw` and can be regenerated later. Raw data is not overwritten.
