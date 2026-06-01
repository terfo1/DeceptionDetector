# Real-Time Eye-Tracking-Based Deception Risk Detection Using Neural Networks

This project is for developing a machine learning system that estimates deception risk from real-time eye-tracking data.

Current project stage: dataset structure design and simple experimental data collection.

At this stage, the project contains dataset documentation, project folders, sample CSV files, and a simple local experiment runner. Model training, real-time inference, API code, frontend code, and live eye-tracking integration will be implemented in later steps.

## Folder structure

```text
data/
  raw/          Raw dataset CSV files
  processed/    Processed datasets created in later steps
  samples/      Example CSV files showing the expected raw format
docs/           Project and dataset documentation
src/            Future source code modules
notebooks/      Future exploratory notebooks
reports/        Future analysis reports
```

## Dataset documentation

The raw dataset format is described in `docs/dataset_format.md`.

## Step 4: Experiment Interface

The project includes a simple Tkinter-based experiment runner for collecting controlled statement-level deception task data. It collects participant, session, trial, and gaze sample rows using the raw CSV format documented in `docs/dataset_format.md`.

The current version uses a mock eye tracker so the interface can be tested without external hardware. Real eye tracker integration will be added later.

Run the app from the project root:

```bash
python -m src.data_collection.experiment_app
```

Raw data is saved to:

```text
data/raw/participants.csv
data/raw/sessions.csv
data/raw/trials.csv
data/raw/gaze_samples.csv
```

The app does not include model training, neural network code, API code, frontend code, or real-time inference.

## Step 5: Preprocessing and Windowing

Raw experiment data can now be validated and converted into processed sliding-window datasets. The preprocessing code checks raw CSV structure, validates dataset relationships, cleans gaze samples, calculates `pupil_mean` and `gaze_velocity`, creates sliding windows, and saves aggregated window features.

This step prepares data for future ML training. No model training, neural network code, API code, frontend code, or real-time inference is implemented yet.

Validate raw data:

```bash
python -m src.preprocessing.validate_raw_data
```

Build processed windows:

```bash
python -m src.preprocessing.build_windows
```

Processed outputs are saved to:

```text
data/processed/windows.csv
data/processed/window_features.csv
data/processed/preprocessing_report.txt
```

## Step 6: Subject-Independent Train/Validation/Test Split

Processed window data can now be split for future ML training. Splitting is done by `participant_id`, not by `window_id` or `trial_id`, so the same participant cannot appear in more than one split. This avoids leakage between train and test data.

No model training, neural network code, API code, frontend code, or real-time inference is implemented yet.

Create subject-independent splits:

```bash
python -m src.training.create_splits
```

Split outputs are saved to:

```text
data/processed/train_windows.csv
data/processed/validation_windows.csv
data/processed/test_windows.csv
data/processed/train_window_features.csv
data/processed/validation_window_features.csv
data/processed/test_window_features.csv
data/processed/split_report.txt
```

## Step 7: Baseline ML Models

Baseline models can now be trained on aggregated window features. This step uses Logistic Regression and Random Forest classifiers to check whether engineered eye-tracking features contain useful signal before future neural sequence models are implemented.

These baselines are not final neural sequence models and must not be interpreted as a universal lie detector. They estimate deception risk only within the controlled experimental pipeline, and the split remains subject-independent.

Train baselines:

```bash
python -m src.models.train_baselines
```

Evaluate baselines on the test split:

```bash
python -m src.models.evaluate_baselines
```

Baseline outputs are saved to:

```text
models/baselines/
reports/baselines/
```

## Final Report Generation

Generate the final Word technical/research report:

```bash
python scripts/generate_final_report.py
```

The report is saved to:

```text
reports/final_report/eye_tracking_deception_report.docx
```

If local PDF conversion support is available, the script also creates:

```text
reports/final_report/eye_tracking_deception_report.pdf
```
