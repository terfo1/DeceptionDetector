# Data Collection Workflow

This module collects controlled eye-tracking experiment data.

## Anonymous Participant IDs

Use anonymous participant IDs such as `P001`. Do not store real names, email addresses, phone numbers, addresses, national IDs, or other direct identifiers.

## Consent Note

The experiment app includes a consent and ethics note. Participants must confirm that they understand the study collects anonymous eye-tracking data for a controlled deception-risk prototype and that it is not a universal lie detector.

## Participant Metadata

Anonymous metadata is stored in:

```text
data/raw/participant_metadata.csv
```

## Calibration Status

The current app records mock calibration status. Real calibration integration will be added only after a real eye tracker SDK is connected.

## Baseline Recording

The app includes a neutral fixation baseline placeholder and records baseline duration metadata.

## Trial Recording

Core trial and gaze data continue to use the existing raw CSV files:

- `data/raw/participants.csv`
- `data/raw/sessions.csv`
- `data/raw/trials.csv`
- `data/raw/gaze_samples.csv`

## Session Quality Report

Quality metadata and reports are written to:

- `data/raw/session_quality.csv`
- `reports/data_collection/latest_session_report.txt`
- `reports/data_collection/latest_session_summary.md`

## Metadata Files

Additional append-only metadata files:

- `data/raw/participant_metadata.csv`
- `data/raw/session_metadata.csv`
- `data/raw/session_quality.csv`

## Commands

Generate the data collection checklist:

```bash
python -m src.data_collection.collection_checklist
```

Run the experiment app:

```bash
python -m src.data_collection.experiment_app
```
