# Dataset Guide

## Raw Dataset Files

- `data/raw/participants.csv`
- `data/raw/participant_metadata.csv`
- `data/raw/sessions.csv`
- `data/raw/session_metadata.csv`
- `data/raw/trials.csv`
- `data/raw/gaze_samples.csv`
- `data/raw/session_quality.csv`

## Processed Dataset Files

- `data/processed/windows.csv`
- `data/processed/window_features.csv`
- `data/processed/train_windows.csv`
- `data/processed/validation_windows.csv`
- `data/processed/test_windows.csv`
- `data/processed/train_window_features.csv`
- `data/processed/validation_window_features.csv`
- `data/processed/test_window_features.csv`
- `data/processed/sequences/train_sequences.npz`
- `data/processed/sequences/validation_sequences.npz`
- `data/processed/sequences/test_sequences.npz`

## Label Rule

- `0` = truthful response
- `1` = deceptive response

The label is based on the experimental instruction, not on a universal truth-detection claim.

## Why Participant-Independent Splits Matter

The same participant must not appear in train and test splits. Subject-independent splitting reduces leakage and gives a more realistic estimate of generalization to unseen participants.
