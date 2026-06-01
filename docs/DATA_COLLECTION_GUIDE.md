# Data Collection Guide

## Before Data Collection

- Prepare the laptop, stable lighting, and enough storage.
- Close unnecessary applications.
- Generate the checklist:

```bash
python -m src.data_collection.collection_checklist
```

## Participant Setup

Use anonymous participant IDs only. Do not collect real names, email addresses, phone numbers, addresses, or direct identifiers.

## Consent and Ethics Note

Before starting, explain that the experiment collects anonymous eye-tracking data for a controlled deception-risk research prototype. The system does not determine lying with certainty and must not be used as an autonomous lie detector.

## Anonymous Participant ID

Assign an ID such as `P001`. Store only minimal metadata such as age group, vision status, and glasses/contact lens status.

## Calibration Status

The current app records calibration status as a prototype field. Real hardware calibration is future work.

## Baseline Recording

Record a short neutral baseline. Avoid lighting changes and distractions during the session.

## Running the Experiment App

```bash
python -m src.data_collection.experiment_app
```

## Session Quality Report

After a session, generate or review:

- `data/raw/session_quality.csv`
- `reports/data_collection/latest_session_report.txt`
- `reports/data_collection/latest_session_summary.md`

## Minimum Dataset Targets

- Technical test: 3 participants
- Prototype demo: 5-10 participants
- Stronger research version: 30+ participants
