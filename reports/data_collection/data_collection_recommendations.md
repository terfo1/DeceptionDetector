# Data Collection Checklist

## 1. Before Participant Arrives
- prepare laptop
- prepare lighting
- close unnecessary apps
- start API only if live inference is needed
- verify storage space
- verify project environment

## 2. Participant Setup
- explain experiment purpose
- explain that it is not a lie detector
- confirm consent
- assign anonymous participant_id
- do not store real name

## 3. Calibration
- check seating position
- check screen distance
- run calibration
- record calibration status

## 4. Baseline
- record neutral fixation baseline
- avoid strong lighting changes
- avoid distractions

## 5. Experiment
- run truth/lie trials
- avoid helping participant during answers
- record session notes if something unusual happens

## 6. After Session
- run session quality report
- check valid_ratio
- check completed trials
- backup raw CSV files
- do not edit raw CSV files manually

## 7. Minimum Dataset Targets
- technical test: 3 participants
- prototype demo: 5-10 participants
- stronger research version: 30+ participants

## 8. Rerun Pipeline After New Data

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
```
