# Commands Cheatsheet

## Data Collection

```bash
python -m src.data_collection.collection_checklist
python -m src.data_collection.experiment_app
```

## Preprocessing

```bash
python -m src.preprocessing.validate_raw_data
python -m src.preprocessing.build_windows
python -m src.training.create_splits
```

## Training

```bash
python -m src.models.train_baselines
python -m src.training.build_sequence_dataset
python -m src.models.train_sequence_models
python -m src.models.train_tcn_model
```

## Evaluation

```bash
python -m src.models.evaluate_baselines
python -m src.models.evaluate_sequence_models
python -m src.models.evaluate_tcn_model
python -m src.analysis.generate_model_comparison
python -m src.analysis.generate_threshold_report
```

## Real-Time Simulation

```bash
python -m src.realtime.run_realtime_simulation --model-type random_forest
python -m src.realtime.run_selected_model_simulation
```

## API

```bash
uvicorn src.api.app:app --reload
```

## Dashboard

```text
web/live_monitor.html
```

## Tracking

```bash
python -m src.tracking.generate_tracking_report
```

## Pipeline

```bash
python -m src.pipeline.run_pipeline --mode full
python -m src.pipeline.run_pipeline --mode full --dry-run
```
