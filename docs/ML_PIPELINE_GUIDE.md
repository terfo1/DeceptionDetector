# ML Pipeline Guide

## Step-by-Step Commands

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
```

## One-Command Runner

```bash
python -m src.pipeline.run_pipeline --mode full
```

## Model Inputs

Baseline models use aggregated window features such as gaze means, blink rate, fixation rate, and velocity statistics.

Neural sequence models use tensors shaped like:

```text
[number_of_windows, time_steps, number_of_features]
```

## Current Interpretation

The current results validate the technical pipeline. They are not sufficient for strong scientific claims because the dataset is still small.
