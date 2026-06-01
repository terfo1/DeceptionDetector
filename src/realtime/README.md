# Real-Time Simulation

This module replays recorded eye-tracking data as if it were arriving online.

It uses:

- a rolling buffer
- causal preprocessing
- a trained model
- a risk decision policy

Run default random forest simulation:

```bash
python -m src.realtime.run_realtime_simulation --model-type random_forest
```

Run GRU simulation:

```bash
python -m src.realtime.run_realtime_simulation --model-type gru
```

Run Causal TCN simulation:

```bash
python -m src.realtime.run_realtime_simulation --model-type causal_tcn
```

Outputs:

- `reports/realtime_simulation/realtime_predictions.csv`
- `reports/realtime_simulation/realtime_trial_summary.csv`
- `reports/realtime_simulation/realtime_simulation_report.txt`
- `reports/realtime_simulation/realtime_simulation_summary.md`

This is not live deployment yet. It is an offline replay simulation using recorded samples.

## Step 14: Model Selection for Live Prototype

This step selects the current model for the live prototype based on previous diagnostics.

Current selection:

- primary model: `random_forest`
- fallback model: `gru`
- disabled model: `lstm`
- experimental model: `causal_tcn`

Validate selected model:

```bash
python -m src.realtime.validate_selected_model
```

Run selected model simulation:

```bash
python -m src.realtime.run_selected_model_simulation
```

Outputs:

- `reports/model_selection/selected_model_config.json`
- `reports/model_selection/model_registry_status.csv`
- `reports/model_selection/model_selection_report.txt`
- `reports/model_selection/model_selection_summary.md`
- `reports/model_selection/selected_model_realtime_predictions.csv`
- `reports/model_selection/selected_model_trial_summary.csv`
- `reports/model_selection/selected_model_simulation_summary.txt`
