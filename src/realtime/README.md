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
