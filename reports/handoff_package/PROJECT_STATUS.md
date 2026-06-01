# Project Status

## Completed Scope

Steps 1-21 are implemented: project definition, protocol, data format, experiment app, preprocessing, subject-independent split, baseline models, neural sequence models, Causal TCN, diagnostics, replay simulation, threshold calibration, model selection, FastAPI/WebSocket API, WebSocket client, live monitor, gaze adapters, data collection workflow, tracking, and pipeline runner.

## Current Model Selection

- Primary model: `random_forest`
- Fallback model: `gru`
- Disabled models: `lstm`
- Experimental models: `causal_tcn`

## API Status

The live inference API is implemented with FastAPI and WebSocket endpoints. It should be started locally with `uvicorn src.api.app:app --reload`.

## Dashboard Status

The minimal HTML/JavaScript dashboard is available at `web/live_monitor.html` and can also be served from `/static/live_monitor.html` when the API is running.

## Current Limitations

The dataset is small, metrics are prototype-level, thresholds are preliminary, and no real eye tracker SDK is integrated yet.
