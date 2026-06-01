# Gaze Source Adapters

This module provides a unified interface for different gaze data sources.

Available adapters:

- `MockGazeAdapter`
- `RecordedCsvGazeAdapter`
- `WebcamGazeAdapter` placeholder
- `RealEyeTrackerAdapter` placeholder

Unified sample schema:

```text
timestamp, gaze_x, gaze_y, pupil_left, pupil_right, blink, fixation, saccade, validity
```

Start API:

```bash
uvicorn src.api.app:app --reload
```

Run mock adapter stream:

```bash
python -m src.gaze_sources.stream_to_api --source mock --duration-seconds 10
```

Run recorded CSV adapter:

```bash
python -m src.gaze_sources.stream_to_api --source recorded_csv --max-trials 3
```

Run one trial:

```bash
python -m src.gaze_sources.stream_to_api --source recorded_csv --trial-id T001
```

Webcam and real eye tracker adapters are placeholders for future integration.
