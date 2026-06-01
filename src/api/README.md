# Live Inference API

This module provides a FastAPI/WebSocket prototype for live deception-risk inference.

This service is not a universal lie detector. It returns risk scores under a controlled experimental protocol.

Run:

```bash
uvicorn src.api.app:app --reload
```

Endpoints:

- `GET /`
- `GET /health`
- `GET /status`
- `POST /predict/window`
- `WebSocket /ws/live`

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict/window ^
  -H "Content-Type: application/json" ^
  -d "{\"samples\":[{\"timestamp\":0.0,\"gaze_x\":0.5,\"gaze_y\":0.5,\"pupil_left\":3.2,\"pupil_right\":3.1,\"blink\":0,\"fixation\":1,\"saccade\":0,\"validity\":1}]}"
```

Prediction with PowerShell:

```powershell
$body = @{
  samples = @(
    @{
      timestamp = 0.0
      gaze_x = 0.5
      gaze_y = 0.5
      pupil_left = 3.2
      pupil_right = 3.1
      blink = 0
      fixation = 1
      saccade = 0
      validity = 1
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/window" -Method POST -Body $body -ContentType "application/json"
```

WebSocket `/ws/live` accepts `start_session`, `start_trial`, `sample`, `end_trial`, `end_session`, and `reset` messages.

## WebSocket Test Client

The test client replays recorded gaze samples through the live WebSocket endpoint.

Before running the client, start the API:

```bash
uvicorn src.api.app:app --reload
```

Then run:

```bash
python scripts/live_ws_test_client.py
```

Examples:

```bash
python scripts/live_ws_test_client.py --max-trials 3
python scripts/live_ws_test_client.py --trial-id T001
python scripts/live_ws_test_client.py --real-time-sleep --sleep-scale 0.5
```

Outputs:

- `reports/live_ws_client/ws_client_predictions.csv`
- `reports/live_ws_client/ws_client_trial_summary.csv`
- `reports/live_ws_client/ws_client_report.txt`
- `reports/live_ws_client/ws_client_summary.md`
