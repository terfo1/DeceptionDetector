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
