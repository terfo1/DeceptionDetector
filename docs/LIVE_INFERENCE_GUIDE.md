# Live Inference Guide

## Start the API

```bash
uvicorn src.api.app:app --reload
```

## Endpoints

- `GET /`
- `GET /health`
- `GET /status`
- `POST /predict/window`
- `WebSocket /ws/live`

The selected model is loaded from the Step 14 model selection configuration. The current primary model is `random_forest`.

Live logs are written to:

```text
reports/live_inference/
```

## PowerShell Examples

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Status:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/status"
```

The API returns deception-risk scores under a controlled protocol. It is not a universal lie detector.
