# Dashboard Guide

## Start API

```bash
uvicorn src.api.app:app --reload
```

## Open Dashboard

Open directly:

```text
web/live_monitor.html
```

Or through FastAPI static serving:

```text
http://127.0.0.1:8000/static/live_monitor.html
```

## Features

- Connect to the WebSocket endpoint.
- Start a session and trial.
- Send mock samples for testing.
- Display probability and smoothed probability.
- Display risk category, valid ratio, sample count, and latency.
- Show recent prediction history and event logs.

The dashboard is a local prototype monitor, not a production frontend.
