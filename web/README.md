# Live Monitor Dashboard

This is a minimal HTML/JavaScript dashboard for monitoring the live WebSocket inference service.

It is a prototype monitor. It does not perform real eye tracking. It can send mock samples to test the WebSocket flow.

## How to Use

1. Start API:

```bash
uvicorn src.api.app:app --reload
```

2. Open:

```text
web/live_monitor.html
```

Or, if the API is running:

```text
http://127.0.0.1:8000/static/live_monitor.html
```

3. Click:

- Connect
- Start Session
- Start Trial
- Start Mock Stream

4. Watch:

- probability
- smoothed probability
- risk category
- latency
- prediction history

The dashboard connects to:

```text
ws://127.0.0.1:8000/ws/live
```
