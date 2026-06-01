# Quick Start

## 1. Install Requirements

```bash
pip install -r requirements.txt
```

## 2. Run Pipeline Dry Run

```bash
python -m src.pipeline.run_pipeline --mode full --dry-run
```

## 3. Start API

```bash
uvicorn src.api.app:app --reload
```

## 4. Check Status

Open:

```text
http://127.0.0.1:8000/status
```

## 5. Open Dashboard

```text
web/live_monitor.html
```

## 6. Run WebSocket Client

```bash
python scripts/live_ws_test_client.py
```
