"""Stream gaze samples from an adapter into the live WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import websockets
from websockets.exceptions import InvalidURI, WebSocketException

from .adapter_factory import create_gaze_adapter
from .base import validate_gaze_sample


PREDICTION_COLUMNS = [
    "received_at",
    "source_type",
    "session_id",
    "trial_id",
    "model_type",
    "timestamp",
    "probability",
    "smoothed_probability",
    "risk_category",
    "valid_ratio",
    "sample_count",
    "latency_ms",
    "status",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream gaze source adapter samples to WebSocket API.")
    parser.add_argument("--source", choices=["mock", "recorded_csv", "webcam", "real_eye_tracker"], default="mock")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/live")
    parser.add_argument("--session-id", default="ADAPTER_SESSION_001")
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--sampling-rate", type=int, default=60)
    parser.add_argument("--real-time-sleep", action="store_true")
    parser.add_argument("--output-dir", default="reports/gaze_sources")
    args = parser.parse_args()

    try:
        asyncio.run(run_stream(args))
    except (ConnectionRefusedError, OSError, WebSocketException, InvalidURI):
        print("Could not connect to WebSocket API. Start it with: uvicorn src.api.app:app --reload")
    except (FileNotFoundError, ValueError, NotImplementedError) as error:
        print(f"Gaze source stream failed: {error}")


async def run_stream(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_seconds = args.duration_seconds
    if args.source == "mock" and duration_seconds is None:
        duration_seconds = 10.0

    adapter = create_gaze_adapter(
        args.source,
        sampling_rate=args.sampling_rate,
        trial_id=args.trial_id,
        max_trials=args.max_trials,
    )
    predictions: list[dict] = []
    errors: list[str] = []
    samples_sent = 0
    current_trial_id = args.trial_id or "ADAPTER_TRIAL_001"

    async with websockets.connect(args.url) as websocket:
        await websocket.send(json.dumps({"type": "start_session", "session_id": args.session_id}))
        await _receive_control(websocket, errors)
        await websocket.send(json.dumps({"type": "start_trial", "trial_id": current_trial_id}))
        await _receive_control(websocket, errors)

        adapter.start()
        started = time.perf_counter()
        previous_timestamp = None
        next_prediction_timestamp = 0.0
        while adapter.is_running():
            if duration_seconds is not None and (time.perf_counter() - started) >= duration_seconds:
                break
            sample = adapter.read_sample()
            if sample is None:
                break
            if args.source == "recorded_csv" and "trial_id" in sample and sample["trial_id"] != current_trial_id:
                await websocket.send(json.dumps({"type": "end_trial"}))
                await _receive_control(websocket, errors)
                current_trial_id = sample["trial_id"]
                await websocket.send(json.dumps({"type": "start_trial", "trial_id": current_trial_id}))
                await _receive_control(websocket, errors)

            is_valid, messages = validate_gaze_sample(sample)
            warning_messages = [message for message in messages if message.startswith("Warning:")]
            errors.extend(warning_messages)
            if not is_valid:
                errors.extend([message for message in messages if not message.startswith("Warning:")])
                continue

            timestamp = float(sample["timestamp"])
            if args.real_time_sleep:
                if previous_timestamp is None:
                    await asyncio.sleep(1.0 / args.sampling_rate if args.source == "mock" else 0.0)
                else:
                    delay = max(0.0, timestamp - previous_timestamp)
                    await asyncio.sleep(delay if args.source == "recorded_csv" else 1.0 / args.sampling_rate)
            previous_timestamp = timestamp

            payload = {
                "type": "sample",
                "data": {
                    key: sample[key]
                    for key in [
                        "timestamp",
                        "gaze_x",
                        "gaze_y",
                        "pupil_left",
                        "pupil_right",
                        "blink",
                        "fixation",
                        "saccade",
                        "validity",
                    ]
                },
            }
            await websocket.send(json.dumps(payload))
            samples_sent += 1
            if timestamp + 1e-9 >= next_prediction_timestamp:
                response = await _receive_prediction(websocket, errors)
                if response and response.get("type") == "prediction":
                    predictions.append(_prediction_row(response, args.source))
                next_prediction_timestamp += 0.5

        adapter.stop()
        await websocket.send(json.dumps({"type": "end_trial"}))
        await _receive_control(websocket, errors)
        await websocket.send(json.dumps({"type": "end_session"}))
        await _receive_control(websocket, errors)

    predictions_df = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    predictions_df.to_csv(output_dir / "adapter_stream_predictions.csv", index=False)
    _write_summary(output_dir / "adapter_stream_summary.txt", args, samples_sent, predictions_df, errors)
    _write_report(output_dir / "adapter_stream_report.md", args, samples_sent, predictions_df, errors, adapter.get_metadata())

    summary = _summary(predictions_df)
    print(f"Source: {args.source}")
    print(f"Samples sent: {samples_sent}")
    print(f"Predictions received: {summary['total_predictions']}")
    print(f"Mean latency ms: {_fmt(summary['mean_latency_ms'])}")
    print(f"Errors: {len(errors)}")


async def _receive_control(websocket, errors: list[str]) -> dict | None:
    try:
        response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10.0))
    except asyncio.TimeoutError:
        errors.append("Timed out waiting for server response.")
        return None
    if response.get("type") == "error":
        errors.append(str(response.get("message", "Unknown server error.")))
    return response


async def _receive_prediction(websocket, errors: list[str]) -> dict | None:
    return await _receive_control(websocket, errors)


def _prediction_row(response: dict, source_type: str) -> dict:
    return {
        "received_at": datetime.now().isoformat(timespec="seconds"),
        "source_type": source_type,
        "session_id": response.get("session_id", ""),
        "trial_id": response.get("trial_id", ""),
        "model_type": response.get("model_type", ""),
        "timestamp": response.get("timestamp", ""),
        "probability": response.get("probability", ""),
        "smoothed_probability": response.get("smoothed_probability", ""),
        "risk_category": response.get("risk_category", ""),
        "valid_ratio": response.get("valid_ratio", ""),
        "sample_count": response.get("sample_count", ""),
        "latency_ms": response.get("latency_ms", ""),
        "status": response.get("status", ""),
    }


def _summary(predictions_df: pd.DataFrame) -> dict:
    if predictions_df.empty:
        return {
            "total_predictions": 0,
            "mean_probability": None,
            "max_probability": None,
            "mean_latency_ms": None,
            "max_latency_ms": None,
            "risk_counts": {},
        }
    return {
        "total_predictions": int(len(predictions_df)),
        "mean_probability": float(pd.to_numeric(predictions_df["probability"]).mean()),
        "max_probability": float(pd.to_numeric(predictions_df["probability"]).max()),
        "mean_latency_ms": float(pd.to_numeric(predictions_df["latency_ms"]).mean()),
        "max_latency_ms": float(pd.to_numeric(predictions_df["latency_ms"]).max()),
        "risk_counts": predictions_df["risk_category"].value_counts().sort_index().to_dict(),
    }


def _write_summary(path: Path, args, samples_sent: int, predictions_df: pd.DataFrame, errors: list[str]) -> None:
    summary = _summary(predictions_df)
    lines = [
        "Adapter Stream Summary",
        f"source_type: {args.source}",
        f"session_id: {args.session_id}",
        f"websocket_url: {args.url}",
        f"samples_sent: {samples_sent}",
        f"predictions_received: {summary['total_predictions']}",
        f"risk_counts: {summary['risk_counts']}",
        f"mean_probability: {_fmt(summary['mean_probability'])}",
        f"max_probability: {_fmt(summary['max_probability'])}",
        f"mean_latency_ms: {_fmt(summary['mean_latency_ms'])}",
        f"max_latency_ms: {_fmt(summary['max_latency_ms'])}",
        f"errors: {len(errors)}",
        "",
        "This is an adapter-level streaming test.",
    ]
    if errors:
        lines.extend(["", "Errors:", *[f"- {error}" for error in errors[:20]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, args, samples_sent: int, predictions_df: pd.DataFrame, errors: list[str], metadata: dict) -> None:
    summary = _summary(predictions_df)
    lines = [
        "# Adapter Stream Report",
        "",
        f"- source_type: `{args.source}`",
        f"- session_id: `{args.session_id}`",
        f"- WebSocket URL: `{args.url}`",
        f"- samples sent: {samples_sent}",
        f"- predictions received: {summary['total_predictions']}",
        f"- risk counts: {summary['risk_counts']}",
        f"- mean probability: {_fmt(summary['mean_probability'])}",
        f"- max probability: {_fmt(summary['max_probability'])}",
        f"- mean latency ms: {_fmt(summary['mean_latency_ms'])}",
        f"- max latency ms: {_fmt(summary['max_latency_ms'])}",
        f"- errors: {len(errors)}",
        "",
        "## Source Metadata",
        "",
        *[f"- {key}: {value}" for key, value in metadata.items()],
        "",
        "This test validates the gaze source adapter layer and WebSocket flow. It is not real eye-tracker deployment.",
    ]
    if errors:
        lines.extend(["", "## Errors", *[f"- {error}" for error in errors[:20]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
