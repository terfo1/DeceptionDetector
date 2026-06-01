"""Replay recorded gaze samples through the live WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import websockets
from websockets.exceptions import InvalidURI, WebSocketException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.websocket_client_utils import (
    build_sample_message,
    get_trial_ids,
    load_raw_stream_data,
    summarize_predictions,
    write_csv_with_header,
)


PREDICTION_COLUMNS = [
    "received_at",
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

TRIAL_SUMMARY_COLUMNS = [
    "received_at",
    "session_id",
    "trial_id",
    "prediction_count",
    "usable_prediction_count",
    "insufficient_data_count",
    "mean_probability",
    "max_probability",
    "final_smoothed_probability",
    "final_risk_category",
    "mean_latency_ms",
    "max_latency_ms",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the live WebSocket inference API.")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/live")
    parser.add_argument("--session-id", default="WS_TEST_001")
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--real-time-sleep", action="store_true")
    parser.add_argument("--sleep-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default="reports/live_ws_client")
    args = parser.parse_args()

    try:
        asyncio.run(run_client(args))
    except (ConnectionRefusedError, OSError, WebSocketException, InvalidURI):
        print("Could not connect to WebSocket API. Start it with: uvicorn src.api.app:app --reload")
    except (FileNotFoundError, ValueError) as error:
        print(f"WebSocket test client failed: {error}")


async def run_client(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gaze_df, trials_df = load_raw_stream_data()
    trial_ids = get_trial_ids(trials_df, max_trials=args.max_trials, trial_id=args.trial_id)
    if not trial_ids:
        raise ValueError("No trials selected for WebSocket replay.")

    prediction_rows: list[dict] = []
    trial_summary_rows: list[dict] = []
    errors: list[str] = []
    completed_trials = 0

    async with websockets.connect(args.url) as websocket:
        await websocket.send(json.dumps({"type": "start_session", "session_id": args.session_id}))
        await _receive_control_message(websocket, errors)

        for trial_id in trial_ids:
            trial_samples = (
                gaze_df[gaze_df["trial_id"].astype(str) == str(trial_id)]
                .copy()
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if trial_samples.empty:
                errors.append(f"No samples found for trial_id={trial_id}")
                continue

            await websocket.send(json.dumps({"type": "start_trial", "trial_id": trial_id}))
            await _receive_control_message(websocket, errors)
            previous_timestamp = None
            next_prediction_timestamp = 0.0

            for _, row in trial_samples.iterrows():
                timestamp = float(row["timestamp"])
                if args.real_time_sleep and previous_timestamp is not None:
                    await asyncio.sleep(max(0.0, timestamp - previous_timestamp) * args.sleep_scale)
                previous_timestamp = timestamp

                await websocket.send(json.dumps(build_sample_message(row)))
                if timestamp + 1e-9 >= next_prediction_timestamp:
                    response = await _receive_prediction_or_error(websocket, errors)
                    if response and response.get("type") == "prediction":
                        prediction_rows.append(_prediction_row(response))
                    next_prediction_timestamp += 0.5

            await websocket.send(json.dumps({"type": "end_trial"}))
            server_trial_summary = await _receive_control_message(websocket, errors)
            trial_summary_rows.append(
                _trial_summary_row(
                    args.session_id,
                    trial_id,
                    pd.DataFrame(prediction_rows),
                    server_trial_summary,
                )
            )
            completed_trials += 1

        await websocket.send(json.dumps({"type": "end_session"}))
        await _receive_control_message(websocket, errors)

    predictions_df = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    trial_summary_df = pd.DataFrame(trial_summary_rows, columns=TRIAL_SUMMARY_COLUMNS)
    write_csv_with_header(output_dir / "ws_client_predictions.csv", prediction_rows, PREDICTION_COLUMNS)
    write_csv_with_header(output_dir / "ws_client_trial_summary.csv", trial_summary_rows, TRIAL_SUMMARY_COLUMNS)
    _write_report(output_dir / "ws_client_report.txt", args, trial_ids, completed_trials, predictions_df, trial_summary_df, errors)
    _write_summary(output_dir / "ws_client_summary.md", args, predictions_df, errors)
    _write_optional_plots(output_dir, predictions_df)

    summary = summarize_predictions(predictions_df)
    print(f"Connected to {args.url}")
    print(f"Trials completed: {completed_trials}")
    print(f"Predictions received: {summary['total_predictions']}")
    print(f"Mean latency ms: {_fmt(summary['mean_latency_ms'])}")
    print(f"Errors: {len(errors)}")


async def _receive_control_message(websocket, errors: list[str]) -> dict | None:
    try:
        response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10.0))
    except asyncio.TimeoutError:
        errors.append("Timed out waiting for server response.")
        return None
    if response.get("type") == "error":
        errors.append(str(response.get("message", "Unknown server error.")))
    return response


async def _receive_prediction_or_error(websocket, errors: list[str]) -> dict | None:
    try:
        response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10.0))
    except asyncio.TimeoutError:
        errors.append("Timed out waiting for prediction response.")
        return None
    if response.get("type") == "error":
        errors.append(str(response.get("message", "Unknown server error.")))
    return response


def _prediction_row(response: dict) -> dict:
    return {
        "received_at": datetime.now().isoformat(timespec="seconds"),
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


def _trial_summary_row(
    session_id: str,
    trial_id: str,
    all_predictions_df: pd.DataFrame,
    server_trial_summary: dict | None,
) -> dict:
    trial_predictions = all_predictions_df[all_predictions_df["trial_id"].astype(str) == str(trial_id)].copy()
    if trial_predictions.empty:
        return {
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "trial_id": trial_id,
            "prediction_count": 0,
            "usable_prediction_count": 0,
            "insufficient_data_count": 0,
            "mean_probability": pd.NA,
            "max_probability": pd.NA,
            "final_smoothed_probability": pd.NA,
            "final_risk_category": "insufficient_data",
            "mean_latency_ms": pd.NA,
            "max_latency_ms": pd.NA,
        }
    probabilities = pd.to_numeric(trial_predictions["probability"], errors="coerce")
    latencies = pd.to_numeric(trial_predictions["latency_ms"], errors="coerce")
    insufficient = int((trial_predictions["risk_category"] == "insufficient_data").sum())
    final_row = trial_predictions.iloc[-1]
    return {
        "received_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "trial_id": trial_id,
        "prediction_count": int(len(trial_predictions)),
        "usable_prediction_count": int(len(trial_predictions) - insufficient),
        "insufficient_data_count": insufficient,
        "mean_probability": float(probabilities.mean()),
        "max_probability": float(probabilities.max()),
        "final_smoothed_probability": final_row.get("smoothed_probability", pd.NA),
        "final_risk_category": final_row.get("risk_category", "insufficient_data"),
        "mean_latency_ms": float(latencies.mean()),
        "max_latency_ms": float(latencies.max()),
    }


def _write_report(
    path: Path,
    args,
    trial_ids: list[str],
    completed_trials: int,
    predictions_df: pd.DataFrame,
    trial_summary_df: pd.DataFrame,
    errors: list[str],
) -> None:
    summary = summarize_predictions(predictions_df)
    lines = [
        "WebSocket Live Client Test Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "1. Overview",
        "This report validates the live WebSocket inference endpoint using recorded gaze samples.",
        "",
        "2. API URL",
        f"- {args.url}",
        "",
        "3. Session Info",
        f"- session_id: {args.session_id}",
        f"- trials requested: {len(trial_ids)}",
        f"- trials completed: {completed_trials}",
        f"- real_time_sleep: {bool(args.real_time_sleep)}",
        "",
        "4. Prediction Summary",
        f"- total predictions received: {summary['total_predictions']}",
        f"- low predictions: {summary['low_count']}",
        f"- medium predictions: {summary['medium_count']}",
        f"- high predictions: {summary['high_count']}",
        f"- insufficient_data predictions: {summary['insufficient_data_count']}",
        f"- mean probability: {_fmt(summary['mean_probability'])}",
        f"- max probability: {_fmt(summary['max_probability'])}",
        f"- mean latency ms: {_fmt(summary['mean_latency_ms'])}",
        f"- max latency ms: {_fmt(summary['max_latency_ms'])}",
        "",
        "5. Trial Summary",
    ]
    if trial_summary_df.empty:
        lines.append("- No trial summaries generated.")
    else:
        for _, row in trial_summary_df.iterrows():
            lines.append(
                f"- {row['trial_id']}: predictions={row['prediction_count']}, "
                f"final_risk={row['final_risk_category']}, "
                f"mean_probability={_fmt(row['mean_probability'])}"
            )
    lines.extend(["", "6. Errors"])
    lines.extend([f"- {error}" for error in errors] if errors else ["None"])
    lines.extend(
        [
            "",
            "7. Interpretation",
            "This confirms the WebSocket API can receive gaze samples and return live risk predictions.",
            "It is still a replay test, not real eye-tracker deployment.",
            "",
            "8. Limitations",
            "- Replayed recorded data, not live hardware.",
            "- Current selected model is prototype-level.",
            "- Dataset is small.",
            "- Predictions are risk indicators, not final lie judgments.",
            "",
            "9. Next Step",
            "- Build a minimal dashboard or CLI monitor.",
            "- Then connect a real eye tracker or webcam gaze estimator.",
            "- Collect more participants.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, args, predictions_df: pd.DataFrame, errors: list[str]) -> None:
    summary = summarize_predictions(predictions_df)
    lines = [
        "# WebSocket Live Client Summary",
        "",
        f"- API URL: `{args.url}`",
        f"- Session ID: `{args.session_id}`",
        f"- Predictions received: {summary['total_predictions']}",
        f"- Mean probability: {_fmt(summary['mean_probability'])}",
        f"- Mean latency ms: {_fmt(summary['mean_latency_ms'])}",
        f"- Risk counts: low={summary['low_count']}, medium={summary['medium_count']}, high={summary['high_count']}, insufficient_data={summary['insufficient_data_count']}",
        f"- Errors: {len(errors)}",
        "",
        "This is an end-to-end replay test of the WebSocket API, not live hardware deployment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_optional_plots(output_dir: Path, predictions_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not available. Plot generation skipped.")
        return
    if predictions_df.empty:
        return
    order = range(1, len(predictions_df) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(list(order), predictions_df["probability"])
    plt.xlabel("Prediction order")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(output_dir / "ws_probability_timeline.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(predictions_df["latency_ms"], bins=15)
    plt.xlabel("Latency ms")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "ws_latency_distribution.png", dpi=150)
    plt.close()

    counts = predictions_df["risk_category"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "ws_risk_category_distribution.png", dpi=150)
    plt.close()


def _fmt(value) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
