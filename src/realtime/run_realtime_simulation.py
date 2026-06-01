"""Run offline replay simulation for realtime deception-risk scoring."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_policy import final_trial_decision, risk_category, smooth_probabilities
from .realtime_buffer import RealtimeGazeBuffer
from .realtime_config import (
    DEFAULT_MODEL_TYPE,
    MIN_VALID_RATIO,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    SUPPORTED_MODEL_TYPES,
    UPDATE_INTERVAL_SECONDS,
    WINDOW_SIZE_SECONDS,
)
from .realtime_predictor import RealtimePredictor
from .stream_simulator import RecordedGazeStreamSimulator


PREDICTION_COLUMNS = [
    "model_type",
    "trial_id",
    "session_id",
    "question_text",
    "instruction",
    "true_label",
    "answer",
    "current_timestamp",
    "window_start",
    "window_end",
    "sample_count",
    "valid_ratio",
    "probability",
    "smoothed_probability",
    "risk_category",
    "latency_ms",
    "original_length",
]

TRIAL_SUMMARY_COLUMNS = [
    "model_type",
    "trial_id",
    "session_id",
    "instruction",
    "true_label",
    "answer",
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
    parser = argparse.ArgumentParser(description="Replay recorded gaze samples for realtime simulation.")
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--real-time-sleep", action="store_true")
    parser.add_argument("--update-interval", type=float, default=UPDATE_INTERVAL_SECONDS)
    args = parser.parse_args()

    try:
        _run_simulation(args)
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as error:
        print(f"Realtime simulation failed: {error}")
    except OSError as error:
        print(f"Realtime simulation file error: {error}")


def _run_simulation(args) -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    gaze_df, trials_df, sessions_df = _load_raw_data()
    _validate_raw_data(gaze_df, trials_df, sessions_df)

    simulator = RecordedGazeStreamSimulator(gaze_df, trials_df)
    available_trials = simulator.get_available_trials()
    if not available_trials:
        raise ValueError("No trials with gaze samples were found.")

    if args.trial_id:
        if args.trial_id not in available_trials:
            raise ValueError(f"Requested trial_id not found in gaze samples: {args.trial_id}")
        trial_ids = [args.trial_id]
    else:
        trial_ids = available_trials
    if args.max_trials is not None:
        trial_ids = trial_ids[: max(0, args.max_trials)]
    if not trial_ids:
        raise ValueError("No trials selected for simulation.")

    predictor = RealtimePredictor(args.model_type)
    predictor.load_model()
    trial_lookup = trials_df.set_index("trial_id", drop=False)

    prediction_rows: list[dict] = []
    trial_summary_rows: list[dict] = []

    print(f"Running realtime replay simulation with model_type={args.model_type}...")
    for trial_id in trial_ids:
        if trial_id not in trial_lookup.index:
            print(f"Warning: trial_id {trial_id} is missing from trials.csv. Skipping.")
            continue
        trial_row = trial_lookup.loc[trial_id]
        rows, summary = _simulate_trial(
            simulator=simulator,
            predictor=predictor,
            trial_id=trial_id,
            trial_row=trial_row,
            update_interval=float(args.update_interval),
            real_time_sleep=bool(args.real_time_sleep),
        )
        prediction_rows.extend(rows)
        if summary is not None:
            trial_summary_rows.append(summary)

    predictions_df = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    summary_df = pd.DataFrame(trial_summary_rows, columns=TRIAL_SUMMARY_COLUMNS)
    predictions_df.to_csv(output_dir / "realtime_predictions.csv", index=False)
    summary_df.to_csv(output_dir / "realtime_trial_summary.csv", index=False)
    _write_report(output_dir / "realtime_simulation_report.txt", args, predictions_df, summary_df, trial_ids)
    _write_summary(output_dir / "realtime_simulation_summary.md", args, predictions_df, summary_df)
    _write_optional_plots(output_dir, predictions_df, summary_df)

    print(f"Trials simulated: {len(summary_df)}")
    print(f"Prediction updates: {len(predictions_df)}")
    print(f"Files saved to {OUTPUT_DIR}")


def _simulate_trial(
    simulator: RecordedGazeStreamSimulator,
    predictor: RealtimePredictor,
    trial_id: str,
    trial_row: pd.Series,
    update_interval: float,
    real_time_sleep: bool,
) -> tuple[list[dict], dict | None]:
    buffer = RealtimeGazeBuffer(WINDOW_SIZE_SECONDS)
    prediction_rows = []
    probabilities = []
    valid_ratios = []
    latencies = []
    next_update_timestamp = 0.0
    sample_seen = False

    for event in simulator.iter_trial_samples(trial_id, real_time_sleep=real_time_sleep):
        sample = dict(event["sample"])
        sample["trial_id"] = trial_id
        sample["timestamp"] = event["timestamp"]
        sample_seen = True
        buffer.add_sample(sample)
        current_timestamp = float(event["timestamp"])
        if current_timestamp + 1e-9 < next_update_timestamp:
            continue

        window_df = buffer.get_window(current_timestamp)
        prediction = predictor.predict(window_df, current_timestamp=current_timestamp)
        probabilities.append(prediction["probability"])
        valid_ratios.append(prediction["valid_ratio"])
        latencies.append(prediction["latency_ms"])
        smoothed = smooth_probabilities(probabilities)
        category = risk_category(smoothed, prediction["valid_ratio"], MIN_VALID_RATIO)

        prediction_rows.append(
            {
                "model_type": predictor.model_type,
                "trial_id": trial_id,
                "session_id": trial_row.get("session_id", ""),
                "question_text": trial_row.get("question_text", ""),
                "instruction": trial_row.get("instruction", ""),
                "true_label": trial_row.get("label", ""),
                "answer": trial_row.get("answer", ""),
                "current_timestamp": round(current_timestamp, 3),
                "window_start": round(max(0.0, current_timestamp - WINDOW_SIZE_SECONDS), 3),
                "window_end": round(current_timestamp, 3),
                "sample_count": prediction["sample_count"],
                "valid_ratio": prediction["valid_ratio"],
                "probability": prediction["probability"],
                "smoothed_probability": smoothed,
                "risk_category": category,
                "latency_ms": prediction["latency_ms"],
                "original_length": prediction["original_length"],
            }
        )
        next_update_timestamp += update_interval

    if not sample_seen:
        print(f"Warning: trial_id {trial_id} has no gaze samples. Skipping.")
        return [], None

    decision = final_trial_decision(probabilities, valid_ratios)
    final_category = risk_category(
        decision["final_smoothed_probability"],
        valid_ratios[-1] if valid_ratios else 0.0,
        MIN_VALID_RATIO,
    )
    summary = {
        "model_type": predictor.model_type,
        "trial_id": trial_id,
        "session_id": trial_row.get("session_id", ""),
        "instruction": trial_row.get("instruction", ""),
        "true_label": trial_row.get("label", ""),
        "answer": trial_row.get("answer", ""),
        "prediction_count": len(probabilities),
        "usable_prediction_count": decision["usable_prediction_count"],
        "insufficient_data_count": decision["insufficient_data_count"],
        "mean_probability": decision["mean_probability"],
        "max_probability": decision["max_probability"],
        "final_smoothed_probability": decision["final_smoothed_probability"],
        "final_risk_category": final_category,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else np.nan,
        "max_latency_ms": float(np.max(latencies)) if latencies else np.nan,
    }
    return prediction_rows, summary


def _load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(RAW_DATA_DIR)
    gaze_path = raw_dir / "gaze_samples.csv"
    trials_path = raw_dir / "trials.csv"
    sessions_path = raw_dir / "sessions.csv"
    for path in [gaze_path, trials_path, sessions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required raw data file is missing: {path}")
    return pd.read_csv(gaze_path), pd.read_csv(trials_path), pd.read_csv(sessions_path)


def _validate_raw_data(gaze_df: pd.DataFrame, trials_df: pd.DataFrame, sessions_df: pd.DataFrame) -> None:
    gaze_required = {
        "sample_id",
        "trial_id",
        "timestamp",
        "gaze_x",
        "gaze_y",
        "pupil_left",
        "pupil_right",
        "blink",
        "fixation",
        "saccade",
        "validity",
    }
    trial_required = {"trial_id", "session_id", "question_text", "instruction", "label", "answer"}
    session_required = {"session_id"}
    _require_columns(gaze_df, gaze_required, "gaze_samples.csv")
    _require_columns(trials_df, trial_required, "trials.csv")
    _require_columns(sessions_df, session_required, "sessions.csv")
    timestamps = pd.to_numeric(gaze_df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("gaze_samples.csv contains invalid timestamps.")


def _require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _write_report(
    path: Path,
    args,
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trial_ids: list[str],
) -> None:
    risk_counts = _risk_counts(predictions_df)
    latency = predictions_df["latency_ms"] if "latency_ms" in predictions_df else pd.Series(dtype=float)
    probabilities = predictions_df["probability"] if "probability" in predictions_df else pd.Series(dtype=float)
    lines = [
        "Real-Time Simulation Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "1. Overview",
        "This report describes simulated online inference using recorded gaze samples replayed trial-by-trial.",
        "The simulation uses a rolling causal buffer and does not train or modify any model.",
        "",
        "2. Model Used",
        f"- model_type: {args.model_type}",
        "",
        "3. Simulation Settings",
        f"- window_size_seconds: {WINDOW_SIZE_SECONDS}",
        f"- update_interval_seconds: {args.update_interval}",
        f"- min_valid_ratio: {MIN_VALID_RATIO}",
        f"- low_threshold: {RISK_LOW_THRESHOLD}",
        f"- high_threshold: {RISK_HIGH_THRESHOLD}",
        f"- real_time_sleep: {bool(args.real_time_sleep)}",
        "",
        "4. Dataset Summary",
        f"- trials simulated: {len(summary_df)}",
        f"- requested trial IDs: {', '.join(trial_ids)}",
        f"- prediction updates: {len(predictions_df)}",
        f"- label distribution: {_label_distribution(summary_df)}",
        f"- model type: {args.model_type}",
        "",
        "5. Latency Summary",
        f"- mean latency ms: {_fmt(latency.mean())}",
        f"- median latency ms: {_fmt(latency.median())}",
        f"- max latency ms: {_fmt(latency.max())}",
        f"- min latency ms: {_fmt(latency.min())}",
        "",
        "6. Risk Score Summary",
        f"- mean probability: {_fmt(probabilities.mean())}",
        f"- max probability: {_fmt(probabilities.max())}",
        f"- low predictions: {risk_counts.get('low', 0)}",
        f"- medium predictions: {risk_counts.get('medium', 0)}",
        f"- high predictions: {risk_counts.get('high', 0)}",
        f"- insufficient_data predictions: {risk_counts.get('insufficient_data', 0)}",
        "",
        "7. Trial-Level Summary",
    ]
    if summary_df.empty:
        lines.append("- No trial summaries were generated.")
    else:
        for _, row in summary_df.iterrows():
            lines.append(
                f"- {row['trial_id']}: final_risk={row['final_risk_category']}, "
                f"mean_probability={_fmt(row['mean_probability'])}, "
                f"max_probability={_fmt(row['max_probability'])}, "
                f"updates={int(row['prediction_count'])}"
            )
    lines.extend(
        [
            "",
            "8. Limitations",
            "- This is a replay simulation, not a live eye-tracker deployment.",
            "- Results are prototype-level because the dataset is small.",
            "- Current model outputs should not be interpreted as final deception judgments.",
            "- Real-time deployment requires live device testing, larger subject-independent dataset, and calibration.",
            "",
            "9. Next Step",
            "- Calibrate thresholds after collecting more data.",
            "- Then build a FastAPI/WebSocket live inference service.",
            "- Then build a dashboard.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, args, predictions_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    risk_counts = _risk_counts(predictions_df)
    latency = predictions_df["latency_ms"] if "latency_ms" in predictions_df else pd.Series(dtype=float)
    lines = [
        "# Real-Time Simulation Summary",
        "",
        f"- Model type: `{args.model_type}`",
        f"- Trials simulated: {len(summary_df)}",
        f"- Prediction updates: {len(predictions_df)}",
        f"- Mean latency ms: {_fmt(latency.mean())}",
        f"- Max latency ms: {_fmt(latency.max())}",
        f"- Risk categories: {risk_counts}",
        "",
        "This is an offline replay simulation using recorded gaze samples, not live deployment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_optional_plots(output_dir: Path, predictions_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not available. Plot generation skipped.")
        return
    if predictions_df.empty:
        return

    plt.figure(figsize=(9, 4))
    for trial_id, group in predictions_df.groupby("trial_id"):
        plt.plot(group["current_timestamp"], group["smoothed_probability"], label=trial_id)
    plt.xlabel("Timestamp")
    plt.ylabel("Smoothed risk probability")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "risk_score_timeline.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(predictions_df["latency_ms"], bins=20)
    plt.xlabel("Latency ms")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_distribution.png", dpi=150)
    plt.close()

    if not summary_df.empty:
        plt.figure(figsize=(8, 4))
        plt.bar(summary_df["trial_id"], summary_df["final_smoothed_probability"])
        plt.ylabel("Final smoothed probability")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "trial_risk_summary.png", dpi=150)
        plt.close()


def _risk_counts(predictions_df: pd.DataFrame) -> dict:
    if predictions_df.empty or "risk_category" not in predictions_df.columns:
        return {}
    return predictions_df["risk_category"].value_counts().sort_index().to_dict()


def _label_distribution(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty or "true_label" not in summary_df.columns:
        return {}
    labels = pd.to_numeric(summary_df["true_label"], errors="coerce")
    return {int(label): int(count) for label, count in labels.value_counts().sort_index().items()}


def _fmt(value) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
