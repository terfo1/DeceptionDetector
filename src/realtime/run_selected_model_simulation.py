"""Run realtime replay simulation with the selected prototype model."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import shutil

import pandas as pd

from .model_registry import check_required_files
from .model_selection_config import DISABLED_MODELS, MODEL_SELECTION_OUTPUT_DIR, SELECTED_MODEL
from .run_realtime_simulation import _run_simulation


def main() -> None:
    try:
        _run()
    except (OSError, ValueError, ModuleNotFoundError) as error:
        print(f"Selected model simulation failed: {error}")


def _run() -> None:
    if SELECTED_MODEL in DISABLED_MODELS:
        raise ValueError(f"Selected model is disabled: {SELECTED_MODEL}")
    status = check_required_files(SELECTED_MODEL)
    if not status["all_files_exist"]:
        raise FileNotFoundError(
            "Selected model is missing required files: " + ", ".join(status["missing_files"])
        )

    output_dir = Path(MODEL_SELECTION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = Namespace(
        model_type=SELECTED_MODEL,
        trial_id=None,
        max_trials=None,
        real_time_sleep=False,
        update_interval=0.5,
    )
    _run_simulation(args)

    source_dir = Path("reports/realtime_simulation")
    predictions_path = source_dir / "realtime_predictions.csv"
    summary_path = source_dir / "realtime_trial_summary.csv"
    if not predictions_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Realtime simulation outputs were not created.")

    selected_predictions = output_dir / "selected_model_realtime_predictions.csv"
    selected_summary = output_dir / "selected_model_trial_summary.csv"
    shutil.copyfile(predictions_path, selected_predictions)
    shutil.copyfile(summary_path, selected_summary)
    _write_simulation_summary(
        output_dir / "selected_model_simulation_summary.txt",
        selected_predictions,
        selected_summary,
    )
    print(f"Selected model simulation complete: {SELECTED_MODEL}")
    print(f"Outputs saved to {MODEL_SELECTION_OUTPUT_DIR}")


def _write_simulation_summary(path: Path, predictions_path: Path, trial_summary_path: Path) -> None:
    predictions = pd.read_csv(predictions_path)
    trial_summary = pd.read_csv(trial_summary_path)
    risk_counts = predictions["risk_category"].value_counts().sort_index().to_dict()
    lines = [
        "Selected Model Simulation Summary",
        f"selected_model: {SELECTED_MODEL}",
        f"trials_simulated: {len(trial_summary)}",
        f"prediction_updates: {len(predictions)}",
        f"mean_latency_ms: {predictions['latency_ms'].mean():.4f}",
        f"max_latency_ms: {predictions['latency_ms'].max():.4f}",
        f"risk_category_counts: {risk_counts}",
        f"mean_probability: {predictions['probability'].mean():.4f}",
        "",
        "This is replay simulation, not live deployment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
