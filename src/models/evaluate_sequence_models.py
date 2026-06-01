"""Evaluate trained LSTM and GRU sequence models on the test split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from .sequence_dataset import EyeTrackingSequenceDataset, load_sequence_npz
from .sequence_model_config import (
    BATCH_SIZE,
    DEVICE,
    MODEL_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
    SEQUENCE_DATA_DIR,
    TEST_FILE,
    THRESHOLD,
)
from .sequence_training_utils import (
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    create_dataloaders,
    evaluate_model,
    get_device,
    load_model_checkpoint,
    predict_with_metadata,
    require_torch,
)
from .train_sequence_models import _distribution, _format_metrics, _validate_sequence_arrays


MODEL_TYPES = ["lstm", "gru"]


def main() -> None:
    try:
        require_torch()
        _run_evaluation()
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as error:
        print(f"Sequence model evaluation failed: {error}")


def _run_evaluation() -> None:
    model_dir = Path(MODEL_OUTPUT_DIR)
    report_dir = Path(REPORT_OUTPUT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "sequence_model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Required sequence model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    test_path = Path(SEQUENCE_DATA_DIR) / TEST_FILE
    if not test_path.exists():
        raise FileNotFoundError(f"Required test sequence file is missing: {test_path}")
    test_arrays = load_sequence_npz(str(test_path))
    _validate_sequence_arrays(test_arrays, "test", require_non_empty=False)

    prediction_path = report_dir / "test_predictions.csv"
    if test_arrays["X"].shape[0] == 0:
        print("Test set is empty. Evaluation skipped.")
        pd.DataFrame(columns=PREDICTION_COLUMNS).to_csv(prediction_path, index=False)
        _update_report(report_dir / "sequence_model_report.txt", test_arrays["y"], {}, [
            "No test metrics were calculated because the test split is empty."
        ])
        return

    input_size = int(config.get("input_size", test_arrays["X"].shape[2]))
    device = get_device(DEVICE)
    test_dataset = EyeTrackingSequenceDataset(str(test_path))
    _, _, test_loader = create_dataloaders(test_dataset, test_dataset, test_dataset, BATCH_SIZE)
    criterion = nn.BCEWithLogitsLoss()

    prediction_frames = []
    metric_rows = []
    test_metrics_by_model = {}

    for model_type in MODEL_TYPES:
        model_path = model_dir / f"{model_type}_model.pt"
        model = load_model_checkpoint(str(model_path), input_size, model_type, device)
        metrics = evaluate_model(model, test_loader, criterion, device, THRESHOLD)
        metrics_row = {"split": "test", "model_name": model_type, **metrics}
        metric_rows.append({column: metrics_row.get(column, np.nan) for column in METRIC_COLUMNS})
        test_metrics_by_model[model_type] = metrics

        prediction_df = predict_with_metadata(model, test_dataset, test_loader, device, THRESHOLD)
        prediction_df["split"] = "test"
        prediction_df["model_name"] = model_type
        prediction_frames.append(prediction_df)
        print(f"Test F1 {model_type}: {metrics['f1']:.4f}")

    pd.concat(prediction_frames, ignore_index=True).to_csv(prediction_path, index=False)
    _upsert_metrics(report_dir / "sequence_model_metrics.csv", metric_rows)

    warnings = []
    if len(test_dataset) < 10:
        warnings.append("Test set is too small for stable neural sequence evaluation.")
    if len(np.unique(test_arrays["y"].astype(int))) < 2:
        warnings.append("Test set has only one class; ROC-AUC and average precision are unavailable.")
    _update_report(report_dir / "sequence_model_report.txt", test_arrays["y"], test_metrics_by_model, warnings)
    print("Test predictions and metrics saved to reports/sequences")


def _upsert_metrics(path: Path, test_rows: list[dict]) -> None:
    new_df = pd.DataFrame(test_rows, columns=METRIC_COLUMNS)
    if path.exists():
        existing_df = pd.read_csv(path)
        if not existing_df.empty:
            keep_mask = ~(
                (existing_df["split"].astype(str) == "test")
                & (existing_df["model_name"].isin(new_df["model_name"]))
            )
            new_df = pd.concat([existing_df.loc[keep_mask], new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def _update_report(
    path: Path,
    test_y: np.ndarray,
    test_metrics: dict[str, dict],
    warnings: list[str],
) -> None:
    existing = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else "Neural Sequence Model Training Report\n"
    )
    marker = "\nTest Evaluation\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"

    lines = [
        "",
        "Test Evaluation",
        f"Test sample count: {int(test_y.shape[0])}",
        f"Test label distribution: {_distribution(test_y)}",
        "Test metrics:",
    ]
    if test_metrics:
        for model_name, metrics in test_metrics.items():
            lines.append(f"- {model_name}: {_format_metrics(metrics)}")
    else:
        lines.append("- No test metrics calculated.")
    lines.append("Warnings:")
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None"])
    path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
