"""Evaluate the trained Causal TCN model on the test sequence split."""

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
)
from .sequence_training_utils import (
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    create_dataloaders,
    evaluate_model,
    get_device,
    predict_with_metadata,
    require_torch,
)
from .tcn_model import CausalTCNClassifier
from .train_sequence_models import _distribution, _format_metrics, _validate_sequence_arrays
from .train_tcn_model import MODEL_NAME, THRESHOLD, _metric_row, _upsert_metrics


def main() -> None:
    try:
        require_torch()
        _run_evaluation()
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as error:
        print(f"Causal TCN evaluation failed: {error}")


def _run_evaluation() -> None:
    model_dir = Path(MODEL_OUTPUT_DIR)
    report_dir = Path(REPORT_OUTPUT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "tcn_model_config.json"
    model_path = model_dir / "tcn_model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Required TCN config is missing: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Required TCN model artifact is missing: {model_path}")

    test_path = Path(SEQUENCE_DATA_DIR) / TEST_FILE
    if not test_path.exists():
        raise FileNotFoundError(f"Required test sequence file is missing: {test_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    test_arrays = load_sequence_npz(str(test_path))
    _validate_sequence_arrays(test_arrays, "test", require_non_empty=False)

    prediction_path = report_dir / "tcn_test_predictions.csv"
    if test_arrays["X"].shape[0] == 0:
        print("Test set is empty. Evaluation skipped.")
        pd.DataFrame(columns=PREDICTION_COLUMNS).to_csv(prediction_path, index=False)
        _update_report(
            report_dir / "tcn_report.txt",
            test_arrays["y"],
            {},
            ["No test metrics were calculated because the test split is empty."],
        )
        return

    device = get_device(DEVICE)
    model = _load_tcn_model(model_path, config, device)
    test_dataset = EyeTrackingSequenceDataset(str(test_path))
    _, _, test_loader = create_dataloaders(test_dataset, test_dataset, test_dataset, BATCH_SIZE)
    criterion = nn.BCEWithLogitsLoss()

    metrics = evaluate_model(model, test_loader, criterion, device, THRESHOLD)
    metric_rows = [_metric_row("test", metrics)]
    _upsert_metrics(report_dir / "tcn_metrics.csv", metric_rows)
    sequence_metrics_path = report_dir / "sequence_model_metrics.csv"
    if sequence_metrics_path.exists():
        _upsert_metrics(sequence_metrics_path, metric_rows)

    prediction_df = predict_with_metadata(model, test_dataset, test_loader, device, THRESHOLD)
    prediction_df["split"] = "test"
    prediction_df["model_name"] = MODEL_NAME
    prediction_df.to_csv(prediction_path, index=False)

    warnings = []
    if len(test_dataset) < 10:
        warnings.append("Test set is too small for stable Causal TCN evaluation.")
    if len(np.unique(test_arrays["y"].astype(int))) < 2:
        warnings.append("Test set has only one class; ROC-AUC and average precision are unavailable.")
    _update_report(report_dir / "tcn_report.txt", test_arrays["y"], metrics, warnings)
    print(f"Test F1 {MODEL_NAME}: {metrics['f1']:.4f}")
    print("TCN test predictions and metrics saved to reports/sequences")


def _load_tcn_model(model_path: Path, config: dict, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    model = CausalTCNClassifier(
        input_size=int(checkpoint_config["input_size"]),
        num_channels=list(checkpoint_config["tcn_channels"]),
        kernel_size=int(checkpoint_config["kernel_size"]),
        dropout=float(checkpoint_config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def _update_report(
    path: Path,
    test_y: np.ndarray,
    test_metrics: dict,
    warnings: list[str],
) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "Causal TCN Training Report\n"
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
        lines.append(f"- {MODEL_NAME}: {_format_metrics(test_metrics)}")
    else:
        lines.append("- No test metrics calculated.")
    lines.append("Warnings:")
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None"])
    path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
