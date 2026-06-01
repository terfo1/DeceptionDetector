"""Train a Causal TCN classifier on fixed-length eye-tracking sequences."""

from __future__ import annotations

import json
import math
from datetime import datetime
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
    DEVICE,
    MODEL_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
    SEQUENCE_DATA_DIR,
    TEST_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
)
from .sequence_training_utils import (
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    calculate_pos_weight,
    create_dataloaders,
    evaluate_model,
    get_device,
    predict_with_metadata,
    require_torch,
    save_model_checkpoint,
    save_training_history,
    set_seed,
    train_one_epoch,
)
from .tcn_model import CausalTCNClassifier
from .train_sequence_models import _distribution, _format_metrics, _validate_sequence_arrays


RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 5

TCN_CHANNELS = [32, 64, 64]
KERNEL_SIZE = 3
DROPOUT = 0.2

MODEL_NAME = "causal_tcn"


def main() -> None:
    try:
        require_torch()
        _run_training()
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as error:
        print(f"Causal TCN training failed: {error}")


def _run_training() -> None:
    print("Training Causal TCN model...")
    set_seed(RANDOM_SEED)

    model_dir = Path(MODEL_OUTPUT_DIR)
    report_dir = Path(REPORT_OUTPUT_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(SEQUENCE_DATA_DIR) / TRAIN_FILE
    validation_path = Path(SEQUENCE_DATA_DIR) / VALIDATION_FILE
    test_path = Path(SEQUENCE_DATA_DIR) / TEST_FILE
    for path in [train_path, validation_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required sequence file is missing: {path}")

    train_arrays = load_sequence_npz(str(train_path))
    validation_arrays = load_sequence_npz(str(validation_path))
    test_arrays = load_sequence_npz(str(test_path))
    _validate_sequence_arrays(train_arrays, "train", require_non_empty=True)
    _validate_sequence_arrays(validation_arrays, "validation", require_non_empty=False)
    _validate_sequence_arrays(test_arrays, "test", require_non_empty=False)

    input_size = int(train_arrays["X"].shape[2])
    config = _tcn_config(input_size)
    (model_dir / "tcn_model_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    train_dataset = EyeTrackingSequenceDataset(str(train_path))
    validation_dataset = EyeTrackingSequenceDataset(str(validation_path))
    test_dataset = EyeTrackingSequenceDataset(str(test_path))
    train_loader, validation_loader, _ = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        BATCH_SIZE,
    )

    print(f"Train sequences: {train_arrays['X'].shape}")
    print(f"Validation sequences: {validation_arrays['X'].shape}")
    print(f"Test sequences: {test_arrays['X'].shape}")
    print(f"Input size: {input_size}")
    print(f"TCN channels: {TCN_CHANNELS}")

    device = get_device(DEVICE)
    pos_weight = calculate_pos_weight(train_arrays["y"].astype(int))
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    validation_available = len(validation_dataset) > 0
    warnings = []
    if not validation_available:
        warning = "Validation set is empty. Best model selection and early stopping are disabled."
        warnings.append(warning)
        print(warning)
    if len(test_dataset) == 0:
        warnings.append("Test set is empty. Test evaluation can be run later when data is available.")

    model, history, validation_metrics = _train_tcn(
        input_size=input_size,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_available=validation_available,
        pos_weight=pos_weight,
        device=device,
        checkpoint_config=config,
    )
    save_training_history(history, str(model_dir / "tcn_training_history.json"))

    prediction_df = pd.DataFrame(columns=PREDICTION_COLUMNS)
    metric_rows = []
    if validation_available:
        prediction_df = predict_with_metadata(
            model,
            validation_dataset,
            validation_loader,
            device,
            THRESHOLD,
        )
        prediction_df["split"] = "validation"
        prediction_df["model_name"] = MODEL_NAME
        metric_rows.append(_metric_row("validation", validation_metrics))

    prediction_df.to_csv(report_dir / "tcn_validation_predictions.csv", index=False)
    pd.DataFrame(metric_rows, columns=METRIC_COLUMNS).to_csv(
        report_dir / "tcn_metrics.csv",
        index=False,
    )
    if metric_rows:
        _upsert_metrics(report_dir / "sequence_model_metrics.csv", metric_rows)

    _write_report(
        report_path=report_dir / "tcn_report.txt",
        input_files=[train_path, validation_path, test_path],
        train_shape=train_arrays["X"].shape,
        validation_shape=validation_arrays["X"].shape,
        test_shape=test_arrays["X"].shape,
        train_y=train_arrays["y"],
        validation_y=validation_arrays["y"],
        validation_metrics=validation_metrics if validation_available else {},
        warnings=warnings,
    )
    print(f"Files saved to {MODEL_OUTPUT_DIR} and {REPORT_OUTPUT_DIR}")


def _train_tcn(
    input_size: int,
    train_loader,
    validation_loader,
    validation_available: bool,
    pos_weight,
    device,
    checkpoint_config: dict,
):
    model = CausalTCNClassifier(
        input_size=input_size,
        num_channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    checkpoint_path = Path(MODEL_OUTPUT_DIR) / "tcn_model.pt"
    history = []
    best_f1 = -math.inf
    best_metrics = {}
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_samples": train_metrics["number_of_samples"],
        }

        if validation_available:
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                criterion,
                device,
                THRESHOLD,
            )
            row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
            current_f1 = validation_metrics["f1"]
            print(
                f"Epoch {epoch}/{EPOCHS} - train loss: {train_metrics['loss']:.4f} - "
                f"Validation F1: {current_f1:.4f}"
            )
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_metrics = validation_metrics
                epochs_without_improvement = 0
                save_model_checkpoint(model, str(checkpoint_path), checkpoint_config)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping Causal TCN at epoch {epoch}.")
                    history.append(row)
                    break
        else:
            print(f"Epoch {epoch}/{EPOCHS} - train loss: {train_metrics['loss']:.4f}")
            save_model_checkpoint(model, str(checkpoint_path), checkpoint_config)

        history.append(row)

    if validation_available and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    elif not checkpoint_path.exists():
        save_model_checkpoint(model, str(checkpoint_path), checkpoint_config)

    return model, history, best_metrics


def _tcn_config(input_size: int) -> dict:
    return {
        "input_size": input_size,
        "tcn_channels": TCN_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "random_seed": RANDOM_SEED,
        "model_type": MODEL_NAME,
        "note": (
            "The model uses causal convolutions and the split is subject-independent "
            "from previous steps."
        ),
    }


def _metric_row(split: str, metrics: dict) -> dict:
    row = {"split": split, "model_name": MODEL_NAME, **metrics}
    return {column: row.get(column, np.nan) for column in METRIC_COLUMNS}


def _upsert_metrics(path: Path, rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    if path.exists():
        existing_df = pd.read_csv(path)
        if not existing_df.empty:
            replace_pairs = set(zip(new_df["split"].astype(str), new_df["model_name"].astype(str)))
            keep_mask = ~existing_df.apply(
                lambda row: (str(row["split"]), str(row["model_name"])) in replace_pairs,
                axis=1,
            )
            new_df = pd.concat([existing_df.loc[keep_mask], new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def _write_report(
    report_path: Path,
    input_files: list[Path],
    train_shape: tuple,
    validation_shape: tuple,
    test_shape: tuple,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    validation_metrics: dict,
    warnings: list[str],
) -> None:
    lines = [
        "Causal TCN Training Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Input sequence files:",
        *[f"- {path}" for path in input_files],
        "",
        f"Train sequence shape: {train_shape}",
        f"Validation sequence shape: {validation_shape}",
        f"Test sequence shape: {test_shape}",
        "",
        "Model architecture:",
        "- Causal Temporal Convolutional Network",
        "- residual temporal blocks with dilated causal Conv1d layers",
        "- mask-aware final valid timestep selection",
        f"TCN channels: {TCN_CHANNELS}",
        f"Kernel size: {KERNEL_SIZE}",
        f"Dropout: {DROPOUT}",
        "",
        "Training hyperparameters:",
        f"- batch_size: {BATCH_SIZE}",
        f"- epochs: {EPOCHS}",
        f"- learning_rate: {LEARNING_RATE}",
        f"- weight_decay: {WEIGHT_DECAY}",
        f"- early_stopping_patience: {EARLY_STOPPING_PATIENCE}",
        f"- threshold: {THRESHOLD}",
        "",
        f"Train label distribution: {_distribution(train_y)}",
        f"Validation label distribution: {_distribution(validation_y)}",
        "",
        "Validation metrics:",
    ]
    if validation_metrics:
        lines.append(f"- {MODEL_NAME}: {_format_metrics(validation_metrics)}")
    else:
        lines.append("- No validation metrics calculated.")

    lines.extend(["", "Warnings:"])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None"])
    lines.extend(
        [
            "",
            "Scientific note:",
            "The Causal TCN is designed for future real-time compatibility because it avoids future-sample leakage. It is still not a universal lie detector.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
