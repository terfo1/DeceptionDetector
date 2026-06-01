"""Train LSTM and GRU classifiers on fixed-length eye-tracking sequences."""

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
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MODEL_OUTPUT_DIR,
    NUM_LAYERS,
    RANDOM_SEED,
    REPORT_OUTPUT_DIR,
    SEQUENCE_DATA_DIR,
    TEST_FILE,
    THRESHOLD,
    TRAIN_FILE,
    VALIDATION_FILE,
    WEIGHT_DECAY,
)
from .sequence_models import RecurrentSequenceClassifier
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


MODEL_TYPES = ["lstm", "gru"]


def main() -> None:
    try:
        require_torch()
        _run_training()
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as error:
        print(f"Sequence model training failed: {error}")


def _run_training() -> None:
    print("Training neural sequence models...")
    set_seed(RANDOM_SEED)

    model_dir = Path(MODEL_OUTPUT_DIR)
    report_dir = Path(REPORT_OUTPUT_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    sequence_dir = Path(SEQUENCE_DATA_DIR)
    train_path = sequence_dir / TRAIN_FILE
    validation_path = sequence_dir / VALIDATION_FILE
    test_path = sequence_dir / TEST_FILE
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
    config = _model_config(input_size)
    (model_dir / "sequence_model_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    train_dataset = EyeTrackingSequenceDataset(str(train_path))
    validation_dataset = EyeTrackingSequenceDataset(str(validation_path))
    test_dataset = EyeTrackingSequenceDataset(str(test_path))
    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        BATCH_SIZE,
    )

    print(f"Train sequences: {train_arrays['X'].shape}")
    print(f"Validation sequences: {validation_arrays['X'].shape}")
    print(f"Test sequences: {test_arrays['X'].shape}")
    print(f"Input size: {input_size}")

    device = get_device(DEVICE)
    pos_weight = calculate_pos_weight(train_arrays["y"].astype(int))
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    warnings = []
    validation_available = len(validation_dataset) > 0
    if not validation_available:
        warning = "Validation set is empty. Best model selection and early stopping are disabled."
        warnings.append(warning)
        print(warning)
    if len(test_dataset) == 0:
        warnings.append("Test set is empty. Test evaluation can be run later when data is available.")

    validation_prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    validation_metrics_by_model: dict[str, dict] = {}

    for model_type in MODEL_TYPES:
        print(f"Training {model_type.upper()}...")
        model, history, best_metrics = _train_single_model(
            model_type=model_type,
            input_size=input_size,
            train_loader=train_loader,
            validation_loader=validation_loader,
            validation_available=validation_available,
            pos_weight=pos_weight,
            device=device,
            checkpoint_config=config,
        )

        save_training_history(
            history,
            str(model_dir / f"{model_type}_training_history.json"),
        )
        if validation_available:
            validation_metrics_by_model[model_type] = best_metrics
            metrics_row = {"split": "validation", "model_name": model_type, **best_metrics}
            metric_rows.append(_ordered_metric_row(metrics_row))

            prediction_df = predict_with_metadata(
                model,
                validation_dataset,
                validation_loader,
                device,
                THRESHOLD,
            )
            prediction_df["split"] = "validation"
            prediction_df["model_name"] = model_type
            validation_prediction_rows.append(prediction_df)

    validation_predictions = (
        pd.concat(validation_prediction_rows, ignore_index=True)
        if validation_prediction_rows
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    validation_predictions.to_csv(report_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(metric_rows, columns=METRIC_COLUMNS).to_csv(
        report_dir / "sequence_model_metrics.csv",
        index=False,
    )

    _write_report(
        report_path=report_dir / "sequence_model_report.txt",
        input_files=[train_path, validation_path, test_path],
        train_shape=train_arrays["X"].shape,
        validation_shape=validation_arrays["X"].shape,
        test_shape=test_arrays["X"].shape,
        input_size=input_size,
        train_y=train_arrays["y"],
        validation_y=validation_arrays["y"],
        test_y=test_arrays["y"],
        validation_metrics=validation_metrics_by_model,
        warnings=warnings,
    )

    print(f"Files saved to {MODEL_OUTPUT_DIR} and {REPORT_OUTPUT_DIR}")


def _train_single_model(
    model_type: str,
    input_size: int,
    train_loader,
    validation_loader,
    validation_available: bool,
    pos_weight,
    device,
    checkpoint_config: dict,
):
    model = RecurrentSequenceClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        model_type=model_type,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    checkpoint_path = Path(MODEL_OUTPUT_DIR) / f"{model_type}_model.pt"

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
                f"validation F1: {current_f1:.4f}"
            )
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_metrics = validation_metrics
                epochs_without_improvement = 0
                save_model_checkpoint(
                    model,
                    str(checkpoint_path),
                    {**checkpoint_config, "model_type": model_type},
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping {model_type.upper()} at epoch {epoch}.")
                    history.append(row)
                    break
        else:
            print(f"Epoch {epoch}/{EPOCHS} - train loss: {train_metrics['loss']:.4f}")
            save_model_checkpoint(
                model,
                str(checkpoint_path),
                {**checkpoint_config, "model_type": model_type},
            )

        history.append(row)

    if validation_available and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    elif not checkpoint_path.exists():
        save_model_checkpoint(
            model,
            str(checkpoint_path),
            {**checkpoint_config, "model_type": model_type},
        )

    return model, history, best_metrics


def _validate_sequence_arrays(arrays: dict, split_name: str, require_non_empty: bool) -> None:
    X = arrays["X"]
    y = arrays["y"]
    masks = arrays["masks"]

    if X.ndim != 3:
        raise ValueError(f"{split_name} X must have shape [N, T, F]. Found {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"{split_name} y must have shape [N]. Found {y.shape}.")
    if masks.ndim != 2:
        raise ValueError(f"{split_name} masks must have shape [N, T]. Found {masks.shape}.")
    if X.shape[0] != y.shape[0] or X.shape[0] != masks.shape[0]:
        raise ValueError(f"{split_name} has inconsistent N across X, y, and masks.")
    if X.shape[1] != masks.shape[1]:
        raise ValueError(f"{split_name} masks time dimension does not match X.")
    if require_non_empty and X.shape[0] == 0:
        raise ValueError("Train sequence set is empty. Cannot train sequence models.")

    labels = pd.to_numeric(pd.Series(y), errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{split_name} labels contain missing or non-numeric values.")
    invalid_labels = sorted(set(labels.astype(int)) - {0, 1})
    if invalid_labels:
        raise ValueError(f"{split_name} labels must be 0 or 1. Found: {invalid_labels}")


def _model_config(input_size: int) -> dict:
    return {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "random_seed": RANDOM_SEED,
        "model_types": MODEL_TYPES,
        "note": "The split is subject-independent from the previous sequence preparation step.",
    }


def _ordered_metric_row(row: dict) -> dict:
    return {column: row.get(column, np.nan) for column in METRIC_COLUMNS}


def _write_report(
    report_path: Path,
    input_files: list[Path],
    train_shape: tuple,
    validation_shape: tuple,
    test_shape: tuple,
    input_size: int,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
    validation_metrics: dict[str, dict],
    warnings: list[str],
) -> None:
    lines = [
        "Neural Sequence Model Training Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Input sequence files:",
        *[f"- {path}" for path in input_files],
        "",
        f"Train sequence shape: {train_shape}",
        f"Validation sequence shape: {validation_shape}",
        f"Test sequence shape: {test_shape}",
        f"Input feature count: {input_size}",
        "",
        "Model architectures:",
        "- lstm: recurrent encoder, dropout, linear logit head",
        "- gru: recurrent encoder, dropout, linear logit head",
        "",
        "Training hyperparameters:",
        f"- batch_size: {BATCH_SIZE}",
        f"- epochs: {EPOCHS}",
        f"- learning_rate: {LEARNING_RATE}",
        f"- weight_decay: {WEIGHT_DECAY}",
        f"- hidden_size: {HIDDEN_SIZE}",
        f"- num_layers: {NUM_LAYERS}",
        f"- dropout: {DROPOUT}",
        f"- early_stopping_patience: {EARLY_STOPPING_PATIENCE}",
        f"- threshold: {THRESHOLD}",
        "",
        f"Train label distribution: {_distribution(train_y)}",
        f"Validation label distribution: {_distribution(validation_y)}",
        f"Test label distribution: {_distribution(test_y)}",
        "",
        "Validation metrics:",
    ]
    if validation_metrics:
        for model_name, metrics in validation_metrics.items():
            lines.append(f"- {model_name}: {_format_metrics(metrics)}")
    else:
        lines.append("- No validation metrics calculated.")

    lines.extend(["", "Warnings:"])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None"])
    lines.extend(
        [
            "",
            "Scientific note:",
            "These neural sequence models estimate deception risk under the controlled experimental protocol. They are not universal lie detectors.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _distribution(y: np.ndarray) -> dict[int, int]:
    if y.size == 0:
        return {}
    labels = pd.Series(y).astype(int)
    counts = labels.value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def _format_metrics(metrics: dict) -> str:
    names = [
        "loss",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "false_positive_rate",
        "false_negative_rate",
    ]
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in names)


if __name__ == "__main__":
    main()
