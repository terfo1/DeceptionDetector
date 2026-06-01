"""Training, evaluation, checkpoint, and prediction helpers for sequence models."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    torch = None
    DataLoader = None

from .sequence_model_config import (
    DROPOUT,
    HIDDEN_SIZE,
    NUM_LAYERS,
)
from .sequence_models import RecurrentSequenceClassifier


PREDICTION_COLUMNS = [
    "split",
    "model_name",
    "window_id",
    "participant_id",
    "trial_id",
    "true_label",
    "predicted_label",
    "predicted_probability",
    "valid_ratio",
    "original_length",
]

METRIC_COLUMNS = [
    "split",
    "model_name",
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
    "number_of_samples",
    "number_of_truth",
    "number_of_lie",
]


def require_torch() -> None:
    """Raise a readable error if PyTorch is unavailable."""
    if torch is None or DataLoader is None:
        raise ModuleNotFoundError(
            "PyTorch is required for sequence model training/evaluation. "
            "Install torch and rerun."
        )


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device(device_config: str):
    """Resolve the configured torch device."""
    require_torch()
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """Create train, validation, and test dataloaders."""
    require_torch()
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def calculate_pos_weight(y_train: np.ndarray):
    """Calculate BCE positive-class weight from train labels."""
    require_torch()
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    if negative_count > 0 and positive_count > 0:
        return torch.tensor([negative_count / positive_count], dtype=torch.float32)
    print("Warning: Training labels contain only one class. pos_weight is disabled.")
    return None


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> dict:
    """Train the model for one epoch."""
    require_torch()
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(X, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples if total_samples else np.nan,
        "number_of_samples": total_samples,
    }


def evaluate_model(model, dataloader, criterion, device, threshold=0.5) -> dict:
    """Evaluate a model and return binary classification metrics."""
    require_torch()
    if len(dataloader.dataset) == 0:
        return _empty_metrics()

    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_batches = []
    probability_batches = []

    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            logits = model(X, mask)
            loss = criterion(logits, y)

            probabilities = torch.sigmoid(logits)
            batch_size = int(y.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            y_true_batches.append(y.detach().cpu().numpy())
            probability_batches.append(probabilities.detach().cpu().numpy())

    y_true = np.concatenate(y_true_batches).astype(int)
    probabilities = np.concatenate(probability_batches)
    predicted = (probabilities >= threshold).astype(int)
    metrics = _classification_metrics(y_true, predicted, probabilities)
    metrics["loss"] = total_loss / total_samples if total_samples else np.nan
    return metrics


def save_training_history(history: list[dict], path: str) -> None:
    """Save training history as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def save_model_checkpoint(model, path: str, config: dict) -> None:
    """Save a model state dict and its configuration."""
    require_torch()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
        },
        output_path,
    )


def load_model_checkpoint(path: str, input_size: int, model_type: str, device):
    """Load a sequence model checkpoint."""
    require_torch()
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Required sequence model artifact is missing: {path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", {})
    model = RecurrentSequenceClassifier(
        input_size=input_size,
        hidden_size=int(config.get("hidden_size", HIDDEN_SIZE)),
        num_layers=int(config.get("num_layers", NUM_LAYERS)),
        dropout=float(config.get("dropout", DROPOUT)),
        model_type=model_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_with_metadata(model, dataset, dataloader, device, threshold=0.5) -> pd.DataFrame:
    """Run model predictions and attach dataset metadata."""
    require_torch()
    if len(dataset) == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["X"].to(device), batch["mask"].to(device))
            probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

    probabilities_array = np.asarray(probabilities, dtype=float)
    predicted = (probabilities_array >= threshold).astype(int)
    return pd.DataFrame(
        {
            "split": "",
            "model_name": "",
            "window_id": dataset.window_ids,
            "participant_id": dataset.participant_ids,
            "trial_id": dataset.trial_ids,
            "true_label": dataset.y.astype(int),
            "predicted_label": predicted,
            "predicted_probability": probabilities_array,
            "valid_ratio": dataset.valid_ratios,
            "original_length": dataset.original_lengths,
        },
        columns=PREDICTION_COLUMNS,
    )


def _classification_metrics(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    has_two_classes = len(np.unique(y_true)) == 2
    return {
        "loss": np.nan,
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities) if has_two_classes else np.nan,
        "average_precision": (
            average_precision_score(y_true, probabilities) if has_two_classes else np.nan
        ),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "number_of_samples": int(len(y_true)),
        "number_of_truth": int((y_true == 0).sum()),
        "number_of_lie": int((y_true == 1).sum()),
    }


def _empty_metrics() -> dict:
    return {
        "loss": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "false_positive_rate": np.nan,
        "false_negative_rate": np.nan,
        "number_of_samples": 0,
        "number_of_truth": 0,
        "number_of_lie": 0,
    }
