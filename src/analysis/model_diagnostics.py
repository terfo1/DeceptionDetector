"""Reusable model comparison and diagnostics functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .comparison_config import (
    MIN_RELIABLE_PARTICIPANTS,
    MIN_RELIABLE_TEST_SAMPLES,
    MIN_RELIABLE_TRAIN_SAMPLES,
    PROCESSED_DATA_DIR,
)


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
    "source_file",
]

PREDICTION_REQUIRED_COLUMNS = [
    "split",
    "model_name",
    "window_id",
    "participant_id",
    "trial_id",
    "true_label",
    "predicted_label",
    "predicted_probability",
]


def load_existing_csv(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load a CSV if possible, returning warnings instead of raising."""
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(), [f"Missing file: {path}"]
    if csv_path.stat().st_size == 0:
        return pd.DataFrame(), [f"Empty file: {path}"]
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), [f"Empty or unreadable CSV: {path}"]
    if df.empty:
        return df, [f"CSV has no rows: {path}"]
    return df, []


def load_all_metrics(metric_files: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Load and combine all available metric files."""
    frames = []
    warnings = []
    for path in metric_files:
        df, file_warnings = load_existing_csv(path)
        warnings.extend(file_warnings)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = path
        if "model_name" in df.columns:
            df["model_name"] = df["model_name"].map(normalize_model_name)
        else:
            warnings.append(f"Metrics file is missing model_name column: {path}")
            continue
        if "split" not in df.columns:
            warnings.append(f"Metrics file is missing split column: {path}")
            continue
        for column in METRIC_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
        frames.append(df[METRIC_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=METRIC_COLUMNS), warnings

    combined = pd.concat(frames, ignore_index=True)
    combined["_row_order"] = np.arange(len(combined))
    combined = combined.sort_values("_row_order").drop_duplicates(
        subset=["split", "model_name"],
        keep="last",
    )
    combined = combined.drop(columns=["_row_order"]).reset_index(drop=True)
    return combined, warnings


def load_all_predictions(prediction_files: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Load and combine all available prediction files."""
    frames = []
    warnings = []
    for path in prediction_files:
        df, file_warnings = load_existing_csv(path)
        warnings.extend(file_warnings)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = path
        missing = [column for column in PREDICTION_REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            warnings.append(f"Prediction file {path} is missing columns: {', '.join(missing)}")
        if "model_name" in df.columns:
            df["model_name"] = df["model_name"].map(normalize_model_name)
        if "split" not in df.columns:
            df["split"] = _infer_split_from_path(path)
            warnings.append(f"Prediction file {path} is missing split column; inferred from file name.")
        frames.append(df)

    if not frames:
        return pd.DataFrame(), warnings
    return pd.concat(frames, ignore_index=True, sort=False), warnings


def diagnose_prediction_behavior(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate prediction behavior diagnostics by split and model."""
    columns = [
        "split",
        "model_name",
        "number_of_predictions",
        "number_of_truth_true",
        "number_of_lie_true",
        "number_predicted_truth",
        "number_predicted_lie",
        "predicted_truth_ratio",
        "predicted_lie_ratio",
        "unique_predicted_labels",
        "probability_min",
        "probability_max",
        "probability_mean",
        "probability_std",
        "collapse_detected",
        "collapse_reason",
    ]
    if predictions_df.empty or not {"split", "model_name", "predicted_label"}.issubset(predictions_df.columns):
        return pd.DataFrame(columns=columns)

    rows = []
    working = predictions_df.copy()
    working["true_label"] = pd.to_numeric(working.get("true_label"), errors="coerce")
    working["predicted_label"] = pd.to_numeric(working["predicted_label"], errors="coerce")
    if "predicted_probability" in working.columns:
        working["predicted_probability"] = pd.to_numeric(
            working["predicted_probability"],
            errors="coerce",
        )
    else:
        working["predicted_probability"] = np.nan

    for (split, model_name), group in working.groupby(["split", "model_name"], dropna=False):
        predicted = group["predicted_label"].dropna().astype(int)
        true_labels = group["true_label"].dropna().astype(int)
        total = int(len(group))
        predicted_truth = int((predicted == 0).sum())
        predicted_lie = int((predicted == 1).sum())
        truth_ratio = predicted_truth / total if total else np.nan
        lie_ratio = predicted_lie / total if total else np.nan
        unique_predictions = int(predicted.nunique()) if len(predicted) else 0

        collapse_detected = False
        collapse_reason = "none"
        if unique_predictions == 1 and total > 0:
            collapse_detected = True
            collapse_reason = "single_class_prediction"
        elif truth_ratio >= 0.95:
            collapse_detected = True
            collapse_reason = "almost_all_truth"
        elif lie_ratio >= 0.95:
            collapse_detected = True
            collapse_reason = "almost_all_lie"

        probabilities = group["predicted_probability"].dropna()
        rows.append(
            {
                "split": split,
                "model_name": model_name,
                "number_of_predictions": total,
                "number_of_truth_true": int((true_labels == 0).sum()),
                "number_of_lie_true": int((true_labels == 1).sum()),
                "number_predicted_truth": predicted_truth,
                "number_predicted_lie": predicted_lie,
                "predicted_truth_ratio": truth_ratio,
                "predicted_lie_ratio": lie_ratio,
                "unique_predicted_labels": unique_predictions,
                "probability_min": probabilities.min() if not probabilities.empty else np.nan,
                "probability_max": probabilities.max() if not probabilities.empty else np.nan,
                "probability_mean": probabilities.mean() if not probabilities.empty else np.nan,
                "probability_std": probabilities.std(ddof=0) if len(probabilities) > 1 else np.nan,
                "collapse_detected": collapse_detected,
                "collapse_reason": collapse_reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def diagnose_dataset_size() -> dict:
    """Summarize split sizes and participant counts from processed window files."""
    warnings = []
    split_frames = {}
    for split in ["train", "validation", "test"]:
        path = Path(PROCESSED_DATA_DIR) / f"{split}_windows.csv"
        df, file_warnings = load_existing_csv(str(path))
        warnings.extend(file_warnings)
        split_frames[split] = df

    info: dict = {"warnings": warnings}
    all_participants: set[str] = set()
    participant_sets: dict[str, set[str]] = {}
    for split, df in split_frames.items():
        info[f"{split}_windows"] = int(len(df))
        if df.empty:
            participant_sets[split] = set()
            info[f"{split}_participants"] = 0
            continue
        if "participant_id" not in df.columns:
            warnings.append(f"{split}_windows.csv is missing participant_id column.")
            participant_sets[split] = set()
            info[f"{split}_participants"] = 0
            continue
        participants = set(df["participant_id"].dropna().astype(str))
        participant_sets[split] = participants
        all_participants |= participants
        info[f"{split}_participants"] = len(participants)

    info["total_windows"] = sum(info.get(f"{split}_windows", 0) for split in ["train", "validation", "test"])
    info["total_unique_participants"] = len(all_participants)
    overlaps = []
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        overlap = participant_sets.get(left, set()) & participant_sets.get(right, set())
        if overlap:
            overlaps.append(f"{left}-{right}: {', '.join(sorted(overlap))}")
    info["subject_independent_split"] = len(overlaps) == 0
    info["participant_overlap_details"] = "; ".join(overlaps) if overlaps else "none"
    info["train_too_small"] = info["train_windows"] < MIN_RELIABLE_TRAIN_SAMPLES
    info["test_too_small"] = info["test_windows"] < MIN_RELIABLE_TEST_SAMPLES
    info["participant_count_too_small"] = (
        info["total_unique_participants"] < MIN_RELIABLE_PARTICIPANTS
    )
    return info


def rank_models(metrics_df: pd.DataFrame, primary_split: str, primary_metric: str) -> pd.DataFrame:
    """Rank models by the primary metric, preferring the requested split."""
    columns = [
        "rank",
        "model_name",
        "split_used",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
        "number_of_samples",
    ]
    if metrics_df.empty:
        return pd.DataFrame(columns=columns)

    split_used = primary_split if (metrics_df["split"] == primary_split).any() else "validation"
    subset = metrics_df[metrics_df["split"] == split_used].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    if primary_metric not in subset.columns:
        subset[primary_metric] = np.nan

    subset[primary_metric] = pd.to_numeric(subset[primary_metric], errors="coerce")
    subset = subset.sort_values(primary_metric, ascending=False, na_position="last").reset_index(drop=True)
    subset["rank"] = np.arange(1, len(subset) + 1)
    subset["split_used"] = split_used
    for column in columns:
        if column not in subset.columns:
            subset[column] = np.nan
    return subset[columns]


def generate_recommendations(
    metrics_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    dataset_info: dict,
) -> list[str]:
    """Generate practical recommendations from metrics and diagnostics."""
    recommendations = [
        "Collect more participants before treating model metrics as scientifically reliable.",
        "Keep the subject-independent split so participants do not appear in more than one split.",
        "After adding participants, retrain all models from scratch for comparable experiments.",
        "Check class balance in every split before interpreting precision, recall, and F1.",
        "Inspect prediction probability distributions, not only hard labels.",
        "Consider threshold tuning later, after a larger validation set is available.",
        "Keep false positive rate as a critical metric because truthful responses flagged as deceptive are ethically sensitive.",
        "Do not interpret current results as final reliability; they validate the technical pipeline only.",
    ]

    collapsed = diagnostics_df[diagnostics_df.get("collapse_detected", False) == True] if not diagnostics_df.empty else pd.DataFrame()
    if not collapsed.empty:
        collapsed_models = ", ".join(sorted(collapsed["model_name"].astype(str).unique()))
        recommendations.append(f"Investigate prediction collapse for: {collapsed_models}.")
    lstm_collapse = (
        not collapsed.empty
        and (collapsed["model_name"].astype(str) == "lstm").any()
    )
    if lstm_collapse:
        recommendations.append("The LSTM collapsed to single-class behavior and should not be selected without retraining on more data.")

    ranking = rank_models(metrics_df, "test", "f1")
    if not ranking.empty:
        best = str(ranking.iloc[0]["model_name"])
        if best == "gru":
            recommendations.append("GRU is currently best by available F1, but this should be treated as prototype-level evidence.")
        else:
            recommendations.append(f"{best} is currently best by available F1, but this should be treated as prototype-level evidence.")
    if "causal_tcn" in set(metrics_df.get("model_name", pd.Series(dtype=str)).astype(str)):
        recommendations.append("The Causal TCN may need substantially more data before its temporal convolution capacity is useful.")

    if dataset_info.get("train_too_small") or dataset_info.get("test_too_small") or dataset_info.get("participant_count_too_small"):
        recommendations.append("Current dataset size is below reliability thresholds; prioritize data collection over architecture changes.")
    return recommendations


def normalize_model_name(value) -> str:
    """Normalize model names from different report files."""
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "logistic": "logistic_regression",
        "logisticregression": "logistic_regression",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "causal_tcn_classifier": "causal_tcn",
        "tcn": "causal_tcn",
    }
    return aliases.get(text, text)


def _infer_split_from_path(path: str) -> str:
    name = Path(path).name.lower()
    if "validation" in name:
        return "validation"
    if "test" in name:
        return "test"
    return "unknown"
