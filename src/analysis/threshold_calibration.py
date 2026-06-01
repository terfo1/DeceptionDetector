"""Threshold calibration and probability diagnostics utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .threshold_config import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    OUTPUT_DIR,
    PREDICTION_FILES,
    REALTIME_PREDICTIONS_FILE,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
)


EXPECTED_COLUMNS = [
    "split",
    "model_name",
    "window_id",
    "participant_id",
    "trial_id",
    "true_label",
    "predicted_label",
    "predicted_probability",
    "source_file",
]


def load_prediction_file(path: str) -> pd.DataFrame:
    """Load one prediction file, returning an empty dataframe for expected issues."""
    df, warnings = _load_prediction_file_with_warnings(path)
    for warning in warnings:
        print(f"Warning: {warning}")
    return df


def load_all_predictions() -> tuple[pd.DataFrame, list[str]]:
    """Load all available offline and realtime prediction files."""
    warnings: list[str] = []
    frames = []
    for path in [*PREDICTION_FILES, REALTIME_PREDICTIONS_FILE]:
        df, file_warnings = _load_prediction_file_with_warnings(path)
        warnings.extend(file_warnings)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS), warnings

    combined = pd.concat(frames, ignore_index=True, sort=False)
    before = len(combined)
    combined["true_label"] = pd.to_numeric(combined.get("true_label"), errors="coerce")
    combined["predicted_probability"] = pd.to_numeric(
        combined.get("predicted_probability"),
        errors="coerce",
    )
    valid_mask = (
        combined["true_label"].isin([0, 1])
        & combined["predicted_probability"].between(0, 1, inclusive="both")
    )
    invalid_count = int((~valid_mask).sum())
    if invalid_count:
        warnings.append(f"Dropped {invalid_count} rows with invalid label or probability values.")
    combined = combined.loc[valid_mask].copy()
    if combined.empty and before > 0:
        warnings.append("All loaded prediction rows were invalid after filtering.")
    combined["true_label"] = combined["true_label"].astype(int)
    if "predicted_label" in combined.columns:
        combined["predicted_label"] = pd.to_numeric(combined["predicted_label"], errors="coerce")
    else:
        combined["predicted_label"] = np.nan
    return combined.reset_index(drop=True), warnings


def compute_probability_diagnostics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize probability distributions by split and model."""
    columns = [
        "split",
        "model_name",
        "n_samples",
        "n_truth",
        "n_lie",
        "probability_min",
        "probability_max",
        "probability_mean",
        "probability_std",
        "probability_median",
        "probability_q10",
        "probability_q25",
        "probability_q75",
        "probability_q90",
        "probability_range",
        "narrow_probability_range",
        "mean_probability_truth",
        "mean_probability_lie",
        "separation",
    ]
    if predictions_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (split, model_name), group in predictions_df.groupby(["split", "model_name"]):
        probabilities = group["predicted_probability"].astype(float)
        truth_probs = group.loc[group["true_label"] == 0, "predicted_probability"].astype(float)
        lie_probs = group.loc[group["true_label"] == 1, "predicted_probability"].astype(float)
        probability_range = float(probabilities.max() - probabilities.min())
        mean_truth = float(truth_probs.mean()) if not truth_probs.empty else np.nan
        mean_lie = float(lie_probs.mean()) if not lie_probs.empty else np.nan
        rows.append(
            {
                "split": split,
                "model_name": model_name,
                "n_samples": int(len(group)),
                "n_truth": int((group["true_label"] == 0).sum()),
                "n_lie": int((group["true_label"] == 1).sum()),
                "probability_min": float(probabilities.min()),
                "probability_max": float(probabilities.max()),
                "probability_mean": float(probabilities.mean()),
                "probability_std": float(probabilities.std(ddof=0)) if len(probabilities) > 1 else np.nan,
                "probability_median": float(probabilities.median()),
                "probability_q10": float(probabilities.quantile(0.10)),
                "probability_q25": float(probabilities.quantile(0.25)),
                "probability_q75": float(probabilities.quantile(0.75)),
                "probability_q90": float(probabilities.quantile(0.90)),
                "probability_range": probability_range,
                "narrow_probability_range": bool(probability_range < 0.15),
                "mean_probability_truth": mean_truth,
                "mean_probability_lie": mean_lie,
                "separation": mean_lie - mean_truth if pd.notna(mean_truth) and pd.notna(mean_lie) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def sweep_thresholds(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate binary metrics across a threshold grid."""
    rows = []
    if predictions_df.empty:
        return pd.DataFrame()
    thresholds = np.round(
        np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP),
        2,
    )
    for (split, model_name), group in predictions_df.groupby(["split", "model_name"]):
        y_true = group["true_label"].astype(int).to_numpy()
        probabilities = group["predicted_probability"].astype(float).to_numpy()
        for threshold in thresholds:
            y_pred = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "split": split,
                    "model_name": model_name,
                    "threshold": float(threshold),
                    "accuracy": accuracy_score(y_true, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
                    "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
                    "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                    "number_of_samples": int(len(y_true)),
                    "number_of_truth": int((y_true == 0).sum()),
                    "number_of_lie": int((y_true == 1).sum()),
                }
            )
    return pd.DataFrame(rows)


def select_best_thresholds(threshold_sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Select preliminary thresholds by F1, balanced accuracy, and conservative FPR."""
    columns = [
        "split",
        "model_name",
        "selected_by",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "number_of_samples",
        "warning",
    ]
    if threshold_sweep_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (split, model_name), group in threshold_sweep_df.groupby(["split", "model_name"]):
        best_f1 = _select_row(group, "f1")
        rows.append(_selection_row(best_f1, "best_f1_threshold", ""))

        best_balanced = _select_row(group, "balanced_accuracy")
        rows.append(_selection_row(best_balanced, "best_balanced_accuracy_threshold", ""))

        candidates = group[group["false_positive_rate"] <= 0.30].copy()
        warning = ""
        if candidates.empty:
            conservative = group.sort_values(
                ["false_positive_rate", "false_negative_rate"],
                ascending=[True, True],
            ).iloc[0]
            warning = "No threshold met false_positive_rate <= 0.30; selected lowest false_positive_rate."
        else:
            conservative = _select_row(candidates, "f1")
        rows.append(_selection_row(conservative, "conservative_threshold", warning))

    return pd.DataFrame(rows, columns=columns)


def analyze_default_risk_bands(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how default low/medium/high risk bands distribute predictions."""
    columns = [
        "split",
        "model_name",
        "low_count",
        "medium_count",
        "high_count",
        "low_ratio",
        "medium_ratio",
        "high_ratio",
        "all_medium_detected",
    ]
    if predictions_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (split, model_name), group in predictions_df.groupby(["split", "model_name"]):
        probabilities = group["predicted_probability"].astype(float)
        total = len(probabilities)
        low = int((probabilities < DEFAULT_LOW_THRESHOLD).sum())
        medium = int(((probabilities >= DEFAULT_LOW_THRESHOLD) & (probabilities < DEFAULT_HIGH_THRESHOLD)).sum())
        high = int((probabilities >= DEFAULT_HIGH_THRESHOLD).sum())
        rows.append(
            {
                "split": split,
                "model_name": model_name,
                "low_count": low,
                "medium_count": medium,
                "high_count": high,
                "low_ratio": low / total if total else np.nan,
                "medium_ratio": medium / total if total else np.nan,
                "high_ratio": high / total if total else np.nan,
                "all_medium_detected": bool(total > 0 and medium / total >= 0.95),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def suggest_preliminary_risk_bands(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Suggest diagnostic quantile-based low/high risk bands."""
    columns = [
        "split",
        "model_name",
        "low_threshold_candidate",
        "high_threshold_candidate",
        "probability_min",
        "probability_max",
        "warning",
    ]
    if predictions_df.empty:
        return pd.DataFrame(columns=columns)

    warning = "Quantile-based risk bands are diagnostic only and should not be used as final thresholds with small datasets."
    rows = []
    for (split, model_name), group in predictions_df.groupby(["split", "model_name"]):
        probabilities = group["predicted_probability"].astype(float)
        rows.append(
            {
                "split": split,
                "model_name": model_name,
                "low_threshold_candidate": float(probabilities.quantile(0.33)),
                "high_threshold_candidate": float(probabilities.quantile(0.66)),
                "probability_min": float(probabilities.min()),
                "probability_max": float(probabilities.max()),
                "warning": warning,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def create_selected_thresholds_json(
    best_thresholds_df: pd.DataFrame,
    risk_bands_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
) -> dict:
    """Create selected-threshold calibration artifact without changing runtime config."""
    payload: dict = {
        "warning": "These thresholds are preliminary and must be recalibrated after collecting more participants.",
        "default_thresholds": {
            "low": DEFAULT_LOW_THRESHOLD,
            "high": DEFAULT_HIGH_THRESHOLD,
        },
        "models": {},
    }
    if best_thresholds_df.empty:
        return payload

    risk_lookup = {
        (row["model_name"], row["split"]): row
        for _, row in risk_bands_df.iterrows()
    } if not risk_bands_df.empty else {}

    for (model_name, split), group in best_thresholds_df.groupby(["model_name", "split"]):
        model_entry = payload["models"].setdefault(str(model_name), {})
        split_entry = model_entry.setdefault(str(split), {})
        for _, row in group.iterrows():
            if row["selected_by"] == "best_f1_threshold":
                split_entry["best_f1_threshold"] = _safe_float(row["threshold"])
            elif row["selected_by"] == "best_balanced_accuracy_threshold":
                split_entry["best_balanced_accuracy_threshold"] = _safe_float(row["threshold"])
            elif row["selected_by"] == "conservative_threshold":
                split_entry["conservative_threshold"] = _safe_float(row["threshold"])
        risk_row = risk_lookup.get((model_name, split))
        if risk_row is not None:
            split_entry["suggested_low_threshold"] = _safe_float(risk_row["low_threshold_candidate"])
            split_entry["suggested_high_threshold"] = _safe_float(risk_row["high_threshold_candidate"])

    payload["probability_diagnostics"] = diagnostics_df.to_dict(orient="records")
    return payload


def save_selected_thresholds(payload: dict, output_path: str | Path | None = None) -> None:
    """Save selected-threshold JSON artifact."""
    path = Path(output_path) if output_path else Path(OUTPUT_DIR) / "selected_thresholds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_prediction_file_with_warnings(path: str) -> tuple[pd.DataFrame, list[str]]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        return pd.DataFrame(), [f"Missing prediction file: {path}"]
    if prediction_path.stat().st_size == 0:
        return pd.DataFrame(), [f"Empty prediction file: {path}"]
    try:
        df = pd.read_csv(prediction_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), [f"Unreadable or empty prediction file: {path}"]
    if df.empty:
        return pd.DataFrame(), [f"Prediction file has no rows: {path}"]

    warnings = []
    df = df.copy()
    df["source_file"] = path
    if prediction_path.name == "realtime_predictions.csv":
        df["predicted_probability"] = df.get("probability")
        df["model_name"] = df.get("model_type")
        df["split"] = "realtime_simulation"
        if "predicted_label" not in df.columns:
            df["predicted_label"] = np.nan
    else:
        missing = [
            column
            for column in ["split", "model_name", "true_label", "predicted_probability"]
            if column not in df.columns
        ]
        if missing:
            warnings.append(f"{path} is missing required columns: {', '.join(missing)}")
            return pd.DataFrame(), warnings

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df["model_name"] = df["model_name"].astype(str).str.strip().str.lower().str.replace("-", "_")
    return df[EXPECTED_COLUMNS], warnings


def _select_row(group: pd.DataFrame, metric: str) -> pd.Series:
    working = group.copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    return working.sort_values(
        [metric, "false_positive_rate", "threshold"],
        ascending=[False, True, True],
        na_position="last",
    ).iloc[0]


def _selection_row(row: pd.Series, selected_by: str, warning: str) -> dict:
    return {
        "split": row["split"],
        "model_name": row["model_name"],
        "selected_by": selected_by,
        "threshold": row["threshold"],
        "accuracy": row["accuracy"],
        "balanced_accuracy": row["balanced_accuracy"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
        "false_positive_rate": row["false_positive_rate"],
        "false_negative_rate": row["false_negative_rate"],
        "number_of_samples": row["number_of_samples"],
        "warning": warning,
    }


def _safe_float(value):
    if pd.isna(value):
        return None
    return float(value)
