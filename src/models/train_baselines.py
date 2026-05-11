"""Train baseline classifiers on aggregated window-level features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .baseline_config import (
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORT_OUTPUT_DIR,
    TARGET_COLUMN,
    TEST_FEATURES_FILE,
    TRAIN_FEATURES_FILE,
    VALIDATION_FEATURES_FILE,
)
from .baseline_utils import (
    calculate_metrics,
    get_feature_columns,
    load_split_features,
    prepare_features,
    save_json,
    validate_splits,
)

PREDICTION_COLUMNS = [
    "split",
    "model_name",
    "window_id",
    "participant_id",
    "trial_id",
    "true_label",
    "predicted_label",
    "predicted_probability",
]
METRIC_COLUMNS = [
    "split",
    "model_name",
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


def main() -> None:
    try:
        _run_training()
    except (FileNotFoundError, ValueError) as error:
        print(f"Baseline training failed: {error}")


def _run_training() -> None:
    model_dir = Path(MODEL_OUTPUT_DIR)
    report_dir = Path(REPORT_OUTPUT_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Training baseline models...")
    train_df, val_df, test_df = load_split_features()
    warnings = validate_splits(train_df, val_df, test_df)

    feature_columns = get_feature_columns(train_df)
    save_json(feature_columns, model_dir / "feature_columns.json")

    X_train, y_train = prepare_features(train_df, feature_columns)
    X_val, y_val = prepare_features(val_df, feature_columns)

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Features used: {len(feature_columns)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if not X_val.empty else X_val
    joblib.dump(scaler, model_dir / "scaler.joblib")

    print("Training Logistic Regression...")
    logistic_regression = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    logistic_regression.fit(X_train_scaled, y_train)
    joblib.dump(logistic_regression, model_dir / "logistic_regression.joblib")

    print("Training Random Forest...")
    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    random_forest.fit(X_train, y_train)
    joblib.dump(random_forest, model_dir / "random_forest.joblib")

    models = {
        "logistic_regression": (logistic_regression, X_val_scaled),
        "random_forest": (random_forest, X_val),
    }

    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    validation_metrics: dict[str, dict] = {}

    if val_df.empty:
        print("Validation set is empty. Validation evaluation skipped.")
    else:
        for model_name, (model, X_eval) in models.items():
            predictions, metrics = _evaluate_model(
                model=model,
                model_name=model_name,
                split_name="validation",
                split_df=val_df,
                X_eval=X_eval,
                y_eval=y_val,
            )
            prediction_rows.extend(predictions)
            metric_rows.append(metrics)
            validation_metrics[model_name] = metrics
            print(f"Validation F1 {model_name}: {metrics['f1']:.4f}")

    pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).to_csv(
        report_dir / "validation_predictions.csv",
        index=False,
    )
    pd.DataFrame(metric_rows, columns=METRIC_COLUMNS).to_csv(
        report_dir / "baseline_metrics.csv",
        index=False,
    )
    _write_report(
        report_dir / "baseline_report.txt",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=feature_columns,
        validation_metrics=validation_metrics,
        warnings=warnings,
    )

    print("Files saved to models/baselines and reports/baselines")


def _evaluate_model(
    model,
    model_name: str,
    split_name: str,
    split_df: pd.DataFrame,
    X_eval,
    y_eval: pd.Series,
) -> tuple[list[dict], dict]:
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    metrics = calculate_metrics(y_eval, y_pred, y_proba)
    metrics.update(_label_counts(split_name, model_name, y_eval))

    rows = []
    for row_index, (_, row) in enumerate(split_df.iterrows()):
        rows.append(
            {
                "split": split_name,
                "model_name": model_name,
                "window_id": row.get("window_id", ""),
                "participant_id": row.get("participant_id", ""),
                "trial_id": row.get("trial_id", ""),
                "true_label": int(y_eval.iloc[row_index]),
                "predicted_label": int(y_pred[row_index]),
                "predicted_probability": float(y_proba[row_index]),
            }
        )
    return rows, metrics


def _label_counts(split_name: str, model_name: str, y: pd.Series) -> dict:
    return {
        "split": split_name,
        "model_name": model_name,
        "number_of_samples": int(len(y)),
        "number_of_truth": int((y == 0).sum()),
        "number_of_lie": int((y == 1).sum()),
    }


def _write_report(
    path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    validation_metrics: dict[str, dict],
    warnings: list[str],
) -> None:
    lines = [
        "Baseline Model Training Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Input files used:",
        f"- {PROCESSED_DATA_DIR}/{TRAIN_FEATURES_FILE}",
        f"- {PROCESSED_DATA_DIR}/{VALIDATION_FEATURES_FILE}",
        f"- {PROCESSED_DATA_DIR}/{TEST_FEATURES_FILE}",
        "",
        f"Train samples: {len(train_df)}",
        f"Validation samples: {len(val_df)}",
        f"Test samples: {len(test_df)}",
        f"Number of features used: {len(feature_columns)}",
        "",
        "Feature columns:",
        *[f"- {column}" for column in feature_columns],
        "",
        f"Train label distribution: {_distribution(train_df)}",
        f"Validation label distribution: {_distribution(val_df)}",
        "",
        "Model names:",
        "- logistic_regression",
        "- random_forest",
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
            "These are baseline models using aggregated window features. They are not neural sequence models and should not be interpreted as a final deception detector.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _distribution(df: pd.DataFrame) -> dict[int, int]:
    if df.empty or TARGET_COLUMN not in df.columns:
        return {}
    counts = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def _format_metrics(metrics: dict) -> str:
    metric_names = [
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
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in metric_names)


if __name__ == "__main__":
    main()
