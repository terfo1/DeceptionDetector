"""Evaluate trained baseline classifiers on the test split."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .baseline_config import MODEL_OUTPUT_DIR, PROCESSED_DATA_DIR, REPORT_OUTPUT_DIR, TEST_FEATURES_FILE
from .baseline_utils import calculate_metrics, load_json, prepare_features
from .train_baselines import METRIC_COLUMNS, PREDICTION_COLUMNS, _distribution, _evaluate_model, _format_metrics


def main() -> None:
    try:
        _run_evaluation()
    except (FileNotFoundError, ValueError) as error:
        print(f"Baseline evaluation failed: {error}")


def _run_evaluation() -> None:
    model_dir = Path(MODEL_OUTPUT_DIR)
    report_dir = Path(REPORT_OUTPUT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    logistic_regression = _load_required(model_dir / "logistic_regression.joblib")
    random_forest = _load_required(model_dir / "random_forest.joblib")
    scaler = _load_required(model_dir / "scaler.joblib")
    feature_columns = load_json(model_dir / "feature_columns.json")

    test_path = Path(PROCESSED_DATA_DIR) / TEST_FEATURES_FILE
    if not test_path.exists():
        raise FileNotFoundError(f"Required test split file is missing: {test_path}")
    test_df = pd.read_csv(test_path)

    prediction_path = report_dir / "test_predictions.csv"
    if test_df.empty:
        print("Test set is empty. Evaluation skipped.")
        pd.DataFrame(columns=PREDICTION_COLUMNS).to_csv(prediction_path, index=False)
        _update_report(report_dir / "baseline_report.txt", test_df, {}, ["Test split is empty."])
        return

    X_test, y_test = prepare_features(test_df, feature_columns)
    X_test_scaled = scaler.transform(X_test)

    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    test_metrics: dict[str, dict] = {}
    models = {
        "logistic_regression": (logistic_regression, X_test_scaled),
        "random_forest": (random_forest, X_test),
    }

    for model_name, (model, X_eval) in models.items():
        predictions, metrics = _evaluate_model(
            model=model,
            model_name=model_name,
            split_name="test",
            split_df=test_df,
            X_eval=X_eval,
            y_eval=y_test,
        )
        prediction_rows.extend(predictions)
        metric_rows.append(metrics)
        test_metrics[model_name] = metrics
        print(f"Test F1 {model_name}: {metrics['f1']:.4f}")

    pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).to_csv(prediction_path, index=False)
    _upsert_metrics(report_dir / "baseline_metrics.csv", metric_rows)

    warnings = []
    if len(test_df) < 10:
        warnings.append("Test set is too small for stable baseline evaluation.")
    if y_test.nunique(dropna=True) < 2:
        warnings.append("Test set has only one class; ROC-AUC and average precision are unavailable.")
    _update_report(report_dir / "baseline_report.txt", test_df, test_metrics, warnings)
    print("Test predictions and metrics saved to reports/baselines")


def _load_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required baseline artifact is missing: {path}")
    return joblib.load(path)


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
    test_df: pd.DataFrame,
    test_metrics: dict[str, dict],
    warnings: list[str],
) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "Baseline Model Training Report\n"
    marker = "\nTest Evaluation\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"

    lines = [
        "",
        "Test Evaluation",
        f"Test sample count: {len(test_df)}",
        f"Test label distribution: {_distribution(test_df)}",
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
