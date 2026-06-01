"""Generate a unified model comparison and diagnostics report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .comparison_config import (
    METRIC_FILES,
    OUTPUT_DIR,
    PREDICTION_FILES,
    PRIMARY_METRIC,
    PRIMARY_SPLIT,
)
from .model_diagnostics import (
    generate_recommendations,
    diagnose_dataset_size,
    diagnose_prediction_behavior,
    load_all_metrics,
    load_all_predictions,
    rank_models,
)


def main() -> None:
    try:
        _run()
    except OSError as error:
        print(f"Model comparison failed due to a file error: {error}")
    except ValueError as error:
        print(f"Model comparison failed: {error}")


def _run() -> None:
    print("Generating model comparison and diagnostics...")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df, metric_warnings = load_all_metrics(METRIC_FILES)
    predictions_df, prediction_warnings = load_all_predictions(PREDICTION_FILES)
    diagnostics_df = diagnose_prediction_behavior(predictions_df)
    dataset_info = diagnose_dataset_size()
    ranking_df = rank_models(metrics_df, PRIMARY_SPLIT, PRIMARY_METRIC)
    recommendations = generate_recommendations(metrics_df, diagnostics_df, dataset_info)

    metrics_df.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    diagnostics_df.to_csv(output_dir / "prediction_diagnostics.csv", index=False)
    ranking_df.to_csv(output_dir / "model_ranking.csv", index=False)

    all_warnings = metric_warnings + prediction_warnings + dataset_info.get("warnings", [])
    _write_text_report(
        output_dir / "model_comparison_report.txt",
        metrics_df,
        diagnostics_df,
        ranking_df,
        dataset_info,
        recommendations,
        all_warnings,
    )
    _write_summary(output_dir / "model_comparison_summary.md", ranking_df, diagnostics_df, dataset_info)
    _write_recommendations(output_dir / "recommendations.md", recommendations)
    _write_optional_plots(output_dir, metrics_df, diagnostics_df, predictions_df)

    best_model = ranking_df.iloc[0]["model_name"] if not ranking_df.empty else "none"
    collapsed_models = _collapsed_model_list(diagnostics_df)
    print(f"Best model by available {PRIMARY_METRIC}: {best_model}")
    print(f"Collapsed models detected: {collapsed_models or 'none'}")
    print(f"Files saved to {OUTPUT_DIR}")


def _write_text_report(
    path: Path,
    metrics_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    dataset_info: dict,
    recommendations: list[str],
    warnings: list[str],
) -> None:
    detected_models = sorted(metrics_df["model_name"].dropna().astype(str).unique()) if not metrics_df.empty else []
    best_line = "No model ranking could be calculated."
    if not ranking_df.empty:
        best = ranking_df.iloc[0]
        best_line = (
            f"Best model by {best['split_used']} F1: {best['model_name']} "
            f"(F1={_fmt(best.get('f1'))})."
        )
        if str(best["model_name"]) == "gru":
            best_line += " On the current prototype dataset, GRU achieved the strongest F1 among available models."

    lines = [
        "Model Comparison and Diagnostics Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "1. Overview",
        "This report compares baseline models and neural sequence models trained in the controlled eye-tracking deception-risk pipeline.",
        "It reads existing metrics and predictions only; no models are trained in this step.",
        "",
        "2. Dataset Size Summary",
        f"- train windows: {dataset_info.get('train_windows', 0)}",
        f"- validation windows: {dataset_info.get('validation_windows', 0)}",
        f"- test windows: {dataset_info.get('test_windows', 0)}",
        f"- total windows: {dataset_info.get('total_windows', 0)}",
        f"- train participants: {dataset_info.get('train_participants', 0)}",
        f"- validation participants: {dataset_info.get('validation_participants', 0)}",
        f"- test participants: {dataset_info.get('test_participants', 0)}",
        f"- total unique participants: {dataset_info.get('total_unique_participants', 0)}",
        f"- subject-independent split: {dataset_info.get('subject_independent_split', False)}",
        f"- participant overlap details: {dataset_info.get('participant_overlap_details', 'unknown')}",
        f"- train too small: {dataset_info.get('train_too_small', True)}",
        f"- test too small: {dataset_info.get('test_too_small', True)}",
        f"- participant count too small: {dataset_info.get('participant_count_too_small', True)}",
        "",
        "3. Available Models",
    ]
    lines.extend([f"- {model}" for model in detected_models] if detected_models else ["- none detected"])
    lines.extend(["", "4. Metrics Summary"])
    lines.extend(_metrics_lines(metrics_df))
    lines.extend(["", "5. Model Ranking", best_line])
    lines.extend(_ranking_lines(ranking_df))
    lines.extend(["", "6. Prediction Behavior Diagnostics"])
    lines.extend(_diagnostic_lines(diagnostics_df))
    if _lstm_single_class_detected(diagnostics_df):
        lines.append(
            "LSTM shows single-class prediction behavior on the evaluated split. "
            "It predicts only the truthful class, resulting in F1 = 0 for the deceptive class."
        )
    lines.extend(
        [
            "",
            "7. False Positive and False Negative Discussion",
            "A false positive is a truthful response predicted as deceptive. A false negative is a deceptive response predicted as truthful.",
            "Both are important, but false positives are especially sensitive ethically because they can wrongly flag truthful behavior as deceptive.",
            "",
            "8. Reliability Assessment",
            _reliability_text(dataset_info),
            "The current results validate the technical pipeline but are not sufficient for strong scientific claims.",
            "",
            "9. Recommendations",
        ]
    )
    lines.extend([f"- {item}" for item in recommendations])
    lines.extend(
        [
            "",
            "10. Next Step",
            "Collect more participants, then rerun preprocessing, splitting, sequence dataset generation, and all training/evaluation scripts.",
            "After enough data is available, perform threshold calibration and real-time simulation.",
            "",
            "Warnings",
        ]
    )
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    ranking_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    dataset_info: dict,
) -> None:
    best = "No model ranking available"
    if not ranking_df.empty:
        best = f"{ranking_df.iloc[0]['model_name']} by {ranking_df.iloc[0]['split_used']} F1"
    collapsed = _collapsed_model_list(diagnostics_df) or "none"
    lines = [
        "# Model Comparison Summary",
        "",
        f"- Best current model: {best}",
        "- Main issue: dataset size is too small for strong scientific reliability.",
        f"- Collapsed models: {collapsed}",
        f"- LSTM collapse note: {'detected' if _lstm_single_class_detected(diagnostics_df) else 'not detected'}",
        f"- Dataset size warning: train_too_small={dataset_info.get('train_too_small')}, test_too_small={dataset_info.get('test_too_small')}, participant_count_too_small={dataset_info.get('participant_count_too_small')}",
        "- Next recommended action: collect more participants and retrain all models from scratch.",
        "",
        "Current results are prototype-level and should not be used for strong scientific claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recommendations(path: Path, recommendations: list[str]) -> None:
    commands = [
        "python -m src.preprocessing.validate_raw_data",
        "python -m src.preprocessing.build_windows",
        "python -m src.training.create_splits",
        "python -m src.models.train_baselines",
        "python -m src.models.evaluate_baselines",
        "python -m src.training.build_sequence_dataset",
        "python -m src.models.train_sequence_models",
        "python -m src.models.evaluate_sequence_models",
        "python -m src.models.train_tcn_model",
        "python -m src.models.evaluate_tcn_model",
        "python -m src.analysis.generate_model_comparison",
    ]
    lines = [
        "# Recommendations",
        "",
        "## Data Collection",
        "- Collect more participants before making scientific claims about model reliability.",
        "- Keep the controlled experimental protocol and preserve participant identifiers for subject-independent splitting.",
        "",
        "## Model Retraining",
        "- Retrain all models from scratch after adding new participants.",
        "- Fine-tuning can be added later, but full retraining is preferred for comparable experiments.",
        "",
        "## Evaluation",
        "- Compare F1, ROC-AUC, false positive rate, and false negative rate together.",
        "- Treat single-class prediction behavior as a model failure mode.",
        "",
        "## Threshold Tuning",
        "- Consider threshold tuning only after a larger validation split is available.",
        "- Keep false positive rate visible because truthful responses flagged as deceptive are ethically sensitive.",
        "",
        "## Real-Time Readiness",
        "- Run offline diagnostics before any real-time simulation.",
        "- Use the Causal TCN as a candidate for future streaming compatibility, but do not deploy it from the current small dataset.",
        "",
        "## Ethical Interpretation",
        "- The system estimates deception risk under a controlled protocol.",
        "- It must not be presented as a universal lie detector.",
        "",
        "## How to update the models after collecting more data",
        "",
        "For clean scientific evaluation, retrain from scratch after adding new participants:",
        "",
        "```bash",
        *commands,
        "```",
        "",
        "## Current Diagnostic Recommendations",
    ]
    lines.extend([f"- {item}" for item in recommendations])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_optional_plots(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not available. Plot generation skipped.")
        return

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy() if not metrics_df.empty else pd.DataFrame()
    if not test_metrics.empty and "f1" in test_metrics.columns:
        test_metrics["f1"] = pd.to_numeric(test_metrics["f1"], errors="coerce")
        test_metrics = test_metrics.dropna(subset=["f1"])
        if not test_metrics.empty:
            plt.figure(figsize=(8, 4))
            plt.bar(test_metrics["model_name"], test_metrics["f1"])
            plt.ylabel("Test F1")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "test_f1_comparison.png", dpi=150)
            plt.close()

    test_diag = diagnostics_df[diagnostics_df["split"] == "test"].copy() if not diagnostics_df.empty else pd.DataFrame()
    if not test_diag.empty:
        x = np.arange(len(test_diag))
        plt.figure(figsize=(8, 4))
        plt.bar(x, test_diag["number_predicted_truth"], label="predicted truth")
        plt.bar(x, test_diag["number_predicted_lie"], bottom=test_diag["number_predicted_truth"], label="predicted lie")
        plt.xticks(x, test_diag["model_name"], rotation=30, ha="right")
        plt.ylabel("Predicted labels")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "predicted_label_distribution.png", dpi=150)
        plt.close()

    if not predictions_df.empty and "predicted_probability" in predictions_df.columns:
        test_predictions = predictions_df[predictions_df["split"] == "test"].copy()
        test_predictions["predicted_probability"] = pd.to_numeric(
            test_predictions["predicted_probability"],
            errors="coerce",
        )
        test_predictions = test_predictions.dropna(subset=["predicted_probability"])
        if not test_predictions.empty:
            plt.figure(figsize=(8, 4))
            for model_name, group in test_predictions.groupby("model_name"):
                plt.hist(group["predicted_probability"], bins=10, alpha=0.5, label=model_name)
            plt.xlabel("Predicted deception probability")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "probability_distribution.png", dpi=150)
            plt.close()


def _metrics_lines(metrics_df: pd.DataFrame) -> list[str]:
    if metrics_df.empty:
        return ["- No metrics available."]
    lines = []
    for _, row in metrics_df.sort_values(["split", "model_name"]).iterrows():
        lines.append(
            f"- {row['split']} / {row['model_name']}: "
            f"F1={_fmt(row.get('f1'))}, ROC-AUC={_fmt(row.get('roc_auc'))}, "
            f"FPR={_fmt(row.get('false_positive_rate'))}, FNR={_fmt(row.get('false_negative_rate'))}, "
            f"n={_fmt(row.get('number_of_samples'), decimals=0)}"
        )
    return lines


def _ranking_lines(ranking_df: pd.DataFrame) -> list[str]:
    if ranking_df.empty:
        return ["- No ranking available."]
    return [
        f"- {int(row['rank'])}. {row['model_name']} ({row['split_used']} F1={_fmt(row.get('f1'))})"
        for _, row in ranking_df.iterrows()
    ]


def _diagnostic_lines(diagnostics_df: pd.DataFrame) -> list[str]:
    if diagnostics_df.empty:
        return ["- No prediction diagnostics available."]
    lines = []
    for _, row in diagnostics_df.sort_values(["split", "model_name"]).iterrows():
        lines.append(
            f"- {row['split']} / {row['model_name']}: predicted truth={int(row['number_predicted_truth'])}, "
            f"predicted lie={int(row['number_predicted_lie'])}, collapse={row['collapse_detected']} "
            f"({row['collapse_reason']})"
        )
    return lines


def _reliability_text(dataset_info: dict) -> str:
    if (
        dataset_info.get("train_too_small")
        or dataset_info.get("test_too_small")
        or dataset_info.get("participant_count_too_small")
    ):
        return "Current results are prototype-level because the train/test window counts or participant count are below reliability thresholds."
    return "Dataset size thresholds are met, but scientific reliability still depends on protocol quality, calibration, and external validation."


def _collapsed_model_list(diagnostics_df: pd.DataFrame) -> str:
    if diagnostics_df.empty:
        return ""
    collapsed = diagnostics_df[diagnostics_df["collapse_detected"] == True]
    if collapsed.empty:
        return ""
    return ", ".join(sorted(collapsed["model_name"].astype(str).unique()))


def _lstm_single_class_detected(diagnostics_df: pd.DataFrame) -> bool:
    if diagnostics_df.empty:
        return False
    lstm_rows = diagnostics_df[
        (diagnostics_df["model_name"].astype(str) == "lstm")
        & (diagnostics_df["collapse_reason"].astype(str) == "single_class_prediction")
        & (diagnostics_df["number_predicted_truth"] > 0)
        & (diagnostics_df["number_predicted_lie"] == 0)
    ]
    return not lstm_rows.empty


def _fmt(value, decimals: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NaN"
        if decimals == 0:
            return str(int(value))
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
