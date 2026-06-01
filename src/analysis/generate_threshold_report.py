"""Generate threshold calibration and probability diagnostics reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .threshold_calibration import (
    analyze_default_risk_bands,
    compute_probability_diagnostics,
    create_selected_thresholds_json,
    load_all_predictions,
    save_selected_thresholds,
    select_best_thresholds,
    suggest_preliminary_risk_bands,
    sweep_thresholds,
)
from .threshold_config import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    OUTPUT_DIR,
    PREDICTION_FILES,
    REALTIME_PREDICTIONS_FILE,
)


def main() -> None:
    try:
        _run()
    except OSError as error:
        print(f"Threshold report generation failed due to a file error: {error}")
    except ValueError as error:
        print(f"Threshold report generation failed: {error}")


def _run() -> None:
    print("Generating threshold calibration and probability diagnostics...")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df, warnings = load_all_predictions()
    diagnostics_df = compute_probability_diagnostics(predictions_df)
    sweep_df = sweep_thresholds(predictions_df)
    selected_df = select_best_thresholds(sweep_df)
    default_bands_df = analyze_default_risk_bands(predictions_df)
    risk_bands_df = suggest_preliminary_risk_bands(predictions_df)
    selected_payload = create_selected_thresholds_json(selected_df, risk_bands_df, diagnostics_df)

    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    diagnostics_df.to_csv(output_dir / "model_probability_diagnostics.csv", index=False)
    selected_df.to_csv(output_dir / "selected_threshold_details.csv", index=False)
    default_bands_df.to_csv(output_dir / "default_risk_band_diagnostics.csv", index=False)
    risk_bands_df.to_csv(output_dir / "preliminary_risk_band_candidates.csv", index=False)
    save_selected_thresholds(selected_payload, output_dir / "selected_thresholds.json")

    _write_report(
        output_dir / "threshold_calibration_report.txt",
        predictions_df,
        diagnostics_df,
        sweep_df,
        selected_df,
        default_bands_df,
        risk_bands_df,
        warnings,
    )
    _write_summary(
        output_dir / "threshold_calibration_summary.md",
        selected_df,
        default_bands_df,
        diagnostics_df,
    )
    _write_recommendations(output_dir / "recommendations.md")
    _write_optional_plots(output_dir, predictions_df, sweep_df, default_bands_df)

    all_medium = _all_medium_labels(default_bands_df)
    print(f"Loaded prediction rows: {len(predictions_df)}")
    print(f"All-medium model/split pairs: {', '.join(all_medium) if all_medium else 'none'}")
    print(f"Files saved to {OUTPUT_DIR}")


def _write_report(
    path: Path,
    predictions_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    default_bands_df: pd.DataFrame,
    risk_bands_df: pd.DataFrame,
    warnings: list[str],
) -> None:
    lines = [
        "Threshold Calibration and Probability Diagnostics Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "1. Overview",
        "This report analyzes model probability outputs and threshold behavior. It does not train models and does not modify production thresholds.",
        "",
        "2. Input Files",
        *[f"- {path}" for path in [*PREDICTION_FILES, REALTIME_PREDICTIONS_FILE]],
        "",
        "Input warnings:",
    ]
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None"])

    lines.extend(["", "3. Probability Diagnostics"])
    if diagnostics_df.empty:
        lines.append("- No valid probability diagnostics could be calculated.")
    else:
        for _, row in diagnostics_df.sort_values(["split", "model_name"]).iterrows():
            narrow = " narrow_range=True" if row["narrow_probability_range"] else ""
            lines.append(
                f"- {row['split']} / {row['model_name']}: "
                f"min={_fmt(row['probability_min'])}, max={_fmt(row['probability_max'])}, "
                f"mean={_fmt(row['probability_mean'])}, std={_fmt(row['probability_std'])}, "
                f"range={_fmt(row['probability_range'])}, separation={_fmt(row['separation'])}{narrow}"
            )
    if _causal_tcn_realtime_narrow_medium(diagnostics_df, default_bands_df):
        lines.append(
            "The Causal TCN real-time simulation produced a narrow probability range, causing all predictions to fall into the medium risk band under the default 0.40/0.70 thresholds."
        )

    lines.extend(["", "4. Default Risk Band Analysis"])
    if default_bands_df.empty:
        lines.append("- No default risk band analysis available.")
    else:
        for _, row in default_bands_df.sort_values(["split", "model_name"]).iterrows():
            lines.append(
                f"- {row['split']} / {row['model_name']}: low={int(row['low_count'])}, "
                f"medium={int(row['medium_count'])}, high={int(row['high_count'])}, "
                f"all_medium={row['all_medium_detected']}"
            )

    lines.extend(
        [
            "",
            "5. Threshold Sweep",
            "Thresholds from 0.05 to 0.95 were tested in 0.01 increments. Metrics were calculated from existing probabilities only.",
        ]
    )
    lines.extend(_selected_threshold_lines(selected_df, selected_by="best_f1_threshold"))
    lines.extend(["", "6. Selected Preliminary Thresholds"])
    lines.extend(_selected_threshold_lines(selected_df, selected_by=None))
    lines.extend(
        [
            "",
            "7. Reliability Warning",
            "Current thresholds are preliminary because dataset size is small and participant count is low. Thresholds should be recalibrated after collecting more participants.",
            "",
            "8. Ethical Note",
            "In deception-risk estimation, false positives are ethically sensitive because truthful responses may be incorrectly flagged as deceptive. Therefore, conservative thresholds may be preferred in high-stakes scenarios.",
            "",
            "9. Recommendations",
            "- Collect more participants.",
            "- Use the validation split for threshold selection, not the test split.",
            "- Keep the test split for final evaluation.",
            "- Recalibrate thresholds after retraining.",
            "- Avoid applying quantile thresholds as final production rules.",
            "- Compare random_forest and GRU for the next simulation.",
            "- Do not select LSTM until collapse is fixed.",
            "- Do not use Causal TCN thresholds seriously until its probability range improves.",
            "",
            "Risk Band Candidates",
        ]
    )
    if risk_bands_df.empty:
        lines.append("- No risk band candidates available.")
    else:
        for _, row in risk_bands_df.sort_values(["split", "model_name"]).iterrows():
            lines.append(
                f"- {row['split']} / {row['model_name']}: "
                f"low_candidate={_fmt(row['low_threshold_candidate'])}, "
                f"high_candidate={_fmt(row['high_threshold_candidate'])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    selected_df: pd.DataFrame,
    default_bands_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
) -> None:
    best = _best_current_model(selected_df)
    all_medium = _all_medium_labels(default_bands_df)
    narrow = _narrow_labels(diagnostics_df)
    lines = [
        "# Threshold Calibration Summary",
        "",
        f"- Best available model by current F1 if detectable: {best}",
        f"- All-medium issue: {', '.join(all_medium) if all_medium else 'none detected'}",
        f"- Narrow probability range: {', '.join(narrow) if narrow else 'none detected'}",
        "- Preliminary threshold warning: thresholds are diagnostic only until more participants are collected.",
        "- Recommended next action: collect more data, retrain, then calibrate thresholds on validation data.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recommendations(path: Path) -> None:
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
        "python -m src.realtime.run_realtime_simulation --model-type random_forest",
        "python -m src.realtime.run_realtime_simulation --model-type gru",
        "python -m src.analysis.generate_threshold_report",
    ]
    lines = [
        "# Threshold Calibration Recommendations",
        "",
        "## Why thresholds are needed",
        "Model probabilities need decision thresholds before they can be mapped into risk categories. The same probability range may behave differently across model families.",
        "",
        "## Why current 0.40/0.70 bands failed for Causal TCN",
        "The Causal TCN currently produces probabilities in a narrow band near 0.5. Under 0.40/0.70 thresholds, nearly all replay predictions become medium, so the bands are not informative for that model.",
        "",
        "## How to recalibrate after more data",
        "Use validation data for threshold selection, preserve test data for final evaluation, and recalibrate after retraining all models from scratch.",
        "",
        "## Which model to use for next real-time simulation",
        "Use random_forest as the default because it currently has the strongest test F1. Also compare GRU because it is the strongest current neural sequence model.",
        "",
        "## Commands to rerun after collecting more data",
        "",
        "```bash",
        *commands,
        "```",
        "",
        "Do not automatically deploy the preliminary thresholds. They are calibration artifacts only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_optional_plots(
    output_dir: Path,
    predictions_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    default_bands_df: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not available. Plot generation skipped.")
        return
    if predictions_df.empty:
        return

    plt.figure(figsize=(9, 4))
    for (split, model), group in predictions_df.groupby(["split", "model_name"]):
        plt.hist(group["predicted_probability"], bins=15, alpha=0.4, label=f"{split}/{model}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "probability_histogram.png", dpi=150)
    plt.close()

    test_sweep = sweep_df[sweep_df["split"].isin(["test", "realtime_simulation"])] if not sweep_df.empty else pd.DataFrame()
    if not test_sweep.empty:
        plt.figure(figsize=(9, 4))
        for model, group in test_sweep.groupby("model_name"):
            plt.plot(group["threshold"], group["f1"], label=model)
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "threshold_f1_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(9, 4))
        for model, group in test_sweep.groupby("model_name"):
            plt.plot(group["threshold"], group["false_positive_rate"], label=f"{model} FPR")
            plt.plot(group["threshold"], group["false_negative_rate"], linestyle="--", label=f"{model} FNR")
        plt.xlabel("Threshold")
        plt.ylabel("Rate")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / "threshold_fpr_fnr_curve.png", dpi=150)
        plt.close()

    if not default_bands_df.empty:
        plot_df = default_bands_df.copy()
        labels = plot_df["split"] + "/" + plot_df["model_name"]
        x = np.arange(len(plot_df))
        plt.figure(figsize=(10, 4))
        plt.bar(x, plot_df["low_count"], label="low")
        plt.bar(x, plot_df["medium_count"], bottom=plot_df["low_count"], label="medium")
        plt.bar(
            x,
            plot_df["high_count"],
            bottom=plot_df["low_count"] + plot_df["medium_count"],
            label="high",
        )
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "risk_band_distribution.png", dpi=150)
        plt.close()


def _selected_threshold_lines(selected_df: pd.DataFrame, selected_by: str | None) -> list[str]:
    if selected_df.empty:
        return ["- No selected thresholds available."]
    df = selected_df if selected_by is None else selected_df[selected_df["selected_by"] == selected_by]
    if df.empty:
        return ["- No selected thresholds available."]
    return [
        f"- {row['split']} / {row['model_name']} / {row['selected_by']}: "
        f"threshold={_fmt(row['threshold'])}, F1={_fmt(row['f1'])}, "
        f"FPR={_fmt(row['false_positive_rate'])}, FNR={_fmt(row['false_negative_rate'])}"
        for _, row in df.sort_values(["split", "model_name", "selected_by"]).iterrows()
    ]


def _causal_tcn_realtime_narrow_medium(diagnostics_df: pd.DataFrame, default_bands_df: pd.DataFrame) -> bool:
    if diagnostics_df.empty or default_bands_df.empty:
        return False
    diag = diagnostics_df[
        (diagnostics_df["split"] == "realtime_simulation")
        & (diagnostics_df["model_name"] == "causal_tcn")
        & (diagnostics_df["narrow_probability_range"] == True)
    ]
    bands = default_bands_df[
        (default_bands_df["split"] == "realtime_simulation")
        & (default_bands_df["model_name"] == "causal_tcn")
        & (default_bands_df["all_medium_detected"] == True)
    ]
    return not diag.empty and not bands.empty


def _all_medium_labels(default_bands_df: pd.DataFrame) -> list[str]:
    if default_bands_df.empty:
        return []
    rows = default_bands_df[default_bands_df["all_medium_detected"] == True]
    return [f"{row['split']}/{row['model_name']}" for _, row in rows.iterrows()]


def _narrow_labels(diagnostics_df: pd.DataFrame) -> list[str]:
    if diagnostics_df.empty:
        return []
    rows = diagnostics_df[diagnostics_df["narrow_probability_range"] == True]
    return [f"{row['split']}/{row['model_name']}" for _, row in rows.iterrows()]


def _best_current_model(selected_df: pd.DataFrame) -> str:
    if selected_df.empty:
        return "not available"
    test_rows = selected_df[
        (selected_df["split"] == "test")
        & (selected_df["selected_by"] == "best_f1_threshold")
    ].copy()
    if test_rows.empty:
        test_rows = selected_df[selected_df["selected_by"] == "best_f1_threshold"].copy()
    if test_rows.empty:
        return "not available"
    best = test_rows.sort_values("f1", ascending=False).iloc[0]
    return f"{best['model_name']} ({best['split']} F1={_fmt(best['f1'])})"


def _fmt(value) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
