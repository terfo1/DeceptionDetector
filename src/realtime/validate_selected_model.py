"""Validate selected model artifacts and write model selection reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .model_registry import check_required_files, validate_model_registry
from .model_selection_config import (
    DISABLED_MODELS,
    EXPERIMENTAL_MODELS,
    FALLBACK_MODEL,
    MODEL_SELECTION_OUTPUT_DIR,
    PRELIMINARY_MODEL_THRESHOLDS,
    SELECTED_MODEL,
)


def main() -> None:
    try:
        _run()
    except (OSError, ValueError) as error:
        print(f"Selected model validation failed: {error}")


def _run() -> None:
    print("Validating model registry...")
    output_dir = Path(MODEL_SELECTION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_df = validate_model_registry()
    registry_df.to_csv(output_dir / "model_registry_status.csv", index=False)

    selected_status = check_required_files(SELECTED_MODEL)
    fallback_status = check_required_files(FALLBACK_MODEL)
    config = _selected_model_config(selected_status, fallback_status)
    (output_dir / "selected_model_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "model_selection_report.txt", registry_df, selected_status, fallback_status)
    _write_summary(output_dir / "model_selection_summary.md", selected_status, fallback_status)

    print(f"Selected model: {SELECTED_MODEL}")
    print(f"Fallback model: {FALLBACK_MODEL}")
    print(f"Disabled models: {', '.join(DISABLED_MODELS)}")
    print(f"Experimental models: {', '.join(EXPERIMENTAL_MODELS)}")
    print(f"All required files for selected model: {'OK' if selected_status['all_files_exist'] else 'MISSING'}")
    print(f"Report saved to {MODEL_SELECTION_OUTPUT_DIR}")


def _selected_model_config(selected_status: dict, fallback_status: dict) -> dict:
    return {
        "selected_model": SELECTED_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "disabled_models": DISABLED_MODELS,
        "experimental_models": EXPERIMENTAL_MODELS,
        "thresholds": {
            model_name: {
                "binary_threshold": values["binary_threshold"],
                "low": values["low"],
                "high": values["high"],
                "note": values["note"],
            }
            for model_name, values in PRELIMINARY_MODEL_THRESHOLDS.items()
        },
        "selected_model_files_ok": selected_status["all_files_exist"],
        "fallback_model_files_ok": fallback_status["all_files_exist"],
        "warnings": [
            "Thresholds are preliminary because the dataset is small.",
            "The selected model is for prototype decision-support only.",
            "The system must not be interpreted as a universal lie detector.",
        ],
    }


def _write_report(path: Path, registry_df, selected_status: dict, fallback_status: dict) -> None:
    lines = [
        "Model Selection Report for Live Prototype",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "1. Overview",
        "This report selects the current model for the live prototype based on existing diagnostics. No models are trained in this step.",
        "",
        "2. Selected Model",
        f"selected_model = {SELECTED_MODEL}",
        "Random Forest is selected because it currently has the strongest overall prototype performance and wider probability distribution.",
        "",
        "3. Fallback Model",
        f"fallback_model = {FALLBACK_MODEL}",
        "GRU is retained as the strongest current neural sequence model.",
        "",
        "4. Disabled Models",
        "lstm is disabled due to prediction collapse.",
        "",
        "5. Experimental Models",
        "causal_tcn is experimental due to narrow probability range and all-medium behavior, while remaining useful for future causal realtime inference.",
        "",
        "6. Registry Status",
    ]
    for _, row in registry_df.iterrows():
        lines.append(
            f"- {row['model_name']}: status={row['status']}, role={row['role']}, "
            f"files_ok={row['all_files_exist']}, missing={row['missing_files'] or 'none'}"
        )
    lines.extend(
        [
            "",
            "7. Threshold Policy",
            "- Runtime thresholds remain default 0.40/0.70 for risk bands.",
            "- Calibration outputs are preliminary.",
            "- No threshold is final until more participants are collected.",
            "- Binary threshold for selected model remains 0.50 for prototype.",
            "",
            "8. Scientific/Ethical Warning",
            "- This is a controlled deception-risk prototype.",
            "- It is not a universal lie detector.",
            "- False positives are ethically sensitive.",
            "- Outputs should be treated as risk indicators only.",
            "",
            "9. Next Step",
            "- Run selected model simulation.",
            "- Then build a FastAPI/WebSocket prototype.",
            "- Collect more participants before strong claims.",
            "",
            f"selected_model_files_ok: {selected_status['all_files_exist']}",
            f"fallback_model_files_ok: {fallback_status['all_files_exist']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, selected_status: dict, fallback_status: dict) -> None:
    lines = [
        "# Model Selection Summary",
        "",
        f"- Selected model: `{SELECTED_MODEL}`",
        f"- Fallback model: `{FALLBACK_MODEL}`",
        f"- Disabled model: `{', '.join(DISABLED_MODELS)}`",
        f"- Experimental model: `{', '.join(EXPERIMENTAL_MODELS)}`",
        "- Why Random Forest was selected: strongest current prototype test F1 and wider probability distribution.",
        f"- Selected model files exist: {selected_status['all_files_exist']}",
        f"- Fallback model files exist: {fallback_status['all_files_exist']}",
        "- Next: run selected model simulation, then build the API prototype after more validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
