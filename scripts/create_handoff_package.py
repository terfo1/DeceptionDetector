"""Create a compact documentation handoff package."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("reports/handoff_package")
SELECTED_MODEL_CONFIG = Path("reports/model_selection/selected_model_config.json")

IMPORTANT_DOCS = [
    "docs/FINAL_PROJECT_OVERVIEW.md",
    "docs/ARCHITECTURE.md",
    "docs/DATA_COLLECTION_GUIDE.md",
    "docs/DATASET_GUIDE.md",
    "docs/ML_PIPELINE_GUIDE.md",
    "docs/MODEL_EVALUATION_GUIDE.md",
    "docs/LIVE_INFERENCE_GUIDE.md",
    "docs/DASHBOARD_GUIDE.md",
    "docs/REPRODUCIBILITY_GUIDE.md",
    "docs/ETHICS_AND_LIMITATIONS.md",
    "docs/FUTURE_WORK.md",
    "docs/TROUBLESHOOTING.md",
    "docs/REPORTS_INDEX.md",
]

IMPORTANT_REPORTS = [
    "reports/model_comparison/model_comparison_report.txt",
    "reports/threshold_calibration/threshold_calibration_report.txt",
    "reports/model_selection/model_selection_report.txt",
    "reports/tracking/tracking_report.txt",
    "reports/pipeline/pipeline_run_report.txt",
]


def main() -> None:
    warnings: list[str] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = _load_model_selection(warnings)

    files = {
        "HANDOFF_README.md": _handoff_readme(),
        "PROJECT_STATUS.md": _project_status(selection),
        "QUICK_START.md": _quick_start(),
        "COMMANDS_CHEATSHEET.md": _commands_cheatsheet(),
        "FILES_INDEX.md": _files_index(),
        "KNOWN_LIMITATIONS.md": _known_limitations(),
        "NEXT_STEPS.md": _next_steps(),
    }

    for filename, content in files.items():
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")

    manifest = _manifest(selection, warnings)
    (OUTPUT_DIR / "handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    zip_path = OUTPUT_DIR / "handoff_package.zip"
    try:
        _create_zip(zip_path)
        print(f"ZIP created: {zip_path}")
    except Exception as exc:
        warnings.append(f"ZIP creation skipped: {exc}")
        print(f"Warning: ZIP creation skipped: {exc}")
        manifest["warnings"] = warnings
        (OUTPUT_DIR / "handoff_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    _print_warnings(warnings)
    print(f"Handoff package created at {OUTPUT_DIR}")


def _load_model_selection(warnings: list[str]) -> dict[str, Any]:
    default = {
        "selected_model": "random_forest",
        "fallback_model": "gru",
        "disabled_models": ["lstm"],
        "experimental_models": ["causal_tcn"],
    }
    if not SELECTED_MODEL_CONFIG.exists():
        warnings.append(f"Missing selected model config: {SELECTED_MODEL_CONFIG}")
        return default
    try:
        data = json.loads(SELECTED_MODEL_CONFIG.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"Invalid selected model config JSON: {exc}")
        return default
    return {
        "selected_model": data.get("selected_model", default["selected_model"]),
        "fallback_model": data.get("fallback_model", default["fallback_model"]),
        "disabled_models": data.get("disabled_models", default["disabled_models"]),
        "experimental_models": data.get("experimental_models", default["experimental_models"]),
    }


def _handoff_readme() -> str:
    return """# Handoff Package

This package summarizes the current project state and points to the detailed documentation and reports needed by another developer, researcher, or supervisor.

The project estimates deception-risk patterns under a controlled experimental protocol. It is not a universal lie detector and does not produce final truth/lie judgments.

Start with:

1. `QUICK_START.md`
2. `PROJECT_STATUS.md`
3. `COMMANDS_CHEATSHEET.md`
4. `KNOWN_LIMITATIONS.md`
5. `NEXT_STEPS.md`
"""


def _project_status(selection: dict[str, Any]) -> str:
    return f"""# Project Status

## Completed Scope

Steps 1-21 are implemented: project definition, protocol, data format, experiment app, preprocessing, subject-independent split, baseline models, neural sequence models, Causal TCN, diagnostics, replay simulation, threshold calibration, model selection, FastAPI/WebSocket API, WebSocket client, live monitor, gaze adapters, data collection workflow, tracking, and pipeline runner.

## Current Model Selection

- Primary model: `{selection['selected_model']}`
- Fallback model: `{selection['fallback_model']}`
- Disabled models: `{', '.join(selection['disabled_models'])}`
- Experimental models: `{', '.join(selection['experimental_models'])}`

## API Status

The live inference API is implemented with FastAPI and WebSocket endpoints. It should be started locally with `uvicorn src.api.app:app --reload`.

## Dashboard Status

The minimal HTML/JavaScript dashboard is available at `web/live_monitor.html` and can also be served from `/static/live_monitor.html` when the API is running.

## Current Limitations

The dataset is small, metrics are prototype-level, thresholds are preliminary, and no real eye tracker SDK is integrated yet.
"""


def _quick_start() -> str:
    return """# Quick Start

## 1. Install Requirements

```bash
pip install -r requirements.txt
```

## 2. Run Pipeline Dry Run

```bash
python -m src.pipeline.run_pipeline --mode full --dry-run
```

## 3. Start API

```bash
uvicorn src.api.app:app --reload
```

## 4. Check Status

Open:

```text
http://127.0.0.1:8000/status
```

## 5. Open Dashboard

```text
web/live_monitor.html
```

## 6. Run WebSocket Client

```bash
python scripts/live_ws_test_client.py
```
"""


def _commands_cheatsheet() -> str:
    return """# Commands Cheatsheet

## Data Collection

```bash
python -m src.data_collection.collection_checklist
python -m src.data_collection.experiment_app
```

## Preprocessing

```bash
python -m src.preprocessing.validate_raw_data
python -m src.preprocessing.build_windows
python -m src.training.create_splits
```

## Training

```bash
python -m src.models.train_baselines
python -m src.training.build_sequence_dataset
python -m src.models.train_sequence_models
python -m src.models.train_tcn_model
```

## Evaluation

```bash
python -m src.models.evaluate_baselines
python -m src.models.evaluate_sequence_models
python -m src.models.evaluate_tcn_model
python -m src.analysis.generate_model_comparison
python -m src.analysis.generate_threshold_report
```

## Real-Time Simulation

```bash
python -m src.realtime.run_realtime_simulation --model-type random_forest
python -m src.realtime.run_selected_model_simulation
```

## API

```bash
uvicorn src.api.app:app --reload
```

## Dashboard

```text
web/live_monitor.html
```

## Tracking

```bash
python -m src.tracking.generate_tracking_report
```

## Pipeline

```bash
python -m src.pipeline.run_pipeline --mode full
python -m src.pipeline.run_pipeline --mode full --dry-run
```
"""


def _files_index() -> str:
    docs = "\n".join(f"- `{path}`" for path in IMPORTANT_DOCS)
    reports = "\n".join(f"- `{path}`" for path in IMPORTANT_REPORTS)
    return f"""# Files Index

## Important Documentation

{docs}

## Important Reports

{reports}

## Runtime Entry Points

- `src.pipeline.run_pipeline`
- `src.api.app`
- `src.data_collection.experiment_app`
- `scripts/live_ws_test_client.py`
- `scripts/create_handoff_package.py`
"""


def _known_limitations() -> str:
    return """# Known Limitations

- The dataset is small, so metrics are prototype-level.
- `random_forest` is selected for the current prototype because it performs best on available test F1.
- `gru` is retained as fallback.
- `lstm` is disabled because prediction collapse was detected.
- `causal_tcn` is not reliable yet because its current probability range is narrow.
- There is no real eye tracker SDK integration yet.
- There is no production security, authentication, database, or Docker packaging.
- The system is not a universal lie detector and must not be used as an autonomous truth/lie decision tool.
"""


def _next_steps() -> str:
    return """# Next Steps

- Collect 30+ participants for a stronger research version.
- Rerun the full pipeline after adding data.
- Recalibrate thresholds on a larger validation set.
- Test with a real eye tracker.
- Improve neural models after more data is available.
- Add Docker packaging.
- Add authentication before any networked use.
- Prepare final thesis/demo report.
"""


def _manifest(selection: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_title": "Real-Time Eye-Tracking-Based Deception Risk Detection Using Neural Networks",
        "selected_model": selection["selected_model"],
        "fallback_model": selection["fallback_model"],
        "disabled_models": selection["disabled_models"],
        "experimental_models": selection["experimental_models"],
        "important_docs": IMPORTANT_DOCS,
        "important_reports": IMPORTANT_REPORTS,
        "key_commands": [
            "python -m src.pipeline.run_pipeline --mode full --dry-run",
            "python -m src.pipeline.run_pipeline --mode full",
            "uvicorn src.api.app:app --reload",
            "python scripts/live_ws_test_client.py",
            "python -m src.tracking.generate_tracking_report",
        ],
        "warnings": [
            *warnings,
            "The project estimates deception risk under controlled experimental conditions and is not a universal lie detector.",
            "Current metrics are prototype-level because the dataset is small.",
        ],
    }


def _create_zip(zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in OUTPUT_DIR.glob("*.md"):
            archive.write(file_path, file_path.name)
        manifest = OUTPUT_DIR / "handoff_manifest.json"
        if manifest.exists():
            archive.write(manifest, manifest.name)
        for doc in IMPORTANT_DOCS:
            path = Path(doc)
            if path.exists():
                archive.write(path, doc)


def _print_warnings(warnings: list[str]) -> None:
    for doc in IMPORTANT_DOCS:
        if not Path(doc).exists():
            warnings.append(f"Missing documentation file: {doc}")
    for report in IMPORTANT_REPORTS:
        if not Path(report).exists():
            warnings.append(f"Missing report file: {report}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    try:
        main()
    except PermissionError as exc:
        print(f"Could not create handoff package due to a permission error: {exc}")
    except Exception as exc:
        print(f"Handoff package generation failed: {exc}")
