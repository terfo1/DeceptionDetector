"""CLI entry point for the one-command project pipeline runner."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .pipeline_config import DEFAULT_MODE, PIPELINE_OUTPUT_DIR, SUPPORTED_MODES
from .pipeline_steps import (
    dry_run_results,
    get_steps_for_mode,
    run_pipeline_steps,
    summarize_pipeline_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run existing project pipeline stages.")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=DEFAULT_MODE)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    try:
        steps = get_steps_for_mode(args.mode)
    except ValueError as exc:
        print(exc)
        return

    output_dir = Path(PIPELINE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline mode: {args.mode}")
    print(f"Selected steps: {len(steps)}")
    for step in steps:
        print(f"- {step['name']}: {'critical' if step['critical'] else 'non-critical'}")

    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    if args.dry_run:
        results = dry_run_results(steps)
        summary = summarize_pipeline_results(results)
        summary["pipeline_status"] = "dry_run"
        print("Dry run complete. No commands were executed.")
    else:
        results = run_pipeline_steps(
            steps,
            stop_on_critical_failure=not args.continue_on_error,
            timeout_seconds=args.timeout_seconds,
        )
        summary = summarize_pipeline_results(results)

    finished_at = datetime.now().isoformat(timespec="seconds")
    total_duration = round(time.perf_counter() - start, 3)
    summary["total_duration_seconds"] = total_duration

    run = {
        "mode": args.mode,
        "dry_run": args.dry_run,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_duration_seconds": total_duration,
        "pipeline_status": summary["pipeline_status"],
        "summary": summary,
        "steps": results,
    }

    _save_outputs(run, output_dir)
    print(f"Pipeline status: {summary['pipeline_status']}")
    print(f"Reports saved to {PIPELINE_OUTPUT_DIR}")


def _save_outputs(run: dict, output_dir: Path) -> None:
    steps_df = pd.DataFrame(run["steps"])
    csv_columns = [
        "step_name",
        "critical",
        "status",
        "return_code",
        "duration_seconds",
        "command",
        "started_at",
        "finished_at",
    ]
    for column in csv_columns:
        if column not in steps_df.columns:
            steps_df[column] = ""
    steps_df[csv_columns].to_csv(output_dir / "pipeline_steps.csv", index=False)

    (output_dir / "latest_pipeline_run.json").write_text(
        json.dumps(run, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pipeline_run_report.txt").write_text(
        _build_text_report(run),
        encoding="utf-8",
    )
    (output_dir / "pipeline_run_summary.md").write_text(
        _build_markdown_summary(run),
        encoding="utf-8",
    )


def _build_text_report(run: dict) -> str:
    summary = run["summary"]
    warnings = _detect_warnings(run["steps"])
    lines = [
        "Pipeline Run Report",
        "",
        "1. Overview",
        f"Mode: {run['mode']}",
        f"Started at: {run['started_at']}",
        f"Finished at: {run['finished_at']}",
        f"Pipeline status: {run['pipeline_status']}",
        f"Total duration seconds: {run['total_duration_seconds']}",
        f"Dry run: {run['dry_run']}",
        "",
        "2. Step Results",
    ]
    for row in run["steps"]:
        lines.extend(
            [
                "",
                f"Step name: {row['step_name']}",
                f"Description: {row['description']}",
                f"Command: {row['command']}",
                f"Critical: {row['critical']}",
                f"Status: {row['status']}",
                f"Return code: {row['return_code']}",
                f"Duration seconds: {row['duration_seconds']}",
                "Stdout tail:",
                row.get("stdout_tail", "") or "",
            ]
        )
        if row.get("stderr_tail"):
            lines.extend(["Stderr tail:", row["stderr_tail"]])

    lines.extend(
        [
            "",
            "3. Summary",
            f"Passed steps: {summary['passed_steps']}",
            f"Failed steps: {summary['failed_steps']}",
            f"Skipped steps: {summary['skipped_steps']}",
            f"Critical failures: {summary['critical_failures']}",
            "",
            "4. Warnings",
            _warnings_text(warnings, summary),
            "",
            "5. Next Actions",
            _next_action(summary),
            "",
        ]
    )
    return "\n".join(lines)


def _build_markdown_summary(run: dict) -> str:
    summary = run["summary"]
    lines = [
        "# Pipeline Run Summary",
        "",
        f"- Mode: {run['mode']}",
        f"- Status: {run['pipeline_status']}",
        f"- Total steps: {summary['total_steps']}",
        f"- Passed: {summary['passed_steps']}",
        f"- Failed: {summary['failed_steps']}",
        f"- Skipped: {summary['skipped_steps']}",
        f"- Critical failures: {summary['critical_failures']}",
        f"- Total duration seconds: {summary['total_duration_seconds']}",
        f"- Next action: {_next_action(summary)}",
        "",
    ]
    return "\n".join(lines)


def _detect_warnings(steps: list[dict]) -> list[str]:
    warnings = []
    combined_output = "\n".join(
        f"{row.get('stdout_tail', '')}\n{row.get('stderr_tail', '')}" for row in steps
    ).lower()
    if any(row["status"] == "failed" and not row.get("critical") for row in steps):
        warnings.append("One or more non-critical steps failed.")
    if "validation set is empty" in combined_output or "test set is empty" in combined_output:
        warnings.append("A validation or test split may be empty.")
    if "too small" in combined_output or "below the preferred reliability threshold" in combined_output:
        warnings.append("Dataset size warnings were detected in stage output.")
    if not warnings:
        warnings.append("None.")
    return warnings


def _warnings_text(warnings: list[str], summary: dict) -> str:
    if summary["failed_steps"] > 0 and summary["critical_failures"] == 0:
        warnings = [*warnings, "Review failed non-critical stages and interpret model metrics carefully."]
    return "\n".join(f"- {warning}" for warning in warnings)


def _next_action(summary: dict) -> str:
    if summary["pipeline_status"] == "success":
        return "Review reports/model_comparison and reports/tracking."
    if summary["pipeline_status"] == "failed":
        return "Fix the first failed critical step and rerun the pipeline."
    if summary["pipeline_status"] == "dry_run":
        return "Review selected steps, then rerun without --dry-run when ready."
    return "Review failed non-critical stages and interpret model metrics carefully."


if __name__ == "__main__":
    try:
        main()
    except PermissionError as exc:
        print(f"Could not write pipeline reports due to a permission error: {exc}")
    except Exception as exc:
        print(f"Pipeline runner failed: {exc}")
