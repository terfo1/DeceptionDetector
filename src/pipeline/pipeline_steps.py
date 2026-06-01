"""Reusable pipeline step selection, execution, and summarization helpers."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime

from .pipeline_config import PIPELINE_STEPS, SUPPORTED_MODES


TAIL_LENGTH = 3000


def get_steps_for_mode(mode: str) -> list[dict]:
    """Return pipeline steps required for a supported mode."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported mode '{mode}'. Supported modes: {', '.join(SUPPORTED_MODES)}"
        )
    return [step for step in PIPELINE_STEPS if mode in step["required_for_modes"]]


def run_command(command: list[str], timeout_seconds: int | None = None) -> dict:
    """Run a subprocess command and capture timing, stdout, and stderr."""
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return_code = result.returncode
        stdout = result.stdout or ""
        stderr = result.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return_code = -1
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = f"Command timed out after {timeout_seconds} seconds."
    except FileNotFoundError as exc:
        return_code = -1
        stdout = ""
        stderr = f"Command not found: {exc}"
    except Exception as exc:
        return_code = -1
        stdout = ""
        stderr = f"Command failed before completion: {exc}"

    finished_at = datetime.now().isoformat(timespec="seconds")
    duration = time.perf_counter() - start
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_seconds": round(duration, 3),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def run_pipeline_steps(
    steps: list[dict],
    stop_on_critical_failure: bool = True,
    timeout_seconds: int | None = None,
) -> list[dict]:
    """Run selected pipeline steps and return normalized result rows."""
    results = []
    critical_failure_seen = False

    for step in steps:
        if critical_failure_seen and stop_on_critical_failure:
            now = datetime.now().isoformat(timespec="seconds")
            results.append(_skipped_result(step, now, "Skipped after critical failure."))
            print(f"Skipping {step['name']} after critical failure.")
            continue

        print(f"Running {step['name']}...")
        command_result = run_command(step["command"], timeout_seconds=timeout_seconds)
        status = "passed" if command_result["return_code"] == 0 else "failed"
        row = _result_row(step, command_result, status)
        results.append(row)

        if status == "failed":
            print(f"Step failed: {step['name']} (return code {row['return_code']})")
            if step.get("critical", False):
                critical_failure_seen = True
        else:
            print(f"Step passed: {step['name']}")

    return results


def summarize_pipeline_results(results: list[dict]) -> dict:
    """Summarize pipeline result rows."""
    passed = sum(1 for row in results if row["status"] == "passed")
    failed = sum(1 for row in results if row["status"] == "failed")
    skipped = sum(1 for row in results if row["status"] == "skipped")
    critical_failures = sum(
        1 for row in results if row["status"] == "failed" and bool(row.get("critical"))
    )
    total_duration = sum(float(row.get("duration_seconds") or 0.0) for row in results)

    if critical_failures > 0:
        pipeline_status = "failed"
    elif failed > 0:
        pipeline_status = "warning"
    else:
        pipeline_status = "success"

    return {
        "total_steps": len(results),
        "passed_steps": passed,
        "failed_steps": failed,
        "skipped_steps": skipped,
        "critical_failures": critical_failures,
        "total_duration_seconds": round(total_duration, 3),
        "pipeline_status": pipeline_status,
    }


def dry_run_results(steps: list[dict]) -> list[dict]:
    """Create skipped rows for a dry run without executing commands."""
    now = datetime.now().isoformat(timespec="seconds")
    return [_skipped_result(step, now, "Dry run only; command was not executed.") for step in steps]


def _result_row(step: dict, command_result: dict, status: str) -> dict:
    return {
        "step_name": step["name"],
        "description": step.get("description", ""),
        "command": " ".join(step["command"]),
        "critical": bool(step.get("critical", False)),
        "status": status,
        "return_code": command_result.get("return_code"),
        "started_at": command_result.get("started_at", ""),
        "finished_at": command_result.get("finished_at", ""),
        "duration_seconds": command_result.get("duration_seconds", 0.0),
        "stdout_tail": _tail(command_result.get("stdout", "")),
        "stderr_tail": _tail(command_result.get("stderr", "")),
    }


def _skipped_result(step: dict, timestamp: str, reason: str) -> dict:
    return {
        "step_name": step["name"],
        "description": step.get("description", ""),
        "command": " ".join(step["command"]),
        "critical": bool(step.get("critical", False)),
        "status": "skipped",
        "return_code": "",
        "started_at": timestamp,
        "finished_at": timestamp,
        "duration_seconds": 0.0,
        "stdout_tail": "",
        "stderr_tail": reason,
    }


def _tail(text: str) -> str:
    if not text:
        return ""
    return text[-TAIL_LENGTH:]
