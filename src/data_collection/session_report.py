"""Generate human-readable reports for data collection sessions."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

from .session_quality import append_session_quality, calculate_session_quality


COLLECTION_STATUS_HEADER = [
    "session_id",
    "participant_id",
    "total_trials",
    "total_gaze_samples",
    "valid_ratio",
    "quality_flag",
    "recommended_action",
    "report_path",
    "created_at",
]


def generate_latest_session_report(
    session_id: str,
    output_dir: str = "reports/data_collection",
) -> dict:
    """Generate latest session report and summary files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sessions = pd.read_csv("data/raw/sessions.csv")
    trials = pd.read_csv("data/raw/trials.csv")
    gaze = pd.read_csv("data/raw/gaze_samples.csv")
    session_rows = sessions[sessions["session_id"].astype(str) == str(session_id)]
    if session_rows.empty:
        raise ValueError(f"session_id not found: {session_id}")
    session = session_rows.iloc[-1]
    session_trials = trials[trials["session_id"].astype(str) == str(session_id)].copy()
    trial_ids = set(session_trials["trial_id"].astype(str)) if not session_trials.empty else set()
    session_gaze = gaze[gaze["trial_id"].astype(str).isin(trial_ids)].copy() if trial_ids else pd.DataFrame()
    quality = calculate_session_quality(session_id)
    append_session_quality(quality)
    recommended_action = _recommended_action(quality["quality_flag"])

    truth_trials = int((session_trials.get("instruction", pd.Series(dtype=str)) == "truth").sum())
    lie_trials = int((session_trials.get("instruction", pd.Series(dtype=str)) == "lie").sum())
    timeout_answers = int((session_trials.get("answer", pd.Series(dtype=str)).astype(str) == "timeout").sum()) if not session_trials.empty else 0
    warnings = _quality_warnings(quality)

    report_path = output_path / "latest_session_report.txt"
    summary_path = output_path / "latest_session_summary.md"
    lines = [
        "Data Collection Session Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "1. Overview",
        f"- session_id: {session_id}",
        f"- participant_id: {session.get('participant_id', '')}",
        f"- date/time: {session.get('date', '')}",
        f"- device: {session.get('device', '')}",
        f"- sampling_rate: {session.get('sampling_rate', '')}",
        f"- calibration_quality: {session.get('calibration_quality', '')}",
        "",
        "2. Trial Summary",
        f"- total trials: {quality['total_trials']}",
        f"- truth trials: {truth_trials}",
        f"- lie trials: {lie_trials}",
        f"- completed trials: {quality['completed_trials']}",
        f"- timeout answers: {timeout_answers}",
        "",
        "3. Gaze Sample Summary",
        f"- total gaze samples: {quality['total_gaze_samples']}",
        f"- valid samples: {quality['valid_sample_count']}",
        f"- invalid samples: {quality['invalid_sample_count']}",
        f"- valid ratio: {quality['valid_ratio']}",
        f"- average samples per trial: {quality['average_samples_per_trial']}",
        "",
        "4. Quality Assessment",
        f"- quality_flag: {quality['quality_flag']}",
        "Warnings:",
        *([f"- {warning}" for warning in warnings] if warnings else ["- none"]),
        "",
        "5. Scientific/Ethical Notes",
        "- Participant data is anonymous.",
        "- The system estimates deception risk under controlled conditions.",
        "- It is not a universal lie detector.",
        "- More participants are required for reliable conclusions.",
        "",
        "6. Recommended Action",
        f"- {recommended_action}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Latest Data Collection Session Summary",
        "",
        f"- Session ID: `{session_id}`",
        f"- Participant ID: `{session.get('participant_id', '')}`",
        f"- Trials: {quality['completed_trials']} / {quality['total_trials']}",
        f"- Gaze samples: {quality['total_gaze_samples']}",
        f"- Valid ratio: {quality['valid_ratio']}",
        f"- Quality flag: `{quality['quality_flag']}`",
        f"- Recommended action: {recommended_action}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _append_collection_status(output_path / "collection_status.csv", quality, recommended_action, report_path)
    return {
        "session_id": session_id,
        "quality": quality,
        "recommended_action": recommended_action,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
    }


def _append_collection_status(path: Path, quality: dict, action: str, report_path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(COLLECTION_STATUS_HEADER)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLLECTION_STATUS_HEADER)
        writer.writerow(
            {
                "session_id": quality["session_id"],
                "participant_id": quality["participant_id"],
                "total_trials": quality["total_trials"],
                "total_gaze_samples": quality["total_gaze_samples"],
                "valid_ratio": quality["valid_ratio"],
                "quality_flag": quality["quality_flag"],
                "recommended_action": action,
                "report_path": report_path,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )


def _recommended_action(quality_flag: str) -> str:
    if quality_flag == "good":
        return "keep session"
    if quality_flag == "warning":
        return "review session"
    return "exclude session if poor quality"


def _quality_warnings(quality: dict) -> list[str]:
    warnings: list[str] = []
    if quality["valid_ratio"] < 0.85:
        warnings.append("valid_ratio is below the preferred 0.85 threshold.")
    if quality["completed_trials"] < quality["total_trials"]:
        warnings.append("not all trials were completed.")
    if quality["total_gaze_samples"] == 0:
        warnings.append("no gaze samples were recorded.")
    return warnings
