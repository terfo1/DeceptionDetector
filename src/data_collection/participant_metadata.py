"""Anonymous participant metadata helpers for data collection."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


PARTICIPANT_METADATA_HEADER = [
    "participant_id",
    "age_group",
    "gender_optional",
    "vision_status",
    "glasses_or_lenses",
    "dominant_eye_optional",
    "notes",
    "consent_confirmed",
    "created_at",
]

AGE_GROUPS = {
    "under_18",
    "18_24",
    "25_34",
    "35_44",
    "45_plus",
    "prefer_not_to_say",
}
VISION_STATUS = {"normal", "corrected_to_normal", "impaired", "prefer_not_to_say"}
GLASSES_OR_LENSES = {"none", "glasses", "contact_lenses", "both", "prefer_not_to_say"}


def create_participant_metadata_row(
    participant_id: str,
    age_group: str = "prefer_not_to_say",
    gender_optional: str = "prefer_not_to_say",
    vision_status: str = "prefer_not_to_say",
    glasses_or_lenses: str = "prefer_not_to_say",
    dominant_eye_optional: str = "prefer_not_to_say",
    notes: str = "",
    consent_confirmed: bool = False,
) -> dict:
    """Create an anonymous participant metadata row."""
    return {
        "participant_id": str(participant_id).strip(),
        "age_group": age_group,
        "gender_optional": gender_optional,
        "vision_status": vision_status,
        "glasses_or_lenses": glasses_or_lenses,
        "dominant_eye_optional": dominant_eye_optional,
        "notes": notes,
        "consent_confirmed": bool(consent_confirmed),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def validate_participant_metadata(row: dict) -> tuple[bool, list[str]]:
    """Validate anonymous participant metadata."""
    messages: list[str] = []
    if not str(row.get("participant_id", "")).strip():
        messages.append("participant_id is required.")
    if row.get("age_group") not in AGE_GROUPS:
        messages.append("age_group is invalid.")
    if row.get("vision_status") not in VISION_STATUS:
        messages.append("vision_status is invalid.")
    if row.get("glasses_or_lenses") not in GLASSES_OR_LENSES:
        messages.append("glasses_or_lenses is invalid.")
    if not bool(row.get("consent_confirmed")):
        messages.append("consent_confirmed must be True.")
    return len(messages) == 0, messages


def append_participant_metadata(
    row: dict,
    path: str = "data/raw/participant_metadata.csv",
) -> list[str]:
    """Append participant metadata and return non-fatal warnings."""
    is_valid, messages = validate_participant_metadata(row)
    if not is_valid:
        raise ValueError("Invalid participant metadata: " + "; ".join(messages))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_file(output_path, PARTICIPANT_METADATA_HEADER)

    warnings: list[str] = []
    try:
        existing = pd.read_csv(output_path)
        if (
            not existing.empty
            and "participant_id" in existing.columns
            and str(row["participant_id"]) in set(existing["participant_id"].astype(str))
        ):
            warnings.append(f"participant_id already exists in metadata: {row['participant_id']}")
    except pd.errors.EmptyDataError:
        pass

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PARTICIPANT_METADATA_HEADER)
        writer.writerow({column: row.get(column, "") for column in PARTICIPANT_METADATA_HEADER})
    return warnings


def ensure_participant_metadata_file(path: str = "data/raw/participant_metadata.csv") -> None:
    """Create participant metadata CSV with headers if it is missing."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_file(output_path, PARTICIPANT_METADATA_HEADER)


def _ensure_file(path: Path, header: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(header)
