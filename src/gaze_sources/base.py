"""Base interface and validation for gaze source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class GazeSourceAdapter(ABC):
    """Abstract interface for sources that emit normalized gaze samples."""

    @abstractmethod
    def start(self) -> None:
        """Start the gaze source."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the gaze source."""

    @abstractmethod
    def read_sample(self) -> dict | None:
        """Return one normalized gaze sample or None if unavailable."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return whether the source is running."""

    @abstractmethod
    def get_source_name(self) -> str:
        """Return a short source name."""

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return source metadata."""


def validate_gaze_sample(sample: dict) -> tuple[bool, list[str]]:
    """Validate the unified gaze sample schema."""
    messages: list[str] = []
    required_numeric = [
        "timestamp",
        "gaze_x",
        "gaze_y",
        "pupil_left",
        "pupil_right",
    ]
    for field in required_numeric:
        if field not in sample:
            messages.append(f"Missing field: {field}")
            continue
        try:
            float(sample[field])
        except (TypeError, ValueError):
            messages.append(f"Field must be numeric: {field}")

    for field in ["blink", "fixation", "saccade", "validity"]:
        if field not in sample:
            messages.append(f"Missing field: {field}")
            continue
        if sample[field] not in {0, 1}:
            messages.append(f"Field must be 0 or 1: {field}")

    for gaze_field in ["gaze_x", "gaze_y"]:
        if gaze_field in sample:
            try:
                value = float(sample[gaze_field])
                if value < 0 or value > 1:
                    messages.append(f"Warning: {gaze_field} outside [0, 1].")
            except (TypeError, ValueError):
                pass

    errors = [message for message in messages if not message.startswith("Warning:")]
    return len(errors) == 0, messages
