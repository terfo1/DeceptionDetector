"""Placeholder real eye tracker SDK adapter."""

from __future__ import annotations

from .base import GazeSourceAdapter


class RealEyeTrackerAdapter(GazeSourceAdapter):
    """Placeholder for future eye tracker SDK integration.

    Future integration plan:
    - connect to Tobii SDK, Pupil Labs, or another device SDK;
    - run calibration;
    - stream gaze samples and pupil diameter;
    - preserve validity flags;
    - normalize timestamps;
    - map device output into the unified gaze sample schema.
    """

    def __init__(self, device_name: str | None = None):
        self.device_name = device_name
        self.running = False

    def start(self) -> None:
        raise NotImplementedError(
            "Real eye tracker SDK integration is not implemented yet. This adapter is a placeholder."
        )

    def stop(self) -> None:
        self.running = False

    def read_sample(self) -> dict | None:
        return None

    def is_running(self) -> bool:
        return self.running

    def get_source_name(self) -> str:
        return "real_eye_tracker"

    def get_metadata(self) -> dict:
        return {
            "source": "real_eye_tracker",
            "device_name": self.device_name,
            "description": "Placeholder for future real eye tracker SDK integration",
        }
