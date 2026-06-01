"""Factory for gaze source adapters."""

from __future__ import annotations

from .base import GazeSourceAdapter
from .mock_adapter import MockGazeAdapter
from .real_eye_tracker_adapter import RealEyeTrackerAdapter
from .recorded_csv_adapter import RecordedCsvGazeAdapter
from .webcam_adapter import WebcamGazeAdapter


def create_gaze_adapter(source_type: str, **kwargs) -> GazeSourceAdapter:
    """Create a gaze adapter for the requested source type."""
    if source_type == "mock":
        return MockGazeAdapter(sampling_rate=kwargs.get("sampling_rate", 60))
    if source_type == "recorded_csv":
        return RecordedCsvGazeAdapter(
            trial_id=kwargs.get("trial_id"),
            max_trials=kwargs.get("max_trials"),
            loop=kwargs.get("loop", False),
        )
    if source_type == "webcam":
        return WebcamGazeAdapter(camera_index=kwargs.get("camera_index", 0))
    if source_type == "real_eye_tracker":
        return RealEyeTrackerAdapter(device_name=kwargs.get("device_name"))
    raise ValueError(f"Unsupported gaze source type: {source_type}")
