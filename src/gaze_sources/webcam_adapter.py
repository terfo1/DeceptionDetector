"""Placeholder webcam gaze adapter."""

from __future__ import annotations

from .base import GazeSourceAdapter


class WebcamGazeAdapter(GazeSourceAdapter):
    """Placeholder for future webcam gaze estimation.

    Future implementation plan:
    - capture webcam frames;
    - detect face and eyes;
    - estimate gaze point;
    - normalize gaze_x/gaze_y to screen coordinates;
    - estimate blink/fixation/saccade if possible;
    - produce the unified gaze sample format.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.running = False

    def start(self) -> None:
        raise NotImplementedError(
            "Webcam gaze estimation is not implemented yet. This adapter is a placeholder for future integration."
        )

    def stop(self) -> None:
        self.running = False

    def read_sample(self) -> dict | None:
        return None

    def is_running(self) -> bool:
        return self.running

    def get_source_name(self) -> str:
        return "webcam"

    def get_metadata(self) -> dict:
        return {
            "source": "webcam",
            "camera_index": self.camera_index,
            "description": "Placeholder for future webcam gaze estimation",
        }
