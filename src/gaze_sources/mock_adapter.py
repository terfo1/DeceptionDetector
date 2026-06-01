"""Synthetic gaze source adapter for pipeline testing."""

from __future__ import annotations

import random

from .base import GazeSourceAdapter


class MockGazeAdapter(GazeSourceAdapter):
    """Generate synthetic normalized gaze samples."""

    def __init__(self, sampling_rate: int = 60):
        self.sampling_rate = int(sampling_rate)
        self.timestamp = 0.0
        self.running = False

    def start(self) -> None:
        self.timestamp = 0.0
        self.running = True

    def stop(self) -> None:
        self.running = False

    def read_sample(self) -> dict | None:
        if not self.running:
            return None
        blink = 1 if random.random() < 0.03 else 0
        saccade = 1 if random.random() < 0.10 else 0
        sample = {
            "timestamp": round(self.timestamp, 6),
            "gaze_x": _clamp(0.5 + random.uniform(-0.04, 0.04), 0.0, 1.0),
            "gaze_y": _clamp(0.5 + random.uniform(-0.04, 0.04), 0.0, 1.0),
            "pupil_left": 3.2 + random.uniform(-0.08, 0.08),
            "pupil_right": 3.2 + random.uniform(-0.08, 0.08),
            "blink": blink,
            "fixation": 0 if saccade else 1,
            "saccade": saccade,
            "validity": 1 if random.random() < 0.98 else 0,
        }
        self.timestamp += 1.0 / self.sampling_rate
        return sample

    def is_running(self) -> bool:
        return self.running

    def get_source_name(self) -> str:
        return "mock"

    def get_metadata(self) -> dict:
        return {
            "source": "mock",
            "sampling_rate": self.sampling_rate,
            "description": "Synthetic gaze source for pipeline testing",
        }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
