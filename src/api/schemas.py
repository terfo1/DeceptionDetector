"""Pydantic schemas for the live inference API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class GazeSample(BaseModel):
    """One gaze sample received by the live inference service."""

    trial_id: str | None = None
    timestamp: float = Field(..., ge=0)
    gaze_x: float
    gaze_y: float
    pupil_left: float
    pupil_right: float
    blink: int = 0
    fixation: int = 0
    saccade: int = 0
    validity: int = 1

    @field_validator("blink", "fixation", "saccade", "validity")
    @classmethod
    def binary_fields_must_be_binary(cls, value: int) -> int:
        if value not in {0, 1}:
            raise ValueError("Value must be 0 or 1.")
        return value


class PredictionRequest(BaseModel):
    """REST request for scoring one current gaze window."""

    session_id: str | None = None
    trial_id: str | None = None
    samples: list[GazeSample]


class PredictionResponse(BaseModel):
    """REST prediction response."""

    model_type: str
    probability: float
    smoothed_probability: float | None
    risk_category: str
    valid_ratio: float
    sample_count: int
    latency_ms: float
    status: str
    message: str


class StreamInputMessage(BaseModel):
    """WebSocket input message."""

    type: Literal[
        "start_session",
        "start_trial",
        "sample",
        "end_trial",
        "end_session",
        "reset",
    ]
    session_id: str | None = None
    trial_id: str | None = None
    data: dict[str, Any] | None = None
