"""Risk categorization policy for replayed online predictions."""

from __future__ import annotations

import numpy as np

from .realtime_config import MIN_VALID_RATIO, RISK_HIGH_THRESHOLD, RISK_LOW_THRESHOLD


def risk_category(probability: float, valid_ratio: float, min_valid_ratio: float) -> str:
    """Map a probability and valid-data ratio to a risk category."""
    if np.isnan(probability) or valid_ratio < min_valid_ratio:
        return "insufficient_data"
    if probability < RISK_LOW_THRESHOLD:
        return "low"
    if probability < RISK_HIGH_THRESHOLD:
        return "medium"
    return "high"


def smooth_probabilities(probabilities: list[float], window_size: int = 3) -> float:
    """Return the moving average of the most recent probabilities."""
    if not probabilities:
        return float("nan")
    values = [value for value in probabilities[-window_size:] if not np.isnan(value)]
    return float(np.mean(values)) if values else float("nan")


def final_trial_decision(probabilities: list[float], valid_ratios: list[float]) -> dict:
    """Summarize trial-level probability behavior."""
    usable_probabilities = [
        probability
        for probability, valid_ratio in zip(probabilities, valid_ratios)
        if not np.isnan(probability) and valid_ratio >= MIN_VALID_RATIO
    ]
    insufficient_data_count = int(sum(1 for valid_ratio in valid_ratios if valid_ratio < MIN_VALID_RATIO))
    return {
        "mean_probability": float(np.mean(usable_probabilities)) if usable_probabilities else np.nan,
        "max_probability": float(np.max(usable_probabilities)) if usable_probabilities else np.nan,
        "final_smoothed_probability": smooth_probabilities(usable_probabilities),
        "usable_prediction_count": len(usable_probabilities),
        "insufficient_data_count": insufficient_data_count,
    }
