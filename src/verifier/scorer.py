"""Weighted score using the five factors documented in the paper."""

import math

DEFAULT_THRESHOLD = 0.80

WEIGHTS = {
    "reachability": 0.20,
    "collision": 0.30,
    "force": 0.15,
    "torque": 0.15,
    "stability": 0.20,
}


def calculate_score(
    reachability: float,
    collision: float,
    force: float,
    torque: float,
    stability: float,
) -> float:
    """Inputs must already be normalized to [0, 1]."""
    values = {
        "reachability": reachability,
        "collision": collision,
        "force": force,
        "torque": torque,
        "stability": stability,
    }
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    return sum(WEIGHTS[name] * value for name, value in values.items())


def meets_threshold(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    if not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError("score and threshold must be finite")
    if not 0 <= score <= 1 or not 0 <= threshold <= 1:
        raise ValueError("score and threshold must be between 0 and 1")
    return score >= threshold
