
"""Uncertainty-aware action verification.

This module is a decision layer, not a physics simulator.
Inputs must come from the project's physical checks and scoring model.
"""

from statistics import mean, pstdev
from typing import Any


DEFAULT_THRESHOLD = 0.80
MIN_SAMPLES = 20
MAX_SCORE = 1.0
MIN_SCORE = 0.0


def _quantile(values: list[float], q: float) -> float:
    """Return a linearly interpolated quantile without external dependencies."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _safe_alternatives(
    violations: list[str],
    uncertain: bool,
    out_of_distribution: bool,
) -> list[str]:
    """Return suggestions, not commands to execute automatically."""
    suggestions = []

    rules = {
        "reach": "Choose a closer target or a reachable robot configuration.",
        "collision": "Re-plan the path with greater obstacle clearance.",
        "force": "Reduce applied force and reassess contact conditions.",
        "torque": "Choose a lower-load configuration and recheck joint limits.",
        "stability": "Move the centre of mass inward and improve support.",
        "slip": "Reassess the grasp, contact friction, and object load.",
    }

    for violation in violations:
        key = violation.lower()
        for term, suggestion in rules.items():
            if term in key and suggestion not in suggestions:
                suggestions.append(suggestion)

    if uncertain:
        suggestions.append(
            "Request more measurements or additional model evaluations "
            "before execution."
        )

    if out_of_distribution:
        suggestions.append(
            "Use a validated model or gather evidence for this unfamiliar "
            "scenario before retrying."
        )

    if not suggestions:
        suggestions.append(
            "Generate a lower-risk candidate, then verify it independently."
        )

    return suggestions


def verify_action(
    *,
    score_samples: list[float],
    hard_gate_violations: list[str] | None = None,
    input_complete: bool = True,
    model_disagreement: bool = False,
    out_of_distribution: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    min_samples: int = MIN_SAMPLES,
    max_sample_std: float = 0.10,
) -> dict[str, Any]:
    """Evaluate an action while accounting for uncertainty.

    score_samples:
        Repeated scores for the same candidate under uncertainty samples.
        Each score must be normalized to [0, 1].

    hard_gate_violations:
        Physical constraint violations reported by the physical gate.
        Do not infer that this list is complete unless all required checks ran.

    Returns:
        A structured decision, reasons, score statistics, and suggestions.

    Decision policy:
        - Known hard violation -> REJECT.
        - Missing/incomplete evidence -> REVIEW.
        - Out-of-distribution input or model disagreement -> REVIEW.
        - Insufficient samples or excessive uncertainty -> REQUEST_ALTERNATIVE.
        - Lower score quantile meets threshold -> ACCEPT.
        - Otherwise -> REQUEST_ALTERNATIVE.
    """
    violations = list(hard_gate_violations or [])

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if max_sample_std < 0:
        raise ValueError("max_sample_std cannot be negative")

    scores = list(score_samples)
    for score in scores:
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("Every score sample must be between 0 and 1")

    # A confirmed hard-constraint violation takes priority.
    if violations:
        return {
            "decision": "REJECT",
            "reason": "One or more reported hard physical constraints failed.",
            "violations": violations,
            "score": None,
            "uncertainty": None,
            "safer_alternatives": _safe_alternatives(
                violations, False, out_of_distribution
            ),
        }

    # Missing evidence must not be mistaken for a safe result.
    if not input_complete:
        return {
            "decision": "REVIEW",
            "reason": "Required input or physical-check evidence is missing.",
            "violations": [],
            "score": None,
            "uncertainty": None,
            "safer_alternatives": [
                "Complete the required physical checks and verify sensor inputs."
            ],
        }

    if out_of_distribution or model_disagreement:
        reasons = []
        if out_of_distribution:
            reasons.append("Input is outside the model's validated operating range.")
        if model_disagreement:
            reasons.append("Models or evaluators disagree about the prediction.")

        return {
            "decision": "REVIEW",
            "reason": " ".join(reasons),
            "violations": [],
            "score": mean(scores) if scores else None,
            "uncertainty": None,
            "safer_alternatives": _safe_alternatives(
                [], True, out_of_distribution
            ),
        }

    if len(scores) < min_samples:
        return {
            "decision": "REQUEST_ALTERNATIVE",
            "reason": (
                f"Only {len(scores)} uncertainty samples were supplied; "
                f"at least {min_samples} are required by this policy."
            ),
            "violations": [],
            "score": mean(scores) if scores else None,
            "uncertainty": None,
            "safer_alternatives": _safe_alternatives([], True, False),
        }

    average = mean(scores)
    std_dev = pstdev(scores)
    lower = _quantile(scores, 0.10)
    upper = _quantile(scores, 0.90)

    stats = {
        "mean": round(average, 4),
        "std_dev": round(std_dev, 4),
        "p10": round(lower, 4),
        "p90": round(upper, 4),
        "sample_count": len(scores),
    }

    if std_dev > max_sample_std:
        return {
            "decision": "REQUEST_ALTERNATIVE",
            "reason": "Predicted safety score varies too much across samples.",
            "violations": [],
            "score": round(average, 4),
            "uncertainty": stats,
            "safer_alternatives": _safe_alternatives([], True, False),
        }

    # Conservative acceptance: even the 10th-percentile score must pass.
    if lower >= threshold:
        return {
            "decision": "ACCEPT",
            "reason": (
                "The score passed the threshold under the configured "
                "uncertainty-sampling policy, and no supplied warning fired."
            ),
            "violations": [],
            "score": round(average, 4),
            "uncertainty": stats,
            "safer_alternatives": [],
        }

    return {
        "decision": "REQUEST_ALTERNATIVE",
        "reason": (
            "The lower score quantile did not meet the acceptance threshold; "
            "generate and verify a safer candidate."
        ),
        "violations": [],
        "score": round(average, 4),
        "uncertainty": stats,
        "safer_alternatives": _safe_alternatives([], True, False),
    }
