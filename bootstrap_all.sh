#!/usr/bin/env bash
set -euo pipefail

# Run from the robot-safety-verification project root.
# Existing target files are backed up before replacement.
for f in \
  src/main.py src/verifier/scorer.py src/verifier/hard_gate.py \
  src/models/seven_dof_arm.py src/safety_verification.py \
  tests/test_scoring.py tests/test_safety_rules.py \
  configs/experiment_config.json README.md
do
  if [[ -f "$f" ]]; then
    cp "$f" "$f.bak"
  fi
done

mkdir -p src/verifier src/models tests configs results data scripts ablation baselines
touch src/__init__.py src/verifier/__init__.py src/models/__init__.py

cat > src/verifier/scorer.py <<'PY'
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
PY

cat > src/verifier/hard_gate.py <<'PY'
"""Documented physical limits. Torque limits must be supplied explicitly."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SafetyLimits:
    max_reach_m: float = 0.72
    min_clearance_m: float = 0.020
    max_required_force_N: float = 40.0
    min_com_margin_m: float = 0.020
    max_slip_probability: float = 0.10


def check_hard_gates(
    *,
    reach_m: float,
    clearance_m: float,
    required_force_N: float,
    com_margin_m: float,
    slip_probability: float,
    joint_torques_Nm: list[float],
    joint_torque_limits_Nm: list[float],
    limits: SafetyLimits = SafetyLimits(),
) -> tuple[bool, list[str]]:
    physical_values = {
        "reach_m": reach_m,
        "clearance_m": clearance_m,
        "required_force_N": required_force_N,
        "com_margin_m": com_margin_m,
        "slip_probability": slip_probability,
    }
    for name, value in physical_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if len(joint_torques_Nm) != len(joint_torque_limits_Nm):
        raise ValueError("Torque values and limits must have equal lengths")
    if not joint_torque_limits_Nm:
        raise ValueError("Verified joint torque limits are required")
    if any(not math.isfinite(x) for x in joint_torques_Nm + joint_torque_limits_Nm):
        raise ValueError("Torque values and limits must be finite")
    if any(x <= 0 for x in joint_torque_limits_Nm):
        raise ValueError("Torque limits must be positive")

    reasons = []
    if reach_m > limits.max_reach_m:
        reasons.append("reach_limit_exceeded")
    if clearance_m < limits.min_clearance_m:
        reasons.append("collision_clearance_too_small")
    if required_force_N > limits.max_required_force_N:
        reasons.append("required_force_exceeded")
    if com_margin_m < limits.min_com_margin_m:
        reasons.append("com_margin_too_small")
    if slip_probability > limits.max_slip_probability:
        reasons.append("slip_probability_too_high")

    for i, (torque, limit) in enumerate(
        zip(joint_torques_Nm, joint_torque_limits_Nm), start=1
    ):
        if abs(torque) > limit:
            reasons.append(f"joint_{i}_torque_exceeded")

    return not reasons, reasons
PY

cat > src/models/seven_dof_arm.py <<'PY'
"""Seven-joint input validation, not a full robot kinematics simulator."""


def validate_seven(values: list[float], label: str = "values") -> None:
    if len(values) != 7:
        raise ValueError(f"{label} must contain exactly 7 values")
    if any(not isinstance(v, (int, float)) for v in values):
        raise ValueError(f"{label} must be numeric")


def within_joint_limits(
    joint_values: list[float],
    lower_limits: list[float],
    upper_limits: list[float],
) -> bool:
    validate_seven(joint_values, "joint_values")
    validate_seven(lower_limits, "lower_limits")
    validate_seven(upper_limits, "upper_limits")
    if any(lo > hi for lo, hi in zip(lower_limits, upper_limits)):
        raise ValueError("Lower joint limit exceeds upper limit")
    return all(
        lo <= value <= hi
        for value, lo, hi in zip(joint_values, lower_limits, upper_limits)
    )
PY

cat > src/safety_verification.py <<'PY'
from src.verifier.scorer import (
    DEFAULT_THRESHOLD, WEIGHTS, calculate_score, meets_threshold
)
from src.verifier.hard_gate import SafetyLimits, check_hard_gates

__all__ = [
    "DEFAULT_THRESHOLD", "WEIGHTS", "calculate_score", "meets_threshold",
    "SafetyLimits", "check_hard_gates",
]
PY

cat > src/main.py <<'PY'
"""Score a CSV with verified, already-normalized factor columns.

This evaluates score thresholding only. It does not generate the paper's
20,000 candidates or claim to reproduce its physical safety experiment.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.verifier.scorer import DEFAULT_THRESHOLD, calculate_score, meets_threshold

COLUMNS = {
    "reachability": "reachability_score",
    "collision": "collision_score",
    "force": "force_score",
    "torque": "torque_score",
    "stability": "stability_score",
}


def evaluate(dataset: Path, threshold: float):
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    results = []
    with dataset.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV is empty or has no header")
        missing = [c for c in COLUMNS.values() if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                "Missing required normalized columns: " + ", ".join(missing)
                + ". Do not guess the mapping; inspect the original dataset."
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                factors = {k: float(row[v]) for k, v in COLUMNS.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid factor on CSV line {line_number}") from exc
            score = calculate_score(**factors)
            results.append({
                "csv_line": line_number,
                "score": score,
                "score_accepted": meets_threshold(score, threshold),
            })

    accepted = sum(r["score_accepted"] for r in results)
    summary = {
        "dataset": str(dataset),
        "threshold": threshold,
        "rows_evaluated": len(results),
        "score_accepted": accepted,
        "score_rejected": len(results) - accepted,
        "score_acceptance_rate": accepted / len(results) if results else None,
        "scope": "Score threshold only; physical hard gates are not evaluated here.",
    }
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "scores.csv")
    args = parser.parse_args()

    results, summary = evaluate(args.dataset, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["csv_line", "score", "score_accepted"]
        )
        writer.writeheader()
        writer.writerows(results)

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved scores: {args.output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
PY

cat > tests/test_scoring.py <<'PY'
import unittest
from src.verifier.scorer import calculate_score, meets_threshold, WEIGHTS


class ScoringTests(unittest.TestCase):
    def test_all_ones(self):
        self.assertAlmostEqual(calculate_score(1, 1, 1, 1, 1), 1.0)

    def test_all_zeros(self):
        self.assertAlmostEqual(calculate_score(0, 0, 0, 0, 0), 0.0)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_threshold_boundary(self):
        self.assertTrue(meets_threshold(0.80, 0.80))
        self.assertFalse(meets_threshold(0.79, 0.80))

    def test_rejects_out_of_range_factor(self):
        with self.assertRaises(ValueError):
            calculate_score(1.1, 1, 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
PY

cat > tests/test_safety_rules.py <<'PY'
import unittest
from src.verifier.hard_gate import check_hard_gates
from src.models.seven_dof_arm import within_joint_limits


class SafetyTests(unittest.TestCase):
    def kwargs(self):
        return dict(
            reach_m=0.70, clearance_m=0.025, required_force_N=20,
            com_margin_m=0.025, slip_probability=0.05,
            joint_torques_Nm=[1.0] * 7,
            joint_torque_limits_Nm=[2.0] * 7,
        )

    def test_safe_example(self):
        safe, reasons = check_hard_gates(**self.kwargs())
        self.assertTrue(safe)
        self.assertEqual(reasons, [])

    def test_clearance_failure(self):
        args = self.kwargs()
        args["clearance_m"] = 0.010
        safe, reasons = check_hard_gates(**args)
        self.assertFalse(safe)
        self.assertIn("collision_clearance_too_small", reasons)

    def test_torque_limit_required(self):
        args = self.kwargs()
        args["joint_torque_limits_Nm"] = []
        with self.assertRaises(ValueError):
            check_hard_gates(**args)

    def test_seven_joint_limits(self):
        self.assertTrue(within_joint_limits([0] * 7, [-1] * 7, [1] * 7))


if __name__ == "__main__":
    unittest.main()
PY

cat > configs/experiment_config.json <<'JSON'
{
  "threshold": 0.80,
  "score_weights": {
    "reachability": 0.20,
    "collision": 0.30,
    "force": 0.15,
    "torque": 0.15,
    "stability": 0.20
  },
  "limits": {
    "max_reach_m": 0.72,
    "min_clearance_m": 0.020,
    "max_required_force_N": 40.0,
    "min_com_margin_m": 0.020,
    "max_slip_probability": 0.10
  },
  "random_seed_reported": 260929,
  "joint_torque_limits_Nm": null,
  "normalization": "Not fully specified in available paper text; must be verified",
  "reproduction_status": "Not yet reproduced"
}
JSON

cat > scripts/run_verification.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_verification.sh path/to/dataset.csv [threshold]"
  exit 2
fi
python -m unittest discover -s tests -v
python src/main.py --dataset "$1" --threshold "${2:-0.80}"
SH
chmod +x scripts/run_verification.sh

cat > ablation/README.md <<'MD'
# Ablation studies

Do not report ablation results until the original protocol, dataset schema,
normalization, and random-seed procedure have been confirmed.
MD

cat > baselines/README.md <<'MD'
# Baselines

Baseline algorithms and metrics must be taken from the source paper or original
repository. No baseline results are fabricated by this scaffold.
MD

cat > README.md <<'MD'
# Robot Safety Verification

## Tests
`python -m unittest discover -s tests -v`

## Score a dataset
`python src/main.py --dataset path/to/dataset.csv --threshold 0.80`

The CSV must contain normalized columns:
`reachability_score, collision_score, force_score, torque_score, stability_score`.

This is a transparent implementation scaffold, not yet an exact reproduction.
The original dataset, score-normalization rules, torque limits, and full
experiment-generation procedure must be verified before comparing paper metrics.
MD

cat > requirements.txt <<'REQ'
# Core implementation uses only Python standard-library modules.
REQ

echo
echo "Files generated. Running unit tests..."
python -m unittest discover -s tests -v

echo
echo "Checking main program..."
python src/main.py --help

echo
echo "Dataset search:"
find data -type f -iname '*.csv' -print

echo
echo "Done. No research experiment was run because the original dataset is not confirmed."
