#!/usr/bin/env python3
"""
Robot Safety Verification Toolkit
---------------------------------
A standalone, dependency-free screening tool.

IMPORTANT:
- Metrics must be normalized safety scores in [0, 1].
- Higher scores mean safer for this screening model.
- This is NOT a physics simulator or certified robot safety controller.
- Real deployment requires robot-specific units, calibrated thresholds,
  validated dynamics, collision checking, and independent safety systems.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

VERSION = "2.0.0"

METRICS = {
    "reachability": 0.20,
    "clearance": 0.30,
    "force": 0.15,
    "torque": 0.15,
    "stability": 0.20,
}

DEFAULT_THRESHOLD = 0.70
MIN_METRIC = 0.0
MAX_METRIC = 1.0

# Generic example limits in radians. Replace with your actual robot's
# manufacturer-specified joint limits before using real hardware.
DEFAULT_JOINT_LIMITS = [(-math.pi, math.pi)] * 7


class SafetyInputError(ValueError):
    """Raised when input data is invalid."""


def parse_finite_number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise SafetyInputError(f"{name}: expected a number, got {value!r}")

    if not math.isfinite(result):
        raise SafetyInputError(f"{name}: NaN and infinity are not allowed")

    return result


def validate_metrics(row):
    """Validate and return the five normalized safety scores."""
    validated = {}

    for name in METRICS:
        if name not in row or row[name] is None or str(row[name]).strip() == "":
            raise SafetyInputError(f"Missing required metric: {name}")

        value = parse_finite_number(row[name], name)

        if not MIN_METRIC <= value <= MAX_METRIC:
            raise SafetyInputError(
                f"{name} must be between 0 and 1; received {value}"
            )

        validated[name] = value

    return validated


def score_action(metrics):
    """Calculate a weighted screening score from 0 to 1."""
    validated = validate_metrics(metrics)
    return sum(
        validated[name] * weight
        for name, weight in METRICS.items()
    )


def verify_action(metrics, threshold=DEFAULT_THRESHOLD):
    """Return a transparent screening decision and reasons."""
    threshold = parse_finite_number(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise SafetyInputError("threshold must be between 0 and 1")

    validated = validate_metrics(metrics)
    score = score_action(validated)

    # Conservative per-metric floor prevents a strong average from
    # hiding a very poor individual safety metric.
    weak_metrics = [
        name for name, value in validated.items()
        if value < threshold
    ]

    passed = score >= threshold and not weak_metrics

    return {
        "score": round(score, 6),
        "threshold": threshold,
        "decision": "PASS_SCREEN" if passed else "REJECT_SCREEN",
        "weak_metrics": ";".join(weak_metrics),
        "reason": (
            "All screening checks passed"
            if passed
            else (
                "Below threshold: " + ", ".join(weak_metrics)
                if weak_metrics
                else "Weighted score below threshold"
            )
        ),
    }


def validate_joints(values, limits=None):
    """Check exactly seven joint values against supplied limits."""
    if limits is None:
        limits = DEFAULT_JOINT_LIMITS

    if len(values) != 7:
        raise SafetyInputError(
            f"Expected exactly 7 joint values; received {len(values)}"
        )

    if len(limits) != 7:
        raise SafetyInputError("Exactly 7 joint-limit pairs are required")

    parsed = [
        parse_finite_number(value, f"joint_{index + 1}")
        for index, value in enumerate(values)
    ]

    violations = []

    for index, (value, pair) in enumerate(zip(parsed, limits), start=1):
        lower, upper = pair

        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and lower < upper
        ):
            raise SafetyInputError(
                f"Invalid limits for joint {index}: {pair}"
            )

        if value < lower or value > upper:
            violations.append(
                f"joint_{index}={value} outside [{lower}, {upper}]"
            )

    return {
        "decision": "PASS_LIMITS" if not violations else "REJECT_LIMITS",
        "joint_count": len(parsed),
        "violations": "; ".join(violations),
    }


def evaluate_csv(input_file, output_file, threshold=DEFAULT_THRESHOLD):
    """Evaluate every CSV row and save decisions to a new CSV file."""
    source = Path(input_file)
    destination = Path(output_file)

    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    if source.resolve() == destination.resolve():
        raise SafetyInputError(
            "Input and output files must differ; refusing to overwrite input"
        )

    required = set(METRICS)
    results = []
    errors = 0

    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise SafetyInputError("CSV is empty or has no header row")

        missing = required - set(reader.fieldnames)
        if missing:
            raise SafetyInputError(
                "CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            result = {
                "row_number": line_number,
                "candidate_id": (
                    row.get("candidate_id", "").strip()
                    or str(line_number - 1)
                ),
            }

            try:
                decision = verify_action(row, threshold)
                result.update(decision)
                result["error"] = ""
            except SafetyInputError as exc:
                errors += 1
                result.update({
                    "score": "",
                    "threshold": threshold,
                    "decision": "INVALID_INPUT",
                    "weak_metrics": "",
                    "reason": "Invalid input",
                    "error": str(exc),
                })

            results.append(result)

    destination.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "row_number", "candidate_id", *METRICS,
        "score", "threshold", "decision",
        "weak_metrics", "reason", "error",
    ]

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for result in results:
            # Preserve input metric values in the report.
            original_row = {}
            with source.open(
                "r", newline="", encoding="utf-8-sig"
            ) as source_handle:
                original_reader = csv.DictReader(source_handle)
                for _ in range(result["row_number"] - 2):
                    next(original_reader, None)
                original_row = next(original_reader, {}) or {}

            output_row = {
                key: original_row.get(key, "")
                for key in METRICS
            }
            output_row.update(result)
            writer.writerow(output_row)

    passed = sum(
        result.get("decision") == "PASS_SCREEN" for result in results
    )
    rejected = sum(
        result.get("decision") == "REJECT_SCREEN" for result in results
    )

    print(f"Version: {VERSION}")
    print(f"Rows processed: {len(results)}")
    print(f"Passed screening: {passed}")
    print(f"Rejected screening: {rejected}")
    print(f"Invalid rows: {errors}")
    print(f"Output saved: {destination}")

    if errors:
        print("Review the 'error' column before relying on these results.")


def run_self_tests():
    """Small deterministic tests for the core validation logic."""
    good = {
        "reachability": 0.95,
        "clearance": 0.95,
        "force": 0.95,
        "torque": 0.95,
        "stability": 0.95,
    }
    result = verify_action(good)
    assert result["decision"] == "PASS_SCREEN"

    bad = dict(good, clearance=0.10)
    assert verify_action(bad)["decision"] == "REJECT_SCREEN"

    for invalid in (
        dict(good, force=float("nan")),
        dict(good, torque=float("inf")),
        dict(good, stability=1.5),
    ):
        try:
            verify_action(invalid)
        except SafetyInputError:
            pass
        else:
            raise AssertionError("Invalid metric was not rejected")

    assert validate_joints([0] * 7)["decision"] == "PASS_LIMITS"

    try:
        validate_joints([0] * 6)
    except SafetyInputError:
        pass
    else:
        raise AssertionError("Wrong joint count was not rejected")

    print("All self-tests passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone robot safety screening toolkit"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    csv_parser = subparsers.add_parser(
        "csv", help="Evaluate normalized metric scores from CSV"
    )
    csv_parser.add_argument("--input", required=True)
    csv_parser.add_argument("--output", default="results.csv")
    csv_parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD
    )

    verify_parser = subparsers.add_parser(
        "verify", help="Verify one candidate's normalized scores"
    )
    for metric in METRICS:
        verify_parser.add_argument(
            f"--{metric}", type=float, required=True
        )
    verify_parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD
    )

    joints_parser = subparsers.add_parser(
        "joints", help="Check seven joint values against example limits"
    )
    joints_parser.add_argument(
        "values", nargs=7, type=float, metavar="JOINT_RAD"
    )

    subparsers.add_parser("self-test", help="Run built-in tests")

    args = parser.parse_args()

    try:
        if args.command == "csv":
            evaluate_csv(args.input, args.output, args.threshold)

        elif args.command == "verify":
            metrics = {
                name: getattr(args, name) for name in METRICS
            }
            result = verify_action(metrics, args.threshold)
            for key, value in result.items():
                print(f"{key}: {value}")
            if result["decision"] != "PASS_SCREEN":
                sys.exit(2)

        elif args.command == "joints":
            result = validate_joints(args.values)
            for key, value in result.items():
                print(f"{key}: {value}")
            if result["decision"] != "PASS_LIMITS":
                sys.exit(2)

        elif args.command == "self-test":
            run_self_tests()

    except (SafetyInputError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()