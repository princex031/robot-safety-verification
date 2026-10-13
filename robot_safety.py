#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

METRICS = ("reachability", "clearance", "force", "torque", "stability")

WEIGHTS = {
    "reachability": 0.20,
    "clearance": 0.30,
    "force": 0.15,
    "torque": 0.15,
    "stability": 0.20,
}

GATES = (
    "reachability_ok",
    "clearance_ok",
    "force_ok",
    "torque_ok",
    "stability_ok",
)


def number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def validate_metrics(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be a JSON object")

    missing = [
        key for key in METRICS
        if key not in metrics or metrics[key] in (None, "")
    ]

    if missing:
        raise ValueError("Missing metrics: " + ", ".join(missing))

    values = {}

    for key in METRICS:
        value = number(metrics[key], key)

        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")

        values[key] = value

    return values


def score_action(metrics):
    values = validate_metrics(metrics)
    return sum(values[k] * WEIGHTS[k] for k in METRICS)


def verify_action(
    metrics,
    gates=None,
    uncertainty=None,
    threshold=0.80,
):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    try:
        values = validate_metrics(metrics)
        score = score_action(values)
    except ValueError as error:
        return {
            "decision": "REVIEW",
            "score": None,
            "hard_gate_passed": False,
            "reasons": [str(error)],
        }

    gates = gates or {}
    reasons = []

    failed = [
        key for key in GATES
        if gates.get(key) is False
    ]

    hard_gate_passed = not failed

    if failed:
        reasons.extend(
            f"{key}: constraint failed" for key in failed
        )

    if uncertainty is not None:
        try:
            uncertainty = number(uncertainty, "uncertainty")
            if not 0 <= uncertainty <= 1:
                raise ValueError(
                    "Uncertainty must be between 0 and 1"
                )
        except ValueError as error:
            return {
                "decision": "REVIEW",
                "score": round(score, 6),
                "hard_gate_passed": hard_gate_passed,
                "reasons": reasons + [str(error)],
            }

    if not hard_gate_passed:
        decision = "REJECT"
    elif uncertainty is not None and uncertainty >= 0.5:
        decision = "REVIEW"
        reasons.append("Uncertainty requires review")
    elif score >= threshold:
        decision = "ACCEPT"
        reasons.append("Score meets threshold")
    else:
        decision = "REJECT"
        reasons.append("Score below threshold")

    return {
        "decision": decision,
        "score": round(score, 6),
        "hard_gate_passed": hard_gate_passed,
        "metric_scores": values,
        "uncertainty": uncertainty,
        "reasons": reasons,
    }


def validate_joints(values, lower, upper, expected_dof=7):
    try:
        values = [number(x, "joint") for x in values]
        lower = [number(x, "lower limit") for x in lower]
        upper = [number(x, "upper limit") for x in upper]
    except ValueError as error:
        return {"valid": False, "reasons": [str(error)]}

    if not (len(values) == len(lower) == len(upper)):
        return {
            "valid": False,
            "reasons": ["Joint arrays must have equal lengths"],
        }

    if expected_dof is not None and len(values) != expected_dof:
        return {
            "valid": False,
            "reasons": [f"Expected {expected_dof} joints"],
        }

    if any(lo > hi for lo, hi in zip(lower, upper)):
        return {
            "valid": False,
            "reasons": ["Invalid joint limits"],
        }

    violations = [
        i + 1
        for i, (q, lo, hi) in enumerate(
            zip(values, lower, upper)
        )
        if q < lo or q > hi
    ]

    return {
        "valid": not violations,
        "joint_count": len(values),
        "violating_joint_indices": violations,
        "reasons": (
            [f"Joint limit violations: {violations}"]
            if violations else []
        ),
    }


def evaluate_csv(input_file, output_file, threshold=0.80):
    results = []

    with open(
        input_file, newline="", encoding="utf-8-sig"
    ) as source:
        reader = csv.DictReader(source)

        if not reader.fieldnames:
            raise ValueError("CSV header is missing")

        for line, row in enumerate(reader, start=2):
            gates = {}

            for key in GATES:
                raw = (row.get(key) or "").strip().lower()

                if raw:
                    if raw not in {
                        "true", "false", "1", "0", "yes", "no"
                    }:
                        raise ValueError(
                            f"Line {line}: invalid boolean for {key}"
                        )

                    gates[key] = raw in {"true", "1", "yes"}

            raw_uncertainty = (
                row.get("uncertainty") or ""
            ).strip()

            try:
                uncertainty = (
                    float(raw_uncertainty)
                    if raw_uncertainty else None
                )
            except ValueError:
                uncertainty = float("nan")

            result = verify_action(
                {key: row.get(key) for key in METRICS},
                gates=gates,
                uncertainty=uncertainty,
                threshold=threshold,
            )

            output_row = dict(row)
            output_row["decision"] = result["decision"]
            output_row["score"] = (
                "" if result["score"] is None
                else result["score"]
            )
            output_row["hard_gate_passed"] = (
                result["hard_gate_passed"]
            )
            output_row["reasons"] = " | ".join(
                result["reasons"]
            )

            results.append(output_row)

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if results:
        with destination.open(
            "w", newline="", encoding="utf-8"
        ) as target:
            writer = csv.DictWriter(
                target, fieldnames=list(results[0].keys())
            )
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "rows_evaluated": len(results),
        "accepted": sum(
            r["decision"] == "ACCEPT" for r in results
        ),
        "rejected": sum(
            r["decision"] == "REJECT" for r in results
        ),
        "review": sum(
            r["decision"] == "REVIEW" for r in results
        ),
        "threshold": threshold,
    }

    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(
        dest="command", required=True
    )

    verify = commands.add_parser("verify")
    verify.add_argument("--metrics", required=True)
    verify.add_argument("--gates", default="{}")
    verify.add_argument("--uncertainty", type=float)
    verify.add_argument("--threshold", type=float, default=0.80)

    joints = commands.add_parser("joints")
    joints.add_argument("--values", required=True)
    joints.add_argument("--lower", required=True)
    joints.add_argument("--upper", required=True)
    joints.add_argument("--dof", type=int, default=7)

    batch = commands.add_parser("csv")
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--threshold", type=float, default=0.80)

    args = parser.parse_args()

    try:
        if args.command == "verify":
            result = verify_action(
                json.loads(args.metrics),
                json.loads(args.gates),
                args.uncertainty,
                args.threshold,
            )
            print(json.dumps(result, indent=2))
            return 0 if result["decision"] == "ACCEPT" else 2

        if args.command == "joints":
            result = validate_joints(
                json.loads(args.values),
                json.loads(args.lower),
                json.loads(args.upper),
                args.dof,
            )
            print(json.dumps(result, indent=2))
            return 0 if result["valid"] else 2

        if args.command == "csv":
            print(json.dumps(
                evaluate_csv(
                    args.input, args.output, args.threshold
                ),
                indent=2,
            ))
            return 0

    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())