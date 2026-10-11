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
