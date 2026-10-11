import csv
import json
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated_reconstruction_20k.csv"
RESULT = ROOT / "results" / "final_summary.json"
CONFIG = ROOT / "results" / "reconstruction_assumptions.json"

N = 20_000
SEED = 260929
THRESHOLD = 0.80
rng = random.Random(SEED)


def clamp(x):
    return max(0.0, min(1.0, x))


def score_candidate(r):
    return (
        0.20 * r["reachability_score"]
        + 0.30 * r["collision_score"]
        + 0.15 * r["force_score"]
        + 0.15 * r["torque_score"]
        + 0.20 * r["stability_score"]
    )


def evaluate_candidate(r):
    safe = all([
        r["reach_m"] <= 0.72,
        r["clearance_m"] >= 0.020,
        r["required_force_n"] <= 40.0,
        r["max_torque_ratio"] <= 1.0,
        r["com_margin_m"] >= 0.020,
        r["slip_probability"] <= 0.10,
    ])

    score = score_candidate(r)
    score_accept = score >= THRESHOLD
    combined_accept = safe and score_accept

    return safe, score, score_accept, combined_accept


class VerifierTests(unittest.TestCase):
    def test_score_weights(self):
        r = {
            "reachability_score": 1.0,
            "collision_score": 1.0,
            "force_score": 1.0,
            "torque_score": 1.0,
            "stability_score": 1.0,
        }
        self.assertAlmostEqual(score_candidate(r), 1.0)

    def test_threshold(self):
        r = {
            "reachability_score": 0.8,
            "collision_score": 0.8,
            "force_score": 0.8,
            "torque_score": 0.8,
            "stability_score": 0.8,
        }
        self.assertGreaterEqual(score_candidate(r), THRESHOLD)

    def test_unsafe_candidate_rejected_by_combined_gate(self):
        r = {
            "reach_m": 0.80,
            "clearance_m": 0.04,
            "required_force_n": 20.0,
            "max_torque_ratio": 0.5,
            "com_margin_m": 0.04,
            "slip_probability": 0.05,
            "reachability_score": 1.0,
            "collision_score": 1.0,
            "force_score": 1.0,
            "torque_score": 1.0,
            "stability_score": 1.0,
        }
        safe, _, _, combined_accept = evaluate_candidate(r)
        self.assertFalse(safe)
        self.assertFalse(combined_accept)


def generate_rows():
    rows = []

    for i in range(1, N + 1):
        reach = rng.uniform(0.05, 0.90)
        clearance = rng.uniform(-0.01, 0.08)
        mass = rng.uniform(0.15, 2.20)
        friction = rng.uniform(0.42, 0.62)
        efficiency = rng.uniform(0.55, 0.95)
        force = mass * 9.81 / (friction * efficiency)

        # Assumed synthetic torque ratios, not robot dynamics.
        torque_ratio = max(rng.uniform(0.30, 1.30) for _ in range(7))
        com_margin = rng.uniform(-0.02, 0.08)
        slip = rng.uniform(0.0, 0.30)

        # Explicit illustrative normalization assumptions.
        r_score = clamp(1.0 - reach / 0.90)
        c_score = clamp((clearance + 0.01) / 0.09)
        f_score = clamp(1.0 - force / 80.0)
        t_score = clamp(1.0 - torque_ratio)
        com_score = clamp((com_margin + 0.02) / 0.10)
        slip_score = clamp(1.0 - slip / 0.30)
        stability_score = 0.5 * com_score + 0.5 * slip_score

        row = {
            "candidate_id": i,
            "reach_m": reach,
            "clearance_m": clearance,
            "mass_kg": mass,
            "friction": friction,
            "grasp_efficiency": efficiency,
            "required_force_n": force,
            "max_torque_ratio": torque_ratio,
            "com_margin_m": com_margin,
            "slip_probability": slip,
            "reachability_score": r_score,
            "collision_score": c_score,
            "force_score": f_score,
            "torque_score": t_score,
            "stability_score": stability_score,
        }

        safe, score, score_accept, combined_accept = evaluate_candidate(row)
        row.update({
            "weighted_score": score,
            "ground_truth_safe": safe,
            "score_only_accept": score_accept,
            "combined_verifier_accept": combined_accept,
        })
        rows.append(row)

    return rows


def metrics(rows, accept_field):
    safe_rows = [r for r in rows if r["ground_truth_safe"]]
    unsafe_rows = [r for r in rows if not r["ground_truth_safe"]]
    accepted = [r for r in rows if r[accept_field]]

    false_safe = sum(r[accept_field] for r in unsafe_rows)
    false_reject = sum(not r[accept_field] for r in safe_rows)

    return {
        "accepted_count": len(accepted),
        "false_safe_count": false_safe,
        "false_rejection_count": false_reject,
        "unsafe_detection_rate_percent": (
            100 * (len(unsafe_rows) - false_safe) / len(unsafe_rows)
            if unsafe_rows else None
        ),
        "false_safe_rate_percent": (
            100 * false_safe / len(unsafe_rows) if unsafe_rows else None
        ),
        "false_rejection_rate_percent": (
            100 * false_reject / len(safe_rows) if safe_rows else None
        ),
    }


def main():
    unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(VerifierTests)
    )
    suite = unittest.TestLoader().loadTestsFromTestCase(VerifierTests)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit("Tests failed; experiment stopped.")

    rows = generate_rows()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)

    with DATA.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in row.items()
            })

    safe_count = sum(r["ground_truth_safe"] for r in rows)
    summary = {
        "experiment_type": "independent_assumption_based_reconstruction",
        "exact_original_paper_reproduction": False,
        "candidate_count": N,
        "seed": SEED,
        "score_threshold": THRESHOLD,
        "safe_candidate_count": safe_count,
        "unsafe_candidate_count": N - safe_count,
        "score_only_evaluation": metrics(rows, "score_only_accept"),
        "combined_gate_and_score_evaluation": metrics(
            rows, "combined_verifier_accept"
        ),
        "dataset": str(DATA.relative_to(ROOT)),
        "notes": [
            "Ground-truth labels use the documented physical thresholds.",
            "Score-only evaluation can reveal false-safe score decisions.",
            "Combined verifier rejects candidates failing any hard safety gate.",
            "Combined false-safe count is zero by construction; it is not independent proof.",
            "Sampling distributions and score normalization are assumptions.",
            "Synthetic torque ratios do not simulate full seven-DOF robot dynamics.",
            "Do not present these metrics as the original paper's results.",
        ],
    }
    RESULT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    CONFIG.write_text(json.dumps({
        "seed": SEED,
        "candidate_count": N,
        "threshold": THRESHOLD,
        "score_weights": {
            "reachability": 0.20,
            "collision": 0.30,
            "force": 0.15,
            "torque": 0.15,
            "stability": 0.20,
        },
        "documented_limits": {
            "max_reach_m": 0.72,
            "min_clearance_m": 0.020,
            "max_force_n": 40.0,
            "min_com_margin_m": 0.020,
            "max_slip_probability": 0.10,
        },
        "status": "assumption-based; not exact reproduction",
    }, indent=2), encoding="utf-8")

    print("\nEXPERIMENT COMPLETE")
    print(f"Candidates: {N}")
    print(f"Ground-truth safe: {safe_count}")
    print(f"Ground-truth unsafe: {N - safe_count}")
    print(f"CSV: {DATA}")
    print(f"Summary: {RESULT}")
    print(f"Assumptions: {CONFIG}")


if __name__ == "__main__":
    main()
