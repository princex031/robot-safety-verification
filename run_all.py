import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reproducibility_results"

DEFAULT_CONFIG = {
    "seed": 20261013,
    "trials": 20000,
    "thresholds": [0.70, 0.75, 0.80, 0.85, 0.90],
    "score_weights": {
        "reachability": 0.20,
        "collision": 0.30,
        "force": 0.15,
        "torque": 0.15,
        "stability": 0.20
    },
    "force_limit_ratio": 1.0,
    "torque_limit_ratio": 1.0,
    "minimum_stability": 0.20,
    "collision_probability": 0.12
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )


def load_config(path):
    if not path.exists():
        write_json(path, DEFAULT_CONFIG)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = set(DEFAULT_CONFIG)
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    weights = config["score_weights"]
    expected = set(DEFAULT_CONFIG["score_weights"])
    if set(weights) != expected:
        raise ValueError("score_weights keys do not match the schema")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Score weights must sum to 1")
    if not config["thresholds"]:
        raise ValueError("At least one threshold is required")
    if any(not 0 <= x <= 1 for x in config["thresholds"]):
        raise ValueError("Thresholds must be in [0, 1]")
    return config


def generate_trial(rng, i, config):
    collision = rng.random() < config["collision_probability"]
    force_ratio = rng.random() * 1.4
    torque_ratio = rng.random() * 1.4
    stability = rng.random()
    reachability = rng.random()

    oracle_unsafe = (
        collision
        or force_ratio > config["force_limit_ratio"]
        or torque_ratio > config["torque_limit_ratio"]
        or stability < config["minimum_stability"]
    )

    w = config["score_weights"]
    score = (
        w["reachability"] * reachability
        + w["collision"] * (0.0 if collision else 1.0)
        + w["force"] * max(
            0.0, 1.0 - force_ratio / config["force_limit_ratio"]
        )
        + w["torque"] * max(
            0.0, 1.0 - torque_ratio / config["torque_limit_ratio"]
        )
        + w["stability"] * stability
    )

    return {
        "trial": i,
        "collision": int(collision),
        "force_ratio": force_ratio,
        "torque_ratio": torque_ratio,
        "stability": stability,
        "reachability": reachability,
        "score": score,
        "oracle_unsafe": int(oracle_unsafe)
    }


def run_experiment(config):
    rng = random.Random(config["seed"])
    rows = []

    for i in range(1, config["trials"] + 1):
        base = generate_trial(rng, i, config)
        for threshold in config["thresholds"]:
            accepted = base["score"] >= threshold
            rows.append({
                **base,
                "seed": config["seed"],
                "threshold": threshold,
                "accepted": int(accepted),
                "false_accept": int(
                    bool(base["oracle_unsafe"]) and accepted
                ),
                "false_reject": int(
                    not bool(base["oracle_unsafe"]) and not accepted
                )
            })
    return rows


def summarize(rows, config):
    reports = []
    for threshold in config["thresholds"]:
        group = [r for r in rows if r["threshold"] == threshold]
        unsafe = sum(r["oracle_unsafe"] for r in group)
        safe = len(group) - unsafe
        accepted = sum(r["accepted"] for r in group)
        false_accept = sum(r["false_accept"] for r in group)
        false_reject = sum(r["false_reject"] for r in group)

        reports.append({
            "threshold": threshold,
            "trials": len(group),
            "oracle_unsafe": unsafe,
            "oracle_safe": safe,
            "accepted": accepted,
            "rejected": len(group) - accepted,
            "false_accept": false_accept,
            "false_reject": false_reject,
            "false_accept_rate_among_unsafe": (
                false_accept / unsafe if unsafe else None
            ),
            "false_reject_rate_among_safe": (
                false_reject / safe if safe else None
            ),
            "acceptance_rate": accepted / len(group) if group else None
        })
    return reports


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="experiment_config.json")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_config(config_path)
    if args.trials is not None:
        if args.trials < 1:
            raise ValueError("--trials must be positive")
        config["trials"] = args.trials
    if args.seed is not None:
        config["seed"] = args.seed

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "experiment_config.json", config)

    start = time.time()
    rows = run_experiment(config)
    reports = summarize(rows, config)

    save_csv(OUT / "all_trials.csv", rows)
    save_csv(OUT / "threshold_summary.csv", reports)

    manifest = {
        "version": VERSION,
        "python": sys.version,
        "seed": config["seed"],
        "trials_per_threshold": config["trials"],
        "thresholds": config["thresholds"],
        "config_sha256": sha256_file(OUT / "experiment_config.json"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "all_trials_sha256": sha256_file(OUT / "all_trials.csv"),
        "summary_sha256": sha256_file(OUT / "threshold_summary.csv"),
        "elapsed_seconds": round(time.time() - start, 3),
        "experiment_type": "synthetic_threshold_sensitivity_test"
    }
    write_json(OUT / "manifest.json", manifest)

    print(json.dumps(reports, indent=2))
    print(f"\nResults saved in: {OUT}")
    print("Run completed.")


if __name__ == "__main__":
    main()