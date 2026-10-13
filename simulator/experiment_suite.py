#!/usr/bin/env python3
"""Reproducible synthetic experiments for robot safety verification.

This is a research simulator, not a validated robot controller.
Labels use a separately implemented constraint predicate. Because both
predicates use sampled physical variables, this is not a fully independent
real-world oracle or proof of physical safety.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CONFIG = ROOT / "configs" / "experiment_protocol.json"

WEIGHTS = {
    "A": {"reach": .20, "collision": .30, "force": .15, "torque": .15, "stability": .20},
    "B": {"reach": .30, "collision": .25, "force": .15, "torque": .10, "stability": .20},
    "C": {"reach": .15, "collision": .35, "force": .20, "torque": .10, "stability": .20},
    "D": {"reach": .20, "collision": .20, "force": .20, "torque": .20, "stability": .20},
}
COMPONENTS = ("reach", "collision", "force", "torque", "stability")


def load_config():
    return json.loads(CONFIG.read_text())


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def sample_episode(rng, horizon=1, split="seen"):
    """Generate an episode from explicit synthetic physical variables."""
    shift = 0.0 if split == "seen" else 0.06
    distance = rng.uniform(0.05, 1.05) + shift
    clearance = rng.uniform(-0.04, 0.10) - shift / 3
    mass = rng.uniform(0.15, 2.20)
    friction = rng.uniform(0.42, 0.62)
    grasp_eff = rng.uniform(0.55, 0.95)
    torque_util = rng.uniform(0.20, 1.35) + shift
    stability_margin = rng.gauss(0.045, 0.022) - shift / 2
    slip_probability = rng.uniform(0.0, 0.25) + shift / 2

    # Multi-step accumulation raises the chance of encountering constraints.
    risk_scale = 1.0 + 0.34 * max(0, horizon - 1)

    force_required = mass * 9.81 / max(0.01, friction * grasp_eff)
    reach = clamp((0.72 - distance) / 0.72)
    collision = clamp(clearance / 0.06)
    force = clamp((40.0 - force_required) / 40.0)
    torque = clamp(1.0 - torque_util)
    stability = clamp((stability_margin - 0.020) / 0.060)
    features = {
        "reach": reach, "collision": collision, "force": force,
        "torque": torque, "stability": stability
    }

    # Separate explicit constraint predicate; not derived from weighted score.
    safe = (
        distance <= 0.72
        and clearance >= 0.020
        and force_required <= 40.0
        and torque_util <= 1.0
        and stability_margin >= 0.020
        and slip_probability <= 0.10
    )
    # Horizon affects accumulated episode-level success, not individual geometry.
    if horizon > 1 and rng.random() < 1 - math.exp(-0.13 * (horizon - 1)):
        safe = False

    # Confidence is intentionally not a safety certificate.
    confidence = clamp(
        0.90 - 0.40 * abs(0.5 - sum(features.values()) / 5)
        - rng.random() * 0.35
    )
    return {
        "safe": bool(safe), "features": features,
        "confidence": confidence, "novelty": split == "unseen",
        "distance_m": distance, "clearance_m": clearance,
        "mass_kg": mass, "force_required_N": force_required,
        "torque_utilization": torque_util,
        "stability_margin_m": stability_margin,
        "slip_probability": slip_probability
    }


def score_episode(ep, weights=None, omit=None, noise=0.0, rng=None):
    weights = weights or WEIGHTS["A"]
    feats = dict(ep["features"])
    if omit:
        feats[omit] = 1.0  # Ablation: component contributes no rejection.
    denom = sum(weights[k] for k in COMPONENTS if k != omit)
    value = sum(weights[k] * feats[k] for k in COMPONENTS if k != omit)
    value = value / denom if denom else 0.0
    if noise and rng:
        value = clamp(value + rng.gauss(0, noise))
    return value


def decide(ep, threshold=.80, weights=None, omit=None, noise=0.0,
           rng=None, mode="full", novelty_deferral=False):
    if novelty_deferral and ep["novelty"]:
        return False, "deferred"
    if mode == "ai_only":
        return ep["confidence"] >= .25, "accepted"
    if mode == "geometry_only":
        f = ep["features"]
        return all(f[k] >= .0 for k in ("reach", "collision")), "accepted"
    if mode == "hard_gate":
        # Constraint-consistency reference; uses the label predicate itself.
        # This is not a fair independent baseline.
        return ep["safe"], "accepted"
    s = score_episode(ep, weights, omit, noise, rng)
    return s >= threshold, "accepted"


def evaluate(episodes, **decision_args):
    tp = fp = tn = fn = accepted_safe = deferred = 0
    for ep in episodes:
        accepted, status = decide(ep, **decision_args)
        if status == "deferred":
            deferred += 1
            accepted = False
        if ep["safe"] and accepted:
            tp += 1
            accepted_safe += 1
        elif ep["safe"] and not accepted:
            fn += 1
        elif not ep["safe"] and accepted:
            fp += 1
        else:
            tn += 1
    total = len(episodes)
    safe_n = tp + fn
    unsafe_n = fp + tn
    return {
        "episodes": total,
        "safe": safe_n,
        "unsafe": unsafe_n,
        "unsafe_rejected": tn,
        "unsafe_accepted": fp,
        "safe_accepted": tp,
        "safe_rejected": fn,
        "detection_pct": 100 * tn / unsafe_n if unsafe_n else 0,
        "false_safe_pct": 100 * fp / unsafe_n if unsafe_n else 0,
        "false_rejection_pct": 100 * fn / safe_n if safe_n else 0,
        "safe_acceptance_pct": 100 * tp / safe_n if safe_n else 0,
        "accepted_actions": tp + fp,
        "accepted_action_unsafe_pct": 100 * fp / (tp + fp) if tp + fp else 0,
        "task_success_proxy_pct": 100 * tp / total if total else 0,
        "deferred": deferred
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def seeded_episodes(n, seed, horizon=1, split="seen"):
    rng = random.Random(seed)
    return [sample_episode(rng, horizon, split) for _ in range(n)]


def run_all(n=None):
    cfg = load_config()
    n = n or cfg["episodes"]
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []

    # Primary: deterministic seed and fixed threshold.
    primary = seeded_episodes(n, cfg["seed"], 1)
    row = evaluate(primary, threshold=.80)
    row.update(experiment="primary", threshold=.80, seed=cfg["seed"])
    summaries.append(row)

    # Horizon sweep.
    for h in cfg["horizons"]:
        eps = seeded_episodes(n, cfg["seed"] + 100 + h, h)
        r = evaluate(eps, threshold=.80)
        r.update(experiment="horizon", horizon=h, seed=cfg["seed"] + 100 + h)
        summaries.append(r)

    # Threshold sweep on the same fixed dataset.
    for t in cfg["thresholds"]:
        r = evaluate(primary, threshold=t)
        r.update(experiment="threshold", threshold=t, seed=cfg["seed"])
        summaries.append(r)

    # Score-estimation noise.
    noise_eps = seeded_episodes(n, cfg["seed"] + 200, 5)
    for sigma in cfg["score_noise"]:
        rr = random.Random(cfg["seed"] + int(sigma * 10000))
        r = evaluate(noise_eps, threshold=.80, noise=sigma, rng=rr)
        r.update(experiment="score_noise", sigma=sigma, seed=cfg["seed"] + 200)
        summaries.append(r)

    # Seen/unseen split.
    for split in ("seen", "unseen"):
        eps = seeded_episodes(n // 2, cfg["seed"] + (300 if split == "seen" else 301),
                              3, split)
        r = evaluate(eps, threshold=.80)
        r.update(experiment="generalization", split=split)
        summaries.append(r)

    # Component ablation.
    abl_eps = seeded_episodes(n, cfg["seed"] + 400, 3)
    for comp in (*COMPONENTS, "none"):
        r = evaluate(abl_eps, threshold=.80, omit=None if comp == "none" else comp)
        r.update(experiment="ablation", omitted_component=comp)
        summaries.append(r)

    # Weight sensitivity.
    for label, weights in WEIGHTS.items():
        r = evaluate(abl_eps, threshold=.80, weights=weights)
        r.update(experiment="weight_sensitivity", weight_set=label,
                 weights_json=json.dumps(weights, sort_keys=True))
        summaries.append(r)

    # Baselines. Hard-gate baseline mirrors labels and is tautological.
    for mode in ("ai_only", "geometry_only", "hard_gate", "full"):
        r = evaluate(abl_eps, threshold=.80, mode=mode)
        r.update(experiment="baseline", baseline=mode)
        summaries.append(r)

    # Confidence vs safety: correlation and contingency counts.
    conf_eps = seeded_episodes(n, cfg["seed"] + 500, 1)
    conf_rows = []
    for ep in conf_eps:
        conf_rows.append({"confidence": ep["confidence"], "safe": int(ep["safe"])})
    safe_conf = [x["confidence"] for x in conf_rows if x["safe"]]
    unsafe_conf = [x["confidence"] for x in conf_rows if not x["safe"]]
    mean_conf = statistics.mean(x["confidence"] for x in conf_rows)
    mean_safe = statistics.mean(safe_conf) if safe_conf else 0
    mean_unsafe = statistics.mean(unsafe_conf) if unsafe_conf else 0
    # Pearson correlation between confidence and binary safety label.
    xs = [x["confidence"] for x in conf_rows]
    ys = [x["safe"] for x in conf_rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / len(xs)
    sx = statistics.pstdev(xs)
    sy = statistics.pstdev(ys)
    corr = cov / (sx * sy) if sx and sy else 0
    summaries.append({
        "experiment": "confidence_vs_safety", "episodes": n,
        "mean_confidence": mean_conf, "mean_confidence_safe": mean_safe,
        "mean_confidence_unsafe": mean_unsafe, "pearson_r": corr,
        "low_confidence_safe": sum(x["safe"] and x["confidence"] < .5 for x in conf_rows),
        "low_confidence_unsafe": sum((not x["safe"]) and x["confidence"] < .5 for x in conf_rows)
    })

    # Conservative uncertainty-aware screen: 5th percentile of score samples.
    uncertainty_eps = seeded_episodes(n, cfg["seed"] + 600, 1)
    for sigma in cfg["uncertainty_noise"]:
        rr = random.Random(cfg["seed"] + int(sigma * 10000) + 600)
        decisions = []
        for ep in uncertainty_eps:
            draws = sorted(score_episode(ep, noise=sigma, rng=rr)
                           for _ in range(cfg["uncertainty_samples"]))
            conservative = draws[max(0, math.ceil(.05 * len(draws)) - 1)]
            decisions.append((ep, conservative >= .80))
        # Convert decisions into metrics without changing the underlying labels.
        fake = []
        tp = fp = tn = fn = 0
        for ep, accepted in decisions:
            if ep["safe"] and accepted: tp += 1
            elif ep["safe"]: fn += 1
            elif accepted: fp += 1
            else: tn += 1
        safe_n, unsafe_n = tp + fn, fp + tn
        r = {
            "experiment": "uncertainty_screen", "sigma": sigma,
            "episodes": n, "safe": safe_n, "unsafe": unsafe_n,
            "unsafe_accepted": fp, "unsafe_rejected": tn,
            "safe_accepted": tp, "safe_rejected": fn,
            "detection_pct": 100*tn/unsafe_n if unsafe_n else 0,
            "false_safe_pct": 100*fp/unsafe_n if unsafe_n else 0,
            "false_rejection_pct": 100*fn/safe_n if safe_n else 0,
            "safe_acceptance_pct": 100*tp/safe_n if safe_n else 0,
            "task_success_proxy_pct": 100*tp/n if n else 0
        }
        summaries.append(r)

    # Independent-validation extension: distinct predicate interface and
    # accepted-action denominator. Simplified oracle; not physical certification.
    ext_n = max(n, 50000)
    ext_eps = seeded_episodes(ext_n, cfg["seed"] + 700, 3)
    for variant, noise, defer in (
        ("shared_rule", 0.0, False),
        ("uncertainty_aware", .08, False),
        ("uncertainty_plus_novelty_deferral", .08, True),
    ):
        rr = random.Random(cfg["seed"] + 701)
        r = evaluate(ext_eps, threshold=.80, noise=noise, rng=rr,
                     novelty_deferral=defer)
        r.update(experiment="independent_validation_extension",
                 variant=variant, denominator="accepted_actions",
                 accepted_action_unsafe_pct=r["accepted_action_unsafe_pct"])
        summaries.append(r)

    # Runtime benchmark (Python simulation only; not controller latency).
    rt_eps = seeded_episodes(n, cfg["seed"] + 800, 3)
    durations = []
    for _ in range(10):
        start = time.perf_counter()
        evaluate(rt_eps, threshold=.80)
        durations.append(time.perf_counter() - start)
    durations_sorted = sorted(durations)
    runtime = {
        "experiment": "runtime_python_simulation",
        "episodes_per_run": n,
        "runs": len(durations),
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "p95_seconds": durations_sorted[min(len(durations)-1, math.ceil(.95*len(durations))-1)],
        "mean_ms_per_episode": 1000*statistics.mean(durations)/n
    }
    summaries.append(runtime)

    write_csv(OUT / "experiment_summary.csv", summaries)
    write_csv(OUT / "primary_episode_sample.csv", [
        {"safe": int(e["safe"]), **e["features"], "confidence": e["confidence"],
         "distance_m": e["distance_m"], "clearance_m": e["clearance_m"],
         "mass_kg": e["mass_kg"], "force_required_N": e["force_required_N"],
         "torque_utilization": e["torque_utilization"],
         "stability_margin_m": e["stability_margin_m"],
         "slip_probability": e["slip_probability"]}
        for e in primary[:min(1000, len(primary))]
    ])

    manifest = {
        "seed": cfg["seed"], "episodes_per_main_experiment": n,
        "python": __import__("sys").version,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((OUT / "experiment_summary.csv").read_bytes()).hexdigest(),
        "note": "Synthetic model outputs; not proof of physical robot safety."
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Generated {len(summaries)} experiment summary rows.")
    print(f"Results: {OUT / 'experiment_summary.csv'}")
    print(f"Manifest: {OUT / 'run_manifest.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 100:
        raise SystemExit("--episodes must be at least 100")
    run_all(args.episodes)


if __name__ == "__main__":
    main()
