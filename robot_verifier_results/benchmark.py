import json, platform, time
import numpy as np
import pandas as pd

SEED = 20261012
N = 20000
rng = np.random.default_rng(SEED)

def make_scenario(name, radius, domain, seed):
    r = np.random.default_rng(seed)
    if domain == "train":
        clearance = r.gamma(2.2, 0.24, N)
        error = r.normal(0, 0.12, N)
        speed = r.uniform(0.15, 0.8, N)
        noise = r.uniform(0.005, 0.045, N)
    elif domain == "transfer":
        clearance = r.gamma(1.7, 0.29, N)
        error = r.normal(0, 0.20, N)
        speed = r.uniform(0.2, 1.0, N)
        noise = r.uniform(0.01, 0.09, N)
    else:
        clearance = r.gamma(1.35, 0.22, N)
        error = r.normal(0, 0.28, N)
        speed = r.uniform(0.25, 1.1, N)
        noise = r.uniform(0.02, 0.13, N)

    penalty = 0.07 * speed + 0.45 * np.abs(error)
    true_clearance = clearance - radius - penalty
    measured = clearance + r.normal(0, noise) - 0.25 - penalty
    uncertainty = noise + 0.025 + 0.03 * np.abs(error)

    return {
        "unsafe": true_clearance < 0.03,
        "clearance": measured,
        "uncertainty": uncertainty
    }

def evaluate(data, policy):
    start = time.perf_counter()
    c, u, unsafe = data["clearance"], data["uncertainty"], data["unsafe"]

    if policy == "no_verifier":
        decision = np.full(N, "accept", dtype="<U6")
    elif policy == "fixed_rule":
        decision = np.where(c >= 0.08, "accept", "reject")
    else:
        decision = np.where(
            c < 0.03, "reject",
            np.where(c < 0.06 + 2.5*u, "defer", "accept")
        )

    latency_us = (time.perf_counter() - start) * 1e6 / N
    accepted = decision == "accept"
    rejected = decision == "reject"
    deferred = decision == "defer"
    n_unsafe = max(1, unsafe.sum())
    n_safe = max(1, (~unsafe).sum())

    return {
        "n": N,
        "input_unsafe_rate": unsafe.mean(),
        "unsafe_accept_rate": (accepted & unsafe).sum()/n_unsafe,
        "safe_reject_rate": (rejected & ~unsafe).sum()/n_safe,
        "defer_rate": deferred.mean(),
        "accept_rate": accepted.mean(),
        "safe_action_accept_rate": (accepted & ~unsafe).sum()/n_safe,
        "policy_time_us_per_proposal": latency_us
    }

scenarios = {
    "seen_robot_train_env": (0.25, "train", 1001),
    "heldout_larger_robot": (0.35, "transfer", 1002),
    "heldout_smaller_robot": (0.15, "transfer", 1003),
    "stress_environment": (0.30, "stress", 1004)
}
policies = ["no_verifier", "fixed_rule", "uncertainty_aware"]
rows, metadata = [], {}

for name, (radius, domain, seed) in scenarios.items():
    data = make_scenario(name, radius, domain, seed)
    metadata[name] = {
        "radius_m": radius,
        "distribution": domain,
        "seed": seed,
        "input_unsafe_rate": float(data["unsafe"].mean())
    }
    for policy in policies:
        row = evaluate(data, policy)
        row.update(scenario=name, policy=policy)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)

with open("summary.json", "w") as f:
    json.dump({
        "seed": SEED,
        "proposals_per_scenario": N,
        "total_proposals": N * len(scenarios),
        "scenario_metadata": metadata,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "limitation": (
            "Synthetic clearance benchmark only; not a physics simulator, "
            "physical robot validation, or formal safety guarantee."
        )
    }, f, indent=2)

print("\n=== ROBOT VERIFIER BENCHMARK ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nSaved: results.csv, summary.json")
print("NOTE: Synthetic benchmark, not physical robot validation.")
