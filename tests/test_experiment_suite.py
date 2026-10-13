import importlib.util
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "experiment_suite", ROOT / "simulator" / "experiment_suite.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class SimulatorTests(unittest.TestCase):
    def test_deterministic_sampling(self):
        a = mod.seeded_episodes(100, 123)
        b = mod.seeded_episodes(100, 123)
        self.assertEqual(a, b)

    def test_weighted_score_in_range(self):
        eps = mod.seeded_episodes(100, 123)
        for ep in eps:
            s = mod.score_episode(ep)
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, 1)

    def test_metrics_sum(self):
        eps = mod.seeded_episodes(1000, 123)
        r = mod.evaluate(eps)
        self.assertEqual(r["safe"] + r["unsafe"], 1000)
        self.assertEqual(r["safe_accepted"] + r["safe_rejected"], r["safe"])
        self.assertEqual(r["unsafe_accepted"] + r["unsafe_rejected"], r["unsafe"])

    def test_weights_sum_to_one(self):
        for weights in mod.WEIGHTS.values():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=8)

    def test_label_is_boolean(self):
        ep = mod.sample_episode(random.Random(5))
        self.assertIsInstance(ep["safe"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
