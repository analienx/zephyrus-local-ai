"""CPU-only mathematical invariants. Run: python -m unittest discover -s tests -v."""
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from spec_math import coupled_acceptance, one_position_output, validate


class SpeculativeProbabilityTests(unittest.TestCase):
    def assert_target_law(self, p, q):
        out, acceptance = one_position_output(p, q)
        for actual, expected in zip(out, p):
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertAlmostEqual(sum(out), 1.0, places=10)
        self.assertAlmostEqual(acceptance, sum(min(a, b) for a, b in zip(p, q)), places=10)

    def test_identical_distributions_always_accept(self):
        p = [0.02, 0.18, 0.3, 0.5]
        output, acceptance = one_position_output(p, p)
        self.assertEqual(output, p)
        self.assertAlmostEqual(acceptance, 1.0)

    def test_restricted_draft_support_preserves_full_target(self):
        self.assert_target_law([0.1, 0.2, 0.3, 0.4], [0.0, 0.75, 0.25, 0.0])
        self.assert_target_law([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self.assert_target_law([0.0, 1.0, 0.0], [0.0, 1.0, 0.0])

    def test_conditional_coupling_has_correct_acceptance(self):
        for p_d in [0.0, 0.01, 0.2, 0.5, 0.99, 1.0]:
            for q_d in [0.01, 0.2, 0.5, 0.99, 1.0]:
                a, extra = coupled_acceptance(p_d, q_d)
                self.assertAlmostEqual(p_d + (1 - p_d) * extra, a)
                self.assertAlmostEqual(a, min(1.0, p_d / q_d))
        with self.assertRaises(ValueError):
            coupled_acceptance(0.5, 0.0)

    def test_random_normalized_proposals_and_target(self):
        rng = random.Random(20260919)
        for size in (2, 3, 16, 128):
            for _ in range(30):
                raw_p = [rng.random() for _ in range(size)]
                raw_q = [rng.random() if rng.random() < 0.5 else 0.0 for _ in range(size)]
                if not any(raw_q):
                    raw_q[0] = 1.0
                p = [x / sum(raw_p) for x in raw_p]
                q = [x / sum(raw_q) for x in raw_q]
                self.assert_target_law(p, q)

    def test_reject_invalid_distributions(self):
        for p, q in [([], []), ([1.0], [0.5]), ([1.0], []),
                     ([float("nan"), 0.0], [1.0, 0.0]), ([1.0, -0.0], [1.0]),
                     ([1.1, -0.1], [0.5, 0.5])]:
            with self.assertRaises(ValueError):
                validate(p, q)


if __name__ == "__main__":
    unittest.main()
