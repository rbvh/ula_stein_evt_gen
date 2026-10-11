"""Checks for weight-tail efficiencies and frozen VEGAS surveys.

Run with python -m unittest test_unweighting.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ula_stein_evt_gen.mc_utils import (
    compute_unweighting_efficiency,
    compute_unweighting_statistics,
)
from ula_stein_evt_gen.vegas import VegasIntegrator


class UnweightingTests(unittest.TestCase):
    def test_weight_contribution_tail_and_acceptance(self):
        # The largest event is 10% of the integral, but only 1/901 events.
        weights = np.array([100.0] + [1.0] * 900 + [0.0] * 99)
        stats = compute_unweighting_statistics(weights, (0.001, 0.01, 0.11))
        self.assertAlmostEqual(stats[0.01]["max_weight_eff"], 100.0)
        self.assertAlmostEqual(stats[0.11]["max_weight_eff"], 1.0)
        self.assertAlmostEqual(stats[0.11]["epsilon"], 1.0)
        self.assertAlmostEqual(stats[0.11]["acceptance"], 0.901)
        self.assertAlmostEqual(stats[0.11]["overweight_integral_fraction"], 0.1)
        self.assertAlmostEqual(stats[0.11]["trials_per_accepted_event"], 1 / 0.901)
        self.assertAlmostEqual(stats[0.0]["epsilon"], 0.01)
        self.assertAlmostEqual(stats[0.0]["inverse_epsilon"], 100.0)

    def test_ties_and_exact_tail_boundary(self):
        weights = np.array([4.0, 4.0, 2.0, 2.0, 0.0])
        stats = compute_unweighting_statistics(weights, (0.5, 2 / 3))
        self.assertAlmostEqual(stats[0.5]["max_weight_eff"], 4.0)
        self.assertEqual(stats[0.5]["overweight_event_fraction"], 0.0)
        self.assertAlmostEqual(stats[2 / 3]["max_weight_eff"], 2.0)
        self.assertAlmostEqual(stats[2 / 3]["overweight_integral_fraction"], 2 / 3)
        self.assertAlmostEqual(stats[2 / 3]["acceptance"], 0.8)

    def test_tail_bound_against_direct_sums_and_scale_invariance(self):
        weights = np.random.default_rng(0).lognormal(size=10_000)
        stats = compute_unweighting_statistics(weights)
        rescaled = compute_unweighting_efficiency(weights * 1e200)
        for a, row in stats.items():
            threshold = row["max_weight_eff"]
            fraction = weights[weights > threshold].sum() / weights.sum()
            self.assertLessEqual(fraction, a + 1e-14)
            self.assertAlmostEqual(row["epsilon"], rescaled[a])
            self.assertAlmostEqual(row["acceptance"], np.minimum(1, weights / threshold).mean())
            self.assertLessEqual(row["acceptance"], 1 + 1e-14)

    def test_invalid_samples_and_fractions(self):
        for weights in ([], [0, 0], [1, -1], [1, np.nan], [np.inf], [[1]]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                compute_unweighting_efficiency(weights)
        for fraction in (-0.1, 1.0, np.nan, np.inf):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                compute_unweighting_efficiency([1, 2], (fraction,))

    def test_frozen_nonuniform_grid_and_ragged_batches(self):
        # An exactly matched target and proposal yield unit importance weights.
        def density(x):
            q0 = jnp.where(x[:, 0] < 0.2, 1 / (2 * 0.2), 1 / (2 * 0.8))
            q1 = jnp.where(x[:, 1] < 0.7, 1 / (2 * 0.7), 1 / (2 * 0.3))
            return q0 * q1

        vegas = VegasIntegrator(density, dim=2, n_bins=2)
        vegas.subdivisions = jnp.array([[0, 0.2, 1], [0, 0.7, 1]])
        grid = np.asarray(vegas.subdivisions).copy()
        messages = []
        survey = vegas.measure_unweighting(jax.random.PRNGKey(0), n_points=1003,
                                          batch_size=200, status_fn=messages.append)
        self.assertEqual(survey["n_trials"], 1003)
        self.assertEqual(len(messages), 6)
        self.assertAlmostEqual(survey["cross_section"], 1.0, places=6)
        for row in survey["efficiencies"].values():
            self.assertAlmostEqual(row["epsilon"], 1.0, places=6)
        np.testing.assert_array_equal(vegas.subdivisions, grid)
        efficiencies = vegas.get_unweighting_efficiency(jax.random.PRNGKey(1),
                                                       n_points=101, batch_size=50)
        self.assertAlmostEqual(efficiencies[0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
