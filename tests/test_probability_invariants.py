import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_engine.bayes_markov import BayesianUpdater
from model_engine.event_modulation import QUpdater


class BayesianInvariantTests(unittest.TestCase):
    def test_prior_is_normalized(self):
        updater = BayesianUpdater([1, 1, 2])
        self.assertTrue(
            np.allclose(updater.get_current_belief(), [0.25, 0.25, 0.5])
        )
        self.assertAlmostEqual(
            float(np.sum(updater.get_current_belief())), 1.0
        )

    def test_negative_prior_rejected(self):
        with self.assertRaises(ValueError):
            BayesianUpdater([1.2, -0.2])

    def test_zero_prior_rejected(self):
        with self.assertRaises(ValueError):
            BayesianUpdater([0.0, 0.0])

    def test_posterior_remains_probability_vector(self):
        updater = BayesianUpdater([0.2] * 5)
        posterior = updater.update([0.1, 0.5, 0.2, 0.15, 0.05])
        self.assertTrue(np.all(posterior >= 0))
        self.assertAlmostEqual(float(np.sum(posterior)), 1.0)

    def test_negative_likelihood_rejected(self):
        updater = BayesianUpdater([0.5, 0.5])
        with self.assertRaises(ValueError):
            updater.update([1.0, -0.1])

    def test_likelihood_shape_must_match_prior(self):
        updater = BayesianUpdater([0.5, 0.5])
        with self.assertRaises(ValueError):
            updater.update([1.0, 0.5, 0.2])


class TransitionInvariantTests(unittest.TestCase):
    def test_stable_softmax_handles_large_q_values(self):
        updater = QUpdater(num_states=5)
        updater.q_table[0] = [10000, 9999, 9998, 9997, 9996]
        policy = updater.get_softmax_policy(0)
        self.assertTrue(np.all(np.isfinite(policy)))
        self.assertTrue(np.all(policy >= 0))
        self.assertAlmostEqual(float(np.sum(policy)), 1.0)

    def test_tau_must_be_positive(self):
        updater = QUpdater(num_states=2)
        with self.assertRaises(ValueError):
            updater.get_softmax_policy(0, tau=0)

    def test_stochastic_policy_stays_on_simplex(self):
        updater = QUpdater(num_states=5)
        updater.q_table[2] = [2.0, -1.0, 0.5, 4.0, 1.0]
        policy = updater.get_softmax_policy(
            2, sigma=0.8, dt=0.25, rng=12345
        )
        self.assertTrue(np.all(policy >= 0))
        self.assertAlmostEqual(float(np.sum(policy)), 1.0)

    def test_seeded_stochastic_policy_is_reproducible(self):
        updater = QUpdater(num_states=3)
        first = updater.get_softmax_policy(1, sigma=0.5, rng=77)
        second = updater.get_softmax_policy(1, sigma=0.5, rng=77)
        self.assertTrue(np.allclose(first, second))

    def test_transition_matrix_is_row_stochastic(self):
        updater = QUpdater(num_states=5)
        updater.q_table[:] = np.arange(25, dtype=float).reshape(5, 5)
        matrix = updater.get_transition_matrix(
            tau=0.7, sigma=0.4, dt=0.5, rng=2026
        )
        self.assertEqual(matrix.shape, (5, 5))
        self.assertTrue(np.all(matrix >= 0))
        self.assertTrue(
            np.allclose(np.sum(matrix, axis=1), np.ones(5))
        )


if __name__ == "__main__":
    unittest.main()
