# =============================================================================
#  Affective Computational Psychology Model
#  Module: bayes_markov.py
#
#  Description:
#      This module implements Bayesian updating for state inference
#      in attachment-based behavioral prediction. It is part of a hybrid
#      model combining Bayesian filtering, Markov transitions, and
#      reinforcement learning to predict and simulate attachment-informed
#      engagement trajectories.
#
#  Author: Corey Vincent Zelinski
#  Email:  Corey.Zelinski@IEEE.org
#  © 2024-2025 Corey Vincent Zelinski. All rights reserved.
# =============================================================================

import numpy as np


class BayesianUpdater:
    def __init__(self, prior_probs):
        """
        Initialize with a vector of prior probabilities over states.
        """
        self.prior = self._normalize_probability_vector(prior_probs, "prior")

    @staticmethod
    def _as_vector(values, label):
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"{label} must be a non-empty one-dimensional vector.")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} must contain only finite values.")
        return vector

    @classmethod
    def _normalize_probability_vector(cls, values, label):
        vector = cls._as_vector(values, label)
        if np.any(vector < 0):
            raise ValueError(f"{label} cannot contain negative probabilities.")

        total = np.sum(vector)
        if total <= 0:
            raise ValueError(f"{label} must have a positive normalization constant.")

        return vector / total

    def normalize(self):
        """
        Normalize the current prior so it sums to 1.
        """
        self.prior = self._normalize_probability_vector(self.prior, "prior")
        return self.prior

    def update(self, likelihoods):
        """
        Apply Bayesian update from posterior sample.

        ARGUMENTS
            likelihoods (list/array), P(E | H_i) for each state H_i
        OUTPUTS
            posterior (np.array), updated belief distribution P(H_i | E)
        """
        likelihoods = self._as_vector(likelihoods, "likelihoods")

        if likelihoods.shape != self.prior.shape:
            raise ValueError(
                f"likelihoods shape {likelihoods.shape} does not match prior "
                f"shape {self.prior.shape}."
            )
        if np.any(likelihoods < 0):
            raise ValueError("likelihoods cannot contain negative values.")

        unnormalized = self.prior * likelihoods
        total = np.sum(unnormalized)
        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                "Invalid Bayesian update: prior times likelihoods yields a "
                "non-positive normalization constant."
            )

        posterior = unnormalized / total
        self.prior = posterior
        return posterior

    def get_current_belief(self):
        """
        Return posterior belief vector.
        """
        return self.prior
