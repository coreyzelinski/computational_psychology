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
        Initialize with a vector of prior probabilities over states
        """
        self.prior = np.array(prior_probs, dtype=np.float64)
        self.normalize()

    def normalize(self):
        """
        All priors must sum to 1
        """
        total = np.sum(self.prior)
        if total > 0:
            self.prior = self.prior / total

    def update(self, likelihoods):
        """
        Apply Bayesian update from posterior sample
        ARGUMENTS
            likelihoods (list/array), P(E | H_i) for each state H_i
        OUTPUTS
            posterior (np.array), updated belief distribution P(H_i | E)
        """
        likelihoods = np.array(likelihoods, dtype=np.float64)
        unnormalized = self.prior * likelihoods
        total = np.sum(unnormalized)
        if total == 0:
            raise ValueError("Invalid Bayesian update: product of prior and likelihood yields zero normalization constant.")
        posterior = unnormalized / total
        self.prior = posterior  # update internal state
        return posterior

    def get_current_belief(self):
        """
        Return posterior belief vector
        """
        return self.prior