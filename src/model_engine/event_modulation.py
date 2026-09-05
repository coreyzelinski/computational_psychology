# =============================================================================
#  Affective Computational Psychology Model
#  Module: event_modulation.py
#
#  Description:
#      This module implements RL logic using the Q-update rule. It provides
#      reward-driven modulation of behavioral engagement dynamics
#      and generates softmax-normalized policy transitions between states.
#
#      The QUpdater class tracks expected values for state-action pairs and
#      updates them based on empirical reward feedback. A softmax function
#      translates Q-values into probabilistic state transitions, adjustable
#      via temperature parameter.
#
#      This module integrates with Bayesian inference and Markov modeling
#      to simulate the behavioral dynamics of human attachment.
#
#  Author: Corey Vincent Zelinski
#  Email:  Corey.Zelinski@IEEE.org
#  (C) Copyright 2024-2025 Corey Vincent Zelinski. All rights reserved.
# =============================================================================

import numpy as np


class QUpdater:
    def __init__(self, num_states, alpha=0.1, gamma=0.95):
        """
        Initialize the Q-table and learning parameters.

        Args:
            num_states (int): Number of engagement states
            alpha (float): Learning rate
            gamma (float): Discount factor
        """
        if int(num_states) != num_states or num_states <= 0:
            raise ValueError("num_states must be a positive integer.")

        self.num_states = int(num_states)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q_table = np.zeros(
            (self.num_states, self.num_states), dtype=np.float64
        )

    def update(self, s, a, r, s_next):
        """
        Perform Q-learning update for a state-action pair.

        Args:
            s (int): Current state index
            a (int): Action or next state index
            r (float): Observed reward
            s_next (int): Resulting state after transition
        """
        max_q_next = np.max(self.q_table[s_next])
        old_value = self.q_table[s, a]
        td_target = r + self.gamma * max_q_next
        self.q_table[s, a] = old_value + self.alpha * (td_target - old_value)

    @staticmethod
    def _validate_policy_parameters(tau, sigma, dt):
        tau = float(tau)
        sigma = float(sigma)
        dt = float(dt)

        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and greater than zero.")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and non-negative.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and greater than zero.")

        return tau, sigma, dt

    @staticmethod
    def _coerce_rng(rng):
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng(rng)

    def get_softmax_policy(self, state, tau=1.0, sigma=0.0, dt=1.0, rng=None):
        """
        Generate softmax-normalized transition probabilities from Q-values.

        Optional stochasticity is applied in Q/logit space before softmax so
        the returned transition vector remains on the probability simplex.

        Args:
            state (int): Current state index
            tau (float): Temperature parameter
            sigma (float): Standard deviation of stochastic Q-value perturbation
            dt (float): Time step used to scale stochastic perturbation
            rng: NumPy Generator or seed for reproducible stochasticity

        Returns:
            np.array: Softmax-normalized vector of transition probabilities
        """
        tau, sigma, dt = self._validate_policy_parameters(tau, sigma, dt)
        q_values = np.asarray(self.q_table[state], dtype=np.float64)

        if not np.all(np.isfinite(q_values)):
            raise ValueError("Q-values must be finite.")

        if sigma:
            generator = self._coerce_rng(rng)
            q_values = q_values + (
                sigma * np.sqrt(dt) * generator.standard_normal(q_values.shape)
            )

        logits = q_values / tau
        logits = logits - np.max(logits)
        exp_q = np.exp(logits)
        normalizer = np.sum(exp_q)

        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("Softmax normalization failed.")

        return exp_q / normalizer

    def get_transition_matrix(self, tau=1.0, sigma=0.0, dt=1.0, rng=None):
        """
        Return a row-stochastic transition matrix for all states.
        """
        tau, sigma, dt = self._validate_policy_parameters(tau, sigma, dt)
        generator = self._coerce_rng(rng) if sigma else None

        return np.vstack(
            [
                self.get_softmax_policy(
                    state,
                    tau=tau,
                    sigma=sigma,
                    dt=dt,
                    rng=generator,
                )
                for state in range(self.num_states)
            ]
        )
