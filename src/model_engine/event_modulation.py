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
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((num_states, num_states), dtype=np.float64)

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

    def get_softmax_policy(self, state, tau=1.0):
        """
        Generate softmax-normalized transition probabilities from Q-values.
        Args:
            state (int): Current state index
            tau (float): Temperature parameter
        Returns:
            np.array: Softmax-normalized vector of transition probabilities
        """
        q_values = self.q_table[state]
        exp_q = np.exp(q_values / tau)
        return exp_q / np.sum(exp_q)
