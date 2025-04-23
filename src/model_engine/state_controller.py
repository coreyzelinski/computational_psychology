# =============================================================================
#  Affective Computational Psychology Model
#  Module: state_controller.py
#
#  Description:
#      Orchestration module for Bayesian and RL components. Coordinates 
#      belief updates, reinforcement signals, and next-state predictions 
#      in the hybrid attachment-behavior simulation model.
#
#      The HybridBehavioralModel class integrates Bayesian inference with 
#      reinforcement learning using empirical reward feedback to simulate 
#      attachment-driven human interaction over time.
#
#  Author: Corey Vincent Zelinski
#  Email:  Corey.Zelinski@IEEE.org
#  © 2025 Corey Vincent Zelinski. All rights reserved.
# =============================================================================

from model_engine.bayes_markov import BayesianUpdater
from model_engine.event_modulation import QUpdater
import numpy as np

class HybridBehavioralModel:
    def __init__(self, initial_probs, alpha=0.1, gamma=0.95, tau=1.0):
        """
        Initialize the hybrid model with Bayesian and RL components.
        ARGUMENTS
            initial_probs (list or np.array): Initial prior over engagement states
            alpha (float): Learning rate for RL
            gamma (float): Discount factor for RL
            tau (float): Temperature parameter for softmax
        """
        self.num_states = len(initial_probs)
        self.bayes = BayesianUpdater(initial_probs)
        self.q_updater = QUpdater(num_states=self.num_states, alpha=alpha, gamma=gamma)
        self.tau = tau

    def observe(self, likelihoods):
        """
        Apply Bayesian update to internal belief state.
        ARGUMENTS
            likelihoods (list or np.array) - Likelihoods P(E | H_i) for each state
        OUTPUTS
            posterior (np.array) - Updated belief distribution
        """
        return self.bayes.update(likelihoods)

    def reinforce(self, s, a, r, s_next):
        """
        Apply reinforcement learning update based on observed reward.
        ARGUMENTS
            s (int), Current state index
            a (int), Action/transition index
            r (float), Observed reward
            s_next (int), Resulting state index
        """
        self.q_updater.update(s, a, r, s_next)

    def predict_next_state(self, current_state):
        """
        Generate transition probabilities using softmax Q-values
        ARGUMENTS
            current_state (int), index of the current state
        OUTPUTS
            np.array, softmax-normalized probabilities for next state
        """
        return self.q_updater.get_softmax_policy(current_state, tau=self.tau)

    def run_step(self, likelihoods, s, a, r, s_next):
        """
        Run full update cycle (Bayesian update + RL update + policy prediction)
        ARGUMENTS
            likelihoods, Observed evidence
            s, Current state
            a, Action taken
            r, Reward received
            s_next, Next state reached
        OUTPUTS
            posterior, predicted_probs - Belief vector and softmax policy
        """
        posterior = self.observe(likelihoods)
        self.reinforce(s, a, r, s_next)
        predicted_probs = self.predict_next_state(s)
        return posterior, predicted_probs
