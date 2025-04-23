# =============================================================================
#  Affective Computational Psychology Model
#  Script: run_simulation.py
#
#  Description:
#      Primary driver script for full-cycle simulation of empirical 
#      behavioral trajectories. Runs multi-step interaction modeling 
#      using HybridBehavioralModel with Bayesian + RL feedback loop.
#
#      Simulates probabilistic observations and reward signals across 
#      discrete time steps, modeling transition adaptation and belief evolution.
#
#  Author: Corey Vincent Zelinski
#  Email:  Corey.Zelinski@IEEE.org
#  (C) Copyright 2024-2025 Corey Vincent Zelinski. All rights reserved.
# =============================================================================

from model_engine.state_controller import HybridBehavioralModel
import numpy as np

def simulate_session(steps=50, spike_step=27, spike_likelihood=None, baseline_likelihood=None):
    # === Setup: Model and Parameters ===
    initial_prior = [0.2] * 5
    model = HybridBehavioralModel(initial_probs=initial_prior, alpha=0.2, gamma=0.9, tau=1.0)

    state_log = []
    belief_log = []

    # Default likelihoods
    if baseline_likelihood is None:
        baseline_likelihood = [0.2, 0.2, 0.2, 0.2, 0.2]
    if spike_likelihood is None:
        spike_likelihood = [0.05, 0.6, 0.15, 0.1, 0.1]

    for t in range(steps):
        # Simulated input: spike on defined step, noise otherwise
        likelihoods = spike_likelihood if t == spike_step else baseline_likelihood

        # Simplified empirical reward model, reinforce state 1 -> 2
        s, a, s_next = 1, 2, 2
        r = 1.0 if t == spike_step else 0.05

        posterior, predicted_probs = model.run_step(likelihoods, s, a, r, s_next)

        state_log.append(predicted_probs)
        belief_log.append(posterior)

    return belief_log, state_log

if __name__ == "__main__":
    belief, transitions = simulate_session()

    print("Final posterior belief distribution:")
    print(np.round(belief[-1], 4))

    print("\nFinal predicted transition probabilities:")
    print(np.round(transitions[-1], 4))
