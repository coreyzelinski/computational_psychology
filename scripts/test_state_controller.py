# =============================================================================
#  Affective Computational Psychology Model
#  Script: test_state_controller.py
#
#  Description:
#      Diagnostic script to test the HybridBehavioralModel. This script simulates 
#      a full update cycle, including Bayesian posterior update, reinforcement 
#      learning (Q-learning/RL), and softmax-based transition prediction.
#
#      This test validates coordination of probabilistic inference and 
#      empirical reward feedback within the attachment-driven interaction model.
#
#  Author: Corey Vincent Zelinski
#  Email:  Corey.Zelinski@IEEE.org
#  © 2024-2025 Corey Vincent Zelinski. All rights reserved.
# =============================================================================

from model_engine.state_controller import HybridBehavioralModel
import numpy as np

# INITIAL PRIOR AND PARAMS

initial_prior = [0.2, 0.2, 0.2, 0.2, 0.2]
likelihood = [0.05, 0.6, 0.2, 0.1, 0.05]  # Observed evidence favors state 1

# STATE TRANSITION & REWARD SETUP
# e.g., subject moved from 1 -> 2 with reward +1.2

s = 1
a = 2
r = 1.2
s_next = 2

# INITIALIZE MODEL

model = HybridBehavioralModel(initial_probs=initial_prior, alpha=0.2, gamma=0.95, tau=1.0)

# EXECUTE FULL UPDATE

posterior, predicted_probs = model.run_step(likelihoods=likelihood, s=s, a=a, r=r, s_next=s_next)

# RETURN OUTPUT RESULTS

print("Posterior belief distribution after Bayesian update:")
print(np.round(posterior, 4))

print("\nPredicted next-state probabilities (softmax from Q-values):")
print(np.round(predicted_probs, 4))
