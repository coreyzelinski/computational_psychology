# =============================================================================
#  Affective Computational Psychology Model
#  Script: test_q_updater.py
#
#  Description:
#      Diagnostic example for testing the Q-learning updates and 
#      softmax policy generation using QUpdater class. Demonstrates 
#      basic state-action updates with artificial reward feedback.
#
#      Part of a hybrid model integrating Bayesian inference, 
#      reinforcement learning, and Markov processes to simulate 
#      attachment-driven dynamics of human interaction.
#
#  Author: Corey Vincent Zelinski
#  Email:  Corey.Zelinski@IEEE.org
#  © 2024-2025 Corey Vincent Zelinski. All rights reserved.
# =============================================================================

from model_engine.event_modulation import QUpdater
import numpy as np

# SIMULATION PARAMS

num_states = 5
alpha = 0.2
gamma = 0.9
tau = 1.0

# INITIALIZE Q-UPDATER

q_updater = QUpdater(num_states=num_states, alpha=alpha, gamma=gamma)

# SIMULATE REWARD-BASED LEARNING
# e.g., from S_1 to S_2 with reward +1.5 resulting in `state 2` (S_2)

q_updater.update(s=1, a=2, r=1.5, s_next=2)

# OUTPUT Q-TABLE AFTER UPDATE

print("Q-table after single update:")
print(np.round(q_updater.q_table, 3))

# GENERATE SOFTMAX POLICY FROM STATE 1

policy = q_updater.get_softmax_policy(state=1, tau=tau)
print("\nSoftmax-normalized policy from state 1:")
print(np.round(policy, 4))
