# =============================================================================
#  Affective Computational Psychology Model
#  Script: test_model.py
#
#  Description:
#      Standalone example script - tests Bayesian state updating leveraging 
#      the BayesianUpdater class. This script simulates a prior belief
#      vector and observed evidence to validate posterior calculation.
#
#      This test is part of a larger framework combining Bayesian inference, 
#      Markov chains, and reinforcement learning for modeling attachment-driven 
#      dynamics of human interaction.
#
#  Author: Corey Vincent Zelinski
#  Email:  Corey.Zelinski@IEEE.org
#  © 2024-2025 Corey Vincent Zelinski. All rights reserved.
# =============================================================================

from model_engine.bayes_markov import BayesianUpdater

# STEP 0 - Prior belief (uniform)
prior = [0.2, 0.2, 0.2, 0.2, 0.2]

# STEP 1 - Likelihood of observation given each state
# (e.g., based on observed behavior or content interaction)
likelihood = [0.1, 0.5, 0.2, 0.15, 0.05]

# STEP 2 - Initialize updater and compute posterior
updater = BayesianUpdater(prior)
posterior = updater.update(likelihood)

# STEP 3 - Output result
print("Posterior belief vector after Bayesian update:")
print(posterior)