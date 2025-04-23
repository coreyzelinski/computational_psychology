
# A Bayesian Hybrid AI Computational Model for Behavioral Prediction

**Author:** Corey Vincent Zelinski  
**Email:** Corey.Zelinski@IEEE.org   
**(C) Copyright 2024–2025 Corey Vincent Zelinski. All rights reserved.**

---

## Overview

This repository presents a hybrid computational model integrating **Bayesian inference**, **Markov chains**, and **reinforcement learning (RL)** for predicting empirically grounded behavioral dynamics in human interaction. It is designed to simulate and analyze dynamic belief updates and reinforcement-adaptive transitions.

---

## Abstract

This model enables iterative probabilistic updates of dyadic-interactive likelihoods (via Bayesian filtering) and adaptive learning from past interactions (via reinforcement learning). We compare the hybrid approach to traditional Markov-based probability matrices and demonstrate that combining probabilistic reasoning with adaptive learning improves prediction accuracy.

The model includes:
- **Probability Generating Function (PGF)** for discrete transitions
- **Continuous Generating Function (CGF)** to account for engagement decay over time
- **Latent-State Model (LSM)** for independent validation

---

## Interactive State Definitions

Let \( S = \{S_0, S_1, S_2, S_3, S_4\} \) represent five discrete behavioral states:

| State | Description |
|-------|-------------|
| \( S_0 \) | **State 0** |
| \( S_1 \) | **State 1**  |
| \( S_2 \) | **State 2**  |
| \( S_3 \) | **State 3** |
| \( S_4 \) | **State 4** |

---

## Computational Methodology

The model proceeds in the following steps:

1. Identify the **current state** \( S_t \)

2. Retrieve the **current transition matrix** \( P \)

The Markov transition matrix \( P \) is defined as:

\[
P = 
\begin{bmatrix}
p_{00} & p_{01} & p_{02} & p_{03} & p_{04} \\
p_{10} & p_{11} & p_{12} & p_{13} & p_{14} \\
p_{20} & p_{21} & p_{22} & p_{23} & p_{24} \\
p_{30} & p_{31} & p_{32} & p_{33} & p_{34} \\
p_{40} & p_{41} & p_{42} & p_{43} & p_{44}
\end{bmatrix}
\]

Each row sums to 1:  
\[
\sum_j p_{ij} = 1 \quad \text{for all } i
\]

This matrix evolves based on empirical data and reinforcement adjustments.

3. Compute the **next transition probabilities** using \( Q \)-learning:

   $$
   Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
   $$
   
   Where:

- \( \alpha \): learning rate
- \( \gamma \): discount factor
- \( r \): reward signal
- \( s' \): next predicted state

Define:
\[
\Delta Q = r + \gamma \max_{a'} Q(s',a') - Q(s,a)
\]

This guides transition probabilities based on reinforcement.

4. Apply the **softmax function** to generate the updated transition probabilities:

   $$
   P(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_b \exp(Q(s,b)/\tau)}
   $$


Where:
- \( \tau \): temperature parameter controlling exploration (low \( \tau \) = sharper preference)

**Example**:  
If \( Q_{12} = 2.5, Q_{13} = 1.2, Q_{14} = 0.8 \), then:

\[
P_{12} = \frac{\exp(2.5 / \tau)}{\exp(2.5 / \tau) + \exp(1.2 / \tau) + \exp(0.8 / \tau)}
\]

5. Introduce **stochastic variability**

To prevent overfitting and reflect empirical volatility:

\[
dP_{ij} = \beta \left( \frac{\exp(Q_{ij} / \tau)}{\sum_k \exp(Q_{ik} / \tau)} - P_{ij} \right) dt + \sigma dW_{ij}
\]

Where:
- \( \beta \): transition rate adaptation coefficient
- \( \sigma dW_{ij} \): stochastic noise (Wiener process component)

This keeps transitions dynamic and empirically sensitive to external perturbations.

6. Update transition matrix \( P \) for the next cycle

---

## Belief Update (Bayesian Filtering)

Beliefs are updated with incoming observations:
$$
P(H_i \mid E) = \frac{P(E \mid H_i) P(H_i)}{\sum_j P(E \mid H_j) P(H_j)}
$$

---

## Spectral and Latent-State Extensions

The model is extended via:
- **Probabilistic Generating Function (PGF)**: captures discrete probabilistic transitions
- **Cumulant Generating Function (CGF)**: integrates long-range decay via fractional derivatives
- **Latent-State Modeling (LSM)**: latent layer for independent validation of modeled dynamics
- **Fast Fourier Transform (FFT) Analysis**: time-domain reactivation detection via spectral energy peaks

---

## Repository Contents

- `bayes_markov.py`: Bayesian inference module
- `event_modulation.py`: Q-learning and softmax reinforcement
- `state_controller.py`: Hybrid behavioral model combining modules
- `run_simulation.py`: Executes full model loop with multi-step simulation
- `notebooks/affective_fft_model.ipynb`: Visualization + FFT diagnostics
- `test_*.py`: Sanity checks for each module

---

## Status

- Core algorithm complete
- Visualization and time-domain simulation working
- First-iteration inclusion of FFT spectral analysis
- AMA-formatted paper and LaTeX doc provides full mathematical description

---

## License

(C) Copyright 2024–2025 Corey Vincent Zelinski.  
All rights reserved. Reuse, redistribution, or derivative work is prohibited without express permission.

---

## Citation

Formal publication pending. For inquiries, contact **Corey.Zelinski@IEEE.org**.

