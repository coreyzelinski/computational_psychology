
# A Bayesian Hybrid AI Computational Model for Behavioral Prediction

**Author:** Corey Vincent Zelinski  
**Email:** Corey.Zelinski@IEEE.org   
**(C) Copyright 2024–2025 Corey Vincent Zelinski. All rights reserved.**

---

## Overview

This repository presents a hybrid computational model integrating **Bayesian inference**, **Markov chains**, and **reinforcement learning (RL)** for predicting empirically grounded behavioral dynamics in human interaction. It is designed to simulate and analyze dynamic belief updates and reinforcement-adaptive transitions.

---

## Abstract

This model enables iterative probabilistic updates of dyadic-interactive likelihoods (via Bayesian filtering) and adaptive learning from past interactions (via reinforcement learning). We compare the hybrid approach to traditional Markov-based probability matrices and demonstrate that combining probabilistic reasoning with adaptive learning improves prediction accuracy. The computational framework includes a hybrid predictive system that integrates Bayesian state inference, reinforcement learning (**RL**), and Markovian transition logic to model engagement dynamics in attachment-informed behavioral systems. Iterative Bayesian updates allow belief revision based on new interaction signals, while reinforcement learning adjusts transition strategies based on empirical rewards over time.

The model includes:
- **Probability Generating Function (PGF)** for discrete transitions
- **Continuous Generating Function (CGF)** to account for engagement decay over time
- **Latent-State Model (LSM)** for independent validation

In addition to its core probabilistic engine, the model supports auxiliary formulations for extended analysis:
- **Probability Generating Function (PGF)** — characterizes discrete state transitions
- **Cumulant Generating Function (CGF)** — introduces time-decaying memory via fractional-order derivatives
- **Latent-State Model (LSM)** — provides an independent representational layer for latent-state resurfacing estimate
- **Spectral Analysis (FFT)** — identifies periodic or symbolic-interactive spikes in belief trajectories
- **Stochastic Differential Equation (SDE)** — governs continuous-time variability in transition dynamics via empirical noise injection
- **Fractional Langevin Equation (FLE)** — models long-memory symbolic dynamics through fractional-order inertia and stochastic forcing terms, affecting delayed symbolic influence on state likelihoods

---

## Analytical Formulations and Extended Diagnostics

The computational model is reinforced independently by a suite of mathematical constructs and formulations:

- **PGF and CGF** model engagement transitions through discrete and continuous temporal structures, modeling decay and symbolic influence on state likelihoods
- The **Latent-State Model (LSM)** provides an orthogonal validation layer, reflecting symbolic and unobserved states that may influence observed human behavior
- **Fast Fourier Transform (FFT)** analysis is applied to posterior trajectories to project or describe cyclical behavioral signals, symbolic periodicities, and phase-delayed behavioral cues
- The **Stochastic Differential Equation (SDE)** component introduces empirical fluctuation dynamics through the realization of a Wiener process, reflecting natural engagement noise
- The **Fractional Langevin Equation (FLE)** encodes symbolic inertia and memory effects, extending the CGF layer with fractional-order kinetic terms and noise coupling

These components allow the model to capture short-term shifts in dyadic-interactive comportment, long-term symbolic memory traces, and state-change patterns. Used in synergy, these provide a unified diagnostic framework for empirical dynamics in social or memory-reactive dyadic systems.

--

## Current Status

- Core algorithm complete (AI model in python code + Jupyter notebook)
- Visualization and time-domain simulation working
- First-iteration inclusion of FFT spectral analysis
- AMA-formatted paper and LaTeX doc provides full mathematical description

---

## TODO: Integrodifferential Refactor and Model Tier Consolidation

- Refactor the existing hybrid architecture to consolidate all continuous-time dynamics (currently partitioned across the cumulant generating function, fatigue decay, and symbolic response latency) into a unified **integrodifferential equation (IDE)** framework.
- Formalize an **IDE-based state propagation mechanism** capable of simultaneously capturing:
  - Local probabilistic transitions (Markovian and RL-informed)
  - Long-range memory or habituation effects (e.g., affective fatigue)
  - Nonlocal symbolic activations (e.g., Dirac-style or spike-recalled states)
- Design the IDE such that it expresses a **convolution kernel over symbolic input history**, leveraging:
  - Recursive empirical reweighting
  - Stochastic re-entry dynamics
  - And transition shaping via fatigue or signal attenuation
- Replace currently disjoint CGF/PGF/LSM layers with an **integrated 3-tier model**:
  1. **Discrete Tier** — Probabilistic Bayesian state transitions and Q-learning logic
  2. **Continuous Tier** — Time-extended fatigue and symbolic decay (modeled via IDE)
  3. **Latent Tier** — Symbolic representations mapped into parameterized interactive windows (Dirac peaks, symbolic lags)

This refactor aims to reduce representational fragmentation and permit **ODE/PDE-style numerical solving** in future simulation stages, enabling deeper convergence diagnostics and eventual differential identity tracking for symbolic parameterization.

---

## Interactive State Definitions

Let S = {S₀, S₁, S₂, S₃, S₄} represent five discrete behavioral states:

| State | Description |
|-------|-------------|
| S₀ | **State 0** |
| S₁ | **State 1**  |
| S₂ | **State 2**  |
| S₃ | **State 3** |
| S₄ | **State 4** |

---

## Computational Methodology

The model proceeds in the following steps:

1. Identify the **current state** Sₜ

2. Retrieve the **current transition matrix** *P*

The Markov transition matrix *P* is defined as:

| *P*     | S₀   | S₁   | S₂   | S₃   | S₄   |
|--------|------|------|------|------|------|
| **S₀** | p₀₀ | p₀₁ | p₀₂ | p₀₃ | p₀₄ |
| **S₁** | p₁₀ | p₁₁ | p₁₂ | p₁₃ | p₁₄ |
| **S₂** | p₂₀ | p₂₁ | p₂₂ | p₂₃ | p₂₄ |
| **S₃** | p₃₀ | p₃₁ | p₃₂ | p₃₃ | p₃₄ |
| **S₄** | p₄₀ | p₄₁ | p₄₂ | p₄₃ | p₄₄ |

Each row sums to 1:  

$$
\sum_j p_{ij} = 1 \quad \text{for all } i
$$

This matrix evolves based on empirical data and reinforcement adjustments.

3. Compute the **next transition probabilities** using *Q*-learning:

   $$
   Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
   $$
   
   Where:

- α: learning rate
- γ: discount factor
- *r*: reward signal
- *s*′: next predicted state

Define:

   $$
   \Delta Q = r + \gamma \max_{a'} Q(s',a') - Q(s,a)
   $$

This guides transition probabilities based on reinforcement.

4. Apply the **softmax function** to generate the updated transition probabilities:

   $$
   P(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_b \exp(Q(s,b)/\tau)}
   $$


Where:
- τ: temperature parameter controlling exploration (low τ = sharper preference)

**Example**:  

If Q₁₂ = 2.5, Q₁₃ = 1.2, Q₁₄ = 0.8, then:

$$
P_{12} = \frac{\exp(2.5 / \tau)}{\exp(2.5 / \tau) + \exp(1.2 / \tau) + \exp(0.8 / \tau)}
$$

5. Introduce **stochastic variability**

To prevent overfitting and reflect empirical volatility:

$$
dP_{ij} = \beta \left( \frac{\exp(Q_{ij} / \tau)}{\sum_k \exp(Q_{ik} / \tau)} - P_{ij} \right) dt + \sigma dW_{ij}
$$

Where:

- β: transition rate adaptation coefficient  
- σ·dWᵢⱼ: stochastic noise (Wiener process component)

This keeps transitions dynamic and empirically sensitive to external perturbations.

6. Update transition matrix *P* for the next cycle

---

## Belief Update (Bayesian Filtering)

Beliefs are updated with incoming observations:

$$
P(H_i \mid E) = \frac{P(E \mid H_i) P(H_i)}{\sum_j P(E \mid H_j) P(H_j)}
$$

---

## Repository Contents

- `bayes_markov.py`: Bayesian inference module
- `event_modulation.py`: Q-learning and softmax reinforcement
- `state_controller.py`: Hybrid behavioral model combining modules
- `run_simulation.py`: Executes full model loop with multi-step simulation
- `notebooks/affective_fft_model.ipynb`: Visualization + FFT diagnostics
- `test_*.py`: Sanity checks for each module

- 

## License

(C) Copyright 2024–2025 Corey Vincent Zelinski.  
All rights reserved. Reuse, redistribution, or derivative work is prohibited without express permission.

---

## Citation

Formal publication pending. For inquiries, contact **Corey.Zelinski@IEEE.org**.

