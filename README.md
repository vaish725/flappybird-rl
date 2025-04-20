# AI Flappy Bird with Reinforcement Learning

**Student:** Vaishnavi Kamdi (G48986897)  
**Course:** AI Algorithms  
**Instructor:** Joseph Goldfrank  

---

## 📌 Project Overview

This project implements and compares two reinforcement learning algorithms—**Deep Q-Networks (DQN)** and **Advantage Actor-Critic (A2C)**—to solve the game of *Flappy Bird*. The agent learns an optimal policy to survive as long as possible, with observations restricted to simulate partial observability.

The project is designed to meet the course requirement of solving a **stochastic sequential decision problem** using a feasible and scalable RL-based approach.

---

## 🧠 Algorithms Used

- **Deep Q-Network (DQN)**
- **Advantage Actor-Critic (A2C)**

Implemented using **PyTorch**, with training metrics such as average survival time per episode, convergence rate and total reward over time.

---

**Natural Language Description of State Space**

In the Flappy Bird environment, the agent (bird) must decide whether to "flap" or "not flap" at each timestep to avoid crashing into pipes or the ground. The state space is defined by the variables that impact the agent's decision-making. To simulate partial observability and keep the state compact, the agent only observes:

- The bird’s vertical position
- The bird’s vertical velocity
- The vertical distance between the bird and the center of the next pipe’s gap

The environment is dynamic and stochastic (pipe positions vary and the gravity affects motion) making outcomes of the actions uncertain. Each state leads to a new configuration depending on the action taken and the game physics (e.g., velocity updates due to gravity or flap).

---

**Mathematical Description**

Let:
- 𝑠<sub>𝑡</sub> be the state at time 𝑡
- 𝑎<sub>𝑡</sub> be the action at time 𝑡
- 𝑟<sub>𝑡</sub> be the reward at time 𝑡
- 𝑜<sub>𝑡</sub> be the observation at time 
- 𝑆,𝐴,𝑂 denote the state, action, and observation spaces respectively

**State Space 𝑆**

Each state 𝑠<sub>𝑡</sub>∈𝑆 is a tuple:

<div align="center">
  𝑠<sub>𝑡</sub>=(𝑦<sub>𝑡</sub>,𝑣<sub>𝑡</sub>,Δ𝑦<sub>𝑡</sub>)
</div>

where:

- 𝑅: set of real numbers
- 𝑦<sub>𝑡</sub>∈𝑅: vertical position of the bird
- 𝑣<sub>𝑡</sub>∈𝑅: vertical velocity of the bird
- Δ𝑦<sub>𝑡</sub>∈𝑅: vertical distance between the bird and the center of the next pipe's gap

**Action Space 𝐴**
<div align = "center">
  𝐴={0,1}
</div>

- 0: Do nothing (bird will fall due to gravity)
- 1: Flap (bird moves upward with fixed velocity boost)

**Observation Space 𝑂**

Since the environment is partially observable:
<div align="center">
  𝑜<sub>𝑡</sub>=(𝑦<sub>𝑡</sub>,𝑣<sub>𝑡</sub>,Δ𝑦<sub>𝑡</sub>)
</div>

Note: In this case, the observation is the same as the reduced state representation.

**Transition Function**

The environment transition function 𝑇:𝑆×𝐴→𝑆 is defined by:

<div align = "center">
  𝑠<sub>𝑡+1</sub>=𝑓(𝑠<sub>𝑡</sub>,𝑎<sub>𝑡</sub>)+𝜖
</div>

where 𝑓 encodes game physics (gravity, flap impulse, pipe movement), and 𝜖 represents stochasticity (e.g., random pipe heights).

**Reward Function**

𝑅(𝑠<sub>𝑡</sub>,𝑎<sub>𝑡</sub>)=>
- (+1) if bird stays alive
  
- (-100) if bird hits a pipe or the ground
 
**Policy**

The RL agent aims to learn a policy 𝜋:𝑂→𝐴 that maximizes the expected cumulative reward:

<div align="center">
  𝐸[∑<sup>𝑇</sup><sub>𝑡=0</sub>𝛾<sup>t</sup>𝑟<sub>t</sub>]
</div>

where

- 𝐸: expected value (summation from 𝑡=0 to 𝑇)
- 𝛾∈[0,1] is the discount factor.

---

## 🎮 Environment

The environment is adapted from [sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird).

---
