# Autonomous Robot Navigation using Deep Reinforcement Learning

## Overview

This project explores autonomous robot navigation using Deep Reinforcement Learning (DRL). The objective is to train intelligent agents capable of navigating through obstacle-rich environments while efficiently reaching a designated goal location.

The project utilizes Deep Q-Networks (DQN) and Soft Actor-Critic (SAC) algorithms implemented using Stable-Baselines3 and Gymnasium. Custom environments, reward shaping strategies, and policy evaluation pipelines were developed to analyze and improve agent performance.

---

## Features

* Custom Gymnasium-based robot navigation environment
* Continuous 2D navigation space with randomized obstacles
* Goal-directed autonomous navigation
* Deep Q-Network (DQN) implementation
* Soft Actor-Critic (SAC) implementation
* Reward shaping for faster convergence
* Automated policy evaluation and model checkpointing
* Trajectory visualization and performance monitoring
* Training reward and testing reward logging

---

## Technologies Used

* Python
* Gymnasium (OpenAI Gym)
* Stable-Baselines3
* PyTorch
* NumPy
* Matplotlib

---

## Environment Description

The environment simulates a robot equipped with localization and sensing capabilities navigating through a 2D space.

### Objectives

* Reach the goal location
* Avoid collisions with obstacles
* Minimize unnecessary movement
* Maximize cumulative reward

### Challenges

* Random obstacle placement
* Continuous state space
* Long-horizon decision making
* Exploration versus exploitation trade-off

---

## Reinforcement Learning Algorithms

### Deep Q-Network (DQN)

Used for learning optimal navigation policies through value-function approximation.

Key concepts:

* Experience Replay
* Target Networks
* Epsilon-Greedy Exploration

### Soft Actor-Critic (SAC)

Used for continuous control and policy optimization.

Key concepts:

* Actor-Critic Architecture
* Entropy Regularization
* Off-Policy Learning

---

## Reward Design

Reward shaping was incorporated to accelerate learning.

Examples include:

* Positive reward for reaching the goal
* Penalty for collisions
* Penalty for inefficient movement
* Intermediate rewards for progress toward the goal

The reward design encourages safe and efficient navigation behavior.

---

## Training Pipeline

1. Initialize custom environment
2. Train DRL agent (DQN or SAC)
3. Periodically evaluate policy performance
4. Save best-performing model checkpoints
5. Log training and testing rewards
6. Visualize learned trajectories

---

## Results

The trained agents successfully learned:

* Goal-directed navigation
* Obstacle avoidance
* Efficient path planning
* Stable policy behavior across multiple scenarios

Performance was evaluated using:

* Cumulative reward
* Success rate
* Trajectory quality
* Policy stability

---

## Repository Structure

```text
.
├── RobotNavigation.py
├── training1.py
├── training2.py
├── training3.py
├── training4.py
├── visualize3.py
├── visualize4.py
├── MODEL3.zip
├── MODEL4.zip
├── training_log.npy
├── testing_log.npy
└── README.md
```

---

## Future Improvements

* Multi-agent navigation
* Dynamic obstacle avoidance
* Curriculum learning
* PPO and TD3 comparisons
* Real-world robot deployment
* Sensor noise simulation

---

## Author

Venkat Akshay Grandhi

B.Tech Artificial Intelligence
Mahindra University

---

## License

This project was developed for educational and research purposes.
