# **Reinforcement Learning Assignment 2 - Lunar Lander DQN Agent**

This project implements a Deep Q-Network (DQN) to train an agent to control a lunar lander, aiming to land it safely on the lunar surface while minimizing
fuel consumption. The agent learns to navigate by interacting with the environment, receiving rewards for stable landings and penalties for crashes.

## **Key Components**
- **Environment**: The agent operates in the continuous state space of the **Lunar Lander** environment from OpenAI Gym, aiming to land between designated flags while minimizing fuel usage and avoiding collisions.
- **DQN Model**: A feed-forward neural network with two hidden layers processes state inputs and predicts Q-values for all possible actions.
- **Target Network**: A secondary, periodically-updated network stabilizes learning by providing more stationary Q-value targets.
- **Replay Buffer**: Experiences are stored in a buffer and sampled randomly to improve training stability by breaking the correlation between consecutive experiences.
- **Agent**: Syncronizes interaction with the environment, storage of experiences, and learning through periodic updates of the target network.
- **Epsilon-Greedy Exploration**: An epsilon-greedy strategy is used to balance exploration (trying new actions) and exploitation (take advantage of learned knowledge).
