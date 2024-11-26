# **Lunar Lander DQN Agent**

This project implements a Deep Q-Network (DQN) to train an agent to land a spacecraft safely in the Lunar Lander environment from OpenAI Gym. The agent learns to navigate the complex physics of the lunar surface by interacting with the environment, receiving rewards for stable landings, and penalties for crashes or poor maneuvers.

## **Project Highlights**
- **Environment**: The agent operates in the continuous state space of the **Lunar Lander** environment, aiming to land between designated flags while minimizing fuel usage and avoiding collisions.
- **Algorithm**: A Deep Q-Network (DQN) is employed, leveraging a neural network to approximate Q-values for action selection.
- **Exploration-Exploitation**: An epsilon-greedy strategy is used to balance exploration (trying new actions) and exploitation (leveraging learned knowledge).
- **Replay Buffer**: Experiences are stored in a buffer and sampled randomly to improve training stability by breaking the correlation between consecutive experiences.
- **Target Network**: A secondary, slowly-updated network stabilizes learning by providing more consistent Q-value targets.

## **Key Components**
1. **DQN Model**: A feed-forward neural network with two hidden layers processes state inputs and predicts Q-values for all possible actions.
2. **Replay Buffer**: Implements experience replay for efficient learning and decorrelation of training samples.
3. **Agent**: Orchestrates interaction with the environment, storage of experiences, and learning through periodic updates of the policy network.
4. **Training**: The agent is trained over multiple episodes, using rewards to improve its policy for achieving stable landings.
5. **Evaluation**: Total rewards are tracked across episodes to monitor the agent's performance and convergence.

## **Project Results**
The agent’s performance is logged across training episodes, with total rewards visualized to track progress. Adjustments to hyperparameters, such as learning rate, exploration rate, and network architecture, may be necessary for optimal convergence.

## **Requirements**
- Python 3.7 or later
- PyTorch
- NumPy
- OpenAI Gym
- Matplotlib

## **How to Run**
1. Clone this repository:
   ```bash
   git clone <repository_url>
