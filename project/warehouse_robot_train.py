import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import environment_class
import robot_class as r


def run_q(episodes, is_training=True, render=False):

    env = gym.make('warehouse-robot', render_mode='human' if render else None)

    if(is_training):

        q = np.zeros(
            (env.unwrapped.grid_rows, env.unwrapped.grid_cols) * (2 * env.unwrapped.num_robots) +
            (env.unwrapped.grid_rows, env.unwrapped.grid_cols) * (env.unwrapped.num_packages) +
            (env.unwrapped.grid_rows, env.unwrapped.grid_cols) * (env.unwrapped.num_targets) +
            (len(r.RobotAction),) * env.unwrapped.num_robots  # Action space for each robot
        )
    else:
        # If testing, load Q Table from file.
        f = open('warehouse_solution.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # Hyperparameters
    learning_rate_a = 0.9   
    discount_factor_g = 0.9 
    epsilon = 1             
    epsilon_decay = 0.99    
    min_epsilon = 0.1      

    # steps_per_episode = np.zeros(episodes)
    step_count=0

    steps_per_episode = []

    for episode in range(episodes):
        if render:
            print(f'Episode {episode + 1}')

        # Reset environment at the beginning of an episode
        state, _ = env.reset()
        terminated = False
        step_count = 0

        while not terminated:
            # Select actions for each robot using epsilon-greedy
            actions = []
            for robot_idx in range(env.unwrapped.num_robots):
                if is_training and random.random() < epsilon:
                    action = random.choice(list(r.RobotAction))
                else:

                    q_state_idx = tuple(state["robots"].flatten()) + tuple(state["package_positions"].flatten()) + tuple(state["target_positions"].flatten())
                    robot_action_idx = slice(robot_idx * len(r.RobotAction), (robot_idx + 1) * len(r.RobotAction))
                    action = np.argmax(q[q_state_idx + robot_action_idx])
                actions.append(action)


            new_state, reward, terminated, truncated, _ = env.step(actions)
            if truncated:
                terminated = True  

            if is_training:
                # Update Q-Table using the Bellman equation
                q_state_action_idx = (
                    tuple(state["robots"].flatten()) +
                    tuple(state["package_positions"].flatten()) +
                    tuple(state["target_positions"].flatten()) +
                    tuple(actions)
                )
                q_new_state_idx = (
                    tuple(new_state["robots"].flatten()) +
                    tuple(new_state["package_positions"].flatten()) +
                    tuple(new_state["target_positions"].flatten())
                )
                q[q_state_action_idx] = q[q_state_action_idx] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[q_new_state_idx]) - q[q_state_action_idx]
                )

            # Update current state
            state = new_state
            step_count += 1

        steps_per_episode.append(step_count)

        # Epsilon decay
        epsilon = max(epsilon * epsilon_decay, min_epsilon)

    env.close()

    # Plot steps per episode (with moving average)
    moving_avg_steps = np.convolve(steps_per_episode, np.ones(100)/100, mode='valid')
    plt.plot(moving_avg_steps)
    plt.xlabel('Episodes')
    plt.ylabel('Average Steps')
    plt.title('Q-Learning Training Progress')
    plt.savefig('warehouse_training_progress.png')

    # Save Q-Table after training
    if is_training:
        with open('warehouse_solution.pkl', 'wb') as f:
            pickle.dump(q, f)

    print("Training complete. Q-Table saved as 'warehouse_solution.pkl'.")

if __name__ == '__main__':

    # Train/test using Q-Learning
    run_q(100, is_training=True, render=False)
    run_q(1, is_training=False, render=True)
