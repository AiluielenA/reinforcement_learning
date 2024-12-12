import gymnasium as gym
from tensorflow.keras import models, layers, optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from collections import deque
import os
from time import strftime
import csv
from environment_class import Environment
from robot_class import Robot, RobotAction
from renderer_class import Renderer


class Logger:
    def __init__(self):
        # Create a unique log directory
        self.log_dir = f"log_{strftime('%Y_%m_%d-%H_%M_%S')}"
        os.makedirs(self.log_dir, exist_ok=True)

        # Paths for logging files
        self.rewards_log_path = os.path.join(self.log_dir, "rewards.csv")
        self.transitions_log_path = os.path.join(self.log_dir, "transitions.csv")
        self.evaluation_log_path = os.path.join(self.log_dir, "evaluation_rewards.csv")
        self.model_save_path = os.path.join(self.log_dir, "trained_model.keras")

        # Initialize the CSV log files
        with open(self.rewards_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "reward"])

        with open(self.transitions_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "state", "action", "next-state", "reward", "done", "truncated"])

        with open(self.evaluation_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "reward"])

    def log_reward(self, episode, reward):
        with open(self.rewards_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, reward])
            
    def log_eval_reward(self, episode, reward):
        with open(self.rewards_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, reward])

    def log_transition(self, episode, state, action, next_state, reward, done, truncated):
        with open(self.transitions_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, state.tolist(), action, next_state.tolist(), reward, done, truncated])

    def save_model(self, model):
        model.save(self.model_save_path)
        
class ExponentialDecay:
    def __init__(self, initial_epsilon, decay_rate, min_epsilon):
        self.initial_epsilon = initial_epsilon
        self.current_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    def get_value(self):
        return max(self.current_epsilon, self.min_epsilon)

    def update(self):
        if self.current_epsilon > self.min_epsilon:
            self.current_epsilon *= self.decay_rate


class DQN:
    def __init__(
            self,
            main_model,
            target_model=None,
            epsilon=None,
            gamma=None,
            action_space=None,
            max_buffer=50000
    ) -> None:
        self.action_space = list(range(len(RobotAction)))  # Actions from 0 to 6
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer_replay = deque(maxlen=max_buffer)
        self.main_model = main_model
        self.target_model = target_model
        self.loss = tf.keras.losses.Huber()
        self.optimizer = optimizers.Adam(learning_rate=0.001)

    def select_action(self, state, num_robots=2):
        if random.random() < self.epsilon.get_value():
            self.epsilon.update()
            return [random.choice(self.action_space) for _ in range(num_robots)]
        else:
            return self.select_best_action(state, num_robots)

    def select_best_action(self, state, num_robots):
        
        print("state", state)
        # Predict Q-values for all actions
        q_values = self.main_model.predict(state[np.newaxis], verbose=0)[0]
        print("Predicted Q-values:", q_values)
        
        # Reshape Q-values: each robot has its own set of Q-values
        num_actions_per_robot = len(RobotAction)
        
        assert q_values.size == num_robots * num_actions_per_robot, \
        f"Q-values size mismatch. Expected {num_robots * num_actions_per_robot}, got {q_values.size}"
    
        q_values_per_robot = np.reshape(q_values, (num_robots, num_actions_per_robot))

        # Select the action with the highest Q-value for each robot
        actions = [np.argmax(q_values_robot) for q_values_robot in q_values_per_robot]
        print("Selected actions:", actions)
        
        return actions      

    def store_transition(self, state, action, next_state, reward, done, truncated):
        self.buffer_replay.append((state, action, next_state, reward, done, truncated))

    def sample_minibatch(self, batch_size):
        indices = np.random.choice(len(self.buffer_replay), size=batch_size)
        return [self.buffer_replay[idx] for idx in indices]

    def train(self, batch_size=32):
        if len(self.buffer_replay) < batch_size:
            return
        
        minibatch = self.sample_minibatch(batch_size)
        states, actions, next_states, rewards, dones, truncates = map(np.array, zip(*minibatch))
        with tf.GradientTape() as tape:
            next_q_values = self.target_model(next_states)
            q_targets = np.max(next_q_values, axis=-1)
            q_targets = rewards + (1 - (dones | truncates)) * tf.squeeze(q_targets) * self.gamma
            current_q_values = self.main_model(states)
            
            num_actions_per_robot = len(RobotAction)
            actions_flattened = actions[:, 0] + actions[:, 1] * num_actions_per_robot
            actions_flattened = tf.cast(actions_flattened, tf.int32)
            actions_flattened = tf.minimum(actions_flattened, num_actions_per_robot * 2 - 1)
            indices = tf.stack([tf.range(len(actions_flattened), dtype=tf.int32), actions_flattened], axis=1)
            
            q_values_selected = tf.gather_nd(current_q_values, indices)
            loss = self.loss(q_values_selected, q_targets)
            
        value_grads = tape.gradient(loss, self.main_model.trainable_variables)
        self.optimizer.apply_gradients(zip(value_grads, self.main_model.trainable_variables))

    def update_target_model(self, weights=None):
        if weights is not None:
            self.target_model.set_weights(weights)
        else:
            self.target_model.set_weights(self.main_model.get_weights())


class CreateNetwork(tf.keras.Model):
    def __init__(self, output_dim):
        super(CreateNetwork, self).__init__()
        self.dens1 = layers.Dense(64, activation=tf.keras.activations.leaky_relu)
        self.dens2 = layers.Dense(64, activation='relu')
        self.dens3 = layers.Dense(output_dim)
        
    def call(self, state):
        x = self.dens1(state)
        x = self.dens2(x)
        x = self.dens3(x)
        return x
    



def train_agent(env, agent, episodes, update_target_freq, logger=None, renderer=None):
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = env.flatten_state(state)
        total_reward = 0
        terminated = False
        steps = 0

        while not terminated  and steps < 10000:
            steps +=1
            
            # Select actions for each robot
            action = agent.select_action(state)
            
            # Step the environment
            next_state, reward, done, truncated, info = env.step(action)
            next_state = env.flatten_state(next_state)
            
            # Store the transition and train
            agent.store_transition(state, action, next_state, reward, done, truncated)
            agent.train(batch_size=32)
            
            # Update state and accumulate rewards
            state = next_state
            total_reward += reward
            
            # Render the environment
            if renderer:
                renderer.render()
            
            terminated = done
        
        # Log rewards and update target network periodically
        logger.log_reward(episode, total_reward)
        if episode % update_target_freq == 0:
            agent.update_target_model()        
            
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon.get_value():.2f}, Steps: {steps}")
    print(f'AVG reward: {np.mean(rewards)}')
    return rewards



def evaluate_agent(env, agent, episodes, logger=None, renderer=None):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        steps = 0
        
        while not terminated:
            actions = [agent.select_best_action(state) for _ in env.robots]
            # action = agent.select_best_action(state)
            next_state, reward, done, truncated, info = env.step(actions)
            next_state = env.flatten_state(next_state)
            
            state = next_state
            total_reward += reward
            
            # Render the environment
            if renderer:
                renderer.render()
            
            terminated = done
                
            
        # Log evaluation rewards
            logger.log_eval_reward(episode, total_reward)
            rewards.append(total_reward)
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

    env.close()

    return rewards


def plot_rewards(root_folder, rewards, title, save_fig=False):
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(title)
    if save_fig:
        plt.savefig(root_folder + "/" + title + '_' + strftime("%Y_%m_%d_%H_%M_%S") +'_' + '.png')
    plt.legend()
    plt.show()
    

# start training 
if __name__ == "__main__":
    # env = gym.make('LunarLander-v3')
    env = Environment(grid_rows=7, grid_cols=7, num_robots=2, num_packages=2, num_targets=2, num_obstacles=4, num_charger=2)
    renderer = Renderer(env, cell_size=64, fps=10)
    
    # Define state and action sizes
    state_dim = env.flatten_state(env._get_observation()).shape[0]
    action_dim = len(RobotAction) * 2

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    
    # Create DQN and supporting components
    main_network = CreateNetwork(action_dim)
    target_network = CreateNetwork(action_dim)
    
    # Ensure both models are built and initialized
    dummy_input = tf.zeros((1, state_dim))  # Dummy input with state dimension
    main_network(dummy_input)
    target_network(dummy_input)
    
    epsilon_schedule = ExponentialDecay(initial_epsilon=1.0, decay_rate=0.995, min_epsilon=0.01)

    agent = DQN(
        main_model=main_network,
        target_model=target_network,
        epsilon=epsilon_schedule,
        gamma=0.99,
        action_space=list(range(len(RobotAction)))
    )
    logger = Logger()
    
    # Training parameters
    training_episodes = 1000
    update_target_freq = 20

    training_rewards = train_agent(env, agent, training_episodes, update_target_freq, logger, renderer)
    plot_rewards(logger.log_dir, training_rewards, "Training Rewards", True)

    # Evaluate the agent
    evaluation_episodes = 100
    evaluation_rewards = evaluate_agent(env, agent, evaluation_episodes, logger, renderer)
    plot_rewards(logger.log_dir, evaluation_rewards, "Evaluation Rewards", True)