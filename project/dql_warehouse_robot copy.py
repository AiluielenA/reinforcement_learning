from tensorflow.keras import layers, optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import os
from time import strftime
import csv
from environment_class import Environment
from robot import RobotAction
from renderer import Renderer
import time

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
        self.loss_log_path = os.path.join(self.log_dir, "loss.csv")

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

        with open(self.loss_log_path, mode='w', newline='') as file:  # Initialize the loss log
            writer = csv.writer(file)
            writer.writerow(["step", "loss"])

    def log_loss(self, step, loss):
        with open(self.loss_log_path, mode='a', newline='') as file:  # Append loss data
            writer = csv.writer(file)
            writer.writerow([step, loss])

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
            max_buffer=30000
    ) -> None:
        self.action_space = list(range(len(RobotAction)))
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
        
        # Predict Q-values for all actions
        q_values = self.main_model.predict(state[np.newaxis], verbose=0)[0]
        
        # Reshape Q-values: each robot has its own set of Q-values
        num_actions_per_robot = len(RobotAction)
        
        assert q_values.size == num_robots * num_actions_per_robot, \
        f"Q-values size mismatch. Expected {num_robots * num_actions_per_robot}, got {q_values.size}"
    
        q_values_per_robot = np.reshape(q_values, (num_robots, num_actions_per_robot))

        # Select the action with the highest Q-value for each robot
        actions = [np.argmax(q_values_robot) for q_values_robot in q_values_per_robot]
        
        return actions      

    def store_transition(self, state, action, next_state, reward, done, truncated):
        self.buffer_replay.append((state, action, next_state, reward, done, truncated))

    def sample_minibatch(self, batch_size):
        indices = np.random.choice(len(self.buffer_replay), size=batch_size)
        return [self.buffer_replay[idx] for idx in indices]

    def train(self, batch_size=32, logger=None, step=0):
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

        if logger:
            logger.log_loss(step, float(loss.numpy()))

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

    monitoring_log_path = os.path.join(logger.log_dir, "training_progress.txt")
    with open(monitoring_log_path, "w") as monitor_file:
        monitor_file.write("Episode, Total Reward, Epsilon, Steps\n")  # Write headers


    for episode in range(episodes):
        state, _ = env.reset()
        state = env.flatten_state(state)
        total_reward = 0
        terminated = False
        step = 0

        while not terminated: #and step < 1000:
            step +=1
            
            # Select actions for each robot
            action = agent.select_action(state)
            
            # Step the environment
            next_state, reward, done, truncated, info = env.step(action)
            next_state = env.flatten_state(next_state)
            
            # Store the transition and train
            agent.store_transition(state, action, next_state, reward, done, truncated)
            agent.train(batch_size=32, logger=logger, step=step) 
            
            # Update state and accumulate rewards
            state = next_state
            total_reward += reward
            
            # Render the environment
            if renderer:
                start_time = time.time()
                renderer.render()
                print(f"Render time: {time.time() - start_time:.4f}s")
            
            terminated = done
            time.sleep(0.01)  # Add a small delay to avoid overloading the CPU
        
        logger.log_reward(episode, total_reward)
        if episode % update_target_freq == 0:
            agent.update_target_model()   

        with open(monitoring_log_path, "a") as monitor_file:
            monitor_file.write(f"{episode + 1}, {total_reward:.2f}, {agent.epsilon.get_value():.2f}, {step}\n")
     
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon.get_value():.2f}, Steps: {step}")
    print(f'AVG reward: {np.mean(rewards)}')
    return rewards



def evaluate_agent(env, agent, episodes, logger=None, renderer=None):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        
        while not terminated:
            actions = [agent.select_best_action(state) for _ in env.robots]
            next_state, reward, done, truncated, info = env.step(actions)
            next_state = env.flatten_state(next_state)
            
            state = next_state
            total_reward += reward
            
            if renderer:
                renderer.render()
            
            terminated = done
                            
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

def plot_loss(log_dir):
    loss_file = os.path.join(log_dir, "loss.csv")
    steps, losses = [], []

    # Read loss data from the CSV file
    with open(loss_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            steps.append(int(row[0]))
            losses.append(float(row[1]))

    plt.plot(steps, losses, label='Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()
  

# start training 
if __name__ == "__main__":
    env = Environment(grid_rows=14, grid_cols=14, num_robots=2, num_packages=4, num_targets=4, num_obstacles=8, num_charger=4)
    renderer = Renderer(env, cell_size=64, fps=5)

    # Define state and action sizes
    state_dim = env.flatten_state(env._get_observation()).shape[0]
    action_dim = len(RobotAction) * 2

    # Create DQN and supporting components
    main_network = CreateNetwork(action_dim)
    target_network = CreateNetwork(action_dim)
    
    # Ensure both models are built and initialized
    state_input = tf.zeros((1, state_dim))  
    main_network(state_input)
    target_network(state_input)
    
    epsilon_schedule = ExponentialDecay(initial_epsilon=1.0, decay_rate=0.99, min_epsilon=0.01)

    agent = DQN(
        main_model=main_network,
        target_model=target_network,
        epsilon=epsilon_schedule,
        gamma=0.9,
        action_space=list(range(len(RobotAction)))
    )
    logger = Logger()
    
    # Training parameters
    training_episodes = 600
    update_target_freq = 20

    training_rewards = train_agent(env, agent, training_episodes, update_target_freq, logger, renderer)
    plot_rewards(logger.log_dir, training_rewards, "Training Rewards", True)


    plot_loss(logger.log_dir)

    # Evaluate the agent
    evaluation_episodes = 100
    evaluation_rewards = evaluate_agent(env, agent, evaluation_episodes, logger, renderer)
    plot_rewards(logger.log_dir, evaluation_rewards, "Evaluation Rewards", True)