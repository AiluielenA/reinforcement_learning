from robot_class import Robot, RobotAction
from package_class import Package
from target_class import Target
from obstacle_class import Obstacle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
try:
    register(
        id='warehouse-robot',
        entry_point='environment_class:Environment',
    )
except gym.error.Error:
    pass


class Environment(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, grid_rows=6, grid_cols=7, num_robots=2, num_packages=2, num_targets=2, num_obstacles=3, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_robots = num_robots
        self.num_packages = num_packages
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.max_steps = 100 # time limit
        self.steps_taken = 0  # Initialize step counter
        self.render_mode = render_mode  # Store the render mode

        self.initialize_environment()
        # self.action_space = spaces.Discrete(len(RobotAction))
        self.action_space = spaces.MultiDiscrete([len(RobotAction)] * self.num_robots) # Independent actions spaces

        self.observation_space = spaces.Dict({
            "robots": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_robots, 2), dtype=np.int32),
            "package_positions": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_packages, 2), dtype=np.int32),
            "target_positions": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_targets, 2), dtype=np.int32),
            "packages": spaces.Box(low=0, high=1, shape=(num_packages,), dtype=np.int32),
            "obstacle_positions": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(len(self.obstacles), 2), dtype=np.int32)
        })
   

    def initialize_environment(self):
        self.robots = []
        self.packages = []
        self.targets = []
        self.obstacles = []

        occupied_positions = []

        # Initialize robots
        for _ in range(self.num_robots):
            robot = Robot(self.grid_rows, self.grid_cols)
            self.robots.append(robot)
            occupied_positions.append(robot.position)

        # Initialize packages
        for _ in range(self.num_packages):
            package = Package(self.grid_rows, self.grid_cols, occupied_positions)
            self.packages.append(package)
            occupied_positions.append(package.position)

        # Initialize targets
        for _ in range(self.num_targets):
            target = Target(self.grid_rows, self.grid_cols, occupied_positions)
            self.targets.append(target)
            occupied_positions.append(target.position)

        # Initialize obstacles
        for _ in range(self.num_obstacles):
            obstacle = Obstacle(self.grid_rows, self.grid_cols, occupied_positions)
            self.obstacles.append(obstacle)
            occupied_positions.append(obstacle.position)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # May be redundant
        # Reset robots, packages, targets
        self.initialize_environment()
        
        self.steps_taken = 0  # Initialize step counter
        
        info = {}
        # Construct the observation state:
        obs = self._get_observation()
        
        return obs, info

    def step(self, robot_actions, fps=5):
        
        reward = 0
        terminated = False
        truncated = False
        
        proximity_threshold_far = 5
        proximity_threshold_close = 2
        
        positions = [tuple(robot.position) for robot in self.robots]
        
        for i, robot in enumerate(self.robots):
            action = RobotAction(robot_actions[i])
            print(f"Robot {i}: Action={action}, Position before move={robot.position}")
            
            # Track whether the move is valid
            valid_move = True
            
            # Handle movement or actions
            if action in [RobotAction.LEFT, RobotAction.RIGHT, RobotAction.UP, RobotAction.DOWN]:
                # Predict the new position based on the action
                new_position = robot.position[:]
                if action == RobotAction.LEFT:
                    new_position[1] -= 1
                elif action == RobotAction.RIGHT:
                    new_position[1] += 1
                elif action == RobotAction.UP:
                    new_position[0] -= 1
                elif action == RobotAction.DOWN:
                    new_position[0] += 1
                # Check if the move is out of bounds
                if (new_position[0] < 0 or new_position[0] >= self.grid_rows or
                    new_position[1] < 0 or new_position[1] >= self.grid_cols):
                    valid_move = False
                    reward -= 3  # Penalize for attempting an out-of-bounds move
                    print(f"Robot {i}: Invalid move out of bounds, Position={robot.position}, Action={action}")
                
                # Check if the new position collides with an obstacle
                elif tuple(new_position) in [tuple(obstacle.position) for obstacle in self.obstacles]:
                    valid_move = False
                    reward -= 3  # Penalize for attempting to move into an obstacle
                    print(f"Robot {i}: Invalid move (obstacle collision), Position={robot.position}, Action={action}, Target={new_position}")
                
                else:
                    # If valid, execute the move
                    robot.move(action)
                    print(f"Robot {i}: Position after move={robot.position}")

            # Track robot positions to detect collisions
            positions[i] = tuple(robot.position)
            collision_detected = len(positions) != len(set(positions))       
                
            # Penalize collisions
            if collision_detected:
                reward = -5 
                print("Penalty collision")

            # Check if the robot picks up a package
            # Handle PICK action
            if action == RobotAction.PICK:
                package_found= False
                for package in self.packages:
                    # The robot doesnt have a pkg and it is in pkg location 
                    if robot.position == package.position and not package.picked: 
                        package_found= False
                        if not robot.has_package:
                            # Give reward for picking the pkg
                            robot.has_package = True
                            package.picked = True
                            reward += 5
                            print("reward pick")   
                        else:
                            # Penalize not picking
                            print("penalty not pick")
                            reward -= 5
                    else:
                        # Calculate proximity to a pkg         
                        distance_pkg = self._manhattan_distance(robot.position,package.position) 
                        # Reward based on the distance            
                        if not robot.has_package: 
                            if distance_pkg <= proximity_threshold_close:
                                print("reward pkg proximity close")
                                reward += 2.5    
                            elif distance_pkg > proximity_threshold_close and distance_pkg < proximity_threshold_far:
                                print("reward pkg proximity far")
                                reward += 0.5
                            else:
                                print("reward pkg proximity too far")
                                reward += 0  ## adjust if is too far  
                    
                    if not package_found:
                        reward -= 3  # Penalize trying to pick when no package is present
                        print(f"Robot {i}: Attempted to pick but no package found at position {robot.position}")
                   

            # Check if the robot delivers a package
            if action == RobotAction.DEPOSIT:
                target_found = False
                for target in self.targets:
                    if robot.position == target.position:   
                        target_found = True 
                        if robot.has_package:
                            robot.has_package = False
                            reward += 10
                            terminated = True
                            # Respawn package at a new random position
                            new_package = Package(
                                self.grid_rows,
                                self.grid_cols,
                                [pkg.position for pkg in self.packages] +
                                [robot.position for robot in self.robots] +
                                [target.position for target in self.targets]
                            )
                            self.packages.append(new_package)
                            print(f"New package spawned at {new_package.position}")
                            
                        else:
                        # Penalty for not depositing
                            print("penalty not target deposit")
                            reward -= 10
                    else:
                        # Calculate proximity to a target         
                        distance_target = self._manhattan_distance(robot.position,target.position) 
                        # Reward based on the distance            
                        if robot.has_package: 
                            if distance_target <= proximity_threshold_close:
                                print("reward target proximity close")
                                reward += 2.5
                            elif distance_target > proximity_threshold_close and distance_target < proximity_threshold_far:
                                reward += 0.5
                                print("reward target proximity far")
                            else:
                                print("reward target proximity too far")
                                reward += 0  ## adjust if is too far   
                    if not target_found:
                        reward -= 3  # Penalize trying to deposit when not at a target
                        print(f"Robot {i}: Attempted to deposit but not at a target position")

                            
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            truncated = True
                    
        obs = self._get_observation()
        
        
        info = {}
                    
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment grid."""
        # if self.render_mode != 'human':
        #     return
        grid = [["." for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        i=1
        for robot in self.robots:
            r, c = robot.position
            # grid[r][c] = "R"
            grid[r][c] = f"R{i}"
            i=i+1

        for package in self.packages:
            if not package.picked:
                r, c = package.position
                grid[r][c] = "P"

        for target in self.targets:
            r, c = target.position
            grid[r][c] = "T"
            
        for obstacle in self.obstacles:
            r, c = obstacle.position
            grid[r][c] = "X"  # Represent obstacles with 'X'


        print("\n".join("".join(row) for row in grid))
        print("\n")

    def _get_observation(self):
        """Construct the observation."""
        return {
            "robots": np.array([robot.position for robot in self.robots], dtype=np.int32),
            "package_positions": np.array([pkg.position for pkg in self.packages], dtype=np.int32),
            "target_positions": np.array([tgt.position for tgt in self.targets], dtype=np.int32),
            "packages": np.array([int(pkg.picked) for pkg in self.packages], dtype=np.int32),
            "obstacle_positions": np.array([obstacle.position for obstacle in self.obstacles], dtype=np.int32)
        }

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# # For unit testing
# if __name__=="__main__":
#     env = gym.make('warehouse-robot', render_mode='human')

#     # Use this to check our custom environment
#     # print("Check environment begin")
#     # check_env(env.unwrapped)
#     # print("Check environment end")

#     # Reset environment
#     obs = env.reset()[0]

#     # Take some random actions
#     for i in range(1,20):
#         rand_action = env.action_space.sample()
#         obs, reward, terminated, truncated, _ = env.step(rand_action)
#         env.render()

#         if terminated or truncated:
#             obs, _ = env.reset()


# for testing the movement
if __name__ == "__main__":
    env = gym.make('warehouse-robot', render_mode='human')

    obs, _ = env.reset()
    print("Initial Observation:", obs)

    for step in range(20):  # Limit steps for debugging
        print(f"\nStep {step + 1}")
        rand_action = env.action_space.sample()
        print(f"Random Actions: {rand_action}")

        obs, reward, terminated, truncated, _ = env.step(rand_action)
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        env.render()

        if terminated or truncated:
            print("Episode ended. Resetting environment.")
            obs, _ = env.reset()


        
        