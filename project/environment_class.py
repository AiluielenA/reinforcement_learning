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
        
        robot_position = []
        
        for i, robot in enumerate(self.robots):
            action = RobotAction(robot_actions[i])

            if action in [RobotAction.LEFT, RobotAction.RIGHT, RobotAction.UP, RobotAction.DOWN]:
                robot_position.append(tuple(robot.move(action)))
            elif action == RobotAction.PICK:
                if robot.has_package:
                    reward -= 1  # Penalize trying to pick while holding a package            
                # Check for picking up a package (First Come First Served policy) 
                for package in self.packages:
                    if robot.position == package.position and not package.picked:
                        robot.has_package = True
                        package.picked = True
                        reward += 5  # Reward for picking up a package
                        break  # Stop checking once the package is picked   
                    elif robot.position == package.position and package.picked:
                        reward -= 2.5  # Penalize trying to pick an unavailable package                      
            elif action == RobotAction.DEPOSIT:
                if not robot.has_package:
                    reward -= 2  # Penalize trying to deposit without a package
                # Check for depositing up a package
                for target in self.targets:
                    if robot.position == target.position and robot.has_package:
                        if not target.occupied:  # Check if the target is unoccupied
                            robot.has_package = False
                            target.occupied = True
                            reward += 10  # Reward for delivering the package
                            terminated = all(target.occupied for target in self.targets) # cooperative termination
                            break  # Only one robot can deposit
                        else:
                            reward -= 5  # Penalize depositing on an already occupied target


        # Track robot positions to detect collisions (MOVED OUTSIDE THE LOOP TO ENSURE BOTH ROBOTS TAKE AN ACTION FIRST)
        # positions = [tuple(robot.position) for robot in self.robots]
        if len(robot_position) != len(set(robot_position)):
            reward -= 5  # Penalize collisions
            
        else:
            for rob_pos in robot_position:
                if rob_pos in [tuple(obstacle.position) for obstacle in self.obstacles]:
                    reward -= 3  # Penalize for attempting to move into an obstacle
                    print(f"Robot {i}: Invalid move (obstacle collision), Position={robot.position}, Action={action}, Target={rob_pos}")
                

        # Penalize robots for being in close proximity to each other
        for i, robot1 in enumerate(self.robots):
            for j, robot2 in enumerate(self.robots):
                if i != j and self._manhattan_distance(robot1.position, robot2.position) == 2:
                    reward -= 2  # Penalize close proximity
    
        for robot in self.robots:
            # Check if the robot picks up a package
            if not robot.has_package:
                for package in self.packages:
                    if not package.picked:
                        # Calculate proximity to a package
                        distance_pkg = self._manhattan_distance(robot.position, package.position)
                        # Reward based on the distance 
                        if distance_pkg < proximity_threshold_close:
                            reward += 2.5
                        elif proximity_threshold_close <= distance_pkg < proximity_threshold_far:
                            reward += 0.5
                        else:
                            reward -= 0.5  ## adjust if is too far
            else:
                for target in self.targets:
                    distance_target = self._manhattan_distance(robot.position, target.position)
                    if distance_target < proximity_threshold_close:
                        reward += 2.5
                    elif proximity_threshold_close <= distance_target < proximity_threshold_far:
                        reward += 0.5
                    else:
                        reward -= 0.5  ## adjust if is too far                

        # Penalize idling (no action)
        if all(action not in [RobotAction.PICK, RobotAction.DEPOSIT] for action in robot_actions):
            reward -= 1

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


        
        