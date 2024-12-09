from robot_class import Robot, RobotAction
from package_class import Package
from target_class import Target
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
    metadata = {'render_modes': ['human']}

    def __init__(self, grid_rows=4, grid_cols=5, num_robots=2, num_packages=2, num_targets=2, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_robots = num_robots
        self.num_packages = num_packages
        self.num_targets = num_targets
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
            "packages": spaces.Box(low=0, high=1, shape=(num_packages,), dtype=np.int32)
        })     

    def initialize_environment(self):
        self.robots = []
        self.packages = []
        self.targets = []
        self.robots = []
        occupied_positions = [robot.position for robot in self.robots]
        # Initialize robots
        for _ in range(self.num_robots):
            self.robots.append(Robot(self.grid_rows, self.grid_cols))

        # Initialize packages and targets, ensuring no overlaps
        occupied_positions = [robot.position for robot in self.robots]
        for _ in range(self.num_packages):
            package = Package(self.grid_rows, self.grid_cols, occupied_positions)
            self.packages.append(package)
            occupied_positions.append(package.position)

        for _ in range(self.num_targets):
            target = Target(self.grid_rows, self.grid_cols, occupied_positions)
            self.targets.append(target)
            occupied_positions.append(target.position)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # May be redundant
        # Reset robots, packages, targets
        self.initialize_environment()
        
        self.steps_taken = 0  # Initialize step counter
        
        info = {}
        # Construct the observation state:
        obs = self._get_observation()
        
        return obs, info

    def step(self, robot_actions):
        
        reward = 0
        terminated = False
        truncated = False
        
        proximity_threshold_far = 5
        proximity_threshold_close = 2
        
        for i, robot in enumerate(self.robots):
            action = RobotAction(robot_actions[i])

            if action in [RobotAction.LEFT, RobotAction.RIGHT, RobotAction.UP, RobotAction.DOWN]:
                robot.move(action)
            elif action == RobotAction.PICK:
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
                # Check for depositing up a package
                for target in self.targets:
                    if robot.position == target.position and robot.has_package:
                        if not target.occupied:  # Check if the target is unoccupied
                            robot.has_package = False
                            target.occupied = True
                            reward += 10  # Reward for delivering the package
                            terminated = True
                            break  # Only one robot can deposit
                        else:
                            reward -= 5  # Penalize depositing on an already occupied target


        # Track robot positions to detect collisions (MOVED OUTSIDE THE LOOP TO ENSURE BOTH ROBOTS TAKE AN ACTION FIRST)
        positions = [tuple(robot.position) for robot in self.robots]
        if len(positions) != len(set(positions)):
            reward -= 5  # Penalize collisions

    
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

        for robot in self.robots:
            r, c = robot.position
            grid[r][c] = "R"

        for package in self.packages:
            if not package.picked:
                r, c = package.position
                grid[r][c] = "P"

        for target in self.targets:
            r, c = target.position
            grid[r][c] = "T"

        print("\n".join("".join(row) for row in grid))
        print("\n")

    def _get_observation(self):
        """Construct the observation."""
        return {
            "robots": np.array([robot.position for robot in self.robots]),
            "package_positions": np.array([pkg.position for pkg in self.packages]),
            "target_positions": np.array([tgt.position for tgt in self.targets]),
            "packages": np.array([int(pkg.picked) for pkg in self.packages])
        }
    
        
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# For unit testing
if __name__=="__main__":
    env = gym.make('warehouse-robot', render_mode='human')

    # Use this to check our custom environment
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    for i in range(1,20):
        rand_action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(rand_action)
        env.render()

        if terminated or truncated:
            obs, _ = env.reset()
        
        