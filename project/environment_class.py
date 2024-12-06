import robot_class as Robot
import package_class as Package
import target_class as Target
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='warehouse-robot',                               
    entry_point='environment_class:Environment',
)

class Environment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_rows=4, grid_cols=5, num_robots=2, num_packages=2, num_targets=2):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_robots = num_robots
        self.num_packages = num_packages
        self.num_targets = num_targets
        self.max_steps = 100 # time limit
        self.steps_taken = 0  # Initialize step counter

        self.initialize_environment()

        # self.action_space = spaces.Discrete(len(Robot.RobotAction))
        self.action_space = spaces.MultiDiscrete([len(Robot.RobotAction)] * self.num_robots) # Independent actions spaces

        self.observation_space = spaces.Dict({
            "robots": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_robots, 2), dtype=np.int32),
            "package_positions": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_packages, 2), dtype=np.int32),
            "target_positions": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(targets, 2), dtype=np.int32),
            "packages": spaces.Box(low=0, high=1, shape=(num_packages,), dtype=np.int32)
        })     

    def initialize_environment(self):
        self.robots = []
        self.packages = []
        self.targets = []
        self.robots = [Robot(self.grid_rows, self.grid_cols) for _ in range(self.num_robots)]
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

        for _ in range(self.num_targetstargets):
            target = Target(self.grid_rows, self.grid_cols, occupied_positions)
            self.targets.append(target)
            occupied_positions.append(target.position)

    def reset(self, seed=None):
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
        
        # Track robot positions to detect collisions
        positions = [tuple(robot.position) for robot in self.robots]
        collision_detected = len(positions) != len(set(positions))   

        for i, robot in enumerate(self.robots):
            robot.move(robot_actions[i])

            # Penalize collisions
            if collision_detected:
                reward = -5        
            # Check if the robot picks up a package
            
            # The robot doesnt have a pkg and it is in pkg location
                # If action == pick -> get a reward
                # else penalize
            # If robot has a pkg and is on pkg location
                # give a zero reward
            # If robot is in target location and has a pkg
                # If action == deposit -> get reward
                # else penalize
            # Give rewards based on robots proximity to pkg and target
            for package in self.packages:
                if not package.picked and robot.position == package.position:
                    robot.has_package = True
                    package.picked = True
                    reward = 5

            # Check if the robot delivers a package
            for target in self.targets:
                if robot.has_package and robot.position == target.position:
                    robot.has_package = False
                    reward = 10
                    terminated = True
            
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                truncated = True
                    
        obs = self._get_observation()
        
        info = {}
                    
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment grid."""
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

# For unit testing
if __name__=="__main__":
    env = gym.make('warehouse-robot-v0', render_mode='human')

    # Use this to check our custom environment
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]