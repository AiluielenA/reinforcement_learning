from robot_class import Robot, RobotAction
from package_class import Package
from target_class import Target
from obstacle_class import Obstacle
from charger_class import Charger
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

    def __init__(self, grid_rows=6, grid_cols=7, num_robots=2, num_packages=2, num_targets=2, num_obstacles=3, num_charger=2, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_robots = num_robots
        self.num_packages = num_packages
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.num_charger = num_charger
        self.max_steps = 100 # time limit
        self.steps_taken = 0  # Initialize step counter
        self.render_mode = render_mode  # Store the render mode
        self.terminated = False

        self.initialize_environment()
        # self.action_space = spaces.Discrete(len(RobotAction))
        self.action_space = spaces.MultiDiscrete([len(RobotAction)] * self.num_robots) # Independent actions spaces

        self.observation_space = spaces.Dict({
            "robots": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_robots, 2), dtype=np.int32),
            "package_positions": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_packages, 2), dtype=np.int32),
            "target_positions": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_targets, 2), dtype=np.int32),
            "packages": spaces.Box(low=0, high=1, shape=(num_packages,), dtype=np.int32),
            "obstacle_positions": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(len(self.obstacles), 2), dtype=np.int32),
            "charger": spaces.Box(low=0, high=max(grid_rows, grid_cols) - 1, shape=(num_charger, 2), dtype=np.int32),
        })
   

    def initialize_environment(self):
        self.robots = []
        self.packages = []
        self.targets = []
        self.obstacles = []
        self.charger = []

        occupied_positions = []

        # Initialize obstacles first
        for _ in range(self.num_obstacles):
            obstacle = Obstacle(self.grid_rows, self.grid_cols, occupied_positions)
            self.obstacles.append(obstacle)
            occupied_positions.append(obstacle.position)

        # Initialize robots
        for _ in range(self.num_robots):
            robot = Robot(self.grid_rows, self.grid_cols, occupied_positions)
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

        # Initialize charging stations
        for _ in range(self.num_charger):
            charger = Charger(self.grid_rows, self.grid_cols, occupied_positions)
            self.charger.append(charger)
            occupied_positions.append(charger)            

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
        truncated = False
        
        proximity_threshold_far = 10
        proximity_threshold_close = 5
        energy_threshold = 40
        
        robot_position = []
        obstacle_positions = [obstacle.position for obstacle in self.obstacles]
        robots_sorted_by_energy = sorted(self.robots, key=lambda r: r.energy)

        for i, robot in enumerate(robots_sorted_by_energy):
            action = RobotAction(robot_actions[self.robots.index(robot)])

            # Consume energy for any action
            robot.energy -= 1
            if robot.energy <= 10:
                reward -= 10  # Penalize robot for running out of energy
                continue

            # Handle charging logic
            if robot.energy <= energy_threshold:
                # Find the closest available charger
                closest_station = min(
                    [ch for ch in self.charger if not ch.occupied],
                    key=lambda ch: self._manhattan_distance(robot.position, ch.position),
                    default=None
                )

                if closest_station:
                    distance_to_station = self._manhattan_distance(robot.position, closest_station.position)

                    if distance_to_station <= proximity_threshold_close:
                        reward += 5  # Close proximity reward
                    elif distance_to_station <= proximity_threshold_far:
                        reward += 2  # Medium proximity reward
                    else:
                        reward -= 1  # Too far, minimal movement

            if action == RobotAction.CHARGE:
                # Check for depositing up a package
                for charger in self.charger:
                    if robot.position == charger.position and robot.energy <= energy_threshold:
                        if not charger.occupied:  # Check if the charger is unoccupied
                            robot.energy = 100 # Recharge
                            charger.occupied = True
                            reward += 10 
                            break  # Only one robot can deposit
                        else:
                            reward -= 5  # Penalize charging on an already occupied target 

            elif action in [RobotAction.LEFT, RobotAction.RIGHT, RobotAction.UP, RobotAction.DOWN]:
                # new potential position
                collision = robot.move(action, obstacle_positions)
                
                # Check for obstacle collision
                if collision:
                    reward -= 3  # Penalize obstacle collision and remain on the same state
                    print(f"Robot {i}: Invalid move (obstacle collision), Position={robot.position}, Action={action}, Targe{collision}")
                    continue  # Skip further processing for this robot's action to prioritize penalty over other moves
                else:
                    robot_position.append(robot.position)                       

            elif action == RobotAction.PICK:
                if robot.has_package:
                    reward -= 1  # Penalize trying to pick while holding a package  
                else:          
                    # Check for picking up a package (First Come First Served policy) 
                    for package in self.packages:
                        if robot.position == package.position and not package.picked:
                            robot.has_package = True
                            package.picked = True
                            reward += 5  # Reward for picking up a package
                            break  # Stop checking once the package is picked   
                        elif robot.position == package.position and package.picked:
                            reward -= 5  # Penalize trying to pick an unavailable package        

            elif action == RobotAction.DEPOSIT:
                if not robot.has_package:
                    reward -= 2  # Penalize trying to deposit without a package
                else:
                    # Check for depositing up a package
                    for target in self.targets:
                        if robot.position == target.position and robot.has_package:
                            if not target.occupied:  # Check if the target is unoccupied
                                robot.has_package = False
                                target.occupied = True
                                for package in self.packages:
                                    if package.picked and not package.delivered_to_target:  # Deliver the picked package
                                        package.position = target.position  # Place the package on the target
                                        package.delivered_to_target = True  # Mark as delivered
                                        break  # Only one package can be delivered at a time
                                reward += 10  # Reward for delivering the package
                                self.terminated = all(target.occupied for target in self.targets) # cooperative termination
                                break  # Only one robot can deposit
                            else:
                                reward -= 5  # Penalize depositing on an already occupied target


        # Track robot positions to detect collisions (MOVED OUTSIDE THE LOOP TO ENSURE BOTH ROBOTS TAKE AN ACTION FIRST)
        # positions = [tuple(robot.position) for robot in self.robots]
        # Convert list elements to tuples for hashability
        if len(robot_position) != len(set(tuple(pos) for pos in robot_position)):
            reward -= 10  # Penalize collisions
            print("Penalty collision")


        # Penalize robots for being in close proximity to each other
        for i, robot1 in enumerate(self.robots):
            for j, robot2 in enumerate(self.robots):
                if i != j and self._manhattan_distance(robot1.position, robot2.position) <= proximity_threshold_close:
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
                            reward += 3
                        elif proximity_threshold_close <= distance_pkg < proximity_threshold_far:
                            reward += 2
                        else:
                            reward -= 1  ## adjust if is too far
            else:
                for target in self.targets:
                    distance_target = self._manhattan_distance(robot.position, target.position)
                    if distance_target < proximity_threshold_close:
                        reward += 3
                    elif proximity_threshold_close <= distance_target < proximity_threshold_far:
                        reward += 2
                    else:
                        reward -= 1  ## adjust if is too far                

        # Penalize idling (no action)
        if all(action not in [RobotAction.PICK, RobotAction.DEPOSIT] for action in robot_actions):
            reward -= 1

        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            truncated = True
                    
        obs = self._get_observation()
        
        
        info = {}
                    
        return obs, reward, truncated, info

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
    terminated = False


    while(not terminated):
        # print(f"\nStep {step + 1}")
        rand_action = env.action_space.sample()
        print(f"Random Actions: {rand_action}")

        obs, reward, terminated, truncated, _ = env.step(rand_action)
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        env.render()

    print("Episode ended. Resetting environment.")
    obs, _ = env.reset()


        
        