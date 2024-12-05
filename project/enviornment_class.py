import robot_class as Robot
import package_class as Package
import target_class as Target
import numpy as np


class Environment:
    def __init__(self, grid_rows=4, grid_cols=5, num_robots=2, num_packages=2, targets=2):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.robots = []
        self.packages = []
        self.targets = []
        
        self.max_steps = 100 # time limit
        self.steps_taken = 0  # Initialize step counter

        # Initialize robots
        for _ in range(num_robots):
            self.robots.append(Robot(grid_rows, grid_cols))

        # Initialize packages and targets, ensuring no overlaps
        occupied_positions = [robot.position for robot in self.robots]
        for _ in range(num_packages):
            package = Package(grid_rows, grid_cols, occupied_positions)
            self.packages.append(package)
            occupied_positions.append(package.position)

        for _ in range(targets):
            target = Target(grid_rows, grid_cols, occupied_positions)
            self.targets.append(target)
            occupied_positions.append(target.position)

    def reset(self):
        self.__init__(self.grid_rows, self.grid_cols, len(self.robots), len(self.packages), len(self.targets))
        self.steps_taken = 0  # Initialize step counter
        
        info = {}
        
        obs = np.concatenate((
            [robot.position for robot in self.robots],
            [target.position for target in self.targets],
            [package.position for package in self.packages],
            [int(package.picked) for package in self.packages]
        )).flatten()
        
        return obs, info

    def step(self, robot_actions):
        
        reward = 0
        terminated = False
        truncated = False
        
        for i, robot in enumerate(self.robots):
            robot.move(robot_actions[i])

            # Check if the robot picks up a package
            for package in self.packages:
                if not package.picked and robot.position == package.position:
                    robot.has_package = True
                    package.picked = True
                    reward = 0.5

            # Check if the robot delivers a package
            for target in self.targets:
                if robot.has_package and robot.position == target.position:
                    robot.has_package = False
                    reward = 1.0
                    terminated = True
            
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                truncated = True

                    
                    
        obs = np.concatenate((
            [robot.position for robot in self.robots],
            [target.position for target in self.targets],
            [package.position for package in self.packages],
            [int(package.picked) for package in self.packages]
        )).flatten()
        
        info = {}
                    
        return obs, reward, terminated, truncated, info



    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])