import random
from enum import Enum



class RobotAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    PICK = 4
    DEPOSIT = 5
    
class Robot:
    def __init__(self, grid_rows, grid_cols, occupied_positions):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.position = self.reset(occupied_positions)

    def reset(self, occupied_positions, seed=None):
        random.seed(seed)
        self.has_package = False
        while True:
            self.position = [
                random.randint(0, self.grid_rows - 1),
                random.randint(0, self.grid_cols - 1)
            ]
            if self.position not in occupied_positions:
                return self.position
            

    # move() needs different logic to control terminate flag in the env.step()
    def move(self, action: RobotAction, obstacle_positions):
        
        print(f"Action received: {action}")
        print(f"Initial position: {self.position}")
        
        if action == RobotAction.LEFT and self.position[1] > 0:
            dummypos = self.position[1] - 1
            if [self.position[0],dummypos] in obstacle_positions:
                return True
            else:
                self.position[1] = dummypos
                return False
        elif action == RobotAction.RIGHT and self.position[1] < self.grid_cols - 1:
            dummypos = self.position[1] + 1
            if [self.position[0],dummypos] in obstacle_positions:
                return True
            else:
                self.position[1] = dummypos
                return False
        elif action == RobotAction.UP and self.position[0] > 0:
            dummypos = self.position[0] - 1
            if [dummypos, self.position[1]] in obstacle_positions:
                return True
            else:
                self.position[0] = dummypos
                return False
        elif action == RobotAction.DOWN and self.position[0] < self.grid_rows - 1:
            dummypos = self.position[0] + 1
            if [dummypos, self.position[1]] in obstacle_positions:
                return True
            else:
                self.position[0] = dummypos
                return False
        
        print(f"Updated position: {self.position}")
        return False
    
    def __str__(self):
        return f"Robot(pos={self.position}, has_package={self.has_package})"