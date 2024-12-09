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
    def __init__(self, grid_rows, grid_cols):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.reset()

    def reset(self, seed=None):
        random.seed(seed)
        self.position = [
            random.randint(0, self.grid_rows - 1),
            random.randint(0, self.grid_cols - 1)
        ]
        self.has_package = False

    # move() needs different logic to control terminate flag in the env.step()
    def move(self, action: RobotAction):
        
        print(f"Action received: {action}")
        print(f"Initial position: {self.position}")
        
        if action == RobotAction.LEFT and self.position[1] > 0:
            self.position[1] -= 1
        elif action == RobotAction.RIGHT and self.position[1] < self.grid_cols - 1:
            self.position[1] += 1
        elif action == RobotAction.UP and self.position[0] > 0:
            self.position[0] -= 1
        elif action == RobotAction.DOWN and self.position[0] < self.grid_rows - 1:
            self.position[0] += 1
        
        print(f"Updated position: {self.position}")
        return self.position
    
    def __str__(self):
        return f"Robot(pos={self.position}, has_package={self.has_package})"