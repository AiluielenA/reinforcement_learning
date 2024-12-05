import random
import enum as Enum


class RobotAction(Enum):
    LEFT=0
    DOWN=1
    RIGHT=2
    UP=3
    
class Robot:
    def __init__(self, grid_rows, grid_cols):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.reset()

    def reset(self):
        self.position = [
            random.randint(0, self.grid_rows - 1),
            random.randint(0, self.grid_cols - 1)
        ]
        self.has_package = False

    def move(self, action: RobotAction):
        if action == RobotAction.LEFT and self.position[1] > 0:
            self.position[1] -= 1
        elif action == RobotAction.RIGHT and self.position[1] < self.grid_cols - 1:
            self.position[1] += 1
        elif action == RobotAction.UP and self.position[0] > 0:
            self.position[0] -= 1
        elif action == RobotAction.DOWN and self.position[0] < self.grid_rows - 1:
            self.position[0] += 1