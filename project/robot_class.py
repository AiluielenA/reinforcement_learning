import random
from enum import Enum



class RobotAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    PICK = 4
    DEPOSIT = 5
    CHARGE = 6
    
class Robot:
    def __init__(self, grid_rows, grid_cols, occupied_positions, energy=150):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.position = self.reset(occupied_positions)
        self.energy = energy
        self.max_energy = 150
        self.has_package = False

    def reset(self, occupied_positions, seed=None):
        random.seed(seed)
        self.has_package = False
        self.max_energy = 150
        self.energy = self.max_energy  # Reset energy to maximum
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
        print(f"Energy before move: {self.energy}")
        
        if self.energy <= 0:
            print("Insufficient energy to move.")
            return False  # Movement fails due to lack of energy
       
        collision = False 
        if action == RobotAction.LEFT and self.position[1] > 0:
            dummypos = self.position[1] - 1
            if [self.position[0],dummypos] in obstacle_positions:
                collision = True
                return True
            else:
                self.position[1] = dummypos
                return False
        elif action == RobotAction.RIGHT and self.position[1] < self.grid_cols - 1:
            dummypos = self.position[1] + 1
            if [self.position[0],dummypos] in obstacle_positions:
                collision = True
                return True
            else:
                self.position[1] = dummypos
                return False
        elif action == RobotAction.UP and self.position[0] > 0:
            dummypos = self.position[0] - 1
            if [dummypos, self.position[1]] in obstacle_positions:
                collision = True
                return True
            else:
                self.position[0] = dummypos
                return False
        elif action == RobotAction.DOWN and self.position[0] < self.grid_rows - 1:
            dummypos = self.position[0] + 1
            if [dummypos, self.position[1]] in obstacle_positions:
                collision = True
                return True
            else:
                self.position[0] = dummypos
                return False
            
        # if not collision:
        #     self.energy -= 1  # Consume energy for valid move
        
        print(f"Updated position: {self.position}")
        return False
    
    def recharge(self):
        """Recharge energy to the maximum."""
        self.energy = self.max_energy
        print(f"Energy recharged to: {self.energy}")
    
    def __str__(self):
        return f"Robot(pos={self.position}, has_package={self.has_package})"