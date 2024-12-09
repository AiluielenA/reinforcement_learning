import random 

class Package:
    def __init__(self, grid_rows, grid_cols, occupied_positions):
        self.position = self.generate_position(grid_rows, grid_cols, occupied_positions)
        self.picked = False
        self.delivered_to_target = False  # Track whether the package is delivered to a target

    def generate_position(self, grid_rows, grid_cols, occupied_positions):
        while True:
            pos = [
                random.randint(0, grid_rows - 1),
                random.randint(0, grid_cols - 1)
            ]
            if pos not in occupied_positions:
                return pos
