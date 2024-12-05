import random 

class Target:
    def __init__(self, grid_rows, grid_cols, occupied_positions):
        self.position = self.generate_position(grid_rows, grid_cols, occupied_positions)

    def generate_position(self, grid_rows, grid_cols, occupied_positions):
        while True:
            pos = [
                random.randint(0, grid_rows - 1),
                random.randint(0, grid_cols - 1)
            ]
            if pos not in occupied_positions:
                return pos
