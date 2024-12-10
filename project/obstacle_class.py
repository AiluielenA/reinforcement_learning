import random

class Obstacle:
    def __init__(self, grid_rows, grid_cols, occupied_positions):
        """Initialize obstacle position, ensuring no overlap with other objects."""
        self.position = self.generate_position(grid_rows, grid_cols, occupied_positions)

    def generate_position(self, grid_rows, grid_cols, occupied_positions):
        while True:
            pos = [
                random.randint(1, grid_rows),
                random.randint(1, grid_rows)
            ]
            if pos not in occupied_positions:
                return pos

    def __str__(self):
        return f"Obstacle(pos={self.position})"
