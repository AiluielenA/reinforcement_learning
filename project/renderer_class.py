from enum import Enum
import pygame
import sys
from os import path
import random
from environment_class import Environment
from robot_class import Robot, RobotAction
from obstacle_class import Obstacle


class Renderer:
    def __init__(self, environment, cell_size=64, fps=10):
        self.environment = environment
        self.cell_size = cell_size
        self.fps = fps  # Set the frames per second
        self.window_size = (
            environment.grid_cols * cell_size,
            environment.grid_rows * cell_size
        )
        pygame.init()
        self.window_surface = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        self.load_sprites()

    def load_sprites(self):
        """Load and scale all sprites."""
        self.robot_img = pygame.image.load("project/sprites/bot_blue.png")
        self.robot_img = pygame.transform.scale(self.robot_img, (self.cell_size, self.cell_size))

        self.package_img = pygame.image.load("project/sprites/package.png")
        self.package_img = pygame.transform.scale(self.package_img, (self.cell_size, self.cell_size))

        self.target_img = pygame.image.load("project/sprites/target.png")
        self.target_img = pygame.transform.scale(self.target_img, (self.cell_size, self.cell_size))

        self.floor_img = pygame.image.load("project/sprites/floor.png")
        self.floor_img = pygame.transform.scale(self.floor_img, (self.cell_size, self.cell_size))

        self.obstacle_img = pygame.image.load("project/sprites/obstacle.png")  # Placeholder sprite for obstacles
        self.obstacle_img = pygame.transform.scale(self.obstacle_img, (self.cell_size, self.cell_size))

    def render(self):
        """Render the current state of the environment."""
        self.window_surface.fill((255, 255, 255))  # Clear the screen with a white background
        
        # Draw floor
        for r in range(self.environment.grid_rows):
            for c in range(self.environment.grid_cols):
                self.window_surface.blit(self.floor_img, (c * self.cell_size, r * self.cell_size))
        
        # Draw targets
        for target in self.environment.targets:
            x, y = target.position
            self.window_surface.blit(self.target_img, (y * self.cell_size, x * self.cell_size))

        # Draw packages
        for package in self.environment.packages:
            if not package.picked:
                x, y = package.position
                self.window_surface.blit(self.package_img, (y * self.cell_size, x * self.cell_size))

        # Draw obstacles
        for obstacle in self.environment.obstacles:  # Ensure `obstacles` are part of the environment
            x, y = obstacle.position
            self.window_surface.blit(self.obstacle_img, (y * self.cell_size, x * self.cell_size))

        # Draw robots
        for robot in self.environment.robots:
            x, y = robot.position
            self.window_surface.blit(self.robot_img, (y * self.cell_size, x * self.cell_size))

        pygame.display.update()
        self.clock.tick(self.fps)  # Limit rendering speed to `fps`


if __name__ == "__main__":
    env = Environment(grid_rows=6, grid_cols=7, num_robots=2, num_packages=2)

    # Add obstacles to the environment (manually for now)
    env.obstacles = [Obstacle(5, 5, [(r.position for r in env.robots)]) for _ in range(3)]  # Add 3 obstacles

    renderer = Renderer(env)

    while True:
        random_actions = [random.choice(list(RobotAction)) for _ in env.robots]
        env.step(random_actions, fps=5)
        renderer.render()
