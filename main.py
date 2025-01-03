# Self-Driving Simulation in Python
import pygame
import random
import neat
import os

# Initialize pygame
pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation with NEAT")

clock = pygame.time.Clock()

maps = [
    pygame.image.load(f"map{i}.png").convert() for i in range(1, 6)
]