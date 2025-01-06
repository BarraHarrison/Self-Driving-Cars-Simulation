import pygame 
import random
import neat
import os

pygame.init()

maps = [pygame.image.load(f"map{i}.png").convert() for i in range(1, 6)]

# Set the screen size on the first map
SCREEN_WIDTH, SCREEN_HEIGHT = maps[0].get_width(), maps[0].get_height()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation with NEAT")

# Colours
WHITE = [255, 255, 255]

# Clock for controlling frame rate
clock = pygame.time.Clock()

CAR_SPRITE = pygame.image.load("car.png")
CAR_WIDTH, CAR_HEIGHT = 50, 30

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 3
        self.angle = 0
        self.image = pygame.transform.scale(CAR_SPRITE, (CAR_WIDTH, CAR_HEIGHT))
        self.fitness = 0

    def draw(self):
        pass

    def move(self, output):
        pass

    def eval_genomes(genomes, config):
        pass

    def run_neat(config_file):
        pass