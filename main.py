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

current_map_index = 0

class Car:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.speed = 3
        self.angle = 0
        self.fitness = 0

    def draw(self):
        car_rect = pygame.Rect(self.x, self.y, 30, 50)
        pygame.draw.rect(screen, self.color, car_rect)

    # Neural Network outputs: [left, right, accelerate, decelerate]
    def move(self, output):
        if output[0] > 0.5:
            self.x -= 2
        if output[1] > 0.5:
            self.x += 2
        if output[2] > 0.5:
            self.y -= 3
        if output[3] > 0.5:
            self.y += 3

        # Keep car within screen boundaries
        self.x = max(0, min(SCREEN_WIDTH - 30, self.x))
        self.y = max(0, min(SCREEN_HEIGHT - 50, self.y))

        # Increment improvement (fitness) for moving forward
        # Survival of the fittest if you will
        self.fitness += 1


def eval_genomes(genomes, config):
    global current_map_index

    cars = []
    nets = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(400, 500, BLUE))
        genome.fitness = 0

    running = True
    while running:
        screen.fill(WHITE)
        screen.blit(maps[current_map_index], (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        for i, car in enumerate(cars):
            inputs = [
                car.x / SCREEN_WIDTH,
                car.y / SCREEN_HEIGHT,
                current_map_index / len(maps),
                random.random(), #Placeholder sensor input
                random.random(),
            ]
            output = nets[i].activate(inputs)
            car.move(output)

            # Check collisions with the map
            if car.y < 0:
                genomes[i][1].fitness += 1000
                running = False

            car.draw()

        pygame.display.flip()
        clock.tick(30)


def run_neat(config_file):
    pass