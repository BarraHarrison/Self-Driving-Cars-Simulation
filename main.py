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
        rotated_image = pygame.transform.scale(self.image, -self.angle)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, rect.topleft)

    def move(self, output):
        # Neural network outputs: [left, right, accelerate, decelerate]
        if output[0] > 0.5:
            self.angle += 2
        if output[1] > 0.5:
            self.angle -= 2
        if output[2] > 0.5:
            self.x += self.speed * pygame.math.Vector2(1, 0).rotate(-self.angle).x
            self.y += self.speed * pygame.math.Vector2(1, 0).rotate(-self.angle).y
        if output[3] > 0.5:
            self.x -= self.speed * pygame.math.Vector2(1, 0).rotate(-self.angle).x
            self.y -= self.speed * pygame.math.Vector2(1, 0).rotate(-self.angle).y

        # Increment fitness as it moves forward
        self.fitness += 0.1

        # Keep the car within the screen boundaries
        self.x = max(0, min(SCREEN_WIDTH, self.x))
        self.y = max(0, min(SCREEN_HEIGHT, self.y))

    def eval_genomes(genomes, config):
        global maps, SCREEN_WIDTH, SCREEN_HEIGHT

        # Using the first map for training
        map_image = maps[0]

        cars = []
        nets = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            cars.append(Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100))
            genome.fitness = 0

        running = True
        while running:
            screen.fill(WHITE)
            screen.blit(map_image, (0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            for i, car in enumerate(cars):
                # Neural Network inputs
                inputs = [
                    car.x / SCREEN_WIDTH,
                    car.y / SCREEN_HEIGHT,
                    car.angle / 360,
                    random.random(), # Placeholder sensor input
                    random.random(), 
                ]

    def run_neat(config_file):
        pass