import pygame
import random
import neat
import os
import math

pygame.init()

maps = [pygame.image.load(f"map{i}.png").convert() for i in range(1, 6)]

SCREEN_WIDTH, SCREEN_HEIGHT = maps[0].get_width(), maps[0].get_height()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation with NEAT")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Load car sprite
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
        self.sensors = []

    def draw(self):
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, rect.topleft)

        for sensor in self.sensors:
            pygame.draw.line(screen, RED, (self.x, self.y), sensor, 1)
            pygame.draw.circle(screen, RED, sensor, 5)

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

        # Increment fitness as the car moves forward
        self.fitness += 0.1

    def detect_collision(self, map_image):
        # Check pixel color at the car'a center
        car_color = map_image.get_at((int(self.x), int(self.y)))
        return car_color != BLACK

    def cast_sensors(self, map_image):
        self.sensors = []
        sensor_angles = [-60, -30, 0, 30, 60]
        for angle in sensor_angles:
            sensor_angle = self.angle + angle
            for dist in range(0, 200, 5): # 200 pixels sensor range
                x = int(self.x + dist * math.cos(math.radians(sensor_angle)))
                y = int(self.y - dist * math.sin(math.radians(sensor_angle)))

                # Stop if out of bounds
                if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
                    break

                # Stop if sensor detects off-track
                if map_image.get_at((x, y)) != BLACK:
                    self.sensors.append((x, y))
                    break
            else:
                # If there is no collision
                self.sensors.append((x, y))


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
            car.cast_sensors(map_image)

            # Neural Network inputs
            inputs = []
            for sensor in car.sensors:
                dist = math.sqrt((sensor[0] - car.x) ** 2 + (sensor[1] - car.y) ** 2)
                inputs.append(dist / 200)

            inputs.append(car.angle / 360)

            # Getting outputs and move the car
            output = nets[i].activate(inputs)
            car.move(output)

            car.draw()

            # Check if we are off-track
            if car.detect_collision(map_image):
                genome = genomes[i][1]
                genome.fitness = car.fitness
                cars.pop(i)
                nets.pop(i)
                genomes.pop(i)

        if not cars:
            running = False

        pygame.display.flip()
        clock.tick(30)

def run_neat(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)
    print("\nBest Genome:\n{!s}".format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run_neat(config_path)
