import pygame
import neat
import os
import math
import random

WIN_WIDTH, WIN_HEIGHT = 800, 600
MAP_PATH = "assets/map1.png"
CAR_IMG_PATH = "assets/car.png"
CHECKPOINT_DIR = "checkpoints"

pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Self-Driving Car with NEAT")
clock = pygame.time.Clock()

MAP_IMAGE = pygame.transform.smoothscale(pygame.image.load(MAP_PATH), (WIN_WIDTH, WIN_HEIGHT))
CAR_IMAGE = pygame.transform.scale(pygame.image.load(CAR_IMG_PATH), (40, 20))

START_POS = (100, 540)

class Car:
    def __init__(self):
        self.x, self.y = START_POS
        self.angle = 0
        self.speed = 2
        self.image = CAR_IMAGE
        self.sensors = []
        self.alive = True
        self.fitness = 0

    def draw(self, win):
        rotated = pygame.transform.rotate(self.image, -self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        win.blit(rotated, rect.topleft)
        for sx, sy in self.sensors:
            pygame.draw.line(win, (255, 0, 0), (self.x, self.y), (sx, sy), 1)
            pygame.draw.circle(win, (0, 255, 0), (sx, sy), 3)

    def move(self, output):
        if output[0] > 0.5: self.angle += 5
        if output[1] > 0.5: self.angle -= 5
        if output[2] > 0.5:
            self.x += self.speed * math.cos(math.radians(self.angle))
            self.y -= self.speed * math.sin(math.radians(self.angle))
        if output[3] > 0.5:
            self.x -= self.speed * math.cos(math.radians(self.angle))
            self.y += self.speed * math.sin(math.radians(self.angle))

    def cast_sensors(self, map_img):
        self.sensors.clear()
        for a in [-60, -30, 0, 30, 60, 90]:
            sensor_angle = self.angle + a
            for d in range(0, 100, 5):
                sx = int(self.x + d * math.cos(math.radians(sensor_angle)))
                sy = int(self.y - d * math.sin(math.radians(sensor_angle)))
                if 0 <= sx < WIN_WIDTH and 0 <= sy < WIN_HEIGHT:
                    if map_img.get_at((sx, sy)) == (255, 255, 255, 255):  # White = off track
                        self.sensors.append((sx, sy))
                        break
                else:
                    self.sensors.append((sx, sy))
                    break

    def check_collision(self, map_img):
        try:
            if map_img.get_at((int(self.x), int(self.y))) == (255, 255, 255, 255):
                self.alive = False
        except:
            self.alive = False

    def update(self, output, map_img):
        if not self.alive:
            return
        self.move(output)
        self.cast_sensors(map_img)
        self.check_collision(map_img)
        self.fitness += 0.1


def eval_genomes(genomes, config):
    nets, cars, ge = [], [], []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car())
        genome.fitness = 0
        ge.append(genome)

    run = True
    while run and any(car.alive for car in cars):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.blit(MAP_IMAGE, (0, 0))

        for i, car in enumerate(cars):
            if not car.alive:
                continue

            car.cast_sensors(MAP_IMAGE)

            inputs = [math.dist((car.x, car.y), s) / 100 for s in car.sensors] + [car.angle / 360]
            car.update(nets[i].activate(inputs), MAP_IMAGE)
            ge[i].fitness = car.fitness
            car.draw(screen)

        pygame.display.update()
        clock.tick(60)


def run_neat(config_file):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix=os.path.join(CHECKPOINT_DIR, "checkpoint-")))

    winner = p.run(eval_genomes, 50)
    print("Best Genome:", winner)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run_neat(config_path)