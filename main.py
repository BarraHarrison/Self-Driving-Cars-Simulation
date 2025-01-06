import pygame
import random
import neat
import os
import math

pygame.init()

# Set a temporary display mode to allow image conversion
TEMP_SCREEN_WIDTH, TEMP_SCREEN_HEIGHT = 800, 600
pygame.display.set_mode((TEMP_SCREEN_WIDTH, TEMP_SCREEN_HEIGHT))

# Load map images and adjust screen size dynamically
maps = [pygame.image.load(f"map{i}.png").convert() for i in range(1, 6)]
SCREEN_WIDTH, SCREEN_HEIGHT = maps[0].get_width(), maps[0].get_height()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation with NEAT")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Load car sprite
CAR_SPRITE = pygame.image.load("car.png")
CAR_WIDTH, CAR_HEIGHT = 50, 30

# Starting position and angle
START_X = SCREEN_WIDTH // 2
START_Y = SCREEN_HEIGHT - 100
START_ANGLE = 0


class Car:
    def __init__(self, x, y, angle=START_ANGLE):
        self.x = x
        self.y = y
        self.speed = 3
        self.angle = angle
        self.image = pygame.transform.scale(CAR_SPRITE, (CAR_WIDTH, CAR_HEIGHT))
        self.fitness = 0
        self.sensors = []

    def draw(self):
        # Draw the car
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, rect.topleft)

        # Draw sensors
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

    # Check for proximity to boundary and adjust behavior
    boundary_proximity = min(
        math.sqrt((sensor[0] - self.x) ** 2 + (sensor[1] - self.y) ** 2)
        for sensor in self.sensors
    )

    if boundary_proximity < 50:  # Close to boundary
        self.fitness -= 0.1 * (50 - boundary_proximity) / 50  # Small penalty
        self.speed = 2  # Reduce speed but allow movement
    else:
        self.speed = 3  # Restore normal speed

    # Increment fitness for moving forward
    self.fitness += 0.1



    def detect_collision(self, map_image):
        # Check the pixel color at the car's center
        car_color = map_image.get_at((int(self.x), int(self.y)))
        return car_color != BLACK

    def cast_sensors(self, map_image):
        self.sensors = []
        sensor_angles = [-120, -90, -60, -30, 0, 30, 60, 90, 120]  # Higher resolution sensors
        for angle in sensor_angles:
            sensor_angle = self.angle + angle
            for dist in range(0, 200, 5):  # 200-pixel sensor range
                x = int(self.x + dist * math.cos(math.radians(sensor_angle)))
                y = int(self.y - dist * math.sin(math.radians(sensor_angle)))

                if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
                    break

                if map_image.get_at((x, y)) != BLACK:
                    self.sensors.append((x, y))
                    break
            else:
                self.sensors.append((x, y))


    def restart(self):
        distance_traveled = math.sqrt((self.x - START_X) ** 2 + (self.y - START_Y) ** 2)
        self.fitness += distance_traveled  
        self.x = START_X + random.randint(-10, 10)  
        self.y = START_Y
        self.angle = START_ANGLE + random.randint(-15, 15)  
        self.fitness -= 50  


def eval_genomes(genomes, config):
    global maps, SCREEN_WIDTH, SCREEN_HEIGHT

    map_image = maps[0]

    cars = []
    nets = []
    ge = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(START_X, START_Y))
        genome.fitness = 0
        ge.append(genome)

    running = True
    while running and len(cars) > 0:
        screen.fill(WHITE)
        screen.blit(map_image, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        for i in range(len(cars) - 1, -1, -1):
            car = cars[i]
            car.cast_sensors(map_image)

            # Neural network inputs
            inputs = []
            for sensor in car.sensors:
                dist = math.sqrt((sensor[0] - car.x) ** 2 + (sensor[1] - car.y) ** 2)
                inputs.append(1 - (dist / 200))  # Normalize and invert distance

            inputs.append(car.angle / 360)

            output = nets[i].activate(inputs)
            car.move(output)

            car.draw()

            # Check collision and fitness update
            if car.detect_collision(map_image):
                ge[i].fitness = car.fitness
                cars.pop(i)
                nets.pop(i)
                ge.pop(i)

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

    checkpoint = neat.Checkpointer(generation_interval=10, time_interval_seconds=None, filename_prefix="checkpoint-")
    p.add_reporter(checkpoint)

    winner = p.run(eval_genomes, 50)

    print("\nBest Genome:\n{!s}".format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    checkpoint_file = 'checkpoint-10'
    if os.path.exists(checkpoint_file):
        print(f"Restoring from checkpoint: {checkpoint_file}")
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        p.run(eval_genomes, 50)
    else:
        run_neat(config_path)
