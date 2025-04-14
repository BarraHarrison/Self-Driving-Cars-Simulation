import pygame
import math
from reward_system import compute_reward

class Car:
    def __init__(self, start_pos):
        self.x, self.y = start_pos
        self.angle = 0
        self.speed = 2
        
        raw_car = pygame.image.load("assets/car.png").convert_alpha()
        scaled_car = pygame.transform.scale(raw_car, (20, 10))
        self.image = pygame.transform.rotate(scaled_car, -self.angle)


        self.sensors = []
        self.sensor_angles = [-60, -30, 0, 30, 60, 90, -90]
        self.alive = True
        self.distance_traveled = 0
        self.prev_position = (self.x, self.y)
        self.velocity = 0
        self.max_speed = 4
        self.rotation_speed = 4
        self.path = []
        self.path_history = []

    def draw(self, screen):
        rotated = pygame.transform.rotate(self.image, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect.topleft)

    def draw_sensors(self, screen):
        """
        Draw green lines from the car to each sensor endpoint for debugging.
        """
        for sensor in self.sensors:
            pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), sensor, 2)
            pygame.draw.circle(screen, (0, 255, 0), sensor, 3)

    def cast_sensors(self, map_image):
        """
        Casts forward-facing sensors (like rays) to detect distance to the nearest obstacle.
        Adds 5 sensors: front, front-left, front-right, left, right
        """
        self.sensors = []
        max_distance = 150

        for angle_offset in self.sensor_angles:
            angle = math.radians(self.angle + angle_offset)
            for distance in range(0, max_distance, 5):
                x = int(self.x + distance * math.cos(angle))
                y = int(self.y - distance * math.sin(angle))

                if x < 0 or x >= map_image.get_width() or y < 0 or y >= map_image.get_height():
                    break

                pixel = map_image.get_at((x, y))[:3]
                if sum(pixel) > 60:
                    break
                
            self.sensors.append((x, y))

    def get_normalized_sensor_distances(self, max_distance=150):
        distances = []
        for x, y in self.sensors:
            dist = math.dist((self.x, self.y), (x, y))
            distances.append(min(dist / max_distance, 1.0))

        return distances


    def move_forward(self):
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)
        self.distance_traveled += math.dist((self.x, self.y), self.prev_position)
        self.prev_position = (self.x, self.y)
        self.path.append((self.x, self.y))
        self.path_history.append((self.x, self.y))

    def get_clear_direction(self, map_image):
        """
        Check which sensor has the highest clear distance and return a direction hint.
        """
        distances = []
        for sensor in self.sensors:
            try:
                pixel = map_image.get_at((int(sensor[0]), int(sensor[1])))[:3]
                if sum(pixel) < 60:
                    dist = math.dist((self.x, self.y), sensor)
                else:
                    dist = 0
            except IndexError:
                dist = 0
            distances.append(dist)

        max_index = distances.index(max(distances))
        
        sensor_angle = self.sensor_angles[max_index]
        if sensor_angle < 0:
            return "left"
        elif sensor_angle > 0:
            return "right"
        else:
            return "forward"


    def draw_path(self, screen):
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                pygame.draw.line(screen, (255, 0, 0), self.path[i-1], self.path[i], 2)


    def brake(self):
        self.speed = max(0, self.speed - 0.5)


    def rotate_left(self):
        if self.speed < 1.0:
            self.angle += self.rotation_speed * 1.5
        else:
            self.angle += self.rotation_speed

    def rotate_right(self):
        if self.speed < 1.0:
            self.angle -= self.rotation_speed * 1.5
        else:
            self.angle -= self.rotation_speed

    def update(self, action, map_image):
        if not self.alive:
            return

        if action[0] > 0.5:
            self.move_forward()
        if action[1] > 0.5:
            self.brake()
        if action[2] > 0.5:
            self.rotate_left()
        if action[3] > 0.5:
            self.rotate_right()

        self.cast_sensors(map_image)

        return compute_reward(self, map_image)