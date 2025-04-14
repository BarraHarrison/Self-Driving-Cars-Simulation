import pygame
import math

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


    def move_forward(self):
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)
        self.distance_traveled += math.dist((self.x, self.y), self.prev_position)
        self.prev_position = (self.x, self.y)

    def rotate_left(self):
        self.angle += self.rotation_speed

    def rotate_right(self):
        self.angle -= self.rotation_speed

    def update(self, action, map_image):
        if not self.alive:
            return

        if action[0] > 0.5:
            self.move_forward()
        if action[1] > 0.5:
            self.velocity = max(self.velocity - 0.2, -self.max_speed)
        if action[2] > 0.5:
            self.rotate_left()
        if action[3] > 0.5:
            self.rotate_right()

        reward = self.compute_reward(map_image)
        return reward

    def compute_reward(self, map_image):
        reward = 1

        try:
            pixel = map_image.get_at((int(self.x), int(self.y)))[:3]
            if pixel[0] < 50 and pixel[1] < 50 and pixel[2] < 50:
                reward += 10
            else:
                reward -= 100
                self.alive = False
        except IndexError:
            reward -= 100
            self.alive = False

        return reward
