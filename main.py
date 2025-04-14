import pygame
import sys
import math
import random
import pickle
import numpy as np
from car import Car
from collections import defaultdict
from reward_system import compute_reward
from save_path import save_path_if_high_reward

WIN_WIDTH, WIN_HEIGHT = 768, 768
START_POS = (420, 640)
MAP_PATH = "assets/new_map.png"
FPS = 60

ACTIONS = [
    (1, 0, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
] 
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.95

pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
clock = pygame.time.Clock()
MAP_IMAGE = pygame.transform.smoothscale(pygame.image.load(MAP_PATH), (WIN_WIDTH, WIN_HEIGHT))

def normalize_sensor_values(sensor_distances, max_distance=150):
    return np.clip(np.array(sensor_distances) / max_distance, 0.0, 1.0)

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(float)

    def get_state(self, car, sensor_distances):
        x_bin = int(car.x // 20)
        y_bin = int(car.y // 20)
        angle_bin = int(car.angle // 45)
        moving = 1 if car.velocity > 0.5 else 0
        sensor_state = tuple(normalize_sensor_values(sensor_distances))
        return (x_bin, y_bin, angle_bin, moving) + sensor_state


    def choose_action(self, state):
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        q_values = [self.q_table[(state, a)] for a in ACTIONS]
        return ACTIONS[q_values.index(max(q_values))]

    def update_q(self, state, action, reward, next_state):
        current_q = self.q_table[(state, action)]
        future_q = max([self.q_table[(next_state, a)] for a in ACTIONS])
        self.q_table[(state, action)] = current_q + ALPHA * (reward + GAMMA * future_q - current_q)

def main():
    agent = QLearningAgent()

    for episode in range(1, 1001):
        car = Car(start_pos=START_POS)
        total_reward = 0
        step = 0

        while car.alive and step < 500:
            screen.blit(MAP_IMAGE, (0, 0))
            car.draw_path(screen)
            car.draw(screen)
            car.draw_sensors(screen)
            pygame.display.flip()

            car.cast_sensors(MAP_IMAGE)
            sensor_distances = [math.dist((car.x, car.y), s) for s in car.sensors]
            sensor_distances = normalize_sensor_values(sensor_distances)

            state = agent.get_state(car, sensor_distances)
            action = agent.choose_action(state)
            car.update(action, MAP_IMAGE)
            reward = compute_reward(car, MAP_IMAGE)
            next_state = agent.get_state(car, sensor_distances)
            agent.update_q(state, action, reward, next_state)

            total_reward += reward
            step += 1
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
        save_path_if_high_reward(car.path_history, total_reward)

    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.q_table), f)

if __name__ == "__main__":
    main()