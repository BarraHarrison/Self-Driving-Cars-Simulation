import pygame
import sys
import math
import random
import pickle
from car import Car
from collections import defaultdict

WIN_WIDTH, WIN_HEIGHT = 768, 768
START_POS = (420, 640)
MAP_PATH = "assets/new_map.png"
FPS = 60

ACTIONS = [(1, 0, 0, 0), 
           (0, 0, 1, 0), 
           (0, 0, 0, 1)] 
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.95

pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
clock = pygame.time.Clock()
MAP_IMAGE = pygame.transform.smoothscale(pygame.image.load(MAP_PATH), (WIN_WIDTH, WIN_HEIGHT))

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(float)

    def get_state(self, car):
        x_bin = int(car.x // 20)
        y_bin = int(car.y // 20)
        angle_bin = int(car.angle // 45)
        return (x_bin, y_bin, angle_bin)

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
    car = Car(start_pos=START_POS)

    for episode in range(1, 1001):
        car = Car(start_pos=START_POS)
        total_reward = 0
        step = 0

        while car.alive and step < 500:
            screen.blit(MAP_IMAGE, (0, 0))
            car.draw(screen)
            pygame.display.flip()

            state = agent.get_state(car)
            action = agent.choose_action(state)
            reward = car.update(action, MAP_IMAGE)
            next_state = agent.get_state(car)
            agent.update_q(state, action, reward, next_state)

            total_reward += reward
            step += 1
            clock.tick(FPS)
            car.draw(screen)
            car.draw_sensors(screen)


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.q_table), f)

if __name__ == "__main__":
    main()