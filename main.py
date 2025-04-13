import numpy as np
import pygame
from car import Car

WIN_WIDTH, WIN_HEIGHT = 768, 768
START_POS = (420, 640)

pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Self-Driving Car RL")
clock = pygame.time.Clock()
MAP_IMAGE = pygame.transform.smoothscale(pygame.image.load("assets/new_map.png"), (WIN_WIDTH, WIN_HEIGHT))

action_space = np.array([
    [1, 0, 0, 0],  
    [0, 1, 0, 0],  
    [1, 0, 1, 0],   
    [1, 0, 0, 1],   
    [0, 0, 1, 0],  
    [0, 0, 0, 1],  
])

car = Car(START_POS)
total_reward = 0
done = False
step = 0

while not done and car.alive and step < 300:
    screen.blit(MAP_IMAGE, (0, 0))

    action = action_space[np.random.randint(0, len(action_space))]
    reward = car.update(action, MAP_IMAGE)
    total_reward += reward
    car.draw(screen)

    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    step += 1

pygame.quit()
print("Total Reward:", total_reward)