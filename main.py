import pygame
import math
import os
from car import Car

WIN_WIDTH, WIN_HEIGHT = 768, 768
MAP_PATH = "assets/new_map.png"
CAR_IMG_PATH = "assets/car.png"

pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Self-Driving Car (RL Setup)")
clock = pygame.time.Clock()

MAP_IMAGE = pygame.transform.smoothscale(pygame.image.load(MAP_PATH), (WIN_WIDTH, WIN_HEIGHT))
CAR_IMAGE = pygame.transform.scale(pygame.image.load(CAR_IMG_PATH), (24, 12))

START_POS = (420, 640)


def main():
    car = Car()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(MAP_IMAGE, (0, 0))

        car.update_manual(MAP_IMAGE)
        car.draw(screen)

        pygame.display.update()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
