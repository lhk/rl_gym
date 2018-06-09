import sys

import pygame
from pygame.locals import *
import numpy as np

import environments.obstacle_car.params as params
from environments.obstacle_car.environment import Environment_Graphical

pygame.init()
clock = pygame.time.Clock()
window = pygame.display.set_mode(params.screen_size)
pygame.display.set_caption("env")

mouse_x, mouse_y = 0, 0

env = Environment_Graphical()
env.reset()

while True:

    frame = (env.render()*255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(frame)
    window.blit(surf, (0, 0))

    acceleration = 0
    steering_angle = 0

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                env.reset()

    keys = pygame.key.get_pressed()
    if keys[K_UP]:
        acceleration = 1
    elif keys[K_DOWN]:
        acceleration = -1
    if keys[K_LEFT]:
        steering_angle = -1
    elif keys[K_RIGHT]:
        steering_angle = 1

    observation, reward, done = env.make_action((acceleration, steering_angle))
    print(reward)
    if done:
        print("collision")

    pygame.display.update()

    clock.tick(10)
