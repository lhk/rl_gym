import sys

import pygame
from pygame.locals import *

import environments.trailer_env.params as params
from environments.trailer_env.environment import Environment

pygame.init()
clock = pygame.time.Clock()
window = pygame.display.set_mode(params.screen_size)
pygame.display.set_caption("Pygame cheat sheet")

mouse_x, mouse_y = 0, 0

env = Environment()
env.new_episode()

while True:

    frame = env.render(return_numpy=False)
    window.blit(frame, (0, 0))

    acceleration = 0
    steering_angle = 0

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[K_UP]:
        acceleration = 1
    elif keys[K_DOWN]:
        acceleration = -1
    if keys[K_LEFT]:
        steering_angle = -1
    elif keys[K_RIGHT]:
        steering_angle = 1
    elif keys[K_SPACE]:
        env.new_episode()

    reward, done = env.make_action((acceleration, steering_angle))
    print(reward)
    if done:
        print("collision")

    pygame.display.update()

    clock.tick(10)
