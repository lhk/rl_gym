import sys

import numpy as np
import pygame
import pygame.locals as pgl

from environments.obstacle_car.environment_vec import Environment_Vec as Environment

canvas_size = (500, 500)
canvas = np.zeros([*canvas_size, 3])

pygame.init()
clock = pygame.time.Clock()
window = pygame.display.set_mode(canvas_size)
pygame.display.set_caption("env")

mouse_x, mouse_y = 0, 0

env = Environment()
env.reset()
total_reward = 0

while True:

    canvas[:] = 0
    env.render_to_canvas(canvas)

    surf = pygame.surfarray.make_surface((canvas * 255).astype(np.uint8))
    window.blit(surf, (0, 0))

    acceleration = 0
    steering_angle = 0

    for event in pygame.event.get():
        if event.type == pgl.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pgl.KEYDOWN:
            if event.key == pgl.K_SPACE:
                print(total_reward)
                total_reward = 0
                env.reset()

    keys = pygame.key.get_pressed()
    if keys[pgl.K_UP]:
        acceleration = 1
    elif keys[pgl.K_DOWN]:
        acceleration = -1
    if keys[pgl.K_LEFT]:
        steering_angle = -1
    elif keys[pgl.K_RIGHT]:
        steering_angle = 1

    observation, reward, done = env.make_action((acceleration, steering_angle))
    total_reward += reward
    if done:
        print("c")
        print("collision")

    pygame.display.update()

    clock.tick(10)
