import sys

import pygame
import pygame.locals as pgl
import numpy as np

import environments.obstacle_car.params_radial as params
from environments.obstacle_car.environment_radial import Environment_Vector

canvas_size = (500, 500)
x_coords = np.arange(canvas_size[0])
y_coords = np.arange(canvas_size[1])
x, y = np.meshgrid(x_coords, y_coords)
coords = np.stack([x, y], axis=-1)

pygame.init()
clock = pygame.time.Clock()
window = pygame.display.set_mode(canvas_size)
pygame.display.set_caption("env")

mouse_x, mouse_y = 0, 0

env = Environment_Vector()
env.reset()

while True:

    canvas = np.zeros([*canvas_size, 3])
    observation = env.render()

    # first observation is speed, throw it away
    # the rest has to be rescaled to the original range
    observation = observation[1:] * params.distance_rescale

    # then we organize it as vectors
    observation = observation.reshape((-1, 2))
    offset = np.array([canvas.shape[0] // 2, canvas.shape[1] // 2])
    observation = (observation + offset).astype(np.int)

    goal = observation[0]
    obstacles = observation[1:]

    goal = np.array(canvas.shape[:2]) - goal
    obstacles = np.array(canvas.shape[:2]) - obstacles

    dist = coords - goal
    dist = np.linalg.norm(dist, axis=-1)
    area = np.where(dist < env.initial_dist * params.max_dist)

    canvas[area[1], area[0], 2] = 0.5

    if np.all(goal > 0) and np.all(goal < canvas.shape[:2]):
        canvas[goal[0] - 5:goal[0] + 5, goal[1] - 5:goal[1] + 5, :] = 1

    for obstacle in obstacles:
        if np.all(obstacle > 0) and np.all(obstacle < canvas.shape[:2]):
            canvas[obstacle[0] - 5:obstacle[0] + 5, obstacle[1] - 5:obstacle[1] + 5, 0] = 1

    # a green dot at the center of the canvas, for our car
    canvas[offset[0] - 5:offset[0] + 5, offset[1] - 5:offset[1] + 5, 1] = 1

    canvas = (canvas * 255).astype(np.uint8)

    surf = pygame.surfarray.make_surface(canvas)
    window.blit(surf, (0, 0))

    acceleration = 0
    steering_angle = 0

    for event in pygame.event.get():
        if event.type == pgl.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pgl.KEYDOWN:
            if event.key == pgl.K_SPACE:
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
    print(reward)
    if done:
        print("collision")

    pygame.display.update()

    clock.tick(10)
