import numpy as np
import pygame

import environments.obstacle_car.colors as colors
import environments.obstacle_car.params as params
import environments.obstacle_car.utils as utils


class Car():
    def __init__(self, pos, speed, rot):
        self.pos = pos
        self.speed = speed
        self.rot = rot

    def update(self, acceleration, steering_angle):
        acceleration = np.clip(acceleration, -1, 1)
        steering_angle = np.clip(steering_angle, -1, 1)

        self.speed += acceleration * params.dT
        self.speed = np.clip(self.speed, params.min_speed, params.max_speed)

        x, y = self.pos
        dx = -np.sin(self.rot / 180 * np.pi) * self.speed * params.dT
        new_x = x + dx
        dy = -np.cos(self.rot / 180 * np.pi) * self.speed * params.dT
        new_y = y + dy

        self.pos[:] = (new_x, new_y)
        self.rot -= self.speed * steering_angle * params.dT * params.steering_factor
