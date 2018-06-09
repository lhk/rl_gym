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

        self.car_speed += acceleration * params.dT
        self.car_speed = np.clip(self.car_speed, params.min_speed, params.max_speed)

        x, y = self.car_position
        dx = -np.sin(self.car_rotation / 180 * np.pi) * self.car_speed * params.dT
        new_x = x + dx
        dy = -np.cos(self.car_rotation / 180 * np.pi) * self.car_speed * params.dT
        new_y = y + dy

        self.car_position[:] = (new_x, new_y)
        self.car_rotation -= self.car_speed * steering_angle * params.dT * params.steering_factor
