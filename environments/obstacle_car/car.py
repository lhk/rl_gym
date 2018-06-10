import numpy as np
import pygame


class Car():
    def __init__(self, pos, speed, rot, params):
        self.pos = pos
        self.speed = speed
        self.rot = rot
        self.params = params

    def update(self, acceleration, steering_angle):
        acceleration = np.clip(acceleration, -1, 1)
        steering_angle = np.clip(steering_angle, -1, 1)

        self.speed += acceleration * self.params.dT
        self.speed = np.clip(self.speed, self.params.min_speed, self.params.max_speed)

        x, y = self.pos
        dx = -np.sin(self.rot / 180 * np.pi) * self.speed * self.params.dT
        new_x = x + dx
        dy = -np.cos(self.rot / 180 * np.pi) * self.speed * self.params.dT
        new_y = y + dy

        self.pos[:] = (new_x, new_y)
        self.rot -= self.speed * steering_angle * self.params.dT * self.params.steering_factor
