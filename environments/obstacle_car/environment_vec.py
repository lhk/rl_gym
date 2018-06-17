import numpy as np
import pygame

import environments.obstacle_car.colors as colors
import environments.obstacle_car.params_radial as params
import environments.obstacle_car.utils as utils

from np_draw.sprite import Sprite
from environments.obstacle_car.car import Car

from skimage.io import imread
from skimage.transform import resize

import gym
from gym import spaces, logger
from gym.utils import seeding


class Environment_Vec(gym.Env):
    def __init__(self, polar_coords=True):

        self.polar_coords = polar_coords

        # the position will be overwritten later
        default_pos = np.zeros((2,))
        self.car = Car(default_pos, 0, 0, params)
        self.car_dim = np.linalg.norm(params.car_size)

        self.goal_pos = default_pos
        self.goal_dim = np.linalg.norm(params.goal_size)

        self.actions = [[0, 0], [0, -1], [0, 1], [1, 0]]
        self.num_actions = len(self.actions)

        self.action_space = spaces.Discrete(self.num_actions)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.steps = 0

        # set up values for dynamics
        self.car.rot = 0
        self.car.speed = 0

        # set up car, obstacle and goal positions
        car_position = np.array([0, 0], dtype=np.float64)
        car_position[0] = self.np_random.uniform(params.car_size[0] / 2, params.screen_size[0] - params.car_size[0] / 2)
        car_position[1] = params.screen_size[1] - params.car_size[1] / 2
        self.car.pos = car_position

        goal_position = np.array([0, 0])
        goal_position[0] = self.np_random.uniform(0, params.screen_size[0] - params.goal_size[0])
        goal_position[1] = params.goal_size[1] / 2
        self.goal_pos = goal_position

        # if the car gets too far away from the goal,
        # we stop the simulation
        # this stop is based on the initial distance
        self.initial_dist = np.linalg.norm(self.car.pos - self.goal_pos)

        # minimum distance an obstacle needs to have from car and goal
        min_dist = (1.5 * self.car_dim + min(params.goal_size))

        self.obstacle_positions = []
        for i in range(params.num_obstacles):
            while True:
                obs_x = self.np_random.rand() * params.screen_size[0]
                obs_y = params.screen_size[1] * 1 / 3 * (1 + self.np_random.rand())
                obstacle_position = np.array([obs_x, obs_y])
                # obstacle must be away from car and goal
                car_dist = np.linalg.norm(obstacle_position - self.car.pos)
                goal_dist = np.linalg.norm(obstacle_position - self.goal_pos)

                if car_dist > min_dist and goal_dist > min_dist:
                    self.obstacle_positions.append(obstacle_position)
                    break

    def get_observation(self, rotated=True):

        # set up a rotation matrix
        if rotated:
            theta = self.car.rot / 180 * np.pi
        else:
            theta = 0

        c, s = np.cos(theta), np.sin(theta)
        mat = np.array(((c, -s), (s, c)))

        # stack obstacle and goal positions
        targets = np.vstack([self.goal_pos, *self.obstacle_positions])

        # origin is car position
        targets = self.car.pos - targets

        # rotate to face car
        targets = (mat @ targets.T).T

        if self.polar_coords:
            distances = np.linalg.norm(targets, axis=1)
            distances = distances / params.distance_rescale
            angles = np.arctan2(targets[:, 0], targets[:, 1])
            distance_angles = np.array(list(zip(distances, angles)))
            observation_vector = np.stack([self.car.speed, *distance_angles.flatten()])
        else:
            targets = targets / params.distance_rescale
            observation_vector = np.stack([self.car.speed, *targets.flatten()])

        return observation_vector

    def step(self, action):
        assert self.action_space.contains(action)
        # internally the action is not a number, but a combination of acceleration and steering
        action = self.actions[action]
        return self.make_action(action)

    def make_action(self, action):
        acceleration, steering_angle = action

        old_dist = np.linalg.norm(self.car.pos - self.goal_pos)
        self.car.update(acceleration, steering_angle)
        new_dist = np.linalg.norm(self.car.pos - self.goal_pos)

        # if params.reward_distance != 0,
        # then the environment rewards you for moving closer to the goal
        dist_reward = (old_dist - new_dist) * params.reward_distance

        observation_vector = self.get_observation()
        targets = observation_vector[1:].reshape((-1, 2))

        if self.polar_coords:
            distances = targets[:, 0]
            distances = distances * params.distance_rescale
        else:
            targets = targets * params.distance_rescale
            distances = np.linalg.norm(targets, axis=1)

        # we have moved out of the simulation domain
        if new_dist > params.max_dist * self.initial_dist:
            return observation_vector, params.reward_collision, True

        rel_goal_dist = distances[0]
        if rel_goal_dist < self.car_dim:
            return observation_vector, params.reward_goal, True

        rel_obs_dist = distances[1:]
        if np.any(rel_obs_dist < self.car_dim):
            return observation_vector, params.reward_collision, True

        self.steps += 1
        if self.steps > params.timeout:
            return observation_vector, params.reward_timestep + dist_reward, True

        return observation_vector, params.reward_timestep + dist_reward, False

    def sample_action(self):
        # for atari, the actions are simply numbers
        return self.np_random.choice(self.num_actions)
