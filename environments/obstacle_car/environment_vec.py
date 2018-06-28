import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import environments.obstacle_car.params as params
from environments.obstacle_car.car import Car


class Environment_Vec(gym.Env):
    def __init__(self, polar_coords=True):

        self.polar_coords = polar_coords

        # the position will be overwritten later
        default_pos = np.zeros((2,))
        self.car = Car(default_pos, 0, 0, params)
        self.car_dim = np.linalg.norm(params.car_size)

        self.goal_pos = default_pos
        self.goal_dim = np.linalg.norm(params.goal_size)

        self.obs_dim = np.linalg.norm(params.obstacle_size)

        self.actions = [[0, 0], [0, -1], [0, 1], [1, 0]]
        self.num_actions = len(self.actions)

        self.action_space = spaces.Discrete(self.num_actions)
        if self.polar_coords:
            min = np.array([params.min_speed,
                            *[0, -np.pi] * 3])

            max = np.array([params.max_speed,
                            *[np.finfo(np.float32).max, +np.pi] * 3])
        else:
            min = np.array([params.min_speed,
                            *[-np.finfo(np.float32).max, -np.finfo(np.float32).max] * 3])

            max = np.array([params.max_speed,
                            *[np.finfo(np.float32).max, np.finfo(np.float32).max] * 3])

        self.observation_space = spaces.Box(min, max)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_to_canvas(self, canvas):
        x_coords = np.arange(canvas.shape[0])
        y_coords = np.arange(canvas.shape[1])
        x, y = np.meshgrid(x_coords, y_coords)
        coords = np.stack([x, y], axis=-1)

        observation = self.get_observation()

        # first observation is speed, throw it away
        # the rest has to be rescaled to the original range
        observation = observation[1:]

        # then we organize it as vectors
        observation = observation.reshape((-1, 2))

        # if the environment is based on polar coordinates, we transform them into cartesian
        if self.polar_coords:
            distances = observation[:, 0]
            angles = observation[:, 1]
            x = distances * np.cos(angles)
            y = distances * np.sin(angles)
            observation = np.stack([y, x], axis=-1)

        observation = observation * params.distance_rescale

        offset = np.array([canvas.shape[0] // 2, canvas.shape[1] // 2])
        observation = (observation + offset).astype(np.int)

        goal = observation[0]
        obstacles = observation[1:]

        goal = np.array(canvas.shape[:2]) - goal
        obstacles = np.array(canvas.shape[:2]) - obstacles

        dist = coords - goal
        dist = np.linalg.norm(dist, axis=-1)
        area = np.where(dist < self.initial_dist * params.max_dist)
        canvas[area[1], area[0], 2] = 0.5

        if np.all(goal > 0) and np.all(goal < canvas.shape[:2]):
            canvas[goal[0] - 5:goal[0] + 5, goal[1] - 5:goal[1] + 5, :] = 1

        for obstacle in obstacles:
            if np.all(obstacle > 0) and np.all(obstacle < canvas.shape[:2]):
                canvas[obstacle[0] - 5:obstacle[0] + 5, obstacle[1] - 5:obstacle[1] + 5, 0] = 1

        # a green dot at the center of the canvas, for our car
        canvas[offset[0] - 5:offset[0] + 5, offset[1] - 5:offset[1] + 5, 1] = 1
        return canvas

    def reset(self):

        self.steps = 0

        # set up values for dynamics
        self.car.rot = 0
        self.car.speed = 0

        # set up car, obstacle and goal positions
        car_position = np.array([0, 0], dtype=np.float64)
        car_position[0] = params.screen_size[
                              0] // 2  # self.np_random.uniform(params.car_size[0] / 2, params.screen_size[0] - params.car_size[0] / 2)
        car_position[1] = params.screen_size[1] - params.car_size[1] / 2
        self.car.pos = car_position

        goal_position = np.array([0, 0])
        goal_position[0] = params.screen_size[
                               0] // 2  # self.np_random.uniform(0, params.screen_size[0] - params.goal_size[0])
        goal_position[1] = params.goal_size[1] / 2
        self.goal_pos = goal_position

        # if the car gets too far away from the goal,
        # we stop the simulation
        # this stop is based on the initial distance
        self.initial_dist = np.linalg.norm(self.car.pos - self.goal_pos)

        # minimum distance an obstacle needs to have from car and goal
        min_dist = (1.5 * self.car_dim + self.goal_dim)

        self.obstacle_positions = []
        for i in range(params.num_obstacles):
            while True:
                obs_x = params.screen_size[0] // 2 + (self.np_random.rand() - 0.5) * params.obs_x_spread
                obs_y = params.screen_size[1] * self.np_random.rand()
                obstacle_position = np.array([obs_x, obs_y])
                # obstacle must be away from car and goal
                car_dist = np.linalg.norm(obstacle_position - self.car.pos)
                goal_dist = np.linalg.norm(obstacle_position - self.goal_pos)

                if car_dist > min_dist and goal_dist > min_dist:
                    self.obstacle_positions.append(obstacle_position)
                    break

        return self.get_observation()

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
        obs, rew, done = self.make_action(action)
        return obs, rew, done, {}

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

        relative_goal_position = self.car.pos - self.goal_pos
        x_distance = abs(relative_goal_position[0])

        if x_distance > params.x_tolerance:
            return observation_vector, params.reward_collision, True

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
        if rel_goal_dist < 1 / 2 * (self.car_dim + self.goal_dim):
            return observation_vector, params.reward_goal, True

        rel_obs_dist = distances[1:]
        if np.any(rel_obs_dist < 1 / 2 * (self.car_dim + self.obs_dim)):
            return observation_vector, params.reward_collision, True

        self.steps += 1
        if self.steps > params.timeout:
            return observation_vector, params.reward_timestep + dist_reward, True

        return observation_vector, params.reward_timestep + dist_reward, False

    def sample_action(self):
        # for atari, the actions are simply numbers
        return self.np_random.choice(self.num_actions)
