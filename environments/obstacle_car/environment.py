import numpy as np
import pygame

import environments.obstacle_car.colors as colors
import environments.obstacle_car.params as params
import environments.obstacle_car.utils as utils

from np_draw.sprite import Sprite
from environments.obstacle_car.car import Car

from skimage.io import imread
from skimage.transform import resize


class Environment_Graphical():
    def __init__(self):

        # set up numpy arrays to be drawn to
        self.canvas = np.zeros((*params.screen_size, 3))

        self.background = np.zeros((*params.screen_size, 3))
        self.background_mask = np.zeros((*params.screen_size,), dtype=np.bool)

        self.foreground = np.zeros((*params.screen_size,3))
        self.foreground_mask = np.zeros((*params.screen_size,), dtype=np.bool)

        # load images and set up their masks
        car_img_transp = imread("environments/obstacle_car/assets/car.png")
        car_img_transp = np.transpose(car_img_transp, [1,0,2])
        car_img_transp = resize(car_img_transp, params.car_size)
        car_img = car_img_transp[:, :, :3]# cut away alpha
        car_mask = (car_img_transp[:, :, 3]>0).astype(np.bool)

        obstacle_img = np.zeros((*params.obstacle_size, 3))
        obstacle_img[:, :, 0] = np.sin(np.linspace(0, 2*np.pi, params.obstacle_size[0])).reshape((-1,1))
        goal_img = np.zeros((*params.goal_size, 3))
        goal_img[:, :, 1]=np.sin(np.linspace(0, 4*np.pi, params.goal_size[1]))

        obstacle_mask = np.ones(params.obstacle_size, dtype=np.bool)
        goal_mask = np.ones(params.goal_size, dtype=np.bool)

        # the position will be overwritten later
        default_pos = np.zeros((2,))
        self.car_sprite = Sprite(car_img, car_mask, default_pos, 0)
        self.obstacle_sprite = Sprite(obstacle_img, obstacle_mask, default_pos, 0)
        self.goal_sprite = Sprite(goal_img, goal_mask, default_pos, 0)

        # car and car_sprite are not the same
        # one is just for graphics, the other is for dynamic movement of the car
        self.car = Car(default_pos, 0, 0, params)

        self.actions = [[0, 0], [0, -1], [0, 1], [1, 0]]
        self.num_actions = len(self.actions)

    def reset(self):

        self.steps = 0

        # set up values for dynamics
        self.car_sprite.set_rotation(0)
        self.car.rot = 0
        self.car.speed = 0

        # set up car, obstacle and goal positions
        car_position = np.array([0, 0], dtype=np.float64)
        car_position[0] = np.random.uniform(params.car_size[0] / 2, params.screen_size[0] - params.car_size[0] / 2)
        car_position[1] = params.screen_size[1] - params.car_size[1] / 2
        self.car_sprite.set_position(car_position)
        self.car.pos = car_position

        goal_position = np.array([0, 0])
        goal_position[0] = np.random.uniform(0, params.screen_size[0] - params.goal_size[0])
        goal_position[1] = params.goal_size[1] / 2
        self.goal_sprite.set_position(goal_position)

        min_dist = (1.5 * self.car_sprite.dim + min(self.goal_sprite.size))

        self.obstacle_positions = []
        for i in range(params.num_obstacles):
            while True:
                obs_x = np.random.random() * params.screen_size[0]
                obs_y = params.screen_size[1] * 1 / 3 * (1 + np.random.random())
                obstacle_position = np.array([obs_x, obs_y])
                # obstacle must be away from car and goal
                car_dist = np.linalg.norm(obstacle_position - self.car.pos)
                goal_dist = np.linalg.norm(obstacle_position - self.goal_sprite.pos)

                if car_dist > min_dist and goal_dist > min_dist:
                    self.obstacle_positions.append(obstacle_position)
                    break

        # render to background
        self.background[:] = 0
        self.background_mask[:] = False

        self.goal_sprite.render(self.background, self.background_mask)
        for obstacle_position in self.obstacle_positions:
            self.obstacle_sprite.set_position(obstacle_position)
            self.obstacle_sprite.render(self.background, self.background_mask)


    def render(self):
        # reset canvas and foreground,
        # background is not rerendered
        self.canvas[:] = self.background
        self.foreground[:] = 0
        self.foreground_mask[:] = False

        # plot the car
        self.car_sprite.render(self.foreground, self.foreground_mask)

        # overlay foreground to canvas
        self.canvas[self.foreground_mask] = self.foreground[self.foreground_mask]

        return (self.canvas * 255).astype(np.uint8)

    def step(self, action):
        # internally the action is not a number, but a combination of acceleration and steering
        action = self.actions[action]
        return self.make_action(action)

    def make_action(self, action):
        acceleration, steering_angle = action

        old_dist = np.linalg.norm(self.car.pos - self.goal_sprite.pos)
        self.car.update(acceleration, steering_angle)
        new_dist = np.linalg.norm(self.car.pos - self.goal_sprite.pos)

        # if params.reward_distance != 0,
        # then the environment rewards you for moving closer to the goal
        dist_reward = (old_dist - new_dist) * params.reward_distance

        x, y = self.car.pos

        border_collision = False
        if x > params.screen_size[0]:
            border_collision = True
            self.car.pos[0] = params.screen_size[0]
        elif x < 0:
            border_collision = True
            self.car.pos[0] = 0

        if y > params.screen_size[1]:
            border_collision = True
            self.car.pos[1] = params.screen_size[1]
        elif y < 0:
            border_collision = True
            self.car.pos[1] = 0

        if border_collision:
            self.car.speed = 0

        # sync dynamics and graphics
        self.car_sprite.set_position(self.car.pos)
        self.car_sprite.set_rotation(-self.car.rot)

        observation = self.render()

        if self.car_sprite.collide(self.goal_sprite):
            return observation, params.reward_goal, True

        if border_collision and params.stop_on_border_collision:
            return observation, params.reward_collision, True

        for obstacle in self.obstacle_positions:
            self.obstacle_sprite.set_position(obstacle)
            if self.car_sprite.collide(self.obstacle_sprite):
                return observation, params.reward_collision, True

        self.steps += 1
        if self.steps > params.timeout:
            return observation, params.reward_timestep + dist_reward, True

        return observation, params.reward_timestep + dist_reward, False

    def sample_action(self):
        # for atari, the actions are simply numbers
        return np.random.choice(self.num_actions)
