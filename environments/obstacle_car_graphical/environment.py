import numpy as np
import pygame

import environments.obstacle_car_graphical.colors as colors
import environments.obstacle_car_graphical.params as params
import environments.obstacle_car_graphical.utils as utils


class Environment():
    def __init__(self):
        pygame.init()

        # the surface on which everything is rendered
        self.view = pygame.Surface(params.screen_size)

        # loading assets
        self.car = pygame.image.load("environments/obstacle_car_graphical/assets/car.png")
        self.car = pygame.transform.smoothscale(self.car, params.car_size)

        self.black = pygame.Surface(params.obstacle_size)
        self.black.fill(colors.black)

        self.green = pygame.Surface(params.goal_size)
        self.green.fill(colors.green)

        # self.actions = [[1, -1], [1, 0], [1, 1],
        #                [0, -1], [0, 0], [0, 1],
        #                [-1, -1], [-1, 0], [-1, 1]]

        self.actions = [[0, 0], [0, -1], [0, 1], [1, 0]]

        self.num_actions = len(self.actions)

    def reset(self):

        self.steps = 0

        # set up car, obstacle and goal positions
        self.car_position = np.array([0, 0], dtype=np.float64)
        self.car_position[0] = np.random.uniform(params.car_size[0] / 2, params.screen_size[0] - params.car_size[0] / 2)
        self.car_position[1] = params.screen_size[1] - params.car_size[1] / 2

        self.goal_position = np.array([0, 0])
        self.goal_position[0] = np.random.uniform(0, params.screen_size[0] - params.goal_size[0])
        self.goal_position[1] = 0  # + params.goal_size[1]/2

        self.car_dim = np.linalg.norm(params.car_size, np.inf)
        self.goal_dim = np.linalg.norm(params.goal_size, np.inf)
        self.obs_dim = np.linalg.norm(params.obstacle_size, np.inf)

        min_dist = (self.car_dim + self.goal_dim)

        self.obstacle_positions = []
        for i in range(params.num_obstacles):
            while True:
                obstacle_position = np.random.rand(2) * params.screen_size
                # obstacle must be away from car and goal
                car_dist = np.linalg.norm(obstacle_position - self.car_position)
                goal_dist = np.linalg.norm(obstacle_position - self.goal_position)

                if car_dist > min_dist and goal_dist > min_dist:
                    self.obstacle_positions.append(obstacle_position)
                    break

        # set up values for dynamics
        self.car_rotation = 0
        self.car_speed = 0

    def render(self, return_numpy=True):
        self.view.fill(colors.white)

        # plot all the obstacles
        self.view.fill(colors.white)
        for obstacle_position in self.obstacle_positions:
            self.view.blit(self.black, obstacle_position)

        # plot the goal
        self.view.blit(self.green, self.goal_position)

        car_rect = self.car.get_rect()
        car_rect.center = self.car_position

        rotated_surface, rotated_rect = utils.rotate(self.car, car_rect, self.car_rotation)
        self.view.blit(rotated_surface, rotated_rect)

        if return_numpy:
            # get numpy surface
            np_arr = pygame.surfarray.array3d(self.view)

            return np_arr

        else:
            return self.view

    def step(self, action):
        # internally the action is not a number, but a combination of acceleration and steering
        action = self.actions[action]
        return self.make_action(action)

    def make_action(self, action):
        acceleration, steering_angle = action
        acceleration = np.clip(acceleration, -1, 1)
        steering_angle = np.clip(steering_angle, -1, 1)

        # TODO: fill with proper single track model
        self.car_speed += acceleration * params.dT
        self.car_speed = np.clip(self.car_speed, params.min_speed, params.max_speed)
        x, y = self.car_position
        dx = -np.sin(self.car_rotation / 180 * np.pi) * self.car_speed * params.dT
        new_x = x + dx
        dy = -np.cos(self.car_rotation / 180 * np.pi) * self.car_speed * params.dT
        new_y = y + dy

        border_collision = False
        if new_x > params.screen_size[0]:
            border_collision = True
            new_x = params.screen_size[0]
        elif new_x < 0:
            border_collision = True
            new_x = 0

        if new_y > params.screen_size[1]:
            border_collision = True
            new_y = params.screen_size[1]
        elif new_y < 0:
            border_collision = True
            new_y = 0

        if border_collision:
            self.car_speed = 0

        # move the car, get the new rendering
        self.car_position[:] = (new_x, new_y)
        self.car_rotation -= self.car_speed * steering_angle * params.dT * params.steering_factor
        observation = self.render()

        if border_collision and params.stop_on_border_collision:
            return observation, params.reward_collision, True

        for obstacle in self.obstacle_positions:
            obstacle = obstacle + np.array(params.obstacle_size) / 2
            dist_obs = np.linalg.norm(obstacle - self.car_position, np.inf)
            if dist_obs < 0.5 * (self.car_dim + self.obs_dim):
                # collision with obstacle
                return observation, params.reward_collision, True

        goal_pos = self.goal_position + np.array(params.goal_size) / 2
        dist_goal = np.linalg.norm(goal_pos - self.car_position, np.inf)
        if dist_goal < 0.5 * (self.car_dim + self.goal_dim):
            # collision with goal, yay
            return observation, params.reward_goal, True

        self.steps += 1
        if self.steps > params.timeout:
            return observation, params.reward_timestep, True

        return observation, params.reward_timestep, False

    def sample_action(self):
        # for atari, the actions are simply numbers
        return np.random.choice(self.num_actions)
